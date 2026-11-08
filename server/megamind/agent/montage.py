"""Montage: a captioned highlight reel of the wearer's memories (ported from Recall's `montage.py`).

`POST /v1/montage` picks candidate tokens (retriever for a query, `store.range` for a time range), asks a
planner for the best few moments (`MontagePlan`), turns each moment into a short clip and renders one mp4
with the time and a title burned in and short fades between clips. `GET /v1/montages/{id}.mp4` serves it
with HTTP Range.

- Planner: Claude (forced `montage_plan` tool call, same key/model/timeout settings as the chat agent) when
  `ANTHROPIC_API_KEY` is set and `MONTAGE_PLANNER` is not `offline`; otherwise, or when the call fails, the
  deterministic offline plan (evenly spaced informative moments, de-duplicated).
- Clip sources: real headset video (`POST /v1/video`, `megamind.video`) when stored clips cover at least
  `VIDEO_MIN_COVERAGE` of the clip (`source="video"`), else the #58 replay ring buffer when it still holds frames
  around the moment (`source="replay"`, real frame rate; both skipped when `PRIVACY_BLUR_FACES` is on, since
  neither is blurred), else a
  timelapse of the stored keyframe thumbnails from that moment on, each held `MONTAGE_TIMELAPSE_STEP_S`
  (`source="timelapse"`, ~1 fps; keyframes are seconds apart).
- Rendering reuses #58: `render_mp4` (with `extra_vf` for captions/fades) per clip, `ReplayCache` for the
  per-key lock, ffmpeg semaphore (montage's own, `MONTAGE_MAX_CONCURRENT_RENDERS`) and pruning, and
  `ranged_file_response` for Range. The cache key is a hash of the plan, so the same plan is rendered once.

Content-filtered ("smart") montages ("montage of all shots of the guy in the black coat") take a separate,
additive path in `smart_montage.py` (gather across history -> Haiku vision check -> one clip per matching moment)
when the query names a subject; time-range montages and subject-less queries keep the path above unchanged.

Wiring (`install`) is additive: the router plus a middleware that clears the montage cache after a
successful `DELETE /v1/memories`. Settings: `MONTAGE_*` env vars (see `.env.example`).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anthropic
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from megamind.agent import smart_montage as smart
from megamind.models import MemoryToken, MontageClip, MontageRequest, MontageResponse
from megamind.replay.render import FFMPEG_HINT, ReplayCache, find_ffmpeg, render_mp4
from megamind.replay.routes import parse_ts, ranged_file_response
from megamind.video.render import render_video, timeline_s
from megamind.video.store import Segment, covered_s

log = logging.getLogger("megamind.montage")

router = APIRouter()

_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)
_ID_RE = re.compile(r"[0-9a-f]{20}")
TIMELAPSE_LOOKAHEAD_S = 120.0  # keyframes after the moment that may join its timelapse
TIMELAPSE_LOOKBACK_S = 30.0  # transcript moments: keyframes this long before the words count too
RANGE_SCAN_LIMIT = 5000  # tokens read at once for a range-only montage; longer ranges are sampled per bucket


class MontageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONTAGE_", env_file=".env", extra="ignore")

    enabled: bool = True
    planner: str = "auto"  # auto (Claude iff ANTHROPIC_API_KEY) | claude | offline
    max_clips: int = 8
    max_candidates: int = 40  # moments shown to the planner
    clip_s: float = 4.0  # default clip length; the planner may pick 2.5-6 s
    pre_s: float = 1.0  # replay clips start this long before the moment
    timelapse_step_s: float = 1.0  # stored keyframes held this long each
    width: int = 960
    height: int = 540
    fps: int = 10
    font: str = ""  # TTF for captions; default: first system font found, else ffmpeg's fontconfig default
    cache_dir: Path | None = None  # default <thumb_dir>/../montages
    cache_max_files: int = 16
    max_concurrent_renders: int = 1
    # content-filtered ("smart") montages, see `smart_montage.py`: "montage of all shots of the guy in the black coat"
    smart: str = "auto"  # auto (subject found AND collection cue: "all/every ...", "shots of") | off (mode="smart")
    smart_max_candidates: int = 60  # retriever candidates sent to the verifier
    smart_max_clips: int = 12
    verifier: str = "auto"  # auto (Claude iff ANTHROPIC_API_KEY) | offline (keep retrieval order, no check)
    verify_model: str = "claude-haiku-5-5"
    verify_batch: int = 8  # candidates (thumbnails) per verifier call
    verify_concurrency: int = 4
    verify_image_px: int = 384
    verify_min_confidence: float = 0.5
    moment_gap_s: float = 90.0  # matches closer than this (same device) are one moment -> one clip


# ---- plan ----


class PlannedClip(BaseModel):
    label: str = Field(description="Moment label, e.g. 'S4'")
    title: str = Field(description="3-7 word on-screen caption, e.g. 'Coffee with a colleague'")
    duration: float = Field(4.0, description="Clip length in seconds, 2.5-6")


class MontagePlan(BaseModel):
    title: str = Field(description="Short title for the whole montage, e.g. 'Your afternoon'")
    clips: list[PlannedClip] = Field(description="Chosen highlights in chronological order")
    narration: str = Field("", description="1-2 sentence spoken reply describing what the montage covers")


PLAN_TOOL = {
    "name": "montage_plan",
    "description": "Return the highlight reel plan.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title for the whole montage, e.g. 'Your afternoon'"},
            "clips": {
                "type": "array",
                "description": "Chosen highlights in chronological order",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Moment label, e.g. 'S4'"},
                        "title": {"type": "string", "description": "3-7 word on-screen caption"},
                        "duration": {"type": "number", "description": "Clip length in seconds, 2.5-6"},
                    },
                    "required": ["label", "title"],
                },
            },
            "narration": {"type": "string", "description": "1-2 plain spoken sentences, no markdown"},
        },
        "required": ["title", "clips", "narration"],
    },
}

SYSTEM = """You are editing a highlight reel from the wearer's first-person AR-headset memories. \
Given time-ordered moments (caption, on-screen text, heard speech), choose the most interesting, varied \
ones: people, conversations, places, meals, notable sights, things the wearer did. Skip near-identical or \
empty moments. Keep it chronological. Prefer moments where something happens or is said. Never identify \
people by name unless the moment's text names them; describe them neutrally. Aim for the requested number \
of clips and answer only by calling the montage_plan tool."""


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _clock(ts: float, zone: ZoneInfo) -> str:
    return datetime.fromtimestamp(ts, zone).strftime("%I:%M %p").lstrip("0")


def _labels(cands: list[MemoryToken]) -> dict[str, MemoryToken]:
    return {f"S{i + 1}": t for i, t in enumerate(sorted(cands, key=lambda t: t.ts))}


def _informative(t: MemoryToken) -> bool:
    return bool(t.ocr_text.strip()) or (bool(t.caption.strip()) and not t.caption.lstrip().startswith("("))


def _short(text: str, n: int = 42) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def evenly(items: list, n: int) -> list:
    """`n` items evenly spaced over `items` (first and last included), order kept."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idx]


def offline_plan(cands: list[MemoryToken], n_clips: int, *, tz: str = "UTC", clip_s: float = 4.0) -> MontagePlan:
    """Deterministic fallback (Recall's): evenly spaced informative moments, identical captions only once."""
    labels = _labels(cands)
    pool = [k for k, t in labels.items() if _informative(t)] or list(labels)
    seen: set[str] = set()
    uniq = []
    for k in pool:  # speech spanning two windows / a static scene shouldn't yield identical clips
        key = " ".join((labels[k].caption or labels[k].ocr_text or k).lower().split())
        if key not in seen:
            seen.add(key)
            uniq.append(k)
    zone = _zone(tz)
    clips = []
    for k in evenly(uniq, n_clips):
        t = labels[k]
        caption = "" if t.caption.lstrip().startswith("(") else t.caption
        raw = caption or t.ocr_text or f"Moment at {_clock(t.ts, zone)}"
        clips.append(PlannedClip(label=k, title=_short(raw), duration=clip_s))
    return MontagePlan(
        title="Your highlights",
        clips=clips,
        narration=f"Here's your highlight reel: {len(clips)} moment{'s' * (len(clips) != 1)} from that period.",
    )


class MontagePlanner:
    """Claude picks the clips when configured; the offline plan otherwise (and on any API/schema failure)."""

    def __init__(self, settings, cfg: MontageSettings, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.cfg = cfg
        self.model = settings.claude_model
        self.tz = settings.wearer_tz
        self._client = client
        mode = cfg.planner.lower()
        if self._client is None and mode != "offline" and settings.anthropic_api_key:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                max_retries=settings.claude_max_retries,
                timeout=settings.claude_timeout_s,
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def plan(self, request: str, cands: list[MemoryToken], n_clips: int) -> tuple[MontagePlan, str]:
        """(plan, "claude" | "offline"). Plan clips reference `_labels(cands)` and are chronological."""
        labels = _labels(cands)
        if self._client is not None:
            try:
                p = await self._claude_plan(request, labels, n_clips)
                order = list(labels)
                seen: set[str] = set()
                clips = []
                for c in sorted((c for c in p.clips if c.label in labels), key=lambda c: order.index(c.label)):
                    if c.label not in seen:
                        seen.add(c.label)
                        clips.append(c)
                if clips:
                    return p.model_copy(update={"clips": clips[: max(n_clips, 1)]}), "claude"
                log.warning("montage planner returned no usable clips; using the offline plan")
            except Exception as e:  # noqa: BLE001  (API/network error, no tool_use block, schema mismatch)
                log.warning("montage planning failed (%s); using the offline plan", e)
        return offline_plan(cands, n_clips, tz=self.tz, clip_s=self.cfg.clip_s), "offline"

    async def _claude_plan(self, request: str, labels: dict[str, MemoryToken], n_clips: int) -> MontagePlan:
        zone = _zone(self.tz)
        lines = []
        for k, t in labels.items():
            parts = [t.caption.strip()]
            if t.ocr_text.strip():
                parts.append(f"text: {t.ocr_text.strip()}")
            body = " | ".join(p for p in parts if p)[:400] or "(no description)"
            lines.append(f"[{k}] {datetime.fromtimestamp(t.ts, zone):%H:%M:%S} ({t.kind}): {body}")
        ctx = f"Request: {request}\nTarget clips: {min(n_clips, len(labels))}\n\nMoments:\n" + "\n".join(lines)
        req = {"model": self.model, "system": SYSTEM, "tools": [PLAN_TOOL]}
        try:
            resp = await self._client.messages.create(
                **req,
                max_tokens=1500,
                tool_choice={"type": "tool", "name": PLAN_TOOL["name"]},
                messages=[{"role": "user", "content": ctx}],
            )
        except anthropic.BadRequestError as e:
            if "tool_choice" not in str(e):
                raise
            # Claude Sonnet 5.5 / Opus 5.5 reject forced tool use (400): ask for the tool instead. Thinking is on
            # there and shares max_tokens, hence the larger budget.
            resp = await self._client.messages.create(
                **req,
                max_tokens=8000,
                tool_choice={"type": "auto"},
                messages=[{"role": "user", "content": ctx + f"\n\nCall {PLAN_TOOL['name']} with your plan."}],
            )
        block = next(b for b in resp.content if getattr(b, "type", "") == "tool_use")
        return MontagePlan.model_validate(block.input)


# ---- clips + render ----


@dataclass(frozen=True)
class ClipSpec:
    token: MemoryToken
    title: str
    when: str  # burned-in time label
    frames: tuple[tuple[float, bytes], ...]  # (ts, jpeg); held until the next frame
    tail_s: float
    source: str  # "video" | "replay" | "timelapse"
    video: tuple[Segment, ...] = ()  # source="video": pieces of stored clips (frames is empty)

    @property
    def duration(self) -> float:
        if self.video:
            return timeline_s(self.video)
        return self.frames[-1][0] - self.frames[0][0] + self.tail_s

    def key(self) -> tuple:
        if self.video:
            segs = tuple((s.clip.id, round(s.start, 3), round(s.end, 3)) for s in self.video)
            return (self.token.id, self.source, self.title, self.when, segs)
        return (self.token.id, self.source, self.title, self.when, self.frames[0][0], self.frames[-1][0],
                len(self.frames), round(self.tail_s, 3))  # fmt: skip


def _esc(text: str) -> str:
    """drawtext text inside single quotes in a filtergraph (Recall's escaping, minus `%`: we pass
    expansion=none, since `\\%` drops the whole line on current ffmpeg) + no control characters."""
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")


def _font(cfg: MontageSettings) -> str | None:
    if cfg.font:
        return cfg.font if Path(cfg.font).is_file() else None
    return next((f for f in _FONTS if Path(f).is_file()), None)


def overlay_vf(clip: ClipSpec, cfg: MontageSettings, *, text: bool = True) -> str:
    """Normalise every clip to WxH (so the parts concat without re-encoding), burn in time + title, fade."""
    w, h = max(2, cfg.width - cfg.width % 2), max(2, cfg.height - cfg.height % 2)  # libx264 needs even sizes
    f = [f"scale={w}:{h}:force_original_aspect_ratio=decrease", f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2", "setsar=1"]
    # render_mp4's concat list holds the last frame ~2x tail_s; trim so the clip (and its fade-out) is exact
    f.append(f"trim=duration={clip.duration:.3f}")
    if text:
        s = h / 720  # Recall's layout was tuned at 720p
        font = _font(cfg)
        ff = f"fontfile='{_esc(font)}':" if font else ""
        bar = round(110 * s)
        f += [
            f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black@0.45:t=fill",
            f"drawtext={ff}expansion=none:text='{_esc(clip.when)}':fontcolor=white@0.85:fontsize={round(28 * s)}"
            f":x={round(40 * s)}:y=h-{round(100 * s)}",
            f"drawtext={ff}expansion=none:text='{_esc(clip.title)}':fontcolor=white:fontsize={round(42 * s)}"
            f":x={round(40 * s)}:y=h-{round(62 * s)}",
        ]
    fade = min(0.3, clip.duration / 4)
    f += [f"fade=t=in:st=0:d={fade:.2f}", f"fade=t=out:st={max(clip.duration - fade, 0):.2f}:d={fade:.2f}"]
    return ",".join(f)


def render_montage(clips: list[ClipSpec], out: Path, *, ffmpeg: str, cfg: MontageSettings, timeout_s=300.0) -> Path:
    """One `render_mp4` per clip (identical size/fps/codec), then a stream-copy concat."""
    if not clips:
        raise ValueError("no clips")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".montage-") as td:
        tmp = Path(td)
        parts = []
        for i, c in enumerate(clips):
            part = tmp / f"p{i:03d}.mp4"
            if c.video:  # same size/fps/codec as render_mp4's parts, so the stream-copy concat still works

                def draw(vf: str, c: ClipSpec = c, part: Path = part) -> None:
                    render_video(c.video, part, ffmpeg=ffmpeg, fps=cfg.fps, max_width=cfg.width, extra_vf=vf)
            else:
                kw = dict(ffmpeg=ffmpeg, fps=cfg.fps, max_width=cfg.width, tail_s=c.tail_s)

                def draw(vf: str, c: ClipSpec = c, part: Path = part, kw: dict = kw) -> None:
                    render_mp4(c.frames, part, extra_vf=vf, **kw)

            try:
                draw(overlay_vf(c, cfg))
            except RuntimeError as e:  # e.g. ffmpeg built without libfreetype (no drawtext): uncaptioned clip
                log.warning("montage: captions failed (%s); rendering clip %d without text", e, i)
                draw(overlay_vf(c, cfg, text=False))
            parts.append(part)
        (tmp / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in parts))
        tmp_out = tmp / "montage.mp4"
        cmd = [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
               "-c", "copy", "-movflags", "+faststart", str(tmp_out)]  # fmt: skip
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg concat failed: {e.stderr.strip()[-500:]}") from e
        os.replace(tmp_out, out)
    return out


class MontageCache(ReplayCache):
    """#58's `ReplayCache` (per-key lock, ffmpeg semaphore, newest-N pruning, `clear`), montage render."""

    def __init__(self, cache_dir: Path, cfg: MontageSettings) -> None:
        super().__init__(cache_dir, cfg.cache_max_files, cfg.max_concurrent_renders)
        self.cfg = cfg
        self.generation = 0  # bumped by `clear` (DELETE /v1/memories): renders started before it are discarded

    def clear(self) -> None:
        self.generation += 1
        super().clear()

    def _render(self, frames, p: Path, ffmpeg: str, fps: int, max_width: int) -> None:  # type: ignore[override]
        render_montage(list(frames), p, ffmpeg=ffmpeg, cfg=self.cfg)


def plan_key(title: str, clips: list[ClipSpec], cfg: MontageSettings) -> str:
    raw = repr((title, [c.key() for c in clips], cfg.width, cfg.height, cfg.fps))
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


async def _timelapse_frames(store, t: MemoryToken, n: int) -> list[MemoryToken]:
    lo = t.ts if t.kind == "frame" else t.ts - TIMELAPSE_LOOKBACK_S
    toks = await store.range(lo, t.ts + TIMELAPSE_LOOKAHEAD_S, limit=n * 4, device_id=t.device_id, kinds=("frame",))
    if t.kind == "frame" and t.thumb_path and all(x.id != t.id for x in toks):
        toks = [t, *toks]
    return [x for x in toks if x.thumb_path and Path(x.thumb_path).is_file()][:n]


async def build_clip(
    st, t: MemoryToken, title: str, duration: float, cfg: MontageSettings, zone: ZoneInfo, *, use_replay=True
):
    """Clip for one moment: stored real video if it covers the clip, else buffered replay frames if any, else a
    timelapse of stored keyframes. `use_replay=False` (face blur on) skips both unblurred sources."""
    dur = min(max(float(duration or cfg.clip_s), 2.5), 6.0)
    video = getattr(st, "video", None) if use_replay else None
    if video is not None:
        t0 = t.ts - cfg.pre_s
        segs = video.segments(t.device_id, t0, t0 + dur)
        if segs and covered_s(segs) >= st.video_settings.min_coverage * dur:
            return ClipSpec(t, title, _clock(t.ts, zone), (), 0.0, "video", video=tuple(segs))
    buf = getattr(st, "replay", None) if use_replay else None
    if buf is not None:
        t0 = t.ts - cfg.pre_s
        frames = []
        for f in buf.frames(t.device_id, t0, t0 + dur):
            try:
                frames.append((f.ts, f.read()))
            except FileNotFoundError:  # spooled frame evicted meanwhile
                continue
        if len(frames) >= 2:
            tail = max(t0 + dur - frames[-1][0], 0.1)
            return ClipSpec(t, title, _clock(t.ts, zone), tuple(frames), tail, "replay")
    step = max(cfg.timelapse_step_s, 0.2)
    toks = await _timelapse_frames(st.store, t, max(1, round(dur / step)))
    frames = []
    for x in toks:
        try:
            frames.append((len(frames) * step, Path(x.thumb_path).read_bytes()))
        except OSError:
            continue
    if not frames:
        return None
    return ClipSpec(t, title, _clock(t.ts, zone), tuple(frames), step, "timelapse")


async def candidates(st, query: str, t0: float | None, t1: float | None, device_id: str | None, n: int):
    n = max(1, n)
    if query:
        res = await st.retriever.retrieve(query, k=n, time_from=t0, time_to=t1, device_id=device_id)
        toks = res.memory_tokens()
    else:
        kinds = ("frame", "transcript")
        toks = await st.store.range(t0, t1, limit=RANGE_SCAN_LIMIT, device_id=device_id, kinds=kinds)
        if len(toks) >= RANGE_SCAN_LIMIT:  # `range` is oldest-first: sample the whole span, not just its start
            lo, hi = toks[0].ts, t1 if t1 is not None else time.time()
            per = max(1, RANGE_SCAN_LIMIT // n)
            toks = []
            for i in range(n):
                a, b = lo + (hi - lo) * i / n, lo + (hi - lo) * (i + 1) / n
                part = await st.store.range(a, b, limit=per, device_id=device_id, kinds=kinds)
                toks += [x for x in part if x.ts < b or i == n - 1]
    toks = sorted({t.id: t for t in toks}.values(), key=lambda t: t.ts)
    return evenly(toks, n)


# ---- HTTP ----


def _ts_param(v: float | str | None, name: str) -> float | None:
    return parse_ts(None if v is None else str(v), name)


@router.post("/montage", response_model=MontageResponse)
async def create_montage(request: Request, body: MontageRequest):
    st = request.app.state
    cache: MontageCache | None = getattr(st, "montage_cache", None)
    if cache is None:
        raise HTTPException(404, "montage is disabled (MONTAGE_ENABLED=false)")
    cfg: MontageSettings = st.montage_settings
    query = (body.query or "").strip()
    t0, t1 = _ts_param(body.time_from, "time_from"), _ts_param(body.time_to, "time_to")
    if not query and t0 is None and t1 is None:
        raise HTTPException(400, "give a query, a time range (time_from/time_to), or both")
    if t0 is not None and t1 is not None and t1 < t0:
        raise HTTPException(400, "time_to must be >= time_from")
    replay_cfg = getattr(st, "replay_settings", None)
    ffmpeg = find_ffmpeg(getattr(replay_cfg, "ffmpeg", "") or "")
    if ffmpeg is None:
        raise HTTPException(503, FFMPEG_HINT)

    gen = cache.generation
    subj = smart.select_mode(body.mode or "", query, cfg.smart, tz=st.settings.wearer_tz)
    if subj is not None:
        return await _smart_montage(st, cache, cfg, ffmpeg, gen, body, subj, t0, t1)
    cands = await candidates(st, query, t0, t1, body.device_id or None, cfg.max_candidates)
    if not cands:
        raise HTTPException(404, "no memories match that query / time range")
    n_clips = min(body.max_clips or cfg.max_clips, max(cfg.max_clips, 1))
    request_text = query or "highlights of this period"
    plan, planner = await st.montage_planner.plan(request_text, cands, n_clips)

    labels = _labels(cands)
    moments = [(labels[c.label], c.title, c.duration) for c in plan.clips]
    clips = await _build_clips(st, cfg, moments)
    title = _short(plan.title, 60) or "Your highlights"
    return await _render_response(st, cache, cfg, ffmpeg, gen, clips, title, plan.narration, planner)


def _use_replay(st) -> bool:
    # buffered replay frames are the raw upload; with face blur on only the (blurred) stored thumbnails are used
    return not (st.settings.privacy_blur_faces or st.settings.privacy_blur_faces_vlm)


async def _build_clips(st, cfg: MontageSettings, moments: list[tuple[MemoryToken, str, float]]) -> list[ClipSpec]:
    zone = _zone(st.settings.wearer_tz)
    clips: list[ClipSpec] = []
    for t, title, duration in moments:
        spec = await build_clip(st, t, _short(title, 60), duration, cfg, zone, use_replay=_use_replay(st))
        if spec is not None:
            clips.append(spec)
    if not clips:
        raise HTTPException(404, "no stored video, buffered frames or stored thumbnails for the chosen moments")
    return clips


async def _render_response(st, cache, cfg, ffmpeg, gen, clips, title, narration, planner, **extra) -> MontageResponse:
    key = plan_key(title, clips, cfg)
    try:
        await cache.get_or_render(key, clips, ffmpeg=ffmpeg, fps=cfg.fps, max_width=cfg.width)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as e:
        if cache.generation != gen:  # a purge deleted a stored video clip mid-render: same answer as below
            raise HTTPException(409, "memories were deleted while the montage was being made; try again") from None
        raise HTTPException(500, f"montage render failed: {e}") from None
    if cache.generation != gen:  # DELETE /v1/memories ran meanwhile: the reel may show deleted moments
        cache.path(key).unlink(missing_ok=True)
        raise HTTPException(409, "memories were deleted while the montage was being made; try again")
    return MontageResponse(
        id=key,
        url=f"/v1/montages/{key}.mp4",
        title=title,
        narration=narration,
        planner=planner,
        clips=[
            MontageClip(
                token_id=c.token.id,
                ts=c.token.ts,
                title=c.title,
                duration_s=round(c.duration, 3),
                source=c.source,
                thumb_url=c.token.thumb_url,
            )
            for c in clips
        ],
        **extra,
    )


async def _smart_montage(st, cache, cfg, ffmpeg, gen, body: MontageRequest, subj, t0, t1) -> MontageResponse:
    """Content-filtered montage (`smart_montage.py`): gather -> verify -> one clip per matching moment."""
    cands = await smart.gather(st, subj, t0, t1, body.device_id or None, cfg)
    verifier = getattr(st, "montage_verifier", None)
    stats = {"verify_errors": 0}
    if verifier is not None:
        verdicts, stats = await verifier.verify(subj.phrase, cands, lambda t: _load_image(st, t))
    else:
        verdicts = [None] * len(cands)
    picks = smart.keep_matches(cands, verdicts, cfg.verify_min_confidence)
    if not picks:
        if stats.get("verify_errors"):  # some batches failed, the checked ones found nothing: not a real "no"
            raise HTTPException(503, smart.unfinished_message(subj.phrase))
        ranged = t0 is not None or t1 is not None or subj.timed
        raise HTTPException(404, smart.no_match_message(subj.phrase, len(cands), ranged))
    cap = max(cfg.smart_max_clips, 1)
    chosen, n_moments = smart.pick_moments(picks, min(body.max_clips or cap, cap), cfg.moment_gap_s)
    zone = _zone(st.settings.wearer_tz)
    moments = []
    for p in chosen:
        caption = "" if p.token.caption.lstrip().startswith("(") else p.token.caption
        title = p.title or caption or p.token.ocr_text or f"Moment at {_clock(p.token.ts, zone)}"
        moments.append((p.token, title, cfg.clip_s))
    clips = await _build_clips(st, cfg, moments)
    verified = any(p.verified for p in picks)
    days = len({datetime.fromtimestamp(p.token.ts, zone).date() for p in picks})
    phrase = subj.phrase
    said = smart.spoken(phrase)
    title = _short(said[:1].upper() + said[1:], 60)
    return await _render_response(
        st,
        cache,
        cfg,
        ffmpeg,
        gen,
        clips,
        title,
        smart.narration(phrase, n_moments, len(clips), days, verified),
        "claude" if verified else "offline",
        mode="smart",
        subject=phrase,
        candidates=len(cands),
        matches=n_moments if verified else 0,
    )


async def _load_image(st, t: MemoryToken) -> bytes | None:
    """JPEG for the verifier: any media tier via `MediaStore.read` (stored thumbnails are already blurred)."""
    media = getattr(st, "media", None)
    if media is not None:
        return await media.read(t)
    try:
        return Path(t.thumb_path).read_bytes() if t.has_image else None
    except OSError:
        return None


@router.api_route("/montages/{montage_id}.mp4", methods=["GET", "HEAD"])
async def get_montage(request: Request, montage_id: str):
    cache: MontageCache | None = getattr(request.app.state, "montage_cache", None)
    if cache is None or not _ID_RE.fullmatch(montage_id):
        raise HTTPException(404)
    p = cache.path(montage_id)
    if not p.is_file():
        raise HTTPException(404, "unknown or expired montage; POST /v1/montage again")
    return ranged_file_response(p, request, "video/mp4", {"Cache-Control": "private, max-age=300"})


class _PurgeMirror:
    """Pure-ASGI middleware: after a successful `DELETE /v1/memories`, drop every rendered montage."""

    def __init__(self, app, state) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "DELETE" or scope["path"].rstrip("/") != "/v1/memories":
            return await self.app(scope, receive, send)
        status: list[int] = []

        async def _send(msg):
            if msg["type"] == "http.response.start":
                status.append(msg["status"])
            await send(msg)

        await self.app(scope, receive, _send)
        if status[:1] == [200]:
            for name in ("montage_cache", "collage_cache"):
                if (cache := getattr(self.state, name, None)) is not None:
                    cache.clear()


def install(app: FastAPI, settings, montage_settings: MontageSettings | None = None) -> None:
    """Attach the montage endpoints to `app` (call from `create_app`). No filesystem side effects until used."""
    cfg = montage_settings or MontageSettings()
    st = app.state
    st.montage_settings = cfg
    st.montage_cache = None
    if cfg.enabled:
        st.montage_cache = MontageCache(cfg.cache_dir or Path(settings.thumb_dir).parent / "montages", cfg)
        st.montage_planner = MontagePlanner(settings, cfg)
        st.montage_verifier = smart.MontageVerifier(settings, cfg)
    app.include_router(router, prefix="/v1")
    app.add_middleware(_PurgeMirror, state=st)
