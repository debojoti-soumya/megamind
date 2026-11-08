"""Live short-term memory: answer "what am I looking at right now / what did I just see" from the raw feed.

Enrichment is async (caption VLM + embeddings), so the last few seconds are often not in the store
yet when the wearer asks about the present. For questions about now / just now (rule-based, see
`is_now_question`), `LiveRetriever` wraps the configured retriever and appends a few extra
evidence items to its result:

* the newest raw frames (default 3, >= 1 s apart) from the moment-replay ring buffer (#58), and
* the newest ambient-audio transcript tokens (#63) captured within ~30 s,

labelled ``NOW-1`` (most recent) .. ``NOW-n`` in their caption and marked live (``reasons=["live"]``).
Items already in the retrieved set are skipped (transcripts by id, frames by device + ts). The
reader sees them like any other token (header + caption + image), so the `Reader` interface and the
`RecallMemory` hand-off are unchanged; their ids are in `all_retrieved`, so citing them is valid.

Ported from the design of Recall's `live.py` (LiveBuffer) and the "LIVE / just now" block in its
`answer.py`. Opt-out with ``LIVE_MEMORY_ENABLED=false``; other knobs are ``LIVE_MEMORY_*``.

Raw frames are written (downscaled, face-blurred when PRIVACY_BLUR_FACES* is on) to a small bounded
cache directory so the reader can read them by path, and are served at ``GET /v1/live/{id}.jpg``
(the citation's ``thumb_url``) only while the frame is still in the replay buffer.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import io
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image
from pydantic_settings import BaseSettings, SettingsConfigDict

from megamind.models import MemoryToken
from megamind.retrieval.base import RetrievalResult, RetrievedToken, RetrieveMode, Retriever

log = logging.getLogger("megamind.live")

LIVE_REASON = "live"


class LiveMemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIVE_MEMORY_", env_file=".env", extra="ignore")

    enabled: bool = True
    frames: int = 3  # newest raw frames shown as NOW items ...
    frame_gap_s: float = 1.0  # ... at least this far apart (the buffer holds ~10 fps of near-duplicates)
    frame_window_s: float = 30.0  # ... and within this long before the newest frame
    max_age_s: float = 120.0  # the newest frame must be this fresh vs. `now`, else it's not "now" (0 = no check)
    transcripts: int = 2  # newest ambient-audio transcript tokens ...
    transcript_window_s: float = 30.0  # ... captured within this long before `now`
    dedupe_s: float = 0.25  # a raw frame this close to an already-retrieved frame token of the device is skipped
    cache_dir: Path | None = None  # default <thumb_dir>/../live
    cache_max_files: int = 64
    max_px: int = 640  # downscale cached frames (the reader downsizes to READER_THUMB_MAX_PX anyway)


# ---- detection ----

_NOW_RE = re.compile(
    r"\b(?:right now|just now|currently|at the moment|at this moment|"
    r"a (?:moment|second|sec|few seconds) ago|moments ago|seconds ago|in front of me|"
    r"current (?:view|scene|surroundings)|"
    r"(?:am i|i'm|i am|are we|we're) (?:looking at|seeing|holding|reading|watching|facing)|"
    r"what (?:do|can) i see|(?:did|have) i just|i just (?:saw|heard|read|said|put|left|picked|did)|"
    r"(?:what|who) just|what(?:'s| is) happening)\b"
    r"|\bnow\s*[?.!]*$",
    re.I,
)

# Raw question of the current chat turn (set by the orchestrator). Claude rewrites the question into
# search keywords, so "what am I looking at right now" may reach the retriever as "current view".
live_question: contextvars.ContextVar[str | None] = contextvars.ContextVar("live_question", default=None)


def is_now_question(text: str | None) -> bool:
    """Rule-based: is this about the present / the last few seconds?"""
    return bool(text) and _NOW_RE.search(text.strip()) is not None


# ---- tokens ----


class LiveFrameToken(MemoryToken):
    """A raw buffered frame posing as a memory token for one recall. Never stored."""

    @property
    def thumb_url(self) -> str:
        return f"/v1/live/{self.id}.jpg"


@dataclass
class _Cached:
    device_id: str
    ts: float
    path: Path


def _frame_id(device_id: str, ts: float) -> str:
    h = hashlib.sha1(f"{device_id}\0{ts:.6f}".encode()).hexdigest()[:10]
    return f"now_{int(ts * 1000)}_{h}"


def _ago(seconds: float) -> str:
    s = max(0.0, seconds)
    return f"{int(round(s))} s ago" if s < 90 else f"{int(round(s / 60))} min ago"


def _downscale(jpeg: bytes, max_px: int) -> bytes:
    try:
        with Image.open(io.BytesIO(jpeg)) as im:
            if max(im.size) <= max_px:
                return jpeg
            im = im.convert("RGB")
            im.thumbnail((max_px, max_px))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except (OSError, ValueError):
        return jpeg


class LiveMemory:
    """Builds NOW items from the replay buffer + store; owns the small frame cache served at /v1/live."""

    def __init__(self, state, cfg: LiveMemorySettings, cache_dir: Path, face_blur=None) -> None:
        self.state = state  # app.state: .replay (ReplayBuffer | None), .store
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.face_blur = face_blur
        self._cache: OrderedDict[str, _Cached] = OrderedDict()

    # -- frame cache --

    def _alive(self, c: _Cached) -> bool:
        buf = getattr(self.state, "replay", None)
        return buf is not None and bool(buf.frames(c.device_id, c.ts, c.ts))

    def lookup(self, frame_id: str) -> Path | None:
        """Path of a cached live frame, if its source frame is still buffered (purges propagate)."""
        c = self._cache.get(frame_id)
        if c is None:
            return None
        if not self._alive(c) or not c.path.is_file():
            self._drop(frame_id)
            return None
        return c.path

    def _drop(self, frame_id: str) -> None:
        c = self._cache.pop(frame_id, None)
        if c is not None:
            c.path.unlink(missing_ok=True)

    async def _materialize(self, device_id: str, frame) -> tuple[str, Path] | None:
        fid = _frame_id(device_id, frame.ts)
        if (c := self._cache.get(fid)) is not None and c.path.is_file():
            self._cache.move_to_end(fid)
            return fid, c.path
        try:
            jpeg = _downscale(frame.read(), self.cfg.max_px)
            if self.face_blur is not None:
                jpeg, _ = await self.face_blur.apply(jpeg)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            p = self.cache_dir / f"{fid}.jpg"
            await asyncio.to_thread(p.write_bytes, jpeg)
        except Exception:  # noqa: BLE001  (live evidence is best-effort; never break recall)
            log.exception("live memory: could not cache frame %s", fid)
            return None
        self._cache[fid] = _Cached(device_id, frame.ts, p)
        while len(self._cache) > max(1, self.cfg.cache_max_files):
            self._drop(next(iter(self._cache)))
        return fid, p

    # -- NOW items --

    def _pick_frames(self, device_id: str | None, now: float):
        buf = getattr(self.state, "replay", None)
        if buf is None or self.cfg.frames <= 0:
            return None, []
        if device_id is None:  # the most recently fed device
            stats = buf.stats()
            device_id = max(stats, key=lambda d: stats[d]["to"]) if stats else None
        newest = buf.newest_ts(device_id) if device_id is not None else None
        if newest is None or (self.cfg.max_age_s > 0 and now - newest > self.cfg.max_age_s):
            return device_id, []
        picked = []
        for f in reversed(buf.frames(device_id, newest - self.cfg.frame_window_s, newest)):
            if not picked or picked[-1].ts - f.ts >= self.cfg.frame_gap_s:
                picked.append(f)
                if len(picked) >= self.cfg.frames:
                    break
        return device_id, picked  # newest first

    async def now_items(self, retrieved: list[RetrievedToken], device_id: str | None, now: float) -> list[MemoryToken]:
        """NOW tokens, newest first and labelled NOW-1.., excluding anything already in `retrieved`."""
        have_ids = {r.token.id for r in retrieved}
        have_frames = [(r.token.device_id, r.token.ts) for r in retrieved if r.token.kind == "frame"]
        items: list[MemoryToken] = []

        dev, frames = self._pick_frames(device_id, now)
        for f in frames:
            if _frame_id(dev, f.ts) in have_ids or any(
                d == dev and abs(ts - f.ts) <= self.cfg.dedupe_s for d, ts in have_frames
            ):
                continue
            got = await self._materialize(dev, f)
            if got is None:
                continue
            fid, path = got
            items.append(
                LiveFrameToken(id=fid, device_id=dev, ts=f.ts, thumb_path=str(path), frame_hash="", kind="frame")
            )

        store = getattr(self.state, "store", None)
        if store is not None and self.cfg.transcripts > 0:
            try:
                rows = await store.range(
                    now - self.cfg.transcript_window_s, now + 5.0, 200, device_id=device_id, kinds=["transcript"]
                )
            except Exception:  # noqa: BLE001
                log.exception("live memory: transcript lookup failed")
                rows = []
            fresh = [t for t in rows if t.id not in have_ids][-self.cfg.transcripts :]
            items.extend(fresh)

        items.sort(key=lambda t: t.ts, reverse=True)
        out = []
        for i, t in enumerate(items, 1):
            if isinstance(t, LiveFrameToken):
                cap = (
                    f"LIVE NOW-{i}: raw camera frame from {_ago(now - t.ts)} (live feed, may not be in memory yet); "
                    "the wearer's current view, look at the image"
                )
            else:
                cap = f"LIVE NOW-{i} ({_ago(now - t.ts)}): {t.caption}"
            out.append(t.model_copy(update={"caption": cap}))
        return out


class LiveRetriever(Retriever):
    """Delegates to `inner`; for now-questions, appends NOW items (score above every retrieved hit)."""

    def __init__(self, inner: Retriever, live: LiveMemory) -> None:
        self.inner = inner
        self.live = live

    def __getattr__(self, name):  # expose the wrapped retriever's extras (cfg, store, ...)
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 12,
        mode: RetrieveMode = "hybrid",
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        now: datetime | None = None,
    ) -> RetrievalResult:
        res = await self.inner.retrieve(
            query, k=k, mode=mode, time_from=time_from, time_to=time_to, device_id=device_id, now=now
        )
        if not (is_now_question(query) or is_now_question(live_question.get())):
            return res
        now_ts = now.timestamp() if now is not None else time.time()
        try:
            items = await self.live.now_items(res.tokens, device_id, now_ts)
        except Exception:  # noqa: BLE001
            log.exception("live memory: skipped")
            return res
        if not items:
            return res
        top = max((r.score for r in res.tokens), default=0.0)
        # NOW-1 ranks first; RetrievalResult.tokens stays chronological
        added = [RetrievedToken(t, top + 1.0 + (len(items) - i) * 1e-3, [LIVE_REASON]) for i, t in enumerate(items)]
        res.tokens = sorted([*res.tokens, *added], key=lambda r: r.token.ts)
        log.info("live memory: +%d NOW item(s) for %r", len(added), query)
        return res


# ---- wiring ----

router = APIRouter()


@router.get("/live/{frame_id}.jpg")
async def get_live_frame(request: Request, frame_id: str):
    live: LiveMemory | None = getattr(request.app.state, "live_memory", None)
    p = live.lookup(frame_id) if live is not None else None
    if p is None:
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


def install(app: FastAPI, settings, live_settings: LiveMemorySettings | None = None) -> None:
    """Call from `create_app`: the GET /v1/live route + settings. `wrap_retriever` does the rest in lifespan."""
    st = app.state
    st.live_settings = live_settings or LiveMemorySettings()
    st.live_memory = None
    app.include_router(router, prefix="/v1")


def wrap_retriever(state, settings, retriever: Retriever) -> Retriever:
    """Lifespan hook: wrap `retriever` with live short-term memory unless LIVE_MEMORY_ENABLED=false."""
    cfg: LiveMemorySettings = getattr(state, "live_settings", None) or LiveMemorySettings()
    if not cfg.enabled:
        return retriever
    from megamind.ingest.privacy import build_face_blur

    cache_dir = cfg.cache_dir or Path(settings.thumb_dir).parent / "live"
    state.live_memory = LiveMemory(state, cfg, cache_dir, face_blur=build_face_blur(settings))
    return LiveRetriever(retriever, state.live_memory)


__all__ = [
    "LIVE_REASON",
    "LiveFrameToken",
    "LiveMemory",
    "LiveMemorySettings",
    "LiveRetriever",
    "install",
    "is_now_question",
    "live_question",
    "wrap_retriever",
]
