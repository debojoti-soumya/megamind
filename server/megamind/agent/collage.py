"""Photo montage ("collage"): collect the stored photos that show something and lay them out as one JPEG.

"Show me pictures of the red car" / "make a collage of every coffee I had today": unlike the mp4 montage
(`montage.py`) this is a single still image, cheap enough to make inside a chat turn and to show on the AR panel.

1. Subject: `smart_montage.detect_subject` (framing and time phrase stripped). A query with only a time phrase
   ("photos from this afternoon") samples that period instead (unverified).
2. Candidates: `smart_montage.gather` (retriever across history, spread in time), checked by the montage
   verifier (Claude Haiku vision, `MONTAGE_VERIFY_*`) when an API key is set; unverified retrieval hits otherwise.
3. One photo per moment (`COLLAGE_MOMENT_GAP_S`), chronological, at most `COLLAGE_MAX_IMAGES`.
4. Render with Pillow: a title header and a grid of tiles, each with its local time and a short caption.

`POST /v1/collage` and the Claude chat tool `make_collage` (`MAKE_COLLAGE_TOOL`, wired by `main.py`) share
`make_collage`. `GET /v1/collages/{id}.jpg` serves the result; `DELETE /v1/memories` clears the cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic_settings import BaseSettings, SettingsConfigDict

from megamind.agent import smart_montage as smart
from megamind.agent.montage import _FONTS, _clock, _load_image, _short, _ts_param, _zone, candidates, evenly
from megamind.models import CollageImage, CollageRequest, CollageResponse, MemoryToken
from megamind.retrieval.planner import parse_time

log = logging.getLogger("megamind.collage")
router = APIRouter()
_ID_RE = re.compile(r"[0-9a-f]{20}")

MAKE_COLLAGE_TOOL = {
    "name": "make_collage",
    "description": (
        "Collect photos from the wearer's visual memory that show something and arrange them into one photo "
        "montage that is displayed to the wearer on their panel. Use it when the user asks to see pictures, "
        "photos or images of something, for a collage or photo montage, or to 'show me all the times' they saw "
        "something. Returns which moments made it into the montage."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What every photo should show, as a short phrase (e.g. 'the red car', 'my dog').",
            },
            "time_from": {"type": ["string", "null"], "description": "ISO-8601 lower bound or null."},
            "time_to": {"type": ["string", "null"], "description": "ISO-8601 upper bound or null."},
        },
        "required": ["query"],
    },
}

COLLAGE_PROMPT = """

You also have make_collage, which gathers the wearer's photos of something into one photo montage shown on
their panel. Use it instead of recall_memory when they ask to see pictures, photos or images of something,
a collage or montage, or all the times they saw something. Then say in one sentence what the montage shows
(how many moments, roughly when); don't describe each photo. If it finds nothing, say so."""

# chat requests that want photos rather than an answer (used to skip the recall prefetch / offline fallback)
_WANTS = re.compile(
    r"\b(collages?|photo\s*montages?|image\s*montages?|montages?|contact\s+sheet"
    r"|(pictures?|photos?|pics|images?|snapshots?)\s+(of|with|from|showing|where)"
    r"|show\s+me\s+(all\s+)?(the\s+)?(pictures?|photos?|pics|images?|snapshots?))\b",
    re.I,
)


def wants_collage(text: str) -> bool:
    return bool(_WANTS.search(text or ""))


class CollageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COLLAGE_", env_file=".env", extra="ignore")

    enabled: bool = True
    agent_tool: bool = True  # offer make_collage to the chat agent
    max_images: int = 9
    moment_gap_s: float = 60.0  # matching photos closer than this (same device) are one moment -> one tile
    tile_px: int = 360  # nominal tile width (rows ~3/4 of it tall); photos keep their own aspect ratio
    quality: int = 85
    font: str = ""
    cache_dir: Path | None = None  # default <thumb_dir>/../collages
    cache_max_files: int = 32


class CollageError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


class CollageCache:
    def __init__(self, cache_dir: Path, max_files: int) -> None:
        self.dir, self.max_files = Path(cache_dir), max(1, max_files)
        self.generation = 0  # bumped by `clear` (DELETE /v1/memories) so in-flight renders are dropped

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.jpg"

    def put(self, key: str, data: bytes) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.path(key)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        files = sorted(self.dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
        for old in files[self.max_files :]:
            old.unlink(missing_ok=True)
        return p

    def clear(self) -> None:
        self.generation += 1
        for f in self.dir.glob("*.jpg") if self.dir.is_dir() else []:
            f.unlink(missing_ok=True)


# ---- render ----


@dataclass(frozen=True)
class Tile:
    token: MemoryToken
    title: str
    jpeg: bytes


def _font(cfg: CollageSettings, size: int):
    for f in [cfg.font, *_FONTS]:
        if f and Path(f).is_file():
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def grid(n: int) -> tuple[int, int]:
    """(columns, rows) for `n` tiles: 1, 2x1, 2x2, 3x2, 3x3, 4x3 ..."""
    cols = 1 if n <= 1 else 2 if n <= 4 else 3 if n <= 9 else 4
    return cols, math.ceil(max(n, 1) / cols)


def _aspect(img: Image.Image) -> float:
    return min(max(img.width / max(img.height, 1), 0.6), 2.2)  # very tall/wide photos are center-cropped to this


def justify(aspects: list[float], width: float, row_h: float, gap: float) -> list[list[int]]:
    """Split photos (in order) into rows that each fill `width` at roughly `row_h` (a justified gallery): the
    row count follows from the total aspect, the split balances the rows' aspect sums (linear partition)."""
    n = len(aspects)
    if n == 0:
        return []
    k = max(1, min(n, round(sum(aspects) * row_h / max(width, 1.0) + 0.25)))  # lean to more, taller rows
    pre = [0.0]
    for a in aspects:
        pre.append(pre[-1] + a)
    target = pre[-1] / k
    inf = float("inf")
    cost = [[inf] * (n + 1) for _ in range(k + 1)]
    back = [[0] * (n + 1) for _ in range(k + 1)]
    cost[0][0] = 0.0
    for r in range(1, k + 1):
        for j in range(r, n + 1):
            for i in range(r - 1, j):
                c = cost[r - 1][i] + (pre[j] - pre[i] - target) ** 2
                if c < cost[r][j]:
                    cost[r][j], back[r][j] = c, i
    rows, j = [], n
    for r in range(k, 0, -1):
        i = back[r][j]
        rows.append(list(range(i, j)))
        j = i
    return rows[::-1]


def _rounded(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=radius, fill=255)
    out = img.convert("RGBA")
    out.putalpha(mask)
    return out


def _shade(w: int, h: int) -> Image.Image:
    """Bottom caption gradient: transparent -> ~70% black."""
    col = Image.linear_gradient("L").resize((1, h)).point(lambda v: int(v * 0.7))
    over = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    over.putalpha(col.resize((w, h)))
    return over


def render_collage(tiles: list[Tile], title: str, subtitle: str, cfg: CollageSettings, tz: str) -> bytes:
    """Justified gallery: every photo keeps its shape (no 4:3 crop), rows fill the width, rounded corners and a
    gradient caption (local time + what it shows) on each photo."""
    zone = _zone(tz)
    cols, _ = grid(len(tiles))
    tw = max(120, cfg.tile_px)
    gap, margin = max(8, tw // 36), max(16, tw // 16)
    width = cols * tw + (cols - 1) * gap
    row_h = tw * 3 / 4 if len(tiles) > 1 else tw * 1.2
    imgs = []
    for t in tiles:
        try:
            imgs.append(ImageOps.exif_transpose(Image.open(io.BytesIO(t.jpeg))).convert("RGB"))
        except (OSError, ValueError):
            imgs.append(Image.new("RGB", (4, 3), (40, 44, 50)))
    aspects = [_aspect(im) for im in imgs]
    rows = justify(aspects, width, row_h, gap)
    if len(tiles) == 1:
        width = round(min(row_h * aspects[0], tw * 2.2))
    layout, y = [], 0
    for row in rows:
        total = sum(aspects[i] for i in row)
        h = (width - gap * (len(row) - 1)) / total
        h = min(h, row_h * (1.6 if len(row) > 1 else 1.15))  # a short last row doesn't blow its photo up
        x = (width - (h * total + gap * (len(row) - 1))) / 2
        for i in row:
            w = h * aspects[i]
            layout.append((i, round(x), round(y), round(x + w) - round(x), round(h)))
            x += w + gap
        y += h + gap
    grid_h = round(y - gap)

    f_title = _font(cfg, max(22, tw // 11))
    f_sub, f_when, f_cap = _font(cfg, max(13, tw // 24)), _font(cfg, max(12, tw // 26)), _font(cfg, max(12, tw // 24))
    header = f_title.size + f_sub.size + 22
    w_canvas, h_canvas = width + 2 * margin, margin * 2 + header + grid_h
    canvas = Image.new("RGB", (w_canvas, h_canvas), (14, 17, 22))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((margin, margin - 4), _fit_text(draw, title, f_title, width), font=f_title, fill=(240, 243, 247))
    unverified = subtitle.endswith(" · unverified")
    sub = subtitle.removesuffix(" · unverified")
    sy = margin + f_title.size + 8
    draw.text((margin, sy), _fit_text(draw, sub, f_sub, width - 120), font=f_sub, fill=(148, 158, 172))
    if unverified:
        cx = margin + draw.textlength(sub, font=f_sub) + 12
        label = "not visually checked"
        cw = draw.textlength(label, font=f_when) + 16
        draw.rounded_rectangle((cx, sy - 2, cx + cw, sy + f_sub.size + 4), radius=8, fill=(255, 196, 0, 40))
        draw.text((cx + 8, sy + 1), label, font=f_when, fill=(255, 205, 90))

    radius = max(8, tw // 28)
    top = margin + header
    for i, x, y, w, h in layout:
        t, img = tiles[i], imgs[i]
        photo = ImageOps.fit(img, (max(w, 1), max(h, 1)), Image.Resampling.LANCZOS)
        band = min(h, f_when.size + f_cap.size + 28)
        photo = photo.convert("RGBA")
        photo.alpha_composite(_shade(w, band), (0, h - band))
        pd = ImageDraw.Draw(photo)
        when = datetime.fromtimestamp(t.token.ts, zone).strftime("%a %b %-d · %H:%M")
        pd.text((10, h - band + 8), when, font=f_when, fill=(255, 210, 110))
        pd.text(
            (10, h - band + 12 + f_when.size), _fit_text(pd, t.title, f_cap, w - 20), font=f_cap, fill=(245, 245, 245)
        )
        canvas.paste(_rounded(photo, radius), (margin + x, top + y), _rounded(photo, radius))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=max(30, min(cfg.quality, 95)), optimize=True)
    return buf.getvalue()


def collage_key(title: str, tiles: list[Tile], cfg: CollageSettings) -> str:
    raw = repr(("v2", title, [(t.token.id, t.title) for t in tiles], cfg.tile_px, cfg.quality))
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


# ---- make ----


def _tile_title(token: MemoryToken, verdict_title: str, zone) -> str:
    caption = "" if token.caption.lstrip().startswith("(") else token.caption
    return _short(verdict_title or caption or token.ocr_text or f"Moment at {_clock(token.ts, zone)}", 60)


def _forms(w: str) -> set[str]:
    """`w` and its singular guesses: "giraffes" -> {giraffes, giraffe, giraff}, "puppies" -> {..., puppy}."""
    out = {w}
    if len(w) > 3 and w.endswith("s"):
        out |= {w[:-1], w[:-2]} if w.endswith("es") else {w[:-1]}
        if w.endswith("ies"):
            out.add(w[:-3] + "y")
    return out


def _caption_matches(picks: list[smart.Pick], *, phrase: str) -> list[smart.Pick]:
    """Unverified picks whose caption names the subject ("dogs" ~ "a dog on the couch"): without the visual check,
    a vector-search neighbour that never mentions it is more likely noise than a photo of it. None when no caption
    does: retrieval always returns its nearest neighbours, so "elephants" would otherwise get any photos at all."""
    want = [_forms(w) for w in smart._content_words(phrase)]
    hits = []
    for p in picks:
        words = {f for w in re.findall(r"[a-z0-9']+", p.token.caption.lower()) for f in _forms(w)}
        if any(f & words for f in want):
            hits.append(p)
    return hits


async def make_collage(
    st,
    query: str,
    t0: float | None = None,
    t1: float | None = None,
    device_id: str | None = None,
    max_images: int | None = None,
) -> CollageResponse:
    """Gather -> verify -> one photo per moment -> render. Raises `CollageError` (404 nothing matches, 503 the
    verifier failed before confirming anything, 409 memories deleted meanwhile, 400 bad request)."""
    cache: CollageCache | None = getattr(st, "collage_cache", None)
    if cache is None:
        raise CollageError(404, "collage is disabled (COLLAGE_ENABLED=false)")
    cfg: CollageSettings = st.collage_settings
    mcfg = st.montage_settings
    tz = st.settings.wearer_tz
    zone = _zone(tz)
    query = " ".join((query or "").split())
    if not query:
        raise CollageError(400, "say what the photos should show")
    if t0 is not None and t1 is not None and t1 < t0:
        raise CollageError(400, "time_to must be >= time_from")
    cap = max(1, min(max_images or cfg.max_images, max(cfg.max_images, 1)))
    gen = cache.generation

    subj = smart.detect_subject(query, tz=tz)
    stats = {"verify_errors": 0}
    if subj is None:  # only a time phrase ("photos from this afternoon"): sample that period
        if t0 is None and t1 is None and (win := parse_time(query, datetime.now(zone))) is not None:
            t0, t1 = win.time_from, win.time_to
        if t0 is None and t1 is None:
            raise CollageError(400, "say what the photos should show, or give a time range")
        cands = [t for t in await candidates(st, "", t0, t1, device_id, mcfg.max_candidates) if t.has_image]
        picks = [smart.Pick(t, "", 0.0, False) for t in cands]
        phrase, title = "", "Your photos"
    else:
        cands = [t for t in await smart.gather(st, subj, t0, t1, device_id, mcfg) if t.has_image]
        verifier = getattr(st, "montage_verifier", None)
        if verifier is not None and cands:
            verdicts, stats = await verifier.verify(subj.phrase, cands, lambda t: _load_image(st, t))
        else:
            verdicts = [None] * len(cands)
        picks = smart.keep_matches(cands, verdicts, mcfg.verify_min_confidence)
        if picks and not any(p.verified for p in picks):
            picks = _caption_matches(picks, phrase=subj.phrase)
        phrase = subj.phrase
        said = smart.spoken(phrase)
        title = _short(said[:1].upper() + said[1:], 60)
    if not picks:
        if subj is not None and stats.get("verify_errors"):
            raise CollageError(503, smart.unfinished_message(phrase))
        ranged = t0 is not None or t1 is not None or bool(subj and subj.timed)
        where = " in that period" if ranged else " in your memories"
        what = f" of {smart.spoken(phrase)}" if phrase else ""
        raise CollageError(404, f"I couldn't find any photos{what}{where}.")
    if subj is None:
        chosen, n_moments = evenly(picks, cap), len(picks)
    else:
        chosen, n_moments = smart.pick_moments(picks, cap, cfg.moment_gap_s)

    tiles: list[Tile] = []
    for p in chosen:
        data = await _load_image(st, p.token)
        if data:
            tiles.append(Tile(p.token, _tile_title(p.token, p.title, zone), data))
    if not tiles:
        raise CollageError(404, "the matching moments have no stored photos left")
    verified = any(p.verified for p in picks)
    days = len({datetime.fromtimestamp(p.token.ts, zone).date() for p in picks})
    if phrase:
        narration = smart.narration(phrase, n_moments, len(tiles), days, verified)
    else:
        narration = f"Here are {len(tiles)} photo{'s' if len(tiles) != 1 else ''} from that period."
    first, last = (datetime.fromtimestamp(x.token.ts, zone) for x in (tiles[0], tiles[-1]))
    span = first.strftime("%a %b %-d %H:%M")
    if len(tiles) > 1:
        span += " – " + last.strftime("%H:%M" if last.date() == first.date() else "%a %b %-d %H:%M")
    subtitle = f"{len(tiles)} moment{'s' if len(tiles) != 1 else ''} · {span}" + ("" if verified else " · unverified")

    key = collage_key(title, tiles, cfg)
    p = cache.path(key)
    if not p.is_file():
        data = await asyncio.to_thread(render_collage, tiles, title, subtitle, cfg, tz)
        if cache.generation != gen:
            raise CollageError(409, "memories were deleted while the collage was being made; try again")
        p = await asyncio.to_thread(cache.put, key, data)
    with Image.open(p) as im:
        width, height = im.size
    log.info("collage %s: %r -> %d tiles (%d candidates, verified=%s)", key, query, len(tiles), len(cands), verified)
    return CollageResponse(
        id=key,
        url=f"/v1/collages/{key}.jpg",
        title=title,
        narration=narration,
        subject=phrase,
        verified=verified,
        candidates=len(cands),
        matches=n_moments,
        width=width,
        height=height,
        images=[
            CollageImage(token_id=t.token.id, ts=t.token.ts, title=t.title, thumb_url=t.token.thumb_url) for t in tiles
        ],
    )


# ---- HTTP ----


@router.post("/collage", response_model=CollageResponse)
async def create_collage(request: Request, body: CollageRequest):
    t0, t1 = _ts_param(body.time_from, "time_from"), _ts_param(body.time_to, "time_to")
    try:
        return await make_collage(request.app.state, body.query, t0, t1, body.device_id or None, body.max_images)
    except CollageError as e:
        raise HTTPException(e.status, e.detail) from None


@router.api_route("/collages/{collage_id}.jpg", methods=["GET", "HEAD"])
async def get_collage(request: Request, collage_id: str):
    cache: CollageCache | None = getattr(request.app.state, "collage_cache", None)
    if cache is None or not _ID_RE.fullmatch(collage_id):
        raise HTTPException(404)
    p = cache.path(collage_id)
    if not p.is_file():
        raise HTTPException(404, "unknown or expired collage; POST /v1/collage again")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


def install(app: FastAPI, settings, collage_settings: CollageSettings | None = None) -> None:
    """Attach the collage endpoints (after `montage.install`, whose verifier and settings it reuses)."""
    cfg = collage_settings or CollageSettings()
    st = app.state
    st.collage_settings = cfg
    st.collage_cache = None
    if cfg.enabled:
        st.collage_cache = CollageCache(
            cfg.cache_dir or Path(settings.thumb_dir).parent / "collages", cfg.cache_max_files
        )
        if getattr(st, "montage_verifier", None) is None:
            st.montage_verifier = smart.MontageVerifier(settings, st.montage_settings)
    app.include_router(router, prefix="/v1")


def agent_tool(st):
    """The chat agent's `make_collage` callable, or None when disabled."""
    cfg = getattr(st, "collage_settings", None)
    if getattr(st, "collage_cache", None) is None or cfg is None or not cfg.agent_tool:
        return None

    async def run(query: str, t0: float | None, t1: float | None, device_id: str | None) -> CollageResponse:
        return await make_collage(st, query, t0, t1, device_id)

    return run
