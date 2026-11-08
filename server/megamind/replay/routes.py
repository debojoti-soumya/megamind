"""`GET /v1/replay.mp4`: the buffered frames in [from, to] as an H.264 mp4, with HTTP Range support."""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from megamind.replay.buffer import BufferedFrame
from megamind.replay.keyframes import kept_frames, kept_tokens
from megamind.replay.render import FFMPEG_HINT, find_ffmpeg

router = APIRouter()

DEFAULT_SPAN_S = 30.0
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeNotSatisfiable(Exception):
    pass


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Single `bytes=` range -> inclusive (start, end). None = ignore the header and send the whole body
    (absent, malformed, multi-range, or last < first, per RFC 9110). Raises RangeNotSatisfiable (-> 416)."""
    if not header or not (m := _RANGE.fullmatch(header.strip())):
        return None
    a, b = m.groups()
    if a == "" and b == "":
        return None
    if a == "":  # suffix range: last N bytes
        n = int(b)
        if n == 0 or size == 0:
            raise RangeNotSatisfiable
        return max(size - n, 0), size - 1
    start = int(a)
    if b and int(b) < start:
        return None
    if start >= size:
        raise RangeNotSatisfiable
    return start, min(int(b), size - 1) if b else size - 1


def ranged_file_response(p: Path, request: Request, media_type: str, headers: dict[str, str] | None = None):
    """Ported from Recall's `video()` handler: 200 + Accept-Ranges, 206 for a range, 416 if unsatisfiable.
    Both bodies are streamed here (not FileResponse) so our parse_range semantics apply, not Starlette's."""
    size = p.stat().st_size
    headers = {"Accept-Ranges": "bytes", **(headers or {})}
    try:
        rng = parse_range(request.headers.get("range"), size)
    except RangeNotSatisfiable:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    start, end = rng if rng is not None else (0, size - 1)

    def body(chunk: int = 1 << 20):
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                data = f.read(min(chunk, left))
                if not data:
                    break
                left -= len(data)
                yield data

    headers["Content-Length"] = str(end - start + 1)
    if rng is None:
        return StreamingResponse(body(), media_type=media_type, headers=headers)
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(body(), status_code=206, media_type=media_type, headers=headers)


def parse_ts(s: str | None, param: str) -> float | None:
    """Unix seconds or ISO-8601 (naive = server local time); nan/inf are rejected."""
    if s is None or s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        try:
            v = datetime.fromisoformat(s).timestamp()
        except ValueError:
            v = math.nan
    if not math.isfinite(v):
        raise HTTPException(400, f"{param}: expected ISO-8601 or unix seconds, got {s!r}")
    return v


@router.api_route("/replay.mp4", methods=["GET", "HEAD"])  # HEAD: some video players probe first
async def replay_mp4(
    request: Request,
    device_id: str | None = None,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
):
    st = request.app.state
    buf = getattr(st, "replay", None)
    if buf is None:
        raise HTTPException(404, "replay is disabled (REPLAY_ENABLED=false)")
    cfg = st.replay_settings
    t1 = parse_ts(to, "to")
    t0 = parse_ts(from_, "from")
    if t1 is None:
        t1 = buf.newest_ts(device_id) if t0 is None else t0 + DEFAULT_SPAN_S
        if t0 is None and (video := getattr(st, "video", None)) is not None:
            # newest footage of either kind: stored real video (megamind.video) can end after the newest frame
            t1 = max((t for t in (t1, video.newest_ts(device_id)) if t is not None), default=None)
        if t1 is None:
            raise HTTPException(404, "no buffered frames")
    if t0 is None:
        t0 = t1 - DEFAULT_SPAN_S
    if t1 < t0:
        raise HTTPException(400, "to must be >= from")
    if t1 - t0 > cfg.max_span_s:
        raise HTTPException(400, f"span {t1 - t0:.0f} s exceeds REPLAY_MAX_SPAN_S={cfg.max_span_s:.0f}")

    dev = buf.resolve_device(t0, t1, device_id)
    frames = buf.frames(dev, t0, t1) if dev is not None else []
    source, tag, toks = "buffer", "", []
    if not frames:  # older than the ring buffer: fall back to the kept keyframes (hot or cold tier)
        dev, toks = await kept_tokens(st, t0, t1, device_id)
    if (video_replay := getattr(st, "video_replay", None)) is not None:
        # real video from the headset (megamind.video) wins when it covers the window; else frames as before
        frame_ts = [f.ts for f in frames] or [t.ts for t in toks]
        if (cut := await video_replay(st, device_id or dev, t0, t1, frame_ts)) is not None:
            return ranged_file_response(cut[0], request, "video/mp4", cut[1])
    if not frames:
        if not toks:
            raise HTTPException(
                404, f"no buffered or kept frames for device={device_id or '*'} in [{t0:.3f}, {t1:.3f}]"
            )
        source = "keyframes"
        tag = "kf|" + hashlib.sha1(",".join(t.id for t in toks).encode()).hexdigest()[:12] + "|"
        # placeholders for the cache key and headers; pixels are read below on a cache miss. Their path never
        # exists, so a clip pruned between the check and the render reads as "evicted; retry" (404), not a crash.
        gone = st.replay_cache.dir / ".unread"
        frames = [BufferedFrame(ts=t.ts, size=0, path=gone) for t in toks]
    ffmpeg = find_ffmpeg(cfg.ffmpeg)
    if ffmpeg is None:
        raise HTTPException(503, FFMPEG_HINT)

    raw = f"{tag}{dev}|{frames[0].ts:.6f}|{frames[-1].ts:.6f}|{len(frames)}|{cfg.out_fps}|{cfg.max_width}"
    key = hashlib.sha1(raw.encode()).hexdigest()[:20]
    if toks and not st.replay_cache.path(key).exists():  # read (decode, for cold frames) only on a cache miss
        frames = await kept_frames(st, toks)
        if not frames:
            raise HTTPException(404, "kept frames are no longer readable")
    try:
        p = await st.replay_cache.get_or_render(key, frames, ffmpeg=ffmpeg, fps=cfg.out_fps, max_width=cfg.max_width)
    except FileNotFoundError:
        raise HTTPException(404, "buffered frames were evicted; retry") from None
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        raise HTTPException(500, f"replay render failed: {e}") from None
    headers = {
        "X-Replay-Device": dev,
        "X-Replay-Start": f"{frames[0].ts:.3f}",  # unix ts at mp4 offset 0
        "X-Replay-Frames": str(len(frames)),
        "X-Replay-Source": source,  # buffer = every received frame; keyframes = only the frames the filter kept
        "Cache-Control": "private, max-age=300",
    }
    return ranged_file_response(p, request, "video/mp4", headers)
