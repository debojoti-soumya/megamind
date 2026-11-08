"""Replay from real video: the hook `GET /v1/replay.mp4` calls (`app.state.video_replay`) before it renders frames.

Video wins when the stored clips cover at least `VIDEO_MIN_COVERAGE` of the window's footage extent (the span from
the earliest to the latest footage of either kind inside `[from, to]`); otherwise, or when cutting fails (clip aged
out meanwhile, ffmpeg error), the hook returns None and the frame-based render runs unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

from fastapi import HTTPException

from megamind.replay.render import FFMPEG_HINT, find_ffmpeg
from megamind.video.render import render_video, timeline_s
from megamind.video.store import Segment, covered_s, privacy_blocks_video

log = logging.getLogger(__name__)

FRAME_HOLD_S = 1.0  # a frame-based replay holds its last frame this long (render_mp4's tail)
MIN_VIDEO_S = 0.5  # less real video than this never wins


def prefer_video(
    segments: Sequence[Segment], frame_ts: Sequence[float], t0: float, t1: float, min_coverage: float
) -> bool:
    covered = covered_s(list(segments))
    if covered < MIN_VIDEO_S:
        return False
    if not frame_ts:
        return True
    lo = min(segments[0].start, max(min(frame_ts), t0))
    hi = max(segments[-1].end, min(max(frame_ts) + FRAME_HOLD_S, t1))
    return covered >= min_coverage * max(hi - lo, 1e-6)


def plan_video(
    state, device_id: str | None, t0: float, t1: float, frame_ts: Sequence[float]
) -> tuple[str, list[Segment]] | None:
    """(device, segments) when stored video should back the window, else None."""
    store = getattr(state, "video", None)
    if store is None or privacy_blocks_video(state.settings):
        return None
    dev = device_id or store.resolve_device(t0, t1)
    if dev is None:
        return None
    segs = store.segments(dev, t0, t1)
    if not segs or not prefer_video(segs, frame_ts, t0, t1, state.video_settings.min_coverage):
        return None
    return dev, segs


async def video_replay(state, device_id: str | None, t0: float, t1: float, frame_ts: Sequence[float]):
    """(mp4 path, headers) cut from stored real video for `[t0, t1]`, or None to fall back to frames."""
    plan = plan_video(state, device_id, t0, t1, frame_ts)
    if plan is None:
        return None
    dev, segs = plan
    replay_cfg = state.replay_settings
    ffmpeg = find_ffmpeg(replay_cfg.ffmpeg)
    if ffmpeg is None:
        raise HTTPException(503, FFMPEG_HINT)
    src_fps = max((s.clip.fps for s in segs), default=0.0)
    fps = max(1, min(state.video_settings.replay_fps, round(src_fps) or state.video_settings.replay_fps))
    raw = "vid|" + dev + "|" + ",".join(f"{s.clip.id}:{s.start:.3f}:{s.end:.3f}" for s in segs)
    key = hashlib.sha1(f"{raw}|{fps}|{replay_cfg.max_width}".encode()).hexdigest()[:20]

    def render(out: Path) -> None:
        if not all(s.clip.path.is_file() for s in segs):
            raise FileNotFoundError("a stored video clip was deleted before rendering")
        render_video(segs, out, ffmpeg=ffmpeg, fps=fps, max_width=replay_cfg.max_width)

    try:
        p = await state.replay_cache.get_or_render(
            key, segs, ffmpeg=ffmpeg, fps=fps, max_width=replay_cfg.max_width, render=render
        )
    except (TimeoutError, FileNotFoundError, RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        log.warning("video replay for %s [%.3f, %.3f] failed (%s); using frames", dev, t0, t1, e)
        return None
    headers = {
        "X-Replay-Device": dev,
        "X-Replay-Start": f"{segs[0].start:.3f}",
        "X-Replay-Frames": str(round(timeline_s(segs) * fps)),
        "X-Replay-Source": "video",
        "X-Replay-Clips": str(len({s.clip.id for s in segs})),
        "Cache-Control": "private, max-age=300",
    }
    return p, headers
