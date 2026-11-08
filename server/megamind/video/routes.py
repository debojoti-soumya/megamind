"""`POST /v1/video`: store one short H.264 mp4 clip from the headset (validated with ffprobe)."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from ulid import ULID

from megamind.models import VideoAccepted, VideoMeta
from megamind.replay.render import FFMPEG_HINT, find_ffmpeg
from megamind.video.store import privacy_blocks_video, probe

log = logging.getLogger(__name__)
router = APIRouter()

MAX_SIDE_PX = 4096


@router.post("/video", response_model=VideoAccepted)
async def post_video(request: Request, video: UploadFile = File(...), meta: str = Form(...)):
    st = request.app.state
    store = getattr(st, "video", None)
    if store is None:
        raise HTTPException(404, "video is disabled (VIDEO_ENABLED=false)")
    cfg = st.video_settings
    try:
        vm = VideoMeta.model_validate(json.loads(meta))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"bad meta: {exc}") from exc
    if privacy_blocks_video(st.settings):  # never keep unblurred real video while face blurring is on
        return VideoAccepted(video_id="", accepted=False, reason="privacy", ts_start=vm.ts_start)

    too_big = f"video clip larger than {cfg.max_clip_bytes} bytes; send shorter clips"
    if video.size is not None and video.size > cfg.max_clip_bytes:
        raise HTTPException(413, too_big)
    seq = store.seq
    clip_id = str(ULID())
    staged = store.staging_path(vm.device_id, clip_id)
    try:
        size = 0
        with open(staged, "wb") as out:
            while data := await video.read(1 << 20):
                size += len(data)
                if size > cfg.max_clip_bytes:
                    raise HTTPException(413, too_big)
                out.write(data)
        if size == 0:
            raise HTTPException(400, "empty video")
        ffmpeg = find_ffmpeg(getattr(getattr(st, "replay_settings", None), "ffmpeg", "") or "")
        try:
            p = await asyncio.to_thread(probe, staged, ffmpeg=ffmpeg)
        except FileNotFoundError:
            raise HTTPException(503, FFMPEG_HINT) from None
        except (ValueError, subprocess.SubprocessError) as e:
            raise HTTPException(415, f"not a decodable mp4/mov video: {e}") from None
        if "mp4" not in p.format_name and "mov" not in p.format_name:
            raise HTTPException(415, f"expected an mp4 container, got {p.format_name or 'unknown'}")
        if p.codec != "h264":
            raise HTTPException(415, f"expected H.264 video, got {p.codec or 'unknown'}")
        if not (0 < p.width <= MAX_SIDE_PX and 0 < p.height <= MAX_SIDE_PX):
            raise HTTPException(415, f"unsupported frame size {p.width}x{p.height}")
        if p.duration <= 0:
            raise HTTPException(415, "video has no measurable duration")
        if p.duration > cfg.max_clip_s:
            raise HTTPException(413, f"video clip longer than {cfg.max_clip_s:g} s; send shorter clips")

        if (dup := store.find_duplicate(vm.device_id, vm.ts_start, p.duration)) is not None:
            return VideoAccepted(
                video_id=dup.id, accepted=True, reason="duplicate", ts_start=dup.ts_start,
                duration_s=round(dup.duration, 3), width=dup.width, height=dup.height,
            )  # fmt: skip
        if store.purged_since(seq, vm.device_id, vm.ts_start + p.duration):
            raise HTTPException(409, "memories were deleted while this clip was uploading; not kept")
        clip = store.commit(staged, clip_id, vm.device_id, vm.ts_start, p)
    finally:
        staged.unlink(missing_ok=True)  # no-op once committed (renamed)
    return VideoAccepted(
        video_id=clip.id, accepted=True, ts_start=clip.ts_start, duration_s=round(clip.duration, 3),
        width=clip.width, height=clip.height,
    )  # fmt: skip
