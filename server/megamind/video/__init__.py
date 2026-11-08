"""Real video from the headset: `POST /v1/video` stores short H.264 clips; replay and montage prefer them.

Ported from Recall (capture-app/quest `VideoRecorder.kt` + the backend's `video` chunk part), adapted to
Megamind's separate-endpoint ingest: the native Quest app (`headset-android/`) uploads each chunk's 30 fps
H.264 mp4 next to its 1 fps JPEGs (`/v1/frames`, which still drive memory) and its audio (`/v1/audio`).

Wiring (`install`) is additive: the `POST /v1/video` router, `app.state.video` (a `VideoStore`), and
`app.state.video_replay`, the hook `GET /v1/replay.mp4` calls before its frame-based render. Without stored
video covering a window, replay and montage behave exactly as before. Real video is never blurred, so while
`PRIVACY_BLUR_FACES(_VLM)` is on uploads are not kept and nothing is served from it. Settings: `VIDEO_*`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from megamind.video.render import render_video, timeline_s
from megamind.video.replay import plan_video, video_replay
from megamind.video.routes import router
from megamind.video.store import Segment, VideoClip, VideoStore, covered_s, privacy_blocks_video, probe

__all__ = [
    "Segment",
    "VideoClip",
    "VideoSettings",
    "VideoStore",
    "build_video_store",
    "covered_s",
    "install",
    "plan_video",
    "privacy_blocks_video",
    "probe",
    "render_video",
    "timeline_s",
    "video_replay",
]


class VideoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIDEO_", env_file=".env", extra="ignore")

    enabled: bool = True
    store_dir: Path | None = None  # default <thumb_dir>/../video
    retention_h: float | None = None  # None -> HOT_RETENTION_H (real video is hot media, #108)
    max_bytes: int = 20 << 30  # whole store; oldest clips are dropped beyond it
    max_clip_bytes: int = 32 << 20  # per upload (a 5 s Quest clip is ~2 MB)
    max_clip_s: float = 60.0  # per upload
    min_coverage: float = 0.5  # replay/montage use video when it covers >= this share of the window
    replay_fps: int = 30  # cap for replay clips cut from real video (montage uses MONTAGE_FPS)

    @field_validator("store_dir", "retention_h", mode="before")
    @classmethod
    def _blank_is_unset(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v


def build_video_store(settings, cfg: VideoSettings) -> VideoStore:
    root = cfg.store_dir or Path(settings.thumb_dir).parent / "video"
    hours = cfg.retention_h if cfg.retention_h is not None else settings.hot_retention_h
    return VideoStore(root, retention_s=float(hours) * 3600.0, max_bytes=cfg.max_bytes)


def install(app: FastAPI, settings, video_settings: VideoSettings | None = None) -> None:
    """Attach `POST /v1/video` and the replay hook to `app` (call from `create_app`, after replay)."""
    cfg = video_settings or VideoSettings()
    st = app.state
    st.video_settings = cfg
    st.video = build_video_store(settings, cfg) if cfg.enabled else None
    st.video_replay = video_replay if cfg.enabled else None  # hooks for megamind.replay (no import cycle)
    st.video_plan = plan_video if cfg.enabled else None
    app.include_router(router, prefix="/v1")
