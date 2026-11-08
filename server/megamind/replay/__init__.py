"""Moment replay: keep every received frame for a short window and serve citations as mp4 clips.

Wiring (`install`) is additive: a frame tap on `POST /v1/frames`, the `GET /v1/replay.mp4` router,
and a middleware that mirrors `DELETE /v1/memories` onto the buffer. `annotate_citations` fills the
optional `replay_url` on chat citations. Settings come from `REPLAY_*` env vars (see `.env.example`).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlencode

from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from megamind.models import Citation
from megamind.replay.buffer import BufferedFrame, ReplayBuffer
from megamind.replay.render import ReplayCache, find_ffmpeg, render_mp4
from megamind.replay.routes import parse_range, parse_ts, router

__all__ = [
    "BufferedFrame",
    "ReplayBuffer",
    "ReplayCache",
    "ReplaySettings",
    "annotate_citations",
    "find_ffmpeg",
    "install",
    "parse_range",
    "render_mp4",
    "replay_url",
]


class ReplaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REPLAY_", env_file=".env", extra="ignore")

    enabled: bool = True
    window_s: float = 900.0  # keep ~15 min per device ...
    max_frames: int = 9000  # ... bounded by count ...
    max_bytes: int = 256 << 20  # ... and bytes (per device)
    spool_dir: Path | None = None  # set to keep frames on disk instead of RAM (survives restarts)
    cache_dir: Path | None = None  # rendered mp4s; default <thumb_dir>/../replay
    cache_max_files: int = 32
    max_span_s: float = 300.0
    pad_before_s: float = 5.0  # citation replay_url window around the cited token's ts
    pad_after_s: float = 5.0
    out_fps: int = 10
    max_width: int = 960
    ffmpeg: str = ""  # explicit ffmpeg binary; default PATH, then imageio-ffmpeg


def replay_url(device_id: str, t0: float, t1: float) -> str:
    return "/v1/replay.mp4?" + urlencode({"device_id": device_id, "from": f"{t0:.3f}", "to": f"{t1:.3f}"})


def annotate_citations(state, cited: list[Citation], device_id: str | None = None) -> None:
    """Set `replay_url` (in place) on citations that have buffered frames around their ts; else leave null."""
    buf: ReplayBuffer | None = getattr(state, "replay", None)
    if buf is None:
        return
    cfg: ReplaySettings = state.replay_settings
    for i, c in enumerate(cited):
        if c.replay_url:
            continue
        t0, t1 = c.ts - cfg.pad_before_s, c.ts + cfg.pad_after_s
        dev = buf.resolve_device(t0, t1, device_id)
        if dev is not None:
            cited[i] = c.model_copy(update={"replay_url": replay_url(dev, t0, t1)})


class _PurgeMirror:
    """Pure-ASGI middleware: after a successful `DELETE /v1/memories`, purge the same window from replay."""

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
        buf = getattr(self.state, "replay", None)
        if buf is None or status[:1] != [200]:
            return
        qs = {k: v[-1] for k, v in parse_qs(scope.get("query_string", b"").decode()).items()}
        try:
            since = parse_ts(qs.get("since"), "since")
        except Exception:  # noqa: BLE001  (route already answered 400 in that case)
            return
        buf.purge(qs.get("device_id") or None, since_ts=since)
        if cache := getattr(self.state, "replay_cache", None):
            cache.clear()


def install(app: FastAPI, settings, replay_settings: ReplaySettings | None = None) -> None:
    """Attach the replay buffer to `app` (call from `create_app`). No filesystem side effects until used."""
    cfg = replay_settings or ReplaySettings()
    st = app.state
    st.replay_settings = cfg
    st.replay = None
    if cfg.enabled:
        st.replay = ReplayBuffer(cfg.window_s, cfg.max_frames, cfg.max_bytes, spool_dir=cfg.spool_dir)
        st.replay_cache = ReplayCache(cfg.cache_dir or Path(settings.thumb_dir).parent / "replay", cfg.cache_max_files)
        st.frame_taps = [*getattr(st, "frame_taps", []), st.replay.tap]
    app.include_router(router, prefix="/v1")
    app.add_middleware(_PurgeMirror, state=st)
