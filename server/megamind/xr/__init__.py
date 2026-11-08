"""Memory Theater: a static WebXR page at `GET /xr` (Quest 3 browser, or any WebGL browser as a 3D preview).

Plain ES modules under `static/`, no build step; three.js loads from a pinned CDN via the page's importmap.
The page only calls the public v1 API (`/v1/memories`, `/v1/chat`, `/v1/thumbs/*`, `/v1/replay.mp4`), so it
adds no server-side behaviour beyond serving files. Wiring (`install`) is additive; `XR_ENABLED=false` turns it off.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings, SettingsConfigDict

STATIC_DIR = Path(__file__).parent / "static"

__all__ = ["STATIC_DIR", "XRSettings", "install"]


class XRSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="XR_", env_file=".env", extra="ignore")

    enabled: bool = True


def install(app: FastAPI, xr_settings: XRSettings | None = None) -> None:
    """Serve the page at `/xr` (and `/xr/`) and its assets under `/xr/static/` (call from `create_app`)."""
    if not (xr_settings or XRSettings()).enabled:
        return
    index = STATIC_DIR / "index.html"

    @app.get("/xr", include_in_schema=False)
    @app.get("/xr/", include_in_schema=False)
    async def xr_page() -> FileResponse:
        return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})

    app.mount("/xr/static", StaticFiles(directory=STATIC_DIR), name="xr-static")
