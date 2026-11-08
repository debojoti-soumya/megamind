"""Browser prototype of the stream-3 AR chat panel, served at /ui.

Emulates the headset on a laptop: webcam as passthrough, hold-Space push-to-talk,
transcript + cited thumbnails, TTS playback, status HUD and optional 1 fps capture.
Talks to the server only through docs/api.md, exactly like the Unity client.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()
_HERE = Path(__file__).parent


@router.get("/ui", include_in_schema=False)
async def ui_index():
    return FileResponse(_HERE / "index.html", media_type="text/html")


@router.get("/ui/3d", include_in_schema=False)
async def ui_model3d():
    """3D memory viewer (three.js + WebXR): /ui/3d?m={id}&f=glb|ply, from `POST /v1/model3d`'s `viewer_url`."""
    return FileResponse(_HERE / "model3d.html", media_type="text/html")
