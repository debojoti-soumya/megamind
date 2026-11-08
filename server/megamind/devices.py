"""`GET /v1/devices` and the chat-side check for a `device_id` with no memories.

Chat only searches the memories of the request's `device_id`, so a client with a made-up id (the browser panel's
`browser-<random>`) silently gets "no memory" while the headset is streaming as `quest3-1`. `resolve_chat_device`
turns that into a `warning` (and, with `CHAT_DEFAULT_DEVICE=latest`, a fallback to the most recently active device).
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Request

from megamind.models import DevicesResponse

router = APIRouter()


@router.get("/devices", response_model=DevicesResponse)
async def list_devices(request: Request):
    return DevicesResponse(devices=await request.app.state.store.devices())


@dataclass
class ChatDevice:
    device_id: str | None  # the device to search (None = all devices)
    warning: str | None = None


async def resolve_chat_device(app, device_id: str | None) -> ChatDevice:
    """The device a chat turn searches. A requested device with no memories while other devices have some gets a
    warning; with `CHAT_DEFAULT_DEVICE=latest` the turn searches the most recently active device instead."""
    if device_id is None or await app.store.count(device_id) > 0:
        return ChatDevice(device_id)
    others = [d.device_id for d in await app.store.devices() if d.device_id != device_id]
    if not others:
        return ChatDevice(device_id)
    if app.settings.chat_default_device == "latest":
        return ChatDevice(others[0], f"device '{device_id}' has no memories; answered from '{others[0]}' instead")
    return ChatDevice(device_id, f"device '{device_id}' has no memories; active devices: {', '.join(others[:5])}")
