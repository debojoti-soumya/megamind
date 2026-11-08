"""Replay fallback: once the ring buffer no longer holds a window (default 15 min), render it from the
keyframes the filter kept instead, read through `MediaStore` so hot JPEGs and cold AV1 frames both work.
The result is keyframe-rate (about one frame every few seconds, each held until the next) rather than
every received frame, but `replay_url` stays usable for memories of any age until their pixels age out."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from megamind.models import MemoryToken
from megamind.replay.buffer import BufferedFrame

MAX_KEYFRAMES = 600  # REPLAY_MAX_SPAN_S=300 at the 20/min frame budget is 100


async def kept_tokens(state: Any, t0: float, t1: float, device_id: str | None = None) -> tuple[str | None, list]:
    """(device, frame tokens with pixels in [t0, t1]); without `device_id`, the device with the most."""
    store, media = getattr(state, "store", None), getattr(state, "media", None)
    if store is None or media is None:
        return None, []
    toks = await store.range(t0, t1, limit=MAX_KEYFRAMES, device_id=device_id, kinds=("frame",))
    toks = [t for t in toks if t.has_image]
    if not toks:
        return None, []
    dev = device_id or Counter(t.device_id for t in toks).most_common(1)[0][0]
    return dev, [t for t in toks if t.device_id == dev]


async def kept_frames(state: Any, tokens: list[MemoryToken]) -> list[BufferedFrame]:
    datas = await asyncio.gather(*(state.media.read(t) for t in tokens))
    return [BufferedFrame(ts=t.ts, size=len(d), data=d) for t, d in zip(tokens, datas) if d]
