"""`GET /v1/memories/recent`: the newest memories, no query needed (Memory Theater cards, debug).

`GET /v1/memories` is a ranked search and (correctly) drops hits with no overlap with `q`, so it cannot list
"what happened lately". This read-only route uses the store's public `range()` primitive instead; it does not
touch retrieval or ranking.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Query, Request

from megamind.models import MemoryHit, SearchResponse

router = APIRouter()

# Look back in widening windows so a busy day doesn't load every token just to return the newest k.
_WINDOWS_S = (900.0, 3 * 3600.0, 24 * 3600.0, 7 * 86400.0, 30 * 86400.0, None)
_RANGE_CAP = 5000  # tokens fetched per window (range() is oldest-first, so the cap bounds the scan)


@router.get("/memories/recent", response_model=SearchResponse)
async def recent_memories(
    request: Request,
    k: int = Query(24, ge=1, le=200),
    device_id: str | None = None,
    kind: list[str] = Query(default=["frame"]),  # repeatable: &kind=frame&kind=transcript
):
    store = request.app.state.store
    now = time.time()
    tokens = []
    for window in _WINDOWS_S:
        since = None if window is None else now - window
        tokens = await store.range(since, None, _RANGE_CAP, device_id=device_id, kinds=kind)
        if len(tokens) >= k:
            break
    newest = sorted(tokens, key=lambda t: (t.ts, t.id), reverse=True)[:k]
    return SearchResponse(results=[MemoryHit.from_token(t, 0.0) for t in newest])
