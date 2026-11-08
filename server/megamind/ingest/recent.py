"""`GET /v1/memories/recent`: the newest memories, no query needed (Memory Theater cards, debug).

`GET /v1/memories` is a ranked search and (correctly) drops hits with no overlap with `q`, so it cannot list
"what happened lately". This read-only route uses the store's public `range()` primitive instead; it does not
touch retrieval or ranking.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Query, Request

from megamind.models import MemoryHit, MemoryToken, SearchResponse

log = logging.getLogger(__name__)
router = APIRouter()

# Look back in widening windows so a busy day doesn't load every token just to return the newest k.
_WINDOWS_S = (900.0, 3 * 3600.0, 24 * 3600.0, 7 * 86400.0, 30 * 86400.0, 365 * 86400.0, None)
_RANGE_CAP = 1000  # tokens per range() page (range() is oldest-first and loads full tokens, so keep pages small)
_MAX_PAGES = 50  # bound on pages walked forward within one window (pathological stores only)


async def _newest_since(store, since: float | None, k: int, device_id: str | None, kinds: list[str]):
    """The newest `k` tokens with `ts >= since`. `range()` is oldest-first and capped, so when a page is full
    we page forward from its last timestamp instead of trusting it (a full page holds the *oldest* tokens)."""
    page = await store.range(since, None, _RANGE_CAP, device_id=device_id, kinds=kinds)
    tail: dict[str, MemoryToken] = {t.id: t for t in page}
    pages = 1
    while len(page) >= _RANGE_CAP:
        cursor = page[-1].ts
        if pages >= _MAX_PAGES:
            log.warning("memories/recent: stopped after %d pages; result may miss the newest tokens", pages)
            break
        page = await store.range(cursor, None, _RANGE_CAP, device_id=device_id, kinds=kinds)
        pages += 1
        if not page or page[-1].ts == cursor:  # > cap tokens share one timestamp: cannot advance
            break
        for t in page:
            tail[t.id] = t
        if len(tail) > k + _RANGE_CAP:  # keep memory bounded: only the newest ones can win
            tail = {t.id: t for t in _sorted_desc(tail.values())[:k]}
    return _sorted_desc(tail.values())[:k]


def _sorted_desc(tokens):
    return sorted(tokens, key=lambda t: (t.ts, t.id), reverse=True)


@router.get("/memories/recent", response_model=SearchResponse)
async def recent_memories(
    request: Request,
    k: int = Query(24, ge=1, le=200),
    device_id: str | None = None,
    kind: list[str] = Query(default=["frame"]),  # repeatable: &kind=frame&kind=transcript
):
    store = request.app.state.store
    now = time.time()
    newest: list[MemoryToken] = []
    for window in _WINDOWS_S:
        since = None if window is None else now - window
        newest = await _newest_since(store, since, k, device_id, kind)
        if len(newest) >= k:
            break
    return SearchResponse(results=[MemoryHit.from_token(t, 0.0) for t in newest])
