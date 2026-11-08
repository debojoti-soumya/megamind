"""`GET /v1/episodes`: the wearer's day as a list of episodes ("what did I do this morning"), newest first.

Read-only over existing tokens: frame `episode_id`s give each episode's span and size, and the consolidator's
`kind="episode"` summary token (when written) gives its summary, entities and representative thumbnail.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from megamind.ingest.router import _iso_to_ts
from megamind.models import EpisodeHit, EpisodeListResponse

router = APIRouter()


@router.get("/episodes", response_model=EpisodeListResponse)
async def list_episodes(request: Request, k: int = Query(50, ge=1, le=500), device_id: str | None = None):
    params = request.query_params
    scan = dict(
        time_from=_iso_to_ts(params.get("from"), "from"),
        time_to=_iso_to_ts(params.get("to"), "to"),
        device_id=device_id,
        vectors=False,
    )
    store = request.app.state.store
    eps: dict[str, EpisodeHit] = {}
    for t in await store.list_tokens(kinds=("frame",), **scan):
        if not t.episode_id:
            continue
        e = eps.get(t.episode_id)
        if e is None:
            eps[t.episode_id] = EpisodeHit(
                episode_id=t.episode_id, device_id=t.device_id, start_ts=t.ts, end_ts=t.ts, n_frames=1
            )
        else:
            e.start_ts, e.end_ts, e.n_frames = min(e.start_ts, t.ts), max(e.end_ts, t.ts), e.n_frames + 1
    for s in await store.list_tokens(kinds=("episode",), **scan):
        if not s.episode_id:
            continue
        e = eps.setdefault(
            s.episode_id,
            EpisodeHit(episode_id=s.episode_id, device_id=s.device_id, start_ts=s.ts, end_ts=s.ts, n_frames=0),
        )
        e.summary, e.entities, e.thumb_url, e.summary_id = s.caption, s.entities, s.thumb_url, s.id
    newest = sorted(eps.values(), key=lambda e: (e.start_ts, e.episode_id), reverse=True)
    return EpisodeListResponse(episodes=newest[:k])
