from __future__ import annotations

import io
import json
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image
from ulid import ULID

from megamind.models import (
    EpisodeListResponse,
    FrameAccepted,
    FrameMeta,
    MemoryToken,
    SearchMode,
    SearchResponse,
)

router = APIRouter()


def _iso_to_ts(s: str | None) -> float | None:
    return datetime.fromisoformat(s).timestamp() if s else None


@router.post("/frames", response_model=FrameAccepted, status_code=202)
async def post_frame(request: Request, image: UploadFile = File(...), meta: str = Form(...)):
    app = request.app.state
    try:
        fm = FrameMeta.model_validate(json.loads(meta))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"bad meta: {exc}") from exc

    jpeg = await image.read()
    try:
        img = Image.open(io.BytesIO(jpeg))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "image is not a decodable JPEG") from exc

    frame_id = str(ULID())
    filt = app.keyframe_filter
    if not filt.has_state(fm.device_id):
        filt.seed(fm.device_id, await app.store.recent_hashes(fm.device_id, fm.ts - app.settings.filter_dhash_window_s))
    decision = filt.decide(img, fm, can_merge=app.worker.can_merge(fm.device_id))
    if decision.reason == "merged":
        await app.worker.attach_transcript(fm.device_id, fm.transcript or "")
    if not decision.accepted:
        return FrameAccepted(frame_id=frame_id, accepted=False, reason=decision.reason)

    thumb_path = app.settings.thumb_dir / f"{frame_id}.jpg"
    thumb_path.write_bytes(jpeg)

    token = MemoryToken(
        id=frame_id,
        device_id=fm.device_id,
        ts=fm.ts,
        pose=fm.pose,
        thumb_path=str(thumb_path),
        frame_hash=decision.frame_hash,
        source=fm.source or "quest",
        trigger=decision.trigger,
        sharpness=decision.sharpness,
        transcript=fm.transcript or "",
    )
    await app.worker.submit(token, jpeg)
    return FrameAccepted(frame_id=frame_id, accepted=True)


@router.get("/memories", response_model=SearchResponse)
async def search_memories(
    request: Request,
    q: str,
    k: int = 12,
    mode: SearchMode = "hybrid",
    device_id: str | None = None,
):
    app = request.app.state
    params = request.query_params
    qvec = await app.text_embedder.embed_query(q)
    ivec = await app.image_embedder.embed_text(q) if app.image_embedder else None
    hits = await app.store.search(
        qvec,
        ivec,
        q,
        k=k,
        mode=mode,
        device_id=device_id,
        time_from=_iso_to_ts(params.get("from")),
        time_to=_iso_to_ts(params.get("to")),
    )
    return SearchResponse(results=hits)


@router.get("/episodes", response_model=EpisodeListResponse)
async def list_episodes(
    request: Request,
    q: str | None = None,
    k: int = 20,
    device_id: str | None = None,
):
    params = request.query_params
    eps = await request.app.state.list_episodes(
        time_from=_iso_to_ts(params.get("from")),
        time_to=_iso_to_ts(params.get("to")),
        query=q,
        k=k,
        device_id=device_id,
    )
    return EpisodeListResponse(episodes=eps)


@router.delete("/memories")
async def purge_memories(request: Request, device_id: str | None = None):
    app = request.app.state
    await app.media.delete_files(await app.store.list_tokens(device_id=device_id))
    n = await app.store.purge(device_id)
    request.app.state.keyframe_filter.reset(device_id)
    request.app.state.episodes.reset(device_id)
    request.app.state.summaries.reset(device_id)
    return {"deleted": n}


@router.get("/thumbs/{token_id}.jpg")
async def get_thumb(request: Request, token_id: str):
    app = request.app.state
    token = await app.store.get(token_id)
    data = await app.media.read(token) if token else None
    if data is None:
        raise HTTPException(404)
    return Response(data, media_type="image/jpeg")


@router.post("/admin/age")
async def age_now(request: Request, hot_older_than_s: float | None = None, cold_older_than_s: float | None = None):
    """Run pixel aging now (demo/ops). Defaults: HOT_RETENTION_H / COLD_RETENTION_DAYS."""
    return await request.app.state.media.age(hot_s=hot_older_than_s, cold_s=cold_older_than_s)


@router.get("/stats")
async def stats(request: Request):
    app = request.app.state
    tiers = await app.store.tier_counts()
    n = sum(tiers.values())
    disk = app.media.disk_bytes()
    counts = dict(app.keyframe_filter.counts)
    seen = sum(counts.values())
    kept = sum(counts.get(t, 0) for t in ("novel", "pose", "heartbeat", "forced", "speech"))
    hot_pf = disk["hot_jpeg"] / max(1, tiers.get("hot", 0))
    cold_pf = disk["cold_chunks"] / max(1, tiers.get("cold", 0)) if tiers.get("cold") else hot_pf / 8
    # small stores are dominated by LanceDB fixed overhead; below 1k rows use an analytic estimate
    # (fp16 vectors + ~1 KB text/metadata) so the 100-day projection isn't inflated
    ie = app.image_embedder
    est_index_pf = 2 * (app.text_embedder.dim + (ie.dim if ie else 0)) + 1024
    index_pf = disk["index"] / n if n >= 1000 else est_index_pf
    per_day = app.settings.filter_budget_per_min * 60 * 16
    hot_days = app.settings.hot_retention_h / 24
    proj = per_day * (hot_days * hot_pf + (100 - hot_days) * cold_pf + 100 * index_pf)
    return {
        "tokens": n,
        "tiers": tiers,
        "filter": {"seen": seen, "kept": kept, "keep_rate": kept / seen if seen else None, "by_reason": counts},
        "worker_queue": app.worker.queue.qsize(),
        "disk_bytes": disk,
        "bytes_per_frame": {"hot": round(hot_pf), "cold": round(cold_pf), "index": round(index_pf)},
        "projected_100d_gb_at_budget": round(proj / 1e9, 2),
        "episodes": len(await app.store.list_episodes(limit=10**9)),
    }
