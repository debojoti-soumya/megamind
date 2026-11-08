from __future__ import annotations

import asyncio
import io
import json
import logging
import math
from datetime import datetime
from typing import get_args

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from ulid import ULID

from megamind.ingest.audio import transcript_sink
from megamind.ingest.worker import purge_all
from megamind.models import FrameAccepted, FrameMeta, MemoryToken, PurgeResponse, SearchMode, SearchResponse
from megamind.store.media import run_aging

log = logging.getLogger(__name__)
router = APIRouter()


def _iso_to_ts(s: str | None, param: str = "from/to") -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        raise HTTPException(400, f"{param}: expected ISO-8601, got {s!r}") from None


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
    for tap in getattr(app, "frame_taps", ()):  # every decodable frame, before the filter (e.g. replay buffer)
        tap(fm, jpeg)
    if fm.transcript and fm.transcript.strip():
        # speech heard around this frame -> a kind="transcript" token, whatever the filter decides below
        await transcript_sink(app).add(fm.device_id, fm.ts, fm.ts, fm.transcript, fm.pose)

    frame_id = str(ULID())
    kf = app.keyframe_filter
    recent = await app.store.recent_hashes(fm.device_id, fm.ts - kf.history_window_s)
    recent += app.pending_hashes.get(fm.device_id, [])
    decision = kf.decide(img, fm, recent)
    if sum(kf.counts.values()) % 60 == 0:
        log.info("filter counts %s", dict(kf.counts))
    if not decision.accepted:
        return FrameAccepted(frame_id=frame_id, accepted=False, reason=decision.reason)

    app.pending_hashes.setdefault(fm.device_id, []).append((fm.ts, decision.frame_hash))
    app.pending_hashes[fm.device_id] = [
        (t, h) for t, h in app.pending_hashes[fm.device_id] if fm.ts - t <= kf.history_window_s
    ]

    thumb_path = app.settings.thumb_dir / f"{frame_id}.jpg"
    thumb_path.write_bytes(jpeg)

    token = MemoryToken(
        id=frame_id,
        device_id=fm.device_id,
        ts=fm.ts,
        pose=fm.pose,
        thumb_path=str(thumb_path),
        frame_hash=decision.frame_hash,
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
        time_from=_iso_to_ts(params.get("from"), "from"),
        time_to=_iso_to_ts(params.get("to"), "to"),
    )
    return SearchResponse(results=hits)


def _since_to_ts(s: str | None) -> float | None:
    """ISO-8601 (naive = server local time) or unix seconds. Absent → None; present-but-blank → 400."""
    if s is None:
        return None
    if not s.strip():
        raise HTTPException(400, "since: empty value; omit the parameter to purge all time")
    try:
        ts = float(s)
    except ValueError:
        return _iso_to_ts(s.strip(), "since")
    if not math.isfinite(ts):
        raise HTTPException(400, f"since: expected a finite unix timestamp, got {s!r}")
    return ts


@router.delete("/memories", response_model=PurgeResponse)
async def purge_memories(request: Request, device_id: str | None = None, since: str | None = None):
    n = await purge_all(request.app.state, device_id, since_ts=_since_to_ts(since))
    return PurgeResponse(deleted=n)


@router.get("/thumbs/{token_id}.jpg")
async def get_thumb(request: Request, token_id: str):
    """Any tier: hot JPEG, cold frame decoded from its AV1 chunk; 404 for text-tier / image-less tokens."""
    app = request.app.state
    p = app.settings.thumb_dir / f"{token_id}.jpg"
    if p.is_file():  # hot frames, episode thumbnails, frames still in the enrichment queue: no store lookup
        return FileResponse(p, media_type="image/jpeg")
    media = getattr(app, "media", None)
    token = await app.store.get(token_id) if media is not None else None
    data = await media.read(token) if token is not None else None  # cold: decoded from its AV1 chunk
    if data is None:
        raise HTTPException(404)
    return Response(data, media_type="image/jpeg")


@router.post("/admin/age")
async def age_media(request: Request, hot_older_than_s: float | None = None, cold_older_than_s: float | None = None):
    """Run one media aging pass now. Defaults: HOT_RETENTION_H / COLD_RETENTION_DAYS."""
    for name, v in (("hot_older_than_s", hot_older_than_s), ("cold_older_than_s", cold_older_than_s)):
        if v is not None and not (math.isfinite(v) and v >= 0):
            raise HTTPException(400, f"{name}: expected a finite number >= 0")
    return await run_aging(request.app.state, hot_s=hot_older_than_s, cold_s=cold_older_than_s)


_REJECT_REASONS = set(get_args(get_args(FrameAccepted.model_fields["reason"].annotation)[0]))
_PROJECTION_DAYS = 100
_WEARING_H_PER_DAY = 16


@router.get("/stats")
async def stats(request: Request):
    """Storage and ingest counters: frame tiers, filter decisions, queue depth, disk use, a 100-day projection."""
    app = request.app.state
    s = app.settings
    media = app.media
    tiers = await app.store.tier_counts()
    n_frames = sum(tiers.values())
    disk = await asyncio.to_thread(media.disk_bytes)  # walks thumbs/, chunks/ and the index: off the loop
    encoder = await media.probe_encoder()
    counts = dict(app.keyframe_filter.counts)
    kept = sum(v for k, v in counts.items() if k not in _REJECT_REASONS)
    seen = sum(counts.values())
    hot_pf = disk["hot_jpeg"] / tiers["hot"] if tiers["hot"] else None
    cold_pf = disk["cold_chunks"] / tiers["cold"] if tiers["cold"] else None
    n_rows = await app.store.count()
    index_pf = disk["index"] / n_rows if n_rows and disk["index"] else None
    proj = None
    if (hot_pf is not None or cold_pf is not None) and s.filter_max_per_min > 0:
        # at the frame budget, HOT_RETENTION_H as JPEG, the rest as AV1; whichever tier is not measured yet is
        # assumed to be 8x the other (JPEG vs AV1, PR #54's measurement)
        per_day = s.filter_max_per_min * 60 * _WEARING_H_PER_DAY
        hot_days = min(_PROJECTION_DAYS, s.hot_retention_h / 24)
        cold_days = min(_PROJECTION_DAYS, s.cold_retention_days) - hot_days
        hpf = hot_pf if hot_pf is not None else cold_pf * 8
        cpf = cold_pf if cold_pf is not None else hpf / 8
        proj = per_day * (hot_days * hpf + max(0.0, cold_days) * cpf + _PROJECTION_DAYS * (index_pf or 0.0))
    worker = app.worker
    return {
        "tokens": n_rows,
        "frames": n_frames,
        "tiers": tiers,
        "filter": {"seen": seen, "kept": kept, "keep_rate": kept / seen if seen else None, "by_reason": counts},
        "worker_queue": worker.queue.qsize(),
        "disk_bytes": disk,
        "bytes_per_frame": {
            "hot": round(hot_pf) if hot_pf is not None else None,
            "cold": round(cold_pf) if cold_pf is not None else None,
            "index": round(index_pf) if index_pf is not None else None,
        },
        "projected_100d_gb_at_budget": round(proj / 1e9, 2) if proj is not None else None,
        "media": {
            "encoder": (encoder or [None, None])[1],
            "hot_retention_h": s.hot_retention_h,
            "cold_retention_days": s.cold_retention_days,
            "cold_decodes": media.decodes,
        },
    }
