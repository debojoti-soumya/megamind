from __future__ import annotations

import io
import json
import logging
import math
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from ulid import ULID

from megamind.ingest.worker import purge_all
from megamind.models import FrameAccepted, FrameMeta, MemoryToken, PurgeResponse, SearchMode, SearchResponse

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
    qvec = await app.text_embedder.embed(q)
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
    p = request.app.state.settings.thumb_dir / f"{token_id}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")
