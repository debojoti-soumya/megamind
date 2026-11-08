"""Ambient audio memory: POST /v1/audio -> timestamped transcript -> kind="transcript" memory tokens.

Speech around the wearer (names, plans) becomes searchable text alongside the visual tokens: each
~`audio_window_s` window of transcript segments is one token whose `ts` is the window's absolute start
and whose `caption` carries the words, so `/v1/memories` and `recall_memory` find it like any other token.
Transcription is synchronous (the response lists the new token ids); no thumbnail, no image vector.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from ulid import ULID

from megamind.agent.stt import Segment, wav_duration
from megamind.ingest.transcribe import ensure_wav
from megamind.models import AudioAccepted, AudioMeta, MemoryToken

log = logging.getLogger(__name__)
router = APIRouter()

CAPTION_PREFIX = "Speech heard nearby: "
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # OpenAI transcription upload cap; ~13 min of 16 kHz mono s16


def merge_segments(segments: list[Segment], window_s: float) -> list[Segment]:
    """Group consecutive segments into windows spanning at most `window_s` (a longer single segment stays whole)."""
    out: list[Segment] = []
    cur: list[Segment] = []
    for s in sorted((s for s in segments if s.text.strip()), key=lambda s: s.start):
        if cur and s.end - cur[0].start > window_s:
            out.append(Segment(cur[0].start, max(c.end for c in cur), " ".join(c.text.strip() for c in cur)))
            cur = []
        cur.append(s)
    if cur:
        out.append(Segment(cur[0].start, max(c.end for c in cur), " ".join(c.text.strip() for c in cur)))
    return out


@router.post("/audio", response_model=AudioAccepted)
async def post_audio(request: Request, audio: UploadFile = File(...), meta: str = Form(...)):
    app = request.app.state
    try:
        am = AudioMeta.model_validate(json.loads(meta))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"bad meta: {exc}") from exc

    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, f"audio chunk larger than {MAX_AUDIO_BYTES} bytes; send shorter chunks")
    # WAV passes through; providers needing 16 kHz (faster-whisper) resample + band-pass themselves
    wav = await asyncio.to_thread(ensure_wav, data)
    if wav is None:
        raise HTTPException(415, "audio is not a WAV and ffmpeg is unavailable or could not decode it")

    audio_id = str(ULID())
    try:
        segments = await app.audio_stt.transcribe_segments(wav)
    except Exception as exc:  # noqa: BLE001
        log.exception("transcription failed for %s", audio_id)
        raise HTTPException(502, f"transcription failed: {exc}") from exc

    windows = merge_segments(segments, app.settings.audio_window_s)
    tokens: list[MemoryToken] = []
    for w in windows:
        caption = f'{CAPTION_PREFIX}"{w.text.strip()}"'
        tokens.append(
            MemoryToken(
                id=str(ULID()),
                device_id=am.device_id,
                ts=am.ts_start + max(0.0, w.start),
                caption=caption,
                entities=["speech"],
                pose=am.pose,
                text_vec=await app.text_embedder.embed(caption),
                thumb_path="",
                frame_hash="",
                kind="transcript",
            )
        )
    if tokens:
        await app.store.add_many(tokens)
    return AudioAccepted(
        audio_id=audio_id,
        duration_s=wav_duration(wav) or 0.0,
        transcript=" ".join(w.text.strip() for w in windows),
        token_ids=[t.id for t in tokens],
    )
