"""Ambient audio memory: POST /v1/audio -> timestamped transcript -> kind="transcript" memory tokens.

Speech around the wearer (names, plans) becomes searchable text alongside the visual tokens: each
~`audio_window_s` window of transcript segments is one token whose `ts` is the window's absolute start
and whose `caption` carries the words, so `/v1/memories` and `recall_memory` find it like any other token.
Transcription is synchronous (the response lists the token ids); no thumbnail, no image vector.

`TranscriptSink` is the one writer of transcript tokens (`/v1/audio` windows and `FrameMeta.transcript` text).
Speech that starts <= `speech_merge_s` after the device's last transcript token ends is appended to that token
(caption rebuilt, re-embedded, upserted by id) while the token spans <= `speech_max_token_s`, so an utterance cut
by a chunk boundary stays one searchable token. The merge target is kept in memory per device (no merge across a
restart) and cleared by purges. Merge rule ported from PR #54 (Devin), moved from frame tokens to transcripts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from ulid import ULID

from megamind.agent.stt import Segment, wav_duration
from megamind.ingest import transcribe
from megamind.ingest.enrich import TextEmbedder
from megamind.models import AudioAccepted, AudioMeta, MemoryToken, Pose
from megamind.store.base import MemoryStore

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


# per-token text bound: FrameMeta.transcript is client text (a form field can be ~1 MB), and a merged token grows
# with every piece, so cap what one token holds; past it, speech starts a new token
MAX_TOKEN_CHARS = 4000


def transcript_caption(text: str) -> str:
    return f'{CAPTION_PREFIX}"{text.strip()}"'


@dataclass
class _Open:
    token: MemoryToken
    end_ts: float
    pieces: list[str] = field(default_factory=list)


class TranscriptSink:
    """Builds, merges and stores `kind="transcript"` tokens. One lock serialises writes (and purges, see
    `purging`), so a merge can never re-insert a token a purge just deleted."""

    def __init__(
        self,
        store: MemoryStore,
        text_embedder: TextEmbedder,
        *,
        merge_s: float = 10.0,
        max_token_s: float = 60.0,
    ) -> None:
        self.store = store
        self.text_embedder = text_embedder
        self.merge_s = merge_s
        self.max_token_s = max_token_s
        self._last: dict[str, _Open] = {}
        self._lock = asyncio.Lock()

    def reset(self, device_id: str | None = None) -> None:
        if device_id is None:
            self._last.clear()
        else:
            self._last.pop(device_id, None)

    @contextlib.asynccontextmanager
    async def purging(self, device_id: str | None = None) -> AsyncIterator[None]:
        """Hold the write lock around a purge: in-flight writes land first (and get purged), later speech
        starts a fresh token instead of merging into a deleted one."""
        async with self._lock:
            self.reset(device_id)
            yield

    def _mergeable(self, last: _Open | None, ts_start: float, ts_end: float, text: str) -> bool:
        if last is None or self.merge_s <= 0:
            return False
        if sum(len(p) + 1 for p in last.pieces) + len(text) > MAX_TOKEN_CHARS:
            return False
        start = last.token.ts
        return start <= ts_start <= last.end_ts + self.merge_s and max(ts_end, last.end_ts) - start <= self.max_token_s

    async def add(
        self,
        device_id: str,
        ts_start: float,
        ts_end: float,
        text: str,
        pose: Pose | None = None,
        *,
        merge: bool = True,
    ) -> MemoryToken | None:
        """Store `text` heard in [ts_start, ts_end]; returns the new or merged-into token (None for blank text)."""
        text = " ".join(text.split())[:MAX_TOKEN_CHARS]
        if not text:
            return None
        ts_end = max(ts_end, ts_start)
        async with self._lock:
            last = self._last.get(device_id)
            if merge and self._mergeable(last, ts_start, ts_end, text):
                assert last is not None
                if last.pieces and last.pieces[-1] == text:  # the same cue re-sent with the next frame
                    last.end_ts = max(last.end_ts, ts_end)
                    return last.token
                pieces = [*last.pieces, text]
                caption = transcript_caption(" ".join(pieces))
                token = last.token.model_copy(
                    update={"caption": caption, "text_vec": await self.text_embedder.embed(caption)}
                )
                await self.store.add(token)  # upsert by id
                self._last[device_id] = _Open(token, max(last.end_ts, ts_end), pieces)
                return token
            caption = transcript_caption(text)
            token = MemoryToken(
                id=str(ULID()),
                device_id=device_id,
                ts=ts_start,
                caption=caption,
                entities=["speech"],
                pose=pose,
                text_vec=await self.text_embedder.embed(caption),
                thumb_path="",
                frame_hash="",
                kind="transcript",
            )
            await self.store.add(token)
            if last is None or ts_start >= last.token.ts:  # never move the merge target backwards in time
                self._last[device_id] = _Open(token, ts_end, [text])
            return token


def transcript_sink(app_state) -> TranscriptSink:
    """The app's sink (created in the lifespan); built lazily for apps/tests that predate it."""
    sink = getattr(app_state, "transcripts", None)
    if sink is None:
        s = app_state.settings
        sink = app_state.transcripts = TranscriptSink(
            app_state.store,
            app_state.text_embedder,
            merge_s=getattr(s, "speech_merge_s", 10.0),
            max_token_s=getattr(s, "speech_max_token_s", 60.0),
        )
    return sink


@router.post("/audio", response_model=AudioAccepted)
async def post_audio(request: Request, audio: UploadFile = File(...), meta: str = Form(...)):
    app = request.app.state
    try:
        am = AudioMeta.model_validate(json.loads(meta))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"bad meta: {exc}") from exc

    too_big = f"audio chunk larger than {MAX_AUDIO_BYTES} bytes; send shorter chunks"
    if audio.size is not None and audio.size > MAX_AUDIO_BYTES:
        raise HTTPException(413, too_big)
    data = await audio.read(MAX_AUDIO_BYTES + 1)  # never pull more than the cap into memory
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, too_big)
    # WAV passes through; providers needing 16 kHz (faster-whisper) resample + band-pass themselves
    wav = await asyncio.to_thread(transcribe.ensure_wav, data)
    if wav is None:
        raise HTTPException(415, "audio is not a WAV and ffmpeg is unavailable or could not decode it")
    duration = await asyncio.to_thread(wav_duration, wav) or 0.0
    if duration > transcribe.MAX_AUDIO_S:
        raise HTTPException(413, f"audio chunk longer than {transcribe.MAX_AUDIO_S:g} s; send shorter chunks")

    audio_id = str(ULID())
    try:
        segments = await app.audio_stt.transcribe_segments(wav)
    except Exception as exc:  # noqa: BLE001
        log.exception("transcription failed for %s", audio_id)
        raise HTTPException(502, f"transcription failed: {exc}") from exc

    windows = merge_segments(segments, app.settings.audio_window_s)
    sink = transcript_sink(app)
    token_ids: list[str] = []
    for i, w in enumerate(windows):
        # within a chunk the windows are already sized by AUDIO_WINDOW_S; only the first may continue the
        # previous chunk's last token (an utterance cut by the chunk boundary)
        t = await sink.add(
            am.device_id,
            am.ts_start + max(0.0, w.start),
            am.ts_start + max(0.0, w.end),
            w.text,
            am.pose,
            merge=i == 0,
        )
        if t is not None and t.id not in token_ids:
            token_ids.append(t.id)
    return AudioAccepted(
        audio_id=audio_id,
        duration_s=duration,
        transcript=" ".join(w.text.strip() for w in windows),
        token_ids=token_ids,
    )
