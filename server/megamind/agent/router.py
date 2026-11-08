from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from ulid import ULID

from megamind.agent import turn_log
from megamind.agent.speech import StreamingSpeaker
from megamind.devices import ChatDevice, resolve_chat_device
from megamind.models import (
    ChatRequest,
    ChatResponse,
    ChatStreamAudio,
    ChatStreamDelta,
    ChatStreamError,
    ChatStreamRetrieval,
    ChatStreamTranscript,
)
from megamind.replay import annotate_citations

log = logging.getLogger(__name__)
router = APIRouter()


@dataclass
class ChatInput:
    text: str
    session_id: str
    device_id: str | None
    voice: bool = False  # multipart: `text` is the speech-to-text transcript


async def _chat_input(request: Request) -> ChatInput:
    """The question from a JSON or multipart chat request (voice is transcribed here); HTTP errors as documented."""
    app = request.app.state
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        try:
            body = ChatRequest.model_validate(await request.json())
        except ValueError as e:  # malformed JSON (JSONDecodeError) or schema mismatch (ValidationError)
            detail = e.errors(include_url=False) if isinstance(e, ValidationError) else f"invalid JSON: {e}"
            raise HTTPException(422, jsonable_encoder(detail)) from None
        if not body.text:
            raise HTTPException(400, "text is required for JSON requests")
        return ChatInput(body.text, body.session_id, body.device_id)
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        session_id = str(form.get("session_id") or "default")
        device_id = form.get("device_id")
        device_id = str(device_id) if device_id else None
        audio = form.get("audio")
        if audio is None or not hasattr(audio, "read"):
            raise HTTPException(400, "audio file is required for multipart requests")
        text = await app.stt.transcribe(await audio.read())
        return ChatInput(text, session_id, device_id, voice=True)
    raise HTTPException(415, "use application/json or multipart/form-data")


async def _finish(app, resp: ChatResponse, dev: ChatDevice, speaker: StreamingSpeaker | None = None) -> ChatResponse:
    """Replay links + TTS + object markers, concurrently (each reads `cited` / `answer`, writes its own field).
    With a `speaker` (streamed turn, TTS_STREAM) most sentences are already rendered: this speaks the tail and joins
    the clips."""

    async def tts() -> None:
        if speaker is not None:
            resp.audio_url = await speaker.finish(resp.answer)
        else:
            resp.audio_url = await app.tts.synthesize(resp.answer, str(ULID()))

    async def markers() -> None:
        locator = getattr(app, "locator", None)
        if locator is not None and resp.cited:
            resp.markers = await locator.markers(resp.transcript, resp.answer, list(resp.cited))

    await asyncio.gather(annotate_citations(app, resp.cited, dev.device_id), tts(), markers())
    resp.device_id_used, resp.warning = dev.device_id, dev.warning
    return resp


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request):
    app = request.app.state
    started = time.time()
    inp = await _chat_input(request)
    dev = await resolve_chat_device(app, inp.device_id)
    resp = await app.agent.chat(inp.session_id, inp.text, dev.device_id)
    resp = await _finish(app, resp, dev)
    turn_log.record(app, started, time.time(), inp, resp.answer)
    return resp


def _sse(event: str, data: dict | BaseModel) -> bytes:
    payload = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(jsonable_encoder(data))
    return f"event: {event}\ndata: {payload}\n\n".encode()


@router.post("/chat/stream")
async def chat_stream(request: Request):
    """Same turn as `POST /chat`, as Server-Sent Events: `transcript` (voice questions, right after speech-to-text),
    `retrieval` (candidate citations, once per tool result), `delta` (answer text), `reset` (drop the streamed
    text), `audio` (TTS of one sentence, in order, while the rest is still being written; `TTS_STREAM`), then
    `final` (exactly the `POST /chat` body) or `error`. Bad requests fail before the stream starts, with the same
    status codes as `POST /chat`."""
    app = request.app.state
    settings = app.settings
    started = time.time()
    inp = await _chat_input(request)
    dev = await resolve_chat_device(app, inp.device_id)
    device_id = dev.device_id
    queue: asyncio.Queue[tuple[str, dict | BaseModel] | None] = asyncio.Queue()

    if inp.voice:  # show what was heard at once (and lets a client spot a voice command and hang up)
        queue.put_nowait(("transcript", ChatStreamTranscript(text=inp.text)))

    async def put(name: str, data: dict) -> None:
        await queue.put((name, data))  # never blocks the turn: replay links are added on the sending side

    speaker = StreamingSpeaker(app.tts, app.settings.audio_dir, put) if app.settings.tts_stream else None

    async def on_event(name: str, data: dict) -> None:
        if speaker is not None:
            if name == "delta":
                await speaker.feed(data["text"])
            elif name == "reset":
                await speaker.reset()
        await put(name, data)

    async def annotate_candidates(cands: list) -> None:
        """replay_url lookups for the `retrieval` event, bounded so answer text queued behind it is not held up."""
        task = asyncio.ensure_future(annotate_citations(app, cands, device_id))
        budget = settings.stream_replay_budget_s
        done, _ = await asyncio.wait([task], timeout=budget if budget > 0 else None)
        if not done:  # links found so far stay (in place); the rest are null here and present in `final`
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        else:
            task.result()

    async def encode(name: str, data: dict | BaseModel) -> bytes:
        if name == "retrieval":
            cands = list(data["candidates"])
            await annotate_candidates(cands)
            data = ChatStreamRetrieval(confidence=data["confidence"], candidates=cands)
        elif name == "delta":
            data = ChatStreamDelta(**data)
        elif name == "audio":
            data = ChatStreamAudio(**data)
        return _sse(name, data)

    async def run() -> None:
        try:
            resp, _ = await app.agent.chat_with_stats(inp.session_id, inp.text, device_id, on_event=on_event)
            final = await _finish(app, resp, dev, speaker)
            turn_log.record(app, started, time.time(), inp, final.answer)
            await queue.put(("final", final))
        except Exception as e:  # noqa: BLE001 - the stream is already 200: report the failure in-band
            log.exception("chat stream failed")
            if speaker is not None:
                await speaker.cancel()
            await queue.put(("error", ChatStreamError(detail=f"{type(e).__name__}: {e}")))
        finally:
            await queue.put(None)

    async def events():
        task = asyncio.create_task(run())
        try:
            while (item := await queue.get()) is not None:
                try:
                    chunk = await encode(*item)
                except Exception as e:  # noqa: BLE001 - e.g. a replay lookup failed: end the stream in-band
                    log.exception("chat stream encode failed")
                    yield _sse("error", ChatStreamError(detail=f"{type(e).__name__}: {e}"))
                    break
                yield chunk
        finally:  # client went away: stop the turn
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if speaker is not None:
                await speaker.cancel()

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events(), media_type="text/event-stream", headers=headers)


@router.get("/audio/{audio_id}.wav")
async def get_audio(request: Request, audio_id: str):
    p = request.app.state.settings.audio_dir / f"{audio_id}.wav"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")


# re-exported for type checkers / docs
__all__ = ["router", "File", "Form", "UploadFile"]
