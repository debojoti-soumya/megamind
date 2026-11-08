from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pydantic import ValidationError
from ulid import ULID

from megamind.models import ChatRequest, ChatResponse
from megamind.replay import annotate_citations

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request):
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
        text, session_id, device_id = body.text, body.session_id, body.device_id
    elif ctype.startswith("multipart/form-data"):
        form = await request.form()
        session_id = str(form.get("session_id") or "default")
        device_id = form.get("device_id")
        device_id = str(device_id) if device_id else None
        audio = form.get("audio")
        if audio is None or not hasattr(audio, "read"):
            raise HTTPException(400, "audio file is required for multipart requests")
        text = await app.stt.transcribe(await audio.read())
    else:
        raise HTTPException(415, "use application/json or multipart/form-data")

    resp = await app.agent.chat(session_id, text, device_id)
    await annotate_citations(app, resp.cited, device_id)
    resp.audio_url = await app.tts.synthesize(resp.answer, str(ULID()))
    return resp


@router.get("/audio/{audio_id}.wav")
async def get_audio(request: Request, audio_id: str):
    p = request.app.state.settings.audio_dir / f"{audio_id}.wav"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")


# re-exported for type checkers / docs
__all__ = ["router", "File", "Form", "UploadFile"]
