"""`POST /v1/stt`: transcript only, so the headset can act on voice commands ("open menu", "pause recording")
without a Claude turn or TTS, and send real questions to `POST /v1/chat/stream` as text (docs/api.md).

The route reuses the chat STT (`app.state.stt`, agent/stt.py: Tim's providers, untouched). `command_hints` is an
optional phrase list the server fuzzy-matches the transcript against (one normalization for every client), so the
headset does not have to re-implement the matcher. Zero cost with the mock provider.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from megamind.agent.stt import wav_duration
from megamind.ingest.audio import MAX_AUDIO_BYTES
from megamind.models import CommandMatch, SttResponse

router = APIRouter()

MAX_HINTS = 200
# words the headset matcher drops before comparing (mirrors VoiceCommands.Normalize in the Unity client)
_FILLERS = frozenset(
    "please hey ok okay megamind can you could would the a my now up".split(),
)
_NOT_WORD = re.compile(r"[^a-z0-9]+")


def normalize_command(text: str) -> str:
    """Lower-case, letters/digits only, filler words removed, single spaces: the key commands are matched on."""
    words = _NOT_WORD.sub(" ", text.lower()).split()
    return " ".join(w for w in words if w not in _FILLERS)


def match_command(
    transcript: str, hints: list[str], *, min_score: float = 0.8, max_words: int = 6
) -> CommandMatch | None:
    """The hint the transcript means, or None. Exact match after normalization scores 1.0; otherwise the best
    `difflib` ratio >= `min_score` wins ("open the menu" ~ "open menu", "paws recording" ~ "pause recording"),
    among hints whose words line up with the transcript's (`_words_align`), so "start recording" never matches
    "stop recording" nor "unmute" "mute". Transcripts longer than `max_words` (after normalization) are questions,
    never commands."""
    t = normalize_command(transcript)
    if not t or len(t.split()) > max_words:
        return None
    best: tuple[float, str] | None = None
    for hint in hints:
        h = normalize_command(hint)
        if not h:
            continue
        if h == t:
            return CommandMatch(phrase=hint, score=1.0)
        score = difflib.SequenceMatcher(None, t, h).ratio()
        if score >= min_score and _words_align(t, h) and (best is None or score > best[0]):
            best = (score, hint)
    return CommandMatch(phrase=best[1], score=round(best[0], 3)) if best else None


def _word_close(a: str, b: str) -> bool:
    """A misheard word ("paws"/"pause", "stopped"/"stop", "pen"/"open"), not a different one ("start"/"stop",
    "unmute"/"mute", "disable"/"enable"): same first letter (or one clipped leading letter) and similar spelling."""
    if a == b:
        return True
    if a[1:] == b or b[1:] == a:
        return True
    return a[0] == b[0] and difflib.SequenceMatcher(None, a, b).ratio() >= 0.6


def _words_align(t: str, h: str) -> bool:
    tw, hw = t.split(), h.split()
    return len(tw) == len(hw) and all(_word_close(a, b) for a, b in zip(tw, hw, strict=True))


def parse_hints(raw: Any) -> list[str]:
    """`command_hints` form field: a JSON array of strings, or phrases separated by newlines / `|`."""
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    hints: Any
    if raw.startswith("["):
        try:
            hints = json.loads(raw)
        except ValueError:
            raise HTTPException(
                400, "command_hints must be a JSON array of strings or a newline/|-separated list"
            ) from None
        if not isinstance(hints, list) or not all(isinstance(h, str) for h in hints):
            raise HTTPException(400, "command_hints must be a JSON array of strings")
    else:
        hints = re.split(r"[\n|]", raw)
    hints = [h.strip() for h in hints if h and h.strip()]
    if len(hints) > MAX_HINTS:
        raise HTTPException(400, f"at most {MAX_HINTS} command_hints")
    return hints


def stt_label(stt: object) -> str:
    """`SttResponse.model`: "mock", or "<provider>:<model>" (OpenAISTT -> "openai:gpt-transcribe")."""
    name = type(stt).__name__
    if name.startswith("Mock"):
        return "mock"
    provider = name.removesuffix("STT").lower()
    model = getattr(stt, "model", None)
    return f"{provider}:{model}" if isinstance(model, str) and model else provider


@router.post("/stt", response_model=SttResponse)
async def stt(request: Request):
    """multipart/form-data: `audio` (the clip `POST /chat` accepts), optional `command_hints`."""
    app = request.app.state
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        raise HTTPException(415, "use multipart/form-data with an `audio` file")
    form = await request.form()
    audio = form.get("audio")
    if audio is None or not hasattr(audio, "read"):
        raise HTTPException(400, "audio file is required")
    hints = parse_hints(form.get("command_hints"))
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(400, "audio file is empty")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, f"audio larger than {MAX_AUDIO_BYTES} bytes")
    try:
        transcript = (await app.stt.transcribe(data)).strip()
    except Exception as e:  # noqa: BLE001 - provider/network failure: tell the client, which can retry or fall back
        raise HTTPException(502, f"STT failed: {type(e).__name__}: {e}") from e
    return SttResponse(
        transcript=transcript,
        duration_s=wav_duration(data),
        model=stt_label(app.stt),
        command=match_command(transcript, hints) if hints else None,
    )
