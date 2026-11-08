"""Speech-to-text providers (stream 3).

Input is always a 16 kHz mono WAV from the headset / browser panel. Providers:
  openai   -> `audio.transcriptions` (whisper-1 / gpt-4o-mini-transcribe)
  deepgram -> REST `listen` endpoint (nova-2)
  mock     -> fixed string, used in tests and before keys are configured
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

import httpx

from megamind.config import Settings

log = logging.getLogger(__name__)


class Segment(NamedTuple):
    """One transcribed span; `start`/`end` are seconds from the start of the clip."""

    start: float
    end: float
    text: str


def wav_duration(wav: bytes) -> float | None:
    """Duration of a RIFF/WAV clip in seconds, or None if it is not a WAV the stdlib can parse.

    Counts the sample data actually present, not the header's claimed frame count (a 244-byte upload can
    claim 2^31 frames at 1 Hz)."""
    import io  # local: keeps this module's import block identical to main's for low-conflict merges
    import wave

    try:
        with wave.open(io.BytesIO(wav)) as w:
            rate, frame_bytes = w.getframerate(), w.getsampwidth() * w.getnchannels()
            if not rate or not frame_bytes:
                return None
            return len(w.readframes(w.getnframes())) // frame_bytes / rate
    except (wave.Error, EOFError, ValueError):
        return None


class STT(ABC):
    @abstractmethod
    async def transcribe(self, wav: bytes) -> str: ...

    async def transcribe_segments(self, wav: bytes) -> list[Segment]:
        """Optional: timestamped segments (used by ambient audio, POST /v1/audio).

        Default: the whole clip as one segment via `transcribe()`, so providers without
        timestamps still work. Empty transcripts yield no segments."""
        text = (await self.transcribe(wav)).strip()
        return [Segment(0.0, wav_duration(wav) or 0.0, text)] if text else []


class MockSTT(STT):
    def __init__(self, script: list[str] | None = None, segment_s: float = 5.0) -> None:
        """`script`: texts for consecutive `segment_s`-long segments (cycled); default "[mock transcript]"."""
        self.script = script or ["[mock transcript]"]
        self.segment_s = segment_s

    async def transcribe_segments(self, wav: bytes) -> list[Segment]:
        dur = wav_duration(wav) or self.segment_s
        n = max(1, int(-(-dur // self.segment_s)))
        return [
            Segment(i * self.segment_s, min((i + 1) * self.segment_s, dur), self.script[i % len(self.script)])
            for i in range(n)
        ]

    async def transcribe(self, wav: bytes) -> str:
        return "[mock transcript]"


class OpenAISTT(STT):
    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.stt_model

    async def transcribe(self, wav: bytes) -> str:
        r = await self.client.audio.transcriptions.create(
            model=self.model,
            file=("q.wav", wav, "audio/wav"),
            response_format="json",
        )
        return (r.text or "").strip()


class DeepgramSTT(STT):
    URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.deepgram_api_key
        self.model = settings.stt_model if settings.stt_model != "whisper-1" else "nova-2"

    async def transcribe(self, wav: bytes) -> str:
        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.post(
                self.URL,
                params={"model": self.model, "smart_format": "true", "language": "en"},
                headers={"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"},
                content=wav,
            )
            r.raise_for_status()
        try:
            return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        except (KeyError, IndexError):
            return ""


def build_stt(settings: Settings) -> STT:
    p = settings.stt_provider.lower()
    if p == "openai" and settings.openai_api_key:
        return OpenAISTT(settings)
    if p == "deepgram" and settings.deepgram_api_key:
        return DeepgramSTT(settings)
    if p != "mock":
        log.warning("STT_PROVIDER=%s but no API key configured; using MockSTT", p)
    return MockSTT()
