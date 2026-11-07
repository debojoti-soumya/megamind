"""Speech-to-text providers (stream 3).

Input is always a 16 kHz mono WAV from the headset / browser panel. Providers:
  openai   -> `audio.transcriptions` (whisper-1 / gpt-4o-mini-transcribe)
  deepgram -> REST `listen` endpoint (nova-2)
  mock     -> fixed string, used in tests and before keys are configured
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx

from megamind.config import Settings

log = logging.getLogger(__name__)


class STT(ABC):
    @abstractmethod
    async def transcribe(self, wav: bytes) -> str: ...


class MockSTT(STT):
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
