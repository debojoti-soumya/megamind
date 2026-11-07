"""Speech-to-text. TODO(stream-3): OpenAI Whisper / Deepgram implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from megamind.config import Settings


class STT(ABC):
    @abstractmethod
    async def transcribe(self, wav: bytes) -> str: ...


class MockSTT(STT):
    async def transcribe(self, wav: bytes) -> str:
        return "[mock transcript]"


def build_stt(settings: Settings) -> STT:
    return MockSTT()
