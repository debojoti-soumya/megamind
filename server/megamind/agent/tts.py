"""Text-to-speech. TODO(stream-3): OpenAI TTS / ElevenLabs; write WAV to settings.audio_dir."""

from __future__ import annotations

from abc import ABC, abstractmethod

from megamind.config import Settings


class TTS(ABC):
    @abstractmethod
    async def synthesize(self, text: str, out_id: str) -> str | None:
        """Return the audio URL (/v1/audio/{id}.wav) or None if TTS is disabled."""


class MockTTS(TTS):
    async def synthesize(self, text: str, out_id: str) -> str | None:
        return None


def build_tts(settings: Settings) -> TTS:
    return MockTTS()
