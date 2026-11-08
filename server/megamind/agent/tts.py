"""Text-to-speech providers (stream 3). Output is a WAV written to settings.audio_dir and
served at /v1/audio/{id}.wav (the headset plays it with UnityWebRequestMultimedia, WAV only).

  openai     -> `audio.speech` with response_format=pcm (24 kHz), wrapped in a WAV header
  elevenlabs -> REST text-to-speech with output_format=pcm_16000, wrapped in a WAV header
  mock       -> no audio (audio_url=null)
"""

from __future__ import annotations

import asyncio
import logging
import re
import struct
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from megamind.config import Settings

log = logging.getLogger(__name__)


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    byte_rate = sample_rate * channels * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        channels * 2,
        16,
        b"data",
        len(pcm),
    )
    return header + pcm


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_GAP_24K = bytes(2 * 24000 * 15 // 100)  # 150 ms of silence


def split_sentences(text: str, min_chars: int = 25) -> list[str]:
    """`text` as sentences for parallel synthesis; fragments shorter than `min_chars` stay with their neighbour."""
    out: list[str] = []
    for part in _SENTENCE_END.split(text.strip()):
        if out and (len(out[-1]) < min_chars or len(part) < min_chars):
            out[-1] = f"{out[-1]} {part}"
        elif part:
            out.append(part)
    return out or [text]


class TTS(ABC):
    @abstractmethod
    async def synthesize(self, text: str, out_id: str) -> str | None:
        """Return the audio URL (/v1/audio/{id}.wav) or None if TTS is disabled."""


class MockTTS(TTS):
    async def synthesize(self, text: str, out_id: str) -> str | None:
        return None


class _FileTTS(TTS):
    def __init__(self, audio_dir: Path) -> None:
        self.audio_dir = audio_dir

    async def synthesize(self, text: str, out_id: str) -> str | None:
        if not text.strip():
            return None
        try:
            wav = await self._render(text)
        except Exception:  # noqa: BLE001
            log.exception("TTS failed; returning no audio")
            return None
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        (self.audio_dir / f"{out_id}.wav").write_bytes(wav)
        return f"/v1/audio/{out_id}.wav"

    @abstractmethod
    async def _render(self, text: str) -> bytes: ...


class OpenAITTS(_FileTTS):
    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        super().__init__(settings.audio_dir)
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.tts_model
        self.voice = settings.tts_voice

    async def _pcm(self, text: str) -> bytes:
        r = await self.client.audio.speech.create(model=self.model, voice=self.voice, input=text, response_format="pcm")
        return r.content

    async def _render(self, text: str) -> bytes:
        # Sentences render concurrently (TTS time ~ the longest sentence, not the sum), joined by a short pause.
        # pcm (24 kHz s16le mono) + our own header: OpenAI's streamed WAV header carries placeholder sizes.
        parts = split_sentences(text)
        pcms = await asyncio.gather(*(self._pcm(p) for p in parts))
        return pcm16_to_wav(_GAP_24K.join(pcms), 24000)


class ElevenLabsTTS(_FileTTS):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.audio_dir)
        self.api_key = settings.elevenlabs_api_key
        self.voice = settings.tts_voice if settings.tts_voice != "alloy" else "21m00Tcm4TlvDq8ikWAM"
        self.model = settings.tts_model if settings.tts_model != "tts-1" else "eleven_turbo_v2_5"

    async def _render(self, text: str) -> bytes:
        async with httpx.AsyncClient(timeout=60.0) as http:
            r = await http.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}",
                params={"output_format": "pcm_16000"},
                headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
                json={"text": text, "model_id": self.model},
            )
            r.raise_for_status()
        return pcm16_to_wav(r.content, 16000)


def build_tts(settings: Settings) -> TTS:
    p = settings.tts_provider.lower()
    if p == "openai" and settings.openai_api_key:
        return OpenAITTS(settings)
    if p == "elevenlabs" and settings.elevenlabs_api_key:
        return ElevenLabsTTS(settings)
    if p != "mock":
        log.warning("TTS_PROVIDER=%s but no API key configured; using MockTTS", p)
    return MockTTS()
