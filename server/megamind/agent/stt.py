"""Speech-to-text. TODO(stream-3): OpenAI Whisper / Deepgram implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

from megamind.config import Settings


class Segment(NamedTuple):
    """One transcribed span; `start`/`end` are seconds from the start of the clip."""

    start: float
    end: float
    text: str


def wav_duration(wav: bytes) -> float | None:
    """Duration of a RIFF/WAV clip in seconds, or None if it is not a WAV the stdlib can parse."""
    import io  # local: keeps this module's import block identical to main's for low-conflict merges
    import wave

    try:
        with wave.open(io.BytesIO(wav)) as w:
            rate = w.getframerate()
            return w.getnframes() / rate if rate else None
    except (wave.Error, EOFError):
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


def build_stt(settings: Settings) -> STT:
    return MockSTT()
