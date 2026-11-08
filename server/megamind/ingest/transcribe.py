"""Timestamped transcription for ambient audio (POST /v1/audio).

Providers implement the `STT` interface (so they could also serve /chat) plus `transcribe_segments`:
  openai          whisper-1 `verbose_json` segments (other models: one segment via the STT default)
  faster-whisper  local, VAD-filtered; needs `pip install -e ".[local-stt]"` and ffmpeg on PATH
  mock            deterministic segments (megamind.agent.stt.MockSTT), used by tests

Decoding (ported from Recall's media.extract_audio): any ffmpeg-decodable input -> 16 kHz mono s16 WAV,
optionally band-passed (80-3800 Hz) + loudness-normalised, which helps Whisper on 8-16 kHz headset mics.
ffmpeg is only needed for non-WAV uploads and for faster-whisper.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import tempfile
import wave
from functools import lru_cache
from pathlib import Path

from megamind.agent.stt import STT, MockSTT, Segment, wav_duration
from megamind.config import Settings

log = logging.getLogger(__name__)

ENHANCE_FILTER = "highpass=f=80,lowpass=f=3800,loudnorm"
NO_SPEECH_MAX = 0.6  # whisper segments more likely silence than speech are dropped (silence hallucinations)


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def to_wav16k(data: bytes, *, enhance: bool = False, timeout_s: float = 120.0) -> bytes | None:
    """Decode any ffmpeg-readable audio/video bytes to 16 kHz mono s16 WAV; None if ffmpeg is missing or fails."""
    if not have_ffmpeg():
        return None
    af = ["-af", ENHANCE_FILTER] if enhance else []
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "in", Path(td) / "out.wav"
        src.write_bytes(data)  # a file, not stdin: mp4/m4a with the moov atom at the end need seeking
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", *af]
        try:
            subprocess.run([*cmd, "-c:a", "pcm_s16le", str(out)], check=True, capture_output=True, timeout=timeout_s)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            log.warning("ffmpeg decode failed: %s", getattr(e, "stderr", b"") or e)
            return None
        return out.read_bytes() if out.is_file() and out.stat().st_size > 44 else None


def ensure_wav(data: bytes, *, enhance: bool = False) -> bytes | None:
    """WAV uploads pass through untouched; anything else is decoded with ffmpeg (None if impossible)."""
    if wav_duration(data) is not None:
        return data
    return to_wav16k(data, enhance=enhance)


def _field(obj, name: str, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


class OpenAIWhisperSTT(STT):
    """OpenAI transcription API; `whisper-1` returns per-segment timestamps via `verbose_json`."""

    def __init__(self, settings: Settings, client=None) -> None:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.client = client
        self.model = settings.audio_stt_model or "whisper-1"

    async def transcribe(self, wav: bytes) -> str:
        r = await self.client.audio.transcriptions.create(
            model=self.model, file=("audio.wav", wav, "audio/wav"), response_format="json"
        )
        return (_field(r, "text") or "").strip()

    async def transcribe_segments(self, wav: bytes) -> list[Segment]:
        if not self.model.startswith("whisper"):  # gpt-4o-*-transcribe has no verbose_json
            return await super().transcribe_segments(wav)
        r = await self.client.audio.transcriptions.create(
            model=self.model,
            file=("audio.wav", wav, "audio/wav"),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        out: list[Segment] = []
        for s in _field(r, "segments") or []:
            text = (_field(s, "text") or "").strip()
            if text and (_field(s, "no_speech_prob") or 0.0) <= NO_SPEECH_MAX:
                out.append(Segment(float(_field(s, "start", 0.0)), float(_field(s, "end", 0.0)), text))
        if not out and not _field(r, "segments"):  # no segment list at all: fall back to the plain text
            text = (_field(r, "text") or "").strip()
            out = [Segment(0.0, wav_duration(wav) or 0.0, text)] if text else []
        return out


@lru_cache(maxsize=2)
def _whisper_model(name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    log.info("loading faster-whisper model %s (%s, %s)", name, device, compute_type)
    return WhisperModel(name, device=device, compute_type=compute_type)


class FasterWhisperSTT(STT):
    """Local faster-whisper (ported from Recall's transcribe.py): ffmpeg 16 kHz decode, VAD, beam 5."""

    def __init__(
        self, model: str = "small.en", *, enhance: bool = True, device: str = "auto", compute_type: str = "int8"
    ) -> None:
        import faster_whisper  # noqa: F401  # fail fast at build time if the extra is missing

        self.model, self.enhance, self.device, self.compute_type = model, enhance, device, compute_type

    def _segments_sync(self, data: bytes) -> list[Segment]:
        import numpy as np

        wav = to_wav16k(data, enhance=self.enhance)
        if wav is None:
            raise RuntimeError("faster-whisper needs ffmpeg on PATH to decode audio")
        with wave.open(io.BytesIO(wav)) as w:  # 16 kHz mono s16 from ffmpeg; avoids PyAV entirely
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        segs, _info = _whisper_model(self.model, self.device, self.compute_type).transcribe(
            pcm, vad_filter=True, beam_size=5, condition_on_previous_text=False
        )
        return [Segment(float(s.start), float(s.end), s.text.strip()) for s in segs if s.text.strip()]

    async def transcribe_segments(self, wav: bytes) -> list[Segment]:
        return await asyncio.to_thread(self._segments_sync, wav)

    async def transcribe(self, wav: bytes) -> str:
        return " ".join(s.text for s in await self.transcribe_segments(wav))


def build_audio_stt(settings: Settings, chat_stt: STT) -> STT:
    """STT for ambient audio. Unknown / unconfigured providers fall back to `chat_stt` with a warning."""
    p = (settings.audio_stt_provider or "").lower().strip()
    if not p:
        if settings.stt_provider.lower() == "openai" and settings.openai_api_key:
            return OpenAIWhisperSTT(settings)
        return chat_stt
    if p == "mock":
        return MockSTT()
    if p == "openai":
        if settings.openai_api_key:
            return OpenAIWhisperSTT(settings)
        log.warning("AUDIO_STT_PROVIDER=openai but OPENAI_API_KEY is empty; using the chat STT")
        return chat_stt
    if p in ("faster-whisper", "faster_whisper", "local"):
        try:
            stt = FasterWhisperSTT(settings.audio_stt_model or "small.en", enhance=settings.audio_enhance)
        except ImportError:
            log.warning('AUDIO_STT_PROVIDER=%s but faster-whisper is not installed (pip install -e ".[local-stt]")', p)
            return chat_stt
        if not have_ffmpeg():
            log.warning("faster-whisper needs ffmpeg on PATH; using the chat STT")
            return chat_stt
        return stt
    log.warning("unknown AUDIO_STT_PROVIDER=%s; using the chat STT", p)
    return chat_stt
