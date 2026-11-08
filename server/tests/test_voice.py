"""Stream 3: voice path (STT -> chat -> TTS -> /v1/audio) and the browser panel at /ui."""

import io
import struct
import wave

import pytest

from megamind.agent import stt as stt_mod
from megamind.agent import tts as tts_mod
from megamind.config import Settings


def _silence_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class FakeTTS(tts_mod._FileTTS):
    async def _render(self, text: str) -> bytes:
        return tts_mod.pcm16_to_wav(b"\x00\x00" * 160, 16000)


def test_pcm16_to_wav_header():
    wav = tts_mod.pcm16_to_wav(b"\x01\x02" * 100, 16000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 100)
    assert struct.unpack("<I", wav[40:44])[0] == 200


def test_provider_factories_fall_back_to_mock_without_keys():
    s = Settings(stt_provider="openai", tts_provider="elevenlabs", _env_file=None)
    assert isinstance(stt_mod.build_stt(s), stt_mod.MockSTT)
    assert isinstance(tts_mod.build_tts(s), tts_mod.MockTTS)
    s = Settings(stt_provider="deepgram", deepgram_api_key="x", _env_file=None)
    assert isinstance(stt_mod.build_stt(s), stt_mod.DeepgramSTT)


def test_stt_faster_whisper_is_keyless():
    pytest.importorskip("faster_whisper")
    from megamind.ingest import transcribe

    s = Settings(stt_provider="faster-whisper", _env_file=None)
    expected = transcribe.FasterWhisperSTT if transcribe.have_ffmpeg() else stt_mod.MockSTT
    assert isinstance(stt_mod.build_stt(s), expected)


async def test_chat_multipart_audio_roundtrip(client):
    c, app = client
    app.state.tts = FakeTTS(app.state.settings.audio_dir)
    r = await c.post(
        "/v1/chat",
        files={"audio": ("q.wav", _silence_wav(), "audio/wav")},
        data={"session_id": "voice", "device_id": "dev1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "[mock transcript]"
    assert body["audio_url"].startswith("/v1/audio/") and body["audio_url"].endswith(".wav")

    a = await c.get(body["audio_url"])
    assert a.status_code == 200 and a.headers["content-type"] == "audio/wav"
    assert a.content[:4] == b"RIFF"


async def test_chat_multipart_requires_audio(client):
    c, _ = client
    r = await c.post("/v1/chat", data={"session_id": "voice"}, files={"x": ("x.txt", b"nope", "text/plain")})
    assert r.status_code == 400


async def test_mock_tts_returns_no_audio(client):
    c, _ = client
    r = await c.post("/v1/chat", json={"session_id": "s", "text": "hello"})
    assert r.status_code == 200 and r.json()["audio_url"] is None


async def test_ui_served(client):
    c, _ = client
    r = await c.get("/ui")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "/v1/chat" in r.text and "/v1/frames" in r.text
