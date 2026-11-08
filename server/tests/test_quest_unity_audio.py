"""The Unity Quest app (headset/Megamind, AmbientAudioRecorder.cs) uploads ambient audio to /v1/audio.

These tests rebuild byte-for-byte what `AudioChunkCodec.cs` encodes (44-byte RIFF header + PCM16 mono, float
samples scaled by 32767 and rounded half-to-even like C# Math.Round, meta JSON with ts_start to 3 decimals) inside
the multipart layout `UnityWebRequest.Post(url, List<IMultipartFormSection>)` writes, and check the server (mock
STT, no OpenAI key) stores it as a heard-speech token that "what did I hear?" retrieves.
"""

import json
import math
import struct
import uuid

import pytest

T0 = 1_762_000_000.25
RATE = 16000
CHUNK_S = 15  # CaptureSettings.audioChunkSeconds default (docs/api.md: ~15-30 s chunks)


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _unity_wav(samples: list[float], rate: int = RATE) -> bytes:
    """AudioChunkCodec.EncodeWav."""
    data = b"".join(struct.pack("<h", max(-32768, min(32767, round(_f32(_f32(v) * _f32(32767.0)))))) for v in samples)
    header = (
        b"RIFF"
        + struct.pack("<i", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<ihhiihh", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<i", len(data))
    )
    return header + data


def _unity_meta(device: str, ts_start: float) -> str:
    """AudioChunkCodec.MetaJson without a pose."""
    return f'{{"device_id":{json.dumps(device)},"ts_start":{ts_start:.3f}}}'


def _unity_multipart(wav: bytes, meta: str, filename: str) -> tuple[bytes, str]:
    """UnityWebRequest.SerializeFormSections: CRLF before every boundary, data sections typed text/plain."""
    boundary = uuid.uuid4().hex[:40]
    out = bytearray()
    out += f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="{filename}"\r\n'.encode()
    out += b"Content-Type: audio/wav\r\n\r\n" + wav
    out += f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="meta"\r\n'.encode()
    out += b"Content-Type: text/plain; encoding=utf-8\r\n\r\n" + meta.encode()
    out += f"\r\n--{boundary}--".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _speech_like(seconds: int = CHUNK_S, rate: int = RATE) -> list[float]:
    """1 s of digital silence, then a 220 Hz tone, with two clipped samples (exercises the clamp)."""
    n = seconds * rate
    s = [0.0 if i < rate else 0.3 * math.sin(2 * math.pi * 220 * i / rate) for i in range(n)]
    s[2 * rate], s[2 * rate + 1] = 1.5, -1.5
    return s


async def _post(c, device: str, ts_start: float, samples: list[float]):
    body, ctype = _unity_multipart(_unity_wav(samples), _unity_meta(device, ts_start), "20251101_122640_250.wav")
    return await c.post("/v1/audio", content=body, headers={"Content-Type": ctype})


def test_unity_wav_layout():
    wav = _unity_wav(_speech_like())
    assert len(wav) == 44 + CHUNK_S * RATE * 2
    assert wav[:4] == b"RIFF" and wav[8:16] == b"WAVEfmt " and wav[36:40] == b"data"
    assert struct.unpack("<ihhiihh", wav[16:36]) == (16, 1, 1, RATE, RATE * 2, 2, 16)
    assert struct.unpack("<hh", wav[44 + 4 * RATE : 44 + 4 * RATE + 4]) == (32767, -32768)  # clamped


async def test_unity_ambient_chunk_accepted_and_heard(client):
    c, app = client
    r = await _post(c, "quest3-unity", T0, _speech_like())
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["audio_id"]
    assert j["duration_s"] == pytest.approx(CHUNK_S)
    # default MockSTT (no OpenAI key): one "[mock transcript]" segment per few seconds of audio
    assert j["transcript"].startswith("[mock transcript]")
    (tid,) = j["token_ids"]
    tok = await app.state.store.get(tid)
    assert tok.kind == "transcript" and tok.device_id == "quest3-unity"
    assert tok.ts == pytest.approx(T0, abs=1e-3)  # ts_start = first sample, same wall clock as frames

    res = await app.state.recall(query="what did I hear?", k=4, now=T0 + 120)
    assert res.all_retrieved and res.all_retrieved[0].id == tid


async def test_unity_consecutive_chunks_merge(client):
    """Back-to-back 15 s chunks (the recorder uploads oldest first) continue one speech token."""
    c, app = client
    first = (await _post(c, "quest3-unity", T0, _speech_like())).json()
    second = (await _post(c, "quest3-unity", T0 + CHUNK_S, _speech_like())).json()
    assert second["token_ids"] == first["token_ids"]
    assert await app.state.store.count() == 1
