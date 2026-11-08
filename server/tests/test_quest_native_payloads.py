"""The native Quest 3 app (headset-android/) talks to /v1/frames, /v1/audio and /v1/video with hand-built multipart.

These tests replay byte-for-byte what `MegamindApi.kt` writes (part order, headers, meta JSON formatting,
a 44-byte WAV header around raw PCM16) so a server-side contract change that would break the app fails here.
"""

import io
import json
import math
import struct
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from megamind.replay import find_ffmpeg

T0 = 1_762_000_000.25
# the app's own test clip (MegamindApiTest): 1 s, 160x120, 30 fps H.264 mp4, shaped like a chunk's video.mp4
APP_CLIP = Path(__file__).resolve().parents[2] / "headset-android/app/src/test/resources/clip.mp4"


def _checkerboard_jpeg(w: int = 640, h: int = 480) -> bytes:
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for y in range(0, h, 24):
        for x in range(0, w, 24):
            v = ((x // 24 + y // 24) % 2) * 160 + 40
            d.rectangle([x, y, x + 23, y + 23], fill=(v, 255 - v, x * 255 // w))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _wav_header(data_bytes: int, rate: int = 16000) -> bytes:
    """Same layout as MegamindApi.wavHeader (PCM16 mono)."""
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_bytes)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", data_bytes)
    )


def _tone_pcm(seconds: float, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    return b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n))


def _multipart(parts: list[tuple[str, str | None, str | None, bytes]]) -> tuple[bytes, str]:
    """MegamindApi.Multipart.writeTo: (name, filename, content_type, body) in order."""
    boundary = "----megamind" + uuid.uuid4().hex
    out = bytearray()
    for name, filename, ctype, body in parts:
        out += f"--{boundary}\r\n".encode()
        disp = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disp += f'; filename="{filename}"'
        out += (disp + "\r\n").encode()
        if ctype is not None:
            out += f"Content-Type: {ctype}\r\n".encode()
        out += b"\r\n" + body + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _frame_body(device: str, ts: float, jpeg: bytes, filename: str = "000000.jpg"):
    meta = f'{{"device_id":{json.dumps(device)},"ts":{ts:.3f}}}'  # MegamindApi.frameMetaJson
    return _multipart([("meta", None, None, meta.encode()), ("image", filename, "image/jpeg", jpeg)])


def _audio_body(device: str, ts_start: float, pcm: bytes, filename: str = "20261107_170000_000.wav"):
    meta = f'{{"device_id":{json.dumps(device)},"ts_start":{ts_start:.3f}}}'  # MegamindApi.audioMetaJson
    wav = _wav_header(len(pcm)) + pcm
    return _multipart([("meta", None, None, meta.encode()), ("audio", filename, "audio/wav", wav)])


def _video_body(device: str, ts_start: float, mp4: bytes, filename: str = "20261107_170000_000.mp4"):
    meta = f'{{"device_id":{json.dumps(device)},"ts_start":{ts_start:.3f}}}'  # MegamindApi.videoMetaJson
    return _multipart([("meta", None, None, meta.encode()), ("video", filename, "video/mp4", mp4)])


async def test_native_app_frame_payload_accepted(client):
    c, app = client
    body, ctype = _frame_body("quest3-native", T0, _checkerboard_jpeg())
    r = await c.post("/v1/frames", content=body, headers={"Content-Type": ctype, "Connection": "close"})
    assert r.status_code == 202, r.text
    j = r.json()
    assert j["accepted"] is True and j["frame_id"]
    # the app counts accepted frames by this exact substring of the compact JSON response
    assert '"accepted":true' in r.text


async def test_native_app_frame_filtered_is_still_2xx(client):
    """A duplicate is dropped by the filter but answered 202, so the app marks it sent and moves on."""
    c, app = client
    jpeg = _checkerboard_jpeg()
    for i, expect in enumerate([True, False]):
        body, ctype = _frame_body("quest3-native", T0 + i, jpeg, f"{i:06d}.jpg")
        r = await c.post("/v1/frames", content=body, headers={"Content-Type": ctype})
        assert r.status_code == 202, r.text
        assert r.json()["accepted"] is expect


async def test_native_app_audio_payload_accepted(client):
    c, app = client
    body, ctype = _audio_body("quest3-native", T0, _tone_pcm(5.0))
    r = await c.post("/v1/audio", content=body, headers={"Content-Type": ctype, "Connection": "close"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["audio_id"]
    assert abs(j["duration_s"] - 5.0) < 1e-6
    assert len(j["token_ids"]) >= 1  # mock STT returns a transcript for any non-empty clip
    tok = await app.state.store.get(j["token_ids"][0])
    assert tok is not None and tok.kind == "transcript"
    assert tok.device_id == "quest3-native"
    assert tok.ts >= T0


async def test_native_app_bad_meta_is_permanent_400(client):
    """The app moves a chunk to chunks_failed on 400/413/415/422 instead of retrying forever."""
    c, _ = client
    body, ctype = _multipart([("meta", None, None, b"{}"), ("image", "x.jpg", "image/jpeg", _checkerboard_jpeg())])
    r = await c.post("/v1/frames", content=body, headers={"Content-Type": ctype})
    assert r.status_code == 400


async def test_native_app_corrupt_jpeg_detail_is_stable(client):
    """The app skips only a corrupt frame, not the whole chunk, on this 400 detail (MegamindApi.UNDECODABLE_JPEG)."""
    c, _ = client
    body, ctype = _frame_body("quest3-native", T0, _checkerboard_jpeg()[:200])
    r = await c.post("/v1/frames", content=body, headers={"Content-Type": ctype})
    assert r.status_code == 400
    assert "not a decodable JPEG" in r.json()["detail"]


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")
async def test_native_app_video_payload_accepted(client):
    c, app = client
    body, ctype = _video_body("quest3-native", T0, APP_CLIP.read_bytes())
    r = await c.post("/v1/video", content=body, headers={"Content-Type": ctype, "Connection": "close"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["accepted"] is True and j["video_id"] and abs(j["duration_s"] - 1.0) < 0.1
    assert len(app.state.video.clips("quest3-native")) == 1
    # Uploader.uploadVideo looks for this exact substring when face blur makes the server discard video
    app.state.settings.privacy_blur_faces = True
    r = await c.post("/v1/video", content=body, headers={"Content-Type": ctype})
    assert r.status_code == 200 and '"reason":"privacy"' in r.text
