"""/v1/frames, /v1/audio and /v1/video share one metadata rule (#129): finite unix-seconds timestamp not far in the
future, non-empty device_id short enough to be a directory name. Bad meta -> 400 `bad meta: ...`, nothing stored."""

from __future__ import annotations

import io
import json
import random
import time
import wave

import pytest
from PIL import Image
from pydantic import ValidationError

from megamind.models import MAX_FUTURE_S, AudioMeta, FrameMeta, VideoMeta


def _noisy(seed: int) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", (96, 72))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(96 * 72)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return buf.getvalue()


NOW = time.time()
BAD = [
    pytest.param('{"device_id": "adv", "TS": NaN}', "finite", id="nan"),
    pytest.param('{"device_id": "adv", "TS": Infinity}', "finite", id="inf"),
    pytest.param('{"device_id": "adv", "TS": 4e10}', "in the future", id="year-3237"),
    pytest.param(f'{{"device_id": "", "TS": {NOW}}}', "device_id must not be empty", id="empty-id"),
    pytest.param(f'{{"device_id": "   ", "TS": {NOW}}}', "device_id must not be empty", id="blank-id"),
    pytest.param(f'{{"device_id": "{"x" * 5000}", "TS": {NOW}}}', "device_id too long", id="5000-char-id"),
]


@pytest.mark.parametrize(("meta", "why"), BAD)
async def test_bad_frame_meta_is_400_and_not_stored(client, meta, why):
    c, app = client
    r = await c.post(
        "/v1/frames", files={"image": ("f.jpg", _noisy(1), "image/jpeg")}, data={"meta": meta.replace("TS", "ts")}
    )
    assert r.status_code == 400, r.text
    assert r.json()["detail"].startswith("bad meta:") and why in r.json()["detail"]
    await app.state.worker.queue.join()
    assert await app.state.store.count() == 0
    assert (await c.get("/v1/devices")).json()["devices"] == []


@pytest.mark.parametrize(("meta", "why"), BAD)
async def test_bad_audio_meta_is_400(client, meta, why):
    c, _ = client
    r = await c.post(
        "/v1/audio", files={"audio": ("a.wav", _wav(), "audio/wav")}, data={"meta": meta.replace("TS", "ts_start")}
    )
    assert r.status_code == 400 and why in r.json()["detail"], r.text


@pytest.mark.parametrize(("meta", "why"), BAD)
def test_video_meta_uses_the_same_rule(meta, why):
    with pytest.raises(ValidationError, match=why):
        VideoMeta.model_validate(json.loads(meta.replace("TS", "ts_start")))


def test_real_uploaders_pass():
    # tools/feeder.py: time.time() or video-timeline floats, --device-id default "feeder"
    FrameMeta.model_validate({"device_id": "feeder", "ts": time.time()})
    FrameMeta.model_validate({"device_id": "feeder", "ts": time.time() + 3600.0})  # --realtime 1 h video, fast feed
    # Unity FrameUploader/AudioChunkCodec ("0.000" seconds) and the native app (unix ms / 1000), ids like quest3-1
    AudioMeta.model_validate(json.loads(f'{{"device_id":"quest3-1","ts_start":{time.time():.3f}}}'))
    FrameMeta.model_validate({"device_id": "quest3-native-" + "a" * 32, "ts": int(time.time() * 1000) / 1000})
    # dataset replays (tools/egolife_eval.py) use past timestamps; unicode ids that fit are fine
    FrameMeta.model_validate({"device_id": "egolife-A1_JAKE", "ts": 1.0})
    VideoMeta.model_validate({"device_id": "Ärger 眼镜", "ts_start": time.time() + MAX_FUTURE_S - 60})
