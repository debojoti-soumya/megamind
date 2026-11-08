"""GET /v1/devices and the chat "device has no memories" warning / CHAT_DEFAULT_DEVICE=latest fallback (mock mode)."""

from __future__ import annotations

import io
import json
import random

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from megamind.config import Settings
from megamind.main import create_app


def _noisy_jpeg(seed: int) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", (96, 72))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(96 * 72)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def _frame(c, seed: int, ts: float, device: str, transcript: str | None = None):
    meta = {"device_id": device, "ts": ts, "transcript": transcript}
    r = await c.post(
        "/v1/frames", files={"image": ("f.jpg", _noisy_jpeg(seed), "image/jpeg")}, data={"meta": json.dumps(meta)}
    )
    assert r.status_code == 202 and r.json()["accepted"]


async def _seed(c, app):
    await _frame(c, 1, 1000.0, "old-dev")
    await _frame(c, 2, 2000.0, "quest3-1")
    await _frame(c, 3, 2010.0, "quest3-1", transcript="the wifi password is hunter2")
    await app.state.worker.queue.join()


async def test_devices_empty(client):
    c, _ = client
    r = await c.get("/v1/devices")
    assert r.status_code == 200 and r.json() == {"devices": []}


async def test_devices_sorted_by_recency_with_counts(client):
    c, app = client
    await _seed(c, app)
    devs = (await c.get("/v1/devices")).json()["devices"]
    assert [d["device_id"] for d in devs] == ["quest3-1", "old-dev"]
    q, old = devs
    assert q["count"] == await app.state.store.count("quest3-1") == 3  # 2 frames + 1 transcript
    assert q["last_frame_ts"] == pytest.approx(2010.0) and q["last_audio_ts"] == pytest.approx(2010.0)
    assert q["last_ts"] == pytest.approx(2010.0)
    assert old == {
        "device_id": "old-dev",
        "count": 1,
        "last_ts": 1000.0,
        "last_frame_ts": 1000.0,
        "last_audio_ts": None,
    }


async def test_chat_warns_when_device_has_no_memories(client):
    c, app = client
    await _seed(c, app)
    body = (await c.post("/v1/chat", json={"session_id": "s", "device_id": "browser-x", "text": "mock caption"})).json()
    assert body["cited"] == [] and body["device_id_used"] == "browser-x"
    assert body["warning"] == "device 'browser-x' has no memories; active devices: quest3-1, old-dev"

    body = (await c.post("/v1/chat", json={"session_id": "s", "device_id": "quest3-1", "text": "mock caption"})).json()
    assert body["cited"] and body["warning"] is None and body["device_id_used"] == "quest3-1"

    body = (await c.post("/v1/chat", json={"session_id": "s", "text": "mock caption"})).json()  # no device = all
    assert body["cited"] and body["warning"] is None and body["device_id_used"] is None


async def test_chat_no_warning_when_store_is_empty(client):
    c, _ = client
    body = (await c.post("/v1/chat", json={"session_id": "s", "device_id": "browser-x", "text": "keys"})).json()
    assert body["warning"] is None and body["device_id_used"] == "browser-x"


async def test_chat_multipart_warns_too(client):
    c, app = client
    await _seed(c, app)
    r = await c.post(
        "/v1/chat",
        data={"session_id": "s", "device_id": "browser-x"},
        files={"audio": ("q.wav", b"RIFF0000WAVE", "audio/wav")},
    )
    assert r.status_code == 200 and r.json()["warning"].startswith("device 'browser-x' has no memories")


def _sse_events(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


async def test_chat_stream_final_carries_warning(client):
    c, app = client
    await _seed(c, app)
    r = await c.post("/v1/chat/stream", json={"session_id": "s", "device_id": "browser-x", "text": "mock caption"})
    name, final = _sse_events(r.text)[-1]
    assert name == "final" and final["warning"].startswith("device 'browser-x' has no memories")
    assert final["device_id_used"] == "browser-x"


@pytest.fixture(params=["memory", "lancedb"])
async def latest_client(request, tmp_path):
    settings = Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend=request.param,
        lancedb_path=tmp_path / "lancedb",
        embed_provider="mock",
        filter_min_gap_s=0.0,
        enrich_provider="mock",
        consolidate_enabled=False,
        chat_default_device="latest",
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app


@pytest.mark.parametrize("path", ["/v1/chat", "/v1/chat/stream"])
async def test_chat_default_device_latest_falls_back(latest_client, path):
    c, app = latest_client
    await _seed(c, app)
    r = await c.post(path, json={"session_id": "s", "device_id": "browser-x", "text": "mock caption"})
    body = _sse_events(r.text)[-1][1] if path.endswith("stream") else r.json()
    assert body["device_id_used"] == "quest3-1"
    assert body["warning"] == "device 'browser-x' has no memories; answered from 'quest3-1' instead"
    tokens = await app.state.store.get_many([cit["id"] for cit in body["cited"]])
    assert tokens and {t.device_id for t in tokens} == {"quest3-1"}

    body = (await c.post("/v1/chat", json={"session_id": "s", "device_id": "old-dev", "text": "mock caption"})).json()
    assert body["device_id_used"] == "old-dev" and body["warning"] is None  # a device with memories is never moved
