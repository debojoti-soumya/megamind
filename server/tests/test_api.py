import asyncio
import io
import json
from datetime import UTC, datetime

from httpx import ASGITransport, AsyncClient
from PIL import Image

from megamind.config import Settings
from megamind.main import create_app


def _jpeg(color, size=(96, 72)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _noisy_jpeg(seed: int) -> bytes:
    import random

    rnd = random.Random(seed)
    img = Image.new("RGB", (96, 72))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(96 * 72)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def _post_frame(c, jpeg, ts, device="dev1"):
    return await c.post(
        "/v1/frames",
        files={"image": ("f.jpg", jpeg, "image/jpeg")},
        data={"meta": json.dumps({"device_id": device, "ts": ts})},
    )


async def test_health(client):
    c, _ = client
    r = await c.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_frame_filter_and_search(client):
    c, app = client
    r = await _post_frame(c, _noisy_jpeg(1), 1000.0)
    assert r.status_code == 202 and r.json()["accepted"]

    r = await _post_frame(c, _noisy_jpeg(1), 1001.0)
    assert r.json() == {**r.json(), "accepted": False, "reason": "duplicate"}

    r = await _post_frame(c, _jpeg((0, 0, 0)), 1002.0)
    assert r.json()["reason"] == "dark"

    r = await _post_frame(c, _noisy_jpeg(2), 1003.0)
    assert r.json()["accepted"]

    await app.state.worker.queue.join()
    assert await app.state.store.count() == 2

    r = await c.get("/v1/memories", params={"q": "mock caption"})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2
    assert r.json()["results"][0]["thumb_url"].startswith("/v1/thumbs/")


async def test_chat_json_fallback(client):
    c, app = client
    await _post_frame(c, _noisy_jpeg(3), 2000.0)
    await app.state.worker.queue.join()
    r = await c.post("/v1/chat", json={"session_id": "s", "text": "mock caption"})
    assert r.status_code == 200
    body = r.json()
    assert body["tool_calls"][0]["name"] == "recall_memory"
    assert body["cited"]


async def test_chat_json_bad_body_is_422_not_500(client):
    c, _ = client
    r = await c.post("/v1/chat", json={"text": "hi"})  # missing session_id (issue #49)
    assert r.status_code == 422
    assert any(err["loc"] == ["session_id"] for err in r.json()["detail"])
    r = await c.post("/v1/chat", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and "invalid JSON" in r.json()["detail"]
    r = await c.post("/v1/chat", json=["hi"])
    assert r.status_code == 422
    # the connection is still usable afterwards (no unhandled exception)
    assert (await c.get("/v1/health")).status_code == 200


async def test_purge(client):
    c, app = client
    await _post_frame(c, _noisy_jpeg(4), 3000.0)
    await app.state.worker.queue.join()
    r = await c.delete("/v1/memories")
    assert r.json()["deleted"] == 1
    await asyncio.sleep(0)


async def test_purge_since_and_device(client):
    c, app = client
    for i, (ts, dev) in enumerate([(1000.0, "dev1"), (2000.0, "dev1"), (3000.0, "dev1"), (3000.0, "dev2")]):
        assert (await _post_frame(c, _noisy_jpeg(10 + i), ts, device=dev)).json()["accepted"]
    await app.state.worker.queue.join()
    assert await app.state.store.count() == 4

    for bad in ("yesterday", "", "   ", "nan", "inf"):
        r = await c.delete("/v1/memories", params={"device_id": "dev1", "since": bad})
        assert r.status_code == 400 and "since" in r.json()["detail"], bad
    r = await c.delete("/v1/memories?since=")
    assert r.status_code == 400
    assert await app.state.store.count() == 4

    r = await c.delete("/v1/memories", params={"device_id": "dev1", "since": "2500"})
    assert r.status_code == 200 and r.json() == {"deleted": 1}

    iso = datetime.fromtimestamp(1500.0, tz=UTC).isoformat()
    r = await c.delete("/v1/memories", params={"since": iso})
    assert r.json() == {"deleted": 2}
    remaining = await app.state.store.range(None, None)
    assert [(t.device_id, t.ts) for t in remaining] == [("dev1", 1000.0)]


async def test_search_bad_time_param_is_400(client):
    c, _ = client
    for param in ("from", "to"):
        r = await c.get("/v1/memories", params={"q": "x", param: "not-a-date"})
        assert r.status_code == 400 and param in r.json()["detail"]
    r = await c.get("/v1/memories", params={"q": "x", "from": "2026-11-07T09:00:00Z"})
    assert r.status_code == 200
    r = await c.get("/v1/health")  # connection still usable after the 400
    assert r.status_code == 200


async def _health(tmp_path, **cfg) -> dict:
    base = dict(thumb_dir=tmp_path / "t", audio_dir=tmp_path / "a", consolidate_enabled=False, _env_file=None)
    app = create_app(Settings(**{**base, **cfg}))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            return (await c.get("/v1/health")).json()["models"]


async def test_health_reports_mocks_when_keys_are_missing(tmp_path):
    models = await _health(
        tmp_path,
        anthropic_api_key="",
        reader_api_key="none",
        embed_provider="openai",
        openai_api_key="",
        stt_provider="deepgram",
        deepgram_api_key="",
        tts_provider="mock",
    )
    assert models == {
        "claude": "fallback (claude-sonnet-4-6 configured, no key)",
        "reader": "mock (vlm configured, no key)",
        "embed": "mock (openai configured, no key)",
        "stt": "mock (deepgram configured, no key)",
        "tts": "mock",
    }


async def test_health_reports_live_providers_and_unwired_ones(tmp_path):
    models = await _health(
        tmp_path,
        anthropic_api_key="k",
        reader_api_key="k",
        embed_provider="openai",
        openai_api_key="k",
        tts_provider="elevenlabs",
        elevenlabs_api_key="k",
    )
    assert models["claude"] == "claude-sonnet-4-6"
    assert models["reader"] == "google/gemma-4-31B-it"
    # keyed but the app still builds a mock for these roles: say so rather than claim the provider is live
    assert models["embed"] == "mock (openai configured, not implemented)"
    assert models["tts"] == "mock (elevenlabs configured, not implemented)"


async def test_health_says_not_installed_for_local_embeddings_without_torch(tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real(name, *a))
    models = await _health(tmp_path, embed_provider="local")
    assert models["embed"] == "mock (local configured, not installed)"
