import asyncio
import io
import json

from PIL import Image


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


async def test_purge(client):
    c, app = client
    await _post_frame(c, _noisy_jpeg(4), 3000.0)
    await app.state.worker.queue.join()
    r = await c.delete("/v1/memories")
    assert r.json()["deleted"] == 1
    await asyncio.sleep(0)


async def test_episodes_endpoint(tmp_path):
    from httpx import ASGITransport, AsyncClient
    from test_episodes import BLUE, RED, _textured

    from megamind.config import Settings
    from megamind.main import create_app

    settings = Settings(
        thumb_dir=tmp_path / "thumbs",
        chunk_dir=tmp_path / "chunks",
        audio_dir=tmp_path / "audio",
        image_embed_provider="mock",
        filter_min_gap_s=0.0,
        filter_budget_per_min=0,
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            for i, (color, ts) in enumerate([(RED, 1000), (RED, 1010), (BLUE, 1020), (BLUE, 1030)]):
                r = await _post_frame(c, _textured(color, i), ts)
                assert r.json()["accepted"], r.json()
            await app.state.worker.queue.join()
            eps = (await c.get("/v1/episodes")).json()["episodes"]
            assert len(eps) == 2 and eps[0]["start_ts"] == 1000 and eps[0]["n_tokens"] == 2
            ranked = (await c.get("/v1/episodes", params={"q": "blue"})).json()["episodes"]
            assert ranked[0]["id"] == eps[1]["id"] and ranked[0]["top_hits"]
            hits = (await c.get("/v1/memories", params={"q": "blue", "mode": "visual"})).json()["results"]
            assert hits[0]["episode_id"] == eps[1]["id"]
