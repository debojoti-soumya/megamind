"""main.py must build embedders via megamind.store.embeddings (issue #71), not the enrich.py stubs."""

import pytest

from megamind.config import Settings
from megamind.ingest.enrich import MockTextEmbedder
from megamind.main import create_app
from megamind.store.embeddings import MockImageEmbedder, OpenAITextEmbedder


def _settings(tmp_path, **kw) -> Settings:
    base = {
        "thumb_dir": tmp_path / "thumbs",
        "audio_dir": tmp_path / "audio",
        "store_backend": "memory",
        "consolidate_enabled": False,
        "openai_api_key": "",
        "_env_file": None,
    }
    return Settings(**(base | kw))


async def _built(settings: Settings):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        return app.state.text_embedder, app.state.image_embedder


async def test_zero_env_is_mock_text_and_no_image(tmp_path):
    text, image = await _built(_settings(tmp_path))
    assert type(text) is MockTextEmbedder
    assert image is None


async def test_openai_provider_with_key_builds_openai_embedder(tmp_path):
    text, _ = await _built(_settings(tmp_path, embed_provider="openai", openai_api_key="sk-test-not-a-real-key"))
    assert isinstance(text, OpenAITextEmbedder)  # constructed only; no request is made


async def test_openai_provider_without_key_falls_back_to_mock(tmp_path):
    text, _ = await _built(_settings(tmp_path, embed_provider="openai"))
    assert type(text) is MockTextEmbedder


async def test_image_embed_provider_is_honoured(tmp_path):
    _, image = await _built(_settings(tmp_path, image_embed_provider="mock"))
    assert isinstance(image, MockImageEmbedder)


@pytest.mark.parametrize("backend", ["memory", "lancedb"])
async def test_bad_openai_key_degrades_instead_of_500(tmp_path, backend):
    import io
    import json

    import httpx
    from httpx import ASGITransport, AsyncClient
    from PIL import Image

    settings = _settings(
        tmp_path,
        store_backend=backend,
        lancedb_path=tmp_path / "lancedb",
        embed_provider="openai",
        openai_api_key="sk-bad",
        filter_min_gap_s=0.0,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        emb = app.state.text_embedder
        assert isinstance(emb, OpenAITextEmbedder)
        await emb._client.aclose()
        emb._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}})),
        )
        img = Image.effect_noise((96, 72), 64).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/v1/frames",
                files={"image": ("f.jpg", buf.getvalue(), "image/jpeg")},
                data={"meta": json.dumps({"device_id": "dev1", "ts": 1000.0})},
            )
            assert r.json()["accepted"]
            await app.state.worker.queue.join()
            assert await app.state.store.count() == 1  # frame stored despite the embed failure

            r = await c.get("/v1/memories", params={"q": "mock caption"})
            assert r.status_code == 200 and r.json()["results"]

            r = await c.post("/v1/chat", json={"session_id": "s", "text": "mock caption"})
            assert r.status_code == 200

            embed = (await c.get("/v1/health")).json()["models"]["embed"]
            assert embed == "openai:text-embedding-3-small (degraded: last error 401)"


async def test_store_image_calibration_is_none_without_siglip(tmp_path):
    app = create_app(_settings(tmp_path, image_embed_provider="mock"))
    async with app.router.lifespan_context(app):
        assert app.state.store.image_calibration is None  # mock image tower: uncalibrated, today's ranking
