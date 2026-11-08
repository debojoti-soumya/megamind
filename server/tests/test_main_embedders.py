"""main.py must build embedders via megamind.store.embeddings (issue #71), not the enrich.py stubs."""

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
