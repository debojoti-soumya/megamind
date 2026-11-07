import pytest
from httpx import ASGITransport, AsyncClient

from megamind.config import Settings
from megamind.main import create_app


@pytest.fixture
async def client(tmp_path):
    settings = Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend="memory",
        embed_provider="mock",
        filter_min_gap_s=0.0,
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app
