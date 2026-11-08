import pytest
from httpx import ASGITransport, AsyncClient

from megamind.config import Settings
from megamind.main import create_app


@pytest.fixture(autouse=True)
def _no_startup_warmup(monkeypatch):
    """No background provider calls from apps built in tests (tests/test_warmup.py opts in with warmup=True)."""
    monkeypatch.setenv("WARMUP", "false")


@pytest.fixture(params=["memory", "lancedb"])
async def client(request, tmp_path):
    settings = Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend=request.param,
        lancedb_path=tmp_path / "lancedb",
        embed_provider="mock",
        filter_min_gap_s=0.0,
        enrich_provider="mock",  # deterministic captions even where tesseract / a reader key is installed
        consolidate_enabled=False,
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app
