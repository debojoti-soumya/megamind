"""LanceDB vector widths come from the configured embedders, not from whichever token is written first (#128)."""

import io
import json
import random

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from megamind.config import Settings
from megamind.main import create_app
from megamind.models import MemoryToken
from megamind.store.lancedb_store import LanceDBStore


def _noisy_jpeg(seed: int) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", (64, 48))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(64 * 48)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _settings(tmp_path, backend: str) -> Settings:
    return Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend=backend,
        lancedb_path=tmp_path / "lancedb",
        embed_provider="mock",
        image_embed_provider="mock",  # 48-d: not the 768-d LanceDB fallback
        filter_min_gap_s=0.0,
        enrich_provider="mock",
        consolidate_enabled=False,
        _env_file=None,
    )


async def _transcript(app, ts: float) -> None:
    await app.state.transcripts.add("d", ts, ts + 2, f"heard something at {ts}", None)


async def _frames(c, app, ts0: float, n: int = 4) -> None:
    for i in range(n):
        r = await c.post(
            "/v1/frames",
            files={"image": ("f.jpg", _noisy_jpeg(100 + i + int(ts0)), "image/jpeg")},
            data={"meta": json.dumps({"device_id": "d", "ts": ts0 + 8 * i})},
        )
        assert r.json()["accepted"], r.text
    await app.state.worker.queue.join()


@pytest.mark.parametrize("backend", ["memory", "lancedb"])
@pytest.mark.parametrize("order", ["audio_first", "frames_first"])
async def test_frames_are_stored_whatever_arrives_first(tmp_path, backend, order):
    app = create_app(_settings(tmp_path, backend))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            if order == "audio_first":
                await _transcript(app, 1000.0)
                await _frames(c, app, 1020.0)
            else:
                await _frames(c, app, 1020.0)
                await _transcript(app, 1100.0)
            health = (await c.get("/v1/health")).json()
            assert health["store_count"] == 5, health
            assert health["status"] == "ok" and health["models"]["store"] == backend
            frames = [t for t in await app.state.store.range(None, None, limit=10) if t.kind == "frame"]
            assert len(frames) == 4 and all(len(t.image_vec) == 48 for t in frames)
            r = await c.get("/v1/memories", params={"q": "red", "mode": "visual"})  # 48-d query vector
            assert r.status_code == 200 and r.json()["results"]


async def test_existing_mis_sized_table_is_degraded_in_health_not_silent(tmp_path):
    s = _settings(tmp_path, "lancedb")
    old = LanceDBStore(s.lancedb_path)  # pre-fix server: transcript first -> image_vec fell back to 768-d
    await old.add(
        MemoryToken(
            id="t1", device_id="d", ts=1.0, thumb_path="", frame_hash="", caption="hello", text_vec=[1.0] + [0.0] * 255
        )
    )
    app = create_app(s)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            health = (await c.get("/v1/health")).json()
            assert health["status"] == "degraded"
            assert "image_vec is 768-d but the configured embedder is 48-d" in health["models"]["store"]
            # a full purge recreates the table at the configured widths and clears the warning
            await c.delete("/v1/memories")
            await _frames(c, app, 2000.0, n=2)
            health = (await c.get("/v1/health")).json()
            assert health["status"] == "ok" and health["store_count"] == 2


def test_empty_mis_sized_table_is_recreated(tmp_path):
    LanceDBStore(tmp_path / "db", text_dim=4, image_dim=768)._ensure_table(
        MemoryToken(id="x", device_id="d", ts=0, thumb_path="", frame_hash="")
    )
    s = LanceDBStore(tmp_path / "db", text_dim=4, image_dim=48)
    assert s.degraded is None
    s._ensure_table(MemoryToken(id="x", device_id="d", ts=0, thumb_path="", frame_hash=""))
    assert s._table.schema.field("image_vec").type.list_size == 48


async def test_dim_mismatch_on_add_is_logged_and_degrades(tmp_path, caplog):
    s = LanceDBStore(tmp_path / "db", text_dim=4, image_dim=4)
    with pytest.raises(ValueError):
        await s.add(
            MemoryToken(
                id="x", device_id="d", ts=0, thumb_path="", frame_hash="", text_vec=[1.0] * 4, image_vec=[1.0] * 8
            )
        )
    assert "image_vec has 8 dims" in (s.degraded or "")
    assert any("refusing token x" in r.getMessage() for r in caplog.records)
