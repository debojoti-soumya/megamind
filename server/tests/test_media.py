import io
import time

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from test_api import _post_frame
from test_episodes import BLUE, RED, _textured

from megamind.config import Settings
from megamind.main import create_app
from megamind.store.media import _pick_encoder

pytestmark = pytest.mark.skipif(_pick_encoder() is None, reason="ffmpeg with an AV1 encoder not installed")


@pytest.mark.parametrize("backend", ["memory", "lancedb"])
async def test_hot_cold_text_tiers(tmp_path, backend):
    settings = Settings(
        store_backend=backend,
        lancedb_path=tmp_path / "lancedb",
        thumb_dir=tmp_path / "thumbs",
        chunk_dir=tmp_path / "chunks",
        audio_dir=tmp_path / "audio",
        image_embed_provider="mock",
        filter_min_gap_s=0.0,
        filter_budget_per_min=0,
        age_interval_s=0,
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            t0 = time.time() - 3600
            frames = [(RED, t0 + i * 5) for i in range(4)] + [(BLUE, t0 + 30 + i * 5) for i in range(4)]
            ids = []
            for i, (color, ts) in enumerate(frames):
                r = await _post_frame(c, _textured(color, i, size=(320, 240)), ts)
                assert r.json()["accepted"], r.json()
                ids.append(r.json()["frame_id"])
            await app.state.worker.queue.join()
            eps = (await c.get("/v1/episodes")).json()["episodes"]
            reps = {e["rep_token_id"] for e in eps}
            assert len(eps) == 2

            r = (await c.post("/v1/admin/age", params={"hot_older_than_s": 0})).json()
            assert r["to_cold"] == 6 and r["chunks"] == 2 and r["chunk_bytes"] < r["jpeg_bytes"]
            st = (await c.get("/v1/stats")).json()
            assert st["tiers"] == {"hot": 2, "cold": 6} and st["filter"]["kept"] == 8
            for fid in ids:
                img = Image.open(io.BytesIO((await c.get(f"/v1/thumbs/{fid}.jpg")).content))
                assert img.size == (320, 240)
            hits = (await c.get("/v1/memories", params={"q": "blue", "mode": "visual"})).json()["results"]
            assert hits and all(h["episode_id"] == eps[1]["id"] for h in hits[:2])

            r = (await c.post("/v1/admin/age", params={"hot_older_than_s": 0, "cold_older_than_s": 0})).json()
            assert r["to_text"] == 6
            for fid in ids:
                code = (await c.get(f"/v1/thumbs/{fid}.jpg")).status_code
                assert code == (200 if fid in reps else 404)
            assert not list((tmp_path / "chunks").glob("*.mp4"))
            hits = (await c.get("/v1/memories", params={"q": "blue", "mode": "visual"})).json()["results"]
            assert hits[0]["episode_id"] == eps[1]["id"]

            assert (await c.delete("/v1/memories")).json()["deleted"] == 8
            assert not list((tmp_path / "thumbs").glob("*.jpg"))
