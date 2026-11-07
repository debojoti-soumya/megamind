import asyncio
import time

from megamind.config import Settings
from megamind.ingest.enrich import build_text_embedder
from megamind.main import create_app
from megamind.models import MemoryToken
from megamind.store import build_store
from megamind.store.consolidate import summary_id


def _settings(tmp_path, **kw) -> Settings:
    return Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend="lancedb",
        lancedb_path=tmp_path / "lancedb",
        embed_provider="mock",
        _env_file=None,
        **kw,
    )


async def _seed_closed_episode(settings: Settings) -> str:
    store = build_store(settings)
    vec = await build_text_embedder(settings).embed("desk")  # ingest always embeds before add (fixes table dims)
    t0 = time.time() - 3600
    frames = [
        MemoryToken(
            id=f"f{i}",
            device_id="dev1",
            ts=t0 + 10 * i,
            caption=f"desk scene {i}",
            thumb_path=f"/nonexistent/f{i}.jpg",
            frame_hash=f"{i:016x}",
            text_vec=vec,
        )
        for i in range(4)
    ]
    await store.add_many(frames)
    (first,) = await store.get_many(["f0"])
    assert first.episode_id
    return first.episode_id


async def test_lifespan_runs_consolidator_and_cancels_on_shutdown(tmp_path):
    settings = _settings(tmp_path, consolidate_interval_s=3600)
    episode_id = await _seed_closed_episode(settings)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        st = app.state
        task = st.consolidate_task
        assert task is not None and not task.done()
        for _ in range(100):
            if await st.store.get(summary_id(episode_id)):
                break
            await asyncio.sleep(0.05)
        summary = await st.store.get(summary_id(episode_id))
        assert summary is not None and summary.kind == "episode"
        assert summary.text_vec is not None
    assert task.cancelled()


async def test_consolidator_disabled(tmp_path):
    settings = _settings(tmp_path, consolidate_enabled=False)
    episode_id = await _seed_closed_episode(settings)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        st = app.state
        assert st.consolidate_task is None
        await asyncio.sleep(0.1)
        assert await st.store.get(summary_id(episode_id)) is None
        written = await st.consolidator.run_once()
        assert [t.id for t in written] == [summary_id(episode_id)]
