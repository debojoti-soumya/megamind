"""Tiered media (store/media.py): hot JPEG -> cold AV1 chunk -> text, purge chunk rewrite, /v1/stats.

Tests that need a real AV1 encoder skip without ffmpeg + libsvtav1/libaom-av1; the rest fake the encoder."""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from megamind.config import Settings
from megamind.main import create_app
from megamind.models import MemoryToken
from megamind.store import media as media_mod
from megamind.store.consolidate import Consolidator, MockSummarizer
from megamind.store.media import MediaStore
from megamind.store.memory_store import InMemoryStore

HAVE_AV1 = media_mod._pick_encoder(media_mod._find_ffmpeg()) is not None
needs_av1 = pytest.mark.skipif(not HAVE_AV1, reason="ffmpeg with an AV1 encoder (libsvtav1/libaom-av1) not installed")
T0 = round(time.time()) - 3600.0  # older than "now", well inside COLD_RETENTION_DAYS


def _frame(seed: int, size=(320, 240)) -> bytes:
    """Distinct, sharp, low-entropy frame (passes the blur and dedupe filters, encodes small)."""
    rnd = np.random.default_rng(seed)
    blocks = rnd.integers(0, 256, (size[1] // 16, size[0] // 16, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blocks).resize(size, Image.Resampling.NEAREST).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _psnr(a: bytes, b: bytes) -> float:
    x = np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), dtype=float)
    y = Image.open(io.BytesIO(b)).convert("RGB")
    y = np.asarray(y.resize((x.shape[1], x.shape[0])), dtype=float)
    mse = ((x - y) ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255**2 / mse)


def _best_match(decoded: bytes, sources: list[bytes]) -> int:
    return int(np.argmax([_psnr(s, decoded) for s in sources]))


@pytest.fixture(params=["memory", "lancedb"])
async def app_client(request, tmp_path):
    settings = Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        chunk_dir=tmp_path / "chunks",
        store_backend=request.param,
        lancedb_path=tmp_path / "lancedb",
        embed_provider="mock",
        filter_min_gap_s=0.0,
        consolidate_enabled=False,
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app


async def _ingest(c, app, jpegs: list[bytes], t0: float = T0, step: float = 5.0) -> list[str]:
    ids = []
    for i, j in enumerate(jpegs):
        r = await c.post(
            "/v1/frames",
            files={"image": ("f.jpg", j, "image/jpeg")},
            data={"meta": json.dumps({"device_id": "dev1", "ts": t0 + i * step})},
        )
        assert r.json()["accepted"], r.json()
        ids.append(r.json()["frame_id"])
    await app.state.worker.queue.join()
    return ids


# ---------------------------------------------------------------------- model


def test_thumb_url_by_tier():
    base = dict(device_id="d", ts=1.0, frame_hash="0" * 16)
    assert MemoryToken(id="h", thumb_path="/x/h.jpg", **base).thumb_url == "/v1/thumbs/h.jpg"
    cold = MemoryToken(id="c", thumb_path="", tier="cold", chunk_id="ck_1", chunk_offset=0, **base)
    assert cold.thumb_url == "/v1/thumbs/c.jpg"
    assert MemoryToken(id="t", thumb_path="", tier="text", **base).thumb_url == ""
    assert MemoryToken(id="s", thumb_path="", kind="transcript", **base).thumb_url == ""
    assert MemoryToken(id="e", thumb_path="", **base).thumb_url == ""  # hot without a file: unchanged behaviour


async def test_live_now_frame_reads_through_media_store(tmp_path):
    """Main's live short-term memory (#98) hands the reader unstored `LiveFrameToken`s from its own cache;
    with tiered media the reader loads every image through `MediaStore.read`, so those must still load."""
    from megamind.live import LiveFrameToken

    jpeg = _frame(7)
    p = tmp_path / "live" / "now_abc.jpg"
    p.parent.mkdir()
    p.write_bytes(jpeg)
    media = MediaStore(Settings(_env_file=None, thumb_dir=tmp_path / "thumbs"), InMemoryStore())
    tok = LiveFrameToken(id="now_abc", device_id="d", ts=T0, thumb_path=str(p), frame_hash="", kind="frame")
    assert tok.thumb_url == "/v1/live/now_abc.jpg"
    assert await media.read(tok) == jpeg
    p.unlink()
    assert await media.read(tok) is None  # evicted from the live cache: no store lookup hit, no crash


# ---------------------------------------------------------------------- no encoder


async def test_without_encoder_aging_skips_cold_but_drops_to_text(app_client, monkeypatch):
    c, app = app_client
    monkeypatch.setattr(media_mod, "_pick_encoder", lambda *_: None)
    ids = await _ingest(c, app, [_frame(i) for i in range(3)])
    before = await app.state.store.get(ids[0])
    assert before.text_vec is not None

    r = (await c.post("/v1/admin/age", params={"hot_older_than_s": 0})).json()
    assert r["skipped"] and r["to_cold"] == 0 and r["to_text"] == 0
    assert (await c.get(f"/v1/thumbs/{ids[0]}.jpg")).status_code == 200

    r = (await c.post("/v1/admin/age", params={"hot_older_than_s": 0, "cold_older_than_s": 0})).json()
    assert r["to_text"] == 3
    after = await app.state.store.get(ids[0])  # the planning scan skips vectors; the rewrite must not drop them
    assert after.tier == "text" and after.text_vec == before.text_vec and after.image_vec == before.image_vec
    assert list((app.state.settings.thumb_dir).glob("*.jpg")) == []
    assert (await c.get(f"/v1/thumbs/{ids[0]}.jpg")).status_code == 404
    hits = (await c.get("/v1/memories", params={"q": "mock caption"})).json()["results"]
    assert len(hits) == 3 and all(h["thumb_url"] == "" for h in hits)  # still searchable, no image
    st = (await c.get("/v1/stats")).json()
    assert st["tiers"] == {"hot": 0, "cold": 0, "text": 3} and st["media"]["encoder"] is None
    assert (await c.post("/v1/admin/age", params={"hot_older_than_s": -1})).status_code == 400


async def test_stats_shape(app_client):
    c, app = app_client
    await _ingest(c, app, [_frame(i) for i in range(2)])
    st = (await c.get("/v1/stats")).json()
    assert st["tokens"] == 2 and st["frames"] == 2 and st["tiers"] == {"hot": 2, "cold": 0, "text": 0}
    assert st["filter"]["seen"] == 2 and st["filter"]["kept"] == 2 and st["filter"]["keep_rate"] == 1.0
    assert st["worker_queue"] == 0
    assert set(st["disk_bytes"]) == {"hot_jpeg", "cold_chunks", "index"} and st["disk_bytes"]["hot_jpeg"] > 0
    assert st["bytes_per_frame"]["hot"] > 0 and st["bytes_per_frame"]["cold"] is None
    assert st["projected_100d_gb_at_budget"] > 0


# ---------------------------------------------------------------------- real AV1


@needs_av1
async def test_hot_cold_text_round_trip(app_client):
    c, app = app_client
    src = [_frame(i) for i in range(4)]
    ids = await _ingest(c, app, src)
    stale = await app.state.store.get(ids[1])  # fetched while hot

    r = (await c.post("/v1/admin/age", params={"hot_older_than_s": 0})).json()
    assert r["to_cold"] == 4 and r["chunks"] == 1 and 0 < r["chunk_bytes"] < r["jpeg_bytes"], r
    cold = await app.state.store.get(ids[1])
    assert cold.text_vec == stale.text_vec and cold.image_vec == stale.image_vec  # vectors survive the rewrite
    assert list(app.state.settings.thumb_dir.glob("*.jpg")) == []
    st = (await c.get("/v1/stats")).json()
    assert st["tiers"] == {"hot": 0, "cold": 4, "text": 0} and st["bytes_per_frame"]["cold"] > 0

    hits = (await c.get("/v1/memories", params={"q": "mock caption"})).json()["results"]
    assert all(h["thumb_url"].startswith("/v1/thumbs/") for h in hits)
    media = app.state.media
    for i, fid in enumerate(ids):
        r = await c.get(f"/v1/thumbs/{fid}.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert _best_match(r.content, src) == i and _psnr(src[i], r.content) > 25
    n = media.decodes
    assert (await c.get(f"/v1/thumbs/{ids[0]}.jpg")).status_code == 200
    assert media.decodes == n  # LRU hit
    assert _best_match(await media.read(stale), src) == 1  # stale hot token re-read from the store

    r = (await c.post("/v1/admin/age", params={"cold_older_than_s": 0})).json()
    assert r["to_text"] == 4
    assert list(media.chunk_dir.glob("*.mp4")) == []
    assert (await c.get(f"/v1/thumbs/{ids[0]}.jpg")).status_code == 404


@needs_av1
async def test_purge_rewrites_chunk_keeping_survivors(app_client):
    c, app = app_client
    src = [_frame(10 + i) for i in range(3)]
    ids = await _ingest(c, app, src)
    await c.post("/v1/admin/age", params={"hot_older_than_s": 0})
    media, store = app.state.media, app.state.store
    old_chunk = (await store.get(ids[0])).chunk_id
    for fid in ids:  # warm the LRU, which must not keep the purged frame either
        await c.get(f"/v1/thumbs/{fid}.jpg")

    r = await c.delete("/v1/memories", params={"since": str(T0 + 10)})  # the last frame only
    assert r.json()["deleted"] == 1
    assert not media.chunk_path(old_chunk).exists()
    chunks = list(media.chunk_dir.glob("*.mp4"))
    assert len(chunks) == 1 and media._count_frames(chunks[0]) == 2
    assert not any(k.startswith(old_chunk) for k in media._cache)
    for i in (0, 1):
        t = await store.get(ids[i])
        assert t.tier == "cold" and t.chunk_id != old_chunk and t.chunk_offset == i
        assert _best_match((await c.get(f"/v1/thumbs/{ids[i]}.jpg")).content, src) == i
    assert (await c.get(f"/v1/thumbs/{ids[2]}.jpg")).status_code == 404
    decoded = [Image.open(io.BytesIO(b)) for b in media._decode_all(chunks[0])]
    assert all(_psnr(src[2], _jpeg(im)) < _psnr(src[i], _jpeg(decoded[i])) for i, im in enumerate(decoded))


def _jpeg(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@needs_av1
async def test_purge_without_encoder_demotes_survivors_to_text(app_client):
    c, app = app_client
    ids = await _ingest(c, app, [_frame(20 + i) for i in range(3)])
    await c.post("/v1/admin/age", params={"hot_older_than_s": 0})
    app.state.media._encoder = None  # e.g. the encoder disappeared after an upgrade
    await c.delete("/v1/memories", params={"since": str(T0 + 10)})
    assert list(app.state.media.chunk_dir.glob("*.mp4")) == []
    assert [(await app.state.store.get(i)).tier for i in ids[:2]] == ["text", "text"]


# ---------------------------------------------------------------------- races (fake encoder, no ffmpeg)


def _media(tmp_path: Path, store) -> MediaStore:
    s = SimpleNamespace(
        thumb_dir=tmp_path / "thumbs", chunk_dir=None, lancedb_path=tmp_path / "l", store_backend="memory",
        hot_retention_h=72, cold_retention_days=100, cold_width=960, cold_crf=50,
    )  # fmt: skip
    s.thumb_dir.mkdir(parents=True, exist_ok=True)
    return MediaStore(s, store)


async def _seed(store, tmp_path: Path, n: int) -> list[MemoryToken]:
    toks = []
    for i in range(n):
        p = tmp_path / "thumbs" / f"t{i}.jpg"
        p.write_bytes(_frame(i, (64, 48)))
        toks.append(MemoryToken(id=f"t{i}", device_id="d", ts=T0 + i, thumb_path=str(p), frame_hash="0" * 16))
    await store.add_many(toks)
    return toks


async def test_aging_concurrent_with_purge_does_not_resurrect(tmp_path):
    store = InMemoryStore()
    m = _media(tmp_path, store)
    await _seed(store, tmp_path, 3)
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow_encode(jpegs, out):
        loop.call_soon_threadsafe(started.set)
        time.sleep(0.3)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")
        return len(jpegs)

    m._ffmpeg, m._encoder, m._encode = "ffmpeg", ["fake"], slow_encode
    task = asyncio.create_task(m.age(hot_s=0))
    await started.wait()
    async with m.lock:  # what EnrichmentWorker.purge does
        await m.purge_window()
        assert await store.purge() == 3
    stats = await task
    assert stats["aborted"] == 1 and stats["to_cold"] == 0
    assert await store.count() == 0
    assert list(m.chunk_dir.glob("*.mp4")) == []  # the chunk holding purged pixels was discarded


async def test_to_text_waiting_on_purge_does_not_resurrect(tmp_path):
    store = InMemoryStore()
    m = _media(tmp_path, store)
    await _seed(store, tmp_path, 2)
    async with m.lock:
        task = asyncio.create_task(m.age(hot_s=0, cold_s=0))
        await asyncio.sleep(0.05)  # aging has listed the tokens and now waits for the lock
        await store.purge()
    await task
    assert await store.count() == 0


async def test_decode_racing_a_chunk_removal_is_not_cached(tmp_path):
    """A cold decode that finishes after a purge removed its chunk must not leave the pixels in the LRU."""
    m = _media(tmp_path, InMemoryStore())
    m._ffmpeg = "ffmpeg"
    m.chunk_dir.mkdir(parents=True)
    path = m.chunk_path("ck_gone")
    path.write_bytes(b"fake")

    async def decode_while_purged(p, offset):
        p.unlink()  # the purge rewrites/deletes the chunk while ffmpeg is still decoding
        return b"pixels"

    m._decode_one = decode_while_purged
    assert await m._read_cold("ck_gone", 0) == b"pixels"
    assert m._cache == {}


async def test_lancedb_list_tokens_can_skip_vectors(tmp_path):
    pytest.importorskip("lancedb")
    from megamind.store.lancedb_store import LanceDBStore

    s = LanceDBStore(tmp_path / "db", text_dim=4, image_dim=4)
    tok = MemoryToken(
        id="a", device_id="d", ts=1.0, thumb_path="/x/a.jpg", frame_hash="0", text_vec=[1, 0, 0, 0],
        image_vec=[0, 1, 0, 0],
    )  # fmt: skip
    await s.add(tok)
    [light] = await s.list_tokens(vectors=False)
    [full] = await s.list_tokens()
    assert light.id == "a" and light.text_vec is None and light.image_vec is None and light.tier == "hot"
    assert full.text_vec == [1, 0, 0, 0] and full.image_vec == [0, 1, 0, 0]


async def test_rewrite_does_not_touch_live_episode_window(tmp_path):
    from megamind.store.episodes import EpisodeTracker

    store = InMemoryStore(episodes=EpisodeTracker())
    m = _media(tmp_path, store)
    old = await _seed(store, tmp_path, 2)
    live = MemoryToken(id="live", device_id="d", ts=T0 + 10_000, thumb_path="", frame_hash="0" * 16)
    await store.add(live)
    before = [ep.id for ep in store.episodes.open_episodes("d")]
    await store.rewrite_many([t.model_copy(update={"tier": "text"}) for t in old])
    assert [ep.id for ep in store.episodes.open_episodes("d")] == before
    assert (await store.get("t0")).tier == "text"
    assert m.disk_bytes()["hot_jpeg"] > 0


# ---------------------------------------------------------------------- consumers


async def test_consolidator_reads_cold_representative_through_loader(tmp_path):
    jpeg = _frame(1)
    frames = [
        MemoryToken(
            id=f"f{i}", device_id="d", ts=T0 + i, caption=f"c{i}", thumb_path="", frame_hash="0" * 16,
            episode_id="ep1", tier="cold", chunk_id="ck_x", chunk_offset=i,
        )
        for i in range(3)
    ]  # fmt: skip

    async def loader(t: MemoryToken) -> bytes | None:
        return jpeg if t.chunk_id == "ck_x" else None

    job = Consolidator(InMemoryStore(), MockSummarizer(), image_loader=loader, thumb_dir=tmp_path / "thumbs")
    tok = await job.summarize_episode(frames)
    assert tok.kind == "episode" and tok.tier == "hot" and Path(tok.thumb_path).read_bytes() == jpeg


async def test_lancedb_migrates_old_tables(tmp_path):
    pytest.importorskip("lancedb")
    import lancedb
    import pyarrow as pa

    from megamind.store.lancedb_store import TABLE_NAME, LanceDBStore

    s = LanceDBStore(tmp_path / "db", text_dim=4, image_dim=4)
    old_schema = pa.schema([f for f in s._schema() if f.name not in ("tier", "chunk_id", "chunk_offset")])
    row = {
        "id": "old", "device_id": "d", "ts": 1.0, "kind": "frame", "caption": "a mug", "ocr_text": "", "entities": [],
        "search_text": "a mug", "pose": None, "thumb_path": "/x/old.jpg", "frame_hash": "0", "episode_id": None,
        "text_vec": None, "image_vec": None,
    }  # fmt: skip
    lancedb.connect(str(tmp_path / "db")).create_table(TABLE_NAME, pa.Table.from_pylist([row], schema=old_schema))

    s = LanceDBStore(tmp_path / "db")
    t = await s.get("old")
    assert t.tier == "hot" and t.chunk_id is None and t.thumb_url == "/v1/thumbs/old.jpg"
    await s.rewrite_many([t.model_copy(update={"tier": "cold", "chunk_id": "ck_1", "chunk_offset": 0})])
    assert [x.id for x in await s.list_tokens(tier="cold", chunk_id="ck_1")] == ["old"]
    assert await s.tier_counts() == {"hot": 0, "cold": 1, "text": 0}


def test_blank_chunk_dir_means_default(monkeypatch, tmp_path):
    monkeypatch.setenv("CHUNK_DIR", "")
    s = Settings(thumb_dir=tmp_path / "thumbs", _env_file=None)
    assert s.chunk_dir is None
    assert MediaStore(s, InMemoryStore()).chunk_dir == tmp_path / "chunks"


def test_decode_all_falls_back_to_vsync_on_old_ffmpeg(tmp_path, monkeypatch):
    """ffmpeg < 5.1 has no `-fps_mode`: retry once with `-vsync passthrough`, then skip the probe (#114)."""
    settings = Settings(thumb_dir=tmp_path / "thumbs", chunk_dir=tmp_path / "chunks")
    media = MediaStore(settings, InMemoryStore())
    media._ffmpeg = "ffmpeg"
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "-fps_mode" in cmd:
            err = b"Unrecognized option 'fps_mode'."
            return SimpleNamespace(returncode=1, stderr=err, check_returncode=lambda: None)
        Path(cmd[-1].replace("%06d", "000001")).write_bytes(b"jpg")
        return SimpleNamespace(returncode=0, stderr=b"", check_returncode=lambda: None)

    monkeypatch.setattr(media_mod.subprocess, "run", fake_run)
    assert media._decode_all(tmp_path / "ck.mp4") == [b"jpg"]
    assert media._decode_all(tmp_path / "ck.mp4") == [b"jpg"]
    assert ["-fps_mode" in c for c in calls] == [True, False, False]
    assert all("-vsync" in c for c in calls[1:])
