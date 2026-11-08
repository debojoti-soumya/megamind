import asyncio

import pytest

from megamind.ingest.enrich import Enricher, Enrichment, ImageEmbedder, MockTextEmbedder
from megamind.ingest.worker import EnrichmentWorker
from megamind.models import MemoryToken
from megamind.store.memory_store import InMemoryStore


class _Image(ImageEmbedder):
    dim = 4

    def __init__(self, started: asyncio.Event | None = None, fail: bool = False) -> None:
        self.started, self.fail, self.calls = started, fail, 0

    async def embed_image(self, jpeg: bytes) -> list[float]:
        self.calls += 1
        if self.started is not None:
            self.started.set()
        await asyncio.sleep(0)
        if self.fail:
            raise RuntimeError("image embed down")
        return [0.0, 1.0, 0.0, 0.0]

    async def embed_text(self, text: str) -> list[float]:
        return [0.0, 1.0, 0.0, 0.0]


class _Enricher(Enricher):
    def __init__(self, wait_for: asyncio.Event | None = None, fail: bool = False) -> None:
        self.wait_for, self.fail = wait_for, fail

    async def enrich(self, jpeg: bytes) -> Enrichment:
        if self.wait_for is not None:  # only completes if the image embed runs concurrently
            await asyncio.wait_for(self.wait_for.wait(), timeout=2)
        if self.fail:
            raise RuntimeError("vlm down")
        return Enrichment(caption="red mug on desk", ocr_text="", entities=["mug"])


def _tok() -> MemoryToken:
    return MemoryToken(id="t1", device_id="d", ts=1.0, thumb_path="", frame_hash="h")


async def test_image_embedding_overlaps_enrichment():
    started = asyncio.Event()
    store = InMemoryStore()
    w = EnrichmentWorker(store, _Enricher(wait_for=started), MockTextEmbedder(), _Image(started))
    await w._process(_tok(), b"jpeg")
    got = await store.get("t1")
    assert got is not None and got.caption == "red mug on desk"
    assert got.image_vec == [0.0, 1.0, 0.0, 0.0] and got.text_vec is not None


@pytest.mark.parametrize(("enrich_fails", "image_fails"), [(True, False), (False, True), (True, True)])
async def test_failed_stage_stores_nothing(enrich_fails, image_fails):
    store = InMemoryStore()
    w = EnrichmentWorker(store, _Enricher(fail=enrich_fails), MockTextEmbedder(), _Image(fail=image_fails))
    with pytest.raises(RuntimeError):
        await w._process(_tok(), b"jpeg")
    await asyncio.sleep(0.01)
    assert await store.count() == 0


async def test_no_image_embedder_still_stores():
    store = InMemoryStore()
    await EnrichmentWorker(store, _Enricher(), MockTextEmbedder(), None)._process(_tok(), b"jpeg")
    got = await store.get("t1")
    assert got is not None and got.image_vec is None


class _SharingEnricher(Enricher):
    """Like LocalEnricher: embeds the JPEG itself with the worker's own image embedder."""

    def __init__(self, image_embedder: ImageEmbedder) -> None:
        self.image_embedder = image_embedder

    async def enrich(self, jpeg: bytes) -> Enrichment:
        v = await self.image_embedder.embed_image(jpeg)
        return Enrichment(caption="mug", ocr_text="", entities=[], image_vec=v, image_vec_from=self.image_embedder)


async def test_enricher_sharing_the_embedder_means_one_image_pass():
    store, img = InMemoryStore(), _Image()
    await EnrichmentWorker(store, _SharingEnricher(img), MockTextEmbedder(), img)._process(_tok(), b"jpeg")
    got = await store.get("t1")
    assert img.calls == 1 and got is not None and got.image_vec == [0.0, 1.0, 0.0, 0.0]


async def test_lag_stats_track_submit_to_store_and_pending():
    store = InMemoryStore()
    w = EnrichmentWorker(store, _Enricher(), MockTextEmbedder(), None, concurrency=3)
    assert w.lag_stats() == {"n": 0, "p50": None, "max": None, "oldest_pending": None}
    await w.submit(_tok(), b"jpeg")
    assert w.lag_stats()["oldest_pending"] is not None
    await w.start()
    try:
        assert len(w._tasks) == 3
        await asyncio.wait_for(w.queue.join(), timeout=2)
    finally:
        await w.stop()
    st = w.lag_stats()
    assert st["n"] == 1 and st["p50"] is not None and st["p50"] >= 0
    assert st["oldest_pending"] is None
    assert await store.count() == 1


async def test_discarded_frames_leave_no_pending_lag():
    w = EnrichmentWorker(InMemoryStore(), _Enricher(), MockTextEmbedder(), None)
    await w.submit(_tok(), b"jpeg")
    assert w.discard() == 1
    assert w.lag_stats()["oldest_pending"] is None
