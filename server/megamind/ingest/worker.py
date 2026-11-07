"""Async enrichment worker: accepted frames -> enriched MemoryToken -> store."""

from __future__ import annotations

import asyncio
import logging

from megamind.ingest.enrich import Enricher, ImageEmbedder, TextEmbedder
from megamind.models import MemoryToken
from megamind.store.base import MemoryStore

log = logging.getLogger(__name__)


class EnrichmentWorker:
    def __init__(
        self,
        store: MemoryStore,
        enricher: Enricher,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder | None,
        concurrency: int = 2,
    ) -> None:
        self.store = store
        self.enricher = enricher
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.queue: asyncio.Queue[tuple[MemoryToken, bytes]] = asyncio.Queue(maxsize=200)
        self._tasks: list[asyncio.Task] = []
        self._concurrency = concurrency

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._run()) for _ in range(self._concurrency)]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()

    async def submit(self, token: MemoryToken, jpeg: bytes) -> None:
        if self.queue.full():
            self.queue.get_nowait()  # drop oldest under backpressure
        await self.queue.put((token, jpeg))

    async def _run(self) -> None:
        while True:
            token, jpeg = await self.queue.get()
            try:
                await self._process(token, jpeg)
            except Exception:  # noqa: BLE001
                log.exception("enrichment failed for %s", token.id)
            finally:
                self.queue.task_done()

    async def _process(self, token: MemoryToken, jpeg: bytes) -> None:
        e = await self.enricher.enrich(jpeg)
        token.caption, token.ocr_text, token.entities = e.caption, e.ocr_text, e.entities
        token.text_vec = await self.text_embedder.embed(f"{e.caption}\n{e.ocr_text}\n{' '.join(e.entities)}")
        if self.image_embedder is not None:
            token.image_vec = await self.image_embedder.embed_image(jpeg)
        await self.store.add(token)
