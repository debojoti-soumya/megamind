"""Async enrichment worker: accepted frames -> enriched MemoryToken -> store."""

from __future__ import annotations

import asyncio
import logging

from megamind.ingest.enrich import Enricher, ImageEmbedder, TextEmbedder
from megamind.ingest.episodes import EpisodeTracker
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
        episodes: EpisodeTracker | None = None,
        concurrency: int = 2,
    ) -> None:
        self.episodes = episodes
        self.store = store
        self.enricher = enricher
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.queue: asyncio.Queue[tuple[MemoryToken, bytes]] = asyncio.Queue(maxsize=200)
        self._tasks: list[asyncio.Task] = []
        self._last: dict[str, MemoryToken] = {}  # device -> last submitted token (speech merge target)
        self._pending: set[str] = set()
        self._concurrency = concurrency

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._run()) for _ in range(self._concurrency)]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()

    async def submit(self, token: MemoryToken, jpeg: bytes) -> None:
        if self.queue.full():
            dropped, _ = self.queue.get_nowait()  # drop oldest under backpressure
            self._pending.discard(dropped.id)
        self._last[token.device_id] = token
        self._pending.add(token.id)
        await self.queue.put((token, jpeg))

    def can_merge(self, device_id: str) -> bool:
        return device_id in self._last

    async def attach_transcript(self, device_id: str, text: str) -> str | None:
        """Append speech to the device's last kept token; re-embed it if it is already stored."""
        t = self._last.get(device_id)
        if t is None:
            return None
        t.transcript = f"{t.transcript} {text}".strip() if t.transcript else text
        if t.id in self._pending:
            return t.id  # _process embeds the final text
        stored = await self.store.get(t.id)
        if stored is None:
            return None
        stored.transcript = t.transcript
        stored.text_vec = await self.text_embedder.embed(stored.search_text())
        await self.store.add(stored)
        return t.id

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
        token.enriched = bool(e.caption or e.ocr_text)
        text = token.search_text()
        token.text_vec = await self.text_embedder.embed(text) if text else None
        if self.image_embedder is not None:
            token.image_vec = await self.image_embedder.embed_image(jpeg)
        episode = await self.episodes.assign(token) if self.episodes is not None else None
        while token.search_text() != text:  # speech merged in while we were embedding
            text = token.search_text()
            token.text_vec = await self.text_embedder.embed(text)
        self._pending.discard(token.id)
        await self.store.add(token)
        if episode is not None:
            await self.store.upsert_episode(episode)
