"""Async enrichment worker: accepted frames -> enriched MemoryToken -> store."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from megamind.ingest.enrich import Enricher, ImageEmbedder, TextEmbedder
from megamind.ingest.privacy import FaceBlur
from megamind.models import MemoryToken
from megamind.store.base import MemoryStore

log = logging.getLogger(__name__)


def _in_window(device_id: str, ts: float, purge_device: str | None, since_ts: float | None) -> bool:
    return (purge_device is None or device_id == purge_device) and (since_ts is None or ts >= since_ts)


def _unlink(path: str) -> None:
    if path:
        Path(path).unlink(missing_ok=True)


class EnrichmentWorker:
    def __init__(
        self,
        store: MemoryStore,
        enricher: Enricher,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder | None,
        concurrency: int = 2,
        face_blur: FaceBlur | None = None,
    ) -> None:
        self.store = store
        self.enricher = enricher
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.face_blur = face_blur
        self.queue: asyncio.Queue[tuple[MemoryToken, bytes]] = asyncio.Queue(maxsize=200)
        self._tasks: list[asyncio.Task] = []
        self._concurrency = concurrency
        self._inflight: dict[str, MemoryToken] = {}  # dequeued, not yet stored
        self._discarded: set[str] = set()  # in-flight ids a purge cancelled
        self._add_lock = asyncio.Lock()  # store.add vs purge: nothing lands after a purge returns

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._run()) for _ in range(self._concurrency)]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()

    async def submit(self, token: MemoryToken, jpeg: bytes) -> None:
        if self.queue.full():
            dropped, _ = self.queue.get_nowait()  # drop oldest under backpressure
            self.queue.task_done()
            _unlink(dropped.thumb_path)
        await self.queue.put((token, jpeg))

    def discard(self, device_id: str | None = None, since_ts: float | None = None) -> int:
        """Drop queued frames and cancel in-flight ones matching a purge window; returns how many.

        Synchronous (no awaits), so it is atomic with respect to the worker tasks.
        """
        kept, dropped = [], 0
        n = self.queue.qsize()
        for _ in range(n):
            token, jpeg = self.queue.get_nowait()
            if _in_window(token.device_id, token.ts, device_id, since_ts):
                _unlink(token.thumb_path)
                dropped += 1
            else:
                kept.append((token, jpeg))
        for item in kept:
            self.queue.put_nowait(item)
        for _ in range(n):  # after re-queueing, so join() never sees a false zero
            self.queue.task_done()
        for tid, token in self._inflight.items():
            if tid not in self._discarded and _in_window(token.device_id, token.ts, device_id, since_ts):
                self._discarded.add(tid)
                dropped += 1
        return dropped

    async def purge(
        self, device_id: str | None = None, *, since_ts: float | None = None, delete_thumbs: bool = True
    ) -> int:
        """`store.purge` that also discards queued / in-flight frames in the same window."""
        async with self._add_lock:
            n = self.discard(device_id, since_ts)
            return n + await self.store.purge(device_id, since_ts=since_ts, delete_thumbs=delete_thumbs)

    async def _run(self) -> None:
        while True:
            token, jpeg = await self.queue.get()
            self._inflight[token.id] = token
            try:
                await self._process(token, jpeg)
            except Exception:  # noqa: BLE001
                log.exception("enrichment failed for %s", token.id)
                # never stored, so no purge would ever find it; and it may be the unblurred upload
                _unlink(token.thumb_path)
            finally:
                self._inflight.pop(token.id, None)
                if token.id in self._discarded:
                    self._discarded.discard(token.id)
                    _unlink(token.thumb_path)
                self.queue.task_done()

    async def _process(self, token: MemoryToken, jpeg: bytes) -> None:
        if self.face_blur is not None:
            blurred, n_faces = await self.face_blur.apply(jpeg)
            if n_faces:
                Path(token.thumb_path).write_bytes(blurred)
                if self.face_blur.upstream:
                    jpeg = blurred
        e = await self.enricher.enrich(jpeg)
        token.caption, token.ocr_text, token.entities = e.caption, e.ocr_text, e.entities
        token.text_vec = await self.text_embedder.embed(f"{e.caption}\n{e.ocr_text}\n{' '.join(e.entities)}")
        if self.image_embedder is not None:
            token.image_vec = await self.image_embedder.embed_image(jpeg)
        async with self._add_lock:
            if token.id in self._discarded:
                return
            await self.store.add(token)


async def purge_all(state: Any, device_id: str | None = None, *, since_ts: float | None = None) -> int:
    """Purge for `DELETE /v1/memories`: store rows + thumbnails, queued/in-flight frames, and the
    ingest dedupe state (pending hashes, keyframe-filter history) that still references them."""
    n = await state.worker.purge(device_id, since_ts=since_ts)
    pending: dict[str, list[tuple[float, str]]] = getattr(state, "pending_hashes", {})
    for dev in [d for d in pending if device_id is None or d == device_id]:
        pending[dev] = [(t, h) for t, h in pending[dev] if since_ts is not None and t < since_ts]
    kf = getattr(state, "keyframe_filter", None)
    if kf is not None:
        kf.reset(device_id)
    return n
