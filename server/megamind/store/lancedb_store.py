"""LanceDB-backed MemoryStore.

TODO(stream-1): implement.
  - table schema mirrors MemoryToken; text_vec (1024) and image_vec (768) as fixed-size vectors
  - create_fts_index on caption + ocr_text for BM25
  - search(): pre-filter by ts/device_id, run vector + fts queries, fuse with RRF
  - recent_hashes(): simple filtered scan
See docs/megamind-design.pdf §4.3 and §5.2.
"""

from __future__ import annotations

from pathlib import Path

from megamind.models import MemoryHit, MemoryToken, SearchMode
from megamind.store.base import MemoryStore


class LanceDBStore(MemoryStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        raise NotImplementedError("LanceDBStore is not implemented yet (TODO stream-1)")

    async def add(self, token: MemoryToken) -> None:
        raise NotImplementedError

    async def get(self, token_id: str) -> MemoryToken | None:
        raise NotImplementedError

    async def count(self, device_id: str | None = None) -> int:
        raise NotImplementedError

    async def recent_hashes(self, device_id: str, since_ts: float) -> list[tuple[float, str]]:
        raise NotImplementedError

    async def search(
        self,
        query_text_vec: list[float] | None,
        query_image_vec: list[float] | None,
        query_text: str,
        *,
        k: int = 12,
        mode: SearchMode = "hybrid",
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
    ) -> list[MemoryHit]:
        raise NotImplementedError

    async def purge(self, device_id: str | None = None) -> int:
        raise NotImplementedError
