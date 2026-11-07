from __future__ import annotations

from abc import ABC, abstractmethod

from megamind.models import MemoryHit, MemoryToken, SearchMode


class MemoryStore(ABC):
    """Storage + hybrid retrieval of memory tokens.

    Implementations must support: time-range pre-filtering, vector search on
    text_vec and image_vec, keyword (BM25) search on caption/ocr_text, and
    reciprocal-rank fusion for `hybrid`.
    """

    @abstractmethod
    async def add(self, token: MemoryToken) -> None: ...

    @abstractmethod
    async def get(self, token_id: str) -> MemoryToken | None: ...

    @abstractmethod
    async def count(self, device_id: str | None = None) -> int: ...

    @abstractmethod
    async def recent_hashes(self, device_id: str, since_ts: float) -> list[tuple[float, str]]:
        """(ts, frame_hash) for accepted frames since `since_ts` — used by the dedupe filter."""

    @abstractmethod
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
    ) -> list[MemoryHit]: ...

    @abstractmethod
    async def purge(self, device_id: str | None = None) -> int: ...
