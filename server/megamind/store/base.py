from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from megamind.models import MemoryHit, MemoryToken, SearchMode

VectorField = Literal["text_vec", "image_vec"]
Scored = tuple[MemoryToken, float]


class MemoryStore(ABC):
    """Storage + retrieval of memory tokens.

    Two layers:
    - **primitives** (`vector_search`, `keyword_search`, `get_many`, `range`, `neighbors`) that
      the retrieval layer composes without knowing the backend. Scores are backend-specific
      and *not* comparable across primitives — fuse rankings with `megamind.store.rrf`.
    - `search()`: the default hybrid (text vector + BM25 [+ image vector], RRF), kept for the
      debug endpoint and as a fallback.

    `vector_search` / `keyword_search` only return enriched tokens (non-empty caption);
    `get_many` / `range` / `neighbors` return everything. `kinds` filters on the token kind
    (`"frame"` for captured keyframes, `"episode"` for consolidation summaries); `None` = all.
    """

    @abstractmethod
    async def add(self, token: MemoryToken) -> None:
        """Insert or replace (by id)."""

    async def add_many(self, tokens: Sequence[MemoryToken]) -> None:
        for t in tokens:
            await self.add(t)

    @abstractmethod
    async def get(self, token_id: str) -> MemoryToken | None: ...

    @abstractmethod
    async def get_many(self, ids: Sequence[str]) -> list[MemoryToken]:
        """Tokens in the order of `ids`; unknown ids are skipped."""

    @abstractmethod
    async def count(self, device_id: str | None = None) -> int: ...

    @abstractmethod
    async def recent_hashes(self, device_id: str, since_ts: float) -> list[tuple[float, str]]:
        """(ts, frame_hash) for accepted frames since `since_ts` — used by the dedupe filter."""

    @abstractmethod
    async def vector_search(
        self,
        vec: list[float],
        k: int,
        *,
        field: VectorField = "text_vec",
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[Scored]:
        """Top-k by cosine similarity on `field`; score in [-1, 1], best first."""

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        k: int,
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[Scored]:
        """Top-k by lexical match on caption + ocr_text + entities (BM25 where available); best first."""

    @abstractmethod
    async def range(
        self,
        time_from: float | None,
        time_to: float | None,
        limit: int = 100,
        *,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[MemoryToken]:
        """Tokens in [time_from, time_to], oldest first, at most `limit`."""

    @abstractmethod
    async def neighbors(self, token_id: str, *, before: int = 2, after: int = 2) -> list[MemoryToken]:
        """Temporal context: the `before` preceding and `after` following tokens of the same device
        and kind, chronological, *including* the anchor. Empty if the anchor is unknown."""

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
    async def purge(
        self, device_id: str | None = None, *, since_ts: float | None = None, delete_thumbs: bool = True
    ) -> int:
        """Delete tokens (optionally one device / only those with ts >= since_ts) and their thumbnails."""
