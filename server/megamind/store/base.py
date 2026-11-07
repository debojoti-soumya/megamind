from __future__ import annotations

from abc import ABC, abstractmethod

from megamind.models import MemoryEpisode, MemoryHit, MemoryToken, SearchMode


class MemoryStore(ABC):
    """Storage + hybrid retrieval of memory tokens.

    Implementations must support: time-range pre-filtering, vector search on
    text_vec and image_vec, keyword (BM25) search on caption/ocr_text, and
    reciprocal-rank fusion for `hybrid`.
    """

    @abstractmethod
    async def add(self, token: MemoryToken) -> None: ...

    async def add_many(self, tokens: list[MemoryToken]) -> None:
        for t in tokens:
            await self.add(t)

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
        per_episode_cap: int | None = None,
    ) -> list[MemoryHit]:
        """`per_episode_cap=None` uses the configured diversity cap; 0 disables it."""

    @abstractmethod
    async def list_tokens(
        self,
        device_id: str | None = None,
        time_to: float | None = None,
        tier: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryToken]:
        """Tokens matching the filters, oldest first (used by media aging)."""

    @abstractmethod
    async def tier_counts(self) -> dict[str, int]: ...

    @abstractmethod
    async def upsert_episode(self, episode: MemoryEpisode) -> None: ...

    @abstractmethod
    async def get_episode(self, episode_id: str) -> MemoryEpisode | None: ...

    @abstractmethod
    async def list_episodes(
        self,
        device_id: str | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
        limit: int = 50,
    ) -> list[MemoryEpisode]:
        """Episodes overlapping [time_from, time_to], oldest first."""

    @abstractmethod
    async def purge(self, device_id: str | None = None) -> int:
        """Delete tokens (and their episodes); returns the number of tokens deleted."""
