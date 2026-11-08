from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from megamind.models import MemoryHit, MemoryToken, SearchMode
from megamind.store.episodes import EpisodeTracker
from megamind.store.rrf import Calibration, Scored

VectorField = Literal["text_vec", "image_vec"]


class MemoryStore(ABC):
    """Storage + retrieval of memory tokens.

    Two layers:
    - **primitives** (`vector_search`, `keyword_search`, `get_many`, `range`, `neighbors`) that
      the retrieval layer composes without knowing the backend. Scores are backend-specific
      and *not* comparable across primitives — fuse rankings with `megamind.store.rrf`.
    - `search()`: the default hybrid (text vector + BM25 [+ image vector], RRF), kept for the
      debug endpoint and as a fallback.

    `vector_search` / `keyword_search` only return enriched tokens (non-empty caption) with a
    positive score (cosine > 0 / at least one matching term), so unrelated queries return `[]`;
    `get_many` / `range` / `neighbors` return everything. `kinds` filters on the token kind
    (`"frame"` for captured keyframes, `"episode"` for consolidation summaries); `None` = all.

    Episodes: when `episodes` (an `EpisodeTracker`) is set, `add`/`add_many` assign an
    `episode_id` to every frame that arrives without one, rebuilding the tracker's window from
    persisted frames on first sight of a device.
    """

    episodes: EpisodeTracker | None = None
    # SigLIP (scale, bias) of the image embedder that wrote `image_vec`, set by main.py when
    # SEARCH_CALIBRATE_IMAGE is on. `search()` and the hybrid retriever then scale each image hit's RRF
    # contribution by sigmoid(scale * cos + bias); None = plain rank credit (the pre-calibration behaviour).
    image_calibration: Calibration | None = None

    async def _episodize(self, tokens: Sequence[MemoryToken]) -> None:
        tracker = self.episodes
        if tracker is None:
            return
        for t in tokens:
            if t.kind != "frame":
                continue
            if not tracker.knows(t.device_id):
                # cover a whole max-length episode so its start (and the length cap) survives a restart
                lookback = max(tracker.gap_s * 2 * tracker.rejoin_last, tracker.max_s + tracker.gap_s, 1800.0)
                recent = await self.range(t.ts - lookback, None, limit=5000, device_id=t.device_id, kinds=("frame",))
                tracker.seed(t.device_id, recent)
            tracker.assign(t)

    @abstractmethod
    async def add(self, token: MemoryToken) -> None:
        """Insert or replace (by id)."""

    async def add_many(self, tokens: Sequence[MemoryToken]) -> None:
        for t in tokens:
            await self.add(t)

    async def rewrite_many(self, tokens: Sequence[MemoryToken]) -> None:
        """Overwrite already-stored tokens in place, without episode assignment (media tiering rewrites old
        frames; re-feeding them to the tracker would disturb the live episode window). Optional; backends
        whose `add_many` episodizes should override it."""
        await self.add_many(tokens)

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

    async def list_tokens(
        self,
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
        tier: str | None = None,
        chunk_id: str | None = None,
        vectors: bool = True,
    ) -> list[MemoryToken]:
        """Every token matching the filters, oldest first (media tiering / stats). Optional: the default is
        built on `range()`; backends may push the `tier` / `chunk_id` filters down. `vectors=False` lets a
        backend skip loading `text_vec` / `image_vec` (planning scans over many rows); such tokens must never
        be written back, since a write replaces the whole row."""
        toks = await self.range(time_from, time_to, limit=10**9, device_id=device_id, kinds=kinds)
        return [t for t in toks if (tier is None or t.tier == tier) and (chunk_id is None or t.chunk_id == chunk_id)]

    async def tier_counts(self, kinds: Sequence[str] | None = ("frame",)) -> dict[str, int]:
        """{"hot": n, "cold": n, "text": n} over `kinds`. Optional; the default scans `list_tokens`."""
        out = {"hot": 0, "cold": 0, "text": 0}
        for t in await self.list_tokens(kinds=kinds):
            out[t.tier] = out.get(t.tier, 0) + 1
        return out

    @abstractmethod
    async def purge(
        self, device_id: str | None = None, *, since_ts: float | None = None, delete_thumbs: bool = True
    ) -> int:
        """Delete tokens (optionally one device / only those with ts >= since_ts) and their thumbnails."""
