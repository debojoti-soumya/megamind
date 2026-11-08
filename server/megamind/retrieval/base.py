"""The pluggable `Retriever` interface. Implementations: `HybridRetriever` (ours), `UCCRetriever` (stub)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from megamind.models import MemoryHit, MemoryToken
from megamind.retrieval.planner import QueryPlan

RetrieveMode = str  # "semantic" | "visual" | "hybrid" | "temporal" | "recent"


@dataclass
class RetrievedToken:
    token: MemoryToken
    score: float
    reasons: list[str] = field(default_factory=list)  # e.g. ["text", "bm25"], ["neighbor"]
    is_context: bool = False  # pulled in by temporal expansion rather than by the query

    def hit(self) -> MemoryHit:
        return MemoryHit.from_token(self.token, self.score)


@dataclass
class RetrievalResult:
    plan: QueryPlan
    tokens: list[RetrievedToken]  # chronological, <= k
    widened: bool = False  # the time window was dropped after an empty first pass
    anchor_ts: float | None = None  # resolved timestamp of `plan.anchor_event`, if any

    @property
    def ranked(self) -> list[RetrievedToken]:
        """Best-first view (what the user would call the top hits)."""
        return sorted(self.tokens, key=lambda r: (r.is_context, -r.score, -r.token.ts))

    def hits(self) -> list[MemoryHit]:
        return [r.hit() for r in self.ranked]

    def memory_tokens(self) -> list[MemoryToken]:
        return [r.token for r in self.tokens]


class Retriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        k: int = 12,
        mode: RetrieveMode = "hybrid",
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        now: datetime | None = None,
    ) -> RetrievalResult: ...
