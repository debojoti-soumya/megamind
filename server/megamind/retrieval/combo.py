"""`RETRIEVER_BACKEND=combo`: route each question to the backend that is best at it, without paying hier's
latency on easy questions.

  pure time / recency ("what was I doing at 11", "what did I just see")   -> hybrid  (~0.1 s)
  hard: another day, sequences, counts, habits ("yesterday", "after lunch",
        "how many times", "usually")                                     -> hier    (+~6 s, best on long days)
  everything else ("where are my keys", "what did the sign say")          -> UCC     (bounded by UCC_TIMEOUT_S)

Missing backends (no key) drop out of the routing; UCC and hier both fall back to hybrid on any failure.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from megamind.retrieval.base import RetrievalResult, RetrieveMode, Retriever
from megamind.retrieval.planner import QueryPlan, plan_query

_HARD = re.compile(
    r"\b(yesterday|last (night|week|time)|days? ago|this week|usually|always|often|rarely|never|"
    r"how (many|often)|every time|first time|in total|before|after|then|between|while|until|since)\b",
    re.IGNORECASE,
)


def is_hard(query: str, plan: QueryPlan, now: datetime, tz: ZoneInfo | None) -> bool:
    """Multi-event, cross-day or habit questions: the ones hier's day/hour/block log answers better."""
    if _HARD.search(query):
        return True
    if plan.time_to is not None:
        local = now.astimezone(tz) if tz else now
        midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
        return plan.time_to < midnight.timestamp()
    return False


class ComboRetriever(Retriever):
    def __init__(
        self,
        hybrid: Retriever,
        *,
        ucc: Retriever | None = None,
        hier: Retriever | None = None,
        tz: str | ZoneInfo | None = None,
    ) -> None:
        self.hybrid, self.ucc, self.hier = hybrid, ucc, hier
        self.tz = ZoneInfo(tz) if isinstance(tz, str) else tz
        self.stats: Counter[str] = Counter()
        self.last_route = ""

    def route(self, query: str, *, now: datetime, time_from: float | None, time_to: float | None) -> str:
        plan = plan_query(query, now=now, tz=self.tz, time_from=time_from, time_to=time_to)
        if plan.mode == "recent":
            return "hybrid"
        if self.hier is not None and is_hard(query, plan, now, self.tz):
            return "hier"
        if plan.route == "hybrid":
            return "hybrid"
        return "ucc" if self.ucc is not None else "hybrid"

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
    ) -> RetrievalResult:
        now = now or datetime.now(UTC)
        name = self.route(query, now=now, time_from=time_from, time_to=time_to)
        self.last_route = name
        self.stats[name] += 1
        backend = {"hybrid": self.hybrid, "ucc": self.ucc, "hier": self.hier}[name]
        return await backend.retrieve(
            query, k=k, mode=mode, time_from=time_from, time_to=time_to, device_id=device_id, now=now
        )
