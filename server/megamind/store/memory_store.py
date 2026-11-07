"""In-memory store. Good enough for tests and the first vertical slice; LanceDBStore for persistence."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path

from megamind.models import MemoryHit, MemoryToken, SearchMode
from megamind.store.base import MemoryStore, Scored, VectorField
from megamind.store.rrf import rrf

_WORD = re.compile(r"[a-z0-9]+")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _keyword_score(query: str, text: str) -> float:
    q = set(_WORD.findall(query.lower()))
    if not q:
        return 0.0
    t = set(_WORD.findall(text.lower()))
    return len(q & t) / len(q)


def _kind(t: MemoryToken) -> str:
    return getattr(t, "kind", None) or "frame"


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._tokens: dict[str, MemoryToken] = {}

    async def add(self, token: MemoryToken) -> None:
        self._tokens[token.id] = token

    async def get(self, token_id: str) -> MemoryToken | None:
        return self._tokens.get(token_id)

    async def get_many(self, ids: Sequence[str]) -> list[MemoryToken]:
        seen: set[str] = set()
        out: list[MemoryToken] = []
        for i in ids:
            if i in self._tokens and i not in seen:
                seen.add(i)
                out.append(self._tokens[i])
        return out

    async def count(self, device_id: str | None = None) -> int:
        if device_id is None:
            return len(self._tokens)
        return sum(1 for t in self._tokens.values() if t.device_id == device_id)

    async def recent_hashes(self, device_id: str, since_ts: float) -> list[tuple[float, str]]:
        return [
            (t.ts, t.frame_hash)
            for t in self._tokens.values()
            if t.device_id == device_id and t.ts >= since_ts and _kind(t) == "frame"
        ]

    def _pool(
        self,
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
        enriched_only: bool = False,
    ) -> list[MemoryToken]:
        return [
            t
            for t in self._tokens.values()
            if (device_id is None or t.device_id == device_id)
            and (time_from is None or t.ts >= time_from)
            and (time_to is None or t.ts <= time_to)
            and (not kinds or _kind(t) in kinds)
            and (not enriched_only or t.caption)
        ]

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
        if field not in ("text_vec", "image_vec"):
            raise ValueError(f"unknown vector field {field!r}")
        scored = [
            (t, _cosine(vec, v))
            for t in self._pool(
                time_from=time_from, time_to=time_to, device_id=device_id, kinds=kinds, enriched_only=True
            )
            if (v := getattr(t, field)) is not None and len(v) == len(vec)
        ]
        scored.sort(key=lambda ts: (ts[1], ts[0].ts), reverse=True)
        return scored[:k]

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
        scored = [
            (t, s)
            for t in self._pool(
                time_from=time_from, time_to=time_to, device_id=device_id, kinds=kinds, enriched_only=True
            )
            if (s := _keyword_score(query, f"{t.caption} {t.ocr_text} {' '.join(t.entities)}")) > 0.0
        ]
        scored.sort(key=lambda ts: (ts[1], ts[0].ts), reverse=True)
        return scored[:k]

    async def range(
        self,
        time_from: float | None,
        time_to: float | None,
        limit: int = 100,
        *,
        device_id: str | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[MemoryToken]:
        pool = self._pool(time_from=time_from, time_to=time_to, device_id=device_id, kinds=kinds)
        pool.sort(key=lambda t: (t.ts, t.id))
        return pool[:limit]

    async def neighbors(self, token_id: str, *, before: int = 2, after: int = 2) -> list[MemoryToken]:
        anchor = self._tokens.get(token_id)
        if anchor is None:
            return []
        series = self._pool(device_id=anchor.device_id, kinds=(_kind(anchor),))
        series.sort(key=lambda t: (t.ts, t.id))
        idx = next(i for i, t in enumerate(series) if t.id == token_id)
        return series[max(0, idx - before) : idx + after + 1]

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
        filters = {"time_from": time_from, "time_to": time_to, "device_id": device_id}
        n = max(k * 4, 48)
        rankings: list[list[Scored]] = []
        if mode in ("semantic", "hybrid"):
            if query_text_vec is not None:
                rankings.append(await self.vector_search(query_text_vec, n, field="text_vec", **filters))
            rankings.append(await self.keyword_search(query_text, n, **filters))
        if mode in ("visual", "hybrid") and query_image_vec is not None:
            rankings.append(await self.vector_search(query_image_vec, n, field="image_vec", **filters))

        tokens: dict[str, MemoryToken] = {}
        for ranking in rankings:
            for t, _ in ranking:
                tokens.setdefault(t.id, t)
        fused = rrf([[t.id for t, _ in ranking] for ranking in rankings])
        ordered = sorted(fused.items(), key=lambda kv: (kv[1], tokens[kv[0]].ts), reverse=True)
        return [MemoryHit.from_token(tokens[i], s) for i, s in ordered[:k]]

    async def purge(
        self, device_id: str | None = None, *, since_ts: float | None = None, delete_thumbs: bool = True
    ) -> int:
        victims = self._pool(device_id=device_id, time_from=since_ts)
        for t in victims:
            del self._tokens[t.id]
            if delete_thumbs:
                p = Path(t.thumb_path)
                if p.is_file():
                    p.unlink(missing_ok=True)
        return len(victims)
