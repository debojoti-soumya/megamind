"""In-memory store. Good enough for tests and the first vertical slice.
TODO(stream-1): replace with LanceDBStore for persistence + real BM25."""

from __future__ import annotations

import math
import re
from collections.abc import Callable

from megamind.models import MemoryHit, MemoryToken, SearchMode
from megamind.store.base import MemoryStore

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


def _ranked(pool: list[MemoryToken], score: Callable[[MemoryToken], float], eps: float = 1e-6) -> list[str]:
    scored = [(score(t), t) for t in pool]
    return [t.id for s, t in sorted(scored, key=lambda st: st[0], reverse=True) if s > eps]


def _rrf(rankings: list[list[str]], c: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, token_id in enumerate(ranking):
            scores[token_id] = scores.get(token_id, 0.0) + 1.0 / (c + rank + 1)
    return scores


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._tokens: dict[str, MemoryToken] = {}

    async def add(self, token: MemoryToken) -> None:
        self._tokens[token.id] = token

    async def get(self, token_id: str) -> MemoryToken | None:
        return self._tokens.get(token_id)

    async def count(self, device_id: str | None = None) -> int:
        if device_id is None:
            return len(self._tokens)
        return sum(1 for t in self._tokens.values() if t.device_id == device_id)

    async def recent_hashes(self, device_id: str, since_ts: float) -> list[tuple[float, str]]:
        return [(t.ts, t.frame_hash) for t in self._tokens.values() if t.device_id == device_id and t.ts >= since_ts]

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
        pool = [
            t
            for t in self._tokens.values()
            if (device_id is None or t.device_id == device_id)
            and (time_from is None or t.ts >= time_from)
            and (time_to is None or t.ts <= time_to)
            and t.caption  # unenriched tokens are not searchable yet
        ]
        if not pool:
            return []

        # Each ranking only keeps tokens with a positive score, so an unrelated query yields no hits
        # (-> recall_memory reports not_found) instead of citing arbitrary memories.
        rankings: list[list[str]] = []
        if mode in ("semantic", "hybrid"):
            if query_text_vec is not None:
                sem = _ranked(pool, lambda t: _cosine(query_text_vec, t.text_vec) if t.text_vec else 0.0)
                rankings.append(sem)
            kw = _ranked(pool, lambda t: _keyword_score(query_text, f"{t.caption} {t.ocr_text} {' '.join(t.entities)}"))
            rankings.append(kw)
        if mode in ("visual", "hybrid") and query_image_vec is not None:
            vis = _ranked(pool, lambda t: _cosine(query_image_vec, t.image_vec) if t.image_vec else 0.0)
            rankings.append(vis)

        fused = _rrf(rankings)
        if not fused:
            return []
        # recency tie-break: newer first among equal scores
        ordered = sorted(fused.items(), key=lambda kv: (kv[1], self._tokens[kv[0]].ts), reverse=True)
        return [MemoryHit.from_token(self._tokens[i], s) for i, s in ordered[:k]]

    async def purge(self, device_id: str | None = None) -> int:
        if device_id is None:
            n = len(self._tokens)
            self._tokens.clear()
            return n
        ids = [i for i, t in self._tokens.items() if t.device_id == device_id]
        for i in ids:
            del self._tokens[i]
        return len(ids)
