"""Thin adapter over `MemoryStore` exposing the primitives the retriever needs.

The storage stream is adding `vector_search`, `keyword_search`, `get_many`, `range` and
`neighbors` to `MemoryStore` (signatures settled in issue #3: the search primitives return
`list[tuple[MemoryToken, float]]`, `neighbors(token_id, *, before, after)` includes the
anchor). Until a store implements one of them, the adapter computes it here from the token
list, so the retriever is unaffected by which store is underneath. The adapter always
exposes `(id, score)` rankings and caches the tokens the store already returned so
`get_many` is free in the common case.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Literal

from megamind.models import MemoryToken
from megamind.store.base import MemoryStore

VectorField = Literal["text_vec", "image_vec"]

_WORD = re.compile(r"[a-z0-9][a-z0-9'$.%-]*")


def tokenize(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        w = w.strip("'.-")
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if len(w) > 5 and w.endswith("ing"):
            w = w[:-3]
        elif len(w) > 4 and w.endswith("ed"):
            w = w[:-2]
        if w:
            out.append(w)
    return out


def token_text(t: MemoryToken) -> str:
    return f"{t.caption} {t.ocr_text} {' '.join(t.entities)}"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def bm25_scores(query: str, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> list[tuple[str, float]]:
    """Okapi BM25 over `docs` (id -> text); positive scores only, best first."""
    q = tokenize(query)
    if not q or not docs:
        return []
    tf = {i: Counter(tokenize(d)) for i, d in docs.items()}
    n = len(docs)
    avgdl = sum(sum(c.values()) for c in tf.values()) / n or 1.0
    df = Counter()
    for c in tf.values():
        df.update(set(c))
    out = []
    for i, c in tf.items():
        dl = sum(c.values())
        s = 0.0
        for term in q:
            f = c.get(term)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        if s > 0:
            out.append((i, s))
    out.sort(key=lambda x: -x[1])
    return out


class StoreAdapter:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self._seen: dict[str, MemoryToken] = {}

    def _rank(self, rows: list) -> list[tuple[str, float]]:
        """Normalise a store ranking (`(token, score)` per #3, or `(id, score)`) to ids."""
        out: list[tuple[str, float]] = []
        for item, score in rows:
            if isinstance(item, MemoryToken):
                self._seen[item.id] = item
                out.append((item.id, float(score)))
            else:
                out.append((str(item), float(score)))
        return out

    # ------------------------------------------------------------------ primitives

    async def vector_search(
        self,
        vec: list[float],
        k: int,
        field: VectorField = "text_vec",
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
    ) -> list[tuple[str, float]]:
        if hasattr(self.store, "vector_search"):
            rows = await self.store.vector_search(
                vec, k, field=field, time_from=time_from, time_to=time_to, device_id=device_id
            )
            return [(i, s) for i, s in self._rank(rows) if s > 0]  # stores may return non-matches at <= 0
        pool = await self._pool(time_from, time_to, device_id)
        scored = [(t.id, cosine(vec, v)) for t in pool if t.caption and (v := getattr(t, field))]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    async def keyword_search(
        self,
        query: str,
        k: int,
        *,
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
    ) -> list[tuple[str, float]]:
        if hasattr(self.store, "keyword_search"):
            rows = await self.store.keyword_search(query, k, time_from=time_from, time_to=time_to, device_id=device_id)
            return self._rank(rows)
        pool = await self._pool(time_from, time_to, device_id)
        return bm25_scores(query, {t.id: token_text(t) for t in pool if t.caption})[:k]

    async def get_many(self, ids: list[str]) -> list[MemoryToken]:
        missing = [i for i in ids if i not in self._seen]
        if missing:
            if hasattr(self.store, "get_many"):
                fetched = await self.store.get_many(missing)
            else:
                fetched = [t for i in missing if (t := await self.store.get(i)) is not None]
            for t in fetched:
                self._seen[t.id] = t
        return [self._seen[i] for i in ids if i in self._seen]

    async def range(
        self,
        time_from: float | None,
        time_to: float | None,
        *,
        limit: int = 100,
        device_id: str | None = None,
    ) -> list[MemoryToken]:
        if hasattr(self.store, "range"):
            return await self.store.range(time_from, time_to, limit, device_id=device_id)
        pool = await self._pool(time_from, time_to, device_id)
        return sorted(pool, key=lambda t: t.ts)[:limit]

    async def neighbors(self, token_id: str, before: int = 1, after: int = 1) -> list[MemoryToken]:
        """Chronological neighbours of `token_id` on the same device, *excluding* the anchor."""
        if hasattr(self.store, "neighbors"):
            rows = await self.store.neighbors(token_id, before=before, after=after)
            return [t for t in rows if t.id != token_id]
        anchor = await self.store.get(token_id)
        if anchor is None:
            return []
        pool = sorted(await self._pool(None, None, anchor.device_id), key=lambda t: t.ts)
        idx = next((i for i, t in enumerate(pool) if t.id == token_id), None)
        if idx is None:
            return []
        return pool[max(0, idx - before) : idx] + pool[idx + 1 : idx + 1 + after]

    # ------------------------------------------------------------------ fallback plumbing

    async def _pool(self, time_from: float | None, time_to: float | None, device_id: str | None) -> list[MemoryToken]:
        """All tokens matching the filters, from whatever the store offers."""
        if hasattr(self.store, "list_tokens"):
            toks = await self.store.list_tokens(device_id=device_id, time_to=time_to)
            return [t for t in toks if time_from is None or t.ts >= time_from]
        raw = getattr(self.store, "_tokens", None)
        if isinstance(raw, dict):
            return [
                t
                for t in raw.values()
                if (device_id is None or t.device_id == device_id)
                and (time_from is None or t.ts >= time_from)
                and (time_to is None or t.ts <= time_to)
            ]
        hits = await self.store.search(
            None, None, "", k=10_000, mode="semantic", time_from=time_from, time_to=time_to, device_id=device_id
        )
        return await self.get_many([h.id for h in hits])
