"""In-memory store: the test double and zero-dependency dev backend. Same fusion as LanceDBStore."""

from __future__ import annotations

from megamind.models import MemoryHit, MemoryToken, SearchMode
from megamind.store.base import MemoryStore
from megamind.store.fusion import POOL, FusionWeights, bm25_rank, cosine, diversify, weighted_rrf


class InMemoryStore(MemoryStore):
    def __init__(self, weights: FusionWeights | None = None) -> None:
        self._tokens: dict[str, MemoryToken] = {}
        self.weights = weights or FusionWeights()

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
        ]
        if not pool:
            return []
        w = self.weights
        rankings: list[tuple[float, list[str]]] = []
        if mode in ("semantic", "hybrid"):
            rankings.append((w.bm25, bm25_rank(query_text, {t.id: t.search_text() for t in pool})[:POOL]))
            if query_text_vec is not None:
                sem = sorted((t for t in pool if t.text_vec), key=lambda t: -cosine(query_text_vec, t.text_vec))
                rankings.append((w.text, [t.id for t in sem[:POOL]]))
        if mode in ("visual", "hybrid") and query_image_vec is not None:
            vis = sorted((t for t in pool if t.image_vec), key=lambda t: -cosine(query_image_vec, t.image_vec))
            rankings.append((w.image_visual_mode if mode == "visual" else w.image, [t.id for t in vis[:POOL]]))

        fused = weighted_rrf(rankings)
        ordered = sorted(fused.items(), key=lambda kv: (kv[1], self._tokens[kv[0]].ts), reverse=True)
        picked = diversify(ordered, {i: self._tokens[i].episode_id for i, _ in ordered}, k, w.per_episode_cap)
        return [MemoryHit.from_token(self._tokens[i], s) for i, s in picked]

    async def purge(self, device_id: str | None = None) -> int:
        if device_id is None:
            n = len(self._tokens)
            self._tokens.clear()
            return n
        ids = [i for i, t in self._tokens.items() if t.device_id == device_id]
        for i in ids:
            del self._tokens[i]
        return len(ids)
