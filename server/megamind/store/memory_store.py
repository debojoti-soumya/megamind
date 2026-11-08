"""In-memory store: the test double and zero-dependency dev backend. Same fusion as LanceDBStore."""

from __future__ import annotations

from megamind.models import MemoryEpisode, MemoryHit, MemorySummary, MemoryToken, SearchMode
from megamind.store.base import MemoryStore
from megamind.store.fusion import (
    POOL,
    SUMMARY_POOL,
    FusionWeights,
    bm25_rank,
    cosine,
    diversify,
    summary_picks,
    weighted_rrf,
)


def _vec_rank(q: list[float], items: list[tuple[str, list[float] | None]]) -> list[tuple[str, float]]:
    """Top-POOL (id, cosine); non-positive similarity is treated as no match."""
    scored = [(cosine(q, v), i) for i, v in items if v]
    return [(i, s) for s, i in sorted(scored, reverse=True)[:POOL] if s > 0]


class InMemoryStore(MemoryStore):
    def __init__(self, weights: FusionWeights | None = None) -> None:
        self._tokens: dict[str, MemoryToken] = {}
        self._episodes: dict[str, MemoryEpisode] = {}
        self._summaries: dict[str, MemorySummary] = {}
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
        per_episode_cap: int | None = None,
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
        rankings: list = []
        if mode in ("semantic", "hybrid"):
            rankings.append((w.bm25, bm25_rank(query_text, {t.id: t.search_text() for t in pool})[:POOL]))
            if query_text_vec is not None:
                rankings.append((w.text, [i for i, _ in _vec_rank(query_text_vec, [(t.id, t.text_vec) for t in pool])]))
        if mode in ("visual", "hybrid") and query_image_vec is not None:
            weight = w.image_visual_mode if mode == "visual" else w.image
            hits = _vec_rank(query_image_vec, [(t.id, t.image_vec) for t in pool])
            rankings.append((weight, [(i, w.image_conf(c)) for i, c in hits]))

        if mode in ("semantic", "hybrid") and self._summaries:
            rankings += self._summary_rankings(
                query_text_vec, query_text, pool, weighted_rrf(rankings), time_from, time_to
            )
        fused = weighted_rrf(rankings)
        cap = w.per_episode_cap if per_episode_cap is None else per_episode_cap
        ordered = sorted(fused.items(), key=lambda kv: (kv[1], self._tokens[kv[0]].ts), reverse=True)
        picked = diversify(ordered, {i: self._tokens[i].episode_id for i, _ in ordered}, k, cap)
        return [MemoryHit.from_token(self._tokens[i], s) for i, s in picked]

    def _summary_rankings(
        self,
        qvec: list[float] | None,
        query_text: str,
        pool: list[MemoryToken],
        base: dict[str, float],
        time_from: float | None,
        time_to: float | None,
    ) -> list:
        by_id = {t.id: t for t in pool}
        sums = [
            s
            for s in self._summaries.values()
            if (time_to is None or s.start_ts <= time_to) and (time_from is None or s.end_ts >= time_from)
        ]
        members = {s.id: [i for i in s.token_ids if i in by_id] for s in sums}
        ranked = [bm25_rank(query_text, {s.id: s.text for s in sums})[:SUMMARY_POOL]]
        if qvec is not None:
            ranked.append([i for i, _ in _vec_rank(qvec, [(s.id, s.text_vec) for s in sums])][:SUMMARY_POOL])
        sim = {
            i: cosine(qvec, by_id[i].text_vec or [0.0])
            for r in ranked
            for sid in r
            for i in members[sid]
            if qvec is not None and by_id[i].text_vec
        }
        return [(self.weights.summary, summary_picks([members[sid] for sid in r], base, sim)) for r in ranked]

    async def list_tokens(
        self,
        device_id: str | None = None,
        time_to: float | None = None,
        tier: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryToken]:
        ts = sorted(
            (
                t
                for t in self._tokens.values()
                if (device_id is None or t.device_id == device_id)
                and (time_to is None or t.ts <= time_to)
                and (tier is None or t.tier == tier)
            ),
            key=lambda t: t.ts,
        )
        return ts[:limit] if limit else ts

    async def tier_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self._tokens.values():
            out[t.tier] = out.get(t.tier, 0) + 1
        return out

    async def upsert_episode(self, episode: MemoryEpisode) -> None:
        self._episodes[episode.id] = episode.model_copy(deep=True)

    async def get_episode(self, episode_id: str) -> MemoryEpisode | None:
        return self._episodes.get(episode_id)

    async def list_episodes(
        self,
        device_id: str | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
        limit: int = 50,
    ) -> list[MemoryEpisode]:
        eps = [
            e
            for e in self._episodes.values()
            if (device_id is None or e.device_id == device_id)
            and (time_from is None or e.end_ts >= time_from)
            and (time_to is None or e.start_ts <= time_to)
        ]
        return sorted(eps, key=lambda e: e.start_ts)[:limit]

    async def upsert_summary(self, summary: MemorySummary) -> None:
        self._summaries[summary.id] = summary.model_copy(deep=True)

    async def list_summaries(
        self, device_id: str | None = None, time_from: float | None = None, time_to: float | None = None
    ) -> list[MemorySummary]:
        out = [
            s
            for s in self._summaries.values()
            if (device_id is None or s.device_id == device_id)
            and (time_from is None or s.end_ts >= time_from)
            and (time_to is None or s.start_ts <= time_to)
        ]
        return sorted(out, key=lambda s: s.start_ts)

    async def purge(self, device_id: str | None = None) -> int:
        if device_id is None:
            n = len(self._tokens)
            self._tokens.clear()
            self._episodes.clear()
            self._summaries.clear()
            return n
        ids = [i for i, t in self._tokens.items() if t.device_id == device_id]
        for i in ids:
            del self._tokens[i]
        for eid in [i for i, e in self._episodes.items() if e.device_id == device_id]:
            del self._episodes[eid]
        for sid in [i for i, x in self._summaries.items() if x.device_id == device_id]:
            del self._summaries[sid]
        return len(ids)
