"""The `recall_memory` tool: retrieve top-k tokens, hand them to the reader, return evidence."""

from __future__ import annotations

from datetime import datetime

from megamind.agent.reader import Reader
from megamind.ingest.enrich import ImageEmbedder, TextEmbedder
from megamind.models import Citation, EpisodeHit, MemoryHit, RecallResult, SearchMode
from megamind.store.base import MemoryStore

RECALL_MEMORY_TOOL = {
    "name": "recall_memory",
    "description": (
        "Search the wearer's visual memory (first-person snapshots captured by their headset) "
        "and get grounded evidence. Use it whenever the user asks about something they saw, "
        "where an object is, what some text said, or what happened earlier. Convert relative "
        "times ('this morning', '10 minutes ago') into ISO-8601 using the current time given "
        "in the system prompt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in natural language."},
            "time_from": {"type": ["string", "null"], "description": "ISO-8601 lower bound or null."},
            "time_to": {"type": ["string", "null"], "description": "ISO-8601 upper bound or null."},
            "k": {"type": "integer", "default": 12, "minimum": 1, "maximum": 24},
            "mode": {"type": "string", "enum": ["semantic", "visual", "hybrid"], "default": "hybrid"},
        },
        "required": ["query"],
    },
}


LIST_EPISODES_TOOL = {
    "name": "list_episodes",
    "description": (
        "List what the wearer was doing over a time range as episodes (contiguous scenes with a "
        "title, time span and representative frame), oldest first. Use for activity/timeline "
        "questions ('what did I do after lunch', 'where was I this morning'). Pass `query` to rank "
        "episodes by relevance instead ('when was I working on the poster')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "time_from": {"type": ["string", "null"], "description": "ISO-8601 lower bound or null."},
            "time_to": {"type": ["string", "null"], "description": "ISO-8601 upper bound or null."},
            "query": {"type": ["string", "null"], "description": "Optional topic to rank episodes by."},
            "k": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
        },
    },
}


def _ts(s: str | None) -> float | None:
    return datetime.fromisoformat(s).timestamp() if s else None


def _cite(h: MemoryHit) -> Citation:
    return Citation(id=h.id, ts=h.ts, caption=h.caption, thumb_url=h.thumb_url)


class RecallMemory:
    def __init__(
        self,
        store: MemoryStore,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder | None,
        reader: Reader,
    ) -> None:
        self.store = store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.reader = reader

    async def __call__(
        self,
        query: str,
        time_from: str | None = None,
        time_to: str | None = None,
        k: int = 12,
        mode: SearchMode = "hybrid",
        device_id: str | None = None,
    ) -> RecallResult:
        k = max(1, min(k, 24))
        qvec = await self.text_embedder.embed_query(query)
        ivec = await self.image_embedder.embed_text(query) if self.image_embedder else None
        hits = await self.store.search(
            qvec,
            ivec,
            query,
            k=k,
            mode=mode,
            time_from=_ts(time_from),
            time_to=_ts(time_to),
            device_id=device_id,
        )
        if not hits:
            return RecallResult(evidence="No memories match.", confidence="not_found", cited=[], all_retrieved=[])
        tokens = [t for h in hits if (t := await self.store.get(h.id)) is not None]
        out = await self.reader.read(query, tokens)
        by_id = {h.id: h for h in hits}
        return RecallResult(
            evidence=out.evidence,
            confidence=out.confidence,
            cited=[_cite(by_id[i]) for i in out.cited_ids if i in by_id],
            all_retrieved=[_cite(h) for h in hits],
        )


class ListEpisodes:
    """Chronological episodes in a range, or episodes ranked by the summed fused score of their tokens."""

    def __init__(self, store: MemoryStore, text_embedder: TextEmbedder, image_embedder: ImageEmbedder | None) -> None:
        self.store = store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder

    async def __call__(
        self,
        time_from: str | float | None = None,
        time_to: str | float | None = None,
        query: str | None = None,
        k: int = 20,
        device_id: str | None = None,
    ) -> list[EpisodeHit]:
        k = max(1, min(k, 50))
        t0 = time_from if isinstance(time_from, float | int) or time_from is None else _ts(time_from)
        t1 = time_to if isinstance(time_to, float | int) or time_to is None else _ts(time_to)
        if not query:
            eps = await self.store.list_episodes(device_id, t0, t1, limit=1000)
            return [EpisodeHit.from_episode(e) for e in eps[-k:]]
        qvec = await self.text_embedder.embed_query(query)
        ivec = await self.image_embedder.embed_text(query) if self.image_embedder else None
        hits = await self.store.search(
            qvec, ivec, query, k=60, time_from=t0, time_to=t1, device_id=device_id, per_episode_cap=0
        )
        scores: dict[str, float] = {}
        tops: dict[str, list[MemoryHit]] = {}
        for h in hits:
            if h.episode_id is None:
                continue
            scores[h.episode_id] = scores.get(h.episode_id, 0.0) + h.score
            if len(tops.setdefault(h.episode_id, [])) < 2:
                tops[h.episode_id].append(h)
        out = []
        for eid, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]:
            e = await self.store.get_episode(eid)
            if e is not None:
                out.append(EpisodeHit.from_episode(e, score=sc, top_hits=tops[eid]))
        return out
