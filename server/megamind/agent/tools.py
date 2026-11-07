"""The `recall_memory` tool: retrieve top-k tokens, hand them to the reader, return evidence."""

from __future__ import annotations

from datetime import datetime

from megamind.agent.reader import Reader
from megamind.ingest.enrich import ImageEmbedder, TextEmbedder
from megamind.models import Citation, MemoryHit, RecallResult, SearchMode
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
        qvec = await self.text_embedder.embed(query)
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
