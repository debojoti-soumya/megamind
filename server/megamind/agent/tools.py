"""The `recall_memory` tool: retrieve top-k tokens, hand them to the reader, return evidence.

Ownership: stream-3 (retrieval) owns `RecallMemory.__call__` up to the ordered token list;
stream-4 (harness) owns the reader call and the `RecallResult` mapping below it.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from megamind.agent.reader import Reader
from megamind.ingest.enrich import ImageEmbedder, TextEmbedder
from megamind.models import Citation, MemoryHit, MemoryToken, RecallResult, SearchMode
from megamind.retrieval import HybridRetriever, Retriever
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
            "query": {
                "type": "string",
                "description": (
                    "Short search keywords, not the full question: the object, place or text to find "
                    "(e.g. 'keys kitchen counter', 'whiteboard text', 'coffee mug')."
                ),
            },
            "time_from": {"type": ["string", "null"], "description": "ISO-8601 lower bound or null."},
            "time_to": {"type": ["string", "null"], "description": "ISO-8601 upper bound or null."},
            "k": {"type": "integer", "default": 12, "minimum": 1, "maximum": 24},
            "mode": {"type": "string", "enum": ["semantic", "visual", "hybrid"], "default": "hybrid"},
        },
        "required": ["query"],
    },
}


DEGRADED_PREFIX = "[reader unavailable: evidence below is from captions only, the images were not inspected]"
OCR_MATCH_PREFIX = "[exact text match: the images were not inspected; answer from the text below]"

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or of on in at to for from with by about is are was were be been am do does did i me my mine "
    "we our you your it its this that these those what which who whom whose when where why how much many there "
    "here have has had any some all today yesterday morning afternoon evening tonight earlier ago last say said".split()
)
# questions where the pixels matter even if the words are printed somewhere
_VISUAL_CUES = frozenset(
    "where who color colour wearing look looked looks see saw seen leave left still image picture photo".split()
)


def keywords(text: str) -> set[str]:
    """Content words, lower-cased, crude plural strip ("keys" == "key"); used for query similarity and OCR match."""
    out = set()
    for w in _WORD_RE.findall(text.lower()):
        if w in _STOPWORDS or (len(w) < 2 and not w.isdigit()):
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def ocr_exact_matches(query: str, tokens: list[MemoryToken], *, top: int = 3) -> list[MemoryToken]:
    """Tokens whose OCR text contains every keyword of `query`, if one of them is a top-`top` hit.

    Conservative on purpose: >= 2 keywords, no visual/location cue words, so the reader could only
    re-read the same printed text."""
    if set(_WORD_RE.findall(query.lower())) & _VISUAL_CUES:
        return []
    terms = keywords(query)
    if len(terms) < 2:
        return []
    matches = [t for t in tokens if t.ocr_text and terms <= keywords(t.ocr_text)]
    if not any(t in matches for t in tokens[:top]):
        return []
    return matches[:4]


def _ocr_evidence(matches: list[MemoryToken]) -> str:
    parts = [f'[{t.id}] text: "{" / ".join(t.ocr_text.split(chr(10)))}" (caption: {t.caption})' for t in matches]
    return f"{OCR_MATCH_PREFIX} " + " ".join(parts)


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
        *,
        retriever: Retriever | None = None,
        tz: str | ZoneInfo | None = None,
    ) -> None:
        self.store = store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.reader = reader
        # Pluggable backend (`UCCRetriever` later); default is our hybrid over the store.
        self.retriever = retriever or HybridRetriever(store, text_embedder, image_embedder, tz=tz)

    async def __call__(
        self,
        query: str,
        time_from: str | None = None,
        time_to: str | None = None,
        k: int = 12,
        mode: SearchMode = "hybrid",
        device_id: str | None = None,
        now: float | None = None,
        skip_reader_on_ocr: bool = False,
    ) -> RecallResult:
        """`skip_reader_on_ocr`: return the matching OCR text instead of calling the reader when it is an
        exact keyword match (Claude path only: the evidence is not phrased for speech)."""
        k = max(1, min(k, 24))
        # --- retrieval (stream-3): plan the query (relative times, type-dependent weights, rewrite),
        # hybrid-rank, dedupe, add temporal context; `mode` and explicit bounds from Claude win over the plan.
        res = await self.retriever.retrieve(
            query,
            k=k,
            mode=mode,
            time_from=_ts(time_from),
            time_to=_ts(time_to),
            device_id=device_id,
            now=datetime.fromtimestamp(now, UTC) if now is not None else None,
        )
        # Hand-off contract (issue #5): full tokens, best-first, context tokens at the tail, <= k.
        hits = res.hits()
        tokens = [r.token for r in res.ranked]
        if not tokens:
            return RecallResult(evidence="No memories match.", confidence="not_found", cited=[], all_retrieved=[])
        # --- reader + result mapping (stream-4) ---
        by_id = {h.id: h for h in hits}
        if skip_reader_on_ocr and mode != "visual" and (matches := ocr_exact_matches(query, tokens)):
            return RecallResult(
                evidence=_ocr_evidence(matches),
                confidence="medium",
                cited=[_cite(by_id[t.id]) for t in matches if t.id in by_id],
                all_retrieved=[_cite(h) for h in hits],
            )
        out = await self.reader.read(query, tokens, now=now)
        evidence = out.evidence
        if out.degraded:
            evidence = f"{DEGRADED_PREFIX} {evidence}"
        return RecallResult(
            evidence=evidence,
            confidence=out.confidence,
            cited=[_cite(by_id[i]) for i in out.cited_ids if i in by_id],
            all_retrieved=[_cite(h) for h in hits],
        )
