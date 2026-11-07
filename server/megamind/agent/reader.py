"""Reader sub-agent: a Qwen-VL model that attends over the concatenated memory tokens
(captions + thumbnails) and answers the sub-question with citations.

TODO(stream-2): implement QwenVLReader against settings.reader_base_url (OpenAI-compatible
chat.completions with image_url data URIs). Prompt: for each token emit
"[id] [local time] caption / ocr" followed by the image; then the query; demand JSON
{"evidence": str, "confidence": high|medium|low|not_found, "cited_ids": [str]}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from megamind.config import Settings
from megamind.models import Confidence, MemoryToken


@dataclass
class ReaderOutput:
    evidence: str
    confidence: Confidence
    cited_ids: list[str] = field(default_factory=list)


class Reader(ABC):
    @abstractmethod
    async def read(self, query: str, tokens: list[MemoryToken]) -> ReaderOutput: ...


class CaptionOnlyReader(Reader):
    """Fallback with no model call: returns the top captions verbatim as evidence."""

    async def read(self, query: str, tokens: list[MemoryToken]) -> ReaderOutput:
        top = tokens[:3]
        lines = [f"[{t.id}] {t.caption}" + (f" (text: {t.ocr_text})" if t.ocr_text else "") for t in top]
        return ReaderOutput(evidence="\n".join(lines), confidence="low", cited_ids=[t.id for t in top])


def build_reader(settings: Settings) -> Reader:
    # TODO(stream-2): return QwenVLReader(settings) when configured
    return CaptionOnlyReader()
