"""Turn an accepted keyframe into caption / OCR / entities + embeddings.

TODO(stream-1):
  - VLMEnricher: call the OpenAI-compatible Qwen-VL endpoint (settings.enrich_url) with the
    JPEG as an image_url data URI and ENRICH_PROMPT; parse the JSON output.
  - Embedders: openai / voyage / local (bge-m3 + SigLIP via open_clip) per settings.embed_provider.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from megamind.config import Settings

ENRICH_PROMPT = """You are describing a single first-person snapshot that will later be searched
to answer questions like "where did I leave my keys" or "what did the sign say".
Return ONLY JSON: {"caption": str, "ocr_text": str, "entities": [str]}.
- caption: 1-3 dense sentences. Name objects and WHERE they are relative to surfaces/furniture,
  the room type, and any people as neutral descriptors (never identities).
- ocr_text: every legible piece of text, verbatim, newline-separated. Empty string if none.
- entities: short noun phrases for objects, places, text labels (max 15)."""


@dataclass
class Enrichment:
    caption: str = ""
    ocr_text: str = ""
    entities: list[str] = field(default_factory=list)


class Enricher(ABC):
    @abstractmethod
    async def enrich(self, jpeg: bytes) -> Enrichment: ...


class MockEnricher(Enricher):
    async def enrich(self, jpeg: bytes) -> Enrichment:
        h = hashlib.sha1(jpeg).hexdigest()[:8]
        return Enrichment(caption=f"mock caption for frame {h}", entities=["mock"])


class TextEmbedder(ABC):
    dim: int

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...


class ImageEmbedder(ABC):
    dim: int

    @abstractmethod
    async def embed_image(self, jpeg: bytes) -> list[float]: ...

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Text tower of the same model, for visual-mode queries."""

    @property
    def calibration(self) -> tuple[float, float] | None:
        """(scale, bias) such that sigmoid(scale * cos + bias) is a match probability, if the model was trained
        with one (SigLIP). None = uncalibrated: image hits keep full RRF credit. Optional; default None."""
        return None


class MockTextEmbedder(TextEmbedder):
    """Deterministic bag-of-words hashing embedding so retrieval tests work offline."""

    dim = 256

    async def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


def build_enricher(settings: Settings) -> Enricher:
    # TODO(stream-1): return VLMEnricher(settings) when settings.enrich_url is reachable
    return MockEnricher()


def build_text_embedder(settings: Settings) -> TextEmbedder:
    # TODO(stream-1): openai / voyage / local
    return MockTextEmbedder()


def build_image_embedder(settings: Settings) -> ImageEmbedder | None:
    # TODO(stream-1): SigLIP via open_clip when embed_provider == "local"
    return None
