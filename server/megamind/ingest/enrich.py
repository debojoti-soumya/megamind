"""Turn an accepted keyframe into caption / OCR / entities + embeddings.

Enricher: any OpenAI-compatible VLM (vLLM Qwen2.5-VL, OpenRouter, OpenAI) via ENRICH_* settings.
Text embedders: openai | voyage | local (sentence-transformers) | mock.
Image embedders: siglip (open_clip, also provides the text tower for visual queries) | mock | none.
All vectors are L2-normalised so cosine == dot product in every store.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from megamind.config import Settings

log = logging.getLogger(__name__)

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

    async def embed_query(self, text: str) -> list[float]:
        """Query-side embedding; asymmetric models (bge) override this with their query prefix."""
        return await self.embed(text)


class ImageEmbedder(ABC):
    dim: int

    @abstractmethod
    async def embed_image(self, jpeg: bytes) -> list[float]: ...

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Text tower of the same model, for visual-mode queries."""

    @property
    def calibration(self) -> tuple[float, float] | None:
        """(scale, bias) such that sigmoid(scale * cos + bias) is a match probability, if the model has one."""
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


def _normalise(v) -> list[float]:
    v = [float(x) for x in v]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


_JSON = re.compile(r"\{.*\}", re.S)


def parse_enrichment(text: str) -> Enrichment:
    m = _JSON.search(text or "")
    if not m:
        return Enrichment(caption=(text or "").strip()[:500])
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Enrichment(caption=(text or "").strip()[:500])
    ents = d.get("entities") or []
    return Enrichment(
        caption=str(d.get("caption") or "").strip(),
        ocr_text=str(d.get("ocr_text") or "").strip(),
        entities=[str(e).strip() for e in ents if str(e).strip()][:15] if isinstance(ents, list) else [],
    )


class VLMEnricher(Enricher):
    """OpenAI-compatible chat.completions with the JPEG as a data URI."""

    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        self.model = settings.enrich_model_name
        self.client = AsyncOpenAI(base_url=settings.enrich_url, api_key=settings.reader_api_key or "none")

    async def enrich(self, jpeg: bytes) -> Enrichment:
        uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        r = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ENRICH_PROMPT},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                }
            ],
        )
        return parse_enrichment(r.choices[0].message.content or "")


class OpenAITextEmbedder(TextEmbedder):
    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}

    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        self.model = settings.embed_model
        self.dim = self._DIMS.get(self.model, 1536)
        self.client = AsyncOpenAI(api_key=settings.openai_api_key or None)

    async def embed(self, text: str) -> list[float]:
        r = await self.client.embeddings.create(model=self.model, input=text[:8000] or " ")
        return _normalise(r.data[0].embedding)


class VoyageTextEmbedder(TextEmbedder):
    def __init__(self, settings: Settings) -> None:
        import httpx

        self.model = settings.embed_model if settings.embed_model.startswith("voyage") else "voyage-3"
        self.dim = 1024
        self.http = httpx.AsyncClient(
            base_url="https://api.voyageai.com/v1",
            headers={"Authorization": f"Bearer {settings.voyage_api_key}"},
            timeout=30,
        )

    async def embed(self, text: str) -> list[float]:
        r = await self.http.post("/embeddings", json={"model": self.model, "input": [text[:8000] or " "]})
        r.raise_for_status()
        return _normalise(r.json()["data"][0]["embedding"])


class LocalTextEmbedder(TextEmbedder):
    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.local_text_model, device="cpu")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.query_prefix = (
            "Represent this sentence for searching relevant passages: " if "bge" in settings.local_text_model else ""
        )
        self._lock = threading.Lock()

    def _embed(self, text: str) -> list[float]:
        with self._lock:
            return _normalise(self.model.encode(text or " ", normalize_embeddings=True))

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed, text)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed, self.query_prefix + text)


class SigLIPImageEmbedder(ImageEmbedder):
    def __init__(self, settings: Settings) -> None:
        import open_clip
        import torch

        self.torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.image_embed_model, pretrained=settings.image_embed_pretrained
        )
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(settings.image_embed_model)
        with torch.no_grad():
            self.dim = int(self.model.encode_text(self.tokenizer(["x"])).shape[-1])
        self._lock = threading.Lock()

    def _image(self, jpeg: bytes) -> list[float]:
        from PIL import Image

        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        with self._lock, self.torch.no_grad():
            return _normalise(self.model.encode_image(self.preprocess(img).unsqueeze(0))[0].tolist())

    def _text(self, text: str) -> list[float]:
        with self._lock, self.torch.no_grad():
            return _normalise(self.model.encode_text(self.tokenizer([text]))[0].tolist())

    async def embed_image(self, jpeg: bytes) -> list[float]:
        return await asyncio.to_thread(self._image, jpeg)

    async def embed_text(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._text, text)

    @property
    def calibration(self) -> tuple[float, float] | None:
        bias = getattr(self.model, "logit_bias", None)
        if bias is None:  # CLIP-style models have no sigmoid bias
            return None
        return float(self.model.logit_scale.exp().detach()), float(bias.detach())


class MockImageEmbedder(ImageEmbedder):
    """Coarse colour layout (4x4 grid x RGB); text tower maps colour words. For offline tests only."""

    dim = 48
    _COLOURS = {"red": (255, 0, 0), "green": (0, 160, 0), "blue": (0, 0, 255), "white": (255, 255, 255)}

    async def embed_image(self, jpeg: bytes) -> list[float]:
        from PIL import Image

        img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((4, 4))
        return _normalise([c / 255.0 for px in img.getdata() for c in px])

    async def embed_text(self, text: str) -> list[float]:
        for word, rgb in self._COLOURS.items():
            if word in text.lower():
                return _normalise([c / 255.0 for c in rgb] * 16)
        return [0.0] * self.dim  # no visual concept in the query: image signal abstains


class LocalEnricher(Enricher):
    """Keyless fallback: Tesseract OCR (exact text) + zero-shot scene caption from the image embedder."""

    def __init__(self, image_embedder: ImageEmbedder | None, labels: list[str] | None = None) -> None:
        from megamind.ingest.episodes import SCENE_LABELS

        self.image_embedder = image_embedder
        self.labels = labels or SCENE_LABELS
        self._label_vecs: list[list[float]] | None = None

    @staticmethod
    def _ocr(jpeg: bytes) -> str:
        import shutil
        import subprocess

        if not shutil.which("tesseract"):
            return ""
        r = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "11"], input=jpeg, capture_output=True)
        words = [w for w in r.stdout.decode(errors="ignore").split() if len(w) >= 3 and any(c.isalnum() for c in w)]
        # psm 11 on photos yields noise; keep tokens that look like words/numbers
        return " ".join(w for w in words if sum(c.isalnum() for c in w) / len(w) >= 0.7)

    async def enrich(self, jpeg: bytes) -> Enrichment:
        ocr = await asyncio.to_thread(self._ocr, jpeg)
        caption = ""
        if self.image_embedder is not None:
            if self._label_vecs is None:
                self._label_vecs = [await self.image_embedder.embed_text(f"a photo of {lbl}") for lbl in self.labels]
            v = await self.image_embedder.embed_image(jpeg)
            best = max(range(len(self.labels)), key=lambda i: sum(a * b for a, b in zip(v, self._label_vecs[i])))
            caption = self.labels[best]
        return Enrichment(caption=caption, ocr_text=ocr, entities=[])


def build_enricher(settings: Settings, image_embedder: ImageEmbedder | None = None) -> Enricher:
    p = settings.enrich_provider
    if p == "vlm" or (p == "auto" and settings.enrich_base_url):
        return VLMEnricher(settings)
    if p == "local":
        return LocalEnricher(image_embedder)
    return MockEnricher()


def build_text_embedder(settings: Settings) -> TextEmbedder:
    p = settings.embed_provider
    if p == "openai":
        return OpenAITextEmbedder(settings)
    if p == "voyage":
        return VoyageTextEmbedder(settings)
    if p == "local":
        return LocalTextEmbedder(settings)
    return MockTextEmbedder()


def build_image_embedder(settings: Settings) -> ImageEmbedder | None:
    p = settings.image_embed_provider
    if p == "siglip":
        return SigLIPImageEmbedder(settings)
    if p == "mock":
        return MockImageEmbedder()
    return None
