"""Embedding providers for memory tokens.

Text: `OpenAITextEmbedder` (text-embedding-3-small, truncated to 1024-d per docs/api.md) | `MockTextEmbedder`.
Image: `OpenCLIPImageEmbedder` (open_clip SigLIP ViT-B/16 768-d default, or ViT-B/32 512-d; CPU; also the text
tower for visual queries)
| `MockImageEmbedder` | None.

`main.py` builds both via the factories here. `OpenAITextEmbedder` never raises on API errors (bad key, 5xx/429
after retries, timeouts): it falls back to a same-dim `MockTextEmbedder` vector for that call and sets
`degraded` (e.g. "last error 401") for /v1/health, so frames are still stored and BM25/UCC search still work.

Env: EMBED_PROVIDER=mock|openai  EMBED_MODEL=text-embedding-3-small  EMBED_DIM=1024  OPENAI_API_KEY
     IMAGE_EMBED_PROVIDER=none|mock|openclip  IMAGE_EMBED_MODEL=ViT-B-16-SigLIP  IMAGE_EMBED_PRETRAINED=webli
     (faster/smaller: IMAGE_EMBED_MODEL=ViT-B-32 IMAGE_EMBED_PRETRAINED=laion2b_s34b_b79k, 512-d)
Measured on 8 CPU threads (960x720 JPEG): SigLIP 6.7 img/s, ViT-B-32 9.1 img/s (model load 4-6 s).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import math
import os
import threading
from typing import Any

import httpx

from megamind.config import Settings
from megamind.ingest.enrich import ImageEmbedder, MockTextEmbedder, TextEmbedder

log = logging.getLogger(__name__)

__all__ = [
    "ImageEmbedder",
    "MockImageEmbedder",
    "MockTextEmbedder",
    "OpenAITextEmbedder",
    "OpenCLIPImageEmbedder",
    "TextEmbedder",
    "build_image_embedder",
    "build_text_embedder",
    "normalize",
]

DEFAULT_TEXT_DIM = 1024
OPENAI_MAX_CHARS = 8000
DEFAULT_IMAGE_MODEL = "ViT-B-16-SigLIP"
DEFAULT_IMAGE_PRETRAINED = "webli"


def normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _setting(settings: Settings, name: str, default: str) -> str:
    """Settings field if config.py has it, else env var, else default (keeps config.py out of this PR)."""
    val = getattr(settings, name, None)
    if val is None or val == "":
        val = os.environ.get(name.upper(), default)
    return str(val)


# ---------------------------------------------------------------------- text


class OpenAITextEmbedder(TextEmbedder):
    """`POST /v1/embeddings` with `dimensions=dim` (text-embedding-3-* support Matryoshka truncation)."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = DEFAULT_TEXT_DIM,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )
        self.degraded: str | None = None  # e.g. "last error 401"; None while the API is answering
        self._fallback = MockTextEmbedder()
        self._fallback.dim = dim  # same width as the real vectors, so the LanceDB table accepts them

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        try:
            vecs = await self._embed_remote(texts)
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
            reason = f"last error {_error_label(e)}"
            if self.degraded is None:
                log.warning("OpenAI embeddings failed (%s); using mock vectors until it recovers", reason)
            self.degraded = reason
            return [await self._fallback.embed(t) for t in texts]
        if self.degraded is not None:
            log.info("OpenAI embeddings recovered after %s", self.degraded)
            self.degraded = None
        return vecs

    async def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        inputs = [(t or " ")[:OPENAI_MAX_CHARS] for t in texts]
        body: dict[str, Any] = {"model": self.model, "input": inputs}
        if self.model.startswith("text-embedding-3"):
            body["dimensions"] = self.dim
        for attempt in range(3):
            r = await self._client.post("/embeddings", json=body)
            if r.status_code in (429, 500, 502, 503) and attempt < 2:
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            r.raise_for_status()
            break
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [normalize(d["embedding"][: self.dim]) for d in data]


def _error_label(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        return str(e.response.status_code)
    if isinstance(e, httpx.TimeoutException):
        return "timeout"
    if isinstance(e, httpx.HTTPError):
        return type(e).__name__
    return "bad response"


# ---------------------------------------------------------------------- image


class MockImageEmbedder(ImageEmbedder):
    """Deterministic 4x4 RGB colour layout (48-d); the text tower maps colour words. Offline tests only."""

    dim = 48
    _COLOURS = {"red": (255, 0, 0), "green": (0, 160, 0), "blue": (0, 0, 255), "white": (255, 255, 255)}

    async def embed_image(self, jpeg: bytes) -> list[float]:
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((4, 4))
            return normalize([c / 255.0 for c in img.tobytes()])
        except Exception:  # noqa: BLE001  (no PIL / not an image: hash fallback, still deterministic)
            h = hashlib.sha256(jpeg).digest()
            return normalize([b / 255.0 for b in (h * 2)[: self.dim]])

    async def embed_text(self, text: str) -> list[float]:
        for word, rgb in self._COLOURS.items():
            if word in text.lower():
                return normalize([c / 255.0 for c in rgb] * 16)
        return normalize([1.0] * self.dim)


class OpenCLIPImageEmbedder(ImageEmbedder):
    """CPU image embedder via open_clip. SigLIP ViT-B-16 (768-d, ~7 img/s on 8 threads) or ViT-B-32 (512-d, ~9 img/s).

    Inference runs in a worker thread under a lock (torch is not re-entrant-safe per model); the event loop
    is never blocked. `torch.set_num_threads` defaults to the CPU count.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_IMAGE_MODEL,
        pretrained: str = DEFAULT_IMAGE_PRETRAINED,
        *,
        threads: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        import open_clip
        import torch

        self.torch = torch
        torch.set_num_threads(threads or max(1, os.cpu_count() or 1))
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir
        )
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model_name = model_name
        with torch.no_grad():
            self.dim = int(self.model.encode_text(self.tokenizer(["x"])).shape[-1])
        self._lock = threading.Lock()

    def _image(self, jpeg: bytes) -> list[float]:
        from PIL import Image

        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        x = self.preprocess(img).unsqueeze(0)
        with self._lock, self.torch.no_grad():
            return normalize(self.model.encode_image(x)[0].tolist())

    def _text(self, text: str) -> list[float]:
        with self._lock, self.torch.no_grad():
            return normalize(self.model.encode_text(self.tokenizer([text]))[0].tolist())

    async def embed_image(self, jpeg: bytes) -> list[float]:
        return await asyncio.to_thread(self._image, jpeg)

    async def embed_text(self, text: str) -> list[float]:
        """Visual-mode query: callers should pass `"a photo of {phrase}"`."""
        return await asyncio.to_thread(self._text, text)


# ---------------------------------------------------------------------- factories


def build_text_embedder(settings: Settings) -> TextEmbedder:
    provider = _setting(settings, "embed_provider", "mock")
    if provider == "openai":
        if settings.openai_api_key:
            return OpenAITextEmbedder(
                settings.openai_api_key,
                _setting(settings, "embed_model", "text-embedding-3-small"),
                int(_setting(settings, "embed_dim", str(DEFAULT_TEXT_DIM))),
            )
        log.warning("EMBED_PROVIDER=openai but OPENAI_API_KEY is empty; using MockTextEmbedder")
    elif provider != "mock":
        log.warning("unknown EMBED_PROVIDER=%s; using MockTextEmbedder", provider)
    return MockTextEmbedder()


def build_image_embedder(settings: Settings) -> ImageEmbedder | None:
    provider = _setting(settings, "image_embed_provider", "none")
    if provider in ("openclip", "siglip", "local"):
        try:
            return OpenCLIPImageEmbedder(
                _setting(settings, "image_embed_model", DEFAULT_IMAGE_MODEL),
                _setting(settings, "image_embed_pretrained", DEFAULT_IMAGE_PRETRAINED),
            )
        except ImportError:
            log.warning(
                "IMAGE_EMBED_PROVIDER=%s but open_clip/torch not installed (pip install -e '.[local-embeddings]')",
                provider,
            )
            return None
    if provider == "mock":
        return MockImageEmbedder()
    return None
