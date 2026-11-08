"""Turn an accepted keyframe into caption / OCR / entities (the embedders live in `megamind.store.embeddings`).

Enrichers (`ENRICH_PROVIDER=auto|vlm|local|mock`, picked by `build_enricher`):
- `VLMEnricher`: any OpenAI-compatible chat endpoint with image input (`ENRICH_BASE_URL`, default the reader's Sail
  Gemma 4) gets the JPEG as a data URI plus `ENRICH_PROMPT`; the JSON answer is parsed tolerantly. One retry on
  429/5xx/timeout, then the fallback enricher, so a VLM outage never drops frames.
- `LocalEnricher`: keyless. Tesseract OCR (pytesseract, `local-enrich` extra + the `tesseract` binary) and a
  zero-shot scene label from the image embedder's text tower (SigLIP).
- `MockEnricher`: deterministic `mock caption for frame <sha>` (tests, no keys, nothing installed).
`auto` = vlm iff an enrich/reader key is set, else local if a real image embedder is configured, else mock.
Ported from PR #54 (Devin) onto main's prompt/config, with httpx instead of the OpenAI SDK.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import logging
import math
import re
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

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
    # optional: the image vector an enricher already computed with the worker's image embedder (LocalEnricher's
    # zero-shot label), so the worker does not embed the same JPEG twice
    image_vec: list[float] | None = field(default=None, repr=False, compare=False)
    image_vec_from: object | None = field(default=None, repr=False, compare=False)  # the embedder that made it


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
        """Vector for a search query. Asymmetric models (e.g. bge's query instruction) override this;
        the default is the document embedding, so existing embedders need no change."""
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


# ---------------------------------------------------------------------- real enrichers

_JSON = re.compile(r"\{.*\}", re.S)
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# zero-shot scene vocabulary for LocalEnricher (from PR #54); prompted as "a photo of <label>"
SCENE_LABELS = [
    "a kitchen", "a living room", "a bedroom", "a bathroom", "an office", "a desk with a computer",
    "a meeting room", "a whiteboard", "a classroom or lecture hall", "a hallway", "a staircase",
    "a street", "a parking lot", "a car interior", "a shop or supermarket", "a restaurant or cafe",
    "a park", "a garden", "a gym", "a lab or workshop", "a library or bookshelf", "a dining table",
    "a laptop screen", "a phone screen", "a poster or sign", "people talking", "a hackathon venue",
    "outdoors in nature", "a store shelf with products", "a doorway or entrance",
]  # fmt: skip


def parse_enrichment(text: str) -> Enrichment:
    """Tolerant parse of the VLM answer (prose / code fences around the object are fine). Raises ValueError
    when there is no JSON object in it."""
    m = _JSON.search(text or "")
    if not m:
        raise ValueError("no JSON object in enricher output")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"bad JSON in enricher output: {e}") from e
    if not isinstance(d, dict):
        raise ValueError("enricher output is not an object")
    ents = d.get("entities") or []
    if isinstance(ents, str):
        ents = re.split(r"[,\n]", ents)
    ocr = d.get("ocr_text") or ""
    if isinstance(ocr, list):
        ocr = "\n".join(str(x) for x in ocr)
    return Enrichment(
        caption=str(d.get("caption") or "").strip(),
        ocr_text=str(ocr).strip(),
        entities=[str(e).strip() for e in ents if str(e).strip()][:15] if isinstance(ents, list) else [],
    )


def tesseract_available() -> bool:
    return importlib.util.find_spec("pytesseract") is not None and shutil.which("tesseract") is not None


class LocalEnricher(Enricher):
    """Keyless: Tesseract OCR (exact text) + zero-shot scene caption from the image embedder's text tower.
    Either half may be missing (no pytesseract/tesseract, or no image embedder); the other still runs."""

    def __init__(
        self, image_embedder: ImageEmbedder | None = None, *, labels: list[str] | None = None, ocr: bool | None = None
    ) -> None:
        self.image_embedder = image_embedder
        self.labels = labels or SCENE_LABELS
        self.ocr = tesseract_available() if ocr is None else ocr
        self._label_vecs: list[list[float]] | None = None
        self._label_lock = asyncio.Lock()

    @property
    def degraded(self) -> str | None:
        """For /v1/health: set when the scene caption cannot run (captions are OCR text, or empty)."""
        if self.image_embedder is not None:
            return None
        return "no image embedder: OCR-only captions" if self.ocr else "no image embedder and no tesseract"

    @staticmethod
    def available(image_embedder: ImageEmbedder | None) -> bool:
        """True with a real (non-mock) image embedder: it writes the caption, and an uncaptioned token is invisible
        to search, so OCR alone does not qualify for `auto`."""
        return image_embedder is not None and not type(image_embedder).__name__.startswith("Mock")

    @staticmethod
    def _ocr(jpeg: bytes) -> str:
        import io

        import pytesseract
        from PIL import Image

        text = pytesseract.image_to_string(Image.open(io.BytesIO(jpeg)), config="--psm 11")
        # psm 11 (sparse text) on photos yields noise: keep tokens that look like words / numbers
        words = [w for w in text.split() if len(w) >= 3 and sum(c.isalnum() for c in w) / len(w) >= 0.7]
        return " ".join(words)

    async def _labels(self) -> list[list[float]]:
        async with self._label_lock:
            if self._label_vecs is None:
                assert self.image_embedder is not None
                self._label_vecs = [await self.image_embedder.embed_text(f"a photo of {lbl}") for lbl in self.labels]
            return self._label_vecs

    async def enrich(self, jpeg: bytes) -> Enrichment:
        ocr = ""
        if self.ocr:
            try:
                ocr = await asyncio.to_thread(self._ocr, jpeg)
            except Exception as e:  # noqa: BLE001  (missing binary, undecodable image)
                log.warning("local OCR failed: %s", e)
        caption = ""
        if self.image_embedder is not None:
            vecs = await self._labels()
            v = await self.image_embedder.embed_image(jpeg)
            best = max(range(len(vecs)), key=lambda i: sum(a * b for a, b in zip(v, vecs[i])))
            caption = self.labels[best]
            return Enrichment(
                caption=caption, ocr_text=ocr, entities=[], image_vec=v, image_vec_from=self.image_embedder
            )
        if ocr:
            caption = f"Text visible: {ocr}"  # search only sees captioned tokens
        return Enrichment(caption=caption, ocr_text=ocr, entities=[])


class VLMEnricher(Enricher):
    """`POST {enrich_url}/chat/completions` with the JPEG as a data URI and ENRICH_PROMPT. Never raises: after
    one retry (429/5xx/timeout/transport) or on an unparseable answer it returns `fallback.enrich(jpeg)` (or an
    empty enrichment), and `degraded` says why for /v1/health until the next success. After a failed call (not
    an unparseable answer) the endpoint is skipped for `cooldown_s`: during an outage every frame would otherwise
    spend ~2x the timeout in retries, the worker queue would fill and drop frames (the fallback exists to avoid
    exactly that)."""

    def __init__(
        self,
        settings: Settings,
        *,
        fallback: Enricher | None = None,
        client: httpx.AsyncClient | None = None,
        retry_wait_s: float = 0.5,
        cooldown_s: float = 30.0,
    ) -> None:
        self.model = settings.enrich_model_name
        self.max_tokens = settings.enrich_max_tokens
        self.fallback = fallback
        self.retry_wait_s = retry_wait_s
        self.cooldown_s = cooldown_s
        self._skip_until = 0.0
        self.degraded: str | None = None
        same_endpoint = not settings.enrich_base_url
        extra = settings.reader_extra_body if same_endpoint else ""
        self.extra_body: dict[str, Any] = json.loads(extra) if extra else {}
        headers = {"Content-Type": "application/json"}
        if settings.enrich_key:
            headers["Authorization"] = f"Bearer {settings.enrich_key}"
        if same_endpoint and settings.reader_completion_window:
            headers["X-Sail-Completion-Window"] = settings.reader_completion_window
        self.client = client or httpx.AsyncClient(
            base_url=settings.enrich_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.enrich_timeout_s, connect=5.0),
        )
        if client is not None:
            self.client.headers.update(headers)

    def _body(self, jpeg: bytes) -> dict[str, Any]:
        uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        return {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ENRICH_PROMPT},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                }
            ],
            **self.extra_body,
        }

    async def _call(self, jpeg: bytes) -> str:
        body = self._body(jpeg)
        last: Exception | None = None
        for attempt in range(2):
            try:
                r = await self.client.post("/chat/completions", json=body)
                if r.status_code in _RETRY_STATUS and attempt == 0:
                    last = httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                    await asyncio.sleep(self.retry_wait_s)
                    continue
                r.raise_for_status()
                return str(r.json()["choices"][0]["message"]["content"] or "")
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
                if attempt == 0:
                    await asyncio.sleep(self.retry_wait_s)
                    continue
                raise
        raise last or RuntimeError("enricher retry loop exhausted")

    async def _fallback(self, jpeg: bytes, why: str, *, cool: bool = False, quiet: bool = False) -> Enrichment:
        self.degraded = why
        if cool and self.cooldown_s > 0:
            self._skip_until = time.monotonic() + self.cooldown_s
        if not quiet:
            using = type(self.fallback).__name__ if self.fallback else "empty enrichment"
            extra = f", skipping the VLM for {self.cooldown_s:g} s" if cool and self.cooldown_s > 0 else ""
            log.warning("VLM enrichment failed (%s); using %s%s", why, using, extra)
        if self.fallback is None:
            return Enrichment()
        try:
            return await self.fallback.enrich(jpeg)
        except Exception:  # noqa: BLE001
            log.exception("fallback enricher failed")
            return Enrichment()

    async def enrich(self, jpeg: bytes) -> Enrichment:
        if time.monotonic() < self._skip_until:
            return await self._fallback(jpeg, self.degraded or "cooling down", quiet=True)
        try:
            text = await self._call(jpeg)
        except httpx.HTTPStatusError as e:
            return await self._fallback(jpeg, f"last error {e.response.status_code}", cool=True)
        except httpx.TimeoutException:
            return await self._fallback(jpeg, "last error timeout", cool=True)
        except httpx.HTTPError as e:
            return await self._fallback(jpeg, f"last error {type(e).__name__}", cool=True)
        except (KeyError, IndexError, TypeError, ValueError) as e:  # malformed completion body
            return await self._fallback(jpeg, f"last error {type(e).__name__}")
        try:
            e = parse_enrichment(text)
        except ValueError:
            return await self._fallback(jpeg, "last error unparseable answer")
        self.degraded = None
        return e


def resolve_enrich_provider(settings: Settings, image_embedder: ImageEmbedder | None = None) -> str:
    """The enricher `build_enricher` will build: vlm | local | mock."""
    p = settings.enrich_provider.lower()
    if p in ("vlm", "local", "mock"):
        return p
    if settings.enrich_key:
        return "vlm"
    return "local" if LocalEnricher.available(image_embedder) else "mock"


def build_enricher(settings: Settings, image_embedder: ImageEmbedder | None = None) -> Enricher:
    p = resolve_enrich_provider(settings, image_embedder)
    if p == "vlm":
        fallback = LocalEnricher(image_embedder) if LocalEnricher.available(image_embedder) else None
        return VLMEnricher(settings, fallback=fallback)
    if p == "local":
        if not LocalEnricher.available(image_embedder):
            log.warning("ENRICH_PROVIDER=local without an image embedder: captions are OCR text only (or empty)")
        return LocalEnricher(image_embedder)
    return MockEnricher()


def build_text_embedder(settings: Settings) -> TextEmbedder:
    # TODO(stream-1): openai / voyage / local
    return MockTextEmbedder()


def build_image_embedder(settings: Settings) -> ImageEmbedder | None:
    # TODO(stream-1): SigLIP via open_clip when embed_provider == "local"
    return None
