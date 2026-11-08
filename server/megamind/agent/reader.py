"""Reader sub-agent: an open-source VLM that attends over the concatenated memory tokens
(captions + OCR + thumbnails) and answers the sub-question with citations.

This is the "attention over memory tokens" component. The context layout is deliberate:

* tokens are laid out **chronologically** (oldest first) so the model can reason about
  order and "after/before";
* every token is a text part ``[id] <absolute UTC> (<n min ago>, +<delta> after first)
  caption: ... / text: ...`` immediately followed by its thumbnail, so the id the model
  cites sits right next to the pixels it saw;
* the question comes last, together with the strict JSON output schema
  ``{"answer", "evidence": [id, ...], "confidence"}`` (kept terse: output tokens dominate reader latency);
* more than ``reader_max_images`` tokens (Sail allows 20 images/request) are split into
  equal chronological chunks read concurrently and merged by confidence.

Any OpenAI-compatible ``/chat/completions`` endpoint works (Sail Research, vLLM on Modal,
Together, OpenRouter). Failures (timeout, HTTP error, unparsable JSON) degrade to
:class:`CaptionOnlyReader` with ``degraded=True`` so the orchestrator can say the images
were not inspected.
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
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from PIL import Image

from megamind.config import Settings
from megamind.models import Confidence, MemoryToken

log = logging.getLogger("megamind.reader")

_CONFIDENCES: tuple[str, ...] = get_args(Confidence)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

READER_SYSTEM_PROMPT = """You are the memory reader of an AR headset assistant. You receive a chronological list of
memory tokens, each a first-person snapshot the wearer's headset captured: an id, when it was
captured, a caption, any text read from the image, and the thumbnail itself. Answer the
question using ONLY these tokens. Look at the images: they override the captions when they
disagree. Never identify people; describe them neutrally.

Be brief: your reply is an intermediate note another model answers from, and every extra word costs
the wearer latency. Reply with one line of raw JSON, no code fence, nothing else:
{"answer":"<one short sentence; say when it was seen>","evidence":["<token id>"],"confidence":"<level>"}
Summary questions (summarize / what did I do / what have I been doing) may use up to three short
clauses, each with its time, so no part of the period is lost. Keep the specifics the question asks
for (text read, object, place). Rules: evidence lists only ids from the list, most relevant first,
at most 3. confidence is "high" (the image clearly shows it), "medium" (caption/text says it but the
image is ambiguous), "low" (a plausible but uncertain match) or "not_found": then answer
"No matching memory." with evidence []."""


# evidence is bare ids: a per-item "why" doubled the output tokens and nothing downstream reads it
# (parse_reader_json still accepts {"id", "why"} objects from older prompts / other models).
READER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": list(_CONFIDENCES)},
    },
    "required": ["answer", "evidence", "confidence"],
    "additionalProperties": False,
}

_JSON_MODES = ("schema", "object", "off")


def response_format(mode: str) -> dict | None:
    if mode == "schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "reader_output", "strict": True, "schema": READER_JSON_SCHEMA},
        }
    if mode == "object":
        return {"type": "json_object"}
    return None


def _downgrade(mode: str) -> str:
    """schema -> object -> off; used when the endpoint rejects a response_format with HTTP 400."""
    i = _JSON_MODES.index(mode) if mode in _JSON_MODES else len(_JSON_MODES) - 1
    return _JSON_MODES[min(i + 1, len(_JSON_MODES) - 1)]


_IMAGE_LOAD_SHARE = 0.35  # of reader_deadline_s spent at most loading images via image_loader (cold decodes)
_MIN_ATTEMPT_S = 1.0  # don't start an attempt with less budget left than the fastest observed call


def _retry_after_s(r: httpx.Response, default: float = 1.0, cap: float = 2.0) -> float:
    try:
        return min(cap, max(0.0, float(r.headers.get("retry-after", default))))
    except ValueError:
        return default


@dataclass
class _ChunkState:
    """Shared by the primary and fallback attempts of one chunk."""

    json_mode: str
    known: set[str]
    calls: int = 0
    last_err: Exception | None = None
    fatal: bool = False  # auth or request-shape error: no model can succeed
    hedged: bool = False  # the fallback ran while the primary was still in flight


@dataclass
class EvidenceItem:
    id: str
    why: str = ""


@dataclass
class ReaderOutput:
    evidence: str
    confidence: Confidence
    cited_ids: list[str] = field(default_factory=list)
    items: list[EvidenceItem] = field(default_factory=list)
    degraded: bool = False
    latency_ms: float = 0.0
    calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    images: list[tuple[str, str]] = field(default_factory=list)  # (token id, data URI) for Claude


class Reader(ABC):
    @abstractmethod
    async def read(self, query: str, tokens: list[MemoryToken], *, now: float | None = None) -> ReaderOutput:
        """`now` (Unix seconds) only affects how times are humanised in the prompt; defaults to wall clock."""


class CaptionOnlyReader(Reader):
    """No model call: the top captions (+ OCR) verbatim as evidence, citing the top 3.

    `top=None` lists every retrieved token in one compact line each: used when the VLM degrades so Claude
    still sees the OCR / caption of hits ranked below 3 (the answer is often there)."""

    def __init__(self, top: int | None = 3, zone: ZoneInfo | None = None) -> None:
        self.top = top
        self.zone = zone

    def _line(self, t: MemoryToken) -> str:
        when = f" {datetime.fromtimestamp(t.ts, self.zone).strftime('%a %H:%M')}:" if self.zone else ""
        line = f"[{t.id}]{when} {t.caption}"
        if t.ocr_text:
            line += f" (text: {' / '.join(t.ocr_text.splitlines())})"
        return line

    async def read(self, query: str, tokens: list[MemoryToken], *, now: float | None = None) -> ReaderOutput:
        shown = tokens if self.top is None else tokens[: self.top]
        return ReaderOutput(
            evidence="\n".join(self._line(t) for t in shown), confidence="low", cited_ids=[t.id for t in tokens[:3]]
        )


def _fmt_delta(seconds: float) -> str:
    s = abs(seconds)
    if s < 60:
        return f"{int(s)} s"
    if s < 3600:
        return f"{int(round(s / 60))} min"
    if s < 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} d"


def _wearer_zone(name: str | None) -> ZoneInfo | None:
    try:
        return ZoneInfo(name) if name else None
    except (ZoneInfoNotFoundError, ValueError):
        return None


def token_header(t: MemoryToken, first_ts: float, now: float, zone: ZoneInfo | None = None) -> str:
    # wearer-local wall time: on the no-LLM path the reader's answer is spoken verbatim
    when = (
        datetime.fromtimestamp(t.ts, zone).strftime("%a %Y-%m-%d %H:%M %Z")
        if zone
        else datetime.fromtimestamp(t.ts, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    ago = _fmt_delta(now - t.ts) + " ago" if now >= t.ts else "in the future"
    delta = "first" if t.ts == first_ts else f"+{_fmt_delta(t.ts - first_ts)} after first"
    line = f"[{t.id}] {when} ({ago}, {delta})\ncaption: {t.caption or '(none)'}"
    if t.ocr_text:
        line += f"\ntext: {t.ocr_text}"
    if t.entities:
        line += f"\nentities: {', '.join(t.entities)}"
    return line


_ENC_CACHE: OrderedDict[tuple[bytes, int], str] = OrderedDict()
_ENC_CACHE_MAX = 256  # ~12 images/turn; 256 x ~30 KB of base64
_ENC_LOCK = threading.Lock()  # encode_jpeg runs in worker threads


def encode_jpeg(data: bytes, max_px: int) -> str:
    """Base64 JPEG data URI, downscaled so the longest side is <= max_px. Raises OSError/ValueError.

    Cached by content digest: the prefetch, Claude's re-queries, follow-ups and the hedge mostly re-send the
    same hits. Keyed on the bytes, so a privacy-blurred rewrite is a miss and a purged frame is never served."""
    key = (hashlib.blake2b(data, digest_size=16).digest(), max_px)
    with _ENC_LOCK:
        if (hit := _ENC_CACHE.get(key)) is not None:
            _ENC_CACHE.move_to_end(key)
            return hit
    with Image.open(io.BytesIO(data)) as im:
        if max(im.size) > max_px:
            im.draft("RGB", (max_px, max_px))  # JPEG: decode at 1/2, 1/4, 1/8 scale (>= max_px) instead of full
            im = im.convert("RGB")
            im.thumbnail((max_px, max_px))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
    uri = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    with _ENC_LOCK:
        _ENC_CACHE[key] = uri
        while len(_ENC_CACHE) > _ENC_CACHE_MAX:
            _ENC_CACHE.popitem(last=False)
    return uri


def clear_encode_cache() -> None:
    with _ENC_LOCK:
        _ENC_CACHE.clear()


def encode_thumb(path: str, max_px: int) -> str | None:
    """`encode_jpeg` of the file at `path`; None if unreadable."""
    try:
        p = Path(path)
        if not path or not p.is_file():
            return None
        return encode_jpeg(p.read_bytes(), max_px)
    except (OSError, ValueError) as e:
        log.warning("thumb %s unreadable: %s", path, e)
        return None


ImageLoader = Callable[[MemoryToken], Awaitable[bytes | None]]


def build_user_content(
    query: str,
    tokens: list[MemoryToken],
    now: float,
    thumb_max_px: int,
    zone: ZoneInfo | None = None,
    images: dict[str, str | None] | None = None,
    sizes: dict[str, int] | None = None,
) -> list[dict]:
    """Interleaved text/image parts, chronological. Image count == number of readable thumbs.
    `images` (token id -> data URI) replaces reading `thumb_path` when given (cold-tier frames);
    `sizes` (token id -> long side) overrides `thumb_max_px` per token when thumbs are read here."""
    ordered = sorted(tokens, key=lambda t: t.ts)
    first_ts = ordered[0].ts if ordered else now
    parts: list[dict] = [
        {"type": "text", "text": f"{len(ordered)} memory tokens, oldest first. Each header is followed by its image."}
    ]
    for t in ordered:
        parts.append({"type": "text", "text": token_header(t, first_ts, now, zone)})
        if images is not None:
            uri = images.get(t.id)
        else:
            uri = encode_thumb(t.thumb_path, sizes.get(t.id, thumb_max_px) if sizes else thumb_max_px)
        if uri:
            parts.append({"type": "image_url", "image_url": {"url": uri}})
        else:
            parts.append({"type": "text", "text": "(image unavailable)"})
    parts.append({"type": "text", "text": f"Question: {query}\nReply with the JSON object only."})
    return parts


def parse_reader_json(text: str, known_ids: set[str]) -> tuple[str, Confidence, list[EvidenceItem]]:
    """Tolerant parse of the model output: strips code fences / prose, validates ids and confidence."""
    m = _JSON_RE.search(text or "")
    if not m:
        raise ValueError("no JSON object in reader output")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("reader output is not an object")
    answer = str(obj.get("answer") or "").strip()
    conf = str(obj.get("confidence") or "low").lower().strip()
    items: list[EvidenceItem] = []
    for e in obj.get("evidence") or []:
        if isinstance(e, dict):
            eid, why = str(e.get("id", "")).strip(), str(e.get("why", "")).strip()
        else:
            eid, why = str(e).strip(), ""
        if eid in known_ids and eid not in {i.id for i in items}:
            items.append(EvidenceItem(eid, why))
    if conf not in _CONFIDENCES:
        conf = "low"
    if not items and conf != "not_found":
        conf = "low" if answer else "not_found"
    if conf == "not_found":
        items = []
        answer = answer or "No matching memory."
    return answer, conf, items  # type: ignore[return-value]


def chunk_chronological(tokens: list[MemoryToken], max_per_chunk: int) -> list[list[MemoryToken]]:
    ordered = sorted(tokens, key=lambda t: t.ts)
    if len(ordered) <= max_per_chunk:
        return [ordered]
    n = math.ceil(len(ordered) / max_per_chunk)
    size = math.ceil(len(ordered) / n)
    return [ordered[i : i + size] for i in range(0, len(ordered), size)]


def merge_outputs(outs: list[ReaderOutput]) -> ReaderOutput:
    """Best confidence wins; evidence of every chunk that found something is kept, best first."""
    rank = {c: i for i, c in enumerate(_CONFIDENCES)}
    found = sorted((o for o in outs if o.confidence != "not_found"), key=lambda o: rank[o.confidence])
    base = found[0] if found else outs[0]
    merged = ReaderOutput(
        evidence=" ".join(o.evidence for o in found) if found else base.evidence,
        confidence=base.confidence,
        degraded=any(o.degraded for o in outs),
        latency_ms=max(o.latency_ms for o in outs),
        calls=sum(o.calls for o in outs),
    )
    for o in found:
        for it in o.items:
            if it.id not in merged.cited_ids:
                merged.cited_ids.append(it.id)
                merged.items.append(it)
    for o in outs:
        for k, v in o.usage.items():
            merged.usage[k] = merged.usage.get(k, 0) + v
    return merged


async def load_images(
    loader: ImageLoader,
    tokens: list[MemoryToken],
    max_px: int,
    budget_s: float,
    sizes: dict[str, int] | None = None,
) -> dict[str, str | None]:
    """Data URIs via `loader`, concurrently, within `budget_s` (cold decodes are the slow case); whatever is not
    ready by then maps to None ("(image unavailable)"). `sizes` (token id -> long side) overrides `max_px`."""

    async def one(t: MemoryToken) -> tuple[str, str | None]:
        px = sizes.get(t.id, max_px) if sizes else max_px
        try:
            data = await loader(t)
            return t.id, await asyncio.to_thread(encode_jpeg, data, px) if data else None
        except (OSError, ValueError) as e:
            log.warning("image for %s unreadable: %s", t.id, e)
            return t.id, None

    tasks = [asyncio.create_task(one(t)) for t in tokens]
    done, pending = await asyncio.wait(tasks, timeout=max(0.5, budget_s)) if tasks else (set(), set())
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
        log.warning("reader: %d/%d images not loaded within the budget", len(pending), len(tokens))
    out: dict[str, str | None] = {t.id: None for t in tokens}
    for task in done:
        if not task.cancelled() and task.exception() is None:
            tid, uri = task.result()
            out[tid] = uri
    return out


class ClaudeVisionReader(Reader):
    """No reader model: the top hits' captions/OCR as evidence plus their images, which the orchestrator (Claude)
    looks at itself in the tool result. Saves the reader round trip (~3 s on Sail) per recall.

    Payload: the best `reader_full_images` hits at `reader_thumb_max_px`, the rest at `reader_small_px` (or no image
    when that is 0). Claude bills ~w*h/750 tokens per image, so 960 -> 512 px cuts a 4:3 frame from ~922 to ~262
    tokens; the tail of the ranked list is mostly temporal context, and every caption/OCR line is still sent."""

    def __init__(self, settings: Settings, image_loader: ImageLoader | None = None) -> None:
        self.image_loader = image_loader
        self.max_images = max(1, settings.reader_max_images)
        self.thumb_max_px = settings.reader_thumb_max_px
        self.full_images = settings.reader_full_images
        self.small_px = settings.reader_small_px
        self.budget_s = settings.reader_deadline_s * _IMAGE_LOAD_SHARE
        self.captions = CaptionOnlyReader(top=None, zone=_wearer_zone(settings.wearer_tz))

    def image_sizes(self, tokens: list[MemoryToken]) -> dict[str, int]:
        """token id -> long side for the tokens that get an image (best first; absent = captions only)."""
        shown = tokens[: self.max_images]
        full = len(shown) if self.full_images <= 0 else self.full_images
        small = min(self.small_px, self.thumb_max_px)
        out: dict[str, int] = {}
        for i, t in enumerate(shown):
            if i < full:
                out[t.id] = self.thumb_max_px
            elif small > 0:
                out[t.id] = small
        return out

    async def read(self, query: str, tokens: list[MemoryToken], *, now: float | None = None) -> ReaderOutput:
        if not tokens:
            return ReaderOutput(evidence="No matching memory.", confidence="not_found")
        t0 = time.perf_counter()
        sizes = self.image_sizes(tokens)
        shown = [t for t in tokens if t.id in sizes]
        if self.image_loader is not None:
            uris = await load_images(self.image_loader, shown, self.thumb_max_px, self.budget_s, sizes)
        else:
            uris = {t.id: await asyncio.to_thread(encode_thumb, t.thumb_path, sizes[t.id]) for t in shown}
        out = await self.captions.read(query, tokens, now=now)
        out.confidence = "medium"
        out.images = [(t.id, u) for t in shown if (u := uris.get(t.id))]
        out.latency_ms = (time.perf_counter() - t0) * 1000
        return out


class VLMReader(Reader):
    """OpenAI-compatible chat.completions reader with image inputs (Qwen3.6 on Sail by default)."""

    def __init__(
        self, settings: Settings, client: httpx.AsyncClient | None = None, image_loader: ImageLoader | None = None
    ) -> None:
        self.settings = settings
        self.image_loader = image_loader  # e.g. MediaStore.read, so cold-tier frames reach the VLM
        self.model = settings.reader_model
        self.fallback_model = settings.reader_fallback_model if settings.reader_fallback_model != self.model else ""
        self.max_images = max(1, settings.reader_max_images)
        self.thumb_max_px = settings.reader_thumb_max_px
        self.full_images = settings.reader_full_images
        self.small_px = min(settings.reader_vlm_small_px, self.thumb_max_px)  # 0 = every image at thumb_max_px
        self.send_temperature = True  # cleared once the endpoint rejects it (claude-haiku-5-5): no 400 per call
        self.max_retries = settings.reader_max_retries
        self.timeout_s = settings.reader_timeout_s
        self.deadline_s = settings.reader_deadline_s
        self.hedge_s = settings.reader_hedge_s
        self.zone = _wearer_zone(settings.wearer_tz)
        self.fallback = CaptionOnlyReader(top=None, zone=self.zone)
        self.extra_body: dict[str, Any] = json.loads(settings.reader_extra_body) if settings.reader_extra_body else {}
        self.json_mode = (
            settings.reader_json_mode.lower() if settings.reader_json_mode.lower() in _JSON_MODES else "off"
        )
        headers = {"Content-Type": "application/json"}
        if settings.reader_api_key and settings.reader_api_key != "none":
            headers["Authorization"] = f"Bearer {settings.reader_api_key}"
        if settings.reader_completion_window:
            headers["X-Sail-Completion-Window"] = settings.reader_completion_window
        self.headers = headers
        self.client = client or httpx.AsyncClient(
            base_url=settings.reader_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.reader_timeout_s, connect=5.0),
        )

    async def read(self, query: str, tokens: list[MemoryToken], *, now: float | None = None) -> ReaderOutput:
        if not tokens:
            return ReaderOutput(evidence="No matching memory.", confidence="not_found")
        t0 = time.perf_counter()
        now = time.time() if now is None else now
        chunks = chunk_chronological(tokens, self.max_images)
        sizes = self.image_sizes(tokens)
        images = await self._load_images(tokens, sizes) if self.image_loader is not None else None
        outs = await asyncio.gather(
            *(
                self._read_chunk(query, c, now, images, sizes, started=t0 if images is not None else None)
                for c in chunks
            )
        )
        out = merge_outputs(list(outs)) if len(outs) > 1 else outs[0]
        out.latency_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "reader model=%s tokens=%d chunks=%d calls=%d conf=%s cited=%d degraded=%s ms=%.0f usage=%s",
            self.model,
            len(tokens),
            len(chunks),
            out.calls,
            out.confidence,
            len(out.cited_ids),
            out.degraded,
            out.latency_ms,
            out.usage,
        )
        return out

    def image_sizes(self, tokens: list[MemoryToken]) -> dict[str, int] | None:
        """token id -> long side, or None = every image at `thumb_max_px` (the default). With READER_VLM_SMALL_PX
        set, only the best `reader_full_images` hits stay at `thumb_max_px` (tokens arrive best-first, temporal
        context at the tail) and the rest shrink, mirroring ClaudeVisionReader; the tail's caption/OCR is unchanged."""
        if self.small_px <= 0 or self.full_images <= 0:
            return None
        return {t.id: self.thumb_max_px if i < self.full_images else self.small_px for i, t in enumerate(tokens)}

    async def _load_images(
        self, tokens: list[MemoryToken], sizes: dict[str, int] | None = None
    ) -> dict[str, str | None]:
        assert self.image_loader is not None
        budget = self.deadline_s * _IMAGE_LOAD_SHARE
        return await load_images(self.image_loader, tokens, self.thumb_max_px, budget, sizes)

    async def _read_chunk(
        self,
        query: str,
        tokens: list[MemoryToken],
        now: float,
        images: dict[str, str | None] | None = None,
        sizes: dict[str, int] | None = None,
        *,
        started: float | None = None,
    ) -> ReaderOutput:
        """Primary model first; if it hasn't answered after `hedge_s` (healthy calls take ~3.1 s, hung ones
        never do) or fails fast, the fallback model runs too and the first good JSON wins. Everything
        shares one wall-clock deadline, after which the chunk degrades to captions-only."""
        # JPEG decode/resize/base64 is ~5-10 ms per image: keep it off the event loop (concurrent turns, Claude I/O)
        content = await asyncio.to_thread(
            build_user_content, query, tokens, now, self.thumb_max_px, self.zone, images, sizes
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": READER_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.settings.reader_max_tokens,
            **({"temperature": 0} if self.send_temperature else {}),
            **self.extra_body,
        }
        st = _ChunkState(json_mode=self.json_mode, known={t.id for t in tokens})
        # one key per (chunk, model), reused on retries so a slow-but-successful attempt is not billed twice
        key = f"megamind-reader-{uuid.uuid4()}"
        t_start = time.perf_counter()
        deadline = (t_start if started is None else started) + self.deadline_s  # image loading counts against it
        hedge_at = t_start + self.hedge_s if self.fallback_model and self.hedge_s > 0 else None

        def launch(model: str, suffix: str, handoff: bool) -> asyncio.Task[ReaderOutput | None]:
            headers = {**self.headers, "Idempotency-Key": key + suffix}
            return asyncio.create_task(self._attempts({**body, "model": model}, headers, deadline, st, handoff))

        primary = launch(self.model, "", handoff=bool(self.fallback_model))
        running: set[asyncio.Task[ReaderOutput | None]] = {primary}
        fallback: asyncio.Task[ReaderOutput | None] | None = None
        result: ReaderOutput | None = None
        try:
            while running:
                wake = deadline if fallback is not None or hedge_at is None else min(deadline, hedge_at)
                done, _ = await asyncio.wait(
                    running, timeout=max(0.0, wake - time.perf_counter()), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    running.discard(task)
                    try:
                        out = task.result()
                    except Exception as e:  # noqa: BLE001 - never let a reader bug kill the voice turn
                        st.last_err, out = e, None
                    if out is not None and result is None:
                        result = out
                        if task is fallback:
                            result.usage["fallback_model"] = 1
                if result is not None or time.perf_counter() >= deadline:
                    break
                hedge_due = hedge_at is not None and time.perf_counter() >= hedge_at
                if fallback is None and self.fallback_model and not st.fatal and (not running or hedge_due):
                    st.hedged = bool(running)
                    if st.hedged:
                        log.info(
                            "reader %s slow after %.1f s, hedging on %s", self.model, self.hedge_s, self.fallback_model
                        )
                    fallback = launch(self.fallback_model, "-fallback", handoff=False)
                    running.add(fallback)
        finally:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
        if result is not None:
            result.calls = st.calls
            if st.hedged:
                result.usage["hedged"] = 1
            return result
        log.warning("reader degraded to captions after %d call(s): %r", st.calls, st.last_err or "deadline")
        fb = await self.fallback.read(query, tokens, now=now)
        fb.degraded, fb.calls = True, st.calls
        return fb

    async def _attempts(
        self, body: dict[str, Any], headers: dict[str, str], deadline: float, st: _ChunkState, handoff: bool
    ) -> ReaderOutput | None:
        """Calls one model until it answers (-> output) or gives up (-> None). `handoff`: the fallback model
        is waiting, so give up on capacity errors / hangs / garbage at once instead of retrying here."""
        model = body["model"]
        attempt = 0
        while attempt <= self.max_retries:
            remaining = deadline - time.perf_counter()
            if remaining < _MIN_ATTEMPT_S:
                st.last_err = st.last_err or TimeoutError("reader deadline")
                return None
            body.pop("response_format", None)
            if fmt := response_format(st.json_mode):
                body["response_format"] = fmt
            st.calls += 1
            t0 = time.perf_counter()
            budget = min(self.timeout_s, remaining)
            try:
                r = await self.client.post(
                    "/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=httpx.Timeout(budget, connect=min(5.0, budget)),
                )
                if r.status_code in (408, 409, 429) or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"reader HTTP {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":  # a cut JSON fails to parse below -> retry / captions
                    log.warning("reader %s hit max_tokens=%s, reply truncated", model, body.get("max_tokens"))
                text = choice["message"]["content"] or ""
                answer, conf, items = parse_reader_json(text, st.known)
                usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, int | float)}
                return ReaderOutput(
                    evidence=answer,
                    confidence=conf,
                    cited_ids=[i.id for i in items],
                    items=items,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    usage=usage,
                )
            except httpx.HTTPStatusError as e:
                st.last_err = e
                code = e.response.status_code
                if code in (401, 403):
                    st.fatal = True  # same key for both models: degrade now
                    log.warning("reader HTTP %d on %s (auth), degrading", code, model)
                    return None
                if code in (400, 422) and "temperature" in body and "temperature" in e.response.text.lower():
                    # newer models (e.g. claude-haiku-5-5) reject `temperature`: drop it, keep the retry budget and
                    # the json_mode (this body is this model's own copy, the other model still sends it)
                    body.pop("temperature")
                    self.send_temperature = False
                    log.warning("reader HTTP %d on %s, retrying without temperature", code, model)
                    continue
                if code in (400, 422):
                    # endpoint rejects the request shape: most likely the response_format. Downgrade
                    # schema -> object -> off (does not consume the retry budget), then give up on both models.
                    if st.json_mode != "off":
                        st.json_mode = _downgrade(st.json_mode)
                        log.warning("reader HTTP %d, retrying with json_mode=%s", code, st.json_mode)
                        continue
                    st.fatal = True
                    return None
                if 400 <= code < 500 and code not in (408, 409, 429):
                    return None  # e.g. 404 unknown model: the other model may still work
                if handoff and code in (429, 503):
                    log.warning("reader HTTP %d on %s, handing over to the fallback model", code, model)
                    return None
                if attempt < self.max_retries:
                    wait = _retry_after_s(e.response)
                    if deadline - time.perf_counter() - wait < _MIN_ATTEMPT_S:
                        return None
                    await asyncio.sleep(wait)
            except httpx.TimeoutException as e:
                st.last_err = e
                if self.fallback_model:
                    return None  # a hung pool: re-queueing on it rarely beats the deadline
            except (httpx.TransportError, KeyError, ValueError, json.JSONDecodeError) as e:
                st.last_err = e
                if handoff:
                    return None
            attempt += 1
            log.warning("reader %s attempt %d/%d failed: %r", model, attempt, self.max_retries + 1, st.last_err)
        return None


def build_reader(settings: Settings, image_loader: ImageLoader | None = None) -> Reader:
    provider = settings.reader_provider.lower()
    if provider == "mock":
        return CaptionOnlyReader()
    if provider == "claude":
        return ClaudeVisionReader(settings, image_loader=image_loader)
    if provider == "auto" and settings.anthropic_api_key:
        return ClaudeVisionReader(settings, image_loader=image_loader)
    has_key = bool(settings.reader_api_key) and settings.reader_api_key != "none"
    if provider == "vlm" or (provider == "auto" and has_key):
        return VLMReader(settings, image_loader=image_loader)
    return CaptionOnlyReader()
