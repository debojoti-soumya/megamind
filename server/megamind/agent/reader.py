"""Reader sub-agent: an open-source VLM that attends over the concatenated memory tokens
(captions + OCR + thumbnails) and answers the sub-question with citations.

This is the "attention over memory tokens" component. The context layout is deliberate:

* tokens are laid out **chronologically** (oldest first) so the model can reason about
  order and "after/before";
* every token is a text part ``[id] <absolute UTC> (<n min ago>, +<delta> after first)
  caption: ... / text: ...`` immediately followed by its thumbnail, so the id the model
  cites sits right next to the pixels it saw;
* the question comes last, together with the strict JSON output schema
  ``{"answer", "evidence": [{"id", "why"}], "confidence"}``;
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
import io
import json
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
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

Reply with strict JSON and nothing else:
{"answer": "<one or two sentences, mention when it was seen>",
 "evidence": [{"id": "<token id>", "why": "<what in this token supports the answer>"}],
 "confidence": "high" | "medium" | "low" | "not_found"}
Rules: cite only ids from the list, most relevant first, at most 4. Use "not_found" with an
empty evidence list and answer "No matching memory." when nothing answers the question.
"high" = the image clearly shows it; "medium" = caption/text says it but the image is
ambiguous; "low" = a plausible but uncertain match."""


READER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "why": {"type": "string"}},
                "required": ["id", "why"],
                "additionalProperties": False,
            },
        },
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


_MIN_ATTEMPT_S = 1.0  # don't start an attempt with less budget left than the fastest observed call


def _retry_after_s(r: httpx.Response, default: float = 1.0, cap: float = 2.0) -> float:
    try:
        return min(cap, max(0.0, float(r.headers.get("retry-after", default))))
    except ValueError:
        return default


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


class Reader(ABC):
    @abstractmethod
    async def read(self, query: str, tokens: list[MemoryToken], *, now: float | None = None) -> ReaderOutput:
        """`now` (Unix seconds) only affects how times are humanised in the prompt; defaults to wall clock."""


class CaptionOnlyReader(Reader):
    """Fallback with no model call: returns the top captions verbatim as evidence."""

    async def read(self, query: str, tokens: list[MemoryToken], *, now: float | None = None) -> ReaderOutput:
        top = tokens[:3]
        lines = [f"[{t.id}] {t.caption}" + (f" (text: {t.ocr_text})" if t.ocr_text else "") for t in top]
        return ReaderOutput(evidence="\n".join(lines), confidence="low", cited_ids=[t.id for t in top])


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


def encode_thumb(path: str, max_px: int) -> str | None:
    """Base64 JPEG data URI, downscaled so the longest side is <= max_px; None if unreadable."""
    try:
        p = Path(path)
        if not path or not p.is_file():
            return None
        data = p.read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            if max(im.size) > max_px:
                im = im.convert("RGB")
                im.thumbnail((max_px, max_px))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=85)
                data = buf.getvalue()
    except (OSError, ValueError) as e:
        log.warning("thumb %s unreadable: %s", path, e)
        return None
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def build_user_content(
    query: str, tokens: list[MemoryToken], now: float, thumb_max_px: int, zone: ZoneInfo | None = None
) -> list[dict]:
    """Interleaved text/image parts, chronological. Image count == number of readable thumbs."""
    ordered = sorted(tokens, key=lambda t: t.ts)
    first_ts = ordered[0].ts if ordered else now
    parts: list[dict] = [
        {"type": "text", "text": f"{len(ordered)} memory tokens, oldest first. Each header is followed by its image."}
    ]
    for t in ordered:
        parts.append({"type": "text", "text": token_header(t, first_ts, now, zone)})
        uri = encode_thumb(t.thumb_path, thumb_max_px)
        if uri:
            parts.append({"type": "image_url", "image_url": {"url": uri, "detail": "low"}})
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


class VLMReader(Reader):
    """OpenAI-compatible chat.completions reader with image inputs (Qwen3.6 on Sail by default)."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.model = settings.reader_model
        self.fallback_model = settings.reader_fallback_model if settings.reader_fallback_model != self.model else ""
        self.max_images = max(1, settings.reader_max_images)
        self.thumb_max_px = settings.reader_thumb_max_px
        self.max_retries = settings.reader_max_retries
        self.timeout_s = settings.reader_timeout_s
        self.deadline_s = settings.reader_deadline_s
        self.zone = _wearer_zone(settings.wearer_tz)
        self.fallback = CaptionOnlyReader()
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
        outs = await asyncio.gather(*(self._read_chunk(query, c, now) for c in chunks))
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

    async def _read_chunk(self, query: str, tokens: list[MemoryToken], now: float) -> ReaderOutput:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": READER_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_content(query, tokens, now, self.thumb_max_px, self.zone)},
            ],
            "max_tokens": self.settings.reader_max_tokens,
            "temperature": 0,
            **self.extra_body,
        }
        json_mode = self.json_mode
        if fmt := response_format(json_mode):
            body["response_format"] = fmt
        # one key per chunk, reused on retries so a slow-but-successful first attempt is not billed twice
        headers = {**self.headers, "Idempotency-Key": f"megamind-reader-{uuid.uuid4()}"}
        known = {t.id for t in tokens}
        calls = 0
        downgrades = 0
        last_err: Exception | None = None
        attempt = 0
        used_fallback = False
        # wall-clock budget for the whole chunk (all attempts): the voice turn must not wait 2x timeout
        deadline = time.perf_counter() + self.deadline_s
        while attempt <= self.max_retries:
            remaining = deadline - time.perf_counter()
            if remaining < _MIN_ATTEMPT_S:
                last_err = last_err or TimeoutError("reader deadline")
                break
            calls += 1
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
                text = data["choices"][0]["message"]["content"] or ""
                answer, conf, items = parse_reader_json(text, known)
                usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, int | float)}
                if used_fallback:
                    usage["fallback_model"] = 1
                return ReaderOutput(
                    evidence=answer,
                    confidence=conf,
                    cited_ids=[i.id for i in items],
                    items=items,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    calls=calls,
                    usage=usage,
                )
            except httpx.HTTPStatusError as e:
                last_err = e
                code = e.response.status_code
                if 400 <= code < 500 and code not in (408, 409, 429):
                    # endpoint rejects the request shape: most likely the response_format. Downgrade
                    # schema -> object -> off (does not consume the retry budget), then give up.
                    if json_mode != "off" and downgrades < len(_JSON_MODES) - 1:
                        json_mode = _downgrade(json_mode)
                        downgrades += 1
                        body.pop("response_format", None)
                        if fmt := response_format(json_mode):
                            body["response_format"] = fmt
                        log.warning("reader HTTP %d, retrying with json_mode=%s", code, json_mode)
                        continue
                    break
                if code in (429, 503) and self.fallback_model and not used_fallback:
                    # capacity problem on the primary pool: retry on the fallback model (different pool)
                    used_fallback, body["model"] = True, self.fallback_model
                    log.warning("reader HTTP %d on %s, retrying on %s", code, self.model, self.fallback_model)
                    continue
                if attempt < self.max_retries:
                    wait = _retry_after_s(e.response)
                    if deadline - time.perf_counter() - wait < _MIN_ATTEMPT_S:
                        break
                    await asyncio.sleep(wait)
            except httpx.TimeoutException as e:
                last_err = e
                if self.fallback_model and not used_fallback:
                    # a slow primary pool: spend the retry on the fallback model rather than queueing again
                    used_fallback, body["model"] = True, self.fallback_model
            except (httpx.TransportError, KeyError, ValueError, json.JSONDecodeError) as e:
                last_err = e
            attempt += 1
            log.warning("reader attempt %d/%d failed: %r", attempt, self.max_retries + 1, last_err)
        log.warning("reader degraded to captions after %d call(s): %r", calls, last_err)
        fb = await self.fallback.read(query, tokens)
        fb.degraded, fb.calls = True, calls
        return fb


def build_reader(settings: Settings) -> Reader:
    provider = settings.reader_provider.lower()
    if provider == "mock":
        return CaptionOnlyReader()
    has_key = bool(settings.reader_api_key) and settings.reader_api_key != "none"
    if provider == "vlm" or (provider == "auto" and has_key):
        return VLMReader(settings)
    return CaptionOnlyReader()
