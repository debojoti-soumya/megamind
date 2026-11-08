"""Object markers: "where is my pen?" -> a bounding box in the cited frame, so the headset can pin it in the room.

After a chat turn whose question asks where something is (`is_where_question`) and whose answer cites a frame
with pixels, one vision call (forced `report_location` tool; ``LOCATE_PROVIDER=openai`` GPT by default,
``anthropic`` for Claude) looks at up to `LOCATE_MAX_FRAMES` cited frames (live NOW frames first, then newest)
and returns the box of the asked-about object. The box goes out as a
`Marker` in `ChatResponse.markers`: normalized image coordinates (origin top-left) plus the frame's `device_id` and
`ts`. The headset keeps the camera pose of every frame it uploaded, turns the box centre into a world ray from that
pose and hit-tests it against the room (docs/api.md, headset ObjectMarkers.cs).

Runs concurrently with TTS in the chat router, and never fails a turn: no key, a timeout or a bad reply means no
markers. Opt out with ``LOCATE_ENABLED=false``; other knobs are ``LOCATE_*``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import time

import anthropic
from PIL import Image
from pydantic_settings import BaseSettings, SettingsConfigDict

from megamind.models import Citation, Marker

log = logging.getLogger("megamind.locate")


class LocateSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOCATE_", env_file=".env", extra="ignore")

    enabled: bool = True  # needs the provider's key as well
    provider: str = "openai"  # openai (OPENAI_API_KEY) | anthropic (ANTHROPIC_API_KEY)
    model: str = ""  # "" -> gpt-4.1 (openai) / CLAUDE_MODEL (anthropic)
    max_frames: int = 2  # cited frames shown to the locator (live NOW frames first, then newest)
    max_age_s: float = 1800.0  # only frames this recent (older ones are from another headset session anyway)
    max_px: int = 1024  # longest side sent (we must know the exact size the model sees)
    timeout_s: float = 8.0
    min_confidence: float = 0.5


_WHERE_RE = re.compile(
    r"\b(?:where(?:'s|'re|'d)?|wheres|find|locate|spot|point (?:to|at|out)|show me|highlight|mark|which way)\b",
    re.I,
)


def is_where_question(text: str | None) -> bool:
    """Rule-based: does the wearer ask where something is (so a marker in the room would help)?"""
    return bool(text) and _WHERE_RE.search(text) is not None


LOCATE_TOOL = {
    "name": "report_location",
    "description": "Report where the object the wearer asked about is in the snapshots.",
    "input_schema": {
        "type": "object",
        "properties": {
            "found": {"type": "boolean", "description": "true iff the object is clearly visible in one snapshot."},
            "image": {"type": "integer", "description": "Index of the snapshot it is in (0-based)."},
            "label": {"type": "string", "description": "One to three words naming the object, e.g. 'blue pen'."},
            "box": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Tight box [x_min, y_min, x_max, y_max] in pixels of that snapshot, origin top-left.",
            },
            "confidence": {"type": "number", "description": "0..1"},
        },
        "required": ["found"],
    },
}

OPENAI_LOCATE_TOOL = {
    "type": "function",
    "function": {
        "name": LOCATE_TOOL["name"],
        "description": LOCATE_TOOL["description"],
        "parameters": LOCATE_TOOL["input_schema"],
    },
}

_PROMPT = """The wearer of an AR headset asked: "{question}"
The assistant answered: "{answer}"

The snapshots above are first-person camera frames from the headset. Find the object the wearer is asking
about. If it is clearly visible, report the snapshot index and a tight bounding box in that snapshot's pixel
coordinates (sizes are given per snapshot; origin top-left). If it is not visible, report found=false.
Do not mark people."""


def _fit(jpeg: bytes, max_px: int) -> tuple[bytes, int, int]:
    """(JPEG, width, height) with the longest side <= max_px."""
    with Image.open(io.BytesIO(jpeg)) as im:
        if max(im.size) <= max_px and im.format == "JPEG":
            return jpeg, im.size[0], im.size[1]
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), im.size[0], im.size[1]


class Locator:
    """Finds the asked-about object in the turn's cited frames. `load(citation)` -> (jpeg, device_id) or None.

    `client` is an `anthropic.AsyncAnthropic` or `openai.AsyncOpenAI` matching `cfg.provider` (tests inject one)."""

    def __init__(self, settings, cfg: LocateSettings, load, client=None) -> None:
        self.cfg = cfg
        self.provider = cfg.provider.lower()
        self.model = cfg.model or (settings.claude_model if self.provider == "anthropic" else "gpt-4.1")
        self.load = load
        self._client = client
        if self._client is None and cfg.enabled:
            if self.provider == "anthropic" and settings.anthropic_api_key:
                self._client = anthropic.AsyncAnthropic(
                    api_key=settings.anthropic_api_key, max_retries=0, timeout=cfg.timeout_s
                )
            elif self.provider == "openai" and settings.openai_api_key:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0, timeout=cfg.timeout_s)

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and self._client is not None

    def candidates(self, cited: list[Citation], now: float) -> list[Citation]:
        frames = [c for c in cited if c.thumb_url and (self.cfg.max_age_s <= 0 or now - c.ts <= self.cfg.max_age_s)]
        frames.sort(key=lambda c: (not c.id.startswith("now_"), -c.ts))
        return frames[: max(0, self.cfg.max_frames)]

    async def markers(
        self, question: str, answer: str, cited: list[Citation], now: float | None = None
    ) -> list[Marker]:
        if not self.enabled or not is_where_question(question):
            return []
        cands = self.candidates(cited, time.time() if now is None else now)
        if not cands:
            return []
        try:
            return await asyncio.wait_for(self._locate(question, answer, cands), self.cfg.timeout_s)
        except Exception as e:  # noqa: BLE001  (markers are best-effort; never break the turn)
            log.warning("locate failed (%s: %s)", type(e).__name__, e)
            return []

    async def _locate(self, question: str, answer: str, cands: list[Citation]) -> list[Marker]:
        shown: list[tuple[Citation, str | None, int, int]] = []
        images: list[tuple[str, str]] = []  # (header, base64 jpeg)
        for c in cands:
            got = await self.load(c)
            if got is None:
                continue
            jpeg, device_id = got
            data, w, h = await asyncio.to_thread(_fit, jpeg, self.cfg.max_px)
            images.append((f"snapshot {len(shown)} ({w}x{h} px):", base64.b64encode(data).decode()))
            shown.append((c, device_id, w, h))
        if not shown:
            return []
        prompt = _PROMPT.format(question=question, answer=answer)
        t0 = time.perf_counter()
        call = self._ask_anthropic if self.provider == "anthropic" else self._ask_openai
        out = await call(images, prompt)
        log.info("locate[%s]: %r -> %s (%.0f ms)", self.provider, question, out, (time.perf_counter() - t0) * 1000)
        m = _to_marker(out, shown, self.cfg.min_confidence)
        return [m] if m is not None else []

    async def _ask_anthropic(self, images: list[tuple[str, str]], prompt: str) -> dict | None:
        content: list[dict] = []
        for header, b64 in images:
            content.append({"type": "text", "text": header})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
        content.append({"type": "text", "text": prompt})
        msg = await self._client.messages.create(
            model=self.model,
            max_tokens=200,
            tools=[LOCATE_TOOL],
            tool_choice={"type": "tool", "name": LOCATE_TOOL["name"]},
            messages=[{"role": "user", "content": content}],
        )
        return next((b.input for b in msg.content if b.type == "tool_use"), None)

    async def _ask_openai(self, images: list[tuple[str, str]], prompt: str) -> dict | None:
        content: list[dict] = []
        for header, b64 in images:
            content.append({"type": "text", "text": header})
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}
            )
        content.append({"type": "text", "text": prompt})
        resp = await self._client.chat.completions.create(
            model=self.model,
            tools=[OPENAI_LOCATE_TOOL],
            tool_choice={"type": "function", "function": {"name": LOCATE_TOOL["name"]}},
            messages=[{"role": "user", "content": content}],
        )
        calls = resp.choices[0].message.tool_calls if resp.choices else None
        return json.loads(calls[0].function.arguments) if calls else None


def _to_marker(out, shown, min_confidence: float) -> Marker | None:
    if not isinstance(out, dict) or not out.get("found"):
        return None
    try:
        idx = int(out.get("image", 0))
        x0, y0, x1, y1 = (float(v) for v in out["box"])
        conf = float(out.get("confidence", 1.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= idx < len(shown) or conf < min_confidence:
        return None
    c, device_id, w, h = shown[idx]
    x0, x1 = sorted((min(max(x0 / w, 0.0), 1.0), min(max(x1 / w, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0 / h, 0.0), 1.0), min(max(y1 / h, 0.0), 1.0)))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    label = str(out.get("label") or "").strip()[:40] or "here"
    return Marker(
        label=label,
        token_id=c.id,
        device_id=device_id,
        ts=c.ts,
        x=round((x0 + x1) / 2, 4),
        y=round((y0 + y1) / 2, 4),
        box=[round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
        confidence=round(min(max(conf, 0.0), 1.0), 3),
    )


def build_locator(state, settings, cfg: LocateSettings | None = None, client=None) -> Locator:
    """Lifespan hook. Frames: live NOW frames from the live cache, stored frames via media (as the reader sees them)."""
    cfg = cfg or LocateSettings()

    async def load(c: Citation) -> tuple[bytes, str | None] | None:
        live = getattr(state, "live_memory", None)
        if c.id.startswith("now_"):  # the cached (downscaled, face-blurred) frame the answering model saw
            src = live.source(c.id) if live is not None else None
            path = live.lookup(c.id) if src is not None else None
            return (await asyncio.to_thread(path.read_bytes), src[0]) if path is not None else None
        token = await state.store.get(c.id)
        if token is None or token.kind != "frame":
            return None
        data = await state.media.read(token)
        return (data, token.device_id) if data else None

    return Locator(settings, cfg, load, client=client)


__all__ = ["LOCATE_TOOL", "LocateSettings", "Locator", "build_locator", "is_where_question"]
