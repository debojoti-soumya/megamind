"""Object markers: "where is my pen?" -> a bounding box in the cited frame, so the headset can pin it in the room.

After a chat turn whose question asks where something is (`is_where_question`) and whose answer cites a frame
with pixels, one vision call (forced `report_location` tool; ``LOCATE_PROVIDER=openai`` GPT by default,
``anthropic`` for Claude) looks at up to `LOCATE_MAX_FRAMES` cited frames (live NOW frames first, then newest)
and returns the box of the asked-about object. The box goes out as a
`Marker` in `ChatResponse.markers`: normalized image coordinates (origin top-left) plus the frame's `device_id` and
`ts`. The headset keeps the camera pose of every frame it uploaded, turns the box centre into a world ray from that
pose and hit-tests it against the room (docs/api.md, headset ObjectMarkers.cs).

Box refinement: vision LLM boxes are loose (13% median centre error for gpt-4.1 on COCO val vs 1.2% for OWLv2), so
when the optional ``locate-refine`` extra (torch + transformers) is installed, an open-vocabulary detector
(`OwlRefiner`, OWLv2) redraws the box in the frame the LLM chose, using the LLM's label as the text query and its
box as a hint (`pick_box`). The LLM still decides whether the object is there and in which frame.
``LOCATE_REFINE=none`` turns it off; a refiner error or timeout keeps the LLM's box.

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
import threading
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
    model: str = ""  # "" -> gpt-5.6-terra (openai) / CLAUDE_MODEL (anthropic)
    max_frames: int = 2  # cited frames shown to the locator (live NOW frames first, then newest)
    max_age_s: float = 1800.0  # only frames this recent (older ones are from another headset session anyway)
    max_px: int = 1024  # longest side sent (we must know the exact size the model sees)
    timeout_s: float = 8.0
    min_confidence: float = 0.5
    reasoning_effort: str = "none"  # gpt-5*/o* only; chat-completions function tools need "none" on gpt-5.4+
    refine: str = "auto"  # auto (OWLv2 when torch + transformers AND a GPU are there) | owlv2 (even on CPU) | none
    refine_model: str = "google/owlv2-base-patch16-ensemble"
    refine_timeout_s: float = 8.0  # on top of timeout_s; ~0.1 s on a GPU, ~9 s on 8 CPU cores (so auto needs a GPU)
    refine_min_score: float = 0.08  # OWLv2 detections below this are ignored
    refine_verify: bool = True  # no detection overlapping the LLM's box -> no marker (the LLM was wrong)


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


Box = tuple[float, float, float, float]


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def pick_box(cands: list[tuple[Box, float]], hint: Box, min_score: float) -> Box | None:
    """Detector candidate for the object the LLM pointed at: score boosted by overlap with the LLM's (loose) box."""
    best = max(
        ((score * (0.5 + _iou(box, hint)), box) for box, score in cands if score >= min_score),
        default=None,
    )
    return best[1] if best else None


class OwlRefiner:
    """OWLv2 open-vocabulary detector (transformers). `detect(jpeg, query)` -> [(box px, score)], blocking."""

    def __init__(self, model_name: str, *, threads: int | None = None) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if threads:
            torch.set_num_threads(threads)
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.model_name = model_name
        self._lock = threading.Lock()

    def detect(self, jpeg: bytes, query: str, min_score: float = 0.0) -> list[tuple[Box, float]]:
        with Image.open(io.BytesIO(jpeg)) as im:
            im = im.convert("RGB")
            w, h = im.size
            with self._lock, self.torch.no_grad():
                inp = self.processor(text=[[f"a photo of a {query}"]], images=im, return_tensors="pt").to(self.device)
                out = self.model(**inp)
                p = self.processor
                post = (
                    p.post_process_grounded_object_detection  # transformers >= 5
                    if hasattr(p, "post_process_grounded_object_detection")
                    else p.post_process_object_detection
                )
                side = max(w, h)  # OWLv2 pads to a square, so boxes come back in padded-square coordinates
                r = post(out, threshold=min_score, target_sizes=self.torch.tensor([[side, side]]))[0]
        order = r["scores"].argsort(descending=True)
        return [
            (
                (
                    max(0.0, float(r["boxes"][i][0])),
                    max(0.0, float(r["boxes"][i][1])),
                    min(float(w), float(r["boxes"][i][2])),
                    min(float(h), float(r["boxes"][i][3])),
                ),
                float(r["scores"][i]),
            )
            for i in order.tolist()
        ]


def build_refiner(cfg: LocateSettings):
    """OwlRefiner per `LOCATE_REFINE`, or None. `auto` also needs a GPU: OWLv2-base takes ~9 s per frame on 8 CPU
    cores, past `refine_timeout_s`, so on a CPU box it would only add the timeout to every where-turn."""
    mode = cfg.refine.lower()
    if mode in ("none", "off", "false", "") or not cfg.enabled:
        return None
    try:
        if mode == "auto":
            import torch

            if not torch.cuda.is_available():
                log.info("LOCATE_REFINE=auto: no GPU, using the LLM's boxes (LOCATE_REFINE=owlv2 forces the detector)")
                return None
        return OwlRefiner(cfg.refine_model)
    except ImportError as e:
        if mode != "auto":
            log.warning("LOCATE_REFINE=%s but torch/transformers missing (%s); using the LLM's boxes", mode, e)
        return None
    except Exception as e:  # noqa: BLE001  (e.g. the model download failed: fall back to the LLM's boxes)
        log.warning("box refiner unavailable (%s: %s); using the LLM's boxes", type(e).__name__, e)
        return None


class Locator:
    """Finds the asked-about object in the turn's cited frames. `load(citation)` -> (jpeg, device_id) or None.

    `client` is an `anthropic.AsyncAnthropic` or `openai.AsyncOpenAI` matching `cfg.provider` (tests inject one).
    `refiner` has `detect(jpeg, query, min_score) -> [(box px, score)]` (`OwlRefiner`), or None (LLM boxes)."""

    def __init__(self, settings, cfg: LocateSettings, load, client=None, refiner=None) -> None:
        self.cfg = cfg
        self.refiner = refiner
        self.provider = cfg.provider.lower()
        self.model = cfg.model or (settings.claude_model if self.provider == "anthropic" else "gpt-5.6-terra")
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
            m, shown, jpegs = await asyncio.wait_for(self._locate(question, answer, cands), self.cfg.timeout_s)
        except Exception as e:  # noqa: BLE001  (markers are best-effort; never break the turn)
            log.warning("locate failed (%s: %s)", type(e).__name__, e)
            return []
        if m is not None and self.refiner is not None:  # own timeout (refine_timeout_s); failure keeps the LLM box
            m = await self._refine(m, shown, jpegs)
        return [m] if m is not None else []

    async def _locate(self, question: str, answer: str, cands: list[Citation]):
        """(LLM marker or None, shown frames, their JPEGs)."""
        shown: list[tuple[Citation, str | None, int, int]] = []
        jpegs: list[bytes] = []
        images: list[tuple[str, str]] = []  # (header, base64 jpeg)
        for c in cands:
            got = await self.load(c)
            if got is None:
                continue
            jpeg, device_id = got
            data, w, h = await asyncio.to_thread(_fit, jpeg, self.cfg.max_px)
            images.append((f"snapshot {len(shown)} ({w}x{h} px):", base64.b64encode(data).decode()))
            jpegs.append(data)
            shown.append((c, device_id, w, h))
        if not shown:
            return None, shown, jpegs
        prompt = _PROMPT.format(question=question, answer=answer)
        t0 = time.perf_counter()
        call = self._ask_anthropic if self.provider == "anthropic" else self._ask_openai
        out = await call(images, prompt)
        log.info("locate[%s]: %r -> %s (%.0f ms)", self.provider, question, out, (time.perf_counter() - t0) * 1000)
        return _to_marker(out, shown, self.cfg.min_confidence), shown, jpegs

    async def _refine(self, m: Marker, shown, jpegs: list[bytes]) -> Marker | None:
        """Redraw `m.box` with the detector in the same frame; a detector failure keeps the LLM's box. With
        `refine_verify`, no detection overlapping the LLM's box means the LLM was wrong (None: no marker)."""
        idx = next(i for i, s in enumerate(shown) if s[0].id == m.token_id)
        _, _, w, h = shown[idx]
        hint = (m.box[0] * w, m.box[1] * h, m.box[2] * w, m.box[3] * h)
        t0 = time.perf_counter()
        try:
            cands = await asyncio.wait_for(
                asyncio.to_thread(self.refiner.detect, jpegs[idx], m.label, self.cfg.refine_min_score),
                self.cfg.refine_timeout_s,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("box refine failed (%s: %s); keeping the LLM's box", type(e).__name__, e)
            return m
        if self.cfg.refine_verify:
            cands = [c for c in cands if _iou(c[0], hint) > 0]
        box = pick_box(cands, hint, self.cfg.refine_min_score)
        log.info("refine %r: %s -> %s (%.0f ms)", m.label, m.box, box, (time.perf_counter() - t0) * 1000)
        if box is None:
            return None if self.cfg.refine_verify else m
        x0, y0, x1, y1 = (min(max(v / d, 0.0), 1.0) for v, d in zip(box, (w, h, w, h)))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return m
        return m.model_copy(
            update={
                "x": round((x0 + x1) / 2, 4),
                "y": round((y0 + y1) / 2, 4),
                "box": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
            }
        )

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
        reasoning = self.cfg.reasoning_effort and self.model.startswith(("gpt-5", "o1", "o3", "o4"))
        extra = {"reasoning_effort": self.cfg.reasoning_effort} if reasoning else {}
        resp = await self._client.chat.completions.create(
            model=self.model,
            **extra,
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
        idx = int(out["image"] if len(shown) > 1 else out.get("image", 0))
        x0, y0, x1, y1 = (float(v) for v in out["box"])
        conf = float(out.get("confidence", 0.0))
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


def build_locator(state, settings, cfg: LocateSettings | None = None, client=None, refiner=None) -> Locator:
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

    loc = Locator(settings, cfg, load, client=client, refiner=refiner)
    if refiner is None and client is None and loc.enabled:  # load the detector off the startup path
        threading.Thread(target=lambda: setattr(loc, "refiner", build_refiner(cfg)), daemon=True).start()
    return loc


__all__ = [
    "LOCATE_TOOL",
    "LocateSettings",
    "Locator",
    "OwlRefiner",
    "build_locator",
    "build_refiner",
    "is_where_question",
    "pick_box",
]
