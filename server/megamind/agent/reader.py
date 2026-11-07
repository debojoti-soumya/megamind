"""Reader sub-agent: looks at the retrieved frames themselves (not just their stored text) and answers
the sub-question with citations. Retrieval only has to put the right frame in the top few; details
such as counts, positions or colours are read from the pixels at question time.

Providers (READER_PROVIDER):
- caption: no model call, returns the stored caption/OCR/transcript of the top tokens.
- vlm: any OpenAI-compatible chat.completions VLM (Qwen2.5-VL on vLLM, OpenRouter, OpenAI); all frames
  in one request, JSON {"answer", "confidence", "cited_ids"}.
- local: a small transformers VLM (SmolVLM) run per frame on CPU; for offline demos and benchmarks.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from megamind.config import Settings
from megamind.models import Confidence, MemoryToken

READ_PROMPT = """You are reading first-person memory snapshots to answer the user's question.
Each snapshot is preceded by a line "[id] time | stored text". Look carefully at the images:
count objects, note where things are relative to furniture/surfaces, read any visible text.
Question: {query}
Return ONLY JSON: {{"answer": str, "confidence": "high"|"medium"|"low"|"not_found", "cited_ids": [str]}}.
Cite only snapshots that actually show the answer. If none do, confidence "not_found"."""

LOCAL_PROMPT = "{header}\nQuestion: {query}\nAnswer in one short sentence using only this image."
_NO = re.compile(r"\b(not visible|no such|cannot (see|tell|determine)|there (is|are) no|i don'?t see)\b", re.I)
_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class ReaderOutput:
    evidence: str
    confidence: Confidence
    cited_ids: list[str] = field(default_factory=list)


class Reader(ABC):
    @abstractmethod
    async def read(self, query: str, tokens: list[MemoryToken]) -> ReaderOutput: ...


def _header(t: MemoryToken) -> str:
    when = datetime.fromtimestamp(t.ts).strftime("%Y-%m-%d %H:%M")
    text = " / ".join(
        p for p in (t.caption, t.ocr_text and f"text: {t.ocr_text}", t.transcript and f"heard: {t.transcript}") if p
    )
    return f"[{t.id}] {when} | {text[:400]}"


class CaptionOnlyReader(Reader):
    """Fallback with no model call: returns the top captions verbatim as evidence."""

    async def read(self, query: str, tokens: list[MemoryToken]) -> ReaderOutput:
        top = tokens[:3]
        lines = [f"[{t.id}] {t.caption}" + (f" (text: {t.ocr_text})" if t.ocr_text else "") for t in top]
        return ReaderOutput(evidence="\n".join(lines), confidence="low", cited_ids=[t.id for t in top])


class _FrameReader(Reader):
    """Base for readers that load each token's pixels from whatever tier they are on."""

    def __init__(self, settings: Settings, media) -> None:
        self.media = media
        self.max_frames = settings.reader_max_frames

    async def _frames(self, tokens: list[MemoryToken]) -> list[tuple[MemoryToken, bytes]]:
        out = []
        for t in tokens[: self.max_frames]:
            jpeg = await self.media.read(t)
            if jpeg:
                out.append((t, jpeg))
        return out


def parse_read(text: str, shown: list[str]) -> ReaderOutput:
    m = _JSON.search(text or "")
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    if not d:
        return ReaderOutput(evidence=(text or "").strip()[:800], confidence="low", cited_ids=shown[:1])
    conf = d.get("confidence") if d.get("confidence") in ("high", "medium", "low", "not_found") else "low"
    cited = [str(i) for i in d.get("cited_ids") or [] if str(i) in shown]
    return ReaderOutput(evidence=str(d.get("answer") or "").strip(), confidence=conf, cited_ids=cited)


class VLMReader(_FrameReader):
    def __init__(self, settings: Settings, media) -> None:
        from openai import AsyncOpenAI

        super().__init__(settings, media)
        self.model = settings.reader_model
        self.client = AsyncOpenAI(base_url=settings.reader_base_url, api_key=settings.reader_api_key or "none")

    async def read(self, query: str, tokens: list[MemoryToken]) -> ReaderOutput:
        frames = await self._frames(tokens)
        if not frames:
            return await CaptionOnlyReader().read(query, tokens)
        content: list[dict] = [{"type": "text", "text": READ_PROMPT.format(query=query)}]
        for t, jpeg in frames:
            uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            content += [{"type": "text", "text": _header(t)}, {"type": "image_url", "image_url": {"url": uri}}]
        r = await self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=400, messages=[{"role": "user", "content": content}]
        )
        return parse_read(r.choices[0].message.content or "", [t.id for t, _ in frames])


class LocalVLMReader(_FrameReader):
    """Asks a small local VLM the question about each frame separately (small models get confused by
    several images at once) and cites the frames whose answer is not a refusal."""

    def __init__(self, settings: Settings, media) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        super().__init__(settings, media)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(settings.local_reader_model)
        self.processor.image_processor.do_image_splitting = settings.local_reader_split
        self.model = AutoModelForImageTextToText.from_pretrained(
            settings.local_reader_model, dtype=torch.float32
        ).eval()
        self._lock = threading.Lock()

    def ask(self, jpeg: bytes, prompt: str, max_new_tokens: int = 48) -> str:
        from PIL import Image

        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.processor(text=text, images=[img], return_tensors="pt")
        with self._lock, self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()

    async def read(self, query: str, tokens: list[MemoryToken]) -> ReaderOutput:
        frames = await self._frames(tokens)
        if not frames:
            return await CaptionOnlyReader().read(query, tokens)
        lines, cited = [], []
        for t, jpeg in frames:
            ans = await asyncio.to_thread(self.ask, jpeg, LOCAL_PROMPT.format(header=_header(t), query=query))
            lines.append(f"[{t.id}] {ans}")
            if ans and not _NO.search(ans):
                cited.append(t.id)
        return ReaderOutput(evidence="\n".join(lines), confidence="medium" if cited else "not_found", cited_ids=cited)


def build_reader(settings: Settings, media=None) -> Reader:
    p = settings.reader_provider
    if p == "vlm" and media is not None:
        return VLMReader(settings, media)
    if p == "local" and media is not None:
        return LocalVLMReader(settings, media)
    return CaptionOnlyReader()
