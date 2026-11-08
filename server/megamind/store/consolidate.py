"""Consolidation: one `kind="episode"` summary token per closed episode.

`Consolidator.run_once()` finds episodes whose last frame is older than `gap_s`, have at least
`min_tokens` frames and no summary yet, asks the `Summarizer` for a one-paragraph caption, and writes
a `MemoryToken(kind="episode", id=f"{episode_id}-summary", ts=<episode start>)` carrying the union of
OCR text, top entities, a text embedding of the summary and the mean image vector, with the
representative frame's thumbnail copied to `<thumb_dir>/<id>.jpg`. Summaries go through the same
`vector_search`/`keyword_search` as frames, so vague queries ("what did I do this morning") hit them.

Wire-up (main.py lifespan, coordinator): `Consolidator(store, build_summarizer(settings), text_embedder,
gap_s=settings.episode_gap_s, min_tokens=settings.consolidate_min_tokens).run_forever(settings.consolidate_interval_s)`
as a background task. Manual: `python -m megamind.store.consolidate [--device D] [--loop]`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import httpx

from megamind.config import Settings, get_settings
from megamind.models import MemoryToken
from megamind.store.base import MemoryStore

log = logging.getLogger(__name__)

SUMMARY_SUFFIX = "-summary"
LOOKBACK_S = 24 * 3600.0
MAX_CAPTIONS = 40
MAX_OCR_CHARS = 1000


def summary_id(episode_id: str) -> str:
    return f"{episode_id}{SUMMARY_SUFFIX}"


def _hhmm(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def _top_entities(frames: list[MemoryToken], n: int = 10) -> list[str]:
    c: Counter[str] = Counter()
    for t in frames:
        c.update(e.strip().lower() for e in t.entities if e.strip())
    return [e for e, _ in c.most_common(n)]


def _ocr_union(frames: list[MemoryToken], max_chars: int = MAX_OCR_CHARS) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for t in frames:
        for line in t.ocr_text.splitlines():
            line = line.strip()
            if line and line.lower() not in seen:
                seen.add(line.lower())
                lines.append(line)
    out = "\n".join(lines)
    return out[:max_chars]


def _mean_vec(vecs: list[list[float]]) -> list[float] | None:
    vecs = [v for v in vecs if v]
    if not vecs:
        return None
    dim = len(vecs[0])
    vecs = [v for v in vecs if len(v) == dim]
    acc = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            acc[i] += x
    return [x / len(vecs) for x in acc]


def _dedup_captions(frames: list[MemoryToken], limit: int = MAX_CAPTIONS) -> list[str]:
    out: list[str] = []
    for t in frames:
        c = t.caption.strip()
        if c and (not out or out[-1].lower() != c.lower()):
            out.append(c)
    if len(out) <= limit:
        return out
    step = len(out) / limit
    return [out[int(i * step)] for i in range(limit)]


# ---------------------------------------------------------------------- summarizers


class Summarizer(Protocol):
    async def summarize(self, frames: list[MemoryToken]) -> str: ...


class MockSummarizer:
    """Deterministic, offline: time span + top entities + first/last captions."""

    async def summarize(self, frames: list[MemoryToken]) -> str:
        caps = _dedup_captions(frames, limit=4)
        ents = _top_entities(frames, 5)
        span = f"{_hhmm(frames[0].ts)}-{_hhmm(frames[-1].ts)}"
        parts = [f"Episode {span}, {len(frames)} moments."]
        if ents:
            parts.append("Involving " + ", ".join(ents) + ".")
        if caps:
            parts.append(" Then ".join(caps) + ".")
        return " ".join(parts)


def _prompt(frames: list[MemoryToken]) -> str:
    lines = [f"[{_hhmm(t.ts)}] {c}" for t, c in zip(frames, _dedup_captions(frames))]
    ocr = _ocr_union(frames, 400)
    body = "\n".join(lines)
    extra = f"\nText seen on screens/signs:\n{ocr}" if ocr else ""
    return (
        "These are chronological captions of what a person wearing a camera saw during one episode. "
        "Write a 2-3 sentence first-person-neutral summary of what happened and where (e.g. 'Worked at a desk "
        "on a laptop, then made coffee in the kitchen'). Mention salient objects, people, places and any readable "
        f"text. No preamble.\n\n{body}{extra}"
    )


@dataclass
class OpenAISummarizer:
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 30.0

    async def summarize(self, frames: list[MemoryToken]) -> str:
        async with httpx.AsyncClient(timeout=self.timeout_s) as c:
            r = await c.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": _prompt(frames)}],
                    "max_tokens": 200,
                    "temperature": 0.2,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()


@dataclass
class AnthropicSummarizer:
    api_key: str
    model: str = "claude-haiku-4-5"
    timeout_s: float = 30.0

    async def summarize(self, frames: list[MemoryToken]) -> str:
        async with httpx.AsyncClient(timeout=self.timeout_s) as c:
            r = await c.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                json={
                    "model": self.model,
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": _prompt(frames)}],
                },
            )
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"]).strip()


def build_summarizer(settings: Settings) -> Summarizer:
    provider = settings.summarizer_provider
    if provider == "openai" and settings.openai_api_key:
        return OpenAISummarizer(settings.openai_api_key)
    if provider == "anthropic" and settings.anthropic_api_key:
        return AnthropicSummarizer(settings.anthropic_api_key)
    if provider != "mock":
        log.warning("summarizer_provider=%s but no API key; falling back to MockSummarizer", provider)
    return MockSummarizer()


# ---------------------------------------------------------------------- job


class TextEmbedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class Consolidator:
    def __init__(
        self,
        store: MemoryStore,
        summarizer: Summarizer,
        text_embedder: TextEmbedder | None = None,
        *,
        gap_s: float = 120.0,
        min_tokens: int = 3,
        lookback_s: float = LOOKBACK_S,
        image_loader: Callable[[MemoryToken], Awaitable[bytes | None]] | None = None,
        thumb_dir: Path | None = None,
    ) -> None:
        self.thumb_dir = Path(thumb_dir) if thumb_dir else None  # where the summary's JPEG copy goes
        self.store = store
        self.image_loader = image_loader  # MediaStore.read: the representative may already be a cold frame
        self.summarizer = summarizer
        self.text_embedder = text_embedder
        self.gap_s = gap_s
        self.min_tokens = max(1, min_tokens)
        self.lookback_s = lookback_s

    async def pending(self, now: float, device_id: str | None = None) -> list[list[MemoryToken]]:
        """Closed, un-summarised episodes as chronological frame lists."""
        since = now - self.lookback_s
        # Plan on vector-less rows (this runs every interval over the whole lookback), then load full rows
        # only for the episodes that will actually be summarised.
        scan = dict(time_from=since, time_to=now, device_id=device_id, vectors=False)
        frames = await self.store.list_tokens(kinds=("frame",), **scan)
        done = {t.episode_id for t in await self.store.list_tokens(kinds=("episode",), **scan)}
        groups: dict[str, list[MemoryToken]] = {}
        for t in frames:
            if t.episode_id:
                groups.setdefault(t.episode_id, []).append(t)
        ready = [
            g
            for ep, g in groups.items()
            if ep not in done and len(g) >= self.min_tokens and g[-1].ts <= now - self.gap_s
        ]
        if not ready:
            return []
        # one time-window scan, not `get_many`: an `id IN (...)` over thousands of ids is the slow path
        lo, hi = min(g[0].ts for g in ready), max(g[-1].ts for g in ready)
        rows = await self.store.list_tokens(time_from=lo, time_to=hi, device_id=device_id, kinds=("frame",))
        full = {t.id: t for t in rows}
        return [[full[t.id] for t in g if t.id in full] for g in ready]

    async def summarize_episode(self, frames: list[MemoryToken]) -> MemoryToken:
        frames = sorted(frames, key=lambda t: (t.ts, t.id))
        episode_id = frames[0].episode_id
        assert episode_id
        caption = await self.summarizer.summarize(frames)
        ocr = _ocr_union(frames)
        entities = _top_entities(frames)
        rep = max(frames, key=lambda t: (len(t.ocr_text), len(t.caption)))
        tid = summary_id(episode_id)
        src = Path(rep.thumb_path) if rep.thumb_path else None
        thumb_dir = self.thumb_dir or (src.parent if src is not None else None)
        thumb_path = thumb_dir / f"{tid}.jpg" if thumb_dir is not None else None
        if thumb_path is not None:
            if src is not None and src.is_file():
                shutil.copyfile(src, thumb_path)
            elif self.image_loader is not None and (data := await self.image_loader(rep)):
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                thumb_path.write_bytes(data)
        text_vec = None
        if self.text_embedder is not None:
            text_vec = await self.text_embedder.embed(f"{caption}\n{ocr}\n{' '.join(entities)}")
        return MemoryToken(
            id=tid,
            device_id=frames[0].device_id,
            ts=frames[0].ts,
            caption=caption,
            ocr_text=ocr,
            entities=entities,
            pose=rep.pose,
            text_vec=text_vec,
            image_vec=_mean_vec([t.image_vec for t in frames if t.image_vec]),
            thumb_path=str(thumb_path) if thumb_path is not None else "",
            frame_hash=rep.frame_hash,
            episode_id=episode_id,
            kind="episode",
        )

    async def run_once(self, now: float | None = None, device_id: str | None = None) -> list[MemoryToken]:
        now = time.time() if now is None else now
        written: list[MemoryToken] = []
        for frames in await self.pending(now, device_id):
            try:
                tok = await self.summarize_episode(frames)
            except Exception:  # noqa: BLE001
                log.exception("consolidation failed for episode %s", frames[0].episode_id)
                continue
            await self.store.add(tok)
            written.append(tok)
        if written:
            log.info("consolidated %d episode(s)", len(written))
        return written

    async def run_forever(self, interval_s: float) -> None:
        while True:
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001
                log.exception("consolidation pass failed")
            await asyncio.sleep(interval_s)


async def _main() -> None:
    from megamind.ingest.enrich import build_text_embedder
    from megamind.store import build_store

    ap = argparse.ArgumentParser(description="Write episode summary tokens for closed episodes.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--loop", action="store_true", help="run every CONSOLIDATE_INTERVAL_S seconds")
    args = ap.parse_args()
    settings = get_settings()
    store = build_store(settings)
    from megamind.store.media import MediaStore  # representatives may already be cold (AV1) frames

    job = Consolidator(
        store,
        build_summarizer(settings),
        build_text_embedder(settings),
        gap_s=settings.episode_gap_s,
        min_tokens=settings.consolidate_min_tokens,
        image_loader=MediaStore(settings, store).read,
        thumb_dir=settings.thumb_dir,
    )
    if args.loop:
        await job.run_forever(settings.consolidate_interval_s)
    else:
        for t in await job.run_once(device_id=args.device):
            print(t.id, _hhmm(t.ts), t.caption)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())
