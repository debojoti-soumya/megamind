"""Sliding-window summaries: every SUMMARY_STEP_S, a SUMMARY_WINDOW_S window of kept frames becomes one
searchable record (time words + distinct frame descriptions + everything said). Questions about what was
going on often match a stretch of speech/activity better than any single frame's text."""

from __future__ import annotations

import math
from datetime import datetime

from megamind.config import Settings
from megamind.ingest.enrich import TextEmbedder
from megamind.models import MemorySummary, MemoryToken
from megamind.store.base import MemoryStore

MAX_DESCRIPTIONS = 6


def time_words(ts: float) -> str:
    d = datetime.fromtimestamp(ts).astimezone()
    h = d.hour
    part = (
        "early morning"
        if h < 9
        else "morning"
        if h < 12
        else "noon lunch"
        if h < 14
        else "afternoon"
        if h < 18
        else "evening"
        if h < 22
        else "night"
    )
    return f"{d:%A %Y-%m-%d %H:%M} {part}"


def summary_text(start_ts: float, members: list[MemoryToken]) -> str:
    seen = list(dict.fromkeys(t.description or t.caption for t in members if t.description or t.caption))
    speech = " ".join(t.transcript for t in members if t.transcript)
    return "\n".join(p for p in (time_words(start_ts), " ".join(seen[:MAX_DESCRIPTIONS]), speech) if p)


class SummaryBuilder:
    def __init__(self, settings: Settings, store: MemoryStore, text_embedder: TextEmbedder) -> None:
        self.size = settings.summary_window_s
        self.step = settings.summary_step_s
        self.store = store
        self.text_embedder = text_embedder
        self._recent: dict[str, dict[str, MemoryToken]] = {}

    def window_starts(self, ts: float) -> list[float]:
        s0 = math.floor(ts / self.step) * self.step
        return [
            s0 - k * self.step for k in range(math.ceil(self.size / self.step)) if s0 - k * self.step + self.size > ts
        ]

    async def add(self, token: MemoryToken) -> list[MemorySummary]:
        """(Re)build the windows containing `token`; call again after its text changes."""
        if self.size <= 0:
            return []
        buf = self._recent.setdefault(token.device_id, {})
        buf[token.id] = token
        horizon = token.ts - 2 * self.size
        for tid in [i for i, t in buf.items() if t.ts < horizon]:
            del buf[tid]
        out = []
        for start in self.window_starts(token.ts):
            members = sorted((t for t in buf.values() if start <= t.ts < start + self.size), key=lambda t: t.ts)
            text = summary_text(start, members)
            s = MemorySummary(
                id=f"{token.device_id}:{int(start)}:{int(self.size)}",
                device_id=token.device_id,
                start_ts=start,
                end_ts=start + self.size,
                text=text,
                text_vec=await self.text_embedder.embed(text),
                token_ids=[t.id for t in members],
            )
            await self.store.upsert_summary(s)
            out.append(s)
        return out

    def reset(self, device_id: str | None = None) -> None:
        if device_id is None:
            self._recent.clear()
        else:
            self._recent.pop(device_id, None)
