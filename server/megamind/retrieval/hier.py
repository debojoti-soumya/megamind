"""Hierarchical LLM recall: an LLM-written memory log (day -> hour -> 10-minute block notes, built from the
stored frame captions, OCR and speech) searched coarse-to-fine on top of the hybrid retriever.

`HierMemory` writes one note per closed block (and per hour / day from those notes) and caches them as JSON.
`HierRetriever.retrieve()`:
  1. runs the base (hybrid) retriever for a wide candidate pool;
  2. stage 1: the LLM reads the day/hour/block log (plus which blocks the hybrid hits fall in) and picks the
     blocks most likely to hold the moment the question is about;
  3. stage 2: the LLM reads the frame/speech lines of those blocks (and of the windows around the hybrid hits),
     each prefixed with its token id, and ranks the token ids that show the answer.
Only ids that were shown are accepted; the hybrid ranking fills the rest. Any LLM failure returns the hybrid
result unchanged, so the backend degrades to `hybrid` instead of failing a recall.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import math
import re
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from megamind.models import MemoryToken
from megamind.retrieval.base import RetrievalResult, RetrievedToken, RetrieveMode, Retriever
from megamind.store.base import MemoryStore

log = logging.getLogger(__name__)

KINDS = ("frame", "transcript")
_SPEECH_RE = re.compile(r'^[^"]*"(.*)"\s*$', re.S)


@dataclass
class HierConfig:
    block_s: float = 600.0  # one LLM note per block
    window_s: float = 120.0  # stage-2 granularity: lines are grouped (and subsampled) per window
    max_blocks: int = 6  # stage-1 picks
    base_pool: int = 30  # hybrid candidates
    hint_windows: int = 10  # windows around the top hybrid hits always shown in stage 2
    lines_per_window: int = 8  # frame lines kept per window (speech lines are always kept)
    caption_chars: int = 200
    speech_chars: int = 300
    note_input_chars: int = 9000  # cap on a block's raw lines sent to the note writer
    min_block_tokens: int = 1
    moment_s: float = 60.0  # picks this close to a better one rank after the distinct moments
    vec_pool: int = 300  # nearest tokens aggregated per block for the vector block ranking
    vec_blocks: int = 4  # top vector-ranked blocks always added to the LLM's picks
    timeout_s: float = 8.0  # wall bound on the LLM stages of one recall (notes + pick + rank); then -> hybrid
    inline_build_max: int = 3  # a query writes at most this many missing block notes itself; more -> background
    # cheap gating: skip the LLM when the hybrid's top hit leads the runner-up by >= this relative margin
    # ((s1 - s2) / s1 of the non-context scores); 0 = always ask the LLM
    gate_margin: float = 0.0
    gate_min_hits: int = 5  # gating also needs this many non-context hybrid hits (a 3-hit result inflates the margin)
    # >0: stage 1 lists block notes only inside this many candidate hours (around the hybrid / vector hits); the other
    # hours show their hour note as [H..], which the pick may name to read that hour's blocks. 0 = every block
    # (the prompt then grows ~15k tokens per day)
    outline_hours: int = 0
    # direct mode: stage 1 also sees the lines around the top hybrid hits and may answer with token ids right away
    # (one LLM call); stage 2 runs only when it asks for blocks instead
    direct: bool = False


class ChatJSON:
    """Minimal OpenAI-compatible `chat/completions` caller that returns the parsed JSON object."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 120.0,
        concurrency: int = 8,
        retries: int = 6,
        transport: httpx.AsyncBaseTransport | None = None,
        reasoning_effort: str = "low",
    ) -> None:
        self.model = model
        self.retries = retries
        self.reasoning_effort = reasoning_effort
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    def _body(self, system: str, user: str, max_tokens: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_tokens,
        }
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            body["reasoning_effort"] = self.reasoning_effort
            body["max_completion_tokens"] = max_tokens + 4000
        else:
            body["temperature"] = 0
        return body

    async def __call__(self, system: str, user: str, *, max_tokens: int = 800) -> dict[str, Any]:
        body = self._body(system, user, max_tokens)
        last: Exception | None = None
        async with self._sem:
            for attempt in range(self.retries):
                try:
                    r = await self._client.post("/chat/completions", json=body)
                    if r.status_code == 429 or r.status_code >= 500:
                        last = httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                        await asyncio.sleep(_retry_after(r, attempt))
                        continue
                    r.raise_for_status()
                    out = json.loads(r.json()["choices"][0]["message"]["content"] or "{}")
                    if not isinstance(out, dict):
                        raise ValueError("LLM answer is not a JSON object")
                    return out
                except (httpx.TimeoutException, httpx.TransportError, json.JSONDecodeError) as e:
                    last = e
                    await asyncio.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"LLM call failed: {last!r}")

    async def aclose(self) -> None:
        await self._client.aclose()


def _retry_after(r: httpx.Response, attempt: int) -> float:
    """Server-suggested wait (Retry-After / x-ratelimit-reset-*), else exponential backoff; capped at 30 s."""
    for h in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        v = r.headers.get(h, "")
        m = re.fullmatch(r"(?:(\d+)m)?(\d+(?:\.\d+)?)(ms|s)?", v.strip())
        if m:
            secs = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1) + 60 * int(m.group(1) or 0)
            return min(30.0, secs + 0.25)
    return min(30.0, 1.5 * 2**attempt)


def speech_text(t: MemoryToken) -> str:
    m = _SPEECH_RE.match(t.caption)
    return m.group(1) if m else t.caption


def token_line(t: MemoryToken, tz: ZoneInfo, cfg: HierConfig) -> str:
    hhmmss = datetime.fromtimestamp(t.ts, tz).strftime("%H:%M:%S")
    if t.kind == "transcript":
        return f"[{t.id}] {hhmmss} heard: {speech_text(t)[: cfg.speech_chars]}"
    ocr = " ".join(t.ocr_text.split())[:80]
    return f"[{t.id}] {hhmmss} seen: {t.caption[: cfg.caption_chars]}" + (f" (text: {ocr})" if ocr else "")


def window_lines(tokens: Sequence[MemoryToken], tz: ZoneInfo, cfg: HierConfig) -> list[str]:
    """Chronological lines for a stretch of tokens: every speech line, frames subsampled evenly per window."""
    by_win: dict[int, list[MemoryToken]] = {}
    for t in tokens:
        by_win.setdefault(math.floor(t.ts / cfg.window_s), []).append(t)
    out: list[MemoryToken] = []
    for w in sorted(by_win):
        toks = by_win[w]
        frames = [t for t in toks if t.kind != "transcript"]
        n = cfg.lines_per_window
        if len(frames) > n:
            step = len(frames) / n
            frames = [frames[int(i * step)] for i in range(n)]
        out.extend(sorted([t for t in toks if t.kind == "transcript"] + frames, key=lambda t: t.ts))
    return [token_line(t, tz, cfg) for t in out]


NOTE_SYSTEM = (
    "You write a factual memory log for the wearer of first-person smart glasses ('I'). Use only the given "
    "lines (what the camera saw, what was heard). Never invent names, objects or events."
)
NOTE_PROMPT = """Lines from {start} to {end} ("seen" = camera caption, "heard" = speech near me):
{lines}

Write the log entry for this stretch. Return ONLY JSON:
{{"summary": str (60-140 words: where I am, who is there (names only if spoken), what each person does, objects \
handled and where they are put, plans/decisions/suggestions said, anything bought/ordered/counted),
"keywords": [str] (up to 20: names, objects, places, activities, notable words heard)}}"""

ROLLUP_PROMPT = """Log entries from {start} to {end}:
{lines}

Summarise this {span}. Return ONLY JSON: {{"summary": str (80-200 words, chronological, keep names, \
activities, objects and decisions), "keywords": [str] (up to 25)}}"""


class HierMemory:
    """Block / hour / day notes over a store, cached in `cache_path` (JSON, keyed by level and start ts)."""

    def __init__(
        self,
        store: MemoryStore,
        llm: ChatJSON,
        *,
        cache_path: Path | str | None = None,
        tz: str | ZoneInfo | None = None,
        config: HierConfig | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.cfg = config or HierConfig()
        self.tz = tz if isinstance(tz, ZoneInfo) else ZoneInfo(tz or "UTC")
        self.cache_path = Path(cache_path) if cache_path else None
        self.notes: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.is_file():
            self.notes = json.loads(self.cache_path.read_text())
        self._lock = asyncio.Lock()
        self._scanned: dict[str | None, float] = {}  # device -> latest `until` already scanned for blocks

    # ------------------------------------------------------------------ build

    def _fmt(self, ts: float, fmt: str = "%a %Y-%m-%d %H:%M") -> str:
        return datetime.fromtimestamp(ts, self.tz).strftime(fmt)

    def _save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.notes))
            tmp.replace(self.cache_path)

    async def _missing(
        self, until: float, device_id: str | None, *, full: bool = False
    ) -> list[tuple[float, list[MemoryToken]]]:
        """Closed blocks before `until` that have tokens but no note yet, oldest first. Only re-reads tokens
        from shortly before the last scan unless `full` (late uploads / failed notes are found by a full scan)."""
        last = self._scanned.get(device_id)
        lo = None if full or last is None else last - 2 * self.cfg.block_s
        toks = await self.store.list_tokens(
            time_from=lo, time_to=until, device_id=device_id, kinds=KINDS, vectors=False
        )
        b = self.cfg.block_s
        blocks: dict[float, list[MemoryToken]] = {}
        for t in toks:
            start = math.floor(t.ts / b) * b
            if start + b <= until:
                blocks.setdefault(start, []).append(t)
        return [
            (s, ts)
            for s, ts in sorted(blocks.items())
            if f"block:{s:.0f}" not in self.notes and len(ts) >= self.cfg.min_block_tokens
        ]

    async def pending(self, until: float, *, device_id: str | None = None) -> int:
        """How many closed blocks before `until` still need a note."""
        return len(await self._missing(until, device_id))

    async def build(
        self, until: float, *, device_id: str | None = None, full: bool = False, rollup: bool = True
    ) -> int:
        """Write the missing notes for every block that closed before `until` (and, with `rollup`, the hour/day
        overviews they complete); returns how many were written."""
        async with self._lock:
            todo = await self._missing(until, device_id, full=full)
            self._scanned[device_id] = max(until, self._scanned.get(device_id, until))
            b = self.cfg.block_s
            written = 0

            async def one(start: float, ts: list[MemoryToken]) -> None:
                nonlocal written
                lines = "\n".join(window_lines(ts, self.tz, self.cfg))[: self.cfg.note_input_chars]
                prompt = NOTE_PROMPT.format(start=self._fmt(start), end=self._fmt(start + b, "%H:%M"), lines=lines)
                try:
                    out = await self.llm(NOTE_SYSTEM, prompt, max_tokens=500)
                except RuntimeError as e:
                    log.warning("hier note for block %s failed: %s", self._fmt(start), e)
                    return
                self.notes[f"block:{start:.0f}"] = _note("block", start, start + b, out, len(ts))
                written += 1

            await asyncio.gather(*(one(s, ts) for s, ts in todo))
            for level, span in (("hour", 3600.0), ("day", 86400.0)) if rollup else ():
                written += await self._rollup(level, span, until)
            self._save()
            return written

    async def _rollup(self, level: str, span: float, until: float) -> int:
        child = "block" if level == "hour" else "hour"
        groups: dict[float, list[dict[str, Any]]] = {}
        for n in self.notes.values():
            if n["level"] == child:
                groups.setdefault(self._floor(n["start"], span), []).append(n)
        todo = [
            (s, ns) for s, ns in sorted(groups.items()) if f"{level}:{s:.0f}" not in self.notes and s + span <= until
        ]

        async def one(start: float, ns: list[dict[str, Any]]) -> bool:
            ns = sorted(ns, key=lambda n: n["start"])
            lines = "\n".join(f"{self._fmt(n['start'], '%H:%M')}: {n['summary']}" for n in ns)
            prompt = ROLLUP_PROMPT.format(start=self._fmt(start), end=self._fmt(start + span), lines=lines, span=level)
            try:
                out = await self.llm(NOTE_SYSTEM, prompt, max_tokens=600)
            except RuntimeError as e:
                log.warning("hier %s note %s failed: %s", level, self._fmt(start), e)
                return False
            self.notes[f"{level}:{start:.0f}"] = _note(level, start, start + span, out, sum(n["n"] for n in ns))
            return True

        return sum(await asyncio.gather(*(one(s, ns) for s, ns in todo)))

    def _floor(self, ts: float, span: float) -> float:
        """Start of the local hour/day containing `ts`."""
        if span >= 86400:
            d = datetime.fromtimestamp(ts, self.tz).replace(hour=0, minute=0, second=0, microsecond=0)
            return d.timestamp()
        return math.floor(ts / span) * span

    # ------------------------------------------------------------------ read

    def blocks(self, until: float, time_from: float | None = None) -> list[dict[str, Any]]:
        """Block notes overlapping [time_from, until), oldest first: stage 1's B0, B1, ... in this order."""
        return sorted(
            (
                n
                for n in self.notes.values()
                if n["level"] == "block" and n["start"] < until and (time_from is None or n["end"] > time_from)
            ),
            key=lambda n: n["start"],
        )

    def hours(self, until: float) -> list[dict[str, Any]]:
        """Hour notes closed by `until`, oldest first: stage 1's H0, H1, ... in this order."""
        return sorted(
            (n for n in self.notes.values() if n["level"] == "hour" and n["end"] <= until), key=lambda n: n["start"]
        )

    def outline(
        self, until: float, blocks: list[dict[str, Any]] | None = None, expand: set[float] | None = None
    ) -> str:
        """The log a stage-1 prompt shows: day and hour overviews, then `blocks` (default: every block before
        `until`) numbered B0, B1, ... in that order. With `expand` (hour starts) only the blocks of those hours (and
        of hours without a note yet) are listed; every other hour note carries an [H..] id (see `hours`)."""
        blocks = self.blocks(until) if blocks is None else blocks
        coarse = sorted(
            (n for n in self.notes.values() if n["level"] in ("day", "hour") and n["end"] <= until),
            key=lambda n: (n["start"], n["level"] != "day"),
        )
        h_idx = {n["start"]: i for i, n in enumerate(self.hours(until))} if expand is not None else {}
        lines = []
        for n in coarse:
            span = (
                self._fmt(n["start"], "%a %Y-%m-%d")
                if n["level"] == "day"
                else (f"{self._fmt(n['start'])}-{self._fmt(n['end'], '%H:%M')}")
            )
            tag = (
                f"[H{h_idx[n['start']]}] "
                if n["level"] == "hour" and expand is not None and n["start"] not in expand
                else ""
            )
            lines.append(f"{tag}{n['level'].upper()} {span}: {n['summary']}")
        lines.append("")
        for i, n in enumerate(blocks):
            hour = self._floor(n["start"], 3600)
            if expand is not None and hour in h_idx and hour not in expand:
                continue
            kw = ", ".join(n.get("keywords", [])[:20])
            lines.append(
                f"[B{i}] {self._fmt(n['start'])}-{self._fmt(n['end'], '%H:%M')}: {n['summary']} (keywords: {kw})"
            )
        return "\n".join(lines)


def _note(level: str, start: float, end: float, out: dict[str, Any], n: int) -> dict[str, Any]:
    kws = out.get("keywords") or []
    return {
        "level": level,
        "start": start,
        "end": end,
        "summary": str(out.get("summary", "")).strip(),
        "keywords": [str(k) for k in kws][:25] if isinstance(kws, list) else [],
        "n": n,
    }


SEARCH_SYSTEM = (
    "You search the wearer's first-person memory log to find the exact moments that answer a question. "
    "Questions are asked at 'now' and are about the past. A question often describes what is happening now "
    "(e.g. 'X is now doing Y, what did X do before?') and asks about an EARLIER related moment: find that earlier "
    "moment, not now. For 'first'/'usually'/'how many' questions consider every matching moment; for "
    "'last time'/'when was' questions the answer is the LATEST matching moment before now (it may be a day or "
    "more earlier). "
    "Use only the log; the hybrid search hints can be wrong."
)
STAGE1_PROMPT = """Memory log (overviews, then 10-minute blocks):
{outline}

Now: {now}
Question: {question}
Keyword/embedding search hits fall in blocks: {hints}

Pick up to {n} blocks most likely to contain the moment(s) that answer the question, best first. An [H..] hour \
whose blocks are not listed can be picked instead to read its blocks.
Return ONLY JSON: {{"reasoning": str (one sentence), "blocks": ["B.." or "H.."]}}"""
STAGE1_DIRECT_PROMPT = """Memory log (overviews, then 10-minute blocks):
{outline}

Now: {now}
Question: {question}

Moments around the keyword/embedding search hits (blocks {hints}), each line "[token id] time seen/heard: ...":
{lines}

If these moments already show or state the moment that answers the question (the evidence a person would look at), \
rank their token ids best first, up to {k}, and return no blocks. Otherwise return up to {n} blocks (best first) \
whose moments should be read instead; an [H..] hour whose blocks are not listed can be named to read its blocks. \
Return ONLY JSON: {{"reasoning": str (one sentence), "ids": [str], "blocks": ["B.." or "H.."]}}"""
STAGE2_PROMPT = """Now: {now}
Question: {question}

Moments from the chosen parts of my memory, each line "[token id] time seen/heard: ...":
{lines}

Rank the token ids of the lines that best show or state the moment that answers the question (the evidence a \
person would look at), best first, up to {k}. Prefer the moment the event actually happens. Return ONLY JSON: \
{{"reasoning": str (one sentence), "ids": [str]}}"""


class HierRetriever(Retriever):
    """`base` (hybrid) candidates re-ranked by a coarse-to-fine LLM search over `memory`'s notes."""

    def __init__(
        self,
        store: MemoryStore,
        base: Retriever,
        memory: HierMemory,
        llm: ChatJSON | None = None,
        *,
        config: HierConfig | None = None,
        auto_build: bool = True,
        rank_llm: ChatJSON | None = None,
    ) -> None:
        self.store = store
        self.base = base
        self.memory = memory
        self.llm = llm or memory.llm  # stage 1 (block pick); also stage 2 unless `rank_llm` is set
        self.rank_llm = rank_llm or self.llm
        self.cfg = config or memory.cfg
        self.auto_build = auto_build
        self.last_trace: dict[str, Any] = {}
        self.stats = {"llm": 0, "gated": 0, "direct": 0, "errors": 0}
        self.recent: deque[dict[str, Any]] = deque(maxlen=200)  # per recall: path + ms, for GET /v1/stats
        self._bg: asyncio.Task | None = None

    def summary(self) -> dict[str, Any]:
        """Counters plus p50/p95 of the LLM stages over the last `recent` recalls (gated ones count as 0 ms)."""

        def pct(key: str) -> dict[str, int] | None:
            xs = sorted(r[key] for r in self.recent if r.get(key) is not None)
            if not xs:
                return None
            return {"p50": xs[len(xs) // 2], "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))]}

        return {
            **self.stats,
            "recent": len(self.recent),
            "llm_ms": pct("llm_ms"),
            "stage1_ms": pct("stage1_ms"),
            "stage2_ms": pct("stage2_ms"),
        }

    async def _ensure_notes(self, until: float, device_id: str | None) -> None:
        """Write a few missing block notes inline; a large backlog (cold start, a long upload) is built in the
        background so the query answers from the notes that exist instead of waiting on dozens of LLM calls.
        Hour/day overviews are never written inline: stage 1 lists the blocks either way, and the two serial
        rollup calls (~2 s each) on top of a block note (~3 s) would eat the whole `timeout_s` budget."""
        if await self.memory.pending(until, device_id=device_id) <= self.cfg.inline_build_max:
            if not await self.memory.build(until, device_id=device_id, rollup=False):
                return
        if self._bg is None or self._bg.done():
            self._bg = asyncio.create_task(self.memory.build(until, device_id=device_id))

    async def run_forever(self, interval_s: float) -> None:
        """Keep notes current in the background (server lifespan); a full rescan every 6th pass."""
        n = 0
        while True:
            try:
                await self.memory.build(time.time(), full=n % 6 == 0)
            except Exception:  # noqa: BLE001
                log.exception("hier note refresh failed")
            n += 1
            await asyncio.sleep(interval_s)

    async def retrieve(
        self,
        query: str,
        *,
        k: int = 12,
        mode: RetrieveMode = "hybrid",
        time_from: float | None = None,
        time_to: float | None = None,
        device_id: str | None = None,
        now: datetime | None = None,
    ) -> RetrievalResult:
        now = now or datetime.now(UTC)
        base = await self.base.retrieve(
            query,
            k=max(k, self.cfg.base_pool),
            mode=mode,
            time_from=time_from,
            time_to=time_to,
            device_id=device_id,
            now=now,
        )
        if mode not in ("hybrid", None) or not query.strip():
            return _truncate(base, k)
        n_hits = sum(not r.is_context for r in base.tokens)
        if self.cfg.gate_margin > 0 and n_hits >= self.cfg.gate_min_hits and top_margin(base) >= self.cfg.gate_margin:
            self.stats["gated"] += 1
            self.last_trace = {"gated": round(top_margin(base), 3)}
            self.recent.append({"path": "gated", "llm_ms": 0})
            return _truncate(base, k)
        t0 = time.perf_counter()
        try:
            stages = self._llm_rank(query, base, k, time_from, time_to, device_id, now)
            picked = await asyncio.wait_for(stages, self.cfg.timeout_s or None)
        except (TimeoutError, RuntimeError, httpx.HTTPError, ValueError, KeyError) as e:
            if isinstance(e, TimeoutError):
                e = TimeoutError(f"LLM stages took > {self.cfg.timeout_s:g} s")
            log.warning("hier recall fell back to hybrid: %s", e)
            self.last_trace = {"error": str(e)}
            self.stats["errors"] += 1
            self.recent.append({"path": "error", "llm_ms": _ms(t0)})
            return _truncate(base, k)
        self.stats["llm"] += 1
        self.last_trace["llm_ms"] = _ms(t0)
        self.recent.append(
            {
                "path": "direct" if self.last_trace.get("direct") else "llm",
                "llm_ms": self.last_trace["llm_ms"],
                "stage1_ms": self.last_trace.get("stage1_ms"),
                "stage2_ms": self.last_trace.get("stage2_ms"),
            }
        )
        tokens = _interleave(_spread(picked, self.cfg.moment_s), base.ranked, k)
        return RetrievalResult(plan=base.plan, tokens=tokens, widened=base.widened, anchor_ts=base.anchor_ts)

    async def _llm_rank(
        self,
        query: str,
        base: RetrievalResult,
        k: int,
        time_from: float | None,
        time_to: float | None,
        device_id: str | None,
        now: datetime,
    ) -> list[RetrievedToken]:
        until = min(now.timestamp(), time_to) if time_to is not None else now.timestamp()
        if self.auto_build:
            await self._ensure_notes(until, device_id)
        blocks = self.memory.blocks(until, time_from)
        now_s = self.memory._fmt(until)
        hits = base.ranked[: self.cfg.hint_windows]
        w = self.cfg.window_s
        hint_spans = [(r.token.ts - w / 2, r.token.ts + w / 2) for r in hits]
        vec_ids, hint_toks = await asyncio.gather(
            self._vector_blocks(query, blocks, time_from, until, device_id),
            self._tokens_in(hint_spans, time_from, until, device_id),
        )
        hint_ids = [f"B{i}" for i in vec_ids]
        for r in hits:
            for i, b in enumerate(blocks):
                if b["start"] <= r.token.ts < b["end"] and f"B{i}" not in hint_ids:
                    hint_ids.append(f"B{i}")
        chosen: list[dict[str, Any]] = []
        if blocks:
            expand = self._expand_hours(hits, [blocks[i] for i in vec_ids]) if self.cfg.outline_hours > 0 else None
            outline = self.memory.outline(until, blocks, expand)
            hours = self.memory.hours(until) if expand is not None else []
            t1 = time.perf_counter()
            if self.cfg.direct and hint_toks:
                lines = self._lines(hint_toks)
                out = await self.llm(
                    SEARCH_SYSTEM,
                    STAGE1_DIRECT_PROMPT.format(
                        now=now_s,
                        question=query,
                        outline=outline,
                        hints=", ".join(hint_ids) or "none",
                        lines="\n".join(lines),
                        k=k,
                        n=self.cfg.max_blocks,
                    ),
                    max_tokens=400,
                )
                picked = self._pick(out, lines, hint_toks, k)
                if picked and not out.get("blocks"):
                    self.stats["direct"] += 1
                    self.last_trace = {"stage1": out, "direct": True, "stage1_ms": _ms(t1)}
                    return picked
            else:
                out = await self.llm(
                    SEARCH_SYSTEM,
                    STAGE1_PROMPT.format(
                        now=now_s,
                        question=query,
                        outline=outline,
                        hints=", ".join(hint_ids) or "none",
                        n=self.cfg.max_blocks,
                    ),
                    max_tokens=300,
                )
            for bid in out.get("blocks", []) or []:
                m = re.fullmatch(r"\[?([BH])(\d+)\]?", str(bid).strip())
                if not m:
                    continue
                i = int(m.group(2))
                if m.group(1) == "B" and i < len(blocks):
                    want = [blocks[i]]
                elif m.group(1) == "H" and i < len(hours):
                    want = [b for b in blocks if hours[i]["start"] <= b["start"] < hours[i]["end"]]
                else:
                    continue
                for b in want:
                    if b not in chosen and len(chosen) < self.cfg.max_blocks:
                        chosen.append(b)
            for i in vec_ids[: self.cfg.vec_blocks]:
                if blocks[i] not in chosen:
                    chosen.append(blocks[i])
            self.last_trace = {
                "stage1": out,
                "blocks": [b["start"] for b in chosen],
                "stage1_ms": _ms(t1),
                "outline_chars": len(outline),
            }

        spans = [(b["start"], min(b["end"], until)) for b in chosen]
        toks = dict(hint_toks)
        toks.update(await self._tokens_in(spans, time_from, until, device_id))
        if not toks:
            return []
        lines = self._lines(toks)
        t2 = time.perf_counter()
        out2 = await self.rank_llm(
            SEARCH_SYSTEM,
            STAGE2_PROMPT.format(now=now_s, question=query, lines="\n".join(lines), k=k),
            max_tokens=400,
        )
        self.last_trace.update(stage2=out2, stage2_ms=_ms(t2))
        return self._pick(out2, lines, toks, k)

    def _expand_hours(self, hits: Sequence[RetrievedToken], vec_blocks: list[dict[str, Any]]) -> set[float]:
        """The hours whose blocks stage 1 gets to read in full: around the top hybrid hits and the top vector-ranked
        blocks, best first, interleaved, up to `outline_hours`."""
        starts = [r.token.ts for r in hits]
        cands = [x for pair in zip_longest(starts, [b["start"] for b in vec_blocks]) for x in pair if x is not None]
        out: set[float] = set()
        for ts in cands:
            out.add(self.memory._floor(ts, 3600))
            if len(out) >= self.cfg.outline_hours:
                break
        return out

    async def _tokens_in(
        self, spans: list[tuple[float, float]], time_from: float | None, until: float, device_id: str | None
    ) -> dict[str, MemoryToken]:
        toks: dict[str, MemoryToken] = {}
        for lo, hi in _merge(spans):
            lo = max(lo, time_from) if time_from is not None else lo
            for t in await self.store.range(lo, min(hi, until), 10**6, device_id=device_id, kinds=KINDS):
                toks[t.id] = t
        return toks

    def _lines(self, toks: dict[str, MemoryToken]) -> list[str]:
        return window_lines(sorted(toks.values(), key=lambda t: t.ts), self.memory.tz, self.cfg)

    @staticmethod
    def _pick(out: dict[str, Any], lines: list[str], toks: dict[str, MemoryToken], k: int) -> list[RetrievedToken]:
        """The LLM's `ids`, in its order, kept to ids that were actually shown."""
        shown = {ln[1 : ln.index("]")] for ln in lines}
        picked: list[RetrievedToken] = []
        for rank, tid in enumerate(out.get("ids", []) or []):
            tid = str(tid).strip().strip("[]")
            if tid in shown and tid in toks and all(p.token.id != tid for p in picked):
                picked.append(RetrievedToken(toks[tid], 2.0 - 0.01 * rank, ["llm"]))
            if len(picked) >= k:
                break
        return picked

    async def _vector_blocks(
        self,
        query: str,
        blocks: list[dict[str, Any]],
        time_from: float | None,
        until: float,
        device_id: str | None,
    ) -> list[int]:
        """Block indices ranked by their best text-vector matches (mean of the top 3 among the `vec_pool` nearest
        tokens): finds blocks whose note left out the detail asked about. Empty when the base has no embedder."""
        embedder = getattr(self.base, "text_embedder", None)
        adapter = getattr(self.base, "adapter", None)
        if embedder is None or adapter is None or not blocks:
            return []
        vec = await self.base._embed_query("text_vec", embedder.embed_query, query)
        hits = await adapter.vector_search(
            vec, self.cfg.vec_pool, "text_vec", time_from=time_from, time_to=until, device_id=device_id
        )
        score = dict(hits)
        starts = [b["start"] for b in blocks]
        per: dict[int, list[float]] = {}
        for t in await adapter.get_many([i for i, _ in hits]):
            i = bisect.bisect_right(starts, t.ts) - 1
            if i >= 0 and t.ts < blocks[i]["end"]:
                per.setdefault(i, []).append(score[t.id])
        top3 = {i: sum(sorted(v)[-3:]) / 3 for i, v in per.items()}
        return sorted(top3, key=lambda i: -top3[i])[: self.cfg.hint_windows]


def _ms(t0: float) -> int:
    return round(1000 * (time.perf_counter() - t0))


def top_margin(res: RetrievalResult) -> float:
    """Relative lead of the best non-context hit over the runner-up, (s1 - s2) / s1; 0 with no hits."""
    s = sorted((r.score for r in res.tokens if not r.is_context), reverse=True)
    if not s or s[0] <= 0:
        return 0.0
    return 1.0 if len(s) == 1 else (s[0] - s[1]) / s[0]


def _spread(picked: list[RetrievedToken], moment_s: float) -> list[RetrievedToken]:
    """Keep the LLM's order but put one pick per distinct moment first, near-duplicates after."""
    lead: list[RetrievedToken] = []
    tail: list[RetrievedToken] = []
    for r in picked:
        near = any(abs(r.token.ts - x.token.ts) < moment_s for x in lead)
        (tail if near else lead).append(r)
    return lead + tail


def _interleave(picked: list[RetrievedToken], base: list[RetrievedToken], k: int) -> list[RetrievedToken]:
    """LLM pick, hybrid hit, LLM pick, ...: the LLM leads, but a strong lexical/vector hit it skipped stays in
    the top few (measured on EgoLife: +4 pts R@5 on day 1, same R@1)."""
    out: list[RetrievedToken] = []
    seen: set[str] = set()
    its = [iter(picked), iter(base)]
    while len(out) < k and its:
        for it in list(its):
            r = next((r for r in it if r.token.id not in seen), None)
            if r is None:
                its.remove(it)
                continue
            seen.add(r.token.id)
            out.append(r)
            if len(out) >= k:
                break
    return out


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _truncate(res: RetrievalResult, k: int) -> RetrievalResult:
    keep = {r.token.id for r in res.ranked[:k]}
    return RetrievalResult(
        plan=res.plan,
        tokens=[r for r in res.tokens if r.token.id in keep],
        widened=res.widened,
        anchor_ts=res.anchor_ts,
    )


def build_hier(
    settings,
    store: MemoryStore,
    base: Retriever,
    api_key: str,
    *,
    cache_path: Path | str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> HierRetriever:
    """`cache_path` / `transport` override the settings (the eval harness keeps its notes out of data/)."""
    url = getattr(settings, "hier_base_url", "https://api.openai.com/v1")
    effort = getattr(settings, "hier_reasoning_effort", "low") or "low"
    kw = dict(base_url=url, transport=transport, reasoning_effort=effort)
    note_llm = ChatJSON(api_key, getattr(settings, "hier_note_model", "gpt-4.1-mini"), concurrency=16, **kw)
    pick_model = getattr(settings, "hier_model", "gpt-4.1-mini")
    rank_model = getattr(settings, "hier_rank_model", "gpt-4.1") or pick_model
    # recall-time calls fail fast (the stage is also bounded by timeout_s); note writing keeps the patient defaults
    fast = dict(kw, timeout_s=float(getattr(settings, "hier_call_timeout_s", 4.0) or 4.0))
    fast["retries"] = max(1, int(getattr(settings, "hier_retries", 2) or 2))
    pick_llm = ChatJSON(api_key, pick_model, **fast)
    rank_llm = pick_llm if rank_model == pick_model else ChatJSON(api_key, rank_model, **fast)
    cfg = HierConfig(
        timeout_s=float(getattr(settings, "hier_timeout_s", 8.0)),
        gate_margin=float(getattr(settings, "hier_gate_margin", 0.0) or 0.0),
        direct=bool(getattr(settings, "hier_direct", False)),
        gate_min_hits=int(getattr(settings, "hier_gate_min_hits", 5) or 0),
        outline_hours=int(getattr(settings, "hier_outline_hours", 0) or 0),
        base_pool=int(getattr(settings, "hier_base_pool", 30) or 30),
        inline_build_max=int(getattr(settings, "hier_inline_build_max", 3) or 0),
        hint_windows=int(getattr(settings, "hier_hint_windows", 10) or 10),
    )
    memory = HierMemory(
        store,
        note_llm,
        cache_path=cache_path or getattr(settings, "hier_cache_path", "") or None,
        tz=getattr(settings, "wearer_tz", None),
        config=cfg,
    )
    return HierRetriever(store, base, memory, pick_llm, rank_llm=rank_llm)
