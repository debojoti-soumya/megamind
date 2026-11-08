"""Claude orchestrator: conversation + tool use over `recall_memory`.

The loop is the standard Anthropic tool-use loop: ``messages.create`` with
``tools=[RECALL_MEMORY_TOOL]``; every ``tool_use`` block is executed through
:class:`RecallMemory`, its result appended as a ``tool_result``, and the model is
called again until it stops with ``end_turn`` (or we hit ``MAX_TOOL_ROUNDS``).
Per-session history is kept in memory, capped at ``MAX_TURNS`` exchanges.

Latency: with ``RECALL_PREFETCH`` the turn starts ``recall_memory`` on the raw question
concurrently with the first model call; Claude's first tool call reuses it when its keywords
mostly come from what the prefetch searched (the question plus the planner's rewrite) and any
time bounds it adds match the planner's window (:func:`prefetch_reusable`), otherwise the
prefetch is cancelled and the tool runs with Claude's arguments. With ``RECALL_FIRST`` a
standalone recall question (:func:`recall_first`) skips that tool-picking call: the prefetch
goes to Claude as its first tool result, and Claude can still re-query.

Without ``ANTHROPIC_API_KEY`` (or when the API fails) the deterministic
``_no_llm_fallback`` path answers straight from the reader's evidence so the
whole system keeps working offline.

Streaming (``POST /v1/chat/stream``): pass ``on_event`` to :meth:`ClaudeAgent.chat_with_stats` and the same
turn runs with ``messages.stream`` instead of ``messages.create`` (identical request parameters, so the same
answer), reporting ``retrieval`` (candidate citations after each tool result), ``delta`` (answer text as it is
generated, ``<cites>`` trailer held back) and ``reset`` (discard the streamed text: that round turned into a tool
call, or the API failed and the caption fallback answers instead). The final :class:`ChatResponse` is unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anthropic
from ulid import ULID

from megamind.agent.tools import OCR_MATCH_PREFIX, RECALL_MEMORY_TOOL, RecallMemory, keywords
from megamind.config import Settings
from megamind.live import live_question
from megamind.models import ChatResponse, Citation, RecallResult, ToolCallRecord
from megamind.retrieval.planner import QueryPlan, plan_query

log = logging.getLogger("megamind.agent")

MAX_TOOL_ROUNDS = 4
MAX_TURNS = 10
MAX_TOKENS = 400
PREFETCH_MIN_OVERLAP = 0.5  # share of Claude's query keywords the prefetch searched for, to reuse it
PREFETCH_MIN_TIME_IOU = 0.5  # Claude's time window vs the planner's window for the raw question
NOT_FOUND_ANSWER = "I don't have a memory of that."
_CITES_RE = re.compile(r"\s*<cites>(.*?)</cites>\s*", re.DOTALL)

SYSTEM_PROMPT = """You are Megamind, a calm assistant living in the wearer's AR headset. Your replies are
spoken aloud by text-to-speech, so answer in one or two short plain sentences: no markdown,
no lists, no token ids, no URLs.

You have one tool, recall_memory, which searches first-person snapshots captured by the
wearer's headset cameras. Call it whenever a question depends on past perception: where an
object is, what some text said, who or what was there, what the wearer was doing earlier.
Pass short keywords as query (the object, place or text), not the whole question.

The wearer's local time is {local} ({tz}); in UTC that is {now}. Resolve relative times
("ten minutes ago", "this morning", "yesterday") against the LOCAL time and pass them as
ISO-8601 with the UTC offset, e.g. {example}. Leave time_from/time_to null when the
question has no time hint. You may call the tool more than once per question, for example to
widen the time window or rephrase the query when the first result is not_found or low
confidence, but stop after a few attempts.

Ground every claim in the evidence the tool returns. Mention when something was seen using
the relative times in the tool result (for example "about twenty minutes ago"). If the tool
returns not_found, or the evidence does not actually answer the question, say
"I don't have a memory of that." and never guess or invent details. Never claim to identify
people; describe them neutrally. If the evidence is marked "reader unavailable", answer from
the captions but add that you could not check the images.

After your spoken answer, on the same message, append the token ids of the evidence your
answer relies on inside a trailer like <cites>id1,id2</cites>, including ids from tool results
earlier in this conversation when you answer from those. The trailer is stripped before
speech. Omit it when you cite nothing."""


@dataclass
class TurnStats:
    """Per-turn telemetry (not part of the API contract; used by logs and tools/eval_agent.py)."""

    request_id: str
    path: str = "claude"
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    total_ms: float = 0.0
    error: str | None = None
    cites_from: str = "reader"
    extra: dict[str, Any] = field(default_factory=dict)


def _zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("unknown WEARER_TZ %r, using UTC", tz_name)
        return ZoneInfo("UTC")


def render_system_prompt(now: float | None, tz_name: str) -> str:
    utc = datetime.fromtimestamp(now, UTC) if now is not None else datetime.now(UTC)
    zone = _zone(tz_name)
    local = utc.astimezone(zone)
    example = local.replace(hour=max(0, local.hour - 1), minute=0, second=0, microsecond=0)
    return SYSTEM_PROMPT.format(
        local=local.strftime("%A %Y-%m-%d %H:%M:%S %Z"),
        tz=zone.key,
        now=utc.isoformat(timespec="seconds"),
        example=example.isoformat(timespec="seconds"),
    )


def _relative(ts: float, now: float) -> str:
    d = max(0.0, now - ts)
    if d < 60:
        return "just now"
    if d < 3600:
        m = int(round(d / 60))
        return f"about {m} minute{'s' if m != 1 else ''} ago"
    if d < 86400:
        h = d / 3600
        return f"about {h:.0f} hour{'s' if round(h) != 1 else ''} ago"
    days = int(d // 86400)
    return f"about {days} day{'s' if days != 1 else ''} ago"


def _tool_result_payload(r: RecallResult, now: float, zone: ZoneInfo) -> str:
    def cite(c: Citation) -> dict:
        return {
            "id": c.id,
            "when": _relative(c.ts, now),
            "time_local": datetime.fromtimestamp(c.ts, zone).strftime("%a %H:%M"),
            "time_utc": datetime.fromtimestamp(c.ts, UTC).isoformat(timespec="seconds"),
            "caption": c.caption,
        }

    payload = {
        "confidence": r.confidence,
        "evidence": r.evidence,
        "cited": [cite(c) for c in r.cited],
        "retrieved_count": len(r.all_retrieved),
        "other_retrieved": [cite(c) for c in r.all_retrieved if c.id not in {x.id for x in r.cited}][:8],
    }
    return json.dumps(payload, ensure_ascii=False)


def _stem(w: str) -> str:
    """Crude suffix strip so "parked"/"parking"/"park" and "eating"/"eat" compare equal."""
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    w = w.removesuffix("e") if len(w) > 3 else w
    return w[:-1] if len(w) > 3 and w[-1] == w[-2] else w


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in keywords(text)}


def _iso_ts(s: object) -> float | None:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp() if s else None


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def prefetch_reusable(args: dict, question: str, plan: QueryPlan | None = None) -> bool:
    """Claude's first recall_memory call can reuse the raw-question prefetch when it keeps the default
    mode / k, its keywords mostly come from what the prefetch searched (the question, plus the planner's
    rewrite `plan.search_text`), and any time bounds it adds overlap the planner's window for the question."""
    if args.get("mode", "hybrid") not in (None, "hybrid") or args.get("k", 12) not in (None, 12):
        return False
    try:
        t_from, t_to = _iso_ts(args.get("time_from")), _iso_ts(args.get("time_to"))
    except ValueError:
        return False
    if t_from is not None or t_to is not None:
        if plan is None or plan.time_from is None or plan.time_to is None:
            return False
        window = (plan.time_from if t_from is None else t_from, plan.time_to if t_to is None else t_to)
        if _iou(window, (plan.time_from, plan.time_to)) < PREFETCH_MIN_TIME_IOU:
            return False
    q = _stems(str(args.get("query") or ""))
    searched = _stems(question) | (_stems(plan.search_text) if plan else set())
    return bool(q) and len(q & searched) / len(q) >= PREFETCH_MIN_OVERLAP


_WORD_RE = re.compile(r"[a-z0-9]+")
_QUESTION_STARTS = frozenset(
    "where what when who whom whose which how did do does is are was were have has had can could remind".split()
)
# pronouns / sequencing words that point at an earlier turn: let Claude read the history first
_FOLLOW_UP_CUES = frozenset(
    "it its that them those they there he she him her after before then also else again".split()
)


def recall_first(question: str, plan: QueryPlan | None) -> bool:
    """True when the question plainly asks about something seen and stands on its own: starts like a question,
    has search terms after planning, and has no follow-up cue ("and ...", "what about ...", "it", "after that")."""
    words = _WORD_RE.findall(question.lower())
    if not words or plan is None or not plan.search_text.strip():
        return False
    if words[0] not in _QUESTION_STARTS or words[:2] in (["what", "about"], ["how", "about"]):
        return False
    return not set(words) & _FOLLOW_UP_CUES


async def _cancel(task: asyncio.Task | None) -> None:
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


EventCallback = Callable[[str, dict], Awaitable[None]]


class _TrailerFilter:
    """Streams answer text minus the `<cites>...</cites>` trailer: holds back a possible partial tag."""

    OPEN, CLOSE = "<cites>", "</cites>"

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    def feed(self, s: str) -> str:
        self.buf += s
        out: list[str] = []
        while True:
            if self.inside:
                j = self.buf.find(self.CLOSE)
                if j < 0:
                    self.buf = self.buf[-(len(self.CLOSE) - 1) :]
                    return "".join(out)
                self.buf, self.inside = self.buf[j + len(self.CLOSE) :], False
                continue
            j = self.buf.find(self.OPEN)
            if j >= 0:
                out.append(self.buf[:j])
                self.buf, self.inside = self.buf[j + len(self.OPEN) :], True
                continue
            keep = next(
                (
                    k
                    for k in range(min(len(self.buf), len(self.OPEN) - 1), 0, -1)
                    if self.OPEN.startswith(self.buf[-k:])
                ),
                0,
            )
            cut = len(self.buf) - keep
            out.append(self.buf[:cut])
            self.buf = self.buf[cut:]
            return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


class _Emitter:
    """Typed wrapper over the streaming callback; remembers whether answer text went out (for `reset`)."""

    def __init__(self, cb: EventCallback) -> None:
        self.cb = cb
        self.dirty = False

    async def delta(self, text: str) -> None:
        if text:
            self.dirty = True
            await self.cb("delta", {"text": text})

    async def reset(self) -> None:
        if self.dirty:
            self.dirty = False
            await self.cb("reset", {})

    async def retrieval(self, r: RecallResult) -> None:
        seen: set[str] = set()
        cands = [c for c in [*r.cited, *r.all_retrieved] if not (c.id in seen or seen.add(c.id))]
        await self.cb("retrieval", {"confidence": r.confidence, "candidates": cands})


def _split_cites(text: str) -> tuple[str, list[str] | None]:
    m = _CITES_RE.search(text)
    if not m:
        return text.strip(), None
    ids = [i.strip() for i in m.group(1).split(",") if i.strip()]
    return _CITES_RE.sub(" ", text).strip(), ids


class ClaudeAgent:
    def __init__(
        self,
        settings: Settings,
        recall: RecallMemory,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.settings = settings
        self.recall = recall
        self.histories: dict[str, list[list[dict]]] = {}
        self.cite_histories: dict[str, list[list[Citation]]] = {}  # per turn, trimmed with `histories`
        self._client = client
        if self._client is None and settings.anthropic_api_key:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                max_retries=settings.claude_max_retries,
                timeout=settings.claude_timeout_s,
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def chat(self, session_id: str, text: str, device_id: str | None) -> ChatResponse:
        resp, _ = await self.chat_with_stats(session_id, text, device_id)
        return resp

    async def chat_with_stats(
        self,
        session_id: str,
        text: str,
        device_id: str | None,
        *,
        now: float | None = None,
        on_event: EventCallback | None = None,
    ) -> tuple[ChatResponse, TurnStats]:
        """`now` pins the wall clock (system prompt, relative times) for deterministic evals. `on_event` streams
        the turn (see the module docstring); without it the turn is exactly the non-streaming one."""
        stats = TurnStats(request_id=str(ULID()))
        emit = _Emitter(on_event) if on_event is not None else None
        live_question.set(text)  # lets live short-term memory see "right now" behind Claude's keyword query
        t0 = time.perf_counter()
        if not self.enabled:
            stats.path = "fallback"
            resp = await self._no_llm_fallback(text, device_id, stats, now=now, emit=emit)
        else:
            try:
                resp = await self._claude_turn(session_id, text, device_id, stats, now=now, emit=emit)
            except (anthropic.APIError, anthropic.APIConnectionError) as e:  # network, 4xx/5xx, timeouts
                log.warning("req=%s claude failed (%s); using caption fallback", stats.request_id, e)
                stats.path, stats.error = "fallback", f"{type(e).__name__}: {e}"
                if emit is not None:
                    await emit.reset()
                resp = await self._no_llm_fallback(text, device_id, stats, now=now, emit=emit)
        stats.total_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "req=%s session=%s path=%s llm_calls=%d tool_calls=%d in_tok=%d out_tok=%d "
            "llm_ms=%.0f tool_ms=%.0f total_ms=%.0f",
            stats.request_id,
            session_id,
            stats.path,
            stats.model_calls,
            stats.tool_calls,
            stats.input_tokens,
            stats.output_tokens,
            stats.llm_ms,
            stats.tool_ms,
            stats.total_ms,
        )
        return resp, stats

    async def _claude_turn(
        self,
        session_id: str,
        text: str,
        device_id: str | None,
        stats: TurnStats,
        *,
        now: float | None = None,
        emit: _Emitter | None = None,
    ) -> ChatResponse:
        assert self._client is not None
        now = time.time() if now is None else now
        turn: list[dict] = [{"role": "user", "content": text}]
        prior = [m for t in self.histories.get(session_id, []) for m in t]
        tool_calls: list[ToolCallRecord] = []
        # ids from earlier turns stay citable: a follow-up answered from history still gets thumbnails
        retrieved: dict[str, Citation] = {c.id: c for t in self.cite_histories.get(session_id, []) for c in t}
        reader_cited: list[Citation] = []
        turn_cites: list[Citation] = []
        answer_text = ""

        prefetch: asyncio.Task[RecallResult] | None = None
        plan: QueryPlan | None = None
        if self.settings.recall_prefetch:
            # overlap retrieval + reader with Claude's first call; most first tool calls are the question's keywords
            prefetch = asyncio.create_task(self._run_tool("recall_memory", {"query": text}, device_id, now))
            stats.extra["prefetch"] = "unused"
            try:
                plan = plan_query(text, now=datetime.fromtimestamp(now, UTC), tz=self.settings.wearer_tz)
            except Exception as e:  # noqa: BLE001 - the plan only widens reuse; never fail the turn on it
                log.warning("req=%s prefetch plan failed: %r", stats.request_id, e)

        def record(use_id: str, name: str, args: dict, r: RecallResult, t_tool: float) -> dict:
            if r.evidence.startswith(OCR_MATCH_PREFIX):
                stats.extra["reader_skipped"] = stats.extra.get("reader_skipped", 0) + 1
            stats.tool_ms += (time.perf_counter() - t_tool) * 1000
            stats.tool_calls += 1
            tool_calls.append(ToolCallRecord(name=name, args=args, confidence=r.confidence))
            for c in [*r.all_retrieved, *r.cited]:
                retrieved.setdefault(c.id, c)
                turn_cites.append(c)
            for c in r.cited:
                if c.id not in {x.id for x in reader_cited}:
                    reader_cited.append(c)
            log.info(
                "req=%s tool=%s args=%s -> %s (%d cited / %d retrieved)",
                stats.request_id,
                name,
                json.dumps(args, ensure_ascii=False),
                r.confidence,
                len(r.cited),
                len(r.all_retrieved),
            )
            payload = _tool_result_payload(r, now, _zone(self.settings.wearer_tz))
            return {"type": "tool_result", "tool_use_id": use_id, "content": payload}

        try:
            if prefetch is not None and self.settings.recall_first and recall_first(text, plan):
                # the question plainly needs recall: hand Claude the prefetch as its first tool result instead of
                # paying a Claude round-trip to pick (almost always) the same call; it can still re-query
                t_tool = time.perf_counter()
                args = {"query": text}
                r, prefetch, stats.extra["prefetch"] = await prefetch, None, "first"
                use_id = f"toolu_recall_{stats.request_id}"
                turn.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": use_id, "name": "recall_memory", "input": args}],
                    }
                )
                turn.append({"role": "user", "content": [record(use_id, "recall_memory", args, r, t_tool)]})
                if emit is not None:
                    await emit.retrieval(r)
            # opt-in prompt caching when the turn opens with Claude's tool-picking call: a breakpoint on the question
            # block on every call of the turn, so call 1 writes tools + system + history + question and the tool
            # round(s) read it back (a cache read needs a breakpoint in the reading request too). The system prompt
            # carries the time, so turns can't share an entry. Request bodies are unchanged when the flag is off.
            cache_question = self.settings.claude_prompt_cache and len(turn) == 1
            for _round in range(MAX_TOOL_ROUNDS + 1):
                t_llm = time.perf_counter()
                messages = prior + turn
                if cache_question:
                    question = {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}
                    messages = [*prior, {"role": "user", "content": [question]}, *turn[1:]]
                req = {
                    "model": self.settings.claude_model,
                    "max_tokens": MAX_TOKENS,
                    "system": self.system_prompt(now),
                    "tools": [RECALL_MEMORY_TOOL],
                    "messages": messages,
                }
                msg = await self._create(req, emit)
                stats.llm_ms += (time.perf_counter() - t_llm) * 1000
                stats.model_calls += 1
                stats.input_tokens += msg.usage.input_tokens
                stats.output_tokens += msg.usage.output_tokens
                for k in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                    if n := getattr(msg.usage, k, None):
                        stats.extra[k] = stats.extra.get(k, 0) + n

                assistant_content = [b.model_dump(exclude_none=True) for b in msg.content]
                turn.append({"role": "assistant", "content": assistant_content})
                texts = [b.text for b in msg.content if b.type == "text"]
                if texts:
                    answer_text = "\n".join(texts)
                uses = [b for b in msg.content if b.type == "tool_use"]
                if msg.stop_reason != "tool_use" or not uses or _round == MAX_TOOL_ROUNDS:
                    break
                if emit is not None:
                    await emit.reset()  # text before a tool call is not the answer

                results: list[dict] = []
                for use in uses:
                    args = dict(use.input) if isinstance(use.input, dict) else {}
                    t_tool = time.perf_counter()
                    if prefetch is not None and use.name == "recall_memory" and prefetch_reusable(args, text, plan):
                        r, stats.extra["prefetch"] = await prefetch, "hit"
                    else:
                        if prefetch is not None:
                            stats.extra["prefetch"] = "miss"
                            await _cancel(prefetch)
                        r = await self._run_tool(use.name, args, device_id, now)
                    prefetch = None
                    results.append(record(use.id, use.name, args, r, t_tool))
                    if emit is not None:
                        await emit.retrieval(r)
                turn.append({"role": "user", "content": results})

        finally:
            await _cancel(prefetch)

        answer, cite_ids = _split_cites(answer_text)
        if not answer:
            answer = NOT_FOUND_ANSWER
        if cite_ids is not None:
            cited = [retrieved[i] for i in cite_ids if i in retrieved]
            stats.cites_from = "claude"
        else:
            cited = reader_cited
        if answer.strip() == NOT_FOUND_ANSWER:
            cited = []

        hist = self.histories.setdefault(session_id, [])
        hist.append(turn)
        del hist[:-MAX_TURNS]
        cite_hist = self.cite_histories.setdefault(session_id, [])
        cite_hist.append(turn_cites)
        del cite_hist[:-MAX_TURNS]
        return ChatResponse(transcript=text, answer=answer, cited=cited, tool_calls=tool_calls)

    async def _create(self, req: dict, emit: _Emitter | None):
        """One Messages API call: `messages.create`, or `messages.stream` (same parameters) forwarding text deltas."""
        assert self._client is not None
        if emit is None:
            return await self._client.messages.create(**req)
        filt = _TrailerFilter()
        n_text = 0
        async with self._client.messages.stream(**req) as stream:
            async for ev in stream:
                if ev.type == "content_block_start" and ev.content_block.type == "text":
                    if n_text:
                        await emit.delta(filt.feed("\n"))  # answer_text joins text blocks with newlines
                    n_text += 1
                elif ev.type == "text":
                    await emit.delta(filt.feed(ev.text))
            msg = await stream.get_final_message()
        await emit.delta(filt.flush())
        return msg

    async def _run_tool(self, name: str, args: dict, device_id: str | None, now: float | None = None) -> RecallResult:
        if name != "recall_memory":
            return RecallResult(evidence=f"Unknown tool {name}.", confidence="not_found", cited=[], all_retrieved=[])
        allowed = {"query", "time_from", "time_to", "k", "mode"}
        kwargs = {k: v for k, v in args.items() if k in allowed and v is not None}
        kwargs.setdefault("query", "")
        try:
            return await self.recall(
                device_id=device_id, now=now, skip_reader_on_ocr=self.settings.reader_skip_on_ocr, **kwargs
            )
        except ValueError as e:  # bad ISO timestamp etc. -> tell the model instead of crashing the turn
            return RecallResult(evidence=f"Tool error: {e}", confidence="not_found", cited=[], all_retrieved=[])

    async def _no_llm_fallback(
        self,
        text: str,
        device_id: str | None,
        stats: TurnStats | None = None,
        *,
        now: float | None = None,
        emit: _Emitter | None = None,
    ) -> ChatResponse:
        """Deterministic path used before Claude is wired: always recall, echo evidence."""
        t_tool = time.perf_counter()
        r = await self.recall(query=text, device_id=device_id, now=now)
        if stats is not None:
            stats.tool_ms += (time.perf_counter() - t_tool) * 1000
            stats.tool_calls += 1
        answer = r.evidence if r.confidence != "not_found" else NOT_FOUND_ANSWER
        if emit is not None:
            await emit.retrieval(r)
            await emit.delta(answer)
        return ChatResponse(
            transcript=text,
            answer=answer,
            cited=r.cited,
            tool_calls=[ToolCallRecord(name="recall_memory", args={"query": text, "k": 12}, confidence=r.confidence)],
        )

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self.histories.clear()
            self.cite_histories.clear()
        else:
            self.histories.pop(session_id, None)
            self.cite_histories.pop(session_id, None)

    def system_prompt(self, now: float | None = None) -> str:
        return render_system_prompt(now, self.settings.wearer_tz)
