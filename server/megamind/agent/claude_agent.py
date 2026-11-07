"""Claude orchestrator: conversation + tool use over `recall_memory`.

The loop is the standard Anthropic tool-use loop: ``messages.create`` with
``tools=[RECALL_MEMORY_TOOL]``; every ``tool_use`` block is executed through
:class:`RecallMemory`, its result appended as a ``tool_result``, and the model is
called again until it stops with ``end_turn`` (or we hit ``MAX_TOOL_ROUNDS``).
Per-session history is kept in memory, capped at ``MAX_TURNS`` exchanges.

Without ``ANTHROPIC_API_KEY`` (or when the API fails) the deterministic
``_no_llm_fallback`` path answers straight from the reader's evidence so the
whole system keeps working offline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import anthropic
from ulid import ULID

from megamind.agent.tools import RECALL_MEMORY_TOOL, RecallMemory
from megamind.config import Settings
from megamind.models import ChatResponse, Citation, RecallResult, ToolCallRecord

log = logging.getLogger("megamind.agent")

MAX_TOOL_ROUNDS = 4
MAX_TURNS = 10
MAX_TOKENS = 400
NOT_FOUND_ANSWER = "I don't have a memory of that."
_CITES_RE = re.compile(r"\s*<cites>(.*?)</cites>\s*", re.DOTALL)

SYSTEM_PROMPT = """You are Megamind, a calm assistant living in the wearer's AR headset. Your replies are
spoken aloud by text-to-speech, so answer in one or two short plain sentences: no markdown,
no lists, no token ids, no URLs.

You have one tool, recall_memory, which searches first-person snapshots captured by the
wearer's headset cameras. Call it whenever a question depends on past perception: where an
object is, what some text said, who or what was there, what the wearer was doing earlier.
Current time (UTC): {now}. Convert relative times ("ten minutes ago", "this morning") into
ISO-8601 time_from/time_to for the tool. You may call the tool more than once per question,
for example to widen the time window or rephrase the query when the first result is
not_found or low confidence, but stop after a few attempts.

Ground every claim in the evidence the tool returns. Mention when something was seen using
the relative times in the tool result (for example "about twenty minutes ago"). If the tool
returns not_found, or the evidence does not actually answer the question, say
"I don't have a memory of that." and never guess or invent details. Never claim to identify
people; describe them neutrally.

After your spoken answer, on the same message, append the token ids of the evidence your
answer relies on inside a trailer like <cites>id1,id2</cites>. The trailer is stripped
before speech. Omit it when you cite nothing."""


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


def _tool_result_payload(r: RecallResult, now: float) -> str:
    def cite(c: Citation) -> dict:
        return {
            "id": c.id,
            "when": _relative(c.ts, now),
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
        self, session_id: str, text: str, device_id: str | None
    ) -> tuple[ChatResponse, TurnStats]:
        stats = TurnStats(request_id=str(ULID()))
        t0 = time.perf_counter()
        if not self.enabled:
            stats.path = "fallback"
            resp = await self._no_llm_fallback(text, device_id, stats)
        else:
            try:
                resp = await self._claude_turn(session_id, text, device_id, stats)
            except (anthropic.APIError, anthropic.APIConnectionError) as e:  # network, 4xx/5xx, timeouts
                log.warning("req=%s claude failed (%s); using caption fallback", stats.request_id, e)
                stats.path, stats.error = "fallback", f"{type(e).__name__}: {e}"
                resp = await self._no_llm_fallback(text, device_id, stats)
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

    async def _claude_turn(self, session_id: str, text: str, device_id: str | None, stats: TurnStats) -> ChatResponse:
        assert self._client is not None
        now = time.time()
        turn: list[dict] = [{"role": "user", "content": text}]
        prior = [m for t in self.histories.get(session_id, []) for m in t]
        tool_calls: list[ToolCallRecord] = []
        retrieved: dict[str, Citation] = {}
        reader_cited: list[Citation] = []
        answer_text = ""

        for _round in range(MAX_TOOL_ROUNDS + 1):
            t_llm = time.perf_counter()
            msg = await self._client.messages.create(
                model=self.settings.claude_model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt(now),
                tools=[RECALL_MEMORY_TOOL],
                messages=prior + turn,
            )
            stats.llm_ms += (time.perf_counter() - t_llm) * 1000
            stats.model_calls += 1
            stats.input_tokens += msg.usage.input_tokens
            stats.output_tokens += msg.usage.output_tokens

            assistant_content = [b.model_dump(exclude_none=True) for b in msg.content]
            turn.append({"role": "assistant", "content": assistant_content})
            texts = [b.text for b in msg.content if b.type == "text"]
            if texts:
                answer_text = "\n".join(texts)
            uses = [b for b in msg.content if b.type == "tool_use"]
            if msg.stop_reason != "tool_use" or not uses or _round == MAX_TOOL_ROUNDS:
                break

            results: list[dict] = []
            for use in uses:
                args = dict(use.input) if isinstance(use.input, dict) else {}
                t_tool = time.perf_counter()
                r = await self._run_tool(use.name, args, device_id)
                stats.tool_ms += (time.perf_counter() - t_tool) * 1000
                stats.tool_calls += 1
                tool_calls.append(ToolCallRecord(name=use.name, args=args, confidence=r.confidence))
                for c in r.all_retrieved:
                    retrieved.setdefault(c.id, c)
                for c in r.cited:
                    retrieved.setdefault(c.id, c)
                    if c.id not in {x.id for x in reader_cited}:
                        reader_cited.append(c)
                log.info(
                    "req=%s tool=%s args=%s -> %s (%d cited / %d retrieved)",
                    stats.request_id,
                    use.name,
                    json.dumps(args, ensure_ascii=False),
                    r.confidence,
                    len(r.cited),
                    len(r.all_retrieved),
                )
                results.append({"type": "tool_result", "tool_use_id": use.id, "content": _tool_result_payload(r, now)})
            turn.append({"role": "user", "content": results})

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
        return ChatResponse(transcript=text, answer=answer, cited=cited, tool_calls=tool_calls)

    async def _run_tool(self, name: str, args: dict, device_id: str | None) -> RecallResult:
        if name != "recall_memory":
            return RecallResult(evidence=f"Unknown tool {name}.", confidence="not_found", cited=[], all_retrieved=[])
        allowed = {"query", "time_from", "time_to", "k", "mode"}
        kwargs = {k: v for k, v in args.items() if k in allowed and v is not None}
        kwargs.setdefault("query", "")
        try:
            return await self.recall(device_id=device_id, **kwargs)
        except ValueError as e:  # bad ISO timestamp etc. -> tell the model instead of crashing the turn
            return RecallResult(evidence=f"Tool error: {e}", confidence="not_found", cited=[], all_retrieved=[])

    async def _no_llm_fallback(self, text: str, device_id: str | None, stats: TurnStats | None = None) -> ChatResponse:
        """Deterministic path used before Claude is wired: always recall, echo evidence."""
        t_tool = time.perf_counter()
        r = await self.recall(query=text, device_id=device_id)
        if stats is not None:
            stats.tool_ms += (time.perf_counter() - t_tool) * 1000
            stats.tool_calls += 1
        answer = r.evidence if r.confidence != "not_found" else NOT_FOUND_ANSWER
        return ChatResponse(
            transcript=text,
            answer=answer,
            cited=r.cited,
            tool_calls=[ToolCallRecord(name="recall_memory", args={"query": text, "k": 12}, confidence=r.confidence)],
        )

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self.histories.clear()
        else:
            self.histories.pop(session_id, None)

    @staticmethod
    def system_prompt(now: float | None = None) -> str:
        ts = datetime.fromtimestamp(now, UTC) if now is not None else datetime.now(UTC)
        return SYSTEM_PROMPT.format(now=ts.isoformat(timespec="seconds"))
