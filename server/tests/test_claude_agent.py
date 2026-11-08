"""Claude orchestrator tests with recorded Anthropic Messages API responses (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
import time

import anthropic
import httpx2 as httpx
import pytest

from megamind.agent.claude_agent import (
    MAX_TOOL_ROUNDS,
    MAX_TURNS,
    NOT_FOUND_ANSWER,
    ClaudeAgent,
    prefetch_reusable,
)
from megamind.agent.reader import CaptionOnlyReader
from megamind.agent.tools import RecallMemory
from megamind.config import Settings
from megamind.ingest.enrich import MockTextEmbedder
from megamind.models import MemoryToken
from megamind.store.memory_store import InMemoryStore

NOW = 1_731_000_000.0


def _message(content: list[dict], stop_reason: str = "end_turn", in_tok: int = 100, out_tok: int = 20) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def text(s: str) -> dict:
    return {"type": "text", "text": s}


def tool_use(args: dict, tid: str = "toolu_1") -> dict:
    return {"type": "tool_use", "id": tid, "name": "recall_memory", "input": args}


class ScriptedAnthropic:
    """Replays a list of Messages API responses; records every request body."""

    def __init__(self, responses: list[dict | Exception]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if isinstance(nxt, dict) and nxt.get("_status"):
            err = {"type": "error", "error": {"type": "api_error", "message": "x"}}
            return httpx.Response(nxt["_status"], json=err)
        return httpx.Response(200, json=nxt)

    def client(self) -> anthropic.AsyncAnthropic:
        http = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return anthropic.AsyncAnthropic(api_key="test", http_client=http, max_retries=0)


async def _store_with_tokens() -> InMemoryStore:
    store = InMemoryStore()
    emb = MockTextEmbedder()
    specs = [
        ("t_keys", NOW - 20 * 60, "Keys on the kitchen counter next to the coffee machine", ""),
        ("t_board", NOW - 60 * 60, "A whiteboard in a meeting room", "Standup 10am\nShip demo Sunday"),
        ("t_mug", NOW - 5 * 60, "A blue coffee mug on a wooden desk beside a laptop", ""),
    ]
    for tid, ts, cap, ocr in specs:
        await store.add(
            MemoryToken(
                id=tid,
                device_id="dev1",
                ts=ts,
                caption=cap,
                ocr_text=ocr,
                entities=[],
                text_vec=await emb.embed(f"{cap} {ocr}"),
                thumb_path=f"/tmp/{tid}.jpg",
                frame_hash="0" * 16,
            )
        )
    return store


@pytest.fixture
async def recall():
    store = await _store_with_tokens()
    return RecallMemory(store, MockTextEmbedder(), None, CaptionOnlyReader())


def _agent(recall: RecallMemory, script: ScriptedAnthropic | None, key: str = "k") -> ClaudeAgent:
    settings = Settings(anthropic_api_key=key, _env_file=None)
    return ClaudeAgent(settings, recall, client=script.client() if script else None)


async def test_no_key_uses_fallback(recall):
    agent = _agent(recall, None, key="")
    assert not agent.enabled
    resp, stats = await agent.chat_with_stats("s", "keys kitchen counter", "dev1")
    assert stats.path == "fallback"
    assert resp.tool_calls[0].name == "recall_memory"
    assert resp.cited and resp.cited[0].id == "t_keys"


async def test_single_tool_call_with_claude_cites(recall):
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "keys", "k": 6})], stop_reason="tool_use", in_tok=500, out_tok=40),
            _message([text("Your keys are on the kitchen counter, about twenty minutes ago. <cites>t_keys</cites>")]),
        ]
    )
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s1", "where did I leave my keys?", "dev1")

    assert resp.answer == "Your keys are on the kitchen counter, about twenty minutes ago."
    assert "<cites>" not in resp.answer
    assert [c.id for c in resp.cited] == ["t_keys"]
    assert resp.tool_calls == [resp.tool_calls[0]] and resp.tool_calls[0].args == {"query": "keys", "k": 6}
    assert resp.tool_calls[0].confidence == "low"  # CaptionOnlyReader
    assert stats.model_calls == 2 and stats.tool_calls == 1 and stats.cites_from == "claude"
    assert stats.input_tokens == 600 and stats.output_tokens == 60

    # request shape: system prompt, tool schema, tool_result fed back
    first, second = script.requests
    assert first["model"] == "claude-sonnet-4-6"
    assert first["tools"][0]["name"] == "recall_memory"
    assert "in UTC that is" in first["system"] and "America/Los_Angeles" in first["system"]
    assert "Pass short keywords as query" in first["system"]
    assert first["messages"] == [{"role": "user", "content": "where did I leave my keys?"}]
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][1]["content"][0]["type"] == "tool_use"
    tr = second["messages"][2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "toolu_1"
    payload = json.loads(tr["content"])
    assert payload["confidence"] == "low"
    assert payload["cited"][0]["id"] == "t_keys"
    assert "ago" in payload["cited"][0]["when"] and payload["cited"][0]["time_local"]


async def test_multi_call_widens_window_and_history_is_kept(recall):
    script = ScriptedAnthropic(
        [
            _message(
                [tool_use({"query": "whiteboard", "time_from": "2030-01-01T00:00:00+00:00", "k": 4}, "toolu_a")],
                stop_reason="tool_use",
            ),
            _message([tool_use({"query": "whiteboard", "time_from": None, "k": 12}, "toolu_b")], "tool_use"),
            _message([text("The whiteboard said standup at ten and ship the demo Sunday. <cites>t_board</cites>")]),
            _message([text("You looked at it about an hour ago.")]),
        ]
    )
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s2", "what did the whiteboard say?", "dev1")
    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0].confidence == "not_found"  # window in the future -> nothing
    assert resp.tool_calls[1].confidence == "low"
    assert [c.id for c in resp.cited] == ["t_board"]
    assert stats.model_calls == 3

    # second turn: history carried into the request, tool blocks included
    resp2, _ = await agent.chat_with_stats("s2", "when was that?", "dev1")
    assert resp2.answer == "You looked at it about an hour ago."
    assert resp2.cited == []  # no trailer, no tool call -> nothing to cite
    msgs = script.requests[-1]["messages"]
    assert msgs[0]["content"] == "what did the whiteboard say?"
    assert msgs[-1] == {"role": "user", "content": "when was that?"}
    assert any(m["role"] == "user" and isinstance(m["content"], list) for m in msgs)  # tool_result kept


async def test_not_found_answer_drops_citations(recall):
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "bicycle"})], stop_reason="tool_use"),
            _message([text(f"{NOT_FOUND_ANSWER} <cites>t_keys</cites>")]),
        ]
    )
    agent = _agent(recall, script)
    resp = await agent.chat("s3", "where is my bicycle?", "dev1")
    assert resp.answer == NOT_FOUND_ANSWER
    assert resp.cited == []


async def test_reader_citations_used_when_claude_omits_trailer(recall):
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "mug"})], stop_reason="tool_use"),
            _message([text("Your mug is on the desk beside the laptop.")]),
        ]
    )
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s4", "where is my mug?", "dev1")
    assert stats.cites_from == "reader"
    assert resp.cited and resp.cited[0].id == "t_mug"


async def test_tool_round_cap(recall):
    script = ScriptedAnthropic(
        [_message([tool_use({"query": "x"}, f"toolu_{i}")], stop_reason="tool_use") for i in range(MAX_TOOL_ROUNDS + 1)]
    )
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s5", "loop forever", "dev1")
    assert stats.model_calls == MAX_TOOL_ROUNDS + 1
    assert stats.tool_calls == MAX_TOOL_ROUNDS
    assert resp.answer == NOT_FOUND_ANSWER


async def test_bad_tool_args_are_reported_not_raised(recall):
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "keys", "time_from": "yesterday-ish"})], stop_reason="tool_use"),
            _message([text(NOT_FOUND_ANSWER)]),
        ]
    )
    agent = _agent(recall, script)
    resp = await agent.chat("s6", "keys?", "dev1")
    assert resp.tool_calls[0].confidence == "not_found"
    payload = json.loads(script.requests[1]["messages"][2]["content"][0]["content"])
    assert payload["evidence"].startswith("Tool error")


async def test_api_error_degrades_to_fallback(recall):
    script = ScriptedAnthropic([{"_status": 500}])
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s7", "keys kitchen counter", "dev1")
    assert stats.path == "fallback" and stats.error
    assert resp.cited and resp.cited[0].id == "t_keys"


async def test_connection_error_degrades_to_fallback(recall):
    script = ScriptedAnthropic([httpx.ConnectError("boom")])
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s8", "keys kitchen counter", "dev1")
    assert stats.path == "fallback"
    assert resp.answer


async def test_history_capped(recall):
    script = ScriptedAnthropic([_message([text(f"reply {i}")]) for i in range(MAX_TURNS + 3)])
    agent = _agent(recall, script)
    for i in range(MAX_TURNS + 3):
        await agent.chat("s9", f"q{i}", "dev1")
    assert len(agent.histories["s9"]) == MAX_TURNS
    assert script.requests[-1]["messages"][0]["content"] == "q2"  # 13 chats, last request carries the 9 previous turns
    agent.reset("s9")
    assert "s9" not in agent.histories


def test_claude_model_default_is_not_the_retiring_sonnet_4_5_and_is_overridable(monkeypatch):
    from megamind.config import Settings

    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    assert Settings(_env_file=None).claude_model == "claude-sonnet-4-6"
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    assert Settings(_env_file=None).claude_model == "claude-sonnet-5-5"


def test_render_system_prompt_local_time_and_bad_tz():
    from megamind.agent.claude_agent import render_system_prompt

    p = render_system_prompt(NOW, "America/Los_Angeles")
    assert "Thursday 2024-11-07 09:20:00 PST" in p and "2024-11-07T17:20:00+00:00" in p
    assert "2024-11-07T08:00:00-08:00" in p  # example ISO with offset
    p = render_system_prompt(NOW, "Not/AZone")
    assert "(UTC)" in p


class SpyRecall:
    """Wraps RecallMemory: records every call's kwargs and adds latency (like a ~3 s Sail reader, scaled down)."""

    def __init__(self, inner: RecallMemory, delay: float = 0.0):
        self.inner, self.delay = inner, delay
        self.calls: list[dict] = []
        self.cancelled = 0

    async def __call__(self, **kw):
        self.calls.append(kw)
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return await self.inner(**kw)


def _spy_agent(spy: SpyRecall, script: ScriptedAnthropic, **cfg) -> ClaudeAgent:
    return ClaudeAgent(Settings(anthropic_api_key="k", _env_file=None, **cfg), spy, client=script.client())


def test_prefetch_reusable():
    q = "where did I leave my keys this morning?"
    assert prefetch_reusable({"query": "keys"}, q)
    assert prefetch_reusable({"query": "keys morning", "k": 12, "mode": "hybrid", "time_from": None}, q)
    assert not prefetch_reusable({"query": "keys", "time_from": "2026-11-07T06:00:00-08:00"}, q)
    assert not prefetch_reusable({"query": "keys", "k": 6}, q)
    assert not prefetch_reusable({"query": "keys", "mode": "visual"}, q)
    assert not prefetch_reusable({"query": "car keys hallway table"}, q)  # 1/4 of Claude's words from the question
    assert not prefetch_reusable({"query": ""}, q)


async def test_prefetch_hit_reuses_raw_question_recall(recall):
    spy = SpyRecall(recall)
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "keys"})], stop_reason="tool_use"),
            _message([text("On the kitchen counter. <cites>t_keys</cites>")]),
        ]
    )
    resp, stats = await _spy_agent(spy, script).chat_with_stats("s", "where did I leave my keys?", "dev1", now=NOW)
    assert [c["query"] for c in spy.calls] == ["where did I leave my keys?"]  # one recall: the prefetch
    assert stats.extra["prefetch"] == "hit" and stats.tool_calls == 1
    assert resp.tool_calls[0].args == {"query": "keys"}  # the record shows what Claude asked for
    assert [c.id for c in resp.cited] == ["t_keys"]


async def test_prefetch_miss_requeries_with_claude_args(recall):
    spy = SpyRecall(recall, delay=0.2)
    args = {"query": "keys", "time_from": "2024-11-07T08:00:00+00:00"}
    script = ScriptedAnthropic(
        [_message([tool_use(args)], stop_reason="tool_use"), _message([text("Counter. <cites>t_keys</cites>")])]
    )
    _, stats = await _spy_agent(spy, script).chat_with_stats("s", "where were my keys this morning", "dev1", now=NOW)
    assert stats.extra["prefetch"] == "miss" and spy.cancelled == 1
    assert [c["query"] for c in spy.calls] == ["where were my keys this morning", "keys"]
    assert spy.calls[1]["time_from"] == args["time_from"]


async def test_prefetch_cancelled_when_claude_answers_without_tools(recall):
    spy = SpyRecall(recall, delay=5.0)
    script = ScriptedAnthropic([_message([text("Hi there.")])])
    t0 = time.perf_counter()
    resp, stats = await _spy_agent(spy, script).chat_with_stats("s", "hello", "dev1")
    assert time.perf_counter() - t0 < 1.0 and resp.answer == "Hi there."
    assert stats.extra["prefetch"] == "unused" and spy.cancelled == 1


async def test_prefetch_cancelled_on_api_error_fallback(recall):
    spy = SpyRecall(recall, delay=0.05)
    script = ScriptedAnthropic([{"_status": 500}])
    resp, stats = await _spy_agent(spy, script).chat_with_stats("s", "keys kitchen counter", "dev1")
    assert stats.path == "fallback" and spy.cancelled == 1 and len(spy.calls) == 2
    assert resp.cited and resp.cited[0].id == "t_keys"


async def test_prefetch_disabled(recall):
    spy = SpyRecall(recall)
    script = ScriptedAnthropic(
        [_message([tool_use({"query": "keys"})], stop_reason="tool_use"), _message([text("Counter.")])]
    )
    _, stats = await _spy_agent(spy, script, recall_prefetch=False).chat_with_stats("s", "where are my keys", "dev1")
    assert [c["query"] for c in spy.calls] == ["keys"] and "prefetch" not in stats.extra


async def test_prefetch_overlaps_recall_with_first_claude_call(recall):
    spy = SpyRecall(recall, delay=0.3)
    script = ScriptedAnthropic(
        [_message([tool_use({"query": "keys"})], stop_reason="tool_use"), _message([text("Counter.")])]
    )
    sync_handler = script.handler

    async def slow_handler(request):
        await asyncio.sleep(0.3)  # first Claude call ~ recall latency
        return sync_handler(request)

    script.handler = slow_handler
    _, stats = await _spy_agent(spy, script).chat_with_stats("s", "where are my keys", "dev1")
    assert stats.extra["prefetch"] == "hit"
    assert stats.total_ms < 1100  # serial would be 0.3 + 0.3 + 0.3 = 0.9 s; overlapped ~0.6 s
    assert stats.tool_ms < 150


async def test_ocr_match_skips_reader_on_claude_path(recall):
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "standup demo", "k": 4})], stop_reason="tool_use"),
            _message([text("Standup at 10am. <cites>t_board</cites>")]),
        ]
    )
    _, stats = await _agent(recall, script).chat_with_stats("s", "when is standup and the demo", "dev1", now=NOW)
    payload = json.loads(script.requests[1]["messages"][2]["content"][0]["content"])
    assert (
        payload["evidence"].startswith("[exact text match") and "Standup 10am / Ship demo Sunday" in payload["evidence"]
    )
    assert stats.extra["reader_skipped"] == 1
