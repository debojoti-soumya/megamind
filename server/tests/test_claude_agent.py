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
    _TrailerFilter,
    parse_textual_tool_call,
    prefetch_reusable,
    recall_first,
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


def _settings(**cfg) -> Settings:
    # Pin the latency flags so RECALL_PREFETCH / READER_SKIP_ON_OCR in the shell env can't change these tests.
    cfg.setdefault("recall_prefetch", True)
    cfg.setdefault("reader_skip_on_ocr", True)
    cfg.setdefault("recall_first", False)  # the scripted tests below drive Claude's first (tool-picking) call
    # these tests assert the pre-caching request shape (clock in the system prompt, plain string question); the
    # cached shape (default on) is covered in test_chat_stream.py
    cfg.setdefault("claude_prompt_cache", False)
    return Settings(_env_file=None, **cfg)


def _agent(recall: RecallMemory, script: ScriptedAnthropic | None, key: str = "k") -> ClaudeAgent:
    settings = _settings(anthropic_api_key=key)
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
    assert first["model"] == "claude-sonnet-5-5"
    assert first["tools"][0]["name"] == "recall_memory"
    assert "in UTC that is" in first["system"] and "America/Los_Angeles" in first["system"]
    assert "Pass short keywords as query" in first["system"]
    assert first["messages"] == [{"role": "user", "content": "where did I leave my keys?"}]
    assert second["messages"][1]["role"] == "assistant"
    # CLAUDE_TOOL_RESULTS_AS_TEXT (default): the tool round goes back as text so Sonnet streams the answer
    assert second["messages"][1]["content"][0] == {"type": "text", "text": 'recall_memory({"query": "keys", "k": 6})'}
    label, tr = second["messages"][2]["content"][:2]
    assert label == {"type": "text", "text": "tool result:"}
    payload = json.loads(tr["text"])
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
    payload = json.loads(script.requests[1]["messages"][2]["content"][1]["text"])
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
    assert Settings(_env_file=None).claude_model == "claude-sonnet-5-5"
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
    return ClaudeAgent(_settings(anthropic_api_key="k", **cfg), spy, client=script.client())


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
    payload = json.loads(script.requests[1]["messages"][2]["content"][1]["text"])
    assert (
        payload["evidence"].startswith("[exact text match") and "Standup 10am / Ship demo Sunday" in payload["evidence"]
    )
    assert stats.extra["reader_skipped"] == 1


def _plan(q: str):
    from datetime import UTC, datetime

    from megamind.retrieval.planner import plan_query

    # 2026-11-07 12:00 in Los Angeles
    return plan_query(q, now=datetime(2026, 11, 7, 20, 0, tzinfo=UTC), tz="America/Los_Angeles")


def test_prefetch_reusable_counts_stems_and_the_planner_rewrite():
    """#70: Claude's paraphrases ("parked car parking spot") searched the same thing as the raw question."""
    q = "where did I park"
    assert not prefetch_reusable({"query": "parked car parking spot"}, q)  # no plan: question words only
    assert prefetch_reusable({"query": "parked car parking spot"}, q, _plan(q))  # rewrite has park/parking/car
    q = "what did I eat for breakfast"
    assert prefetch_reusable({"query": "breakfast food eating"}, q, _plan(q))
    q = "how much was the total?"
    assert not prefetch_reusable({"query": "self-checkout total price receipt"}, q, _plan(q))


def test_prefetch_reusable_accepts_time_bounds_matching_the_planner_window():
    q = "when did I make coffee this morning"
    plan = _plan(q)  # 05:00-12:00 local
    morning = {"query": "coffee", "time_from": "2026-11-07T06:00:00-08:00", "time_to": "2026-11-07T12:00:00-08:00"}
    assert prefetch_reusable(morning, q, plan)
    assert prefetch_reusable({**morning, "time_to": None}, q, plan)  # open end -> planner's end
    yesterday = {**morning, "time_from": "2026-11-06T06:00:00-08:00", "time_to": "2026-11-06T12:00:00-08:00"}
    assert not prefetch_reusable(yesterday, q, plan)
    assert not prefetch_reusable({**morning, "time_from": "not a time"}, q, plan)
    q = "where are my keys"  # no time in the question: Claude's bounds can't match the prefetch
    assert not prefetch_reusable({"query": "keys", "time_from": "2026-11-07T06:00:00-08:00"}, q, _plan(q))


async def test_follow_up_answered_from_history_keeps_its_citations(recall):
    """#70: turn 2 answers from turn 1's tool result with no tool call; its <cites> must still resolve."""
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "keys"})], stop_reason="tool_use"),
            _message([text("On the kitchen counter. <cites>t_keys</cites>")]),
            _message([text("Next to the coffee machine. <cites>t_keys</cites>")]),
        ]
    )
    agent = _spy_agent(SpyRecall(recall), script, recall_prefetch=False)
    first = await agent.chat("s", "where are my keys", "dev1")
    follow = await agent.chat("s", "and what was next to them?", "dev1")
    assert [c.id for c in first.cited] == ["t_keys"]
    assert not follow.tool_calls and [c.id for c in follow.cited] == ["t_keys"]
    other = ScriptedAnthropic([_message([text("Hmm. <cites>t_keys</cites>")])])
    agent._client = other.client()
    assert not (await agent.chat("other-session", "what was next to them?", "dev1")).cited  # no cross-session leak
    agent.reset("s")
    assert "s" not in agent.cite_histories


def test_recall_first_only_for_standalone_recall_questions():
    for q in ("where did I leave my keys this morning?", "what was the wifi password", "Who did I meet at lunch?"):
        assert recall_first(q, _plan(q)), q
    for q in (
        "and where did I put my keys after that?",  # follow-up: Claude reads the history first
        "how much was it?",
        "thanks!",
        "tell me a joke",
        "what did I just see?",  # nothing left to search after planning
        "",
    ):
        assert not recall_first(q, _plan(q)), q


async def test_recall_first_skips_the_tool_picking_call(recall):
    spy = SpyRecall(recall)
    script = ScriptedAnthropic([_message([text("On the kitchen counter. <cites>t_keys</cites>")])])
    agent = _spy_agent(spy, script, recall_first=True)
    resp, stats = await agent.chat_with_stats("s", "where did I leave my keys?", "dev1", now=NOW)
    assert len(script.requests) == 1 and stats.extra["prefetch"] == "first" and stats.tool_calls == 1
    assert [c["query"] for c in spy.calls] == ["where did I leave my keys?"]
    msgs = script.requests[0]["messages"]
    # the result rides on the question, not a synthetic tool_result: Sonnet streams its answer only then (#138)
    assert [m["role"] for m in msgs] == ["user"] and "tool_result" not in json.dumps(msgs)
    blocks = msgs[0]["content"]
    assert blocks[0] == {"type": "text", "text": "where did I leave my keys?"} and "recall_memory" in blocks[1]["text"]
    assert "t_keys" in json.dumps(blocks[2:])
    assert [c.id for c in resp.cited] == ["t_keys"] and resp.tool_calls[0].args == {
        "query": "where did I leave my keys?"
    }

    # the synthetic pair is valid history for the next turn, and a follow-up goes through Claude's own pick
    script.responses = [_message([text("Still the counter. <cites>t_keys</cites>")])]
    resp, stats = await agent.chat_with_stats("s", "and after that?", "dev1", now=NOW)
    assert stats.extra.get("prefetch") != "first" and [c.id for c in resp.cited] == ["t_keys"]
    assert [m["role"] for m in script.requests[1]["messages"]] == ["user", "assistant", "user"]


async def test_recall_first_lets_claude_re_query(recall):
    spy = SpyRecall(recall)
    script = ScriptedAnthropic(
        [
            _message([tool_use({"query": "mug", "mode": "visual"})], stop_reason="tool_use"),
            _message([text("Blue, on the desk. <cites>t_mug</cites>")]),
        ]
    )
    resp, stats = await _spy_agent(spy, script, recall_first=True).chat_with_stats(
        "s", "what colour was my mug?", "dev1", now=NOW
    )
    assert len(script.requests) == 2 and stats.tool_calls == 2
    assert [c["query"] for c in spy.calls] == ["what colour was my mug?", "mug"]
    assert [c.id for c in resp.cited] == ["t_mug"]


async def test_vision_reader_sends_images_to_claude_and_strips_them_from_history(tmp_path):
    from PIL import Image

    from megamind.agent.reader import ClaudeVisionReader

    store = await _store_with_tokens()
    for tid in ("t_keys", "t_board", "t_mug"):
        Image.new("RGB", (960, 720), (200, 30, 30)).save(f"/tmp/{tid}.jpg", format="JPEG")
    reader = ClaudeVisionReader(_settings(reader_provider="claude"))
    recall = RecallMemory(store, MockTextEmbedder(), None, reader)
    script = ScriptedAnthropic(
        [
            _message([text("On the kitchen counter. <cites>t_keys</cites>")]),
            _message([text("Next to the coffee machine. <cites>t_keys</cites>")]),
        ]
    )
    agent = ClaudeAgent(_settings(anthropic_api_key="k", recall_first=True), recall, client=script.client())
    resp = await agent.chat("s", "where did I leave my keys?", "dev1")
    assert [c.id for c in resp.cited] == ["t_keys"]
    content = script.requests[0]["messages"][0]["content"]
    kinds = [b["type"] for b in content]
    assert kinds[:3] == ["text", "text", "text"] and kinds.count("image") >= 1
    img = next(b for b in content if b["type"] == "image")
    assert img["source"]["media_type"] == "image/jpeg" and img["source"]["data"]
    assert "look at them" in script.requests[0]["system"]

    await agent.chat("s", "and what was next to them?", "dev1")
    history = json.dumps(script.requests[1]["messages"])
    assert '"type": "image"' not in history and "(image omitted)" in history


async def test_tool_call_written_as_text_runs_as_a_tool_round(recall):
    # CLAUDE_TOOL_RESULTS_AS_TEXT shows the model its earlier calls as `recall_memory({...})` text; when it imitates
    # that instead of emitting a tool_use block (seen live), the call runs as a tool round and is never the answer
    script = ScriptedAnthropic(
        [
            _message([text('recall_memory({"query": "keys", "k": 6})')]),
            _message([text("Counter. <cites>t_keys</cites>")]),
        ]
    )
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s1", "where did I leave my keys?", "dev1")

    assert resp.answer == "Counter." and [c.id for c in resp.cited] == ["t_keys"]
    assert stats.model_calls == 2 and stats.tool_calls == 1 and stats.extra["textual_tool_calls"] == 1
    assert resp.tool_calls[0].args == {"query": "keys", "k": 6}
    second = script.requests[1]  # the recovered round goes back like a real one (text-flattened by default)
    assert second["messages"][1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": 'recall_memory({"query": "keys", "k": 6})'}],
    }
    assert second["messages"][2]["content"][0] == {"type": "text", "text": "tool result:"}
    assert json.loads(second["messages"][2]["content"][1]["text"])["cited"][0]["id"] == "t_keys"


async def test_tool_call_written_as_text_at_the_round_cap_is_not_the_answer(recall):
    script = ScriptedAnthropic(
        [
            *[
                _message([tool_use({"query": "x"}, f"toolu_{i}")], stop_reason="tool_use")
                for i in range(MAX_TOOL_ROUNDS)
            ],
            _message([text('recall_memory({"query": "x"})')]),
        ]
    )
    agent = _agent(recall, script)
    resp, stats = await agent.chat_with_stats("s1", "where did I leave my keys?", "dev1")

    assert resp.answer == NOT_FOUND_ANSWER
    assert stats.model_calls == MAX_TOOL_ROUNDS + 1 and stats.extra["textual_tool_calls"] == 1
    # the history keeps the model's text (no dangling tool_use without a result)
    assert agent.histories["s1"][0][-1]["content"] == [{"type": "text", "text": 'recall_memory({"query": "x"})'}]


def test_parse_textual_tool_call():
    names = ["recall_memory", "make_collage"]
    assert parse_textual_tool_call('recall_memory({"query": "keys", "k": 6})', names) == (
        "recall_memory",
        {"query": "keys", "k": 6},
    )
    assert parse_textual_tool_call('\n make_collage({"query": "sunset"}) ', names) == (
        "make_collage",
        {"query": "sunset"},
    )
    assert parse_textual_tool_call('search({"query": "a"})', names) is None  # unknown tool
    assert parse_textual_tool_call("I recall memory of the keys on the counter.", names) is None
    assert parse_textual_tool_call("recall_memory(query)", names) is None  # not JSON args
    assert parse_textual_tool_call('recall_memory({"query": "a"}) and more', names) is None  # not the whole text


def test_trailer_filter_holds_back_a_tool_call_written_as_text():
    f = _TrailerFilter(hold=["recall_memory("])
    assert f.feed("recall_") == "" and f.feed('memory({"query": "keys"})') == "" and f.flush() == ""
    f = _TrailerFilter(hold=["recall_memory("])  # a real answer that merely starts like the tool name
    assert f.feed("recall") == "" and f.feed(" the keys: counter.") == "recall the keys: counter." and f.flush() == ""
    f = _TrailerFilter(hold=["recall_memory("])  # ordinary answers and the cites trailer behave as before
    assert f.feed("Your keys") == "Your keys" and f.feed(" <cites>t</cites>") == " " and f.flush() == ""
    f = _TrailerFilter()  # no hold list (non-agent callers): unchanged behaviour
    assert f.feed('recall_memory({"q": 1})') == 'recall_memory({"q": 1})'
