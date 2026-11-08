"""Hierarchical LLM recall (retrieval/hier.py) with a scripted LLM: notes per closed block, coarse-to-fine
ranking, id validation, time bounds and the hybrid fallback."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from megamind.models import MemoryToken
from megamind.retrieval import build_retriever
from megamind.retrieval.base import RetrievalResult, RetrievedToken, Retriever
from megamind.retrieval.hier import ChatJSON, HierConfig, HierMemory, HierRetriever, window_lines
from megamind.retrieval.planner import plan_query
from megamind.store.memory_store import InMemoryStore

T0 = 1_767_600_000.0  # a block/hour boundary (divisible by 3600)


def tok(i: str, ts: float, caption: str, kind: str = "frame") -> MemoryToken:
    return MemoryToken(id=i, device_id="d", ts=ts, caption=caption, thumb_path="", frame_hash="", kind=kind)


class FakeLLM:
    def __init__(self, stage1=None, stage2=None, fail: bool = False) -> None:
        self.stage1, self.stage2, self.fail = stage1 or {}, stage2 or {}, fail
        self.calls: list[str] = []

    async def __call__(self, system: str, user: str, *, max_tokens: int = 800) -> dict:
        self.calls.append(user)
        if self.fail:
            raise RuntimeError("down")
        if "Pick up to" in user:
            return self.stage1
        if "Rank the token ids" in user:
            return self.stage2
        return {"summary": f"note {len(self.calls)}", "keywords": ["k"]}


class FixedRetriever(Retriever):
    def __init__(self, toks: list[MemoryToken]) -> None:
        self.toks = toks

    async def retrieve(self, query, *, k=12, mode="hybrid", time_from=None, time_to=None, device_id=None, now=None):
        got = [RetrievedToken(t, 1.0 / (i + 1), ["bm25"]) for i, t in enumerate(self.toks[:k])]
        return RetrievalResult(plan=plan_query(query, now=now or datetime.now(UTC)), tokens=got)


async def _store() -> InMemoryStore:
    s = InMemoryStore()
    await s.add_many(
        [
            tok("a1", T0 + 30, "kitchen, a person cuts onions"),
            tok("a2", T0 + 90, 'I heard: "pass me the knife"', kind="transcript"),
            tok("b1", T0 + 700, "living room, keys on the sofa"),
            tok("c1", T0 + 1300, "hallway, shoes by the door"),  # block 2 is still open at `until`
        ]
    )
    return s


async def test_notes_only_for_closed_blocks_and_hours():
    store, llm = await _store(), FakeLLM()
    mem = HierMemory(store, llm, tz="UTC")
    n = await mem.build(T0 + 1250)
    assert n == 2 and set(mem.notes) == {f"block:{T0:.0f}", f"block:{T0 + 600:.0f}"}
    assert await mem.build(T0 + 1250) == 0  # cached
    assert await mem.build(T0 + 3600 + 1) >= 2  # block 2 closes, then the hour rolls up
    assert f"hour:{T0:.0f}" in mem.notes and "pass me the knife" in llm.calls[0]


async def test_notes_cache_round_trips(tmp_path):
    store = await _store()
    path = tmp_path / "notes.json"
    await HierMemory(store, FakeLLM(), cache_path=path, tz="UTC").build(T0 + 1250)
    again = FakeLLM()
    mem = HierMemory(store, again, cache_path=path, tz="UTC")
    assert await mem.build(T0 + 1250) == 0 and not again.calls and len(mem.notes) == 2


async def test_llm_ids_rank_first_and_unknown_or_future_ids_are_dropped():
    store = await _store()
    llm = FakeLLM(stage1={"blocks": ["B1", "B9"]}, stage2={"ids": ["bogus", "c1", "b1", "a1"]})
    base = FixedRetriever([await store.get("a1"), await store.get("a2")])
    r = HierRetriever(store, base, HierMemory(store, llm, tz="UTC"))
    now = datetime.fromtimestamp(T0 + 1250, UTC)
    res = await r.retrieve("where are my keys", k=3, time_to=T0 + 1250, now=now)
    ranked = [x.token.id for x in res.ranked]
    assert ranked[:2] == ["b1", "a1"] and "c1" not in ranked and "bogus" not in ranked
    assert ranked[2] == "a2" and res.ranked[0].reasons == ["llm"]
    stage2 = llm.calls[-1]
    assert "[b1]" in stage2 and "[c1]" not in stage2  # c1 is after `now`: never shown


async def test_llm_failure_falls_back_to_base():
    store = await _store()
    base = FixedRetriever([await store.get("a1"), await store.get("b1")])
    r = HierRetriever(store, base, HierMemory(store, FakeLLM(fail=True), tz="UTC"))
    res = await r.retrieve("keys", k=1, now=datetime.fromtimestamp(T0 + 1250, UTC))
    assert [x.token.id for x in res.ranked] == ["a1"] and "error" in r.last_trace


def test_window_lines_keep_speech_and_subsample_frames():
    toks = [tok(f"f{i}", T0 + i, f"frame {i}") for i in range(20)] + [tok("s", T0 + 5, '"hi"', kind="transcript")]
    lines = window_lines(sorted(toks, key=lambda t: t.ts), HierMemory(None, None, tz="UTC").tz, HierConfig())
    assert len(lines) == HierConfig().lines_per_window + 1 and any(
        ln.startswith("[s]") and "heard" in ln for ln in lines
    )


async def test_chat_json_retries_then_parses():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        if len(seen) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ids": ["x"]}'}}]})

    llm = ChatJSON("k", "gpt-4.1", transport=httpx.MockTransport(handler))
    assert await llm("s", "u") == {"ids": ["x"]}
    assert len(seen) == 2 and seen[0]["response_format"] == {"type": "json_object"} and seen[0]["temperature"] == 0
    await llm.aclose()


@pytest.mark.parametrize("key,cls", [("sk-test", HierRetriever), ("", None)])
def test_build_retriever_selects_hier_only_with_a_key(key, cls):
    s = SimpleNamespace(retriever_backend="hier", openai_api_key=key, hier_cache_path="", wearer_tz="UTC")
    r, _ = build_retriever(s, InMemoryStore(), None)
    assert isinstance(r, cls) if cls else type(r).__name__ == "HybridRetriever"


def test_spread_puts_distinct_moments_first():
    from megamind.retrieval.hier import _spread

    def rt(i, ts):
        return RetrievedToken(
            MemoryToken(id=f"t{i}", device_id="d", ts=ts, caption="x", thumb_path="", frame_hash=""), 1.0, []
        )

    out = _spread([rt(0, 100), rt(1, 110), rt(2, 500), rt(3, 130), rt(4, 900)], 60)
    assert [r.token.id for r in out] == ["t0", "t2", "t4", "t1", "t3"]


def test_interleave_keeps_llm_first_and_dedupes():
    from megamind.retrieval.hier import _interleave

    def rt(i):
        return RetrievedToken(
            MemoryToken(id=f"t{i}", device_id="d", ts=float(i), caption="x", thumb_path="", frame_hash=""), 1.0, []
        )

    out = _interleave([rt(1), rt(2)], [rt(2), rt(3), rt(4), rt(5)], 5)
    assert [r.token.id for r in out] == ["t1", "t2", "t3", "t4", "t5"]


def test_retry_after_honours_headers():
    from megamind.retrieval.hier import _retry_after

    assert _retry_after(httpx.Response(429, headers={"retry-after": "2"}), 0) == 2.25
    assert _retry_after(httpx.Response(429, headers={"x-ratelimit-reset-tokens": "1m3s"}), 0) == 30.0
    assert _retry_after(httpx.Response(429, headers={"x-ratelimit-reset-tokens": "250ms"}), 0) == 0.5
    assert _retry_after(httpx.Response(500), 2) == 6.0
