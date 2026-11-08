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
from megamind.retrieval.adapter import StoreAdapter
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


class VecBase(FixedRetriever):
    """A base that exposes an embedder + adapter like HybridRetriever, so vector block ranking runs."""

    def __init__(self, toks, store) -> None:
        super().__init__(toks)
        self.adapter = StoreAdapter(store)
        self.text_embedder = SimpleNamespace(embed_query=self._embed)

    async def _embed(self, text: str) -> list[float]:
        return [0.0, 1.0]

    async def _embed_query(self, field, embed, text):
        return await embed(text)


async def test_vector_ranked_blocks_join_the_llm_picks():
    store = InMemoryStore()
    a1 = tok("a1", T0 + 30, "kitchen, a person cuts onions").model_copy(update={"text_vec": [1.0, 0.0]})
    b1 = tok("b1", T0 + 700, "living room, keys on the sofa").model_copy(update={"text_vec": [0.0, 1.0]})
    await store.add_many([a1, b1])
    llm = FakeLLM(stage1={"blocks": ["B0"]}, stage2={"ids": ["b1"]})
    r = HierRetriever(store, VecBase([a1], store), HierMemory(store, llm, tz="UTC"), config=HierConfig(vec_blocks=1))
    res = await r.retrieve("where are my keys", k=2, now=datetime.fromtimestamp(T0 + 1300, UTC))
    assert r.last_trace["blocks"] == [T0, T0 + 600]  # B0 from the LLM, B1 from the vector ranking
    assert "[b1]" in llm.calls[-1] and res.ranked[0].token.id == "b1"
    assert "B1" in llm.calls[-2].split("search hits fall in blocks:")[1]


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


async def test_large_note_backlog_builds_in_background_not_inline():
    store = await _store()
    llm = FakeLLM(stage1={"blocks": []}, stage2={"ids": []})
    mem = HierMemory(store, llm, tz="UTC")
    r = HierRetriever(store, FixedRetriever([await store.get("a1")]), mem, config=HierConfig(inline_build_max=1))
    now = datetime.fromtimestamp(T0 + 1250, UTC)
    assert await mem.pending(now.timestamp()) == 2
    await r.retrieve("keys", k=1, now=now)
    assert r._bg is not None
    await r._bg
    assert await mem.pending(now.timestamp()) == 0 and len(mem.notes) == 2
    r.cfg.inline_build_max = 3
    await store.add_many([tok("c2", T0 + 1400, "hallway, a coat")])
    await r.retrieve("coat", k=1, now=datetime.fromtimestamp(T0 + 1900, UTC))
    assert f"block:{T0 + 1200:.0f}" in mem.notes  # a small backlog is written inline


async def test_gate_margin_skips_the_llm_when_the_hybrid_is_confident():
    from megamind.retrieval.hier import top_margin

    store = await _store()
    base = FixedRetriever([await store.get("a1"), await store.get("b1")])  # scores 1.0, 0.5 -> margin 0.5
    assert top_margin(await base.retrieve("keys")) == pytest.approx(0.5)
    llm = FakeLLM(stage2={"ids": ["b1"]})
    now = datetime.fromtimestamp(T0 + 1250, UTC)
    cfg = HierConfig(gate_margin=0.4, gate_min_hits=2)
    gated = HierRetriever(store, base, HierMemory(store, llm, tz="UTC"), llm, config=cfg)
    res = await gated.retrieve("keys", k=1, now=now)
    assert [x.token.id for x in res.ranked] == ["a1"]
    assert gated.stats == {"llm": 0, "gated": 1, "direct": 0, "errors": 0}
    # two hits are too few to trust the lead under the default gate_min_hits: the LLM runs
    llm2 = FakeLLM(stage1={"blocks": ["B0"]}, stage2={"ids": ["a1"]})
    few = HierRetriever(store, base, HierMemory(store, llm2, tz="UTC"), llm2, config=HierConfig(gate_margin=0.4))
    await few.retrieve("keys", k=1, now=now)
    assert few.stats["gated"] == 0 and few.stats["llm"] == 1
    assert not any("Rank the token ids" in c for c in llm.calls)
    open_ = HierRetriever(store, base, HierMemory(store, llm, tz="UTC"), llm, config=HierConfig(gate_margin=0.6))
    await open_.retrieve("keys", k=1, now=now)
    assert open_.stats["gated"] == 0


async def test_llm_failure_is_counted():
    store = await _store()
    base = FixedRetriever([await store.get("a1")])
    r = HierRetriever(store, base, HierMemory(store, FakeLLM(fail=True), tz="UTC"))
    await r.retrieve("keys", k=1, now=datetime.fromtimestamp(T0 + 1250, UTC))
    assert r.stats["errors"] == 1


def test_build_hier_splits_pick_and_rank_models_and_passes_reasoning_effort():
    from megamind.retrieval.hier import build_hier

    s = SimpleNamespace(
        hier_model="gpt-5.4-nano",
        hier_rank_model="gpt-4.1",
        hier_reasoning_effort="none",
        hier_base_pool=60,
        hier_cache_path="",
        wearer_tz="UTC",
    )
    r = build_hier(s, InMemoryStore(), FixedRetriever([]), "k")
    assert (r.llm.model, r.rank_llm.model, r.cfg.base_pool) == ("gpt-5.4-nano", "gpt-4.1", 60)
    assert r.llm._body("s", "u", 10)["reasoning_effort"] == "none"
    assert "reasoning_effort" not in r.rank_llm._body("s", "u", 10)
    same = build_hier(
        SimpleNamespace(hier_model="gpt-4.1", hier_cache_path=""), InMemoryStore(), FixedRetriever([]), "k"
    )
    assert same.rank_llm is same.llm and same.llm._body("s", "u", 10)["temperature"] == 0


async def test_stage2_uses_the_rank_model():
    store = await _store()
    base = FixedRetriever([await store.get("a1"), await store.get("b1")])
    pick, rank = FakeLLM(stage1={"blocks": ["B1"]}), FakeLLM(stage2={"ids": ["b1"]})
    r = HierRetriever(store, base, HierMemory(store, pick, tz="UTC"), pick, rank_llm=rank)
    res = await r.retrieve("keys", k=1, now=datetime.fromtimestamp(T0 + 1250, UTC))
    assert [x.token.id for x in res.ranked] == ["b1"]
    assert any("Pick up to" in c for c in pick.calls) and not any("Rank the token ids" in c for c in pick.calls)
    assert any("Rank the token ids" in c for c in rank.calls)


async def _hier_over(llm, timeout_s: float):
    store = await _store()
    base = FixedRetriever([await store.get("a1"), await store.get("b1")])
    memory = HierMemory(store, FakeLLM(), tz="UTC")
    return HierRetriever(store, base, memory, llm, config=HierConfig(timeout_s=timeout_s))


async def test_hung_llm_server_falls_back_to_hybrid_within_timeout():
    import asyncio
    import time

    held = []

    async def never_answer(reader, writer):
        held.append(writer)  # accept, read the request, never respond
        await reader.read(65536)

    server = await asyncio.start_server(never_answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    llm = ChatJSON("k", "gpt-4.1-mini", base_url=f"http://127.0.0.1:{port}/v1", timeout_s=60, retries=2)
    r = await _hier_over(llm, timeout_s=0.3)
    t0 = time.perf_counter()
    res = await r.retrieve("keys", k=1, now=datetime.fromtimestamp(T0 + 1250, UTC))
    assert time.perf_counter() - t0 < 2.0
    assert [x.token.id for x in res.ranked] == ["a1"] and held
    assert r.stats["errors"] == 1 and "took > 0.3 s" in r.last_trace["error"]
    await llm.aclose()
    for w in held:
        w.close()
    server.close()


async def test_llm_500s_fall_back_to_hybrid_within_timeout():
    import time

    llm = ChatJSON("k", "gpt-4.1-mini", transport=httpx.MockTransport(lambda req: httpx.Response(500)), retries=2)
    r = await _hier_over(llm, timeout_s=0.3)
    t0 = time.perf_counter()
    res = await r.retrieve("keys", k=1, now=datetime.fromtimestamp(T0 + 1250, UTC))
    assert time.perf_counter() - t0 < 2.0 and [x.token.id for x in res.ranked] == ["a1"] and r.stats["errors"] == 1
    await llm.aclose()


def test_build_hier_recall_calls_fail_fast_and_notes_stay_patient():
    from megamind.retrieval.hier import build_hier

    s = SimpleNamespace(hier_cache_path="", wearer_tz="UTC")
    r = build_hier(s, None, None, "sk-test")
    assert r.cfg.timeout_s == 8.0
    assert (r.llm.retries, r.llm._client.timeout.read) == (2, 4.0)
    assert (r.rank_llm.retries, r.rank_llm._client.timeout.read) == (2, 4.0)
    assert (r.memory.llm.retries, r.memory.llm._client.timeout.read) == (6, 120.0)
    tight = build_hier(
        SimpleNamespace(**vars(s), hier_timeout_s=3, hier_call_timeout_s=1, hier_retries=1), None, None, "k"
    )
    assert (tight.cfg.timeout_s, tight.llm.retries, tight.llm._client.timeout.read) == (3.0, 1, 1.0)


async def test_block_ids_count_from_time_from():
    store = InMemoryStore()
    await store.add_many([tok(f"t{i}", T0 + 600 * i + 30, f"scene {i}") for i in range(12)])
    llm = FakeLLM(stage1={"blocks": ["B0"]}, stage2={"ids": ["t7"]})
    mem = HierMemory(store, llm, tz="UTC")
    await mem.build(T0 + 600 * 12 + 3600)
    base = FixedRetriever([await store.get("t11")])
    r = HierRetriever(store, base, mem, llm, config=HierConfig(vec_blocks=0), auto_build=False)
    now = datetime.fromtimestamp(T0 + 600 * 12 + 3600, UTC)
    res = await r.retrieve("scene", k=2, time_from=T0 + 600 * 7, now=now)
    assert r.last_trace["blocks"][0] == T0 + 600 * 7  # the B0 the prompt showed: the first block in range, not T0
    assert res.ranked[0].token.id == "t7"
    assert "[B0] " + mem._fmt(T0 + 600 * 7) in llm.calls[-2]


class DirectLLM(FakeLLM):
    """Scripted stage 1 for direct mode: `direct` is the one-call answer, `stage1`/`stage2` the fallback path."""

    def __init__(self, direct, **kw) -> None:
        super().__init__(**kw)
        self.direct = direct

    async def __call__(self, system: str, user: str, *, max_tokens: int = 800) -> dict:
        if "return no blocks" in user:
            self.calls.append(user)
            return self.direct
        return await super().__call__(system, user, max_tokens=max_tokens)


async def _direct(llm):
    store = await _store()
    base = FixedRetriever([await store.get("a1"), await store.get("b1")])
    mem = HierMemory(store, FakeLLM(), tz="UTC")
    r = HierRetriever(store, base, mem, llm, config=HierConfig(direct=True))
    return r, await r.retrieve("keys", k=2, now=datetime.fromtimestamp(T0 + 1250, UTC))


async def test_direct_mode_answers_from_the_hint_moments_in_one_call():
    llm = DirectLLM({"ids": ["b1", "zz"], "blocks": []})
    r, res = await _direct(llm)
    assert [x.token.id for x in res.ranked][:1] == ["b1"]
    assert len(llm.calls) == 1 and "[b1]" in llm.calls[0] and "[a1]" in llm.calls[0]
    assert "Rank the token ids" not in llm.calls[0]
    assert r.stats == {"llm": 1, "gated": 0, "direct": 1, "errors": 0} and r.last_trace["direct"] is True


async def test_direct_mode_falls_through_to_stage_2_when_blocks_are_requested():
    llm = DirectLLM({"ids": [], "blocks": ["B1"]}, stage2={"ids": ["b1"]})
    r, res = await _direct(llm)
    assert [x.token.id for x in res.ranked][:1] == ["b1"]
    assert len(llm.calls) == 2 and "Rank the token ids" in llm.calls[1]
    assert r.stats["direct"] == 0 and r.last_trace["blocks"] == [T0 + 600] and "stage2" in r.last_trace


async def test_direct_mode_ignores_ids_when_blocks_are_also_requested():
    # ids + blocks together means "read more": the ids are not trusted as the final answer
    llm = DirectLLM({"ids": ["a1"], "blocks": ["B1"]}, stage2={"ids": ["b1"]})
    r, res = await _direct(llm)
    assert [x.token.id for x in res.ranked][:1] == ["b1"] and r.stats["direct"] == 0


async def _week_store() -> InMemoryStore:
    # three hours of blocks (18 blocks); every closed hour gets a note
    s = InMemoryStore()
    await s.add_many([tok(f"t{i}", T0 + 600 * i + 30, f"scene {i}") for i in range(18)])
    return s


async def test_pruned_outline_lists_blocks_of_expanded_hours_and_tags_the_rest():
    store = await _week_store()
    mem = HierMemory(store, FakeLLM(), tz="UTC")
    until = T0 + 3 * 3600
    await mem.build(until)
    hours = mem.hours(until)
    assert [h["start"] for h in hours] == [T0, T0 + 3600, T0 + 7200]
    full = mem.outline(until)
    assert "[B0]" in full and "[B17]" in full and "[H" not in full
    pruned = mem.outline(until, expand={T0 + 3600})
    assert "[H0] HOUR" in pruned and "[H2] HOUR" in pruned and "[H1]" not in pruned
    assert "[B6]" in pruned and "[B11]" in pruned and "[B0]" not in pruned and "[B12]" not in pruned
    assert len(pruned) < len(full)


async def test_outline_hours_lets_stage_1_pick_an_hour_and_reads_its_blocks():
    store = await _week_store()
    llm = FakeLLM(stage1={"blocks": ["H0", "B13"]}, stage2={"ids": ["t2"]})
    mem = HierMemory(store, llm, tz="UTC")
    until = T0 + 3 * 3600
    await mem.build(until)
    base = FixedRetriever([await store.get("t13"), await store.get("t14")])
    r = HierRetriever(store, base, mem, llm, config=HierConfig(outline_hours=1, max_blocks=4))
    res = await r.retrieve("scene", k=1, now=datetime.fromtimestamp(until, UTC))
    assert [x.token.id for x in res.ranked][:1] == ["t2"]
    prompt = next(c for c in llm.calls if "Pick up to" in c)
    assert "[H0] HOUR" in prompt and "[H1] HOUR" in prompt and "[H2]" not in prompt  # hits are in hour 2
    assert "[B13]" in prompt and "[B0]" not in prompt
    # H0 expands to its blocks up to max_blocks; B13 no longer fits
    assert r.last_trace["blocks"] == [T0 + 600 * i for i in range(4)]
    assert r.last_trace["outline_chars"] == len(prompt.split("\n\nNow:")[0].split("blocks):\n", 1)[1])
    assert r.last_trace["stage1_ms"] >= 0 and r.last_trace["stage2_ms"] >= 0
