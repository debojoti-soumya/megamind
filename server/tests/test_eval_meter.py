"""The eval's paid-call meter: cache replay, $ accounting and the cross-run budget ledger."""

import json

import httpx
import pytest

from megamind.retrieval.eval_meter import BudgetExceeded, Meter, MeteredTransport, openai_cost


def _openai(calls: list) -> httpx.MockTransport:
    def handle(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        usage = {"prompt_tokens": 1000, "completion_tokens": 100}
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": usage})

    return httpx.MockTransport(handle)


def test_openai_cost_uses_model_prefix_and_cached_tokens():
    u = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "prompt_tokens_details": {"cached_tokens": 500_000}}
    assert openai_cost("gpt-4.1-2025-04-14", u) == pytest.approx(1.25)
    assert openai_cost("gpt-4.1-mini", {"prompt_tokens": 1_000_000}) == pytest.approx(0.40)
    assert openai_cost("unknown", {"prompt_tokens": 5}) == 0.0


async def test_cache_replays_without_spend_and_ledger_persists(tmp_path):
    calls: list = []
    body = {"model": "gpt-4.1", "messages": []}
    for _ in range(2):  # second process: same cache dir
        m = Meter(tmp_path, replay_latency=False)
        async with httpx.AsyncClient(transport=MeteredTransport(m, "openai", _openai(calls))) as c:
            r = await c.post("https://api.openai.com/v1/chat/completions", json=body)
            assert r.json()["usage"]["completion_tokens"] == 100
        assert m.all.llm_calls == 1 and m.all.usd == pytest.approx(0.0028)
    assert len(calls) == 1 and m.live.calls == 0
    assert json.loads((tmp_path / "ledger.json").read_text())["openai_usd"] == pytest.approx(0.0028)


async def test_budget_blocks_live_calls_and_ucc_counts_only_facet_selects(tmp_path):
    calls: list = []
    m = Meter(tmp_path, max_usd=0.001, max_ucc=1, replay_latency=False)
    async with httpx.AsyncClient(transport=MeteredTransport(m, "openai", _openai(calls))) as c:
        await c.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4.1", "n": 1})
        with pytest.raises(BudgetExceeded):
            await c.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4.1", "n": 2})
    ucc = httpx.MockTransport(lambda req: httpx.Response(200, json={"rows": []}))
    async with httpx.AsyncClient(transport=MeteredTransport(m, "ucc", ucc)) as c:
        await c.post("https://u/v1/corpora/x/queries/select", json={"fields": ["id"]})  # listing: free
        await c.post("https://u/v1/corpora/x/queries/select", json={"columns": {"a": 1}})
        with pytest.raises(BudgetExceeded):
            await c.post("https://u/v1/corpora/x/queries/select", json={"columns": {"b": 1}})
    assert m.live.ucc_queries == 1 and len(calls) == 1
