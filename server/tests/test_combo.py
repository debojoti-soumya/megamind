from __future__ import annotations

from datetime import UTC, datetime

from megamind.config import Settings
from megamind.retrieval import build_retriever
from megamind.retrieval.base import RetrievalResult
from megamind.retrieval.combo import ComboRetriever
from megamind.store.memory_store import InMemoryStore

NOW = datetime(2026, 11, 7, 20, 0, tzinfo=UTC)


class Fake:
    def __init__(self, name: str) -> None:
        self.name, self.calls = name, []

    async def retrieve(self, query, **kw):
        self.calls.append(query)
        return RetrievalResult(plan=None, tokens=[])


def _combo(ucc=True, hier=True):
    h, u, r = Fake("hybrid"), Fake("ucc") if ucc else None, Fake("hier") if hier else None
    return ComboRetriever(h, ucc=u, hier=r, tz="America/Los_Angeles"), h, u, r


async def test_routes_easy_hard_and_temporal_questions():
    combo, h, u, r = _combo()
    for q, want in [
        ("where are my keys", "ucc"),
        ("what did the lobby sign say", "ucc"),
        ("what was I doing at 11am", "hybrid"),
        ("what did I just see?", "hybrid"),
        ("what did I eat yesterday", "hier"),
        ("where did I put my keys after lunch", "hier"),
        ("how many times did I check my phone", "hier"),
    ]:
        await combo.retrieve(q, now=NOW)
        assert combo.last_route == want, q
    assert combo.stats == {"ucc": 2, "hybrid": 2, "hier": 3}
    assert len(u.calls) == 2 and len(r.calls) == 3 and len(h.calls) == 2


async def test_missing_backends_drop_out():
    combo, h, _, r = _combo(ucc=False)
    await combo.retrieve("where are my keys", now=NOW)
    await combo.retrieve("what did I eat yesterday", now=NOW)
    assert [combo.stats["hybrid"], combo.stats["hier"]] == [1, 1] and r.calls == ["what did I eat yesterday"]
    combo, h, u, _ = _combo(hier=False)
    await combo.retrieve("what did I eat yesterday", now=NOW)
    assert combo.last_route == "hybrid"  # no hier: a pure-time question stays on hybrid
    await combo.retrieve("where did I put my keys after lunch", now=NOW)
    assert combo.last_route == "ucc"


def test_build_retriever_combo_needs_a_key():
    store = InMemoryStore()
    retr, idx = build_retriever(Settings(_env_file=None, retriever_backend="combo"), store, None)
    assert not isinstance(retr, ComboRetriever) and idx is None
    retr, idx = build_retriever(Settings(_env_file=None, retriever_backend="combo", openai_api_key="k"), store, None)
    assert isinstance(retr, ComboRetriever) and retr.hier is not None and retr.ucc is None and idx is None
