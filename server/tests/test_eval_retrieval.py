"""The committed synthetic day must keep retrieval quality above a floor (guards against weight regressions)."""

from megamind.retrieval.eval import DEFAULT_DATASET, build_store, evaluate, load_dataset, make_retriever


async def test_dataset_is_consistent():
    ds = load_dataset(DEFAULT_DATASET)
    ids = {t.id for t in ds.tokens}
    assert len(ids) == len(ds.tokens)
    for c in ds.cases:
        assert c.gold and set(c.gold) <= ids, c.q
    assert ds.tokens == sorted(ds.tokens, key=lambda t: t.ts)


async def test_quality_floor_with_mock_embedder():
    ds = load_dataset(DEFAULT_DATASET)
    store = await build_store(ds.tokens)
    rep = await evaluate(make_retriever(store, ds), ds, k=8)
    s = rep.summary()
    assert s["mrr"] >= 0.75, s
    assert s["recall@3"] >= 0.8, s
    assert s["handoff_recall"] >= 0.85, s
