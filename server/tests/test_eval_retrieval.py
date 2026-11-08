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


V2 = DEFAULT_DATASET.with_name("retrieval_day_v2.json")


async def test_v2_dataset_is_consistent_and_regenerates_identically():
    import subprocess
    import sys
    from collections import Counter

    ds = load_dataset(V2)
    ids = {t.id for t in ds.tokens}
    assert 300 <= len(ds.tokens) <= 600 and 60 <= len(ds.cases) <= 100
    assert ds.tokens == sorted(ds.tokens, key=lambda t: t.ts) and len(ids) == len(ds.tokens)
    assert any(t.kind == "transcript" for t in ds.tokens)
    for c in ds.cases:
        assert c.groups and all(g and set(g) <= ids for g in c.groups), c.q
    types = Counter(c.qtype for c in ds.cases)
    assert {"locate", "read", "people", "heard", "temporal", "time_range", "recency", "multi_hop"} <= set(types)
    gen = V2.with_name("gen_retrieval_day.py")
    subprocess.run([sys.executable, str(gen), "--check"], check=True)


async def test_v2_hybrid_floor_and_ops_columns():
    ds = load_dataset(V2)
    rep = await evaluate(make_retriever(await build_store(ds.tokens), ds), ds, k=8)
    assert rep.summary()["handoff_recall"] >= 0.55, rep.summary()
    ops = rep.ops()
    assert ops["usd_per_q"] == 0 and ops["failures"] == 0 and ops["p95_ms"] >= ops["p50_ms"]
    assert set(rep.by_type()) >= {"read", "multi_hop"}


def test_group_recall_counts_any_member():
    from megamind.retrieval.eval import CaseResult, EvalCase

    c = EvalCase(q="q", gold=["a", "b", "c"], groups=[["a", "b"], ["c"]])
    r = CaseResult(case=c, ranked_ids=["b", "x", "c"], handoff_ids=["b"], first_rank=1)
    assert r.recall_at(1) == 0.5 and r.recall_at(3) == 1.0 and r.handoff_recall == 0.5


def test_blank_captions_is_seeded_and_keeps_ocr():
    from megamind.retrieval.eval import blank_captions

    ds = load_dataset(V2)
    a, b = blank_captions(ds.tokens, 0.2, seed=7), blank_captions(ds.tokens, 0.2, seed=7)
    assert a == b
    blanked = [t for t, o in zip(a, ds.tokens, strict=True) if o.caption and not t.caption]
    frames = [t for t in ds.tokens if t.kind == "frame" and t.caption]
    assert abs(len(blanked) - 0.2 * len(frames)) <= 1 and all(t.kind == "frame" for t in blanked)
