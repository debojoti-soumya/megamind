#!/usr/bin/env python
"""Evaluate retrieval on the synthetic "day" dataset.

    cd server && . .venv/bin/activate
    python tools/eval_retrieval.py                 # Recall@k / MRR with the mock embedder (offline)
    python tools/eval_retrieval.py --misses        # also print every question whose first gold id is not rank 1
    python tools/eval_retrieval.py --grid          # sweep weight multipliers / RRF c, best configs first
    EMBED_PROVIDER=openai python tools/eval_retrieval.py --real   # use the configured real embedders
    UCC_API_KEY=... python tools/eval_retrieval.py --backend ucc  # sync the day to a UCC corpus, rank with facets
    OPENAI_API_KEY=... python tools/eval_retrieval.py --backend hier   # LLM block notes + two-stage pick/rank
    python tools/eval_retrieval.py --backend all --real --dataset tools/data/retrieval_day_v2.json --by-type
    python tools/eval_retrieval.py --backend all --blank-captions 0.2  # robustness: a poor enricher

Paid calls (OpenAI embeddings / chat, UCC facet selects) go through a metered, on-disk cache in `.eval_cache/`
(responses replayed with their recorded latency, so reruns are free) and a ledger enforcing --max-usd / --max-ucc
across runs. hier's notes are cached in `.eval_cache/hier/`, never in data/. Backends whose key is missing are
skipped with a note, so `--backend all` runs offline with zero paid calls.

The dataset is `tools/data/retrieval_day.json`; edit it to add tokens or questions (gold = token ids).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megamind.retrieval.eval import (
    DEFAULT_DATASET,
    blank_captions,
    build_store,
    evaluate,
    grid_search,
    load_dataset,
    make_retriever,
    tokens_digest,
)
from megamind.retrieval.eval_meter import Meter

CACHE = Path(__file__).resolve().parents[1] / ".eval_cache"
# name -> (needs, description); layered rows are prototypes behind config flags (HIER_GATE_MARGIN, UCC_CANDIDATES)
BACKENDS = {
    "hybrid": ("", "vector+BM25+OCR+temporal RRF"),
    "ucc": ("ucc", "UCC facets over the window, raced vs hybrid (prod 2 s timeout)"),
    "ucc-notimeout": ("ucc", "same answers, 30 s timeout (UCC's own quality)"),
    "ucc-rerank": ("ucc", "UCC reranks hybrid top-N (UCC_CANDIDATES)"),
    "hier": ("openai", "LLM notes + 2-stage pick/rank over hybrid top-30"),
    "hier-gated": ("openai", "hier only when hybrid top-1/top-2 margin < HIER_GATE_MARGIN"),
}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--k", type=int, default=8, help="tokens handed to the reader per question")
    ap.add_argument("--misses", action="store_true", help="print questions not answered at rank 1")
    ap.add_argument("--grid", action="store_true", help="sweep weights / rrf_c and print the top configs")
    ap.add_argument("--real", action="store_true", help="use the configured real embedders (EMBED_PROVIDER)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--backend", choices=[*BACKENDS, "all"], default="hybrid", help="'all': every backend, one table")
    ap.add_argument("--only", help="comma list of backends for --backend all")
    ap.add_argument("--by-type", action="store_true", help="also print R@1 / MRR per question type")
    ap.add_argument("--blank-captions", type=float, default=0.0, help="blank this share of frame captions (seeded)")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (cost control)")
    ap.add_argument("--stride", type=int, default=1, help="every Nth question (a cheaper, type-spread subset)")
    ap.add_argument("--corpus", help="UCC corpus (default: megamind-eval-<tokens hash>, kept 3 days)")
    ap.add_argument("--candidates", type=int, default=30, help="ucc-rerank: hybrid top-N UCC reranks")
    ap.add_argument("--gate-margin", type=float, default=0.25, help="hier-gated: skip the LLM at this top-1 lead")
    ap.add_argument(
        "--hier",
        action="append",
        default=[],
        metavar="LABEL:k=v,...",
        help="extra hier row; keys note/pick/rank (models), effort, pool, hints, gate; e.g. m:pick=gpt-5-mini",
    )
    ap.add_argument("--max-usd", type=float, default=3.0, help="OpenAI budget across all runs (ledger)")
    ap.add_argument("--max-ucc", type=int, default=300, help="UCC facet-query budget across all runs (ledger)")
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--live-latency", action="store_true", help="don't replay recorded latency on cache hits")
    args = ap.parse_args()

    from megamind.config import get_settings

    s = get_settings()
    meter = Meter(args.cache_dir, max_usd=args.max_usd, max_ucc=args.max_ucc, replay_latency=not args.live_latency)
    text_emb = image_emb = None
    if args.real:
        from megamind.ingest.enrich import build_image_embedder, build_text_embedder
        from megamind.store.embeddings import OpenAITextEmbedder

        if s.embed_provider == "openai" and s.openai_api_key:
            text_emb = OpenAITextEmbedder(
                s.openai_api_key, s.embed_model, int(s.embed_dim), transport=meter.transport("openai")
            )
        else:
            text_emb = build_text_embedder(s)
        image_emb = build_image_embedder(s)
        print(f"embedders: {s.embed_provider}:{s.embed_model} image={type(image_emb).__name__}", file=sys.stderr)

    ds = load_dataset(args.dataset)
    if args.blank_captions:
        ds.tokens = blank_captions(ds.tokens, args.blank_captions, seed=7)
    ds.cases = ds.cases[:: max(1, args.stride)]
    if args.limit:
        ds.cases = ds.cases[: args.limit]
    store = await build_store(ds.tokens, text_emb, image_emb)
    setup = meter.all.copy()

    if args.grid:
        rows = await grid_search(store, ds, k=args.k, text_embedder=text_emb, image_embedder=image_emb)
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            print(f"{'params':<55} {'mrr':>6} {'r@1':>6} {'r@3':>6} {'r@5':>6} {'handoff':>8}")
            for params, summ in rows[:12]:
                p = " ".join(f"{k}={v:g}" for k, v in params.items())
                print(
                    f"{p:<55} {summ['mrr']:>6.3f} {summ['recall@1']:>6.3f} {summ['recall@3']:>6.3f} "
                    f"{summ['recall@5']:>6.3f} {summ['handoff_recall']:>8.3f}"
                )
        return 0

    names = list(BACKENDS) if args.backend == "all" else [args.backend]
    if args.only:
        names = [n for n in args.only.split(",") if n in BACKENDS]
    if args.backend == "hybrid" and args.hier and not args.only:
        names = []
    specs = dict(parse_hier_spec(x) for x in args.hier)
    names += list(specs)
    have = {"": True, "ucc": bool(s.ucc_api_key), "openai": bool(getattr(s, "hier_api_key", "") or s.openai_api_key)}
    reports, notes = {}, {}
    for name in names:
        need = BACKENDS[name][0] if name in BACKENDS else "openai"
        if not have[need]:
            print(f"skip {name}: no {need.upper()} key", file=sys.stderr)
            continue
        hybrid = make_retriever(store, ds, text_embedder=text_emb, image_embedder=image_emb)
        retriever, notes[name] = await build_backend(name, args, s, store, ds, hybrid, meter, specs.get(name))
        reports[name] = await evaluate(retriever, ds, k=args.k, meter=meter)
        stats = getattr(retriever, "stats", None)
        if stats:
            print(f"{name} stats: {stats}", file=sys.stderr)

    if args.json:
        print(json.dumps(as_json(reports, notes, meter, setup, ds), indent=1))
        return 0
    cap = f", {args.blank_captions:.0%} captions blanked" if args.blank_captions else ""
    print(f"{len(ds.tokens)} tokens, {len(ds.cases)} questions, k={args.k}{cap}")
    if len(reports) == 1 and args.backend != "all":
        rep = next(iter(reports.values()))
        for name, v in rep.summary().items():
            print(f"  {name:<16}{v:.3f}")
    else:
        print_table(reports, args.k)
    if args.by_type:
        print_by_type(reports)
    emb = setup
    print(
        f"\nsetup (embedding the day): {emb.calls} calls ${emb.usd:.4f}; "
        f"hier notes: {', '.join(f'{n} {v}' for n, v in notes.items() if v) or '-'}; "
        f"live this run: {meter.live.calls} calls ${meter.live.usd:.4f} {meter.live.ucc_queries} UCC; "
        f"ledger total: ${meter.ledger.get('openai_usd', 0):.4f} OpenAI, {meter.ledger.get('ucc_queries', 0)} UCC"
    )
    if args.misses:
        for name, rep in reports.items():
            for r in rep.misses():
                print(f"\n{name} [{r.first_rank or '-'}] {r.case.q!r}  gold={r.case.groups}")
                print(f"     ranked={r.ranked_ids[:6]}  handoff={r.handoff_ids}")
    return 0


def print_table(reports, k: int) -> None:
    head = f"{'backend':<14} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'MRR':>5} {'hand':>5} {'p50ms':>6} {'p95ms':>6}"
    print(head + f" {'calls/q':>7} {'LLM/q':>5} {'UCC/q':>5} {'$/q':>8} {'fail':>4} {'hyb':>4}")
    for name, rep in reports.items():
        sm, op = rep.summary(), rep.ops()
        print(
            f"{name:<14} {sm['recall@1']:>5.3f} {sm['recall@3']:>5.3f} {sm['recall@5']:>5.3f} {sm['mrr']:>5.3f} "
            f"{sm['handoff_recall']:>5.3f} {op['p50_ms']:>6} {op['p95_ms']:>6} {op['calls_per_q']:>7.2f} "
            f"{op['llm_calls_per_q']:>5.2f} {op['ucc_per_q']:>5.2f} {op['usd_per_q']:>8.5f} {op['failures']:>4} "
            f"{op['hybrid_only']:>4}"
        )
    print("(hand = handoff recall@k; fail = paid call errored/timed out -> hybrid; hyb = routed/gated to hybrid)")


def print_by_type(reports) -> None:
    types = sorted({t for rep in reports.values() for t in rep.by_type()})
    print("\nR@1 / MRR by question type")
    print(f"{'type':<11} {'n':>3} " + " ".join(f"{n:>14}" for n in reports))
    for t in types:
        cells, n = [], 0
        for rep in reports.values():
            sub = rep.by_type().get(t)
            n = len(sub.results) if sub else n
            cells.append(f"{sub.summary()['recall@1']:.2f} / {sub.summary()['mrr']:.2f}" if sub else "-")
        print(f"{t:<11} {n:>3} " + " ".join(f"{c:>14}" for c in cells))


def as_json(reports, notes, meter, setup, ds) -> dict:
    out = {"n": len(ds.cases), "tokens": len(ds.tokens), "setup_usd": setup.usd, "ledger": meter.ledger}
    for name, rep in reports.items():
        types = {t: {**r.summary(), "n": len(r.results)} for t, r in rep.by_type().items()}
        out[name] = {"summary": rep.summary(), "ops": rep.ops(), "by_type": types, "notes": notes.get(name)}
    return out


HIER_KEYS = {
    "note": "hier_note_model",
    "pick": "hier_model",
    "rank": "hier_rank_model",
    "effort": "hier_reasoning_effort",
    "pool": "hier_base_pool",
    "hints": "hier_hint_windows",
    "gate": "hier_gate_margin",
}


def parse_hier_spec(spec: str) -> tuple[str, dict]:
    label, _, kv = spec.partition(":")
    over = {}
    for part in filter(None, kv.split(",")):
        k, _, v = part.partition("=")
        if k not in HIER_KEYS:
            raise SystemExit(f"--hier {spec!r}: unknown key {k!r} (one of {', '.join(HIER_KEYS)})")
        over[HIER_KEYS[k]] = int(v) if k in ("pool", "hints") else float(v) if k == "gate" else v
    return f"hier[{label}]", over


async def build_backend(name, args, s, store, ds, hybrid, meter, overrides=None):
    """(retriever, one-time setup note) for one bake-off row."""
    if name == "hybrid":
        return hybrid, None
    if name.startswith("ucc"):
        from megamind.retrieval.ucc import UCCConfig, UCCRetriever

        client = await ucc_client(args, s, store, ds, meter)
        timeout = 30.0 if name == "ucc-notimeout" else s.ucc_timeout_s
        cands = args.candidates if name == "ucc-rerank" else 0
        return UCCRetriever(client, hybrid, tz=ds.tz, config=UCCConfig(timeout_s=timeout, candidates=cands)), None
    from megamind.retrieval.hier import build_hier

    key = getattr(s, "hier_api_key", "") or s.openai_api_key
    gate = args.gate_margin if name == "hier-gated" else 0.0
    settings = s.model_copy(update={"hier_gate_margin": gate, "wearer_tz": ds.tz, **(overrides or {})})
    digest = tokens_digest(ds.tokens)
    note = settings.hier_note_model
    if note.startswith(("gpt-5", "o1", "o3", "o4")):
        note += f"-{settings.hier_reasoning_effort}"
    cache = args.cache_dir / "hier" / f"notes-{digest}-{note}.json"
    hier = build_hier(settings, store, hybrid, key, cache_path=cache, transport=meter.transport("openai"))
    before = meter.all.copy()
    written = await hier.memory.build(ds.now.timestamp())
    d = meter.all.minus(before)
    return hier, f"{written} notes written ({d.llm_calls} LLM calls, ${d.usd:.4f})" if written else "notes cached"


async def ucc_client(args, s, store, ds, meter):
    """Push the (possibly blanked) day to a UCC corpus once per content hash; uploads are not metered."""
    import time

    from megamind.retrieval.ucc import UCCClient, UCCIndexer

    corpus = args.corpus or f"megamind-eval-{tokens_digest(ds.tokens)}"
    client = UCCClient(
        s.ucc_api_key, corpus, base_url=s.ucc_base_url, keep_alive_s=3 * 86400, transport=meter.transport("ucc")
    )
    pushed = await UCCIndexer(client, store, tz=ds.tz).sync_once()
    want = len([t for t in ds.tokens if t.caption.strip()])
    deadline = time.monotonic() + 240
    while len(await client.token_ids()) < want and time.monotonic() < deadline:
        await asyncio.sleep(3)  # uploads become queryable once the engine has read them
    print(f"ucc corpus {client.corpus}: pushed {pushed}, ready {len(await client.token_ids())}/{want}", file=sys.stderr)
    return client


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
