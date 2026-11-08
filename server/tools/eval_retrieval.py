#!/usr/bin/env python
"""Evaluate retrieval on the synthetic "day" dataset.

    cd server && . .venv/bin/activate
    python tools/eval_retrieval.py                 # Recall@k / MRR with the mock embedder (offline)
    python tools/eval_retrieval.py --misses        # also print every question whose first gold id is not rank 1
    python tools/eval_retrieval.py --grid          # sweep weight multipliers / RRF c, best configs first
    EMBED_PROVIDER=openai python tools/eval_retrieval.py --real   # use the configured real embedders
    UCC_API_KEY=... python tools/eval_retrieval.py --backend ucc  # sync the day to a UCC corpus, rank with facets

The dataset is `tools/data/retrieval_day.json`; edit it to add tokens or questions (gold = token ids).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megamind.retrieval.eval import DEFAULT_DATASET, build_store, evaluate, grid_search, load_dataset, make_retriever


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--k", type=int, default=8, help="tokens handed to the reader per question")
    ap.add_argument("--misses", action="store_true", help="print questions not answered at rank 1")
    ap.add_argument("--grid", action="store_true", help="sweep weights / rrf_c and print the top configs")
    ap.add_argument("--real", action="store_true", help="use build_text_embedder/build_image_embedder from settings")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--backend", choices=["hybrid", "ucc"], default="hybrid")
    ap.add_argument("--corpus", help="UCC corpus (default: megamind-eval-<dataset hash>, kept 3 days)")
    ap.add_argument("--candidates", type=int, default=0, help="UCC: rerank the hybrid's top-N (0 = whole window)")
    args = ap.parse_args()

    text_emb = image_emb = None
    if args.real:
        from megamind.config import get_settings
        from megamind.ingest.enrich import build_text_embedder
        from megamind.store.embeddings import build_image_embedder

        s = get_settings()
        text_emb, image_emb = build_text_embedder(s), build_image_embedder(s)
        print(f"embedders: {s.embed_provider}:{s.embed_model} image={type(image_emb).__name__}", file=sys.stderr)

    ds = load_dataset(args.dataset)
    store = await build_store(ds.tokens, text_emb, image_emb)

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

    retriever = make_retriever(store, ds, text_embedder=text_emb, image_embedder=image_emb)
    if args.backend == "ucc":
        retriever = await ucc_retriever(args, store, ds, retriever)
    rep = await evaluate(retriever, ds, k=args.k)
    if args.backend == "ucc":
        print(f"ucc stats: {retriever.stats}", file=sys.stderr)
    if args.json:
        print(json.dumps({"summary": rep.summary(), "n": len(rep.results)}))
    else:
        print(f"{len(ds.tokens)} tokens, {len(ds.cases)} questions, k={args.k}")
        for name, v in rep.summary().items():
            print(f"  {name:<16}{v:.3f}")
    if args.misses:
        for r in rep.misses():
            print(f"\n[{r.first_rank or '-'}] {r.case.q!r}  gold={r.case.gold}")
            print(f"     ranked={r.ranked_ids[:6]}  handoff={r.handoff_ids}")
    return 0


async def ucc_retriever(args, store, ds, hybrid):
    """Push the synthetic day to a UCC corpus (once per dataset version) and wrap `hybrid` with UCC ranking."""
    import hashlib
    import time

    from megamind.config import get_settings
    from megamind.retrieval.ucc import UCCClient, UCCConfig, UCCIndexer, UCCRetriever

    s = get_settings()
    if not s.ucc_api_key:
        raise SystemExit("--backend ucc needs UCC_API_KEY")
    digest = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()[:10]
    client = UCCClient(
        s.ucc_api_key, args.corpus or f"megamind-eval-{digest}", base_url=s.ucc_base_url, keep_alive_s=3 * 86400
    )
    pushed = await UCCIndexer(client, store, tz=ds.tz).sync_once()
    deadline = time.monotonic() + 180
    while len(await client.token_ids()) < len([t for t in ds.tokens if t.caption]) and time.monotonic() < deadline:
        await asyncio.sleep(3)  # uploads become queryable once the engine has read them
    print(f"ucc corpus {client.corpus}: pushed {pushed}, ready {len(await client.token_ids())}", file=sys.stderr)
    return UCCRetriever(client, hybrid, tz=ds.tz, config=UCCConfig(timeout_s=30.0, candidates=args.candidates))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
