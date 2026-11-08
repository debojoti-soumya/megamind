"""Latency of `HybridRetriever.retrieve` (the retrieval half of `recall_memory`) at demo scale, per stage.

    python tools/bench_retrieval.py                       # 3k tokens / 1 wearer and 12k / 4 wearers, memory + lancedb
    python tools/bench_retrieval.py --n 3000 --devices 1 --backends lancedb --reps 30

Synthetic day from `megamind.store.bench.synth_tokens` (one kept frame every ~8 s per device), hashed 256-d text
vectors (the mock embedder's size), no image tower (`IMAGE_EMBED_PROVIDER=none`, the demo default). Each adapter
primitive and the query embedder are wrapped with timers, so the table shows where a retrieve() spends its time.
No network, no keys.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from megamind.ingest.enrich import MockTextEmbedder
from megamind.retrieval import HybridRetriever
from megamind.store.bench import synth_tokens
from megamind.store.memory_store import InMemoryStore

QUERIES = {
    "content": ["where did I leave my keys", "laptop on the desk", "passport", "mug on the kitchen counter"],
    "ocr": ["what was the wifi password", "room 204", "what did the menu say"],
    "temporal": ["what was I doing at 11", "what was I doing an hour ago"],
    "recent": ["what did I just see?", "what was I looking at?"],
}
STAGES = ("embed", "keyword_search", "vector_search", "ocr_search", "get_many", "neighbors", "range")


def _instrument(retr: HybridRetriever, acc: dict[str, float]) -> None:
    def wrap(obj, name: str, key: str) -> None:
        fn = getattr(obj, name)

        async def timed(*a, **kw):
            t = time.perf_counter()
            try:
                return await fn(*a, **kw)
            finally:
                acc[key] += time.perf_counter() - t

        setattr(obj, name, timed)

    for s in STAGES[1:]:
        wrap(retr.adapter, s, s)
    if retr.text_embedder is not None:
        wrap(retr.text_embedder, "embed", "embed")


async def _store(backend: str, n: int, devices: int, tmp: Path):
    toks = synth_tokens(n, text_dim=MockTextEmbedder.dim, image_dim=8, devices=devices)
    emb = MockTextEmbedder()
    for t in toks:  # hashed caption vectors so the text leg ranks something meaningful
        t.text_vec = await emb.embed(f"{t.caption} {t.ocr_text}")
        t.image_vec = None
    if backend == "memory":
        st = InMemoryStore()
    else:
        from megamind.store.lancedb_store import LanceDBStore

        st = LanceDBStore(tmp / f"bench-{n}-{devices}")
    for i in range(0, len(toks), 500):
        await st.add_many(toks[i : i + 500])
    if hasattr(st, "optimize"):
        await st.optimize()
    return st, toks


async def bench(backend: str, n: int, devices: int, reps: int, tmp: Path) -> list[str]:
    st, toks = await _store(backend, n, devices, tmp)
    retr = HybridRetriever(st, MockTextEmbedder(), None, tz="America/Los_Angeles")
    acc: dict[str, float] = defaultdict(float)
    _instrument(retr, acc)
    now = datetime.fromtimestamp(toks[-1].ts + 30, UTC)
    rows = [f"\n## {backend}: {n} tokens, {devices} device(s), {reps} reps per query type"]
    rows.append(f"{'type':<9} {'p50 ms':>7} {'p95 ms':>7}  " + " ".join(f"{s:>14}" for s in STAGES))
    for qtype, qs in QUERIES.items():
        await retr.retrieve(qs[0], k=12, device_id="quest-0", now=now)  # warm-up
        acc.clear()
        samples = []
        for r in range(reps):
            t = time.perf_counter()
            await retr.retrieve(qs[r % len(qs)], k=12, device_id="quest-0", now=now)
            samples.append((time.perf_counter() - t) * 1000)
        samples.sort()
        p95 = samples[min(len(samples) - 1, round(0.95 * (len(samples) - 1)))]
        stages = " ".join(f"{acc[s] * 1000 / reps:>14.1f}" for s in STAGES)
        rows.append(f"{qtype:<9} {statistics.median(samples):>7.1f} {p95:>7.1f}  {stages}")
    return rows


async def run(args) -> None:
    sizes = [(args.n, args.devices)] if args.n else [(3000, 1), (12000, 4)]
    with tempfile.TemporaryDirectory() as d:
        for backend in args.backends:
            for n, dev in sizes:
                print("\n".join(await bench(backend, n, dev, args.reps, Path(d))), flush=True)
    print("\n(stage columns: mean ms per retrieve spent in that primitive; fusion legs overlap)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--backends", nargs="+", default=["memory", "lancedb"])
    ap.add_argument("--reps", type=int, default=20)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
