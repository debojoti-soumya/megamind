"""Latency benchmark for every `MemoryStore` primitive, in-memory vs LanceDB, on synthetic tokens.

    python -m megamind.store.bench                      # 10k tokens, both backends, 20 queries per primitive
    python -m megamind.store.bench --n 50000 --backends lancedb --queries 50
    python -m megamind.store.bench --n 2000 --json out.json

Tokens simulate one wearer-day: one keyframe every ~8 s from 08:00 on two devices, templated captions/OCR
(BM25 has something to match), unit-norm random `text_vec` (default 1024-d, OpenAI) and `image_vec`
(768-d, SigLIP), ~1 episode per 15 frames. No embedder or network is used, so absolute recall is
meaningless; only latency is measured. LanceDB writes to a temp dir unless `--path` is given
(it must not exist or be empty — the benchmark never touches a real store).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np

from megamind.models import MemoryToken, Pose
from megamind.store.base import MemoryStore
from megamind.store.memory_store import InMemoryStore

OBJECTS = "keys wallet phone laptop mug badge charger notebook glasses umbrella headphones passport".split()
PLACES = "kitchen counter desk sofa hallway car office shelf bed table window bathroom".split()
ACTIONS = "picking up|putting down|looking at|walking past|holding|reading".split("|")
SIGNS = ["EXIT", "WELCOME", "ROOM 204", "CAFE OPEN 7-3", "WIFI megamind-guest", "PARKING LEVEL 2", "MENU $4.50"]
QUERIES = [
    "where did I leave my keys",
    "laptop on the desk",
    "wifi password",
    "mug kitchen counter",
    "room 204",
    "passport",
]


def synth_tokens(n: int, *, text_dim: int, image_dim: int, seed: int = 0, devices: int = 2) -> list[MemoryToken]:
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    tv = rng.standard_normal((n, text_dim), dtype=np.float32)
    tv /= np.linalg.norm(tv, axis=1, keepdims=True)
    iv = rng.standard_normal((n, image_dim), dtype=np.float32)
    iv /= np.linalg.norm(iv, axis=1, keepdims=True)
    t0 = 1_794_067_200.0  # 2026-11-07 08:00 PST
    out: list[MemoryToken] = []
    for i in range(n):
        obj, place = rnd.choice(OBJECTS), rnd.choice(PLACES)
        out.append(
            MemoryToken(
                id=f"bench{i:07d}",
                device_id=f"quest-{i % devices}",
                ts=t0 + 8.0 * (i // devices) + rnd.random(),
                caption=f"{rnd.choice(ACTIONS)} the {obj} on the {place}",
                ocr_text=rnd.choice(SIGNS) if rnd.random() < 0.3 else "",
                entities=[obj, place],
                pose=Pose(x=rnd.uniform(-5, 5), y=1.6, z=rnd.uniform(-5, 5), yaw=rnd.uniform(0, 360)),
                text_vec=tv[i].tolist(),
                image_vec=iv[i].tolist(),
                thumb_path=f"/nonexistent/bench{i:07d}.jpg",
                frame_hash=f"{rnd.getrandbits(64):016x}",
                episode_id=f"ep{i // (15 * devices):05d}",
            )
        )
    return out


def _stats(samples_s: list[float]) -> dict[str, float]:
    ms = sorted(s * 1000.0 for s in samples_s)
    p95 = ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))]
    return {"n": len(ms), "p50_ms": statistics.median(ms), "p95_ms": p95, "mean_ms": statistics.fmean(ms)}


async def _time(fn: Callable[[int], Awaitable[Any]], reps: int) -> dict[str, float]:
    await fn(0)  # warm-up (index load, lazy table open)
    samples = []
    for r in range(reps):
        start = time.perf_counter()
        await fn(r)
        samples.append(time.perf_counter() - start)
    return _stats(samples)


async def bench_store(
    store: MemoryStore, tokens: list[MemoryToken], *, queries: int, k: int = 12, batch: int = 500, seed: int = 1
) -> dict[str, dict[str, float]]:
    rnd = random.Random(seed)
    results: dict[str, dict[str, float]] = {}

    start = time.perf_counter()
    for i in range(0, len(tokens), batch):
        await store.add_many(tokens[i : i + batch])
    ingest_s = time.perf_counter() - start
    results["add_many (total ingest)"] = {
        "n": len(tokens),
        "total_s": ingest_s,
        "per_token_ms": ingest_s * 1000.0 / max(1, len(tokens)),
    }
    if hasattr(store, "optimize"):
        start = time.perf_counter()
        await store.optimize()
        results["optimize"] = {"n": 1, "total_s": time.perf_counter() - start}

    lo, hi = tokens[0].ts, tokens[-1].ts
    hour = (lo + (hi - lo) / 2, lo + (hi - lo) / 2 + 3600)
    qtext = [tokens[rnd.randrange(len(tokens))].text_vec for _ in range(queries + 1)]
    qimg = [tokens[rnd.randrange(len(tokens))].image_vec for _ in range(queries + 1)]
    ids = [t.id for t in tokens]
    frames = ("frame",)

    cases: dict[str, Callable[[int], Awaitable[Any]]] = {
        "vector_search text": lambda r: store.vector_search(qtext[r], k, kinds=frames),
        "vector_search image": lambda r: store.vector_search(qimg[r], k, field="image_vec", kinds=frames),
        "vector_search text +1h window": lambda r: store.vector_search(
            qtext[r], k, time_from=hour[0], time_to=hour[1], kinds=frames
        ),
        "keyword_search": lambda r: store.keyword_search(QUERIES[r % len(QUERIES)], k, kinds=frames),
        "keyword_search +device": lambda r: store.keyword_search(
            QUERIES[r % len(QUERIES)], k, device_id="quest-0", kinds=frames
        ),
        "range 1h": lambda r: store.range(hour[0], hour[1], limit=1000),
        "range day (consolidator)": lambda r: store.range(lo, hi, limit=1_000_000, kinds=frames),
        "get_many 12": lambda r: store.get_many(rnd.sample(ids, min(12, len(ids)))),
        "neighbors ±2": lambda r: store.neighbors(rnd.choice(ids)),
        "search hybrid": lambda r: store.search(qtext[r], qimg[r], QUERIES[r % len(QUERIES)], k=k),
        "count": lambda r: store.count(),
    }
    for name, fn in cases.items():
        results[name] = await _time(fn, queries)
    return results


def _fmt(results: dict[str, dict[str, dict[str, float]]]) -> str:
    backends = list(results)
    names = list(dict.fromkeys(name for res in results.values() for name in res))
    head = f"{'primitive':34s}" + "".join(f"{b + ' p50 / p95 ms':>28s}" for b in backends)
    lines = [head, "-" * len(head)]
    for name in names:
        row = f"{name:34s}"
        for b in backends:
            s = results[b].get(name, {})
            if "p50_ms" in s:
                row += f"{s['p50_ms']:>15.2f} / {s['p95_ms']:<10.2f}"
            elif "per_token_ms" in s:
                row += f"{s['total_s']:.2f} s total ({s['per_token_ms']:.3f} ms/tok)".rjust(28)
            elif "total_s" in s:
                row += f"{s['total_s'] * 1000:.1f} ms total".rjust(28)
            else:
                row += " " * 28
        lines.append(row)
    return "\n".join(lines)


async def run(
    n: int,
    *,
    backends: list[str],
    queries: int,
    text_dim: int,
    image_dim: int,
    path: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    tokens = synth_tokens(n, text_dim=text_dim, image_dim=image_dim, seed=seed)
    out: dict[str, Any] = {"n": n, "queries": queries, "text_dim": text_dim, "image_dim": image_dim, "results": {}}
    for backend in backends:
        if backend == "memory":
            out["results"][backend] = await bench_store(InMemoryStore(), tokens, queries=queries)
            continue
        from megamind.store.lancedb_store import LanceDBStore

        with tempfile.TemporaryDirectory(prefix="megamind-bench-") as tmp:
            p = Path(path) if path else Path(tmp) / "lancedb"
            if p.exists() and any(p.iterdir()):
                raise SystemExit(f"--path {p} is not empty; refusing to benchmark into an existing store")
            store = LanceDBStore(p, text_dim=text_dim, image_dim=image_dim)
            out["results"][backend] = await bench_store(store, tokens, queries=queries)
    return out


def _positive_int(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=_positive_int, default=10_000, help="synthetic tokens (default 10000)")
    ap.add_argument("--queries", type=_positive_int, default=20, help="timed calls per primitive (default 20)")
    ap.add_argument("--backends", default="memory,lancedb", help="comma list of memory,lancedb")
    ap.add_argument("--text-dim", type=_positive_int, default=1024)
    ap.add_argument("--image-dim", type=_positive_int, default=768)
    ap.add_argument("--path", type=Path, help="LanceDB dir (must be empty; default: temp dir)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="also write raw results here")
    a = ap.parse_args(argv)
    backends = [b.strip() for b in a.backends.split(",") if b.strip()]
    if bad := set(backends) - {"memory", "lancedb"}:
        ap.error(f"unknown backend(s): {', '.join(sorted(bad))}")
    out = asyncio.run(
        run(a.n, backends=backends, queries=a.queries, text_dim=a.text_dim, image_dim=a.image_dim, path=a.path)
    )
    print(f"n={a.n} tokens, {a.queries} queries/primitive, text_dim={a.text_dim}, image_dim={a.image_dim}")
    print(_fmt(out["results"]))
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
