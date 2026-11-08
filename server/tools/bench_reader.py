#!/usr/bin/env python3
"""Reader latency bench: what one recall_memory reader call costs, split into payload build and model time.

    cd server && . .venv/bin/activate
    python tools/bench_reader.py                                   # offline: payload build time / size only
    python tools/bench_reader.py --frame-px 960 --thumb-px 512,256 # headset-sized frames, two downscales
    python tools/bench_reader.py --live --limit 8 \\
        --models google/gemma-4-31B-it,nvidia/Gemma-4-31B-IT-NVFP4  # one call per (model, px, question): $$

The token lists are what `recall_memory` hands the reader for the first `--limit` questions of the eval day
(tools/data/agent_day.json, mock embeddings). Live calls go through `VLMReader` with no fallback, no hedge and
no retries, so each row is exactly one request to one model; a call that misses `--timeout` counts as a hang.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent import DEFAULT_DATASET, load_day, render_thumbs, seed_store  # noqa: E402

import megamind.agent.reader as reader_mod  # noqa: E402
from megamind.agent.reader import CaptionOnlyReader, Reader, ReaderOutput, VLMReader, build_user_content  # noqa: E402
from megamind.agent.tools import RecallMemory  # noqa: E402
from megamind.config import Settings  # noqa: E402
from megamind.ingest.enrich import MockTextEmbedder  # noqa: E402
from megamind.models import MemoryToken  # noqa: E402


class CapturingReader(Reader):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[MemoryToken]]] = []

    async def read(self, query: str, tokens: list[MemoryToken], *, now: float | None = None) -> ReaderOutput:
        self.calls.append((query, list(tokens)))
        return await CaptionOnlyReader().read(query, tokens, now=now)


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))] if xs else float("nan")


async def capture(day, limit: int) -> list[tuple[str, list[MemoryToken]]]:
    emb = MockTextEmbedder()
    cap = CapturingReader()
    recall = RecallMemory(await seed_store(day.tokens, emb), emb, None, cap)
    for case in day.cases[:limit]:
        await recall(query=case.q, device_id=day.device_id, now=day.now)
    return [c for c in cap.calls if c[1]]


def bench_build(calls, now: float, px: int, zone) -> tuple[float, int]:
    t0 = time.perf_counter()
    size = 0
    for q, toks in calls:
        parts = build_user_content(q, toks, now, px, zone)
        size += sum(len(p.get("image_url", {}).get("url", "")) for p in parts)
    return (time.perf_counter() - t0) * 1000 / len(calls), size // len(calls)


async def bench_live(calls, now: float, model: str, px: int, timeout: float) -> list[dict]:
    base = Settings()
    s = base.model_copy(
        update={
            "reader_model": model,
            "reader_fallback_model": "",
            "reader_hedge_s": 0.0,
            "reader_max_retries": 0,
            "reader_timeout_s": timeout,
            "reader_deadline_s": timeout + 0.5,
            "reader_thumb_max_px": px,
        }
    )
    reader = VLMReader(s)
    rows = []
    for q, toks in calls:
        out = await reader.read(q, toks, now=now)
        rows.append(
            {
                "q": q,
                "ms": out.latency_ms,
                "hang": out.degraded,
                "conf": out.confidence,
                "prompt": out.usage.get("prompt_tokens", 0),
                "completion": out.usage.get("completion_tokens", 0),
                "images": len(toks),
            }
        )
    await reader.client.aclose()
    return rows


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--frame-px", type=int, default=960, help="width of the rendered frames (headset: 960)")
    ap.add_argument("--thumb-px", default="512", help="comma list of READER_THUMB_MAX_PX values to compare")
    ap.add_argument("--live", action="store_true", help="call READER_BASE_URL with READER_API_KEY ($$)")
    ap.add_argument("--models", default="", help="comma list (default: READER_MODEL)")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    settings = Settings()
    pxs = [int(x) for x in args.thumb_px.split(",") if x]
    with tempfile.TemporaryDirectory(prefix="megamind-bench-") as tmp:
        day = load_day(args.dataset, Path(tmp))
        render_thumbs(day, Path(tmp), width=args.frame_px)
        calls = await capture(day, args.limit)
        n_img = statistics.mean(len(t) for _, t in calls)
        print(f"{len(calls)} reader calls, {n_img:.1f} images each, frames {args.frame_px}px wide\n")
        print("| thumb px | build ms/call (cold) | build ms/call (warm) | image payload KB/call |\n|--:|--:|--:|--:|")
        for px in pxs:
            reader_mod.clear_encode_cache()
            ms, size = bench_build(calls, day.now, px, day.tz)
            warm, _ = bench_build(calls, day.now, px, day.tz)
            print(f"| {px} | {ms:.0f} | {warm:.0f} | {size / 1024:.0f} |")
        if not args.live:
            return 0
        if settings.reader_api_key in ("", "none"):
            print("--live needs READER_API_KEY", file=sys.stderr)
            return 2
        models = [m for m in args.models.split(",") if m] or [settings.reader_model]
        print("\n| model | thumb px | n | p50 ms | p95 ms | max ms | hangs | prompt tok | completion tok |")
        print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
        for model in models:
            for px in pxs:
                rows = await bench_live(calls, day.now, model, px, args.timeout)
                ok = [r["ms"] for r in rows if not r["hang"]]
                print(
                    f"| {model} | {px} | {len(rows)} | {pct(ok, 50):.0f} | {pct(ok, 95):.0f} | "
                    f"{max(ok, default=float('nan')):.0f} | {sum(r['hang'] for r in rows)} | "
                    f"{statistics.mean(r['prompt'] for r in rows):.0f} | "
                    f"{statistics.mean(r['completion'] for r in rows):.0f} |",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    raise SystemExit(asyncio.run(main()))
