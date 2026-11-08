#!/usr/bin/env python
"""Sweep how many / how large / which snapshots Claude sees per recall (READER_PROVIDER=claude), via eval_agent.

    cd server && . .venv/bin/activate
    python tools/sweep_reader_images.py                               # offline smoke: mock path, $0
    python tools/sweep_reader_images.py --live --configs px960_n12,px768_n6_div --json sweep.json   # $$$
    python tools/sweep_reader_images.py --live --dataset tools/data/long_week.json --questions 1,2,5 --pixels-only

Config names: `px<max px>_n<max images>` plus `_div` (READER_IMAGE_SELECT=diverse) and `_pre` (Haiku 4.5 picks
<= READER_PRESELECT_MAX of them first). Frames are rendered once at `--thumb-width` (1280 = headset upload) into
`--thumb-dir`, and the store is seeded once, so configs differ only in what Claude is shown.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_agent  # noqa: E402

DEFAULT_CONFIGS = "px960_n12,px768_n12,px512_n12,px960_n8,px960_n6,px960_n4,px768_n6,px768_n6_div,px960_n12_pre"
_NAME = re.compile(r"px(\d+)_n(\d+)((?:_div|_pre)*)$")


def config_overrides(name: str) -> list[str]:
    m = _NAME.match(name)
    if not m:
        raise SystemExit(f"bad config {name!r} (want px<px>_n<count>[_div][_pre])")
    out = ["READER_PROVIDER=claude", f"READER_THUMB_MAX_PX={m[1]}", f"READER_MAX_IMAGES={m[2]}"]
    out.append(f"READER_IMAGE_SELECT={'diverse' if '_div' in m[3] else 'rank'}")
    out.append(f"READER_PRESELECT_MODEL={'claude-haiku-4-5' if '_pre' in m[3] else ''}")
    return out


def row(name: str, s: dict) -> str:
    f = eval_agent._fmt
    return (
        f"| {name} | {s['accuracy']:.0%} | {f(s['citation_precision'])} / {f(s['citation_recall'])} "
        f"| {s['input_tokens_per_turn']:.0f} | {s['images_per_turn']:.1f} | ${s['cost_per_turn_usd']:.4f} "
        f"| {f(s['claude_call_p50_ms'], 0)} / {f(s['claude_call_p95_ms'], 0)} "
        f"| {s['latency_p50_ms']:.0f} / {s['latency_p95_ms']:.0f} |"
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=eval_agent.DEFAULT_DATASET)
    ap.add_argument("--configs", default=DEFAULT_CONFIGS)
    ap.add_argument("--live", action="store_true", help="real Claude from ANTHROPIC_API_KEY ($$$)")
    ap.add_argument("--questions", default="", help="comma-separated 1-based question numbers")
    ap.add_argument("--pixels-only", action="store_true", help="hide captions/OCR from Claude")
    ap.add_argument("--thumb-width", type=int, default=1280)
    ap.add_argument("--thumb-dir", type=Path, default=Path(".eval_cache/sweep_thumbs"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra Settings override")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    base = eval_agent.with_overrides(eval_agent.eval_settings(args.live), args.set)
    if args.live and not base.anthropic_api_key:
        raise SystemExit("--live needs ANTHROPIC_API_KEY")
    thumbs = args.thumb_dir / f"{args.dataset.stem}_{args.thumb_width}"
    day = eval_agent.select_questions(eval_agent.load_day(args.dataset, thumbs), args.questions)
    if not (thumbs / f"{day.tokens[-1].id}.jpg").is_file():
        eval_agent.render_thumbs(day, thumbs, width=args.thumb_width)
    store = await eval_agent.seed_store(day.tokens, eval_agent.build_text_embedder(base))
    print(
        "| config | accuracy | cite P / R | input tok/turn | images/turn | $/turn | Claude call p50 / p95 ms "
        "| turn p50 / p95 ms |\n|---|--:|--:|--:|--:|--:|--:|--:|"
    )
    results = {}
    for name in args.configs.split(","):
        settings = eval_agent.with_overrides(base, config_overrides(name))
        if not args.live:
            settings = settings.model_copy(update={"anthropic_api_key": "", "reader_provider": "mock"})
        turns = await eval_agent.run(day, settings, store=store, pixels_only=args.pixels_only)
        summ = eval_agent.summarize(turns)
        results[name] = {"summary": summ, "turns": [eval_agent.asdict(t) for t in turns]}
        print(row(name, summ), flush=True)
    total = sum(r["summary"]["cost_usd"] for r in results.values())
    print(f"\nestimated spend: ${total:.4f}", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
