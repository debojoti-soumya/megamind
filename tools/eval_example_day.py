#!/usr/bin/env python3
"""Ask a running server the example day's gold questions and score the answers and citations.

    python tools/make_example_day.py && sh server/data/example_day/feed.sh     # generate + feed (see tools/README.md)
    python tools/eval_example_day.py --gold server/data/example_day/example_day.json
    python tools/eval_example_day.py --modality visual --json out.json

A question is correct when the answer contains any `expect` substring (number-normalised, same scorer as
server/tools/eval_agent.py) or, for `__not_found__`, when the agent declines. A citation is a hit when its
timestamp falls inside a gold shot's clip window (`at` .. `at + dur_s`, plus --slack seconds). Server-side
token ids are random, so scoring by time is the only way to line live citations up with the gold shots.

Requires: pip install httpx (and the server package for number-normalised scoring).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = REPO / "server" / "data" / "example_day" / "example_day.json"
NOT_FOUND = "__not_found__"


def _scorer():
    """eval_agent.score_answer (handles "two"/"2", "level three"/"level 3") when the server package imports."""
    try:
        mod = sys.modules.get("eval_agent")
        if mod is None:
            spec = importlib.util.spec_from_file_location("eval_agent", REPO / "server" / "tools" / "eval_agent.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["eval_agent"] = mod  # @dataclass resolves annotations through sys.modules[cls.__module__]
            try:
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            except BaseException:
                sys.modules.pop("eval_agent", None)
                raise
        return lambda answer, expect: mod.score_answer(answer, mod.Case(q="", gold=[], expect=expect))
    except Exception as exc:  # noqa: BLE001 - fall back to plain substrings
        print(f"(plain substring scoring: {exc})", file=sys.stderr)

        def plain(answer: str, expect: list[str]) -> bool:
            a = answer.lower()
            if NOT_FOUND in expect:
                return "don't have a memory" in a or "no memory" in a or "didn't see" in a
            return any(e.lower() in a for e in expect)

        return plain


def windows(gold: dict) -> dict[str, tuple[float, float]]:
    tz, day = ZoneInfo(gold["tz"]), dt.date.fromisoformat(gold["date"])
    out = {}
    for e in gold["stream"]:
        h, m, s = (int(x) for x in e.get("at", e["time"] + ":00").split(":"))
        t0 = dt.datetime.combine(day, dt.time(h, m, s), tzinfo=tz).timestamp()
        out[e["id"]] = (t0, t0 + float(e.get("dur_s", 60)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--device-id", default=None, help="default: the gold file's device_id")
    ap.add_argument("--modality", choices=["visual", "speech", "both", "none"], action="append", help="filter")
    ap.add_argument("--slack", type=float, default=5.0, help="seconds of slack around each gold clip window")
    ap.add_argument("--json", type=Path, default=None, help="write per-question records here")
    args = ap.parse_args()

    gold = json.loads(args.gold.read_text())
    win = windows(gold)
    device = args.device_id or gold.get("device_id")
    score = _scorer()
    rows = []
    with httpx.Client(base_url=args.server, timeout=120.0) as http:
        for i, q in enumerate(gold["questions"]):
            if args.modality and q.get("modality") not in args.modality:
                continue
            t0 = time.time()
            r = http.post("/v1/chat", json={"session_id": f"example-day-{i}", "device_id": device, "text": q["q"]})
            r.raise_for_status()
            body = r.json()
            cited = [c["ts"] for c in body.get("cited", [])]
            spans = [win[g] for g in q["gold"]]
            hits = [ts for ts in cited if any(a - args.slack <= ts <= b + args.slack for a, b in spans)]
            rows.append(
                {
                    "q": q["q"],
                    "modality": q.get("modality"),
                    "answer": body["answer"],
                    "correct": score(body["answer"], q["expect"]),
                    "cited": len(cited),
                    "cite_hit": bool(hits) if q["gold"] else not cited,
                    "cite_precision": len(hits) / len(cited) if cited else None,
                    "ms": round((time.time() - t0) * 1000),
                }
            )
            row = rows[-1]
            mark = "✓" if row["correct"] else "✗"
            print(f"{mark} [{row['modality']:<6}] {q['q']}\n    -> {body['answer'][:160]}  (cited {len(cited)}, "
                  f"hit={row['cite_hit']}, {row['ms']} ms)")  # fmt: skip

    if not rows:
        print("no questions selected")
        return 1
    print(f"\naccuracy {sum(r['correct'] for r in rows)}/{len(rows)}, citation hit {sum(r['cite_hit'] for r in rows)}"
          f"/{len(rows)}")  # fmt: skip
    for mod in sorted({r["modality"] for r in rows}):
        sel = [r for r in rows if r["modality"] == mod]
        print(f"  {mod:<6} {sum(r['correct'] for r in sel)}/{len(sel)} correct")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
