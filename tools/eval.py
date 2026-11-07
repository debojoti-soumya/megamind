#!/usr/bin/env python3
"""Gold-question eval against a running server: hit@1 / hit@5 on /v1/memories, episode purity.

    python tools/eval.py --gold /tmp/day/gold.json --start 1731000000 [--server http://localhost:8000]

A hit is correct if its timestamp falls inside the expected scene's [start, end) window.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--start", type=float, required=True, help="same --start given to feeder.py")
    ap.add_argument("--device-id", default=None)
    args = ap.parse_args()
    gold = json.load(open(args.gold))
    win = {s["scene"]: (args.start + s["start_s"], args.start + s["end_s"]) for s in gold["scenes"]}

    def scene_of(ts: float) -> str | None:
        return next((n for n, (a, b) in win.items() if a <= ts < b), None)

    h1 = h5 = 0
    with httpx.Client(base_url=args.server, timeout=60) as http:
        for item in gold["questions"]:
            params = {"q": item["q"], "mode": item["mode"], "k": 5}
            if args.device_id:
                params["device_id"] = args.device_id
            res = http.get("/v1/memories", params=params).json()["results"]
            got = [scene_of(r["ts"]) for r in res]
            ok1, ok5 = bool(got) and got[0] == item["scene"], item["scene"] in got
            h1, h5 = h1 + ok1, h5 + ok5
            print(f"{'OK ' if ok1 else ('~5 ' if ok5 else 'XX ')} [{item['mode']:6}] {item['q'][:48]:48} -> {got[:3]}")

        eps = http.get("/v1/episodes", params={"k": 50}).json()["episodes"]
        purity = []
        for e in eps:
            s = Counter(scene_of(e["start_ts"] + f * (e["end_ts"] - e["start_ts"]) / 4) for f in range(5))
            purity.append(s.most_common(1)[0][1] / 5)
            print(f"  episode {e['n_tokens']:3d} tok  {e['title'][:30]:30}  scenes={dict(s)}")
        stats = http.get("/v1/stats").json()
    n = len(gold["questions"])
    print(f"\nhit@1 {h1}/{n}   hit@5 {h5}/{n}   episodes {len(eps)} (scenes {len(win)})", end="")
    print(f"   mean purity {sum(purity) / max(1, len(purity)):.2f}")
    print(json.dumps({k: stats[k] for k in ("tokens", "filter", "bytes_per_frame", "projected_100d_gb_at_budget")}))


if __name__ == "__main__":
    main()
