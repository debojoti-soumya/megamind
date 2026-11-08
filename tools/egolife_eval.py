#!/usr/bin/env python3
"""Long real-glasses eval: EgoLife (Meta Aria, 30 s clips, bilingual transcripts) + EgoLifeQA.

    python tools/egolife_eval.py extract --clips /tmp/egolife/hf/A1_JAKE/DAY1 --out /tmp/egolife/frames
    python tools/egolife_eval.py ingest --frames /tmp/egolife/frames --srt-dir /tmp/egolife/srt
    python tools/egolife_eval.py eval --qa /tmp/egolife/EgoLifeQA_A1_JAKE.json --day DAY1

Clip names encode wall time (DAY1_A1_JAKE_11100000 = 11:10:00.00); every day maps onto a fixed epoch so
EgoLifeQA query_time / target_time line up with token ts. A retrieval hit is a top-k token within
--window seconds of target_time, searching only memories from before query_time.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

DAY0 = datetime(2026, 1, 5, tzinfo=UTC).timestamp()  # DAY1 00:00
CLIP_RE = re.compile(r"DAY(\d+)_\w+?_(\d{8})")


def wall_ts(day: int, hhmmssff: str) -> float:
    h, m, s, f = (int(hhmmssff[i : i + 2]) for i in (0, 2, 4, 6))
    return DAY0 + (day - 1) * 86400 + h * 3600 + m * 60 + s + f / 100


def qa_ts(t: dict) -> float:
    return wall_ts(int(t["date"].removeprefix("DAY")), t["time"])


def extract(clips: Path, out: Path, fps: float, width: int) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def one(p: Path) -> int:
        d = out / p.stem
        if d.exists() and any(d.iterdir()):
            return 0
        d.mkdir(exist_ok=True)
        vf = f"fps={fps},scale={width}:-2"
        cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", str(p), "-vf", vf, "-q:v", "4", str(d / "%04d.jpg")]
        subprocess.run(cmd, check=True)
        return 1

    clips_list = sorted(clips.glob("*.mp4"))
    with ThreadPoolExecutor(6) as ex:
        done = sum(ex.map(one, clips_list))
    print(f"extracted {done} new clips ({len(clips_list)} total) -> {out}")


def load_cues(srt_dir: Path, day: int) -> list[tuple[float, str]]:
    """Hour-file SRTs (Transcript_HH.srt) -> [(abs ts, English line)]."""
    cues = []
    for f in sorted(srt_dir.glob("Transcript_*.srt")):
        hour = int(f.stem.split("_")[1])
        for block in re.split(r"\n\s*\n", f.read_text(encoding="utf-8", errors="ignore")):
            m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->", block)
            if not m:
                continue
            lines = [ln.strip() for ln in block.splitlines()[2:] if ln.strip()]
            en = [ln for ln in lines if re.search(r"[A-Za-z]{3}", ln.split(":", 1)[-1])]
            if en:
                off = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
                cues.append((DAY0 + (day - 1) * 86400 + hour * 3600 + off, " ".join(en)))
    return sorted(cues)


def ingest(server: str, frames: Path, srt_dir: Path | None, fps: float, device: str, max_queue: int) -> None:
    clips = sorted(frames.iterdir())
    first = CLIP_RE.search(clips[0].name)
    cues = load_cues(srt_dir, int(first[1])) if srt_dir else []
    sent = kept = 0
    reasons: dict[str, int] = {}
    t0 = time.time()
    with httpx.Client(base_url=server, timeout=30) as http:
        for clip in clips:
            m = CLIP_RE.search(clip.name)
            start = wall_ts(int(m[1]), m[2])
            for i, jpg in enumerate(sorted(clip.glob("*.jpg"))):
                ts = start + i / fps
                meta: dict = {"device_id": device, "ts": ts, "source": "egolife"}
                due = [c for c in cues if c[0] <= ts]
                if due:
                    meta["transcript"] = " ".join(c[1] for c in due)
                    cues = cues[len(due) :]
                body = http.post(
                    "/v1/frames",
                    files={"image": ("f.jpg", jpg.read_bytes(), "image/jpeg")},
                    data={"meta": json.dumps(meta)},
                ).json()
                sent += 1
                kept += bool(body.get("accepted"))
                r = body.get("reason") or body.get("trigger") or "ok"
                reasons[r] = reasons.get(r, 0) + 1
                if kept and kept % 25 == 0 and body.get("accepted"):
                    while http.get("/v1/stats").json().get("worker_queue", 0) > max_queue:
                        time.sleep(1)
            print(f"\r{clip.name} sent {sent} kept {kept} {time.time() - t0:.0f}s", end="", flush=True)
        while http.get("/v1/stats").json().get("worker_queue", 0) > 0:
            time.sleep(2)
        stats = http.get("/v1/stats").json()
    print(f"\nsent {sent}, kept {kept} ({kept / max(1, sent):.1%}); reasons {reasons}")
    print(json.dumps(stats, indent=1)[:2000])


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def evaluate(server: str, qa_path: Path, day: str, windows: list[float], k: int, out: Path | None) -> dict:
    qa = [
        q for q in json.loads(qa_path.read_text()) if q["target_time"]["date"] == day and q["query_time"]["date"] == day
    ]
    w_max = max(windows)
    rows = []
    with httpx.Client(base_url=server, timeout=60) as http:
        for q in qa:
            tq, tt = qa_ts(q["query_time"]), qa_ts(q["target_time"])
            res = http.get("/v1/memories", params={"q": q["question"], "k": k, "to": iso(tq)}).json()["results"]
            dts = [abs(r["ts"] - tt) for r in res]
            rows.append(
                {
                    "id": q["ID"],
                    "type": q["type"],
                    "need_audio": q["need_audio"],
                    "q": q["question"],
                    "dt_top1": dts[0] if dts else None,
                    "best_rank": {str(w): next((i + 1 for i, d in enumerate(dts) if d <= w), None) for w in windows},
                    "top": [{"id": r["id"], "dt": round(r["ts"] - tt), "caption": r["caption"]} for r in res[:5]],
                }
            )
            br = rows[-1]["best_rank"][str(w_max)]
            print(f"{'OK' if br == 1 else (f'@{br}' if br else 'XX'):3} {q['type']:12} {q['question'][:60]}")
    summary: dict = {}
    for t in sorted({r["type"] for r in rows}) + ["ALL"]:
        rs = [r for r in rows if t in ("ALL", r["type"])]
        summary[t] = {"n": len(rs)}
        for w in windows:
            ranks = [r["best_rank"][str(w)] for r in rs]
            summary[t][f"±{w:g}s"] = {
                "R@1": round(sum(r == 1 for r in ranks) / len(rs), 2),
                "R@5": round(sum(bool(r) and r <= 5 for r in ranks) / len(rs), 2),
                f"R@{k}": round(sum(bool(r) for r in ranks) / len(rs), 2),
            }
    print(json.dumps(summary, indent=1))
    if out:
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return summary


def reepisode(lancedb_path: Path) -> None:
    """Re-run online episode assignment over a stored LanceDB (server stopped), e.g. after tuning EPISODE_*."""
    import asyncio
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
    from megamind.config import Settings
    from megamind.ingest.episodes import EpisodeTracker
    from megamind.store.lancedb_store import LanceDBStore

    async def run() -> None:
        s = Settings()
        store = LanceDBStore(lancedb_path, text_dim=384, image_dim=768)
        tracker = EpisodeTracker(s, None)
        toks = await store.list_tokens()
        store._e.delete("true")
        eps = {}
        for t in toks:
            e = await tracker.assign(t)
            eps[e.id] = e
        await store.add_many(toks)
        for e in eps.values():
            await store.upsert_episode(e)
        print(f"{len(toks)} tokens -> {len(eps)} episodes")

    asyncio.run(run())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--clips", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--fps", type=float, default=1.0)
    e.add_argument("--width", type=int, default=960)
    i = sub.add_parser("ingest")
    i.add_argument("--server", default="http://localhost:8000")
    i.add_argument("--frames", type=Path, required=True)
    i.add_argument("--srt-dir", type=Path, default=None)
    i.add_argument("--fps", type=float, default=1.0)
    i.add_argument("--device-id", default="aria-jake")
    i.add_argument("--max-queue", type=int, default=60)
    r = sub.add_parser("reepisode")
    r.add_argument("--lancedb", type=Path, required=True)
    v = sub.add_parser("eval")
    v.add_argument("--server", default="http://localhost:8000")
    v.add_argument("--qa", type=Path, required=True)
    v.add_argument("--day", default="DAY1")
    v.add_argument("--windows", default="30,120,600")
    v.add_argument("--k", type=int, default=12)
    v.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.clips, a.out, a.fps, a.width)
    elif a.cmd == "ingest":
        ingest(a.server, a.frames, a.srt_dir, a.fps, a.device_id, a.max_queue)
    elif a.cmd == "reepisode":
        reepisode(a.lancedb)
    else:
        evaluate(a.server, a.qa, a.day, [float(x) for x in a.windows.split(",")], a.k, a.out)


if __name__ == "__main__":
    main()
