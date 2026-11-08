#!/usr/bin/env python3
"""Long real-glasses retrieval eval: EgoLife (Meta Aria, 30 s clips, bilingual transcripts) + EgoLifeQA.
Ported from PR #54 and adapted to main (separate transcript tokens, /v1/audio, main's EpisodeTracker).

    python tools/egolife_eval.py fetch                                   # HF_TOKEN; accept the dataset terms first
    python tools/egolife_eval.py extract --clips <dir of DAY1_A1_JAKE_*.mp4> --out ~/.cache/megamind/egolife/frames
    python tools/egolife_eval.py ingest --frames ~/.cache/megamind/egolife/frames --srt-dir <dir of Transcript_HH.srt>
    python tools/egolife_eval.py ingest --frames ... --audio --clips <clip dir>       # speech via POST /v1/audio
    python tools/egolife_eval.py eval --qa <EgoLifeQA_A1_JAKE.json> --day DAY1 --out egolife.json
    python tools/egolife_eval.py eval --qa ... --offline --lancedb server/data/lancedb --calibrate off --speech-link 0
    python tools/egolife_eval.py reepisode --lancedb server/data/lancedb --dry-run    # server stopped

Clip names encode wall time (DAY1_A1_JAKE_11100000 = 11:10:00.00); every day maps onto a fixed epoch so EgoLifeQA
query_time / target_time line up with token ts. A retrieval hit is a returned token within ±W seconds of a
target time, searching only memories from before query_time. Two baselines need no retrieval: `recent` (the k
newest memories before query_time) and `chance` (k uniformly random earlier memories, in closed form).

`eval` scores `GET /v1/memories?to=<query_time>` on a running server (the `/v1/memories` path), or with `--offline`
the in-process `HybridRetriever` over a stored LanceDB (the `recall_memory` path) with A/B switches:
`--calibrate on|off` (PR 1's calibrated image fusion) and `--speech-link S` (PR 3's cross-kind context).

EgoLife is research-only and gated: accept its terms on Hugging Face, keep the data outside the repo (default
~/.cache/megamind/egolife) and never commit or redistribute it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from feeder import parse_srt  # noqa: E402

DEFAULT_DIR = Path.home() / ".cache" / "megamind" / "egolife"
DEFAULT_REPO = "lmms-lab/EgoLife"
DAY0 = datetime(2026, 1, 5, tzinfo=UTC).timestamp()  # DAY1 00:00
CLIP_RE = re.compile(r"DAY(\d+)_\w+?_(\d{8})")
TOKENS_TABLE = "memory_tokens"  # LanceDBStore's table on main


# --------------------------------------------------------------------------- time mapping (pure, CI-tested)


def wall_ts(day: int, hhmmssff: str) -> float:
    """EgoLife "HHMMSSFF" (FF = hundredths) on DAY<day> -> unix seconds on the fixed eval calendar."""
    h, m, s, f = (int(hhmmssff[i : i + 2]) for i in (0, 2, 4, 6))
    return DAY0 + (day - 1) * 86400 + h * 3600 + m * 60 + s + f / 100


def day_no(date: str) -> int:
    return int(date.upper().removeprefix("DAY"))


def clip_start(name: str) -> float | None:
    """`DAY1_A1_JAKE_11100000[.mp4]` -> its first frame's ts; None for names that don't match."""
    m = CLIP_RE.search(name)
    return wall_ts(int(m[1]), m[2]) if m else None


def qa_ts(t: dict) -> float:
    return wall_ts(day_no(t["date"]), t["time"])


def targets(t: dict) -> list[float]:
    """All target timestamps of a question (`time_list` when the answer happened several times)."""
    times = t.get("time_list") or [t["time"]]
    return [wall_ts(day_no(t["date"]), x) for x in times]


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


# --------------------------------------------------------------------------- transcripts (pure, CI-tested)


def english(line: str) -> bool:
    """Bilingual EgoLife cues carry a Chinese and an English line; keep lines with a Latin word after `Speaker:`."""
    return bool(re.search(r"[A-Za-z]{3}", line.split(":", 1)[-1]))


def load_cues(srt_dir: Path, day: int) -> list[tuple[float, str]]:
    """Hour-file SRTs (`Transcript_HH.srt`, cue times relative to the hour) -> [(abs ts, English text)]."""
    cues = []
    for f in sorted(srt_dir.glob("Transcript_*.srt")):
        try:
            hour = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        base = DAY0 + (day - 1) * 86400 + hour * 3600
        for start, _end, lines in parse_srt(f.read_text(encoding="utf-8", errors="ignore")):
            en = [ln for ln in lines if english(ln)]
            if en:
                cues.append((base + start, " ".join(en)))
    return sorted(cues)


def take_due(cues: list[tuple[float, str]], ts: float) -> tuple[str | None, list[tuple[float, str]]]:
    """Cues at or before `ts` -> (joined text, remaining cues); each cue is sent once."""
    n = 0
    while n < len(cues) and cues[n][0] <= ts:
        n += 1
    if not n:
        return None, cues
    return " ".join(c[1] for c in cues[:n]), cues[n:]


# --------------------------------------------------------------------------- scoring (pure, CI-tested)


def first_hit(ts_list: list[float], tgts: list[float], window: float) -> int | None:
    """1-based rank of the first timestamp within ±window of any target, else None."""
    return next((i + 1 for i, ts in enumerate(ts_list) if min(abs(ts - t) for t in tgts) <= window), None)


def chance_at(before: list[float], tgts: list[float], window: float, n: int) -> float:
    """P(at least one of n uniformly drawn earlier memories is a hit) (with replacement)."""
    if not before:
        return 0.0
    p = sum(min(abs(x - t) for t in tgts) <= window for x in before) / len(before)
    return 1 - (1 - p) ** n


def lag_days(q: dict) -> int:
    return day_no(q["query_time"]["date"]) - day_no(q["target_time"]["date"])


def select_questions(qa: list[dict], day: str, max_lag: int = 0) -> list[dict]:
    return [q for q in qa if q["query_time"]["date"].upper() == day.upper() and 0 <= lag_days(q) <= max_lag]


def score_question(q: dict, got: list[dict], before: list[float], windows: list[float], k: int) -> dict:
    """One EgoLifeQA question vs the retrieved memories `got` ([{id, ts, caption, ...}], best first) and the
    sorted timestamps `before` of every memory up to query_time (empty = no baselines)."""
    tgts = targets(q["target_time"])
    ts_list = [r["ts"] for r in got]
    row: dict = {
        "id": q["ID"],
        "type": q["type"],
        "need_audio": bool(q.get("need_audio")),
        "lag_days": lag_days(q),
        "q": q["question"],
        "dt_top1": min(abs(ts_list[0] - t) for t in tgts) if ts_list else None,
        "best_rank": {str(w): first_hit(ts_list, tgts, w) for w in windows},
        "top": [
            {"id": r["id"], "dt": round(r["ts"] - tgts[0]), "kind": r.get("kind", "frame"), "caption": r.get("caption")}
            for r in got[:5]
        ],
    }
    if before:
        newest = before[::-1][:k]
        row["recent_rank"] = {str(w): first_hit(newest, tgts, w) for w in windows}
        row["chance"] = {str(w): {str(n): chance_at(before, tgts, w, n) for n in (1, 5, k)} for w in windows}
    return row


def summarize_rows(rows: list[dict], windows: list[float], k: int, max_lag: int = 0) -> dict:
    """Per question type (+ audio/no-audio, + per lag day, + ALL): R@1/R@5/R@k at each window, with the
    recent/chance baselines when every row has them."""
    groups = {t: [r for r in rows if r["type"] == t] for t in sorted({r["type"] for r in rows})}
    groups["need_audio"] = [r for r in rows if r["need_audio"]]
    groups["no_audio"] = [r for r in rows if not r["need_audio"]]
    if max_lag:
        groups |= {f"lag{d}d": [r for r in rows if r["lag_days"] == d] for d in range(max_lag + 1)}
    groups["ALL"] = rows
    ns = sorted({1, 5, k})
    summary: dict = {}
    for name, rs in groups.items():
        if not rs:
            continue
        summary[name] = {"n": len(rs)}
        for w in windows:
            ranks = [r["best_rank"][str(w)] for r in rs]
            m: dict = {f"R@{n}": round(sum(bool(x) and x <= n for x in ranks) / len(rs), 2) for n in ns}
            if all("chance" in r for r in rs):
                rec = [r["recent_rank"][str(w)] for r in rs]
                m["recent"] = {f"R@{n}": round(sum(bool(x) and x <= n for x in rec) / len(rs), 2) for n in ns}
                m["chance"] = {f"R@{n}": round(sum(r["chance"][str(w)][str(n)] for r in rs) / len(rs), 2) for n in ns}
            summary[name][f"±{w:g}s"] = m
    return summary


# --------------------------------------------------------------------------- fetch


def allow_patterns(person: str, day: str) -> list[str]:
    """`huggingface_hub` allow_patterns (fnmatch: `*` also matches `/`) for one person-day: the clips, the
    hour-file transcripts and the QA file, wherever the dataset keeps them."""
    return [
        f"{person}/{day}/*.mp4",
        f"*{person}*{day}*.srt",
        f"*Transcript*{person}*{day}*",
        f"*EgoLifeQA*{person}*.json",
    ]


def locate(root: Path, person: str, day: str) -> dict:
    """Find clips / transcript dir / QA json under a download root (layout-agnostic)."""
    clips = sorted(p for p in root.rglob(f"{day}_{person}_*.mp4"))
    srts = sorted(root.rglob("Transcript_*.srt"))
    srts = [p for p in srts if day in str(p) or person in str(p)] or srts
    qa = sorted(p for p in root.rglob("*.json") if "EgoLifeQA" in p.name and person in p.name)
    return {
        "clips": str(clips[0].parent) if clips else None,
        "n_clips": len(clips),
        "srt_dir": str(srts[0].parent) if srts else None,
        "qa": str(qa[0]) if qa else None,
    }


def fetch(out: Path, repo: str, person: str, day: str, patterns: list[str] | None) -> dict:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("fetch needs huggingface_hub: pip install huggingface_hub") from None
    out.mkdir(parents=True, exist_ok=True)
    pats = patterns or allow_patterns(person, day)
    print(f"downloading {repo} {pats} -> {out} (gated: accept the terms on huggingface.co first)", flush=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(out / "hf"),
        allow_patterns=pats,
        token=os.environ.get("HF_TOKEN") or None,
    )
    found = locate(out / "hf", person, day)
    print(json.dumps(found, indent=1))
    if not found["n_clips"]:
        print("no clips matched: check the dataset layout and pass --pattern (repeatable)")
    return found


# --------------------------------------------------------------------------- extract / ingest


def extract(clips: Path, out: Path, fps: float, width: int) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def one(p: Path) -> int:
        d = out / p.stem
        if d.exists() and any(d.iterdir()):
            return 0
        d.mkdir(exist_ok=True)
        cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", str(p), "-vf", f"fps={fps},scale={width}:-2"]
        subprocess.run([*cmd, "-q:v", "4", str(d / "%04d.jpg")], check=True)
        return 1

    clips_list = sorted(clips.glob("*.mp4"))
    with ThreadPoolExecutor(6) as ex:
        done = sum(ex.map(one, clips_list))
    print(f"extracted {done} new clips ({len(clips_list)} total) -> {out}")


def wait_drained(
    http: httpx.Client,
    *,
    max_queue: int = 0,
    poll_s: float = 2.0,
    stable_polls: int = 3,
    timeout_s: float = 3600.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Block until the enrichment worker has caught up. Uses `/v1/stats.worker_queue` when the server has it
    (tiered-media PR); otherwise waits for `/v1/health.store_count` to stay unchanged for `stable_polls` polls.
    Returns which signal was used ("stats" | "store_count" | "timeout")."""
    deadline = time.monotonic() + timeout_s
    last, same = None, 0
    while time.monotonic() < deadline:
        r = http.get("/v1/stats")
        if r.status_code == 200 and "worker_queue" in r.json():
            if r.json()["worker_queue"] <= max_queue:
                return "stats"
        else:
            if max_queue > 0:
                return "store_count"  # throttling needs a queue size; without one, don't block mid-ingest
            count = http.get("/v1/health").json().get("store_count")
            same = same + 1 if count == last else 0
            last = count
            if same >= stable_polls:
                return "store_count"
        sleep(poll_s)
    return "timeout"


def ingest(
    server: str,
    frames: Path,
    *,
    srt_dir: Path | None,
    fps: float,
    device: str,
    max_queue: int,
    audio_clips: Path | None = None,
    audio_chunk: float = 15.0,
) -> dict:
    """Post every extracted frame to /v1/frames on the clip's wall clock. Speech goes either as SRT cues on
    `FrameMeta.transcript` (stored as transcript tokens by servers with the speech PR; ignored by older ones) or,
    with `audio_clips`, as each clip's audio track on POST /v1/audio (needs ffmpeg and server-side STT)."""
    from feeder import AudioSender, extract_audio

    clips = sorted(d for d in frames.iterdir() if d.is_dir() and clip_start(d.name) is not None)
    if not clips:
        raise SystemExit(f"no DAYn_<person>_HHMMSSFF clip folders in {frames}")
    cues = load_cues(srt_dir, int(CLIP_RE.search(clips[0].name)[1])) if srt_dir else []
    sent = kept = 0
    reasons: dict[str, int] = {}
    audio_tokens = 0
    t0 = time.time()
    with httpx.Client(base_url=server, timeout=60) as http:
        health = http.get("/v1/health").json()
        for clip in clips:
            start = clip_start(clip.name)
            for i, jpg in enumerate(sorted(clip.glob("*.jpg"))):
                ts = start + i / fps
                meta: dict = {"device_id": device, "ts": ts}
                said, cues = take_due(cues, ts) if cues else (None, cues)
                if said:
                    meta["transcript"] = said
                body = http.post(
                    "/v1/frames",
                    files={"image": ("f.jpg", jpg.read_bytes(), "image/jpeg")},
                    data={"meta": json.dumps(meta)},
                ).json()
                sent += 1
                kept += bool(body.get("accepted"))
                r = body.get("reason") or "kept"
                reasons[r] = reasons.get(r, 0) + 1
                if body.get("accepted") and kept % 25 == 0:
                    wait_drained(http, max_queue=max_queue)
            if audio_clips is not None:
                mp4 = audio_clips / f"{clip.name}.mp4"
                pcm = extract_audio(str(mp4)) if mp4.is_file() else None
                if pcm:
                    snd = AudioSender(http, pcm, start, device, audio_chunk)
                    snd.flush()
                    audio_tokens += snd.tokens
            print(f"\r{clip.name} sent {sent} kept {kept} {time.time() - t0:.0f}s", end="", flush=True)
        how = wait_drained(http)
        stats = http.get("/v1/stats")
        final = http.get("/v1/health").json()
    out = {
        "sent": sent,
        "kept": kept,
        "kept_frac": round(kept / max(1, sent), 4),
        "reasons": reasons,
        "audio_transcript_tokens": audio_tokens,
        "drain": how,
        "store_count": final.get("store_count"),
        "models": health.get("models"),
        "stats": stats.json() if stats.status_code == 200 else None,
    }
    if srt_dir:
        out["note"] = "transcripts were sent as FrameMeta.transcript; servers without the speech PR ignore them"
    print("\n" + json.dumps(out, indent=1)[:3000])
    return out


# --------------------------------------------------------------------------- eval


def memory_timestamps(lancedb_path: Path) -> list[float]:
    """Sorted ts of every stored token (for the recent/chance baselines)."""
    import lancedb

    tbl = lancedb.connect(str(lancedb_path)).open_table(TOKENS_TABLE)
    return sorted(tbl.to_arrow().column("ts").to_pylist())


class HTTPBackend:
    def __init__(self, server: str, k: int) -> None:
        self.http = httpx.Client(base_url=server, timeout=60)
        self.k = k

    def describe(self) -> dict:
        h = self.http.get("/v1/health")
        return {"backend": "http /v1/memories", "health": h.json() if h.status_code == 200 else None}

    def search(self, question: str, tq: float) -> list[dict]:
        r = self.http.get("/v1/memories", params={"q": question, "k": self.k, "to": iso(tq)})
        r.raise_for_status()
        return r.json()["results"]


class OfflineBackend:
    """main's HybridRetriever over a stored LanceDB (server stopped), with embedders from the server's env."""

    def __init__(self, lancedb_path: Path, k: int, calibrate: str, speech_link: float | None) -> None:
        import asyncio

        sys.path.insert(0, str(ROOT / "server"))
        # A/B switches also go in as env settings, so a Settings-driven implementation sees them too.
        os.environ["RETRIEVER_BACKEND"] = "hybrid"  # UCC needs its cloud corpus; offline is our retriever
        if speech_link is not None:
            os.environ["SPEECH_LINK_S"] = str(speech_link)
        if calibrate != "default":
            os.environ["SEARCH_CALIBRATE_IMAGE"] = "true" if calibrate == "on" else "false"
        from retrieval_bench import _name, calibration_of, set_calibration

        from megamind.config import Settings
        from megamind.retrieval import build_retriever
        from megamind.store.embeddings import build_image_embedder, build_text_embedder
        from megamind.store.lancedb_store import LanceDBStore

        self.loop = asyncio.new_event_loop()
        s = Settings()
        self.text, self.image = build_text_embedder(s), build_image_embedder(s)
        self.store = LanceDBStore(lancedb_path)
        self.retriever, _ = build_retriever(s, self.store, self.text, self.image)
        cfg = self.retriever.cfg
        self.notes: list[str] = []
        if speech_link is not None:
            if hasattr(cfg, "speech_link_s"):
                cfg.speech_link_s = speech_link
            elif not hasattr(s, "speech_link_s"):
                self.notes.append("--speech-link ignored: no speech_link_s setting (speech PR not merged)")
        cal = calibration_of(self.image)
        if calibrate != "default":
            if cal is None:
                self.notes.append("--calibrate ignored: image embedder has no calibration (PR 1 / no SigLIP)")
            set_calibration(self.retriever, self.store, self.image, calibrate == "on")
        self.k = k
        self.config = {
            "backend": "offline HybridRetriever",
            "text_embedder": _name(self.text),
            "image_embedder": _name(self.image),
            "image_calibration": cal,
            "calibrate": calibrate,
            "speech_link_s": getattr(cfg, "speech_link_s", None),
            "notes": self.notes,
        }
        for n in self.notes:
            print("note:", n)

    def describe(self) -> dict:
        return self.config

    def search(self, question: str, tq: float) -> list[dict]:
        res = self.loop.run_until_complete(
            self.retriever.retrieve(question, k=self.k, time_to=tq, now=datetime.fromtimestamp(tq, UTC))
        )
        return [
            {"id": r.token.id, "ts": r.token.ts, "kind": r.token.kind, "caption": r.token.caption} for r in res.ranked
        ]


def evaluate(
    backend,
    qa_path: Path,
    day: str,
    windows: list[float],
    k: int,
    out: Path | None,
    lancedb_path: Path | None,
    max_lag: int = 0,
) -> dict:
    qa = select_questions(json.loads(qa_path.read_text()), day, max_lag)
    all_ts = memory_timestamps(lancedb_path) if lancedb_path else []
    w_max = max(windows)
    rows = []
    for q in qa:
        tq = qa_ts(q["query_time"])
        got = backend.search(q["question"], tq)
        before = [x for x in all_ts if x <= tq]
        row = score_question(q, got, before, windows, k)
        rows.append(row)
        br = row["best_rank"][str(w_max)]
        print(f"{'OK' if br == 1 else (f'@{br}' if br else 'XX'):3} {q['type']:14} {q['question'][:60]}")
    summary = summarize_rows(rows, windows, k, max_lag)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    config = {"qa": str(qa_path), "day": day, "windows": windows, "k": k, "max_lag": max_lag, **backend.describe()}
    if out:
        out.write_text(json.dumps({"config": config, "summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
        print("report ->", out)
    return summary


# --------------------------------------------------------------------------- reepisode


def reassign(tokens: list, tracker) -> dict:
    """Re-run `tracker` over frame tokens (in ts order, episode ids cleared) -> stats. Mutates `episode_id`."""
    frames = sorted((t for t in tokens if t.kind == "frame"), key=lambda t: (t.device_id, t.ts, t.id))
    spans: dict[str, list[float]] = {}
    for t in frames:
        t.episode_id = None
        eid = tracker.assign(t)
        lo_hi = spans.setdefault(eid, [t.ts, t.ts])
        lo_hi[0], lo_hi[1] = min(lo_hi[0], t.ts), max(lo_hi[1], t.ts)
    durs = sorted(hi - lo for lo, hi in spans.values())
    return {
        "frames": len(frames),
        "episodes": len(spans),
        "median_s": durs[len(durs) // 2] if durs else 0.0,
        "max_s": durs[-1] if durs else 0.0,
    }


def reepisode(lancedb_path: Path, dry_run: bool) -> dict:
    """Re-run main's online episode assignment (current EPISODE_* settings) over a stored LanceDB, e.g. after
    tuning thresholds. The server must be stopped. Consolidator summaries of the old episodes are not rebuilt."""
    import asyncio

    sys.path.insert(0, str(ROOT / "server"))
    from megamind.config import Settings
    from megamind.store import build_episode_tracker
    from megamind.store.lancedb_store import LanceDBStore

    async def run() -> dict:
        store = LanceDBStore(lancedb_path)  # no tracker: add_many must not re-assign
        toks = await store.range(None, None, limit=10_000_000, kinds=("frame",))
        stats = reassign(toks, build_episode_tracker(Settings()))
        if not dry_run:
            await store.add_many(toks)
        return stats

    stats = asyncio.run(run())
    print(json.dumps({**stats, "written": not dry_run}, indent=1))
    return stats


# --------------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(description="EgoLife retrieval eval for megamind")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download one person-day from Hugging Face (HF_TOKEN)")
    f.add_argument("--out", type=Path, default=DEFAULT_DIR)
    f.add_argument("--repo", default=DEFAULT_REPO)
    f.add_argument("--person", default="A1_JAKE")
    f.add_argument("--day", default="DAY1")
    f.add_argument("--pattern", action="append", default=None, help="override allow_patterns (repeatable)")
    e = sub.add_parser("extract", help="clips -> JPEG frames (ffmpeg)")
    e.add_argument("--clips", type=Path, required=True)
    e.add_argument("--out", type=Path, default=DEFAULT_DIR / "frames")
    e.add_argument("--fps", type=float, default=1.0)
    e.add_argument("--width", type=int, default=960)
    i = sub.add_parser("ingest", help="post frames (+ speech) to a running server")
    i.add_argument("--server", default="http://localhost:8000")
    i.add_argument("--frames", type=Path, default=DEFAULT_DIR / "frames")
    i.add_argument("--srt-dir", type=Path, default=None, help="Transcript_HH.srt dir -> FrameMeta.transcript")
    i.add_argument("--audio", action="store_true", help="post each clip's audio to /v1/audio instead")
    i.add_argument("--clips", type=Path, default=None, help="clip dir (with --audio)")
    i.add_argument("--fps", type=float, default=1.0, help="must match extract --fps")
    i.add_argument("--device-id", default="aria-jake")
    i.add_argument("--max-queue", type=int, default=60)
    i.add_argument("--out", type=Path, default=None, help="write ingest stats JSON here")
    r = sub.add_parser("reepisode", help="re-run episode assignment over a stored LanceDB (server stopped)")
    r.add_argument("--lancedb", type=Path, required=True)
    r.add_argument("--dry-run", action="store_true")
    v = sub.add_parser("eval", help="score EgoLifeQA")
    v.add_argument("--server", default="http://localhost:8000")
    v.add_argument("--qa", type=Path, required=True)
    v.add_argument("--day", default="DAY1")
    v.add_argument("--windows", default="30,120,600")
    v.add_argument("--k", type=int, default=12)
    v.add_argument("--out", type=Path, default=None)
    v.add_argument("--lancedb", type=Path, default=None, help="token store: recent/chance baselines (+ --offline)")
    v.add_argument("--max-lag", type=int, default=0, help="also score questions answered up to N days earlier")
    v.add_argument("--offline", action="store_true", help="in-process HybridRetriever over --lancedb, no server")
    v.add_argument("--calibrate", choices=["default", "on", "off"], default="default", help="offline: PR 1 A/B")
    v.add_argument("--speech-link", type=float, default=None, help="offline: SPEECH_LINK_S for PR 3 A/B (0 = off)")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.out, a.repo, a.person, a.day, a.pattern)
    elif a.cmd == "extract":
        extract(a.clips, a.out, a.fps, a.width)
    elif a.cmd == "ingest":
        if a.audio and a.clips is None:
            raise SystemExit("--audio needs --clips (the .mp4 directory)")
        res = ingest(
            a.server,
            a.frames,
            srt_dir=None if a.audio else a.srt_dir,
            fps=a.fps,
            device=a.device_id,
            max_queue=a.max_queue,
            audio_clips=a.clips if a.audio else None,
        )
        if a.out:
            a.out.write_text(json.dumps(res, indent=1))
    elif a.cmd == "reepisode":
        reepisode(a.lancedb, a.dry_run)
    else:
        if a.offline and a.lancedb is None:
            raise SystemExit("--offline needs --lancedb")
        backend = (
            OfflineBackend(a.lancedb, a.k, a.calibrate, a.speech_link) if a.offline else HTTPBackend(a.server, a.k)
        )
        evaluate(backend, a.qa, a.day, [float(x) for x in a.windows.split(",")], a.k, a.out, a.lancedb, a.max_lag)


if __name__ == "__main__":
    main()
