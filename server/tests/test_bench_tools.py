"""Offline checks for the benchmark tools ported from PR #54 (tools/retrieval_bench.py, tools/egolife_eval.py,
tools/feeder.py --srt): dataset loaders, generators, time mapping and scoring math, on tiny committed fixtures.
No network, no torch; the benches' --prepare / fetch are never run here."""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import httpx
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
FIX = Path(__file__).resolve().parent / "fixtures" / "bench"
sys.path.insert(0, str(TOOLS))

import egolife_eval as ego  # noqa: E402
import feeder  # noqa: E402
import retrieval_bench as rb  # noqa: E402

from megamind.models import MemoryToken  # noqa: E402
from megamind.retrieval.hybrid import HybridConfig  # noqa: E402
from megamind.store.embeddings import MockImageEmbedder, MockTextEmbedder  # noqa: E402
from megamind.store.episodes import EpisodeTracker  # noqa: E402


def _coco() -> list[dict]:
    return rb.coco_subset(
        json.loads((FIX / "coco_instances.json").read_text()), json.loads((FIX / "coco_captions.json").read_text())
    )


# --------------------------------------------------------------------------- retrieval_bench: loaders


def test_coco_subset_keeps_indoor_photos_with_fractional_areas():
    sub = _coco()
    assert [r["file"] for r in sub] == ["000000000101.jpg", "000000000202.jpg"]  # 303 has only a person
    first = sub[0]
    assert first["captions"][0] == "Three cups of coffee on a wooden dining table."  # stripped
    cups = [o for o in first["objects"] if o["cat"] == "cup"]
    assert len(cups) == 3 and cups[0]["area"] == pytest.approx(2000 / (640 * 480))
    assert _coco() == sub  # deterministic


def test_coco_subset_caps_n_deterministically():
    inst = json.loads((FIX / "coco_instances.json").read_text())
    caps = json.loads((FIX / "coco_captions.json").read_text())
    one = rb.coco_subset(inst, caps, n=1, seed=7)
    assert len(one) == 1 and one == rb.coco_subset(inst, caps, n=1, seed=7)


def test_flickr_rows_numbers_files_in_row_order():
    rows = rb.flickr_rows([json.loads((FIX / "flickr_rows.json").read_text())])
    assert [r["file"] for r in rows] == ["0000.jpg", "0001.jpg"]
    assert rows[1]["captions"][0] == "Two children ride bicycles."
    assert rows[0]["src"].endswith("/0.jpg")


def test_count_and_find_questions_from_annotations():
    sub = _coco()
    counts = rb.count_questions(sub, random.Random(0))
    assert [(c["img"], c["cat"], c["n"]) for c in counts] == [("000000000101.jpg", "cup", 3)]
    assert counts[0]["q"].startswith("how many cups were there when I saw a table set with mugs")
    assert rb.find_questions(sub) == []  # every find target needs >= 2 valid photos
    assert rb.supports_of(sub[0], "cup") >= {"table"}


# --------------------------------------------------------------------------- retrieval_bench: generators


def test_make_facts_is_deterministic_and_has_hard_negatives():
    a, b = rb.make_facts(random.Random(1)), rb.make_facts(random.Random(1))
    assert a == b
    assert a != rb.make_facts(random.Random(2))
    kinds = {f["kind"] for f in a}
    assert kinds == {"ocr", "speech"}
    cats = {c for f in a for c, _, _ in f["queries"]}
    assert {"ocr_exact", "ocr_noisy", "ocr_paraphrase", "speech", "speech_paraphrase", "time_window"} <= cats
    noisy = [f for f in a if any(c == "ocr_noisy" for c, _, _ in f["queries"])]
    assert noisy and all(f["text"] != f["clean"] for f in noisy)
    parking = [f for f in a if f["queries"][0][0] == "time_window"]
    assert len(parking) == 8
    for f in parking:  # every parking sign sits inside its own day's window, and only there
        lo, hi = f["queries"][0][2]
        assert lo <= f["ts"] < hi
    speech = [f for f in a if f["kind"] == "speech"]
    assert len({f["queries"][0][1] for f in speech}) == len(speech)  # one question per (person, topic)


def test_says_number_reads_the_first_number():
    assert rb.says_number("There were three cups, maybe 4.", 3)
    assert rb.says_number("[coco:1] 2 bottles on the table", 2)
    assert not rb.says_number("I saw 4 cups and two plates", 2)
    assert not rb.says_number("no idea", 1)


# --------------------------------------------------------------------------- retrieval_bench: scoring


def test_score_r1_r5_mrr():
    assert rb.score(["a", "b"], {"a"}) == (1, 1, 1.0)
    assert rb.score(["x", "y", "a"], {"a"}) == (0, 1, pytest.approx(1 / 3))
    assert rb.score([f"x{i}" for i in range(6)] + ["a"], {"a"}) == (0, 0, pytest.approx(1 / 7))
    assert rb.score([f"x{i}" for i in range(10)] + ["a"], {"a"}) == (0, 0, 0.0)  # beyond K
    assert rb.score([], {"a"}) == (0, 0, 0.0)
    assert rb.score(["b", "a"], {"a", "b"}) == (1, 1, 1.0)  # any gold id counts


def test_summarize_means_and_all_row():
    t = rb.summarize({"c1": {"s": [(1, 1, 1.0), (0, 1, 0.5)]}, "c2": {"s": [(0, 0, 0.0), (0, 0, 0.0)]}})
    assert t["c1"]["s"] == {"n": 2, "r1": 0.5, "r5": 1.0, "mrr": 0.75}
    assert t["ALL"]["s"] == {"n": 4, "r1": 0.25, "r5": 0.5, "mrr": 0.375}


def test_load_weights_merges_over_planner_defaults(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"activity": {"image": 0.2}}))
    w = rb.load_weights(p)
    assert w["activity"].image == 0.2 and w["activity"].text == 1.0
    assert rb.load_weights(None) is None


def test_calibration_toggle_hides_and_restores():
    class Calibrated(MockImageEmbedder):
        calibration = (10.0, -12.0)

    emb = Calibrated()
    assert rb.calibration_of(emb) == (10.0, -12.0)
    assert rb.calibration_of(MockImageEmbedder()) is None
    assert rb.calibration_of(None) is None
    assert all("calibrat" in k for k in rb.calibration_knobs(HybridConfig()))  # [] on main without PR 1

    class R:
        cfg = HybridConfig()
        image_embedder = emb

    class S:
        image_calibration = (10.0, -12.0)

    r, s = R(), S()
    rb.set_calibration(r, s, emb, False)
    assert rb.calibration_of(r.image_embedder) is None and s.image_calibration is None
    rb.set_calibration(r, s, emb, True)
    assert r.image_embedder is emb and s.image_calibration == (10.0, -12.0)


async def test_bench_runs_end_to_end_with_mock_embedders(tmp_path):
    """Plumbing smoke test: tiny synthetic corpus, mock embedders, in-memory store, through HybridRetriever."""
    from PIL import Image

    sub = _coco()
    rows = rb.flickr_rows([json.loads((FIX / "flickr_rows.json").read_text())])
    for folder, items in (("coco", sub), ("flickr", rows)):
        (tmp_path / folder / "img").mkdir(parents=True)
        for i, r in enumerate(items):
            Image.new("RGB", (160, 120), (40 * i, 120, 200)).save(tmp_path / folder / "img" / r["file"])
    (tmp_path / "coco/subset.json").write_text(json.dumps(sub))
    (tmp_path / "flickr/captions.json").write_text(
        json.dumps([{"file": r["file"], "captions": r["captions"]} for r in rows])
    )

    class Calibrated(MockImageEmbedder):
        calibration = (10.0, -12.0)

    out = tmp_path / "report.json"
    rep = await rb.run(
        tmp_path,
        text=MockTextEmbedder(),
        image=Calibrated(),
        out=out,
        configs=("keyless", "captioned"),
        store_backend="memory",
        max_questions=25,
        quiet=True,
    )
    assert json.loads(out.read_text())["config"]["signals"] == rep["config"]["signals"]
    assert {"bm25", "text_vec", "image_vec", "hybrid", "hybrid_nocal", "store_search"} <= set(rep["config"]["signals"])
    assert rep["config"]["image_calibration"] == (10.0, -12.0)
    for config in ("keyless", "captioned"):
        assert rep["retrieval"][config]["ALL"]["hybrid"]["n"] == 25
        assert rep["config"][f"corpus_tokens_{config}"] > 4
    assert rep["planner"]["captioned"]  # query types recorded per category


# --------------------------------------------------------------------------- feeder --srt / --start


def test_feeder_parses_srt_and_maps_cues_to_frames():
    cues = feeder.load_srt(FIX / "cues.srt")
    assert cues == [
        (1.5, 3.0, "Priya: the demo moved to Friday."),
        (4.0, 6.25, "Marcus: let's get ramen after the meeting."),
    ]
    said, rest = feeder.due_cues(cues, 1.0)
    assert said is None and rest == cues
    said, rest = feeder.due_cues(cues, 2.0)  # first sampled frame after the cue starts carries it
    assert said == "Priya: the demo moved to Friday." and len(rest) == 1
    said, rest = feeder.due_cues(rest, 10.0)
    assert said.startswith("Marcus") and rest == []
    assert feeder.due_cues(rest, 20.0) == (None, [])  # each cue goes out once


def test_feeder_parse_srt_skips_malformed_blocks():
    text = "1\n00:00:01,000 --> 00:00:02,000\nhello\n\ngarbage block\n\n3\n00:00:05.500 --> 00:00:06.000\n\n"
    assert feeder.parse_srt(text) == [(1.0, 2.0, ["hello"])]


def test_feeder_help_lists_start_and_srt_for_feed_sh():
    """make_example_day's feed.sh greps `feeder.py --help` for these flags; --help must work without cv2."""
    import contextlib
    import io

    buf = io.StringIO()
    argv = sys.argv
    sys.argv = ["feeder.py", "--help"]
    try:
        with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
            feeder.main()
    finally:
        sys.argv = argv
    assert "--start" in buf.getvalue() and "--srt" in buf.getvalue()


# --------------------------------------------------------------------------- egolife: time mapping + transcripts


def test_wall_ts_and_clip_names():
    assert ego.wall_ts(1, "11100050") == ego.DAY0 + 11 * 3600 + 10 * 60 + 0.5
    assert ego.wall_ts(2, "00000000") == ego.DAY0 + 86400
    assert ego.clip_start("DAY1_A1_JAKE_11100000.mp4") == ego.wall_ts(1, "11100000")
    assert ego.clip_start("DAY3_A2_ALICE_09000000") == ego.wall_ts(3, "09000000")
    assert ego.clip_start("notes.txt") is None
    assert ego.day_no("day4") == 4


def test_targets_use_time_list_when_present():
    qa = json.loads((FIX / "egolife_qa.json").read_text())
    assert ego.targets(qa[0]["target_time"]) == [ego.wall_ts(1, "11150000")]
    assert ego.targets(qa[1]["target_time"]) == [ego.wall_ts(1, "11100000"), ego.wall_ts(1, "11400000")]


def test_load_cues_from_hour_files(tmp_path):
    shutil.copy(FIX / "cues.srt", tmp_path / "Transcript_11.srt")
    (tmp_path / "Transcript_12.srt").write_text("1\n00:00:02,000 --> 00:00:03,000\n舒瑞：我们走吧\nShure: let's go\n\n")
    (tmp_path / "Transcript_xx.srt").write_text("1\n00:00:02,000 --> 00:00:03,000\nignored\n\n")
    cues = ego.load_cues(tmp_path, day=1)
    base = ego.DAY0 + 11 * 3600
    assert cues[0] == (base + 1.5, "Priya: the demo moved to Friday.")
    assert cues[1] == (base + 4.0, "Marcus: let's get ramen after the meeting.")
    assert cues[2] == (ego.DAY0 + 12 * 3600 + 2.0, "Shure: let's go")  # Chinese line dropped
    said, rest = ego.take_due(cues, base + 5.0)
    assert said.startswith("Priya") and "Marcus" in said and len(rest) == 1
    assert ego.take_due(rest, base) == (None, rest)


def test_allow_patterns_and_locate(tmp_path):
    pats = ego.allow_patterns("A1_JAKE", "DAY1")
    assert "A1_JAKE/DAY1/*.mp4" in pats and any("EgoLifeQA" in p for p in pats)
    (tmp_path / "A1_JAKE/DAY1").mkdir(parents=True)
    (tmp_path / "A1_JAKE/DAY1/DAY1_A1_JAKE_11100000.mp4").write_bytes(b"")
    (tmp_path / "Transcript/A1_JAKE/DAY1").mkdir(parents=True)
    (tmp_path / "Transcript/A1_JAKE/DAY1/Transcript_11.srt").write_text("")
    (tmp_path / "EgoLifeQA").mkdir()
    (tmp_path / "EgoLifeQA/EgoLifeQA_A1_JAKE.json").write_text("[]")
    found = ego.locate(tmp_path, "A1_JAKE", "DAY1")
    assert found["n_clips"] == 1 and found["clips"].endswith("A1_JAKE/DAY1")
    assert found["srt_dir"].endswith("Transcript/A1_JAKE/DAY1")
    assert found["qa"].endswith("EgoLifeQA_A1_JAKE.json")


# --------------------------------------------------------------------------- egolife: scoring + baselines


def test_first_hit_and_chance():
    t = [100.0]
    assert ego.first_hit([500.0, 110.0, 100.0], t, 30) == 2
    assert ego.first_hit([500.0], t, 30) is None
    assert ego.first_hit([200.0], [100.0, 190.0], 30) == 1  # any target counts
    before = [100.0, 1000.0, 2000.0, 3000.0]  # 1 of 4 within ±30 s
    assert ego.chance_at(before, t, 30, 1) == pytest.approx(0.25)
    assert ego.chance_at(before, t, 30, 2) == pytest.approx(1 - 0.75**2)
    assert ego.chance_at([], t, 30, 5) == 0.0


def test_score_question_and_summary_with_baselines():
    qa = json.loads((FIX / "egolife_qa.json").read_text())
    q0, q1 = qa
    t0 = ego.targets(q0["target_time"])[0]
    got = [
        {"id": "a", "ts": t0 + 500, "caption": "x"},
        {"id": "b", "ts": t0 + 20, "caption": "y", "kind": "transcript"},
    ]
    before = sorted([t0 - 4000, t0 + 10, t0 + 290])
    row = ego.score_question(q0, got, before, [30.0, 600.0], k=12)
    assert row["best_rank"] == {"30.0": 2, "600.0": 1}
    assert row["dt_top1"] == pytest.approx(500)
    assert row["top"][1]["kind"] == "transcript" and row["top"][0]["kind"] == "frame"
    assert row["recent_rank"]["30.0"] == 2  # newest first: t0+290 (miss), then t0+10 (hit)
    assert row["chance"]["30.0"]["1"] == pytest.approx(1 / 3)
    t1 = ego.targets(q1["target_time"])
    row1 = ego.score_question(q1, [{"id": "c", "ts": t1[1] + 5, "caption": "z"}], before, [30.0, 600.0], k=12)
    assert row1["best_rank"]["30.0"] == 1 and row1["need_audio"] is True
    s = ego.summarize_rows([row, row1], [30.0, 600.0], k=12)
    assert s["ALL"]["n"] == 2
    assert s["ALL"]["±30s"]["R@1"] == 0.5 and s["ALL"]["±30s"]["R@5"] == 1.0
    assert s["need_audio"]["n"] == 1 and s["no_audio"]["n"] == 1
    assert "recent" in s["ALL"]["±30s"] and "chance" in s["ALL"]["±30s"]
    assert set(s) >= {"EntityLog", "EventRecall"}


def test_summary_without_baselines_and_question_selection():
    qa = json.loads((FIX / "egolife_qa.json").read_text())
    assert len(ego.select_questions(qa, "DAY1")) == 2
    assert ego.select_questions(qa, "DAY2") == []
    later = {**qa[0], "query_time": {"date": "DAY2", "time": "10000000"}}
    assert ego.select_questions([later], "DAY2") == []  # lag 1 day excluded by default
    assert ego.select_questions([later], "DAY2", max_lag=1) == [later]
    row = ego.score_question(qa[0], [], [], [30.0], k=12)
    assert row["dt_top1"] is None and "chance" not in row
    s = ego.summarize_rows([row], [30.0], k=12, max_lag=1)
    assert "recent" not in s["ALL"]["±30s"] and s["lag0d"]["n"] == 1


# --------------------------------------------------------------------------- egolife: drain + reepisode


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://t", transport=httpx.MockTransport(handler))


def test_wait_drained_prefers_stats_queue():
    queue = iter([5, 2, 0])

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/stats"
        return httpx.Response(200, json={"worker_queue": next(queue)})

    assert ego.wait_drained(_client(handler), sleep=lambda s: None) == "stats"


def test_wait_drained_falls_back_to_stable_store_count():
    counts = iter([3, 5, 7, 7, 7, 7])
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/stats":
            return httpx.Response(404, json={"detail": "Not Found"})
        seen.append(1)
        return httpx.Response(200, json={"store_count": next(counts)})

    assert ego.wait_drained(_client(handler), stable_polls=3, sleep=lambda s: None) == "store_count"
    assert len(seen) == 6
    # mid-ingest throttling without a queue size never blocks
    assert ego.wait_drained(_client(handler), max_queue=60, sleep=lambda s: None) == "store_count"


def test_reassign_reruns_episode_tracker():
    def tok(i: int, ts: float, vec: list[float], kind: str = "frame") -> MemoryToken:
        return MemoryToken(
            id=f"t{i}", device_id="d", ts=ts, thumb_path="", frame_hash="", image_vec=vec, kind=kind, episode_id="old"
        )

    a, b = [1.0, 0.0], [0.0, 1.0]
    toks = [tok(0, 0, a), tok(1, 30, a), tok(2, 60, b), tok(3, 1000, b), tok(4, 1010, b, kind="transcript")]
    stats = ego.reassign(toks, EpisodeTracker(gap_s=120, split_sim=0.8, rejoin_sim=0.85))
    assert stats["frames"] == 4 and stats["episodes"] == 3  # visual split at 60 s, gap split at 1000 s
    assert toks[0].episode_id == toks[1].episode_id != toks[2].episode_id
    assert toks[4].episode_id == "old"  # non-frame tokens untouched
    assert stats["max_s"] == 30


async def _lancedb_with(tmp_path, toks):
    from megamind.store.lancedb_store import LanceDBStore

    store = LanceDBStore(tmp_path / "db", vector_index_min_rows=10_000)
    await store.add_many(toks)
    return tmp_path / "db"


def test_offline_backend_runs_hybrid_in_utc_and_reports_calibration(tmp_path, monkeypatch):
    """--offline: main's retriever over a stored LanceDB, planner clock in UTC (the eval calendar), and "default"
    calibration mirrors the server (store.image_calibration set from the embedder) instead of silently off."""
    import asyncio

    pytest.importorskip("lancedb")
    for k in ("RETRIEVER_BACKEND", "WEARER_TZ", "SEARCH_CALIBRATE_IMAGE", "SPEECH_LINK_S"):
        monkeypatch.setenv(k, "")  # OfflineBackend writes these; monkeypatch restores them afterwards
        monkeypatch.delenv(k)
    monkeypatch.setenv("EMBED_PROVIDER", "mock")
    monkeypatch.setenv("IMAGE_EMBED_PROVIDER", "none")
    emb = MockTextEmbedder()
    base = ego.wall_ts(1, "11000000")

    async def toks():
        out = []
        for i, cap in enumerate(["red mug on the desk", "walking in the park", "red mug in the sink"]):
            t = MemoryToken(id=f"t{i}", device_id="d", ts=base + 600 * i, caption=cap, thumb_path="", frame_hash="")
            t.text_vec = await emb.embed(cap)
            out.append(t)
        return out

    path = asyncio.run(_lancedb_with(tmp_path, asyncio.run(toks())))
    be = ego.OfflineBackend(path, k=5, calibrate="default", speech_link=None)
    assert str(be.retriever.tz) == "UTC"  # not the server's WEARER_TZ default (America/Los_Angeles)
    got = be.search("red mug", base + 900)  # only memories up to query time
    assert [r["id"] for r in got] and {r["id"] for r in got} <= {"t0", "t1"}
    assert got[0]["id"] == "t0" and got[0]["kind"] == "frame"
    assert be.describe()["calibrated"] is False  # no image embedder -> nothing to calibrate
