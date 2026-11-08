"""The committed example-day gold file (tools/make_example_day.py) loads with both eval harnesses' loaders.

The generator itself needs network + macOS `say` + ffmpeg and never runs here; only its stdlib gold builder does.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from megamind.retrieval.eval import build_store, evaluate, load_dataset, make_retriever

SERVER = Path(__file__).resolve().parents[1]
GOLD = SERVER / "tools" / "data" / "example_day.json"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


eval_agent = _load("eval_agent", SERVER / "tools" / "eval_agent.py")


def test_gold_loads_with_agent_harness():
    day = eval_agent.load_day(GOLD)
    ids = [t.id for t in day.tokens]
    assert len(ids) == len(set(ids)) >= 10
    assert day.tokens == sorted(day.tokens, key=lambda t: t.ts)
    assert day.now > day.tokens[-1].ts
    assert len(day.cases) >= 12
    for c in day.cases:
        assert c.expect and set(c.gold) <= set(ids), c.q
        assert c.not_found == (not c.gold), c.q
    assert sum(c.not_found for c in day.cases) >= 1


def test_gold_extra_fields():
    raw = json.loads(GOLD.read_text())
    assert {q["modality"] for q in raw["questions"]} <= {"visual", "speech", "both", "none"}
    for e in raw["stream"]:
        assert e["at"].startswith(e["time"]) and e["dur_s"] > 0, e["id"]
        assert isinstance(e["said"], list)
    by_id = {e["id"]: e for e in raw["stream"]}
    for q in raw["questions"]:  # speech-only questions must have their answer actually spoken in a gold shot
        if q["modality"] == "speech":
            said = " ".join(s for g in q["gold"] for s in by_id[g]["said"]).lower()
            assert any(x.lower() in said for x in q["expect"]), q["q"]
        if q["modality"] == "visual":
            seen = " ".join(by_id[g]["caption"] + " " + by_id[g]["ocr"] for g in q["gold"]).lower()
            assert any(x.lower() in seen for x in q["expect"]), q["q"]


async def test_gold_loads_with_retrieval_harness():
    ds = load_dataset(GOLD)
    assert ds.cases and all(c.gold for c in ds.cases)  # must-decline questions are skipped
    rep = await evaluate(make_retriever(await build_store(ds.tokens), ds), ds, k=8)
    assert set(rep.summary()) >= {"mrr", "recall@1", "handoff_recall"}


def test_committed_gold_matches_generator():
    raw = json.loads(GOLD.read_text())
    gen = _load("make_example_day", SERVER.parent / "tools" / "make_example_day.py")
    assert GOLD.read_text() == gen.dump_gold(gen.build_gold(raw["date"], raw["tz"])), (
        "regenerate: python tools/make_example_day.py --gold-only --date 2026-11-08 --tz America/Los_Angeles "
        "--gold server/tools/data/example_day.json"
    )
