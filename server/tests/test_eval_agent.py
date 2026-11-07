"""Offline tests for tools/eval_agent.py: dataset, scoring, and a full 20-question run on the mock path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_agent", ROOT / "tools" / "eval_agent.py")
eval_agent = importlib.util.module_from_spec(spec)
sys.modules["eval_agent"] = eval_agent
spec.loader.exec_module(eval_agent)  # type: ignore[union-attr]

Case = eval_agent.Case


def test_dataset_is_a_scripted_day_with_20_questions():
    day = eval_agent.load_day()
    assert len(day.cases) == 20
    assert len(day.tokens) >= 30 and day.tokens == sorted(day.tokens, key=lambda t: t.ts)
    ids = {t.id for t in day.tokens}
    for c in day.cases:
        assert set(c.gold) <= ids, c.q
        assert c.expect, c.q
    assert sum(c.not_found for c in day.cases) == 1
    assert day.now > day.tokens[-1].ts  # "now" is after the last memory


def test_score_answer_substrings_and_not_found():
    assert eval_agent.score_answer("They're in the ceramic bowl in the hallway.", Case("q", ["e04"], ["bowl"]))
    assert not eval_agent.score_answer("On the desk.", Case("q", ["e04"], ["bowl", "hallway"]))
    nf = Case("q", [], [eval_agent.NOT_FOUND])
    assert eval_agent.score_answer("I don't have a memory of that.", nf)
    assert not eval_agent.score_answer("You fed the cat at 7.", nf)


@pytest.mark.parametrize(
    ("spoken", "canonical"),
    [
        ("ten thirty", "1030"),
        ("ten oh five", "1005"),
        ("four ninety-nine", "499"),
        ("four dollars and ninety-nine cents", "499 cents"),
        ("twenty-three forty-seven", "2347"),
        ("four eighty", "480"),
        ("four hundred and eighty", "480"),
        ("twelve hundred", "1200"),
        ("two thousand and twenty", "2020"),
        ("four one five, five five five, zero one eight eight", "4155550188"),
        ("route twenty-two", "route 22"),
        ("someone often", "someone often"),
    ],
)
def test_normalize_numbers_spoken_to_digits(spoken, canonical):
    assert eval_agent.normalize_numbers(spoken) == canonical


@pytest.mark.parametrize(
    ("answer", "expect"),
    [
        ("Your demo is at ten thirty on Thursday.", "10:30"),
        ("The oat milk was four ninety-nine.", "4.99"),
        ("Your total came to twenty-three dollars and forty-seven cents.", "23.47"),
        ("The slide showed a p95 of four hundred eighty milliseconds.", "480"),
        ("It's four one five, five five five, zero one eight eight.", "415-555-0188"),
        ("You took the twenty-two bus.", "22"),
        ("Level three, zone B.", "level 3"),
        ("It said sprint fourteen.", "sprint 14"),
        ("The demo is at 10:30.", "10:30"),
    ],
)
def test_score_answer_matches_spoken_numbers(answer, expect):
    assert eval_agent.score_answer(answer, Case("q", [], [expect]))


@pytest.mark.parametrize(
    ("answer", "expect"),
    [
        ("Your demo is at eleven thirty.", "10:30"),
        ("The oat milk was five ninety-nine.", "4.99"),
        ("You took the two bus.", "22"),
        ("It was route one twenty-two.", "22"),
        ("About forty-eight milliseconds.", "480"),
    ],
)
def test_score_answer_rejects_wrong_numbers(answer, expect):
    assert not eval_agent.score_answer(answer, Case("q", [], [expect]))


def test_citation_scores():
    assert eval_agent.citation_scores(["a", "b"], ["a"]) == (0.5, 1.0)
    assert eval_agent.citation_scores([], ["a"]) == (None, 0.0)
    assert eval_agent.citation_scores(["a"], []) == (0.0, None)


def test_render_thumbs_writes_a_jpeg_per_token(tmp_path):
    day = eval_agent.load_day(thumb_dir=tmp_path)
    eval_agent.render_thumbs(day, tmp_path)
    files = sorted(tmp_path.glob("*.jpg"))
    assert len(files) == len(day.tokens) and files[0].stat().st_size > 500


@pytest.mark.asyncio
async def test_offline_run_scores_every_question_without_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-ignored")  # --live not set -> must be blanked
    settings = eval_agent.eval_settings(live=False)
    assert settings.anthropic_api_key == "" and settings.reader_provider == "mock"
    day = eval_agent.load_day()
    turns = await eval_agent.run(day, settings)
    assert len(turns) == 20 and all(t.path == "fallback" for t in turns)
    assert all(t.model_calls == 0 and t.cost_usd == 0 for t in turns)
    summ = eval_agent.summarize(turns)
    # caption-only baseline on hashed embeddings: retrieval+captions should still answer most OCR questions
    assert summ["accuracy"] >= 0.5
    assert summ["tool_calls"] == 20 and summ["latency_p95_ms"] < 2000
    # relative-time questions plan against the scripted day's `now`, not the wall clock
    lunch = next(t for t in turns if t.q == "what did I have for lunch")
    assert lunch.correct and {"l01", "l02"} & set(lunch.cited)
    md = eval_agent.markdown(turns, summ, "t")
    assert md.count("\n| ") == 21 and "accuracy" in md  # header + 20 rows


@pytest.mark.asyncio
async def test_run_respects_limit_and_pins_now(monkeypatch):
    seen = []
    day = eval_agent.load_day()
    settings = eval_agent.eval_settings(live=False)
    from megamind.agent.claude_agent import ClaudeAgent

    orig = ClaudeAgent.chat_with_stats

    async def spy(self, session_id, text, device_id, *, now=None):
        seen.append(now)
        return await orig(self, session_id, text, device_id, now=now)

    monkeypatch.setattr(ClaudeAgent, "chat_with_stats", spy)
    turns = await eval_agent.run(day, settings, limit=3)
    assert len(turns) == 3 and seen == [day.now] * 3
