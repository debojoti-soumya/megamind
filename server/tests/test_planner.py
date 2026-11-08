from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from megamind.retrieval.planner import detect_type, parse_anchor, plan_query, rewrite

TZ = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 11, 8, 10, 30, tzinfo=TZ)  # Sunday 10:30 local


def _local(y, mo, d, h, m=0):
    return datetime(y, mo, d, h, m, tzinfo=TZ).timestamp()


def test_minutes_ago_window_centered():
    p = plan_query("where did I leave my keys 10 minutes ago?", now=NOW)
    center = NOW.timestamp() - 600
    assert p.time_from == pytest.approx(center - 300)
    assert p.time_to == pytest.approx(center + 300)
    assert p.time_hint == "10 minutes ago"
    assert p.query_type == "locate_object"
    assert "keys" in p.search_text and "minutes" not in p.search_text


def test_word_numbers_and_half_an_hour():
    p = plan_query("what was on the whiteboard about half an hour ago", now=NOW)
    assert p.time_from == pytest.approx(NOW.timestamp() - 1800 - 900)
    p = plan_query("the sign I saw a couple of hours ago", now=NOW)
    assert NOW.timestamp() - 2 * 3600 - 0.35 * 2 * 3600 == pytest.approx(p.time_from)


def test_last_n_minutes_ends_now():
    p = plan_query("what did I read in the last 20 minutes", now=NOW)
    assert p.time_to == pytest.approx(NOW.timestamp())
    assert p.time_from == pytest.approx(NOW.timestamp() - 1200)


def test_this_morning():
    p = plan_query("who did I meet this morning", now=NOW)
    assert p.time_from == _local(2026, 11, 8, 5)
    assert p.time_to == pytest.approx(NOW.timestamp())  # clamped: it is still morning
    assert p.query_type == "people"


def test_yesterday_around_lunch():
    p = plan_query("what did the menu say yesterday around lunch", now=NOW)
    assert p.time_from == _local(2026, 11, 7, 11, 30)
    assert p.time_to == _local(2026, 11, 7, 14)
    assert p.query_type == "read_text"
    assert "menu" in p.search_text and "yesterday" not in p.search_text and "lunch" not in p.search_text


def test_last_night_spans_midnight():
    p = plan_query("where did I put my wallet last night", now=NOW)
    assert p.time_from == _local(2026, 11, 7, 19)
    assert p.time_to == _local(2026, 11, 8, 4)


def test_clock_time_pm_and_bare():
    p = plan_query("what was I doing at 9am", now=NOW)
    assert p.time_from == _local(2026, 11, 8, 8, 15) and p.time_to == _local(2026, 11, 8, 9, 45)
    p = plan_query("the document I looked at around 3 pm yesterday", now=NOW)
    assert p.time_from == _local(2026, 11, 7, 14, 15)
    p = plan_query("what happened at 8", now=NOW)  # 8 am today is the most recent 8 o'clock
    assert p.time_from == _local(2026, 11, 8, 7, 15)


def test_weekday_resolves_to_most_recent_past():
    p = plan_query("what did I buy on friday", now=NOW)
    assert p.time_from == _local(2026, 11, 6, 0) and p.time_to == _local(2026, 11, 7, 0)


def test_evening_in_future_rolls_back_a_day():
    p = plan_query("what did I cook this evening", now=NOW)  # asked at 10:30, evening hasn't happened
    assert p.time_from == _local(2026, 11, 7, 17)


def test_pure_temporal_query_has_empty_text_and_temporal_mode():
    p = plan_query("what was I doing 10 minutes ago?", now=NOW)
    assert p.search_text == ""
    assert p.mode == "temporal"
    assert p.query_type == "activity"


def test_anchor_event_detected_without_time():
    p = plan_query("where were my glasses before I left the office?", now=NOW)
    assert p.anchor_event == "left the office"
    assert not p.has_time
    assert p.search_text.split()[0] == "glasses"


def test_explicit_bounds_override_parsed_and_strip_phrase():
    p = plan_query("keys 10 minutes ago", now=NOW, time_from=1.0, time_to=2.0)
    assert (p.time_from, p.time_to) == (1.0, 2.0)
    assert p.explicit_time and p.search_text == "keys"


def test_no_time_expression():
    p = plan_query("where are my keys", now=NOW)
    assert not p.has_time and p.time_hint is None
    assert p.search_text == "keys"


@pytest.mark.parametrize(
    "q,t",
    [
        ("what did the sign say", "read_text"),
        ("how much was the oat milk", "read_text"),
        ("what's the wifi password", "read_text"),
        ("where did I leave my keys", "locate_object"),
        ("where's my phone", "locate_object"),
        ("who was I talking to", "people"),
        ("what was I doing", "activity"),
        ("coffee machine", "generic"),
    ],
)
def test_detect_type(q, t):
    assert detect_type(q) == t


def test_rewrite_keeps_content_words():
    assert rewrite("Hey, do you remember where I put the blue notebook?") == "blue notebook"
    assert rewrite("what did the whiteboard say about the demo") == "whiteboard demo"


def test_weights_follow_type():
    p = plan_query("what did the sign say", now=NOW)
    assert p.weights.bm25 > p.weights.text > p.weights.image
    assert p.mode == "semantic"


def test_parse_anchor_none():
    assert parse_anchor("where are my keys") is None


async def test_async_rewriter_hook():
    from megamind.retrieval.planner import plan_query_async

    async def llm(q: str) -> str:
        return "house keys keychain"

    p = await plan_query_async("where are my keys", rewriter=llm, now=NOW)
    assert p.search_text == "house keys keychain"

    async def broken(q: str) -> str:
        raise RuntimeError("down")

    p = await plan_query_async("where are my keys", rewriter=broken, now=NOW)
    assert p.search_text == "keys"


def test_routing_pure_time_questions_stay_on_hybrid():
    for q in ("what was I doing 10 minutes ago?", "what did I do after lunch", "what was I doing at 9"):
        assert plan_query(q, now=NOW).route == "hybrid", q
    for q in ("where are my keys", "what did the whiteboard say this morning", "who did I meet at lunch"):
        assert plan_query(q, now=NOW).route == "ucc", q


def test_meal_as_subject_searches_for_food():
    p = plan_query("what did I have for lunch", now=NOW)
    assert p.mode == "hybrid" and p.route == "ucc"
    assert "lunch" in p.search_text and "food" in p.search_text
    assert p.time_from == _local(2026, 11, 7, 11, 30)  # 10:30 now: today's lunch hasn't happened yet
    p = plan_query("what did I do after lunch", now=NOW)
    assert p.search_text == "" and p.mode == "temporal" and p.time_focus == "start"


@pytest.mark.parametrize(
    "q", ["what did I just see?", "what was I looking at?", "what was that?", "what just happened", "what was I doing?"]
)
def test_contentless_question_without_time_is_recent_mode(q):
    p = plan_query(q, now=NOW)
    assert p.search_text == "" and not p.has_time
    assert p.mode == "recent" and p.route == "hybrid"


def test_contentless_question_with_time_stays_temporal():
    p = plan_query("what happened a minute ago?", now=NOW)
    assert p.search_text == "" and p.has_time and p.mode == "temporal" and p.route == "hybrid"


@pytest.mark.parametrize(
    "q",
    ["what did I see?", "what did I notice?", "latest views", "my surroundings", "recent sights", "what's going on"],
)
def test_generic_perception_words_alone_are_recent_mode(q):
    p = plan_query(q, now=NOW)
    assert p.search_text == "" and p.mode == "recent" and p.route == "hybrid"


@pytest.mark.parametrize("q", ["where is my camera", "what did I see on the whiteboard", "recent emails"])
def test_generic_words_next_to_a_subject_keep_the_subject(q):
    p = plan_query(q, now=NOW)
    assert p.search_text and p.mode != "recent"
