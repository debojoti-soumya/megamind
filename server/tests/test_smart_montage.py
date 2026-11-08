"""Content-filtered ("smart") montages: subject detection, verification filtering, zero-match honesty, and the
classic time-range path left unchanged. Offline: a fake Anthropic client verifies, the render is stubbed."""

import io
import json
import random
import re
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from megamind.agent import smart_montage as smart
from megamind.agent.montage import MontagePlan, MontageSettings, PlannedClip
from megamind.agent.smart_montage import MontageVerifier, Pick, Verdict, detect_subject, select_mode
from megamind.config import Settings
from megamind.main import create_app
from megamind.models import MemoryToken

T0 = 1_731_000_000.0
COAT = "montage of all shots of the guy in the black coat"


def _jpeg(seed: int) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", (96, 72), (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _tok(i: int, caption: str, ts: float | None = None, thumb: str = "", device="dev1") -> MemoryToken:
    return MemoryToken(
        id=f"t{i:03d}",
        device_id=device,
        ts=T0 + 300 * i if ts is None else ts,
        caption=caption,
        thumb_path=thumb,
        frame_hash=f"{i:016x}",
    )


# ---- subject detection ----


@pytest.mark.parametrize(
    "query,phrase,cue",
    [
        (COAT, "the guy in the black coat", True),
        ("Make a supercut of coffee cups from the past 3 hours", "coffee cups", True),
        ("every time I was at the cafe today", "I was at the cafe", True),
        ("show me all the photos with my dog", "my dog", True),
        ("clips of the red car on Monday", "the red car", True),
        ("desk scene", "desk scene", False),
        ("make a montage of my afternoon at the park", "the park", False),  # no orphan "my at the park"
        ("make a video of this morning in the kitchen", "the kitchen", False),
        ("montage of all shots of the guy in the black coat yesterday", "the guy in the black coat", True),
        ("montage of my trip to Paris last week", "my trip to Paris", False),  # plain "montage of": no cue
    ],
)
def test_detect_subject(query, phrase, cue):
    s = detect_subject(query)
    assert s is not None and s.phrase == phrase and s.cue is cue
    assert phrase.split()[-1] in s.search  # the retriever query keeps the time phrase for its own planner


@pytest.mark.parametrize(
    "query",
    ["make me a montage of my afternoon", "my day", "highlights of this period", "the past few hours", "all of today",
     "montage of yesterday", "montage of everything since 2pm", "a reel of 3:30 to 5pm", "every morning this week",
     "montage of all day", "", "   "],
)  # fmt: skip
def test_time_only_requests_have_no_subject(query):
    assert detect_subject(query) is None


def test_select_mode():
    assert select_mode("", "", "auto") is None  # time-range-only request: always classic
    assert select_mode("", "my afternoon", "auto") is None
    assert select_mode("", "desk scene", "auto") is None  # subject but no collection cue: classic (pre-smart)
    assert select_mode("", "what happened at work", "auto") is None
    assert select_mode("", "reel of my run", "auto") is None  # an episode: the classic highlight reel
    assert select_mode("", "montage of all day at the beach", "auto") is None  # "all day" is a span, not a cue
    assert select_mode(None, COAT, "auto").phrase == "the guy in the black coat"  # collection cue
    assert select_mode("", "every time I was at the cafe", "auto").phrase == "I was at the cafe"
    assert select_mode("classic", COAT, "auto") is None
    assert select_mode("auto", COAT, "off") is None
    assert select_mode("smart", "desk scene", "off").phrase == "desk scene"
    assert select_mode("smart", "my afternoon", "auto") is None  # nothing to filter by


def test_spread_caps_one_long_visit():
    visit = [_tok(i, "coat", ts=T0 + i) for i in range(10)]  # one visit, best-first
    other = [_tok(20 + i, "coat", ts=T0 + 10_000 * (i + 1)) for i in range(3)]
    out = smart.spread(visit + other, 6, gap_s=90)
    assert [t.id for t in out] == ["t000", "t001", "t002", "t020", "t021", "t022"]
    assert len(smart.spread(visit, 6, gap_s=90)) == 6  # second pass fills by rank


# ---- verifier ----


class _FakeMessages:
    """Matches a candidate iff its caption mentions 'black coat'; records every call."""

    def __init__(self, fail_batches=(), drop_labels=(), exc=None, stop_reason="end_turn"):
        self.calls, self.fail_batches, self.drop_labels, self.exc = [], set(fail_batches), set(drop_labels), exc
        self.stop_reason = stop_reason

    async def create(self, **kw):
        self.calls.append(kw)
        if self.exc or len(self.calls) - 1 in self.fail_batches:
            raise self.exc or RuntimeError("overloaded")
        verdicts = []
        for b in kw["messages"][0]["content"]:
            m = re.match(r"\[(C\d+)\] .*caption: (.*)", b.get("text", "")) if b["type"] == "text" else None
            if m and m.group(1) not in self.drop_labels:
                hit = "black coat" in m.group(2)
                verdicts.append({"label": m.group(1), "match": hit, "confidence": 0.9 if hit else 0.8, "title": "Coat"})
        text = json.dumps({"verdicts": verdicts})
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=text)])


def _verifier(messages, **cfg) -> MontageVerifier:
    return MontageVerifier(
        Settings(_env_file=None), MontageSettings(_env_file=None, **cfg), client=SimpleNamespace(messages=messages)
    )


async def test_verifier_batches_with_images_and_filters():
    cands = [_tok(i, "a man in a black coat" if i % 3 == 0 else "an empty hallway") for i in range(20)]
    msgs = _FakeMessages()

    async def load(t):
        return _jpeg(int(t.id[1:]))

    verdicts, stats = await _verifier(msgs).verify("the guy in the black coat", cands, load)
    assert len(msgs.calls) == 3 and stats == {"verify_calls": 3, "verify_errors": 0}  # 8 + 8 + 4
    call = msgs.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    imgs = [b for b in call["messages"][0]["content"] if b["type"] == "image"]
    assert len(imgs) == 8 and imgs[0]["source"]["media_type"] == "image/jpeg" and imgs[0]["source"]["type"] == "base64"
    assert "Subject: the guy in the black coat" in call["messages"][0]["content"][0]["text"]
    picks = smart.keep_matches(cands, verdicts, 0.5)
    assert [p.token.id for p in picks] == [t.id for t in cands if int(t.id[1:]) % 3 == 0]
    assert all(p.verified for p in picks)


async def test_verifier_failed_batch_dropped_and_skipped_label_dropped():
    cands = [_tok(i, "a man in a black coat") for i in range(10)]
    verdicts, stats = await _verifier(_FakeMessages(fail_batches={1}, drop_labels={"C2"})).verify("x", cands)
    assert stats["verify_errors"] == 1
    assert verdicts[1].match is False  # shown to the model and not confirmed
    assert verdicts[8] is None and verdicts[9] is None  # second batch's call failed: unchecked
    picks = smart.keep_matches(cands, verdicts, 0.5)
    assert picks and all(p.verified for p in picks)  # once something was checked, unchecked ones are not mixed in
    assert not {"t001", "t008", "t009"} & {p.token.id for p in picks}
    all_failed = smart.keep_matches(cands, [None] * len(cands), 0.5)  # nothing checked: offline rule
    assert [p.token.id for p in all_failed] == [t.id for t in cands] and not any(p.verified for p in all_failed)


async def test_verifier_truncated_or_refused_output_counts_as_unchecked():
    cands = [_tok(i, "a man in a black coat") for i in range(3)]
    for reason in ("max_tokens", "refusal"):
        verdicts, stats = await _verifier(_FakeMessages(stop_reason=reason)).verify("x", cands)
        assert verdicts == [None, None, None] and stats["verify_errors"] == 1


async def test_verifier_low_confidence_and_offline():
    cands = [_tok(1, "black coat"), _tok(2, "black coat")]
    v = [Verdict(label="C1", match=True, confidence=0.3), Verdict(label="C2", match=True, confidence=0.7)]
    assert [p.token.id for p in smart.keep_matches(cands, v, 0.5)] == ["t002"]
    offline = MontageVerifier(Settings(anthropic_api_key="", _env_file=None), MontageSettings(_env_file=None))
    assert not offline.enabled
    verdicts, stats = await offline.verify("x", cands)
    assert verdicts == [None, None] and stats["verify_calls"] == 0
    keyed_off = MontageVerifier(
        Settings(anthropic_api_key="k", _env_file=None), MontageSettings(verifier="offline", _env_file=None)
    )
    assert not keyed_off.enabled


def test_pick_moments_groups_orders_and_caps():
    def p(i, ts, score):
        return Pick(_tok(i, "c", ts=ts), "", score, True)

    picks = [p(1, T0 + 1000, 1.5), p(2, T0, 1.2), p(3, T0 + 30, 1.9), p(4, T0 + 5000, 1.1)]
    chosen, n = smart.pick_moments(picks, 12, gap_s=90)
    assert n == 3 and [c.token.id for c in chosen] == ["t003", "t001", "t004"]  # best of each moment, by time
    many = [p(i, T0 + 1000 * i, 1.5) for i in range(30)]
    chosen, n = smart.pick_moments(many, 12, gap_s=90)
    assert n == 30 and len(chosen) == 12 and chosen[0].token.id == "t000" and chosen[-1].token.id == "t029"


def test_narration_and_no_match_text():
    assert smart.narration("the guy in the black coat", 3, 3, 1, True) == (
        "Found 3 moments matching the guy in the black coat."
    )
    assert "here are 12 of them" in smart.narration("x", 20, 12, 2, True) and "across 2 days" in smart.narration(
        "x", 20, 12, 2, True
    )
    assert "not visually checked" in smart.narration("x", 5, 5, 1, False)
    assert smart.spoken("I was at my desk") == "you were at your desk"
    assert "couldn't finish checking your memories for your dog" in smart.unfinished_message("my dog")
    msg = smart.no_match_message("the guy in the black coat", 37, False)
    assert msg.startswith("I couldn't find any moments matching the guy in the black coat") and "37" in msg


# ---- HTTP ----


@pytest.fixture
async def app_client(tmp_path, monkeypatch):
    settings = Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend="memory",
        embed_provider="mock",
        consolidate_enabled=False,
        anthropic_api_key="",
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        monkeypatch.setattr("megamind.agent.montage.find_ffmpeg", lambda *_: "ffmpeg")
        rendered = []

        def _render(frames, p, *_a):  # stubbed ffmpeg: a placeholder mp4
            rendered.append(list(frames))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00\x00\x00\x18ftypmp42")

        app.state.montage_cache._render = _render
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app, rendered


COAT_IDX = (1, 4, 5, 9)  # t004 and t005 are 300 s apart: separate moments (gap 90 s)


async def _seed(app, n=12):
    st = app.state
    for i in range(n):
        p = st.settings.thumb_dir / f"t{i:03d}.jpg"
        p.write_bytes(_jpeg(i))
        cap = "a man in a black coat by the door" if i in COAT_IDX else f"a desk with a laptop, scene {i}"
        await st.store.add(_tok(i, cap, thumb=str(p)))


class _StubPlanner:
    def __init__(self):
        self.calls = []

    async def plan(self, request, cands, n_clips):
        self.calls.append((request, len(cands), n_clips))
        return MontagePlan(
            title="Desk day", narration="Two.", clips=[PlannedClip(label="S2", title="Second")]
        ), "claude"


async def test_smart_montage_keeps_only_verified_matches_chronologically(app_client):
    c, app, rendered = app_client
    await _seed(app)
    app.state.montage_verifier = _verifier(msgs := _FakeMessages())
    app.state.montage_planner = planner = _StubPlanner()
    r = await c.post("/v1/montage", json={"query": COAT, "device_id": "", "time_from": "", "max_clips": 0, "mode": ""})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["mode"] == "smart" and j["planner"] == "claude" and j["subject"] == "the guy in the black coat"
    assert [cl["token_id"] for cl in j["clips"]] == [f"t{i:03d}" for i in COAT_IDX]
    assert j["matches"] == 4 and 4 <= j["candidates"] <= 12
    assert j["narration"] == "Found 4 moments matching the guy in the black coat."
    assert j["title"] == "The guy in the black coat"
    assert {cl["title"] for cl in j["clips"]} == {"Coat"}
    assert planner.calls == [] and msgs.calls and len(rendered) == 1
    assert (await c.get(j["url"])).status_code == 200


async def test_smart_montage_caps_clips(app_client):
    c, app, _ = app_client
    await _seed(app)
    app.state.montage_verifier = _verifier(_FakeMessages())
    r = await c.post("/v1/montage", json={"query": COAT, "max_clips": 2})
    j = r.json()
    assert r.status_code == 200 and [cl["token_id"] for cl in j["clips"]] == ["t001", "t009"]  # evenly spread
    assert "here are 2 of them" in j["narration"]
    app.state.montage_settings.smart_max_clips = 3
    j = (await c.post("/v1/montage", json={"query": COAT})).json()
    assert len(j["clips"]) == 3


async def test_smart_montage_zero_matches_is_honest_404(app_client):
    c, app, rendered = app_client
    await _seed(app)
    app.state.montage_verifier = _verifier(_FakeMessages())
    r = await c.post("/v1/montage", json={"query": "montage of every shot of the red umbrella"})
    assert r.status_code == 404
    assert r.json()["detail"].startswith("I couldn't find any moments matching the red umbrella in your memories")
    assert rendered == [] and not list(app.state.montage_cache.dir.glob("*.mp4"))


async def test_partial_verifier_failure_with_no_match_is_503_not_none(app_client):
    c, app, rendered = app_client
    await _seed(app)
    no_verdicts = {f"C{i}" for i in range(1, 61)}  # every checked candidate comes back unconfirmed
    app.state.montage_verifier = _verifier(msgs := _FakeMessages({1}, no_verdicts), verify_batch=2)
    r = await c.post("/v1/montage", json={"query": COAT})
    assert len(msgs.calls) >= 2
    assert r.status_code == 503 and "couldn't finish checking" in r.json()["detail"] and rendered == []


async def test_no_match_with_time_phrase_says_that_period(app_client):
    c, app, _ = app_client
    await _seed(app)
    app.state.montage_verifier = _verifier(_FakeMessages())
    r = await c.post("/v1/montage", json={"query": "every shot of the red umbrella last week"})
    assert r.status_code == 404 and "in that period" in r.json()["detail"]


async def test_smart_montage_respects_time_range(app_client):
    c, app, _ = app_client
    await _seed(app)
    app.state.montage_verifier = _verifier(_FakeMessages())
    body = {"query": COAT, "time_from": T0 + 300 * 3, "time_to": T0 + 300 * 6}
    j = (await c.post("/v1/montage", json=body)).json()
    assert [cl["token_id"] for cl in j["clips"]] == ["t004", "t005"]
    body = {"query": COAT, "time_from": T0 + 300 * 10, "time_to": T0 + 300 * 11}
    r = await c.post("/v1/montage", json=body)
    assert r.status_code == 404 and "in that period" in r.json()["detail"]


async def test_smart_montage_offline_keeps_retrieval_order(app_client):
    c, app, _ = app_client
    await _seed(app)
    assert not app.state.montage_verifier.enabled  # no ANTHROPIC_API_KEY
    app.state.montage_planner = planner = _StubPlanner()
    r = await c.post("/v1/montage", json={"query": COAT})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["mode"] == "smart" and j["planner"] == "offline" and j["matches"] == 0
    assert "not visually checked" in j["narration"] and planner.calls == []
    ts = [cl["ts"] for cl in j["clips"]]
    assert ts == sorted(ts) and 1 <= len(ts) <= 12


async def test_all_verifier_calls_failing_falls_back_to_offline(app_client):
    c, app, _ = app_client
    await _seed(app)
    app.state.montage_verifier = _verifier(_FakeMessages(exc=RuntimeError("down")))
    j = (await c.post("/v1/montage", json={"query": COAT})).json()
    assert j["mode"] == "smart" and j["planner"] == "offline" and j["clips"]


async def test_time_range_montage_unchanged_even_with_verifier(app_client):
    c, app, _ = app_client
    await _seed(app)
    app.state.montage_verifier = _verifier(msgs := _FakeMessages())
    app.state.montage_planner = planner = _StubPlanner()
    r = await c.post("/v1/montage", json={"time_from": T0 - 1, "time_to": T0 + 300 * 12, "query": "", "mode": ""})
    assert r.status_code == 200, r.text
    j = r.json()
    assert planner.calls == [("highlights of this period", 12, 8)] and msgs.calls == []
    assert j["mode"] == "classic" and j["planner"] == "claude" and j["title"] == "Desk day"
    assert j["subject"] == "" and j["candidates"] == 0 and j["matches"] == 0
    assert [cl["token_id"] for cl in j["clips"]] == ["t001"]
    # a subject-less query keeps the classic path (here: the retriever finds nothing -> the classic 404)
    r = await c.post("/v1/montage", json={"query": "make me a highlight reel"})
    assert r.status_code == 404 and r.json()["detail"] == "no memories match that query / time range"
    # an explicit classic request with a subject: classic planner
    planner.calls.clear()
    r = await c.post("/v1/montage", json={"query": COAT, "mode": "classic"})
    assert r.status_code == 200 and r.json()["mode"] == "classic" and len(planner.calls) == 1
    assert msgs.calls == []


async def test_plain_query_stays_classic_with_or_without_key(app_client):
    c, app, _ = app_client
    await _seed(app)
    app.state.montage_planner = planner = _StubPlanner()
    r = await c.post("/v1/montage", json={"query": "desk scene"})
    assert r.status_code == 200 and r.json()["mode"] == "classic" and len(planner.calls) == 1
    app.state.montage_verifier = _verifier(msgs := _FakeMessages())  # a configured verifier changes nothing here
    r = await c.post("/v1/montage", json={"query": "desk scene"})
    assert r.status_code == 200 and r.json()["mode"] == "classic" and len(planner.calls) == 2 and msgs.calls == []
    r = await c.post("/v1/montage", json={"query": "desk scene", "mode": "smart"})
    assert r.status_code == 200 and r.json()["mode"] == "smart"


async def test_mode_validation(app_client):
    c, app, _ = app_client
    r = await c.post("/v1/montage", json={"query": COAT, "mode": "fancy"})
    assert r.status_code == 422


async def test_smart_setting_off(app_client):
    c, app, _ = app_client
    await _seed(app)
    app.state.montage_settings.smart = "off"
    app.state.montage_planner = planner = _StubPlanner()
    r = await c.post("/v1/montage", json={"query": COAT})
    assert r.status_code == 200 and r.json()["mode"] == "classic" and len(planner.calls) == 1
