"""Photo montage (collage): gather -> verify -> one photo per moment -> one JPEG; the HTTP endpoints, the purge
hook, and the chat agent's make_collage tool. Offline: stub verifier, scripted Anthropic client."""

import io
import json
import random

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from test_claude_agent import ScriptedAnthropic, _message, _settings, _store_with_tokens, text

from megamind.agent.claude_agent import ClaudeAgent
from megamind.agent.collage import CollageError, grid, wants_collage
from megamind.agent.reader import CaptionOnlyReader
from megamind.agent.smart_montage import Verdict
from megamind.agent.tools import RecallMemory
from megamind.config import Settings
from megamind.ingest.enrich import MockTextEmbedder
from megamind.main import create_app
from megamind.models import CollageImage, CollageResponse, MemoryToken

T0 = 1_731_000_000.0
COAT_IDX = (1, 4, 5, 9)


def _jpeg(seed: int) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", (160, 120), (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
async def app_client(tmp_path):
    settings = Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        store_backend="memory",
        embed_provider="mock",
        consolidate_enabled=False,
        anthropic_api_key="",
        tts_provider="mock",
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        st = app.state
        for i in range(12):
            p = st.settings.thumb_dir / f"t{i:03d}.jpg"
            p.write_bytes(_jpeg(i))
            cap = "a man in a black coat by the door" if i in COAT_IDX else f"a desk with a laptop, scene {i}"
            tok = MemoryToken(
                id=f"t{i:03d}",
                device_id="dev1",
                ts=T0 + 300 * i,
                caption=cap,
                thumb_path=str(p),
                frame_hash=f"{i:016x}",
            )
            await st.store.add(tok)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app


class _StubVerifier:
    """Confirms the coat captions, like Haiku would."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    async def verify(self, subject, cands, load_image=None):
        self.calls += 1
        if self.fail:
            return [None] * len(cands), {"verify_calls": 1, "verify_errors": 1}
        assert all([await load_image(t) for t in cands])
        out = [Verdict(label=f"C{i}", match="coat" in t.caption, confidence=0.9, title="Man in black coat")
               for i, t in enumerate(cands)]  # fmt: skip
        return out, {"verify_calls": 1, "verify_errors": 0}


def test_grid_and_cue():
    assert [grid(n) for n in (1, 2, 4, 5, 9, 10)] == [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3)]
    assert wants_collage("show me pictures of my dog")
    assert wants_collage("make a collage of every coffee I had")
    assert wants_collage("photo montage of the beach")
    assert not wants_collage("where did I leave my keys?")
    assert not wants_collage("what did the whiteboard say")


async def test_collage_verified_matches_one_jpeg(app_client):
    c, app = app_client
    app.state.montage_verifier = _StubVerifier()
    r = await c.post("/v1/collage", json={"query": "pictures of the guy in the black coat"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["verified"] and j["subject"] == "the guy in the black coat"
    assert [i["token_id"] for i in j["images"]] == [f"t{i:03d}" for i in COAT_IDX]  # chronological, one per moment
    assert j["images"][0]["title"] == "Man in black coat" and j["images"][0]["thumb_url"] == "/v1/thumbs/t001.jpg"
    assert "4 moments" in j["narration"]
    img = await c.get(j["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    im = Image.open(io.BytesIO(img.content))
    assert im.size == (j["width"], j["height"]) and j["width"] > j["height"] > 0  # 2x2 grid + header
    again = await c.post("/v1/collage", json={"query": "pictures of the guy in the black coat"})
    assert again.json()["id"] == j["id"]  # same picks -> same cached image


async def test_collage_caps_images(app_client):
    c, app = app_client
    app.state.montage_verifier = _StubVerifier()
    r = await c.post("/v1/collage", json={"query": "the black coat", "max_images": 2})
    assert r.status_code == 200 and len(r.json()["images"]) == 2 and r.json()["matches"] == 4


async def test_collage_offline_is_unverified(app_client):
    c, app = app_client
    app.state.montage_verifier = None
    r = await c.post("/v1/collage", json={"query": "man in a black coat"})
    assert r.status_code == 200
    j = r.json()
    assert not j["verified"] and 1 <= len(j["images"]) <= 9 and "not visually checked" in j["narration"]


async def test_collage_no_match_and_verifier_failure(app_client):
    c, app = app_client
    app.state.montage_verifier = _StubVerifier()
    r = await c.post("/v1/collage", json={"query": "photos of a giraffe"})
    assert r.status_code == 404 and "giraffe" in r.json()["detail"]
    app.state.montage_verifier = _StubVerifier(fail=True)  # every call failed -> unverified fallback, not "none"
    r = await c.post("/v1/collage", json={"query": "photos of the black coat"})
    assert r.status_code == 200 and not r.json()["verified"]


async def test_collage_time_range_only_samples_period(app_client):
    c, _ = app_client
    r = await c.post("/v1/collage", json={"query": "photos", "time_from": T0 + 900, "time_to": T0 + 2100})
    assert r.status_code == 200, r.text
    assert {i["token_id"] for i in r.json()["images"]} == {"t003", "t004", "t005", "t006", "t007"}
    r = await c.post("/v1/collage", json={"query": "photos"})
    assert r.status_code == 400


async def test_purge_clears_collages(app_client):
    c, app = app_client
    app.state.montage_verifier = _StubVerifier()
    j = (await c.post("/v1/collage", json={"query": "the black coat"})).json()
    assert (await c.get(j["url"])).status_code == 200
    assert (await c.delete("/v1/memories", params={"device_id": "dev1"})).status_code == 200
    assert (await c.get(j["url"])).status_code == 404
    assert (await c.get("/v1/collages/not-an-id.jpg")).status_code == 404


async def test_chat_without_key_makes_collage(app_client):
    c, app = app_client
    app.state.montage_verifier = _StubVerifier()
    r = await c.post("/v1/chat", json={"session_id": "s", "text": "show me photos of the guy in the black coat"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["tool_calls"][0]["name"] == "make_collage"
    assert j["collage"]["url"].startswith("/v1/collages/") and len(j["collage"]["images"]) == 4
    assert [x["id"] for x in j["cited"]] == [f"t{i:03d}" for i in COAT_IDX]
    plain = (await c.post("/v1/chat", json={"session_id": "s", "text": "where is the laptop?"})).json()
    assert plain["collage"] is None and plain["tool_calls"][0]["name"] == "recall_memory"


@pytest.fixture
async def recall():
    return RecallMemory(await _store_with_tokens(), MockTextEmbedder(), None, CaptionOnlyReader())


def _fake_collage(calls):
    async def run(query, t0, t1, device_id):
        calls.append((query, t0, t1, device_id))
        if query == "giraffe":
            raise CollageError(404, "I couldn't find any photos of giraffe in your memories.")
        imgs = [CollageImage(token_id="t_mug", ts=1_730_999_700.0, title="Blue mug", thumb_url="/v1/thumbs/t_mug.jpg")]
        url = f"/v1/collages/{'a' * 20}.jpg"
        narration = "Found 1 moment matching coffee mugs."
        return CollageResponse(id="a" * 20, url=url, title="Mugs", narration=narration, verified=True, images=imgs)

    return run


async def test_claude_calls_make_collage(recall):
    calls = []
    use = {"type": "tool_use", "id": "toolu_1", "name": "make_collage", "input": {"query": "coffee mug"}}
    script = ScriptedAnthropic(
        [_message([use], stop_reason="tool_use"), _message([text("Here's one moment with your mug.")])]
    )
    agent = ClaudeAgent(_settings(anthropic_api_key="k"), recall, client=script.client(), collage=_fake_collage(calls))
    resp, stats = await agent.chat_with_stats("s", "show me pictures of my coffee mug", "dev1")
    assert calls == [("coffee mug", None, None, "dev1")]
    assert stats.extra.get("prefetch") is None  # a photo request skips the recall prefetch
    first = script.requests[0]
    assert [t["name"] for t in first["tools"]] == ["recall_memory", "make_collage"]
    assert "make_collage" in first["system"]
    assert resp.collage is not None and resp.collage.id == "a" * 20
    assert [c.id for c in resp.cited] == ["t_mug"]
    payload = json.loads(script.requests[1]["messages"][-1]["content"][0]["content"])
    assert payload["confidence"] == "high" and "Photo montage shown" in payload["evidence"]


async def test_cached_system_prompt_keeps_the_collage_instructions(recall):
    script = ScriptedAnthropic([_message([text("ok")])])
    settings = _settings(anthropic_api_key="k", claude_prompt_cache=True)
    agent = ClaudeAgent(settings, recall, client=script.client(), collage=_fake_collage([]))
    await agent.chat_with_stats("s", "show me pictures of my coffee mug", "dev1")
    system = script.requests[0]["system"]
    assert isinstance(system, list) and "make_collage" in system[0]["text"]


async def test_claude_collage_not_found_has_no_collage(recall):
    use = {"type": "tool_use", "id": "toolu_1", "name": "make_collage", "input": {"query": "giraffe"}}
    script = ScriptedAnthropic([_message([use], stop_reason="tool_use"), _message([text("No giraffe photos.")])])
    agent = ClaudeAgent(_settings(anthropic_api_key="k"), recall, client=script.client(), collage=_fake_collage([]))
    resp = await agent.chat("s", "show me pictures of a giraffe", "dev1")
    assert resp.collage is None
    payload = json.loads(script.requests[1]["messages"][-1]["content"][0]["content"])
    assert payload["confidence"] == "not_found"


async def test_no_collage_tool_keeps_requests_unchanged(recall):
    script = ScriptedAnthropic([_message([text("Hi.")])])
    agent = ClaudeAgent(_settings(anthropic_api_key="k"), recall, client=script.client())
    await agent.chat("s", "hello", "dev1")
    assert [t["name"] for t in script.requests[0]["tools"]] == ["recall_memory"]
    assert "make_collage" not in script.requests[0]["system"]


@pytest.mark.parametrize(
    ("query", "phrase"),
    [
        ("a collage of giraffes", "giraffes"),
        ("make a photo montage of every cat", "cat"),
        ("show me images of the red car", "the red car"),
        ("pictures of my dog", "my dog"),
    ],
)
def test_collage_framing_is_not_the_subject(query, phrase):
    from megamind.agent.smart_montage import detect_subject

    assert detect_subject(query).phrase == phrase


def test_unverified_picks_keep_captions_naming_the_subject():
    from megamind.agent.collage import _caption_matches
    from megamind.agent.smart_montage import Pick

    def pick(i, cap):
        return Pick(
            MemoryToken(id=f"p{i}", device_id="d", ts=T0 + i, caption=cap, thumb_path="", frame_hash=f"{i:016x}"),
            "",
            0.5,
            False,
        )

    picks = [pick(0, "A herd of giraffe in a field"), pick(1, "A red stop sign"), pick(2, "Two puppies playing")]
    assert [p.token.id for p in _caption_matches(picks, phrase="giraffes")] == ["p0"]
    assert [p.token.id for p in _caption_matches(picks, phrase="a puppy")] == ["p2"]
    assert _caption_matches(picks, phrase="a zebra") == []  # nothing names it: retrieval noise, not zebras


def test_layout_hero_leads_and_tiles_fit():
    from megamind.agent.collage import layout_tiles

    aspects = [1.5, 0.75, 1.33, 1.0, 1.78, 0.8]
    boxes = layout_tiles(aspects, hero=3, width=1100, row_h=300, gap=10)
    assert sorted(b[0] for b in boxes) == list(range(len(aspects)))
    first = boxes[0]
    assert first[0] == 3 and first[1] == 0 and first[2] == 0
    assert first[3] > 1100 * 0.55
    for _i, x, y, w, h in boxes:
        assert x >= 0 and y >= 0 and x + w <= 1100 and w > 0 and h > 0
    for a in boxes:
        for b in boxes:
            if a is not b:
                assert a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1] or a[2] + a[4] <= b[2] or b[2] + b[4] <= a[2]


def test_choose_hero_prefers_best_match():
    from megamind.agent.collage import Tile, choose_hero

    tok = MemoryToken(id="h", device_id="dev1", ts=T0, caption="x", thumb_path="", frame_hash="0")
    tiles = [Tile(tok, "t", _jpeg(i), s) for i, s in enumerate([0.2, 0.9, 0.5])]
    assert choose_hero(tiles) == 1
    assert choose_hero([]) == 0
