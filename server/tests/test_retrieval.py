"""Pipeline-level retrieval: frames go in through POST /v1/frames (filter -> enrich -> embed -> store) and
each query must return the exact token id the server assigned, among hard-negative lookalikes."""

import io
import json
import random
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from megamind.config import Settings
from megamind.main import create_app
from megamind.models import MemoryToken
from megamind.store.lancedb_store import LanceDBStore

T0 = 1_750_000_000.0

# (transcript, query) pairs; every transcript has a near-duplicate decoy elsewhere in the list.
SPEECH = [
    ("remind me to call mom about the flight tickets to denver", "what did I need to call mom about"),
    ("remind me to call dad about the train tickets to boston", "call dad train tickets"),
    ("the wifi password at the cafe is falcon-2026", "falcon-2026"),
    ("the wifi password at the hotel is eagle-7781", "wifi password at the hotel"),
    ("we parked on level 3 spot B12", "parked on level 3"),
    ("we parked on level 5 spot C7", "which spot on level 5"),
    ("dentist appointment moved to tuesday 4pm", "dentist appointment moved"),
    ("haircut appointment is on friday 10am", "haircut"),
    ("sam said the demo is on friday", "what did sam say"),
    ("priya said the launch slipped to march", "what did priya say about the launch"),
    ("honestly the pasta here is way too salty", "which food was too salty"),
    ("the soup here is way too cold", "the soup was cold"),
]


def _jpeg(seed: int, tint: tuple[int, int, int] | None = None) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", (96, 72))
    if tint:
        px = [tuple(max(0, min(255, c + rnd.randint(-30, 30))) for c in tint) for _ in range(96 * 72)]
    else:
        px = [(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(96 * 72)]
    img.putdata(px)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(params=["memory", "lancedb"])
async def client(request, tmp_path):
    settings = Settings(
        thumb_dir=tmp_path / "thumbs",
        chunk_dir=tmp_path / "chunks",
        audio_dir=tmp_path / "audio",
        lancedb_path=tmp_path / "ldb",
        store_backend=request.param,
        embed_provider="mock",
        image_embed_provider="mock",
        enrich_provider="mock",
        age_interval_s=0,
        _env_file=None,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, app


async def _post(c, jpeg: bytes, ts: float, **meta) -> str:
    r = await c.post(
        "/v1/frames",
        files={"image": ("f.jpg", jpeg, "image/jpeg")},
        data={"meta": json.dumps({"device_id": "dev1", "ts": ts, **meta})},
    )
    body = r.json()
    assert r.status_code == 202 and body["accepted"], body
    return body["frame_id"]


async def _search(c, q: str, **params) -> list[str]:
    r = await c.get("/v1/memories", params={"q": q, "k": 5, **params})
    assert r.status_code == 200
    return [h["id"] for h in r.json()["results"]]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


async def test_speech_tokens_return_exact_id(client):
    c, app = client
    ids = [await _post(c, _jpeg(i), T0 + i * 15, transcript=t) for i, (t, _) in enumerate(SPEECH)]
    await app.state.worker.queue.join()
    assert await app.state.store.count() == len(SPEECH)
    for fid, (transcript, q) in zip(ids, SPEECH, strict=True):
        got = await _search(c, q)
        assert got[0] == fid, f"{q!r}: expected {transcript!r}"
        tok = await app.state.store.get(fid)
        assert tok.transcript == transcript and tok.trigger == "speech" and tok.text_vec and tok.image_vec


async def test_time_window_picks_the_right_day(client):
    c, app = client
    day = 86_400
    a = await _post(c, _jpeg(100), T0, transcript="my gym locker number is 41")
    b = await _post(c, _jpeg(101), T0 + day, transcript="my gym locker number is 17")
    await app.state.worker.queue.join()
    assert set(await _search(c, "gym locker number")) == {a, b}
    assert (await _search(c, "gym locker number", **{"from": _iso(T0 + day - 60), "to": _iso(T0 + day + 60)})) == [b]
    assert (await _search(c, "gym locker number", to=_iso(T0 + 60))) == [a]


async def test_visual_only_tokens(client):
    c, app = client
    colours = {"red": (230, 30, 30), "green": (30, 170, 30), "blue": (30, 30, 230)}
    ids = {
        name: await _post(c, _jpeg(200 + i, rgb), T0 + i * 10, force=True)
        for i, (name, rgb) in enumerate(colours.items())
    }
    await app.state.worker.queue.join()
    for name, fid in ids.items():
        assert (await _search(c, f"something {name}", mode="visual"))[0] == fid


async def test_float16_vectors_keep_identity(tmp_path):
    """Every token must be its own nearest neighbour after the float16 round-trip, among 300 random vectors."""
    rnd = random.Random(7)

    def unit(dim):
        v = [rnd.gauss(0, 1) for _ in range(dim)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    store = LanceDBStore(tmp_path / "ldb", text_dim=384, image_dim=768)
    toks = [
        MemoryToken(
            id=f"t{i}",
            device_id="d",
            ts=T0 + i,
            thumb_path="x",
            frame_hash=f"{i:016x}",
            text_vec=unit(384),
            image_vec=unit(768),
        )
        for i in range(300)
    ]
    await store.add_many(toks)
    for t in toks[::15]:
        back = await store.get(t.id)
        assert max(abs(a - b) for a, b in zip(back.text_vec, t.text_vec, strict=True)) < 1e-3
        assert (await store.search(t.text_vec, None, "", k=1, mode="semantic"))[0].id == t.id
        assert (await store.search(None, t.image_vec, "", k=1, mode="visual"))[0].id == t.id


async def test_continuous_speech_is_merged_and_searchable(client):
    c, app = client
    first = await _post(c, _jpeg(900), T0, transcript="okay then we need a stopwatch")
    r = await c.post(
        "/v1/frames",
        files={"image": ("f.jpg", _jpeg(901), "image/jpeg")},
        data={"meta": json.dumps({"device_id": "dev1", "ts": T0 + 3, "transcript": "the clapperboard is upstairs"})},
    )
    assert r.json()["reason"] == "merged"
    await app.state.worker.queue.join()
    r = await c.post(
        "/v1/frames",
        files={"image": ("f.jpg", _jpeg(902), "image/jpeg")},
        data={"meta": json.dumps({"device_id": "dev1", "ts": T0 + 6, "transcript": "screwdriver on the second floor"})},
    )
    assert r.json()["reason"] == "merged"  # merged after the token was stored: re-embedded in place
    for q in ("clapperboard upstairs", "screwdriver second floor", "stopwatch"):
        hits = (await c.get("/v1/memories", params={"q": q, "k": 3})).json()["results"]
        assert hits[0]["id"] == first, q
    assert await app.state.store.count() == 1
