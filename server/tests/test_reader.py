"""VLM reader tests: request layout, chunking/merge, JSON parsing, degradation (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from zoneinfo import ZoneInfo

import httpx
import pytest
from PIL import Image

from megamind.agent.reader import (
    CaptionOnlyReader,
    VLMReader,
    build_reader,
    chunk_chronological,
    parse_reader_json,
    token_header,
)
from megamind.agent.tools import DEGRADED_PREFIX, RecallMemory
from megamind.config import Settings
from megamind.ingest.enrich import MockTextEmbedder
from megamind.models import MemoryToken
from megamind.store.memory_store import InMemoryStore

NOW = 1_731_000_000.0


def _settings(**kw) -> Settings:
    base = dict(reader_base_url="http://reader.test/v1", reader_api_key="sk-test", reader_max_retries=1, _env_file=None)
    base.update(kw)
    return Settings(**base)


def _thumb(tmp_path, name: str, size=(1280, 720)) -> str:
    p = tmp_path / f"{name}.jpg"
    Image.new("RGB", size, (120, 40, 200)).save(p, format="JPEG")
    return str(p)


def _tokens(tmp_path, n: int, start=NOW - 3600, step=300.0, thumbs=True) -> list[MemoryToken]:
    toks = []
    for i in range(n):
        toks.append(
            MemoryToken(
                id=f"t{i:02d}",
                device_id="dev1",
                ts=start + i * step,
                caption=f"caption {i}",
                ocr_text="LOCKER 42" if i == 1 else "",
                entities=["mug"] if i == 2 else [],
                thumb_path=_thumb(tmp_path, f"t{i:02d}") if thumbs else str(tmp_path / "missing.jpg"),
                frame_hash="0" * 16,
            )
        )
    return toks


def _completion(obj: dict | str, usage: dict | None = None) -> dict:
    content = obj if isinstance(obj, str) else json.dumps(obj)
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "google/gemma-4-31B-it",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 1200, "completion_tokens": 60, "total_tokens": 1260},
    }


class Scripted:
    """Replays `responses` in request order, or (`by_chunk`) answers each chunk by its oldest token id: chunks
    build their payloads in worker threads, so their request order is not fixed."""

    def __init__(self, responses=(), by_chunk: dict | None = None):
        self.responses = list(responses)
        self.by_chunk = by_chunk
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.by_chunk is not None:
            first = json.loads(request.content)["messages"][1]["content"][1]["text"][1:4]
            nxt = self.by_chunk[first]
        else:
            nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if isinstance(nxt, int):
            return httpx.Response(nxt, json={"error": {"message": "nope"}})
        return httpx.Response(200, json=nxt)

    def reader(self, settings: Settings | None = None) -> VLMReader:
        settings = settings or _settings()
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.handler),
            base_url=settings.reader_base_url,
            headers={"Authorization": f"Bearer {settings.reader_api_key}"},
        )
        return VLMReader(settings, client=client)

    def body(self, i: int = 0) -> dict:
        return json.loads(self.requests[i].content)


async def test_layout_is_chronological_interleaved_and_cited(tmp_path):
    toks = _tokens(tmp_path, 3)
    shuffled = [toks[2], toks[0], toks[1]]  # retrieval rank order
    s = Scripted(
        [
            _completion(
                {
                    "answer": "Locker 42, about 55 min ago.",
                    "evidence": [{"id": "t01", "why": "sign"}],
                    "confidence": "high",
                }
            )
        ]
    )
    out = await s.reader().read("which locker did I use?", shuffled, now=NOW)

    assert out.evidence == "Locker 42, about 55 min ago."
    assert out.confidence == "high" and out.cited_ids == ["t01"] and out.items[0].why == "sign"
    assert not out.degraded and out.calls == 1 and out.usage["prompt_tokens"] == 1200

    req, body = s.requests[0], s.body()
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "google/gemma-4-31B-it" and body["temperature"] == 0
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["required"] == ["answer", "evidence", "confidence"]
    assert req.headers["idempotency-key"].startswith("megamind-reader-")
    assert body["messages"][0]["role"] == "system"
    parts = body["messages"][1]["content"]
    texts = [p["text"] for p in parts if p["type"] == "text"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 3
    # header, then image, chronological ids
    headers = [t for t in texts if t.startswith("[t")]
    assert [h[1:4] for h in headers] == ["t00", "t01", "t02"]
    assert "first)" in headers[0] and "+5 min after first" in headers[1] and "+10 min after first" in headers[2]
    assert "(1.0 h ago, first)" in headers[0] and "text: LOCKER 42" in headers[1] and "entities: mug" in headers[2]
    assert parts[1]["type"] == "text" and parts[2]["type"] == "image_url" and parts[3]["type"] == "text"
    assert texts[-1].startswith("Question: which locker did I use?")
    uri = images[0]["image_url"]["url"]
    assert uri.startswith("data:image/jpeg;base64,")
    # thumbnail downscaled to <= reader_thumb_max_px (960: the headset upload width)
    import base64

    im = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert max(im.size) <= 960


async def test_missing_thumb_sends_text_only(tmp_path):
    toks = _tokens(tmp_path, 2, thumbs=False)
    s = Scripted([_completion({"answer": "x", "evidence": [{"id": "t00", "why": ""}], "confidence": "medium"})])
    out = await s.reader().read("q", toks)
    parts = s.body()["messages"][1]["content"]
    assert not [p for p in parts if p["type"] == "image_url"]
    assert sum("(image unavailable)" in p["text"] for p in parts) == 2
    assert out.confidence == "medium"


async def test_chunking_over_image_limit_merges(tmp_path):
    toks = _tokens(tmp_path, 24)
    assert [len(c) for c in chunk_chronological(toks, 20)] == [12, 12]
    s = Scripted(
        by_chunk={
            "t00": _completion({"answer": "No matching memory.", "evidence": [], "confidence": "not_found"}),
            "t12": _completion(
                {
                    "answer": "Keys on the counter.",
                    "evidence": [{"id": "t15", "why": "keys visible"}, {"id": "zzz", "why": "hallucinated id"}],
                    "confidence": "high",
                }
            ),
        }
    )
    out = await s.reader().read("keys?", toks)
    assert len(s.requests) == 2
    assert out.confidence == "high" and out.cited_ids == ["t15"] and out.calls == 2
    assert out.evidence == "Keys on the counter."
    assert out.usage["prompt_tokens"] == 2400
    ids_by_req = [
        [p["text"][1:4] for p in s.body(i)["messages"][1]["content"] if p["type"] == "text" and p["text"][:2] == "[t"]
        for i in range(2)
    ]
    assert sorted(ids_by_req) == [[f"t{i:02d}" for i in range(12)], [f"t{i:02d}" for i in range(12, 24)]]


async def test_two_found_chunks_concatenate_best_first(tmp_path):
    toks = _tokens(tmp_path, 22)
    s = Scripted(
        by_chunk={
            "t00": _completion(
                {"answer": "Early: mug on desk.", "evidence": [{"id": "t02", "why": ""}], "confidence": "low"}
            ),
            "t11": _completion(
                {"answer": "Later: mug in sink.", "evidence": [{"id": "t20", "why": ""}], "confidence": "high"}
            ),
        }
    )
    out = await s.reader().read("mug?", toks)
    assert out.confidence == "high"
    assert out.evidence == "Later: mug in sink. Early: mug on desk."
    assert out.cited_ids == ["t20", "t02"]


def test_parse_reader_json_tolerates_fences_and_bad_ids():
    obj = '{"answer": "Yes", "evidence": [{"id": "a", "why": "w"}, {"id": "nope"}, "b"], "confidence": "HIGH"}'
    raw = f"Sure!\n```json\n{obj}\n```"
    answer, conf, items = parse_reader_json(raw, {"a", "b"})
    assert answer == "Yes" and conf == "high" and [i.id for i in items] == ["a", "b"]

    answer, conf, items = parse_reader_json('{"answer": "Maybe", "evidence": [], "confidence": "weird"}', {"a"})
    assert conf == "low" and items == []

    answer, conf, items = parse_reader_json('{"answer": "", "evidence": [], "confidence": "not_found"}', {"a"})
    assert conf == "not_found" and answer == "No matching memory."

    answer, conf, items = parse_reader_json(
        '{"answer": "x", "evidence": [{"id": "a"}], "confidence": "not_found"}', {"a"}
    )
    assert conf == "not_found" and items == []

    with pytest.raises(ValueError):
        parse_reader_json("I cannot help with that.", {"a"})


async def test_timeout_degrades_to_captions(tmp_path):
    toks = _tokens(tmp_path, 2)
    s = Scripted([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")])
    out = await s.reader().read("q", toks)
    assert out.degraded and out.calls == 2
    assert out.confidence == "low" and out.cited_ids == ["t00", "t01"]
    assert out.evidence.startswith("[t00] ") and "caption 0" in out.evidence


async def test_5xx_then_success_retries(tmp_path):
    toks = _tokens(tmp_path, 1)
    s = Scripted([503, _completion({"answer": "ok", "evidence": [{"id": "t00"}], "confidence": "medium"})])
    out = await s.reader(_settings(reader_fallback_model="")).read("q", toks)
    assert not out.degraded and out.calls == 2 and out.evidence == "ok"


async def test_400_downgrades_json_mode_schema_object_off_then_gives_up(tmp_path):
    toks = _tokens(tmp_path, 1)
    s = Scripted([400, 400, 400])
    out = await s.reader().read("q", toks)
    assert s.body(0)["response_format"]["type"] == "json_schema"
    assert s.body(1)["response_format"] == {"type": "json_object"}
    assert "response_format" not in s.body(2)
    assert out.degraded and out.calls == 3
    keys = {r.headers["idempotency-key"] for r in s.requests}
    assert len(keys) == 1  # same key across retries of one chunk


async def test_429_honours_retry_after_then_succeeds(tmp_path, monkeypatch):
    import megamind.agent.reader as mod

    slept: list[float] = []

    async def fake_sleep(x):
        slept.append(x)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    toks = _tokens(tmp_path, 1)

    class S(Scripted):
        def handler(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "slow down"})
            return httpx.Response(
                200, json=_completion({"answer": "ok", "evidence": [{"id": "t00"}], "confidence": "high"})
            )

    s = S([])
    out = await s.reader(_settings(reader_fallback_model="")).read("q", toks)
    assert not out.degraded and out.calls == 2
    assert slept == [2.0]  # capped so the voice answer still arrives


async def test_503_switches_to_fallback_model(tmp_path):
    toks = _tokens(tmp_path, 1)
    s = Scripted([503, _completion({"answer": "ok", "evidence": [{"id": "t00"}], "confidence": "high"})])
    out = await s.reader().read("q", toks)
    assert not out.degraded and out.calls == 2 and out.usage["fallback_model"] == 1
    assert s.body(0)["model"] == "google/gemma-4-31B-it"
    assert s.body(1)["model"] == "nvidia/Gemma-4-31B-IT-NVFP4"


async def test_5xx_on_both_models_degrades(tmp_path):
    toks = _tokens(tmp_path, 1)
    s = Scripted([503, 503, 500])
    out = await s.reader().read("q", toks)
    assert out.degraded and out.calls == 3  # primary, fallback, retry


async def test_garbage_output_degrades(tmp_path):
    toks = _tokens(tmp_path, 1)
    s = Scripted([_completion("I am not JSON"), _completion("still not"), _completion("nope")])
    out = await s.reader().read("q", toks)
    assert out.degraded and out.calls == 3  # primary hands over at once; the fallback gets its retry


async def test_empty_tokens_short_circuits(tmp_path):
    s = Scripted([])
    out = await s.reader().read("q", [])
    assert out.confidence == "not_found" and not s.requests


async def test_extra_body_and_no_key_header(tmp_path):
    toks = _tokens(tmp_path, 1)
    settings = _settings(
        reader_api_key="none",
        reader_json_mode="off",
        reader_completion_window="asap",
        reader_extra_body='{"chat_template_kwargs": {"enable_thinking": false}}',
    )
    s = Scripted([_completion({"answer": "ok", "evidence": [{"id": "t00"}], "confidence": "high"})])
    client = httpx.AsyncClient(transport=httpx.MockTransport(s.handler), base_url=settings.reader_base_url)
    await VLMReader(settings, client=client).read("q", toks)
    body = s.body()
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and "response_format" not in body
    assert s.requests[0].headers["x-sail-completion-window"] == "asap"
    assert "authorization" not in s.requests[0].headers


def test_build_reader_selection():
    assert isinstance(build_reader(Settings(_env_file=None)), CaptionOnlyReader)  # default key "none"
    assert isinstance(build_reader(_settings()), VLMReader)
    assert isinstance(build_reader(_settings(reader_provider="mock")), CaptionOnlyReader)
    assert isinstance(build_reader(_settings(reader_api_key="none", reader_provider="vlm")), VLMReader)


async def test_recall_memory_marks_degraded_evidence(tmp_path):
    store = InMemoryStore()
    emb = MockTextEmbedder()
    for t in _tokens(tmp_path, 2):
        t.text_vec = await emb.embed(t.caption)
        await store.add(t)
    s = Scripted([httpx.ConnectError("down"), httpx.ConnectError("down")])
    recall = RecallMemory(store, emb, None, s.reader())
    r = await recall(query="caption 1")
    assert r.evidence.startswith(DEGRADED_PREFIX)
    assert r.confidence == "low" and r.cited


def test_token_header_uses_wearer_local_time():
    t = MemoryToken(id="t", device_id="d", ts=1_762_644_600.0, caption="c", thumb_path="", frame_hash="0" * 16)
    la = ZoneInfo("America/Los_Angeles")
    assert token_header(t, t.ts, t.ts + 60, la).startswith("[t] Sat 2025-11-08 15:30 PST (1 min ago")
    assert "2025-11-08 23:30:00 UTC" in token_header(t, t.ts, t.ts + 60)


async def test_timeout_retries_once_on_the_fallback_model(tmp_path):
    toks = _tokens(tmp_path, 1)
    ok = _completion({"answer": "ok", "evidence": [{"id": "t00"}], "confidence": "high"})
    s = Scripted([httpx.ReadTimeout("slow"), ok])
    out = await s.reader().read("q", toks)
    assert not out.degraded and out.calls == 2 and out.usage["fallback_model"] == 1
    assert s.body(0)["model"] == "google/gemma-4-31B-it" and s.body(1)["model"] == "nvidia/Gemma-4-31B-IT-NVFP4"


async def test_deadline_caps_total_reader_time(tmp_path):
    """No second attempt once less than a useful attempt's worth of the wall budget is left."""
    toks = _tokens(tmp_path, 1)

    class Slow(Scripted):
        async def handler(self, request):
            self.requests.append(request)
            await asyncio.sleep(0.6)
            raise httpx.ReadTimeout("slow")

    s = Slow([])
    out = await s.reader(_settings(reader_deadline_s=1.5)).read("q", toks)
    assert out.degraded and out.calls == 1 and len(s.requests) == 1


async def test_retry_after_longer_than_the_deadline_degrades_instead_of_sleeping(tmp_path, monkeypatch):
    import megamind.agent.reader as mod

    slept = []

    async def fake_sleep(x):
        slept.append(x)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    toks = _tokens(tmp_path, 1)
    s = Scripted([429, _completion({"answer": "ok", "evidence": [{"id": "t00"}], "confidence": "high"})])
    out = await s.reader(_settings(reader_fallback_model="", reader_deadline_s=1.8)).read("q", toks)
    assert out.degraded and out.calls == 1 and slept == []


class ByModel(Scripted):
    """Async handler: per-model delay (None = hang until cancelled) then a canned answer."""

    def __init__(self, delays: dict[str, float | None], answers: dict[str, object] | None = None):
        super().__init__([])
        self.delays, self.answers = delays, answers or {}

    async def handler(self, request):
        self.requests.append(request)
        model = json.loads(request.content)["model"]
        delay = self.delays[model]
        await asyncio.sleep(3600 if delay is None else delay)
        ans = self.answers.get(model, {"answer": f"from {model}", "evidence": [{"id": "t00"}], "confidence": "high"})
        if isinstance(ans, int):
            return httpx.Response(ans, json={"error": {"message": "nope"}})
        return httpx.Response(200, json=_completion(ans))


PRIMARY, FALLBACK = "google/gemma-4-31B-it", "nvidia/Gemma-4-31B-IT-NVFP4"


def test_default_deadlines_leave_room_for_the_fallback(monkeypatch):
    for var in ("READER_HEDGE_S", "READER_DEADLINE_S", "READER_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    healthy_p50_s, healthy_p95_s = 3.1, 3.25  # Sail Gemma asap, both models (tools/bench_reader.py, issue #85)
    assert s.reader_hedge_s >= healthy_p95_s > healthy_p50_s  # healthy primaries rarely pay for a second call
    assert s.reader_hedge_s + healthy_p95_s + 0.5 <= s.reader_deadline_s <= s.reader_timeout_s


async def test_hung_primary_is_hedged_on_the_fallback_model(tmp_path):
    toks = _tokens(tmp_path, 1)
    s = ByModel({PRIMARY: None, FALLBACK: 0.1})
    t0 = asyncio.get_running_loop().time()
    out = await s.reader(_settings(reader_hedge_s=0.2, reader_deadline_s=2.0, reader_timeout_s=2.0)).read("q", toks)
    elapsed = asyncio.get_running_loop().time() - t0
    assert not out.degraded and out.evidence == f"from {FALLBACK}"
    assert out.usage["fallback_model"] == 1 and out.usage["hedged"] == 1 and out.calls == 2
    assert elapsed < 1.0  # hedge (0.2) + fallback (0.1), not the 2 s deadline
    keys = {r.headers["idempotency-key"] for r in s.requests}
    assert len(keys) == 2  # the hedge is a different request, not a replay of the primary


async def test_primary_finishing_after_the_hedge_still_wins(tmp_path):
    toks = _tokens(tmp_path, 1)
    s = ByModel({PRIMARY: 0.3, FALLBACK: None})
    out = await s.reader(_settings(reader_hedge_s=0.1, reader_deadline_s=2.0, reader_timeout_s=2.0)).read("q", toks)
    assert out.evidence == f"from {PRIMARY}" and out.usage["hedged"] == 1 and "fallback_model" not in out.usage


async def test_fast_primary_never_starts_the_fallback(tmp_path):
    toks = _tokens(tmp_path, 1)
    s = ByModel({PRIMARY: 0.05, FALLBACK: 0.05})
    out = await s.reader(_settings(reader_hedge_s=0.5)).read("q", toks)
    assert out.evidence == f"from {PRIMARY}" and out.calls == 1 and "hedged" not in out.usage


async def test_both_models_hung_degrade_at_the_deadline(tmp_path):
    toks = _tokens(tmp_path, 4)
    s = ByModel({PRIMARY: None, FALLBACK: None})
    t0 = asyncio.get_running_loop().time()
    out = await s.reader(_settings(reader_hedge_s=0.1, reader_deadline_s=1.2, reader_timeout_s=5.0)).read("q", toks)
    assert out.degraded and out.calls == 2
    assert asyncio.get_running_loop().time() - t0 < 1.6


async def test_auth_error_degrades_without_retrying(tmp_path):
    toks = _tokens(tmp_path, 1)
    s = Scripted([401])
    out = await s.reader().read("q", toks)
    assert out.degraded and out.calls == 1 and len(s.requests) == 1


async def test_degraded_evidence_lists_every_hit_with_its_ocr(tmp_path):
    """#68: the answer was in rank 4-5; captions-only used to pass only the top 3."""
    toks = _tokens(tmp_path, 6)
    toks[4].ocr_text = "VISITORS CHECK IN\nAT FRONT DESK"
    s = Scripted([httpx.ConnectError("down")] * 4)
    out = await s.reader(_settings(wearer_tz="America/Los_Angeles")).read("lobby sign", toks)
    assert out.degraded
    lines = out.evidence.splitlines()
    assert len(lines) == 6 and lines[4].startswith("[t04] ")
    assert "(text: VISITORS CHECK IN / AT FRONT DESK)" in lines[4]
    assert out.cited_ids == ["t00", "t01", "t02"]


async def test_caption_only_reader_default_stays_top_3(tmp_path):
    out = await CaptionOnlyReader().read("q", _tokens(tmp_path, 6, thumbs=False))
    assert len(out.evidence.splitlines()) == 3 and out.evidence.startswith("[t00] caption 0")


async def test_image_loader_supplies_cold_tier_frames(tmp_path):
    """Cold frames have no thumb_path; the reader gets their pixels through `image_loader` (MediaStore.read)."""
    toks = [
        t.model_copy(update={"thumb_path": "", "tier": "cold", "chunk_id": "ck_1", "chunk_offset": i})
        for i, t in enumerate(_tokens(tmp_path, 2))
    ]
    jpeg = open(_thumb(tmp_path, "decoded"), "rb").read()
    asked: list[str] = []

    async def loader(t: MemoryToken) -> bytes | None:
        asked.append(t.id)
        return jpeg if t.chunk_offset == 0 else None

    s = Scripted([_completion({"answer": "x", "evidence": [{"id": "t00", "why": ""}], "confidence": "high"})])
    r = s.reader()
    r.image_loader = loader
    await r.read("q", toks, now=NOW)
    parts = s.body()["messages"][1]["content"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert sorted(asked) == ["t00", "t01"] and len(images) == 1
    assert sum("(image unavailable)" in p.get("text", "") for p in parts) == 1


async def test_slow_image_loader_is_bounded_by_the_deadline(tmp_path):
    toks = _tokens(tmp_path, 1)

    async def hung(_t: MemoryToken) -> bytes | None:
        await asyncio.sleep(30)
        return None

    s = Scripted([_completion({"answer": "x", "evidence": [{"id": "t00", "why": ""}], "confidence": "high"})])
    r = s.reader(_settings(reader_deadline_s=2.0, reader_timeout_s=2.0))
    r.image_loader = hung
    out = await asyncio.wait_for(r.read("q", toks, now=NOW), timeout=5)
    assert out.confidence == "high"
    assert "(image unavailable)" in json.dumps(s.body()["messages"][1]["content"])


def test_build_reader_passes_the_image_loader():
    async def loader(_t):
        return None

    r = build_reader(_settings(reader_provider="vlm"), image_loader=loader)
    assert isinstance(r, VLMReader) and r.image_loader is loader


def test_encode_jpeg_is_cached_by_content(tmp_path):
    from megamind.agent.reader import _ENC_CACHE, clear_encode_cache, encode_thumb

    p = tmp_path / "f.jpg"
    Image.new("RGB", (960, 720), (200, 10, 10)).save(p, "JPEG")
    clear_encode_cache()
    a = encode_thumb(str(p), 512)
    assert a and encode_thumb(str(p), 512) == a and len(_ENC_CACHE) == 1
    with Image.open(io.BytesIO(base64.b64decode(a.split(",", 1)[1]))) as im:
        assert im.size == (512, 384)
    Image.new("RGB", (960, 720), (10, 10, 200)).save(p, "JPEG")  # privacy blur rewrites the thumb in place
    b = encode_thumb(str(p), 512)
    assert b not in (None, a) and len(_ENC_CACHE) == 2
    p.unlink()  # purge: never served from the cache
    assert encode_thumb(str(p), 512) is None


async def test_claude_vision_reader_returns_images_best_first(tmp_path):
    from megamind.agent.reader import ClaudeVisionReader, build_reader

    def jpg(name: str, size=(1280, 960)) -> str:
        p = tmp_path / f"{name}.jpg"
        Image.new("RGB", size, (40, 90, 160)).save(p, format="JPEG")
        return str(p)

    tokens = [
        MemoryToken(
            id=f"v{i}",
            device_id="d",
            ts=1000.0 + i,
            caption=f"cap {i}",
            ocr_text="EXIT" if i == 0 else "",
            thumb_path=jpg(f"v{i}") if i != 1 else "",
            frame_hash="0" * 16,
        )
        for i in range(4)
    ]
    settings = Settings(_env_file=None, reader_provider="claude", reader_max_images=3)
    reader = build_reader(settings)
    assert isinstance(reader, ClaudeVisionReader)
    out = await reader.read("exit sign", tokens, now=2000.0)
    assert [i for i, _ in out.images] == ["v0", "v2"]  # rank order, capped at 3, unreadable v1 skipped
    with Image.open(io.BytesIO(base64.b64decode(out.images[0][1].split(",", 1)[1]))) as im:
        assert max(im.size) == 960
    assert out.confidence == "medium" and out.cited_ids == ["v0", "v1", "v2"] and "EXIT" in out.evidence
    empty = await reader.read("x", [])
    assert empty.confidence == "not_found" and not empty.images
