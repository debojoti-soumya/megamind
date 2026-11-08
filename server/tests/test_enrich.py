import base64
import io
import json

import httpx
import pytest
from PIL import Image

from megamind.config import Settings
from megamind.ingest import enrich
from megamind.ingest.enrich import (
    ENRICH_PROMPT,
    Enricher,
    Enrichment,
    ImageEmbedder,
    LocalEnricher,
    MockEnricher,
    VLMEnricher,
    build_enricher,
    parse_enrichment,
    resolve_enrich_provider,
)


def _jpeg(color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buf, format="JPEG")
    return buf.getvalue()


def _settings(**kw) -> Settings:
    base = dict(reader_base_url="http://vlm.test/v1", reader_api_key="sk-test", _env_file=None)
    return Settings(**{**base, **kw})


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


GOOD = {"caption": "Keys on a kitchen counter by the kettle.", "ocr_text": "BOSCH", "entities": ["keys", "kettle"]}


class _Fallback(Enricher):
    def __init__(self) -> None:
        self.calls = 0

    async def enrich(self, jpeg: bytes) -> Enrichment:
        self.calls += 1
        return Enrichment(caption="a kitchen", ocr_text="", entities=[])


def _vlm(handler, fallback=None, **kw) -> VLMEnricher:
    client = httpx.AsyncClient(base_url="http://vlm.test/v1", transport=httpx.MockTransport(handler))
    return VLMEnricher(_settings(**kw), fallback=fallback, client=client, retry_wait_s=0)


# ------------------------------------------------------------------ parsing


def test_parse_enrichment_tolerates_fences_and_prose():
    fenced = "Sure! Here you go:\n```json\n" + json.dumps(GOOD) + "\n```"
    e = parse_enrichment(fenced)
    assert e == Enrichment(caption=GOOD["caption"], ocr_text="BOSCH", entities=["keys", "kettle"])
    e = parse_enrichment('{"caption": "x", "ocr_text": ["A", "B"], "entities": "mug, desk ,"}')
    assert e.ocr_text == "A\nB" and e.entities == ["mug", "desk"]
    e = parse_enrichment(json.dumps({"caption": "x", "entities": [f"e{i}" for i in range(30)]}))
    assert len(e.entities) == 15
    for bad in ("no json here", "{not json}", "[1, 2]", ""):
        with pytest.raises(ValueError):
            parse_enrichment(bad)


# ------------------------------------------------------------------ VLMEnricher


async def test_vlm_enricher_good_response_and_request_shape():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=_completion(json.dumps(GOOD)))

    e = await _vlm(handler, reader_completion_window="asap", reader_extra_body='{"top_k": 1}').enrich(_jpeg())
    assert e.caption == GOOD["caption"] and e.ocr_text == "BOSCH" and e.entities == ["keys", "kettle"]
    (req,) = seen
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert req.headers["x-sail-completion-window"] == "asap"
    body = json.loads(req.content)
    assert body["model"] == "google/gemma-4-31B-it" and body["temperature"] == 0 and body["top_k"] == 1
    text, image = body["messages"][0]["content"]
    assert text == {"type": "text", "text": ENRICH_PROMPT}
    assert image["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(_jpeg()).decode()


async def test_vlm_enricher_fenced_json():
    fenced = "```json\n" + json.dumps(GOOD) + "\n```"
    vlm = _vlm(lambda req: httpx.Response(200, json=_completion(fenced)))
    assert (await vlm.enrich(_jpeg())).caption == GOOD["caption"]
    assert vlm.degraded is None


async def test_vlm_enricher_garbage_falls_back():
    fb = _Fallback()
    vlm = _vlm(lambda req: httpx.Response(200, json=_completion("I cannot help with that.")), fallback=fb)
    assert (await vlm.enrich(_jpeg())).caption == "a kitchen"
    assert fb.calls == 1 and vlm.degraded == "last error unparseable answer"
    no_fb = _vlm(lambda req: httpx.Response(200, json={"unexpected": True}))
    assert await no_fb.enrich(_jpeg()) == Enrichment()  # empty, logged; never raises


async def test_vlm_enricher_retries_5xx_then_falls_back():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(503, json={"error": "overloaded"})

    fb = _Fallback()
    vlm = _vlm(handler, fallback=fb)
    assert (await vlm.enrich(_jpeg())).caption == "a kitchen"
    assert len(calls) == 2 and fb.calls == 1 and vlm.degraded == "last error 503"


async def test_vlm_enricher_retry_recovers_and_clears_degraded():
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=req)
        return httpx.Response(200, json=_completion(json.dumps(GOOD)))

    vlm = _vlm(handler)
    vlm.degraded = "last error 503"
    assert (await vlm.enrich(_jpeg())).caption == GOOD["caption"]
    assert len(calls) == 2 and vlm.degraded is None


async def test_vlm_enricher_4xx_is_not_retried():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(401, json={"error": "bad key"})

    vlm = _vlm(handler)
    assert await vlm.enrich(_jpeg()) == Enrichment()
    assert len(calls) == 1 and vlm.degraded == "last error 401"


async def test_vlm_enricher_separate_endpoint_does_not_get_reader_key():
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=_completion(json.dumps(GOOD)))

    vlm = _vlm(handler, enrich_base_url="http://other.test/v1", reader_extra_body='{"top_k": 1}')
    await vlm.enrich(_jpeg())
    assert "authorization" not in seen[0].headers and "top_k" not in json.loads(seen[0].content)
    assert _settings(enrich_base_url="http://other.test/v1", enrich_api_key="sk-enrich").enrich_key == "sk-enrich"
    assert _settings(reader_api_key="none").enrich_key == ""


# ------------------------------------------------------------------ LocalEnricher


class _StubSigLIP(ImageEmbedder):
    """Red images look like kitchens; everything else like an office."""

    dim = 2

    async def embed_image(self, jpeg: bytes) -> list[float]:
        r, g, b = Image.open(io.BytesIO(jpeg)).convert("RGB").getpixel((0, 0))
        return [1.0, 0.0] if r > 128 else [0.0, 1.0]

    async def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0] if "kitchen" in text else [0.0, 1.0] if "office" in text else [0.1, 0.1]


async def test_local_enricher_zero_shot_caption_without_ocr():
    le = LocalEnricher(_StubSigLIP(), ocr=False)
    assert (await le.enrich(_jpeg((220, 10, 10)))).caption == "a kitchen"
    assert (await le.enrich(_jpeg((10, 10, 220)))).caption == "an office"
    assert le.degraded is None
    assert LocalEnricher(None, ocr=False).degraded  # nothing to run: reported, captions empty
    assert await LocalEnricher(None, ocr=False).enrich(_jpeg()) == Enrichment()


async def test_local_enricher_ocr_only_captions_with_text(monkeypatch):
    monkeypatch.setattr(LocalEnricher, "_ocr", staticmethod(lambda jpeg: "EXIT 42"))
    le = LocalEnricher(None, ocr=True)
    assert le.degraded == "no image embedder: OCR-only captions"
    e = await le.enrich(_jpeg())
    assert e.caption == "Text visible: EXIT 42" and e.ocr_text == "EXIT 42"
    with_scene = await LocalEnricher(_StubSigLIP(), ocr=True).enrich(_jpeg((220, 10, 10)))
    assert with_scene.caption == "a kitchen" and with_scene.ocr_text == "EXIT 42"


async def test_local_enricher_ocr():
    if not enrich.tesseract_available():
        pytest.skip("pytesseract / tesseract not installed (local-enrich extra)")
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", (640, 160), "white")
    ImageDraw.Draw(img).text((20, 40), "EXIT 42", fill="black", font=ImageFont.load_default(size=72))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    e = await LocalEnricher(None).enrich(buf.getvalue())
    assert "EXIT" in e.ocr_text


# ------------------------------------------------------------------ provider selection


def test_build_enricher_selection(monkeypatch):
    keyless = dict(reader_api_key="none")
    assert isinstance(build_enricher(_settings(**keyless)), MockEnricher)  # today's default stays mock
    assert isinstance(build_enricher(_settings(**keyless), _StubSigLIP()), LocalEnricher)  # real image tower
    from megamind.store.embeddings import MockImageEmbedder

    assert isinstance(build_enricher(_settings(**keyless), MockImageEmbedder()), MockEnricher)  # mock tower: no

    vlm = build_enricher(_settings(), _StubSigLIP())  # reader key -> VLM, local as its fallback
    assert isinstance(vlm, VLMEnricher) and isinstance(vlm.fallback, LocalEnricher)
    assert build_enricher(_settings()).fallback is None
    assert isinstance(build_enricher(_settings(enrich_provider="mock")), MockEnricher)
    assert isinstance(build_enricher(_settings(enrich_provider="local", **keyless)), LocalEnricher)
    assert isinstance(build_enricher(_settings(enrich_provider="vlm", **keyless)), VLMEnricher)  # keyless vLLM
    assert resolve_enrich_provider(_settings(enrich_provider="AUTO", **keyless)) == "mock"

    monkeypatch.setattr(enrich, "tesseract_available", lambda: True)
    assert resolve_enrich_provider(_settings(**keyless)) == "mock"  # OCR alone would leave most frames uncaptioned
    assert resolve_enrich_provider(_settings(**keyless), _StubSigLIP()) == "local"


async def test_worker_stores_vlm_enrichment(tmp_path):
    """End to end through the app: frames get the VLM's caption, so BM25 can find them."""
    from httpx import ASGITransport, AsyncClient

    from megamind.main import create_app

    app = create_app(
        _settings(thumb_dir=tmp_path / "t", audio_dir=tmp_path / "a", consolidate_enabled=False, filter_min_gap_s=0)
    )
    async with app.router.lifespan_context(app):
        worker = app.state.worker
        assert isinstance(worker.enricher, VLMEnricher)
        worker.enricher = _vlm(lambda req: httpx.Response(200, json=_completion(json.dumps(GOOD))))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            import random

            rnd = random.Random(3)
            img = Image.new("RGB", (96, 72))
            img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(96 * 72)])
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            r = await c.post(
                "/v1/frames",
                files={"image": ("f.jpg", buf.getvalue(), "image/jpeg")},
                data={"meta": json.dumps({"device_id": "dev1", "ts": 1000.0})},
            )
            assert r.json()["accepted"]
            await worker.queue.join()
            hits = (await c.get("/v1/memories", params={"q": "kettle"})).json()["results"]
            assert hits and hits[0]["caption"] == GOOD["caption"] and hits[0]["ocr_text"] == "BOSCH"
