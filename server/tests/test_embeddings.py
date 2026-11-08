import json
import math
from types import SimpleNamespace

import httpx
import pytest

from megamind.config import Settings
from megamind.ingest.enrich import MockTextEmbedder
from megamind.main import image_calibration
from megamind.store.embeddings import (
    MockImageEmbedder,
    OpenAITextEmbedder,
    OpenCLIPImageEmbedder,
    build_image_embedder,
    build_text_embedder,
    normalize,
    siglip_calibration,
)


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


async def test_openai_text_embedder_requests_dimensions_and_normalises():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        n = len(seen["body"]["input"])
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [3.0, 4.0] + [0.0] * 6} for i in reversed(range(n))]}
        )

    emb = OpenAITextEmbedder("sk-test", dim=8, transport=httpx.MockTransport(handler))
    v = await emb.embed("keys on the counter")
    assert seen["url"].endswith("/v1/embeddings") and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "text-embedding-3-small" and seen["body"]["dimensions"] == 8
    assert len(v) == 8 and abs(_norm(v) - 1.0) < 1e-9 and v[:2] == [0.6, 0.8]
    many = await emb.embed_many(["a", "b", ""])
    assert len(many) == 3 and seen["body"]["input"] == ["a", "b", " "]


async def test_openai_text_embedder_retries_on_429():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    emb = OpenAITextEmbedder("sk", dim=2, transport=httpx.MockTransport(handler))
    assert await emb.embed("x") == [1.0, 0.0] and calls["n"] == 2


async def test_mock_image_embedder_is_deterministic_and_colour_aware():
    import io

    from PIL import Image

    def jpeg(rgb):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), rgb).save(buf, format="JPEG")
        return buf.getvalue()

    emb = MockImageEmbedder()
    red = await emb.embed_image(jpeg((255, 0, 0)))
    assert red == await emb.embed_image(jpeg((255, 0, 0))) and len(red) == emb.dim
    q_red, q_blue = await emb.embed_text("a red mug"), await emb.embed_text("a blue mug")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(red, q_red) > dot(red, q_blue)
    assert abs(_norm(await emb.embed_image(b"not a jpeg")) - 1.0) < 1e-9


def test_factories_fall_back_to_mock_without_keys(monkeypatch):
    monkeypatch.delenv("IMAGE_EMBED_PROVIDER", raising=False)
    assert isinstance(build_text_embedder(Settings(embed_provider="mock")), MockTextEmbedder)
    assert isinstance(build_text_embedder(Settings(embed_provider="openai", openai_api_key="")), MockTextEmbedder)
    real = build_text_embedder(
        Settings(embed_provider="openai", openai_api_key="sk-x", embed_model="text-embedding-3-small")
    )
    assert isinstance(real, OpenAITextEmbedder) and real.dim == 1024
    assert build_image_embedder(Settings()) is None
    monkeypatch.setenv("IMAGE_EMBED_PROVIDER", "mock")
    assert isinstance(build_image_embedder(Settings()), MockImageEmbedder)


def test_normalize_zero_vector():
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


async def test_openclip_embedder_cpu_throughput():
    import io
    import time

    from PIL import Image

    pytest.importorskip("open_clip", reason="open_clip not installed")
    from megamind.store.embeddings import OpenCLIPImageEmbedder

    emb = OpenCLIPImageEmbedder()
    buf = io.BytesIO()
    Image.new("RGB", (960, 720), (120, 80, 40)).save(buf, format="JPEG")
    jpeg = buf.getvalue()
    v = await emb.embed_image(jpeg)
    assert len(v) == emb.dim == 768 and abs(_norm(v) - 1.0) < 1e-5
    t0 = time.perf_counter()
    for _ in range(5):
        await emb.embed_image(jpeg)
    fps = 5 / (time.perf_counter() - t0)
    assert fps >= 1.0, f"{fps:.2f} frames/s"
    q = await emb.embed_text("a photo of a brown wall")
    assert len(q) == 768


def _failing(status: int, calls: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] = calls.get("n", 0) + 1
        if calls.get("ok"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0] + [0.0] * 7}]})
        return httpx.Response(status, json={"error": {"message": "nope"}})

    return httpx.MockTransport(handler)


async def test_openai_embedder_degrades_to_same_dim_mock_on_401(caplog):
    calls: dict = {}
    emb = OpenAITextEmbedder("sk-bad", dim=8, transport=_failing(401, calls))
    v = await emb.embed("keys on the counter")
    assert len(v) == 8 and abs(_norm(v) - 1.0) < 1e-9
    assert emb.degraded == "last error 401" and calls["n"] == 1  # 401 is not retried
    mock = MockTextEmbedder()
    mock.dim = 8
    assert v == await mock.embed("keys on the counter")  # deterministic, so BM25-like matching still works
    assert len(await emb.embed_many(["a", "b"])) == 2
    assert sum("OpenAI embeddings failed" in r.message for r in caplog.records) == 1  # logged once

    calls["ok"] = True
    assert await emb.embed("x") == [1.0] + [0.0] * 7
    assert emb.degraded is None


async def test_openai_embedder_degrades_after_retries_on_503(monkeypatch):
    async def no_sleep(_s):
        return None

    monkeypatch.setattr("megamind.store.embeddings.asyncio.sleep", no_sleep)
    calls: dict = {}
    emb = OpenAITextEmbedder("sk", dim=8, transport=_failing(503, calls))
    assert len(await emb.embed("x")) == 8
    assert calls["n"] == 3 and emb.degraded == "last error 503"


async def test_openai_embedder_degrades_on_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns failure", request=request)

    emb = OpenAITextEmbedder("sk", dim=8, transport=httpx.MockTransport(handler))
    assert len(await emb.embed("x")) == 8 and emb.degraded == "last error ConnectError"


# ---------------------------------------------------------------- SigLIP calibration (port of PR #54), no torch


class _FakeTensor:
    def __init__(self, v: float) -> None:
        self.v = v

    def detach(self):
        return self

    def item(self) -> float:
        return self.v


def _fake_openclip(model) -> OpenCLIPImageEmbedder:
    emb = object.__new__(OpenCLIPImageEmbedder)  # skip __init__: no open_clip / torch needed
    emb.model = model
    return emb


def test_siglip_calibration_from_model_attributes():
    siglip = SimpleNamespace(logit_scale=_FakeTensor(math.log(117.3)), logit_bias=_FakeTensor(-12.9))
    scale, bias = _fake_openclip(siglip).calibration
    assert scale == pytest.approx(117.3) and bias == pytest.approx(-12.9)
    assert siglip_calibration(SimpleNamespace(logit_scale=math.log(10.0), logit_bias=-1.0)) == pytest.approx(
        (10.0, -1.0)
    )


def test_clip_models_and_mocks_are_uncalibrated():
    clip = SimpleNamespace(logit_scale=_FakeTensor(math.log(100.0)))  # CLIP: scale but no sigmoid bias
    assert _fake_openclip(clip).calibration is None
    assert MockImageEmbedder().calibration is None


def test_main_sets_store_calibration_only_when_enabled():
    class _Cal(MockImageEmbedder):
        calibration = (117.3, -12.9)

    on = Settings(_env_file=None)
    off = Settings(_env_file=None, search_calibrate_image=False)
    assert image_calibration(on, _Cal()) == (117.3, -12.9)
    assert image_calibration(off, _Cal()) is None
    assert image_calibration(on, None) is None
    assert image_calibration(on, MockImageEmbedder()) is None
