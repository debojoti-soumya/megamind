import json
import math

import httpx
import pytest

from megamind.config import Settings
from megamind.ingest.enrich import MockTextEmbedder
from megamind.store.embeddings import (
    MockImageEmbedder,
    OpenAITextEmbedder,
    build_image_embedder,
    build_text_embedder,
    normalize,
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
