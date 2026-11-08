"""`LocalTextEmbedder` (bge-small via sentence-transformers) and the optional `TextEmbedder.embed_query`, without
torch: a fake model stands in for SentenceTransformer."""

from __future__ import annotations

import sys

from megamind.config import Settings
from megamind.ingest.enrich import TextEmbedder
from megamind.models import MemoryToken
from megamind.retrieval.hybrid import HybridRetriever
from megamind.store import InMemoryStore
from megamind.store.embeddings import (
    BGE_QUERY_PREFIX,
    LocalTextEmbedder,
    MockTextEmbedder,
    build_text_embedder,
)


class FakeST:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, normalize_embeddings=True):
        self.seen += list(texts)
        return [[float(len(t)), 0.0, 1.0] for t in texts]


async def test_local_embedder_prefixes_queries_only():
    m = FakeST()
    emb = LocalTextEmbedder("BAAI/bge-small-en-v1.5", model=m)
    assert emb.dim == 3 and emb.model == "BAAI/bge-small-en-v1.5"
    doc = await emb.embed("keys on the table")
    q = await emb.embed_query("where are my keys")
    assert m.seen == ["keys on the table", BGE_QUERY_PREFIX + "where are my keys"]
    for v in (doc, q):
        assert abs(sum(x * x for x in v) - 1.0) < 1e-9
    assert len(await emb.embed_many(["a", "b"])) == 2
    assert m.seen[-2:] == ["a", "b"]


async def test_non_bge_model_gets_no_prefix_and_empty_text_is_safe():
    m = FakeST()
    emb = LocalTextEmbedder("sentence-transformers/all-MiniLM-L6-v2", model=m)
    await emb.embed_query("hello")
    await emb.embed("")
    assert m.seen == ["hello", " "]


async def test_default_embed_query_is_embed():
    emb = MockTextEmbedder()
    assert await emb.embed_query("red mug") == await emb.embed("red mug")


def test_build_local_falls_back_to_mock_without_sentence_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    emb = build_text_embedder(Settings(_env_file=None, embed_provider="local"))
    assert isinstance(emb, MockTextEmbedder)


async def test_hybrid_retriever_embeds_the_query_with_embed_query():
    class Recording(MockTextEmbedder):
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def embed_query(self, text: str) -> list[float]:
            self.queries.append(text)
            return await self.embed(text)

    emb = Recording()
    store = InMemoryStore()
    t = MemoryToken(id="t1", device_id="d", ts=1000.0, caption="red mug on desk", thumb_path="", frame_hash="")
    t.text_vec = await emb.embed(t.caption)
    await store.add(t)
    res = await HybridRetriever(store, emb).retrieve("red mug")
    assert emb.queries and [r.token.id for r in res.tokens] == ["t1"]
    assert isinstance(emb, TextEmbedder)


def test_local_embed_model_is_read_from_the_env_file(tmp_path, monkeypatch):
    """LOCAL_EMBED_MODEL in server/.env (as .env.example shows) must reach the embedder, not only a shell export."""
    import megamind.store.embeddings as E

    monkeypatch.delenv("LOCAL_EMBED_MODEL", raising=False)
    env = tmp_path / ".env"
    env.write_text("EMBED_PROVIDER=local\nLOCAL_EMBED_MODEL=BAAI/bge-base-en-v1.5\n")
    seen = []
    monkeypatch.setattr(E, "LocalTextEmbedder", lambda name: seen.append(name) or MockTextEmbedder())
    build_text_embedder(Settings(_env_file=env))
    assert seen == ["BAAI/bge-base-en-v1.5"]
