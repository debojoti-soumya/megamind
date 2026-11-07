"""`recall_memory` wiring: the retrieval half of `RecallMemory` honours mode / time bounds / k and hands the
reader a best-first list of full tokens (hand-off contract in issue #5). The reader is faked."""

from datetime import UTC, datetime

from megamind.agent.reader import Reader, ReaderOutput
from megamind.agent.tools import RECALL_MEMORY_TOOL, RecallMemory
from megamind.ingest.enrich import ImageEmbedder, MockTextEmbedder
from megamind.models import MemoryToken
from megamind.retrieval import HybridConfig, HybridRetriever
from megamind.store.memory_store import InMemoryStore

NOW = datetime.now(UTC).timestamp()
STREAM = [
    ("t0", 3600, "Kitchen counter with a coffee machine and a mug.", "", "ff00ff00ff00ff00"),
    ("t1", 3540, "Keys lying on the kitchen counter next to the coffee machine.", "", "ff00ff00ff00ff01"),
    ("t2", 3480, "Hallway with a coat rack and a closed front door.", "", "0000000000000000"),
    ("t3", 2700, "Office desk with a laptop; a whiteboard behind.", "DEMO FRIDAY 5PM", "1234123412341234"),
    ("t4", 1800, "Two people sitting at a conference table talking.", "", "abcdabcdabcdabcd"),
    ("t5", 1200, "Keys on a wooden desk beside a notebook.", "", "5555555555555555"),
    ("t6", 600, "A grocery shelf with cartons of oat milk and a price tag.", "OAT MILK $4.99", "9999999999999999"),
]


class FakeReader(Reader):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[MemoryToken]]] = []

    async def read(self, query: str, tokens: list[MemoryToken]) -> ReaderOutput:
        self.calls.append((query, tokens))
        return ReaderOutput(evidence=f"saw {tokens[0].id}", confidence="high", cited_ids=[tokens[0].id, "bogus"])


class FakeImageEmbedder(ImageEmbedder):
    def __init__(self) -> None:
        self.text_calls = 0

    async def embed_image(self, jpeg: bytes) -> list[float]:
        return [1.0, 0.0]

    async def embed_text(self, text: str) -> list[float]:
        self.text_calls += 1
        return [1.0, 0.0]


async def _setup(image_embedder=None, **cfg):
    store = InMemoryStore()
    emb = MockTextEmbedder()
    for tid, age, cap, ocr, fh in STREAM:
        await store.add(
            MemoryToken(
                id=tid,
                device_id="dev",
                ts=NOW - age,
                caption=cap,
                ocr_text=ocr,
                thumb_path=f"/tmp/{tid}.jpg",
                frame_hash=fh,
                text_vec=await emb.embed(f"{cap} {ocr}"),
                image_vec=[1.0, 0.0] if image_embedder else None,
            )
        )
    reader = FakeReader()
    retr = HybridRetriever(store, emb, image_embedder, tz="UTC", config=HybridConfig(**cfg))
    return RecallMemory(store, emb, image_embedder, reader, retriever=retr), reader


async def test_hands_reader_full_tokens_best_first_and_maps_result():
    recall, reader = await _setup()
    res = await recall(query="where are my keys", k=4)
    ((q, tokens),) = reader.calls
    assert q == "where are my keys"
    assert all(isinstance(t, MemoryToken) for t in tokens)
    assert tokens[0].id == "t5"  # most recent sighting first, not chronological
    assert len(tokens) <= 4
    assert res.confidence == "high" and res.evidence == "saw t5"
    assert [c.id for c in res.cited] == ["t5"]  # 'bogus' filtered out
    assert [c.id for c in res.all_retrieved] == [t.id for t in tokens]


async def test_context_tokens_come_after_ranked_hits():
    recall, reader = await _setup()
    await recall(query="what did the whiteboard say", k=6)
    ((_, tokens),) = reader.calls
    assert tokens[0].id == "t3"
    neighbours = {"t2", "t4"}
    ranked_ids = [t.id for t in tokens]
    assert neighbours & set(ranked_ids)  # temporal expansion added context
    assert all(ranked_ids.index(n) > 0 for n in neighbours & set(ranked_ids))


async def test_honours_k_and_default_retriever():
    store = InMemoryStore()
    emb = MockTextEmbedder()
    for tid, age, cap, ocr, fh in STREAM:
        await store.add(
            MemoryToken(
                id=tid,
                device_id="dev",
                ts=NOW - age,
                caption=cap,
                ocr_text=ocr,
                thumb_path="/tmp/x.jpg",
                frame_hash=fh,
                text_vec=await emb.embed(cap),
            )
        )
    reader = FakeReader()
    recall = RecallMemory(store, emb, None, reader)  # builds its own HybridRetriever
    assert isinstance(recall.retriever, HybridRetriever)
    res = await recall(query="keys", k=2)
    assert len(reader.calls[0][1]) == 2 and len(res.all_retrieved) == 2
    res = await recall(query="keys", k=99)
    assert len(res.all_retrieved) <= 24


async def test_explicit_iso_bounds_win_over_recency():
    recall, reader = await _setup()
    lo = datetime.fromtimestamp(NOW - 3700, UTC).isoformat()
    hi = datetime.fromtimestamp(NOW - 3000, UTC).isoformat()
    res = await recall(query="where are my keys", time_from=lo, time_to=hi, k=4)
    assert reader.calls[0][1][0].id == "t1"
    assert all(NOW - 3700 <= c.ts <= NOW - 3000 for c in res.all_retrieved)


async def test_relative_time_in_the_question_is_parsed():
    recall, reader = await _setup()
    await recall(query="where were my keys about an hour ago", k=3)
    assert reader.calls[0][1][0].id == "t1"


async def test_visual_mode_uses_image_tower():
    img = FakeImageEmbedder()
    recall, reader = await _setup(image_embedder=img)
    await recall(query="oat milk", mode="visual", k=3)
    assert img.text_calls == 1
    assert reader.calls[0][1]


async def test_semantic_mode_skips_image_tower():
    img = FakeImageEmbedder()
    recall, _ = await _setup(image_embedder=img)
    await recall(query="oat milk", mode="semantic", k=3)
    assert img.text_calls == 0


async def test_no_match_returns_not_found_without_calling_reader():
    recall, reader = await _setup()
    res = await recall(query="zebra trombone", k=3)
    assert res.confidence == "not_found" and res.all_retrieved == [] and reader.calls == []


def test_tool_schema_unchanged():
    assert RECALL_MEMORY_TOOL["input_schema"]["properties"]["mode"]["enum"] == ["semantic", "visual", "hybrid"]
