import pytest

from megamind.agent.reader import VLMReader, parse_read
from megamind.config import Settings
from megamind.models import MemoryToken
from megamind.store.fusion import FusionWeights, tokenize, weighted_rrf
from megamind.store.memory_store import InMemoryStore


def _tok(i: str, **kw) -> MemoryToken:
    return MemoryToken(id=i, device_id="d", ts=1000.0, frame_hash="0" * 16, thumb_path="", **kw)


def test_parse_read_keeps_only_shown_citations():
    out = parse_read('ok {"answer": "3 cups", "confidence": "high", "cited_ids": ["a", "zzz"]}', ["a", "b"])
    assert (out.evidence, out.confidence, out.cited_ids) == ("3 cups", "high", ["a"])
    raw = parse_read("three cups on the table", ["a", "b"])
    assert raw.confidence == "low" and raw.cited_ids == ["a"] and "three" in raw.evidence


@pytest.mark.asyncio
async def test_vlm_reader_sends_retrieved_frames():
    class Media:
        async def read(self, t):
            return b"\xff\xd8jpeg-" + t.id.encode() if t.id != "gone" else None

    sent = {}

    class Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                async def create(**kw):
                    sent.update(kw)
                    msg = type(
                        "M", (), {"content": '{"answer": "on the couch", "confidence": "high", "cited_ids": ["b"]}'}
                    )
                    return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    r = VLMReader(Settings(_env_file=None, reader_max_frames=3), Media())
    r.client = Client()
    out = await r.read("where are my keys", [_tok("a"), _tok("gone"), _tok("b"), _tok("c")])
    images = [p for p in sent["messages"][0]["content"] if p["type"] == "image_url"]
    assert len(images) == 2  # max 3 tokens, one has no pixels left
    assert out.cited_ids == ["b"] and out.evidence == "on the couch"


def test_tokenize_matches_lance_fts_normalisation():
    assert tokenize("Where did I park the car?") == ["park", "car"]
    assert tokenize("PARKING level 3") == ["park", "level", "3"]


def test_rrf_multiplier_and_image_calibration():
    assert weighted_rrf([(1.0, [("a", 0.0), ("b", 1.0)])])["a"] == 0.0
    w = FusionWeights(image_logit_scale=117.3, image_logit_bias=-12.9)
    assert w.image_conf(0.15) > 0.9 and w.image_conf(0.05) < 0.01
    assert FusionWeights().image_conf(-1.0) == 1.0


@pytest.mark.asyncio
async def test_weak_image_matches_do_not_outrank_text_hits():
    w = FusionWeights(image=2.0, image_logit_scale=117.3, image_logit_bias=-12.9)
    s = InMemoryStore(w)
    text_vec, img = [1.0, 0.0], [1.0, 0.0, 0.0]
    await s.add(_tok("sign", ocr_text="wifi password falcon", text_vec=[0.9, 0.1], image_vec=[0.0, 1.0, 0.0]))
    await s.add(_tok("photo", caption="a cafe", text_vec=[0.5, 0.5], image_vec=[0.06, 1.0, 0.0]))
    hits = await s.search(text_vec, img, "wifi password", mode="hybrid", k=2)
    assert hits[0].id == "sign"
