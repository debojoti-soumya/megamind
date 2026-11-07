import io
import random

from PIL import Image

from megamind.config import Settings
from megamind.ingest.enrich import MockImageEmbedder
from megamind.ingest.episodes import EpisodeTracker
from megamind.models import MemoryToken, Pose


def _textured(color, seed, size=(96, 72)) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([tuple(max(0, min(255, c + rnd.randint(-40, 40))) for c in color) for _ in range(size[0] * size[1])])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


RED, BLUE = (200, 40, 40), (40, 40, 200)


async def _tok(emb, i, ts, color, **kw):
    return MemoryToken(
        id=f"t{i}",
        device_id="d",
        ts=ts,
        thumb_path="",
        frame_hash="0",
        image_vec=await emb.embed_image(_textured(color, i)),
        sharpness=100.0,
        **kw,
    )


async def test_split_on_visual_change_gap_and_rejoin():
    emb = MockImageEmbedder()
    tr = EpisodeTracker(Settings(_env_file=None), emb, labels=["a red room", "a blue room"])
    a = [await tr.assign(await _tok(emb, i, 10.0 * i, RED)) for i in range(3)]
    assert len({e.id for e in a}) == 1 and a[-1].n_tokens == 3 and a[-1].title == "a red room"
    b = await tr.assign(await _tok(emb, 3, 40, BLUE))
    assert b.id != a[0].id and b.title == "a blue room"
    back = await tr.assign(await _tok(emb, 4, 50, RED))
    assert back.id == a[0].id and back.n_tokens == 4
    late = await tr.assign(await _tok(emb, 5, 50 + 121, RED))
    assert late.id != a[0].id


async def test_split_on_movement_and_entity_titles():
    tr = EpisodeTracker(Settings(_env_file=None), None)

    def tok(i, ts, x, ents):
        return MemoryToken(
            id=f"t{i}",
            device_id="d",
            ts=ts,
            thumb_path="",
            frame_hash="0",
            pose=Pose(x=x, y=1.6, z=0, yaw=0),
            entities=ents,
        )

    e1 = await tr.assign(tok(1, 0, 0.0, ["keys", "mug"]))
    e2 = await tr.assign(tok(2, 5, 0.5, ["keys"]))
    assert e1.id == e2.id and e2.title.startswith("keys")
    e3 = await tr.assign(tok(3, 10, 5.0, ["car"]))
    assert e3.id != e1.id
