import asyncio
import io
import json
import random
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from megamind.config import Settings
from megamind.ingest import privacy
from megamind.ingest.enrich import Enricher, Enrichment, MockTextEmbedder
from megamind.ingest.privacy import FaceBlur, blur_faces, build_face_blur
from megamind.ingest.worker import EnrichmentWorker
from megamind.models import MemoryToken
from megamind.store.memory_store import InMemoryStore

BOX = (30, 20, 40, 40)  # x, y, w, h


def _noisy_jpeg(seed: int = 0, size=(128, 96)) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(size[0] * size[1])])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _stddev(jpeg: bytes, box) -> float:
    img = Image.open(io.BytesIO(jpeg)).convert("L")
    return ImageStat.Stat(img.crop(box)).stddev[0]


def fake_detector(img):
    return [BOX]


def no_faces(img):
    return []


# --- blur_faces -------------------------------------------------------------------------------


def test_blur_faces_blurs_only_detected_region():
    raw = _noisy_jpeg()
    out, n = blur_faces(raw, fake_detector)
    assert n == 1 and out != raw
    x, y, w, h = BOX
    face = (x + 5, y + 5, x + w - 5, y + h - 5)
    assert _stddev(out, face) < 0.3 * _stddev(raw, face)  # noise inside the face box is smoothed away
    corner = (100, 70, 128, 96)
    assert _stddev(out, corner) > 0.8 * _stddev(raw, corner)  # background keeps its detail


def test_blur_faces_no_face_returns_input_unchanged():
    raw = _noisy_jpeg()
    assert blur_faces(raw, no_faces) == (raw, 0)


def test_blur_faces_box_at_edge_and_undecodable_input():
    raw = _noisy_jpeg()
    out, n = blur_faces(raw, lambda img: [(-10, -10, 30, 30), (120, 90, 50, 50)])
    assert n == 2 and Image.open(io.BytesIO(out)).size == (128, 96)
    assert blur_faces(b"not a jpeg", fake_detector) == (b"not a jpeg", 0)


def test_missing_opencv_is_a_noop(monkeypatch):
    monkeypatch.setattr(privacy, "haar_detector", lambda: None)
    raw = _noisy_jpeg()
    assert blur_faces(raw) == (raw, 0)
    assert build_face_blur(Settings(privacy_blur_faces=True, _env_file=None)) is None


def test_build_face_blur_flags(monkeypatch):
    monkeypatch.setattr(privacy, "haar_detector", lambda: fake_detector)
    assert build_face_blur(Settings(_env_file=None)) is None  # default off
    fb = build_face_blur(Settings(privacy_blur_faces=True, _env_file=None))
    assert fb is not None and fb.detector is fake_detector and not fb.upstream
    fb = build_face_blur(Settings(privacy_blur_faces_vlm=True, _env_file=None))  # implies thumbnail blur
    assert fb is not None and fb.upstream


def test_real_haar_detector_smoke():
    pytest.importorskip("cv2")
    privacy.haar_detector.cache_clear()
    det = privacy.haar_detector()
    assert det is not None
    raw = _noisy_jpeg()
    assert blur_faces(raw, det) == (raw, 0)  # noise has no faces: bytes untouched, no re-encode
    flat = io.BytesIO()
    Image.new("RGB", (320, 240), (200, 180, 160)).save(flat, format="JPEG")
    assert det(Image.open(flat)) == []


def test_real_haar_detector_blurs_a_real_face():
    pytest.importorskip("cv2")
    skdata = pytest.importorskip("skimage.data")  # public-domain NASA portrait, not committed to the repo
    buf = io.BytesIO()
    Image.fromarray(skdata.astronaut()).resize((480, 480)).save(buf, format="JPEG", quality=90)
    raw = buf.getvalue()
    privacy.haar_detector.cache_clear()
    det = privacy.haar_detector()
    boxes = det(Image.open(io.BytesIO(raw)))
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    out, n = blur_faces(raw, det)
    assert n == 1
    face = (x + w // 4, y + h // 4, x + 3 * w // 4, y + 3 * h // 4)
    assert _stddev(out, face) < 0.5 * _stddev(raw, face)
    assert det(Image.open(io.BytesIO(out))) == []  # no longer detectable as a face


def test_blur_faces_keeps_exif_orientation():
    img = Image.open(io.BytesIO(_noisy_jpeg()))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    out, n = blur_faces(buf.getvalue(), fake_detector)
    assert n == 1 and Image.open(io.BytesIO(out)).getexif().get(0x0112) == 6


# --- worker -----------------------------------------------------------------------------------


class RecordingEnricher(Enricher):
    def __init__(self) -> None:
        self.seen: list[bytes] = []

    async def enrich(self, jpeg: bytes) -> Enrichment:
        self.seen.append(jpeg)
        return Enrichment(caption="a person at a desk")


class GatedEnricher(Enricher):
    """Blocks every call until `gate` is set, so frames stay in flight."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.entered = 0

    async def enrich(self, jpeg: bytes) -> Enrichment:
        self.entered += 1
        await self.gate.wait()
        return Enrichment(caption="late caption")


def _token(tmp_path: Path, i: int, jpeg: bytes, *, device="dev1", ts=1000.0) -> MemoryToken:
    thumb = tmp_path / f"f{i}.jpg"
    thumb.write_bytes(jpeg)
    return MemoryToken(id=f"f{i}", device_id=device, ts=ts, thumb_path=str(thumb), frame_hash=f"{i:016x}")


async def _until(pred, steps: int = 200) -> None:
    for _ in range(steps):
        if pred():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached")


@pytest.mark.parametrize("upstream", [False, True])
async def test_worker_blurs_thumbnail_and_optionally_vlm_input(tmp_path, upstream):
    store, enricher = InMemoryStore(), RecordingEnricher()
    worker = EnrichmentWorker(
        store, enricher, MockTextEmbedder(), None, face_blur=FaceBlur(fake_detector, upstream=upstream)
    )
    await worker.start()
    raw = _noisy_jpeg()
    tok = _token(tmp_path, 0, raw)
    await worker.submit(tok, raw)
    await worker.queue.join()
    await worker.stop()

    stored = Path(tok.thumb_path).read_bytes()
    assert stored != raw and _stddev(stored, (35, 25, 65, 55)) < 0.3 * _stddev(raw, (35, 25, 65, 55))
    assert enricher.seen == [stored if upstream else raw]
    assert await store.count() == 1


async def test_worker_without_faces_leaves_thumbnail_untouched(tmp_path):
    store, enricher = InMemoryStore(), RecordingEnricher()
    worker = EnrichmentWorker(store, enricher, MockTextEmbedder(), None, face_blur=FaceBlur(no_faces, upstream=True))
    await worker.start()
    raw = _noisy_jpeg()
    tok = _token(tmp_path, 0, raw)
    await worker.submit(tok, raw)
    await worker.queue.join()
    await worker.stop()
    assert Path(tok.thumb_path).read_bytes() == raw and enricher.seen == [raw]


async def test_failed_frame_thumbnail_is_removed(tmp_path):
    """A frame whose blur/enrichment fails is never stored, so no purge could reach its raw thumbnail."""

    def broken_detector(img):
        raise RuntimeError("cascade exploded")

    store = InMemoryStore()
    worker = EnrichmentWorker(store, RecordingEnricher(), MockTextEmbedder(), None, face_blur=FaceBlur(broken_detector))
    await worker.start()
    raw = _noisy_jpeg()
    tok = _token(tmp_path, 0, raw)
    await worker.submit(tok, raw)
    await asyncio.wait_for(worker.queue.join(), 5)
    await worker.stop()
    assert await store.count() == 0 and not Path(tok.thumb_path).exists()


async def test_worker_purge_discards_queued_and_inflight(tmp_path):
    store, enricher = InMemoryStore(), GatedEnricher()
    worker = EnrichmentWorker(store, enricher, MockTextEmbedder(), None, concurrency=1)
    await worker.start()
    jpeg = _noisy_jpeg()
    inflight = _token(tmp_path, 0, jpeg, ts=1000.0)
    queued = _token(tmp_path, 1, jpeg, ts=2000.0)
    old = _token(tmp_path, 2, jpeg, ts=500.0)  # before the `since` window
    other = _token(tmp_path, 3, jpeg, device="dev2", ts=2000.0)  # another device
    for t in (inflight, queued, old, other):
        await worker.submit(t, jpeg)
    await _until(lambda: enricher.entered == 1)

    assert await worker.purge("dev1", since_ts=900.0) == 2
    enricher.gate.set()
    await asyncio.wait_for(worker.queue.join(), 5)
    await worker.stop()

    assert sorted(t.id for t in await store.range(None, None)) == ["f2", "f3"]
    assert not Path(inflight.thumb_path).exists() and not Path(queued.thumb_path).exists()
    assert Path(old.thumb_path).exists() and Path(other.thumb_path).exists()


async def test_submit_backpressure_drops_oldest_and_its_thumbnail(tmp_path):
    worker = EnrichmentWorker(InMemoryStore(), RecordingEnricher(), MockTextEmbedder(), None)
    worker.queue = asyncio.Queue(maxsize=1)  # not started: items stay queued
    jpeg = _noisy_jpeg()
    first, second = _token(tmp_path, 0, jpeg), _token(tmp_path, 1, jpeg)
    await worker.submit(first, jpeg)
    await worker.submit(second, jpeg)
    assert not Path(first.thumb_path).exists() and worker.queue.qsize() == 1
    assert worker.discard() == 1
    await asyncio.wait_for(worker.queue.join(), 1)  # task accounting stays consistent


# --- API: DELETE /v1/memories vs frames still in the worker -------------------------------------


async def _post_frame(c, jpeg, ts, device="dev1"):
    return await c.post(
        "/v1/frames",
        files={"image": ("f.jpg", jpeg, "image/jpeg")},
        data={"meta": json.dumps({"device_id": device, "ts": ts})},
    )


async def test_purge_race_nothing_stored_after_delete(client):
    c, app = client
    worker = app.state.worker
    worker.enricher = GatedEnricher()  # every frame stalls mid-enrichment
    frames = [_noisy_jpeg(seed) for seed in (1, 2, 3)]
    for i, jpeg in enumerate(frames):  # 2 workers -> two in flight, one still queued
        assert (await _post_frame(c, jpeg, 1000.0 + i)).json()["accepted"]
    await _until(lambda: worker.enricher.entered == 2)
    assert worker.queue.qsize() == 1

    r = await c.delete("/v1/memories")
    assert r.status_code == 200 and r.json() == {"deleted": 3}

    worker.enricher.gate.set()
    await asyncio.wait_for(worker.queue.join(), 5)
    assert await app.state.store.count() == 0
    assert list(app.state.settings.thumb_dir.iterdir()) == []

    # dedupe state of the purged frames is gone too: the same picture is a new memory now
    r = await _post_frame(c, frames[0], 1010.0)
    assert r.json()["accepted"], r.json()
    await asyncio.wait_for(worker.queue.join(), 5)
    assert await app.state.store.count() == 1


async def test_purge_race_respects_device_scope(client):
    c, app = client
    worker = app.state.worker
    worker.enricher = GatedEnricher()
    assert (await _post_frame(c, _noisy_jpeg(5), 1000.0, device="dev1")).json()["accepted"]
    assert (await _post_frame(c, _noisy_jpeg(6), 1000.0, device="dev2")).json()["accepted"]
    await _until(lambda: worker.enricher.entered == 2)

    assert (await c.delete("/v1/memories", params={"device_id": "dev1"})).json() == {"deleted": 1}
    worker.enricher.gate.set()
    await asyncio.wait_for(worker.queue.join(), 5)
    assert [t.device_id for t in await app.state.store.range(None, None)] == ["dev2"]
