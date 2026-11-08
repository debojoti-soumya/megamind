"""Frame-ingest path: kept-hash lookup reads the store once per device, then stays in memory."""

from __future__ import annotations

import io
import json
import random

from PIL import Image

from megamind.ingest.filter import dhash
from megamind.models import MemoryToken


def _noisy(seed: int) -> bytes:
    rnd = random.Random(seed)
    img = Image.new("RGB", (96, 72))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(96 * 72)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


async def _post(c, jpeg: bytes, ts: float, device: str = "dev1") -> dict:
    r = await c.post(
        "/v1/frames",
        files={"image": ("f.jpg", jpeg, "image/jpeg")},
        data={"meta": json.dumps({"device_id": device, "ts": ts})},
    )
    assert r.status_code == 202
    return r.json()


def _count_scans(store) -> list[str]:
    calls: list[str] = []
    orig = store.recent_hashes

    async def wrapped(device_id, since_ts):
        calls.append(device_id)
        return await orig(device_id, since_ts)

    store.recent_hashes = wrapped
    return calls


async def test_frames_kept_before_restart_still_dedupe(client):
    c, app = client
    jpeg = _noisy(1)
    h = dhash(Image.open(io.BytesIO(jpeg)))
    await app.state.store.add(MemoryToken(id="old", device_id="dev1", ts=1000.0, thumb_path="", frame_hash=h))
    calls = _count_scans(app.state.store)

    assert (await _post(c, jpeg, 1010.0))["reason"] == "duplicate"
    assert (await _post(c, _noisy(2), 1012.0))["accepted"]
    assert (await _post(c, _noisy(2), 1014.0))["reason"] == "duplicate"  # kept in this process, not yet stored
    assert (await _post(c, _noisy(3), 1016.0, device="dev2"))["accepted"]
    assert calls == ["dev1", "dev2"]  # one store read per device, not one per frame


async def test_purge_forgets_cached_hashes(client):
    c, app = client
    assert (await _post(c, _noisy(1), 1000.0))["accepted"]
    await app.state.worker.queue.join()
    r = await c.delete("/v1/memories", params={"device_id": "dev1"})
    assert r.status_code == 200
    assert (await _post(c, _noisy(1), 1003.0))["accepted"]


async def test_since_purge_keeps_older_cached_hashes(client):
    c, app = client
    assert (await _post(c, _noisy(1), 1000.0))["accepted"]
    assert (await _post(c, _noisy(2), 1005.0))["accepted"]
    await app.state.worker.queue.join()
    assert (await c.delete("/v1/memories", params={"since": "1004"})).status_code == 200
    assert (await _post(c, _noisy(1), 1010.0))["reason"] == "duplicate"  # 1000 survived the purge
    assert (await _post(c, _noisy(2), 1012.0))["accepted"]  # 1005 was purged
