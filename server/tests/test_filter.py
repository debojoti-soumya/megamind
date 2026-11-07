"""Synthetic-image tests for megamind.ingest.filter (no network, no models)."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter

from megamind.config import Settings
from megamind.ingest.filter import KeyframeFilter, dhash, hamming, sharpness
from megamind.models import FrameMeta, Pose

W, H = 320, 240


def scene(seed: int, size=(W, H)) -> Image.Image:
    """A textured 'room': random coloured rectangles + fine noise, deterministic per seed."""
    rnd = random.Random(seed)
    img = Image.new("RGB", size, (rnd.randrange(60, 200), rnd.randrange(60, 200), rnd.randrange(60, 200)))
    px = img.load()
    for _ in range(25):
        x0, y0 = rnd.randrange(size[0]), rnd.randrange(size[1])
        x1, y1 = min(size[0], x0 + rnd.randrange(10, 120)), min(size[1], y0 + rnd.randrange(10, 120))
        c = (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
        for x in range(x0, x1):
            for y in range(y0, y1):
                px[x, y] = c
    noise = np.random.default_rng(seed).integers(-25, 25, (size[1], size[0], 3))
    return Image.fromarray(np.clip(np.asarray(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8))


def blurred(img: Image.Image, radius: float = 6.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


class Harness:
    def __init__(self, s: Settings | None = None):
        self.f = KeyframeFilter(s or settings())
        self.kept: dict[str, list[tuple[float, str]]] = {}

    def push(self, img: Image.Image, ts: float, pose: Pose | None = None, device="d"):
        kept = self.kept.setdefault(device, [])
        d = self.f.decide(img, FrameMeta(device_id=device, ts=ts, pose=pose), list(kept))
        if d.accepted:
            kept.append((ts, d.frame_hash))
        return d


def test_dhash_and_hamming_basics():
    a, b = scene(1), scene(2)
    assert hamming(dhash(a), dhash(a)) == 0
    assert hamming(dhash(a), dhash(b)) > 10
    assert hamming(dhash(a), dhash(blurred(a, 2))) <= 4  # dHash is robust to mild blur


def test_sharpness_separates_blur():
    sharp = sharpness(scene(1))
    assert sharp > 200
    assert sharpness(blurred(scene(1))) < 40
    assert sharpness(Image.new("RGB", (W, H), (120, 120, 120))) < 1
    # directional (pan) motion blur: horizontal box blur, vertical detail intact
    a = np.asarray(scene(1), dtype=np.float32)
    k = 25
    pan = np.cumsum(np.pad(a, ((0, 0), (k, 0), (0, 0))), axis=1)
    pan = (pan[:, k:] - pan[:, :-k]) / k
    assert sharpness(Image.fromarray(pan.astype(np.uint8))) < 40


def test_dark_rejected_first():
    h = Harness()
    d = h.push(Image.new("RGB", (W, H), (5, 5, 5)), 0.0)
    assert (d.accepted, d.reason) == (False, "dark")


def test_blurry_rejected_even_if_novel():
    h = Harness()
    assert h.push(scene(1), 0.0).accepted
    d = h.push(blurred(scene(2)), 10.0)
    assert (d.accepted, d.reason) == (False, "blurry")
    assert d.sharpness < 40


def test_first_then_too_soon_then_duplicate():
    h = Harness()
    d = h.push(scene(1), 0.0)
    assert d.accepted and d.trigger == "first"
    assert h.push(scene(2), 1.0).reason == "too_soon"
    assert h.push(scene(1), 5.0).reason == "duplicate"
    d = h.push(scene(3), 6.0)
    assert d.accepted and d.trigger == "novel"


def test_settled_frame_kept_after_blurry_pan():
    """Kept A, then a blurry pan frame (rejected), then the first stable frame of scene B: keep it
    even though it is inside the min gap - that is the 'put keys on the counter' moment."""
    h = Harness()
    assert h.push(scene(1), 0.0).accepted
    assert h.push(blurred(scene(2)), 0.6).reason == "blurry"
    d = h.push(scene(2), 1.2)
    assert d.accepted and d.trigger == "settled"
    # but a stable frame that is *still the old scene* stays a duplicate / too_soon
    assert h.push(blurred(scene(2)), 1.8).reason == "blurry"
    assert not h.push(scene(2), 2.4).accepted


def test_settled_frame_after_fast_scene_change_without_blur():
    h = Harness()
    assert h.push(scene(1), 0.0).accepted
    assert h.push(scene(2), 0.6).reason == "too_soon"  # changed but too soon -> prev_unstable
    d = h.push(scene(2), 1.2)
    assert d.accepted and d.trigger == "settled"


def test_pose_turn_forces_keep_and_still_pose_tightens_dedupe():
    h = Harness()
    p0 = Pose(x=0, y=1.6, z=0, yaw=0)
    assert h.push(scene(1), 0.0, p0).accepted
    d = h.push(scene(2), 0.5, Pose(x=0, y=1.6, z=0, yaw=25))
    assert d.accepted and d.trigger == "pose"
    # walk 0.6 m -> keep
    d = h.push(scene(3), 1.0, Pose(x=0.6, y=1.6, z=0, yaw=25))
    assert d.accepted and d.trigger == "pose"
    # head still, hash close-ish (noise re-seeded) -> duplicate even though hamming >= 6
    base = scene(3)
    noise = np.random.default_rng(99).integers(-40, 40, (H, W, 3))
    jitter = Image.fromarray(np.clip(np.asarray(base, dtype=np.int16) + noise, 0, 255).astype(np.uint8))
    dist = hamming(dhash(base), dhash(jitter))
    assert 0 < dist < 10
    d = h.push(jitter, 5.0, Pose(x=0.65, y=1.6, z=0, yaw=27))
    assert d.reason == "duplicate"
    # same image, head turned 10 deg (not enough to force) -> dedupe applies normally
    d2 = h.push(jitter, 10.0, Pose(x=0.65, y=1.6, z=0, yaw=37))
    assert d2.accepted == (dist >= 6)


def test_heartbeat_keeps_static_scene():
    h = Harness()
    assert h.push(scene(1), 0.0).accepted
    assert h.push(scene(1), 10.0).reason == "duplicate"
    d = h.push(scene(1), 31.0)
    assert d.accepted and d.trigger == "heartbeat"


def test_long_window_dedupe_for_static_scene():
    h = Harness(settings(filter_heartbeat_s=0))
    assert h.push(scene(1), 0.0).accepted
    assert h.push(scene(1), 100.0).reason == "duplicate"  # outside 30 s window, inside 600 s / <3
    d = h.push(scene(1), 700.0)
    assert d.accepted


def test_per_minute_budget():
    h = Harness(settings(filter_min_gap_s=0, filter_max_per_min=5, filter_heartbeat_s=0))
    reasons = [h.push(scene(i), i * 1.0).reason for i in range(1, 9)]
    assert reasons[:5] == [None] * 5
    assert reasons[5:] == ["budget"] * 3
    # tokens refill: one minute later novel frames flow again
    assert h.push(scene(50), 70.0).accepted


def test_state_is_per_device_and_counts_recorded():
    h = Harness()
    assert h.push(scene(1), 0.0, device="a").accepted
    assert h.push(scene(1), 0.5, device="b").accepted
    assert h.f.counts["first"] == 2
    h.f.reset("a")
    assert h.push(scene(1), 5.0, device="a").reason == "duplicate"  # recent hashes still provided by caller


def test_restart_uses_store_history_for_gap():
    """After a restart the in-memory state is empty; `recent` from the store still enforces too_soon/dedupe."""
    f = KeyframeFilter(settings())
    d = f.decide(scene(2), FrameMeta(device_id="d", ts=1000.5), [(1000.0, dhash(scene(1)))])
    assert d.reason == "too_soon"
    d = f.decide(scene(1), FrameMeta(device_id="d", ts=1005.0), [(1000.0, dhash(scene(1)))])
    assert d.reason == "duplicate"


def test_speed():
    import time

    img = scene(7, size=(960, 720))
    f = KeyframeFilter(settings())
    t0 = time.perf_counter()
    for i in range(20):
        f.decide(img, FrameMeta(device_id="d", ts=float(i)), [])
    assert (time.perf_counter() - t0) / 20 < 0.05
