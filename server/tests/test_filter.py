import random

from PIL import Image, ImageFilter

from megamind.config import Settings
from megamind.ingest.filter import KeyframeFilter, sharpness
from megamind.models import FrameMeta, Pose


def _noisy(seed: int, size=(160, 120)) -> Image.Image:
    rnd = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(rnd.randrange(256),) * 3 for _ in range(size[0] * size[1])])
    return img


def _meta(ts, pose=None, force=False):
    return FrameMeta(device_id="d", ts=ts, pose=pose, force=force)


def _filter(**kw) -> KeyframeFilter:
    return KeyframeFilter(Settings(_env_file=None, **kw))


def test_dark_and_blurry():
    f = _filter()
    assert f.decide(Image.new("RGB", (160, 120), (5, 5, 5)), _meta(0)).reason == "dark"
    blurred = _noisy(1).filter(ImageFilter.GaussianBlur(6))
    assert sharpness(blurred) < sharpness(_noisy(1))
    assert f.decide(blurred, _meta(0)).reason == "blurry"
    assert f.decide(blurred, _meta(0, force=True)).accepted


def test_gap_duplicate_heartbeat():
    f = _filter()
    a = _noisy(1)
    assert f.decide(a, _meta(0)).trigger == "novel"
    assert f.decide(_noisy(2), _meta(1)).reason == "too_soon"
    assert f.decide(a, _meta(5)).reason == "duplicate"
    d = f.decide(a, _meta(31))
    assert d.accepted and d.trigger == "heartbeat"


def test_compares_against_kept_history():
    f = _filter()
    a, b = _noisy(1), _noisy(2)
    assert f.decide(a, _meta(0)).accepted
    assert f.decide(b, _meta(3)).accepted
    assert f.decide(a, _meta(6)).reason == "duplicate"


def test_pose_trigger_overrides_duplicate():
    f = _filter()
    a = _noisy(1)
    assert f.decide(a, _meta(0, Pose(x=0, y=1.6, z=0, yaw=350))).accepted
    assert f.decide(a, _meta(3, Pose(x=0, y=1.6, z=0, yaw=5))).reason == "duplicate"
    d = f.decide(a, _meta(6, Pose(x=0, y=1.6, z=0, yaw=40)))
    assert d.accepted and d.trigger == "pose"
    d = f.decide(a, _meta(9, Pose(x=1.0, y=1.6, z=0, yaw=40)))
    assert d.trigger == "pose"


def test_budget_caps_novel_frames():
    f = _filter(filter_budget_per_min=6, filter_budget_burst=6, filter_heartbeat_s=0)
    decisions = [f.decide(_noisy(i), _meta(i * 2.0)) for i in range(60)]
    kept = sum(d.accepted for d in decisions)
    # 120 s of novel frames every 2 s: burst of 6 + 6/min refill over 2 min
    assert 15 <= kept <= 19
    assert any(d.reason == "budget" for d in decisions)


def test_seed_restores_state():
    f = _filter()
    a = _noisy(1)
    from megamind.ingest.filter import dhash

    f.seed("d", [(0.0, dhash(a))])
    assert f.decide(a, _meta(5)).reason == "duplicate"


def test_speech_frames_always_kept():
    f = _filter()
    a = _noisy(1)
    assert f.decide(a, _meta(0)).accepted
    m = FrameMeta(device_id="d", ts=0.5, transcript="call mom about the tickets")
    d = f.decide(a, m)
    assert d.accepted and d.trigger == "speech"
    dark = FrameMeta(device_id="d", ts=0.7, transcript="lights are off")
    assert f.decide(Image.new("RGB", (160, 120), (5, 5, 5)), dark).trigger == "speech"
    assert f.counts["speech"] == 2
