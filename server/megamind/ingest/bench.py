"""Latency benchmark for the frame-ingest path (`POST /v1/frames`) on a synthetic 1 fps head-cam clip.

    python -m megamind.ingest.bench                        # 240 frames, memory + lancedb, 960x720 q75
    python -m megamind.ingest.bench --frames 600 --backends lancedb --json out.json
    python -m megamind.ingest.bench --frames-dir data/replay/<device>   # real frames: calibrate thresholds

The clip is deterministic and needs no camera, network or models: textured "rooms" seen through a
960x720 window, cycling through the cases the filter must handle - a static scene (duplicates), a head
turn (horizontal motion blur + yaw change), a dark moment, a dwell where an object is put down (a novel
frame) and a walk to the next room (pose move). Two numbers are reported:

- stage: per-frame CPU of each filter step (JPEG decode, dHash, luma, sharpness, `decide`), measured in
  isolation so regressions in `ingest/filter.py` show up without HTTP noise;
- e2e: wall time per `POST /v1/frames` through the ASGI app (multipart parse, decode, store lookup,
  filter, thumbnail write, enqueue), with mock enrichment, as frames/s and p50/p95 ms per backend.

`--frames-dir` replaces the synthetic clip with real JPEGs (e.g. a `REPLAY_SPOOL_DIR` device folder, whose
`<ts>_<ulid>.jpg` names carry the capture time) and also prints sharpness / luma percentiles, which is
how `FILTER_MIN_SHARPNESS` / `FILTER_MIN_LUMA` get calibrated on headset footage.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from megamind.config import Settings
from megamind.ingest.filter import KeyframeFilter, dhash, mean_luma, sharpness
from megamind.models import FrameMeta, Pose

T0 = 1_794_067_200.0  # 2026-11-07 08:00 PST


def _room(seed: int, w: int, h: int) -> np.ndarray:
    """A wide textured panorama: smooth shading + coloured rectangles + sensor noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 110 + 50 * np.sin(xx / rng.uniform(30, 90)) * np.cos(yy / rng.uniform(30, 90))
    img = np.repeat(base[..., None], 3, axis=2) + rng.uniform(-40, 40, 3)
    for _ in range(60):
        x0, y0 = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
        x1, y1 = x0 + int(rng.integers(20, 260)), y0 + int(rng.integers(20, 200))
        img[y0:y1, x0:x1] = rng.integers(0, 256, 3)
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _jpeg(a: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def synth_clip(
    n: int = 240, *, width: int = 960, quality: int = 75, seed: int = 0, device_id: str = "bench-quest"
) -> list[tuple[bytes, FrameMeta]]:
    """`n` frames at 1 fps: (jpeg, meta). 30 s per room: 12 s static, 4 s turn, 2 s dark, 12 s dwell
    with an object put down at 22 s."""
    height = width * 3 // 4
    pano_w = width * 3
    rng = np.random.default_rng(seed + 1)
    rooms: dict[int, np.ndarray] = {}
    out: list[tuple[bytes, FrameMeta]] = []
    for i in range(n):
        room, t = divmod(i, 30)
        if room not in rooms:
            rooms.clear()
            rooms[room] = _room(seed * 1000 + room, pano_w, height)
        pano = rooms[room]
        x, yaw, blur = 0, 90.0 * room, 0
        if 12 <= t < 16:  # head turn: window slides across the panorama, horizontally smeared
            x, yaw, blur = (t - 11) * width // 3, yaw + (t - 11) * 15.0, 25
        elif t >= 16:
            x, yaw = 4 * width // 3, yaw + 60.0
        a = pano[:, x : x + width].astype(np.int16)
        a = np.clip(a + rng.normal(0, 3, a.shape), 0, 255).astype(np.uint8)  # per-frame sensor noise
        if blur:
            c = np.cumsum(np.pad(a.astype(np.float32), ((0, 0), (blur, 0), (0, 0))), axis=1)
            a = ((c[:, blur:] - c[:, :-blur]) / blur).astype(np.uint8)
        if t >= 22:  # something put down on the counter: a new, stable scene
            a[height // 4 : height // 4 + height // 2, width // 4 : width // 4 + width // 2] = (230, 40, 40)
        if t in (16, 17):  # lights off / lens covered
            a = (a // 12).astype(np.uint8)
        pose = Pose(x=room * 3.0, y=1.6, z=0.0, yaw=yaw % 360.0)
        out.append((_jpeg(a, quality), FrameMeta(device_id=device_id, ts=T0 + i, pose=pose)))
    return out


def load_frames(directory: Path, *, device_id: str = "bench-quest") -> list[tuple[bytes, FrameMeta]]:
    """Real frames from a folder of JPEGs, sorted by name; `ts` from a `<float>_...jpg` prefix, else 1 fps."""
    out: list[tuple[bytes, FrameMeta]] = []
    for i, p in enumerate(sorted(Path(directory).glob("*.jp*g"))):
        try:
            ts = float(p.stem.split("_", 1)[0])
        except ValueError:
            ts = T0 + i
        out.append((p.read_bytes(), FrameMeta(device_id=device_id, ts=ts)))
    out.sort(key=lambda f: f[1].ts)
    return out


def metric_percentiles(clip: list[tuple[bytes, FrameMeta]]) -> dict[str, dict[str, float]]:
    """Distribution of the filter's pixel metrics over a clip, for threshold calibration."""
    vals: dict[str, list[float]] = {"sharpness": [], "mean_luma": []}
    for jpeg, _ in clip:
        img = Image.open(io.BytesIO(jpeg))
        vals["sharpness"].append(sharpness(img))
        vals["mean_luma"].append(mean_luma(img))
    return {k: {f"p{q}": float(np.percentile(v, q)) for q in (5, 10, 25, 50, 75, 90)} for k, v in vals.items()}


def _stats(samples_s: list[float]) -> dict[str, float]:
    ms = sorted(s * 1000.0 for s in samples_s)
    p95 = ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))]
    total = sum(samples_s)
    return {
        "n": len(ms),
        "p50_ms": statistics.median(ms),
        "p95_ms": p95,
        "mean_ms": statistics.fmean(ms),
        "fps": len(ms) / total if total > 0 else float("inf"),
    }


def bench_stages(clip: list[tuple[bytes, FrameMeta]], settings: Settings) -> dict[str, dict[str, float]]:
    """CPU per filter step, each timed in isolation over the whole clip."""

    def decode(jpeg: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(jpeg))
        img.load()
        return img

    imgs = [decode(j) for j, _ in clip]
    samples: dict[str, list[float]] = {k: [] for k in ("decode", "dhash", "mean_luma", "sharpness", "decide")}
    for (jpeg, _), img in zip(clip, imgs, strict=True):
        for name, fn, arg in (
            ("decode", decode, jpeg),
            ("dhash", dhash, img),
            ("mean_luma", mean_luma, img),
            ("sharpness", sharpness, img),
        ):
            start = time.perf_counter()
            fn(arg)
            samples[name].append(time.perf_counter() - start)
    kf = KeyframeFilter(settings)
    kept: list[tuple[float, str]] = []
    for (_, meta), img in zip(clip, imgs, strict=True):
        start = time.perf_counter()
        d = kf.decide(img, meta, [(t, h) for t, h in kept if meta.ts - t <= kf.history_window_s])
        samples["decide"].append(time.perf_counter() - start)
        if d.accepted:
            kept.append((meta.ts, d.frame_hash))
    out = {k: _stats(v) for k, v in samples.items()}
    out["decide"]["kept"] = len(kept)
    return out


async def bench_e2e(clip: list[tuple[bytes, FrameMeta]], backend: str, tmp: Path) -> dict[str, Any]:
    from httpx import ASGITransport, AsyncClient

    from megamind.main import create_app

    settings = Settings(
        _env_file=None,
        thumb_dir=tmp / "thumbs",
        audio_dir=tmp / "audio",
        store_backend=backend,
        lancedb_path=tmp / "lancedb",
        embed_provider="mock",
        image_embed_provider="none",
        reader_provider="mock",
        retriever_backend="hybrid",
        ucc_api_key="",
        consolidate_enabled=False,
    )
    app = create_app(settings)
    samples: list[float] = []
    reasons: dict[str, int] = {}
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://bench") as c:
            for jpeg, meta in clip:
                start = time.perf_counter()
                r = await c.post(
                    "/v1/frames",
                    files={"image": ("f.jpg", jpeg, "image/jpeg")},
                    data={"meta": meta.model_dump_json()},
                )
                samples.append(time.perf_counter() - start)
                body = r.json()
                key = "accepted" if body.get("accepted") else str(body.get("reason") or r.status_code)
                reasons[key] = reasons.get(key, 0) + 1
            await app.state.worker.queue.join()
    return {**_stats(samples), "reasons": dict(sorted(reasons.items()))}


async def run(
    frames: int,
    *,
    backends: list[str],
    width: int = 960,
    quality: int = 75,
    seed: int = 0,
    frames_dir: Path | None = None,
) -> dict[str, Any]:
    if frames_dir is not None:
        clip = load_frames(frames_dir)
        if not clip:
            raise SystemExit(f"no *.jpg frames in {frames_dir}")
    else:
        clip = synth_clip(frames, width=width, quality=quality, seed=seed)
    sizes = [len(j) for j, _ in clip]
    out: dict[str, Any] = {
        "frames": len(clip),
        "source": str(frames_dir) if frames_dir is not None else "synthetic",
        "width": width,
        "quality": quality,
        "jpeg_kb_mean": statistics.fmean(sizes) / 1024,
        "stages": bench_stages(clip, Settings(_env_file=None)),
        "e2e": {},
    }
    if frames_dir is not None:
        out["metrics"] = metric_percentiles(clip)
    for backend in backends:
        with tempfile.TemporaryDirectory(prefix="megamind-ingest-bench-") as tmp:
            out["e2e"][backend] = await bench_e2e(clip, backend, Path(tmp))
    return out


def _fmt(out: dict[str, Any]) -> str:
    lines = [f"{'stage / backend':<22}{'n':>6}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}{'frames/s':>10}"]
    rows = [(f"stage {k}", v) for k, v in out["stages"].items()] + [(f"e2e {k}", v) for k, v in out["e2e"].items()]
    for name, r in rows:
        lines.append(
            f"{name:<22}{r['n']:>6}{r['p50_ms']:>10.2f}{r['p95_ms']:>10.2f}{r['mean_ms']:>10.2f}{r['fps']:>10.0f}"
        )
    for k, v in out["e2e"].items():
        lines.append(f"e2e {k} outcomes: {v['reasons']}")
    for k, v in out.get("metrics", {}).items():
        lines.append(f"{k} percentiles: " + " ".join(f"{q}={x:.1f}" for q, x in v.items()))
    return "\n".join(lines)


def _positive_int(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=_positive_int, default=240, help="clip length at 1 fps (default 240)")
    ap.add_argument("--backends", default="memory,lancedb", help="comma list of memory,lancedb")
    ap.add_argument("--width", type=_positive_int, default=960, help="frame width, 4:3 (default 960)")
    ap.add_argument("--quality", type=_positive_int, default=75, help="JPEG quality (default 75)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--frames-dir", type=Path, help="benchmark real JPEGs from this folder instead (+ metric percentiles)"
    )
    ap.add_argument("--json", type=Path, help="also write raw results here")
    a = ap.parse_args(argv)
    backends = [b.strip() for b in a.backends.split(",") if b.strip()]
    if bad := set(backends) - {"memory", "lancedb"}:
        ap.error(f"unknown backend(s): {', '.join(sorted(bad))}")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out = asyncio.run(
        run(a.frames, backends=backends, width=a.width, quality=a.quality, seed=a.seed, frames_dir=a.frames_dir)
    )
    clip = out["source"] if a.frames_dir else f"synthetic {a.width}x{a.width * 3 // 4} q{a.quality}"
    print(f"{out['frames']} frames ({clip}), mean JPEG {out['jpeg_kb_mean']:.0f} KB")
    print(_fmt(out))
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
