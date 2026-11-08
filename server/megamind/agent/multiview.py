"""Multi-view 3D: many stored photos of one object -> VGGT -> a coloured point cloud with a back and sides.

VGGT (Meta, pretrained, feed-forward; no per-object fitting) predicts a world point and a confidence for every
pixel of every photo at once. `clean_points` keeps confident pixels inside the detector's box of the object in
each photo, drops the support plane (table, plinth, floor) and strays, and stands the object upright on y=0.
Weights `facebook/VGGT-1B` are CC-BY-NC-4.0 (non-commercial)."""

from __future__ import annotations

import importlib.util
import io
import threading

import numpy as np
from PIL import Image, ImageOps

SIDE = 518  # VGGT's input width; heights are multiples of its 14 px patches
PATCH = 14


def frame_size(w: int, h: int) -> tuple[int, int, int]:
    """(resized height, crop top, cropped height) for VGGT's 'crop' preprocessing of a w x h photo."""
    nh = max(PATCH, round(h * (SIDE / w) / PATCH) * PATCH)
    top = (nh - SIDE) // 2 if nh > SIDE else 0
    return nh, top, min(nh, SIDE)


def prepare(images: list[bytes], boxes: list[list[float] | None]) -> tuple[np.ndarray, np.ndarray]:
    """Photos -> float32 (N,3,H,W) in [0,1] at one common size (the first photo's), and (N,H,W) box masks
    (the box grown 10%; all True for a photo without a box). A photo of another shape is centre-cropped to the
    first one's aspect ratio rather than stretched."""
    first = ImageOps.exif_transpose(Image.open(io.BytesIO(images[0])))
    fw, fh = first.size
    nh, top, ch = frame_size(fw, fh)
    out = np.empty((len(images), 3, ch, SIDE), np.float32)
    masks = np.ones((len(images), ch, SIDE), bool)
    for i, (data, box) in enumerate(zip(images, boxes, strict=True)):
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        w, h = img.size
        cw, chh = (h * fw / fh, h) if w * fh > h * fw else (w, w * fh / fw)
        ox, oy = (w - cw) / 2, (h - chh) / 2
        img = img.resize((SIDE, nh), Image.BICUBIC, box=(ox, oy, ox + cw, oy + chh))
        out[i] = (np.asarray(img, np.float32)[top : top + ch] / 255.0).transpose(2, 0, 1)
        if box is not None:
            x0, y0, x1, y1 = box
            gx, gy = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            sx, sy = SIDE / cw, nh / chh
            v, u = np.mgrid[0:ch, 0:SIDE]
            v = v + top
            masks[i] = (
                (u >= (x0 - gx - ox) * sx)
                & (u <= (x1 + gx - ox) * sx)
                & (v >= (y0 - gy - oy) * sy)
                & (v <= (y1 + gy - oy) * sy)
            )
    return out, masks


def _plane(p: np.ndarray, thr: float, rng: np.random.Generator, iters: int = 300) -> tuple[np.ndarray, np.ndarray, int]:
    """RANSAC: (unit normal, point on plane, inlier count) of the dominant plane."""
    sample = p[rng.choice(len(p), min(len(p), 40000), replace=False)]
    best = (np.array([0.0, -1.0, 0.0]), p[0], -1)
    for _ in range(iters):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        k = int((np.abs((sample - a) @ n) < thr).sum())
        if k > best[2]:
            best = (n, a, k)
    n, a, _ = best
    return n, a, int((np.abs((p - a) @ n) < thr).sum())


def clean_points(
    pts: np.ndarray,
    conf: np.ndarray,
    rgb: np.ndarray,
    masks: np.ndarray | None = None,
    *,
    conf_pct: float = 50.0,
    max_points: int = 300_000,
    min_plane: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-pixel world points (N,H,W,3), confidences (N,H,W), colours (N,H,W,3 uint8) -> the object's points
    (float32, y up, standing on y=0, centred in x/z) and colours. Raises ValueError when nothing is left."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(pts).all(-1) & np.isfinite(conf)
    if masks is not None:
        ok &= masks
    if ok.any():  # against the object's own points: a confident background mustn't push them all out
        ok &= conf >= np.percentile(conf[ok], conf_pct)
    p, c = pts[ok].astype(np.float64), rgb[ok]
    if len(p) < 500:
        raise ValueError("too few confident points of the object")
    scale = float(np.median(np.linalg.norm(p - np.median(p, 0), axis=1))) or 1.0
    thr = 0.02 * scale
    n, a, inliers = _plane(p, thr, rng)
    plane = inliers >= min_plane * len(p)
    if plane:
        d = (p - a) @ n
        off = np.abs(d) >= thr
        if off.any() and np.median(d[off]) < 0:  # the object is on the side most off-plane points are
            n, d = -n, -d
        keep = d > 1.5 * thr
        p, c = p[keep], c[keep]
        if len(p) < 500:
            raise ValueError("only a flat surface was found, not an object on it")
    else:
        n = np.array([0.0, -1.0, 0.0])  # VGGT's world is the first camera's (OpenCV: y down)
    ctr = np.median(p, 0)
    r = np.linalg.norm(p - ctr, axis=1)
    keep = r < np.percentile(r, 90) * 1.3
    p, c = p[keep], c[keep]
    y = n / np.linalg.norm(n)
    x = np.cross([0.0, 0.0, 1.0], y)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    q = (p - ctr) @ np.stack([x, y, z]).T
    q[:, 1] -= (a - ctr) @ y if plane else q[:, 1].min()
    q[:, 0] -= np.median(q[:, 0])
    q[:, 2] -= np.median(q[:, 2])
    if len(q) > max_points:
        idx = rng.choice(len(q), max_points, replace=False)
        q, c = q[idx], c[idx]
    return q.astype(np.float32), c.astype(np.uint8), {"points": len(q), "plane_removed": bool(plane)}


def to_points_ply(points: np.ndarray, colors: np.ndarray) -> bytes:
    head = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode()
    rec = np.empty(len(points), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = points, colors
    return head + rec.tobytes()


def ply_vertex_count(data: bytes) -> int:
    head = data.split(b"end_header\n", 1)[0].decode()
    return next(int(line.split()[2]) for line in head.splitlines() if line.startswith("element vertex"))


class VGGTRunner:
    """VGGT loaded on first use; one reconstruction at a time (~10 GB RAM for 16 photos on CPU)."""

    def __init__(self, weights: str = "facebook/VGGT-1B") -> None:
        self.weights = weights
        self._lock = threading.Lock()
        self._model = None

    @staticmethod
    def missing() -> str | None:
        gone = [m for m in ("torch", "vggt") if importlib.util.find_spec(m) is None]
        if gone:
            return (
                f"multi-view 3D needs {' and '.join(gone)} (pip install git+https://github.com/facebookresearch/vggt)"
            )
        return None

    def _load(self):
        if self._model is None:
            import os

            import torch
            from vggt.models.vggt import VGGT

            if os.path.isfile(self.weights):
                m = VGGT()
                m.load_state_dict(torch.load(self.weights, map_location="cpu"))
            else:
                m = VGGT.from_pretrained(self.weights)
            self._model = m.eval()
        return self._model

    def run(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,3,H,W) in [0,1] -> world points (N,H,W,3) and their confidences (N,H,W)."""
        import torch

        with self._lock:
            model = self._load()
            with torch.inference_mode():
                pred = model(torch.from_numpy(images))
        return pred["world_points"][0].float().numpy(), pred["world_points_conf"][0].float().numpy()
