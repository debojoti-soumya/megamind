"""3D from memory: "make a 3D model of my mug" -> the best stored photo of it -> the object cut out -> 3D -> viewer.

1. Photos: `collage.find_photos` (same subject detection, Haiku verifier and caption filter as the photo montage).
2. Best frame + object box: Grounding DINO (zero-shot, the subject as prompt) over the top `MODEL3D_FRAMES_TRIED`
   photos; the most confident detection wins (whole-frame box when it finds nothing).
3. Mask: SAM (box prompt), the fullest of its confident masks.
4. 3D, `MODEL3D_BACKEND`:
   - `local` (default, CPU, ~10 s): Depth Anything V2 relative depth; the masked pixels are back-projected into a
     vertex-coloured mesh (.glb) or, `MODEL3D_LOCAL_FORMAT=splat`, one small Gaussian per pixel (.ply splat).
     A relief: only the side the camera saw, scaled to `MODEL3D_SIZE_M`, not metric.
   - `remote`: POST image + mask + prompt to `MODEL3D_REMOTE_URL` (e.g. SAM 3D Objects on a GPU box), which answers
     with a .glb mesh or a .ply Gaussian splat of the whole object.
5. `GET /v1/models3d/{id}.{glb|ply}`; `/ui/3d?m={id}&f={glb|ply}` is a three.js viewer with WebXR (Quest Browser:
   "View in AR" places it in the room over passthrough).

Heavy deps are optional (`pip install -e ".[local-3d]"` + CPU torch); without them the endpoint answers 503.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import secrets
import struct
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic_settings import BaseSettings, SettingsConfigDict

from megamind.agent import smart_montage as smart
from megamind.agent.collage import CollageError, find_photos
from megamind.agent.montage import _load_image, _ts_param, _zone
from megamind.models import Model3DRequest, Model3DResponse

log = logging.getLogger("megamind.model3d")
router = APIRouter()
_ID_RE = re.compile(r"[0-9a-f]{20}")
_MEDIA = {"glb": "model/gltf-binary", "ply": "application/octet-stream"}

MAKE_3D_TOOL = {
    "name": "make_3d_model",
    "description": (
        "Turn something the wearer saw into a 3D model they can look at in the headset: finds the best stored "
        "photo of it, cuts the object out and lifts it into 3D. Use it when the user asks for a 3D model, a 3D "
        "view, a hologram or a Gaussian splat of something, or to see something 'in 3D'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The object, as a short phrase (e.g. 'my blue mug')."},
            "time_from": {"type": ["string", "null"], "description": "ISO-8601 lower bound or null."},
            "time_to": {"type": ["string", "null"], "description": "ISO-8601 upper bound or null."},
        },
        "required": ["query"],
    },
}

MODEL3D_PROMPT = """

You also have make_3d_model, which turns an object from the wearer's photos into a 3D model shown on their
panel. Use it when they ask for a 3D model, a 3D view, a hologram or a splat of something. Then say in one
sentence what was modelled and when the photo is from; if it is only a relief (full_3d false), say it only
shows the side they saw. If it fails, say why."""

_WANTS = re.compile(r"\b(?:(?:3d|3-d|three[- ]d)\b(?!\s*-?print)|holograms?\b|(?:gaussian\s+)?splats?\b)", re.I)
_FRAMING = re.compile(
    r"^\s*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(?:make|create|build|generate|give|show|turn|render|reconstruct)?\s*(?:me\s+)?(?:an?\s+|the\s+)?"
    r"(?:(?:3d|3-d|three[- ]d)\s*(?:model|version|view|scan|object|mesh)?|hologram|(?:gaussian\s+)?splats?)\s*"
    r"(?:of|from|for)?\s+",
    re.I,
)
_SUFFIX = re.compile(
    r"\s+(?:in|into|as)\s+(?:an?\s+)?"
    r"(?:(?:3d|3-d|three[- ]d)(?:\s+(?:model|version|view|scan|object|mesh))?|hologram|(?:gaussian\s+)?splats?)\b",
    re.I,
)


def wants_3d(text: str) -> bool:
    return bool(_WANTS.search(text or ""))


def strip_3d_framing(query: str) -> str:
    """'make a 3D model of my mug' -> 'my mug'; 'show me the red car in 3d' -> 'show me the red car'."""
    q = " ".join(_SUFFIX.sub(" ", " ".join((query or "").split())).split())
    return _FRAMING.sub("", q, count=1).strip(" ?.!")


class Model3DSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL3D_", env_file=".env", extra="ignore")

    enabled: bool = True
    agent_tool: bool = True  # offer make_3d_model to the chat agent
    backend: str = "local"  # local | remote
    local_format: str = "glb"  # local backend output: glb (coloured mesh) | splat (.ply Gaussian splat)
    remote_url: str = ""  # backend=remote: POST multipart image, mask (png, optional), prompt -> .glb or .ply
    remote_token: str = ""  # sent as "Authorization: Bearer ..."
    remote_timeout_s: float = 300.0
    detector: str = "IDEA-Research/grounding-dino-tiny"
    detector_threshold: float = 0.25
    sam: str = "facebook/sam-vit-base"
    depth: str = "depth-anything/Depth-Anything-V2-Small-hf"
    frames_tried: int = 3  # best photos the detector compares to pick the frame
    max_side: int = 384  # mesh resolution: vertices along the photo's longest side
    size_m: float = 0.4  # longest side of the shown object (relief scale is not metric)
    fov_deg: float = 64.0  # horizontal field of view assumed for back-projection
    cache_dir: Path | None = None  # default <thumb_dir>/../models3d
    cache_max_files: int = 16


# ---- geometry (pure numpy) ----


@dataclass
class Mesh:
    positions: np.ndarray  # (N, 3) float32, metres, y up, facing +z
    colors: np.ndarray  # (N, 3) uint8
    faces: np.ndarray  # (M, 3) uint32


def _erode(m: np.ndarray, n: int) -> np.ndarray:
    for _ in range(n):
        e = m.copy()
        e[1:] &= m[:-1]
        e[:-1] &= m[1:]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        e[0], e[-1], e[:, 0], e[:, -1] = False, False, False, False
        m = e
    return m


def relief_mesh(
    rgb: np.ndarray, mask: np.ndarray, disparity: np.ndarray, *, fov_deg: float = 64.0, size_m: float = 0.4
) -> Mesh:
    """Back-project the masked pixels of one photo into a mesh: relative disparity (bigger = nearer, any scale)
    -> depth, two triangles per pixel quad whose corners are all in the mask without a depth jump (silhouette and
    background bleed are cut), centred on the floor (y=0), longest side `size_m`."""
    h, w = mask.shape
    d = disparity.astype(np.float64)
    d = (d - d.min()) / (d.max() - d.min() + 1e-9)
    z = 1.0 / (0.15 + 0.85 * d)
    m = _erode(mask.astype(bool), 2)
    if m.sum() < 16:
        raise ValueError("the object mask is too small to model")
    zm = np.median(z[m])
    mad = np.median(np.abs(z[m] - zm)) + 1e-6
    m &= np.abs(z - zm) < 4 * 1.4826 * mad  # depth outliers: background bleeding into the mask
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    v, u = np.mgrid[0:h, 0:w]
    pts = np.stack([(u - w / 2) * z / f, -(v - h / 2) * z / f, -z], -1)
    idx = np.full((h, w), -1, np.int64)
    idx[m] = np.arange(int(m.sum()))
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    zq = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]])
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (e >= 0) & ((zq.max(0) - zq.min(0)) < 0.04 * zq.mean(0))
    faces = np.concatenate([np.stack([a[ok], c[ok], b[ok]], 1), np.stack([b[ok], c[ok], e[ok]], 1)])
    if not len(faces):
        raise ValueError("the object has no surface to model")
    used = np.unique(faces)
    remap = np.full(int(m.sum()), -1, np.int64)
    remap[used] = np.arange(len(used))
    verts = pts[m][used]
    cols = rgb[m][used]
    lo, hi = verts.min(0), verts.max(0)
    verts = verts - np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    verts *= size_m / max(float((hi - lo).max()), 1e-9)
    return Mesh(verts.astype(np.float32), cols.astype(np.uint8), remap[faces].astype(np.uint32))


def to_glb(mesh: Mesh, name: str = "memory") -> bytes:
    """Binary glTF 2.0: one unlit, double-sided, vertex-coloured triangle mesh (three.js / Unity glTFast load it)."""
    pos = np.ascontiguousarray(mesh.positions, "<f4")
    col = np.ascontiguousarray(np.c_[mesh.colors, np.full(len(mesh.colors), 255, np.uint8)], np.uint8)
    ind = np.ascontiguousarray(mesh.faces, "<u4").ravel()
    n, k = len(pos), len(ind)
    blob = pos.tobytes() + col.tobytes() + ind.tobytes()
    doc = {
        "asset": {"version": "2.0", "generator": "megamind model3d"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {"name": name, "primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "material": 0}]}
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1},
                "doubleSided": True,
                "extensions": {"KHR_materials_unlit": {}},
            }
        ],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 12 * n, "target": 34962},
            {"buffer": 0, "byteOffset": 12 * n, "byteLength": 4 * n, "target": 34962},
            {"buffer": 0, "byteOffset": 16 * n, "byteLength": 4 * k, "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": n,
                "type": "VEC3",
                "min": pos.min(0).tolist(),
                "max": pos.max(0).tolist(),
            },
            {"bufferView": 1, "componentType": 5121, "normalized": True, "count": n, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5125, "count": k, "type": "SCALAR"},
        ],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )


def to_splat_ply(mesh: Mesh) -> bytes:
    """3D Gaussian splat .ply (the INRIA layout splat viewers read): one isotropic, opaque Gaussian per vertex,
    sized to the local vertex spacing, its colour as the degree-0 SH coefficient."""
    p = mesh.positions.astype(np.float32)
    n = len(p)
    e = mesh.faces.astype(np.int64)
    edge = np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1) if len(e) else np.array([0.002])
    s = np.log(max(float(np.median(edge)), 1e-5) * 0.75)
    sh = (mesh.colors.astype(np.float32) / 255.0 - 0.5) / 0.28209479
    rows = np.zeros((n, 14), np.float32)
    rows[:, 0:3] = p
    rows[:, 3:6] = sh
    rows[:, 6] = 4.0  # logit opacity ~0.98
    rows[:, 7:10] = s
    rows[:, 10] = 1.0  # rotation quaternion w
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"]
    props += ["rot_0", "rot_1", "rot_2", "rot_3"]
    head = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
    head += "".join(f"property float {q}\n" for q in props) + "end_header\n"
    return head.encode() + rows.astype("<f4").tobytes()


# ---- models (lazy, CPU) ----


@dataclass
class Lifted:
    data: bytes
    fmt: str  # glb | ply
    kind: str  # mesh | splat
    full_3d: bool  # False: a relief of the side the camera saw
    box: list[float] | None = None
    score: float = 0.0
    stats: dict = field(default_factory=dict)


class LocalModels:
    """Grounding DINO + SAM + Depth Anything, loaded on first use, one inference at a time."""

    def __init__(self, cfg: Model3DSettings) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._m: dict | None = None

    @staticmethod
    def missing() -> str | None:
        gone = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
        if gone:
            return f"3D needs {' and '.join(gone)} (pip install -e '.[local-3d]' plus CPU torch)"
        return None

    def _load(self) -> dict:
        if self._m is None:
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                SamModel,
                SamProcessor,
            )

            c = self.cfg
            self._m = {
                "dp": AutoProcessor.from_pretrained(c.detector),
                "dm": AutoModelForZeroShotObjectDetection.from_pretrained(c.detector).eval(),
                "sp": SamProcessor.from_pretrained(c.sam),
                "sm": SamModel.from_pretrained(c.sam).eval(),
                "zp": AutoImageProcessor.from_pretrained(c.depth),
                "zm": AutoModelForDepthEstimation.from_pretrained(c.depth).eval(),
            }
        return self._m

    def detect(self, img: Image.Image, prompt: str) -> tuple[list[float], float]:
        """(best box x0,y0,x1,y1 in pixels, score); the centre 80% of the frame and 0 when nothing is found."""
        import torch

        m = self._load()
        with self._lock, torch.no_grad():
            inp = m["dp"](images=img, text=prompt.lower().strip(" .") + ".", return_tensors="pt")
            out = m["dm"](**inp)
            res = m["dp"].post_process_grounded_object_detection(
                out,
                inp["input_ids"],
                threshold=self.cfg.detector_threshold,
                text_threshold=self.cfg.detector_threshold,
                target_sizes=[img.size[::-1]],
            )[0]
        if len(res["scores"]):
            i = int(res["scores"].argmax())
            return [float(x) for x in res["boxes"][i]], float(res["scores"][i])
        w, h = img.size
        return [0.1 * w, 0.1 * h, 0.9 * w, 0.9 * h], 0.0

    def segment(self, img: Image.Image, box: list[float]) -> np.ndarray:
        import torch

        m = self._load()
        with self._lock, torch.no_grad():
            inp = m["sp"](img, input_boxes=[[box]], return_tensors="pt")
            out = m["sm"](**inp, multimask_output=True)
            masks = m["sp"].image_processor.post_process_masks(
                out.pred_masks, inp["original_sizes"], inp["reshaped_input_sizes"]
            )[0][0]
        scores = out.iou_scores[0, 0]
        area = masks.flatten(1).float().mean(1)
        k = int((scores * area.sqrt() * (scores > scores.max() - 0.08)).argmax())  # fullest confident mask
        return masks[k].numpy().astype(bool)

    def disparity(self, img: Image.Image) -> np.ndarray:
        import torch

        m = self._load()
        with self._lock, torch.no_grad():
            pred = m["zm"](**m["zp"](img, return_tensors="pt")).predicted_depth
            out = torch.nn.functional.interpolate(pred[None], size=img.size[::-1], mode="bicubic")[0, 0]
        return out.numpy()


def _open(data: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def _small(img: Image.Image, side: int) -> Image.Image:
    s = side / max(img.size)
    return img if s >= 1 else img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)


def _png(mask: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buf, format="PNG")
    return buf.getvalue()


def choose_frame(models: LocalModels, images: list[bytes], prompt: str) -> tuple[int, list[float], float]:
    """Index of the photo where the detector is surest it sees `prompt`, with that box; the first photo when
    nothing is detected anywhere."""
    best = (0, None, -1.0)
    for i, data in enumerate(images):
        box, score = models.detect(_open(data), prompt)
        if score > best[2]:
            best = (i, box, score)
    if best[1] is None:
        box, score = models.detect(_open(images[0]), prompt)
        best = (0, box, score)
    return best  # type: ignore[return-value]


def lift_local(models: LocalModels, cfg: Model3DSettings, data: bytes, box: list[float]) -> Lifted:
    img = _open(data)
    mask = models.segment(img, box)
    small = _small(img, cfg.max_side)
    mask_s = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(small.size, Image.NEAREST)) > 127
    disp = models.disparity(small)
    mesh = relief_mesh(np.asarray(small), mask_s, disp, fov_deg=cfg.fov_deg, size_m=cfg.size_m)
    stats = {"vertices": len(mesh.positions), "faces": len(mesh.faces), "mask_px": int(mask.sum())}
    if cfg.local_format == "splat":
        return Lifted(to_splat_ply(mesh), "ply", "splat", False, box, stats=stats)
    return Lifted(to_glb(mesh), "glb", "mesh", False, box, stats=stats)


async def lift_remote(cfg: Model3DSettings, data: bytes, mask: np.ndarray | None, prompt: str) -> Lifted:
    """`MODEL3D_REMOTE_URL` contract: multipart `image` (jpeg), optional `mask` (png, white = object) and `prompt`;
    answers with a binary glTF (mesh) or a .ply (Gaussian splat when it has `f_dc_0`, else points)."""
    files = {"image": ("frame.jpg", data, "image/jpeg")}
    if mask is not None:
        files["mask"] = ("mask.png", _png(mask), "image/png")
    headers = {"Authorization": f"Bearer {cfg.remote_token}"} if cfg.remote_token else {}
    async with httpx.AsyncClient(timeout=cfg.remote_timeout_s) as client:
        r = await client.post(cfg.remote_url, files=files, data={"prompt": prompt}, headers=headers)
    if r.status_code != 200:
        raise CollageError(502, f"the 3D service failed ({r.status_code})")
    body = r.content
    if body[:4] == b"glTF":
        return Lifted(body, "glb", "mesh", True)
    if body[:4] == b"ply\n":
        head = body[: body.find(b"end_header") + 10]
        return Lifted(body, "ply", "splat" if b"f_dc_0" in head else "points", True)
    raise CollageError(502, "the 3D service answered with neither a .glb nor a .ply")


# ---- cache ----


class Model3DCache:
    def __init__(self, cache_dir: Path, max_files: int) -> None:
        self.dir, self.max_files = Path(cache_dir), max(1, max_files)
        self.generation = 0  # bumped by `clear` (DELETE /v1/memories)
        self.meta: dict[str, Model3DResponse] = {}

    def find(self, key: str) -> Path | None:
        for ext in _MEDIA:
            if (p := self.dir / f"{key}.{ext}").is_file():
                return p
        return None

    def put(self, key: str, fmt: str, data: bytes) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"{key}.{fmt}"
        tmp = p.with_suffix(f".{os.getpid()}.{secrets.token_hex(4)}.tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        files = sorted(
            (f for ext in _MEDIA for f in self.dir.glob(f"*.{ext}")), key=lambda f: f.stat().st_mtime, reverse=True
        )
        for old in files[self.max_files :]:
            old.unlink(missing_ok=True)
            self.meta.pop(old.stem, None)
        return p

    def clear(self) -> None:
        self.generation += 1
        self.meta.clear()
        for ext in _MEDIA:
            for f in self.dir.glob(f"*.{ext}") if self.dir.is_dir() else []:
                f.unlink(missing_ok=True)


# ---- pipeline ----


async def make_model3d(
    st, query: str, t0: float | None = None, t1: float | None = None, device_id: str | None = None
) -> Model3DResponse:
    """Find the photos -> best frame -> mask -> 3D. Raises `CollageError` (as `find_photos`; 503 3D deps or
    service missing, 502 the 3D service failed, 422 nothing to model in the photo, 409 memories deleted)."""
    cache: Model3DCache | None = getattr(st, "model3d_cache", None)
    if cache is None:
        raise CollageError(404, "3D models are disabled (MODEL3D_ENABLED=false)")
    cfg: Model3DSettings = st.model3d_settings
    models: LocalModels = st.model3d_models
    remote = cfg.backend == "remote"
    if remote and not cfg.remote_url:
        raise CollageError(503, "MODEL3D_BACKEND=remote needs MODEL3D_REMOTE_URL")
    missing = models.missing()
    if missing and not remote:
        raise CollageError(503, missing)
    subject_q = strip_3d_framing(query)
    found = await find_photos(st, subject_q, t0, t1, device_id)
    if not found.phrase:
        raise CollageError(400, "say which object to make in 3D")
    phrase = found.phrase
    prompt = " ".join(smart._content_words(phrase)) or phrase
    gen = cache.generation

    picks = found.picks[: max(1, cfg.frames_tried)]
    images, kept = [], []
    for p in picks:
        if data := await _load_image(st, p.token):
            images.append(data)
            kept.append(p)
    if not images:
        raise CollageError(404, "the matching moments have no stored photos left")

    box, score, i = None, 0.0, 0
    if not missing:
        i, box, score = await asyncio.to_thread(choose_frame, models, images, prompt)
    pick, data = kept[i], images[i]
    tag = f"remote:{cfg.remote_url}" if remote else f"local:{cfg.local_format}:{cfg.max_side}:{cfg.size_m}"
    key = hashlib.sha256(repr(("v1", tag, pick.token.id, prompt)).encode()).hexdigest()[:20]

    p = cache.find(key)
    if p is None or key not in cache.meta:
        try:
            if remote:
                mask = await asyncio.to_thread(models.segment, _open(data), box) if box is not None else None
                lifted = await lift_remote(cfg, data, mask, prompt)
                lifted.box, lifted.score = box, score
            else:
                lifted = await asyncio.to_thread(lift_local, models, cfg, data, box)
                lifted.score = score
        except ValueError as e:
            raise CollageError(422, f"I couldn't model {smart.spoken(phrase)}: {e}") from None
        except httpx.HTTPError as e:
            raise CollageError(502, f"the 3D service is unreachable ({type(e).__name__})") from None
        if cache.generation != gen:
            raise CollageError(409, "memories were deleted while the model was being made; try again")
        p = cache.put(key, lifted.fmt, lifted.data)  # no await since the generation check: a purge can't slip in
        zone = _zone(st.settings.wearer_tz)
        when = datetime.fromtimestamp(pick.token.ts, zone).strftime("%A at %-I:%M %p")
        said = smart.spoken(phrase)
        narration = f"Here's {said} in 3D, from {when}."
        if not lifted.full_3d:
            narration += " It only shows the side you saw."
        if score == 0.0 and not missing:
            narration += f" I couldn't spot {said} exactly, so it's the middle of the photo."
        cache.meta[key] = Model3DResponse(
            id=key,
            url=f"/v1/models3d/{key}.{lifted.fmt}",
            viewer_url=f"/ui/3d?m={key}&f={lifted.fmt}",
            format=lifted.fmt,
            kind=lifted.kind,
            full_3d=lifted.full_3d,
            backend=cfg.backend,
            title=found.title,
            narration=narration,
            subject=phrase,
            verified=pick.verified,
            token_id=pick.token.id,
            ts=pick.token.ts,
            thumb_url=pick.token.thumb_url,
            box=[round(v, 1) for v in lifted.box] if lifted.box else None,
            detection_score=round(score, 3),
            candidates=len(found.candidates),
            vertices=int(lifted.stats.get("vertices", 0)),
            size_bytes=len(lifted.data),
        )
        log.info("model3d %s: %r -> %s %s (%s)", key, query, lifted.kind, p.name, lifted.stats)
    return cache.meta[key]


# ---- HTTP ----


@router.post("/model3d", response_model=Model3DResponse)
async def create_model3d(request: Request, body: Model3DRequest):
    t0, t1 = _ts_param(body.time_from, "time_from"), _ts_param(body.time_to, "time_to")
    try:
        return await make_model3d(request.app.state, body.query, t0, t1, body.device_id or None)
    except CollageError as e:
        raise HTTPException(e.status, e.detail) from None


@router.api_route("/models3d/{model_id}.{fmt}", methods=["GET", "HEAD"])
async def get_model3d(request: Request, model_id: str, fmt: str):
    cache: Model3DCache | None = getattr(request.app.state, "model3d_cache", None)
    if cache is None or fmt not in _MEDIA or not _ID_RE.fullmatch(model_id):
        raise HTTPException(404)
    p = cache.dir / f"{model_id}.{fmt}"
    if not p.is_file():
        raise HTTPException(404, "unknown or expired 3D model; POST /v1/model3d again")
    return FileResponse(p, media_type=_MEDIA[fmt], headers={"Cache-Control": "private, max-age=300"})


def install(app: FastAPI, settings, model3d_settings: Model3DSettings | None = None) -> None:
    """Attach the 3D endpoints (after `collage.install`, whose photo matching it reuses)."""
    cfg = model3d_settings or Model3DSettings()
    st = app.state
    st.model3d_settings = cfg
    st.model3d_cache = None
    st.model3d_models = LocalModels(cfg)
    if cfg.enabled:
        st.model3d_cache = Model3DCache(
            cfg.cache_dir or Path(settings.thumb_dir).parent / "models3d", cfg.cache_max_files
        )
    app.include_router(router, prefix="/v1")


def agent_tool(st):
    """The chat agent's `make_3d_model` callable, or None when disabled."""
    cfg = getattr(st, "model3d_settings", None)
    if getattr(st, "model3d_cache", None) is None or cfg is None or not cfg.agent_tool:
        return None

    async def run(query: str, t0: float | None, t1: float | None, device_id: str | None) -> Model3DResponse:
        return await make_model3d(st, query, t0, t1, device_id)

    return run
