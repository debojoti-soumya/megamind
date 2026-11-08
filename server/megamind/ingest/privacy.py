"""Bystander face blur for keyframes (ported from Recall's ``privacy.blur_faces``).

The wearer is behind the camera, so every detected face is a bystander. When enabled
(``PRIVACY_BLUR_FACES``), the enrichment worker blurs faces in the stored thumbnail, which is
also what the headset shows and what the reader VLM receives at query time. With
``PRIVACY_BLUR_FACES_VLM`` the blurred frame is also what the caption VLM / image embedder see,
so raw face pixels never leave the server.

Detection uses OpenCV's Haar cascade (``pip install -e ".[privacy]"``), imported lazily; without
OpenCV the blur is a no-op with a warning. The blur itself is done with Pillow, and the detector
is a plain callable, so tests can inject a fake one without OpenCV.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image, ImageFilter

from megamind.config import Settings

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # x, y, w, h in pixels
FaceDetector = Callable[[Image.Image], Sequence[Box]]

_PAD = 0.15  # grow each box so hairline/chin are covered too


@lru_cache(maxsize=1)
def haar_detector() -> FaceDetector | None:
    """OpenCV frontal-face Haar cascade, or ``None`` (with a warning) if OpenCV is not installed."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        log.warning('face blur requested but OpenCV is missing (pip install -e ".[privacy]"); faces are NOT blurred')
        return None

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    lock = threading.Lock()  # one classifier shared by the worker's threads

    def detect(img: Image.Image) -> list[Box]:
        gray = np.asarray(img.convert("L"))
        with lock:
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        return [tuple(int(v) for v in f) for f in faces]

    return detect


def blur_boxes(img: Image.Image, boxes: Sequence[Box]) -> Image.Image:
    """Gaussian-blur each (padded) box of an RGB image in place and return it."""
    W, H = img.size
    for x, y, w, h in boxes:
        pad = int(_PAD * max(w, h))
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
        if x1 <= x0 or y1 <= y0:
            continue
        region = img.crop((x0, y0, x1, y1))
        img.paste(region.filter(ImageFilter.GaussianBlur(radius=max(6, max(w, h) // 6))), (x0, y0))
    return img


def blur_faces(jpeg: bytes, detector: FaceDetector | None = None) -> tuple[bytes, int]:
    """Return ``(jpeg, n_faces)`` with every detected face blurred.

    The input bytes are returned unchanged (no re-encode) when no face is found, when the
    image cannot be decoded, or when no detector is available.
    """
    detector = detector or haar_detector()
    if detector is None:
        return jpeg, 0
    try:
        src = Image.open(io.BytesIO(jpeg))
        img = src.convert("RGB")
    except Exception:  # noqa: BLE001
        return jpeg, 0
    boxes = list(detector(img))
    if not boxes:
        return jpeg, 0
    buf = io.BytesIO()
    exif = src.info.get("exif")  # keep the orientation tag: boxes are in stored-pixel coordinates
    blur_boxes(img, boxes).save(buf, format="JPEG", quality=90, **({"exif": exif} if exif else {}))
    return buf.getvalue(), len(boxes)


@dataclass
class FaceBlur:
    """What the worker applies: always the stored thumbnail; the VLM/embedder input iff ``upstream``."""

    detector: FaceDetector
    upstream: bool = False

    async def apply(self, jpeg: bytes) -> tuple[bytes, int]:
        return await asyncio.to_thread(blur_faces, jpeg, self.detector)


def build_face_blur(settings: Settings) -> FaceBlur | None:
    if not (settings.privacy_blur_faces or settings.privacy_blur_faces_vlm):
        return None
    detector = haar_detector()
    if detector is None:
        return None
    return FaceBlur(detector, upstream=settings.privacy_blur_faces_vlm)
