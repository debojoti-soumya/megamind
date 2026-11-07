"""Keyframe filter: perceptual-hash dedupe, minimum gap, darkness check."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from megamind.config import Settings


def dhash(img: Image.Image, size: int = 8) -> str:
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{size * size // 4}x}"


def hamming(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def mean_luma(img: Image.Image) -> float:
    return float(np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32).mean())


@dataclass
class FilterDecision:
    accepted: bool
    reason: str | None
    frame_hash: str


class KeyframeFilter:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def decide(
        self,
        img: Image.Image,
        ts: float,
        recent: list[tuple[float, str]],
    ) -> FilterDecision:
        h = dhash(img)
        if mean_luma(img) < self.s.filter_min_luma:
            return FilterDecision(False, "dark", h)
        if recent:
            last_ts = max(t for t, _ in recent)
            if ts - last_ts < self.s.filter_min_gap_s:
                return FilterDecision(False, "too_soon", h)
            window = [fh for t, fh in recent if ts - t <= self.s.filter_dhash_window_s]
            if any(hamming(h, fh) < self.s.filter_dhash_max_distance for fh in window):
                return FilterDecision(False, "duplicate", h)
        return FilterDecision(True, None, h)
