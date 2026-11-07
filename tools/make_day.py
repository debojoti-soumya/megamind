#!/usr/bin/env python3
"""Synthesize a first-person "day" video from scene photos + a gold question set, for eval.

    python tools/make_day.py --scenes /tmp/scenes --out /tmp/day   # -> day.mp4, gold.json

Each scene gets `--seconds` of 1 fps handheld-style footage (jittered crops, zoom, noise,
exposure drift). Some scenes carry a rendered sign/note so exact-text questions have answers.
gold.json holds scene time offsets (seconds from video start) and questions; tools/eval.py
maps them to absolute timestamps with the same --start that tools/feeder.py used.
Requires: pip install opencv-python-headless numpy pillow
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIGNS = {
    "office": ["WIFI: falcon-2026", "PW on sticky note"],
    "meeting_room": ["PROJECT FALCON", "DEMO FRIDAY 5PM"],
    "grocerystore": ["OAT MILK $4.99", "AISLE 7"],
    "library": ["QUIET ZONE", "ROOM 214"],
    "gym": ["LOCKER 42"],
}
QUESTIONS = [
    ("what is the wifi password", "hybrid", "office"),
    ("when is the demo for project falcon", "hybrid", "meeting_room"),
    ("how much did the oat milk cost", "hybrid", "grocerystore"),
    ("which locker number did I use", "hybrid", "gym"),
    ("what room number was the quiet zone", "hybrid", "library"),
    ("when was I in the kitchen", "visual", "kitchen"),
    ("bookshelves full of books", "visual", "library"),
    ("treadmills and exercise machines", "visual", "gym"),
    ("a bed with pillows", "visual", "bedroom"),
    ("a sofa and a tv in the living room", "visual", "livingroom"),
    ("eating at a restaurant with tables", "visual", "restaurant"),
    ("a long hallway with doors", "visual", "corridor"),
    ("supermarket shelves", "hybrid", "grocerystore"),
    ("people sitting around a conference table", "hybrid", "meeting_room"),
]
ORDER = ["bedroom", "kitchen", "corridor", "office", "meeting_room", "restaurant",
         "library", "gym", "grocerystore", "livingroom"]  # fmt: skip


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def add_sign(img: Image.Image, lines: list[str]) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    f = font(max(18, img.width // 22))
    w = max(d.textlength(t, font=f) for t in lines) + 30
    h = len(lines) * (f.size + 10) + 20
    x, y = img.width * 0.12, img.height * 0.2
    d.rectangle([x, y, x + w, y + h], fill=(250, 248, 235), outline=(40, 40, 40), width=3)
    for i, t in enumerate(lines):
        d.text((x + 15, y + 10 + i * (f.size + 10)), t, fill=(20, 20, 20), font=f)
    return img


def handheld(img: np.ndarray, rnd: random.Random, t: float, out_wh: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    z = 0.78 + 0.08 * np.sin(t / 7) + rnd.uniform(-0.02, 0.02)
    cw, ch = int(w * z), int(h * z)
    cx = int((w - cw) * (0.5 + 0.4 * np.sin(t / 11) + rnd.uniform(-0.05, 0.05)))
    cy = int((h - ch) * (0.5 + 0.3 * np.cos(t / 9) + rnd.uniform(-0.05, 0.05)))
    crop = img[max(0, cy) : max(0, cy) + ch, max(0, cx) : max(0, cx) + cw]
    frame = cv2.resize(crop, out_wh, interpolation=cv2.INTER_AREA).astype(np.float32)
    frame *= 0.9 + 0.15 * np.sin(t / 5) + rnd.uniform(-0.03, 0.03)
    frame += np.random.default_rng(int(t * 1000)).normal(0, 6, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=Path, required=True, help="dir with <scene>_<n>.jpg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=int, default=45)
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(0)
    out_wh = (args.width, args.width * 9 // 16)
    vw = cv2.VideoWriter(str(args.out / "day.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, out_wh)
    scenes, t = [], 0
    for name in ORDER:
        photos = sorted(args.scenes.glob(f"{name}_*.jpg"))
        if not photos:
            continue
        start = t
        for i in range(args.seconds):
            src = photos[(i * len(photos)) // args.seconds]
            pil = Image.open(src).convert("RGB")
            if pil.width < 800:
                pil = pil.resize((800, int(pil.height * 800 / pil.width)))
            if name in SIGNS and i % 15 < 10:
                pil = add_sign(pil, SIGNS[name])
            frame = handheld(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), rnd, t, out_wh)
            vw.write(frame)
            t += 1
        scenes.append({"scene": name, "start_s": start, "end_s": t})
    vw.release()
    present = {s["scene"] for s in scenes}
    gold = {
        "scenes": scenes,
        "questions": [{"q": q, "mode": m, "scene": s} for q, m, s in QUESTIONS if s in present],
    }
    (args.out / "gold.json").write_text(json.dumps(gold, indent=1))
    print(f"{t} frames, {len(scenes)} scenes -> {args.out}/day.mp4")


if __name__ == "__main__":
    main()
