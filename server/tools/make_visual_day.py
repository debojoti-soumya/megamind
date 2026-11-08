#!/usr/bin/env python3
"""Generate tools/data/visual_day.json: a day where the *captions are vague* ("Desk in the office.") and only the
pixels say which frame answers the question: colours of objects, counts, objects the caption never names.
Each item carries a `scene` (shapes drawn by eval_agent.render_thumbs), so SigLIP image vectors and Claude's
own look at the snapshot are what can find / read the answer. Deterministic.

    python tools/make_visual_day.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "visual_day.json"
COLORS = {
    "red": [210, 40, 40],
    "blue": [40, 80, 220],
    "green": [40, 170, 70],
    "yellow": [240, 210, 30],
    "purple": [140, 60, 190],
    "orange": [245, 140, 30],
}
GREYS = [[120, 120, 120], [150, 150, 150], [95, 95, 95]]


def mug(c, x=360, y=240):
    return [
        {"shape": "rect", "xy": [x, y, x + 90, y + 110], "color": c},
        {"shape": "ring", "xy": [x + 80, y + 25, x + 130, y + 80], "color": c},
    ]


def umbrella(c, x=420, y=140):
    return [
        {"shape": "pie", "xy": [x, y, x + 260, y + 220], "color": c},
        {"shape": "rect", "xy": [x + 125, y + 110, x + 135, y + 300], "color": [60, 60, 60]},
    ]


def car(c, x=260, y=300):
    return [
        {"shape": "rect", "xy": [x, y, x + 420, y + 130], "color": c},
        {"shape": "rect", "xy": [x + 80, y - 70, x + 330, y], "color": c},
        {"shape": "circle", "xy": [x + 50, y + 100, x + 140, y + 190], "color": [30, 30, 30]},
        {"shape": "circle", "xy": [x + 280, y + 100, x + 370, y + 190], "color": [30, 30, 30]},
    ]


def apples(n, c=COLORS["red"]):
    return [{"shape": "circle", "xy": [180 + i * 110, 330, 270 + i * 110, 420], "color": c} for i in range(n)]


def balloons(n, c):
    out = []
    for i in range(n):
        x = 150 + i * 120
        out += [
            {"shape": "ellipse", "xy": [x, 120, x + 80, 220], "color": c},
            {"shape": "line", "xy": [x + 40, 220, x + 40, 400], "color": [70, 70, 70]},
        ]
    return out


def book(c):
    return [
        {"shape": "rect", "xy": [400, 300, 560, 340], "color": c},
        {"shape": "rect", "xy": [300, 340, 700, 360], "color": [110, 80, 50]},
    ]


def build(seed: int = 3) -> dict:
    rng = random.Random(seed)
    stream, n = [], 0

    def add(hhmm, caption, scene, bg, tag=None):
        nonlocal n
        n += 1
        tid = f"v{n:04d}"
        stream.append(
            {
                "id": tid,
                "time": hhmm,
                "caption": caption,
                "ocr": "",
                "entities": [],
                "episode": caption.split()[0].lower(),
                "scene": {"bg": bg, "shapes": scene},
            }
        )
        return tid

    places = [
        ("Desk in the office.", [205, 200, 190]),
        ("Kitchen table.", [215, 195, 165]),
        ("Parking garage.", [140, 140, 145]),
        ("Living room.", [190, 200, 210]),
        ("Street outside.", [170, 180, 190]),
        ("Bedroom nightstand.", [200, 190, 200]),
    ]
    gold = {}
    # targets (vague captions; the answer is only in the pixels)
    targets = [
        ("09:10", 0, mug(COLORS["green"]), "mug", "what color was the mug on my desk this morning", ["green"]),
        ("10:40", 1, apples(3), "apples", "how many apples were on the kitchen table", ["three", "3"]),
        (
            "12:05",
            4,
            umbrella(COLORS["yellow"]),
            "umbrella",
            "what color was the umbrella I saw on the street",
            ["yellow"],
        ),
        ("13:30", 2, car(COLORS["blue"]), "car", "what color was the car next to me in the parking garage", ["blue"]),
        (
            "15:00",
            3,
            balloons(4, COLORS["purple"]),
            "balloons",
            "how many balloons were in the living room",
            ["four", "4"],
        ),
        ("15:01", 3, [], None, None, None),
        ("16:20", 5, book(COLORS["orange"]), "book", "what color was the book on my nightstand", ["orange"]),
        (
            "17:45",
            1,
            apples(5, COLORS["green"]),
            "apples2",
            "how many green apples were on the table this evening",
            ["five", "5"],
        ),
        ("18:10", 0, mug(COLORS["red"]), "mug2", "what color was the mug on my desk this evening", ["red"]),
    ]
    for hhmm, pi, scene, tag, *_ in targets:
        if tag:
            gold[tag] = add(hhmm, places[pi][0], scene, places[pi][1])
    # distractors: same places, same vague captions, grey clutter
    for m in range(8 * 60, 19 * 60, 3):
        cap, bg = rng.choice(places)
        shapes = [
            {
                "shape": rng.choice(["rect", "circle"]),
                "xy": [
                    x := rng.randrange(60, 700),
                    y := rng.randrange(60, 480),
                    x + rng.randrange(40, 160),
                    y + rng.randrange(40, 160),
                ],
                "color": rng.choice(GREYS),
            }
            for _ in range(rng.randrange(1, 4))
        ]
        add(f"{m // 60:02d}:{m % 60:02d}", cap, shapes, bg)
    stream.sort(key=lambda e: e["time"])
    questions = [{"q": q, "gold": [gold[t]], "expect": ex} for _, _, _, t, q, ex in targets if t]
    questions.append({"q": "did I see a pink flamingo today", "gold": [], "expect": ["__not_found__"]})
    return {
        "date": "2026-11-08",
        "tz": "America/Los_Angeles",
        "now": "19:30",
        "device_id": "eval-device",
        "_comment": "vague captions; answers only in the pixels (tools/make_visual_day.py)",
        "stream": stream,
        "questions": questions,
    }


if __name__ == "__main__":
    d = build()
    OUT.write_text(json.dumps(d, indent=0))
    print(f"{OUT}: {len(d['stream'])} memories, {len(d['questions'])} questions")
