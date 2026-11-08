#!/usr/bin/env python3
"""Generate tools/data/coco_day.json: real photos (COCO val2017, CC BY 4.0) with *vague* captions, so only the
pixels answer the question: counts, which animal / vehicle / food. Images are downloaded to ~/.cache/coco/img
(never committed); eval_agent copies an item's `image` into its thumbnail.

    python tools/make_coco_day.py --ann ~/.cache/coco/annotations/instances_val2017.json
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "coco_day.json"
IMG = Path.home() / ".cache" / "coco" / "img"
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
NOT_FOUND = "fire hydrant"  # absent from every picked photo (checked against the annotations)
NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, default=Path.home() / ".cache/coco/annotations/instances_val2017.json")
    ap.add_argument("--distractors", type=int, default=160)
    a = ap.parse_args()
    d = json.loads(a.ann.read_text())
    cat = {c["id"]: c["name"] for c in d["categories"]}
    per: dict[int, Counter] = defaultdict(Counter)
    for an in d["annotations"]:
        if not an.get("iscrowd") and an["area"] > 1500:
            per[an["image_id"]][cat[an["category_id"]]] += 1
    imgs = {i["id"]: i for i in d["images"]}
    rng = random.Random(11)
    ids = sorted(per)
    rng.shuffle(ids)

    def pick(pred):
        for i in ids:
            if i not in used and pred(per[i]):
                used.add(i)
                return i
        raise SystemExit("no image")

    used: set[int] = set()
    T = []  # (time, caption, image_id, question, expect)
    i = pick(lambda c: c["dining table"] and 3 <= c["person"] <= 5 and len(c) <= 6)
    T.append(
        (
            "12:30",
            "Lunch at a restaurant.",
            i,
            "how many people were at the table at lunch",
            [NUM[per[i]["person"]], str(per[i]["person"])],
        )
    )
    i = pick(lambda c: 2 <= c["banana"] <= 5 and c["person"] == 0)
    T.append(
        (
            "08:10",
            "Kitchen counter.",
            i,
            "how many bananas were there this morning",
            [NUM[per[i]["banana"]], str(per[i]["banana"])],
        )
    )
    for t, name, q in [
        ("16:40", "dog", "what animal did I see at the park"),
        ("17:20", "horse", "what animal did I see on the trail"),
    ]:
        i = pick(lambda c, n=name: c[n] >= 1 and not (ANIMALS - {n}) & set(c))
        T.append((t, "Outdoors, walking.", i, q, [name]))
    i = pick(lambda c: c["bus"] >= 1 and c["truck"] == 0)
    T.append(("08:45", "Street outside.", i, "what vehicle was at the stop this morning", ["bus"]))
    i = pick(lambda c: c["pizza"] >= 1 and c["person"] <= 1)
    T.append(("19:30", "Dinner.", i, "what did I have for dinner", ["pizza"]))
    i = pick(lambda c: c["umbrella"] >= 2)
    T.append(("14:15", "Street outside.", i, "what were people holding on the street this afternoon", ["umbrella"]))
    i = pick(lambda c: 2 <= c["cup"] <= 4 and c["person"] == 0)
    T.append(
        (
            "10:05",
            "Office kitchen.",
            i,
            "how many cups were on the counter in the office kitchen",
            [NUM[per[i]["cup"]], str(per[i]["cup"])],
        )
    )
    i = pick(lambda c: c["laptop"] >= 1 and c["cat"] >= 1)
    T.append(("21:10", "Home office.", i, "what animal was next to my laptop tonight", ["cat"]))
    places = [
        "Kitchen counter.",
        "Street outside.",
        "Outdoors, walking.",
        "Office kitchen.",
        "Home office.",
        "Dinner.",
        "Lunch at a restaurant.",
    ]
    # distractors must not hold another answer: no animal at all (else "what animal" is ambiguous), no cups /
    # buses / pizza / umbrellas / bananas, and nothing the not-found question asks about
    avoid = ANIMALS | {"bus", "pizza", "umbrella", "banana", "cup", NOT_FOUND}
    stream, n = [], 0
    IMG.mkdir(parents=True, exist_ok=True)

    def add(hhmm, cap, iid):
        nonlocal n
        n += 1
        p = IMG / imgs[iid]["file_name"]
        if not p.exists():
            urllib.request.urlretrieve(imgs[iid]["coco_url"], p)
        stream.append(
            {
                "id": f"c{n:04d}",
                "time": hhmm,
                "caption": cap,
                "ocr": "",
                "entities": [],
                "episode": cap.split()[0].lower(),
                "image": str(p),
            }
        )
        return f"c{n:04d}"

    qs = []
    for t, cap, iid, q, ex in T:
        qs.append({"q": q, "gold": [add(t, cap, iid)], "expect": ex})
    m = 7 * 60
    for _ in range(a.distractors):
        iid = pick(lambda c: not (avoid & set(c)))
        m += rng.randrange(2, 6)
        add(f"{min(m, 22 * 60) // 60:02d}:{min(m, 22 * 60) % 60:02d}", rng.choice(places), iid)
    stream.sort(key=lambda e: e["time"])
    qs.append({"q": f"did I see a {NOT_FOUND} today", "gold": [], "expect": ["__not_found__"]})
    OUT.write_text(
        json.dumps(
            {
                "date": "2026-11-08",
                "tz": "America/Los_Angeles",
                "now": "22:30",
                "device_id": "eval-device",
                "_comment": "COCO val2017 photos (CC BY 4.0), vague captions; tools/make_coco_day.py",
                "stream": stream,
                "questions": qs,
            },
            indent=0,
        )
    )
    print(f"{OUT}: {len(stream)} memories, {len(qs)} questions")


if __name__ == "__main__":
    main()
