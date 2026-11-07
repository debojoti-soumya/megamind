#!/usr/bin/env python3
"""Retrieval-quality benchmark for memory tokens, with the real embedders (bge-small + SigLIP).

Stage 1 (retrieval): does the search return the exact token? Per question type and per signal
(BM25 / text vector / image vector / fused), R@1, R@5, MRR.
Stage 2 (reading, --reader local): send the retrieved frames to a VLM and check it extracts the
detail (object counts, where an object was), versus giving it the correct frame directly ("oracle").

Corpus (all in one store, so every question competes with every token):
- COCO val2017: 500 indoor photos with object boxes and 5 human captions each.
- Flickr30k 1k-test: 300 photos with 5 human captions each.
- ~100 synthetic sign/OCR and speech tokens with hard-negative lookalikes (same template,
  different entity), some with OCR noise, plus same-question-different-day tokens for time filters.
Two configs: "keyless" (photo tokens have no caption, image vector only) and "captioned" (caption 0
stored as if written by the VLM enricher; questions use a different caption).

    python tools/retrieval_bench.py --prepare --data /tmp/mmbench      # downloads ~330 MB once
    python tools/retrieval_bench.py --data /tmp/mmbench --reader local --out /tmp/mmbench/report.json
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import io
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from megamind.agent.reader import LocalVLMReader  # noqa: E402
from megamind.config import Settings  # noqa: E402
from megamind.ingest.enrich import LocalTextEmbedder, SigLIPImageEmbedder  # noqa: E402
from megamind.models import MemoryToken  # noqa: E402
from megamind.store.fusion import FusionWeights  # noqa: E402
from megamind.store.lancedb_store import LanceDBStore  # noqa: E402
from megamind.store.memory_store import InMemoryStore  # noqa: E402

DAY = 86_400.0
T0 = 1_760_000_000.0
FIND_OBJECTS = {
    "cell phone": "phone",
    "remote": "tv remote",
    "laptop": "laptop",
    "backpack": "backpack",
    "handbag": "handbag",
    "umbrella": "umbrella",
    "scissors": "scissors",
    "book": "book",
    "teddy bear": "teddy bear",
    "suitcase": "suitcase",
    "keyboard": "keyboard",
    "clock": "clock",
    "vase": "vase",
    "bottle": "water bottle",
    "cup": "coffee cup",
    "toothbrush": "toothbrush",
}
COUNT_OBJECTS = {
    "cup": "cups",
    "bottle": "bottles",
    "chair": "chairs",
    "bowl": "bowls",
    "wine glass": "wine glasses",
    "book": "books",
    "donut": "donuts",
    "banana": "bananas",
    "orange": "oranges",
    "laptop": "laptops",
}
SUPPORTS = {
    "dining table": ["table"],
    "couch": ["couch", "sofa"],
    "bed": ["bed"],
    "chair": ["chair"],
    "bench": ["bench"],
    "sink": ["sink"],
    "tv": ["tv", "television"],
    "refrigerator": ["fridge", "refrigerator"],
    "oven": ["oven", "stove"],
    "microwave": ["microwave"],
    "toilet": ["toilet"],
    "laptop": ["laptop"],
    "keyboard": ["keyboard", "desk"],
    "suitcase": ["suitcase"],
    "backpack": ["backpack"],
}
NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
SIGNALS = ("fused", "fused_nocal", "bm25", "text_vec", "image_vec")


# --------------------------------------------------------------------------- data
def prepare(data: Path) -> None:
    import io
    import urllib.request
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    coco, flickr = data / "coco", data / "flickr"
    (coco / "img").mkdir(parents=True, exist_ok=True)
    (flickr / "img").mkdir(parents=True, exist_ok=True)
    if not (coco / "subset.json").exists():
        print("downloading COCO annotations (250 MB)...")
        raw = urllib.request.urlopen("http://images.cocodataset.org/annotations/annotations_trainval2017.zip").read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        inst = json.loads(z.read("annotations/instances_val2017.json"))
        caps = json.loads(z.read("annotations/captions_val2017.json"))
        cats = {c["id"]: c["name"] for c in inst["categories"]}
        by, capby = collections.defaultdict(list), collections.defaultdict(list)
        for a in inst["annotations"]:
            by[a["image_id"]].append(a)
        for c in caps["annotations"]:
            capby[c["image_id"]].append(c["caption"].strip())
        imgs = {i["id"]: i for i in inst["images"]}
        indoor = set(FIND_OBJECTS) | set(COUNT_OBJECTS) | set(SUPPORTS)
        cand = [i for i, a in by.items() if len({cats[x["category_id"]] for x in a} & indoor) >= 2]
        random.Random(0).shuffle(cand)
        sel = sorted(cand[:500])
        out = []
        for i in sel:
            im = imgs[i]
            area = im["width"] * im["height"]
            objs = [
                {
                    "cat": cats[a["category_id"]],
                    "bbox": a["bbox"],
                    "area": a["area"] / area,
                    "crowd": a["iscrowd"],
                }
                for a in by[i]
            ]
            out.append(
                {
                    "file": im["file_name"],
                    "id": i,
                    "captions": capby[i],
                    "objects": objs,
                }
            )
        with ThreadPoolExecutor(24) as ex:
            list(
                ex.map(
                    lambda r: urllib.request.urlretrieve(
                        f"http://images.cocodataset.org/val2017/{r['file']}",
                        coco / "img" / r["file"],
                    ),
                    out,
                )
            )
        (coco / "subset.json").write_text(json.dumps(out))
    if not (flickr / "captions.json").exists():
        base = "https://datasets-server.huggingface.co/rows?dataset=nlphuji/flickr_1k_test_image_text_retrieval"
        rows = []
        for off in (0, 100, 200):
            rows += [
                r["row"]
                for r in json.load(urllib.request.urlopen(f"{base}&config=TEST&split=test&offset={off}&length=100"))[
                    "rows"
                ]
            ]
        with ThreadPoolExecutor(16) as ex:
            list(
                ex.map(
                    lambda ir: urllib.request.urlretrieve(ir[1]["image"]["src"], flickr / "img" / f"{ir[0]:04d}.jpg"),
                    enumerate(rows),
                )
            )
        (flickr / "captions.json").write_text(
            json.dumps([{"file": f"{i:04d}.jpg", "captions": r["caption"]} for i, r in enumerate(rows)])
        )
    print("data ready in", data)


def _ocr_noise(text: str, rnd: random.Random, rate: float = 0.12) -> str:
    sub = {
        "o": "0",
        "O": "0",
        "l": "1",
        "I": "1",
        "e": "c",
        "S": "5",
        "s": "5",
        "B": "8",
        "a": "o",
        "m": "rn",
    }
    out = []
    for ch in text:
        r = rnd.random()
        if r < rate / 3:
            continue
        out.append(sub.get(ch, ch) if r < rate else ch)
    return "".join(out) + rnd.choice(["", " ~", " |", " ee"])


def make_facts(rnd: random.Random) -> list[dict]:
    """Sign and speech tokens. Each template has several instances that differ only in the entity."""
    words = [
        "falcon",
        "maple",
        "cobalt",
        "orchid",
        "summit",
        "harbor",
        "willow",
        "quartz",
        "ember",
        "lotus",
    ]
    facts: list[dict] = []

    def add(kind, text, qs, ts=None, noisy=False):
        stored = _ocr_noise(text, rnd) if noisy else text
        if noisy:
            qs = [("ocr_noisy" if c == "ocr_exact" else c, q, w) for c, q, w in qs]
        facts.append({"kind": kind, "text": stored, "clean": text, "ts": ts, "queries": qs})

    for i, p in enumerate(
        [
            "cafe",
            "hotel",
            "library",
            "airport lounge",
            "coworking space",
            "gym",
            "office",
            "hostel",
        ]
    ):
        pw = f"{words[i]}-{rnd.randint(1000, 9999)}"
        add(
            "ocr",
            f"WIFI {p.upper()} network password: {pw}",
            [
                ("ocr_exact", f"wifi password at the {p}", None),
                ("ocr_paraphrase", f"internet login for the {p}", None),
            ],
            noisy=i % 2 == 1,
        )
    items = [
        "oat milk",
        "almond butter",
        "sourdough bread",
        "greek yogurt",
        "coffee beans",
        "avocados",
        "olive oil",
        "free range eggs",
        "dark chocolate",
        "basmati rice",
    ]
    for i, it in enumerate(items):
        add(
            "ocr",
            f"{it.upper()} ${rnd.randint(1, 19)}.{rnd.randint(10, 99)} PER UNIT",
            [
                ("ocr_exact", f"how much did the {it} cost", None),
                ("ocr_paraphrase", f"what did I pay for {it}", None),
            ],
            noisy=i % 2 == 1,
        )
    cities = [
        "denver",
        "boston",
        "seattle",
        "chicago",
        "austin",
        "miami",
        "portland",
        "atlanta",
    ]
    for i, c in enumerate(cities):
        add(
            "ocr",
            f"FLIGHT UA{rnd.randint(100, 999)} TO {c.upper()} GATE {rnd.choice('ABCDE')}{rnd.randint(1, 40)} "
            f"BOARDING {rnd.randint(6, 22)}:{rnd.choice(['05', '20', '35', '50'])}",
            [
                ("ocr_exact", f"what gate was my flight to {c}", None),
                ("ocr_paraphrase", f"where did I board the plane to {c}", None),
            ],
            noisy=i % 2 == 1,
        )
    for n in ["patel", "nguyen", "garcia", "kim", "okafor", "rossi"]:
        add(
            "ocr",
            f"DR. {n.upper()} - CARDIOLOGY - ROOM {rnd.randint(100, 499)} FLOOR {rnd.randint(1, 6)}",
            [
                ("ocr_exact", f"which room is dr {n} in", None),
                ("ocr_paraphrase", f"where is dr {n}'s office", None),
            ],
        )
    topics = {
        "demo": (
            "presentation",
            [
                "is on friday at 5pm",
                "moved to monday morning",
                "got pushed to next week",
            ],
        ),
        "budget": (
            "spending limit",
            ["is capped at 40k", "was approved by finance", "needs another review"],
        ),
        "offsite": (
            "team retreat",
            ["is in lake tahoe", "moved to june", "was cancelled this year"],
        ),
    }
    for person in ["sam", "priya", "alex", "jordan", "mei", "omar"]:
        for topic, (syn, values) in topics.items():
            add(
                "speech",
                f"{person} said the {topic} {rnd.choice(values)}",
                [
                    ("speech", f"what did {person} say about the {topic}", None),
                    (
                        "speech_paraphrase",
                        f"what did {person} tell me about the {syn}",
                        None,
                    ),
                ],
            )
    for d in range(8):
        ts = T0 + d * DAY + 12 * 3600
        add(
            "ocr",
            f"PARKING LEVEL {rnd.randint(1, 6)} SPOT {rnd.choice('ABCDEFG')}{rnd.randint(1, 60)}",
            [
                (
                    "time_window",
                    "where did I park the car",
                    (T0 + d * DAY, T0 + (d + 1) * DAY),
                )
            ],
            ts=ts,
        )
    return facts


def count_questions(coco: list[dict], rnd: random.Random) -> list[dict]:
    qs = []
    for r in coco:
        for cat, plural in COUNT_OBJECTS.items():
            inst = [o for o in r["objects"] if o["cat"] == cat]
            if 2 <= len(inst) <= 5 and all(o["area"] > 0.004 and not o["crowd"] for o in inst):
                qs.append(
                    {
                        "img": r["file"],
                        "cat": cat,
                        "n": len(inst),
                        "q": f"how many {plural} were there when I saw {r['captions'][1].rstrip('.').lower()}",
                    }
                )
    rnd.shuffle(qs)
    return qs


def find_questions(coco: list[dict]) -> list[dict]:
    qs = []
    for cat, name in FIND_OBJECTS.items():
        valid = {r["file"] for r in coco if any(o["cat"] == cat and o["area"] > 0.003 for o in r["objects"])}
        if len(valid) >= 2:
            qs.append(
                {
                    "cat": cat,
                    "q": f"where did I leave my {name}",
                    "valid": valid,
                    "base_rate": len(valid) / len(coco),
                }
            )
    return qs


def supports_of(r: dict, cat: str) -> set[str]:
    """Words naming annotated furniture/objects under or around the target object (for grading 'where')."""
    out = set()
    for t in (o for o in r["objects"] if o["cat"] == cat):
        cx, cy = t["bbox"][0] + t["bbox"][2] / 2, t["bbox"][1] + t["bbox"][3] / 2
        for o in r["objects"]:
            x, y, w, h = o["bbox"]
            if (
                o["cat"] in SUPPORTS
                and o["cat"] != cat
                and x - 0.1 * w <= cx <= x + 1.1 * w
                and y - 0.3 * h <= cy <= y + 1.1 * h
            ):
                out.update(SUPPORTS[o["cat"]])
    return out


def render_fact(f: dict, photo: Path, rnd: random.Random) -> bytes:
    """A frame for a fact token: the sign's text on a card inside a scene, or (speech) a shifted,
    mirrored crop of a scene, i.e. what the glasses saw while someone was talking."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    img = Image.open(photo).convert("RGB")
    w, h = img.size
    img = ImageOps.mirror(img.crop((int(w * 0.1), int(h * 0.1), w, h))).resize((960, 540))
    if f["kind"] == "ocr":
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=30)
        lines, cur = [], ""
        for word in f["clean"].split():
            cur = f"{cur} {word}".strip()
            if len(cur) > 22:
                lines.append(cur)
                cur = ""
        lines.append(cur)
        x0, y0 = rnd.randint(40, 300), rnd.randint(40, 200)
        box = [x0, y0, x0 + 440, y0 + 50 * len(lines) + 30]
        d.rectangle(box, fill=(245, 245, 235), outline=(30, 30, 30), width=4)
        for k, line in enumerate(lines):
            d.text((x0 + 20, y0 + 20 + 50 * k), line, fill=(10, 10, 10), font=font)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def says_number(text: str, n: int) -> bool:
    text = re.sub(r"^\[[^\]]*\]\s*", "", text.strip())  # drop the "[token id]" prefix
    found = [int(x) for x in re.findall(r"\b\d+\b", text)] + [
        v for k, v in NUM.items() if re.search(rf"\b{k}\b", text.lower())
    ]
    return bool(found) and found[0] == n


# --------------------------------------------------------------------------- bench
class Bench:
    def __init__(self, data: Path, cache: Path, weights: FusionWeights | None = None) -> None:
        s = Settings(_env_file=None)
        self.text = LocalTextEmbedder(s)
        self.image = SigLIPImageEmbedder(s)
        self.data, self.cache = data, cache
        self.coco = json.loads((data / "coco/subset.json").read_text())
        self.flickr = json.loads((data / "flickr/captions.json").read_text())
        self._qcache: dict[tuple[str, str], list[float]] = {}
        scale, bias = self.image.calibration or (None, 0.0)
        base = weights or FusionWeights()
        self.weights = dataclasses.replace(base, image_logit_scale=scale, image_logit_bias=bias)
        self.nocal = dataclasses.replace(base, image_logit_scale=None)

    async def image_vecs(self, paths: list[Path]) -> dict[str, list[float]]:
        import numpy as np

        f = self.cache / "siglip_images.npz"
        have = dict(np.load(f)) if f.exists() else {}
        todo = [p for p in paths if p.name not in have]
        for i, p in enumerate(todo):
            have[p.name] = np.asarray(await self.image.embed_image(p.read_bytes()), dtype=np.float32)
            if i % 100 == 99:
                print(f"  siglip {i + 1}/{len(todo)}", flush=True)
        if todo:
            np.savez(f, **have)
        return {p.name: have[p.name].tolist() for p in paths}

    async def q(self, kind: str, text: str) -> list[float]:
        key = (kind, text)
        if key not in self._qcache:
            self._qcache[key] = await (self.text.embed_query(text) if kind == "t" else self.image.embed_text(text))
        return self._qcache[key]

    async def build(self, captioned: bool, rnd: random.Random) -> tuple[list[MemoryToken], list[dict]]:
        cpaths = [self.data / "coco/img" / r["file"] for r in self.coco]
        fpaths = [self.data / "flickr/img" / r["file"] for r in self.flickr]
        self.counts = count_questions(self.coco, random.Random(2))[:40]
        facts = make_facts(random.Random(1))
        busy = {c["img"] for c in self.counts}
        spare = [p for p, r in zip(cpaths, self.coco, strict=True) if r["file"] not in busy]
        fdir = self.cache / "facts"
        fdir.mkdir(exist_ok=True)
        frnd = random.Random(4)
        fact_paths = []
        for i, f in enumerate(facts):
            fp = fdir / f"fact_{i:03d}.jpg"
            if not fp.exists():
                fp.write_bytes(render_fact(f, frnd.choice(spare), frnd))
            fact_paths.append(fp)
        ivec = await self.image_vecs(cpaths + fpaths + fact_paths)
        toks: list[MemoryToken] = []

        async def tok(tid, path, caption="", ocr="", transcript="", ts=None, image=True):
            t = MemoryToken(
                id=tid,
                device_id="glasses",
                ts=ts or T0 + rnd.random() * 10 * DAY,
                thumb_path=str(path),
                frame_hash="0" * 16,
                caption=caption,
                ocr_text=ocr,
                transcript=transcript,
                image_vec=ivec[Path(path).name] if image else None,
            )
            st = t.search_text()
            t.text_vec = await self.text.embed(st) if st else None
            toks.append(t)

        for r, p in zip(self.coco, cpaths, strict=True):
            await tok(f"coco:{r['file']}", p, caption=r["captions"][0] if captioned else "")
        for r, p in zip(self.flickr, fpaths, strict=True):
            await tok(f"flickr:{r['file']}", p, caption=r["captions"][0] if captioned else "")
        qs: list[dict] = []
        for i, (f, fp) in enumerate(zip(facts, fact_paths, strict=True)):
            tid = f"fact:{i}"
            await tok(
                tid,
                fp,
                ocr=f["text"] if f["kind"] == "ocr" else "",
                transcript=f["text"] if f["kind"] == "speech" else "",
                ts=f["ts"],
            )
            qs += [{"cat": c, "q": q, "gold": {tid}, "window": w} for c, q, w in f["queries"]]
        for r in self.flickr:
            qs.append(
                {
                    "cat": "visual_scene",
                    "q": r["captions"][2],
                    "gold": {f"flickr:{r['file']}"},
                    "window": None,
                }
            )
        for c in self.counts:
            qs.append(
                {
                    "cat": "count_target",
                    "q": c["q"],
                    "gold": {f"coco:{c['img']}"},
                    "window": None,
                }
            )
        self.finds = find_questions(self.coco)
        for f in self.finds:
            qs.append(
                {
                    "cat": "object_find",
                    "q": f["q"],
                    "gold": {f"coco:{x}" for x in f["valid"]},
                    "window": None,
                    "base_rate": f["base_rate"],
                }
            )
        return toks, qs

    async def search(self, store, item: dict, signal: str, k: int = 5):
        q = item["q"]
        w = item["window"] or (None, None)
        kw = {"k": k, "time_from": w[0], "time_to": w[1], "per_episode_cap": 0}
        if signal == "bm25":
            return await store.search(None, None, q, mode="semantic", **kw)
        if signal == "text_vec":
            return await store.search(await self.q("t", q), None, "", mode="semantic", **kw)
        if signal == "image_vec":
            return await store.search(None, await self.q("i", q), "", mode="visual", **kw)
        if signal == "fused_nocal":
            store.weights, keep = self.nocal, store.weights
            try:
                return await store.search(await self.q("t", q), await self.q("i", q), q, mode="hybrid", **kw)
            finally:
                store.weights = keep
        return await store.search(await self.q("t", q), await self.q("i", q), q, mode="hybrid", **kw)


def score(ranked: list[str], gold: set[str]) -> tuple[int, int, float]:
    for i, tid in enumerate(ranked):
        if tid in gold:
            return int(i == 0), int(i < 5), 1.0 / (i + 1)
    return 0, 0, 0.0


async def run(data: Path, reader: str | None, out: Path | None, limit_reader: int, image_w: float | None) -> dict:
    cache = data / "cache"
    cache.mkdir(exist_ok=True)
    b = Bench(data, cache, FusionWeights(image=image_w) if image_w is not None else None)
    report: dict = {"retrieval": {}, "reader": {}, "image_weight": b.weights.image}
    for config in ("keyless", "captioned"):
        toks, qs = await b.build(config == "captioned", random.Random(3))
        with tempfile.TemporaryDirectory() as td:
            lance = LanceDBStore(Path(td), b.text.dim, b.image.dim, b.weights)
            await lance.add_many(toks)
            mem = InMemoryStore(b.weights)
            await mem.add_many(toks)
            res: dict = collections.defaultdict(lambda: collections.defaultdict(list))
            lat = []
            for item in qs:
                for sig in SIGNALS:
                    t = time.perf_counter()
                    hits = await b.search(lance, item, sig)
                    if sig == "fused":
                        lat.append(time.perf_counter() - t)
                        item["top"] = [h.id for h in hits]
                    res[item["cat"]][sig].append(score([h.id for h in hits], item["gold"]))
                hits32 = await b.search(mem, item, "fused")
                res[item["cat"]]["fused_fp32"].append(score([h.id for h in hits32], item["gold"]))
            table = {}
            for cat, sigs in res.items():
                table[cat] = {
                    sig: {
                        "n": len(v),
                        "r1": sum(x[0] for x in v) / len(v),
                        "r5": sum(x[1] for x in v) / len(v),
                        "mrr": sum(x[2] for x in v) / len(v),
                    }
                    for sig, v in sigs.items()
                }
                if cat == "object_find":
                    table[cat]["random_r1"] = sum(i["base_rate"] for i in qs if i["cat"] == cat) / len(sigs["fused"])
            report["retrieval"][config] = table
            report["latency_ms_p50"] = sorted(lat)[len(lat) // 2] * 1000
            report["corpus_tokens"] = len(toks)
            if reader and config == "captioned":
                report["reader"] = await read_stage(b, lance, toks, qs, limit_reader)
            if reader and config == "keyless":
                report["reader_keyless"] = await read_stage(b, lance, toks, qs, limit_reader)
    print_report(report)
    if out:
        out.write_text(json.dumps(report, indent=1))
    return report


async def read_stage(b: Bench, store, toks: list[MemoryToken], qs: list[dict], limit: int) -> dict:
    class Media:
        async def read(self, t):
            return Path(t.thumb_path).read_bytes() if t.thumb_path else None

    s = Settings(_env_file=None, reader_max_frames=3)
    if not hasattr(b, "reader"):
        b.reader = LocalVLMReader(s, Media())
    by_id = {t.id: t for t in toks}
    coco = {r["file"]: r for r in b.coco}
    out = {
        "count": {"oracle": 0, "retrieved": 0, "n": 0},
        "where": {"found_object": 0, "named_place": 0, "n": 0},
    }
    for c in b.counts[:limit]:
        gold = by_id[f"coco:{c['img']}"]
        o = await b.reader.read(c["q"], [gold])
        top = [by_id[i] for i in next(x for x in qs if x["q"] == c["q"])["top"][:3]]
        r = await b.reader.read(c["q"], top)
        first = r.evidence.splitlines()[0] if r.evidence else ""
        out["count"]["n"] += 1
        out["count"]["oracle"] += says_number(o.evidence, c["n"])
        out["count"]["retrieved"] += says_number(first.split("] ", 1)[-1], c["n"]) and top[0].id == gold.id
    for f in b.finds:
        top = [by_id[i] for i in next(x for x in qs if x["q"] == f["q"])["top"][:3]]
        r = await b.reader.read(f["q"], top)
        out["where"]["n"] += 1
        for line in r.evidence.splitlines():
            tid, _, ans = line[1:].partition("] ")
            img = tid.removeprefix("coco:")
            if img in f["valid"]:
                out["where"]["found_object"] += 1
                sup = supports_of(coco[img], f["cat"])
                out["where"]["named_place"] += any(re.search(rf"\b{w}\b", ans.lower()) for w in sup)
                break
    return out


def print_report(r: dict) -> None:
    for config, table in r["retrieval"].items():
        print(f"\n== {config}  ({r['corpus_tokens']} tokens, image weight {r['image_weight']})   R@1 / R@5 / MRR")
        print(f"{'category':20}" + "".join(f"{s:>20}" for s in (*SIGNALS, "fused_fp32")))
        for cat, sigs in table.items():
            row = "".join(
                f"{sigs[s]['r1']:6.2f}{sigs[s]['r5']:6.2f}{sigs[s]['mrr']:6.2f}  " for s in (*SIGNALS, "fused_fp32")
            )
            extra = f"  random R@1 {sigs['random_r1']:.2f}" if "random_r1" in sigs else ""
            print(f"{cat:16} n={sigs['fused']['n']:3d} {row}{extra}")
    print(f"\nfused search p50 latency {r['latency_ms_p50']:.1f} ms")
    for k in ("reader_keyless", "reader"):
        if r.get(k):
            print(k, json.dumps(r[k]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/tmp/mmbench"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--reader", choices=["local"], default=None)
    ap.add_argument("--reader-limit", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--image-weight", type=float, default=None, help="hybrid image weight (default: server's)")
    a = ap.parse_args()
    if a.prepare:
        prepare(a.data)
        return
    asyncio.run(run(a.data, a.reader, a.out, a.reader_limit, a.image_weight))


if __name__ == "__main__":
    main()
