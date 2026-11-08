#!/usr/bin/env python3
"""Retrieval-quality benchmark on real photos, through main's real retrieval path (ported from PR #54).

Stage 1 (retrieval): does the search put the exact token first? Per question category and per signal, R@1, R@5
and MRR (over the top 10):
  - `bm25`, `text_vec`, `image_vec`: the store primitives on their own (what each signal can do alone);
  - `hybrid`: `HybridRetriever.retrieve` (planner types/weights, weighted RRF, recency prior, dedupe) = the
    `recall_memory` path; `hybrid_nocal` is the same with the image embedder's calibration hidden (PR 1 toggle);
  - `hybrid_fp16`: `hybrid` over a float16-vector LanceDB table (when the store supports `vector_dtype`, PR 1);
  - `store_search`: `MemoryStore.search` (the `/v1/memories` path) for comparison.
Stage 2 (reading, `--reader vlm`, needs READER_API_KEY): main's `VLMReader` on the retrieved frames vs on the
right frame alone ("oracle"), for object counts and "where did I leave X".

Corpus (one store, so every question competes with every token):
  - COCO val2017: 500 indoor photos with object boxes and 5 human captions each;
  - Flickr30k 1k-test: 300 photos with 5 human captions each;
  - ~100 synthetic sign (OCR) tokens and speech (`kind="transcript"`) tokens with hard-negative lookalikes (same
    template, different entity), half the signs with OCR noise, plus same-question-different-day parking signs
    for time filters.
Two configs:
  - `keyless`: what main stores with no enricher today (`MockEnricher` caption); signs keep their OCR text;
  - `captioned`: COCO/Flickr caption 0 stored as if written by the VLM enricher; questions use another caption.

    python tools/retrieval_bench.py --prepare                      # downloads ~330 MB once (never committed)
    python tools/retrieval_bench.py --out bench.json               # needs .[local-embeddings] (+ CPU torch)
    python tools/retrieval_bench.py --text mock --image mock --limit 40   # plumbing smoke run, no torch

Numbers are not comparable to the ones posted on PR #54 (different filter, enricher, fusion and weights): the
JSON report records the full configuration so runs can be compared with each other.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import hashlib
import io
import json
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from megamind.ingest.audio import CAPTION_PREFIX  # noqa: E402
from megamind.ingest.enrich import ImageEmbedder, MockEnricher, TextEmbedder  # noqa: E402
from megamind.models import MemoryToken  # noqa: E402
from megamind.retrieval.adapter import StoreAdapter  # noqa: E402
from megamind.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from megamind.retrieval.planner import WEIGHTS_BY_TYPE, Weights, plan_query  # noqa: E402

DEFAULT_DATA = Path.home() / ".cache" / "megamind" / "bench"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{file}"
FLICKR_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=nlphuji/flickr_1k_test_image_text_retrieval"
    "&config=TEST&split=test&offset={offset}&length={length}"
)
DAY = 86_400.0
T0 = 1_760_000_000.0
NOW = T0 + 11 * DAY  # every token is older than this; fixed so the recency prior is deterministic
K = 10  # ranked depth scored (R@1, R@5, MRR@10)
DEVICE = "glasses"
SIGN_CAPTION = "A printed sign on a card held up in front of a scene."
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
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


# --------------------------------------------------------------------------- dataset parsing (pure, CI-tested)


def coco_subset(instances: dict, captions: dict, n: int = 500, seed: int = 0) -> list[dict]:
    """COCO `instances_*.json` + `captions_*.json` -> up to `n` photos showing >= 2 indoor object classes,
    as [{file, id, width, height, captions, objects: [{cat, bbox, area (fraction of image), crowd}]}], sorted
    by image id. Deterministic for a given `seed`."""
    cats = {c["id"]: c["name"] for c in instances["categories"]}
    by: dict[int, list[dict]] = collections.defaultdict(list)
    capby: dict[int, list[str]] = collections.defaultdict(list)
    for a in instances["annotations"]:
        by[a["image_id"]].append(a)
    for c in captions["annotations"]:
        capby[c["image_id"]].append(c["caption"].strip())
    imgs = {i["id"]: i for i in instances["images"]}
    indoor = set(FIND_OBJECTS) | set(COUNT_OBJECTS) | set(SUPPORTS)
    cand = sorted(i for i, a in by.items() if i in imgs and len({cats[x["category_id"]] for x in a} & indoor) >= 2)
    random.Random(seed).shuffle(cand)
    out = []
    for i in sorted(cand[:n]):
        im = imgs[i]
        area = float(im["width"] * im["height"]) or 1.0
        objs = [
            {"cat": cats[a["category_id"]], "bbox": a["bbox"], "area": a["area"] / area, "crowd": a.get("iscrowd", 0)}
            for a in by[i]
        ]
        out.append({"file": im["file_name"], "id": i, "captions": capby[i], "objects": objs})
    return out


def flickr_rows(pages: list[dict]) -> list[dict]:
    """HF datasets-server `/rows` pages of nlphuji/flickr_1k_test_image_text_retrieval -> [{file, captions, src}].
    Files are numbered in row order (0000.jpg, ...)."""
    rows = [r["row"] for page in pages for r in page["rows"]]
    out = []
    for i, r in enumerate(rows):
        caps = r["caption"] if isinstance(r["caption"], list) else [r["caption"]]
        out.append({"file": f"{i:04d}.jpg", "captions": [c.strip() for c in caps], "src": r["image"]["src"]})
    return out


def prepare(data: Path, n_coco: int = 500, n_flickr: int = 300) -> None:
    """Download COCO val2017 annotations + `n_coco` photos and `n_flickr` Flickr 1k-test photos into `data`."""
    import urllib.request
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    coco, flickr = data / "coco", data / "flickr"
    (coco / "img").mkdir(parents=True, exist_ok=True)
    (flickr / "img").mkdir(parents=True, exist_ok=True)
    if not (coco / "subset.json").exists():
        print("downloading COCO annotations (~250 MB)...", flush=True)
        with urllib.request.urlopen(COCO_ANN_URL) as resp:  # noqa: S310 (fixed https/http dataset URL)
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
        sub = coco_subset(
            json.loads(z.read("annotations/instances_val2017.json")),
            json.loads(z.read("annotations/captions_val2017.json")),
            n_coco,
        )
        with ThreadPoolExecutor(16) as ex:
            list(
                ex.map(
                    lambda r: urllib.request.urlretrieve(COCO_IMG_URL.format(file=r["file"]), coco / "img" / r["file"]),
                    sub,
                )
            )
        (coco / "subset.json").write_text(json.dumps(sub))
    if not (flickr / "captions.json").exists():
        print("downloading Flickr 1k-test rows...", flush=True)
        pages = []
        for off in range(0, n_flickr, 100):
            url = FLICKR_ROWS_URL.format(offset=off, length=min(100, n_flickr - off))
            with urllib.request.urlopen(url) as resp:  # noqa: S310
                pages.append(json.load(resp))
        rows = flickr_rows(pages)
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda r: urllib.request.urlretrieve(r["src"], flickr / "img" / r["file"]), rows))
        (flickr / "captions.json").write_text(
            json.dumps([{"file": r["file"], "captions": r["captions"]} for r in rows])
        )
    print("data ready in", data)


# --------------------------------------------------------------------------- synthetic facts (pure, CI-tested)


def _ocr_noise(text: str, rnd: random.Random, rate: float = 0.12) -> str:
    sub = {"o": "0", "O": "0", "l": "1", "I": "1", "e": "c", "S": "5", "s": "5", "B": "8", "a": "o", "m": "rn"}
    out = []
    for ch in text:
        r = rnd.random()
        if r < rate / 3:
            continue
        out.append(sub.get(ch, ch) if r < rate else ch)
    return "".join(out) + rnd.choice(["", " ~", " |", " ee"])


def make_facts(rnd: random.Random) -> list[dict]:
    """Sign (`ocr`) and `speech` facts. Each template has several instances that differ only in the entity, so
    every question has hard negatives. Returns [{kind, text (as stored), clean, ts|None, queries: [(cat, q,
    window|None)]}]; deterministic for a seeded `rnd`."""
    words = ["falcon", "maple", "cobalt", "orchid", "summit", "harbor", "willow", "quartz", "ember", "lotus"]
    facts: list[dict] = []

    def add(kind, text, qs, ts=None, noisy=False):
        stored = _ocr_noise(text, rnd) if noisy else text
        if noisy:
            qs = [("ocr_noisy" if c == "ocr_exact" else c, q, w) for c, q, w in qs]
        facts.append({"kind": kind, "text": stored, "clean": text, "ts": ts, "queries": qs})

    places = ["cafe", "hotel", "library", "airport lounge", "coworking space", "gym", "office", "hostel"]
    for i, p in enumerate(places):
        pw = f"{words[i]}-{rnd.randint(1000, 9999)}"
        add(
            "ocr",
            f"WIFI {p.upper()} network password: {pw}",
            [("ocr_exact", f"wifi password at the {p}", None), ("ocr_paraphrase", f"internet login for the {p}", None)],
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
            [("ocr_exact", f"how much did the {it} cost", None), ("ocr_paraphrase", f"what did I pay for {it}", None)],
            noisy=i % 2 == 1,
        )
    cities = ["denver", "boston", "seattle", "chicago", "austin", "miami", "portland", "atlanta"]
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
            [("ocr_exact", f"which room is dr {n} in", None), ("ocr_paraphrase", f"where is dr {n}'s office", None)],
        )
    topics = {
        "demo": ("presentation", ["is on friday at 5pm", "moved to monday morning", "got pushed to next week"]),
        "budget": ("spending limit", ["is capped at 40k", "was approved by finance", "needs another review"]),
        "offsite": ("team retreat", ["is in lake tahoe", "moved to june", "was cancelled this year"]),
    }
    for person in ["sam", "priya", "alex", "jordan", "mei", "omar"]:
        for topic, (syn, values) in topics.items():
            add(
                "speech",
                f"{person} said the {topic} {rnd.choice(values)}",
                [
                    ("speech", f"what did {person} say about the {topic}", None),
                    ("speech_paraphrase", f"what did {person} tell me about the {syn}", None),
                ],
            )
    for d in range(8):
        add(
            "ocr",
            f"PARKING LEVEL {rnd.randint(1, 6)} SPOT {rnd.choice('ABCDEFG')}{rnd.randint(1, 60)}",
            [("time_window", "where did I park the car", (T0 + d * DAY, T0 + (d + 1) * DAY))],
            ts=T0 + d * DAY + 12 * 3600,
        )
    return facts


def count_questions(coco: list[dict], rnd: random.Random) -> list[dict]:
    qs = []
    for r in coco:
        if len(r["captions"]) < 2:
            continue
        for cat, plural in COUNT_OBJECTS.items():
            inst = [o for o in r["objects"] if o["cat"] == cat]
            if 2 <= len(inst) <= 5 and all(o["area"] > 0.004 and not o["crowd"] for o in inst):
                desc = r["captions"][1].rstrip(".").lower()
                qs.append(
                    {
                        "img": r["file"],
                        "cat": cat,
                        "n": len(inst),
                        "q": f"how many {plural} were there when I saw {desc}",
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
                {"cat": cat, "q": f"where did I leave my {name}", "valid": valid, "base_rate": len(valid) / len(coco)}
            )
    return qs


def supports_of(r: dict, cat: str) -> set[str]:
    """Words naming annotated furniture/objects under or around the target object (for grading 'where')."""
    out: set[str] = set()
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


def says_number(text: str, n: int) -> bool:
    """The first number (digits or a word up to ten) in `text` equals `n`."""
    text = re.sub(r"^\[[^\]]*\]\s*", "", text.strip())  # drop a leading "[token id]"
    found = [(m.start(), int(m[0])) for m in re.finditer(r"\b\d+\b", text)]
    found += [(m.start(), v) for k, v in NUM.items() for m in re.finditer(rf"\b{k}\b", text.lower())]
    return bool(found) and min(found)[1] == n


# --------------------------------------------------------------------------- scoring (pure, CI-tested)


def score(ranked: list[str], gold: set[str], k: int = K) -> tuple[int, int, float]:
    """(hit@1, hit@5, reciprocal rank) of the first gold id within the top `k` of `ranked`."""
    for i, tid in enumerate(ranked[:k]):
        if tid in gold:
            return int(i == 0), int(i < 5), 1.0 / (i + 1)
    return 0, 0, 0.0


def summarize(results: dict[str, dict[str, list[tuple[int, int, float]]]]) -> dict:
    """{category: {signal: [score tuples]}} -> {category: {signal: {n, r1, r5, mrr}}}, plus an "ALL" category
    per signal over every question that signal answered."""
    table: dict = {}
    pooled: dict[str, list] = collections.defaultdict(list)
    for cat, sigs in results.items():
        table[cat] = {}
        for sig, v in sigs.items():
            pooled[sig] += v
            table[cat][sig] = _agg(v)
    if results:
        table["ALL"] = {sig: _agg(v) for sig, v in pooled.items()}
    return table


def _agg(v: list[tuple[int, int, float]]) -> dict:
    n = len(v) or 1
    return {
        "n": len(v),
        "r1": round(sum(x[0] for x in v) / n, 4),
        "r5": round(sum(x[1] for x in v) / n, 4),
        "mrr": round(sum(x[2] for x in v) / n, 4),
    }


# --------------------------------------------------------------------------- rendering


def render_fact(f: dict, photo: Path, rnd: random.Random) -> bytes:
    """A frame for a sign fact: the sign's text on a card inside a (mirrored, cropped) scene."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    img = Image.open(photo).convert("RGB")
    w, h = img.size
    img = ImageOps.mirror(img.crop((int(w * 0.1), int(h * 0.1), w, h))).resize((960, 540))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=30)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    lines, cur = [], ""
    for word in f["clean"].split():
        cur = f"{cur} {word}".strip()
        if len(cur) > 22:
            lines.append(cur)
            cur = ""
    lines.append(cur)
    x0, y0 = rnd.randint(40, 300), rnd.randint(40, 200)
    d.rectangle([x0, y0, x0 + 440, y0 + 50 * len(lines) + 30], fill=(245, 245, 235), outline=(30, 30, 30), width=4)
    for k, line in enumerate(lines):
        d.text((x0 + 20, y0 + 20 + 50 * k), line, fill=(10, 10, 10), font=font)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


# --------------------------------------------------------------------------- calibration / dtype toggles


class _HideCalibration(ImageEmbedder):
    """The same image embedder with `calibration = None`: PR 1 defines that as uncalibrated fusion."""

    calibration = None

    def __init__(self, inner: ImageEmbedder) -> None:
        self.inner = inner
        self.dim = inner.dim

    async def embed_image(self, jpeg: bytes) -> list[float]:
        return await self.inner.embed_image(jpeg)

    async def embed_text(self, text: str) -> list[float]:
        return await self.inner.embed_text(text)


def calibration_of(emb: ImageEmbedder | None) -> tuple[float, float] | None:
    cal = getattr(emb, "calibration", None) if emb is not None else None
    return (float(cal[0]), float(cal[1])) if cal else None


def calibration_knobs(cfg: HybridConfig) -> list[str]:
    """Boolean HybridConfig fields that switch calibrated image fusion (PR 1); [] on main without PR 1."""
    return [f.name for f in dataclasses.fields(cfg) if "calibrat" in f.name and isinstance(getattr(cfg, f.name), bool)]


def set_calibration(retriever: HybridRetriever, store, base: ImageEmbedder | None, on: bool) -> None:
    if base is None:
        return
    retriever.image_embedder = base if on else _HideCalibration(base)
    if hasattr(store, "image_calibration"):
        store.image_calibration = calibration_of(base) if on else None
    for name in calibration_knobs(retriever.cfg):
        setattr(retriever.cfg, name, on)


def supports_vector_dtype() -> bool:
    import inspect

    from megamind.store.lancedb_store import LanceDBStore

    return "vector_dtype" in inspect.signature(LanceDBStore.__init__).parameters


# --------------------------------------------------------------------------- embedders


def build_embedders(text: str, image: str) -> tuple[TextEmbedder, ImageEmbedder | None]:
    from megamind.store import embeddings as E

    if text == "local":
        t: TextEmbedder = E.LocalTextEmbedder()
    elif text == "openai":
        from megamind.config import Settings

        s = Settings()
        if not s.openai_api_key:
            raise SystemExit("--text openai needs OPENAI_API_KEY")
        t = E.OpenAITextEmbedder(s.openai_api_key, s.embed_model, s.embed_dim)
    else:
        t = E.MockTextEmbedder()
    if image == "openclip":
        i: ImageEmbedder | None = E.OpenCLIPImageEmbedder()
    elif image == "mock":
        i = E.MockImageEmbedder()
    else:
        i = None
    return t, i


def _name(obj) -> str:
    if obj is None:
        return "none"
    return f"{type(obj).__name__}:{getattr(obj, 'model', None) or getattr(obj, 'model_name', '')}".rstrip(":")


def doc_text(t: MemoryToken) -> str:
    """What the enrichment worker embeds for a token (worker.py)."""
    return f"{t.caption}\n{t.ocr_text}\n{' '.join(t.entities)}"


# --------------------------------------------------------------------------- bench


class Bench:
    def __init__(
        self,
        data: Path,
        cache: Path,
        text: TextEmbedder,
        image: ImageEmbedder | None,
        *,
        limit: int | None = None,
        seed: int = 0,
    ) -> None:
        self.text, self.image = text, image
        self.data, self.cache = data, cache
        self.seed = seed
        self.coco = json.loads((data / "coco/subset.json").read_text())[:limit]
        self.flickr = json.loads((data / "flickr/captions.json").read_text())[:limit]
        self._qcache: dict[tuple[str, str], list[float]] = {}

    async def image_vecs(self, paths: list[Path]) -> dict[str, list[float]]:
        if self.image is None:
            return {}
        import numpy as np

        f = self.cache / f"imgvec_{hashlib.sha1(_name(self.image).encode()).hexdigest()[:10]}.npz"
        have = dict(np.load(f)) if f.exists() else {}
        todo = [p for p in paths if p.name not in have]
        for i, p in enumerate(todo):
            have[p.name] = np.asarray(await self.image.embed_image(p.read_bytes()), dtype=np.float32)
            if i % 100 == 99:
                print(f"  image vectors {i + 1}/{len(todo)}", flush=True)
        if todo:
            np.savez(f, **have)
        return {p.name: [float(x) for x in have[p.name]] for p in paths}

    async def qvec(self, kind: str, text: str) -> list[float]:
        key = (kind, text)
        if key not in self._qcache:
            self._qcache[key] = await (self.text.embed_query(text) if kind == "t" else self.image.embed_text(text))
        return self._qcache[key]

    async def build(self, captioned: bool) -> tuple[list[MemoryToken], list[dict]]:
        rnd = random.Random(self.seed + 3)
        cpaths = [self.data / "coco/img" / r["file"] for r in self.coco]
        fpaths = [self.data / "flickr/img" / r["file"] for r in self.flickr]
        self.counts = count_questions(self.coco, random.Random(self.seed + 2))[:40]
        facts = make_facts(random.Random(self.seed + 1))
        busy = {c["img"] for c in self.counts}
        spare = [p for p, r in zip(cpaths, self.coco) if r["file"] not in busy] or cpaths
        fdir = self.cache / "facts"
        fdir.mkdir(parents=True, exist_ok=True)
        frnd = random.Random(self.seed + 4)
        fact_paths: dict[int, Path] = {}
        for i, f in enumerate(facts):
            if f["kind"] != "ocr":
                continue
            fp = fdir / f"fact_{self.seed}_{i:03d}.jpg"
            photo = frnd.choice(spare)
            if not fp.exists():
                fp.write_bytes(render_fact(f, photo, frnd))
            fact_paths[i] = fp
        ivec = await self.image_vecs(cpaths + fpaths + list(fact_paths.values()))
        mock = MockEnricher()
        toks: list[MemoryToken] = []

        async def frame(tid: str, path: Path, caption: str | None, ocr: str = "", ts: float | None = None) -> None:
            if caption is None:  # keyless: exactly what main's pipeline stores with no enricher
                e = await mock.enrich(path.read_bytes())
                caption, entities = e.caption, e.entities
            else:
                entities = []
            t = MemoryToken(
                id=tid,
                device_id=DEVICE,
                ts=ts if ts is not None else T0 + rnd.random() * 10 * DAY,
                caption=caption,
                ocr_text=ocr,
                entities=entities,
                thumb_path=str(path),
                frame_hash=hashlib.sha1(tid.encode()).hexdigest()[:16],
                image_vec=ivec.get(path.name),
            )
            t.text_vec = await self.text.embed(doc_text(t))
            toks.append(t)

        for r, p in zip(self.coco, cpaths):
            await frame(f"coco:{r['file']}", p, r["captions"][0] if captioned else None)
        for r, p in zip(self.flickr, fpaths):
            await frame(f"flickr:{r['file']}", p, r["captions"][0] if captioned else None)
        qs: list[dict] = []
        for i, f in enumerate(facts):
            tid = f"fact:{i}"
            if f["kind"] == "ocr":
                await frame(tid, fact_paths[i], SIGN_CAPTION if captioned else None, ocr=f["text"], ts=f["ts"])
            else:  # speech: a separate transcript token, as POST /v1/audio stores it (#63)
                caption = f'{CAPTION_PREFIX}"{f["text"]}"'
                t = MemoryToken(
                    id=tid,
                    device_id=DEVICE,
                    ts=f["ts"] if f["ts"] is not None else T0 + rnd.random() * 10 * DAY,
                    caption=caption,
                    entities=["speech"],
                    thumb_path="",
                    frame_hash="",
                    kind="transcript",
                )
                t.text_vec = await self.text.embed(caption)
                toks.append(t)
            qs += [{"cat": c, "q": q, "gold": {tid}, "window": w} for c, q, w in f["queries"]]
        for r in self.flickr:
            if len(r["captions"]) > 2:
                qs.append(
                    {"cat": "visual_scene", "q": r["captions"][2], "gold": {f"flickr:{r['file']}"}, "window": None}
                )
        for c in self.counts:
            qs.append({"cat": "count_target", "q": c["q"], "gold": {f"coco:{c['img']}"}, "window": None, "count": c})
        self.finds = find_questions(self.coco)
        for f in self.finds:
            qs.append(
                {
                    "cat": "object_find",
                    "q": f["q"],
                    "gold": {f"coco:{x}" for x in f["valid"]},
                    "window": None,
                    "base_rate": f["base_rate"],
                    "find": f,
                }
            )
        return toks, qs

    async def primitive(self, adapter: StoreAdapter, item: dict, signal: str) -> list[str]:
        lo, hi = item["window"] or (None, None)
        flt = {"time_from": lo, "time_to": hi}
        if signal == "bm25":
            rows = await adapter.keyword_search(item["q"], K, **flt)
        elif signal == "text_vec":
            rows = await adapter.vector_search(await self.qvec("t", item["q"]), K, "text_vec", **flt)
        else:
            rows = await adapter.vector_search(await self.qvec("i", item["q"]), K, "image_vec", **flt)
        return [i for i, _ in rows]

    async def hybrid(self, retriever: HybridRetriever, item: dict) -> list[str]:
        lo, hi = item["window"] or (None, None)
        res = await retriever.retrieve(
            item["q"], k=K, time_from=lo, time_to=hi, device_id=None, now=datetime.fromtimestamp(NOW, UTC)
        )
        return [r.token.id for r in res.ranked]

    async def store_search(self, store, item: dict) -> list[str]:
        lo, hi = item["window"] or (None, None)
        iv = await self.qvec("i", item["q"]) if self.image is not None else None
        hits = await store.search(await self.qvec("t", item["q"]), iv, item["q"], k=K, time_from=lo, time_to=hi)
        return [h.id for h in hits]


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def load_weights(path: Path | None) -> dict[str, Weights] | None:
    """JSON {query_type: {text, bm25, image, recency, ocr, ocr_bm25}} -> planner weight overrides (merged over
    the planner's defaults for that type)."""
    if path is None:
        return None
    raw = json.loads(path.read_text())
    return {t: dataclasses.replace(WEIGHTS_BY_TYPE.get(t, Weights()), **w) for t, w in raw.items()}


async def run(
    data: Path,
    *,
    text: TextEmbedder,
    image: ImageEmbedder | None,
    reader: str = "none",
    reader_limit: int = 30,
    out: Path | None = None,
    weights: dict[str, Weights] | None = None,
    configs: tuple[str, ...] = ("keyless", "captioned"),
    limit: int | None = None,
    seed: int = 0,
    store_backend: str = "lancedb",
    max_questions: int | None = None,
    quiet: bool = False,
) -> dict:
    from megamind.store.lancedb_store import LanceDBStore
    from megamind.store.memory_store import InMemoryStore

    cache = data / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    b = Bench(data, cache, text, image, limit=limit, seed=seed)
    cal = calibration_of(image)
    fp16 = store_backend == "lancedb" and supports_vector_dtype()
    hcfg = HybridConfig(expand_top=0, weights_by_type=weights)  # expand_top=0: score the ranking, not context
    report: dict = {
        "config": {
            "git": _git_rev(),
            "text_embedder": _name(text),
            "image_embedder": _name(image),
            "image_calibration": cal,
            "calibration_knobs": calibration_knobs(hcfg) + (["store.image_calibration"] if cal else []),
            "float16_store": fp16,
            "hybrid_config": {k: v for k, v in dataclasses.asdict(hcfg).items() if k != "weights_by_type"},
            "weights_by_type": {t: dataclasses.asdict(w) for t, w in {**WEIGHTS_BY_TYPE, **(weights or {})}.items()},
            "k": K,
            "now": NOW,
            "seed": seed,
            "limit": limit,
            "store": store_backend,
            "max_questions": max_questions,
        },
        "retrieval": {},
        "planner": {},
        "latency_ms_p50": {},
    }
    signals = ["bm25", "text_vec", *(["image_vec"] if image is not None else []), "hybrid"]
    if cal is not None:
        signals.append("hybrid_nocal")
    if fp16:
        signals.append("hybrid_fp16")
    signals.append("store_search")
    report["config"]["signals"] = signals
    if cal is None and not quiet:
        print("image calibration unavailable (no image embedder, or PR 1 not merged): hybrid_nocal skipped")
    if store_backend == "lancedb" and not fp16 and not quiet:
        print("LanceDBStore has no vector_dtype (PR 1 not merged): float16 comparison skipped")

    for config in configs:
        toks, qs = await b.build(config == "captioned")
        if max_questions is not None:
            qs = random.Random(seed).sample(qs, min(max_questions, len(qs)))
        with tempfile.TemporaryDirectory() as td:
            kw = {"text_dim": text.dim, "image_dim": image.dim if image is not None else None}
            if fp16:
                kw["vector_dtype"] = "float32"
            store = LanceDBStore(Path(td) / "fp32", **kw) if store_backend == "lancedb" else InMemoryStore()
            await store.add_many(toks)
            store16 = None
            if fp16:
                store16 = LanceDBStore(Path(td) / "fp16", **{**kw, "vector_dtype": "float16"})
                await store16.add_many(toks)
            retriever = HybridRetriever(store, text, image, tz="UTC", config=hcfg)
            r16 = HybridRetriever(store16, text, image, tz="UTC", config=hcfg) if store16 is not None else None
            adapter = StoreAdapter(store)
            res: dict = collections.defaultdict(lambda: collections.defaultdict(list))
            lat: dict[str, list[float]] = collections.defaultdict(list)
            qtypes: dict = collections.defaultdict(collections.Counter)
            for item in qs:
                qtypes[item["cat"]][
                    plan_query(item["q"], now=datetime.fromtimestamp(NOW, UTC), tz="UTC").query_type
                ] += 1
                for sig in signals:
                    t = time.perf_counter()
                    if sig in ("bm25", "text_vec", "image_vec"):
                        ranked = await b.primitive(adapter, item, sig)
                    elif sig == "hybrid":
                        set_calibration(retriever, store, image, True)
                        ranked = await b.hybrid(retriever, item)
                        item["top"] = ranked
                    elif sig == "hybrid_nocal":
                        set_calibration(retriever, store, image, False)
                        ranked = await b.hybrid(retriever, item)
                        set_calibration(retriever, store, image, True)
                    elif sig == "hybrid_fp16":
                        set_calibration(r16, store16, image, True)
                        ranked = await b.hybrid(r16, item)
                    else:
                        if hasattr(store, "image_calibration"):
                            store.image_calibration = cal
                        ranked = await b.store_search(store, item)
                    lat[sig].append((time.perf_counter() - t) * 1000)
                    res[item["cat"]][sig].append(score(ranked, item["gold"]))
            table = summarize(res)
            if "object_find" in table:
                finds = [i for i in qs if i["cat"] == "object_find"]
                table["object_find"]["random_r1"] = round(sum(i["base_rate"] for i in finds) / len(finds), 4)
            report["retrieval"][config] = table
            report["planner"][config] = {c: dict(v) for c, v in qtypes.items()}
            report["latency_ms_p50"][config] = {s: round(statistics.median(v), 2) for s, v in lat.items()}
            report["config"][f"corpus_tokens_{config}"] = len(toks)
            report["config"][f"questions_{config}"] = len(qs)
            if reader == "vlm":
                report.setdefault("reader", {})[config] = await read_stage(b, toks, qs, reader_limit)
    if not quiet:
        print_report(report)
    if out:
        out.write_text(json.dumps(report, indent=1))
        if not quiet:
            print("report ->", out)
    return report


async def read_stage(b: Bench, toks: list[MemoryToken], qs: list[dict], limit: int) -> dict:
    """Main's VLMReader on the retrieved top 3 vs on the right frame alone, for counts and "where"."""
    from megamind.agent.reader import VLMReader
    from megamind.config import Settings

    if not hasattr(b, "reader"):
        b.reader = VLMReader(Settings())
    by_id = {t.id: t for t in toks}
    coco = {r["file"]: r for r in b.coco}
    out = {
        "count": {"oracle": 0, "retrieved": 0, "gold_in_top3": 0, "n": 0},
        "where": {"found_object": 0, "named_place": 0, "n": 0},
    }
    for item in [i for i in qs if i["cat"] == "count_target"][:limit]:
        c = item["count"]
        gold = by_id[f"coco:{c['img']}"]
        top = [by_id[i] for i in item["top"][:3]]
        o = await b.reader.read(c["q"], [gold], now=NOW)
        r = await b.reader.read(c["q"], top, now=NOW)
        out["count"]["n"] += 1
        out["count"]["oracle"] += says_number(o.evidence, c["n"])
        out["count"]["retrieved"] += says_number(r.evidence, c["n"])
        out["count"]["gold_in_top3"] += gold.id in {t.id for t in top}
    for item in [i for i in qs if i["cat"] == "object_find"]:
        f = item["find"]
        top = [by_id[i] for i in item["top"][:3]]
        r = await b.reader.read(f["q"], top, now=NOW)
        out["where"]["n"] += 1
        hit = next((i.removeprefix("coco:") for i in r.cited_ids if i.removeprefix("coco:") in f["valid"]), None)
        if hit:
            out["where"]["found_object"] += 1
            sup = supports_of(coco[hit], f["cat"])
            out["where"]["named_place"] += any(re.search(rf"\b{w}\b", r.evidence.lower()) for w in sup)
    return out


def print_report(r: dict) -> None:
    sigs = r["config"]["signals"]
    for config, table in r["retrieval"].items():
        n = r["config"].get(f"corpus_tokens_{config}")
        print(f"\n== {config} ({n} tokens)   R@1 / R@5 / MRR@{K}")
        print(f"{'category':22}" + "".join(f"{s:>20}" for s in sigs))
        for cat, row in table.items():
            cells = "".join(
                f"{row[s]['r1']:6.2f}{row[s]['r5']:6.2f}{row[s]['mrr']:6.2f}  " if s in row else " " * 20 for s in sigs
            )
            extra = f"  random R@1 {row['random_r1']:.2f}" if "random_r1" in row else ""
            first = next(iter(v for k, v in row.items() if isinstance(v, dict)), {"n": 0})
            print(f"{cat:16} n={first['n']:3d} {cells}{extra}")
        print("p50 ms:", json.dumps(r["latency_ms_p50"].get(config, {})))
    for config, rd in (r.get("reader") or {}).items():
        print("reader", config, json.dumps(rd))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help=f"dataset/cache dir (default {DEFAULT_DATA})")
    ap.add_argument("--prepare", action="store_true", help="download COCO + Flickr into --data and exit")
    ap.add_argument("--text", choices=["local", "openai", "mock"], default="local", help="text embedder")
    ap.add_argument("--image", choices=["openclip", "mock", "none"], default="openclip", help="image embedder")
    ap.add_argument("--reader", choices=["none", "vlm"], default="none", help="vlm: main's VLMReader (READER_API_KEY)")
    ap.add_argument("--reader-limit", type=int, default=30)
    ap.add_argument("--weights", type=Path, default=None, help="JSON planner weight overrides per query type")
    ap.add_argument("--configs", default="keyless,captioned")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N COCO and N Flickr photos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", choices=["lancedb", "memory"], default="lancedb", help="store backend under test")
    ap.add_argument("--max-questions", type=int, default=None, help="score a seeded sample of N questions")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    a = ap.parse_args()
    if a.prepare:
        prepare(a.data)
        return
    if not (a.data / "coco/subset.json").exists():
        raise SystemExit(f"no dataset in {a.data}: run with --prepare first")
    text, image = build_embedders(a.text, a.image)
    asyncio.run(
        run(
            a.data,
            text=text,
            image=image,
            reader=a.reader,
            reader_limit=a.reader_limit,
            out=a.out,
            weights=load_weights(a.weights),
            configs=tuple(c for c in a.configs.split(",") if c),
            limit=a.limit,
            seed=a.seed,
            store_backend=a.store,
            max_questions=a.max_questions,
        )
    )


if __name__ == "__main__":
    main()
