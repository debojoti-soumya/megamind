#!/usr/bin/env python
"""End-to-end eval of the agent loop (Claude -> recall_memory -> reader) on a scripted day.

    cd server && . .venv/bin/activate
    python tools/eval_agent.py                      # offline: no-key fallback path + mock reader, 20 questions
    python tools/eval_agent.py --json out.json      # also dump per-turn records
    python tools/eval_agent.py --live --limit 5     # real Claude/reader from env keys (costs money; ask first)

Dataset: `tools/data/agent_day.json` — same schema as retrieval's `retrieval_day.json` (the stream *is* that
synthetic day) plus `questions[].expect`: any-of substrings a correct spoken answer must contain, or
`["__not_found__"]` when the agent must decline. Scores: answer correctness, citation precision/recall
against `gold` ids; per turn latency (total / LLM / tool / reader) and token usage -> USD.

Without `--live` the harness blanks `ANTHROPIC_API_KEY` and forces the mock reader so it can never spend money.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megamind.agent.claude_agent import NOT_FOUND_ANSWER, ClaudeAgent, TurnStats  # noqa: E402
from megamind.agent.reader import Reader, ReaderOutput, build_reader  # noqa: E402
from megamind.agent.tools import RecallMemory  # noqa: E402
from megamind.config import Settings  # noqa: E402
from megamind.ingest.enrich import MockTextEmbedder, build_image_embedder, build_text_embedder  # noqa: E402
from megamind.models import MemoryToken  # noqa: E402
from megamind.store.memory_store import InMemoryStore  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "agent_day.json"
NOT_FOUND = "__not_found__"
# USD per 1M tokens (Claude Sonnet 4.5 list price); reader price is per-provider, default 0 until Sail publishes one
CLAUDE_PRICE_IN, CLAUDE_PRICE_OUT = 3.0, 15.0


@dataclass
class Case:
    q: str
    gold: list[str]
    expect: list[str]

    @property
    def not_found(self) -> bool:
        return NOT_FOUND in self.expect


@dataclass
class Day:
    tokens: list[MemoryToken]
    cases: list[Case]
    now: float
    tz: ZoneInfo
    device_id: str


def _local(d: date, hhmm: str, tz: ZoneInfo) -> datetime:
    h, m = hhmm.split(":")
    return datetime.combine(d, time(int(h), int(m)), tzinfo=tz)


PHOTOS: dict[str, str] = {}  # token id -> real photo path (tools/make_coco_day.py)
SCENES: dict[str, dict] = {}  # token id -> drawn scene (tools/make_visual_day.py); else a caption card


def _draw_scene(scene: dict, width: int):
    from PIL import Image, ImageDraw

    k = width / 960
    img = Image.new("RGB", (width, round(720 * k)), tuple(scene.get("bg", [220, 220, 220])))
    dr = ImageDraw.Draw(img)
    for sh in scene["shapes"]:
        xy, c = [v * k for v in sh["xy"]], tuple(sh["color"])
        kind = sh["shape"]
        if kind == "rect":
            dr.rectangle(xy, fill=c)
        elif kind in ("circle", "ellipse"):
            dr.ellipse(xy, fill=c)
        elif kind == "ring":
            dr.ellipse(xy, outline=c, width=max(2, round(12 * k)))
        elif kind == "pie":
            dr.pieslice(xy, 180, 360, fill=c)
        elif kind == "line":
            dr.line(xy, fill=c, width=max(1, round(4 * k)))
    return img


def load_day(path: Path | str = DEFAULT_DATASET, thumb_dir: Path | None = None) -> Day:
    raw = json.loads(Path(path).read_text())
    tz = ZoneInfo(raw.get("tz", "America/Los_Angeles"))
    d = date.fromisoformat(raw["date"])
    device_id = raw.get("device_id", "eval-device")
    tokens = []
    for e in raw["stream"]:
        thumb = thumb_dir / f"{e['id']}.jpg" if thumb_dir else Path(f"/nonexistent/{e['id']}.jpg")
        tokens.append(
            MemoryToken(
                id=e["id"],
                device_id=device_id,
                kind=e.get("kind", "frame"),
                ts=_local(d + timedelta(days=e.get("day", 0)), e["time"], tz).timestamp(),
                caption=e["caption"],
                ocr_text=e.get("ocr", ""),
                entities=e.get("entities", []),
                thumb_path=str(thumb),
                frame_hash=f"{abs(hash(e['id'])) % (1 << 64):016x}",
                episode_id=e.get("episode"),
            )
        )
    for t, e in zip(tokens, raw["stream"], strict=True):
        if "scene" in e:
            SCENES[t.id] = e["scene"]
        if "image" in e:
            PHOTOS[t.id] = e["image"]
    cases = [Case(q=q["q"], gold=list(q.get("gold", [])), expect=list(q.get("expect", []))) for q in raw["questions"]]
    now = _local(d + timedelta(days=raw.get("now_day", 0)), raw["now"], tz).timestamp()
    return Day(tokens=tokens, cases=cases, now=now, tz=tz, device_id=device_id)


def render_thumbs(day: Day, out_dir: Path, width: int = 512) -> None:
    """Synthetic frames: the caption + OCR text drawn on a card, so a real VLM has pixels to read.
    `width` > 512 scales the card up (e.g. 960 = headset frame size) so the reader has to downscale it."""
    from PIL import Image, ImageDraw, ImageFont

    out_dir.mkdir(parents=True, exist_ok=True)
    scale = width / 512
    font = ImageFont.load_default(size=round(11 * scale)) if scale != 1 else None
    for t in day.tokens:
        if t.id in PHOTOS:
            ph = Image.open(PHOTOS[t.id]).convert("RGB")
            ph.thumbnail((width, width))
            ph.save(t.thumb_path, "JPEG", quality=85)
            continue
        if t.id in SCENES:
            _draw_scene(SCENES[t.id], width).save(t.thumb_path, "JPEG", quality=80)
            continue
        img = Image.new("RGB", (width, round(384 * scale)), (235, 235, 225))
        dr = ImageDraw.Draw(img)
        y = 16 * scale
        for line in [t.caption, *t.ocr_text.splitlines()]:
            for i in range(0, len(line), 48):
                dr.text((16 * scale, y), line[i : i + 48], fill=(20, 20, 20), font=font)
                y += 18 * scale
        img.save(t.thumb_path, "JPEG", quality=80)


async def seed_store(tokens: list[MemoryToken], text_embedder=None, image_embedder=None) -> InMemoryStore:
    emb = text_embedder or MockTextEmbedder()
    store = InMemoryStore()
    texts = [f"{t.caption} {t.ocr_text}" for t in tokens]
    if hasattr(emb, "embed_many") and len(tokens) > 64:  # long datasets: batched provider calls
        vecs = [v for i in range(0, len(texts), 256) for v in await emb.embed_many(texts[i : i + 256])]
    else:
        vecs = [await emb.embed(x) for x in texts]
    for t, v in zip(tokens, vecs, strict=True):
        upd: dict = {"text_vec": v}
        if image_embedder is not None and Path(t.thumb_path).is_file():
            upd["image_vec"] = await image_embedder.embed_image(Path(t.thumb_path).read_bytes())
        await store.add(t.model_copy(update=upd))
    return store


class RecordingReader(Reader):
    """Wraps the real reader so per-turn reader latency/usage/degradation can be attributed."""

    def __init__(self, inner: Reader) -> None:
        self.inner = inner
        self.outputs: list[ReaderOutput] = []

    async def read(self, query: str, tokens: list[MemoryToken], *, now: float | None = None) -> ReaderOutput:
        out = await self.inner.read(query, tokens, now=now)
        self.outputs.append(out)
        return out

    def drain(self) -> list[ReaderOutput]:
        outs, self.outputs = self.outputs, []
        return outs


# --------------------------------------------------------------------------- scoring


_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_NUM_WORD = "|".join(sorted([*_UNITS, *_TENS, "hundred", "thousand"], key=len, reverse=True))
_NUM_RUN = re.compile(rf"\b(?:{_NUM_WORD})(?:(?:\s+and\s+|\s+oh\s+|[\s-]+)(?:{_NUM_WORD}))*\b")
# digit groups separated by these are one spoken number: "4.99" / "4 99" / "4 dollars and 99", "10:30" / "10 30"
_DIGIT_JOIN = re.compile(r"(\d)(?:[:.,\-/]|,? |\s+(?:dollars?|bucks?|point)(?:\s+and)?\s+)(?=\d)")


def _run_to_digits(run: str) -> str:
    """One run of number words -> space-separated numbers, splitting where speech starts a new number:
    "four ninety nine" -> "4 99", "four hundred and eighty" -> "480", "ten oh five" -> "10 0 5"."""
    out: list[int] = []
    big = small = 0
    last = ""  # previous word kind: unit | tens | hundred | thousand

    def flush() -> None:
        nonlocal big, small, last
        if last:
            out.append(big + small)
        big = small = 0
        last = ""

    for w in re.split(r"[\s-]+", run):
        if w == "and":
            if last not in ("hundred", "thousand"):
                flush()
            continue
        if w == "oh":
            w = "zero"
        if w in _UNITS:
            v = _UNITS[w]
            if not ((last == "tens" and v < 10) or last in ("hundred", "thousand")):
                flush()
            small, last = small + v, "unit"
        elif w in _TENS:
            if last not in ("hundred", "thousand"):
                flush()
            small, last = small + _TENS[w], "tens"
        elif w == "hundred" and last in ("unit", "tens") and 0 < small < 100:
            small, last = small * 100, "hundred"
        elif w == "thousand" and last in ("unit", "tens", "hundred") and small and not big:
            big, small, last = small * 1000, 0, "thousand"
        else:
            flush()
            out.append({"hundred": 100, "thousand": 1000}[w])
    flush()
    return " ".join(map(str, out))


def normalize_numbers(text: str) -> str:
    """Canonical form for number matching: TTS-style answers spell numbers out ("ten thirty", "four
    ninety-nine") while expects are digits ("10:30", "4.99"); both map to joined digit runs ("1030", "499")."""
    s = _NUM_RUN.sub(lambda m: _run_to_digits(m.group(0)), " ".join(text.lower().split()))
    prev = None
    while prev != s:
        prev, s = s, _DIGIT_JOIN.sub(r"\1", s)
    return s


def _number_match(expect: str, answer_norm: str) -> bool:
    e = normalize_numbers(expect)
    if not any(c.isdigit() for c in e):
        return False
    return re.search(rf"(?<!\d){re.escape(e)}(?!\d)", answer_norm) is not None


def score_answer(answer: str, case: Case) -> bool:
    a = " ".join(answer.lower().split())
    if case.not_found:
        return NOT_FOUND_ANSWER.lower() in a or "don't have a memory" in a or "no memory" in a
    if any(e.lower() in a for e in case.expect):
        return True
    a_norm = normalize_numbers(answer)
    return any(_number_match(e, a_norm) for e in case.expect)


def citation_scores(cited: list[str], gold: list[str]) -> tuple[float | None, float | None]:
    """(precision, recall); None when undefined (no citations / no gold)."""
    c, g = set(cited), set(gold)
    precision = len(c & g) / len(c) if c else None
    recall = len(c & g) / len(g) if g else None
    return precision, recall


def claude_cost(stats: TurnStats, price_in: float, price_out: float) -> float:
    return (stats.input_tokens * price_in + stats.output_tokens * price_out) / 1e6


@dataclass
class Turn:
    i: int
    q: str
    answer: str
    path: str
    correct: bool
    cited: list[str]
    gold: list[str]
    cite_precision: float | None
    cite_recall: float | None
    total_ms: float
    llm_ms: float
    tool_ms: float
    reader_ms: float
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    reader_tokens: int
    reader_degraded: bool
    cost_usd: float
    request_id: str
    error: str | None = None
    extra: dict = field(default_factory=dict)  # TurnStats.extra: prefetch hit/miss, reader_skipped
    extra: dict = field(default_factory=dict)


def summarize(turns: list[Turn]) -> dict:
    n = len(turns)
    if not n:
        return {}
    prec = [t.cite_precision for t in turns if t.cite_precision is not None]
    rec = [t.cite_recall for t in turns if t.cite_recall is not None]
    lat = sorted(t.total_ms for t in turns)
    p95 = lat[min(n - 1, int(round(0.95 * (n - 1))))]
    return {
        "n": n,
        "accuracy": sum(t.correct for t in turns) / n,
        "citation_precision": statistics.fmean(prec) if prec else None,
        "citation_recall": statistics.fmean(rec) if rec else None,
        "declined_correctly": sum(t.correct for t in turns if not t.gold),
        "latency_p50_ms": statistics.median(lat),
        "latency_p95_ms": p95,
        "latency_mean_ms": statistics.fmean(lat),
        "reader_ms_mean": statistics.fmean(t.reader_ms for t in turns),
        "model_calls": sum(t.model_calls for t in turns),
        "tool_calls": sum(t.tool_calls for t in turns),
        "input_tokens": sum(t.input_tokens for t in turns),
        "output_tokens": sum(t.output_tokens for t in turns),
        "reader_tokens": sum(t.reader_tokens for t in turns),
        "cost_usd": sum(t.cost_usd for t in turns),
        "degraded_turns": sum(t.reader_degraded for t in turns),
        "paths": {p: sum(t.path == p for t in turns) for p in sorted({t.path for t in turns})},
    }


# --------------------------------------------------------------------------- run


def eval_settings(live: bool) -> Settings:
    s = Settings()
    if not live:  # never spend money by accident
        s = s.model_copy(update={"anthropic_api_key": "", "reader_provider": "mock"})
    return s


async def eval_retriever(settings: Settings, store, text_emb, image_emb, day: Day):
    """`settings.retriever_backend` (hybrid when unset). UCC gets the day pushed to a throwaway eval corpus first;
    hier gets every block/hour/day note written up front, so neither pays index time inside a measured turn."""
    if settings.retriever_backend == "hybrid":
        return None
    import hashlib
    import time as _time

    from megamind.retrieval import build_retriever
    from megamind.retrieval.hier import HierRetriever

    digest = hashlib.sha256(json.dumps([t.id for t in day.tokens]).encode()).hexdigest()[:10]
    settings = settings.model_copy(update={"ucc_corpus": f"megamind-eval-agent-{digest}", "hier_cache_path": ""})
    retriever, indexer = build_retriever(settings, store, text_emb, image_emb)
    if indexer is not None:
        indexer.client.keep_alive_s = 3 * 86400
        await indexer.sync_once()
        want = len([t for t in day.tokens if t.caption])
        deadline = _time.monotonic() + float(os.environ.get("EVAL_UCC_WAIT_S", "180"))
        while len(await indexer.client.token_ids()) < want and _time.monotonic() < deadline:
            await asyncio.sleep(3)  # uploads become queryable once the engine has read them
    hier = retriever if isinstance(retriever, HierRetriever) else getattr(retriever, "hier", None)
    if hier is not None:
        await hier.memory.build(day.now, full=True)
    print(f"retriever: {type(retriever).__name__}", file=sys.stderr)
    return retriever


async def run(
    day: Day,
    settings: Settings,
    *,
    limit: int | None = None,
    price_in: float = CLAUDE_PRICE_IN,
    price_out: float = CLAUDE_PRICE_OUT,
    reader_price: float = 0.0,
    agent: ClaudeAgent | None = None,
) -> list[Turn]:
    text_emb = build_text_embedder(settings)
    image_emb = build_image_embedder(settings)
    store = await seed_store(day.tokens, text_emb, image_emb)
    reader = RecordingReader(build_reader(settings))
    retriever = await eval_retriever(settings, store, text_emb, image_emb, day)
    if agent is None:
        agent = ClaudeAgent(settings, RecallMemory(store, text_emb, image_emb, reader, retriever=retriever))
    else:  # test hook: caller-built agent (e.g. mocked Anthropic client); re-point its tool at our store
        agent.recall = RecallMemory(store, text_emb, image_emb, reader, retriever=retriever)

    turns: list[Turn] = []
    routes: list[str] = []
    for i, case in enumerate(day.cases[: limit or None]):
        resp, stats = await agent.chat_with_stats(f"eval-{i}", case.q, day.device_id, now=day.now)
        outs = reader.drain()
        if retriever is not None:
            routes.append(f"{i + 1}:{getattr(retriever, 'last_route', '') or type(retriever).__name__}")
        cited = [c.id for c in resp.cited]
        p, r = citation_scores(cited, case.gold)
        reader_tokens = sum(o.usage.get("total_tokens", 0) for o in outs)
        turns.append(
            Turn(
                i=i + 1,
                q=case.q,
                answer=resp.answer,
                path=stats.path,
                correct=score_answer(resp.answer, case),
                cited=cited,
                gold=case.gold,
                cite_precision=p,
                cite_recall=r,
                total_ms=stats.total_ms,
                llm_ms=stats.llm_ms,
                tool_ms=stats.tool_ms,
                reader_ms=sum(o.latency_ms for o in outs),
                model_calls=stats.model_calls,
                tool_calls=stats.tool_calls,
                input_tokens=stats.input_tokens,
                output_tokens=stats.output_tokens,
                reader_tokens=reader_tokens,
                reader_degraded=any(o.degraded for o in outs),
                cost_usd=claude_cost(stats, price_in, price_out) + reader_tokens * reader_price / 1e6,
                request_id=stats.request_id,
                error=stats.error,
                extra=dict(stats.extra),
            )
        )
    if retriever is not None:
        ucc = getattr(retriever, "ucc", None) or retriever
        print(f"routes: {' '.join(routes)}", file=sys.stderr)
        print(
            f"retriever stats: {dict(getattr(retriever, 'stats', {}))} ucc: {getattr(ucc, 'stats', None)}",
            file=sys.stderr,
        )
    return turns


def _fmt(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def markdown(turns: list[Turn], summ: dict, title: str) -> str:
    rows = [
        f"### {title}",
        "",
        "| # | question | path | ok | cited | gold | cite P | ms | llm ms | reader ms | in/out tok | $ |",
        "|--:|---|---|:-:|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    for t in turns:
        rows.append(
            f"| {t.i} | {t.q} | {t.path}{' (degraded)' if t.reader_degraded else ''} | {'✓' if t.correct else '✗'} "
            f"| {','.join(t.cited) or '-'} | {','.join(t.gold) or '(none)'} | {_fmt(t.cite_precision)} "
            f"| {t.total_ms:.0f} | {t.llm_ms:.0f} | {t.reader_ms:.0f} | {t.input_tokens}/{t.output_tokens} "
            f"| {t.cost_usd:.4f} |"
        )
    rows += [
        "",
        f"**{summ['n']} questions** — accuracy **{summ['accuracy']:.0%}**, citation precision "
        f"{_fmt(summ['citation_precision'])}, citation recall {_fmt(summ['citation_recall'])}; "
        f"latency p50 {summ['latency_p50_ms']:.0f} ms / p95 {summ['latency_p95_ms']:.0f} ms; "
        f"{summ['model_calls']} LLM calls, {summ['tool_calls']} tool calls, "
        f"{summ['input_tokens']}/{summ['output_tokens']} Claude tokens in/out, {summ['reader_tokens']} reader tokens, "
        f"${summ['cost_usd']:.4f} total; degraded reader turns: {summ['degraded_turns']}.",
    ]
    return "\n".join(rows)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--live", action="store_true", help="use ANTHROPIC_API_KEY / READER_API_KEY from the env ($$$)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (use with --live)")
    ap.add_argument("--no-thumbs", action="store_true", help="skip rendering synthetic frames (captions only)")
    ap.add_argument("--backend", choices=["hybrid", "combo", "ucc", "hier"], default=None, help="RETRIEVER_BACKEND")
    ap.add_argument("--thumb-width", type=int, default=512, help="synthetic frame width (960 = headset upload)")
    ap.add_argument("--price-in", type=float, default=CLAUDE_PRICE_IN, help="USD per 1M Claude input tokens")
    ap.add_argument("--price-out", type=float, default=CLAUDE_PRICE_OUT, help="USD per 1M Claude output tokens")
    ap.add_argument("--reader-price", type=float, default=0.0, help="USD per 1M reader tokens (blended)")
    ap.add_argument("--json", type=Path, default=None, help="write per-turn records + summary here")
    ap.add_argument("--answers", action="store_true", help="print each answer under the table")
    args = ap.parse_args()

    settings = eval_settings(args.live)
    if args.backend:
        settings = settings.model_copy(update={"retriever_backend": args.backend})
    if args.live and not (settings.anthropic_api_key or settings.reader_api_key not in ("", "none")):
        print("--live but no ANTHROPIC_API_KEY/READER_API_KEY in the environment", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="megamind-eval-") as tmp:
        day = load_day(args.dataset, None if args.no_thumbs else Path(tmp))
        if not args.no_thumbs:
            render_thumbs(day, Path(tmp), width=args.thumb_width)
        turns = await run(
            day,
            settings,
            limit=args.limit,
            price_in=args.price_in,
            price_out=args.price_out,
            reader_price=args.reader_price,
        )
    summ = summarize(turns)
    mode = (
        f"live: claude={settings.claude_model}, reader={settings.reader_model} ({settings.reader_provider})"
        if args.live
        else "offline: no-key fallback (recall + caption reader), mock embeddings"
    )
    print(markdown(turns, summ, f"eval_agent — {mode}"))
    if args.answers:
        print()
        for t in turns:
            print(f"{t.i:>2}. {t.q}\n    -> {t.answer}")
    if args.json:
        args.json.write_text(json.dumps({"mode": mode, "summary": summ, "turns": [asdict(t) for t in turns]}, indent=1))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    raise SystemExit(asyncio.run(main()))
