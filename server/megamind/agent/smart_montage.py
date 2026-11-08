"""Content-filtered ("smart") montages (ported from Recall's `montage.py` smart-montage section).

"Montage of all shots of the guy in the black coat" is not a highlight reel of a time period: every clip must
show the subject. The smart path:

1. `detect_subject` pulls the subject out of the request ("the guy in the black coat"); a request with no
   content words ("make a montage of my afternoon") has none and keeps the classic path unchanged.
2. `gather` asks the existing retriever for a large candidate pool across all history (or the requested range),
   then spreads it over time so one long visit can't fill every slot (at most `MONTAGE_SMART_MAX_CANDIDATES`).
3. `MontageVerifier` checks each candidate with a cheap vision call (Claude Haiku, `MONTAGE_VERIFY_MODEL`):
   caption + OCR text + thumbnail, `MONTAGE_VERIFY_BATCH` candidates per call, calls run concurrently.
   Without an API key (or for a batch whose call failed) candidates are kept unverified in retrieval order.
4. `pick_moments` groups matches into moments (`MONTAGE_MOMENT_GAP_S`), keeps the most confident token of each,
   orders them chronologically and caps them at `MONTAGE_SMART_MAX_CLIPS` (evenly spread in time if more).
   No match -> the caller answers an honest 404 ("no moments showing ...") instead of a reel of near-misses.

`montage.create_montage` decides per request (`select_mode`) and renders the picks with the classic clip and
render pipeline, so caching, purge handling, face-blur and Range serving are shared.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anthropic
from pydantic import BaseModel, Field, ValidationError

from megamind.models import MemoryToken
from megamind.retrieval.planner import parse_time

log = logging.getLogger("megamind.montage.smart")

# Words that carry no subject (Recall's list + megamind's "shots/frames/photos" framing): if nothing else is left,
# the request is a plain time-range montage.
_NON_SUBJECT = set(
    """
make me a an the of my from for all every everything video videos montage montages highlight highlights reel reels
supercut supercuts compilation recap clip clips please create show give generate build cut put together can you
could would i want like quick short little best bits moments moment day days today yesterday tonight morning
afternoon evening night week weekend this that past last few couple hours hour minutes minute whole entire so far
since until till around between before after earlier recently time times in on at to and with what happened did do
just now while was were whats it its me our we us shots shot frames frame photos photo pictures picture scenes scene
footage memories memory period stuff things everything anything any each some
monday tuesday wednesday thursday friday saturday sunday am pm one two three four five six seven eight nine ten
""".split()
)
# "every time", "all the", "shots of", "montage of": the wearer wants a collection of matching moments
_CUE = re.compile(
    r"\b(all|every|each|any|whenever)\b"
    r"|\b(shots?|clips?|times|moments?|scenes?|frames?|photos?|pictures?|footage)\s+(of|with|where|when|showing)\b"
    r"|\b(montage|supercut|compilation|reel)\s+(of|with|showing|featuring)\b",
    re.I,
)
# Leading framing stripped (repeatedly) before the subject phrase: "make me a montage of all shots of ..."
_FRAMING = re.compile(
    r"^(?:every\s+time|times\s+when|whenever|please|can you|could you|would you|i want|i'd like|show me|give me|"
    r"make me|make|create|build|generate|"
    r"cut|put together|a|an|me|quick|short|little|"
    r"(?:video\s+)?(?:montage|supercut|compilation|highlight reel|highlights?|reel|recap|video)s?"
    r"(?:\s+(?:of|with|showing|featuring))?|"
    r"(?:all|every|each|any)(?:\s+of)?(?:\s+the)?(?:\s+(?:shots?|clips?|moments?|scenes?|frames?|photos?|"
    r"pictures?|footage))?(?:\s+(?:of|with|showing|where|when))?|"
    r"(?:shots?|clips?|moments?|scenes?|frames?|photos?|pictures?|footage)\s+(?:of|with|showing|where|when)"
    r")\b[\s,:]*",
    re.I,
)
_TRAILING = re.compile(r"(?:[\s,;:.!?]+|\b(?:from|during|on|in|over|since|at|for|of|and)\s*)+$", re.I)


@dataclass(frozen=True)
class Subject:
    phrase: str  # what every clip must show, e.g. "the guy in the black coat" (time phrase removed)
    search: str  # retriever query: the phrase with any time expression kept, so the retriever's planner sees it
    cue: bool  # the request asked for a collection ("all", "every time", "shots of", "montage of" ...)


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _content_words(text: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _NON_SUBJECT and not w.isdigit() and len(w) > 1
    ]


def detect_subject(query: str, *, tz: str = "UTC", now: datetime | None = None) -> Subject | None:
    """The montage's content subject, or None for a plain time-range montage ("my afternoon", "the past 3 hours").

    Deterministic (no API call): strip the request framing ("make a montage of all shots of"), drop the time
    expression the retriever's planner recognises, and require at least one content word in what remains."""
    q = " ".join((query or "").replace("’", "'").split())
    if not q:
        return None
    rest = q
    while True:
        m = _FRAMING.match(rest)
        if not m or not m.group(0):
            break
        rest = rest[m.end() :]
    rest = rest.strip(" ,.;:!?")
    if not _content_words(rest):
        return None
    phrase = rest
    win = parse_time(rest, (now or datetime.now(UTC)).astimezone(_zone(tz)))
    if win is not None:
        a, b = win.span
        phrase = f"{rest[:a]} {rest[b:]}"
    phrase = _TRAILING.sub("", " ".join(phrase.split())).strip()
    if not _content_words(phrase):  # only a time phrase was left ("lunch" alone stays: it has content words)
        return None
    return Subject(phrase=phrase, search=rest, cue=bool(_CUE.search(q)))


def select_mode(mode: str, query: str, verifier_enabled: bool, setting: str, *, tz: str = "UTC") -> Subject | None:
    """The subject when this request takes the smart path, else None (classic path, unchanged).

    - request `mode`: "classic" never; "smart" whenever a subject is found; "" / "auto" = the `MONTAGE_SMART`
      setting. Setting "off": never (unless the request says "smart"); "auto" (default): a subject is found AND
      either the request asks for a collection ("all shots of ...", "every time ...") or the Claude verifier
      is configured. A time-range-only request has no query, hence no subject: always classic.
    """
    mode = (mode or "auto").strip().lower()
    if mode == "classic" or not (query or "").strip():
        return None
    subj = detect_subject(query, tz=tz)
    if subj is None:
        return None
    if mode == "smart":
        return subj
    if (setting or "auto").strip().lower() == "off":
        return None
    return subj if (subj.cue or verifier_enabled) else None


# ---- gather ----


def spread(ranked: list[MemoryToken], k: int, gap_s: float, per_cluster: int = 3) -> list[MemoryToken]:
    """Recall's pool spreading: cluster best-first candidates in time (`gap_s`), take at most `per_cluster` per
    cluster first, then fill up to `k` by rank. Keeps rank order."""
    chron = sorted(ranked, key=lambda t: (t.device_id, t.ts))
    cluster: dict[str, tuple[str, int]] = {}
    cid, prev = 0, None
    for t in chron:
        if prev is None or t.device_id != prev.device_id or t.ts - prev.ts > gap_s:
            cid += 1
        cluster[t.id] = (t.device_id, cid)
        prev = t
    first, rest, per = [], [], {}
    for t in ranked:
        c = cluster[t.id]
        if per.get(c, 0) < per_cluster:
            per[c] = per.get(c, 0) + 1
            first.append(t)
        else:
            rest.append(t)
    return (first + rest)[: max(k, 0)]


async def gather(
    st, subj: Subject, t0: float | None, t1: float | None, device_id: str | None, cfg
) -> list[MemoryToken]:
    """Best-first candidates for the subject across all history (or [t0, t1]) via the existing retriever."""
    k = max(1, cfg.smart_max_candidates)
    res = await st.retriever.retrieve(subj.search, k=k * 2, time_from=t0, time_to=t1, device_id=device_id)
    ranked = [r.token for r in res.ranked]
    ranked = list({t.id: t for t in ranked}.values())  # dict keeps first (best) occurrence
    return spread(ranked, k, cfg.moment_gap_s)


# ---- verify ----


class Verdict(BaseModel):
    label: str = Field(description="Candidate label, e.g. 'C3'")
    match: bool = Field(description="True only if this moment clearly shows the subject")
    confidence: float = Field(0.0, description="0-1 confidence in the match decision")
    title: str = Field("", description="3-7 word on-screen caption for this moment")


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Candidate label, e.g. 'C3'"},
                    "match": {"type": "boolean", "description": "True only if the moment clearly shows the subject"},
                    "confidence": {"type": "number", "description": "0-1 confidence in the match decision"},
                    "title": {"type": "string", "description": "3-7 word on-screen caption for this moment"},
                },
                "required": ["label", "match", "confidence", "title"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

VERIFY_SYSTEM = """You check candidate moments from the wearer's first-person AR-headset memories for a \
montage. Each candidate has a label, a time, its caption, any text read from the image, and usually its camera \
frame. Decide for EACH candidate whether the moment really shows the requested subject. Be strict: \
similar-but-different things, mere mentions of a different thing, dark/blocked frames and guesses don't count. \
But the wearer speaks casually: treat everyday paraphrases as the same thing (guy = man, coat = jacket, cafe = \
coffee shop, cup = mug) as long as the distinguishing details (colour, kind of thing, place) agree. A caption \
that clearly names the subject counts even if the frame is unclear, and a frame that clearly shows it counts even \
if the caption doesn't mention it. Never identify people by name unless the text names them; describe them \
neutrally. Return one verdict per candidate, as JSON."""

ImageLoader = Callable[[MemoryToken], Awaitable[bytes | None]]


class MontageVerifier:
    """Haiku vision check of montage candidates. `verify` returns one entry per candidate: a `Verdict`, or None
    when the candidate was not checked (no client, or its batch's call failed): the caller keeps those (offline
    rule: retrieval order)."""

    def __init__(self, settings, cfg, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.cfg = cfg
        self.model = cfg.verify_model
        self.tz = settings.wearer_tz
        self._client = client
        if self._client is None and cfg.verifier.lower() != "offline" and settings.anthropic_api_key:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                max_retries=settings.claude_max_retries,
                timeout=settings.claude_timeout_s,
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def verify(
        self, subject: str, cands: list[MemoryToken], load_image: ImageLoader | None = None
    ) -> tuple[list[Verdict | None], dict]:
        stats = {"verify_calls": 0, "verify_errors": 0}
        if self._client is None or not cands:
            return [None] * len(cands), stats
        size = max(1, self.cfg.verify_batch)
        labelled = [(f"C{i + 1}", t) for i, t in enumerate(cands)]
        batches = [labelled[i : i + size] for i in range(0, len(labelled), size)]
        sem = asyncio.Semaphore(max(1, self.cfg.verify_concurrency))

        async def run(batch):
            async with sem:
                stats["verify_calls"] += 1
                try:
                    return await self._verify_batch(subject, batch, load_image)
                except Exception as e:  # noqa: BLE001  (API/network error, unparsable or schema-mismatched output)
                    log.warning("montage verify batch failed (%s); keeping its %d candidates unverified", e, len(batch))
                    stats["verify_errors"] += 1
                    return None

        results = await asyncio.gather(*(run(b) for b in batches))
        out: list[Verdict | None] = []
        for batch, verdicts in zip(batches, results, strict=True):
            if verdicts is None:
                out += [None] * len(batch)
                continue
            by_label = {v.label.strip().strip("[]").upper(): v for v in verdicts}
            # a label the model skipped counts as "no match": it was shown and not confirmed
            out += [by_label.get(lab) or Verdict(label=lab, match=False) for lab, _ in batch]
        return out, stats

    async def _verify_batch(self, subject: str, batch, load_image: ImageLoader | None) -> list[Verdict]:
        from megamind.agent.reader import encode_jpeg

        zone = _zone(self.tz)
        content: list[dict] = [{"type": "text", "text": f"Subject: {subject}\nCandidates follow ({len(batch)})."}]
        for lab, t in batch:
            parts = [f"caption: {t.caption.strip() or '(none)'}"]
            if t.ocr_text.strip():
                parts.append(f"text: {t.ocr_text.strip()[:300]}")
            data = await load_image(t) if load_image is not None else None
            block = None
            if data:
                try:
                    uri = encode_jpeg(data, self.cfg.verify_image_px)
                    block = {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": uri.split(",", 1)[1]},
                    }
                except (OSError, ValueError):
                    block = None
            when = datetime.fromtimestamp(t.ts, zone).strftime("%a %Y-%m-%d %H:%M")
            frame = "1 frame" if block else "no frame"
            content.append({"type": "text", "text": f"[{lab}] {when} ({t.kind}, {frame}): " + " | ".join(parts)[:700]})
            if block:
                content.append(block)
        content.append({"type": "text", "text": "Return one verdict per candidate label."})
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=VERIFY_SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": VERIFY_SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
        text = next(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            return [Verdict.model_validate(v) for v in json.loads(text)["verdicts"]]
        except (ValueError, KeyError, TypeError, ValidationError) as e:
            raise ValueError(f"bad verifier output: {e}") from None


# ---- pick ----


@dataclass(frozen=True)
class Pick:
    token: MemoryToken
    title: str  # verifier title ("" when unverified)
    score: float  # confidence (verified) or a rank-derived score (unverified)
    verified: bool


def keep_matches(cands: list[MemoryToken], verdicts: list[Verdict | None], min_confidence: float) -> list[Pick]:
    """Verified matches (confidence >= threshold) + unchecked candidates; best-first rank order is kept."""
    out = []
    n = len(cands)
    for i, (t, v) in enumerate(zip(cands, verdicts, strict=True)):
        if v is None:
            out.append(Pick(t, "", 0.5 * (n - i) / max(n, 1), False))  # unverified: retrieval rank decides
        elif v.match and v.confidence >= min_confidence:
            out.append(Pick(t, v.title.strip(), 1.0 + v.confidence, True))  # verified beats unverified in a moment
    return out


def pick_moments(picks: list[Pick], max_clips: int, gap_s: float) -> tuple[list[Pick], int]:
    """(one pick per moment, chronological, at most `max_clips` evenly spread in time; number of moments)."""
    from megamind.agent.montage import evenly

    groups: list[list[Pick]] = []
    last: Pick | None = None
    for p in sorted(picks, key=lambda p: (p.token.device_id, p.token.ts)):
        if last is None or p.token.device_id != last.token.device_id or p.token.ts - last.token.ts > gap_s:
            groups.append([])
        groups[-1].append(p)
        last = p
    best = sorted((max(g, key=lambda p: p.score) for g in groups), key=lambda p: p.token.ts)
    return evenly(best, max(1, max_clips)), len(groups)


_SECOND_PERSON = [(r"\bI was\b", "you were"), (r"\bI am\b", "you are"), (r"\bI'm\b", "you're"),
                  (r"\bI\b", "you"), (r"\bmy\b", "your"), (r"\bme\b", "you"), (r"\bmine\b", "yours"),
                  (r"\bmyself\b", "yourself")]  # fmt: skip


def spoken(phrase: str) -> str:
    """The subject as the assistant says it back: "I was at the cafe" -> "you were at the cafe"."""
    for pat, rep in _SECOND_PERSON:
        phrase = re.sub(pat, rep, phrase, flags=re.I)
    return phrase


def narration(subject: str, n_moments: int, n_clips: int, days: int, verified: bool) -> str:
    subject = spoken(subject)
    if not verified:
        return (
            f"Here are {n_clips} likely moment{'s' if n_clips != 1 else ''} matching {subject} (not visually checked)."
        )
    text = f"Found {n_moments} moment{'s' if n_moments != 1 else ''} matching {subject}"
    if days > 1:
        text += f" across {days} days"
    return text + ("." if n_clips == n_moments else f"; here are {n_clips} of them, spread across that time.")


def no_match_message(subject: str, n_candidates: int, ranged: bool) -> str:
    where = " in that period" if ranged else " in your memories"
    checked = f" (checked {n_candidates} candidate moment{'s' if n_candidates != 1 else ''})" if n_candidates else ""
    return f"I couldn't find any moments matching {spoken(subject)}{where}{checked}, so there's no montage to make."
