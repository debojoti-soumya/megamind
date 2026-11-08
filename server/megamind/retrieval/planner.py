"""Query planning: temporal expressions -> time window, query type -> fusion weights,
question -> retrieval-friendly text.

Everything here is rule-based and deterministic so it runs offline and in tests. An
optional async `rewriter` hook lets an LLM (Claude) replace the rewrite step later.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

QueryType = Literal["locate_object", "read_text", "activity", "people", "generic"]
PlanMode = Literal["semantic", "visual", "hybrid", "temporal"]

DEFAULT_TZ = "America/Los_Angeles"


@dataclass
class Weights:
    """Relative weights of the ranked lists fused by RRF."""

    text: float = 1.0
    bm25: float = 1.0
    image: float = 0.6  # CLIP-family towers are weak on dense text (research digest)
    recency: float = 0.15  # pull toward newer tokens among near-equal scores
    ocr: float = 0.0  # bonus for tokens that carry OCR text (read-text questions)
    ocr_bm25: float = 0.0  # extra BM25 list over `ocr_text` only (research digest: read-text path)


WEIGHTS_BY_TYPE: dict[str, Weights] = {
    "locate_object": Weights(text=1.0, bm25=1.0, image=0.8, recency=0.3),
    "read_text": Weights(text=0.8, bm25=1.6, image=0.3, recency=0.1, ocr=0.6, ocr_bm25=2.0),
    "activity": Weights(text=1.0, bm25=0.6, image=1.0, recency=0.2),
    "people": Weights(text=1.0, bm25=0.6, image=1.0, recency=0.15),
    "generic": Weights(),
}


TimeFocus = Literal["start", "center", "end"]
# Which backend may answer. "hybrid": nothing content-bearing to judge (pure time window / anchor-only, e.g.
# "what was I doing at 11", "what did I do after lunch") -> the answer is "frames from the window", which a
# relevance model (UCC facets) cannot improve and would only add latency/cost. "ucc": content words exist, so a
# configured UCC backend ranks them (the hybrid stays its fallback).
PlanRoute = Literal["ucc", "hybrid"]


@dataclass
class QueryPlan:
    query: str
    search_text: str
    query_type: QueryType
    time_from: float | None = None
    time_to: float | None = None
    time_hint: str | None = None
    anchor_event: str | None = None
    weights: Weights = field(default_factory=Weights)
    mode: PlanMode = "hybrid"
    explicit_time: bool = False
    time_focus: TimeFocus = "center"  # "after lunch" -> start, "before I left" -> end
    route: PlanRoute = "ucc"

    @property
    def has_time(self) -> bool:
        return self.time_from is not None or self.time_to is not None


# --------------------------------------------------------------------------- temporal

_NUM_WORDS = {
    "a": 1,
    "an": 1,
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
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "couple": 2,
    "couple of": 2,
    "few": 5,
    "half an": 0.5,
    "half a": 0.5,
}
_UNIT_S = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400, "week": 604800}
_NUM = (
    r"(?P<num>\d+(?:\.\d+)?|a few|a couple of|a couple|couple of|half an|half a|an|a|one|two|three|four|five|"
    r"six|seven|eight|nine|ten|fifteen|twenty|thirty|forty|fifty|sixty)"
)
_UNIT = r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)"
_APPROX = r"(?:about|around|roughly|approximately|some|like|maybe)?\s*"

_RE_AGO = re.compile(rf"\b{_APPROX}{_NUM}\s+{_UNIT}\s+(?:ago|back|earlier)\b", re.I)
_RE_LAST_N = re.compile(rf"\b(?:in|within|over|during)?\s*the\s+(?:last|past)\s+{_NUM}?\s*{_UNIT}\b", re.I)
_RE_LAST_UNIT = re.compile(rf"\b(?:in|within|over|during)?\s*the\s+(?:last|past)\s+{_UNIT}\b", re.I)
_RE_CLOCK = re.compile(
    r"\b(?:at|around|about|near|by)\s+(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ampm>am|pm|a\.m\.|p\.m\.|o'clock)?\b",
    re.I,
)
_RE_WEEKDAY = re.compile(
    r"\b(?:last|on|this)?\s*(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I
)
_RE_ANCHOR = re.compile(
    r"\b(?P<rel>before|after|when|while|since|until|right before|right after|just before|just after)\s+"
    r"(?P<event>i\s+(?:\w+\s*){1,6}?)(?=[?.,!]|$|\s+(?:and|but|or)\b)",
    re.I,
)

# Named day parts as (start_hour, end_hour) in local time. `end` may exceed 24 (next day).
_DAY_PARTS: dict[str, tuple[float, float]] = {
    "early morning": (4, 9),
    "this morning": (5, 12),
    "morning": (5, 12),
    "around noon": (11.5, 12.5),
    "at noon": (11.5, 12.5),
    "noon": (11.5, 12.5),
    "midday": (11, 14),
    "lunchtime": (11.5, 14),
    "lunch time": (11.5, 14),
    "around lunch": (11.5, 14),
    "at lunch": (11.5, 14),
    "lunch": (11.5, 14),
    "breakfast": (6, 10),
    "this afternoon": (12, 18),
    "afternoon": (12, 18),
    "this evening": (17, 24),
    "evening": (17, 24),
    "dinner": (17.5, 21),
    "tonight": (17, 24),
    "last night": (19, 28),  # 19:00 yesterday .. 04:00 today, resolved with day_offset=-1
    "night": (19, 28),
}
_RE_YESTERDAY = re.compile(r"\byesterday\b", re.I)
_RE_TODAY = re.compile(r"\b(?:today|earlier today)\b", re.I)
_RE_EARLIER = re.compile(r"\bearlier\b", re.I)
_RE_RECENT = re.compile(r"\b(?:just now|a moment ago|moments ago|a second ago|right now|recently|lately)\b", re.I)
_RE_DAYPART = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in _DAY_PARTS), key=len, reverse=True)) + r")\b", re.I
)
_RE_LAST_WEEK = re.compile(r"\b(?:last|past|this) week\b", re.I)
_RE_REL_AFTER = re.compile(r"\b(?:right |just )?(?:after|since|following)\s+$", re.I)
_RE_REL_BEFORE = re.compile(r"\b(?:right |just )?(?:before|until|till|prior to)\s+$", re.I)
_RE_FOCUS_START = re.compile(r"\b(?:after|since|following)\b", re.I)
_RE_FOCUS_END = re.compile(r"\b(?:before|until|till|prior to)\b", re.I)
_RE_MEAL = re.compile(r"\b(breakfast|lunch|dinner)\b", re.I)
_RE_TIME_FILLER = re.compile(r"\b(?:at|around|about|on|in|during|sometime|some time)\s*$", re.I)


def _num(s: str | None) -> float:
    if not s:
        return 1.0
    s = s.strip().lower()
    if s in _NUM_WORDS:
        return _NUM_WORDS[s]
    s = re.sub(r"^a\s+", "", s)
    if s in _NUM_WORDS:
        return _NUM_WORDS[s]
    try:
        return float(s)
    except ValueError:
        return 1.0


def _unit_seconds(unit: str) -> int:
    u = unit.lower().rstrip("s")
    return _UNIT_S.get(u, 60)


def _local_day(now: datetime, offset_days: int = 0) -> datetime:
    return datetime.combine((now + timedelta(days=offset_days)).date(), time(0), tzinfo=now.tzinfo)


def _hours(day: datetime, h: float) -> datetime:
    return day + timedelta(hours=h)


@dataclass
class TimeWindow:
    time_from: float | None
    time_to: float | None
    hint: str
    span: tuple[int, int]  # char span in the query that expressed the time


def parse_time(query: str, now: datetime) -> TimeWindow | None:
    """Resolve a relative temporal expression in `query` to a [from, to] window.

    `now` must be timezone-aware in the wearer's timezone. Returns None if no temporal
    expression is found. The upper bound is clamped to `now`.
    """
    win = _parse_time(query, now)
    if win is not None:
        win.hint = win.hint.strip()
    return win


def _parse_time(q: str, now: datetime) -> TimeWindow | None:  # noqa: C901 - a long but flat rule table
    now_ts = now.timestamp()

    m = _RE_AGO.search(q)
    if m:
        n, unit = _num(m.group("num")), _unit_seconds(m.group("unit"))
        offset = n * unit
        center = now_ts - offset
        if unit >= 86400:
            day = _local_day(now, -int(round(n * unit / 86400)))
            return TimeWindow(day.timestamp(), (day + timedelta(days=1)).timestamp(), m.group(0), m.span())
        half = max(0.5 * offset, 120.0) if unit < 3600 else max(0.35 * offset, 20 * 60)
        return TimeWindow(center - half, min(now_ts, center + half), m.group(0), m.span())

    m = _RE_LAST_N.search(q) or _RE_LAST_UNIT.search(q)
    if m:
        n = _num(m.groupdict().get("num")) if m.groupdict().get("num") else 1.0
        offset = n * _unit_seconds(m.group("unit"))
        return TimeWindow(now_ts - offset, now_ts, m.group(0), m.span())

    m = _RE_RECENT.search(q)
    if m:
        word = m.group(0).lower()
        offset = 30 * 60 if word in ("recently", "lately") else 10 * 60
        return TimeWindow(now_ts - offset, now_ts, m.group(0), m.span())

    day_offset = 0
    hint_parts: list[tuple[int, int]] = []
    my = _RE_YESTERDAY.search(q)
    if my:
        day_offset = -1
        hint_parts.append(my.span())
    mw = _RE_WEEKDAY.search(q)
    if mw and not my:
        target = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].index(
            mw.group("day").lower()
        )
        back = (now.weekday() - target) % 7
        day_offset = -back if back else (0 if "this" in mw.group(0).lower() else -7)
        hint_parts.append(mw.span())

    md = _RE_DAYPART.search(q)
    mc = _RE_CLOCK.search(q)
    if mc and not (mc.group("ampm") is None and mc.group("m") is None and int(mc.group("h")) > 12):
        h = int(mc.group("h"))
        mm = int(mc.group("m") or 0)
        ampm = (mc.group("ampm") or "").lower().replace(".", "")
        if ampm == "pm" and h < 12:
            h += 12
        elif ampm == "am" and h == 12:
            h = 0
        elif not ampm and h <= 12:
            # no am/pm: pick the most recent occurrence that is not in the future
            cand = _hours(_local_day(now, day_offset), h + mm / 60)
            if cand > now and day_offset == 0:
                h = h + 12 if h < 12 and _hours(_local_day(now), h + 12 + mm / 60) <= now else h
                if _hours(_local_day(now, day_offset), h + mm / 60) > now:
                    day_offset -= 1
        center = _hours(_local_day(now, day_offset), h + mm / 60)
        half = timedelta(minutes=45)
        spans = hint_parts + [mc.span()]
        lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
        return TimeWindow((center - half).timestamp(), min(now_ts, (center + half).timestamp()), q[lo:hi], (lo, hi))

    if md:
        start_h, end_h = _DAY_PARTS[md.group(1).lower()]
        part = md.group(1).lower()
        if part in ("last night", "night") and not my:
            day_offset = -1
        day = _local_day(now, day_offset)
        lo_dt, hi_dt = _hours(day, start_h), _hours(day, end_h)
        if day_offset == 0 and lo_dt > now:
            # e.g. "this evening" asked at 10 am -> most likely yesterday evening
            day = _local_day(now, -1)
            lo_dt, hi_dt = _hours(day, start_h), _hours(day, end_h)
        spans = hint_parts + [md.span()]
        lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
        mid = lo_dt + (hi_dt - lo_dt) / 2
        rel = _RE_REL_BEFORE.search(q[:lo]) or _RE_REL_AFTER.search(q[:lo])
        if rel:
            lo = rel.start()
            if _RE_REL_AFTER.match(rel.group(0)):
                lo_dt, hi_dt = mid, now
            else:
                lo_dt, hi_dt = day, mid
        return TimeWindow(lo_dt.timestamp(), min(now_ts, hi_dt.timestamp()), q[lo:hi], (lo, hi))

    if my or mw:
        day = _local_day(now, day_offset)
        lo, hi = min(s[0] for s in hint_parts), max(s[1] for s in hint_parts)
        return TimeWindow(day.timestamp(), min(now_ts, (day + timedelta(days=1)).timestamp()), q[lo:hi], (lo, hi))

    m = _RE_TODAY.search(q) or _RE_EARLIER.search(q)
    if m:
        day = _local_day(now)
        return TimeWindow(day.timestamp(), now_ts, m.group(0), m.span())

    m = _RE_LAST_WEEK.search(q)
    if m:
        return TimeWindow(now_ts - 7 * 86400, now_ts, m.group(0), m.span())

    return None


def parse_anchor(query: str) -> str | None:
    """'before I left the office' -> 'left the office'. Unresolved here: the retriever may
    run a second search for the anchor event and use its timestamp as a bound."""
    m = _RE_ANCHOR.search(query)
    if not m:
        return None
    event = re.sub(r"^i\s+", "", m.group("event").strip(), flags=re.I)
    return event.strip() or None


# --------------------------------------------------------------------------- query type

_TYPE_PATTERNS: list[tuple[QueryType, re.Pattern[str]]] = [
    (
        "locate_object",
        re.compile(
            r"\b(where (?:did|do|have|is|are|was|were|'s|s)?\s*(?:i|my|the)?\s*(?:leave|left|put|place|placed|"
            r"set|drop|dropped|park|parked|lose|lost|last see|last saw|see|saw|keep|kept)?|where(?:'s| is| are| was)|"
            r"left my|leave my|put my|lost my|find my|misplaced|last seen|location of|"
            r"which (?:room|shelf|drawer|table))\b",
            re.I,
        ),
    ),
    (
        "read_text",
        re.compile(
            r"\b(what (?:did|does|do) (?:the |that |it |they )?\w*\s*(?:say|read|show)|say|said|read|written|"
            r"text|sign|label|price|cost|how much|wifi|wi-fi|password|passcode|pin|code|number|phone|"
            r"address|plate|menu|title|headline|receipt|ticket|room number|flight|gate|time (?:did|does) the|"
            r"spelled|spelt|name of the|brand|serial|url|website|email|note said|sticky)\b",
            re.I,
        ),
    ),
    (
        "people",
        re.compile(
            r"\b(who|whom|person|people|someone|somebody|guy|woman|man|lady|kid|colleague|friend|"
            r"meet|met|meeting with|talk(?:ed|ing)? (?:to|with)|spoke (?:to|with)|conversation|wearing|face)\b",
            re.I,
        ),
    ),
    (
        "activity",
        re.compile(
            r"\b(what (?:was|were|am|have) i (?:doing|up to|working on|looking at|watching|eating|reading|holding)|"
            r"what did i (?:do|eat|see|watch|read|work on|cook|buy|look at)|what happened|what was going on|"
            r"what was (?:i|that|there)|summar(?:y|ize|ise)|recap|how did i|did i|was i|have i|where was i|"
            r"where have i been|what have i)\b",
            re.I,
        ),
    ),
]


def detect_type(query: str) -> QueryType:
    q = query.lower()
    for qtype, pat in _TYPE_PATTERNS:
        if pat.search(q):
            return qtype
    return "generic"


# --------------------------------------------------------------------------- rewrite

_FILLER = re.compile(
    r"\b(?:hey|hi|ok|okay|so|um|uh|please|can you|could you|would you|do you|did you|will you|"
    r"tell me|show me|remind me|help me|let me know|i want to know|i need to know|i wonder|i think|"
    r"do you remember|do you recall|remember|recall|try to|happen to|by any chance|"
    r"megamind|assistant)\b",
    re.I,
)
_SCAFFOLD = re.compile(
    r"\b(?:what|where|when|who|which|how|whom|whose|why|is|are|was|were|am|be|been|being|do|does|did|"
    r"have|has|had|the|a|an|my|your|our|their|his|her|its|i|me|we|you|they|them|it|this|that|these|those|"
    r"there|here|of|to|in|on|at|for|with|from|by|into|onto|about|and|or|but|if|then|than|"
    r"last|see|saw|seen|look|looked|looking|leave|left|put|place|placed|set|find|lose|lost|keep|kept|"
    r"go|went|get|got|say|said|says|read|show|showed|written|write|doing|done|up|down|earlier|again|"
    r"some|any|something|anything|thing|things|stuff|exactly|really|just|actually|like|kind|sort|"
    r"before|after|since|until|till|while|during|following|prior|"
    r"take|took|taken|happen|happened|eat|ate|eaten|meet|met|talk|talked|hand|handed|"
    r"s|t|m|re|ve|ll|d|again|ago|around|about)\b",
    re.I,
)
_PUNCT = re.compile(r"[^\w\s$%#@&'./:-]")
_MULTISPACE = re.compile(r"\s+")
_POSSESSIVE = re.compile(r"\b(\w+)'s\b", re.I)


def rewrite(query: str, time_span: tuple[int, int] | None = None, anchor: str | None = None) -> str:
    """Strip the temporal phrase, question scaffolding and filler; keep content words.

    Returns '' when nothing content-bearing is left (e.g. 'what was I doing 10 minutes ago'),
    which the retriever treats as a pure temporal query.
    """
    q = query
    if time_span:
        q = q[: time_span[0]] + " " + q[time_span[1] :]
        q = _RE_TIME_FILLER.sub("", q)
    if anchor:
        q = re.sub(
            r"\b(?:right |just )?(?:before|after|when|while|since|until)\s+i\s+" + re.escape(anchor), " ", q, flags=re.I
        )
    q = _POSSESSIVE.sub(r"\1", q)
    q = _PUNCT.sub(" ", q)
    q = _FILLER.sub(" ", q)
    q = _SCAFFOLD.sub(" ", q)
    q = _MULTISPACE.sub(" ", q).strip(" .,?!'").lower()
    words = q.split()
    extra = [e for w in words for e in _EXPAND.get(w, "").split() if e not in words]
    return " ".join(words + extra)


# Query words -> what a caption/OCR is likely to say instead (kept tiny and obvious on purpose).
_EXPAND: dict[str, str] = {
    "park": "parking garage car",
    "parked": "parking garage car",
    "car": "parking",
    "lunch": "food plate restaurant menu",
    "dinner": "food plate restaurant menu",
    "breakfast": "food plate kitchen",
    "phone": "phone screen",
    "glasses": "reading glasses",
    "spectacles": "glasses",
    "colleague": "person",
    "someone": "person",
    "person": "people",
    "people": "person",
}


Rewriter = Callable[[str], Awaitable[str]]


def _mode_for(qtype: QueryType, search_text: str, has_time: bool) -> PlanMode:
    if not search_text and has_time:
        return "temporal"
    if qtype == "read_text":
        return "semantic"
    return "hybrid"


def plan_query(
    query: str,
    *,
    now: datetime | None = None,
    tz: str | ZoneInfo | None = None,
    time_from: float | None = None,
    time_to: float | None = None,
) -> QueryPlan:
    """Rule-based plan. Explicit `time_from`/`time_to` (e.g. from Claude's tool call) win over
    anything parsed from the text, but the temporal phrase is still removed from `search_text`."""
    zone = ZoneInfo(tz) if isinstance(tz, str) else (tz or ZoneInfo(DEFAULT_TZ))
    now_local = (now or datetime.now(UTC)).astimezone(zone)

    win = parse_time(query, now_local)
    anchor = parse_anchor(query)
    if anchor and win:
        anchor = re.sub(re.escape(win.hint), " ", anchor, flags=re.I).strip() or None
    qtype = detect_type(query)
    text = rewrite(query, win.span if win else None, anchor)
    if qtype == "people" and not any(w in text.split() for w in ("person", "people")):
        text = f"{text} person".strip()
    focus: TimeFocus = "start" if _RE_FOCUS_START.search(query) else "end" if _RE_FOCUS_END.search(query) else "center"
    meal = _RE_MEAL.search(win.hint) if win and not text and focus == "center" else None
    if meal:  # "what did I have for lunch": the meal is the subject, not only the time -> search for food
        text = rewrite(meal.group(1))

    explicit = time_from is not None or time_to is not None
    if explicit:
        tf, tt, hint = time_from, time_to, None
    elif win:
        tf, tt, hint = win.time_from, win.time_to, win.hint
    else:
        tf = tt = hint = None
    has_time = tf is not None or tt is not None
    mode = _mode_for(qtype, text, has_time)

    return QueryPlan(
        query=query,
        search_text=text,
        query_type=qtype,
        time_from=tf,
        time_to=tt,
        time_hint=hint,
        anchor_event=anchor,
        weights=WEIGHTS_BY_TYPE[qtype],
        mode=mode,
        explicit_time=explicit,
        time_focus=focus,
        route="hybrid" if mode == "temporal" else "ucc",
    )


async def plan_query_async(query: str, *, rewriter: Rewriter | None = None, **kwargs) -> QueryPlan:
    """Same as `plan_query`, with an optional LLM rewrite hook applied to `search_text`.
    Falls back to the rule-based rewrite if the hook fails or returns nothing."""
    plan = plan_query(query, **kwargs)
    if rewriter is not None:
        try:
            out = (await rewriter(query)).strip()
        except Exception:  # noqa: BLE001 - the hook is best-effort
            out = ""
        if out:
            plan.search_text = out
            plan.mode = _mode_for(plan.query_type, out, plan.has_time)
            plan.route = "hybrid" if plan.mode == "temporal" else "ucc"
    return plan
