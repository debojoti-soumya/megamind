#!/usr/bin/env python3
"""Generate tools/data/long_week.json: a synthetic 7-day egocentric week (~2 memories / minute while awake, plus
conversation transcripts) with hard long-horizon questions: cross-day lookups, counts, before/after, what changed,
latest-of-many and not-found. Deterministic (seeded); same schema as agent_day.json plus per-item `day` / `kind`.

    python tools/make_long_week.py            # writes tools/data/long_week.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "long_week.json"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
START = "2026-11-02"  # a Monday

FILLER = {
    "home": [
        "Living room with a grey sofa and a coffee table.",
        "Hallway with shoes by the door.",
        "Bathroom mirror and sink.",
        "Bedroom with an unmade bed.",
        "Kitchen with dishes in the sink.",
        "Phone screen showing a news app.",
        "Window view of the street, overcast.",
    ],
    "office": [
        "Desk with a laptop showing a code editor.",
        "Open-plan office with people at desks.",
        "Laptop screen with a Slack channel.",
        "Coffee machine in the office kitchenette.",
        "Monitor showing a dashboard with graphs.",
        "Office corridor with glass meeting rooms.",
    ],
    "street": [
        "Walking on a sidewalk past parked cars.",
        "Crosswalk at a busy intersection.",
        "Storefronts along a city street.",
        "Inside a bus, passengers seated.",
    ],
    "gym": ["Gym floor with treadmills.", "Dumbbell rack at the gym.", "Locker room benches."],
    "store": ["Grocery aisle with cereal boxes.", "Produce section with apples and bananas.", "Checkout lane."],
    "cafe": ["Cafe counter with pastries.", "Table at a cafe with a latte."],
}

BREAKFAST = [
    "oatmeal with blueberries",
    "scrambled eggs on toast",
    "pancakes with maple syrup",
    "a bagel with cream cheese",
    "greek yogurt with granola",
    "avocado toast",
    "a smoothie bowl",
]
LUNCH = [
    ("Sweetgreen", "Marcus"),
    ("Taqueria El Sol", "Priya"),
    ("Pho Saigon", "Marcus"),
    ("Taqueria El Sol", "Jordan"),
    ("Sushi Ko", "Priya"),
    (None, None),
    (None, None),
]
DINNER = [
    "spaghetti bolognese",
    "chicken curry",
    "a mushroom risotto",
    "salmon with rice",
    "homemade pizza",
    "takeout ramen",
    "a lentil soup",
]
TV = ["The Bear", "Severance", "Shogun", "Slow Horses", "Severance", "Planet Earth", "The Bear"]
KEYS = [
    "the ceramic bowl in the hallway",
    "the kitchen counter next to the fruit bowl",
    "the pocket of the blue jacket",
    "the bedside table",
    "the ceramic bowl in the hallway",
    "the top of the shoe rack",
    "the desk drawer in the study",
]
BUS = ["Route 22", "Route 22", "Route 14", "Route 22", "Route 38", None, None]
PARK = [None, None, None, None, None, "Level 2 Zone C", "Level 4 Zone A"]  # weekend driving
GYM = {0, 2, 4, 5}  # Mon, Wed, Fri, Sat -> 4 times
GROCERY = {
    1: ("Safeway", "$31.80", "oat milk 4.99, bread 3.49, eggs 5.29"),
    3: ("Trader Joe's", "$47.12", "oat milk 3.99, salmon 12.99, rice 4.49"),
    6: ("Whole Foods", "$22.65", "bananas 1.99, coffee beans 14.99"),
}
BOARD = [
    "Sprint 15: ingest, latency",
    "Sprint 15: ingest, latency, demo prep",
    "Sprint 15: latency, demo prep, QA",
    "Sprint 15: demo prep, QA, launch Dec 3",
    "Sprint 15: QA, launch Dec 3, retro Friday",
    None,
    None,
]


def build(seed: int = 7) -> dict:
    rng = random.Random(seed)
    stream: list[dict] = []
    gold: dict[str, list[str]] = {}
    n = 0

    def add(
        day: int,
        hhmm: str,
        caption: str,
        *,
        ocr: str = "",
        ents=(),
        episode: str = "",
        kind: str = "frame",
        tag: str | None = None,
    ) -> str:
        nonlocal n
        n += 1
        tid = f"d{day}_{n:05d}"
        e = {
            "id": tid,
            "day": day,
            "time": hhmm,
            "caption": caption,
            "ocr": ocr,
            "entities": list(ents),
            "episode": f"d{day}-{episode}",
        }
        if kind != "frame":
            e["kind"] = kind
        stream.append(e)
        if tag:
            gold.setdefault(tag, []).append(tid)
        return tid

    def fill(day: int, start: int, end: int, place: str, episode: str) -> None:
        for t in range(start, end):
            for _ in range(rng.choice((1, 2, 2, 3))):  # ~2 memories / minute: the headset's 20/min cap, filtered
                add(day, f"{t // 60:02d}:{t % 60:02d}", rng.choice(FILLER[place]), episode=episode)

    for d in range(7):
        wd = d < 5
        hm = lambda m: f"{m // 60:02d}:{m % 60:02d}"  # noqa: E731
        # morning at home
        fill(d, 7 * 60, 7 * 60 + 25, "home", "home-am")
        add(
            d,
            "07:26",
            f"Breakfast plate with {BREAKFAST[d]} on the kitchen table.",
            ents=["breakfast"],
            episode="home-am",
            tag=f"breakfast{d}",
        )
        if d in (0, 2, 4, 6):
            add(
                d,
                "07:31",
                "Weekly pill organizer on the counter, one compartment open.",
                ocr=["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][d],
                ents=["pills"],
                episode="home-am",
                tag=f"pills{d}",
            )
        if d == 1:
            add(
                d,
                "07:40",
                "Opened envelope on the table: electricity bill.",
                ocr="PG&E  Amount due $84.30  Due Nov 20",
                ents=["bill"],
                episode="home-am",
                tag="bill",
            )
            add(
                d,
                "07:44",
                "Passport on the bookshelf in the study, next to a plant.",
                ents=["passport"],
                episode="home-am",
                tag="passport",
            )
        fill(d, 7 * 60 + 45, 8 * 60 + 15, "home", "home-am")
        if wd:
            add(
                d,
                "08:20",
                f"Bus stop sign showing {BUS[d]} downtown.",
                ocr=f"{BUS[d]} DOWNTOWN",
                ents=["bus stop"],
                episode="commute",
                tag=f"bus{d}",
            )
            fill(d, 8 * 60 + 21, 9 * 60, "street", "commute")
            arrive = 9 * 60 + rng.randrange(5, 50)
            add(
                d,
                hm(arrive),
                "Office lobby with a reception desk.",
                ocr="VISITORS CHECK IN AT FRONT DESK",
                episode="office-am",
                tag=f"arrive{d}",
            )
            fill(d, arrive + 1, 10 * 60 + 30, "office", "office-am")
            if BOARD[d]:
                add(
                    d,
                    "10:35",
                    "Whiteboard in the meeting room after standup.",
                    ocr=BOARD[d],
                    ents=["whiteboard"],
                    episode="office-am",
                    tag=f"board{d}",
                )
            if d == 3:
                add(
                    d,
                    "10:50",
                    "Conversation in the meeting room.",
                    ocr="",
                    ents=["Marcus"],
                    episode="office-am",
                    kind="transcript",
                    tag="marcus",
                )
                stream[-1]["caption"] = "Marcus: heads up, the launch moved to December 3rd, legal needs another week."
            if d == 4:
                add(
                    d,
                    "10:55",
                    "Yellow sticky note on the monitor.",
                    ocr="TODO: send Priya the slide deck, book flights, call dentist",
                    ents=["sticky note"],
                    episode="office-am",
                    tag="todo",
                )
                add(
                    d,
                    "11:05",
                    "Conversation at Priya's desk.",
                    ents=["Priya"],
                    episode="office-am",
                    kind="transcript",
                    tag="priya",
                )
                stream[-1]["caption"] = (
                    "Priya: can you send me the slide deck before Monday? I need it for the board meeting."
                )
            fill(d, 11 * 60 + 10, 12 * 60 + 25, "office", "office-am")
            place, who = LUNCH[d]
            if place:
                add(
                    d,
                    "12:30",
                    f"Lunch at {place} with {who} across the table.",
                    ocr=place.upper(),
                    ents=[place, who],
                    episode="lunch",
                    tag=f"lunch{d}",
                )
            fill(d, 13 * 60 + 15, 17 * 60 + 30, "office", "office-pm")
            if d == 2:
                add(
                    d,
                    "15:10",
                    "Cafe chalkboard near the register.",
                    ocr="WIFI: bluebottle / pourover2026",
                    ents=["cafe"],
                    episode="office-pm",
                    tag="wifi",
                )
                add(
                    d,
                    "15:12",
                    "Coffee with Elena from the design team at the cafe.",
                    ents=["Elena", "cafe"],
                    episode="office-pm",
                    tag="elena",
                )
            if d == 3:
                add(
                    d,
                    "16:40",
                    "Phone screen with a calendar reminder.",
                    ocr="Dentist - Dr. Kim - Nov 12, 4:00 PM",
                    ents=["reminder"],
                    episode="office-pm",
                    tag="dentist",
                )
            fill(d, 17 * 60 + 31, 18 * 60, "street", "commute-pm")
        else:
            add(
                d,
                "09:30",
                f"Parking garage pillar sign: {PARK[d]}.",
                ocr=PARK[d].upper(),
                ents=["parking"],
                episode="errands",
                tag=f"park{d}",
            )
            fill(d, 9 * 60 + 31, 10 * 60 + 30, "street", "errands")
            if d == 6:
                add(
                    d,
                    "10:40",
                    "Reading a paperback on the sofa: 'Project Hail Mary'.",
                    ocr="PROJECT HAIL MARY  ANDY WEIR",
                    ents=["book"],
                    episode="home-sun",
                    tag="book",
                )
            fill(d, 11 * 60, 17 * 60, "home", "home-day")
        if d in GYM:
            g0 = 18 * 60 + 5
            add(
                d,
                hm(g0),
                "Gym entrance with a membership scanner.",
                ocr="FITLAB  WELCOME",
                ents=["gym"],
                episode="gym",
                tag=f"gym{d}",
            )
            fill(d, g0 + 1, g0 + 50, "gym", "gym")
            if d == 4:
                add(
                    d,
                    hm(g0 + 55),
                    "Smoothie bar right outside the gym, a green smoothie on the counter.",
                    ents=["smoothie"],
                    episode="gym",
                    tag="aftergym",
                )
        if d in GROCERY:
            store, total, items = GROCERY[d]
            fill(d, 19 * 60, 19 * 60 + 15, "store", "grocery")
            add(
                d,
                "19:16",
                f"Receipt from {store} at checkout.",
                ocr=f"{store.upper()}  {items}  TOTAL {total}",
                ents=["receipt", store],
                episode="grocery",
                tag=f"receipt{d}",
            )
        if d == 3:
            add(
                d,
                "19:40",
                "Package delivered, placed in the hall closet on the top shelf.",
                ocr="AMAZON",
                ents=["package"],
                episode="home-pm",
                tag="package",
            )
        add(d, "20:00", f"Dinner: {DINNER[d]} on the table.", ents=["dinner"], episode="home-pm", tag=f"dinner{d}")
        if d == 6:
            add(
                d,
                "19:45",
                "Stirring a pot of lentil soup on the stove.",
                ents=["cooking"],
                episode="home-pm",
                tag="cook6",
            )
        fill(d, 20 * 60 + 5, 21 * 60, "home", "home-pm")
        add(
            d,
            "21:05",
            f"TV showing {TV[d]} title card.",
            ocr=TV[d].upper(),
            ents=["TV"],
            episode="home-pm",
            tag=f"tv{d}",
        )
        fill(d, 21 * 60 + 10, 22 * 60 + 30, "home", "home-pm")
        add(d, "22:35", f"Keys on {KEYS[d]}.", ents=["keys"], episode="home-pm", tag=f"keys{d}")
        if d == 6:
            break  # "now" is Sunday 20:30: nothing after it
    stream = [e for e in stream if not (e["day"] == 6 and e["time"] > "20:30")]
    ids = {e["id"] for e in stream}
    g = lambda *tags: [t for tag in tags for t in gold.get(tag, []) if t in ids]  # noqa: E731

    Q = [
        ("what did I have for breakfast on Tuesday", g("breakfast1"), ["eggs"]),
        ("how many times did I go to the gym this week", g("gym0", "gym2", "gym4", "gym5"), ["four", "4"]),
        ("where did I park yesterday", g("park5"), ["level 2", "zone c"]),
        ("what did the whiteboard say on Monday", g("board0"), ["ingest"]),
        ("what changed on the whiteboard between Monday and Friday", g("board0", "board4"), ["launch", "retro", "qa"]),
        ("who did I have lunch with on Wednesday", g("lunch2"), ["marcus"]),
        ("where did I leave my keys on Thursday night", g("keys3"), ["bedside"]),
        ("what was the total at Trader Joe's", g("receipt3"), ["47.12"]),
        ("which store did I buy oat milk at most recently", g("receipt3"), ["trader joe"]),
        ("what did Marcus say about the launch date", g("marcus"), ["december 3", "dec 3"]),
        ("what TV show did I watch on Friday night", g("tv4"), ["severance"]),
        ("what did I do right after the gym on Friday", g("aftergym"), ["smoothie"]),
        ("what did I eat for dinner the day before yesterday", g("dinner4"), ["pizza"]),
        ("which bus route did I take on Wednesday", g("bus2"), ["14"]),
        ("where did I last see my passport", g("passport"), ["bookshelf", "study"]),
        ("what was the wifi password at the cafe", g("wifi"), ["pourover2026"]),
        ("who did I have coffee with at the cafe", g("elena"), ["elena"]),
        ("how much was the electricity bill", g("bill"), ["84.30", "84.3"]),
        ("what book was I reading this morning", g("book"), ["hail mary"]),
        ("where did I put the package that arrived on Thursday", g("package"), ["closet"]),
        ("when is my dentist appointment", g("dentist"), ["nov 12", "november 12", "12th"]),
        ("what was on my to-do sticky note", g("todo"), ["slide deck", "flights", "dentist"]),
        ("what did Priya ask me to send", g("priya"), ["slide deck", "deck"]),
        ("what restaurant did I go to most for lunch this week", g("lunch1", "lunch3"), ["taqueria", "el sol"]),
        ("did I see a red umbrella this week", [], ["__not_found__"]),
        (
            "which days did I take my pills",
            g("pills0", "pills2", "pills4", "pills6"),
            ["monday", "wednesday", "friday", "sunday"],
        ),
        ("what did I cook tonight", g("cook6"), ["lentil"]),
        (
            "what time did I get to the office on Thursday",
            g("arrive3"),
            [stream[[e["id"] for e in stream].index(gold["arrive3"][0])]["time"].lstrip("0")],
        ),
        ("where were my keys on Monday night", g("keys0"), ["bowl", "hallway"]),
        ("what did I watch on TV the night before last", g("tv4"), ["severance"]),
    ]
    questions = [{"q": q, "gold": gl, "expect": ex} for q, gl, ex in Q]
    return {
        "date": START,
        "tz": "America/Los_Angeles",
        "now": "20:30",
        "now_day": 6,
        "device_id": "eval-device",
        "_comment": "7-day synthetic week (tools/make_long_week.py); items carry `day` (0=Monday), optional `kind`.",
        "stream": stream,
        "questions": questions,
    }


if __name__ == "__main__":
    data = build()
    OUT.write_text(json.dumps(data, indent=0))
    print(f"{OUT}: {len(data['stream'])} memories, {len(data['questions'])} questions")
