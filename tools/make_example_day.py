#!/usr/bin/env python3
# ruff: noqa: E501  (photo credits, scene scripts and ground-truth captions are data tables)
"""Build a realistic head-cam "example day" from Creative Commons photos + TTS dialogue, plus its gold questions.

Ported from Recall's scripts/make_example_day.py. Each scene is a slow, slightly shaky head-cam pan/push-in
over real photos (CC0 / CC BY / CC BY-SA, credited in CREDITS.md next to the output) with spoken dialogue from
macOS `say` (one voice per person) over faint room tone. Eleven scenes span 07:42-19:36 on one date.

    python tools/make_example_day.py                          # -> server/data/example_day/ (gitignored)
    python tools/make_example_day.py --out /tmp/day --date 2026-11-08 --tz America/Los_Angeles
    python tools/make_example_day.py --gold-only --date 2026-11-08 --tz America/Los_Angeles \\
        --gold server/tools/data/example_day.json             # regenerate the committed gold file (no deps)

Output (in --out):
    scenes/<date>_<HH-MM-SS>_<slug>.mp4   one clip per scene, 1280x720 @ 25 fps, 16 kHz mono AAC
    scenes/<date>_<HH-MM-SS>_<slug>.srt   what was said in that clip (times relative to the clip)
    example_day.json                      gold questions, same schema as server/tools/data/agent_day.json
    feed.sh                               pushes every scene through tools/feeder.py at its wall-clock time
    scenes.json, CREDITS.md               scene index and photo credits

Then: `sh <out>/feed.sh` against a running server, and `python tools/eval_example_day.py --gold <out>/example_day.json`.
Requires macOS (`say`), ffmpeg/ffprobe, and internet (downloads 13 photos once into <out>/_photos). Never run in CI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "server" / "data" / "example_day"
DEVICE_ID = "example-day"
UA = {"User-Agent": "MegamindHackathonDemo/0.1 (example footage builder)"}

# key -> (url, title, creator, licence, source page)
PHOTOS = {
    "coffee_mug": ("https://live.staticflickr.com/1893/44381116112_4453cecc5c_b.jpg", "Black coffee in a retro mug", "freestocks.org", "CC0 1.0", "https://www.flickr.com/photos/135396164@N05/44381116112"),
    "keys_wallet": ("https://live.staticflickr.com/4117/4829251012_9a824e4843_b.jpg", "crash landing: wallet, keys", "bagaball", "CC BY 2.0", "https://www.flickr.com/photos/33756523@N05/4829251012"),
    "bus_stop": ("https://upload.wikimedia.org/wikipedia/commons/3/3c/Sandra_Street_47_bus_stop_timetable_sign%2C_Piccadilly_St_Geebung_DSCF5589.jpg", "Sandra Street 47 bus stop timetable sign, Piccadilly St Geebung", "John Robert McPherson", "CC BY-SA 4.0", "https://commons.wikimedia.org/w/index.php?curid=98218828"),
    "bus_schedule": ("https://live.staticflickr.com/2098/2417321223_b3bf0c1777_b.jpg", "Detail of schedule on information sign", "Oran Viriyincy", "CC BY-SA 2.0", "https://www.flickr.com/photos/95482862@N00/2417321223"),
    "office": ("https://live.staticflickr.com/5050/5265471026_03a3ca35fd_b.jpg", "PAX Coworking Jelly #6", "yutaka-f", "CC BY 2.0", "https://www.flickr.com/photos/32235035@N06/5265471026"),
    "whiteboard": ("https://cdn.stocksnap.io/img-thumbs/960w/YR89OQFMT1.jpg", "Whiteboard Post-It notes", "Startup Stock Photos", "CC0 1.0", "https://stocksnap.io/photo/whiteboard-post-itnotes-YR89OQFMT1"),
    "ramen_a": ("https://live.staticflickr.com/3400/4568517633_2f44a5c26f_b.jpg", "bowl of ramen at Laksa Ramen, Sao Paulo", "Paul Keller", "CC BY 2.0", "https://www.flickr.com/photos/18259771@N00/4568517633"),
    "ramen_b": ("https://live.staticflickr.com/436/20126460812_ebb254a225_b.jpg", "Bowl of ramen at Hiro Handmade Ramen", "amg.lincluden", "CC BY-SA 2.0", "https://www.flickr.com/photos/39226993@N06/20126460812"),
    "latte": ("https://live.staticflickr.com/8160/7663486588_80dccd3ed4_b.jpg", "Artistic Coffee from French Press Coffee in Santa Barbara", "MamaOT", "CC BY 2.0", "https://www.flickr.com/photos/80135679@N02/7663486588"),
    "desk": ("https://live.staticflickr.com/3546/3653560187_0fee48000d_b.jpg", "My current desk setup at work", "grggrssmr", "CC BY-SA 2.0", "https://www.flickr.com/photos/24700112@N05/3653560187"),
    "car_park": ("https://upload.wikimedia.org/wikipedia/commons/6/64/Newbury_Kennet_Centre_Multi-Storey_Car_Park_Level_3_-_geograph.org.uk_-_7924541.jpg", "Newbury Kennet Centre Multi-Storey Car Park Level 3", "Oscar Taylor", "CC BY-SA 2.0", "https://commons.wikimedia.org/w/index.php?curid=170394230"),
    "hall_table": ("https://live.staticflickr.com/1350/908738639_7a523461e8_b.jpg", "The Items Maczter Carries v1.0", "maczter", "CC BY 2.0", "https://www.flickr.com/photos/66215925@N00/908738639"),
    "living_room": ("https://live.staticflickr.com/3209/3156625773_1198e36e0c.jpg", "Our old apartment!", "longlostcousin", "CC BY 2.0", "https://www.flickr.com/photos/67469019@N00/3156625773"),
}  # fmt: skip

ME, PRIYA, MARCUS = "Daniel", "Samantha", "Rishi"  # macOS voices; the people are fictional

# Scene: (start HH:MM:SS, slug, episode, shots, lines)
#   shot = (token id, photo, seconds, (focus_x, focus_y), (zoom_start, zoom_end))
#   line = (voice, seconds into the scene, text)
SCENES = [
    ("07:42:10", "kitchen", "home-am", [("kitchen", "coffee_mug", 14, (0.5, 0.5), (1.0, 1.15))],
     [(ME, 2, "Morning. First coffee of the day, I definitely need it.")]),
    ("07:58:30", "bedroom", "home-am", [("bedroom", "keys_wallet", 20, (0.3, 0.5), (1.0, 1.35))],
     [(ME, 3, "Okay, my keys and wallet are on the rug by the bed. Don't forget them.")]),
    ("08:31:05", "bus-stop", "commute-am",
     [("bus-stop", "bus_stop", 12, (0.45, 0.35), (1.0, 1.6)), ("bus-schedule", "bus_schedule", 14, (0.3, 0.45), (1.1, 1.35))],
     [(ME, 15, "Looks like the twenty four to Discovery Park comes in a few minutes.")]),
    ("09:12:40", "office-intro", "office-am", [("office-intro", "office", 26, (0.45, 0.5), (1.0, 1.2))],
     [(PRIYA, 2, "Hi, you must be new here. I'm Priya Shah, I lead the design team."),
      (ME, 9, "Nice to meet you, Priya."),
      (PRIYA, 12, "Grab a seat, we're about to go over the launch plan.")]),
    ("09:27:15", "whiteboard", "office-am", [("whiteboard", "whiteboard", 24, (0.3, 0.4), (1.0, 1.3))],
     [(PRIYA, 2, "So, week three is design, week four we build the admin A P I, and we ship the beta at the end of week seven.")]),
    ("12:36:20", "lunch", "lunch",
     [("lunch-a", "ramen_a", 14, (0.5, 0.5), (1.0, 1.15)), ("lunch-b", "ramen_b", 16, (0.5, 0.45), (1.05, 1.25))],
     [(ME, 2, "I'll have the miso ramen with an extra egg, please."),
      (MARCUS, 8, "Same for me. I'm Marcus, by the way, I'm on the data team."),
      (ME, 15, "Good to meet you, Marcus.")]),
    ("14:18:45", "coffee", "office-pm", [("coffee", "latte", 14, (0.5, 0.5), (1.0, 1.2))],
     [(ME, 2, "Second coffee. A flat white this time.")]),
    ("15:04:10", "desk", "office-pm", [("desk", "desk", 22, (0.5, 0.55), (1.0, 1.15))],
     [(ME, 3, "Back at my desk. I'm putting my wallet in the top drawer so I don't lose it.")]),
    # No dialogue: only the sign says where the car is (the retroactive, visual-only question).
    ("17:52:30", "car-park", "commute-pm", [("car-park", "car_park", 18, (0.39, 0.36), (1.0, 2.6))], []),
    ("18:41:00", "hallway", "home-pm", [("hallway", "hall_table", 20, (0.45, 0.62), (1.0, 1.3))],
     [(ME, 2, "Home. Emptying my pockets onto the hallway table. Keys, phone, watch.")]),
    ("19:35:20", "evening", "home-pm", [("evening", "living_room", 16, (0.5, 0.5), (1.0, 1.12))],
     [(ME, 3, "Finally, time to relax.")]),
]  # fmt: skip

# What an ideal captioner/OCR would put on each shot's memory token (pixels only; speech lives in "said").
TRUTH = {
    "kitchen": ("Close-up of black coffee in a patterned retro mug on a table.", "", ["coffee", "mug"]),
    "bedroom": ("Low view of a bunch of keys and a black leather wallet lying on a cream shag rug.", "", ["keys", "wallet", "rug"]),
    "bus-stop": ("Yellow bus stop pole with a timetable sign on a quiet suburban street.", "PICCADILLY / HAYMARKET\nTRANSLINK\n13 12 30", ["bus stop", "timetable", "sign"]),
    "bus-schedule": ("Close-up of a bus timetable panel listing route 24 to Discovery Park.", "24 to Discovery Park\nvia Seattle Center, Elliott Ave W, Pier 91\nBus leaves 4th Ave S & S Jackson St at:\n402 to Lynnwood Transit Center", ["bus timetable", "route 24"]),
    "office-intro": ("Meeting room: several people seated at a table with laptops open; a whiteboard on the back wall.", "", ["meeting room", "people", "laptop", "whiteboard"]),
    "whiteboard": ("Whiteboard project plan with rows of blue sticky notes, one row per week.", "WEEK 3: Design\nWEEK 4: Design\nWEEK 5: Design\nWEEK 6: Dev\nWEEK 7: Dev\nADMIN API\nInfrastructure Recs: Laravel + Backbone", ["whiteboard", "sticky notes", "project plan"]),
    "lunch-a": ("A bowl of ramen with noodles, a soft egg and toppings on a restaurant table.", "", ["ramen", "bowl", "egg", "restaurant"]),
    "lunch-b": ("A second bowl of ramen across the table in a ramen restaurant.", "", ["ramen", "bowl", "restaurant"]),
    "coffee": ("A latte with leaf-shaped latte art in a cup on a saucer at a cafe.", "", ["latte", "coffee", "cup", "cafe"]),
    "desk": ("Office cubicle desk with a monitor, a laptop on a stand, keyboard, desk phone and a mug; keys and a black pouch at the front edge.", "", ["desk", "monitor", "laptop", "desk phone", "mug", "keys"]),
    "car-park": ("Multi-storey car park lift lobby with brick walls, a lift door and signs reading Level 3.", "Level 3", ["car park", "lift", "sign"]),
    "hallway": ("Pocket contents laid out on a wooden table: car keys with a remote fob, a wallet, a watch, a phone, a notebook, a pen, glasses and a pocket knife.", "", ["keys", "wallet", "watch", "phone", "notebook", "glasses", "table"]),
    "evening": ("Living room with a dark green sofa, an armchair, polished wooden floor and a small Christmas tree.", "", ["living room", "sofa", "armchair", "christmas tree"]),
}  # fmt: skip

# Gold questions. modality: "visual" (answerable from pixels), "speech" (only from what was said: needs
# transcripts on tokens), "both" (either), "none" (must decline). expect = any-of substrings (eval_agent schema).
QUESTIONS = [
    ("Where did I leave my keys?", ["hallway"], ["hallway", "hall table", "wooden table"], "both"),
    ("Where were my keys this morning?", ["bedroom"], ["rug", "carpet", "bedroom", "floor"], "both"),
    ("What's the name of the person who leads the design team?", ["office-intro"], ["Priya"], "speech"),
    ("Who did I have lunch with?", ["lunch-a", "lunch-b"], ["Marcus"], "speech"),
    ("What did I eat for lunch?", ["lunch-a", "lunch-b"], ["ramen"], "both"),
    ("Which bus stop was I at this morning?", ["bus-stop"], ["Piccadilly", "Haymarket"], "visual"),
    ("Which bus was I waiting for?", ["bus-schedule"], ["24", "Discovery Park"], "both"),
    ("When are we shipping the beta?", ["whiteboard"], ["week 7", "week seven", "seventh week"], "both"),
    ("What is week 3 for on the whiteboard plan?", ["whiteboard"], ["design"], "visual"),
    ("How many coffees did I have today?", ["kitchen", "coffee"], ["two", "twice", "2 coffee", "2 cups"], "both"),
    ("What did I drink this afternoon?", ["coffee"], ["flat white", "latte"], "both"),
    ("Where did I put my wallet at work?", ["desk"], ["drawer"], "speech"),
    ("What level did I park on?", ["car-park"], ["level 3", "level three", "third"], "visual"),
    ("What did I do after I got home this evening?", ["hallway", "evening"], ["relax", "sofa", "couch", "living room", "pockets"], "both"),
    ("Where did I put my passport?", [], ["__not_found__"], "none"),
]  # fmt: skip

NOW = "19:40"  # "now" for offline evals: just after the last scene
W, H, FPS = 1280, 720, 25


# --------------------------------------------------------------------------- gold (stdlib only, no I/O)


def _hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _secs(hms: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def shot_table() -> list[dict]:
    """One row per shot (= one gold memory token): absolute start, duration and the lines said during it."""
    rows = []
    for start, slug, episode, shots, lines in SCENES:
        off = 0.0
        for tid, photo, secs, _focus, _zoom in shots:
            said = [text for _v, at, text in lines if off <= at < off + secs]
            rows.append({"id": tid, "scene": slug, "episode": episode, "photo": photo,
                         "at": _hms(_secs(start) + off), "dur_s": secs, "said": said})  # fmt: skip
            off += secs
    return rows


def build_gold(date: str, tz: str) -> dict:
    """The gold file: server/tools/data/agent_day.json schema (date/tz/now/device_id/stream/questions).

    Extra keys the loaders ignore: stream[].at/dur_s (exact clip window, for scoring live citations),
    stream[].said (dialogue, for speech-aware ingest), questions[].modality.
    """
    stream = []
    for r in shot_table():
        caption, ocr, entities = TRUTH[r["id"]]
        stream.append({"id": r["id"], "time": r["at"][:5], "at": r["at"], "dur_s": r["dur_s"],
                       "episode": r["episode"], "caption": caption, "ocr": ocr, "entities": entities,
                       "said": r["said"]})  # fmt: skip
    questions = [{"q": q, "gold": gold, "expect": expect, "modality": mod} for q, gold, expect, mod in QUESTIONS]
    return {
        "date": date,
        "tz": tz,
        "now": NOW,
        "device_id": DEVICE_ID,
        "_comment": "Realistic example day (tools/make_example_day.py): CC photos + TTS dialogue. Same schema as "
        "agent_day.json; stream captions/OCR are ground truth for each shot, 'said' is its dialogue. "
        "questions[].modality: visual | speech (needs transcripts) | both | none (must decline).",
        "stream": stream,
        "questions": questions,
    }


def dump_gold(gold: dict) -> str:
    return json.dumps(gold, indent=1, ensure_ascii=False) + "\n"


def local_tz_name() -> str:
    try:  # macOS/Linux: /etc/localtime -> .../zoneinfo/Area/City
        return str(Path("/etc/localtime").resolve()).split("zoneinfo/", 1)[1]
    except (IndexError, OSError):
        return "America/Los_Angeles"


# --------------------------------------------------------------------------- media (macOS say + ffmpeg)


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def fetch(photos_dir: Path) -> None:
    photos_dir.mkdir(parents=True, exist_ok=True)
    for key, (url, *_rest) in PHOTOS.items():
        f = photos_dir / f"{key}.jpg"
        if f.exists() and f.stat().st_size > 10_000:
            continue
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        if data[:2] != b"\xff\xd8":
            raise RuntimeError(f"{key}: not a JPEG (rate limited?) - retry in a minute")
        f.write_bytes(data)
        print("downloaded", key)


def render_shot(src: Path, secs: float, focus, zoom, out: Path, seed: int) -> None:
    """Head-cam style move over a still: cover-crop to 16:9 around the focus, eased push-in, small sway."""
    n = int(secs * FPS)
    fx, fy = focus
    z0, z1 = zoom
    pre = f"scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160:'(iw-3840)*{fx}':'(ih-2160)*{fy}'"
    ease = f"(1-cos(PI*on/{n}))/2"
    zexpr = f"{z0}+({z1}-{z0})*{ease}"
    sway_x = f"(18*sin(on/{11 + seed % 5})+6*sin(on/3.1+{seed}))"
    sway_y = f"(12*sin(on/{13 + seed % 3}+1)+5*sin(on/2.7+{seed}))"
    xexpr = f"max(0,min(iw-iw/zoom,(iw-iw/zoom)*{fx}+{sway_x}))"
    yexpr = f"max(0,min(ih-ih/zoom,(ih-ih/zoom)*{fy}+{sway_y}))"
    vf = (
        f"{pre},zoompan=z='{zexpr}':x='{xexpr}':y='{yexpr}':d={n}:s={W}x{H}:fps={FPS},"
        f"eq=brightness=0.01*sin(t*0.7):saturation=1.05,noise=alls=4:allf=t,format=yuv420p"
    )
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(src), "-vf", vf, "-frames:v", str(n),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", str(out)])  # fmt: skip


def duration(path: Path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(out.strip() or 0)


def scene_audio(lines, total: float, td: Path, out: Path) -> list[tuple[float, float, str]]:
    """Mix TTS lines over brown-noise room tone; returns subtitle cues (start, end, text) in clip seconds."""
    inputs, filters, labels, cues = [], [], [], []
    for i, (voice, at, text) in enumerate(lines):
        aiff = td / f"l{i}.aiff"
        run(["say", "-v", voice, "-r", "175", "-o", str(aiff), text])
        cues.append((float(at), min(total, at + duration(aiff)), text))
        inputs += ["-i", str(aiff)]
        ms = int(at * 1000)
        filters.append(f"[{i}:a]aresample=16000,aformat=channel_layouts=mono,adelay={ms}|{ms},volume=1.0[s{i}]")
        labels.append(f"[s{i}]")
    k = len(lines)
    inputs += ["-f", "lavfi", "-t", f"{total}", "-i", "anoisesrc=color=brown:amplitude=0.015:sample_rate=16000"]
    filters.append(f"[{k}:a]lowpass=f=900,aformat=channel_layouts=mono[room]")
    filters.append(f"{''.join(labels)}[room]amix=inputs={k + 1}:duration=longest:normalize=0,atrim=0:{total}[a]")
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filters), "-map", "[a]",
         "-ar", "16000", "-ac", "1", str(out)])  # fmt: skip
    return cues


def srt(cues: list[tuple[float, float, str]]) -> str:
    def ts(s: float) -> str:
        ms = int(round(s * 1000))
        return f"{ms // 3_600_000:02d}:{ms % 3_600_000 // 60_000:02d}:{ms % 60_000 // 1000:02d},{ms % 1000:03d}"

    return "".join(f"{i}\n{ts(a)} --> {ts(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(cues, 1))


def make_scene(idx: int, shots, lines, photos: Path, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tdn:
        td = Path(tdn)
        parts = []
        for j, (_tid, photo, secs, focus, zoom) in enumerate(shots):
            p = td / f"v{j}.mp4"
            render_shot(photos / f"{photo}.jpg", secs, focus, zoom, p, seed=idx * 7 + j)
            parts.append(p)
        total = sum(s[2] for s in shots)
        lst = td / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        video, audio = td / "video.mp4", td / "audio.wav"
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video)])
        cues = scene_audio(lines, total, td, audio)
        run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "48k", "-shortest", "-movflags", "+faststart", str(dest)])  # fmt: skip
    dest.with_suffix(".srt").write_text(srt(cues))


def feed_script(date: str, tz: str, clips: list[tuple[str, str]]) -> str:
    """POSIX sh that feeds each clip with its wall-clock start (needs feeder --start/--srt; degrades without)."""
    zone = ZoneInfo(tz)
    day = dt.date.fromisoformat(date)
    lines = [
        "#!/bin/sh",
        "# Generated by tools/make_example_day.py: feed the example day to a running Megamind server.",
        "#   SERVER=http://localhost:8000 FEEDER=tools/feeder.py sh feed.sh",
        "# With a feeder that supports --start/--srt, frames carry the scene's real clock time (and its",
        "# dialogue as FrameMeta.transcript); otherwise they are stamped 'now' and the dialogue is skipped.",
        "set -e",
        'HERE=$(cd "$(dirname "$0")" && pwd)',
        "SERVER=${SERVER:-http://localhost:8000}",
        "FEEDER=${FEEDER:-tools/feeder.py}",
        "PY=${PYTHON:-python}",
        'HELP=$("$PY" "$FEEDER" --help)',
        'case "$HELP" in *--start*) HAS_START=1 ;; *) HAS_START= ; echo "feeder has no --start: using now" >&2 ;; esac',
        'case "$HELP" in *--srt*) HAS_SRT=1 ;; *) HAS_SRT= ;; esac',
        "feed() {",
        f'  set -- "$1" "$2" --server "$SERVER" --device-id {DEVICE_ID} --fps 1 --realtime',
        "  clip=$1; start=$2; shift 2",
        '  [ -n "$HAS_START" ] && set -- "$@" --start "$start"',
        '  [ -n "$HAS_SRT" ] && [ -s "$HERE/scenes/${clip%.mp4}.srt" ] && set -- "$@" --srt "$HERE/scenes/${clip%.mp4}.srt"',
        '  echo "$clip"',
        '  "$PY" "$FEEDER" --source "$HERE/scenes/$clip" "$@"',
        "}",
    ]
    for clip, hms in clips:
        h, m, s = (int(x) for x in hms.split(":"))
        epoch = dt.datetime.combine(day, dt.time(h, m, s), tzinfo=zone).timestamp()
        lines.append(f"feed {shlex.quote(clip)} {epoch:.0f}   # {date} {hms} {tz}")
    return "\n".join(lines) + "\n"


def credits_md() -> str:
    out = [
        "# Example day - photo credits",
        "",
        "Scenes are animated from these Creative Commons photos (crop, pan/zoom, colour and grain added = adapted).",
        "Dialogue is synthetic (macOS text-to-speech); the people named are fictional.",
        "",
    ]
    out += [f"- **{title}** by {creator}, {lic} - {src}" for _url, title, creator, lic, src in PHOTOS.values()]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="day the scenes happen on (YYYY-MM-DD)")
    ap.add_argument("--tz", default=local_tz_name(), help="IANA zone of the scene times (match the server's)")
    ap.add_argument("--gold", type=Path, default=None, help="gold file path (default <out>/example_day.json)")
    ap.add_argument("--gold-only", action="store_true", help="only write the gold file (no network/ffmpeg/say)")
    a = ap.parse_args()
    ZoneInfo(a.tz)  # fail fast on a bad zone
    dt.date.fromisoformat(a.date)

    gold_path = a.gold or a.out / "example_day.json"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(dump_gold(build_gold(a.date, a.tz)))
    print("wrote", gold_path)
    if a.gold_only:
        return

    photos, scenes = a.out / "_photos", a.out / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    fetch(photos)
    clips = []
    for idx, (start, slug, _episode, shots, lines) in enumerate(SCENES):
        name = f"{a.date}_{start.replace(':', '-')}_{slug}.mp4"
        clips.append((name, start))
        dest = scenes / name
        if dest.exists() and dest.with_suffix(".srt").exists():
            print("exists", name)
            continue
        make_scene(idx, shots, lines, photos, dest)
        print("made", name)
    (a.out / "feed.sh").write_text(feed_script(a.date, a.tz, clips))
    (a.out / "CREDITS.md").write_text(credits_md())
    index = [
        {"clip": c, "time": s[0], "scene": s[1], "seconds": sum(x[2] for x in s[3]), "said": [t for _v, _at, t in s[4]]}
        for c, s in zip((c for c, _ in clips), SCENES)
    ]
    (a.out / "scenes.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"done: {len(clips)} scenes in {scenes}\nnext: sh {a.out / 'feed.sh'}")


if __name__ == "__main__":
    main()
