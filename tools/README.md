# tools/

| script | what it does |
|---|---|
| `feeder.py` | Acts as the headset: pushes webcam or video frames to `POST /v1/frames` |
| `ask.py` | CLI chat client for `POST /v1/chat` |
| `make_example_day.py` | Builds a realistic head-cam day (CC photos + TTS dialogue) and its gold questions |
| `eval_example_day.py` | Asks a running server the example day's questions and scores answers and citations |

Install the server first: `cd server && pip install -e ".[dev,tools]"`. Run every command from the repo root.

## Example day (realistic eval data)

The offline evals (`server/tools/eval_{retrieval,agent}.py`) run on hand-written captions. The example day
runs the real pipeline instead: real photos go through the feeder, filter, captioner and agent.

It is 11 scenes between 07:42 and 19:36. Each scene is a slow, shaky head-cam pan over Creative Commons
photos, with dialogue spoken by macOS `say` voices. The day includes coffee, keys and a wallet on a bedroom
rug, a bus stop, meeting Priya (who leads the design team), a whiteboard plan, ramen with Marcus, a second
coffee, the wallet going into a desk drawer, and a car park "Level 3" sign that nobody mentions. It ends
with keys on the hallway table and a relaxing evening. The people are fictional, and the photo credits are
written to `CREDITS.md` next to the output.

```bash
# 1. generate (macOS only: needs `say`, ffmpeg and internet; ~2 min; ~25 MB, never committed)
python tools/make_example_day.py                 # -> server/data/example_day/ (gitignored)
#    --date YYYY-MM-DD (default today)   --tz Area/City (default: this machine's zone; match the server)

# 2. feed it to a running server (uvicorn megamind.main:app --port 8000)
SERVER=http://localhost:8000 sh server/data/example_day/feed.sh

# 3. score it (live server; real answers need ANTHROPIC_API_KEY / READER_API_KEY on the server)
python tools/eval_example_day.py --gold server/data/example_day/example_day.json
python tools/eval_example_day.py --gold server/data/example_day/example_day.json --modality visual
```

The generator writes `scenes/*.mp4`, `scenes/*.srt` (what was said in each clip), `example_day.json`,
`feed.sh`, `scenes.json` and `CREDITS.md`.

- **Timestamps.** `feed.sh` passes `--start <epoch>` (each scene's clock time) and `--srt` (its dialogue)
  to `feeder.py` when the feeder supports those flags. Otherwise every frame is stamped "now" and the
  dialogue is skipped. Questions that depend on the time of day ("this morning", "this afternoon") need
  real timestamps, a matching `--tz`, and a server clock after 19:36 on `--date`.
- **Gold file.** `example_day.json` uses the same schema as `server/tools/data/agent_day.json`. Each
  `stream` entry is one shot, with ground-truth caption, OCR and entities. A few extra keys are ignored
  by the loaders: `at`/`dur_s` (the exact clip window, which the live scorer matches against citation
  timestamps), `said` (the shot's dialogue) and `questions[].modality`:
  - `visual`: answerable from pixels.
  - `speech`: needs transcripts on tokens.
  - `both`: answerable either way.
  - `none`: the agent must decline.
- **Offline.** A copy for 2026-11-08 is committed as `server/tools/data/example_day.json`. The offline
  harnesses run on it as-is:
  `cd server && python tools/eval_agent.py --dataset tools/data/example_day.json` (or `eval_retrieval.py`).
  After editing scenes or questions, regenerate that copy with:
  `python tools/make_example_day.py --gold-only --date 2026-11-08 --tz America/Los_Angeles --gold server/tools/data/example_day.json`.
  `server/tests/test_example_day.py` checks that the copy matches the generator.
