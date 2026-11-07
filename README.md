# Megamind

Egocentric memory for Meta Quest 3. The headset's passthrough cameras are sampled
continuously; keyframes are filtered, described, embedded and stored as **memory
tokens**. A Claude agent exposes a `recall_memory` tool that retrieves tokens and
hands them to a Qwen-VL **reader** sub-agent, which attends over them and returns
grounded evidence. The wearer talks to the agent through a voice-first AR panel.

Read [`docs/megamind-design.pdf`](docs/megamind-design.pdf) first. The API contract
in [`docs/api.md`](docs/api.md) is the only interface between headset and server.

## Layout

```
docs/      design doc (LaTeX + PDF), API contract
server/    FastAPI backend: ingest, memory store, agent (Python 3.11)
tools/     feeder.py (webcam/video -> /v1/frames), ask.py (CLI chat)
headset/   Unity project for Quest 3 (capture + AR chat)
```

## Quick start (server)

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # fill in keys
uvicorn megamind.main:app --reload --port 8000
curl localhost:8000/v1/health
```

Feed it frames from a laptop webcam (no headset needed):

```bash
python tools/feeder.py --server http://localhost:8000 --source 0 --fps 1
python tools/ask.py --server http://localhost:8000 "what did I just see?"
```

Keyless local stack (BGE text + SigLIP image embeddings, Tesseract OCR, zero-shot captions):
`EMBED_PROVIDER=local IMAGE_EMBED_PROVIDER=siglip ENRICH_PROVIDER=local`. Point `ENRICH_BASE_URL`
at a Qwen-VL / OpenAI-compatible server for real captions + OCR.

Gold-question eval on a synthetic first-person "day" (9 indoor scenes, signs with exact text):

```bash
python tools/make_day.py --scenes <dir of scene_n.jpg> --out /tmp/day
START=$(($(date +%s) - 3600))
python tools/feeder.py --source /tmp/day/day.mp4 --fps 1 --realtime --start $START
python tools/eval.py --gold /tmp/day/gold.json --start $START   # hit@1/hit@5, episode purity, stats
curl -XPOST "localhost:8000/v1/admin/age?hot_older_than_s=600"  # force hot JPEG -> AV1 chunks
```

## Streams and owners

| Stream | Code | Status |
|---|---|---|
| 1. Capture + tokenization | `headset/.../FrameUploader.cs`, `server/megamind/ingest`, `server/megamind/store` | skeleton |
| 2. Retrieval tool + reader | `server/megamind/agent/{tools,reader}.py` | skeleton |
| 3. AR chat | `headset/.../ChatClient.cs`, `ChatPanel.cs`, `server/megamind/agent` | skeleton |

Every `TODO(stream-N)` comment in the code marks work owned by that stream.

## Rules

- Headset and server communicate **only** through `docs/api.md`. A contract change
  is one PR that updates `api.md`, `server/megamind/models.py` and the C# DTOs together.
- Everything server-side must be testable with `curl` / the feeder, without hardware.
- No model training. No on-device inference.
