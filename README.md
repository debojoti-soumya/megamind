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

Or open the browser prototype of the AR panel at <http://localhost:8000/ui>: webcam as
passthrough, hold **Space** to talk, cited thumbnails, TTS playback, status HUD, and an optional
1 fps capture toggle (settings ⚙) so the whole loop runs on one laptop. Real STT/TTS need
`OPENAI_API_KEY` (or Deepgram / ElevenLabs keys) in `server/.env`; without keys the mock
providers are used.

## Streams and owners

| Stream | Code | Status |
|---|---|---|
| 1. Capture + tokenization | `headset/.../FrameUploader.cs`, `server/megamind/ingest`, `server/megamind/store` | skeleton |
| 2. Retrieval tool + reader | `server/megamind/agent/{tools,reader}.py` | skeleton |
| 3. AR chat | `headset/.../ChatClient.cs`, `ChatPanel.cs`, `PanelAnchor.cs`, `SettingsPanel.cs`, `server/megamind/agent/{stt,tts}.py`, `server/megamind/ui` | prototype (browser `/ui` works end-to-end; Unity scripts untested on device) |

Every `TODO(stream-N)` comment in the code marks work owned by that stream.

## Rules

- Headset and server communicate **only** through `docs/api.md`. A contract change
  is one PR that updates `api.md`, `server/megamind/models.py` and the C# DTOs together.
- Everything server-side must be testable with `curl` / the feeder, without hardware.
- No model training. No on-device inference.
