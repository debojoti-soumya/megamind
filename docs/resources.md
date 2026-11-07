# Hackathon resources → how Megamind uses them

Source: [2026 Neo Hackathon](https://neo-community.notion.site/2026-Neo-Hackathon-40c2ad94154180fba622dd4d978e1c7e). Team of 4 ⇒ per-person credits ×4.

| Resource | Amount (team) | Use in Megamind |
|---|---|---|
| **Anthropic** | $200/pp → $800 | **Claude Sonnet 4.5 orchestrator** (`/chat`, tool-use loop). ~$0.01–0.03 per turn; budget is not a constraint. Also fallback captioner if the open VLM path is flaky. |
| **Sail Research** (OpenAI/Anthropic-compatible open-model inference) | $500/pp → $2000 | **Qwen3.6-35B-A3B** (vision, `Qwen/Qwen3.6-35B-A3B`, flex window, $0.05/$0.40 per 1M) as the **enricher** (caption/OCR/entities for every keyframe — background job, latency-tolerant). **Gemma 4 31B IT** (vision, default window, low latency) or Kimi K3.1 Pro as the interactive **reader** if Qwen's flex latency is too high for live Q&A. Base URL `https://api.sailresearch.com/v1`. |
| **Modal** | $500/pp → $2000 | Plan B for the reader/enricher: self-host `Qwen2.5-VL-7B-Instruct` on vLLM (A100/H100, OpenAI-compatible) if Sail's Qwen latency or availability disappoints. Also SigLIP image embeddings and Whisper STT as serverless endpoints, and optionally host the FastAPI server itself (public URL ⇒ no Tailscale/ngrok needed for the headset). |
| **OpenAI** | $100/pp → $400 | `text-embedding-3-small` text embeddings, Whisper STT, `tts-1` TTS. Cheap, reliable, already the default in `.env.example`. |
| **Team card** | $100 | ElevenLabs (nicer TTS voice) or Deepgram (fast streaming STT) if OpenAI's are too slow for the demo. |
| **Hardware** | 4× Quest 3, 6× Jetson Orin Nano | Extra Quest 3s ⇒ parallel headset dev + multi-wearer demo (memories tagged by `device_id`). Jetson not needed (server does the heavy lifting). |
| **Cognition / Devin** | unlimited | Parallel sub-agent sessions per stream. |
| **Warp** | free Build plan | Team terminal. |
| xAI, ByteDance | TBD | Ignore unless announced. |

## Recommended default configuration

```
CLAUDE_MODEL=claude-sonnet-4-5                      # Anthropic credits
ENRICH_BASE_URL=https://api.sailresearch.com/v1     # Sail credits
ENRICH_MODEL=Qwen/Qwen3.6-35B-A3B                   # vision, cheap, background
READER_BASE_URL=https://api.sailresearch.com/v1
READER_MODEL=Qwen/Qwen3.6-35B-A3B                   # try first; swap to google/gemma-4-31B-it if slow
EMBED_PROVIDER=openai  STT_PROVIDER=openai  TTS_PROVIDER=openai   # OpenAI credits
```

Sail image-input limits: ≤20 images/request, JPEG/PNG/WebP, base64 or public URL. Our reader sends ≤12 thumbnails per call, so this fits; thumbnails should be sent as base64 (the server is not on the public internet).

## Schedule and rules that affect us

- Hacking window: **Fri 9:00 PM PT → Sun 8:00 AM PT** (Devpost submission due 8 AM). ~35 hours.
- Demos **Sun 10:30 AM–12:00 PM, science-fair style**, 3+ judges per team, live demo focused. Record a backup video for Devpost.
- Awards: Top 3 overall + *Most Technically Impressive* + *Most Delightful*.
- Rules: planning docs and software installs may be done in advance; **code may not be written before 9 PM Friday**. Teams must link the GitHub repo with all source created at the hackathon.
