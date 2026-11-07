# Hackathon resources → how Megamind uses them

Source: [2026 Neo Hackathon](https://neo-community.notion.site/2026-Neo-Hackathon-40c2ad94154180fba622dd4d978e1c7e). Team of 4 ⇒ per-person credits ×4.

| Resource | Amount (team) | Use in Megamind |
|---|---|---|
| **Anthropic** | $200/pp → $800 | **Claude Sonnet 4.5 orchestrator** (`/chat`, tool-use loop). ~$0.01–0.03 per turn; budget is not a constraint. Also fallback captioner if the open VLM path is flaky. |
| **Sail Research** (OpenAI/Anthropic-compatible open-model inference) | $500/pp → $2000 | **Gemma 4 31B IT** (vision, `google/gemma-4-31B-it`, `asap` window, $0.40/$0.60 per 1M) as the interactive **reader** (Chat Completions, `response_format=json_schema`); fallback `nvidia/Gemma-4-31B-IT-NVFP4` (asap, $0.14/$0.40). Same model on the `balanced` window ($0.12/$0.60) as the **enricher** (caption/OCR/entities per keyframe — background, latency-tolerant). **Qwen3.6-35B-A3B** (`Qwen/Qwen3.6-35B-A3B`, $0.05/$0.40) is **flex-only**: not available on Chat Completions or foreground Responses, only `background: true` Responses (202 + poll) or Batch — usable as a cheaper enricher via polling, never as the live reader. Base URL `https://api.sailresearch.com/v1`. Rate limits are per-org (reader and enricher share them). |
| **Modal** | $500/pp → $2000 | Plan B for the reader/enricher: self-host `Qwen2.5-VL-7B-Instruct` on vLLM (A100/H100, OpenAI-compatible) if Sail's Qwen latency or availability disappoints. Also SigLIP image embeddings and Whisper STT as serverless endpoints, and optionally host the FastAPI server itself (public URL ⇒ no Tailscale/ngrok needed for the headset). |
| **OpenAI** | $100/pp → $400 | `text-embedding-3-small` text embeddings, Whisper STT, `tts-1` TTS. Cheap, reliable, already the default in `.env.example`. |
| **Team card** | $100 | ElevenLabs (nicer TTS voice) or Deepgram (fast streaming STT) if OpenAI's are too slow for the demo. |
| **Hardware** | 4× Quest 3, 6× Jetson Orin Nano | Extra Quest 3s ⇒ parallel headset dev + multi-wearer demo (memories tagged by `device_id`). Jetson not needed (server does the heavy lifting). |
| **Cognition / Devin** | unlimited | Parallel sub-agent sessions per stream. |
| **Warp** | free Build plan | Team terminal. |
| xAI, ByteDance | TBD | Ignore unless announced. |

## Recommended default configuration

```
CLAUDE_MODEL=claude-sonnet-4-6                      # Anthropic credits
ENRICH_BASE_URL=https://api.sailresearch.com/v1     # Sail credits
ENRICH_MODEL=google/gemma-4-31B-it                  # vision; balanced window (X-Sail-Completion-Window: balanced)
READER_BASE_URL=https://api.sailresearch.com/v1
READER_MODEL=google/gemma-4-31B-it                  # asap window; fallback nvidia/Gemma-4-31B-IT-NVFP4 (Qwen3.6 is flex-only → 400 on chat completions)
EMBED_PROVIDER=openai  STT_PROVIDER=openai  TTS_PROVIDER=openai   # OpenAI credits
```

Sail image-input limits: ≤20 images/request, ≤20 MB per image, JPEG/PNG/WebP, base64 or public URL (URL fetch must finish in 10 s). Use `max_retries=0` on the OpenAI client plus an `Idempotency-Key` header and your own backoff (Sail bills retried requests); honour `Retry-After` on 429/503. Source: `research/reports/04-sail-api-limits-latency.md`. Our reader sends ≤12 thumbnails per call, so this fits; thumbnails should be sent as base64 (the server is not on the public internet).

## Schedule and rules that affect us

- Hacking window: **Fri 9:00 PM PT → Sun 8:00 AM PT** (Devpost submission due 8 AM). ~35 hours.
- Demos **Sun 10:30 AM–12:00 PM, science-fair style**, 3+ judges per team, live demo focused. Record a backup video for Devpost.
- Awards: Top 3 overall + *Most Technically Impressive* + *Most Delightful*.
- Rules: planning docs and software installs may be done in advance; **code may not be written before 9 PM Friday**. Teams must link the GitHub repo with all source created at the hackathon.
