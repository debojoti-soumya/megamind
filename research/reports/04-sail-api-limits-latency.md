# 04 — Sail Research API: limits, completion windows and what they mean for enricher vs. reader (answers issue #14)

*Research bot (stream-5), 2026-11-07. Applies to: **storage** (stream-2 enricher), **retrieval** (stream-3 reader), **harness/coordinator** (model choice). All facts below are from docs.sailresearch.com as of today unless marked (?).*

## Question
What are Sail's rate limits, image limits and latency characteristics for `Qwen/Qwen3.6-35B-A3B` vs `google/gemma-4-31B-it`? Decides enricher concurrency and whether the reader needs the Gemma fallback.

## Findings (primary: Sail docs)

### 1. Qwen3.6-35B-A3B is **flex-only** on Sail — it cannot be the interactive reader
- Pricing page lists `Qwen/Qwen3.6-35B-A3B` under **"Flex-only models — served with the `flex` completion window exclusively"** ($0.05 in / $0.02 cached / $0.40 out per 1M).
- **`flex` is not compatible with Chat Completions, Messages, or foreground Responses requests.** To use it you must call the **Responses API with `background: true`** (returns `202`, then poll `GET /v1/responses/{id}`) or the **Batch API** (`POST /v1/batches`, always flex).
- `flex` = "best-effort scheduling… batch jobs, evals, offline". No latency SLO is published; expect seconds-to-minutes and high variance (?). Results can expire 10 min after first successful delivery.
- ⇒ **Our `docs/resources.md` default (`READER_MODEL=Qwen/Qwen3.6-35B-A3B` via chat completions) will return HTTP 400.** The reader (interactive, ≤ 2.5 s budget) must be a model with an `asap` window.

### 2. Gemma 4 31B IT has all three windows and image input
| Model ID | Window | In / cached / out ($ per 1M) |
|---|---|---|
| `google/gemma-4-31B-it` | asap (default) | 0.40 / 0.20 / 0.60 |
| | balanced | 0.12 / 0.08 / 0.60 |
| | flex | 0.06 / 0.02 / 0.30 |
| `nvidia/Gemma-4-31B-IT-NVFP4` (NVFP4 quant) | asap | 0.14 / 0.07 / 0.40 |
| | balanced | 0.11 / 0.06 / 0.32 |
| | flex | 0.07 / 0.04 / 0.20 |
| `Qwen/Qwen3.6-35B-A3B` | **flex only** | 0.05 / 0.02 / 0.40 |

Gemma 4 31B (HF card): dense 30.7B, **256K context**, modalities text + image (+ audio natively, but Sail does not accept audio input), built-in reasoning mode, "Object detection, Document/PDF parsing, screen and UI understanding". Qwen3.6-35B-A3B (HF card): MoE 35B total / 3B active, 262K context, image-text-to-text, thinking mode on by default (`enable_thinking`), MMMU-Pro vision 75.3.

### 3. Image limits (all models)
- **Max 20 images per request**; 256 MiB total request body; **max 20 MB per image** (pre-base64); JPEG/PNG/WebP/GIF; no pixel-dimension limit.
- base64 data URIs or **public** URLs (URL fetch must complete in 10 s or the request fails with 400) — our server is not public ⇒ **send base64** (as planned).
- `detail: "auto" | "low" | "high"` is accepted (effect per model unspecified (?)).
- Error messages are explicit: `too many images: maximum 20 images per request`, `model '<id>' does not support image input`.

### 4. Rate limits and errors
- Limits are **per organization** (all keys share), on requests/min **and** concurrency; numbers are not published. "Accounts using only promotional sign-up credits are allowed a limited number of requests per minute" — hackathon credits may or may not count as promotional (?); **purchasing any credits removes that cap**, Pro plan gives higher limits.
- `429` → honour `Retry-After`. `503 model_capacity_unavailable` → asap request refused *before* inference (Sail waits briefly for capacity, then refuses — no queueing). `503 service_overloaded` can happen in any window.
- Retry guidance: exponential backoff 1-2-4 s + jitter, cap attempts and total time; **send an `Idempotency-Key`** on first attempt and reuse on retries, or disable SDK auto-retries (`client.with_options(max_retries=0)`) to avoid double-billing.
- Streams can return 200 and still fail mid-generation — handle error events.

### 5. API feature support relevant to us
- **Chat Completions:** `response_format` supports `json_object` **and** `json_schema` → use `json_schema` for enricher/reader output. `stream: true` SSE supported. Not supported: `stop`, `seed`, `logit_bias`, `frequency/presence_penalty`, `logprobs`.
- **Responses API:** `text.format` `json_schema` only (no `json_object`); function tools; `background: true` works only with balanced/flex; `reasoning.effort` none…max; **no `previous_response_id`** (send full input every time); `prompt_cache_key` routing hint; all inputs get a 30-min cache write for free.
- Anthropic Messages API is **Beta** — stick to OpenAI-compatible endpoints.

## Concrete recommendations

| # | Recommendation | Stream |
|---|---|---|
| 1 | **Reader = `google/gemma-4-31B-it`, window `asap`, Chat Completions, `response_format=json_schema`, `stream=false`.** Set `READER_MODEL=google/gemma-4-31B-it` as the default in `.env.example`/`resources.md`; drop Qwen3.6 as the interactive reader. If asap capacity 503s during the demo, fall back to **`nvidia/Gemma-4-31B-IT-NVFP4` asap** (same API, cheaper, likely different capacity pool (?)), then Claude Haiku/Sonnet vision. | retrieval, harness |
| 2 | **Enricher option A (simplest, recommended for hackathon): Gemma 4 31B `balanced` via Chat Completions** (`metadata.completion_window: "balanced"` or header `X-Sail-Completion-Window: balanced`). Same code path as the reader; $0.12/1M input. A 960-px JPEG + prompt ≈ 1–2K tokens ⇒ ~$0.0003/frame; 1 000 frames/h ≈ $0.30/h. Latency: unspecified, "wider scheduling window" — likely seconds to tens of seconds (?), acceptable for an async worker. | storage |
| 3 | **Enricher option B (cheapest): Qwen3.6-35B-A3B via Responses API `background: true` + `flex`,** poll `GET /v1/responses/{id}` with backoff (1, 2, 4… s, cap 30 s), treat `completed/incomplete/failed/cancelled` as terminal, read `output_text`. Only worth it if (A) is rate-limited or too slow; it adds a polling state machine and unbounded latency. Do **not** use the Batch API for live ingest (it is for offline jobs) — but it is ideal for re-enriching a recorded demo walk overnight. | storage |
| 4 | **Enricher concurrency:** start at **4 in-flight requests**, adaptive: on `429/503` halve and sleep `Retry-After` (default 2 s); on 20 consecutive successes +1 up to 8. Shared org limits mean the reader competes with the enricher — **reserve headroom: pause the enricher queue while a `/chat` turn is in flight** (one asyncio.Event). | storage |
| 5 | **Idempotency + no SDK retries:** `client = OpenAI(base_url=…, max_retries=0)`, add `Idempotency-Key: <frame_id>` header, own retry loop (3 attempts, 1-2-4 s + jitter, 60 s total). For the reader: 1 retry max, then degrade to caption-only evidence so the voice answer still arrives. | storage, retrieval |
| 6 | **Thumbnails to the reader:** ≤ 12 images (well under 20), JPEG q≈70 at 640–960 px long side, base64. Set `detail: "low"` for the reader (speed) unless an OCR question is detected, then `"high"` (?). Keep the enricher at full 960 px for OCR. | retrieval |
| 7 | **Thinking/reasoning off for latency:** Chat Completions on Gemma: `reasoning_effort` is not in the supported param list, so control via prompt / `chat_template_kwargs` is unverified (?) — measure reader latency with and without a "do not think, answer directly" instruction on day 1. For Qwen3.6 (thinking on by default), pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` if the Responses path accepts it (?), otherwise cap `max_output_tokens` ≈ 400 for captions. | retrieval, storage |
| 8 | **Day-1 measurement (10 min, needs `READER_API_KEY`):** script 20 reader calls (12 images each) and 50 enricher calls on `asap`, `balanced`, (`flex` background) and record p50/p95 latency and any 429/503 — Sail also exposes a **Usage API with latency endpoints** (`/usage`, `/usage-endpoints`) that can confirm server-side numbers. Put results in `research/digest/` and set `ENRICH_WINDOW` accordingly. | harness |
| 9 | **Budget sanity:** at Gemma asap prices a reader call (12 images ≈ 10K tokens in, 300 out) ≈ $0.004; 500 demo questions ≈ $2. Enricher at balanced ≈ $0.30/h of wear. Sail credits ($2000) are not a constraint; **rate limits are**. | coordinator |

### Open questions for the coordinator
- Do the hackathon Sail credits count as "promotional" (lower RPM) — check the dashboard once the key arrives; if so, ask Sail/organisers to lift, or put $5 on the card.
- Confirm the Sail model list at key time: `GET https://api.sailresearch.com/v1/models` (401 without a key).

## Sources
- Completion windows (flex ⇒ background Responses/Batch only): https://docs.sailresearch.com/completion-windows
- Pricing incl. "Flex-only models": https://docs.sailresearch.com/pricing
- Models: https://docs.sailresearch.com/models
- Images (20/request, 20 MB, base64/URL, detail): https://docs.sailresearch.com/images
- Rate limits (org-shared, promotional cap, 429/503): https://docs.sailresearch.com/rate-limits
- Retrying requests (backoff, idempotency, uncertain submissions): https://docs.sailresearch.com/retries
- API support matrix (json_schema, streaming, background, unsupported params): https://docs.sailresearch.com/support
- Sending requests at scale (Batch/background): https://docs.sailresearch.com/requests_at_scale
- Gemma 4 31B IT model card: https://huggingface.co/google/gemma-4-31B-it ; Qwen3.6-35B-A3B model card: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
