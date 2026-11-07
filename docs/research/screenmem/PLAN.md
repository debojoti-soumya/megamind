# Screen-memory ("Coast-like") — 18-hour hackathon plan, backed by measurements

## 0. TL;DR

- **100 days in 100 GB is feasible** for the pixel tier on a single 2560×1920 display *if* you (a) capture at ≤1 fps, (b) drop unchanged frames, (c) encode kept frames as low-fps video (HEVC/AV1), and (d) age pixels down (full-res → 1920 → text-only). Measured on 13.7 min of scripted browsing/typing/YouTube: **0.19–1.1 GB per 16-hour day** depending on codec/CRF (table below). Text + embeddings tier is **~8 MB/day (≤ 46 MB with SQLite/FTS overhead)** — essentially free. Budget sketch for 100 days: ~15 GB recent pixels + ~25 GB aged pixels + ~3 GB text/vectors + optional ~12 GB audio = **~55 GB, 45 GB headroom**.
- **The big lever is not the codec, it's what you don't store**: 70 % of 1-fps frames were byte-identical to the previous one; only 13–19 % survive change detection (threshold matters — see §3.4).
- **"Tokenization/context-aware compression" should be a *retrieval* tier, not the storage tier.** Store pixels with a codec (5–30 KB/frame); store *small* retrieval vectors (int8 MiniLM 384 B/frame, pooled ColPali-style patches ~5 KB/frame); compute VLM visual tokens *at query time* from the codec frames. Continuous visual tokens (64 tokens × 1280-d fp16 = 164 KB/frame) are 5–30× *larger* than the compressed pixels they describe. The exception worth researching later is **discrete** screen tokenizers (DeepSeek-OCR-style optical compression, VQ codes ≈ 0.1–0.5 KB/frame) — not an 18-hour item.
- **Exact text must come from OCR/accessibility, never from a VLM.** SmolVLM-500M on CPU hallucinated page topics; it took 7–11 s/frame. Fine for daily summaries; unacceptable as evidence.
- **Do in 18 h:** capture daemon (1 fps + change detection + window title/URL) → HEVC chunks + PNG keyframes → OCR + FTS5 + MiniLM → episode segmentation → hybrid search → LLM answer with cited frames → timeline/chat UI (+ MCP/CLI). **Skip:** audio, learned tokenizers, encryption-at-rest beyond FileVault, Windows/Linux, agent actions.

## 1. What Coast and the others actually are (research summary)

| Product | Capture | Index | Query surface | Storage claim | Lesson |
|---|---|---|---|---|---|
| **Coast Local** (Attention Eng., Mac, Jul 2026 preview, ~5 ppl, $1.25M) | screenshot every ~2 s; bundle id, window title, web domain; optional AX tree | OCR → FTS; "activity segments" | **CLI** (`coast query fts / sample / image`, `coast usage`) consumed by Claude Code/Codex/Cursor — *no native chat* | retention period + storage cap, old video compressed/deleted | Dumb local index + external agents; endgame is workflow automation. Reviewer: unprompted recall ≈ 0/wk, "where did my week go" is the real demand |
| **Rewind** (2022–25 → Limitless → Meta, capture disabled Dec 2025) | SCK screenshot every 2 s of focused display, Apple Vision OCR | SQLite (frames, segments, search_content w/ bbox) + H.264 chunks at 0.5 fps | timeline + chat (GPT-4) | "3,750×"; real 12–14 GB/mo | Pivoted to meetings/pendant; desktop recall alone didn't sustain the business |
| **MS Recall** | periodic snapshots, focused window | OCR + vector DB, VBS enclave, TPM keys | Windows search UI | 25/75/150 GB caps; 90-day default | Security redesign after TotalRecall; SIT filter misses things; "malware with user privs reads everything — by design" |
| **Screenpipe** (YC S26, Rust) | **event-driven** (focus, click, scroll end, typing pause, clipboard, idle 5 s) + AX tree; OCR fallback | SQLite FTS; JPEGs; audio 30 s chunks w/ Whisper | API :3030 + MCP + plugins | 5–10 GB/mo event-driven (older naive version 5–10 GB/day) | Abandoned "record video + OCR every frame" (dup data, CPU heat); AX tree >> OCR |
| Pensieve / OpenRecall / Windrecorder | 5 s screenshots, change detect, OCR, optional VLM | SQLite + embeddings | web UI | ~400 MB/day, ~8 GB/mo | Same architecture converges everywhere |

Capture APIs: macOS **ScreenCaptureKit** (window exclusion, zero-copy) + **AXUIElement**; Windows **Graphics.Capture** (gives dirty rects) + UIA; Linux xdg-desktop-portal/PipeWire (fragmented, Wayland-dependent). Apple Silicon: HW **HEVC/H.264 encode**, AV1 decode only (M3+), **no AV1 encode**, no HEVC-SCC → on Mac use HEVC via VideoToolbox; AV1 (palette + IntraBC = ideal for text) is software-only.

## 2. Creative additions (ranked by value / 18-h cost)

1. **Episodes, not frames.** Segment the stream by (app, title/URL, visual change) → one record with span, text, representative frame. Makes "what was I doing at 3 pm" and summaries trivial; cuts retrieval units 10–50×. *(do it)*
2. **"Resume where I left off" / end-of-day journal** auto-generated from episodes — the continuous-value loop Coast's reviewer says recall lacks. *(do it, it's one LLM call over episodes)*
3. **Cited answers**: every answer returns frame ids + timestamps + OCR crop; UI shows the frame. Hallucination guard. *(do it)*
4. **Agent/MCP surface**: expose `search_text`, `sample_episodes(range)`, `get_frame(id)`, `summarize(range)` as MCP tools so Claude Code/Cursor can use your memory (Coast's actual wedge). *(cheap, high demo value)*
5. **Clipboard + URL + typing-burst events as first-class evidence** (typing changes <0.5 % of pixels and was *missed* by our 0.02 thumbnail threshold — see §3.4). *(do: keystroke-count / clipboard triggers)*
6. **Fidelity aging ("memory forgetting")**: pixels full-res 7–14 d → 1920 → 1280 → keyframes-only → text-only; text/vectors forever. Directly mirrors DeepSeek-OCR's proposal of shrinking older context. *(cron job, cheap)*
7. **Privacy tiers**: exclude apps/domains/private windows; secure-text fields via AX; regex SIT filter (cards, keys) before persisting OCR text; global hotkey pause; visible indicator. *(do the exclusion list + pause; the rest later)*
8. Later: meetings/audio with consent prompts; "how did I fix this before" procedural search over terminal/IDE episodes; semantic dedup across days ("you've opened this doc 14 times"); time-accounting dashboards; cross-device sync (encrypted); region-of-interest VLM Q&A ("what did the chart say").

## 3. Compression: what we measured (this VM: Ubuntu, 8 vCPU, no GPU, 2560×1920 X display)

Corpus: 819 lossless PNG frames at 1 fps over 13.7 min (920 MB raw) of scripted Chrome activity: 12 pages of Wikipedia/HN/GitHub/docs/arXiv/BBC with scrolling, 49 s of typing into a textarea, 60 s of YouTube (Big Buck Bunny), then more browsing. Ground truth: `events.jsonl` (URLs, typed strings, timestamps). Scripts in `/home/ubuntu/screenmem/`.

### 3.1 Straight 1-fps video (all 819 frames)

| Encode | MB (13.7 min) | GB / 16-h day |
|---|---|---|
| raw PNG | 920 | 64.7 |
| H.264 full 2560 crf28 | 34.6 | 2.43 |
| H.264 1280-wide crf28 | 12.3 | 0.87 |
| HEVC full crf28 / crf32 | 40.5 / 31.5 | 2.85 / 2.22 |
| HEVC 1920-wide crf30 | 23.4 | 1.64 |
| HEVC 1280-wide crf30 | 13.0 | 0.92 |
| AV1 1920-wide crf40 | 11.1 | 0.78 |
| AV1 full crf40 | 14.1 | 0.99 |

Per-phase bitrate (HEVC 1920 crf30): browsing+scrolling **32 KB/s**, typing **4 KB/s**, YouTube **42 KB/s**; AV1 1920: 16.5 / 2.7 / **10.2** KB/s (AV1 handles natural video *better* than text; x265 the opposite).

### 3.2 Change-detected keyframes only (109/819 frames, 13 %, thumbnail MAD > 0.02)

| Encode of kept frames | MB | GB / 16-h day | OCR word recall (§3.3) |
|---|---|---|---|
| HEVC full crf28 | 29.1 | 2.04 | 0.92 |
| HEVC 1920 crf30 | 16.3 | 1.14 | 0.91 |
| HEVC 1920 crf34 | 12.1 | 0.85 | 0.90 |
| HEVC 1280 crf30 | 9.2 | 0.65 | 0.87 |
| **AV1 1920 crf40** | **4.1** | **0.28** | **0.90** |
| AV1 1920 crf48 | 2.7 | 0.19 | 0.87 |
| WebP 1920 q60 stills (random access) | 15.8 | 1.11 | 0.90 |

Encode time on 8 vCPU for 13.7 min of footage: x264 24–29 s, x265 24–39 s, libaom-av1 cpu-used 6: 133–253 s (≈2–3× realtime at 1 fps — OK as a background job, not on battery).

### 3.3 OCR fidelity vs compression
Tesseract at full 2560×1920 on 16 sampled keyframes; each variant decoded and upscaled back to 2560×1920; metric = word recall / precision vs OCR of the lossless PNG (mean ~300 words/frame; Tesseract's own noise is included, so even a perfect codec would score <100 %).

| Variant | word recall | word precision | bytes/frame (kept-frame encode) |
|---|---|---|---|
| JPEG full q60 | 0.93 | 0.89 | ~250 KB |
| HEVC full crf28 | 0.92 | 0.91 | 267 KB |
| WebP full q50 | 0.91 | 0.89 | ~200 KB |
| HEVC 1920 crf30 | 0.91 | 0.89 | 149 KB |
| HEVC 1920 crf34 | 0.90 | 0.89 | 111 KB |
| WebP 1920 q60 | 0.90 | 0.86 | 145 KB |
| **AV1 1920 crf40** | **0.90** | 0.87 | **37 KB** |
| PNG 1920 (lossless, resolution loss only) | 0.88 | 0.87 | — |
| AV1 1920 crf48 | 0.87 | 0.84 | 25 KB |
| HEVC 1280 crf30 | 0.87 | 0.86 | 84 KB |

Reading: downscaling 2560→1920 costs ~4 points of OCR recall and is the dominant loss; codec quantization at these CRFs costs ≤1–2 points on top. AV1 1920 crf40 keeps 90 % word recall at **4× fewer bytes than HEVC crf30** — the archive-tier pick. 1280-wide is where small UI text starts to go. (Earlier RapidOCR run gave 36–77 % — an artifact of its default `limit_side_len=736` downscale; set it to ≥2560 or tile the screen.) **Important corollary: OCR the lossless frame at capture time, before any lossy step — the text tier must never be derived from the archive tier.**

### 3.4 Dedup / lossless tier
- 70 % of consecutive 1-fps frames were **byte-identical**; median changed-tile fraction 0 %, mean 9.6 %.
- RDP-style lossless (64×64 changed tiles, XOR vs previous, zstd-3): browsing 138 KB/s (8 GB/16 h), typing **12.5 KB/s (0.7 GB/16 h)**, YouTube 2.8 MB/s (161 GB/16 h). → Lossless is viable only as a short rolling window for text-heavy work, must auto-switch to codec on video.
- **Threshold trap:** at MAD > 0.02 the typing phase kept 2/49 frames — the typed notes were almost entirely lost from the pixel *and* OCR tiers (2 of 5 typed-note questions became unanswerable). At > 0.002 it keeps 47/49 typing frames and ~31 % of all frames. Use a low threshold **plus** input-event triggers (keystroke bursts, clipboard, focus change), and dirty-rect area rather than global MAD.

### 3.5 Text / vector tier
Typing-safe change detection (MAD > 0.002) kept 156/819 frames (19 %); Tesseract full-res OCR took 0.17 s/frame wall on 8 cores (1.7 s single-thread); ~254 words/kept frame.

| Item | bytes (13.7 min) | MB / 16-h day |
|---|---|---|
| OCR text, all kept frames, zstd-19 | 53.9 KB | **3.8** |
| OCR text, delta lines vs previous frame, zstd-19 | 51.6 KB | 3.6 (delta barely helps after zstd; helps the LLM context more than disk) |
| MiniLM 384-d int8 per kept frame | 59.9 KB | 4.2 |
| MiniLM binary | 7.5 KB | 0.5 |
| SQLite (text + FTS5 + fp16 vectors, unoptimized, 4 KB pages) | 655 KB | 46 — index overhead dominates; still < 5 GB / 100 days |

Text + vectors for 100 days ≈ 0.4–5 GB. Storage is a pixel problem, not a text problem.

### 3.6 Query quality on the text tier (16 gold questions; retrieval hit = a returned frame's OCR contains the answer)
| Retrieval over text tier | v1: RapidOCR @736 px, MAD>0.02 (109 frames) | v2: Tesseract full-res, MAD>0.002 (156 frames) |
|---|---|---|
| answerable at all (gold answer present in some frame's OCR) | 87.5 % (14/16) | **100 %** |
| FTS5 (BM25) hit@1 / hit@5 | 75 % / 75 % | **87.5 % / 93.8 %** |
| MiniLM dense hit@1 / hit@5 | 62.5 % / 75 % | 68.8 % / **100 %** |

The two v1 misses were the typed invoice number and Lisbon budget — lost by the change-detection threshold, not by compression. Exact questions favour FTS; semantic ones favour dense; union@5 = 100 %. A VLM caption tier (SmolVLM-500M, 7–11 s/frame CPU) was tested qualitatively: good at "what kind of page is this", hallucinated an HN front page as a law-firm article → captions are a ranking signal, never evidence.

### 3.7 100 GB / 100 days budget (1 display, 16 active h/day, change-detected 1 fps)

| Tier | Policy | GB |
|---|---|---|
| Recent pixels (days 0–14) | HEVC 1920 crf30, HW encoder, random-access GOP ≤ 60 | 14 × 1.1 ≈ **15** |
| Aged pixels (days 15–100) | re-encode to AV1 1920 crf40–48 (or HEVC 1280 crf30 on Mac if no AV1 encode time) | 86 × 0.3 ≈ **26** |
| Episode keyframes (WebP q60 1920, 1 per episode, kept forever) | ~300 episodes/day × 100 KB | **3** |
| OCR text + AX JSON (zstd, delta vs previous frame) | ≤ 40 MB/day | **≤ 4** |
| Vectors: MiniLM int8 per kept frame + pooled visual patch vectors per keyframe | 384 B × 18 k + 5 KB × 300 /day | **≤ 1** |
| Audio (optional, Opus 16 kbps VAD-gated) | ~5 h speech/day | ~12 |
| **Total** | | **~50–60 GB** |
Worst cases that break it: 2–3 displays (×2–3), 5K Retina (×2 pixels), full-screen video all day (×3–5 with x265; AV1 helps). Mitigation: per-display budget, fps 0.2 during video, resolution cap 1920-wide for the archive tier.

## 4. Context-aware compression / tokenization — research synthesis and recommendation

**Taxonomy of what "tokenization" can mean here, with measured/literature costs per frame:**

| Representation | Bytes/frame | Exact text? | Queryable by LLM? | Notes |
|---|---|---|---|---|
| HEVC/AV1 frame in chunk | 5–30 KB | yes (after decode+OCR) | after decode → VLM | random access needs keyframes; HW encode on Mac |
| WebP keyframe | ~100 KB @1920 | yes | yes | random access, 5× more than video |
| OCR text + bboxes (delta, zstd) | 0.2–2 KB | **yes** | yes, cheapest tokens (~300 words ≈ 400 LLM tokens) | the authoritative evidence tier |
| AX tree JSON (roles/names/values/bounds, diffed) | 1–10 KB | **yes, structured** | yes | 100× cheaper than OCR to produce; fails on canvas/remote/video |
| MiniLM text embedding int8 | 384 B | no | retrieval only | binary 48 B retains ~92–96 % w/ rescoring |
| ColPali/ColQwen patch vectors | 1,030 × 128 d ≈ 130 KB fp16; pooled 38 vectors ≈ 5 KB int8 | no | late-interaction visual retrieval | 98 % perf at 12 % memory (Light-ColPali) |
| VLM visual tokens, continuous (SigLIP/FastViT/DeepSeek-OCR Tiny 64 tok × 1280 fp16) | 64–400 tokens → 160 KB–1 MB | reconstructable ~97 % at ≤10× (DeepSeek-OCR) | yes, directly | **bigger than the pixels** — don't persist; compute at query time |
| Learned discrete tokens (TiTok 32–128 codes; Cosmos DV 8×16×16; VQ) | 0.1–2 KB | **no** (trained on natural images, text legibility not preserved) | via decoder | research direction: GUI-specific tokenizer trained to preserve glyphs |
| GUI-adaptive tokens (AQuaUI quadtree) | −30 % tokens at 99 % GUI-agent accuracy | n/a | yes | training-free; uses nonuniform density + temporal reuse |
| Bounded KV memory (StreamMem / KeyRec) | fixed budget per session (~10 % of tokens) | no | yes | query-agnostic saliency; novelty add/merge/evict |
| Generative codec (DiSCo: text + sketch → diffusion) | <1 KB | **hallucinates** | — | thumbnails only, never evidence |

**Design recommendation (hybrid "codec for pixels, text for truth, tokens for retrieval"):**
1. Pixels live in the codec tier; they are the only thing that can be re-derived from (re-OCR with better models later, re-embed, re-tokenize). Age them down, never generate them.
2. Text truth = AX tree when available, else OCR with bboxes. Store **deltas** (new lines vs previous frame) — cuts text ~2–4× before zstd.
3. Retrieval tokens = small per-frame text embedding + per-*episode* pooled visual patch vectors (for "the page with the red chart"). Both quantized int8/binary. Version every vector with its model id.
4. Query-time "cross-attention": planner (LLM) → filters (time/app) + FTS + dense → candidate episodes → **late-interaction rerank** (query tokens × stored patch vectors) → pull 3–8 frames/crops from codec → VLM/LLM answers with citations. Cross-attention is applied to a few hundred candidates, not to 100 days.
5. Context-aware *spatial* tokenization at query time: feed the VLM OCR-bbox/AX-bounds-guided crops (text regions at full res, chrome/background at low res; unchanged regions reused across frames) — the AQuaUI idea with free side-information we already have. Cuts VLM input tokens ~2–3× vs whole screenshots.
6. Post-hackathon research: a **GUI-specific discrete tokenizer** (VQ/FSQ trained on screenshots with an OCR-consistency loss) targeting ~256–512 codes/frame that preserves glyph legibility — this is the piece that would make the pixel tier ~10× smaller than AV1 and let an LLM attend over months. Nobody ships this yet; DeepSeek-OCR's 97 %-at-10× result is the existence proof.

## 5. The 18-hour build plan

**Stack (fastest path):** Python daemon (mss or `screencapture`/ScreenCaptureKit via `pyobjc`, or FFmpeg avfoundation on Mac) → SQLite (WAL) + FTS5 + `sqlite-vec`/numpy → FastAPI → small React/HTML timeline+chat → OpenAI/Anthropic or local Ollama for answers. Embeddings: `all-MiniLM-L6-v2` (CPU, 384-d). OCR: Apple Vision (Mac, via pyobjc) or Tesseract/RapidOCR(limit_side_len=2560!).

| Hour | Deliverable | Acceptance |
|---|---|---|
| 0–2 | `capture.py`: 1 fps screenshot of focused display, 64×48 thumbnail MAD > 0.002 **or** input/clipboard/focus event → keep; record app, title, URL (AppleScript/AX), timestamp to `frames` table; PNG to spool | runs 10 min, ~30 % frames kept, titles correct |
| 2–4 | `index.py` worker: OCR kept frames (bbox+text), MiniLM embed, FTS5 insert; **episode segmentation** (new episode on app/title/URL change or MAD > 0.1; store span, rep frame) | `SELECT` finds typed sentence within 5 s of typing |
| 4–6 | `pack.py`: every 5 min encode spool → HEVC chunk (HW on Mac: `-c:v hevc_videotoolbox`, `-g 60`), keep WebP keyframe per episode, delete PNGs; `frame_locator(frame_id) → (chunk, offset)` | disk ≤ 1.2 GB/16 h; frame fetch < 300 ms |
| 6–9 | Hybrid search API: `/search?q&from&to&app` = FTS5 ∪ dense (top-50) → rerank by BM25+cosine+recency → episodes; `/frame/{id}` | 16-question gold set ≥ 80 % hit@5 |
| 9–12 | `/ask`: planner prompt → filters+queries → evidence pack (episode text + ≤4 keyframes) → LLM → answer with `[frame 123 @ 14:32]` citations; `/summary?day` journal | answers cite real frames; no answer when no evidence |
| 12–15 | UI: timeline (episodes as blocks, click → keyframe + OCR), chat with citations, pause toggle, exclusion list; MCP server exposing search/sample/get_frame/summarize | demo: "what was the invoice number I typed", "what did I do after lunch", Claude Code uses the MCP |
| 15–17 | Aging cron: re-encode chunks > N days to AV1/1280; delete-range tool that removes frames/text/vectors/chunks; storage dashboard | GB/day projection on dashboard |
| 17–18 | Demo script, README with the numbers above, fallback video | |

Cut order if behind: aging cron → MCP → summary → UI polish. Never cut: capture, OCR+FTS, cited answers.

## 6. Open questions you must answer *during* the hackathon (decide in the first hour)
1. Mac only? (ScreenCaptureKit + AX + HW HEVC make Mac the only sane 18-h target.)
2. LLM: cloud (fast, better) vs local Ollama (privacy story)? Recommend cloud for the demo with an "excluded apps never leave the device" rule; text-only evidence sent by default, images only on explicit visual questions.
3. Focused display only (Rewind) or all displays? Focused only.
4. Capture cadence: 1 fps + events, or 2 s like Coast? 1 fps + events (we measured the cost; it's fine).
5. Which demo questions? Pick 10 covering exact (numbers/URLs), semantic, temporal, summary — build the gold set first and test against it hourly.
6. Retention UI promise: "pixels 14 days full, 100 days reduced, text forever" — state it in the UI.

## 7. Everything else you'll need to answer before this is a product (appendix)
See `plan_questions.md` (A product/wedge · B capture · C compression/representation · D retrieval/LLM · E privacy/security/legal · F evaluation) — ~70 questions. The ones with the biggest consequences: who is user #1 and what is the *daily* loop (Coast's reviewer: recall ≈ 0/week, accounting is daily); malware-with-user-privileges threat model (Recall's unsolved problem); other people's data in calls/chats; EU/UK workplace monitoring (Limitless left the EU); deletion guarantees across derived tiers; model-version migration of embeddings/tokens.

## 8. Limitations of these measurements
Single display, X11 VM, synthetic Chrome-only workload (no IDE/terminal/Slack/spreadsheets), 13.7 min extrapolated to days, software encoders (HW encoders are ~10–30 % less efficient at equal quality), Tesseract/RapidOCR rather than Apple Vision, no audio. Treat GB/day as ±50 % until measured on a real Mac week.
