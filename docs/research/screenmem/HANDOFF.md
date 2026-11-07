# HANDOFF — full context for screenmem

Origin: Devin session https://app.devin.ai/sessions/c87d8bdda7bf4c09a038cb6b6782aaf6 (Debojoti's personal org, 2026-11-07).
This document is the complete brain-dump: thesis, research, measured numbers, decisions, architecture, Mac state, gotchas, next steps.
Detailed research lives in `docs/research/` (PLAN.md = hackathon plan; APPENDIX_open_questions.md = full question catalogue;
APPENDIX_research_notes.md / research_notes.md = literature + competitor notes). Raw benchmark results + scripts: `docs/bench/`.

---
## 1. What we are building (and why)
A **local-first screen memory for macOS**, inspired by Coast by Attention / Rewind / Microsoft Recall / Screenpipe:
record the screen continuously, compress it hard, turn it into text + vectors, and let an LLM (and agents via MCP) answer
*any* question about what the user did on their computer, with cited frames.

Hackathon framing (18 h, not a company; must be fun, nice, somewhat unique). Chosen angle:
> **"Coast indexes your screen. We tokenize it."** Screen history as *model context*, not a search index: pixels age down through
> codec tiers (lossless → HEVC → AV1 → text-only) while exact text (OCR + Accessibility tree) and small retrieval vectors are kept forever;
> episodes (not frames) are the unit the model attends over; agents consume it via MCP.

Demo beats (in priority order): ask-bar with clickable cited frames → episode timeline + "Resume where I left off" → end-of-day journal →
Claude Code/Cursor asking the MCP ("what was the error in my terminal 20 min ago?") → live storage/fidelity dashboard with 100-day projection →
(stretch) visual questions via query-time VLM, day-to-video recap, déjà-vu alerts.

### Explicit user requirements
- Episodes, not frames: segment by (app, title/URL, visual change) → record with span, text, representative frame. Done (`episodes` table).
- Resume card + end-of-day journal = one LLM call over episodes. Done (`ask.resume_card`, `ask.daily_summary`).
- AX tree as second text source (exact values/URLs; OCR fallback). Done (`ax_mac.py`), password fields skipped.
- "It has to work really well and answer all sorts of questions" → query planner + gold set (see §6).
- Local-first privacy: only OCR/AX text goes to the LLM; images only for explicitly visual questions; exclusions + pause + delete-range.

---
## 2. Competitor / landscape notes (summary; details in docs/research)
- **Coast by Attention**: Mac, local, continuous capture, "ask anything", heading toward agent memory/automation; reviewer critique: recall tools
  lack a continuous-value loop (nothing comes back to you unless you ask) → our journal/resume card answer that.
- **Rewind** (now Limitless): Mac, 1 fps-ish, local SQLite+FTS, compressed ~3750× via dedupe + HEVC; pivoted to pendant/audio.
- **Microsoft Recall**: Windows Copilot+ PCs, snapshots every few seconds, OCR + semantic index, big privacy backlash → opt-in, encrypted, filters.
- **Screenpipe**: open-source Rust, screen+audio 24/7, OCR (Apple Vision on Mac), SQLite FTS, plugins ("pipes"), AX capture; ~15 GB/month.
- Everyone does screenshot → OCR → FTS. Nobody exposes fidelity aging, episodes as model context, or agent-native MCP as the primary surface.

---
## 3. Storage research & benchmark (the "100 days in 100 GB" question) — ANSWERED: yes
Benchmark on the Devin VM: 2560×1920 synthetic-but-realistic desk work (browsing Wikipedia/docs, terminal, editor typing, a full-screen video phase), 1 fps, 13.7 min, 823 frames; results in `docs/bench/res_*.json`.

| strategy | GB per 16-h day | note |
|---|---|---|
| raw PNG | 64.7 | ground truth |
| H.264 full-res crf28 | 2.43 | |
| H.264 1280w crf28 | 0.87 | |
| HEVC full-res crf28 | 2.85 | x265 sw |
| HEVC 1920w crf30 | 1.64 | |
| HEVC 1280w crf30 | 0.92 | |
| AV1 1920w crf40 | 0.78 | no change detection |
| **HEVC 1920w crf30 + change detection** | **1.14** | hot tier |
| **AV1 1920w crf40 + change detection** | **0.28** | cold tier |
| AV1 crf48 + change detection | 0.19 | floor before OCR recall drops |
| lossless tiles (XOR+zstd) | fine for static text, **161 GB/day during video** | rejected as sole representation |

- 70.2 % of consecutive 1-fps frames were byte-identical; mean changed-tile fraction 9.6 %; ~19 % of frames survive change detection.
- Live on Debojoti's Mac (3024×1964 → 1920w HEVC videotoolbox q45): **16 KB/frame**, ~800 kept frames/hour → projection **0.7–0.9 GB per 16-h day**; dashboard projects 100 days live.
- Text + AX + MiniLM vectors ≈ 5–8 MB/day (SQLite). Kept forever.
- Budget: 14 days hot HEVC (~15 GB) + 86 days cold AV1 (~26 GB) + text/vectors (<5 GB) ≈ **55 GB / 100 days**, with headroom for video-heavy days.
- **Change-detection threshold trap**: at the "obvious" threshold the typing phase lost 47/49 frames and 2/5 typed-note questions became unanswerable.
  Fix: low pixel threshold (0.002 mean abs diff on a 64×48 gray thumb) + focus-change trigger + 30 s heartbeat; (keystroke/clipboard triggers are TODO).

### OCR fidelity vs compression
- **Trap**: RapidOCR default `limit_side_len=736` silently downscales 2560-px frames → bogus fidelity numbers. Re-ran with full-res Tesseract (`OMP_THREAD_LIMIT=1` when parallel).
- Corrected: 90–93 % OCR word recall across practical lossy codecs; AV1 1920w crf40 keeps ~90 % at 37 KB/frame. **Downscaling hurts recall more than codec quantization.**
- Design rule: **OCR the lossless frame at capture time; never OCR from the archive.** Archive pixels are for humans/VLMs, text is the truth.
- Apple Vision OCR on M4 Pro: 0.45 s/frame warm (11 s cold start), ~550 words/128 boxes on a browser screenshot; chosen for Mac.

### Retrieval quality (16-question gold set on benchmark data)
- FTS5: hit@1 87.5 %, hit@5 93.8 %. MiniLM (all-MiniLM-L6-v2) dense: hit@1 68.8 %, hit@5 100 %. **Union hit@5 = 100 %.** → hybrid with RRF.
- VLM captions (query-time, gpt-4o-mini class) useful for coarse ranking but hallucinated a page topic → ranking signal only, never evidence.

---
## 4. Context-aware compression / tokenization research — conclusions
(see docs/research/APPENDIX_research_notes.md for citations: TiTok, Cosmos tokenizer, FSQ/VQ, ToMe token merging, ColPali/ColQwen late interaction, StreamMem/KeyRec KV memory, DeepSeek-OCR "optical compression", semantic codecs)
- Persisting VLM visual tokens is **bigger** than compressed pixels (~164 KB/frame for a 2k-token fp16 sequence vs 5–37 KB HEVC/AV1). Storing tokens loses the storage argument.
- Therefore: **store pixels in a codec (cheap, lossy, aged), store exact text (AX+OCR) and small int8/binary vectors for retrieval, compute visual tokens at query time** on the few retrieved frames (OCR-bbox-guided crops to cut tokens 5–10×).
- Multi-vector (ColPali-style) patch embeddings for *visual* retrieval: ~128 dims × ~1k patches/page; with binarization ≈ 16 KB/frame — viable per episode keyframe, not per frame. Planned as Modal GPU stretch (ColQwen2).
- A GUI-specific discrete tokenizer (learned, text-aware, 100–300 tokens/screen) is the real research item post-hackathon; the "optical compression" line (DeepSeek-OCR: 10× fewer vision tokens than text tokens at 97 % decode accuracy) suggests screens could be stored as ~100 tokens each (~200 bytes) with near-lossless text — this is the long-term "100 days as model context" story.
- KV-cache memory (StreamMem etc.) not practical to persist across days (GBs/hour); episodes + retrieval remain the right abstraction.

---
## 5. Architecture (repo layout)
```
screenmem/config.py    paths (~/.screenmem), defaults, .env loading (OPENAI_API_KEY), config.json overrides
screenmem/db.py        SQLite schema: frames (evidence), frames_fts (FTS5), episodes, chunks, events, summaries
screenmem/capture.py   1 fps screencapture (mac) / mss / ffmpeg x11grab; change detection; focus trigger; heartbeat; app/title/URL via osascript + AX; AX text; exclusions; pause
screenmem/ax_mac.py    focused-window AX text walk (title/value/description; skips AXSecureTextField), URL via AXDocument/AXURL/web area/address bar
screenmem/indexer.py   Apple Vision OCR (Tesseract elsewhere) → ocr_text/boxes; episode assignment (app+domain+title key, 120 s gap); MiniLM embeddings
screenmem/packer.py    spool PNG → HEVC chunk (hevc_videotoolbox q45, 1920w, 1 fps CFR; libx265/libsvtav1 elsewhere) + WebP keyframe per episode; extract_frame by index
screenmem/search.py    time-phrase parser, app hints, FTS (quoted phrases / AND-first then OR) ∪ dense, RRF, per-episode diversity cap
screenmem/ask.py       planner routes fact / activity / visual; evidence pack; cited answers [frame:ID]; daily_summary; resume_card; images only on visual route
screenmem/api.py       FastAPI: /api/search /ask /episodes /frame/{id}(.jpg) /summary /resume /stats /pause /exclude /delete_range; serves ui/
screenmem/mcp_server.py MCP tools: search_screen, ask_screen, timeline, get_frame_text  (claude mcp add screenmem -- .venv/bin/python -m screenmem.mcp_server)
ui/index.html          ask bar, cited thumbnails, episode timeline, resume card, journal, storage bar + 100-day projection, pause, frame viewer
setup_mac.sh / run.sh  Mac setup probes; launch capture+indexer+packer+API (port 8787)
```
Data: `~/.screenmem/{memory.sqlite, spool/, chunks/, keyframes/, logs/, config.json}`.

---
## 6. Current state on Debojoti's Mac (M4 Pro, macOS 26.2, 20 GB free disk)
- Checkout `~/screenmem-app` tracks `main`; venv `.venv` (Python 3.12 via uv; anaconda python lacks pyobjc frameworks); ffmpeg via brew (hevc_videotoolbox + libsvtav1).
- Permissions (Screen Recording, Accessibility, Automation) granted to **Ghostty**; everything is launched from Ghostty.
- Running since 2026-11-07 ~11:50 PST: capture, indexer, packer, API (http://localhost:8787). First chunk: 87 frames → 1.4 MB.
- Verified: screencapture 3024×1964 ✓, app/title ✓, URL (Zen via AXDocument) ✓, AX text 4.8k chars/0.3 s ✓, Vision OCR ✓, HEVC HW ✓, FTS search on real data ✓.
- **Missing: `~/screenmem-app/.env` with `OPENAI_API_KEY=`** (a key was pasted in chat → told to revoke; new key must be created by Debojoti). Without it /ask, /resume, /summary return raw evidence.
- Gold set (`docs/gold_questions.json`, `python -m screenmem.evalset`): 8/8 hit@1 on real Mac data (2026-11-07 12:06).
- Fidelity aging implemented (`packer.age_chunks`, runs in the packer loop; `POST /api/age?force=true` = demo button in the dashboard): hot HEVC -> cold AV1 -> text-only. **Measured on a real Mac chunk (108 frames):** HEVC videotoolbox q45@1920 = 13.7 KB/frame; AV1 crf45@1920 = 12.2 KB (useless), crf52 = 8.1 KB, crf58 = 5.2 KB, crf45@1280 = 6.3 KB -> cold tier set to **crf50 @1280 (~4.5 KB/frame, ~3x)**. Lesson: the HW HEVC hot tier is already aggressive, so cold must downscale, not just re-quantize (OCR already happened on lossless pixels, so recall is unaffected).
- CLIP ViT-B/32 image embeddings (`frames.img_emb`, fp16 512-d = 1 KB/frame) computed at index time from the lossless frame, backfilled from keyframes/chunks for old frames; fused into `search()` as a third RRF signal (weight 0.6). Mac: 0.2 s/img. Makes retrieval work for frames with no text (glasses/phone video).
- `screenmem/ingest_video.py`: source-agnostic ingester (any mp4 -> change-detected 1 fps frames with `source`, optional whisper transcript -> `frames.transcript`, searchable as "[speech] ..."). Same indexer/episodes/tiers/search downstream.
- Also done: episode-level retrieval (`search_episodes`, used by the activity route for topic questions without a time range); CLIP episode split (`clip_split`); zero-shot CLIP scene labels as camera episode titles; `clean_ocr`.
- Not yet done: UI polish; MCP demo with Claude Code; keystroke/clipboard triggers; speech-topic episode split; int8 vector quantization; real Meta glasses export tested.

### Camera-video compression (for glasses) — measured 2026-11-07 on 1080p test footage (Jellyfish = real camera, Big Buck Bunny = animation), 2 fps samples, bytes/frame
| encoder | Jellyfish | BBB |
|---|---:|---:|
| x265 1920 crf28 | 69 KB | 170 KB |
| x265 1280 crf30 | 36 KB | 61 KB |
| AV1 1280 crf40 | 10.8 KB | 15.8 KB |
| AV1 1280 crf50 | 6.1 KB | 7.6 KB |
| AV1 960 crf50 | 4.4 KB | 4.5 KB |
Change detection (0.02 on 64x40 gray) keeps 100% of real camera frames (vs 19-30% on screens) → glasses ≈ 57.6k frames per 16 h at 1 fps.
Per 16 h day: HEVC 1280 = 2.1–3.5 GB (too much for hot tier at 100 days), AV1 1280 crf40 = 0.6–0.9 GB, crf50 = 0.35–0.44 GB.
**Glasses tiers:** hot = HEVC 1280 crf30 for 3 days (~7–10 GB), cold = AV1 1280 crf50 (100 days ≈ 35–44 GB), text/speech/CLIP vectors forever (~2.5 KB/frame → 14 GB/100 d; quantize to int8 to halve). 100 days fits in 100 GB with ~2× headroom; at 0.5 fps it's trivial. Camera frames carry far less text than screens: budget OCR only for frames whose CLIP says "document/screen/sign", rely on CLIP + speech for the rest.

### Glasses path validated on test footage (2026-11-07 12:25, VM)
`ingest_video.py` on Jellyfish (real camera) + Big Buck Bunny (+ JFK speech track muxed in) → 30 frames, 3 episodes (split by CLIP, no metadata), faster-whisper `small` transcript attached to frames → `docs/gold_glasses.json` **8/8 hit@1** (visual-only questions via CLIP, speech via FTS on `[speech]`). OCR on camera frames is mostly noise → `clean_ocr(strict=True)` for non-screen sources.

### Direction change (2026-11-07 12:20): Meta glasses, not screen
Debojoti: "we are planning to switch to meta glasses instead of screen ... iterate on the common part instead of the screen part ... work more on compression, embedding and query things". So: the screen recorder is now just one *sensor*; invest in `db/indexer/packer/search/ask` (tiers, embeddings, planner) and in `ingest_video.py` as the glasses path. Glasses differences to design for: no app/title/URL/AX metadata (episodes must split on visual/CLIP change + speech topic), audio transcript is a first-class text source, frames are photos (CLIP matters more, OCR less), higher per-frame entropy (camera noise -> change detection needs a higher threshold, codec rates ~3-5x screen).

### Gotchas (hard-won)
- AX: `AXUIElementCreateSystemWide` + `kAXFocusedApplicationAttribute` → error -25204; use `NSWorkspace.frontmostApplication().processIdentifier()` + `AXUIElementCreateApplication(pid)`.
- osascript: multi-line `tell application appVar` fails to compile browser terminology → one literal script per browser (`capture.BROWSER_URL_SCRIPTS`); Zen/Firefox have no AppleScript URL → AX `AXDocument`.
- macOS lacks GNU `timeout`. mss fails on the VM's X11; ffmpeg x11grab fallback.
- Devin local-machine commands get interrupted by new chat messages → keep them short, re-run.
- Zen's AX text includes the whole tab sidebar (all tab titles) → noisy but searchable; consider weighting OCR over AX for "what was on screen".

---
## 7. Next steps (priority)
1. `.env` key → verify /ask, /resume, /summary end-to-end on real data; build a 25-question gold set from Debojoti's real day (exact terminal errors, pages read, "what was I doing at …", URLs, resume); measure hit@1/5 and citation correctness; fix planner failures.
2. Episodes for text-less sources: split on CLIP cosine < ~0.8 to the episode's rep frame + speech-topic shift; rep frame = medoid in CLIP space. (Screen episodes already rejoin after brief switches and split on OCR-text change.)
3. Embedding quality: evaluate CLIP vs SigLIP-2 (better text-in-image) on a glasses-style gold set; int8/binary quantize vectors; episode-level embeddings (mean of frames) so the planner can retrieve episodes directly.
3b. Compression for camera video: re-measure tiers on a real glasses clip (expect HEVC ~40-80 KB/frame at 1080p; test 720p cold tier); scene-cut-aware chunking.
4. MCP demo with Claude Code; hotkey ask-bar (thin window over localhost:8787); menu-bar (rumps) if time.
5. Stretch: ColQwen2 visual retrieval over episode keyframes (Modal), day-to-video recap, déjà-vu matching, Universal Constructor corpus (only with privacy decision).

## 8. Credits available (user): Modal, OpenAI API, Sail, Warp, Universal Constructor. Only OpenAI is wired (gpt-4o-mini text; gpt-4o for visual route).
