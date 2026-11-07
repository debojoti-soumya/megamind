# Open questions catalogue (must be answered before/while building)

## A. Product / wedge
1. Who is user #1: solo developer? founder/knowledge worker? researcher? Enterprise employee (consent regime changes everything)?
2. Primary job: episodic recall ("what was that link") vs continuous accounting ("where did my week go") vs agent memory (Claude Code/Cursor reading your context) vs workflow automation (Coast's stated endgame) vs meeting memory (where Rewind/Limitless found demand)?
3. What is the daily-use loop? Rewind/Coast evidence: pure recall is used rarely; products die if the only loop is "search when you forgot". Candidates: morning brief, end-of-day journal, auto time-sheet, "resume where I left off", agent context injection.
4. Which questions must be answerable *exactly* (invoice #, URL, code snippet, number in a spreadsheet) vs approximately (what was I working on Tuesday afternoon)? This decides which tiers are allowed to be lossy.
5. Surfaces: timeline UI, chat, CLI (agent-friendly like Coast), MCP server, browser extension, OS-level "ask about my screen" hotkey?
6. Platform order: macOS (ScreenCaptureKit + AX, Neural Engine, FileVault) first? Windows (competes with Recall) ? Linux (fragmented: portal/PipeWire/Wayland)?
7. Is audio (mic/system/meetings) in v1? Doubles value and quadruples legal risk (two-party consent, EU).
8. Pricing/what is paid: local-only product has no server cost but needs a reason to pay (models? sync? team features?).
9. Retention default and how users understand it (e.g., "pixels 14 days, text forever").
10. What is the "undo": can a user delete a timespan and prove everything derived from it is gone?

## B. Capture
1. Display scope: all monitors, focused display only (Rewind), focused window only (Recall tab-level)?
2. Sampling: fixed interval (2 s Coast/Rewind, 5 s Pensieve) vs event-driven (Screenpipe: focus change, click, scroll end, typing pause, clipboard, idle heartbeat) vs OS dirty-region driven?
3. What signal says "nothing changed" cheaply (hash of downscaled frame, OS dirty rects, AX tree hash)?
4. Metadata per frame: app bundle id, window title, URL (browser extension vs AX vs AppleScript), tab id, document path, git branch, clipboard, input-event counts, idle state, geolocation/network (wifi SSID = place), calendar event.
5. Accessibility tree: capture always, on change, or only when OCR confidence is low? How big is it (KB/frame)? How to diff it?
6. Which apps break AX/OCR: games, video, remote desktop, Electron canvas (Figma), terminals with ligatures, dark-mode low-contrast?
7. Audio: system audio vs mic, VAD gating, speaker diarization, meeting-app detection, consent prompt.
8. Lifecycle: sleep/wake, lock screen, display hot-plug, resolution change, disk-full, battery saver, thermal throttle -> degrade gracefully (lower rate, text-only mode).
9. Resource budget: CPU % (Rewind 20-40% core), RAM (Screenpipe ~600 MB), battery delta/hour, disk write amplification (SSD wear at 1 write/s for years).
10. Capture pipeline must never block UI thread; what is the back-pressure strategy when OCR/VLM lags?

## C. Compression / representation
1. What fidelity target per tier: pixel-exact (lossless tiles) / visually lossless (HEVC CRF ~24) / OCR-preserving (HEVC CRF 30-34 @1920) / thumbnail (recognise the screen) / text-only / summary-only?
2. What fraction of time is the screen unchanged? (Measured here: see tiles.py). Dedup is the biggest lever.
3. Still images vs video chunks: random access + simpler deletion vs 5-20x better ratio.
4. Codec: H.264 (HW everywhere), HEVC (HW on Apple, best Mac option), AV1 (best for screen content: palette + IntraBC; SW-only on Mac). Keyframe interval vs seek cost.
5. Resolution policy: full-res for N days -> 1920 -> 1280 as content ages ("memory forgetting", DeepSeek-OCR idea applied to pixels).
6. Lossless-change-tile tier for recent window (RDP-style XOR+zstd): bytes/hour on real work? Breaks on video — detect and switch to codec.
7. Text tier: OCR bbox+text per frame deduped against previous frame (only new lines), zstd dictionary compression. Does text tier approach ~1 MB/day?
8. AX JSON tier: role/name/value/bounds; diff-encode. Compare to OCR on token count & accuracy.
9. Embedding tier: which model (MiniLM 384-d / bge-small / nomic / SigLIP for images / ColPali patches)? fp16 vs int8 vs binary; Matryoshka truncation; one vector per frame vs per segment vs per text block.
10. Visual-token tier (the "tokenization" research): store model-ready visual tokens (SigLIP/FastViT patch features, DeepSeek-OCR style 64-400 tokens per screen) so that queries can run a VLM over hundreds of screens without re-encoding pixels. Which encoder, how many tokens, which quantizer (FSQ/int8), is it readable 2 years later (model versioning)?
11. Adaptive tokenization for GUI: variable-density tokens (AQuaUI quadtree), region-of-interest from OCR bboxes/AX bounds, temporal reuse of unchanged regions.
12. Bounded memory (StreamMem/KeyRec): per-session fixed token budget with add/merge/evict by novelty — how to make eviction decisions query-agnostic without losing rare-but-important moments?
13. Generative/semantic codecs (diffusion reconstruct from text+sketch): tempting but hallucinates — acceptable only for thumbnails, never as evidence.
14. How to version every derived representation (ocr_v, emb_model, tok_model) and re-derive from pixel tier when pixels still exist, or accept frozen quality when they don't.
15. Where does the 100-day/100 GB budget go: pixels vs text vs vectors vs tokens? (Budget table in plan.)

## D. Retrieval / query / LLM
1. Query types: exact lookup, semantic lookup, temporal ("Tuesday after lunch"), aggregate ("how long on X"), visual ("the chart with red bars"), procedural ("how did I fix the build"), cross-session ("what was I doing before I got distracted").
2. Query planner: LLM -> structured plan (time filter, app filter, FTS terms, embedding query, need_pixels?) -> hybrid retrieval -> rerank (cross-encoder / ColPali late interaction / VLM) -> evidence pack -> answer with citations (frame ids + timestamps + crops).
3. Segmenting the stream into "episodes" (app+title+URL+visual change) -> one record per episode with time span, text, representative frame. Episodes are the retrieval unit, frames are the evidence unit.
4. How many frames/crops can the LLM see per answer (local 7B VLM budget ~4-8 images)? -> need crop-level retrieval (OCR bbox -> crop) not whole screens.
5. Cross-attention search: (a) late-interaction over patch tokens (ColPali) for visual queries; (b) query-conditioned token selection over stored visual tokens before VLM; (c) cross-encoder reranking over text. Which gives best accuracy per ms on CPU/ANE?
6. Local vs cloud model: default local (privacy promise), optional user-approved cloud for hard questions with redaction of excluded apps.
7. Confidence/uncertainty: when OCR disagrees with AX, when no evidence found; never fabricate — show "no memory of this".
8. Summaries (daily journal) are derived artefacts; regenerate when models improve; store generation prompt + model version.
9. Agent actions on memory (Coast's direction): permission model, dry-run, confirmation.
10. Latency targets: FTS <50 ms, semantic <300 ms, VLM answer <10 s on M-series.

## E. Privacy / security / legal
1. Threat model: lost laptop (FileVault+own encryption), malware with user privileges (Recall's "by design" problem — anything the user can read, malware can read), malicious local app querying the API, cloud-sync leaks, backups (Time Machine), crash dumps/swap/log files, model caches, thumbnails.
2. Encryption: SQLCipher/age-encrypted chunks, keys in Secure Enclave/TPM tied to biometric; unlock-on-query vs always-unlocked for indexing (background indexer needs keys -> enclave design like Recall VBS).
3. Exclusions: apps, sites/domains, private windows, categories (banking/health/adult), password fields via AX (secure text fields are already hidden), meeting apps unless consented. Default-on list.
4. Sensitive data detection before persistence: regex SITs (cards, SSN, API keys), ML PII, on-screen password managers. Known failure modes (Recall missed bank routing numbers). Policy when detection is uncertain: drop frame, keep text-only, blur region?
5. Other people's data: screen share, chats, video calls — their faces/words captured without consent. Blur faces? Don't store call windows? Legal exposure differs by jurisdiction.
6. Workplace: UK ICO/GDPR Art 88/BDSG/French CSE: DPIA, lawful basis, employee information. Limitless pulled out of EU. Decide: consumer-only initially?
7. Audio: two-party consent states, wiretap laws; Zoom notice requirements.
8. Deletion guarantees: delete timespan -> pixels, OCR, embeddings, tokens, summaries, FTS index, WAL, caches. Need tombstones + verification tool. Video chunks containing the span must be re-encoded or chunked small enough.
9. Local API/MCP auth: unix socket with peer-pid check + per-client tokens + scopes (read text, read pixels, read excluded apps = never).
10. Visible indicator, global hotkey pause, auto-pause rules (screen share on, private window, specific wifi).
11. Export: open format (SQLite + media) so users aren't locked in; Rewind couldn't export video.
12. Multi-user / shared machines: per-OS-user stores; never cross.

## F. Evaluation
1. Corpus: real multi-day logs from consenting testers across browsing/coding/spreadsheets/chat/meetings/video/idle; plus synthetic scripted sessions with ground truth (this benchmark).
2. Question set: exact (numbers, ids, URLs), semantic, temporal, aggregate, visual; ~200 Qs with gold evidence frame ids.
3. Metrics: storage GB/active-hour and GB/calendar-day; Recall@k of gold frame/episode; exact-answer accuracy; OCR word recall vs lossless; hallucination rate (answers with no supporting evidence); time-to-index; query latency; CPU/battery; privacy FP/FN on seeded secrets.
4. Quality-vs-bytes curve for each tier; pick operating point for 100 GB / 100 days.
5. Worst cases: full-screen video, animated dashboards, multi-monitor, 4K/5K Retina, fast scrolling, dark themes.
6. Model-upgrade regression: re-derive tiers and compare answers.
