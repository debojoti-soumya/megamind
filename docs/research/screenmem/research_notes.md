# Research notes (facts gathered, with sources)

## Coast (Attention Engineering / Attention Inc.)  coast.app
- Launched local preview Jul 28 2026 (Aidan Guo, Julian Windeck). ~5 people, $1.25M pre-seed (Village Global, Liquid 2). Stated direction: desktop-native AI assistant that learns workflows & automates repetitive tasks; consumer recorder = substrate/groundwork. (cronushq review, runtimewire)
- Mac-only; "fully local memory for you and your agents"; local inference on Apple Neural Engine.
- Mechanics (cronushq review): screenshot ~every 2 s; OCR per frame; FTS index of text + metadata (bundle ID, web domain, window title); optional macOS accessibility tree; CLI `~/.local/bin/coast`: `coast query fts`, `coast usage top-applications`, `coast query sample --tr` (one representative frame per activity segment), `coast query image`. Timerange syntax; filters automatch (safari -> com.apple.safari).
- NOT a chatbot: no NL question box. Agents (Claude Code/Codex/Cursor) shell out to the CLI. Index is dumb+local; intelligence lives elsewhere.
- FAQ: nothing uploaded; pause; exclude apps/websites/categories; incognito auto-excluded; retention period + storage limit, "older video is compressed or permanently deleted". Recommends FileVault. Cannot recover captures.
- Critique (Cronus): recall demand is rare/episodic ("approximately zero" unprompted uses in weeks); accounting (where did my week go) is continuous demand. App-level time totals ("Chrome 2h36m") don't answer "doing what". Privacy bargain: local-and-total = catastrophic blast radius, low probability.

## Rewind (2022-2025, -> Limitless -> acquired by Meta Dec 2025; Rewind capture disabled Dec 19 2025)
- Teardown (kevinchen.co): ScreenCaptureKit screenshots every 2 s of focused display; Apple Vision OCR; PNG temp then FFmpeg H.264 chunks ~5 min long at 0.5 fps; SQLite (frame, segment, search_content, node w/ bbox). Raw PNG 1-2 MB/s at 3024x1964 text; >10 MB/frame on video.
- Claimed 3,750x compression (10.5 GB raw -> 2.8 MB, 1,302 frames). Real-world: avg 14 GB/month (range 1-39 GB/month); users reported 12 GB/month, 24 GB over 2 months. CPU 20-40% of one core while recording. Added own encryption on top of FileVault. No video export.
- Pivot to Limitless pendant Apr 2024 (meeting transcription), Meta acquisition Dec 2025; service pulled from EU/UK/Brazil/China/Israel/Korea/Turkey (regulatory). Lesson: desktop recall alone didn't sustain a business; audio/meetings had clearer demand.

## Microsoft Recall
- Default storage allocation 25 GB (256 GB SSD) / 75 GB / 150 GB; retention 30/60/90(default)/180/unlimited days; "25 GB stores ~3 months".
- Security redesign (Sep 2024): opt-in; snapshots+vector DB always encrypted; keys in TPM tied to Windows Hello ESS; processing in VBS enclave; rate-limiting/anti-hammering. TotalRecall (2024) dumped plaintext SQLite; TotalRecall Reloaded (Apr 2026) rides along user auth and extracts everything -> MS says "by design".
- Sensitive Information Filtering (Purview SIT classifiers: credit cards, SSNs, API keys, 100+ types) default on; hit-and-miss (Ars: checks w/ routing numbers slipped through). App/website exclusion, in-focus-tab only; embedded content leaks.

## Screenpipe (YC S26, Rust, source-available)
- Event-driven capture (app switch, focus, click/scroll, typing pause, clipboard, idle fallback ~5 s) + accessibility tree walk at same timestamp; OCR fallback (remote desktop, games); JPEG frames + SQLite FTS; audio 30 s chunks w/ speaker ID (Whisper/Parakeet); PII redaction model (MLX/DirectML); API port 3030 + MCP. Claims ~600 MB RAM, 5-10 GB/month (event-driven) vs older 5-10 GB/day; HN criticism: consent, local security, CPU ("space heater"), signal-to-noise, agents acting on data. Enterprise: workflow discovery.
- Naive "record video + OCR every frame" abandoned: duplicate data, CPU heat, discards OS structure.

## Others
- OpenRecall (AGPL, Python, doctr OCR + sentence-transformers, change detection). Pensieve (Apache, every 5 s, 40-400 KB/shot, ~400 MB/day at 10 h 2560x1440, ~8 GB/month; optional VLM via Ollama; structured VLM JSON plugin). Windrecorder (Windows, 3 s screenshots -> video every 15 min, index only changed scenes, OCR engines RapidOCR/WeChat/Tesseract, activity stats). Rem (Mac, 2 s, dead 2024). Dayflow (Mac, AI-written timeline). Cronus (time tracker: reads title/url/text, keeps no screenshot, syncs metadata).
- mech.app estimate: 1 shot/2 s @200 KB -> ~8 GB/day; audio 16 kHz mono ~1.4 GB/day.

## Capture APIs
- macOS ScreenCaptureKit (window exclusion, zero-copy), Windows Graphics.Capture (dirty regions metadata, D3D11), Linux xdg-desktop-portal + PipeWire (DMA-BUF). Rust: scap (CapSoftware), windows-capture, screencapturekit-rs. Dirty rects = OS refresh hints, good side-info for change detection and encoder guidance (Fluendo VVenC prototype).
- Hardware codecs: Apple Silicon has HEVC/H.264 encode, AV1 decode only (M3+), NO AV1 encoder; HEVC SCC not exposed by VideoToolbox. AV1 main profile includes palette mode + IntraBC (screen-content tools) — great for text/UI but software-only on Mac; Intel Arc/NVIDIA 40xx/AMD 7000 have AV1 HW encode.

## Legal/consent
- UK ICO monitoring guidance: DPIA, lawful basis, least-intrusive means, inform workers. GDPR Art 6/13/88; CJEU C-34/21 weakens §26 BDSG. France: inform/consult CSE. Two-party consent states (US) for audio; Zoom calls recorded without notice = exposure. Limitless withdrew from EU/UK.

## Retrieval / tokenization literature
- DeepSeek-OCR (Oct 2025): "contexts optical compression" — 10 text tokens -> 1 vision token at 97% decode precision (<10x), ~90% at 10-12x, ~60% at 20x. Tiny 512px=64 tokens, Small 640=100, Base 1024=256, Large 1280=400 tokens. Proposes memory-forgetting by progressively downscaling older pages (fewer tokens). Directly relevant: store old screens as low-token "optical" representations.
- ColPali/ColQwen2: ~1,030 patch vectors x128-d per page (late interaction). Qdrant pooling 1030->38 vectors; Light-ColPali keeps 98.2% perf at 11.8% memory, 94.6% at 2.8%.
- Embedding quantization (HF): int8 4x smaller ~no loss; binary 32x, ~92.5% retained (96% w/ rescoring). Matryoshka dims: 8x dim cut + int8 = 32x at 94.4%.
- Visual tokenizers: TiTok 1D: 256x256 image -> 32-128 discrete tokens (recon FID ~1-2). Cosmos tokenizer: 8x/16x spatial, 4x/8x temporal, up to 2048x; continuous or discrete (FSQ). Trained on natural imagery — text legibility not a design goal.
- GUI token reduction: Qwen3-VL encodes 1920x1080 screenshot into ~2,040 visual tokens; AQuaUI adaptive quadtree keeps 99% perf at -29.5% tokens (training-free; uses non-uniform density of GUI screenshots + temporal continuity across consecutive screenshots); STP-GUI 21% tokens keeps 77.7%.
- Streaming memory: StreamMem (query-agnostic KV cache compression via cross-attention saliency to generic query tokens, fixed-size memory); KeyRec (recent visual cache + event bank w/ novelty-based add-merge-evict; 10% token budget); Video-EM (episodic memory, keyframes as temporally ordered episodes).
- Generative/semantic codecs: DiSCo (CVPR 2026) text + degraded video + sketches -> diffusion reconstruct, 2-10x better perceptual at low bitrates; GVSC. Caveat: hallucinated detail unacceptable for a memory system (needs faithful text).
- Screen content coding: HEVC SCC/AV1 palette + IntraBC; remote-desktop codecs: dirty tiles + XOR vs previous + ZRLE/zstd lossless (kilobytes/frame for IDE/terminal/browser; breaks on video).
- FastVLM (Apple, CVPR 2025): FastViTHD encoder, fewer tokens, 85x faster TTFT than LLaVA-OV-0.5B; 0.5B/1.5B/7B MLX+CoreML checkpoints -> on-device screen captioning realistic on Mac.
- Accessibility tree vs OCR vs VLM: complementary "router"; a11y ~100x faster, deterministic, structured; fails for remote desktop/games/canvas apps.
