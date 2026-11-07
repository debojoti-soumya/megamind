# 02 — Keyframe selection for egocentric capture at ~1 fps

*Research bot (stream-5), 2026-11-07. Applies to: **capture** (stream-1: `headset/.../FrameUploader.cs`, `server/megamind/ingest/filter.py`). Uncertain claims marked (?).*

## Question
We sample the passthrough camera at 1 fps and want to keep only frames worth captioning (each kept frame costs a VLM call + an embedding + storage) while never losing the frame that answers "where did I put X". Which cheap heuristics work on egocentric video, and what do EgoLife / Project Aria pipelines do?

## What the literature and products do
- **1 fps is the industry default for "understanding" sampling.** Gemini's video API samples at 1 fps for descriptions ("works well for most content… may miss details in videos with rapid motion"). EgoLife expands its captions with representative frames *sampled at 1 FPS* and summarises per 30-s clip. Project Aria's default low-power recording profile (profile 0) records **RGB at 1 fps** (8 MP) alongside 10 fps SLAM cams. Nobody in this space tries to caption every frame.
- **Lifelogging cameras (SenseCam / Narrative, 1 frame per 30–60 s) spawned the "photostream event segmentation" literature:** cluster consecutive frames by visual features into *events*, pick one representative keyframe per event (Bolaños et al. 2015, 1505.01130; Garcia del Molino et al. 2018, 1808.02289 — "Contextual Event Segmentation", LSTM predicts visual context; boundary where past-context ≠ future-context; 16 % F-measure over prior clustering). The recurring failure mode they name: **"temporary changes in sight direction, e.g. at a meeting"** fragment one event into many. For us this means: head turns produce near-duplicate *places* seen from different angles — a dHash miss but a semantic duplicate.
- **Long-video QA (2025–26): keyframe selection is query-time, not ingest-time.** FocusGraph (2603.04349) first picks query-relevant *clips* from compact graph captions, then extracts keyframes from those clips with a motion-based retention rule (patch-wise sparse optical flow), then sends ≤ N frames to the MLLM; they show MLLM accuracy *degrades* with more frames. EgoLife stores every 30-s clip caption and lets retrieval choose. ⇒ Over-keeping at ingest is recoverable (retrieval filters); under-keeping is not.
- **Standard low-level filters (battle-tested, OpenCV/PIL only):**
  - *Near-duplicate:* dHash 8×8 (64-bit). Krawetz (author of dHash): Hamming 0 = same, **1–10 = variation of same image, > 10 = different image**. Our threshold `< 6` is on the strict side (keeps more). imagehash lib implements dhash/phash/whash identically.
  - *Blur / motion blur:* variance of the Laplacian on grayscale (Pech-Pacheco et al. 2000; OpenCV `cv2.Laplacian(gray, CV_64F).var()`), typical cut-off ~100 for 640-px photos but **content-dependent — calibrate on your own frames**. Egocentric frames blur during head turns; the blurry frame is also the one whose dHash differs most, so a blur gate *before* the dedupe check prevents "novel because blurry" false keeps.
  - *Scene change:* PySceneDetect `ContentDetector` = mean |Δ| of HSV channels between consecutive frames, default threshold **27.0** (0–255 scale); `AdaptiveDetector` (ratio to rolling mean, default 3.0) exists specifically "to mitigate false detections in situations such as fast camera motions". At 1 fps consecutive frames differ a lot more than at 30 fps, so raise the threshold or use the adaptive ratio.
  - *Dark / covered lens:* mean luminance (we use < 20) — fine; also catches headset-doffed-on-table.

## Concrete recommendations for Megamind (capture)

1. **Keep 1 fps on the headset; do not add client-side smarts.** Headset CPU/GPU is the scarce resource, and the server can always discard. (Gemini/EgoLife/Aria all sample ≈1 fps.)
2. **Reorder the server filter to: dark → blur → too_soon → dedupe**, and add a `"blurry"` reason (needs a one-word addition to the `reason` enum in `docs/api.md`; worth it). Blur gate: `laplacian_var(gray_480px) < BLUR_MIN` with `BLUR_MIN` env-tunable, default ≈ 60–100 (?) — calibrate on 50 real passthrough frames in the first hour of device access (passthrough cams are noisier/softer than photos, so the right value may be lower). Avoid `opencv` as a hard dep if unwanted: PIL `ImageFilter.FIND_EDGES` + variance is an adequate proxy (?).
3. **"First stable frame" heuristic** (what the user actually wants to remember is where the head *settled*, not the pan): keep frame *t* if `hamming(h_t, h_{t-1}) ≤ STABLE_MAX` (≈ 4) **and** the previous frame was *rejected for motion/blur or large change*. Implementation: track `last_hamming` per device in the filter state; a frame that is both sharp and similar to its predecessor, following a frame that was not, is a "settled" frame → accept even if `too_soon`. This replaces a fixed 2-s gap with a motion-aware one.
4. **Two-tier dedupe window.** Keep `hamming < 6 within 30 s → duplicate`. Add a longer, looser tier: `hamming < 3 within 10 min → duplicate` for static scenes (desk, wall) that re-enter view — these are the frames that would otherwise dominate the store and dilute retrieval (lifelog literature's "meeting" fragmentation problem). Optional third tier using SigLIP cosine > 0.97 within 10 min, only if the image embedder is cheap (it is, if run locally) — catches same-place-different-angle duplicates dHash misses. (?) thresholds untested on passthrough footage.
5. **Never drop a frame that contains hands + a new object.** Not implementable cheaply at the filter; instead rely on (3): object placement events ("put keys on counter") are preceded by head/hand motion and followed by a stable view — the settled-frame rule keeps exactly those. If the enricher is async anyway, bias toward keeping (expected 10–30 % keep rate is fine; 1 fps × 30 % × 1 h ≈ 1 000 tokens, trivial for LanceDB and ≈ $0.5 of Qwen3.6 on Sail).
6. **Store the rejected-frame counters** (`dark/blurry/too_soon/duplicate`) in `/health` or logs — this is the first thing to look at when "it didn't remember X" happens in the demo, and judges like seeing the keep rate.
7. **Thumbnail for the reader ≠ frame for the hash.** Compute dHash/blur on a 160-px grayscale downscale (fast, more robust to sensor noise); store the 960-px JPEG as the thumbnail the reader sees (OCR needs resolution — see report 04).
8. **Pose-assisted dedupe (cheap, we already send pose):** if `|Δyaw| < 5°` and `|Δposition| < 0.15 m` vs. the last accepted frame and dHash < 10 → duplicate. Head pose from `OVRCameraRig` is free and far more reliable than image hashing for "I haven't moved".
9. **Do not do event segmentation at ingest.** Episode boundaries (for summaries, report 06) are better computed later from timestamps + caption similarity than from pixel heuristics; the 2018 CES paper needed an LSTM to do it well.

### Suggested defaults (env vars)
```
FILTER_MIN_LUMA=20            # keep
FILTER_BLUR_MIN=80            # laplacian variance on 480px gray; calibrate day-1
FILTER_MIN_GAP_S=2.0          # keep, but bypassed by settled-frame rule
FILTER_DHASH_MAX_DISTANCE=6   # keep (30 s window)
FILTER_DHASH_LONG_DISTANCE=3  # new, 600 s window
FILTER_STABLE_MAX_DISTANCE=4  # new, settled-frame rule
```

## Sources
- Gemini video understanding (1 fps default): https://ai.google.dev/gemini-api/docs/video-understanding
- EgoLife (1 FPS representative frames, 30-s clips): https://arxiv.org/abs/2503.03803 §3
- Project Aria hardware spec / profile 0 (RGB 1 fps): https://facebookresearch.github.io/projectaria_tools/docs/tech_spec/hardware_spec
- dHash and Hamming thresholds (Krawetz, "Kind of Like That"): https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html ; imagehash: https://github.com/JohannesBuchner/imagehash
- Variance of Laplacian blur detection (OpenCV walkthrough; Pech-Pacheco 2000): https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
- PySceneDetect detectors (ContentDetector threshold 27, AdaptiveDetector 3.0): https://www.scenedetect.com/docs/latest/api/detectors.html
- Bolaños et al. 2015, egocentric photostream keyframes: https://arxiv.org/abs/1505.01130
- Garcia del Molino et al. 2018, Contextual Event Segmentation: https://arxiv.org/abs/1808.02289
- FocusGraph 2026, query-time keyframe selection for egocentric long-video QA: https://arxiv.org/abs/2603.04349
