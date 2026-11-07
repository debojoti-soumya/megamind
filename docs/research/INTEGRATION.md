# Integrating the screenmem research into megamind

Megamind (Quest 3 passthrough capture -> memory tokens -> Claude + Qwen-VL reader -> AR chat; design doc v0.1) has the same core problem screenmem studied: continuous egocentric capture, filtering, compact storage, hybrid retrieval and grounded answers. This note lists what we measured and how it changes or confirms each part of the design. Numbers are measured unless marked *estimate*. Raw data and scripts: `screenmem/`.

## TL;DR — recommended changes to design v0.1
1. **Keep D2 (memory token = record, not learned vector).** We measured it: persisting VLM visual tokens costs ~164 KB/frame (2k tokens fp16), 13–35x more than the compressed frame itself (4.5–12 KB). Store pixels + text + small vectors; let the Qwen reader create visual tokens at query time.
2. **Add episodes to the MVP, not as a stretch.** Retrieval over episodes cuts the number of units by roughly 10–50x (*estimate*) and makes "what did I do after lunch" a single query. We have a working, calibrated segmenter for text-less camera frames (SigLIP/CLIP cosine split + zero-shot scene label).
3. **Make retrieval three-signal RRF with weights:** BM25 on caption+OCR+speech (1.0), text-vector (1.0), image-vector via SigLIP text tower (0.6). Text-only and image-only each miss things; fused hit@5 was 100% on our gold sets.
4. **Treat captions as a ranking signal, never as evidence.** VLM captions hallucinate; the reader must see the thumbnail. Verbatim OCR is the only text evidence that is reliable.
5. **Add ambient speech transcripts as a first-class text field** (`transcript`), if the team accepts the privacy trade-off. Speech answered "what did someone say about X" questions that vision alone cannot.
6. **Add storage tiers instead of keeping every JPEG forever.** Recent thumbnails as JPEG (reader needs them fast), older ones packed into AV1 video chunks (~6–8 KB/frame measured on camera video vs ~80–120 KB JPEG), text + vectors kept forever.
7. **Ship a gold-question eval from hour 3** (`expect_token_ids` / `expect_episodes` / substrings) and run it on every prompt or threshold change.

## 1. Filtering (design §4.2)
| Finding | Implication for megamind |
|---|---|
| On screens, 70% of 1 fps frames were byte-identical and ~19% survived change detection. On real camera footage (1080p test clip) a 0.02 mean-abs-diff threshold on a 64x40 thumbnail kept **100%** of frames: camera noise + head motion means "nothing changed" almost never happens. | The 10–30% keep rate in §4.2 is unverified for passthrough video. Measure it with `tools/feeder.py` on hour 3. If dHash<6 keeps too much, raise the threshold or lower the sample rate rather than accepting 1 fps of near-duplicates. |
| Threshold trap: at the "obvious" threshold the typing phase lost 47/49 frames and 2/5 typed-note questions became unanswerable. | Never filter only on pixel difference. Keep a frame on events too: head-pose change (yaw > ~20°, or translation > ~0.5 m, cheap since pose is already sent), a speech segment starting, a 30 s heartbeat. |
| Dark-frame rejection is cheap and correct. | Keep §4.2's luminance < 20 rule. |

## 2. Memory token (design §4.3) — suggested fields
Add to `MemoryToken`:
- `episode_id` — filled at ingest by the segmenter (see §4), not by a 10-minute stretch job.
- `transcript: str | null` — speech overlapping the frame (faster-whisper `small` int8 locally ran fine; segments attached to the nearest kept frames).
- `source: str` — `quest`, `feeder`, `phone`, so the same pipeline ingests fallback video.
- `chunk_id`, `chunk_offset` — where the pixels live once the JPEG is aged into a video chunk.

Field costs per token (measured or exact): text_vec 1024 fp32 = 4 KB (1 KB at int8); SigLIP image_vec 768 fp32 = 3 KB (0.8 KB at int8); caption+OCR typically < 1 KB. All of it together is smaller than one thumbnail, so vectors are not the storage problem.

OCR: run it on the full-quality frame at ingest, before any re-compression. Lossy codecs kept ~90–93% OCR word recall, downscaling cost more than codec quantisation; OCR on an archived frame is measurably worse. Also strip OCR noise on camera frames (we saw garbage like `(\n7\ni\n| WW` on non-text scenes; a line filter requiring a real word and >= 60% alphanumerics fixed it). If Qwen-VL does OCR inside the enrichment call, still keep the line filter.

Gotcha: RapidOCR's default `limit_side_len=736` silently downscales images and wrecks recall; use full resolution.

## 3. Storage and compression
Per-frame sizes measured on 1080p camera-style test video (Jellyfish = real camera, Big Buck Bunny = animation), bytes per frame:

| Encoding | Jellyfish | BBB |
|---|---:|---:|
| x265 1920 crf28 | 69 KB | 170 KB |
| x265 1280 crf30 | 36 KB | 61 KB |
| AV1 1280 crf40 | 10.8 KB | 15.8 KB |
| AV1 1280 crf50 | 6.1 KB | 7.6 KB |
| AV1 960 crf50 | 4.4 KB | 4.5 KB |

These clips were consecutive 1 fps frames. After filtering, kept frames are further apart and less similar, so expect AV1 cost per kept frame to be higher (*estimate*: up to 2x).

Daily volume for megamind, 16 h of wear, 1 fps sample (*estimate*, using the design's 80–120 KB JPEG):
| Keep rate | Kept frames/day | JPEG only | JPEG 3 days, then AV1 1280 crf50 |
|---|---:|---:|---|
| 10% | 5.8k | 0.5–0.7 GB/day, 50–70 GB/100 d | ~2 GB hot + ~4–9 GB for 100 days cold |
| 30% | 17k | 1.4–2.1 GB/day, 140–210 GB/100 d | ~6 GB hot + ~10–25 GB for 100 days cold |
| 100% (what we saw on camera) | 57.6k | 4.6–6.9 GB/day | ~18 GB hot + ~35–90 GB cold |

For the hackathon demo (hours, not days) JPEG-only is fine. Tiering is a strong "we thought about 24/7" slide; it already exists in screenmem's `packer.py` (`age_chunks`: hot -> AV1 -> text-only, encoder auto-detects `libsvtav1` vs `libaom-av1`) and ports with small changes. Lesson from real data: when the hot tier is already efficient, the cold tier must downscale (1280) and not just re-quantise; AV1 crf40 at the same width came out *larger* than hardware HEVC once.

## 4. Episodes (design §4.3 "consolidation", promote to MVP)
Screenmem segmenter, adapted for camera frames with no app/title metadata:
- New episode when time gap > 120 s, or image-embedding cosine to the episode's representative frame < **0.80**. Calibration on real *screen* keyframes (not camera yet): within an episode min 0.83 / median 0.99; across different content 0.73–0.90. Re-tune on Quest footage with the feeder.
- Brief detours rejoin the previous episode if it is among the last 4 episodes and within the gap.
- Representative frame = the frame with the most text (or medoid in embedding space for text-less scenes).
- Episode title from zero-shot classification of the rep frame against ~20 scene prompts ("a kitchen", "a whiteboard with writing", "a street outdoors", ...). Worked on test footage ("the sea or underwater", "a cartoon or animation"). Gives the timeline and journal something to say before any VLM call.
- Optional: one Claude/Qwen call per closed episode for a summary; `head pose` centroid per episode for "where" questions.

Retrieval over episodes = sum of RRF scores of their frames, returned in time order. That answers "what was I doing after lunch", "what did I do about the whiteboard", and day journals with one call.

## 5. Retrieval (design §5.2)
Measured on screenmem's 16-question gold set:
| Method | hit@1 | hit@5 |
|---|---:|---:|
| BM25 (FTS5) | 87.5% | 93.8% |
| Dense text (MiniLM) | 68.8% | 100% |
| Union | — | 100% |
Glasses-style gold set (visual-only + speech questions on test footage): 8/8 hit@1 with three-signal fusion.

Details that mattered:
- BM25 query: AND of terms first, fall back to OR if empty. Stop-word strip the natural-language question first.
- Parse time phrases ("at 3pm", "yesterday morning", "20 minutes ago") into a ts filter before searching; Claude can also pass `time_from`/`time_to` as in §5.1.
- Image vector weight below text (0.6): CLIP-family models are weak on dense text, strong on objects/scenes/colours.
- Diversity: at most 2–3 tokens per episode in top-k, otherwise 12 near-identical frames of the same desk fill the reader's context.
- Route by question type before retrieval: `fact` (top tokens), `activity` (episodes over a time range or topic), `visual` (image-vector heavy + send thumbnails). Cheap regexes worked; Claude's tool call can carry `mode` as in §5.1.

## 6. Reader / "attention over tokens" (design §5.3)
Research conclusion (see `screenmem/APPENDIX_research_notes.md`): the honest and cheap version of "the model attends over memory" is exactly D2/D3 — retrieve records, give the VLM real image + text tokens at query time. Improvements:
- Feed the reader episode rep frames plus the top frames, not 12–24 random hits; prefix each with `[id] [time] [episode title] OCR/speech`.
- For text questions ("name on the whiteboard"), crop the region around the OCR boxes and send the crop at high resolution instead of the whole downscaled thumbnail.
- Require citations by token id and mark speech evidence separately (`[speech]`) from vision.
- Stretch, if a GPU is available: ColQwen2/ColPali-style late interaction over episode keyframes for hard visual queries. Not needed for the demo.

## 7. Evaluation
Port `screenmem/evalset.py`: a JSON list of `{q, expect: [substrings] | expect_token_ids | expect_episodes}`, reports hit@1/hit@k. Write 20–30 questions from the scripted demo walk (keys, whiteboard, a person, a sign, "what did I do after X") on hour 3 and re-run after every change to filtering thresholds, caption prompt or fusion weights.

## 8. What to port from screenmem (all Python, ~1–2 h each, *estimate*)
| screenmem file | megamind target | notes |
|---|---|---|
| `ingest_video.py` | `tools/feeder.py` | any mp4 -> 1 fps, change-detected frames, optional whisper transcript |
| `indexer.py` `assign_episode`, `scene_label`, `clean_ocr` | `ingest/worker.py` | episode segmentation + labels + OCR filter |
| `search.py` | `store/lancedb_store.py` query side | RRF fusion weights, AND/OR BM25, time parsing, episode diversity, `search_episodes` |
| `ask.py` `classify`, `episode_pack` | `agent/tools.py`, `agent/reader.py` | routing + evidence formatting with citations |
| `packer.py` `age_chunks` | `store/` stretch | hot JPEG -> AV1 chunk -> text-only |
| `evalset.py` + gold JSON | `server/tests/` | retrieval regression test |

## 9. Open questions this adds
- Actual keep rate and dHash threshold on Quest passthrough frames (measure on hour 3).
- SigLIP vs CLIP for image_vec: SigLIP is expected to be better on text-in-image; untested here.
- Record ambient audio or not (privacy of bystanders vs "what did they say" questions).
- Per-frame Qwen enrichment cost at the real keep rate (17k frames/day at 30% is 17k VLM calls); consider captioning only episode rep frames + frames with OCR text.
