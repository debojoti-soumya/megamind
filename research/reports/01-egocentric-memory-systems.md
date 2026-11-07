# 01 — Egocentric episodic memory systems: what to copy from EgoLife/EgoRAG, Vinci, Ego4D

*Research bot (stream-5), 2026-11-07. Primary sources checked; uncertain claims marked (?).*

## Question
Which memory structures and retrieval strategies have actually worked for "assistant that remembers what I saw", and which of them should Megamind copy in the next ~30 hours?

## What the literature does

### Ego4D Episodic Memory (Grauman et al., CVPR 2022)
- Defines our task exactly: *given past egocentric video + a query, localize the window where the answer is visible.* Three query types: **NLQ** (text query → temporal window), **VQ** (image of an object → last time/place it was seen, with bbox), **MQ** (activity name → all windows).
- NLQ was built from **13 templates** that are a ready-made demo-question list: *Where is object X before/after event Y? Where is object X? What did I put in X? How many X's? What X did I Y? In what location did I see object X? What X is Y? Where is my object X? Where did I put X? Who did I interact with when I did activity X? Who did I talk to in location X? When did I interact with person with role X?*
- All queries are **past tense, asked "at the end of the video"**, and ambiguity is resolved to the **last** relevant occurrence. Metric is recall@k at temporal IoU — i.e. the system is judged on *finding the right moment*, not on fluent prose.
- Takeaway: "where is X" ⇒ answer with the *most recent* sighting by default; the time of the sighting is part of the answer.

### EgoLife / EgoButler = EgoGPT + EgoRAG (Yang et al., CVPR 2025)
- 300 h, 6 people × 7 days on Meta Aria glasses. QA types: **EntityLog** (object details: last use, location, price), **EventRecall**, HabitInsight, RelationMap, TaskMaster. Each question has a "certification length" (look-back: 2 h → 24 h+).
- **Memory bank** = per-30-s-clip dense captions (visual+audio) + **hourly and daily LLM summaries** stored with timestamps ("multi-level memory bank"). Stored in **Chroma with OpenCLIP embeddings** (`OpenCLIPEmbeddingFunction`) — i.e. the text captions are embedded with CLIP's text tower, not a text-embedding model.
- **Retrieval is two-step and time-first:**
  1. An LLM reads the day summaries and picks *one day*, then reads that day's hour summaries and picks *one [start,end] window* (`get_day_prompt` / `get_hour_prompt` in `EgoRAG/egorag/utils/prompts.py`). Only then is vector search run, **restricted to that window** (top-3 clips).
  2. For "last time…" questions a different path: an LLM rewrites the question into **short verb+noun keywords** ("use microwave", "take out from refrigerator") and a **search-before time** ("question time − 5 min" unless the question implies otherwise); Chroma returns 50 candidates with `end_time <= search_time`; an LLM re-ranker picks the best id.
- **Evidence cards:** each retrieved clip is checked by a VLM with a per-clip prompt ("If relevant: `I can provide evidence. Evidence: …`, else `I can't provide evidence.`"), then the answerer sees only `{time, evidence_info}` cards. Timestamps are formatted `DAYX_HHMMSSFF`, and the answer prompt spells out how to compute "yesterday"/"day before yesterday" from question time.
- **Results (Table 6, EgoLifeQA):** with look-backs > 2 h EgoRAG lifts accuracy from ~25–29 → 35–39 (Gemini-1.5-Pro on a 30-min window: 15–28). Caption quality is the dominant factor (Table 7: human captions > any model; visual+audio > visual-only > audio-only).
- **Stated weakness (§5):** single-pass retrieval, "if EgoRAG cannot locate supporting evidence it simply fails" — no re-query/iteration.

### Vinci (OpenGVLab, 2024/2025)
- Always-on assistant on phone/glasses; fine-tuned EgoVideo-VL. **Memory module** = VLM writes a short *action* description for each snapshot ("pour the water into the pot") + timestamp into a **FIFO text queue**; at query time the whole queue is put into the LLM context (no vector search). Temporal grounding ("when/where did I lose my keys") reported > 80 % accuracy on a 10-object controlled test; 0.6–0.7 s response latency. No evidence frames are shown to the user; no persistence beyond the queue. Hardware-agnostic, web UI with live stream + chat history.

## Concrete recommendations for Megamind

| # | Recommendation | Stream | Why / source |
|---|---|---|---|
| 1 | **Default to "most recent match" for where-is-X.** Add a recency prior in RRF (e.g. a third ranked list ordered by `ts` desc, or multiply fused score by `exp(-age/τ)`, τ≈2 h) and let `recall_memory` return hits sorted so the newest relevant token is first. | retrieval | Ego4D resolves ambiguity to the *last* occurrence; EgoRAG hard-filters `end_time <= question_time`. |
| 2 | **Query rewriting before search:** have Claude pass `query` as short verb+noun / object keywords ("keys counter", "whiteboard text") rather than the full question. Put this in the tool description so Claude does it; no extra LLM call. | harness, retrieval | EgoRAG `query_prompt`: "only verb + direct noun, remove locations/adjectives". |
| 3 | **Time-first, then semantics.** Keep `time_from/time_to` pre-filter (already in contract). If a query has no time hint, run a second, *wider* pass only if the first returns `not_found` — that is the "iterative retrieval" EgoRAG admits it lacks. Implement as: pass 1 = last 2 h, pass 2 = all time. | retrieval | EgoLife §5 failure analysis; certification-length stats (most QAs ≤ 24 h). |
| 4 | **Evidence-card pattern for the reader.** Reader prompt per token: `[id] [YYYY-MM-DD HH:MM:SS local] caption / ocr` + thumbnail; ask for `relevant: bool, evidence: str` per token *and* a final answer. Pass only the relevant cards (not all 12 captions) to Claude. Already the plan; the EgoRAG prompts confirm the format works with weaker VLMs. | retrieval | `video_evidence_prompt`, `with_evidence` in EgoRAG. |
| 5 | **Human-readable local timestamps in context.** Format ts as local time with a relative hint (`14:32 (21 min ago)`) in the reader and Claude contexts; EgoRAG had to teach the model how to compute "yesterday". Keep Unix floats only in the API. | harness, retrieval | EgoRAG `with_evidence` prompt. |
| 6 | **Caption quality is the lever, not the retriever.** Spend prompt-tuning time on the enricher: object + *surface/location* + verbatim OCR + room type. EgoLife Table 7: swapping captioners moves accuracy more than retrieval tweaks. Add a 1-line "action" field ("person puts keys on counter") like Vinci — short action phrases retrieve well for "what did I do" questions. | capture | EgoLife Table 7; Vinci memory module. |
| 7 | **Hourly/episode summaries are a real win for long look-backs** (EgoRAG +10 pts beyond 2 h). For the demo (minutes–hours of capture) this is optional; see report 06. Cheap version: every 10 min, one LLM call over the captions → one `episode` token stored with `episode_id`, retrievable by the same search. | storage | EgoRAG multi-level memory bank. |
| 8 | **Embed captions with a text model, not CLIP text tower, for the semantic path.** EgoRAG used OpenCLIP for text-to-text similarity; CLIP's text encoder is weak on long captions (77-token limit). Our plan (text-embedding-3-small for `text_vec`, SigLIP for `image_vec`) is better; keep it. | storage | EgoRAG `Chroma.py`; CLIP 77-token context (known limitation). |
| 9 | **Demo script = Ego4D NLQ templates.** Use "Where did I put X?", "What did I put in X?", "What did the sign/whiteboard say?", "Who did I talk to in the kitchen?" as the rehearsed question set; they are the canonical, judge-recognisable episodic-memory queries. | ui, harness | Ego4D Table 5. |
| 10 | **Show the frame.** None of EgoRAG/Vinci show the evidence frame to the user; Ego4D VQ explicitly asks for the *where/when* of the last sighting. Our cited-thumbnail panel is a genuine differentiator — make sure the panel displays `ts` ("seen 21 min ago") next to the thumbnail. | ui | Ego4D VQ definition; prior-art gap. |

### What *not* to copy
- EgoRAG's LLM-pick-a-day → LLM-pick-an-hour cascade (2 extra LLM calls, ~3 s) — unnecessary for < 1 day of memory; Claude's `time_from/time_to` args already do this in one step.
- Vinci's FIFO-in-context memory — does not scale past a few hundred tokens and has no retrieval; fine for a 10-minute demo but our vector store already covers it.
- Fine-tuned egocentric VLMs (EgoGPT, EgoVideo-VL) — no training in scope; hosted Qwen/Gemma are adequate captioners per EgoLife Table 7's Gemini/Qwen rows (?: exact numbers for Qwen2-VL not re-checked).

## Sources
- Ego4D: https://arxiv.org/abs/2110.07058 (§5.1 Episodic Memory; App. F Table 5 NLQ templates)
- EgoLife: https://arxiv.org/abs/2503.03803 (§4.2 EgoRAG, Tables 6–7, §5 analysis); code https://github.com/EvolvingLMMs-Lab/EgoLife/tree/main/EgoRAG (`egorag/agents/RagAgent.py`, `egorag/utils/prompts.py`, `egorag/database/Chroma.py`)
- Vinci: https://arxiv.org/abs/2412.21080 and https://arxiv.org/abs/2503.04250 (§3 memory module, §5 temporal grounding); code https://github.com/OpenGVLab/vinci
