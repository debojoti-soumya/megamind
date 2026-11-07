# 03 — Multimodal RAG over image+text memories: hybrid retrieval, fusion, temporal expansion, query rewriting, evaluation

*Research bot (stream-5), 2026-11-07. Applies to: **retrieval** (stream-3, `agent/tools.py`), **storage** (stream-2, `store/*`, embedders), **harness** (stream-4, Claude tool description). Uncertain claims marked (?).*

## Question
Our store holds tokens with `text_vec` (caption+OCR+entities), `image_vec` (SigLIP), BM25-able text, and `ts`. How should `recall_memory` fuse these for the question shapes we care about — "where did I leave X", "what did the sign say", "what was I doing at 3 pm" — and how do we measure whether it works?

## What the literature and products do
- **Hybrid (dense + lexical) beats either alone, and the gain is largest on exact strings.** Anthropic's Contextual Retrieval write-up: BM25 "is particularly effective for queries that include unique identifiers" (their example: error code *TS-999*); combining contextual embeddings + contextual BM25 cut failed retrievals by 49 %, +reranking 67 %. For us the "unique identifier" case is **OCR text** (room numbers, product names, whiteboard words) — the semantic embedding of a caption blurs them, BM25 on `ocr_text` nails them.
- **Reciprocal Rank Fusion** (Cormack, Clarke & Büttcher, SIGIR 2009): `score(d) = Σ_r 1/(k + rank_r(d))`, k = 60. Parameter-free, robust to incomparable score scales, consistently beats CombSUM/CombMNZ. LanceDB ships it (`RRFReranker`) for `query_type="hybrid"` with full-text search + vector in one call, including **prefilter** on scalar columns (our `ts`/`device_id`). We already do RRF in `InMemoryStore` — keep it, extend the ranked lists (below).
- **Text→image retrieval with a dual encoder** (CLIP/SigLIP; SigLIP = sigmoid loss, better at small batch, released models in big_vision) is the standard way to answer *visual* queries ("the red mug") without captions. Caveat: CLIP-family text towers are weak on long/compositional queries and have a 64–77-token limit, so feed them **short noun phrases**, not the user's sentence.
- **Vision-centric RAG benchmarks** (MRAG-Bench, 2410.08182; VisRAG, 2410.10594) find LVLMs gain more from retrieved *images* than from retrieved *text* for visually-grounded questions, and that parsing images to text before retrieval loses information. Supports our design (reader re-looks at thumbnails) and argues for including a couple of *visually* retrieved neighbours even when the text path looks confident.
- **Query rewriting.** HyDE (2212.10496): have the LLM write a hypothetical *document* and embed that; works because the corpus is documents, not questions. Our analogue: rewrite the question into **what the caption would say** ("keys on the kitchen counter next to a coffee machine") before embedding — Claude can do this inside the tool call with no extra round-trip. EgoRAG (report 01) instead rewrites to verb+noun keywords for BM25. Do both: one dense query (caption-like), one sparse query (keywords).
- **Temporal handling in egocentric QA:** Ego4D NLQ resolves to the *last* occurrence; EgoRAG hard-filters to `t ≤ question time` and narrows to a window first; all retrieved units are **windows**, not single frames (30 s clips), and the answer model sees neighbours. Nothing in the literature retrieves single frames and stops.
- **Evaluation:** Ego4D NLQ uses **R@1/R@5 at tIoU 0.3/0.5** (did a retrieved window overlap the gold window); EgoLifeQA is 4-way multiple choice with a gold `target_time`; retrieval literature uses Recall@k / MRR / nDCG. For a system whose reader sees k tokens, the retrieval metric that matters is **hit@k = P(gold frame or a frame within ±Δt of it is among the k retrieved)**.

## Concrete recommendations for Megamind

### Retrieval (stream-3)
1. **Four ranked lists into RRF, not two:** (a) cosine on `text_vec` with the *caption-like* rewrite; (b) BM25 over `caption + ocr_text + entities` with the *keyword* rewrite; (c) SigLIP text→`image_vec` with a ≤ 8-word noun phrase; (d) **recency** (all candidates in the time window ordered by `ts` desc). k = 60. Weight (d) at 0.5 if it dominates (?). This encodes "most recent sighting wins ties" (Ego4D) without a hand-tuned decay.
2. **Hard time pre-filter first** (`ts ≤ now`, `time_from/time_to` if given, `device_id`), then vector/BM25 on the survivors — LanceDB supports prefilter natively; the in-memory store already does it. Never post-filter (loses recall when the window is narrow).
3. **Temporal expansion after fusion:** for each of the top-m (m = 4) fused hits, add its temporal neighbours within ±20 s (max 2 per hit) so the reader sees the moment *before/after* the placement ("Where is X before/after event Y"). Cap total at k. Mark them `neighbor_of=<id>` in `all_retrieved` so the reader prompt can group them.
4. **Two-pass on `not_found`**: pass 1 restricted to the last 2 h (or the Claude-supplied window), pass 2 unrestricted; return `confidence` from the pass that succeeded and tell Claude which pass it was (EgoRAG's missing "iterative retrieval").
5. **OCR path:** when the query looks like a text question (contains *say/read/sign/number/label/written/price*), add a BM25 list restricted to `ocr_text` with phrase matching, and boost it ×2 in RRF. Store OCR verbatim, lower-cased copy for BM25 (unicode-normalised), original for display.
6. **`mode` semantics:** `semantic` = (a)+(b)+(d); `visual` = (c)+(d); `hybrid` = all. Default `hybrid`; Claude should pick `visual` only when the user describes appearance ("the blue folder").
7. **Return a small, ordered, de-duplicated set:** collapse hits with `hamming(frame_hash) < 4` within 60 s to one representative (keep the sharpest/most-recent) before the k cut — otherwise 12 slots fill with the same desk.

### Storage / embeddings (stream-2)
8. **Embed `caption + "\n" + ocr_text + "\n" + ", ".join(entities)`** as one text (already planned) — but prepend a one-line context header `"{local time}, {room/place guess}: "` à la Contextual Retrieval, so "kitchen this morning" matches even if the caption omits the room.
9. **BM25 needs a tokenizer that keeps alphanumerics** ("B-204", "WD-40"): LanceDB FTS (tantivy) default is fine; for the in-memory fallback use `re.findall(r"[a-z0-9][a-z0-9\-\.]*", text.lower())`.
10. **SigLIP: use `ViT-B-16-SigLIP` or `so400m` via open_clip; 768-d matches the contract.** Compute on the 960-px thumbnail; normalise. For `visual` mode queries embed with the *text* tower using the template `"a photo of {phrase}"`.

### Harness (stream-4)
11. **Tool description should ask Claude for three strings, not one:** `query` (caption-like sentence for dense), `keywords` (2–4 words for BM25; optional — the retrieval side can fall back to `query`), `visual_phrase` (optional noun phrase). This fits in the existing contract if the extra fields are *optional* args of `recall_memory`; if the contract must stay frozen, do the rewrite server-side with one cheap Haiku call (~300 ms) (?).
12. **Always pass time bounds when the user says any time word**; when none, pass nothing and let the two-pass logic work.

### Evaluation — hit@k on a gold set (stream-4; full protocol in the eval report answering issue #13)
13. `tools/eval.py`: a JSONL of `{question, gold_ts, tolerance_s, type}` (types: where-object, ocr-text, who/what-activity, when-event); metric = **hit@k** (any retrieved `ts` within `±tolerance_s` of `gold_ts`, k ∈ {1, 3, 12}) + **MRR**, computed against `GET /memories?q=…` so it needs no LLM. Also log mean retrieval latency. Target before demo: hit@12 ≥ 0.9, hit@3 ≥ 0.7 on a 20–30 question set built from the demo walk.

## Sources
- Cormack, Clarke, Büttcher 2009, Reciprocal Rank Fusion: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- LanceDB hybrid search (FTS + vector, RRFReranker, prefilter): https://docs.lancedb.com/search/hybrid-search
- Anthropic, Contextual Retrieval (BM25 + embeddings, −49 %/−67 % failures): https://www.anthropic.com/news/contextual-retrieval
- HyDE: https://arxiv.org/abs/2212.10496
- MRAG-Bench: https://arxiv.org/abs/2410.08182 ; VisRAG: https://arxiv.org/abs/2410.10594
- SigLIP: https://arxiv.org/abs/2303.15343 ; open_clip models: https://github.com/mlfoundations/open_clip
- Multimodal RAG survey: https://arxiv.org/abs/2502.08826
- Ego4D NLQ metrics (R@k, tIoU): https://arxiv.org/abs/2110.07058 §5.1 ; EgoLife/EgoRAG: https://arxiv.org/abs/2503.03803 §4.2 (see report 01)
