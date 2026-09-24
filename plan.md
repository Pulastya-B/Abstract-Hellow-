# Business Entity Resolution Challenge — Implementation Plan

## Context

This is the Amazon ML Challenge 2026 hackathon problem: **Business Entity Resolution**. Given three independent, noisy sources of business records (`business_name`, `business_address`, `country`), the task is to find, for every Source-1 (deduplicated reference) entity, which Source-2 and/or Source-3 records refer to the same real-world business. A Source-1 entity may match zero, one, or many records.

The dataset is large — this is not a toy problem:

| File | Rows |
|---|---|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| train_ground_truth | 2,206,821 |
| test_source1 | 1,732,544 |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

~3GB of TSV total. Naive pairwise comparison (2.2M × 10M+) is impossible — a strong **blocking/candidate-generation** stage is the single highest-leverage piece of this pipeline, since it sets a hard recall ceiling nothing downstream can recover from.

Key facts verified directly against the real files (all row counts, ratios, and schema below re-confirmed against the actual TSVs — not assumptions):

- **Schema**: `train_source1/2/3` and `test_source1/2/3` each have exactly 4 columns: `entity_id` (e.g. `S1-925783039`, `S2-...`, `S3-...`), `business_name`, `business_address`, `country`. `train_ground_truth.tsv` has exactly 2 columns: `source1_entity_id`, `matched_entity_ids` — the second field is a **comma-separated list** of S2-/S3-prefixed IDs (empty string = singleton), not a single ID. This exact list-per-row shape is also what both output TSVs must use (see Output contract below). No hidden/extra columns exist in any file.
- **Singletons**: 123,247 of 2,206,821 S1 training entities (5.6%) have *no* match (empty `matched_entity_ids`). F_0.5 is computed **per S1 entity, macro-averaged, including singletons** — a correct empty prediction scores 1.0, a false match on a true singleton scores 0.0. Getting singletons right is worth as much as getting a real match right.
- **Match-count shape**: of matched entities, the vast majority have 2-5 matches (long tail out to 11); total matched IDs across train ground truth = 7,638,365 exploded (S1, matched_id) pairs — i.e. ~73% of train S2/S3 records are true matches to *some* S1 entity, ~27% are unlinked "noise."
- **S2/S3 IDs are exclusively owned — verified structural invariant**: across all 7,638,365 ground-truth (S1, matched_id) pairs, **zero** S2/S3 IDs are claimed by more than one S1 entity. The true mapping is strictly many-to-one on the S2/S3 side. This is not a heuristic — it's confirmed on the full training ground truth — and is exploited directly by the global conflict-resolution step in §6d below.
- **Country is an open set**: train covers US (1,323,633) and India (883,188) only; test adds **France** (259,452 of 1,732,544 test S1 rows, ~15%) which never appears in training. Nothing in the pipeline may hardcode logic to `{US, India}`.
- **Missing data**: `business_address` is never empty in Source 1, but is empty in ~3.3% of Source 2 and ~3.3% of Source 3 records (train figures; test expected similar). Missingness must be modeled explicitly, not treated as a mismatch.
- **Script/language noise is real, not hypothetical**: India-country names in Source 2 contain native Devanagari script (e.g. "राम मार्केटिंग प्राइवेट लिमिटेड") alongside Latin-script equivalents elsewhere. France test addresses are French-language. Plus the documented noise: legal-suffix variants (Corp/Corporation, Pvt/Private, Ltd/Limited), DBA names, punctuation, word-order transpositions, typos, address abbreviations, landmark references, missing PIN/state, municipal numbering variance.
- **Scoring**: F_0.5 (β=0.5) — precision weighted 2× over recall — confirmed verbatim against README: `F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`, macro-averaged per S1 entity including singletons. Model must be conservative/precision-heavy, not recall-maximizing.
- **Hard constraints**: fair-play rules ban external databases/APIs/services used **to look up or resolve business identities** (commercial ER APIs, registry/geocoding lookups, internet data augmentation) — they do not explicitly address downloading frozen pretrained model weights. Constraint 5 caps the *final model* at MIT/Apache-2.0 license, ≤8B params. Using pretrained open-weight models (embeddings, rerankers) with frozen weights is a reasonable reading of what's allowed given that cap, but this is our inference, not explicit rule text — verify each model's actual license on its model card before final packaging (see §5/§5b), and if in doubt, check the competition FAQ/organizers before submission.
- **Output contract**: confirmed against `utils/validate_submission.py` source. Both output files are **one row per S1 entity** with a comma-separated ID list: `matching_results.tsv` = `[source1_entity_id, matched_entity_ids]` (the only file actually scored on the leaderboard), `candidate_pairs.tsv` = `[source1_entity_id, candidate_entity_ids]` (not scored by the public validator, but still required in the submission zip). The validator additionally rejects: missing/extra/duplicate S1 rows, duplicate IDs within a list, S1-prefixed self-matches, and non-S2/S3-prefixed IDs — `io_utils.py`'s writer should enforce these structurally rather than relying on catching them via validator re-runs. `--check-ids` (checks predicted IDs exist in test_source2/3) is off by default due to memory cost, but the real scorer treats an unknown ID as score-lowering rather than a hard reject — run it at least once before final submission anyway, since an ID-hallucination bug would otherwise pass local validation silently.
- Final submission is a zip named `<team_name>_submission.zip` containing `output/{matching_results.tsv, candidate_pairs.tsv}`, `code/business_entity_resolution/{src/, README.md, requirements.txt}`, and `Documentation_template.md` at the zip root (required sections: Executive Summary; Methodology incl. Problem Analysis/Solution Strategy; Candidate Generation/Blocking; Matching Model; Results & Error Analysis incl. macro F_0.5; Conclusion; Appendices).

No starter code exists yet — only `README.md` (problem statement), `Documentation_template.md` (methodology template), and `utils/validate_submission.py` (local format validator). Everything under `code/business_entity_resolution/src/` needs to be built.

**Decisions already made with the user:**
- Approach: **full ensemble pipeline** — two independently-trained matchers (a symbolic LightGBM on engineered similarity features, and a neural classifier on embedding-derived features), a cross-encoder reranker on a bounded candidate subset, and a stacked meta-model that combines all signals into a calibrated final probability. This supersedes an earlier, simpler "one connected system" design — the user explicitly opted for the heavier ensemble after reviewing the tradeoffs (more capability ceiling, more engineering/compute cost — see §5b and §8 for the compute risk this introduces and how it's mitigated).
- Platform: **Kaggle Notebooks** primary (GPU available, ~30h/week quota, session limits ~9-12h — plan must checkpoint across sessions via Kaggle Datasets).
- Ambition: go straight for the strongest pipeline — no throwaway minimal baseline — but built in independently-runnable phases so a valid, submittable output exists at every milestone. The ensemble stages (§6b onward) are strictly additive on top of a working single-matcher baseline (§6a), so a time or compute shortfall degrades gracefully rather than leaving nothing runnable.
- The user will run everything themselves on Kaggle GPUs; nothing gets executed locally by the assistant.

---

## Plan

### 1. Scoped EDA (`01-eda.ipynb`, sampled + full ground truth, target <2h)

- Token-count/char-length histograms for `business_name` by source × country (sets min-token-overlap thresholds).
- Join `train_ground_truth` to S1/S2/S3, compute token Jaccard / rapidfuzz ratio / char-3gram Jaccard for the ~2M true-positive pairs vs. an equal-size random-pair sample — this single artifact drives every threshold decision downstream.
- Blocking-key collision risk: count S1 entities sharing an identical normalized name / first token per country (flags common-name over-generation, e.g. "City Medical Store").
- Postal-code presence rate per country (US 5/9-digit ZIP, India 6-digit PIN, France 5-digit) via regex — decides whether postal code can be a primary blocking key.
- Devanagari incidence rate in India-country names by source — sizes how much the transliteration path matters vs. relying on the embedding model alone.

### 2. Normalization (`src/normalize.py`, `src/transliterate.py`)

Built with **polars** (not pandas, for the 5M+ row tables) + **regex** + **rapidfuzz**. Produces two representations per text field: `normalized` (aggressive, for similarity/blocking) and `core` (legal-suffix stripped).

- NFKC normalization → casefold → punctuation normalization (`&`→` and `, collapse whitespace).
- **Devanagari→Latin**: detect U+0900–U+097F codepoints; transliterate via `indic_transliteration` (Devanagari→IAST→ASCII diacritic strip) — deterministic script mapping, no network call, so it is compliant with the no-external-lookup rule. Keep the *original-script* string too, for the embedding model.
- **Legal-suffix dictionary**, hand-curated from the training vocabulary itself (`corp/corporation`, `pvt/private`, `ltd/limited`, `llc`, `llp`, `inc`, etc.) → canonicalized `normalized_name` and suffix-stripped `core_name`. Fail-open (unknown token = no-op), so it degrades safely for France rather than requiring hardcoded SARL/SAS rules.
- **Address abbreviation dictionary** (`rd→road`, `st→street`, `ave→avenue`, ...).
- **Landmark extraction** (`near|opp|behind|next to` + phrase) into a separate column, excluded from the core address comparison but kept as a `has_landmark` feature.
- **Postal-code extraction** via per-country regex, applied only within its country partition.
- **Locality/region heuristic**: comma-split, take trailing segments — same rule for every country, so France needs no special-casing.
- Tokenize to token-list and char-3gram-shingle columns.

Output written to parquet immediately (checkpointed as a `normalized_v1` Kaggle Dataset).

### 3. Blocking / candidate generation (`src/blocking.py`)

**Country-partitioned throughout** — reduces the largest country bucket from 2.2M×10M to ≤~1.3M×3.3M, the single biggest scalability lever, and is safe since `country` is a reliable provided field.

Within each partition, union several independent, cheap blocking keys (each catches a different noise pattern), then merge per-S1:

1. **Token inverted index** (primary): explode normalized-name tokens (stopwords/suffix tokens excluded from vocabulary), require ≥1 shared significant token.
2. **Postal-code exact block**: free, high-precision, unioned in whenever both sides have an extracted code.
3. **Phonetic block**: Double Metaphone (`jellyfish`) on the first significant token — catches typos.
4. **MinHash/LSH on char 3-grams** (`datasketch`) — catches transpositions/heavy typos with no shared tokens.
5. **Sorted-neighborhood** on normalized name within country (fixed window) — cheap extra recall margin.
6. **Embedding ANN block** (ties to §5): FAISS per-country index over S2+S3 name embeddings, top-k≈30 query per S1 — the mechanism that handles Devanagari↔Latin and generalizes to unseen-country France without any country-specific keys.

**Track block provenance per candidate** — record which of the keys above (1-6) surfaced each (S1, candidate) pair. This feeds a `num_blocks_agreeing` feature in §4: a candidate independently found by several unrelated blocking methods is much stronger evidence than one found by a single fuzzy rule, and it's free to compute since it's just bookkeeping during the union.

**Mandatory recall-ceiling check before moving on**: on train, verify what fraction of ground-truth positive IDs appear anywhere in the unioned raw pool (target ≥97%); tune key thresholds against this number, while tracking avg-candidates/S1 (reduction ratio) to keep the pool tractable.

**Final pruning → `candidate_pairs.tsv`**: score the unioned pool with a cheap weighted-sum of fast signals (token Jaccard + rapidfuzz ratio + postal bonus), rank per S1, cap at top-N (~30). This pruned, ranked set — not the raw union — is what gets written to `candidate_pairs.tsv` and what the matchers score, matching the spec's requirement that it be the *last* stage before the model. **Output format**: one row per S1 entity, `candidate_entity_ids` as a comma-separated list of the top-N IDs (mirrors `CANDIDATE_HEADER` in `validate_submission.py`) — the S1-row-with-list shape, not a flat pairwise table.

### 4. Feature engineering (`src/features.py`, computed only over the pruned candidate set)

Country used only as (a) an upstream partition key and (b) a passthrough categorical feature — never hardcoded logic — so the models can learn any residual country effect and France still works at inference despite being unseen in training rows.

- **Name**: exact-normalized/exact-core match, token Jaccard, rapidfuzz `token_sort_ratio`/`token_set_ratio` (word-order robust), rapidfuzz `ratio`, Jaro-Winkler, char-3gram Jaccard, first-token match, length ratio, phonetic-code equality, **name embedding cosine similarity**.
- **Address**: tri-state postal match (match/mismatch/**unknown** — never collapsed into mismatch), address token Jaccard, char-3gram similarity, `token_sort_ratio`, street-number match, locality overlap, `has_landmark` flag, **address embedding cosine similarity**, explicit `address_missing_left/right` indicators.
- **Meta**: `source` (S2/S3, categorical), `country` (categorical, passthrough), `candidate_score_rank_within_s1`, `num_candidates_for_s1` (flags collision-risk entities from EDA §1), **`num_blocks_agreeing`** (from §3's provenance tracking).

These are the shared engineered features that feed Matcher A directly (§6a) and, via cheap cross-candidate aggregation per S1 (counts/ranks above), also inform the meta-model's per-entity statistics in §6c — kept distinct from the *probability-based* per-S1 statistics (max/2nd-max/gap of model scores), which can only be computed after Matcher A/B run and belong in §6c, not here.

### 5. Embedding component (`src/embed.py`)

- **Model**: `intfloat/multilingual-e5-base` (stated MIT license, 278M params, covers Hindi/English/French, retrieval-tuned) as the target; prototype first with `paraphrase-multilingual-MiniLM-L12-v2` (stated Apache-2.0, 118M, faster) if e5-base proves too slow at 20M-record scale, then swap in for the final run. Both are claimed to satisfy the ≤8B-param/MIT-Apache constraint — **verify the actual license on each model's HuggingFace model card before final packaging**, since this has not been independently confirmed against the primary source.
- **Batching**: `sentence-transformers`, fp16, batch ~256-512, sharded across Kaggle's T4×2 via `CUDA_VISIBLE_DEVICES`. Expect low single-digit hours for the full ~20M+ short-text corpus (name + address, per source/country partition).
- **Persistence**: fp16 `.npy` (memmap-able) + id-order parquet, checkpointed as an `embeddings_v1` Kaggle Dataset so this expensive step is only recomputed if upstream normalization changes.
- **Three uses** (was two): FAISS per-country ANN index for blocking key 6 above; cosine similarity feature computed only for the pruned candidate pairs (bounded cost) and fed into Matcher A (§4); and raw/derived embedding vectors (cosine sims per field + difference vectors) as the input feature set for Matcher B (§6a).

### 5b. Reranker component (`src/rerank.py`) — new, part of the ensemble upgrade

- **Model**: a multilingual cross-encoder reranker (e.g. `BAAI/bge-reranker-v2-m3`, stated Apache-2.0, ~568M params) that scores a candidate pair jointly (name+address of S1 vs. candidate) rather than comparing independent embeddings — this captures interaction signal a cosine-similarity feature structurally cannot. **License not independently verified — confirm on the actual HF model card before use**, same caveat as §5.
- **This is the plan's single biggest compute risk — must be scoped, not assumed.** A cross-encoder can't use ANN; it requires a full forward pass per pair. At the existing top-N≈30 pruning cap across ~3.9M total S1 entities (train+test), reranking *every* pruned candidate is up to ~117M pairwise calls. At an optimistic ~1000 pairs/sec combined on Kaggle's T4×2, that's ~32 hours — at or beyond the entire weekly GPU quota by itself.
- **Mitigation (mandatory, not optional)**:
  1. Rerank only a **small top-K per S1** (e.g. top 5–8 by combined Matcher A/B score), not the full ~30-candidate pool — cuts volume ~4-6×.
  2. **Skip reranking for confidently-resolved S1 entities** — only rerank where Matcher A/B disagree or the top candidate's combined score falls in an uncertain band (e.g. roughly [0.15, 0.85], to be tuned from validation data once Matcher A/B exist). Obvious singletons and obvious strong matches don't need the expensive signal.
  3. Treat reranker coverage as **partial** — the meta-model (§6c) must accept a "reranker available" flag plus an imputed/missing value for uncovered pairs, not assume universal coverage.
  4. **Measure real throughput on a ~100K-pair sample at the start of milestone M5**, before committing to the full-scale run, and re-derive the top-K/band cutoffs from that measured number rather than the estimate above.

### 6. Matching: ensemble design (`src/matcher_a.py`, `src/matcher_b.py`, `src/stack.py`, `src/postprocess.py`, `src/train.py`)

#### 6a. Matcher A — symbolic (LightGBM)

- **LightGBM** binary classifier on the §4 engineered features — histogram-based speed at tens-of-millions-of-rows scale, native categorical support (unseen "France" routes to a learned default split rather than erroring — see §7 LOCO diagnostic for why this is checked directly rather than assumed), GPU build available on Kaggle.
- **Training rows**: positives = the 7,638,365 exploded ground-truth (S1, matched_id) pairs (explicitly explode `train_ground_truth.tsv`'s comma-separated `matched_entity_ids` field — do not treat it as a flat pair list); negatives = drawn from the *same* pruned blocking pool (not random pairs, so the training distribution matches inference), split hard (high blocking-score, non-match — what F_0.5 punishes hardest, e.g. same normalized first token/postal code but not a true match) vs. easy, target ratio ~1:3-1:5. Include the full negative-only candidate pools for train singleton entities, since F_0.5's singleton weighting requires the model to have seen "correctly predict nothing" cases.
- **Trained via K-fold (K=5) entity-level cross-validation** (see §7) rather than a single train/val split — required so that its predictions can be used out-of-fold (OOF) downstream in §6c without leakage.

#### 6b. Matcher B — neural (embedding-feature classifier)

- A lightweight classifier (logistic regression or small MLP — keep it simple, this is a second, *diverse* signal for the ensemble, not a place to over-invest) trained on embedding-derived features from §5: cosine similarity per field (name/address/combined), and raw difference/concatenation vectors.
- Same K-fold OOF training scheme as Matcher A.
- **Ablate before committing further**: after M4.5, confirm Matcher B actually adds signal over Matcher A alone (e.g. does the meta-model in §6c weight it meaningfully, does validation macro F_0.5 improve) before investing further ensemble time — if it doesn't, that's a legitimate reason to simplify back toward a single matcher, not a reason to force the extra complexity in.

#### 6c. Meta-model (stacker) and threshold search

- A simple second-stage model (logistic regression or shallow LightGBM, to avoid overfitting on a small number of meta-features) trained on: Matcher A's OOF probability, Matcher B's OOF probability, reranker score where available (§5b, with a missingness flag), `num_blocks_agreeing`, candidate-distribution stats (rank, count, from §4), and exact-match flags → a single calibrated final probability per candidate pair.
- **Both Matcher A and Matcher B must feed this with out-of-fold predictions, never in-fold** — training the meta-model (or the singleton gate below) on in-fold scores lets it see artificially confident predictions that don't reflect real test-time score distributions, a classic stacking leakage bug. This is a real complexity increase (K-fold training instead of one split) but LightGBM training is cheap (<30 min/fold per §8), so K=5 costs roughly 2-3h total — acceptable.
- **Per-S1 aggregation**: from the meta-model's final probabilities, compute per-entity statistics — max probability, 2nd-max probability, gap between them, counts above several probability thresholds.
- **Threshold selection targets macro F_0.5 directly**, not accuracy and not a fixed cutoff: jointly grid-search a `singleton_threshold` (below which an entity is predicted empty) and a `match_threshold` (above which a candidate is included), optionally per-source (S2 vs S3), maximizing the exact held-out per-entity macro F_0.5 — not just singleton-classification accuracy, since the real objective is end-to-end score, and a couple of misclassified singletons that happen to be non-singletons with real links cost far more than raw accuracy suggests.

#### 6d. Global conflict resolution (`src/postprocess.py`) — new, justified by verified data

Confirmed on the full training ground truth (see Context): every S2/S3 ID that appears in any `matched_entity_ids` list belongs to exactly one S1 entity — never more than one. Per-entity independent thresholding can still, in principle, assign the same S2/S3 ID into two different S1 entities' final output lists even though the true data never does this. **Post-process step**: after per-S1 thresholding, if any S2/S3 ID appears in more than one S1's output list, keep it only under the S1 with the higher meta-model confidence and drop it from the others. Essentially free to implement, pure precision gain under F_0.5, and directly justified by a verified structural property of the data rather than an assumption.

### 7. Validation methodology (`src/validate_f05.py`)

- **Entity-level K-fold split** (K=5): stratified by country and by ground-truth match-count bucket (0 / 1 / 2-3 / 4-6 / 7+) to preserve singleton rate and tail shape across folds, and to support the OOF requirement in §6a-c. Exclude each fold's held-out S1 IDs from that fold's training entirely.
- **Leave-one-country-out (LOCO) diagnostic** — separate, one-off, not part of the main K-fold loop: train on India-only (or US-only) and validate on the other country treated as fully unseen. This is the only way to directly test how well the pipeline handles a genuinely novel `country` category — the France scenario at test time — since a country-stratified K-fold split structurally always has every training country represented in some fold, and therefore can never simulate "this country was never seen at all." Run once before trusting France-side test performance; if it reveals the `country` categorical feature degrades badly on unseen values, the mitigation is to lean more on the country-agnostic signals (embeddings, reranker, string features) that already generalize by design, and de-emphasize or drop the raw categorical feature.
- **Self-scoring**: implement the exact formula (`F_0.5 = 1.25·P·R / (0.25·P + R)`; empty-vs-empty = 1.0; any prediction on a true empty = 0.0), macro-averaged over all held-out entities including singletons — scored against the *full pipeline's realistic output*: through the real blocking pool (not an oracle candidate set) **and after §6d's global conflict resolution**, so it reflects the same recall-ceiling loss and dedup effects the real leaderboard will see. One function, reused by the threshold search in §6c and as a standalone pre-submission CLI check.

### 8. Scalability / Kaggle compute plan

| Phase | Tooling | Rough budget |
|---|---|---|
| EDA | polars/pandas, sampled | <2h |
| Normalization (~20M rows) | polars vectorized + regex → parquet | 30-60 min |
| Blocking (per country) | polars explode/group_by, datasketch, jellyfish, joblib | 2-4h |
| Embedding (~20-40M short texts) | sentence-transformers fp16, T4×2 sharded | 1-3h |
| ANN blocking + pruning | faiss-gpu/cpu IVF | <1h |
| Feature computation (pruned pairs) | rapidfuzz via joblib chunking | 1-2h |
| Matcher A + B training (K=5 OOF each) | LightGBM (GPU histogram optional) + lightweight classifier | 2-3h |
| Reranker inference (bounded top-K/band, §5b) | cross-encoder, batched, T4×2 | **Unverified — measure on 100K-pair sample first; budget several hours pending that number, this is the plan's largest open compute risk** |
| Meta-model + joint threshold search | logistic regression / shallow LightGBM, grid search | <1h |
| Full test inference (~50M scored pairs worst case, excl. reranker) | vectorized predict | <30 min |

Convert every TSV to parquet immediately after normalization; use polars/duckdb (not pandas) for any 5M+ row operation given ~16GB Kaggle RAM. **Checkpoint every phase as a versioned Kaggle Dataset** (`normalized_v1`, `candidates_raw_v1`, `embeddings_v1`, `features_v1`, `matcher_a_v1`, `matcher_b_v1`, `rerank_v1`, `stack_v1`) so a new session loads the prior phase via "Add Data" instead of recomputing — this is what survives the ~9-12h session limit / ~30h weekly GPU quota. Kaggle "Internet: On" is only used for `pip install` of OSS libraries (polars, rapidfuzz, sentence-transformers, faiss, datasketch, jellyfish, indic_transliteration, lightgbm) and downloading pretrained model weights — package/weight installs, not entity lookups, so this stays compliant with the fair-play rule as read in Context (verify licenses per §5/§5b regardless).

### 9. Repo structure

```
code/business_entity_resolution/src/
├── config.py          # paths, blocking thresholds, top-N cap, K-fold config, hyperparameters
├── io_utils.py         # TSV<->parquet I/O; exact-format output writer (dedup + S2/S3-prefix enforcement built in)
├── normalize.py        # NFKC/casefold, suffix + abbreviation dicts, postal/locality/landmark extraction
├── transliterate.py    # Devanagari->IAST->ASCII wrapper, isolated/swappable
├── blocking.py         # inverted index, phonetic, MinHash, sorted-neighborhood, postal block, provenance tracking, pruning
├── embed.py            # batch embedding, FAISS index build/query, cosine feature, Matcher B feature export
├── features.py         # full pairwise feature set for Matcher A + shared meta features
├── rerank.py            # cross-encoder scoring on bounded top-K/uncertain-band subset (§5b)
├── matcher_a.py          # LightGBM, K-fold OOF training
├── matcher_b.py           # embedding-feature classifier, K-fold OOF training
├── stack.py                # meta-model training + joint (singleton, match) threshold search vs macro F_0.5
├── postprocess.py          # global one-to-one conflict resolution (§6d)
├── train.py                 # orchestrates K-fold OOF generation for Matcher A/B, calls stack.py
├── infer.py                  # end-to-end: blocking → Matcher A/B → rerank → stack → threshold → postprocess → both TSVs
└── validate_f05.py           # exact macro F_0.5 scorer (CLI + importable), scores full post-postprocess output
```

Kaggle notebooks, one per phase (each consumes the prior phase's Dataset, so any phase re-runs in isolation without burning session time on unaffected upstream work):

`01-eda.ipynb` → `02-normalize.ipynb` → `03-blocking.ipynb` (+ `04-embeddings.ipynb`, independent, can run in parallel) → `05-candidate-prune-features.ipynb` (consumes 03+04) → `06-matcher-a.ipynb` → `06b-matcher-b.ipynb` (can run in parallel with 06 once features/embeddings exist) → `07-rerank.ipynb` (consumes 06+06b's scores to pick the targeted top-K/band) → `08-stack-and-postprocess.ipynb` → `09-infer-submit.ipynb` (writes both TSVs, runs `validate_submission.py` as its last cell).

### 10. Execution roadmap (a valid submission exists at every milestone)

- **M0** — Scoped EDA (§1), confirms normalization/blocking assumptions.
- **M1** (first submittable) — Crude complete skeleton: minimal normalization + single token-overlap blocking key (country-partitioned) + hand-weighted string-similarity threshold, no ML yet. Produces both output TSVs, passes `validate_submission.py`, gets a first leaderboard number. Exercises the full pipeline shape so later milestones only swap internals.
- **M2** — Full normalization + multi-key blocking union (with provenance tracking for `num_blocks_agreeing`), recall ceiling ≥95% measured on train, proper top-N-pruned `candidate_pairs.tsv`, still rule-based scoring. Resubmit — confirms blocking correctness before investing in embeddings/matchers.
- **M3** — Add embedding component + FAISS ANN blocking key + cosine feature; re-measure recall ceiling/reduction ratio (expect gains on India Devanagari cases and on France generalization).
- **M4** — Matcher A (LightGBM) full feature set + K-fold entity split + `validate_f05.py` self-scoring; replace rule-based scorer with the tuned classifier. Resubmit. **This is a fully valid, independently-submittable checkpoint** — if everything past this point runs out of time or the reranker compute budget doesn't pan out, M4's single-model output is what ships; the ensemble stages below are additive, not a rewrite that leaves nothing runnable if cut short.
- **M4.5** — Matcher B (embedding-feature classifier) via the same K-fold OOF scheme; run the ablation from §6b before proceeding further.
- **M5** — Reranker stage: measure real throughput on a 100K-pair sample (§5b), derive the top-K/uncertain-band cutoffs from that number, implement targeted reranking.
- **M5.5** — Meta-model stacker + joint (singleton_threshold, match_threshold) grid search directly against held-out macro F_0.5 (§6c) — replaces Matcher A's standalone threshold search from M4.
- **M5.75** — Global conflict-resolution post-process (§6d).
- **M6** — Final polish: full test inference, `validate_submission.py --check-ids`, license verification for every pretrained model used (§5/§5b), filled `Documentation_template.md`, pinned `requirements.txt`, `code/.../README.md`, zip packaging exactly per the README's structure.

If time runs out after M2, that is already a validator-passing, precision-tunable submission; if it runs out after M4, that is a working single-matcher ML submission — no milestone leaves the user without a legitimate result to submit.

---

## Verification

- After every phase, run `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` from `student_resource/` — must print `PASS` before considering that milestone done.
- After M2 onward, run `src/validate_f05.py` against the held-out train split and record the macro F_0.5 — this is the number to track milestone-over-milestone (not the public leaderboard alone, which is a subset and rate-limited). From M5.5 onward this must be scored against the full post-stacking, post-conflict-resolution output, not a raw single-matcher score.
- Recall-ceiling check (§3) must be re-run any time blocking keys/thresholds change, and must stay ≥95% before trusting a drop in leaderboard score to mean "the model is wrong" rather than "candidates were lost upstream."
- Run the leave-one-country-out diagnostic (§7) once before trusting France-side performance — a country-stratified K-fold split alone cannot catch this failure mode.
- Spot-check `candidate_pairs.tsv` ⊇ `matching_results.tsv` per S1 entity manually on a sample before each submission (the validator also warns on this, but confirm it's near-zero, not just non-fatal).
- Spot-check that no S2/S3 ID appears in more than one S1 entity's final `matching_results.tsv` list after §6d's post-process — should be exactly zero given the verified data invariant.
- Run `validate_submission.py --check-ids` at least once before final submission despite its memory cost — it's the only local check that catches ID-hallucination bugs, which otherwise silently lower score without failing validation.
- Verify the actual license (not the assumed one) of every pretrained model used — `multilingual-e5-base`/`-large`, the reranker — on its HuggingFace model card before the final zip, since this gates the ≤8B-param/MIT-Apache hard constraint.
- Before the final zip, confirm `code/business_entity_resolution/` is genuinely runnable standalone (fresh Kaggle session, only `requirements.txt` + the checkpointed Datasets) end-to-end to `output/matching_results.tsv` + `output/candidate_pairs.tsv`.
