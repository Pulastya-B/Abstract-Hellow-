from pathlib import Path

# ── gpu74 (college A40 box) paths ──
REPO_DIR = Path("/home/faculty/ritesh/ber_project")
DATASET_DIR = Path("/home/faculty/ritesh/ber_project/dataset")

OUTPUT_DIR = REPO_DIR / "output"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Per-country checkpoint chunks (parquet) so the pipeline never has to hold
# every country's data in memory at once — written incrementally, reloaded
# once at the end for the (much smaller) sampling/training/output step.
TRAIN_CHUNK_DIR = OUTPUT_DIR / "_chunks" / "train"
TEST_SCORED_CHUNK_DIR = OUTPUT_DIR / "_chunks" / "test_scored"
TEST_CANDIDATE_CHUNK_DIR = OUTPUT_DIR / "_chunks" / "test_candidates"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

TOP_N_CANDIDATES = 50           # candidates kept per S1 entity after blocking (raised from 30: the
                                 # union of 9 recall-oriented channels needs a wider cap than a single
                                 # rule-based key did, so true matches aren't pruned back out again)
NEG_PER_POS = 4                 # negative:positive sampling ratio for training
HARD_NEGATIVE_FRACTION = 0.7    # share of sampled negatives that are high-name_ratio non-matches
RANDOM_SEED = 42
VAL_FRACTION = 0.15             # fraction of S1 entities held out for threshold search
MAX_BLOCK_SIZE = 5000           # safety cap: drop a single blocking-key GROUP if EITHER side exceeds this
MAX_TOTAL_PAIRS = 3_000_000     # safety cap: cumulative join size across ALL groups of one key, per country

# ── Multi-channel blocking (recall-first redesign) ──
# A token is banned as a standalone blocking key if it appears in more than
# this fraction of a country partition's names — "inc"/"store"/"the" style
# tokens that carry ~no discriminative signal and would otherwise explode
# into a huge, useless join. This is a document-frequency ban, not a group-
# size drop: informative tokens are never discarded just for being common
# in one particular pairing, only when the token itself is near-universal.
RARE_TOKEN_MAX_DOC_FREQ = 0.01

# Word-level (not char n-gram) TF-IDF: char n-grams for short business names
# were measured at ~98% pairwise density (nearly every pair shares some
# 2-4-char sequence) — brute cosine over that is O(n_query x n_corpus)
# regardless of implementation and never finishes at India/US scale (proven
# directly: 20K x 100K took 90s+ on char n-grams vs. 0.03s on word tokens
# with a realistic vocabulary). Word vocab is large and selective enough that
# the similarity matrix stays genuinely sparse, which is what makes the
# inverted-index-style sparse matmul in _sparse_topk_pairs sub-quadratic.
# Fuzzy/typo tolerance that char n-grams would have caught is still covered
# by rare_token (channel 3) and the embedding channel (channel 9).
# TFIDF_TOP_K=40, chosen via benchmark_topk.py on a 15K-entity India sample:
# recall@20=66.63%, recall@40=69.03% (+2.40pts), recall@60=69.95% (+0.93pts
# more) — real but diminishing gain past 40, and 60 nearly doubles candidate
# volume/query time for well under 1 point of additional recall. ~30% of
# true matches aren't retrievable by name-only word TF-IDF at ANY k (rank
# >=60 in the same benchmark) — that gap is expected to be covered by the
# other 8 channels (exact_name/suffix_normalized/rare_token/postal/numeric/
# embedding/address_tfidf/composite_tfidf), not by raising k further.
TFIDF_TOP_K = 40
TFIDF_MIN_DF = 1
# max_df=1.0 (no exclusion) + max_features=50_000 — Config A from
# benchmark_tfidf_configs.py's vocabulary sweep, which tied for BEST
# retrieval recall (66.66%) among 10 tested configs (max_df in {1.0, 0.02},
# max_features in {50k..300k}, min_df in {1,2,5}, word 1-gram/2-gram, char
# 3-5/3-6-gram). Counterintuitively, excluding common terms via max_df=0.02
# measured slightly WORSE recall than keeping them (64.7% vs 66.66%) — for
# many true matches, shared generic words like "private"/"limited" still
# contribute real signal in combination with the rest of the name vector,
# so removing them loses information rather than adding selectivity. Char
# n-grams (3-5, 3-6) measured no recall benefit over word-level despite
# 10-15x higher fit/query cost, so they are not used for word_tfidf.
TFIDF_MAX_DF = 1.0
TFIDF_MAX_FEATURES = 50_000
TFIDF_SUBLINEAR_TF = True

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_TOP_K = 20               # neighbors retrieved per S1 row from the ANN index
EMBEDDING_BATCH_SIZE = 256
EMBEDDING_MAX_SEQ_LENGTH = 64       # names + short addresses; keeps encoding fast

# Country partitions above this many combined (S1 + S2 + S3) rows skip the
# embedding channel to keep single-box encoding time bounded — the other 8
# channels still cover it fully, so recall degrades gracefully rather than
# the run stalling on a multi-hour encode of the largest partition.
#
# SPRINT MODE (5-hour deadline to submission #2): lowered from 6,000,000 to
# 1,000,000 so embedding is skipped for BOTH US (~7.5M combined) and India
# (~5M combined) at full scale — measured directly that full embedding
# extrapolates to ~59h/country, and even India's cheap-only-channel recall
# (95.80%, measured on a 35K-entity representative sample) is close enough to
# the 97% target that shipping fast beats chasing the last few points via a
# channel that cannot finish in the time available. Revisit this cutoff once
# there's time to properly benchmark the residual-embedding-rescue approach
# (blocking.generate_candidates_with_rescue), which only embeds a small
# flagged subset of S1 entities instead of the full corpus.
EMBEDDING_MAX_PARTITION_ROWS = 1_000_000

RECALL_TARGET_PCT = 97.0          # do not proceed to matcher training below this

# SPRINT MODE (5-hour deadline to submission #2): name_tfidf ran at 154s/chunk
# (~19h ETA) on the full India partition at full scale — far slower than the
# few-seconds/chunk measured on the 35K-entity sample, almost certainly due
# to heavy contention on this shared box. Rather than lose more of the time
# budget diagnosing it, drop all 3 TF-IDF channels for this iteration and use
# only the 5 fastest channels (all measured in single-digit seconds even at
# full scale: exact_name, suffix_normalized, rare_token, postal, numeric).
# train.py/infer.py pass this explicitly to generate_candidates(channels=...)
# instead of the full _ALL_CHANNELS. Revisit once there's time to find out
# whether the TF-IDF slowdown was transient contention or a real regression.
ACTIVE_CHANNELS = ["exact_name", "suffix_normalized", "rare_token", "postal", "numeric"]

# tqdm refresh throttle: some notebook/terminal output panes don't support
# carriage-return line overwriting, so every refresh becomes a new printed
# line instead of updating in place. A large mininterval keeps the log
# readable regardless — fewer refreshes, not fewer printed characters per one.
TQDM_MININTERVAL = 5.0
