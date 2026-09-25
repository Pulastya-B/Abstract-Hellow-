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
TFIDF_TOP_K = 20                  # neighbors retrieved per S1 row, per TF-IDF channel
TFIDF_MIN_DF = 1
TFIDF_MAX_FEATURES = 50_000       # cap vocabulary size per country partition (memory guard)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_TOP_K = 20               # neighbors retrieved per S1 row from the ANN index
EMBEDDING_BATCH_SIZE = 256
EMBEDDING_MAX_SEQ_LENGTH = 64       # names + short addresses; keeps encoding fast

# Country partitions above this many combined (S1 + S2 + S3) rows skip the
# embedding channel to keep single-box encoding time bounded — the other 8
# channels still cover it fully, so recall degrades gracefully rather than
# the run stalling on a multi-hour encode of the largest partition.
EMBEDDING_MAX_PARTITION_ROWS = 6_000_000

RECALL_TARGET_PCT = 97.0          # do not proceed to matcher training below this

# tqdm refresh throttle: some notebook/terminal output panes don't support
# carriage-return line overwriting, so every refresh becomes a new printed
# line instead of updating in place. A large mininterval keeps the log
# readable regardless — fewer refreshes, not fewer printed characters per one.
TQDM_MININTERVAL = 5.0
