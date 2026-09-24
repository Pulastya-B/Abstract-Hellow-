from pathlib import Path

# ── EDIT THESE TWO LINES for your environment ──
# Where the cloned repo lives (this determines where outputs get written)
REPO_DIR = Path("/content/<your-repo-name>")
# Where the dataset lives — NOT in the git repo (dataset/ is gitignored), so
# this must point at wherever you separately mounted/downloaded it (Drive, etc.)
DATASET_DIR = Path("/content/drive/MyDrive/student_resource/dataset")

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

TOP_N_CANDIDATES = 30           # candidates kept per S1 entity after blocking
NEG_PER_POS = 4                 # negative:positive sampling ratio for training
HARD_NEGATIVE_FRACTION = 0.7    # share of sampled negatives that are high-name_ratio non-matches
RANDOM_SEED = 42
VAL_FRACTION = 0.15             # fraction of S1 entities held out for threshold search
MAX_BLOCK_SIZE = 5000           # safety cap: skip a blocking key if EITHER side's group exceeds this
MAX_PAIR_PRODUCT = 2_000_000    # safety cap: skip a blocking key if the join size (n_s1 * n_candidates) exceeds this

# tqdm refresh throttle: some notebook/terminal output panes don't support
# carriage-return line overwriting, so every refresh becomes a new printed
# line instead of updating in place. A large mininterval keeps the log
# readable regardless — fewer refreshes, not fewer printed characters per one.
TQDM_MININTERVAL = 2.0
