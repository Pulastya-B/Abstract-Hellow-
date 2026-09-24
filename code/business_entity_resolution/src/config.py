from pathlib import Path

# ── EDIT THESE TWO LINES to match wherever you mounted the data on Kaggle/Colab ──
BASE_DIR = Path("/content/student_resource/student_resource")
OUTPUT_DIR = BASE_DIR / "output"

DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

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
MAX_BLOCK_SIZE = 20000          # safety cap: skip a blocking key if either side's group exceeds this
