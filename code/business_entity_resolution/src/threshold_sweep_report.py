"""
Fast standalone threshold-sweep report — reuses the ALREADY-COMPUTED val
candidates (output/_chunks/val_candidates/*.parquet) and the ALREADY-TRAINED
model (output/matcher_a.txt) from the last train.py run. No blocking, no
feature engineering, no retraining — just reloads and re-scores at each
requested threshold, so this is fast regardless of blocker runtime.

train.py's own threshold search already swept 0.50-0.99 in steps of 0.01 and
picked the best (0.83, per the last run's report) — this script exists to
print the full per-threshold table at the specific values requested, since
train.py only reports the winner.
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import numpy as np
import polars as pl
import lightgbm as lgb

import config
import io_utils
import validate_f05
from features import FEATURE_COLS

REQUESTED_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.83, 0.85, 0.90]


def main():
    print("Loading model + threshold...")
    model = lgb.Booster(model_file=str(config.OUTPUT_DIR / "matcher_a.txt"))

    print("Loading full ground-truth entity index...")
    all_s1_ids, true_by_s1 = validate_f05.load_full_gt_index(config.TRAIN_GT)

    print("Re-deriving the same train/val entity split train.py used (same seed)...")
    train_entity_ids, val_entity_ids = validate_f05.entity_train_val_split(
        all_s1_ids, config.VAL_FRACTION, config.RANDOM_SEED,
    )
    print(f"  {len(val_entity_ids):,} val entities")

    print("Loading val candidates (already-computed features, no recomputation)...")
    val_chunk_dir = config.TRAIN_CHUNK_DIR.parent / "val_candidates"
    val_candidates = pl.scan_parquet(str(val_chunk_dir / "*.parquet")).collect()
    print(f"  {val_candidates.height:,} val candidate rows, "
          f"{val_candidates['source1_entity_id'].n_unique():,} entities with >=1 candidate")

    print("Loading S1 country lookup...")
    s1_full = io_utils.read_source(config.TRAIN_S1)
    s1_country = dict(zip(s1_full["entity_id"].to_list(), s1_full["country"].to_list()))
    del s1_full

    print("Scoring val candidates with the existing model...")
    X_val = val_candidates.select(FEATURE_COLS).to_numpy().astype("float64")
    probs = model.predict(X_val) if X_val.shape[0] else np.array([])
    val_candidates = val_candidates.with_columns(pl.Series("prob", probs)) if X_val.shape[0] else val_candidates

    print(f"\n{'='*100}\nTHRESHOLD SWEEP (existing candidates + existing model, no retraining)\n{'='*100}")
    print(f"{'threshold':>10}{'macro_F0.5':>12}{'precision':>11}{'recall':>9}"
          f"{'singleton_acc':>15}{'avg_pred/S1':>13}")

    results = []
    for t in REQUESTED_THRESHOLDS:
        preds = val_candidates.filter(pl.col("prob") >= t) if val_candidates.height else val_candidates
        predicted_by_s1 = {}
        for s1id, cid in zip(preds["source1_entity_id"].to_list(), preds["candidate_id"].to_list()):
            predicted_by_s1.setdefault(s1id, set()).add(cid)
        result = validate_f05.score_entities(val_entity_ids, true_by_s1, predicted_by_s1, s1_country)
        results.append((t, result))
        print(f"{t:>10.2f}{result['macro_f05']:>12.4f}{result['macro_precision']:>11.4f}"
              f"{result['macro_recall']:>9.4f}{result['singleton_accuracy']:>15.4f}"
              f"{result['avg_predicted_match_count']:>13.3f}")

    best_t, best_r = max(results, key=lambda x: x[1]["macro_f05"])
    print(f"\nBest among requested thresholds: {best_t:.2f} (macro_F0.5={best_r['macro_f05']:.4f})")
    print("(train.py's own finer 0.50-0.99 step-0.01 sweep may have found a different optimum — "
          "check output/threshold.txt for that value.")
    with open(config.OUTPUT_DIR / "threshold.txt") as f:
        saved_threshold = f.read().strip()
    print(f" Currently saved threshold.txt: {saved_threshold})")


if __name__ == "__main__":
    main()
