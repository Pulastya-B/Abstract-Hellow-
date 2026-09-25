"""
v4 training: single LightGBM matcher, now scored against the corrected,
competition-faithful validation framework (validate_f05.py) instead of the
old survivor-only approximation.

Key fix vs. v3: the train/val ENTITY split is drawn from the FULL S1
population in the ground truth file BEFORE blocking or negative sampling run
— not from whatever happened to survive into the sampled feature rows. Every
val S1 entity is scored, including ones blocking gave zero candidates to
(correctly counted as a recall miss, not silently dropped from the average).

Processes ONE COUNTRY AT A TIME end to end (read -> normalize -> block ->
features -> label), checkpointing each country's engineered features to a
parquet chunk and freeing memory before moving to the next — unchanged from
v3, this part was never the source of the validation bias.
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import gc
import time

import numpy as np
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

import config
import io_utils
from io_utils import read_ground_truth, explode_ground_truth
from normalize import normalize_df
from blocking import generate_candidates
from features import add_features, FEATURE_COLS
import validate_f05


def build_dataset(train_entity_ids: set, val_entity_ids: set, gt_pairs: set):
    """
    Runs blocking + features for every country, over the FULL S1 population
    (both train_entity_ids and val_entity_ids) — val entities need candidates
    generated too, exactly as they would at real inference time, so blocking
    recall can be measured on them independently of the matcher.

    Returns:
      train_rows: sampled (positive + hard/easy negative) feature rows for
                  TRAIN entities only — this is what the model trains on.
      val_candidates: ALL candidate rows for VAL entities (no negative
                  sampling — every val entity's full candidate set is scored).
      dataset_stats, candidate_by_s1 (val-only, for blocking-recall scoring)
    """
    countries = sorted(io_utils.list_countries(config.TRAIN_S1))
    print(f"Countries: {countries}")

    config.TRAIN_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    n_candidate_pairs_total = 0
    n_recalled_total = 0
    n_gt_pairs = len(gt_pairs)

    val_chunk_dir = config.TRAIN_CHUNK_DIR.parent / "val_candidates"
    val_chunk_dir.mkdir(parents=True, exist_ok=True)

    for country in tqdm(countries, desc="processing countries"):
        country_t0 = time.time()
        print(f"\n>>> [{country}] starting: normalize -> block -> features -> label", flush=True)

        s1c_full = normalize_df(io_utils.scan_source_country(config.TRAIN_S1, country), label=f"train_s1[{country}]")
        s2c = normalize_df(io_utils.scan_source_country(config.TRAIN_S2, country), label=f"train_s2[{country}]")
        s3c = normalize_df(io_utils.scan_source_country(config.TRAIN_S3, country), label=f"train_s3[{country}]")
        print(f">>> [{country}] normalization done at +{time.time()-country_t0:.0f}s "
              f"(S1={s1c_full.height:,} S2={s2c.height:,} S3={s3c.height:,})", flush=True)

        # Blocking runs over EVERY S1 entity in this country (train + val) —
        # val entities must go through the exact same candidate-generation
        # path they would at real inference time, so blocking recall on them
        # is measured honestly rather than assumed.
        candidates_c = generate_candidates(
            s1c_full, s2c, s3c, top_n=config.TOP_N_CANDIDATES,
            max_block_size=config.MAX_BLOCK_SIZE, max_total_pairs=config.MAX_TOTAL_PAIRS,
        )
        n_candidate_pairs_total += candidates_c.height
        print(f">>> [{country}] blocking done at +{time.time()-country_t0:.0f}s "
              f"({candidates_c.height:,} candidate pairs)", flush=True)

        others_c = pl.concat(
            [s2c.with_columns(pl.lit("S2").alias("src")), s3c.with_columns(pl.lit("S3").alias("src"))],
            how="vertical_relaxed",
        )
        feats_c = add_features(candidates_c, s1c_full, others_c)
        print(f">>> [{country}] feature engineering done at +{time.time()-country_t0:.0f}s", flush=True)

        labels = [
            1 if (s1id, cid) in gt_pairs else 0
            for s1id, cid in zip(feats_c["source1_entity_id"].to_list(), feats_c["candidate_id"].to_list())
        ]
        feats_c = feats_c.with_columns(pl.Series("label", labels))
        n_recalled_total += sum(labels)

        is_val = pl.col("source1_entity_id").is_in(list(val_entity_ids))
        val_feats_c = feats_c.filter(is_val)
        train_feats_c = feats_c.filter(~is_val)

        train_feats_c.write_parquet(config.TRAIN_CHUNK_DIR / f"{country}.parquet")
        val_feats_c.write_parquet(val_chunk_dir / f"{country}.parquet")
        print(f">>> [{country}] FINISHED in {time.time()-country_t0:.0f}s total "
              f"({sum(labels):,} true pairs recalled)", flush=True)

        del s1c_full, s2c, s3c, candidates_c, others_c, feats_c, labels, val_feats_c, train_feats_c
        gc.collect()

    recall_ceiling_pct = 100 * n_recalled_total / max(n_gt_pairs, 1)
    print(f"  recall check (train+val entities combined): {n_recalled_total}/{n_gt_pairs} true pairs "
          f"survived blocking ({recall_ceiling_pct:.1f}%) — target >={config.RECALL_TARGET_PCT}%")

    print("Reloading TRAIN engineered features for sampling...")
    train_feats = pl.scan_parquet(str(config.TRAIN_CHUNK_DIR / "*.parquet")).collect()

    pos = train_feats.filter(pl.col("label") == 1)
    neg = train_feats.filter(pl.col("label") == 0)
    n_neg = min(len(neg), len(pos) * config.NEG_PER_POS)

    n_hard = int(n_neg * config.HARD_NEGATIVE_FRACTION)
    n_easy = n_neg - n_hard
    neg_sorted = neg.sort("name_ratio", descending=True)
    hard_neg = neg_sorted.head(n_hard)
    remaining = neg_sorted.slice(n_hard, len(neg_sorted) - n_hard)
    n_easy = min(n_easy, len(remaining))
    easy_neg = remaining.sample(n=n_easy, seed=config.RANDOM_SEED) if n_easy > 0 else remaining.head(0)
    neg = pl.concat([hard_neg, easy_neg], how="vertical_relaxed")

    train_rows = pl.concat([pos, neg], how="vertical_relaxed")
    print(f"  training rows: {len(pos)} positive, {len(neg)} negative "
          f"({len(hard_neg)} hard / {len(easy_neg)} easy)")

    print("Reloading VAL candidates (full pool, no sampling)...")
    val_candidates = pl.scan_parquet(str(val_chunk_dir / "*.parquet")).collect()

    dataset_stats = {
        "n_ground_truth_pairs": n_gt_pairs,
        "n_candidate_pairs": n_candidate_pairs_total,
        "n_positives_recalled": n_recalled_total,
        "recall_ceiling_pct": recall_ceiling_pct,
        "n_train_positive_rows": len(pos),
        "n_train_negative_rows": len(neg),
    }
    return train_rows, val_candidates, dataset_stats


def train_model():
    print("Loading full ground-truth entity index...")
    all_s1_ids, true_by_s1 = validate_f05.load_full_gt_index(config.TRAIN_GT)
    gt = read_ground_truth(config.TRAIN_GT)
    pos_pairs = explode_ground_truth(gt)
    gt_pairs = set(zip(pos_pairs["source1_entity_id"].to_list(), pos_pairs["candidate_id"].to_list()))
    del gt, pos_pairs
    gc.collect()

    # Entity split drawn from the FULL S1 population, before any blocking or
    # sampling — this is the fix: val_ids no longer depends on what happens
    # to survive into the sampled feature rows.
    train_entity_ids, val_entity_ids = validate_f05.entity_train_val_split(
        all_s1_ids, config.VAL_FRACTION, config.RANDOM_SEED,
    )
    print(f"Full entity split: {len(train_entity_ids):,} train / {len(val_entity_ids):,} val "
          f"(out of {len(all_s1_ids):,} total S1 entities)")

    train_rows, val_candidates, dataset_stats = build_dataset(train_entity_ids, val_entity_ids, gt_pairs)

    s1_full = io_utils.read_source(config.TRAIN_S1)
    s1_country = dict(zip(s1_full["entity_id"].to_list(), s1_full["country"].to_list()))
    del s1_full
    gc.collect()

    candidate_by_s1 = {}
    for s1id, cid in zip(val_candidates["source1_entity_id"].to_list(), val_candidates["candidate_id"].to_list()):
        candidate_by_s1.setdefault(s1id, set()).add(cid)

    blocking_recall = validate_f05.score_blocking_recall(val_entity_ids, true_by_s1, candidate_by_s1)

    X_train = train_rows.select(FEATURE_COLS).to_numpy().astype("float64")
    y_train = train_rows["label"].to_numpy()

    print(f"Training LightGBM on {X_train.shape[0]:,} rows x {X_train.shape[1]} features...", flush=True)
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        objective="binary",
        random_state=config.RANDOM_SEED,
        verbose=1,
    )
    model.fit(X_train, y_train)
    print(f"LightGBM training done in {time.time()-t0:.0f}s", flush=True)

    print(f"Scoring VAL candidates ({val_candidates.height:,} rows, "
          f"{val_candidates['source1_entity_id'].n_unique():,} entities with >=1 candidate)...")
    X_val = val_candidates.select(FEATURE_COLS).to_numpy().astype("float64")
    probs = model.predict_proba(X_val)[:, 1] if X_val.shape[0] else np.array([])
    val_candidates = val_candidates.with_columns(pl.Series("prob", probs)) if X_val.shape[0] else val_candidates

    print("Searching threshold for macro F_0.5 over the FULL val population "
          "(including val entities blocking gave zero candidates to)...")
    best_threshold, best_score = None, None
    for t in tqdm(np.arange(0.50, 0.99, 0.01), desc="threshold search", mininterval=config.TQDM_MININTERVAL):
        preds = val_candidates.filter(pl.col("prob") >= float(t)) if val_candidates.height else val_candidates
        predicted_by_s1 = {}
        for s1id, cid in zip(preds["source1_entity_id"].to_list(), preds["candidate_id"].to_list()):
            predicted_by_s1.setdefault(s1id, set()).add(cid)
        result = validate_f05.score_entities(val_entity_ids, true_by_s1, predicted_by_s1, s1_country)
        if best_score is None or result["macro_f05"] > best_score["macro_f05"]:
            best_threshold, best_score = float(t), result

    print(f"Best threshold: {best_threshold:.2f}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(config.OUTPUT_DIR / "matcher_a.txt"))
    with open(config.OUTPUT_DIR / "threshold.txt", "w") as f:
        f.write(str(best_threshold))

    validate_f05.print_report(blocking_recall, best_score, best_threshold, dataset_stats)

    return model, {"blocking_recall": blocking_recall, "matcher_scores": best_score, "threshold": best_threshold}


if __name__ == "__main__":
    train_model()
