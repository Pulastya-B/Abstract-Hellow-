"""
v2 training: single LightGBM matcher (plan.md §6a, without the K-fold OOF/
ensemble machinery from the later milestones — that's layered in once this
baseline is proven), now with hard-negative-prioritized sampling. Builds
labeled rows from the pruned candidate pool, trains, and grid-searches a
threshold directly against macro F_0.5.
"""

import numpy as np
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

import config
from io_utils import read_source, read_ground_truth, explode_ground_truth
from normalize import normalize_df
from blocking import generate_candidates
from features import add_features, FEATURE_COLS


def build_dataset():
    print("Loading + normalizing train sources...")
    s1 = normalize_df(read_source(config.TRAIN_S1), label="train_s1")
    s2 = normalize_df(read_source(config.TRAIN_S2), label="train_s2")
    s3 = normalize_df(read_source(config.TRAIN_S3), label="train_s3")
    gt = read_ground_truth(config.TRAIN_GT)
    pos_pairs = explode_ground_truth(gt)
    print(f"  {len(pos_pairs)} ground-truth positive pairs")

    print("Blocking...")
    candidates = generate_candidates(
        s1, s2, s3, top_n=config.TOP_N_CANDIDATES,
        max_block_size=config.MAX_BLOCK_SIZE, max_pair_product=config.MAX_PAIR_PRODUCT,
    )
    print(f"  {len(candidates)} candidate pairs generated")

    others = pl.concat(
        [
            s2.with_columns(pl.lit("S2").alias("src")),
            s3.with_columns(pl.lit("S3").alias("src")),
        ],
        how="vertical_relaxed",
    )

    print("Computing features...")
    feats = add_features(candidates, s1, others)

    pos_set = set(zip(pos_pairs["source1_entity_id"].to_list(), pos_pairs["candidate_id"].to_list()))
    labels = [
        1 if (s1id, cid) in pos_set else 0
        for s1id, cid in zip(feats["source1_entity_id"].to_list(), feats["candidate_id"].to_list())
    ]
    feats = feats.with_columns(pl.Series("label", labels))

    n_pos = int(sum(labels))
    n_recalled = n_pos
    recall_ceiling_pct = 100 * n_recalled / max(len(pos_pairs), 1)
    print(f"  recall check: {n_recalled}/{len(pos_pairs)} true pairs survived blocking "
          f"({recall_ceiling_pct:.1f}%) — should be checked against "
          f"the >=97% target in plan.md §3 before trusting downstream numbers")

    pos = feats.filter(pl.col("label") == 1)
    neg = feats.filter(pl.col("label") == 0)
    n_neg = min(len(neg), len(pos) * config.NEG_PER_POS)

    # Hard-negative-prioritized sampling: mostly high-name_ratio non-matches
    # (what F_0.5 punishes hardest), plus a smaller random slice for diversity —
    # not a uniform random sample, per plan.md §6a.
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

    dataset_stats = {
        "n_ground_truth_pairs": len(pos_pairs),
        "n_candidate_pairs": len(candidates),
        "n_positives_recalled": n_recalled,
        "recall_ceiling_pct": recall_ceiling_pct,
        "n_train_positive_rows": len(pos),
        "n_train_negative_rows": len(neg),
    }
    return train_rows, pos_pairs, dataset_stats


def entity_split(rows: pl.DataFrame):
    s1_ids = rows["source1_entity_id"].unique().to_list()
    rng = np.random.default_rng(config.RANDOM_SEED)
    rng.shuffle(s1_ids)
    n_val = int(len(s1_ids) * config.VAL_FRACTION)
    val_ids = set(s1_ids[:n_val])
    val_mask = pl.col("source1_entity_id").is_in(list(val_ids))
    return rows.filter(~val_mask), rows.filter(val_mask), val_ids


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)


def score_at_threshold(val_rows: pl.DataFrame, probs: np.ndarray, threshold: float, val_ids: set, gt_pairs: set) -> dict:
    """Full per-entity breakdown at a given threshold: macro F_0.5, mean precision/recall,
    and singleton accuracy (fraction of true-empty entities correctly predicted empty)."""
    val = val_rows.with_columns(pl.Series("prob", probs))
    preds = val.filter(pl.col("prob") >= threshold)

    pred_by_s1 = {}
    for s1id, cid in zip(preds["source1_entity_id"].to_list(), preds["candidate_id"].to_list()):
        pred_by_s1.setdefault(s1id, set()).add(cid)

    true_by_s1 = {}
    for s1id, cid in gt_pairs:
        if s1id in val_ids:
            true_by_s1.setdefault(s1id, set()).add(cid)

    f_scores, precisions, recalls = [], [], []
    n_true_singletons = n_true_singletons_correct = 0
    for s1id in val_ids:
        pred = pred_by_s1.get(s1id, set())
        true = true_by_s1.get(s1id, set())
        if not true:
            n_true_singletons += 1
            if not pred:
                f_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
                n_true_singletons_correct += 1
            else:
                f_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
        else:
            tp = len(pred & true)
            precision = tp / len(pred) if pred else 0.0
            recall = tp / len(true) if true else 0.0
            f_scores.append(f_beta(precision, recall))
            precisions.append(precision)
            recalls.append(recall)

    return {
        "threshold": threshold,
        "macro_f05": float(np.mean(f_scores)) if f_scores else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "n_true_singletons": n_true_singletons,
        "singleton_accuracy": n_true_singletons_correct / n_true_singletons if n_true_singletons else 0.0,
    }


def macro_f05(val_rows: pl.DataFrame, probs: np.ndarray, threshold: float, val_ids: set, gt_pairs: set) -> float:
    return score_at_threshold(val_rows, probs, threshold, val_ids, gt_pairs)["macro_f05"]


def train_model():
    train_rows, pos_pairs, dataset_stats = build_dataset()
    train_split, val_split, val_ids = entity_split(train_rows)
    print(f"Train/val entity split: {train_split['source1_entity_id'].n_unique()} / "
          f"{len(val_ids)} entities")

    X_train = train_split.select(FEATURE_COLS).to_numpy().astype("float64")
    y_train = train_split["label"].to_numpy()
    X_val = val_split.select(FEATURE_COLS).to_numpy().astype("float64")

    print("Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        objective="binary",
        random_state=config.RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]

    gt_pairs = set(zip(pos_pairs["source1_entity_id"].to_list(), pos_pairs["candidate_id"].to_list()))

    print("Searching threshold for macro F_0.5...")
    print("NOTE: this validation set only covers S1 entities that had >=1 candidate "
          "survive blocking+sampling; it is an approximation of the real recall-ceiling-"
          "aware score from plan.md §7, not the full-pool version yet.")
    best_result = None
    for t in tqdm(np.arange(0.50, 0.99, 0.01), desc="threshold search"):
        result = score_at_threshold(val_split, probs, float(t), val_ids, gt_pairs)
        if best_result is None or result["macro_f05"] > best_result["macro_f05"]:
            best_result = result

    print(f"Best threshold: {best_result['threshold']:.2f}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(config.OUTPUT_DIR / "matcher_a.txt"))
    with open(config.OUTPUT_DIR / "threshold.txt", "w") as f:
        f.write(str(best_result["threshold"]))

    metrics = {**dataset_stats, **best_result, "n_train_entities": train_split["source1_entity_id"].n_unique(),
               "n_val_entities": len(val_ids)}

    print("\n===== TRAINING SCORES =====")
    print(f"  Ground-truth positive pairs:      {metrics['n_ground_truth_pairs']:,}")
    print(f"  Candidate pairs after blocking:   {metrics['n_candidate_pairs']:,}")
    print(f"  Recall ceiling (blocking):        {metrics['recall_ceiling_pct']:.1f}%  (target >=97%)")
    print(f"  Train/val entities:               {metrics['n_train_entities']:,} / {metrics['n_val_entities']:,}")
    print(f"  Best threshold:                   {metrics['threshold']:.2f}")
    print(f"  Val macro F_0.5 (approx):         {metrics['macro_f05']:.4f}")
    print(f"  Val mean precision:               {metrics['mean_precision']:.4f}")
    print(f"  Val mean recall:                  {metrics['mean_recall']:.4f}")
    print(f"  Val singleton accuracy:           {metrics['singleton_accuracy']:.4f}  "
          f"({metrics['n_true_singletons']:,} true singletons in val)")
    print(f"  Saved model + threshold to:       {config.OUTPUT_DIR}")
    print("============================\n")

    return model, metrics


if __name__ == "__main__":
    train_model()
