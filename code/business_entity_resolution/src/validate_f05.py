"""
Competition-faithful validation framework (plan.md §7 / §11 self-scoring).

Replaces the old approximation in train.py's score_at_threshold(), which only
scored S1 entities that happened to survive INTO the sampled train_rows —
i.e. entities that had >=1 candidate after blocking AND (for negatives)
survived NEG_PER_POS/hard-negative sampling. Any S1 entity whose true matches
were entirely lost by blocking, or whose only candidate rows got dropped by
negative sampling, silently never appeared in val_ids at all, inflating the
score by hiding blocking's own losses inside a validation set already
filtered down to blocking's survivors.

Fix: the val/train ENTITY split happens first, directly on the full S1
population from the ground truth file — before blocking or sampling ever
run. Every val S1 entity is then scored, including ones with zero surviving
candidates (scored as recall=0 if they have true matches they never got a
chance at, or singleton-correct if they truly have none).

This module is also the reusable K-fold entity-split utility for out-of-fold
(OOF) training: Matcher A / Matcher B / any future meta-model or singleton
detector must be scored on OOF predictions, never in-fold, or they'll see
artificially confident scores that don't reflect real test-time distributions.
"""

import numpy as np
import polars as pl

import config
import io_utils


def load_full_gt_index(gt_path):
    """
    Returns:
      all_s1_ids: list of every S1 entity id in the ground truth file
      true_by_s1: dict s1_id -> frozenset of true matched ids (empty set = singleton)
    """
    gt = io_utils.read_ground_truth(gt_path)
    all_s1_ids = gt["source1_entity_id"].to_list()
    match_lists = gt["matched_entity_ids"].to_list()
    true_by_s1 = {}
    for s1id, matches in zip(all_s1_ids, match_lists):
        if matches:
            true_by_s1[s1id] = frozenset(matches.split(","))
        else:
            true_by_s1[s1id] = frozenset()
    return all_s1_ids, true_by_s1


def entity_train_val_split(all_s1_ids: list, val_fraction: float, seed: int):
    """Single train/val split over the FULL S1 population (not blocking survivors)."""
    ids = list(all_s1_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def entity_kfold_split(all_s1_ids: list, k: int, seed: int):
    """
    Yields (train_ids, val_ids) for K entity-disjoint folds over the FULL S1
    population. Used for out-of-fold (OOF) training: Matcher A/B and any
    downstream meta-model/singleton-detector must be trained on K-1 folds and
    predict on the held-out fold, so their OOF predictions never leak
    in-sample confidence into the meta-model or threshold search.
    """
    ids = list(all_s1_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    folds = np.array_split(np.array(ids, dtype=object), k)
    for i in range(k):
        val_ids = set(folds[i].tolist())
        train_ids = set(ids) - val_ids
        yield train_ids, val_ids


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)


def _cardinality_bucket(n_true: int) -> str:
    if n_true == 0:
        return "0_singleton"
    if n_true == 1:
        return "1_single_match"
    if n_true <= 3:
        return "2-3_matches"
    if n_true <= 6:
        return "4-6_matches"
    return "7+_matches"


def _source_mix_bucket(true_ids: frozenset) -> str:
    has_s2 = any(i.startswith("S2-") for i in true_ids)
    has_s3 = any(i.startswith("S3-") for i in true_ids)
    if not true_ids:
        return "singleton"
    if has_s2 and has_s3:
        return "both_S2_S3"
    if has_s2:
        return "S2_only"
    return "S3_only"


def score_entities(
    val_ids: set,
    true_by_s1: dict,
    predicted_by_s1: dict,
    s1_country: dict = None,
) -> dict:
    """
    Competition-faithful per-S1 macro F0.5, computed over EVERY val_ids entity
    — including ones absent from predicted_by_s1 entirely (treated as an
    empty prediction, exactly as the real scorer/validator would see a
    missing-but-required row filled with an empty match list).

    val_ids: full set of validation S1 entity ids (not just ones with candidates)
    true_by_s1: s1_id -> frozenset of true matched ids (from load_full_gt_index)
    predicted_by_s1: s1_id -> set of predicted matched ids (only for ids present;
                      absent ids are treated as predicting empty)
    s1_country: optional s1_id -> country, for the by-country breakdown
    """
    f_scores, precisions, recalls = [], [], []
    n_true_singletons = n_true_singletons_correct = n_singleton_false_positive = 0
    predicted_counts = []
    per_cardinality = {}
    per_country = {}
    per_source_mix = {}

    for s1id in val_ids:
        true_ids = true_by_s1.get(s1id, frozenset())
        pred_ids = predicted_by_s1.get(s1id, set())
        predicted_counts.append(len(pred_ids))

        if not true_ids:
            n_true_singletons += 1
            if not pred_ids:
                f, p, r = 1.0, 1.0, 1.0
                n_true_singletons_correct += 1
            else:
                f, p, r = 0.0, 0.0, 0.0
                n_singleton_false_positive += 1
        else:
            tp = len(pred_ids & true_ids)
            p = tp / len(pred_ids) if pred_ids else 0.0
            r = tp / len(true_ids) if true_ids else 0.0
            f = f_beta(p, r)

        f_scores.append(f)
        precisions.append(p)
        recalls.append(r)

        card_bucket = _cardinality_bucket(len(true_ids))
        per_cardinality.setdefault(card_bucket, []).append(f)

        mix_bucket = _source_mix_bucket(true_ids)
        per_source_mix.setdefault(mix_bucket, []).append(f)

        if s1_country is not None:
            c = s1_country.get(s1id, "UNKNOWN")
            per_country.setdefault(c, []).append(f)

    def _agg(d):
        return {k: {"macro_f05": float(np.mean(v)), "n": len(v)} for k, v in d.items()}

    return {
        "n_val_entities": len(val_ids),
        "macro_f05": float(np.mean(f_scores)) if f_scores else 0.0,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "n_true_singletons": n_true_singletons,
        "singleton_accuracy": n_true_singletons_correct / n_true_singletons if n_true_singletons else 0.0,
        "singleton_false_positive_rate": n_singleton_false_positive / n_true_singletons if n_true_singletons else 0.0,
        "avg_predicted_match_count": float(np.mean(predicted_counts)) if predicted_counts else 0.0,
        "score_by_cardinality": _agg(per_cardinality),
        "score_by_country": _agg(per_country) if s1_country is not None else None,
        "score_by_source_mix": _agg(per_source_mix),
    }


def score_blocking_recall(val_ids: set, true_by_s1: dict, candidate_by_s1: dict) -> dict:
    """
    Blocking-only recall: for each val S1, what fraction of its TRUE matches
    survived into the candidate set (before any matcher/threshold is applied).
    Reported separately from matcher recall per the requirement to not
    conflate "blocking never gave the matcher a chance" with "the matcher
    scored a surfaced candidate too low."
    """
    n_true_total = 0
    n_recovered_total = 0
    per_entity_recall = []
    for s1id in val_ids:
        true_ids = true_by_s1.get(s1id, frozenset())
        if not true_ids:
            continue
        cand_ids = candidate_by_s1.get(s1id, set())
        recovered = len(true_ids & cand_ids)
        n_true_total += len(true_ids)
        n_recovered_total += recovered
        per_entity_recall.append(recovered / len(true_ids))

    return {
        "n_true_pairs": n_true_total,
        "n_recovered_pairs": n_recovered_total,
        "pair_level_recall_pct": 100 * n_recovered_total / n_true_total if n_true_total else 0.0,
        "entity_level_mean_recall_pct": 100 * float(np.mean(per_entity_recall)) if per_entity_recall else 0.0,
        "n_entities_with_true_matches": len(per_entity_recall),
    }


def print_report(blocking_recall: dict, matcher_scores: dict, threshold: float, dataset_stats: dict = None):
    print("\n" + "=" * 100)
    print("CORRECTED BASELINE VALIDATION REPORT (full S1 population, no survivor-only bias)")
    print("=" * 100)

    if dataset_stats:
        print("\n--- Dataset ---")
        for k, v in dataset_stats.items():
            print(f"  {k}: {v}")

    print("\n--- Blocking recall (independent of matcher/threshold) ---")
    print(f"  True pairs (val entities only):          {blocking_recall['n_true_pairs']:,}")
    print(f"  Recovered by blocking:                   {blocking_recall['n_recovered_pairs']:,}")
    print(f"  Pair-level recall:                        {blocking_recall['pair_level_recall_pct']:.2f}%")
    print(f"  Entity-level mean recall:                  {blocking_recall['entity_level_mean_recall_pct']:.2f}%")
    print(f"  Val entities with >=1 true match:          {blocking_recall['n_entities_with_true_matches']:,}")

    print(f"\n--- Final prediction quality (threshold={threshold:.2f}), FULL val population ---")
    print(f"  Val entities scored:                       {matcher_scores['n_val_entities']:,}")
    print(f"  Macro F_0.5:                                {matcher_scores['macro_f05']:.4f}")
    print(f"  Macro precision:                            {matcher_scores['macro_precision']:.4f}")
    print(f"  Macro recall:                               {matcher_scores['macro_recall']:.4f}")
    print(f"  True singletons:                            {matcher_scores['n_true_singletons']:,}")
    print(f"  Singleton accuracy:                         {matcher_scores['singleton_accuracy']:.4f}")
    print(f"  Singleton false-positive rate:               {matcher_scores['singleton_false_positive_rate']:.4f}")
    print(f"  Avg predicted match count:                  {matcher_scores['avg_predicted_match_count']:.3f}")

    print("\n--- Score by true-cardinality bucket ---")
    for bucket, s in sorted(matcher_scores["score_by_cardinality"].items()):
        print(f"  {bucket:<20} macro_F0.5={s['macro_f05']:.4f}  (n={s['n']:,})")

    if matcher_scores["score_by_country"]:
        print("\n--- Score by country ---")
        for c, s in sorted(matcher_scores["score_by_country"].items()):
            print(f"  {c:<12} macro_F0.5={s['macro_f05']:.4f}  (n={s['n']:,})")

    print("\n--- Score by source mix (S2-only / S3-only / both / singleton) ---")
    for mix, s in sorted(matcher_scores["score_by_source_mix"].items()):
        print(f"  {mix:<14} macro_F0.5={s['macro_f05']:.4f}  (n={s['n']:,})")

    print("=" * 100)
