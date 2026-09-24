"""
v2 test inference: multi-key blocking -> features (incl. block-agreement +
cross-source consistency) -> scored with the trained matcher_a.txt -> both
output TSVs in the exact validator format.
"""

import polars as pl
import lightgbm as lgb

import config
from io_utils import read_source, write_results
from normalize import normalize_df
from blocking import generate_candidates
from features import add_features, FEATURE_COLS


def run_inference():
    print("Loading + normalizing test sources...")
    s1 = normalize_df(read_source(config.TEST_S1))
    s2 = normalize_df(read_source(config.TEST_S2))
    s3 = normalize_df(read_source(config.TEST_S3))

    print("Blocking...")
    candidates = generate_candidates(s1, s2, s3, top_n=config.TOP_N_CANDIDATES)

    others = pl.concat(
        [
            s2.with_columns(pl.lit("S2").alias("src")),
            s3.with_columns(pl.lit("S3").alias("src")),
        ],
        how="vertical_relaxed",
    )

    print("Computing features...")
    feats = add_features(candidates, s1, others)

    print("Scoring...")
    model = lgb.Booster(model_file=str(config.OUTPUT_DIR / "matcher_a.txt"))
    with open(config.OUTPUT_DIR / "threshold.txt") as f:
        threshold = float(f.read().strip())

    X = feats.select(FEATURE_COLS).to_numpy().astype("float64")
    probs = model.predict(X)
    feats = feats.with_columns(pl.Series("prob", probs))

    print("Writing candidate_pairs.tsv...")
    write_results(
        s1["entity_id"],
        candidates.select(["source1_entity_id", "candidate_id"]),
        config.OUTPUT_DIR / "candidate_pairs.tsv",
        "candidate_entity_ids",
    )

    print(f"Writing matching_results.tsv (threshold={threshold:.2f})...")
    matches = feats.filter(pl.col("prob") >= threshold).select(["source1_entity_id", "candidate_id"])
    write_results(
        s1["entity_id"],
        matches,
        config.OUTPUT_DIR / "matching_results.tsv",
        "matched_entity_ids",
    )

    n_test_entities = s1.height
    matched_entities = matches["source1_entity_id"].n_unique()
    n_matched_pairs = matches.height
    n_predicted_singletons = n_test_entities - matched_entities
    stats = {
        "n_test_entities": n_test_entities,
        "n_matched_entities": matched_entities,
        "n_matched_pairs": n_matched_pairs,
        "n_predicted_singletons": n_predicted_singletons,
        "predicted_singleton_rate": 100 * n_predicted_singletons / max(n_test_entities, 1),
    }

    print("\n===== INFERENCE SCORES =====")
    print(f"  Test S1 entities:                 {stats['n_test_entities']:,}")
    print(f"  Entities with >=1 predicted match: {stats['n_matched_entities']:,}")
    print(f"  Total predicted match pairs:       {stats['n_matched_pairs']:,}")
    print(f"  Predicted singletons:              {stats['n_predicted_singletons']:,} "
          f"({stats['predicted_singleton_rate']:.1f}%  — train true rate was 5.6%, sanity-check against that)")
    print("=============================\n")

    return stats


if __name__ == "__main__":
    run_inference()
