"""
v3 test inference: same per-country streaming pattern as train.py's
build_dataset() — one country at a time (read -> normalize -> block ->
features -> score), checkpointed to parquet, then reloaded once at the end
(small: scored pairs + candidate pairs only, not raw text) to write both
output TSVs in the exact validator format.
"""

import gc

import polars as pl
import lightgbm as lgb
from tqdm import tqdm

import config
import io_utils
from io_utils import write_results
from normalize import normalize_df
from blocking import generate_candidates
from features import add_features, FEATURE_COLS


def run_inference():
    print("Loading model + threshold...")
    model = lgb.Booster(model_file=str(config.OUTPUT_DIR / "matcher_a.txt"))
    with open(config.OUTPUT_DIR / "threshold.txt") as f:
        threshold = float(f.read().strip())

    countries = sorted(io_utils.list_countries(config.TEST_S1))
    print(f"Countries: {countries}")

    config.TEST_SCORED_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    config.TEST_CANDIDATE_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    all_s1_ids = []

    for country in tqdm(countries, desc="processing countries"):
        s1c_raw = io_utils.scan_source_country(config.TEST_S1, country)
        all_s1_ids.extend(s1c_raw["entity_id"].to_list())

        s1c = normalize_df(s1c_raw, label=f"test_s1[{country}]")
        s2c = normalize_df(io_utils.scan_source_country(config.TEST_S2, country), label=f"test_s2[{country}]")
        s3c = normalize_df(io_utils.scan_source_country(config.TEST_S3, country), label=f"test_s3[{country}]")

        candidates_c = generate_candidates(
            s1c, s2c, s3c, top_n=config.TOP_N_CANDIDATES,
            max_block_size=config.MAX_BLOCK_SIZE, max_pair_product=config.MAX_PAIR_PRODUCT,
        )

        others_c = pl.concat(
            [s2c.with_columns(pl.lit("S2").alias("src")), s3c.with_columns(pl.lit("S3").alias("src"))],
            how="vertical_relaxed",
        )
        feats_c = add_features(candidates_c, s1c, others_c)

        X = feats_c.select(FEATURE_COLS).to_numpy().astype("float64")
        probs = model.predict(X)
        feats_c = feats_c.with_columns(pl.Series("prob", probs))

        feats_c.select(["source1_entity_id", "candidate_id", "prob"]).write_parquet(
            config.TEST_SCORED_CHUNK_DIR / f"{country}.parquet"
        )
        candidates_c.select(["source1_entity_id", "candidate_id"]).write_parquet(
            config.TEST_CANDIDATE_CHUNK_DIR / f"{country}.parquet"
        )

        del s1c_raw, s1c, s2c, s3c, candidates_c, others_c, feats_c, X, probs
        gc.collect()

    print("Reloading scored + candidate pairs (small: no raw text)...")
    all_scored = pl.scan_parquet(str(config.TEST_SCORED_CHUNK_DIR / "*.parquet")).collect()
    all_candidates = pl.scan_parquet(str(config.TEST_CANDIDATE_CHUNK_DIR / "*.parquet")).collect()
    all_s1_series = pl.Series("entity_id", all_s1_ids)

    print("Writing candidate_pairs.tsv...")
    write_results(
        all_s1_series, all_candidates,
        config.OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids",
    )

    print(f"Writing matching_results.tsv (threshold={threshold:.2f})...")
    matches = all_scored.filter(pl.col("prob") >= threshold).select(["source1_entity_id", "candidate_id"])
    write_results(
        all_s1_series, matches,
        config.OUTPUT_DIR / "matching_results.tsv", "matched_entity_ids",
    )

    n_test_entities = len(all_s1_ids)
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
