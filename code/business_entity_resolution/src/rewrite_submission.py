"""
Fast fix-and-rewrite: reuses the already-scored test parquet chunks from the
last infer.py run (no re-blocking, no re-scoring) to regenerate both output
TSVs with io_utils.write_results' quote_style="never" fix — the previous
files had Polars' default CSV quoting wrap empty singleton rows in a literal
"" (two double-quote characters) instead of a truly empty field, which the
validator's plain split(",") read as a garbage non-S2/S3-prefixed "ID".
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import polars as pl

import config
import io_utils
from io_utils import write_results


def main():
    print("Reloading already-scored test candidates and threshold...")
    with open(config.OUTPUT_DIR / "threshold.txt") as f:
        threshold = float(f.read().strip())
    print(f"  threshold={threshold:.2f}")

    all_scored = pl.scan_parquet(str(config.TEST_SCORED_CHUNK_DIR / "*.parquet")).collect()
    all_candidates = pl.scan_parquet(str(config.TEST_CANDIDATE_CHUNK_DIR / "*.parquet")).collect()

    print("Reloading required S1 entity list...")
    all_s1_ids = []
    for country in sorted(io_utils.list_countries(config.TEST_S1)):
        all_s1_ids.extend(io_utils.scan_source_country(config.TEST_S1, country)["entity_id"].to_list())
    all_s1_series = pl.Series("entity_id", all_s1_ids)

    print("Re-writing candidate_pairs.tsv with quote_style='never' fix...")
    write_results(
        all_s1_series, all_candidates,
        config.OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids",
    )

    print(f"Re-writing matching_results.tsv (threshold={threshold:.2f})...")
    matches = all_scored.filter(pl.col("prob") >= threshold).select(["source1_entity_id", "candidate_id"])
    write_results(
        all_s1_series, matches,
        config.OUTPUT_DIR / "matching_results.tsv", "matched_entity_ids",
    )

    print("Done. Re-run utils/validate_submission.py to confirm.")


if __name__ == "__main__":
    main()
