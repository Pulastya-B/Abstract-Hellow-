"""
v2 feature set: rapidfuzz string similarity + block-agreement provenance +
per-S1 meta features (rank, candidate count) + a cross-source consistency
signal (does this S1 also have strong evidence in the OTHER source?).

The consistency feature here is a simplified proxy for the full idea (S2
candidate vs. S3 candidate mutual similarity) — it uses "best same-S1 score
in the other source" instead, which is much cheaper to compute and still
captures the core signal: ~85% of non-singleton S1 entities have matches in
both S2 and S3 (see plan.md), so an S1 with strong S3 evidence but a weak S2
candidate is itself informative about that S2 candidate's plausibility.

No embedding cosine similarity yet (plan.md §5) — later addition.
"""

import polars as pl
from rapidfuzz import fuzz

FEATURE_COLS = [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "addr_ratio",
    "addr_token_set_ratio",
    "exact_name_match",
    "is_source2",
    "addr_missing",
    "num_blocks_agreeing",
    "block_exact_match",
    "candidate_rank_within_s1",
    "num_candidates_for_s1",
    "other_source_best_name_ratio",
]


def _ratio(a, b):
    return fuzz.ratio(a, b) / 100.0


def _token_sort(a, b):
    return fuzz.token_sort_ratio(a, b) / 100.0


def _token_set(a, b):
    return fuzz.token_set_ratio(a, b) / 100.0


def add_features(pairs: pl.DataFrame, s1: pl.DataFrame, others: pl.DataFrame) -> pl.DataFrame:
    """
    pairs: [source1_entity_id, candidate_id, candidate_source,
            num_blocks_agreeing, block_exact_match] (from blocking.generate_candidates)
    s1, others: normalized frames with entity_id, normalized_name,
                normalized_address, address_missing, country
    """
    s1n = s1.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("normalized_name").alias("s1_name"),
        pl.col("normalized_address").alias("s1_addr"),
    ])
    on = others.select([
        pl.col("entity_id").alias("candidate_id"),
        pl.col("normalized_name").alias("cand_name"),
        pl.col("normalized_address").alias("cand_addr"),
        pl.col("address_missing").alias("cand_addr_missing"),
    ])

    df = pairs.join(s1n, on="source1_entity_id").join(on, on="candidate_id")

    s1_names = df["s1_name"].to_list()
    cand_names = df["cand_name"].to_list()
    s1_addrs = df["s1_addr"].to_list()
    cand_addrs = df["cand_addr"].to_list()

    name_ratio = [_ratio(a, b) for a, b in zip(s1_names, cand_names)]
    name_tsort = [_token_sort(a, b) for a, b in zip(s1_names, cand_names)]
    name_tset = [_token_set(a, b) for a, b in zip(s1_names, cand_names)]
    addr_ratio = [_ratio(a, b) for a, b in zip(s1_addrs, cand_addrs)]
    addr_tset = [_token_set(a, b) for a, b in zip(s1_addrs, cand_addrs)]
    exact_name = [1.0 if a == b and a != "" else 0.0 for a, b in zip(s1_names, cand_names)]

    df = df.with_columns([
        pl.Series("name_ratio", name_ratio),
        pl.Series("name_token_sort_ratio", name_tsort),
        pl.Series("name_token_set_ratio", name_tset),
        pl.Series("addr_ratio", addr_ratio),
        pl.Series("addr_token_set_ratio", addr_tset),
        pl.Series("exact_name_match", exact_name),
    ])

    df = df.with_columns([
        (pl.col("candidate_source") == "S2").cast(pl.Int8).alias("is_source2"),
        pl.col("cand_addr_missing").cast(pl.Int8).alias("addr_missing"),
        pl.col("block_exact_match").cast(pl.Int8),
        pl.col("num_blocks_agreeing").cast(pl.Int64),
    ])

    # per-S1 meta: rank (best name match first) and candidate pool size
    df = df.with_columns([
        pl.col("name_ratio")
        .rank(method="ordinal", descending=True)
        .over("source1_entity_id")
        .alias("candidate_rank_within_s1"),
        pl.len().over("source1_entity_id").alias("num_candidates_for_s1"),
    ])

    # cross-source consistency proxy: this S1's best name_ratio in the OTHER source
    per_source_best = (
        df.group_by(["source1_entity_id", "candidate_source"])
        .agg(pl.col("name_ratio").max().alias("best_ratio"))
    )
    s2_best = per_source_best.filter(pl.col("candidate_source") == "S2").select([
        "source1_entity_id", pl.col("best_ratio").alias("s2_best"),
    ])
    s3_best = per_source_best.filter(pl.col("candidate_source") == "S3").select([
        "source1_entity_id", pl.col("best_ratio").alias("s3_best"),
    ])
    df = df.join(s2_best, on="source1_entity_id", how="left").join(s3_best, on="source1_entity_id", how="left")
    df = df.with_columns([
        pl.col("s2_best").fill_null(0.0),
        pl.col("s3_best").fill_null(0.0),
    ])
    df = df.with_columns(
        pl.when(pl.col("candidate_source") == "S2")
        .then(pl.col("s3_best"))
        .otherwise(pl.col("s2_best"))
        .alias("other_source_best_name_ratio")
    )

    return df
