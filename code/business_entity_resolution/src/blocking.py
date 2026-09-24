"""
v2 blocking: country-partitioned, multi-key union (first token / last token /
sorted-token signature). Each candidate's `num_blocks_agreeing` is tracked as
provenance — a candidate independently found by several keys is much stronger
evidence than one found by a single fuzzy rule (plan.md §3's design).

Still simplified vs. the full plan.md key set (no postal/phonetic/MinHash/
embedding ANN yet) — additive layers on top of this.
"""

import polars as pl


def _first_token(name: str) -> str:
    parts = name.split()
    return parts[0] if parts else ""


def _last_token(name: str) -> str:
    parts = name.split()
    return parts[-1] if parts else ""


def _sorted_tokens(name: str) -> str:
    parts = name.split()
    return " ".join(sorted(parts)) if parts else ""


BLOCK_KEY_FUNCS = {
    "first_token": _first_token,
    "last_token": _last_token,
    "sorted_tokens": _sorted_tokens,
}


def _run_single_block(s1b: pl.DataFrame, ob: pl.DataFrame, key_fn, key_name: str, max_block_size: int) -> pl.DataFrame:
    s1k = s1b.with_columns(
        pl.col("s1_name").map_elements(key_fn, return_dtype=pl.Utf8).alias("block_key")
    ).filter(pl.col("block_key") != "")
    obk = ob.with_columns(
        pl.col("cand_name").map_elements(key_fn, return_dtype=pl.Utf8).alias("block_key")
    ).filter(pl.col("block_key") != "")

    # Safety cap: an overly common key (e.g. "the", "sri") can blow up the join.
    # Drop such keys from this blocking pass rather than let them explode.
    key_counts = obk.group_by(["country", "block_key"]).len()
    safe_keys = key_counts.filter(pl.col("len") <= max_block_size).select(["country", "block_key"])
    obk = obk.join(safe_keys, on=["country", "block_key"], how="inner")

    joined = s1k.join(obk, on=["country", "block_key"], how="inner")
    joined = joined.with_columns((pl.col("s1_name") == pl.col("cand_name")).alias("exact_match"))
    return joined.select([
        "source1_entity_id", "candidate_id", "candidate_source", "exact_match",
    ]).with_columns(pl.lit(key_name).alias("block_method"))


def generate_candidates(
    s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame,
    top_n: int, max_block_size: int = 20000,
) -> pl.DataFrame:
    """
    Returns [source1_entity_id, candidate_id, candidate_source,
    num_blocks_agreeing, block_exact_match] — up to top_n candidates per S1,
    ranked by exact-match first, then by how many independent blocking keys
    agreed on the candidate.
    """
    others = pl.concat(
        [
            s2.with_columns(pl.lit("S2").alias("src")),
            s3.with_columns(pl.lit("S3").alias("src")),
        ],
        how="vertical_relaxed",
    )

    s1b = s1.select([
        pl.col("entity_id").alias("source1_entity_id"),
        "country",
        pl.col("normalized_name").alias("s1_name"),
    ])
    ob = others.select([
        pl.col("entity_id").alias("candidate_id"),
        "country",
        pl.col("normalized_name").alias("cand_name"),
        pl.col("src").alias("candidate_source"),
    ])

    hits = [
        _run_single_block(s1b, ob, fn, name, max_block_size)
        for name, fn in BLOCK_KEY_FUNCS.items()
    ]
    unioned = pl.concat(hits, how="vertical_relaxed")

    agg = unioned.group_by(["source1_entity_id", "candidate_id"]).agg([
        pl.col("candidate_source").first(),
        pl.col("block_method").n_unique().alias("num_blocks_agreeing"),
        pl.col("exact_match").max().alias("block_exact_match"),
    ])

    agg = agg.sort(
        ["source1_entity_id", "block_exact_match", "num_blocks_agreeing"],
        descending=[False, True, True],
    )
    agg = agg.group_by("source1_entity_id", maintain_order=True).head(top_n)

    return agg.select([
        "source1_entity_id", "candidate_id", "candidate_source",
        "num_blocks_agreeing", "block_exact_match",
    ])
