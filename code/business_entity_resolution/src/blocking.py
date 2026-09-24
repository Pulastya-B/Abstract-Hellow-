"""
v3 blocking: country-partitioned, multi-key union (first token / last token /
sorted-token signature), vectorized (no Python-level map_elements) and with a
join-size safety cap on BOTH sides of every key, not just one — the previous
version could still blow up (and did: OOM-crashed a 12GB Colab runtime) if a
common token had thousands of hits on the S1 side even when the S2/S3 side
was capped, since the join cost is the PRODUCT of both sides.

Each candidate's `num_blocks_agreeing` is tracked as provenance — a candidate
independently found by several keys is much stronger evidence than one found
by a single fuzzy rule (plan.md §3's design). Processes one country at a time
to keep each join's working set bounded, since country partitions here range
from ~250K (France, test-only) to ~1.3M (US) S1 entities.

Still simplified vs. the full plan.md key set (no postal/phonetic/MinHash/
embedding ANN yet) — additive layers on top of this.
"""

import polars as pl
from tqdm import tqdm


def _add_block_keys(df: pl.DataFrame, name_col: str) -> pl.DataFrame:
    tokens = pl.col(name_col).str.split(" ")
    return df.with_columns([
        tokens.list.first().fill_null("").alias("key_first"),
        tokens.list.last().fill_null("").alias("key_last"),
        tokens.list.sort().list.join(" ").fill_null("").alias("key_sorted"),
    ])


_KEY_COLS = [("first_token", "key_first"), ("last_token", "key_last"), ("sorted_tokens", "key_sorted")]


def _block_one_key(
    s1_part: pl.DataFrame, ob_part: pl.DataFrame, key_col: str, key_name: str,
    max_block_size: int, max_pair_product: int,
) -> pl.DataFrame:
    empty_result = pl.DataFrame(
        schema={
            "source1_entity_id": pl.Utf8, "candidate_id": pl.Utf8,
            "candidate_source": pl.Utf8, "exact_match": pl.Boolean, "block_method": pl.Utf8,
        }
    )

    s1k = s1_part.filter(pl.col(key_col) != "").select(["source1_entity_id", key_col, "s1_name"])
    obk = ob_part.filter(pl.col(key_col) != "").select(["candidate_id", "candidate_source", key_col, "cand_name"])

    if s1k.height == 0 or obk.height == 0:
        return empty_result

    s1_counts = s1k.group_by(key_col).len().rename({"len": "n_s1"})
    ob_counts = obk.group_by(key_col).len().rename({"len": "n_ob"})
    sizes = s1_counts.join(ob_counts, on=key_col, how="inner")

    # Symmetric safety cap: both individual side sizes AND their product must
    # stay bounded, since join cost scales with the product, not either side alone.
    safe = sizes.filter(
        (pl.col("n_s1") <= max_block_size)
        & (pl.col("n_ob") <= max_block_size)
        & (pl.col("n_s1").cast(pl.Int64) * pl.col("n_ob").cast(pl.Int64) <= max_pair_product)
    ).select(key_col)

    s1k = s1k.join(safe, on=key_col, how="inner")
    obk = obk.join(safe, on=key_col, how="inner")

    joined = s1k.join(obk, on=key_col, how="inner")
    joined = joined.with_columns((pl.col("s1_name") == pl.col("cand_name")).alias("exact_match"))
    return joined.select(["source1_entity_id", "candidate_id", "candidate_source", "exact_match"]).with_columns(
        pl.lit(key_name).alias("block_method")
    )


def generate_candidates(
    s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame,
    top_n: int, max_block_size: int = 5000, max_pair_product: int = 2_000_000,
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

    s1b = _add_block_keys(
        s1.select([
            pl.col("entity_id").alias("source1_entity_id"),
            "country",
            pl.col("normalized_name").alias("s1_name"),
        ]),
        "s1_name",
    )
    ob = _add_block_keys(
        others.select([
            pl.col("entity_id").alias("candidate_id"),
            "country",
            pl.col("normalized_name").alias("cand_name"),
            pl.col("src").alias("candidate_source"),
        ]),
        "cand_name",
    )

    results = []
    countries = s1b["country"].unique().to_list()
    for country in tqdm(countries, desc="blocking by country"):
        s1c = s1b.filter(pl.col("country") == country)
        obc = ob.filter(pl.col("country") == country)
        if s1c.height == 0 or obc.height == 0:
            continue
        tqdm.write(f"  country={country}: {s1c.height:,} S1 x {obc.height:,} candidates")
        for key_name, key_col in tqdm(_KEY_COLS, desc=f"  keys[{country}]", leave=False):
            hit = _block_one_key(s1c, obc, key_col, key_name, max_block_size, max_pair_product)
            if hit.height:
                results.append(hit)

    unioned = pl.concat(results, how="vertical_relaxed")

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
