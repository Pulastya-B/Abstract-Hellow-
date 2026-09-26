import polars as pl


def read_source(path) -> pl.DataFrame:
    """Read a source1/2/3 TSV: entity_id, business_name, business_address, country."""
    return pl.read_csv(path, separator="\t")


def list_countries(path) -> list:
    """Cheap: scan just the country column instead of loading the whole file."""
    return (
        pl.scan_csv(path, separator="\t")
        .select("country")
        .unique()
        .collect()["country"]
        .to_list()
    )


def scan_source_country(path, country: str) -> pl.DataFrame:
    """
    Read only one country's rows of a source1/2/3 TSV, via lazy scan + filter
    pushdown — keeps peak memory bounded to one country's slice instead of
    materializing the whole (multi-million-row) table first.
    """
    return (
        pl.scan_csv(path, separator="\t", low_memory=True)
        .filter(pl.col("country") == country)
        .collect()
    )


def read_ground_truth(path) -> pl.DataFrame:
    """Read train_ground_truth.tsv: source1_entity_id, matched_entity_ids (comma list)."""
    return pl.read_csv(path, separator="\t")


def explode_ground_truth(gt: pl.DataFrame) -> pl.DataFrame:
    """One row per (source1_entity_id, candidate_id) true positive pair. Singletons drop out."""
    return (
        gt.filter(pl.col("matched_entity_ids").str.len_chars() > 0)
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .rename({"matched_entity_ids": "candidate_id"})
        .select(["source1_entity_id", "candidate_id"])
    )


def write_results(all_s1_ids: pl.Series, pairs: pl.DataFrame, path, list_col: str):
    """
    Write the validator's required format: one row per S1 entity (every required S1
    id present, empty string if none), comma-separated candidate list, deduped,
    S2-/S3- prefix only.

    pairs must have columns [source1_entity_id, candidate_id].
    """
    clean = (
        pairs.filter(
            pl.col("candidate_id").str.starts_with("S2-")
            | pl.col("candidate_id").str.starts_with("S3-")
        )
        .unique(subset=["source1_entity_id", "candidate_id"])
    )

    grouped = clean.group_by("source1_entity_id").agg(
        pl.col("candidate_id").sort().str.concat(",").alias(list_col)
    )

    full = pl.DataFrame({"source1_entity_id": all_s1_ids}).join(
        grouped, on="source1_entity_id", how="left"
    )
    full = full.with_columns(pl.col(list_col).fill_null(""))

    path = str(path)
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # quote_style="never": Polars' default quoting wraps an empty string in
    # literal "" characters (two double-quotes), not a truly empty field.
    # The validator's plain split(",") then reads that quoted-empty as a
    # single garbage "ID" (literally '""'), which fails the S2-/S3- prefix
    # check on every true singleton row. Confirmed directly against
    # validate_submission.py's output on the first full-scale inference run.
    full.write_csv(path, separator="\t", quote_style="never")
