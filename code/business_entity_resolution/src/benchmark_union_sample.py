"""
Representative multi-country UNION blocking benchmark — deadline-constrained
(~2 days), so this runs the REAL blocking.generate_candidates() pipeline
(not a reimplementation) on a stratified 50-100K S1 sample spanning US +
India, instead of the full 883K+1.7M-entity country-by-country run.

Stratification targets (per user requirement):
  - both US and India represented
  - mix of true singleton / non-singleton S1 entities
  - mix of S2-only / S3-only / both-source true matches
  - natural mix of name length / script / address availability (achieved by
    random sampling WITHIN each stratum, not cherry-picking)

Channels: all 9 from blocking.py, using channels=None (the real union) plus
per-channel breakdown via repeated calls with channels=[single] — reuses the
production _ALL_CHANNELS list and generate_candidates() so this measures the
actual pipeline behavior, not an approximation of it. Char n-gram TF-IDF is
NOT a separate channel in blocking.py (already excluded per the
benchmark_tfidf_configs.py finding: no recall benefit over word-level TF-IDF,
substantial extra compute) — nothing to add here.

Does NOT touch the embedding channel's country-partition-size cutoff
(config.EMBEDDING_MAX_PARTITION_ROWS) — the sample is small enough that
embedding will run if the channel list includes it.
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import argparse
import time

import numpy as np
import polars as pl
from tqdm import tqdm

import config
import io_utils
from normalize import normalize_df
from blocking import generate_candidates, _ALL_CHANNELS

SAMPLE_N_S1_PER_COUNTRY = 35_000   # ~70K total across US+India, within the requested 50-100K range
CORPUS_RANDOM_PER_COUNTRY = 400_000  # random S2/S3 rows added per country beyond the true-match set
RANK_BUCKETS = [1, 5, 10, 20, 40, 60]


def stratified_sample_country(country: str):
    print(f"\n{'='*90}\nLoading + sampling {country}\n{'='*90}")
    gt = io_utils.read_ground_truth(config.TRAIN_GT)
    s1_full_raw = io_utils.scan_source_country(config.TRAIN_S1, country)
    country_s1_ids = set(s1_full_raw["entity_id"].to_list())
    gt_country = gt.filter(pl.col("source1_entity_id").is_in(list(country_s1_ids)))

    has_match = gt_country.filter(pl.col("matched_entity_ids").str.len_chars() > 0)
    singletons = gt_country.filter(pl.col("matched_entity_ids").str.len_chars() == 0)

    # keep the real singleton ratio rather than forcing 50/50 — closer to
    # the true test-time distribution (~5.6% singleton rate per plan.md)
    target_n = min(SAMPLE_N_S1_PER_COUNTRY, gt_country.height)
    true_singleton_rate = singletons.height / max(gt_country.height, 1)
    n_singleton = int(target_n * true_singleton_rate)
    n_match = target_n - n_singleton

    sample_match = has_match.sample(n=min(n_match, has_match.height), seed=config.RANDOM_SEED)
    sample_singleton = singletons.sample(n=min(n_singleton, singletons.height), seed=config.RANDOM_SEED + 1)
    sample_gt = pl.concat([sample_match, sample_singleton]).unique(subset=["source1_entity_id"])
    sample_s1_ids = set(sample_gt["source1_entity_id"].to_list())

    true_by_s1 = {}
    for s1id, matches in zip(sample_gt["source1_entity_id"].to_list(), sample_gt["matched_entity_ids"].to_list()):
        true_by_s1[s1id] = frozenset(matches.split(",")) if matches else frozenset()

    n_s2_only = sum(1 for v in true_by_s1.values() if v and all(i.startswith("S2-") for i in v))
    n_s3_only = sum(1 for v in true_by_s1.values() if v and all(i.startswith("S3-") for i in v))
    n_both = sum(1 for v in true_by_s1.values() if v and not all(i.startswith("S2-") for i in v)
                 and not all(i.startswith("S3-") for i in v))
    print(f"  sampled {len(sample_s1_ids):,} S1 ({sum(1 for v in true_by_s1.values() if not v):,} singletons, "
          f"{n_s2_only:,} S2-only, {n_s3_only:,} S3-only, {n_both:,} both)")

    s1_sample = normalize_df(
        s1_full_raw.filter(pl.col("entity_id").is_in(list(sample_s1_ids))), label=f"s1[{country}]",
    )

    all_true_ids = set()
    for ids in true_by_s1.values():
        all_true_ids |= ids

    s2_full = io_utils.scan_source_country(config.TRAIN_S2, country)
    s3_full = io_utils.scan_source_country(config.TRAIN_S3, country)

    s2_true = s2_full.filter(pl.col("entity_id").is_in(list(all_true_ids)))
    s3_true = s3_full.filter(pl.col("entity_id").is_in(list(all_true_ids)))
    s2_random = s2_full.filter(~pl.col("entity_id").is_in(list(all_true_ids))).sample(
        n=min(CORPUS_RANDOM_PER_COUNTRY, s2_full.height), seed=config.RANDOM_SEED,
    )
    s3_random = s3_full.filter(~pl.col("entity_id").is_in(list(all_true_ids))).sample(
        n=min(CORPUS_RANDOM_PER_COUNTRY, s3_full.height), seed=config.RANDOM_SEED,
    )

    s2_sample = normalize_df(pl.concat([s2_true, s2_random]).unique(subset=["entity_id"]), label=f"s2[{country}]")
    s3_sample = normalize_df(pl.concat([s3_true, s3_random]).unique(subset=["entity_id"]), label=f"s3[{country}]")

    print(f"  corpus: {s2_sample.height:,} S2 + {s3_sample.height:,} S3")
    return s1_sample, s2_sample, s3_sample, true_by_s1


def candidate_stats(candidates: pl.DataFrame, s1_ids: list):
    counts = candidates.group_by("source1_entity_id").len()["len"].to_numpy() if candidates.height else np.array([])
    n_missing = max(len(s1_ids) - counts.size, 0)
    full_counts = np.concatenate([counts, np.zeros(n_missing)]) if counts.size or n_missing else np.array([0])
    return {
        "n_candidates": candidates.height,
        "avg_per_s1": float(full_counts.mean()),
        "p95_per_s1": float(np.percentile(full_counts, 95)),
        "p99_per_s1": float(np.percentile(full_counts, 99)),
    }


def recall_against(candidates: pl.DataFrame, true_by_s1: dict):
    cand_pairs = set(zip(candidates["source1_entity_id"].to_list(), candidates["candidate_id"].to_list())) \
        if candidates.height else set()
    n_true = n_recovered = 0
    for s1id, true_ids in true_by_s1.items():
        if not true_ids:
            continue
        n_true += len(true_ids)
        n_recovered += sum(1 for t in true_ids if (s1id, t) in cand_pairs)
    return n_true, n_recovered, (100 * n_recovered / n_true if n_true else 0.0)


def run_for_country(country: str):
    s1_sample, s2_sample, s3_sample, true_by_s1 = stratified_sample_country(country)
    s1_ids = s1_sample["entity_id"].to_list()

    print(f"\n--- {country}: per-channel breakdown (cumulative incremental recall) ---")
    already_recovered = set()
    rows = []
    for ch in _ALL_CHANNELS:
        t0 = time.time()
        ch_candidates = generate_candidates(
            s1_sample, s2_sample, s3_sample, top_n=config.TOP_N_CANDIDATES,
            max_block_size=config.MAX_BLOCK_SIZE, max_total_pairs=config.MAX_TOTAL_PAIRS,
            channels=[ch],
        )
        elapsed = time.time() - t0
        n_true, n_recovered, recall_pct = recall_against(ch_candidates, true_by_s1)
        cand_pairs = set(zip(ch_candidates["source1_entity_id"].to_list(), ch_candidates["candidate_id"].to_list())) \
            if ch_candidates.height else set()
        recovered_pairs = {
            (s1id, t) for s1id, true_ids in true_by_s1.items() for t in true_ids if (s1id, t) in cand_pairs
        }
        incremental = recovered_pairs - already_recovered
        already_recovered |= recovered_pairs
        stats = candidate_stats(ch_candidates, s1_ids)

        rows.append({
            "channel": ch, "candidates": stats["n_candidates"], "recall_pct": recall_pct,
            "incremental_pairs": len(incremental),
            "incremental_pct": 100 * len(incremental) / n_true if n_true else 0.0,
            "avg_per_s1": stats["avg_per_s1"], "p95_per_s1": stats["p95_per_s1"], "p99_per_s1": stats["p99_per_s1"],
            "runtime_s": elapsed,
        })
        print(f"  [{ch:<18}] candidates={stats['n_candidates']:>9,}  recall={recall_pct:>6.2f}%  "
              f"incremental={len(incremental):>6,} ({rows[-1]['incremental_pct']:>5.2f}%)  "
              f"avg/S1={stats['avg_per_s1']:>5.1f}  p95={stats['p95_per_s1']:>4.0f}  "
              f"p99={stats['p99_per_s1']:>4.0f}  time={elapsed:>6.1f}s")

    print(f"\n--- {country}: FULL UNION (all channels together) ---")
    t0 = time.time()
    union_candidates = generate_candidates(
        s1_sample, s2_sample, s3_sample, top_n=config.TOP_N_CANDIDATES,
        max_block_size=config.MAX_BLOCK_SIZE, max_total_pairs=config.MAX_TOTAL_PAIRS,
        channels=None,
    )
    union_elapsed = time.time() - t0
    n_true, n_recovered, union_recall = recall_against(union_candidates, true_by_s1)
    union_stats = candidate_stats(union_candidates, s1_ids)

    n_s1 = len(s1_ids)
    n_corpus = s2_sample.height + s3_sample.height
    reduction_ratio = 1 - (union_stats["n_candidates"] / (n_s1 * n_corpus)) if n_s1 * n_corpus else 0.0

    print(f"  UNION recall: {n_recovered:,}/{n_true:,} = {union_recall:.2f}%")
    print(f"  candidates: {union_stats['n_candidates']:,}  avg/S1={union_stats['avg_per_s1']:.1f}  "
          f"p95={union_stats['p95_per_s1']:.0f}  p99={union_stats['p99_per_s1']:.0f}")
    print(f"  reduction ratio vs naive {n_s1:,}x{n_corpus:,}: {reduction_ratio:.6f}")
    print(f"  union runtime (this sample): {union_elapsed:.1f}s")

    return {
        "country": country, "rows": rows, "union_recall": union_recall,
        "union_stats": union_stats, "union_elapsed": union_elapsed,
        "n_true": n_true, "n_recovered": n_recovered,
        "sample_n_s1": n_s1, "sample_n_corpus": n_corpus,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", type=str, default="US,India")
    args = parser.parse_args()
    countries = args.countries.split(",")

    results = [run_for_country(c) for c in countries]

    print(f"\n\n{'='*100}\nOVERALL SUMMARY (all sampled countries combined)\n{'='*100}")
    total_true = sum(r["n_true"] for r in results)
    total_recovered = sum(r["n_recovered"] for r in results)
    overall_recall = 100 * total_recovered / total_true if total_true else 0.0
    print(f"Combined UNION recall: {total_recovered:,}/{total_true:,} = {overall_recall:.2f}%  "
          f"(target >= {config.RECALL_TARGET_PCT}%)")

    for r in results:
        print(f"  {r['country']:<10} recall={r['union_recall']:.2f}%  "
              f"candidates={r['union_stats']['n_candidates']:,}  "
              f"avg/S1={r['union_stats']['avg_per_s1']:.1f}  "
              f"sample={r['sample_n_s1']:,} S1 x {r['sample_n_corpus']:,} corpus  "
              f"runtime={r['union_elapsed']:.1f}s")

    print(f"\n--- Rough full-scale runtime extrapolation (scaled by real S1 x real corpus size vs. "
          f"this sample's S1 x corpus size; likely pessimistic since larger real corpora may have "
          f"MORE shared structure to exploit, not less) ---")
    for r in results:
        country = r["country"]
        real_n_s1 = io_utils.scan_source_country(config.TRAIN_S1, country).height
        real_n_s2 = io_utils.scan_source_country(config.TRAIN_S2, country).height
        real_n_s3 = io_utils.scan_source_country(config.TRAIN_S3, country).height
        real_n_corpus = real_n_s2 + real_n_s3
        sample_pairs = r["sample_n_s1"] * r["sample_n_corpus"]
        real_pairs = real_n_s1 * real_n_corpus
        scale = real_pairs / sample_pairs if sample_pairs else float("nan")
        print(f"  {country}: real={real_n_s1:,} S1 x {real_n_corpus:,} corpus "
              f"({scale:.1f}x this sample's pair volume) "
              f"-> estimated full union runtime ~{r['union_elapsed'] * scale / 60:.1f} min")

    print(f"\n{'='*100}\nDECISION\n{'='*100}")
    if overall_recall >= config.RECALL_TARGET_PCT:
        print(f"CASE A: UNION recall {overall_recall:.2f}% >= {config.RECALL_TARGET_PCT}% target.")
        print("  -> FREEZE blocking. Proceed to matcher training (do NOT redesign further).")
    elif overall_recall >= 90.0:
        print(f"CASE B: UNION recall {overall_recall:.2f}% is 90-97%.")
        print("  -> Identify the channel with the largest remaining incremental-recall gap")
        print("     (see per-channel breakdown above) and implement ONE targeted fix, then re-benchmark.")
    else:
        print(f"CASE C: UNION recall {overall_recall:.2f}% is below 90%.")
        print("  -> Do not add many speculative channels. Identify the dominant failure mode from the")
        print("     per-channel breakdown and implement the single highest-value retrieval mechanism.")


if __name__ == "__main__":
    main()
