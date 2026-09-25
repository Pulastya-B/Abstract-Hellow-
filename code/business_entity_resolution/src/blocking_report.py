"""
Blocking benchmark: measures true-match candidate recall per channel and for
the full union, against the COMPLETE training ground truth (not the
candidate-survivor-only validation approximation train.py's threshold search
uses). This is the source of truth for whether blocking clears the >=97%
recall-ceiling bar in plan.md §3 before any matcher training is trusted.

Runs blocking once per channel-subset (each individual channel, then the
full union) so per-channel and incremental recall can be measured, then
reports the full stats block requested: per-channel/union recall, candidate
volume percentiles, reduction ratio, and recall broken out by country,
singleton-vs-non-singleton, and source pair (S1-S2 vs S1-S3).

Usage:
    python blocking_report.py [--no-embedding] [--countries US,India]
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import argparse
import gc

import numpy as np
import polars as pl
from tqdm import tqdm

import config
import io_utils
from io_utils import read_ground_truth, explode_ground_truth
from normalize import normalize_df
from blocking import generate_candidates, _ALL_CHANNELS


def _load_all_countries(path, countries_filter=None):
    countries = sorted(io_utils.list_countries(path))
    if countries_filter:
        countries = [c for c in countries if c in countries_filter]
    return countries


def _candidates_for_channels(s1, s2, s3, channels, use_embedding):
    return generate_candidates(
        s1, s2, s3, top_n=config.TOP_N_CANDIDATES,
        max_block_size=config.MAX_BLOCK_SIZE, max_total_pairs=config.MAX_TOTAL_PAIRS,
        use_embedding=use_embedding, channels=channels,
    )


def _recall_stats(candidates: pl.DataFrame, gt_pairs: set, gt_by_s1: dict, s1_country: dict, singleton_ids: set):
    """
    candidates: [source1_entity_id, candidate_id, candidate_source, ...]
    Returns overall recall plus breakdowns by country / source / singleton status.
    """
    cand_pairs = set(zip(candidates["source1_entity_id"].to_list(), candidates["candidate_id"].to_list()))

    n_gt = len(gt_pairs)
    n_recovered = len(gt_pairs & cand_pairs)
    overall_recall = 100 * n_recovered / n_gt if n_gt else 0.0

    # by country
    by_country = {}
    country_gt = {}
    for s1id, cid in gt_pairs:
        c = s1_country.get(s1id, "UNKNOWN")
        country_gt.setdefault(c, set()).add((s1id, cid))
    for c, pairs in country_gt.items():
        recovered = len(pairs & cand_pairs)
        by_country[c] = {
            "n_gt": len(pairs), "n_recovered": recovered,
            "recall_pct": 100 * recovered / len(pairs) if pairs else 0.0,
        }

    # by source (S2 vs S3, based on candidate_id prefix)
    by_source = {}
    for prefix in ("S2", "S3"):
        pairs = {(s1id, cid) for s1id, cid in gt_pairs if cid.startswith(prefix)}
        recovered = len(pairs & cand_pairs)
        by_source[prefix] = {
            "n_gt": len(pairs), "n_recovered": recovered,
            "recall_pct": 100 * recovered / len(pairs) if pairs else 0.0,
        }

    # singleton vs non-singleton: recall only meaningful for non-singletons
    # (a true singleton has no gt pairs by definition); report candidate
    # over-generation rate for singletons instead (false-candidate exposure).
    non_singleton_gt_s1 = {s1id for s1id, _ in gt_pairs}
    singleton_with_candidates = len(
        {s1id for s1id in singleton_ids if s1id in set(candidates["source1_entity_id"].to_list())}
    )

    return {
        "overall_recall_pct": overall_recall,
        "n_gt": n_gt,
        "n_recovered": n_recovered,
        "by_country": by_country,
        "by_source": by_source,
        "n_singletons": len(singleton_ids),
        "n_singletons_with_any_candidate": singleton_with_candidates,
    }


def _candidate_volume_stats(candidates: pl.DataFrame, n_s1_total: int):
    counts = candidates.group_by("source1_entity_id").len()["len"].to_numpy()
    if counts.size == 0:
        counts = np.array([0])
    # entities with zero candidates at all don't appear in `candidates` —
    # pad with zeros so percentiles reflect the true per-S1 distribution
    n_missing = max(n_s1_total - counts.size, 0)
    full_counts = np.concatenate([counts, np.zeros(n_missing, dtype=counts.dtype)])
    return {
        "total_candidate_pairs": int(candidates.height),
        "avg_candidates_per_s1": float(full_counts.mean()),
        "p50": float(np.percentile(full_counts, 50)),
        "p90": float(np.percentile(full_counts, 90)),
        "p95": float(np.percentile(full_counts, 95)),
        "p99": float(np.percentile(full_counts, 99)),
        "max": int(full_counts.max()),
    }


def run_benchmark(countries_filter=None, use_embedding=True):
    print("Loading ground truth...")
    gt = read_ground_truth(config.TRAIN_GT)
    pos_pairs_df = explode_ground_truth(gt)
    gt_pairs = set(zip(pos_pairs_df["source1_entity_id"].to_list(), pos_pairs_df["candidate_id"].to_list()))
    all_s1_ids = set(gt["source1_entity_id"].to_list())
    singleton_ids = set(
        gt.filter(pl.col("matched_entity_ids").str.len_chars() == 0)["source1_entity_id"].to_list()
    )
    print(f"  {len(gt_pairs):,} ground-truth positive pairs across {len(all_s1_ids):,} S1 entities "
          f"({len(singleton_ids):,} singletons)")
    del gt, pos_pairs_df
    gc.collect()

    countries = _load_all_countries(config.TRAIN_S1, countries_filter)
    print(f"Countries: {countries}")

    s1_country = {}
    all_channel_candidates = {ch: [] for ch in _ALL_CHANNELS}
    union_candidates = []
    n_s1_total = 0

    for country in tqdm(countries, desc="countries"):
        s1c = normalize_df(io_utils.scan_source_country(config.TRAIN_S1, country), label=f"s1[{country}]")
        s2c = normalize_df(io_utils.scan_source_country(config.TRAIN_S2, country), label=f"s2[{country}]")
        s3c = normalize_df(io_utils.scan_source_country(config.TRAIN_S3, country), label=f"s3[{country}]")
        n_s1_total += s1c.height
        for s1id in s1c["entity_id"].to_list():
            s1_country[s1id] = country

        print(f"\n--- country={country}: computing full union (all channels) ---")
        union_c = _candidates_for_channels(s1c, s2c, s3c, None, use_embedding)
        union_candidates.append(union_c)

        print(f"--- country={country}: computing per-channel candidates ---")
        for ch in _ALL_CHANNELS:
            if ch == "embedding" and not use_embedding:
                continue
            ch_c = _candidates_for_channels(s1c, s2c, s3c, [ch], use_embedding)
            all_channel_candidates[ch].append(ch_c)
            del ch_c
            gc.collect()

        del s1c, s2c, s3c, union_c
        gc.collect()

    union_all = pl.concat(union_candidates, how="vertical_relaxed") if union_candidates else pl.DataFrame()

    print("\n" + "=" * 100)
    print("BLOCKING BENCHMARK REPORT")
    print("=" * 100)

    # ── per-channel + union recall table ──
    rows = []
    already_recovered = set()
    channel_order = [c for c in _ALL_CHANNELS if all_channel_candidates[c]]
    for ch in channel_order:
        ch_df = pl.concat(all_channel_candidates[ch], how="vertical_relaxed")
        ch_pairs = set(zip(ch_df["source1_entity_id"].to_list(), ch_df["candidate_id"].to_list()))
        recovered = ch_pairs & gt_pairs
        incremental = recovered - already_recovered
        rows.append({
            "block": ch,
            "candidate_count": ch_df.height,
            "true_match_recall_pct": 100 * len(recovered) / len(gt_pairs) if gt_pairs else 0.0,
            "incremental_recall_pct": 100 * len(incremental) / len(gt_pairs) if gt_pairs else 0.0,
            "contribution_pairs": len(incremental),
        })
        already_recovered |= recovered
        del ch_df
        gc.collect()

    union_pairs = set(zip(union_all["source1_entity_id"].to_list(), union_all["candidate_id"].to_list())) \
        if union_all.height else set()
    union_recovered = union_pairs & gt_pairs
    rows.append({
        "block": "UNION (all channels)",
        "candidate_count": union_all.height,
        "true_match_recall_pct": 100 * len(union_recovered) / len(gt_pairs) if gt_pairs else 0.0,
        "incremental_recall_pct": None,
        "contribution_pairs": None,
    })

    print(f"\n{'block':<24}{'candidate_count':>18}{'true_match_recall_%':>22}"
          f"{'incremental_recall_%':>24}{'contribution_pairs':>20}")
    for r in rows:
        inc = f"{r['incremental_recall_pct']:.2f}" if r["incremental_recall_pct"] is not None else "-"
        contrib = f"{r['contribution_pairs']:,}" if r["contribution_pairs"] is not None else "-"
        print(f"{r['block']:<24}{r['candidate_count']:>18,}{r['true_match_recall_pct']:>21.2f}%"
              f"{inc:>24}{contrib:>20}")

    # ── volume stats on the union ──
    vol = _candidate_volume_stats(union_all, n_s1_total)
    reduction_ratio = 1 - (vol["total_candidate_pairs"] / (n_s1_total * max(1, 1))) if n_s1_total else 0.0

    print(f"\n--- Candidate volume (union) ---")
    print(f"  Total candidate pairs:            {vol['total_candidate_pairs']:,}")
    print(f"  Avg candidates per S1:            {vol['avg_candidates_per_s1']:.2f}")
    print(f"  p50 / p90 / p95 / p99 per S1:      {vol['p50']:.0f} / {vol['p90']:.0f} / {vol['p95']:.0f} / {vol['p99']:.0f}")
    print(f"  Max candidates per S1:             {vol['max']:,}")

    # ── recall breakdowns ──
    stats = _recall_stats(union_all, gt_pairs, {}, s1_country, singleton_ids)
    print(f"\n--- Recall ceiling (union, overall) ---")
    print(f"  {stats['n_recovered']:,} / {stats['n_gt']:,} true pairs recovered "
          f"({stats['overall_recall_pct']:.2f}%)  target >= {config.RECALL_TARGET_PCT}%")

    print(f"\n--- Recall by country ---")
    for c, s in sorted(stats["by_country"].items()):
        print(f"  {c:<10} {s['n_recovered']:>10,} / {s['n_gt']:<10,} ({s['recall_pct']:.2f}%)")

    print(f"\n--- Recall by source pair ---")
    for src, s in stats["by_source"].items():
        print(f"  S1-{src}: {s['n_recovered']:>10,} / {s['n_gt']:<10,} ({s['recall_pct']:.2f}%)")

    print(f"\n--- Singleton exposure ---")
    print(f"  True singletons in train:                    {stats['n_singletons']:,}")
    print(f"  Singletons with >=1 candidate surfaced:       {stats['n_singletons_with_any_candidate']:,} "
          f"({100 * stats['n_singletons_with_any_candidate'] / max(stats['n_singletons'], 1):.2f}%) "
          f"— lower is better here (means blocking correctly finds nothing for them, "
          f"leaving the decision to the matcher's threshold, not a forced false candidate)")

    verdict = "PASS" if stats["overall_recall_pct"] >= config.RECALL_TARGET_PCT else "FAIL"
    print(f"\n{'=' * 100}")
    print(f"VERDICT: {verdict} — overall recall {stats['overall_recall_pct']:.2f}% "
          f"({'meets' if verdict == 'PASS' else 'does NOT meet'} the >= {config.RECALL_TARGET_PCT}% target)")
    print("=" * 100)

    return {"rows": rows, "volume": vol, "recall_stats": stats, "verdict": verdict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embedding", action="store_true")
    parser.add_argument("--countries", type=str, default=None, help="comma-separated country filter")
    args = parser.parse_args()
    countries_filter = set(args.countries.split(",")) if args.countries else None
    run_benchmark(countries_filter=countries_filter, use_embedding=not args.no_embedding)
