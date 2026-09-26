"""
Residual embedding rescue benchmark — reuses the exact stratified US+India
sample from benchmark_union_sample.py (same seeds) to directly compare:

  1. cheap-only (8 non-embedding channels, no embedding at all)
  2-4. residual embedding rescue under 3 policies (zero_candidates,
       low_count, low_agreement) — embedding runs ONLY for the S1 subset
       each policy flags, not universally

against the full-embedding-for-everyone baseline already measured in
benchmark_union_sample.py (combined recall 97.36%, ~59h/country extrapolated
full-scale runtime, embedding channel alone = 551s of the 1872s India run).

Goal: find the best recall/runtime tradeoff — full embedding is provably too
slow for a 2-day deadline; this measures how much recall rescue policies
recover at a small fraction of the full embedding workload.
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import time

import numpy as np
import polars as pl

import config
from benchmark_union_sample import stratified_sample_country, recall_against, candidate_stats
from blocking import generate_candidates, generate_candidates_with_rescue, _ALL_CHANNELS

CHEAP_CHANNELS = [c for c in _ALL_CHANNELS if c != "embedding"]


def compute_cheap_candidates(s1_sample, s2_sample, s3_sample):
    """
    Computed ONCE per country and reused across cheap-only scoring and every
    rescue policy — the 8 cheap channels don't depend on the rescue policy at
    all, so running them once per policy (4x redundant work; the 3 TF-IDF
    channels alone measured 5-11 min each at this sample size) wastes most of
    the benchmark's runtime on identical repeated computation.
    """
    print("\n--- Computing cheap candidates (once, reused for cheap-only + all rescue policies) ---")
    t0 = time.time()
    candidates = generate_candidates(
        s1_sample, s2_sample, s3_sample, top_n=config.TOP_N_CANDIDATES,
        max_block_size=config.MAX_BLOCK_SIZE, max_total_pairs=config.MAX_TOTAL_PAIRS,
        channels=CHEAP_CHANNELS,
    )
    elapsed = time.time() - t0
    print(f"  cheap channels done in {elapsed:.1f}s — {candidates.height:,} candidate rows")
    return candidates, elapsed


def run_cheap_only(cheap_candidates, cheap_elapsed, true_by_s1, s1_ids):
    print("\n--- CHEAP-ONLY (8 non-embedding channels) ---")
    n_true, n_recovered, recall_pct = recall_against(cheap_candidates, true_by_s1)
    stats = candidate_stats(cheap_candidates, s1_ids)
    print(f"  recall={recall_pct:.2f}% ({n_recovered:,}/{n_true:,})  candidates={stats['n_candidates']:,}  "
          f"avg/S1={stats['avg_per_s1']:.1f}  p95={stats['p95_per_s1']:.0f}  p99={stats['p99_per_s1']:.0f}  "
          f"runtime={cheap_elapsed:.1f}s")
    return {
        "name": "cheap_only", "recall_pct": recall_pct, "n_true": n_true, "n_recovered": n_recovered,
        "stats": stats, "runtime_s": cheap_elapsed, "n_embedded": 0, "embed_fraction": 0.0,
    }


def run_rescue_policy(policy, cheap_candidates, s1_sample, s2_sample, s3_sample, true_by_s1, s1_ids, **policy_kwargs):
    print(f"\n--- RESCUE POLICY: {policy} {policy_kwargs} ---")
    t0 = time.time()
    candidates, rescue_stats = generate_candidates_with_rescue(
        s1_sample, s2_sample, s3_sample, top_n=config.TOP_N_CANDIDATES,
        max_block_size=config.MAX_BLOCK_SIZE, max_total_pairs=config.MAX_TOTAL_PAIRS,
        rescue_policy=policy, cheap_candidates=cheap_candidates, **policy_kwargs,
    )
    elapsed = time.time() - t0  # embedding-only time now, since cheap candidates are reused
    n_true, n_recovered, recall_pct = recall_against(candidates, true_by_s1)
    stats = candidate_stats(candidates, s1_ids)
    print(f"  rescued {rescue_stats['n_rescued']:,}/{rescue_stats['n_total_s1']:,} S1s "
          f"({100*rescue_stats['rescue_fraction']:.1f}%)")
    print(f"  recall={recall_pct:.2f}% ({n_recovered:,}/{n_true:,})  candidates={stats['n_candidates']:,}  "
          f"avg/S1={stats['avg_per_s1']:.1f}  p95={stats['p95_per_s1']:.0f}  p99={stats['p99_per_s1']:.0f}  "
          f"embedding_stage_runtime={elapsed:.1f}s")
    return {
        "name": f"rescue_{policy}", "recall_pct": recall_pct, "n_true": n_true, "n_recovered": n_recovered,
        "stats": stats, "runtime_s": elapsed,
        "n_embedded": rescue_stats["n_rescued"], "embed_fraction": rescue_stats["rescue_fraction"],
    }


def main():
    countries = ["US", "India"]
    all_results = {c: [] for c in countries}

    for country in countries:
        print(f"\n{'='*100}\n{country}\n{'='*100}")
        s1_sample, s2_sample, s3_sample, true_by_s1 = stratified_sample_country(country)
        s1_ids = s1_sample["entity_id"].to_list()

        cheap_candidates, cheap_elapsed = compute_cheap_candidates(s1_sample, s2_sample, s3_sample)

        cheap_result = run_cheap_only(cheap_candidates, cheap_elapsed, true_by_s1, s1_ids)
        all_results[country].append(cheap_result)

        # total_runtime_s for a rescue policy = shared cheap-stage cost (paid
        # once) + that policy's own embedding-only stage — NOT just the
        # embedding stage alone, since a real end-to-end run always pays for
        # both stages. Cheap-only's own runtime already includes cheap_elapsed.
        for policy, kwargs in [
            ("zero_candidates", {}),
            ("low_count", {"min_count_threshold": 5}),
            ("low_agreement", {"min_agreement_threshold": 2}),
        ]:
            r = run_rescue_policy(policy, cheap_candidates, s1_sample, s2_sample, s3_sample,
                                   true_by_s1, s1_ids, **kwargs)
            r["embedding_stage_runtime_s"] = r["runtime_s"]
            r["runtime_s"] = cheap_elapsed + r["runtime_s"]  # total end-to-end cost
            all_results[country].append(r)

    print(f"\n\n{'='*100}\nSUMMARY\n{'='*100}")
    print(f"{'policy':<20}{'country':<8}{'recall%':>9}{'embed%':>9}{'candidates':>12}{'avg/S1':>8}{'runtime_s':>11}")
    combined = {}
    for country, results in all_results.items():
        for r in results:
            print(f"{r['name']:<20}{country:<8}{r['recall_pct']:>8.2f}%{100*r['embed_fraction']:>8.1f}%"
                  f"{r['stats']['n_candidates']:>12,}{r['stats']['avg_per_s1']:>8.1f}{r['runtime_s']:>11.1f}")
            combined.setdefault(r["name"], {"n_true": 0, "n_recovered": 0, "runtime_s": 0.0, "embed_fractions": []})
            combined[r["name"]]["n_true"] += r["n_true"]
            combined[r["name"]]["n_recovered"] += r["n_recovered"]
            combined[r["name"]]["runtime_s"] += r["runtime_s"]
            combined[r["name"]]["embed_fractions"].append(r["embed_fraction"])

    print(f"\n{'='*100}\nCOMBINED (US + India)\n{'='*100}")
    print(f"{'policy':<20}{'recall%':>9}{'avg_embed%':>12}{'total_runtime_s':>18}")
    for name, agg in combined.items():
        recall = 100 * agg["n_recovered"] / agg["n_true"] if agg["n_true"] else 0.0
        avg_embed = 100 * np.mean(agg["embed_fractions"])
        print(f"{name:<20}{recall:>8.2f}%{avg_embed:>11.1f}%{agg['runtime_s']:>18.1f}")

    print(f"\n{'='*100}\nRECOMMENDATION\n{'='*100}")
    cheap_recall = 100 * combined["cheap_only"]["n_recovered"] / combined["cheap_only"]["n_true"]
    print(f"cheap-only combined recall: {cheap_recall:.2f}%")
    if cheap_recall >= 97.0:
        print("-> cheap-only already clears the 97% target. RECOMMENDATION: cheap-only "
              "(drop embedding entirely, maximum speed, no further complexity needed).")
    elif cheap_recall >= 96.0:
        best_rescue = max(
            (n for n in combined if n.startswith("rescue_")),
            key=lambda n: 100 * combined[n]["n_recovered"] / combined[n]["n_true"],
        )
        best_recall = 100 * combined[best_rescue]["n_recovered"] / combined[best_rescue]["n_true"]
        best_embed_pct = 100 * np.mean(combined[best_rescue]["embed_fractions"])
        print(f"-> cheap-only is 96-97%. Best rescue policy: {best_rescue} "
              f"(recall={best_recall:.2f}%, only {best_embed_pct:.1f}% of S1s embedded). "
              f"RECOMMENDATION: residual embedding with {best_rescue}.")
    else:
        print(f"-> cheap-only is below 96% ({cheap_recall:.2f}%). RECOMMENDATION: do not rely on embedding "
              f"rescue alone — investigate which cheap channel has the largest recall gap "
              f"(see benchmark_union_sample.py's per-channel breakdown) and improve it first.")


if __name__ == "__main__":
    main()
