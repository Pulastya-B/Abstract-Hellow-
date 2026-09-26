"""
Sprint-mode char q-gram incremental recall benchmark — time-boxed, per the
user's explicit constraint: implementation+smoke test <=30min, this
benchmark <=45min, total <=90min. If char_qgram doesn't show meaningful
incremental recall (target: >10 points) by the time this finishes, abandon
it and proceed with the existing 5-channel cheap blocker (55.21% measured
full-scale recall) rather than continuing to iterate.

Reuses the same stratified US+India sample as benchmark_union_sample.py.
Compares:
  - existing cheap blocker (5 channels: exact_name, suffix_normalized,
    rare_token, postal, numeric) — this is what's currently live in
    config.ACTIVE_CHANNELS and produced the 55.21% full-scale recall
  - existing cheap blocker + char_qgram (6 channels)

Reports old recall, new recall, incremental recall, candidate volume,
runtime, and a full-scale runtime extrapolation.
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import time

import config
from benchmark_union_sample import stratified_sample_country, recall_against, candidate_stats
from blocking import generate_candidates

EXISTING_CHANNELS = list(config.ACTIVE_CHANNELS)
WITH_QGRAM_CHANNELS = EXISTING_CHANNELS + ["char_qgram"]


def run_for_country(country: str):
    print(f"\n{'='*100}\n{country}\n{'='*100}")
    s1_sample, s2_sample, s3_sample, true_by_s1 = stratified_sample_country(country)
    s1_ids = s1_sample["entity_id"].to_list()

    print(f"\n--- EXISTING cheap blocker ({EXISTING_CHANNELS}) ---")
    t0 = time.time()
    old_candidates = generate_candidates(
        s1_sample, s2_sample, s3_sample, top_n=config.TOP_N_CANDIDATES,
        max_block_size=config.MAX_BLOCK_SIZE, max_total_pairs=config.MAX_TOTAL_PAIRS,
        channels=EXISTING_CHANNELS,
    )
    old_elapsed = time.time() - t0
    n_true, n_recovered_old, old_recall = recall_against(old_candidates, true_by_s1)
    old_stats = candidate_stats(old_candidates, s1_ids)
    print(f"  recall={old_recall:.2f}% ({n_recovered_old:,}/{n_true:,})  "
          f"candidates={old_stats['n_candidates']:,}  avg/S1={old_stats['avg_per_s1']:.1f}  "
          f"runtime={old_elapsed:.1f}s")

    print(f"\n--- EXISTING + char_qgram ({WITH_QGRAM_CHANNELS}) ---")
    t0 = time.time()
    new_candidates = generate_candidates(
        s1_sample, s2_sample, s3_sample, top_n=config.TOP_N_CANDIDATES,
        max_block_size=config.MAX_BLOCK_SIZE, max_total_pairs=config.MAX_TOTAL_PAIRS,
        channels=WITH_QGRAM_CHANNELS,
    )
    new_elapsed = time.time() - t0
    n_true2, n_recovered_new, new_recall = recall_against(new_candidates, true_by_s1)
    new_stats = candidate_stats(new_candidates, s1_ids)
    print(f"  recall={new_recall:.2f}% ({n_recovered_new:,}/{n_true2:,})  "
          f"candidates={new_stats['n_candidates']:,}  avg/S1={new_stats['avg_per_s1']:.1f}  "
          f"runtime={new_elapsed:.1f}s")

    incremental_pct = new_recall - old_recall
    print(f"\n  INCREMENTAL recall from char_qgram: {incremental_pct:+.2f} points")

    real_n_s1 = None
    try:
        import io_utils
        real_n_s1 = io_utils.scan_source_country(config.TRAIN_S1, country).height
    except Exception:
        pass

    return {
        "country": country, "n_true": n_true, "n_recovered_old": n_recovered_old,
        "n_recovered_new": n_recovered_new, "old_recall": old_recall, "new_recall": new_recall,
        "incremental_pct": incremental_pct, "old_stats": old_stats, "new_stats": new_stats,
        "old_elapsed": old_elapsed, "new_elapsed": new_elapsed, "sample_n_s1": len(s1_ids),
        "real_n_s1": real_n_s1,
    }


def main():
    countries = ["US", "India"]
    results = [run_for_country(c) for c in countries]

    print(f"\n\n{'='*100}\nSUMMARY\n{'='*100}")
    total_true = sum(r["n_true"] for r in results)
    total_old = sum(r["n_recovered_old"] for r in results)
    total_new = sum(r["n_recovered_new"] for r in results)
    old_combined = 100 * total_old / total_true if total_true else 0.0
    new_combined = 100 * total_new / total_true if total_true else 0.0

    for r in results:
        print(f"  {r['country']:<10} old_recall={r['old_recall']:.2f}%  new_recall={r['new_recall']:.2f}%  "
              f"incremental={r['incremental_pct']:+.2f}pts  "
              f"old_candidates={r['old_stats']['n_candidates']:,}  new_candidates={r['new_stats']['n_candidates']:,}  "
              f"qgram_extra_runtime={r['new_elapsed']-r['old_elapsed']:.1f}s")

    print(f"\nCombined old recall: {old_combined:.2f}%")
    print(f"Combined new recall: {new_combined:.2f}%")
    print(f"Combined incremental: {new_combined - old_combined:+.2f} points")

    print(f"\n--- Full-scale runtime extrapolation for char_qgram alone ---")
    for r in results:
        if r["real_n_s1"]:
            scale = r["real_n_s1"] / r["sample_n_s1"]
            qgram_time = r["new_elapsed"] - r["old_elapsed"]
            print(f"  {r['country']}: sample={r['sample_n_s1']:,} S1, real={r['real_n_s1']:,} S1 "
                  f"({scale:.1f}x) -> estimated qgram full-scale runtime ~{qgram_time*scale/60:.1f} min")

    print(f"\n{'='*100}\nDECISION\n{'='*100}")
    incremental = new_combined - old_combined
    if incremental >= 10.0:
        print(f"Incremental recall {incremental:+.2f} points >= 10 point target. "
              f"RECOMMENDATION: FREEZE char_qgram channel, proceed to LightGBM.")
    else:
        print(f"Incremental recall {incremental:+.2f} points < 10 point target. "
              f"RECOMMENDATION: char_qgram adds little recall for its cost — "
              f"abandon it, proceed with the existing 5-channel blocker as-is.")


if __name__ == "__main__":
    main()
