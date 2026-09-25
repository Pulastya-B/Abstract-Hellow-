"""
Isolates TOP_K as the single variable, holding Config A (word TF-IDF,
max_df=1.0, max_features=50_000, sublinear_tf=True, dtype=float32) fixed —
per user instruction, following the benchmark_tfidf_configs.py vocabulary
sweep which showed max_df/max_features/min_df move recall by only ~5
points (62-67%) while avg_candidates/p99 were pinned at exactly top_k=20
for every config, meaning top_k itself may be the dominant constraint on
retrieval recall, not vocabulary composition.

Retrieves the SAME query at k=60 once (the largest k under test), then
truncates that single retrieval to 20/40/60 to measure top_k=20/40/60 —
avoids three separate retrievals with different random tie-breaking, and
gives the exact per-true-match RANK (a k=20-only retrieval can't tell you
whether a missed match was at rank 21 or rank 400 — this can).

Uses the identical S1 sample, corpus, and ground-truth pairs as
benchmark_tfidf_configs.py's config A run (same seeds), so results are
directly comparable.
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import time
import tracemalloc

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from benchmark_tfidf_configs import load_sample

CONFIG_A_KWARGS = dict(
    analyzer="word", ngram_range=(1, 1), min_df=1, max_df=1.0, max_features=50_000,
    sublinear_tf=True, dtype=np.float32,
)
TOP_K_VALUES = [20, 40, 60]
MAX_K = max(TOP_K_VALUES)
RANK_BUCKETS = [1, 5, 10, 20, 40, 60]


def ranked_topk(query_matrix, corpus_matrix, k):
    """
    Like blocking._sparse_topk_pairs, but returns candidates SORTED by
    descending similarity within each row (rank 0 = best match) instead of
    an unordered top-k — needed to answer "was the true match at rank 3 or
    rank 19" rather than just "was it in the top 20 at all."

    Returns a list of length n_query; each element is a list of
    (corpus_col_idx, similarity) tuples sorted best-first, length <= k.
    """
    n_query = query_matrix.shape[0]
    query_norms = np.sqrt(np.asarray(query_matrix.multiply(query_matrix).sum(axis=1))).ravel()
    query_norms[query_norms == 0] = 1.0
    corpus_norms = np.sqrt(np.asarray(corpus_matrix.multiply(corpus_matrix).sum(axis=1))).ravel()
    corpus_norms[corpus_norms == 0] = 1.0
    corpus_T = corpus_matrix.T.tocsr()

    chunk_size = 2000
    results = [None] * n_query
    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        sims = query_matrix[start:end].dot(corpus_T).tocsr()
        for local_row in range(sims.shape[0]):
            row_start, row_end = sims.indptr[local_row], sims.indptr[local_row + 1]
            if row_start == row_end:
                results[start + local_row] = []
                continue
            cols = sims.indices[row_start:row_end]
            vals = sims.data[row_start:row_end] / (query_norms[start + local_row] * corpus_norms[cols])
            k_eff = min(k, cols.shape[0])
            if k_eff < cols.shape[0]:
                top_local = np.argpartition(-vals, k_eff - 1)[:k_eff]
                cols, vals = cols[top_local], vals[top_local]
            order = np.argsort(-vals)
            results[start + local_row] = list(zip(cols[order].tolist(), vals[order].tolist()))
    return results


def main():
    s1_sample, s2_sample, s3_sample, true_by_s1 = load_sample()

    s1_texts = s1_sample["normalized_name"].to_list()
    s1_ids = s1_sample["entity_id"].to_list()

    others = pl.concat([s2_sample, s3_sample], how="vertical_relaxed")
    corpus_texts = others["normalized_name"].to_list()
    corpus_ids = others["entity_id"].to_list()

    print(f"\nFitting Config A vectorizer (fixed, unchanged): {CONFIG_A_KWARGS}")
    tracemalloc.start()
    t0 = time.time()
    vec = TfidfVectorizer(**CONFIG_A_KWARGS)
    corpus_matrix = vec.fit_transform(corpus_texts).tocsr()
    s1_matrix = vec.transform(s1_texts).tocsr()
    fit_time = time.time() - t0
    print(f"  fit+transform: {fit_time:.2f}s, vocab={len(vec.vocabulary_):,}")

    print(f"\nRetrieving top-{MAX_K} ranked candidates for every S1 (single retrieval, "
          f"truncated for each top_k value under test)...")
    t0 = time.time()
    ranked = ranked_topk(s1_matrix, corpus_matrix, MAX_K)
    retrieval_time = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  retrieval (k={MAX_K}) time: {retrieval_time:.2f}s, peak memory: {peak/1e6:.1f} MB")

    # Rank-distribution of true matches: for every true (S1, match) pair,
    # find the match's rank in its S1's ranked candidate list (None if
    # absent from the top-MAX_K entirely).
    corpus_id_to_col = {cid: i for i, cid in enumerate(corpus_ids)}
    rank_of_true_match = []  # 0-indexed rank, or None if not in top-MAX_K
    for s1id, ranked_row in zip(s1_ids, ranked):
        true_ids = true_by_s1.get(s1id, frozenset())
        if not true_ids:
            continue
        col_to_rank = {col: rank for rank, (col, _sim) in enumerate(ranked_row)}
        for true_id in true_ids:
            true_col = corpus_id_to_col.get(true_id)
            if true_col is None:
                continue  # true match wasn't even in the sampled corpus (shouldn't happen, load_sample guarantees it)
            rank_of_true_match.append(col_to_rank.get(true_col, None))

    n_true_total = len(rank_of_true_match)
    print(f"\n{'='*90}\nRANK DISTRIBUTION of {n_true_total:,} true matches "
          f"(within top-{MAX_K} retrieval, Config A)\n{'='*90}")
    prev_bound = 0
    cum_found = 0
    for bound in RANK_BUCKETS:
        in_bucket = sum(1 for r in rank_of_true_match if r is not None and prev_bound <= r < bound)
        cum_found += in_bucket
        print(f"  rank [{prev_bound:>3}, {bound:>3}): {in_bucket:>6,} true matches "
              f"({100*in_bucket/n_true_total:.2f}%)  cumulative: {100*cum_found/n_true_total:.2f}%")
        prev_bound = bound
    not_retrieved = sum(1 for r in rank_of_true_match if r is None)
    print(f"  rank >= {MAX_K} / not retrieved: {not_retrieved:,} ({100*not_retrieved/n_true_total:.2f}%)")

    print(f"\n{'='*90}\nPER-TOP_K COMPARISON (same retrieval, truncated)\n{'='*90}")
    print(f"{'top_k':>6}{'recall%':>10}{'incr_vs_20':>12}{'n_candidates':>14}{'avg/S1':>9}"
          f"{'p50':>6}{'p95':>6}{'p99':>6}{'max':>6}")

    baseline_recall = None
    for k in TOP_K_VALUES:
        n_recovered = sum(1 for r in rank_of_true_match if r is not None and r < k)
        recall_pct = 100 * n_recovered / n_true_total if n_true_total else 0.0
        if baseline_recall is None:
            baseline_recall = recall_pct
        incr = recall_pct - baseline_recall

        counts = np.array([min(len(row), k) for row in ranked])
        n_candidates = int(counts.sum())

        print(f"{k:>6}{recall_pct:>9.2f}%{incr:>11.2f}%{n_candidates:>14,}{counts.mean():>9.1f}"
              f"{np.percentile(counts,50):>6.0f}{np.percentile(counts,95):>6.0f}"
              f"{np.percentile(counts,99):>6.0f}{counts.max():>6.0f}")

    # storage size estimate: candidate rows x (~2 string IDs, ~20 bytes each avg)
    print(f"\nEstimated candidate storage (rows x ~40 bytes for 2 string IDs):")
    for k in TOP_K_VALUES:
        n_candidates = int(sum(min(len(row), k) for row in ranked))
        print(f"  top_k={k}: {n_candidates:,} rows -> ~{n_candidates * 40 / 1e6:.1f} MB")

    print(f"\nShared costs (same for all top_k values, since retrieval already computed at k={MAX_K}):")
    print(f"  TF-IDF fit+transform time: {fit_time:.2f}s")
    print(f"  Retrieval time (k={MAX_K}, reused via truncation): {retrieval_time:.2f}s")
    print(f"  Peak RAM during retrieval: {peak/1e6:.1f} MB")

    print(f"\n{'='*90}\nDECISION\n{'='*90}")
    recalls = {}
    for k in TOP_K_VALUES:
        n_recovered = sum(1 for r in rank_of_true_match if r is not None and r < k)
        recalls[k] = 100 * n_recovered / n_true_total if n_true_total else 0.0
    print(f"  recall@20={recalls[20]:.2f}%  recall@40={recalls[40]:.2f}%  recall@60={recalls[60]:.2f}%")
    gain_40 = recalls[40] - recalls[20]
    gain_60 = recalls[60] - recalls[40]
    print(f"  gain 20->40: {gain_40:+.2f} pts   gain 40->60: {gain_60:+.2f} pts")
    if gain_40 < 1.0:
        print("  RECOMMENDATION: recall barely improves beyond top_k=20 — STOP increasing top_k, keep 20.")
    elif gain_60 < 1.0:
        print("  RECOMMENDATION: top_k=40 gives a real gain over 20, but 60 adds little further — use top_k=40.")
    else:
        print("  RECOMMENDATION: recall keeps improving through top_k=60 — consider testing even higher k, "
              "weighed against the candidate-volume growth shown above.")


if __name__ == "__main__":
    main()
