"""
Sampled TF-IDF configuration benchmark — vocabulary controls (max_df,
max_features, min_df, sublinear_tf) x word/char analyzers, measured against
REAL retrieval recall on a representative sample, not just vocabulary stats.

Root cause already fixed (normalize.py's Devanagari mark-stripping bug) and
confirmed via diagnose_tfidf.py: tokenization is now correct, but vocabulary
composition is still bad — max_features=50_000 keeps the highest-document-
frequency words (sklearn's default behavior), and the top 20 are dominated
by generic legal-suffix terms ("limited" 26.1% of all India S2 rows,
"private" 26.1%, "ltd" 15.2%, "pvt" 9.4%) that carry almost no discriminative
signal, while ~243K rarer/more-informative words get dropped entirely. This
benchmark finds vocabulary settings that fix that tradeoff, verified against
real candidate recall — not assumed from vocabulary stats alone.

Does NOT touch normalize.py's suffix-stripping design: core_name (used by
the suffix_normalized channel) already strips suffixes only via an explicit
dictionary match, not a blanket deletion from every representation — the
raw normalized_name (with suffixes intact) remains a separate, independent
representation, per the "don't globally delete legal terms" requirement.

Runs on a SAMPLE (not the full 5M+ dataset) so multiple configs can be
compared quickly. Reports vocabulary/density/timing stats AND real
retrieval recall against the training ground truth for the sampled S1
entities, for every requested (max_features, max_df, min_df, analyzer,
ngram_range) combination.
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import time
import tracemalloc

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

import config
import io_utils
from normalize import normalize_df
from blocking import _sparse_topk_pairs

SAMPLE_N_S1 = 15_000          # S1 entities sampled for the benchmark
SAMPLE_TOP_K = 20             # candidates retrieved per S1, matches config.TFIDF_TOP_K


def load_sample():
    print("Loading India ground truth + full S1/S2/S3 (normalized once, reused across configs)...")
    gt = io_utils.read_ground_truth(config.TRAIN_GT)
    s1_full = normalize_df(io_utils.scan_source_country(config.TRAIN_S1, "India"), label="s1")
    s2_full = normalize_df(io_utils.scan_source_country(config.TRAIN_S2, "India"), label="s2")
    s3_full = normalize_df(io_utils.scan_source_country(config.TRAIN_S3, "India"), label="s3")

    india_s1_ids = set(s1_full["entity_id"].to_list())
    gt_india = gt.filter(pl.col("source1_entity_id").is_in(list(india_s1_ids)))

    # Stratified sample: half from entities WITH matches (recall is only
    # measurable where there's something to recover), half random (includes
    # singletons, matches the real distribution's ~5.6% singleton rate).
    has_match = gt_india.filter(pl.col("matched_entity_ids").str.len_chars() > 0)
    n_with_match = min(SAMPLE_N_S1 // 2, has_match.height)
    sample_with_match = has_match.sample(n=n_with_match, seed=config.RANDOM_SEED)

    n_random = SAMPLE_N_S1 - n_with_match
    sample_random = gt_india.sample(n=min(n_random, gt_india.height), seed=config.RANDOM_SEED + 1)

    sample_gt = pl.concat([sample_with_match, sample_random]).unique(subset=["source1_entity_id"])
    sample_s1_ids = set(sample_gt["source1_entity_id"].to_list())

    true_by_s1 = {}
    for s1id, matches in zip(sample_gt["source1_entity_id"].to_list(), sample_gt["matched_entity_ids"].to_list()):
        true_by_s1[s1id] = frozenset(matches.split(",")) if matches else frozenset()

    s1_sample = s1_full.filter(pl.col("entity_id").is_in(list(sample_s1_ids)))

    # Corpus must include every true match for the sampled S1s (otherwise
    # recall is unmeasurable — a candidate that was never in the corpus
    # can't be "recalled"), plus a large random slice of S2/S3 to keep the
    # corpus size/composition realistic for density/timing measurements.
    all_true_ids = set()
    for ids in true_by_s1.values():
        all_true_ids |= ids

    s2_true = s2_full.filter(pl.col("entity_id").is_in(list(all_true_ids)))
    s3_true = s3_full.filter(pl.col("entity_id").is_in(list(all_true_ids)))

    n_random_corpus = 300_000
    s2_random = s2_full.filter(~pl.col("entity_id").is_in(list(all_true_ids))).sample(
        n=min(n_random_corpus, s2_full.height), seed=config.RANDOM_SEED,
    )
    s3_random = s3_full.filter(~pl.col("entity_id").is_in(list(all_true_ids))).sample(
        n=min(n_random_corpus, s3_full.height), seed=config.RANDOM_SEED,
    )

    s2_sample = pl.concat([s2_true, s2_random]).unique(subset=["entity_id"])
    s3_sample = pl.concat([s3_true, s3_random]).unique(subset=["entity_id"])

    print(f"Sample: {s1_sample.height:,} S1 entities, corpus = {s2_sample.height:,} S2 + {s3_sample.height:,} S3")
    print(f"  ({sum(1 for v in true_by_s1.values() if v)} have >=1 true match, "
          f"{sum(1 for v in true_by_s1.values() if not v)} are singletons)")

    return s1_sample, s2_sample, s3_sample, true_by_s1


def measure_recall(row_pos, col_idx, s1_ids_arr, corpus_ids_arr, true_by_s1):
    predicted_by_s1 = {}
    for r, c in zip(row_pos, col_idx):
        predicted_by_s1.setdefault(s1_ids_arr[r], set()).add(corpus_ids_arr[c])

    n_true_total = n_recovered_total = 0
    for s1id, true_ids in true_by_s1.items():
        if not true_ids:
            continue
        recovered = len(true_ids & predicted_by_s1.get(s1id, set()))
        n_true_total += len(true_ids)
        n_recovered_total += recovered

    recall_pct = 100 * n_recovered_total / n_true_total if n_true_total else 0.0

    counts = np.array([len(v) for v in predicted_by_s1.values()] +
                       [0] * (len(s1_ids_arr) - len(predicted_by_s1)))
    return {
        "recall_pct": recall_pct, "n_true_total": n_true_total, "n_recovered_total": n_recovered_total,
        "avg_candidates": float(counts.mean()) if counts.size else 0.0,
        "p95_candidates": float(np.percentile(counts, 95)) if counts.size else 0.0,
        "p99_candidates": float(np.percentile(counts, 99)) if counts.size else 0.0,
    }


def run_config(name, s1_texts, corpus_texts, s1_ids_arr, corpus_ids_arr, true_by_s1, vectorizer_kwargs, top_k):
    print(f"\n{'='*90}\nCONFIG: {name}\n  {vectorizer_kwargs}\n{'='*90}")

    tracemalloc.start()
    t0 = time.time()
    vec = TfidfVectorizer(**vectorizer_kwargs)
    try:
        corpus_matrix = vec.fit_transform(corpus_texts).tocsr()
        s1_matrix = vec.transform(s1_texts).tocsr()
    except ValueError as e:
        print(f"  SKIPPED (vectorizer error: {e})")
        tracemalloc.stop()
        return None
    fit_time = time.time() - t0

    vocab_size = len(vec.vocabulary_)
    doc_freqs = np.asarray((corpus_matrix > 0).sum(axis=0)).ravel()
    doc_freq_ratio = doc_freqs / corpus_matrix.shape[0]
    generic_mask = doc_freq_ratio > 0.02  # arbitrary "generic term" cutoff for reporting
    pct_generic = 100 * generic_mask.sum() / vocab_size if vocab_size else 0.0

    inv_vocab = {v: k for k, v in vec.vocabulary_.items()}
    top_idx = np.argsort(-doc_freqs)[:15]
    top_terms = [(inv_vocab[i], f"{100*doc_freq_ratio[i]:.1f}%") for i in top_idx]

    t0 = time.time()
    row_pos, col_idx = _sparse_topk_pairs(s1_matrix, corpus_matrix, top_k, chunk_size=2000)
    query_time = time.time() - t0

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    sample_sims = s1_matrix[:min(500, s1_matrix.shape[0])].dot(corpus_matrix.T).tocsr()
    density = sample_sims.nnz / (sample_sims.shape[0] * corpus_matrix.shape[0])

    recall = measure_recall(row_pos, col_idx, s1_ids_arr, corpus_ids_arr, true_by_s1)

    print(f"  vocabulary size:        {vocab_size:,}")
    print(f"  top 15 terms (term: doc-freq%): {top_terms}")
    print(f"  %% vocab 'generic' (df>2%): {pct_generic:.1f}%")
    print(f"  matrix density (500-row sample): {density:.6f}")
    print(f"  peak memory:             {peak / 1e6:.1f} MB")
    print(f"  fit+transform time:      {fit_time:.2f}s")
    print(f"  query (topk) time:       {query_time:.2f}s")
    print(f"  RETRIEVAL RECALL:        {recall['recall_pct']:.2f}% "
          f"({recall['n_recovered_total']}/{recall['n_true_total']} true pairs)")
    print(f"  avg candidates/S1:       {recall['avg_candidates']:.1f}")
    print(f"  p95 / p99 candidates/S1: {recall['p95_candidates']:.0f} / {recall['p99_candidates']:.0f}")

    return {
        "name": name, "vocab_size": vocab_size, "pct_generic": pct_generic, "density": density,
        "peak_memory_mb": peak / 1e6, "fit_time": fit_time, "query_time": query_time, **recall,
    }


def main():
    s1_sample, s2_sample, s3_sample, true_by_s1 = load_sample()

    s1_texts = s1_sample["normalized_name"].to_list()
    s1_ids_arr = np.asarray(s1_sample["entity_id"].to_list(), dtype=object)

    others = pl.concat([s2_sample, s3_sample], how="vertical_relaxed")
    corpus_texts = others["normalized_name"].to_list()
    corpus_ids_arr = np.asarray(others["entity_id"].to_list(), dtype=object)

    configs = [
        ("A: word(1,1) max_feat=50k max_df=1.0", dict(
            analyzer="word", ngram_range=(1, 1), min_df=1, max_df=1.0, max_features=50_000,
            sublinear_tf=True, dtype=np.float32)),
        ("B: word(1,1) max_feat=50k max_df=0.02", dict(
            analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.02, max_features=50_000,
            sublinear_tf=True, dtype=np.float32)),
        ("C: word(1,1) max_feat=100k max_df=0.02", dict(
            analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.02, max_features=100_000,
            sublinear_tf=True, dtype=np.float32)),
        ("D: word(1,1) max_feat=200k max_df=0.02", dict(
            analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.02, max_features=200_000,
            sublinear_tf=True, dtype=np.float32)),
        ("E: word(1,1) max_feat=300k max_df=0.02", dict(
            analyzer="word", ngram_range=(1, 1), min_df=1, max_df=0.02, max_features=300_000,
            sublinear_tf=True, dtype=np.float32)),
        ("F: word(1,1) max_feat=200k max_df=0.02 min_df=2", dict(
            analyzer="word", ngram_range=(1, 1), min_df=2, max_df=0.02, max_features=200_000,
            sublinear_tf=True, dtype=np.float32)),
        ("G: word(1,1) max_feat=200k max_df=0.02 min_df=5", dict(
            analyzer="word", ngram_range=(1, 1), min_df=5, max_df=0.02, max_features=200_000,
            sublinear_tf=True, dtype=np.float32)),
        ("H: word(1,2) max_feat=200k max_df=0.02", dict(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.02, max_features=200_000,
            sublinear_tf=True, dtype=np.float32)),
        ("I: char(3,5) max_feat=200k max_df=0.02", dict(
            analyzer="char", ngram_range=(3, 5), min_df=2, max_df=0.02, max_features=200_000,
            sublinear_tf=True, dtype=np.float32)),
        ("J: char(3,6) max_feat=200k max_df=0.02", dict(
            analyzer="char", ngram_range=(3, 6), min_df=2, max_df=0.02, max_features=200_000,
            sublinear_tf=True, dtype=np.float32)),
    ]

    results = []
    for name, kwargs in configs:
        r = run_config(name, s1_texts, corpus_texts, s1_ids_arr, corpus_ids_arr, true_by_s1, kwargs, SAMPLE_TOP_K)
        if r:
            results.append(r)

    print(f"\n\n{'='*100}\nSUMMARY (sorted by retrieval recall, descending)\n{'='*100}")
    print(f"{'config':<45}{'vocab':>9}{'%generic':>10}{'density':>10}{'recall%':>9}{'avg_cand':>10}{'p99':>7}")
    for r in sorted(results, key=lambda x: -x["recall_pct"]):
        print(f"{r['name']:<45}{r['vocab_size']:>9,}{r['pct_generic']:>9.1f}%{r['density']:>10.5f}"
              f"{r['recall_pct']:>8.2f}%{r['avg_candidates']:>10.1f}{r['p99_candidates']:>7.0f}")

    best_word = max((r for r in results if "word" in r["name"].lower() or r["name"][2] == "w"), key=lambda x: x["recall_pct"], default=None)
    best_char = max((r for r in results if "char" in r["name"].lower()), key=lambda x: x["recall_pct"], default=None)
    print(f"\nBest word config: {best_word['name'] if best_word else 'N/A'}")
    print(f"Best char config: {best_char['name'] if best_char else 'N/A'}")


if __name__ == "__main__":
    main()
