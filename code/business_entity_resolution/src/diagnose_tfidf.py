"""
One-off diagnostic: measures real India-partition TF-IDF vocabulary size and
similarity-matrix density, to find out why name_tfidf estimated 24+ hours on
real data vs. minutes on synthetic benchmark data during development.

Run from src/: python diagnose_tfidf.py
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import time

from sklearn.feature_extraction.text import TfidfVectorizer

import config
import io_utils
from normalize import normalize_df

print("Loading + normalizing India S1/S2 (small sample of S2 for density check)...")
s1 = normalize_df(io_utils.scan_source_country(config.TRAIN_S1, "India"), label="s1")
s2 = normalize_df(io_utils.scan_source_country(config.TRAIN_S2, "India"), label="s2")

s1_names = s1["normalized_name"].to_list()
s2_names = s2["normalized_name"].to_list()

print(f"\nS1 rows: {len(s1_names):,}, S2 rows: {len(s2_names):,}")
print(f"Sample S1 names: {s1_names[:5]}")
print(f"Sample S2 names: {s2_names[:5]}")

vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1, max_features=50_000)
t0 = time.time()
s2_matrix = vec.fit_transform(s2_names)
print(f"\nVectorizer fit on S2 took {time.time()-t0:.1f}s")
print(f"Vocabulary size: {len(vec.vocabulary_):,}")
print(f"S2 matrix shape: {s2_matrix.shape}, nnz: {s2_matrix.nnz:,}, "
      f"avg nonzero per row: {s2_matrix.nnz / s2_matrix.shape[0]:.2f}")

# Check whether max_features is truncating vocab (and if so, what got kept
# vs dropped) — TfidfVectorizer's max_features keeps the HIGHEST document-
# frequency terms, which is the opposite of what a selective retrieval index
# wants: the most common words get kept, the most discriminative (rare) ones
# get dropped.
vec_uncapped = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1)
vec_uncapped.fit(s2_names)
true_vocab_size = len(vec_uncapped.vocabulary_)
print(f"\nTRUE (uncapped) vocabulary size: {true_vocab_size:,} "
      f"(capped run kept only {len(vec.vocabulary_):,} = "
      f"{100*len(vec.vocabulary_)/true_vocab_size:.1f}% of it)")

# document frequency of the top 20 most common words that made the cut
import numpy as np
doc_freqs = np.asarray((s2_matrix > 0).sum(axis=0)).ravel()
top_idx = np.argsort(-doc_freqs)[:20]
inv_vocab = {v: k for k, v in vec.vocabulary_.items()}
print("\nTop 20 highest-document-frequency words IN the capped vocabulary:")
for i in top_idx:
    print(f"  {inv_vocab[i]!r}: appears in {doc_freqs[i]:,} / {s2_matrix.shape[0]:,} rows "
          f"({100*doc_freqs[i]/s2_matrix.shape[0]:.1f}%)")

s1_matrix = vec.transform(s1_names)
print(f"S1 matrix shape: {s1_matrix.shape}, nnz: {s1_matrix.nnz:,}, "
      f"avg nonzero per row: {s1_matrix.nnz / s1_matrix.shape[0]:.2f}")

# Small-scale density test: first 2000 S1 rows against first 20000 S2 rows
n_s1_sample = min(2000, s1_matrix.shape[0])
n_s2_sample = min(20000, s2_matrix.shape[0])
print(f"\nTiming a {n_s1_sample}x{n_s2_sample} sparse matmul sample...")
t0 = time.time()
sims = s1_matrix[:n_s1_sample].dot(s2_matrix[:n_s2_sample].T).tocsr()
elapsed = time.time() - t0
density = sims.nnz / (n_s1_sample * n_s2_sample)
print(f"  took {elapsed:.2f}s, nnz={sims.nnz:,}, density={density:.6f}")

full_pairs = s1_matrix.shape[0] * s2_matrix.shape[0]
sample_pairs = n_s1_sample * n_s2_sample
scale_factor = full_pairs / sample_pairs
print(f"\nFull India S1xS2 pair count: {full_pairs:,} ({scale_factor:.1f}x this sample)")
print(f"Naive linear-scaling estimate for full matmul: {elapsed * scale_factor:.1f}s "
      f"({elapsed * scale_factor / 60:.1f} min)")
print("\nIf density is high (e.g. >0.01) OR the naive estimate is still huge,")
print("the vocabulary is not selective enough for this approach at this scale.")
