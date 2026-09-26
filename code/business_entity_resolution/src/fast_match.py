"""
Retrieval-first matcher.

Every S2/S3 record belongs to at most one S1 entity (0 violations across all
7,638,365 train ground-truth pairs). So instead of blocking pairs from the S1
side, each S2/S3 record is used as a query against a per-country word BM25
index over S1 "name + address". Its top-10 S1 hits are each scored by a
LightGBM pair model, and the record is assigned to its highest-probability
candidate if that probability clears a threshold tuned for F0.5; otherwise it
is assigned to nobody. S1 entities that no record claims come out empty,
which is exactly the right answer for singletons.

Measured on train (3,000 true records per country, full S1 index): the true
S1 is ranked #1 by BM25 for 91.4% (India) / 95.4% (US) of records and is in
the top 10 for 95.7% / 98.2% (word TF-IDF: 89.0/94.9 and 94.2/97.8). Scoring
all 10 lets the model recover owners that BM25 ranked 2nd-10th. Name-token
blocking, by contrast, capped recall at 30%. Address carries the match when
names are in another script (Telugu, Kannada, Gujarati, Devanagari) or
replaced outright, so no transliteration is needed for retrieval.

Retrieval, feature computation and prediction all run inside the worker
processes, chunk by chunk, so the whole pipeline scales with --n-jobs.

Usage, v2 (cached; retrieval runs once, everything after it takes minutes):
  python fast_match.py --data-dir D --out-dir O --stage extract --split train --countries India
  python fast_match.py --data-dir D --out-dir O --stage extract --split train --countries US --sample 300000
  python fast_match.py --data-dir D --out-dir O --stage extract --split test
  python fast_match.py --data-dir D --out-dir O --stage fit       # exact leaderboard metric + rule tuning
  python fast_match.py --data-dir D --out-dir O --stage predict   # writes both submission TSVs

Usage, v1 (single run, no cache; produced the 0.928 submission):
  python fast_match.py --data-dir /path/to/dataset --out-dir /path/to/output --n-jobs 24
  python fast_match.py ... --max-queries 20000      # smoke test: caps queries per (country, source)
  python fast_match.py ... --stage train            # only fit + save the model
  python fast_match.py ... --stage test             # reuse a saved model, write submission TSVs
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.utils import default_process
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

TOKEN_PATTERN = r"(?u)\b\w+\b"
MAX_DF = 0.05           # drop tokens in >5% of a country's S1 docs ("road", state codes, "pvt")
BM25_K1 = 1.2
BM25_B = 0.75
TOP_K = 10              # S1 candidates retrieved and scored per record
CANDIDATE_OUT_K = 5     # candidates per record written to candidate_pairs.tsv (plus the assigned one);
                        # all 10 would be ~100M ids at full test scale, heavy for the validator/scorer
TASK_CHUNK = 2000       # records per worker task
SUB_CHUNK = 256         # records per sparse matmul inside a task (bounds per-product memory)
SEED = 42

PAIR_FEATURES = [
    "rank", "n_cands", "score", "score1", "gap_to_top", "ratio_to_top", "gap_to_next", "rec_margin12",
    "name_tset", "name_ratio", "name_partial", "addr_tset", "addr_ratio",
    "num_jacc", "num_first_eq", "name_len_ratio",
    "name_tset_gap_to_best", "addr_tset_gap_to_best",
    "q_addr_empty", "q_nonlatin", "s_nonlatin", "is_s2",
]
COL = {f: i for i, f in enumerate(PAIR_FEATURES)}

_DIGITS = re.compile(r"\d+")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── I/O ──────────────────────────────────────────────────────────────────────

def read_source(path, country=None):
    lf = pl.scan_csv(path, separator="\t", infer_schema_length=0, quote_char=None)
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    return lf.select(["entity_id", "business_name", "business_address", "country"]).collect().with_columns(
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
    )


def list_countries(path):
    return sorted(
        pl.scan_csv(path, separator="\t", infer_schema_length=0, quote_char=None)
        .select("country").unique().drop_nulls().collect()["country"].to_list()
    )


def doc_text(df):
    return (df["business_name"] + " " + df["business_address"]).to_list()


def write_lists(all_s1_ids, pairs, path, col):
    """One row per required S1 id, comma-joined deduped S2/S3 ids (empty = no match)."""
    grouped = (
        pairs.filter(pl.col("cand_id").str.starts_with("S2-") | pl.col("cand_id").str.starts_with("S3-"))
        .group_by("source1_entity_id")
        .agg(pl.col("cand_id").unique().sort())
        .with_columns(pl.col("cand_id").list.join(",").alias(col))
        .select(["source1_entity_id", col])
    )
    full = (
        pl.DataFrame({"source1_entity_id": all_s1_ids})
        .join(grouped, on="source1_entity_id", how="left")
        .with_columns(pl.col(col).fill_null(""))
    )
    # quote_style="never": polars otherwise writes empty strings as "" (quoted), which the
    # validator reads as a literal ID and rejects. IDs never contain tabs/newlines/quotes.
    full.write_csv(path, separator="\t", quote_style="never")
    log(f"wrote {path}  ({full.height:,} rows, {full.filter(pl.col(col) != '').height:,} non-empty)")


# ── Worker: retrieval + pair features (+ prediction) for one chunk ──────────

def _nonlatin(s):
    return float(any(ch.isalpha() and ord(ch) > 0x24F for ch in s))


def _digit_info(addr):
    d = _DIGITS.findall(addr)
    return (set(d), d[0].lstrip("0")) if d else (None, None)


_W = {}


def _init_worker(XT, s1_names, s1_addrs, model_path):
    # Everything about S1 that every candidate comparison needs is prepared
    # once per worker here, instead of once per (record, candidate) pair.
    _W["XT"] = XT
    _W["sn"] = [default_process(x) for x in s1_names]
    _W["sa"] = [default_process(x) for x in s1_addrs]
    _W["s_nonlatin"] = np.array([_nonlatin(x) for x in s1_names], dtype=np.float32)
    _W["s_digits"] = [_digit_info(x) for x in s1_addrs]
    _W["booster"] = lgb.Booster(model_file=model_path) if model_path else None


def _topk(Q):
    n = Q.shape[0]
    idx = np.full((n, TOP_K), -1, dtype=np.int64)
    sc = np.zeros((n, TOP_K), dtype=np.float32)
    for s in range(0, n, SUB_CHUNK):
        R = (Q[s:s + SUB_CHUNK] @ _W["XT"]).tocsr()
        for i in range(R.shape[0]):
            lo, hi = R.indptr[i], R.indptr[i + 1]
            if lo == hi:
                continue
            d, c = R.data[lo:hi], R.indices[lo:hi]
            if len(d) > TOP_K:
                sel = np.argpartition(-d, TOP_K)[:TOP_K]
                d, c = d[sel], c[sel]
            order = np.argsort(-d)
            k = len(order)
            idx[s + i, :k] = c[order]
            sc[s + i, :k] = d[order]
    return idx, sc


def _pair_features(q_names, q_addrs, q_is_s2, idx, sc):
    n = idx.shape[0]
    F = np.zeros((n, TOP_K, len(PAIR_FEATURES)), dtype=np.float32)
    valid = idx >= 0
    n_cands = valid.sum(axis=1).astype(np.float32)

    # score-shape features, vectorized over (record, rank)
    next_sc = np.concatenate([sc[:, 1:], np.zeros((n, 1), dtype=np.float32)], axis=1)
    next_valid = np.concatenate([valid[:, 1:], np.zeros((n, 1), dtype=bool)], axis=1)
    F[:, :, COL["rank"]] = np.arange(TOP_K, dtype=np.float32)[None, :]
    F[:, :, COL["n_cands"]] = n_cands[:, None]
    F[:, :, COL["score"]] = sc
    F[:, :, COL["score1"]] = sc[:, :1]
    F[:, :, COL["gap_to_top"]] = sc[:, :1] - sc
    F[:, :, COL["ratio_to_top"]] = np.where(sc[:, :1] > 0, sc / np.maximum(sc[:, :1], 1e-9), 0)
    F[:, :, COL["gap_to_next"]] = np.where(next_valid, sc - next_sc, sc)
    F[:, :, COL["rec_margin12"]] = (sc[:, 0] - sc[:, 1])[:, None]
    F[:, :, COL["is_s2"]] = q_is_s2[:, None]

    sn, sa, s_nonlatin, s_digits = _W["sn"], _W["sa"], _W["s_nonlatin"], _W["s_digits"]
    for i in range(n):
        a_name = default_process(q_names[i])
        a_addr = default_process(q_addrs[i])
        a_empty = not a_addr
        a_digits, a_first = _digit_info(q_addrs[i])
        F[i, :, COL["q_addr_empty"]] = float(a_empty)
        F[i, :, COL["q_nonlatin"]] = _nonlatin(q_names[i])
        for r in range(TOP_K):
            j = idx[i, r]
            if j < 0:
                break
            b_name, b_addr = sn[j], sa[j]
            row = F[i, r]
            row[COL["s_nonlatin"]] = s_nonlatin[j]
            row[COL["name_tset"]] = fuzz.token_set_ratio(a_name, b_name)
            row[COL["name_ratio"]] = fuzz.ratio(a_name, b_name)
            row[COL["name_partial"]] = fuzz.partial_ratio(a_name, b_name)
            if not a_empty and b_addr:
                row[COL["addr_tset"]] = fuzz.token_set_ratio(a_addr, b_addr)
                row[COL["addr_ratio"]] = fuzz.ratio(a_addr, b_addr)
            else:
                row[COL["addr_tset"]] = -1
                row[COL["addr_ratio"]] = -1
            b_digits, b_first = s_digits[j]
            if a_digits and b_digits:
                row[COL["num_jacc"]] = len(a_digits & b_digits) / len(a_digits | b_digits)
                row[COL["num_first_eq"]] = float(a_first == b_first)
            else:
                row[COL["num_jacc"]] = -1
                row[COL["num_first_eq"]] = -1
            la, lb = len(a_name), len(b_name)
            row[COL["name_len_ratio"]] = min(la, lb) / max(la, lb) if max(la, lb) else 0

    # how far each candidate is from the best candidate of the same record
    for f, gap in (("name_tset", "name_tset_gap_to_best"), ("addr_tset", "addr_tset_gap_to_best")):
        vals = np.where(valid, F[:, :, COL[f]], -np.inf)
        F[:, :, COL[gap]] = np.where(valid, vals.max(axis=1, keepdims=True) - F[:, :, COL[f]], 0)

    return F


def _process_chunk(task):
    Q, q_names, q_addrs, q_is_s2 = task
    idx, sc = _topk(Q)
    F = _pair_features(q_names, q_addrs, q_is_s2, idx, sc)
    booster = _W["booster"]
    if booster is None:
        return idx, sc, F
    n = idx.shape[0]
    probs = booster.predict(F.reshape(n * TOP_K, -1), num_threads=1).reshape(n, TOP_K)
    probs[idx < 0] = 0.0
    return idx, sc, probs.astype(np.float32)


def iter_chunks(Q, q, XT, s1, n_jobs, model_path, desc):
    """Yields (start_row, result) in row order; result is (idx, sc, F) or (idx, sc, probs)."""
    qn, qa = q["business_name"].to_list(), q["business_address"].to_list()
    qs2 = (q["src"] == "S2").to_numpy().astype(np.float32)
    starts = list(range(0, q.height, TASK_CHUNK))
    tasks = [(Q[s:s + TASK_CHUNK], qn[s:s + TASK_CHUNK], qa[s:s + TASK_CHUNK], qs2[s:s + TASK_CHUNK]) for s in starts]
    initargs = (XT, s1["business_name"].to_list(), s1["business_address"].to_list(), model_path)
    if n_jobs <= 1:
        _init_worker(*initargs)
        for s, t in zip(starts, tqdm(tasks, desc=desc, mininterval=10)):
            yield s, _process_chunk(t)
    else:
        # "spawn", not Linux's default fork: after the parent has trained LightGBM,
        # its OpenMP runtime is not fork-safe, and forked workers hang forever on
        # their first booster.predict (observed: stuck at 0% on the first test chunk).
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx,
                                 initializer=_init_worker, initargs=initargs) as ex:
            results = ex.map(_process_chunk, tasks)
            for s, r in zip(starts, tqdm(results, total=len(tasks), desc=desc, mininterval=10)):
                yield s, r


def run_chunks(Q, q, XT, s1, n_jobs, model_path, desc):
    """Returns (idx, sc, F) without a model, (idx, sc, probs) with one; rows align with q."""
    results = [r for _, r in iter_chunks(Q, q, XT, s1, n_jobs, model_path, desc)]
    return tuple(np.concatenate([r[k] for r in results]) for k in range(3))


def build_index(s1, max_df=None):
    """
    BM25 document weights over S1 "name + address", returned transposed for Q @ XT.
    Measured on the same 3,000 true records per country as the TF-IDF baseline:
    recall@1 India 89.0% -> 91.4%, US 94.9% -> 95.4%; recall@10 India 94.2% -> 95.7%,
    US 97.8% -> 98.2%; same speed.
    """
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, max_df=max_df or MAX_DF, dtype=np.float32)
    tf = vec.fit_transform(doc_text(s1)).tocsr()
    n = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    dl = np.asarray(tf.sum(axis=1)).ravel()
    norm = BM25_K1 * (1 - BM25_B + BM25_B * dl / max(float(dl.mean()), 1e-9))
    rows = np.repeat(np.arange(n), np.diff(tf.indptr))
    w = idf[tf.indices] * tf.data * (BM25_K1 + 1) / (tf.data + norm[rows])
    W = sp.csr_matrix((w.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)
    return vec, W.T.tocsr()


def encode_queries(vec, df):
    Q = vec.transform(doc_text(df)).tocsr()
    Q.data[:] = 1.0  # BM25 sums over the distinct query terms
    return Q


# ── Train: pair model over every (record, top-10 candidate) ──────────────────

def f05(tp, fp, n_pos):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_pos if n_pos else 0.0
    return (1.25 * p * r / (0.25 * p + r) if p + r else 0.0), p, r


def train_stage(data_dir, out_dir, n_jobs, sample_per_country):
    train_dir = os.path.join(data_dir, "train")
    gt = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t",
                     infer_schema_length=0, quote_char=None)
    owners = (
        gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .rename({"matched_entity_ids": "cand_id", "source1_entity_id": "owner"})
    )

    Fs, labels, valids, has_owners = [], [], [], []
    for country in list_countries(os.path.join(train_dir, "train_source1.tsv")):
        t0 = time.time()
        s1 = read_source(os.path.join(train_dir, "train_source1.tsv"), country)
        vec, XT = build_index(s1)
        s1_ids = np.array(s1["entity_id"].to_list(), dtype=object)
        q = pl.concat([
            read_source(os.path.join(train_dir, "train_source2.tsv"), country).with_columns(pl.lit("S2").alias("src")),
            read_source(os.path.join(train_dir, "train_source3.tsv"), country).with_columns(pl.lit("S3").alias("src")),
        ])
        q = q.sample(n=min(sample_per_country, q.height), seed=SEED)
        log(f"[train {country}] index {s1.height:,} S1, querying {q.height:,} sampled S2/S3 records")

        Q = encode_queries(vec, q)
        idx, sc, F = run_chunks(Q, q, XT, s1, n_jobs, None, desc=f"train {country}")

        owner_map = dict(owners.join(q.select(pl.col("entity_id").alias("cand_id")), on="cand_id", how="semi")
                         .select(["cand_id", "owner"]).iter_rows())
        rec_owner = np.array([owner_map.get(r) for r in q["entity_id"].to_list()], dtype=object)
        valid = idx >= 0
        cand_ids = s1_ids[np.where(valid, idx, 0)]
        y = np.asarray(cand_ids == rec_owner[:, None], dtype=bool) & valid
        has_owner = np.array([o is not None for o in rec_owner], dtype=bool)
        log(f"[train {country}] records with an owner: {has_owner.mean():.1%} | owner at BM25 rank 1: "
            f"{y[has_owner, 0].mean():.1%} | owner in top-{TOP_K}: {y[has_owner].any(axis=1).mean():.1%} "
            f"({time.time() - t0:.0f}s)")

        Fs.append(F)
        labels.append(y)
        valids.append(valid)
        has_owners.append(has_owner)
        del s1, vec, XT, q, Q

    F = np.concatenate(Fs)          # (records, K, features)
    y = np.concatenate(labels)      # (records, K)
    valid = np.concatenate(valids)
    has_owner = np.concatenate(has_owners)

    rng = np.random.default_rng(SEED)
    val_rec = rng.random(F.shape[0]) < 0.2          # split by RECORD so a record's 10 rows stay together
    tr_mask = valid & ~val_rec[:, None]
    va_mask = valid & val_rec[:, None]
    log(f"[train] fitting LightGBM on {int(tr_mask.sum()):,} (record, candidate) rows "
        f"({int(y[tr_mask].sum()):,} positive)")
    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        random_state=SEED, n_jobs=max(1, n_jobs), verbose=-1,
    )
    model.fit(F[tr_mask], y[tr_mask].astype(np.int8))

    probs = np.zeros(y.shape, dtype=np.float32)
    probs[va_mask] = model.predict_proba(F[va_mask])[:, 1]
    pv, yv, ov = probs[val_rec], y[val_rec], has_owner[val_rec]
    best = pv.argmax(axis=1)
    p_best = pv[np.arange(len(best)), best]
    correct = yv[np.arange(len(best)), best]
    n_pos = int(ov.sum())
    log(f"[train] held-out: owner in top-{TOP_K} for {yv[ov].any(axis=1).mean():.1%} of owned records; "
        f"model's best pick is the owner for {correct[ov].mean():.1%} (BM25 rank 1 alone: {yv[ov, 0].mean():.1%})")

    best_f = (0.0, 0.5, 0.0, 0.0)
    for t in np.arange(0.05, 0.96, 0.01):
        a = p_best >= t
        f, prec, rec = f05(int((a & correct).sum()), int((a & ~correct).sum()), n_pos)
        if f > best_f[0]:
            best_f = (f, float(t), prec, rec)
    f, t, prec, rec = best_f
    log(f"[train] held-out link-level F0.5={f:.4f}  precision={prec:.4f}  recall={rec:.4f}  threshold={t:.2f}")
    log("[train] (link-level proxy for the per-entity macro F0.5 on the leaderboard, not identical to it)")

    os.makedirs(out_dir, exist_ok=True)
    model.booster_.save_model(os.path.join(out_dir, "fast_model.txt"))
    with open(os.path.join(out_dir, "fast_threshold.txt"), "w") as fh:
        fh.write(f"{t}\n")
    log(f"[train] saved fast_model.txt + fast_threshold.txt to {out_dir}")


# ── Test: score every test S2/S3 record's top-10, write both submission TSVs ─

def test_stage(data_dir, out_dir, n_jobs, max_queries):
    test_dir = os.path.join(data_dir, "test")
    model_path = os.path.join(out_dir, "fast_model.txt")
    lgb.Booster(model_file=model_path)  # fail fast if the model is missing, before any heavy work
    with open(os.path.join(out_dir, "fast_threshold.txt")) as fh:
        threshold = float(fh.read().strip())
    log(f"[test] threshold={threshold:.2f}")

    all_s1_ids, match_parts, cand_parts = [], [], []
    for country in list_countries(os.path.join(test_dir, "test_source1.tsv")):
        s1 = read_source(os.path.join(test_dir, "test_source1.tsv"), country)
        all_s1_ids.extend(s1["entity_id"].to_list())
        vec, XT = build_index(s1)
        s1_ids = np.array(s1["entity_id"].to_list(), dtype=object)
        log(f"[test {country}] index {s1.height:,} S1")

        for src, fname in (("S2", "test_source2.tsv"), ("S3", "test_source3.tsv")):
            t0 = time.time()
            q = read_source(os.path.join(test_dir, fname), country).with_columns(pl.lit(src).alias("src"))
            if max_queries:
                q = q.head(max_queries)
            if q.height == 0:
                continue
            Q = encode_queries(vec, q)
            idx, _, probs = run_chunks(Q, q, XT, s1, n_jobs, model_path, desc=f"test {country}/{src}")

            rows = np.arange(len(idx))
            best = probs.argmax(axis=1)
            p_best = probs[rows, best]
            best_idx = idx[rows, best]
            take = (best_idx >= 0) & (p_best >= threshold)

            rec_ids = np.array(q["entity_id"].to_list(), dtype=object)
            for k in range(CANDIDATE_OUT_K):
                ok = idx[:, k] >= 0
                cand_parts.append(pl.DataFrame({
                    "source1_entity_id": s1_ids[idx[ok, k]].tolist(), "cand_id": rec_ids[ok].tolist()}))
            assigned = pl.DataFrame({
                "source1_entity_id": s1_ids[best_idx[take]].tolist(), "cand_id": rec_ids[take].tolist()})
            match_parts.append(assigned)
            cand_parts.append(assigned)  # guarantees matches ⊆ candidates even when the pick ranked 6th-10th
            log(f"[test {country}/{src}] {q.height:,} records, {int(take.sum()):,} assigned "
                f"({take.mean():.1%}), {int((take & (best > 0)).sum()):,} of them to a non-rank-1 candidate "
                f"({time.time() - t0:.0f}s)")
            del q, Q, idx, probs
        del s1, vec, XT

    os.makedirs(out_dir, exist_ok=True)
    empty = pl.DataFrame(schema={"source1_entity_id": pl.Utf8, "cand_id": pl.Utf8})
    matches = pl.concat(match_parts) if match_parts else empty
    cands = pl.concat(cand_parts) if cand_parts else empty
    write_lists(all_s1_ids, matches, os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids")
    write_lists(all_s1_ids, cands, os.path.join(out_dir, "candidate_pairs.tsv"), "candidate_entity_ids")
    n_matched = matches["source1_entity_id"].n_unique()
    log(f"[test] S1 entities with >=1 match: {n_matched:,} / {len(all_s1_ids):,} "
        f"(predicted singleton rate {1 - n_matched / max(len(all_s1_ids), 1):.1%}; train truth is 5.6%)")


# ══ v2: cached extract -> fit (exact leaderboard metric + entity-aware rules) -> predict ══════
#
# Retrieval is ~94% of the runtime (profiled: 2.7-3.0 ms/record vs 0.15-0.19 ms for features),
# so it runs exactly once per dataset in `extract`, which saves every record's top-10 candidate
# ids and pair features to disk. `fit` and `predict` then work from that cache in minutes.

V2_MAX_DF = 0.02        # profiled: 2.7x faster retrieval on India than 0.05 for -0.7pt recall@10
V2_MODEL = "fast_model_v2.txt"
V2_PARAMS = "fast_params_v2.json"
PREDICT_CHUNK = 200_000


def _cache_path(cache_dir, split, country):
    return os.path.join(cache_dir, f"{split}_{country}")


def list_caches(cache_dir, split):
    if not os.path.isdir(cache_dir):
        return []
    pre = split + "_"
    return sorted(d[len(pre):] for d in os.listdir(cache_dir)
                  if d.startswith(pre) and os.path.exists(os.path.join(cache_dir, d, "done")))


def extract_stage(data_dir, cache_dir, split, countries, n_jobs, sample, max_df):
    d = os.path.join(data_dir, split)
    s1_path = os.path.join(d, f"{split}_source1.tsv")
    for country in countries or list_countries(s1_path):
        out = _cache_path(cache_dir, split, country)
        if os.path.exists(os.path.join(out, "done")):
            log(f"[extract {split} {country}] cache already complete, skipping ({out})")
            continue
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        s1 = read_source(s1_path, country)
        vec, XT = build_index(s1, max_df)
        q = pl.concat([
            read_source(os.path.join(d, f"{split}_source2.tsv"), country).with_columns(pl.lit("S2").alias("src")),
            read_source(os.path.join(d, f"{split}_source3.tsv"), country).with_columns(pl.lit("S3").alias("src")),
        ])
        sampled = bool(sample) and q.height > sample
        if sampled:
            q = q.sample(n=sample, seed=SEED)
        n = q.height
        log(f"[extract {split} {country}] index {s1.height:,} S1 (max_df={max_df}), "
            f"{'a sample of' if sampled else 'ALL'} {n:,} S2/S3 records")
        q.select(pl.col("entity_id").alias("rec_id"), "src").write_parquet(os.path.join(out, "records.parquet"))
        s1.select("entity_id").write_parquet(os.path.join(out, "s1.parquet"))
        # written chunk by chunk as results arrive, so the full feature tensor never sits in RAM
        idx_mm = np.lib.format.open_memmap(os.path.join(out, "idx.npy"), mode="w+", dtype=np.int32, shape=(n, TOP_K))
        F_mm = np.lib.format.open_memmap(os.path.join(out, "F.npy"), mode="w+", dtype=np.float16,
                                         shape=(n, TOP_K, len(PAIR_FEATURES)))
        Q = encode_queries(vec, q)
        for s, (idx, _, F) in iter_chunks(Q, q, XT, s1, n_jobs, None, desc=f"extract {split} {country}"):
            idx_mm[s:s + len(idx)] = idx
            F_mm[s:s + len(idx)] = F
        idx_mm.flush()
        F_mm.flush()
        del idx_mm, F_mm
        with open(os.path.join(out, "meta.json"), "w") as fh:
            json.dump({"split": split, "country": country, "n_records": n, "n_s1": s1.height, "sampled": sampled,
                       "top_k": TOP_K, "features": PAIR_FEATURES, "max_df": max_df}, fh)
        open(os.path.join(out, "done"), "w").close()
        log(f"[extract {split} {country}] done in {(time.time() - t0) / 60:.1f} min -> {out}")
        del s1, vec, XT, q, Q


def load_cache(cache_dir, split, country):
    p = _cache_path(cache_dir, split, country)
    with open(os.path.join(p, "meta.json")) as fh:
        meta = json.load(fh)
    recs = pl.read_parquet(os.path.join(p, "records.parquet"))
    return {
        "dir": p, "meta": meta, "country": country,
        "rec_ids": recs["rec_id"], "is_s2": (recs["src"] == "S2").to_numpy(),
        "s1_ids": pl.read_parquet(os.path.join(p, "s1.parquet"))["entity_id"],
        "idx": np.load(os.path.join(p, "idx.npy"), mmap_mode="r"),
        "F": np.load(os.path.join(p, "F.npy"), mmap_mode="r"),
    }


def load_owners(data_dir):
    gt = pl.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), separator="\t",
                     infer_schema_length=0, quote_char=None)
    return (
        gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .rename({"matched_entity_ids": "cand_id", "source1_entity_id": "owner"})
    )


def owner_rows(cache, owners):
    """Row (in this cache's S1 index) of each record's true owner; -1 if it has none."""
    recs = pl.DataFrame({"rec_id": cache["rec_ids"]}).with_row_index("row")
    s1 = pl.DataFrame({"owner": cache["s1_ids"]}).with_row_index("s1_row")
    m = (recs.join(owners, left_on="rec_id", right_on="cand_id", how="left")
         .join(s1, on="owner", how="left").sort("row"))
    return m["s1_row"].fill_null(-1).cast(pl.Int64).to_numpy()


def predict_probs(booster, cache, rows_mask=None, n_threads=0):
    idx_mm, F_mm = cache["idx"], cache["F"]
    n = idx_mm.shape[0]
    out = np.zeros((n, TOP_K), dtype=np.float32)
    for s in tqdm(range(0, n, PREDICT_CHUNK), desc=f"predict {cache['country']}", mininterval=10):
        e = min(n, s + PREDICT_CHUNK)
        v = np.asarray(idx_mm[s:e]) >= 0
        if rows_mask is not None:
            v &= rows_mask[s:e, None]
        if v.any():
            F = np.asarray(F_mm[s:e], dtype=np.float32)
            out[s:e][v] = booster.predict(F[v], num_threads=n_threads)
    return out


def assign(idx, probs, is_s2, n_s1, t_high, t_sib, t_first, sib_other_source):
    """
    Entity-aware assignment. Each record can only go to its highest-probability candidate:
      A. p >= t_high
      B. t_sib <= p < t_high, if that S1 already holds a stage-A record
         (from the other source only, when sib_other_source)
      C. p >= t_first, for S1 entities still empty after A+B: their single most likely record
    Why: per entity F0.5 = 1.25k / (k + f + 0.25n), so an entity's first correct link is worth
    far more than its 4th, and a wrong link on an empty non-singleton costs nothing, which a
    single global threshold cannot express. Returns (S1 row per record or -1, stage 0/1/2/3).
    """
    rows = np.arange(len(idx))
    b = probs.argmax(axis=1)
    p = probs[rows, b]
    e = np.asarray(idx[rows, b], dtype=np.int64)
    valid = (e >= 0) & (p > 0)
    ec = np.where(valid, e, 0)
    stage = np.zeros(len(idx), dtype=np.int8)
    stage[valid & (p >= t_high)] = 1
    if t_sib < t_high:
        a = stage == 1
        if sib_other_source:
            c2 = np.bincount(e[a & is_s2], minlength=n_s1)
            c3 = np.bincount(e[a & ~is_s2], minlength=n_s1)
            has = np.where(is_s2, c3[ec], c2[ec]) > 0
        else:
            has = np.bincount(e[a], minlength=n_s1)[ec] > 0
        stage[valid & (stage == 0) & (p >= t_sib) & has] = 2
    if t_first < t_high:
        cnt = np.bincount(e[stage > 0], minlength=n_s1)
        cand = np.where(valid & (stage == 0) & (p >= t_first) & (cnt[ec] == 0))[0]
        if len(cand):
            order = cand[np.lexsort((-p[cand], e[cand]))]
            first = order[np.r_[True, e[order][1:] != e[order][:-1]]]
            stage[first] = 3
    return np.where(stage > 0, e, -1), stage


def entity_f05(assigned_e, owner, n_true):
    """Exact per-S1 leaderboard F0.5 (singletons: 1 if left empty, else 0)."""
    n = len(n_true)
    a = assigned_e >= 0
    correct = a & (assigned_e == owner)
    k = np.bincount(assigned_e[correct], minlength=n)
    f = np.bincount(assigned_e[a & ~correct], minlength=n)
    return np.where(n_true == 0, (f == 0).astype(np.float64), 1.25 * k / np.maximum(k + f + 0.25 * n_true, 1e-9))


def _evaluate(worlds, params):
    tot = cnt = 0
    for w in worlds:
        e, _ = assign(w["idx"], w["probs"], w["cache"]["is_s2"], len(w["n_true"]), **params)
        F = entity_f05(e, w["owner"], w["n_true"])
        tot += F.sum()
        cnt += len(F)
    return tot / cnt


def _report(worlds, params, label):
    for w in worlds:
        e, stage = assign(w["idx"], w["probs"], w["cache"]["is_s2"], len(w["n_true"]), **params)
        F = entity_f05(e, w["owner"], w["n_true"])
        sing = w["n_true"] == 0
        log(f"[{label} {w['cache']['country']}] leaderboard-style F0.5={F.mean():.4f} | singletons "
            f"{F[sing].mean():.3f} (n={sing.sum():,}) | non-singletons {F[~sing].mean():.3f}, of which "
            f"{(F[~sing] == 0).mean():.1%} score 0 | links by stage A/B/C: "
            f"{(stage == 1).sum():,}/{(stage == 2).sum():,}/{(stage == 3).sum():,}")


def tune(worlds):
    def single(t):
        return {"t_high": float(t), "t_sib": float(t), "t_first": float(t), "sib_other_source": False}

    base = {float(t): _evaluate(worlds, single(t)) for t in np.round(np.arange(0.30, 0.951, 0.025), 3)}
    t_base = max(base, key=base.get)
    log(f"[tune] one global threshold (the submitted method): best t={t_base:.3f} -> F0.5 {base[t_base]:.4f}")
    _report(worlds, single(t_base), "tune single-threshold")

    best, best_score = single(t_base), base[t_base]

    def consider(cand):
        nonlocal best, best_score
        s = _evaluate(worlds, cand)
        if s > best_score:
            best, best_score = cand, s

    for rnd in range(2):
        for other in (False, True):
            for ts in np.round(np.arange(0.05, best["t_high"], 0.05), 3):
                consider(dict(best, t_sib=float(ts), sib_other_source=other))
        for tf in np.round(np.arange(0.05, best["t_high"], 0.05), 3):
            consider(dict(best, t_first=float(tf)))
        for th in np.round(np.arange(max(0.30, best["t_high"] - 0.15), min(0.976, best["t_high"] + 0.151), 0.025), 3):
            consider(dict(best, t_high=float(th), t_sib=min(best["t_sib"], float(th)),
                          t_first=min(best["t_first"], float(th))))
        log(f"[tune] round {rnd + 1}: {best} -> F0.5 {best_score:.4f}")
    _report(worlds, best, "tune entity-aware")
    return best, best_score, base[t_base], t_base


def _fit_lgb(parts, mask_fn, n_threads):
    X = np.concatenate([p["X"][mask_fn(p)] for p in parts])
    y = np.concatenate([p["y"][mask_fn(p)] for p in parts])
    log(f"[fit] LightGBM on {len(y):,} (record, candidate) rows ({int(y.sum()):,} positive)")
    m = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        random_state=SEED, n_jobs=n_threads if n_threads > 0 else -1, verbose=-1,
    )
    m.fit(X, y)
    return m.booster_


def fit_stage(data_dir, cache_dir, out_dir, fit_sample, n_threads):
    owners = load_owners(data_dir)
    rng = np.random.default_rng(SEED)
    parts, worlds = [], []
    for country in list_caches(cache_dir, "train"):
        c = load_cache(cache_dir, "train", country)
        owner = owner_rows(c, owners)
        n, n_s1, full = len(owner), c["meta"]["n_s1"], not c["meta"]["sampled"]
        # fold by S1 ENTITY: every record of an entity lands in the same fold, so the model that
        # scores an entity's records for evaluation has never seen any of them
        ent_fold = rng.integers(0, 2, n_s1)
        rec_fold = np.where(owner >= 0, ent_fold[np.maximum(owner, 0)], rng.integers(0, 2, n))
        sel = np.sort(rng.choice(n, size=min(fit_sample, n), replace=False))
        idx_s = np.asarray(c["idx"][sel])
        valid = idx_s >= 0
        y = (idx_s == owner[sel][:, None]) & valid
        parts.append({"X": np.asarray(c["F"][sel], dtype=np.float32)[valid], "y": y[valid].astype(np.int8),
                      "fold": np.broadcast_to(rec_fold[sel][:, None], valid.shape)[valid], "full": full})
        idx_all = np.asarray(c["idx"])
        has = owner >= 0
        in_top = (idx_all == owner[:, None]).any(axis=1)
        log(f"[fit {country}] {n:,} records ({'full world' if full else 'sample'}), {has.mean():.1%} have an owner, "
            f"owner in top-{TOP_K}: {in_top[has].mean():.1%}; fitting on {len(sel):,} of them")
        if full:
            worlds.append({"cache": c, "idx": idx_all, "owner": owner, "rec_fold": rec_fold,
                           "n_true": np.bincount(owner[has], minlength=n_s1)})

    os.makedirs(out_dir, exist_ok=True)
    result = {"params": {"t_high": 0.78, "t_sib": 0.78, "t_first": 0.78, "sib_other_source": False}}
    if worlds:
        excl = []
        for f in (0, 1):
            log(f"[fit] evaluation model {f + 1}/2 (never sees fold-{f} entities)")
            excl.append(_fit_lgb(parts, lambda p, f=f: (p["fold"] != f) if p["full"] else np.ones(len(p["y"]), bool),
                                 n_threads))
        for w in worlds:
            w["probs"] = sum(predict_probs(excl[f], w["cache"], rows_mask=(w["rec_fold"] == f), n_threads=n_threads)
                             for f in (0, 1))
        params, score, base_score, t_base = tune(worlds)
        result = {"params": params, "leaderboard_style_f05": score,
                  "single_threshold_f05": base_score, "single_threshold": t_base,
                  "evaluated_on": [w["cache"]["country"] for w in worlds]}
        log(f"[fit] leaderboard-style F0.5 on held-out entities: single threshold {base_score:.4f} -> "
            f"entity-aware {score:.4f}")
    else:
        log("[fit] WARNING: no full-world train cache (extract one with --split train and no --sample); "
            "cannot evaluate or tune, using the old single threshold 0.78")

    log("[fit] final model on all sampled records")
    _fit_lgb(parts, lambda p: np.ones(len(p["y"]), bool), n_threads).save_model(os.path.join(out_dir, V2_MODEL))
    with open(os.path.join(out_dir, V2_PARAMS), "w") as fh:
        json.dump(result, fh, indent=2)
    log(f"[fit] saved {V2_MODEL} + {V2_PARAMS} to {out_dir}")


def predict_stage(data_dir, cache_dir, out_dir, n_threads, overrides):
    model_path = os.path.join(out_dir, V2_MODEL)
    booster = lgb.Booster(model_file=model_path)
    with open(os.path.join(out_dir, V2_PARAMS)) as fh:
        params = json.load(fh)["params"]
    params.update({k: v for k, v in overrides.items() if v is not None})
    log(f"[predict] assignment params: {params}")

    test_dir = os.path.join(data_dir, "test")
    missing = set(list_countries(os.path.join(test_dir, "test_source1.tsv"))) - set(list_caches(cache_dir, "test"))
    if missing:
        raise SystemExit(f"no complete test cache for {sorted(missing)}: run --stage extract --split test first")
    all_s1_ids = read_source(os.path.join(test_dir, "test_source1.tsv"))["entity_id"].to_list()

    match_parts, cand_parts = [], []
    for country in list_caches(cache_dir, "test"):
        c = load_cache(cache_dir, "test", country)
        probs_path = os.path.join(c["dir"], "probs_v2.npy")
        if os.path.exists(probs_path) and os.path.getmtime(probs_path) > os.path.getmtime(model_path):
            probs = np.load(probs_path)
            log(f"[predict {country}] reusing saved probabilities (model unchanged)")
        else:
            probs = predict_probs(booster, c, n_threads=n_threads)
            np.save(probs_path, probs)
        idx = np.asarray(c["idx"])
        n_s1 = c["meta"]["n_s1"]
        e, stage = assign(idx, probs, c["is_s2"], n_s1, **params)
        s1_ids = np.array(c["s1_ids"].to_list(), dtype=object)
        rec_ids = np.array(c["rec_ids"].to_list(), dtype=object)
        a = e >= 0
        m = pl.DataFrame({"source1_entity_id": s1_ids[e[a]].tolist(), "cand_id": rec_ids[a].tolist()})
        match_parts.append(m)
        cand_parts.append(m)  # guarantees matches ⊆ candidates
        for k in range(CANDIDATE_OUT_K):
            ok = idx[:, k] >= 0
            cand_parts.append(pl.DataFrame({"source1_entity_id": s1_ids[idx[ok, k]].tolist(),
                                            "cand_id": rec_ids[ok].tolist()}))
        empty_s1 = (np.bincount(e[a], minlength=n_s1) == 0).mean()
        log(f"[predict {country}] {len(e):,} records, {a.mean():.1%} assigned (stage A {(stage == 1).sum():,}, "
            f"B {(stage == 2).sum():,}, C {(stage == 3).sum():,}); S1 left empty: {empty_s1:.1%}")

    empty = pl.DataFrame(schema={"source1_entity_id": pl.Utf8, "cand_id": pl.Utf8})
    matches = pl.concat(match_parts) if match_parts else empty
    cands = pl.concat(cand_parts) if cand_parts else empty
    write_lists(all_s1_ids, matches, os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids")
    write_lists(all_s1_ids, cands, os.path.join(out_dir, "candidate_pairs.tsv"), "candidate_entity_ids")
    n_matched = matches["source1_entity_id"].n_unique()
    log(f"[predict] S1 entities with >=1 match: {n_matched:,} / {len(all_s1_ids):,} "
        f"(predicted singleton rate {1 - n_matched / max(len(all_s1_ids), 1):.1%}; train truth is 5.6%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--out-dir", required=True, help="where the model and both TSVs are written")
    ap.add_argument("--stage", choices=["extract", "fit", "predict", "all", "train", "test"], default="all",
                    help="v2: extract / fit / predict.  v1 (single run, no cache): all / train / test")
    ap.add_argument("--n-jobs", type=int, default=max(1, min(24, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--train-sample", type=int, default=150_000, help="v1: sampled S2/S3 records per train country")
    ap.add_argument("--max-queries", type=int, default=0, help="v1 smoke test: cap test queries per (country, source)")
    ap.add_argument("--cache-dir", default=None, help="v2: feature cache location (default: <out-dir>/cache)")
    ap.add_argument("--split", choices=["train", "test"], help="v2 extract: which dataset to extract")
    ap.add_argument("--countries", default="", help="v2 extract: comma-separated, e.g. India,US (default: all)")
    ap.add_argument("--sample", type=int, default=0, help="v2 extract: only this many random records per country")
    ap.add_argument("--max-df", type=float, default=V2_MAX_DF, help="v2 extract: BM25 common-word cutoff")
    ap.add_argument("--fit-sample", type=int, default=200_000, help="v2 fit: records per train cache used for fitting")
    ap.add_argument("--t-high", type=float, help="v2 predict: override the tuned threshold for stage A")
    ap.add_argument("--t-sib", type=float, help="v2 predict: override stage B threshold")
    ap.add_argument("--t-first", type=float, help="v2 predict: override stage C threshold")
    args = ap.parse_args()
    cache_dir = args.cache_dir or os.path.join(args.out_dir, "cache")

    log(f"stage={args.stage}  n_jobs={args.n_jobs}  top_k={TOP_K}")
    t0 = time.time()
    if args.stage == "extract":
        if not args.split:
            ap.error("--stage extract needs --split train or --split test")
        countries = [c.strip() for c in args.countries.split(",") if c.strip()]
        extract_stage(args.data_dir, cache_dir, args.split, countries, args.n_jobs, args.sample, args.max_df)
    elif args.stage == "fit":
        fit_stage(args.data_dir, cache_dir, args.out_dir, args.fit_sample, args.n_jobs)
    elif args.stage == "predict":
        predict_stage(args.data_dir, cache_dir, args.out_dir, args.n_jobs,
                      {"t_high": args.t_high, "t_sib": args.t_sib, "t_first": args.t_first})
    else:
        if args.stage in ("all", "train"):
            train_stage(args.data_dir, args.out_dir, args.n_jobs, args.train_sample)
        if args.stage in ("all", "test"):
            test_stage(args.data_dir, args.out_dir, args.n_jobs, args.max_queries)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
