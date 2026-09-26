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

Usage (run from anywhere):
  python fast_match.py --data-dir /path/to/dataset --out-dir /path/to/output --n-jobs 24
  python fast_match.py ... --max-queries 20000      # smoke test: caps queries per (country, source)
  python fast_match.py ... --stage train            # only fit + save the model
  python fast_match.py ... --stage test             # reuse a saved model, write submission TSVs
"""

import argparse
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
    full.write_csv(path, separator="\t")
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


def run_chunks(Q, q, XT, s1, n_jobs, model_path, desc):
    """Returns (idx, sc, F) without a model, (idx, sc, probs) with one; rows align with q."""
    qn, qa = q["business_name"].to_list(), q["business_address"].to_list()
    qs2 = (q["src"] == "S2").to_numpy().astype(np.float32)
    tasks = [
        (Q[s:s + TASK_CHUNK], qn[s:s + TASK_CHUNK], qa[s:s + TASK_CHUNK], qs2[s:s + TASK_CHUNK])
        for s in range(0, q.height, TASK_CHUNK)
    ]
    initargs = (XT, s1["business_name"].to_list(), s1["business_address"].to_list(), model_path)
    if n_jobs <= 1:
        _init_worker(*initargs)
        results = [_process_chunk(t) for t in tqdm(tasks, desc=desc, mininterval=10)]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker, initargs=initargs) as ex:
            results = list(tqdm(ex.map(_process_chunk, tasks), total=len(tasks), desc=desc, mininterval=10))
    return tuple(np.concatenate([r[k] for r in results]) for k in range(3))


def build_index(s1):
    """
    BM25 document weights over S1 "name + address", returned transposed for Q @ XT.
    Measured on the same 3,000 true records per country as the TF-IDF baseline:
    recall@1 India 89.0% -> 91.4%, US 94.9% -> 95.4%; recall@10 India 94.2% -> 95.7%,
    US 97.8% -> 98.2%; same speed.
    """
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, max_df=MAX_DF, dtype=np.float32)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--out-dir", required=True, help="where the model and both TSVs are written")
    ap.add_argument("--stage", choices=["all", "train", "test"], default="all")
    ap.add_argument("--n-jobs", type=int, default=max(1, min(24, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--train-sample", type=int, default=150_000, help="sampled S2/S3 records per train country")
    ap.add_argument("--max-queries", type=int, default=0, help="smoke test: cap test queries per (country, source)")
    args = ap.parse_args()

    log(f"n_jobs={args.n_jobs}  top_k={TOP_K}")
    t0 = time.time()
    if args.stage in ("all", "train"):
        train_stage(args.data_dir, args.out_dir, args.n_jobs, args.train_sample)
    if args.stage in ("all", "test"):
        test_stage(args.data_dir, args.out_dir, args.n_jobs, args.max_queries)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
