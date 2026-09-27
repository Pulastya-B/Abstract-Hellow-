"""
Retrieval-first matcher.

Every S2/S3 record belongs to at most one S1 entity (0 violations across all
7,638,365 train ground-truth pairs). So instead of blocking pairs from the S1
side, each S2/S3 record is used as a query against a per-country word BM25
index over S1 "name + address", plus exact name/address-key matches. Its
top-15 S1 hits are each scored by a LightGBM pair model; a second, record-level
model reads the spread of those probabilities and decides whether the best
pick is safe, against thresholds tuned for F0.5 in a train world whose sibling
decoys are weighted to test's density. Unaccepted records are assigned to
nobody. S1 entities that no record claims come out empty, which is exactly the
right answer for singletons.

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
  python fast_match.py --data-dir D --out-dir O --stage vocab     # learn which words the noise adds/drops
  python fast_match.py --data-dir D --out-dir O --stage extract --split train --countries India
  python fast_match.py --data-dir D --out-dir O --stage extract --split train --countries US --sample 300000
  python fast_match.py --data-dir D --out-dir O --stage extract --split test
  python fast_match.py --data-dir D --out-dir O --stage fit       # exact leaderboard metric + rule tuning
  python fast_match.py --data-dir D --out-dir O --stage analyze   # error types + examples on held-out train
  python fast_match.py --data-dir D --out-dir O --stage predict   # writes both submission TSVs
  (or run all of it: bash ../run_pipeline.sh)

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
from rapidfuzz.distance import Levenshtein
from rapidfuzz.utils import default_process
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from textnorm import (address_parts, core_compare, core_extras, expand_abbrev, has_repeated_word, house_match,
                      is_subsequence, legal_form, name_core, noise_evidence, number_compare, numbers, skeleton,
                      strip_latin_accents)

TOKEN_PATTERN = r"(?u)\b\w+\b"
MAX_DF = 0.05           # drop tokens in >5% of a country's S1 docs ("road", state codes, "pvt")
BM25_K1 = 1.2
BM25_B = 0.75
TOP_K = 15              # S1 candidates retrieved and scored per record
MAX_INJECT = 4          # extra candidates from exact name/address keys that BM25's top-K missed
INJECT_CAP = 3          # a key shared by more S1 entities than this is too ambiguous to inject
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
    # v3: script-independent names (textnorm.skeleton), address parts, competition among candidates
    "name_skel_tset", "name_skel_ratio", "addr_skel_tset",
    "house_match", "seg_jacc", "state_match", "postal_match",
    "name_skel_gap_to_best", "addr_skel_gap_to_best", "house_dupes", "n_strong_addr",
    # v4: distinctive name tokens (textnorm.name_core): what's left unmatched on each side, and how rare
    "core_extra_q", "core_extra_s", "core_extra_q_idf", "core_extra_s_idf", "core_match_frac", "core_nospace",
    "num_common", "q_domain",
    # v5: learned noise vocabulary (--stage vocab): are the unmatched words ones the noise adds/drops?
    "extra_q_noise_min", "extra_q_distinct", "extra_s_drop_min", "extra_s_distinct", "q_dup_word",
    # v6: sibling decoys (numbers changed vs digits dropped, legal form), exact-key agreement
    "num_changed_q", "num_changed_s", "num_presence", "house_dig_lev", "house_dig_subseq",
    "legal_same", "legal_trans", "exact_name", "exact_glued", "exact_addr",
]
VOCAB_FILE = "noise_vocab.json"
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


def _country_of(df):
    return df["country"][0] if df.height else ""


def prep_name(s):
    return default_process(strip_latin_accents(s))


def prep_addr(s, country):
    return default_process(expand_abbrev(strip_latin_accents(s).lower(), country))


def bm25_preprocess(s):
    return strip_latin_accents(s).lower()


def doc_text(df):
    # The name's consonant skeleton is appended so Indian-script names can be retrieved via their
    # transliteration. Measured on 6,000 India records vs the full S1 index: Indian-script
    # recall@10 88.6% -> 95.9%, recall@1 78.3% -> 91.2%; Latin unchanged (97.2% -> 97.3%).
    country = _country_of(df)
    return [f"{n} {expand_abbrev(bm25_preprocess(a), country)} {skeleton(n)}"
            for n, a in zip(df["business_name"].to_list(), df["business_address"].to_list())]


# ── exact keys: a second retrieval channel next to BM25 ─────────────────────
#
# BM25 drops words in >5% of S1 docs, so "Apex Tech Pvt-Ltd" (no address) scores the same against every
# "Apex Tech ..." and its exact twin can fall outside the top-K; a made-up trade name ("Tavowexlyra") with
# the owner's exact address competes with every business on that street. These keys catch both.

def name_keys(name):
    """-> (sorted core tokens + legal form, core tokens glued together: 'creativeglobal.com' = 'Creative Global')."""
    core = name_core(name)
    glued = "".join(core)
    return (" ".join(sorted(core)) + "|" + legal_form(name) if core else "", glued if len(glued) >= 6 else "")


def addr_key(prepped):
    """Sorted tokens of a prepped address; only addresses with a number are specific enough."""
    toks = prepped.split()
    return " ".join(sorted(toks)) if len(toks) >= 3 and any(c.isdigit() for c in prepped) else ""


def _keys_chunk(args):
    names, addrs, country = args
    return [(*name_keys(n), addr_key(prep_addr(a, country))) for n, a in zip(names, addrs)]


def _all_keys(df, country, n_jobs):
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    step = 20_000
    tasks = [(names[s:s + step], addrs[s:s + step], country) for s in range(0, len(names), step)]
    if n_jobs <= 1 or len(tasks) == 1:
        return [k for t in tasks for k in _keys_chunk(t)]
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as ex:
        return [k for part in ex.map(_keys_chunk, tasks) for k in part]


def injections(s1, q, n_jobs=1):
    """(records, MAX_INJECT) S1 rows sharing an exact key with each record (-1 = none)."""
    country = _country_of(s1)
    maps = ({}, {}, {})
    for j, keys in enumerate(_all_keys(s1, country, n_jobs)):
        for m, k in zip(maps, keys):
            if k:
                m.setdefault(k, []).append(j)
    maps = tuple({k: v for k, v in m.items() if len(v) <= INJECT_CAP} for m in maps)
    out = np.full((q.height, MAX_INJECT), -1, dtype=np.int32)
    for i, keys in enumerate(_all_keys(q, country, n_jobs)):
        got = []
        for m, k in zip(maps, keys):
            for j in m.get(k, ()) if k else ():
                if j not in got:
                    got.append(j)
        if got:
            out[i, :min(len(got), MAX_INJECT)] = got[:MAX_INJECT]
    return out


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


def _addr_info(addr, country):
    house, segs, state, postal, joined = address_parts(addr, country)
    return house, segs, state, postal, skeleton(joined)


def _init_worker(XT, s1_names, s1_addrs, model_path, country="", vocab=None):
    # Everything about S1 that every candidate comparison needs is prepared
    # once per worker here, instead of once per (record, candidate) pair.
    _W["XT"] = XT
    _W["country"] = country
    _W["vocab"] = vocab or {}
    _W["sn"] = [prep_name(x) for x in s1_names]
    _W["sa"] = [prep_addr(x, country) for x in s1_addrs]
    _W["s_nonlatin"] = np.array([_nonlatin(x) for x in s1_names], dtype=np.float32)
    _W["s_digits"] = [_digit_info(x) for x in s1_addrs]
    _W["s_nums"] = [numbers(x) for x in s1_addrs]
    _W["s_legal"] = [legal_form(x) for x in s1_names]
    _W["s_skel"] = [skeleton(x) for x in s1_names]
    _W["s_addr"] = [_addr_info(x, country) for x in s1_addrs]
    _W["s_core"] = [name_core(x) for x in s1_names]
    df = {}
    for toks in _W["s_core"]:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    n = max(len(s1_names), 1)
    _W["idf"] = {t: float(np.log(n / c)) for t, c in df.items()}
    _W["idf_unseen"] = float(np.log(n))
    _W["booster"] = lgb.Booster(model_file=model_path) if model_path else None


def _topk(Q, inj=None):
    """BM25 top-K per record; exact-key candidates (inj) BM25 missed replace its lowest-ranked ones."""
    n = Q.shape[0]
    idx = np.full((n, TOP_K), -1, dtype=np.int64)
    sc = np.zeros((n, TOP_K), dtype=np.float32)
    for s in range(0, n, SUB_CHUNK):
        R = (Q[s:s + SUB_CHUNK] @ _W["XT"]).tocsr()
        for i in range(R.shape[0]):
            lo, hi = R.indptr[i], R.indptr[i + 1]
            d_all, c_all = R.data[lo:hi], R.indices[lo:hi]
            d, c = d_all, c_all
            if len(d) > TOP_K:
                sel = np.argpartition(-d, TOP_K)[:TOP_K]
                d, c = d[sel], c[sel]
            if inj is not None:
                extra = [j for j in inj[s + i] if j >= 0 and j not in c]
                if extra:
                    keep = TOP_K - len(extra)
                    if len(d) > keep:
                        o = np.argsort(-d)[:keep]
                        d, c = d[o], c[o]
                    es = [float(d_all[c_all == j][0]) if (c_all == j).any() else 0.0 for j in extra]
                    d = np.concatenate([d, np.array(es, dtype=d.dtype)])
                    c = np.concatenate([c, np.array(extra, dtype=c.dtype)])
            if not len(d):
                continue
            order = np.argsort(-d, kind="stable")
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
    s_skel, s_addr, country = _W["s_skel"], _W["s_addr"], _W["country"]
    s_core, idf, idf_unseen, vocab = _W["s_core"], _W["idf"], _W["idf_unseen"], _W["vocab"]
    s_nums, s_legal = _W["s_nums"], _W["s_legal"]
    legal_rates = vocab.get("legal", {})
    for i in range(n):
        a_name = prep_name(q_names[i])
        a_addr = prep_addr(q_addrs[i], country)
        a_empty = not a_addr
        a_digits, a_first = _digit_info(q_addrs[i])
        a_skel = skeleton(q_names[i])
        a_house, a_segs, a_state, a_postal, a_addr_skel = _addr_info(q_addrs[i], country)
        a_core = name_core(q_names[i])
        a_nums = numbers(q_addrs[i])
        a_legal = legal_form(q_names[i])
        a_core_key = sorted(a_core)
        a_glued = "".join(a_core)
        a_akey = addr_key(a_addr)
        a_hdig = re.sub(r"\D", "", a_house)
        low = q_names[i].lower()
        F[i, :, COL["q_addr_empty"]] = float(a_empty)
        F[i, :, COL["q_nonlatin"]] = _nonlatin(q_names[i])
        F[i, :, COL["q_domain"]] = float("www." in low or ".com" in low or ".in" in low)
        F[i, :, COL["q_dup_word"]] = float(has_repeated_word(q_names[i]))
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
            row[COL["name_skel_tset"]] = fuzz.token_set_ratio(a_skel, s_skel[j])
            row[COL["name_skel_ratio"]] = fuzz.ratio(a_skel, s_skel[j])
            b_house, b_segs, b_state, b_postal, b_addr_skel = s_addr[j]
            if not a_empty and b_addr:
                row[COL["addr_tset"]] = fuzz.token_set_ratio(a_addr, b_addr)
                row[COL["addr_ratio"]] = fuzz.ratio(a_addr, b_addr)
                row[COL["addr_skel_tset"]] = fuzz.token_set_ratio(a_addr_skel, b_addr_skel)
                row[COL["seg_jacc"]] = len(a_segs & b_segs) / max(len(a_segs | b_segs), 1)
                row[COL["state_match"]] = float(a_state == b_state) if a_state and b_state else -1
            else:
                for f in ("addr_tset", "addr_ratio", "addr_skel_tset", "seg_jacc", "state_match"):
                    row[COL[f]] = -1
            row[COL["house_match"]] = house_match(a_house, b_house)
            row[COL["postal_match"]] = float(a_postal == b_postal) if a_postal and b_postal else -1
            core_f, xq, xs = core_compare(a_core, s_core[j], idf, idf_unseen, fuzz.ratio, fuzz.partial_ratio)
            (row[COL["core_extra_q"]], row[COL["core_extra_s"]], row[COL["core_extra_q_idf"]],
             row[COL["core_extra_s_idf"]], row[COL["core_match_frac"]], row[COL["core_nospace"]]) = core_f
            (row[COL["extra_q_noise_min"]], row[COL["extra_q_distinct"]],
             row[COL["extra_s_drop_min"]], row[COL["extra_s_distinct"]]) = noise_evidence(xq, xs, vocab)
            b_digits, b_first = s_digits[j]
            if a_digits and b_digits:
                row[COL["num_jacc"]] = len(a_digits & b_digits) / len(a_digits | b_digits)
                row[COL["num_first_eq"]] = float(a_first == b_first)
                row[COL["num_common"]] = len(a_digits & b_digits)
            else:
                row[COL["num_jacc"]] = -1
                row[COL["num_first_eq"]] = -1
                row[COL["num_common"]] = -1
            la, lb = len(a_name), len(b_name)
            row[COL["name_len_ratio"]] = min(la, lb) / max(la, lb) if max(la, lb) else 0

            (row[COL["num_changed_q"]], row[COL["num_changed_s"]],
             row[COL["num_presence"]]) = number_compare(a_nums, s_nums[j])
            b_hdig = re.sub(r"\D", "", b_house)
            if a_hdig and b_hdig:
                row[COL["house_dig_lev"]] = Levenshtein.distance(a_hdig, b_hdig)
                short, long_ = sorted((a_hdig, b_hdig), key=len)
                row[COL["house_dig_subseq"]] = float(is_subsequence(short, long_))
            else:
                row[COL["house_dig_lev"]] = -1
                row[COL["house_dig_subseq"]] = -1
            b_legal = s_legal[j]
            if a_legal and b_legal:
                row[COL["legal_same"]] = float(a_legal == b_legal)
                row[COL["legal_trans"]] = legal_rates.get(f"{b_legal}>{a_legal}", -1.0)
            else:
                row[COL["legal_same"]] = -1
                row[COL["legal_trans"]] = -1
            b_core = s_core[j]
            row[COL["exact_name"]] = float(bool(a_core) and a_core_key == sorted(b_core) and a_legal == b_legal)
            row[COL["exact_glued"]] = float(len(a_glued) >= 6 and a_glued == "".join(b_core))
            row[COL["exact_addr"]] = float(bool(a_akey) and a_akey == addr_key(b_addr))

    # competition: how far each candidate is from the record's best candidate on each signal
    for f, gap in (("name_tset", "name_tset_gap_to_best"), ("addr_tset", "addr_tset_gap_to_best"),
                   ("name_skel_tset", "name_skel_gap_to_best"), ("addr_skel_tset", "addr_skel_gap_to_best")):
        vals = np.where(valid, F[:, :, COL[f]], -np.inf)
        F[:, :, COL[gap]] = np.where(valid, vals.max(axis=1, keepdims=True) - F[:, :, COL[f]], 0)
    # ...and how many candidates are equally plausible on address (ambiguous if > 1)
    F[:, :, COL["house_dupes"]] = ((F[:, :, COL["house_match"]] == 1) & valid).sum(axis=1, keepdims=True)
    F[:, :, COL["n_strong_addr"]] = ((F[:, :, COL["addr_skel_tset"]] >= 90) & valid).sum(axis=1, keepdims=True)

    return F


def _process_chunk(task):
    Q, q_names, q_addrs, q_is_s2, inj = task
    idx, sc = _topk(Q, inj)
    F = _pair_features(q_names, q_addrs, q_is_s2, idx, sc)
    booster = _W["booster"]
    if booster is None:
        return idx, sc, F
    n = idx.shape[0]
    probs = booster.predict(F.reshape(n * TOP_K, -1), num_threads=1).reshape(n, TOP_K)
    probs[idx < 0] = 0.0
    return idx, sc, probs.astype(np.float32)


def iter_chunks(Q, q, XT, s1, n_jobs, model_path, desc, vocab=None, inj=None):
    """Yields (start_row, result) in row order; result is (idx, sc, F) or (idx, sc, probs)."""
    qn, qa = q["business_name"].to_list(), q["business_address"].to_list()
    qs2 = (q["src"] == "S2").to_numpy().astype(np.float32)
    starts = list(range(0, q.height, TASK_CHUNK))
    tasks = [(Q[s:s + TASK_CHUNK], qn[s:s + TASK_CHUNK], qa[s:s + TASK_CHUNK], qs2[s:s + TASK_CHUNK],
              None if inj is None else inj[s:s + TASK_CHUNK]) for s in starts]
    country = s1["country"][0] if s1.height else ""
    initargs = (XT, s1["business_name"].to_list(), s1["business_address"].to_list(), model_path, country, vocab)
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
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, max_df=max_df or MAX_DF, dtype=np.float32,
                          preprocessor=bm25_preprocess)
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


def extract_stage(data_dir, cache_dir, split, countries, n_jobs, sample, max_df, vocab_path):
    if not os.path.exists(vocab_path):
        raise SystemExit(f"{vocab_path} not found: run --stage vocab first (same --out-dir)")
    with open(vocab_path) as fh:
        vocab = json.load(fh)
    log(f"[extract] noise vocabulary: {len(vocab['added']):,} record words, {len(vocab['dropped']):,} S1 words")
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
        t1 = time.time()
        inj = injections(s1, q, n_jobs)
        log(f"[extract {split} {country}] exact-key candidates for {(inj[:, 0] >= 0).mean():.1%} of records "
            f"({time.time() - t1:.0f}s)")
        for s, (idx, _, F) in iter_chunks(Q, q, XT, s1, n_jobs, None, desc=f"extract {split} {country}",
                                          vocab=vocab, inj=inj):
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


def vocab_stage(data_dir, out_dir, n_pairs, min_count=30):
    """
    Learn, from train TRUE pairs, how often the dataset's noise adds each word to a record name and drops
    each word from an S1 name (measured on 300K pairs: "smt"/"shri"/"praivet"/"limitet" are added ~100% of
    the times they appear, "group"/"care"/"services" dropped 25-35%). Written to noise_vocab.json.
    Only words seen >= min_count times are kept, so no single pair's label leaks into a feature.
    """
    train_dir = os.path.join(data_dir, "train")
    owners = load_owners(data_dir)
    if owners.height > n_pairs:
        owners = owners.sample(n=n_pairs, seed=SEED)

    def names(fname, ids, id_col):
        return (pl.scan_csv(os.path.join(train_dir, fname), separator="\t", infer_schema_length=0, quote_char=None)
                .join(pl.LazyFrame({"entity_id": ids}), on="entity_id", how="semi")
                .select(pl.col("entity_id").alias(id_col), pl.col("business_name").fill_null(""))
                .collect())

    cand_ids = owners["cand_id"].to_list()
    recs = pl.concat([names(f"train_source{k}.tsv", cand_ids, "cand_id") for k in (2, 3)])
    s1 = names("train_source1.tsv", owners["owner"].unique().to_list(), "owner")
    pairs = (owners.join(recs.rename({"business_name": "rn"}), on="cand_id")
             .join(s1.rename({"business_name": "sn"}), on="owner"))
    log(f"[vocab] learning noise vocabulary from {pairs.height:,} train true pairs")

    added, dropped, rec_tok, s1_tok, legal_pair, legal_s1 = {}, {}, {}, {}, {}, {}

    def bump(d, toks):
        for t in toks:
            d[t] = d.get(t, 0) + 1

    for rn, sn in tqdm(pairs.select("rn", "sn").iter_rows(), total=pairs.height, desc="vocab", mininterval=10):
        q, s = name_core(rn), name_core(sn)
        bump(rec_tok, q)
        bump(s1_tok, s)
        if q and s:
            _, xq, xs = core_extras(q, s, fuzz.ratio)
            bump(added, xq)
            bump(dropped, xs)
        lq, ls = legal_form(rn), legal_form(sn)
        if lq and ls:
            bump(legal_pair, [f"{ls}>{lq}"])
            bump(legal_s1, [ls])
    vocab = {"added": {t: added.get(t, 0) / c for t, c in rec_tok.items() if c >= min_count},
             "dropped": {t: dropped.get(t, 0) / c for t, c in s1_tok.items() if c >= min_count},
             # P(record's legal form | S1's legal form) on true pairs: "pvt>ltd" is common noise, "ltd>llp" is not
             "legal": {k: c / legal_s1[k.split(">")[0]] for k, c in legal_pair.items()
                       if legal_s1[k.split(">")[0]] >= min_count}}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, VOCAB_FILE), "w") as fh:
        json.dump(vocab, fh)
    top = sorted(((r, t) for t, r in vocab["added"].items() if rec_tok[t] >= 200), reverse=True)[:15]
    log(f"[vocab] {len(vocab['added']):,} record words / {len(vocab['dropped']):,} S1 words kept; most often "
        f"added by noise: {', '.join(f'{t} {r:.0%}' for r, t in top)}")
    keep = sorted(((k, r) for k, r in vocab["legal"].items() if k.split(">")[0] == k.split(">")[1]), key=lambda x: -x[1])
    log(f"[vocab] legal form kept by the noise: {', '.join(f'{k.split(chr(62))[0]} {r:.0%}' for k, r in keep)}")


# Features computed when a cache is LOADED (not stored in it), so they can be added without re-extracting.
# Records with no address (~3-4%) can only be matched by name, and several S1 entities often share that
# exact name ("City Constructions LLP" in Delhi and in Gujarat): the model needs to know whether a name is
# unique among S1 entities (confident) or shared (ambiguous; abstaining is right under F0.5).
DERIVED_FEATURES = ["s_name_freq", "name_dupes"]
MODEL_FEATURES = PAIR_FEATURES + DERIVED_FEATURES


def _s1_name_freq(data_dir, cache):
    """For each S1 row of the cache: how many S1 entities of that country share its core name."""
    split = cache["meta"]["split"]
    s1 = read_source(os.path.join(data_dir, split, f"{split}_source1.tsv"), cache["country"])
    keys = [" ".join(sorted(name_core(n))) for n in s1["business_name"].to_list()]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    by_id = dict(zip(s1["entity_id"].to_list(), (counts[k] for k in keys)))
    return np.array([by_id.get(e, 1) for e in cache["s1_ids"].to_list()], dtype=np.float32)


def with_derived(cache, idx_rows, F_rows):
    """(n, K, len(PAIR_FEATURES)) float32 -> (n, K, len(MODEL_FEATURES))."""
    valid = idx_rows >= 0
    freq = np.where(valid, cache["s1_freq"][np.maximum(idx_rows, 0)], 0).astype(np.float32)
    dupes = ((F_rows[:, :, COL["name_skel_tset"]] >= 95) & valid).sum(axis=1, keepdims=True)
    dupes = np.broadcast_to(dupes, idx_rows.shape).astype(np.float32)
    return np.concatenate([F_rows, freq[..., None], dupes[..., None]], axis=2)


def load_cache(cache_dir, split, country, data_dir):
    p = _cache_path(cache_dir, split, country)
    with open(os.path.join(p, "meta.json")) as fh:
        meta = json.load(fh)
    if meta.get("features") != PAIR_FEATURES:
        raise SystemExit(f"{p} was built with a different feature set than this code "
                         f"({len(meta.get('features', []))} vs {len(PAIR_FEATURES)} features). "
                         f"Use a new --out-dir (or delete that cache) and run --stage extract again.")
    recs = pl.read_parquet(os.path.join(p, "records.parquet"))
    cache = {
        "dir": p, "meta": meta, "country": country,
        "rec_ids": recs["rec_id"], "is_s2": (recs["src"] == "S2").to_numpy(),
        "s1_ids": pl.read_parquet(os.path.join(p, "s1.parquet"))["entity_id"],
        "idx": np.load(os.path.join(p, "idx.npy"), mmap_mode="r"),
        "F": np.load(os.path.join(p, "F.npy"), mmap_mode="r"),
    }
    cache["s1_freq"] = _s1_name_freq(data_dir, cache)
    return cache


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
        ix = np.asarray(idx_mm[s:e])
        v = ix >= 0
        if rows_mask is not None:
            v &= rows_mask[s:e, None]
        if v.any():
            F = with_derived(cache, ix, np.asarray(F_mm[s:e], dtype=np.float32))
            out[s:e][v] = booster.predict(F[v], num_threads=n_threads)
    return out


def assign(idx, probs, is_s2, n_s1, t_high, t_sib, t_first, sib_other_source, score=None):
    """
    Entity-aware assignment. Each record can only go to its highest-probability candidate:
      A. p >= t_high
      B. t_sib <= p < t_high, if that S1 already holds a stage-A record
         (from the other source only, when sib_other_source)
      C. p >= t_first, for S1 entities still empty after A+B: their single most likely record
    Why: per entity F0.5 = 1.25k / (k + f + 0.25n), so an entity's first correct link is worth
    far more than its 4th, and a wrong link on an empty non-singleton costs nothing, which a
    single global threshold cannot express. p is `score` when given (the record-level model's
    confidence that the pick is right), else the pick's pair probability.
    Returns (S1 row per record or -1, stage 0/1/2/3).
    """
    rows = np.arange(len(idx))
    b = probs.argmax(axis=1)
    p_pair = probs[rows, b]
    p = p_pair if score is None else score
    e = np.asarray(idx[rows, b], dtype=np.int64)
    valid = (e >= 0) & (p_pair > 0)
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


def entity_f05(assigned_e, owner, n_true, w_unowned=1.0):
    """
    Exact per-S1 leaderboard F0.5 (singletons: 1 if left empty, else 0), with false links made by
    UNOWNED records counted w_unowned times.

    Why: unowned records are generated sibling businesses (same street as a real S1, number changed), and
    test has ~2x as many of them per S1 as train (address-key mixture analysis: test unowned records hit an
    S1's street at the same rate as train's, i.e. they are all sibling-type, at 2.44 vs 1.22 per S1). So a
    false link seen in train stands for ~w of them in test. Singletons score 0/1, so there the expectation is
    kept at the group level: a singleton hit only by unowned records scores 1 - w (can be < 0; means only).
    """
    n = len(n_true)
    a = assigned_e >= 0
    correct = a & (assigned_e == owner)
    k = np.bincount(assigned_e[correct], minlength=n)
    fo = np.bincount(assigned_e[a & ~correct & (owner >= 0)], minlength=n)
    fu = np.bincount(assigned_e[a & (owner < 0)], minlength=n)
    single = np.where(fo > 0, 0.0, 1.0 - w_unowned * (fu > 0))
    return np.where(n_true == 0, single, 1.25 * k / np.maximum(k + fo + w_unowned * fu + 0.25 * n_true, 1e-9))


def decoy_weight(cache_dir, world):
    """Test unowned records per S1 / train unowned records per S1, assuming test has train's owned records per
    S1 (3.46 India; the address-key mixture analysis on test India gave 3.42)."""
    has = world["owner"] >= 0
    n_s1 = len(world["n_true"])
    owned, unowned = has.sum() / n_s1, max((~has).sum() / n_s1, 1e-9)
    metas = {}
    for c in list_caches(cache_dir, "test"):
        with open(os.path.join(_cache_path(cache_dir, "test", c), "meta.json")) as fh:
            metas[c] = json.load(fh)
    if not metas:
        return 2.0, None, owned, unowned
    use = [metas[world["cache"]["country"]]] if world["cache"]["country"] in metas else list(metas.values())
    rps = sum(m["n_records"] for m in use) / sum(m["n_s1"] for m in use)
    return float(np.clip((rps - owned) / unowned, 1.0, 4.0)), rps, owned, unowned


def _evaluate(worlds, params):
    tot = cnt = 0
    for w in worlds:
        e, _ = assign(w["idx"], w["probs"], w["cache"]["is_s2"], len(w["n_true"]), **params, score=w.get("score"))
        F = entity_f05(e, w["owner"], w["n_true"], w["w"])
        tot += F.sum()
        cnt += len(F)
    return tot / cnt


def _report(worlds, params, label):
    for w in worlds:
        e, stage = assign(w["idx"], w["probs"], w["cache"]["is_s2"], len(w["n_true"]), **params, score=w.get("score"))
        F = entity_f05(e, w["owner"], w["n_true"], w["w"])
        sing = w["n_true"] == 0
        log(f"[{label} {w['cache']['country']}] leaderboard-style F0.5={F.mean():.4f} (decoys x{w['w']:.2f}) | "
            f"singletons {F[sing].mean():.3f} (n={sing.sum():,}) | non-singletons {F[~sing].mean():.3f}, of which "
            f"{(F[~sing] <= 0).mean():.1%} score 0 | links by stage A/B/C: "
            f"{(stage == 1).sum():,}/{(stage == 2).sum():,}/{(stage == 3).sum():,}")


def tune(worlds, label):
    def single(t):
        return {"t_high": float(t), "t_sib": float(t), "t_first": float(t), "sib_other_source": False}

    base = {float(t): _evaluate(worlds, single(t)) for t in np.round(np.arange(0.30, 0.991, 0.025), 3)}
    t_base = max(base, key=base.get)
    log(f"[tune {label}] one global threshold: best t={t_base:.3f} -> F0.5 {base[t_base]:.4f}")

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
        for th in np.round(np.arange(max(0.30, best["t_high"] - 0.15), min(0.991, best["t_high"] + 0.151), 0.025), 3):
            consider(dict(best, t_high=float(th), t_sib=min(best["t_sib"], float(th)),
                          t_first=min(best["t_first"], float(th))))
        log(f"[tune {label}] round {rnd + 1}: {best} -> F0.5 {best_score:.4f}")
    _report(worlds, best, f"tune {label}")
    return best, best_score, base[t_base], t_base


def _fit_lgb(X, y, n_threads, label, **overrides):
    log(f"[fit] LightGBM ({label}) on {len(y):,} rows ({int(y.sum()):,} positive, {X.shape[1]} features)")
    kw = dict(n_estimators=1000, learning_rate=0.05, num_leaves=127, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
              random_state=SEED, n_jobs=n_threads if n_threads > 0 else -1, verbose=-1)
    kw.update(overrides)
    m = lgb.LGBMClassifier(**kw)
    m.fit(X, y)
    return m.booster_


def _fit_pairs(parts, mask_fn, n_threads, label):
    X = np.concatenate([p["X"][mask_fn(p)] for p in parts])
    y = np.concatenate([p["y"][mask_fn(p)] for p in parts])
    return _fit_lgb(X, y, n_threads, label)


REC_MODEL = "fast_recmodel_v2.txt"
REC_BASE = ["p1", "p2", "p3", "p_gap12", "p_sum", "n_p50", "pick_rank", "n_valid"]
REC_FEATURES = REC_BASE + [f"best_{f}" for f in MODEL_FEATURES] + [f"second_{f}" for f in MODEL_FEATURES]
REC_PARAMS = dict(n_estimators=600, num_leaves=63, min_child_samples=200, colsample_bytree=0.8)


def record_features(cache, probs):
    """
    One row per record for the record-level model: how the pair model's probability is spread over the
    record's candidates, plus the full feature rows of its top-2 candidates. It learns when a confident pick
    is still unsafe: two S1 entities at p=1.00 with the same name (the record has no address to decide),
    a sibling whose street matches but whose number doesn't.
    """
    idx_mm, F_mm = cache["idx"], cache["F"]
    n = len(probs)
    out = np.zeros((n, len(REC_FEATURES)), dtype=np.float32)
    nb = len(REC_BASE)
    nf = len(MODEL_FEATURES)
    for s in range(0, n, PREDICT_CHUNK):
        e = min(n, s + PREDICT_CHUNK)
        ix = np.asarray(idx_mm[s:e])
        P = probs[s:e]
        F = with_derived(cache, ix, np.asarray(F_mm[s:e], dtype=np.float32))
        r = np.arange(e - s)
        b = P.argmax(axis=1)
        P2 = P.copy()
        P2[r, b] = -1.0
        b2 = P2.argmax(axis=1)
        top = -np.sort(-P, axis=1)[:, :3]
        o = out[s:e]
        o[:, :nb] = np.column_stack([top[:, 0], top[:, 1], top[:, 2], top[:, 0] - top[:, 1], P.sum(axis=1),
                                     (P > 0.5).sum(axis=1), b, (ix >= 0).sum(axis=1)])
        o[:, nb:nb + nf] = F[r, b]
        o[:, nb + nf:] = F[r, b2]
    return out


def record_scores(booster, cache, probs, n_threads=0):
    out = np.zeros(len(probs), dtype=np.float32)
    for s in range(0, len(probs), PREDICT_CHUNK):
        e = min(len(probs), s + PREDICT_CHUNK)
        sub = {**cache, "idx": cache["idx"][s:e], "F": cache["F"][s:e]}
        out[s:e] = booster.predict(record_features(sub, probs[s:e]), num_threads=n_threads)
    return out


def fit_stage(data_dir, cache_dir, out_dir, fit_sample, n_threads, decoy_w):
    owners = load_owners(data_dir)
    rng = np.random.default_rng(SEED)
    parts, worlds = [], []
    for country in list_caches(cache_dir, "train"):
        c = load_cache(cache_dir, "train", country, data_dir)
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
        parts.append({"cache": c, "owner": owner, "rec_fold": rec_fold,
                      "X": with_derived(c, idx_s, np.asarray(c["F"][sel], dtype=np.float32))[valid],
                      "y": y[valid].astype(np.int8),
                      "fold": np.broadcast_to(rec_fold[sel][:, None], valid.shape)[valid]})
        idx_all = np.asarray(c["idx"])
        has = owner >= 0
        in_top = (idx_all == owner[:, None]).any(axis=1)
        log(f"[fit {country}] {n:,} records ({'full world' if full else 'sample'}), {has.mean():.1%} have an owner, "
            f"owner in top-{TOP_K}: {in_top[has].mean():.1%}; fitting on {len(sel):,} of them")
        if full:
            worlds.append({"cache": c, "idx": idx_all, "owner": owner, "part": len(parts) - 1,
                           "n_true": np.bincount(owner[has], minlength=n_s1)})

    os.makedirs(out_dir, exist_ok=True)
    result = {"params": {"t_high": 0.78, "t_sib": 0.78, "t_first": 0.78, "sib_other_source": False},
              "use_record_model": False}
    if worlds:
        # 1. pair model, out-of-fold: every train record scored by the model that never saw its entity
        excl = [_fit_pairs(parts, lambda p, f=f: p["fold"] != f, n_threads, f"pair, without fold {f}")
                for f in (0, 1)]
        for p in parts:
            p["probs"] = sum(predict_probs(excl[f], p["cache"], rows_mask=(p["rec_fold"] == f), n_threads=n_threads)
                             for f in (0, 1))
            idx_all = np.asarray(p["cache"]["idx"])
            rows = np.arange(len(idx_all))
            pick = idx_all[rows, p["probs"].argmax(axis=1)]
            p["R"] = record_features(p["cache"], p["probs"])
            p["ry"] = ((pick == p["owner"]) & (pick >= 0)).astype(np.int8)
            p["rvalid"] = p["probs"].max(axis=1) > 0
        del excl

        # 2. record-level model on those out-of-fold probabilities, itself out-of-fold for tuning
        for p in parts:
            p["rscore"] = np.zeros(len(p["ry"]), dtype=np.float32)
        for f in (0, 1):
            tr = [p["rvalid"] & (p["rec_fold"] != f) for p in parts]
            m = _fit_lgb(np.concatenate([p["R"][t] for p, t in zip(parts, tr)]),
                         np.concatenate([p["ry"][t] for p, t in zip(parts, tr)]), n_threads,
                         f"record, without fold {f}", **REC_PARAMS)
            for p in parts:
                te = p["rvalid"] & (p["rec_fold"] == f)
                p["rscore"][te] = m.predict(p["R"][te], num_threads=n_threads)
        log("[fit] record model, final (all folds)")
        allr = [p["rvalid"] for p in parts]
        _fit_lgb(np.concatenate([p["R"][t] for p, t in zip(parts, allr)]),
                 np.concatenate([p["ry"][t] for p, t in zip(parts, allr)]), n_threads, "record, final",
                 **REC_PARAMS).save_model(os.path.join(out_dir, REC_MODEL))
        for p in parts:
            del p["R"]

        # 3. tune the assignment rules on a test-like world: unowned (sibling) false links weighted up
        for w in worlds:
            p = parts[w["part"]]
            w["probs"], w["score"] = p["probs"], p["rscore"]
            np.save(os.path.join(w["cache"]["dir"], "probs_oof.npy"), w["probs"])  # for --stage analyze
            np.save(os.path.join(w["cache"]["dir"], "rec_oof.npy"), w["score"])
            wt, rps, owned, unowned = decoy_weight(cache_dir, w)
            w["w"] = decoy_w if decoy_w > 0 else wt
            log(f"[fit {w['cache']['country']}] train: {owned:.2f} owned + {unowned:.2f} unowned records per S1; "
                f"test: {rps if rps is None else round(rps, 2)} records per S1 -> unowned false links weighted "
                f"x{w['w']:.2f}{' (--decoy-weight)' if decoy_w > 0 else ''}")
        pair_worlds = [dict(w, score=None) for w in worlds]
        p_params, p_score, p_base, p_t = tune(pair_worlds, "pair-prob")
        r_params, r_score, r_base, r_t = tune(worlds, "record-model")
        use_rec = r_score > p_score
        params, score = (r_params, r_score) if use_rec else (p_params, p_score)
        chosen = worlds if use_rec else pair_worlds
        as_is = _evaluate([dict(w, w=1.0) for w in chosen], params)
        old = _evaluate([dict(w, w=1.0, score=None) for w in worlds],
                        {"t_high": .725, "t_sib": .7, "t_first": .725, "sib_other_source": False})
        log(f"[fit] leaderboard-style F0.5 on held-out entities, test-like world (decoys weighted): pair prob "
            f"{p_score:.4f} vs record model {r_score:.4f} -> using {'record model' if use_rec else 'pair prob'}")
        log(f"[fit] same params on the as-is train world (comparable to v5's 0.9698): {as_is:.4f}; "
            f"v5 rule (0.725 on pair prob) on this model, as-is world: {old:.4f}")
        result = {"params": params, "use_record_model": bool(use_rec), "decoy_weight": [w["w"] for w in worlds],
                  "leaderboard_style_f05": score, "as_is_world_f05": as_is,
                  "pair_prob": {"f05": p_score, "params": p_params, "single_threshold": p_t, "single_f05": p_base},
                  "record_model": {"f05": r_score, "params": r_params, "single_threshold": r_t, "single_f05": r_base},
                  "evaluated_on": [w["cache"]["country"] for w in worlds]}
    else:
        log("[fit] WARNING: no full-world train cache (extract one with --split train and no --sample); "
            "cannot evaluate or tune, using the old single threshold 0.78")

    log("[fit] final pair model on all sampled records")
    _fit_pairs(parts, lambda p: np.ones(len(p["y"]), bool), n_threads, "pair, final").save_model(
        os.path.join(out_dir, V2_MODEL))
    with open(os.path.join(out_dir, V2_PARAMS), "w") as fh:
        json.dump(result, fh, indent=2)
    log(f"[fit] saved {V2_MODEL} + {REC_MODEL} + {V2_PARAMS} to {out_dir}")


def _texts(path, ids):
    lf = pl.scan_csv(path, separator="\t", infer_schema_length=0, quote_char=None)
    ids_lf = pl.LazyFrame({"entity_id": sorted(set(ids))})
    return {r[0]: (r[1], r[2]) for r in lf.join(ids_lf, on="entity_id", how="semi")
            .select(["entity_id", "business_name", "business_address"]).collect().iter_rows()}


def analyze_stage(data_dir, cache_dir, out_dir, n_examples):
    """Counts and real examples of every error type on held-out train entities (needs --stage fit first)."""
    owners = load_owners(data_dir)
    with open(os.path.join(out_dir, V2_PARAMS)) as fh:
        saved = json.load(fh)
    params, use_rec = saved["params"], saved.get("use_record_model", False)
    weights = dict(zip(saved.get("evaluated_on", []), saved.get("decoy_weight", [])))
    train_dir = os.path.join(data_dir, "train")
    rng = np.random.default_rng(SEED)
    for country in list_caches(cache_dir, "train"):
        c = load_cache(cache_dir, "train", country, data_dir)
        pp = os.path.join(c["dir"], "probs_oof.npy")
        if c["meta"]["sampled"] or not os.path.exists(pp):
            continue
        lines = []

        def say(msg=""):
            print(msg, flush=True)
            lines.append(msg)

        probs, idx, owner = np.load(pp), np.asarray(c["idx"]), owner_rows(c, owners)
        score = np.load(os.path.join(c["dir"], "rec_oof.npy")) if use_rec else None
        n_s1, rows = c["meta"]["n_s1"], np.arange(len(idx))
        e, stage = assign(idx, probs, c["is_s2"], n_s1, **params, score=score)
        best = probs.argmax(axis=1)
        pick, p_best = idx[rows, best], (probs[rows, best] if score is None else score)
        has = owner >= 0
        owner_rank = np.where((idx == owner[:, None]) & has[:, None], np.arange(TOP_K)[None, :], TOP_K).min(axis=1)
        in_top = owner_rank < TOP_K
        indic = np.asarray(c["F"][:, 0, COL["q_nonlatin"]]) > 0
        cats = {
            "correct link": has & (e == owner),
            "REJECTED: owner was the model's pick, below threshold": has & (e < 0) & in_top & (pick == owner),
            f"REJECTED: owner in top-{TOP_K} but the model preferred another S1": has & (e < 0) & in_top & (pick != owner),
            "WRONG LINK: assigned to a different S1": has & (e >= 0) & (e != owner),
            f"RETRIEVAL MISS: owner not in top-{TOP_K}, nothing assigned": has & ~in_top & (e < 0),
            "unowned record, correctly left alone": ~has & (e < 0),
            "FALSE LINK: unowned record assigned to an S1": ~has & (e >= 0),
        }
        say(f"\n══════ ERROR ANALYSIS: train {country} (held-out predictions, params {params}, "
            f"{'record model' if use_rec else 'pair prob'} decides) ══════")
        say(f"{len(idx):,} records: {has.sum():,} owned, {(~has).sum():,} unowned; "
            f"{indic.mean():.1%} have an Indian-script name")
        say(f"{'category':62s} {'records':>10s} {'% of group':>10s} {'% Latin':>8s} {'% Indic':>8s} {'% S2':>6s}")
        for name, m in cats.items():
            grp = has if name[0] in "cRW" else ~has
            lat, ind = grp & ~indic, grp & indic
            say(f"{name:62s} {m.sum():10,} {m.sum() / max(grp.sum(), 1):10.2%} "
                f"{(m & lat).sum() / max(lat.sum(), 1):8.2%} {(m & ind).sum() / max(ind.sum(), 1):8.2%} "
                f"{c['is_s2'][m].mean() if m.any() else 0:6.1%}")

        n_true = np.bincount(owner[has], minlength=n_s1)
        Fe = entity_f05(e, owner, n_true)
        sing = n_true == 0
        say(f"\nper-entity (leaderboard formula, as-is train world): mean F0.5 {Fe.mean():.4f} | "
            f"singletons {Fe[sing].mean():.3f} ({sing.sum():,}) | non-singletons {Fe[~sing].mean():.3f}")
        wt = weights.get(country, 1.0)
        Fw = entity_f05(e, owner, n_true, wt)
        say(f"per-entity, test-like world (unowned false links x{wt:.2f}): mean F0.5 {Fw.mean():.4f} | "
            f"singletons {Fw[sing].mean():.3f} | non-singletons {Fw[~sing].mean():.3f}")
        lost = 1 - Fe
        say(f"points lost (as-is): {lost.sum() / len(Fe):.4f} total = singletons given a false link "
            f"{lost[sing].sum() / len(Fe):.4f} + real entities scoring 0 {lost[~sing & (Fe == 0)].sum() / len(Fe):.4f} "
            f"+ partly-found real entities {lost[~sing & (Fe > 0)].sum() / len(Fe):.4f}")

        picks = {name: rng.choice(np.where(m)[0], size=min(n_examples, int(m.sum())), replace=False)
                 for name, m in cats.items() if name[0] in "RWF" and m.any()}
        all_rows = np.concatenate(list(picks.values())) if picks else np.array([], dtype=np.int64)
        s1_ids, rec_ids = c["s1_ids"].to_list(), c["rec_ids"].to_list()
        need_s1 = {s1_ids[j] for i in all_rows for j in (owner[i], pick[i], e[i]) if j >= 0}
        need_rec = {rec_ids[i] for i in all_rows}
        s1_txt = _texts(os.path.join(train_dir, "train_source1.tsv"), need_s1)
        rec_txt = {**_texts(os.path.join(train_dir, "train_source2.tsv"), need_rec),
                   **_texts(os.path.join(train_dir, "train_source3.tsv"), need_rec)}

        def fmt(t):
            return f"{t[0]} | {t[1]}" if t else "?"

        def feats(i, r):
            if r >= TOP_K:
                return ""
            f = c["F"][i, r]
            return ("  ".join(f"{k}={float(f[COL[k]]):.0f}" for k in ("name_tset", "name_skel_tset", "addr_tset",
                                                                      "addr_skel_tset", "score", "num_changed_q",
                                                                      "num_changed_s"))
                    + "  " + "  ".join(f"{k}={float(f[COL[k]]):.2f}" for k in ("house_match", "seg_jacc",
                                                                               "legal_trans")))

        for name, sel in picks.items():
            say(f"\n── {name}: {len(sel)} random examples ──")
            for i in sel:
                o, pk = owner[i], pick[i]
                say(f"[{'S2' if c['is_s2'][i] else 'S3'}] best p={p_best[i]:.2f} "
                    f"(pair p={probs[i, best[i]]:.2f}) stage={stage[i]} "
                    f"owner rank={'-' if owner_rank[i] >= TOP_K else owner_rank[i] + 1}")
                say(f"   record : {fmt(rec_txt.get(rec_ids[i]))}")
                if o >= 0:
                    r = owner_rank[i]
                    say(f"   owner  : {fmt(s1_txt.get(s1_ids[o]))}"
                        + (f"   (p={probs[i, r]:.2f})  {feats(i, r)}" if r < TOP_K else ""))
                if pk >= 0 and pk != o:
                    say(f"   picked : {fmt(s1_txt.get(s1_ids[pk]))}   (p={p_best[i]:.2f})  {feats(i, best[i])}")

        path = os.path.join(out_dir, f"error_analysis_{country}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        log(f"[analyze] report also written to {path}")


def predict_stage(data_dir, cache_dir, out_dir, n_threads, overrides):
    model_path = os.path.join(out_dir, V2_MODEL)
    booster = lgb.Booster(model_file=model_path)
    with open(os.path.join(out_dir, V2_PARAMS)) as fh:
        saved = json.load(fh)
    params, use_rec = saved["params"], saved.get("use_record_model", False)
    params.update({k: v for k, v in overrides.items() if v is not None})
    rec_booster = lgb.Booster(model_file=os.path.join(out_dir, REC_MODEL)) if use_rec else None
    log(f"[predict] assignment params: {params}; decided by {'record model' if use_rec else 'pair prob'}")

    test_dir = os.path.join(data_dir, "test")
    missing = set(list_countries(os.path.join(test_dir, "test_source1.tsv"))) - set(list_caches(cache_dir, "test"))
    if missing:
        raise SystemExit(f"no complete test cache for {sorted(missing)}: run --stage extract --split test first")
    all_s1_ids = read_source(os.path.join(test_dir, "test_source1.tsv"))["entity_id"].to_list()

    match_parts, cand_parts = [], []
    for country in list_caches(cache_dir, "test"):
        c = load_cache(cache_dir, "test", country, data_dir)
        probs_path = os.path.join(c["dir"], "probs_v2.npy")
        if os.path.exists(probs_path) and os.path.getmtime(probs_path) > os.path.getmtime(model_path):
            probs = np.load(probs_path)
            log(f"[predict {country}] reusing saved probabilities (model unchanged)")
        else:
            probs = predict_probs(booster, c, n_threads=n_threads)
            np.save(probs_path, probs)
        idx = np.asarray(c["idx"])
        n_s1 = c["meta"]["n_s1"]
        score = record_scores(rec_booster, c, probs, n_threads) if use_rec else None
        e, stage = assign(idx, probs, c["is_s2"], n_s1, **params, score=score)
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
    ap.add_argument("--stage", choices=["vocab", "extract", "fit", "analyze", "predict", "all", "train", "test"],
                    default="all", help="v2: vocab / extract / fit / analyze / predict.  v1 (no cache): all / train / test")
    ap.add_argument("--examples", type=int, default=12, help="v2 analyze: examples printed per error type")
    ap.add_argument("--vocab-pairs", type=int, default=2_000_000, help="v2 vocab: train true pairs to learn from")
    ap.add_argument("--n-jobs", type=int, default=max(1, min(24, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--train-sample", type=int, default=150_000, help="v1: sampled S2/S3 records per train country")
    ap.add_argument("--max-queries", type=int, default=0, help="v1 smoke test: cap test queries per (country, source)")
    ap.add_argument("--cache-dir", default=None, help="v2: feature cache location (default: <out-dir>/cache)")
    ap.add_argument("--split", choices=["train", "test"], help="v2 extract: which dataset to extract")
    ap.add_argument("--countries", default="", help="v2 extract: comma-separated, e.g. India,US (default: all)")
    ap.add_argument("--sample", type=int, default=0, help="v2 extract: only this many random records per country")
    ap.add_argument("--max-df", type=float, default=V2_MAX_DF, help="v2 extract: BM25 common-word cutoff")
    ap.add_argument("--fit-sample", type=int, default=1_500_000, help="v2 fit: records per train cache used for fitting")
    ap.add_argument("--decoy-weight", type=float, default=0,
                    help="v2 fit: weight of false links made by unowned records when tuning (test has ~2x the "
                         "unowned records per S1 of train); 0 = derive it from the train and test caches")
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
        extract_stage(args.data_dir, cache_dir, args.split, countries, args.n_jobs, args.sample, args.max_df,
                      os.path.join(args.out_dir, VOCAB_FILE))
    elif args.stage == "vocab":
        vocab_stage(args.data_dir, args.out_dir, args.vocab_pairs)
    elif args.stage == "fit":
        fit_stage(args.data_dir, cache_dir, args.out_dir, args.fit_sample, args.n_jobs, args.decoy_weight)
    elif args.stage == "analyze":
        analyze_stage(args.data_dir, cache_dir, args.out_dir, args.examples)
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
