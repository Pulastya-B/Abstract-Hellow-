"""
v5 blocking: multi-channel recall-first redesign (replaces the v4 first/last/
sorted-token-only design that measured only 30.3% candidate recall on train —
LightGBM can never recover a true match that blocking never surfaces).

Channels (all UNIONed, never intersected — recall is the priority here, the
matcher downstream is what narrows precision):

  1. exact_name        — exact normalized-name match
  2. suffix_normalized  — exact match after legal-suffix stripping (core_name)
  3. rare_token         — inverted index on informative name tokens only
                          (tokens above RARE_TOKEN_MAX_DOC_FREQ in a country
                          partition are banned as standalone keys — "inc",
                          "store", "the" never form a block by themselves)
  4. name_tfidf         — word-level TF-IDF cosine retrieval on business name
  5. address_tfidf      — word-level TF-IDF cosine retrieval on address
  6. composite_tfidf    — word-level TF-IDF cosine retrieval on name+address
                          concatenated (catches cases where neither field
                          alone is close enough but the pair together is)
  7. postal             — exact postal/PIN code match
  8. numeric            — exact house/building-number + locality match
  9. embedding          — multilingual sentence-embedding ANN (ties channels
                          across script/language boundaries, e.g. Devanagari
                          vs. Latin, and generalizes to unseen countries like
                          France with no country-specific rules)

Every candidate's contributing channels are tracked (`block_methods`,
`num_blocks_agreeing`) — bookkeeping during the union, not an extra pass.

Processed one country at a time (as before) to keep each channel's working
set bounded to a single partition.
"""

import _thread_limits  # noqa: F401 — must be the first import; see that module's docstring

import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import config

# Query chunk size for the sparse top-k search (channels 4/5/6): bounds peak
# memory of each intermediate sparse similarity block instead of computing
# the full n_s1 x n_ob sparse product in one call.
TFIDF_QUERY_CHUNK_SIZE = 2000

_ALL_CHANNELS = [
    "exact_name", "suffix_normalized", "rare_token", "char_qgram", "name_tfidf",
    "address_tfidf", "composite_tfidf", "postal", "numeric", "embedding",
]

_embedding_model = None  # lazy-loaded singleton, shared across countries


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
        _embedding_model.max_seq_length = config.EMBEDDING_MAX_SEQ_LENGTH
        tqdm.write(f"  embedding model loaded on device={device}")
    return _embedding_model


def _empty_hits() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "source1_entity_id": pl.Utf8, "candidate_id": pl.Utf8,
        "candidate_source": pl.Utf8, "block_method": pl.Utf8,
    })


def _run_channel(name: str, fn, *args, **kwargs) -> pl.DataFrame:
    """
    Wraps a channel call with start/finish heartbeat logging. Without this,
    a slow-but-working channel (e.g. TF-IDF/embedding on a large country
    partition) prints NOTHING between "blocking by country: 0/1" and the
    channel finishing — indistinguishable in the log from a genuine hang.
    This cost real wall-clock time twice already: a run was killed by a
    labmate who saw no log output for several minutes and assumed it was
    stuck, when the underlying process may have been working correctly.
    """
    tqdm.write(f"    [{name}] starting...")
    t0 = time.time()
    result = fn(*args, **kwargs)
    tqdm.write(f"    [{name}] done in {time.time() - t0:.1f}s — {result.height:,} candidate rows")
    return result


# ── Channel 1/2: exact-match blocks ────────────────────────────────────────

def _exact_match_block(s1: pl.DataFrame, ob: pl.DataFrame, key_col: str, method: str) -> pl.DataFrame:
    s1k = s1.filter(pl.col(key_col) != "").select(["source1_entity_id", key_col])
    obk = ob.filter(pl.col(key_col) != "").select(["candidate_id", "candidate_source", key_col])
    if s1k.height == 0 or obk.height == 0:
        return _empty_hits()
    joined = s1k.join(obk, on=key_col, how="inner")
    return joined.select(["source1_entity_id", "candidate_id", "candidate_source"]).with_columns(
        pl.lit(method).alias("block_method")
    )


def _capped_join_block(
    s1: pl.DataFrame, ob: pl.DataFrame, key_cols: list, method: str,
    max_block_size: int, max_total_pairs: int,
) -> pl.DataFrame:
    """
    Exact-match join on key_cols, with the same two-layer safety cap
    _rare_token_block uses: drop any single key-value GROUP whose join would
    exceed max_block_size on either side, then bound the CUMULATIVE join
    size across all remaining groups to max_total_pairs (smallest/most-
    discriminative groups kept first).

    Added after _numeric_block (house_number + locality, no cap at all)
    produced 722M candidate rows from a single country partition — a
    locality-extraction bug made "locality" collapse onto a handful of
    Indian state names shared by hundreds of thousands of records, and nothing
    stopped the resulting (house_number, locality) join from exploding. The
    locality bug is fixed at the source (normalize.py), but this cap is the
    second, independent line of defense: any join-based channel needs one,
    since blocking keys derived from real-world text can degrade unpredictably
    for reasons that only show up on the full data, not on samples.
    """
    s1k = s1.filter(pl.all_horizontal([pl.col(c) != "" for c in key_cols])).select(
        ["source1_entity_id"] + key_cols
    )
    obk = ob.filter(pl.all_horizontal([pl.col(c) != "" for c in key_cols])).select(
        ["candidate_id", "candidate_source"] + key_cols
    )
    if s1k.height == 0 or obk.height == 0:
        return _empty_hits()

    s1_counts = s1k.group_by(key_cols).len().rename({"len": "n_s1"})
    ob_counts = obk.group_by(key_cols).len().rename({"len": "n_ob"})
    sizes = s1_counts.join(ob_counts, on=key_cols, how="inner")
    sizes = sizes.filter((pl.col("n_s1") <= max_block_size) & (pl.col("n_ob") <= max_block_size))
    sizes = sizes.with_columns((pl.col("n_s1") * pl.col("n_ob")).alias("pair_count"))
    sizes = sizes.sort("pair_count").with_columns(pl.col("pair_count").cum_sum().alias("cum_pairs"))
    safe = sizes.filter(pl.col("cum_pairs") <= max_total_pairs).select(key_cols)
    if safe.height == 0:
        return _empty_hits()

    s1k = s1k.join(safe, on=key_cols, how="inner")
    obk = obk.join(safe, on=key_cols, how="inner")

    joined = s1k.join(obk, on=key_cols, how="inner")
    return joined.select(["source1_entity_id", "candidate_id", "candidate_source"]).with_columns(
        pl.lit(method).alias("block_method")
    )


# ── Channel 3: rare-token inverted index ───────────────────────────────────

def _rare_token_block(s1: pl.DataFrame, ob: pl.DataFrame, max_block_size: int, max_total_pairs: int) -> pl.DataFrame:
    tqdm.write("      [rare_token] tokenizing...")
    s1_tok = s1.select(["source1_entity_id", "s1_name"]).filter(pl.col("s1_name") != "").with_columns(
        pl.col("s1_name").str.split(" ").alias("tokens")
    ).explode("tokens").filter(pl.col("tokens") != "")
    ob_tok = ob.select(["candidate_id", "candidate_source", "cand_name"]).filter(pl.col("cand_name") != "").with_columns(
        pl.col("cand_name").str.split(" ").alias("tokens")
    ).explode("tokens").filter(pl.col("tokens") != "")

    if s1_tok.height == 0 or ob_tok.height == 0:
        return _empty_hits()

    tqdm.write(f"      [rare_token] computing doc frequency over {s1_tok.height + ob_tok.height:,} token rows...")
    # Document frequency of each token across BOTH sides combined — a token
    # is banned as a standalone key once it's near-universal, not merely
    # because one particular join of it happens to be large.
    n_docs = s1.height + ob.height
    doc_freq = (
        pl.concat([
            s1_tok.select(["source1_entity_id", "tokens"]).unique().select("tokens"),
            ob_tok.select(["candidate_id", "tokens"]).unique().select("tokens"),
        ])
        .group_by("tokens").len().rename({"len": "df"})
        .with_columns((pl.col("df") / n_docs).alias("doc_freq_ratio"))
    )
    informative = doc_freq.filter(pl.col("doc_freq_ratio") <= config.RARE_TOKEN_MAX_DOC_FREQ).select("tokens")
    if informative.height == 0:
        return _empty_hits()
    tqdm.write(f"      [rare_token] {informative.height:,} informative tokens kept")

    s1_tok = s1_tok.join(informative, on="tokens", how="inner")
    ob_tok = ob_tok.join(informative, on="tokens", how="inner")

    tqdm.write("      [rare_token] sizing token groups against pair-count budget...")
    s1_counts = s1_tok.group_by("tokens").len().rename({"len": "n_s1"})
    ob_counts = ob_tok.group_by("tokens").len().rename({"len": "n_ob"})
    sizes = s1_counts.join(ob_counts, on="tokens", how="inner")
    sizes = sizes.filter((pl.col("n_s1") <= max_block_size) & (pl.col("n_ob") <= max_block_size))
    sizes = sizes.with_columns((pl.col("n_s1") * pl.col("n_ob")).alias("pair_count"))
    sizes = sizes.sort("pair_count").with_columns(pl.col("pair_count").cum_sum().alias("cum_pairs"))
    safe = sizes.filter(pl.col("cum_pairs") <= max_total_pairs).select("tokens")
    if safe.height == 0:
        return _empty_hits()

    s1_tok = s1_tok.join(safe, on="tokens", how="inner")
    ob_tok = ob_tok.join(safe, on="tokens", how="inner")
    tqdm.write(f"      [rare_token] joining {safe.height:,} safe token keys...")

    joined = s1_tok.join(ob_tok, on="tokens", how="inner")
    return joined.select(["source1_entity_id", "candidate_id", "candidate_source"]).unique().with_columns(
        pl.lit("rare_token").alias("block_method")
    )


# ── Char q-gram inverted-index retrieval ────────────────────────────────────
#
# Sprint-mode addition: word-level TF-IDF/rare_token miss transliteration
# variants and typos that share no whole word at all. Full char n-gram TF-IDF
# (dense cosine) was already proven too slow (~98% pairwise density on short
# business names). This channel gets fuzzy/typo-tolerant matching a
# different way: an inverted index over char n-grams with explicit IDF-like
# weighting and a document-frequency exclusion for near-universal grams
# (mirrors _rare_token_block's proven-fast Polars-join pattern instead of a
# dense similarity matrix), ranking candidates by summed weight of SHARED
# grams only — cost scales with actual gram overlap, not full n_s1 x n_ob.

def _extract_qgrams(name_col: str, n: int) -> pl.Expr:
    """Character n-grams over a name, via n shifted-substring columns unioned
    — avoids a Python-level sliding-window loop over every row."""
    length = pl.col(name_col).str.len_chars()
    grams = [pl.col(name_col).str.slice(i, n) for i in range(0, 30)]  # cap at 30 positions; names are short
    return pl.concat_list(grams).list.eval(pl.element().filter(pl.element().str.len_chars() == n))


def _char_qgram_block(
    s1: pl.DataFrame, ob: pl.DataFrame, name_col_s1: str, name_col_ob: str,
    top_k: int, max_doc_freq: float = 0.05,
    max_block_size: int = 5000, max_total_pairs: int = 3_000_000,
) -> pl.DataFrame:
    s1n = s1.select(["source1_entity_id", pl.col(name_col_s1).alias("name")]).filter(pl.col("name") != "")
    obn = ob.select(["candidate_id", "candidate_source", pl.col(name_col_ob).alias("name")]).filter(pl.col("name") != "")
    if s1n.height == 0 or obn.height == 0:
        return _empty_hits()

    def _grams_df(df: pl.DataFrame, id_cols: list, n_values: tuple) -> pl.DataFrame:
        parts = []
        for n in n_values:
            g = df.select(id_cols + [_extract_qgrams("name", n).alias("grams")]).explode("grams")
            parts.append(g.filter(pl.col("grams").is_not_null()))
        return pl.concat(parts, how="vertical_relaxed").unique()

    s1_grams = _grams_df(s1n, ["source1_entity_id"], (3, 4))
    ob_grams = _grams_df(obn, ["candidate_id", "candidate_source"], (3, 4))
    if s1_grams.height == 0 or ob_grams.height == 0:
        return _empty_hits()

    # IDF-like weighting + exclusion of near-universal grams, computed over
    # the corpus side (same spirit as _rare_token_block's doc-frequency ban,
    # applied to grams instead of whole words).
    n_ob_docs = obn.height
    gram_df = ob_grams.group_by("grams").agg(pl.col("candidate_id").n_unique().alias("df"))
    gram_df = gram_df.with_columns((pl.col("df") / n_ob_docs).alias("doc_freq_ratio"))
    gram_df = gram_df.filter(pl.col("doc_freq_ratio") <= max_doc_freq)
    if gram_df.height == 0:
        return _empty_hits()
    gram_df = gram_df.with_columns((-(pl.col("doc_freq_ratio").log())).alias("idf_weight"))
    gram_weights = gram_df.select(["grams", "idf_weight"])

    s1_grams = s1_grams.join(gram_weights, on="grams", how="inner")
    ob_grams = ob_grams.join(gram_weights, on="grams", how="inner")
    if s1_grams.height == 0 or ob_grams.height == 0:
        return _empty_hits()

    # Same two-layer join-size cap as _rare_token_block/_capped_join_block:
    # doc-freq filtering alone doesn't bound join size — a gram surviving the
    # doc-freq cutoff can still individually connect thousands of S1 rows to
    # thousands of corpus rows (small alphabet -> few possible 3/4-grams,
    # each shared by many names). This is what made the first version of this
    # channel hang past 120s on a mere 20K x 100K synthetic test.
    s1_counts = s1_grams.group_by("grams").len().rename({"len": "n_s1"})
    ob_counts = ob_grams.group_by("grams").len().rename({"len": "n_ob"})
    sizes = s1_counts.join(ob_counts, on="grams", how="inner")
    sizes = sizes.filter((pl.col("n_s1") <= max_block_size) & (pl.col("n_ob") <= max_block_size))
    sizes = sizes.with_columns((pl.col("n_s1") * pl.col("n_ob")).alias("pair_count"))
    sizes = sizes.sort("pair_count").with_columns(pl.col("pair_count").cum_sum().alias("cum_pairs"))
    safe_grams = sizes.filter(pl.col("cum_pairs") <= max_total_pairs).select("grams")
    if safe_grams.height == 0:
        return _empty_hits()

    s1_grams = s1_grams.join(safe_grams, on="grams", how="inner")
    ob_grams = ob_grams.join(safe_grams, on="grams", how="inner")

    joined = s1_grams.join(ob_grams, on="grams", how="inner")
    scored = joined.group_by(["source1_entity_id", "candidate_id"]).agg([
        pl.col("candidate_source").first(),
        pl.col("idf_weight").sum().alias("score"),
    ])

    scored = scored.sort(["source1_entity_id", "score"], descending=[False, True])
    top = scored.group_by("source1_entity_id", maintain_order=True).head(top_k)

    return top.select(["source1_entity_id", "candidate_id", "candidate_source"]).with_columns(
        pl.lit("char_qgram").alias("block_method")
    )


# ── Channels 4/5/6: TF-IDF cosine retrieval ────────────────────────────────

def _sparse_topk_pairs(query_matrix, corpus_matrix, top_k, chunk_size):
    """
    Sparse top-k cosine retrieval — genuinely sub-quadratic, unlike both
    sklearn's brute-force NearestNeighbors AND a naive chunked-dense-matmul
    approach (both tried first; both are O(n_query x n_corpus) regardless of
    implementation, which never finishes at India/US scale, millions x
    millions).

    The fix: `query @ corpus.T` between two SPARSE CSR matrices produces a
    SPARSE result via scipy — its cost is proportional to the number of
    shared nonzero dimensions actually present (i.e. query/corpus pairs that
    share >=1 vocabulary term), not the full n_query x n_corpus product. This
    is the standard inverted-index behavior for sparse vectors: two documents
    with zero terms in common contribute exactly zero compute, instead of
    being densified into an explicit zero. NEVER call .todense()/.toarray()
    on the similarity result — that's what silently turns this back into the
    same O(n^2) cost this function exists to avoid. This only works because
    the caller uses WORD-level TF-IDF (large, selective vocabulary keeps the
    result genuinely sparse) — char n-grams were measured at ~98% pairwise
    density for short business names, which defeats this entirely.

    Chunking the query side only bounds peak memory of intermediate CSR
    buffers, not compute — the sparse matmul itself is what makes this fast.

    Returns (row_positions, col_indices) arrays — flat, ragged-safe: each row
    may have fewer than top_k hits if fewer than top_k corpus docs share any
    n-gram with it at all (which is correct: there is no meaningful
    similarity to report for a pair with zero token overlap).
    """
    n_query = query_matrix.shape[0]

    query_norms = np.sqrt(np.asarray(query_matrix.multiply(query_matrix).sum(axis=1))).ravel()
    query_norms[query_norms == 0] = 1.0
    corpus_norms = np.sqrt(np.asarray(corpus_matrix.multiply(corpus_matrix).sum(axis=1))).ravel()
    corpus_norms[corpus_norms == 0] = 1.0

    corpus_T = corpus_matrix.T.tocsr()

    all_rows, all_cols = [], []
    n_chunks = max(1, (n_query + chunk_size - 1) // chunk_size)
    chunk_starts = tqdm(
        range(0, n_query, chunk_size), total=n_chunks, desc="      tfidf chunks", leave=False,
        mininterval=config.TQDM_MININTERVAL,
    ) if n_chunks > 1 else range(0, n_query, chunk_size)
    for start in chunk_starts:
        end = min(start + chunk_size, n_query)
        chunk = query_matrix[start:end]
        sims = chunk.dot(corpus_T).tocsr()  # SPARSE result — cost ~ shared n-grams only

        for local_row in range(sims.shape[0]):
            row_start, row_end = sims.indptr[local_row], sims.indptr[local_row + 1]
            if row_start == row_end:
                continue  # zero corpus docs share any n-gram with this query row
            cols = sims.indices[row_start:row_end]
            vals = sims.data[row_start:row_end] / (query_norms[start + local_row] * corpus_norms[cols])
            k = min(top_k, cols.shape[0])
            if k < cols.shape[0]:
                top_local = np.argpartition(-vals, k - 1)[:k]
                cols = cols[top_local]
            all_rows.append(np.full(cols.shape[0], start + local_row, dtype=np.int64))
            all_cols.append(cols)

    if not all_rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(all_rows), np.concatenate(all_cols)


def _tfidf_topk_block(
    s1_ids: list, s1_texts: list, ob_ids: list, ob_sources: list, ob_texts: list,
    top_k: int, method: str, chunk_size: int = TFIDF_QUERY_CHUNK_SIZE,
) -> pl.DataFrame:
    if not s1_texts or not ob_texts:
        return _empty_hits()

    # word-level, not char n-gram — see config.py's TFIDF_TOP_K/TFIDF_MAX_DF
    # comments for the full benchmark trail: char n-grams measured ~98%
    # pairwise density for short business names (O(n_query x n_corpus)
    # regardless of implementation, never finishes at India/US scale) AND no
    # recall benefit over word-level once real data was tested. Config A
    # (max_df=1.0, no common-term exclusion) tied for best measured recall
    # among 10 vocabulary configs tested via benchmark_tfidf_configs.py.
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 1),
        min_df=config.TFIDF_MIN_DF, max_df=config.TFIDF_MAX_DF,
        max_features=config.TFIDF_MAX_FEATURES, sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        dtype=np.float32,
    )
    try:
        ob_matrix = vectorizer.fit_transform(ob_texts).tocsr()
        s1_matrix = vectorizer.transform(s1_texts).tocsr()
    except ValueError:
        return _empty_hits()

    if ob_matrix.shape[0] == 0 or ob_matrix.nnz == 0 or s1_matrix.nnz == 0:
        return _empty_hits()

    row_pos, col_idx = _sparse_topk_pairs(s1_matrix, ob_matrix, top_k, chunk_size)
    if row_pos.shape[0] == 0:
        return _empty_hits()

    ob_ids_arr = np.asarray(ob_ids, dtype=object)
    ob_sources_arr = np.asarray(ob_sources, dtype=object)
    s1_ids_arr = np.asarray(s1_ids, dtype=object)

    rows_s1 = s1_ids_arr[row_pos]
    rows_cand = ob_ids_arr[col_idx]
    rows_src = ob_sources_arr[col_idx]

    return pl.DataFrame({
        "source1_entity_id": rows_s1, "candidate_id": rows_cand, "candidate_source": rows_src,
    }).unique().with_columns(pl.lit(method).alias("block_method"))


# ── Channel 7/8: postal / numeric exact blocks ─────────────────────────────

def _postal_block(s1: pl.DataFrame, ob: pl.DataFrame, max_block_size: int, max_total_pairs: int) -> pl.DataFrame:
    return _capped_join_block(s1, ob, ["postal_code"], "postal", max_block_size, max_total_pairs)


def _numeric_block(s1: pl.DataFrame, ob: pl.DataFrame, max_block_size: int, max_total_pairs: int) -> pl.DataFrame:
    return _capped_join_block(
        s1, ob, ["house_number", "locality"], "numeric", max_block_size, max_total_pairs,
    )


# ── Channel 9: multilingual embedding ANN ──────────────────────────────────

def _embedding_block(
    s1_ids: list, s1_texts: list, ob_ids: list, ob_sources: list, ob_texts: list, top_k: int,
) -> pl.DataFrame:
    if not s1_texts or not ob_texts:
        return _empty_hits()

    import faiss

    tqdm.write("      [embedding] loading model (first call downloads weights)...")
    model = _get_embedding_model()

    tqdm.write(f"      [embedding] encoding {len(ob_texts):,} corpus texts...")
    ob_emb = model.encode(
        ob_texts, batch_size=config.EMBEDDING_BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")
    tqdm.write(f"      [embedding] encoding {len(s1_texts):,} query texts...")
    s1_emb = model.encode(
        s1_texts, batch_size=config.EMBEDDING_BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")

    tqdm.write("      [embedding] building FAISS index and searching...")
    faiss.omp_set_num_threads(8)  # explicit cap — see the module-level OMP_NUM_THREADS comment
    dim = ob_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(ob_emb)
    k = min(top_k, ob_emb.shape[0])
    _, indices = index.search(s1_emb, k)

    ob_ids_arr = np.asarray(ob_ids, dtype=object)
    ob_sources_arr = np.asarray(ob_sources, dtype=object)
    s1_ids_arr = np.asarray(s1_ids, dtype=object)

    valid = indices >= 0
    row_pos = np.repeat(np.arange(len(s1_ids_arr)), indices.shape[1])[valid.ravel()]
    col_idx = indices.ravel()[valid.ravel()]

    rows_s1 = s1_ids_arr[row_pos]
    rows_cand = ob_ids_arr[col_idx]
    rows_src = ob_sources_arr[col_idx]

    return pl.DataFrame({
        "source1_entity_id": rows_s1, "candidate_id": rows_cand, "candidate_source": rows_src,
    }).unique().with_columns(pl.lit("embedding").alias("block_method"))


# ── Orchestration ──────────────────────────────────────────────────────────

def generate_candidates(
    s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame,
    top_n: int, max_block_size: int = 5000, max_total_pairs: int = 3_000_000,
    use_embedding: bool = True, channels: list = None,
) -> pl.DataFrame:
    """
    Returns [source1_entity_id, candidate_id, candidate_source,
    num_blocks_agreeing, block_exact_match, block_methods] — up to top_n
    candidates per S1, ranked by exact-match first, then by how many
    independent channels agreed on the candidate.

    `channels`: optional subset of _ALL_CHANNELS to run (used by the
    blocking benchmark to measure per-channel and incremental recall).
    """
    active_channels = channels if channels is not None else list(_ALL_CHANNELS)

    others = pl.concat(
        [s2.with_columns(pl.lit("S2").alias("src")), s3.with_columns(pl.lit("S3").alias("src"))],
        how="vertical_relaxed",
    )

    s1p = s1.select([
        pl.col("entity_id").alias("source1_entity_id"), "country",
        pl.col("normalized_name").alias("s1_name"),
        pl.col("core_name").alias("s1_core_name"),
        pl.col("normalized_address").alias("s1_addr"),
        "house_number", "locality", "postal_code",
    ])
    obp = others.select([
        pl.col("entity_id").alias("candidate_id"), "country",
        pl.col("normalized_name").alias("cand_name"),
        pl.col("core_name").alias("cand_core_name"),
        pl.col("normalized_address").alias("cand_addr"),
        pl.col("src").alias("candidate_source"),
        "house_number", "locality", "postal_code",
    ])

    results = []
    countries = s1p["country"].unique().to_list()
    for country in tqdm(countries, desc="blocking by country", mininterval=config.TQDM_MININTERVAL):
        s1c = s1p.filter(pl.col("country") == country)
        obc = obp.filter(pl.col("country") == country)
        if s1c.height == 0 or obc.height == 0:
            continue
        tqdm.write(f"  country={country}: {s1c.height:,} S1 x {obc.height:,} candidates")

        if "exact_name" in active_channels:
            hit = _run_channel(
                "exact_name", _exact_match_block,
                s1c.with_columns(pl.col("s1_name").alias("key")),
                obc.with_columns(pl.col("cand_name").alias("key")),
                "key", "exact_name",
            )
            if hit.height:
                results.append(hit)

        if "suffix_normalized" in active_channels:
            hit = _run_channel(
                "suffix_normalized", _exact_match_block,
                s1c.with_columns(pl.col("s1_core_name").alias("key")),
                obc.with_columns(pl.col("cand_core_name").alias("key")),
                "key", "suffix_normalized",
            )
            if hit.height:
                results.append(hit)

        if "rare_token" in active_channels:
            hit = _run_channel("rare_token", _rare_token_block, s1c, obc, max_block_size, max_total_pairs)
            if hit.height:
                results.append(hit)

        if "char_qgram" in active_channels:
            hit = _run_channel(
                "char_qgram", _char_qgram_block, s1c, obc, "s1_name", "cand_name",
                config.CHAR_QGRAM_TOP_K, config.CHAR_QGRAM_MAX_DOC_FREQ,
            )
            if hit.height:
                results.append(hit)

        if "postal" in active_channels:
            hit = _run_channel("postal", _postal_block, s1c, obc, max_block_size, max_total_pairs)
            if hit.height:
                results.append(hit)

        if "numeric" in active_channels:
            hit = _run_channel("numeric", _numeric_block, s1c, obc, max_block_size, max_total_pairs)
            if hit.height:
                results.append(hit)

        if "name_tfidf" in active_channels:
            hit = _run_channel(
                "name_tfidf", _tfidf_topk_block,
                s1c["source1_entity_id"].to_list(), s1c["s1_name"].to_list(),
                obc["candidate_id"].to_list(), obc["candidate_source"].to_list(), obc["cand_name"].to_list(),
                config.TFIDF_TOP_K, "name_tfidf",
            )
            if hit.height:
                results.append(hit)

        if "address_tfidf" in active_channels:
            hit = _run_channel(
                "address_tfidf", _tfidf_topk_block,
                s1c["source1_entity_id"].to_list(), s1c["s1_addr"].to_list(),
                obc["candidate_id"].to_list(), obc["candidate_source"].to_list(), obc["cand_addr"].to_list(),
                config.TFIDF_TOP_K, "address_tfidf",
            )
            if hit.height:
                results.append(hit)

        if "composite_tfidf" in active_channels:
            s1_comp = (s1c["s1_name"] + " " + s1c["s1_addr"]).to_list()
            ob_comp = (obc["cand_name"] + " " + obc["cand_addr"]).to_list()
            hit = _run_channel(
                "composite_tfidf", _tfidf_topk_block,
                s1c["source1_entity_id"].to_list(), s1_comp,
                obc["candidate_id"].to_list(), obc["candidate_source"].to_list(), ob_comp,
                config.TFIDF_TOP_K, "composite_tfidf",
            )
            if hit.height:
                results.append(hit)

        if "embedding" in active_channels and use_embedding:
            partition_rows = s1c.height + obc.height
            if partition_rows <= config.EMBEDDING_MAX_PARTITION_ROWS:
                s1_comp = (s1c["s1_name"] + " " + s1c["s1_addr"]).to_list()
                ob_comp = (obc["cand_name"] + " " + obc["cand_addr"]).to_list()
                hit = _run_channel(
                    "embedding", _embedding_block,
                    s1c["source1_entity_id"].to_list(), s1_comp,
                    obc["candidate_id"].to_list(), obc["candidate_source"].to_list(), ob_comp,
                    config.EMBEDDING_TOP_K,
                )
                if hit.height:
                    results.append(hit)
            else:
                tqdm.write(
                    f"  country={country}: skipping embedding channel "
                    f"({partition_rows:,} rows > EMBEDDING_MAX_PARTITION_ROWS)"
                )

    if not results:
        return pl.DataFrame(schema={
            "source1_entity_id": pl.Utf8, "candidate_id": pl.Utf8, "candidate_source": pl.Utf8,
            "num_blocks_agreeing": pl.Int64, "block_exact_match": pl.Boolean, "block_methods": pl.Utf8,
        })

    unioned = pl.concat(results, how="vertical_relaxed")

    agg = unioned.group_by(["source1_entity_id", "candidate_id"]).agg([
        pl.col("candidate_source").first(),
        pl.col("block_method").unique().alias("_methods"),
        pl.col("block_method").n_unique().alias("num_blocks_agreeing"),
        (pl.col("block_method") == "exact_name").any().alias("block_exact_match"),
    ])
    agg = agg.with_columns(pl.col("_methods").list.sort().list.join(",").alias("block_methods")).drop("_methods")

    agg = agg.sort(
        ["source1_entity_id", "block_exact_match", "num_blocks_agreeing"],
        descending=[False, True, True],
    )
    agg = agg.group_by("source1_entity_id", maintain_order=True).head(top_n)

    return agg.select([
        "source1_entity_id", "candidate_id", "candidate_source",
        "num_blocks_agreeing", "block_exact_match", "block_methods",
    ])


# ── Residual embedding rescue ───────────────────────────────────────────────
#
# The embedding channel dominates runtime: measured directly on a 35K-S1
# sample, it took 551s vs. single-digit-seconds-to-low-minutes for every
# other channel combined, and its cost scales ~linearly with (n_s1 + n_ob)
# text-encoding volume. At full country scale that extrapolates to ~59 hours
# PER COUNTRY, infeasible against a 2-day deadline. But embedding is also the
# only channel that generalizes across script/language boundaries (Devanagari
# vs. Latin) without relying on shared tokens at all — so dropping it
# entirely risks losing exactly the hardest, most script-divergent true
# matches the other 8 (all lexical/token-based) channels structurally cannot
# find. Rescue policy: run the 8 cheap channels for EVERY S1 (fast), then run
# embedding only for the subset of S1 entities the cheap channels served
# poorly — a small fraction of the full corpus, not all of it.

_RESCUE_POLICIES = {"zero_candidates", "low_count", "low_score", "low_agreement"}


def _select_rescue_ids(
    cheap_candidates: pl.DataFrame, all_s1_ids: list, policy: str,
    min_count_threshold: int = 5, min_agreement_threshold: int = 2,
) -> set:
    """
    Returns the subset of all_s1_ids that should be sent to embedding rescue,
    per the given policy:
      - "zero_candidates": only S1s with NO cheap candidates at all (cheapest,
        most conservative — misses S1s that got a few bad candidates but no
        good one)
      - "low_count": S1s with fewer than min_count_threshold cheap candidates
        (covers zero_candidates plus thin coverage)
      - "low_agreement": S1s whose best cheap candidate has fewer than
        min_agreement_threshold independent channels agreeing on it (weak
        lexical consensus, even if candidate count is nominally high)
    """
    all_ids = set(all_s1_ids)
    if cheap_candidates.height == 0:
        return all_ids  # nothing cheap found anything; rescue everyone

    if policy == "zero_candidates":
        has_candidates = set(cheap_candidates["source1_entity_id"].to_list())
        return all_ids - has_candidates

    if policy == "low_count":
        counts = cheap_candidates.group_by("source1_entity_id").len()
        thin = set(counts.filter(pl.col("len") < min_count_threshold)["source1_entity_id"].to_list())
        has_candidates = set(counts["source1_entity_id"].to_list())
        return (all_ids - has_candidates) | thin

    if policy == "low_agreement":
        best_agreement = cheap_candidates.group_by("source1_entity_id").agg(
            pl.col("num_blocks_agreeing").max().alias("best_agreement")
        )
        weak = set(
            best_agreement.filter(pl.col("best_agreement") < min_agreement_threshold)["source1_entity_id"].to_list()
        )
        has_candidates = set(best_agreement["source1_entity_id"].to_list())
        return (all_ids - has_candidates) | weak

    raise ValueError(f"unknown rescue policy: {policy!r} (expected one of {_RESCUE_POLICIES})")


def generate_candidates_with_rescue(
    s1: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame,
    top_n: int, max_block_size: int = 5000, max_total_pairs: int = 3_000_000,
    rescue_policy: str = "low_count", min_count_threshold: int = 5, min_agreement_threshold: int = 2,
    cheap_candidates: pl.DataFrame = None,
) -> tuple:
    """
    Two-stage candidate generation:
      Stage 1: all 8 cheap (non-embedding) channels, for every S1 entity.
      Stage 2: embedding retrieval ONLY for S1 entities the rescue policy
               flags as poorly served by stage 1.

    `cheap_candidates`: pass a previously-computed cheap-channel result
    (e.g. from a prior generate_candidates(channels=cheap_channels) call) to
    skip recomputing stage 1 — the 8 cheap channels don't depend on the
    rescue policy at all, so comparing multiple policies on the same S1/S2/S3
    input should compute them ONCE and reuse, not once per policy (the 3
    TF-IDF channels alone measured 5-11 min each at 35K-S1 sample scale).

    Returns (final_candidates, rescue_stats) — rescue_stats has n_rescued,
    rescue_fraction, and n_total_s1 for reporting.
    """
    if cheap_candidates is None:
        cheap_channels = [c for c in _ALL_CHANNELS if c != "embedding"]
        cheap_candidates = generate_candidates(
            s1, s2, s3, top_n=top_n, max_block_size=max_block_size, max_total_pairs=max_total_pairs,
            channels=cheap_channels,
        )

    all_s1_ids = s1["entity_id"].to_list()
    rescue_ids = _select_rescue_ids(
        cheap_candidates, all_s1_ids, rescue_policy, min_count_threshold, min_agreement_threshold,
    )
    tqdm.write(
        f"  rescue policy={rescue_policy}: {len(rescue_ids):,}/{len(all_s1_ids):,} S1 entities "
        f"({100*len(rescue_ids)/max(len(all_s1_ids),1):.1f}%) sent to embedding"
    )

    rescue_stats = {
        "n_total_s1": len(all_s1_ids), "n_rescued": len(rescue_ids),
        "rescue_fraction": len(rescue_ids) / max(len(all_s1_ids), 1),
    }

    if not rescue_ids:
        return cheap_candidates, rescue_stats

    s1_rescue = s1.filter(pl.col("entity_id").is_in(list(rescue_ids)))
    embedding_candidates = generate_candidates(
        s1_rescue, s2, s3, top_n=top_n, max_block_size=max_block_size, max_total_pairs=max_total_pairs,
        channels=["embedding"],
    )

    if embedding_candidates.height == 0:
        return cheap_candidates, rescue_stats

    if cheap_candidates.height == 0:
        return embedding_candidates, rescue_stats

    # Re-explode each side's already-joined block_methods string back into
    # individual (S1, candidate, method) rows and re-aggregate from scratch —
    # simpler and less error-prone than trying to merge two already-aggregated
    # frames (a pair found by both a cheap channel AND embedding, though rare,
    # needs its methods unioned and its agreement count recomputed correctly).
    def _reexplode(df: pl.DataFrame) -> pl.DataFrame:
        return df.select([
            "source1_entity_id", "candidate_id", "candidate_source",
            pl.col("block_methods").str.split(",").alias("block_method"),
        ]).explode("block_method")

    unioned = pl.concat([_reexplode(cheap_candidates), _reexplode(embedding_candidates)], how="vertical_relaxed")

    agg = unioned.group_by(["source1_entity_id", "candidate_id"]).agg([
        pl.col("candidate_source").first(),
        pl.col("block_method").unique().alias("_methods"),
        pl.col("block_method").n_unique().alias("num_blocks_agreeing"),
        (pl.col("block_method") == "exact_name").any().alias("block_exact_match"),
    ])
    agg = agg.with_columns(pl.col("_methods").list.sort().list.join(",").alias("block_methods")).drop("_methods")

    agg = agg.sort(
        ["source1_entity_id", "block_exact_match", "num_blocks_agreeing"],
        descending=[False, True, True],
    )
    agg = agg.group_by("source1_entity_id", maintain_order=True).head(top_n)

    final = agg.select([
        "source1_entity_id", "candidate_id", "candidate_source",
        "num_blocks_agreeing", "block_exact_match", "block_methods",
    ])
    return final, rescue_stats
