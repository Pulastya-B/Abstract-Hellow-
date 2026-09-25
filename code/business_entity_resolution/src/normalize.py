"""
v1 normalization: NFKC/casefold, punctuation strip, legal-suffix strip.

Deliberately simplified vs. the full plan.md design (§2) to get a trainable
pipeline running fast: no Devanagari transliteration, no postal-code/landmark
extraction, no address-abbreviation dictionary yet. Those are additive
improvements to layer in once this baseline is training and validating.
"""

import re
import unicodedata

import polars as pl
from tqdm import tqdm

import config

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt ltd.", "private ltd",
    "limited", "ltd", "llc", "llp", "inc", "incorporated", "corporation",
    "corp", "co", "company", "sarl", "sas", "sa",
]

_WS_RE = re.compile(r"\s+")

# Combining-mark categories (matras, virama, anusvara, ...) that Python's
# stdlib \w does NOT include even under re.UNICODE. A regex punctuation
# strip using [^\w\s] therefore treats these as punctuation and blanks them
# out — for Devanagari (and other Indic scripts built from base consonants +
# combining vowel signs), this shatters every word into its bare consonant
# skeleton, e.g. "मार्केटिंग" -> "म र क ट ग". Confirmed directly against
# India's real S2 data: this was silently destroying the vocabulary that
# name_tfidf/rare_token/exact_name all depend on, collapsing tens of
# thousands of Devanagari business names into single-character tokens (a
# handful of common consonants), which is what made word-level TF-IDF
# density spike on India specifically (measured 24.7% vs. ~0.2% on ASCII-only
# synthetic test data) and the whole channel take 24+ hours instead of
# minutes. `unicodedata.category(ch) in ("Mn", "Mc")` explicitly preserves
# these mark characters instead of stripping them.
_COMBINING_MARK_CATEGORIES = frozenset({"Mn", "Mc"})
_SUFFIX_PATTERNS = [re.compile(r"\b" + re.escape(s) + r"\b") for s in LEGAL_SUFFIXES]

# House/building number: leading digit run at the start of the address, or
# right after a comma-separated segment boundary (covers "123 Main St" and
# "Flat 4B, Main St" style leading numeric/alnum unit markers).
_HOUSE_NUM_RE = re.compile(r"(?:^|\b)(\d{1,6}[a-z]?)\b")

# Postal code patterns, applied per-country. Kept permissive (extraction, not
# validation) since business_address is free text with inconsistent formatting.
_POSTAL_PATTERNS = {
    "US": re.compile(r"\b(\d{5})(?:-\d{4})?\b"),
    "India": re.compile(r"\b(\d{6})\b"),
    "France": re.compile(r"\b(\d{5})\b"),
}
_POSTAL_FALLBACK_RE = re.compile(r"\b(\d{5,6})\b")


def _strip_punct_unicode_safe(s: str) -> str:
    """
    Replaces punctuation with a space while preserving combining marks (see
    _COMBINING_MARK_CATEGORIES above) — a per-character categorize-and-filter
    pass instead of a [^\\w\\s] regex, since stdlib \\w silently excludes
    Devanagari matras/virama and would otherwise shatter Indic-script words.
    """
    out = []
    for ch in s:
        if ch.isalnum() or ch.isspace() or unicodedata.category(ch) in _COMBINING_MARK_CATEGORIES:
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def _normalize_str(s) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = s.replace("&", " and ")
    s = _strip_punct_unicode_safe(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _strip_suffixes(s: str) -> str:
    for pattern in _SUFFIX_PATTERNS:
        s = pattern.sub("", s)
    return _WS_RE.sub(" ", s).strip()


def _extract_house_number(addr: str) -> str:
    m = _HOUSE_NUM_RE.search(addr)
    return m.group(1) if m else ""


def _extract_postal(addr: str, country: str) -> str:
    pattern = _POSTAL_PATTERNS.get(country, _POSTAL_FALLBACK_RE)
    m = pattern.search(addr)
    if m:
        return m.group(1)
    m = _POSTAL_FALLBACK_RE.search(addr)
    return m.group(1) if m else ""


def _extract_locality(addr: str) -> str:
    """
    Second-to-last non-numeric comma-separated segment — cheap, country-
    agnostic proxy for CITY, same rule for every country so France needs no
    special-casing.

    Originally took the LAST non-numeric segment, which for addresses
    formatted "..., City, State" (extremely common in the India partition)
    grabbed the STATE, not the city — a handful of state names shared by
    hundreds of thousands of records. That collapsed _numeric_block's
    (house_number, locality) join key onto a few giant buckets and produced
    722M candidate rows from a single channel on India alone (blocking.py's
    per-channel timing log caught this directly). City is far more
    discriminative than state/region, so preferring the second-to-last
    segment fixes the join fan-out at the source; _numeric_block's own
    per-group/cumulative pair-count caps (mirroring _rare_token_block's)
    are the second, independent line of defense in case any single
    locality value is still too common for a given country's data.
    """
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    non_numeric = [p for p in parts if not _WS_RE.sub("", p).isdigit()]
    if len(non_numeric) >= 2:
        return non_numeric[-2]
    if non_numeric:
        return non_numeric[-1]
    return ""


def normalize_df(df: pl.DataFrame, label: str = "") -> pl.DataFrame:
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    countries = df["country"].to_list()
    tag = f"[{label}] " if label else ""

    tq = dict(mininterval=config.TQDM_MININTERVAL)
    norm_names = [_normalize_str(n) for n in tqdm(names, desc=f"{tag}normalize names", **tq)]
    core_names = [_strip_suffixes(n) for n in tqdm(norm_names, desc=f"{tag}strip suffixes", **tq)]

    # locality/house-number extraction run on the RAW address (before punctuation
    # is stripped) since comma boundaries and digit adjacency carry structure
    # that normalization would otherwise destroy.
    raw_addrs = [str(a) if a is not None else "" for a in addrs]
    localities = [_extract_locality(a) for a in tqdm(raw_addrs, desc=f"{tag}extract locality", **tq)]
    house_nums = [_extract_house_number(a) for a in tqdm(raw_addrs, desc=f"{tag}extract house#", **tq)]
    postals = [
        _extract_postal(a, c) for a, c in tqdm(zip(raw_addrs, countries), total=len(raw_addrs),
                                                 desc=f"{tag}extract postal", **tq)
    ]

    norm_addrs = [_normalize_str(a) for a in tqdm(addrs, desc=f"{tag}normalize addresses", **tq)]
    norm_localities = [_normalize_str(loc) for loc in tqdm(localities, desc=f"{tag}normalize locality", **tq)]

    return df.with_columns([
        pl.Series("normalized_name", norm_names),
        pl.Series("core_name", core_names),
        pl.Series("normalized_address", norm_addrs),
        pl.Series("locality", norm_localities),
        pl.Series("house_number", house_nums),
        pl.Series("postal_code", postals),
        (
            pl.col("business_address").is_null()
            | (pl.col("business_address").cast(pl.Utf8) == "")
        ).alias("address_missing"),
    ])
