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

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt ltd.", "private ltd",
    "limited", "ltd", "llc", "llp", "inc", "incorporated", "corporation",
    "corp", "co", "company", "sarl", "sas", "sa",
]

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_SUFFIX_PATTERNS = [re.compile(r"\b" + re.escape(s) + r"\b") for s in LEGAL_SUFFIXES]


def _normalize_str(s) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _strip_suffixes(s: str) -> str:
    for pattern in _SUFFIX_PATTERNS:
        s = pattern.sub("", s)
    return _WS_RE.sub(" ", s).strip()


def normalize_df(df: pl.DataFrame) -> pl.DataFrame:
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()

    norm_names = [_normalize_str(n) for n in names]
    core_names = [_strip_suffixes(n) for n in norm_names]
    norm_addrs = [_normalize_str(a) for a in addrs]

    return df.with_columns([
        pl.Series("normalized_name", norm_names),
        pl.Series("core_name", core_names),
        pl.Series("normalized_address", norm_addrs),
        (
            pl.col("business_address").is_null()
            | (pl.col("business_address").cast(pl.Utf8) == "")
        ).alias("address_missing"),
    ])
