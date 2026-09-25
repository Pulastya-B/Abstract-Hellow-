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

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
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
    """Last non-numeric comma-separated segment — cheap, country-agnostic
    proxy for city/region, same rule for every country so France needs no
    special-casing."""
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    for part in reversed(parts):
        stripped = _WS_RE.sub("", part)
        if stripped and not stripped.isdigit():
            return part
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
