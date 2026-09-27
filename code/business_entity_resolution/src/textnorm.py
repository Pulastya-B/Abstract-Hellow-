"""
Script-independent name/address normalization, standard library only (no external services).

Why: in India ~10-20% of S2/S3 names are in an Indian script while S1 names are Latin, so
`Arihant Software` vs `अरिहंत सॉफ्टवेयर` scored a name similarity of 6, indistinguishable
from a wrong match. `indic_to_latin` transliterates every Brahmic script in U+0900-U+0DFF
(Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) using the
character names in Python's own `unicodedata` tables, which follow one pattern for all of them
("TELUGU LETTER KA", "DEVANAGARI VOWEL SIGN AA", "KANNADA SIGN VIRAMA"). `skeleton` then reduces
both spellings to a consonant skeleton, so transliteration choices (v/w, ph/f, sh/s, vowel
length) stop mattering: `స్వస్తిక్` -> "svastik" -> "svstk" and `Swastik` -> "svstk".
"""

import re
import unicodedata
from functools import lru_cache

_VOWELS = {
    "A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "E": "e", "EE": "e", "AI": "ai",
    "O": "o", "OO": "o", "AU": "au", "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li",
    "VOCALIC LL": "li", "CANDRA E": "e", "CANDRA O": "o", "SHORT E": "e", "SHORT O": "o", "SHORT A": "a",
    "CANDRA A": "a", "OE": "o", "OOE": "o", "UE": "u", "UUE": "u", "AW": "au", "PRISHTHAMATRA E": "e",
}
# retroflex / script-specific consonants folded to the Latin spelling Indian businesses use
_CONS = {
    "c": "ch", "ch": "chh", "tt": "t", "tth": "th", "dd": "d", "ddh": "dh", "nn": "n", "nnn": "n",
    "ss": "sh", "ny": "n", "ng": "n", "ll": "l", "lll": "l", "rr": "r", "q": "k", "khh": "kh",
    "ghh": "gh", "dddh": "r", "yy": "y", "jnya": "gy",
}
_NO_CASE = {"‌", "‍"}


def _is_indic(ch):
    return "ऀ" <= ch <= "෿"


def has_indic(s):
    return any(_is_indic(ch) for ch in s)


def indic_to_latin(s):
    """Transliterate Brahmic-script text to plain Latin; non-Indic characters pass through."""
    if not s or not has_indic(s):
        return s
    out = []
    pending = False  # a consonant's inherent "a", emitted only if no vowel sign / virama follows

    def flush():
        nonlocal pending
        if pending:
            out.append("a")
            pending = False

    for ch in s:
        if not _is_indic(ch):
            if ch in _NO_CASE:
                continue
            pending = False  # word-final inherent vowel is silent (राम -> ram)
            out.append(ch)
            continue
        name = unicodedata.name(ch, "")
        rest = name.split(" ", 1)[1] if " " in name else ""
        if rest.startswith("LETTER CHILLU "):
            flush()
            out.append(rest[len("LETTER CHILLU "):].lower())
        elif rest.startswith("LETTER "):
            key = rest[len("LETTER "):]
            if key in _VOWELS:
                flush()
                out.append(_VOWELS[key])
            else:
                flush()
                base = key.split()[-1].lower()
                if base.endswith("a") and len(base) > 1:
                    base = base[:-1]
                out.append(_CONS.get(base, base))
                pending = True
        elif rest.startswith("VOWEL SIGN "):
            pending = False
            out.append(_VOWELS.get(rest[len("VOWEL SIGN "):], ""))
        elif rest.startswith("SIGN VIRAMA"):
            pending = False
        elif rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU"):
            flush()
            out.append("n")
        elif rest == "SIGN VISARGA":
            flush()
            out.append("h")
        elif rest.startswith("DIGIT "):
            flush()
            out.append(str(unicodedata.digit(ch, 0)))
        # nukta, length marks, dandas, avagraha: no sound of their own
    return "".join(out)


_LATIN_MARKS = re.compile(r"[\u0300-\u036f]")


def strip_latin_accents(s):
    """'ÀMICALE' -> 'AMICALE', 'Límited' -> 'Limited'. Only Latin combining marks are removed, so Indian
    scripts (whose vowel signs are also combining marks) keep their spelling."""
    return _LATIN_MARKS.sub("", unicodedata.normalize("NFKD", s)) if s else s


_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_REPEAT = re.compile(r"(.)\1+")
_FOLD = (("ph", "f"), ("w", "v"), ("ck", "k"), ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "j"),
         ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("gh", "g"), ("kh", "k"), ("jh", "j"), ("y", "i"))


def ascii_fold(s):
    s = unicodedata.normalize("NFKD", indic_to_latin(s).lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def skeleton(s):
    """Consonant skeleton per word (first letter kept): robust to transliteration and vowel spelling."""
    s = _NON_ALNUM.sub(" ", ascii_fold(s))
    for a, b in _FOLD:
        s = s.replace(a, b)
    words = []
    for w in s.split():
        w = _REPEAT.sub(r"\1", w)
        words.append(w[0] + re.sub(r"[aeiouh]", "", w[1:]))
    return " ".join(words)


# ── names: core tokens ───────────────────────────────────────────────────────
#
# The error analysis showed two failure modes that whole-string similarity can't see:
#  - true matches with name noise: "alampataanimal.com" vs "Alampata Animal Pvt. Ltd.",
#    "YB Sreivsesmrvtae Ltd" vs "YB Servicesprivate Ltd", "[CENTER]", "M/s", "Private Private"
#  - sister companies at the same address: "Solutions Mindtrack Horologicals Overseas [LLP]" vs
#    "... Horologicals LLP" (token_set_ratio 100, but "Overseas" says it's a different business)
# So names are reduced to their distinctive core tokens, matched word by word (typo- and
# shuffle-tolerant), and whatever is left UNMATCHED on either side becomes the signal.

_FILLER = {
    "pvt", "private", "ltd", "limited", "llp", "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "cie", "ei", "center", "centre", "mr", "mrs", "ms", "m", "s",
    "the", "and", "of", "www", "com", "net", "org", "india", "france", "usa", "us",
}
_DOMAIN = re.compile(r"\.(?:co\.in|com|in|net|org|biz|info|fr|us)\b")
_DOTTED = re.compile(r"(?<![a-z0-9])[a-z](?:\.[a-z])+\.?(?![a-z0-9])")


def _undot_legal(s):
    """'e.u.r.l.' -> 'eurl', 's.a.s' -> 'sas'; other dotted initials ('r.k.') are left alone."""
    def fix(m):
        t = m.group().replace(".", "")
        return t if t in _FILLER else m.group()
    return _DOTTED.sub(fix, s)


def name_core(name):
    """Distinctive name tokens, in order, deduplicated: legal forms, fillers, domains and numbers removed."""
    s = _DOMAIN.sub(" ", _undot_legal(ascii_fold(name)).replace("www.", " "))
    seen, out = set(), []
    for t in _NON_ALNUM.sub(" ", s).split():
        if t not in _FILLER and not t.isdigit() and t not in seen:
            seen.add(t)
            out.append(t)
    return out


@lru_cache(maxsize=1_000_000)
def _word_skeleton(t):
    return skeleton(t)


def _tok_match(a, b, ratio):
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    if ratio(a, b) >= 85 or ratio("".join(sorted(a)), "".join(sorted(b))) >= 90:  # typo / letter shuffle
        return True
    # transliterated English words ("kanstrakshans" vs "constructions", "phud" vs "food")
    sa, sb = _word_skeleton(a), _word_skeleton(b)
    return min(len(sa), len(sb)) >= 3 and ratio(sa, sb) >= 80


def core_extras(qa, sb, ratio):
    """-> (share of record tokens matched, unmatched record tokens, unmatched S1 tokens)."""
    mq = [any(_tok_match(t, u, ratio) for u in sb) for t in qa]
    xs = [u for u in sb if not any(_tok_match(u, t, ratio) for t in qa)]
    return sum(mq) / len(qa), [t for t, m in zip(qa, mq) if not m], xs


def core_compare(qa, sb, idf, default_idf, ratio, partial):
    """
    -> ((unmatched record tokens, unmatched S1 tokens, max rarity of each unmatched side,
         share of record tokens matched, similarity of the space-less cores), xq, xs).
    Rarity is IDF over S1 names: an unmatched rare word ("overseas") means a different business far more
    than an unmatched common one ("traders"). The space-less comparison catches glued names
    ("alampataanimal" vs "alampata animal").
    """
    if not qa or not sb:
        return (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0), [], []
    frac, xq, xs = core_extras(qa, sb, ratio)
    return (float(len(xq)), float(len(xs)),
            max((idf.get(t, default_idf) for t in xq), default=0.0),
            max((idf.get(u, default_idf) for u in xs), default=0.0),
            frac, float(partial("".join(qa), "".join(sb)))), xq, xs


def noise_evidence(xq, xs, vocab):
    """
    Learned from train true pairs (fast_match --stage vocab): how often the dataset's noise ADDS each
    word to a record name ("smt": 100%, "praivet": 100%) or DROPS it from the S1 name ("group": 35%).
    An unmatched word that noise often adds is harmless; one it never adds ("overseas") signals a
    different business. -> (min add-rate of record extras, sum of (1 - add-rate), same for S1 extras
    with drop-rates); -1 / 0 when a side has no extras. Unseen words count as rate 0.
    """
    added, dropped = vocab.get("added", {}), vocab.get("dropped", {})
    aq = [added.get(t, 0.0) for t in xq]
    ds = [dropped.get(u, 0.0) for u in xs]
    return (min(aq) if aq else -1.0, sum(1 - r for r in aq),
            min(ds) if ds else -1.0, sum(1 - r for r in ds))


_WORDS = re.compile(r"\w+")


def has_repeated_word(name):
    w = _WORDS.findall(name.lower())
    return any(a == b for a, b in zip(w, w[1:]))


# Legal form as a class. Measured on train India: a true record keeps its S1's class 92.1% of the time
# (the common change is "Private Limited" -> "Limited"), a sibling business at the same address only 74.3%.
_LEGAL_TOK = {
    "llp": "llp", "llc": "llc", "inc": "inc", "incorporated": "inc", "pvt": "pvt", "private": "pvt", "pvtltd": "pvt",
    "ltd": "ltd", "limited": "ltd", "corp": "corp", "corporation": "corp", "sarl": "sarl", "sasu": "sasu",
    "sas": "sas", "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc", "co": "co", "company": "co", "cie": "co",
    # transliterated from Indian scripts (see the noise vocabulary: "praivet", "praibhet", "piraivet")
    "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "praivat": "pvt", "limitet": "ltd", "limited": "ltd",
}
_FILLER.update(_LEGAL_TOK)
_LEGAL_ORDER = ["llp", "llc", "inc", "pvt", "ltd", "corp", "sarl", "sasu", "sas", "eurl", "sa", "sci", "snc", "co"]


def legal_form(name):
    """Legal-form class of a business name ('pvt', 'llp', 'sarl', ...), '' if none."""
    found = {_LEGAL_TOK[t] for t in _NON_ALNUM.sub(" ", _undot_legal(ascii_fold(name))).split() if t in _LEGAL_TOK}
    return next((c for c in _LEGAL_ORDER if c in found), "")


# ── numbers ──────────────────────────────────────────────────────────────────
#
# Unowned records are mostly generated sibling businesses: same street as a real S1 entity, house number
# CHANGED (582 -> 595). Measured on train India, same street: a number substituted in 70.6% of siblings vs
# 1.2% of true pairs, whose number noise is dropping/adding digits or whole numbers (1093 vs 01093, 13 vs 9-13).

_NUM = re.compile(r"\d+")


def numbers(addr):
    """Numbers in an address, leading zeros dropped, Indian-script digits folded to ASCII."""
    return [n.lstrip("0") or "0" for n in _NUM.findall(ascii_fold(addr))] if addr else []


def number_compare(a, b):
    """-> (record numbers unrelated to every S1 number, the same from the S1 side, presence 0-3).
    'Related' = equal or one contains the other, which is how the noise drops/adds digits."""
    presence = float((1 if a else 0) + (2 if b else 0))
    if not a or not b:
        return -1.0, -1.0, presence
    ca = sum(1 for x in set(a) if not any(x in y or y in x for y in b))
    cb = sum(1 for y in set(b) if not any(x in y or y in x for x in a))
    return float(ca), float(cb), presence


def is_subsequence(a, b):
    it = iter(b)
    return all(ch in it for ch in a)


# ── record-name cleanup: leetspeak, glued domain/handle names ────────────────
#
# Retrieval misses on train India (40K owned records): after Indian-script and no-address records, most are
# names glued into a domain or handle ("foundationunitedprivate.com" = United Foundation Private Limited,
# "@si1versolution", "Hardiktraderscom"), which word-level BM25 cannot match at all.

_LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"}
_LEET_RE = re.compile(r"(?<=[A-Za-z])[013457](?=[A-Za-z])|(?<=[A-Za-z]{3})[013457]\b|\b[013457](?=[A-Za-z]{3})")


def unleet(s):
    """'si1ver' -> 'silver', 'Ind0' -> 'Indo', '5hakti' -> 'shakti': digits between letters, ending a 3+ letter
    word, or starting one ('4Th', '1St' are left alone)."""
    return _LEET_RE.sub(lambda m: _LEET[m.group()], s)


def segment(word, vocab, max_len=20):
    """Split a glued lowercase word into common known words (vocab: word -> cost, lower is more common), or
    None. Every piece must be a known word of 3+ letters (one 2-letter piece allowed), so typos and made-up
    trade names ('lgisstics', 'veohalosol') are left whole instead of being cut into junk."""
    n = len(word)
    best = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            c = vocab.get(word[j:i])
            if c is not None and i - j >= 2 and best[j] + c < best[i]:
                best[i], back[i] = best[j] + c, j
    if best[n] == float("inf"):
        return None
    parts, i = [], n
    while i > 0:
        parts.append(word[back[i]:i])
        i = back[i]
    parts.reverse()
    if len(parts) < 2 or sum(len(p) == 2 for p in parts) > 1:
        return None
    return parts


# ── addresses: abbreviations ─────────────────────────────────────────────────

_ABBR_DEFAULT = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "ln": "lane", "ste": "suite",
    "apt": "apartment", "flr": "floor", "fl": "floor", "bldg": "building", "nr": "near", "opp": "opposite",
    "hwy": "highway", "pkwy": "parkway", "sq": "square", "extn": "extension", "ext": "extension", "sec": "sector",
    "cir": "circle", "twr": "tower", "plz": "plaza",
}
_ABBR = {
    "France": {
        "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "blvd": "boulevard", "boul": "boulevard",
        "all": "allee", "al": "allee", "imp": "impasse", "ch": "chemin", "chem": "chemin", "pl": "place",
        "rte": "route", "crs": "cours", "fg": "faubourg", "fbg": "faubourg", "sq": "square", "pte": "porte",
        "res": "residence", "st": "saint", "ste": "sainte", "qu": "quai",
    },
}
_WORD = re.compile(r"\b[a-z]+\b")


def expand_abbrev(text, country):
    """Street-type abbreviations to full words ('63 r. de dieppe' -> '63 rue. de dieppe'); expects lowercase."""
    abbr = _ABBR.get(country, _ABBR_DEFAULT)
    return _WORD.sub(lambda m: abbr.get(m.group(), m.group()), text)


# ── addresses ────────────────────────────────────────────────────────────────

_HOUSE = re.compile(r"\d+[a-z]?(?:\s*[-/]\s*\d+[a-z]?)*")
_POSTAL = {"US": re.compile(r"\b(\d{5})(?:-\d{4})?\b"), "India": re.compile(r"\b(\d{6})\b"),
           "France": re.compile(r"\b(\d{5})\b")}

INDIA_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "keralam": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland",
    "od": "odisha", "or": "odisha", "orissa": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim",
    "tn": "tamil nadu", "tg": "telangana", "ts": "telangana", "tr": "tripura", "up": "uttar pradesh",
    "uk": "uttarakhand", "ut": "uttarakhand", "uttaranchal": "uttarakhand", "wb": "west bengal",
    "paschimbanga": "west bengal", "pashchimbanga": "west bengal", "dl": "delhi", "jk": "jammu and kashmir",
    "la": "ladakh", "ch": "chandigarh", "py": "puducherry", "pondicherry": "puducherry",
    "an": "andaman and nicobar islands", "dn": "dadra and nagar haveli", "dd": "daman and diu",
    "ld": "lakshadweep",
}
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
_STATE_MAPS = {"India": INDIA_STATES, "US": US_STATES}


def house_number(addr):
    """First number-like token, leading zeros dropped: '#002-3-349/2' -> '2-3-349/2'."""
    m = _HOUSE.search(ascii_fold(addr))
    if not m:
        return ""
    return re.sub(r"\d+", lambda d: str(int(d.group())), re.sub(r"\s+", "", m.group()))


def house_match(a, b):
    """1 same, 0.5 one extends the other ('1-35-178' vs '1-35-178/3'), 0 different, -1 missing."""
    if not a or not b:
        return -1.0
    if a == b:
        return 1.0
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return 0.5 if long_.startswith(short) and long_[len(short)] in "-/" else 0.0


def address_parts(addr, country):
    """
    -> (house number, set of segment skeletons, state skeleton, postal code, full normalized text).
    Segments are comma-separated pieces; a segment that is a state abbreviation is expanded
    (HR -> haryana, TX -> texas) before skeletonizing, so reformatted addresses line up.
    """
    if not addr:
        return "", frozenset(), "", "", ""
    states = _STATE_MAPS.get(country, {})
    state_names = set(states.values())
    segs = []
    for raw in addr.split(","):
        # numbers are compared separately (house number, postal), so segments keep only words
        t = " ".join(w for w in _NON_ALNUM.sub(" ", ascii_fold(raw)).split() if not any(c.isdigit() for c in w))
        if t:
            key = t.replace(" ", "")
            segs.append(states[key] if key in states else expand_abbrev(t, country))
    # the state is wherever a known state name appears (addresses get reordered: "TX, DALLAS"),
    # else the last segment (regions/native-script states still line up via their skeleton).
    # No state list for the country (France): unknown. The "last segment" guess said "different state"
    # for 73% of near-certain France matches (India 8%, US 0.2%), which the model reads as a mismatch.
    if states:
        state = next((s for s in reversed(segs) if s in state_names), segs[-1] if segs else "")
    else:
        state = ""
    # postal codes are searched after the first segment, so a 5-digit US house number isn't one
    rest = addr.split(",", 1)[1] if "," in addr else ""
    m = _POSTAL.get(country, _POSTAL["France"]).findall(rest)
    return house_number(addr), frozenset(skeleton(s) for s in segs), skeleton(state), m[-1] if m else "", \
        ", ".join(segs)
