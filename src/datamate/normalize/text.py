"""Question text normalisation (docs/ARCHITECTURE.md §1 step 1).

normalize(q) -> folded canonical text: lowercase, no diacritics, Telex typos repaired, teencode expanded,
numbers de-grouped ('12.000.000' -> '12000000'). The parser works on this text; names shown in answers are
always taken from the shop data, never from this text.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from unidecode import unidecode

_VOCAB_PATH = Path(__file__).with_name("vocab.txt")

# ascii words that are legitimate as typed (teencode / english) and must not be Telex-decoded
_KEEP = {"revenue", "total", "bill", "order", "size", "top", "cod", "check", "best", "seller", "oa", "zalo", "vat",
         "voucher", "nv", "kg", "ml", "m", "l", "mm", "cm", "m3", "m2", "ok"}

# teencode / abbreviations on folded text; applied in order. \b-anchored.
_TEEN = [
    (r"\b(dthu|dt|revenue|doanh so)\b", "doanh thu"),
    (r"\b(bnhiu|bnhieu|bnh|bn|bao nhiu|bao nhieeu)\b", "bao nhieu"),
    (r"\b(hqua|bua qua|hom wa)\b", "hom qua"),
    (r"\b(hnay|bua nay|hum nay)\b", "hom nay"),
    (r"\b(dc|đc|dduoc|duoc)\b", "duoc"),
    (r"\b(trc|truoc)\b", "truoc"),
    (r"\bth (truoc|roi|lien truoc|nay|sau)\b", r"thang \1"),
    (r"\bvs\b", "voi"),
    (r"\bsp\b", "san pham"),
    (r"\btm\b", "tien mat"),
    (r"\bck\b", "chuyen khoan"),
    (r"\b(bill|order|hoa don)s?\b", "don"),
    (r"\btotal\b", "tong cong"),
    (r"\b(k|hok|khg|kh0ng)\b(?= (ke don|ban|co|duoc|ai|can))", "khong"),
    (r"\bhong\b(?= (ke don|duoc|co ai))|(?<=ban )hong\b", "khong"),  # 'hông' teencode; 'họng'/'Hồng' elsewhere
    (r"\b(ko|hok|khoong|khg)\b", "khong"),
    (r"\b(la|ke) (j|chi|gi)\b", r"\1 gi"),
    (r"\bnhiu\b", "nhieu"),
    (r"\b(gium|giup|dum)\b", "giup"),
    (r"\bmon\b", "mon"),
]
_TEEN = [(re.compile(a), b) for a, b in _TEEN]
# conversational fillers at the edges (no meaning for the parser)
_LEAD = re.compile(r"^((e|ee|ê|oi|alo|check giup|coi giup|xem giup|cho (toi|minh|em) (hoi|biet)|cho hoi|cho (toi|minh)"
                   r"|giup (minh|toi|em)?|lam on)\s+)+")
_TAIL = re.compile(r"(\s+(z|vay|nhi|ta|ha|a|nha|do|the|hả|hen|ne|di|vaay|vay a|nhe))+\s*$")


@lru_cache(maxsize=1)
def vocab() -> frozenset:
    return frozenset(w.strip() for w in _VOCAB_PATH.read_text(encoding="utf8").split() if w.strip())


def telex_fix(tok: str, protect: frozenset = frozenset()) -> str:
    """Repair one folded ascii token typed with Telex but without IME: 'thangs'->'thang', 'dduwowcj'->'duoc'."""
    V = vocab()
    if tok in V or tok in _KEEP or tok in protect or not tok.isalpha():
        return tok
    cands = [tok]
    if len(tok) > 1 and tok[-1] in "sfrxj":
        cands.append(tok[:-1])
    out = []
    for c in cands:
        d = c.replace("dd", "d")
        d = re.sub(r"(?<=[aou])w", "", d)
        d = re.sub(r"([aeo])\1", r"\1", d)
        out.append(d)
        # tone letter placed before the final consonant: 'tieefn' -> 'tien'
        m = re.match(r"^(.*[aeiouy])([sfrxj])([cmnptg]h?|ng|nh)$", d)
        if m:
            out.append(m.group(1) + m.group(3))
    for d in out:
        if d in V:
            return d
    return tok


def fold_text(s: str) -> str:
    """Lowercase + strip diacritics + collapse spaces (no typo repair)."""
    s = unicodedata.normalize("NFC", s or "").lower()
    return re.sub(r"\s+", " ", unidecode(s)).strip()


def _tokens(q: str) -> list[str]:
    s = unicodedata.normalize("NFC", q or "").lower()
    s = s.replace("≥", " tu ").replace("≤", " toi da ").replace("→", " , ").replace("–", "-").replace("—", "-")
    s = unidecode(s)
    # numbers: '12.000.000' / '1,400' (thousand groups) -> plain digits; keep dates d/m/y and decimals '0,9'
    s = re.sub(r"(?<![\d/])(\d{1,3}(?:[.,]\d{3})+)(?![\d/])(?!\s*(?:mm|ml|m\b|%))",
               lambda m: re.sub(r"[.,]", "", m.group(1)), s)
    # 'm3', 'd16', '500ml', '11.7m' stay single tokens (same tokenisation for questions and catalog names)
    return re.findall(r"[a-z]+\d+[a-z]*|\d+(?:[.,]\d+)?[a-z]*|[a-z]+|[^\sa-z\d]", s)


def _join(toks: list[str]) -> str:
    s = " ".join(toks)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+([?.,!:;])", r"\1", s)
    s = re.sub(r"\(\s*", "(", s)
    return re.sub(r"\s*\)", ")", s)


def norm_name(s: str) -> str:
    """Catalog names in the same token space as normalize() (no typo repair, no teencode)."""
    return _join(_tokens(s)).strip()


def normalize_pair(q: str, protect: frozenset = frozenset()) -> tuple[str, str]:
    """(typo-repaired text before teencode expansion, fully normalised text). Names are linked on both, since
    teencode may rewrite a token that belongs to a product name ('Total', 'SP')."""
    toks = [telex_fix(t, protect) if t.isalpha() else t for t in _tokens(q)]
    pre = _join(toks)
    return pre, _finish(pre)


def normalize(q: str, protect: frozenset = frozenset()) -> str:
    return normalize_pair(q, protect)[1]


def _finish(s: str) -> str:
    for rx, rep in _TEEN:
        s = rx.sub(rep, s)
    s = re.sub(r"[?!.]+\s*$", "", s).strip()
    s = _LEAD.sub("", s)
    s = _TAIL.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


__all__ = ["normalize", "normalize_pair", "norm_name", "fold_text", "telex_fix", "vocab"]
