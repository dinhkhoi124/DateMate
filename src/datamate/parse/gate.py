"""Coverage gate (D10): answer only questions whose wording the rule parser has seen.

A rule parser maps an unfamiliar question to the nearest familiar op and answers it confidently (public v0.2:
precision ~55% on industries absent from train). The gate removes everything the parser can explain without wording
knowledge - catalog entities, time expressions, numbers - and abstains if any remaining word is outside the known
vocabulary (words of train-dev questions the pipeline answers correctly, built by analysis/build_gate_vocab.py).
"""
from __future__ import annotations

import re
from pathlib import Path

from datamate.link.catalog import Linker

VOCAB_PATH = Path(__file__).with_name("gate_vocab.txt")
_WORD = re.compile(r"[a-z]+")


def load_vocab(path: Path = VOCAB_PATH) -> frozenset[str] | None:
    if not path.exists():
        return None
    words = (w.strip() for w in path.read_text(encoding="utf8").splitlines())
    return frozenset(w for w in words if w and not w.startswith("#"))


def _mask(text: str, spans) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(max(a, 0), min(b, len(chars))):
            chars[i] = " "
    return "".join(chars)


def residual(t: str, pre: str | None, lk: Linker, periods) -> list[str]:
    """Words of the normalised question left after masking time expressions and catalog mentions (sorted, unique)."""
    spans = [p.span for p in periods]
    for d in (lk.products, lk.customers, lk.staff, lk.categories, lk.ingredients, lk.suppliers):
        spans += [(h.start, h.end) for h in d.find(t, limit=50)]
    words = set(_WORD.findall(_mask(t, spans)))
    if pre is not None and pre != t:  # names linked only before teencode repair: drop their words too
        for d in (lk.products, lk.customers):
            for h in d.find(pre, limit=50):
                words -= set(_WORD.findall(pre[h.start:h.end]))
    return sorted(words)


def unknown(words: list[str], vocab: frozenset[str] | None) -> list[str]:
    if vocab is None:
        return []
    return [w for w in words if w not in vocab]
