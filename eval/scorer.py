"""Exact typed-match scorer, mirroring rulebook §9.

score(pred, gold) -> +1 (correct), -1 (wrong), 0 (abstain / missing).
`pred` is a submission row {"id","type","answer"[,"reason"]}; `gold` is an answers.jsonl row
{"id","type","answer","alias"}.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def norm_str(s: str) -> str:
    """Case/whitespace/punctuation-insensitive; diacritics kept (they must be right)."""
    s = unicodedata.normalize("NFC", s).casefold()
    return _PUNCT.sub("", s)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _num_eq(a, b) -> bool:
    # only float noise is forgiven: money must match to the đồng, percent to 0.01
    return _is_num(a) and _is_num(b) and abs(a - b) <= 1e-6 + 1e-12 * abs(b)


def _str_eq(pred, gold, aliases=()) -> bool:
    if not isinstance(pred, str):
        return False
    p = norm_str(pred)
    return any(p == norm_str(g) for g in (gold, *aliases) if isinstance(g, str))


def _elem_eq(pred, gold, aliases=()) -> bool:
    if isinstance(gold, str):
        return _str_eq(pred, gold, aliases)
    if isinstance(gold, bool):
        return isinstance(pred, bool) and pred == gold
    if _is_num(gold):
        return _num_eq(pred, gold)
    if isinstance(gold, list):
        return isinstance(pred, list) and len(pred) == len(gold) and all(_elem_eq(p, g) for p, g in zip(pred, gold))
    return pred == gold


def _table_eq(pred, gold, row_aliases) -> bool:
    if not (isinstance(pred, dict) and isinstance(pred.get("columns"), list) and isinstance(pred.get("rows"), list)):
        return False
    gcols, pcols = gold["columns"], pred["columns"]
    if len(gcols) != len(pcols) or len(pred["rows"]) != len(gold["rows"]):
        return False
    # columns matched by name, else by position
    order = []
    for i, c in enumerate(gcols):
        order.append(pcols.index(c) if c in pcols else i)
    if sorted(order) != list(range(len(gcols))):
        order = list(range(len(gcols)))
    remaining = [r for r in pred["rows"] if isinstance(r, list) and len(r) == len(pcols)]
    if len(remaining) != len(gold["rows"]):
        return False
    for gi, grow in enumerate(gold["rows"]):
        al = row_aliases[gi] if gi < len(row_aliases) else []
        hit = None
        for ri, prow in enumerate(remaining):
            pr = [prow[j] for j in order]
            if _elem_eq(pr[0], grow[0], al) and all(_elem_eq(p, g) for p, g in zip(pr[1:], grow[1:])):
                hit = ri
                break
        if hit is None:
            return False
        remaining.pop(hit)
    return True


def is_correct(pred: dict, gold: dict) -> bool:
    gt, pt = gold["type"], pred.get("type")
    if pt != gt:
        return False
    if gt == "unanswerable":
        return True
    pa, ga, al = pred.get("answer"), gold["answer"], gold.get("alias") or []
    if gt in ("number", "percent"):
        return _num_eq(pa, ga)
    if gt == "date":
        return isinstance(pa, str) and pa == ga
    if gt == "boolean":
        return isinstance(pa, bool) and pa == ga
    if gt == "string":
        return _str_eq(pa, ga, al)
    if gt == "list":
        if not isinstance(pa, list) or len(pa) != len(ga):
            return False
        return all(_elem_eq(p, g, al[i] if i < len(al) else ()) for i, (p, g) in enumerate(zip(pa, ga)))
    if gt == "table":
        return _table_eq(pa, ga, al)
    return False


def score(pred: dict | None, gold: dict) -> int:
    if pred is None or pred.get("type") in (None, "abstain"):
        return 0
    return 1 if is_correct(pred, gold) else -1


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def evaluate(preds: list[dict], golds: list[dict]) -> dict:
    pm = {p["id"]: p for p in preds}
    s = [score(pm.get(g["id"]), g) for g in golds]
    n = len(golds)
    right, wrong = s.count(1), s.count(-1)
    return {"n": n, "correct": right, "wrong": wrong, "abstain": n - right - wrong,
            "PA": (right - wrong) / n if n else 0.0,
            "precision": right / (right + wrong) if right + wrong else 0.0}


if __name__ == "__main__":
    import sys
    golds = load_jsonl(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "data/train/train/answers.jsonl")
    print(evaluate(load_jsonl(sys.argv[1]), golds))
