"""Model-code result -> submission answer (rulebook §9 formats) and the vote key used to compare samples (D20).

`canon` turns whatever `solve()` returned (numpy scalars, Timestamps, DataFrames…) into plain JSON values of the
declared type, or raises ValueError when it cannot (that sample is discarded, never guessed).
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import date, datetime

TYPES = ("number", "percent", "date", "string", "list", "table", "boolean", "unanswerable", "abstain")


def _num(v):
    if hasattr(v, "item") and not isinstance(v, (list, dict, str)):   # numpy scalar
        v = v.item()
    if isinstance(v, bool):
        raise ValueError("bool is not a number")
    if isinstance(v, str):
        v = float(v.replace(",", "").strip())
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        raise ValueError("nan/inf")
    v = round(v, 2)
    return int(v) if v == int(v) else v


def _scalar(v):
    """Plain JSON scalar for list/table cells."""
    if hasattr(v, "item") and not isinstance(v, (list, dict, str)):
        v = v.item()
    if isinstance(v, (datetime,)):
        return v.date().isoformat() if (v.hour, v.minute, v.second) == (0, 0, 0) else v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if hasattr(v, "isoformat"):            # pandas Timestamp
        return v.isoformat()
    if isinstance(v, float):
        if math.isnan(v):
            raise ValueError("nan cell")
        v = round(v, 2)
        return int(v) if v == int(v) else v
    if v is None or isinstance(v, (int, str, bool)):
        return unicodedata.normalize("NFC", v) if isinstance(v, str) else v
    return str(v)


def _date(v) -> str:
    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()
    if isinstance(v, datetime):
        v = v.date()
    if isinstance(v, date):
        return v.isoformat()
    s = str(v).strip()
    if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", s):
        return s
    raise ValueError(f"bad date {s!r}")


def canon(res) -> dict:
    """{'type', 'answer'} (or unanswerable/abstain) in submission shape. Raises ValueError on anything malformed."""
    if not isinstance(res, dict) or res.get("type") not in TYPES:
        raise ValueError("solve() must return {'type': ..., 'answer': ...}")
    t = res["type"]
    if t == "abstain":
        return {"type": "abstain"}
    if t == "unanswerable":
        return {"type": "unanswerable", "reason": str(res.get("reason") or res.get("answer") or "")}
    if "answer" not in res:
        raise ValueError("missing answer")
    a = res["answer"]
    if t in ("number", "percent"):
        return {"type": t, "answer": _num(a)}
    if t == "date":
        return {"type": t, "answer": _date(a)}
    if t == "boolean":
        if hasattr(a, "item"):
            a = a.item()
        if not isinstance(a, bool):
            raise ValueError("boolean answer must be bool")
        return {"type": t, "answer": a}
    if t == "string":
        if a is None:
            raise ValueError("empty string answer")
        return {"type": t, "answer": unicodedata.normalize("NFC", str(_scalar(a)))}
    if t == "list":
        if hasattr(a, "tolist"):
            a = a.tolist()
        if not isinstance(a, (list, tuple)):
            raise ValueError("list answer must be a list")
        return {"type": t, "answer": [_scalar(x) for x in a]}
    if t == "table":
        if hasattr(a, "to_dict") and hasattr(a, "columns"):   # DataFrame
            a = {"columns": [str(c) for c in a.columns], "rows": a.values.tolist()}
        if not isinstance(a, dict) or "columns" not in a or "rows" not in a:
            raise ValueError("table answer must be {'columns', 'rows'}")
        return {"type": t, "answer": {"columns": [str(c) for c in a["columns"]],
                                      "rows": [[_scalar(x) for x in r] for r in a["rows"]]}}
    raise ValueError(t)


def _fold(s: str) -> str:
    """Scorer-style string comparison (§9: case, whitespace, punctuation insensitive)."""
    s = unicodedata.normalize("NFC", s).casefold()
    return re.sub(r"[\s\W_]+", "", s)


def vote_key(row: dict) -> str:
    """Two samples vote together iff their keys are equal (mirrors eval/scorer.py exact typed match)."""
    t = row["type"]
    if t in ("abstain", "unanswerable"):
        return t
    a = row["answer"]

    def k(x):
        return _fold(x) if isinstance(x, str) else x
    if t == "string":
        a = k(a)
    elif t == "list":
        a = [k(x) for x in a]
    elif t == "table":
        a = {"columns": [_fold(c) for c in a["columns"]], "rows": sorted(json.dumps([k(x) for x in r]) for r in a["rows"])}
    return json.dumps([t, a], ensure_ascii=False, sort_keys=True)
