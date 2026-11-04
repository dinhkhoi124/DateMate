"""Validate a submission.jsonl before uploading (docs/SPEC.md §3).

python eval/submit_check.py <submission.jsonl> [questions.jsonl]
Exit 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

TYPES = {"number", "percent", "date", "boolean", "string", "list", "table", "unanswerable", "abstain"}
DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_row(r: dict) -> list[str]:
    t, a, errs = r.get("type"), r.get("answer"), []
    if t not in TYPES:
        return [f"type không hợp lệ: {t!r}"]
    if t in ("number", "percent") and not _num(a):
        errs.append(f"{t} phải là JSON number: {a!r}")
    if t == "percent" and _num(a) and round(a, 2) != a:
        errs.append(f"percent chưa làm tròn 2 chữ số: {a}")
    if t == "date" and not (isinstance(a, str) and DATE.match(a)):
        errs.append(f"date sai format: {a!r}")
    if t == "boolean" and not isinstance(a, bool):
        errs.append(f"boolean phải là true/false: {a!r}")
    if t == "string" and not (isinstance(a, str) and a.strip()):
        errs.append("string rỗng hoặc không phải chuỗi")
    if t == "list" and not isinstance(a, list):
        errs.append("list phải là mảng")
    if t == "table":
        if not (isinstance(a, dict) and isinstance(a.get("columns"), list) and isinstance(a.get("rows"), list)):
            errs.append("table thiếu columns/rows")
        else:
            for row in a["rows"]:
                if not isinstance(row, list) or len(row) != len(a["columns"]):
                    errs.append(f"dòng table sai số cột: {row!r}")
                elif any(isinstance(c, float) and not math.isfinite(c) for c in row):
                    errs.append("ô NaN/Inf trong table")
    if t == "unanswerable" and "reason" in r and not isinstance(r["reason"], str):
        errs.append("reason phải là chuỗi")
    return errs


def main() -> int:
    sub = Path(sys.argv[1])
    raw = sub.read_bytes()
    errors: list[str] = []
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("file có BOM UTF-8")
    rows, seen = [], Counter()
    for i, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if "NaN" in line or "Infinity" in line:
            errors.append(f"dòng {i}: chứa NaN/Infinity")
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"dòng {i}: JSON lỗi ({e})")
            continue
        if "id" not in r:
            errors.append(f"dòng {i}: thiếu id")
            continue
        seen[r["id"]] += 1
        errors += [f"dòng {i} ({r['id']}): {e}" for e in check_row(r)]
        rows.append(r)
    errors += [f"id trùng: {k} ×{v}" for k, v in seen.items() if v > 1]

    qpath = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if qpath:
        qids = {json.loads(l)["id"] for l in qpath.read_text(encoding="utf8").splitlines() if l.strip()}
        extra = set(seen) - qids
        missing = qids - set(seen)
        errors += [f"id không có trong questions: {x}" for x in sorted(extra)[:20]]
        if missing:
            print(f"⚠ thiếu {len(missing)} câu (tính là abstain)")

    types = Counter(r.get("type") for r in rows)
    n = len(rows) or 1
    print("phân bố type:", dict(types.most_common()))
    print(f"abstain: {types['abstain'] / n:.1%}  unanswerable: {types['unanswerable'] / n:.1%}")
    if types["unanswerable"] / n > 0.05:
        print("⚠ unanswerable > 5% (train chỉ 2.3%), kiểm tra lại gate")
    for e in errors[:50]:
        print("✗", e)
    print("PASS" if not errors else f"FAIL ({len(errors)} lỗi)")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
