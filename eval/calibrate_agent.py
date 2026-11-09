"""Calibrate the code-gen agent (D20) on CPU from a GPU run's trace.jsonl (eval/run_eval.py --trace).

Re-votes the recorded per-sample results for every min_votes m and every rule x code-gen policy, without running
the model or the sandbox again, and reports PA / precision / coverage. Also counts sample failure modes
(no code, runtime error, timeout) and the most frequent error messages, to guide prompt and helper fixes.

    python eval/calibrate_agent.py eval/runs/<run>/trace.jsonl [--errors 10]

Holdout traces: totals only (CLAUDE.md rule 2) -- use --totals-only.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from datamate.agent.agent import combine  # noqa: E402
from datamate.agent.answer import vote_key  # noqa: E402
from eval.scorer import load_jsonl, score  # noqa: E402

POLICIES = ("codegen", "rule_first", "agree")


def revote(results: list[dict], m: int, allow_unans: bool) -> dict | None:
    rows = [r["result"] for r in results if r.get("status") == "ok" and isinstance(r.get("result"), dict)]
    if not rows:
        return None
    c = Counter(vote_key(r) for r in rows)
    top, votes = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    row = next(r for r in rows if vote_key(r) == top)
    if votes < m or row["type"] == "abstain" or (row["type"] == "unanswerable" and not allow_unans):
        return None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--answers", default=str(ROOT / "data" / "train" / "train" / "answers.jsonl"))
    ap.add_argument("--errors", type=int, default=8, help="show N most frequent runtime error messages")
    ap.add_argument("--totals-only", action="store_true", help="holdout: no per-shop or error detail")
    args = ap.parse_args()

    golds = {g["id"]: g for g in load_jsonl(args.answers)}
    tr = [t for t in load_jsonl(args.trace) if t["id"] in golds]
    k = max((len(t.get("output") or []) for t in tr), default=0)
    print(f"{len(tr)} questions, k={k} samples")

    status, errs = Counter(), Counter()
    for t in tr:
        for r in t.get("output") or []:
            status[r.get("status")] += 1
            if r.get("status") == "error":
                errs[re.sub(r"\d+", "N", str(r.get("result")))[:90]] += 1
    tot = sum(status.values()) or 1
    print("samples: " + "  ".join(f"{s}={n} ({100 * n / tot:.0f}%)" for s, n in status.most_common()))

    print(f"\n{'policy':12}{'m':>3}{'unans':>6}{'ok':>6}{'wrong':>6}{'abst':>6}{'PA':>8}{'prec':>8}")
    best = None
    for allow in (False, True):
        for m in range(1, max(k, 1) + 1):
            for pol in POLICIES:
                c = w = a = 0
                for t in tr:
                    rule = {"id": t["id"], **((t.get("inputs") or {}).get("rule_answer") or {"type": "abstain"})}
                    row, _ = combine(rule, revote(t.get("output") or [], m, allow), pol)
                    s = score(row, golds[t["id"]])
                    c, w, a = c + (s == 1), w + (s == -1), a + (s == 0)
                n = c + w + a
                pa, prec = (c - w) / n, c / (c + w) if c + w else 0
                print(f"{pol:12}{m:3}{'y' if allow else 'n':>6}{c:6}{w:6}{a:6}{pa:8.3f}{prec:8.3f}")
                if best is None or pa > best[0]:
                    best = (pa, pol, m, allow, prec)
    print(f"\nbest PA {best[0]:.3f}: policy={best[1]} m={best[2]} unanswerable={'on' if best[3] else 'off'} "
          f"precision={best[4]:.3f}")

    if not args.totals_only:
        print("\ncode-gen alone (m=ceil(k/2)+1 or k) by shop: ok/wrong/abst")
        m = min(k, k // 2 + 2)
        by = {}
        for t in tr:
            row = revote(t.get("output") or [], m, False)
            s = score({"id": t["id"], **row} if row else None, golds[t["id"]])
            by.setdefault(t["business_id"], Counter())[s] += 1
        for b, cn in sorted(by.items()):
            print(f"  {b}: {cn[1]}/{cn[-1]}/{cn[0]}")
        if errs:
            print(f"\ntop runtime errors ({sum(errs.values())}):")
            for e, n in errs.most_common(args.errors):
                print(f"  {n:4}  {e}")


if __name__ == "__main__":
    main()
