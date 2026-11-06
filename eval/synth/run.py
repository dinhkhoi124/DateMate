"""Unseen-shop proxy (PLAN 04-06/11 step 1): ingest and end-to-end measured separately.

PYTHONPATH=src .venv/Scripts/python eval/synth/run.py [--transforms rename nodoc ...] [--shops biz_104 ...] [--no-e2e]

- ingest: fingerprint of the Store built from the synthetic shop vs the one built from the original (counts, money,
  monthly revenue, statuses, capabilities). A mismatch means the data was read differently - no parser can fix it.
- e2e: the shop's dev questions on the synthetic shop vs on the original (production config, gate on).
Dev shops only (CLAUDE.md rule 2). Writes eval/runs/<ts>_synth/report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from datamate.execute.facts import facts_of  # noqa: E402
from datamate.ingest.store import Store  # noqa: E402
from datamate.pipeline import Context, answer  # noqa: E402
from eval.scorer import load_jsonl, score  # noqa: E402
from eval.synth.make import TRAIN, TRANSFORMS, make  # noqa: E402


def _r(x):
    return None if x is None else round(float(x), 2)


def fingerprint(st: Store) -> dict:
    """Comparable summary of what ingest understood."""
    fp = {}
    f = facts_of(st)
    O = f.O
    fp["orders.n"] = len(O)
    for k in ("valid", "cancelled", "returned"):
        fp[f"orders.{k}"] = int(O[k].sum())
    fp["orders.rev_valid"] = _r(O.loc[O["valid"], "rev"].sum())
    fp["orders.with_customer"] = int(O["ckey"].notna().sum())
    fp["orders.customers"] = int(O["ckey"].nunique())
    fp["orders.with_time"] = int(O["tod"].notna().sum())
    fp["orders.pay"] = dict(sorted(Counter(str(v) for v in O["pay"]).items()))
    fp["orders.channel"] = dict(sorted(Counter(str(v) for v in O["channel"]).items())) if "channel" in O else {}
    d = O["d"].values
    ok = ~np.isnat(d)
    fp["orders.date_min"] = str(d[ok].min()) if ok.any() else None
    fp["orders.date_max"] = str(d[ok].max()) if ok.any() else None
    months = defaultdict(float)
    for dd, v, rv in zip(d, O["valid"].values, O["rev"].values):
        if v and not np.isnat(dd):
            months[str(dd)[:7]] += rv
    fp["orders.rev_by_month"] = {k: _r(v) for k, v in sorted(months.items())}
    L = f.L
    fp["lines.n"] = len(L)
    if len(L):
        fp["lines.qty_base_valid"] = _r(L.loc[L["valid"], "qty_base"].sum())
        fp["lines.amount_valid"] = _r(L.loc[L["valid"], "amount"].sum())
        fp["lines.products"] = int(L["product_key"].nunique())
    fp["products.n"] = len(f.P)
    for name in ("payments", "imports", "import_lines", "stock", "lots", "deliveries", "expenses"):
        df = getattr(st, name)
        fp[f"{name}.n"] = 0 if df is None else len(df)
    if st.payments is not None and len(st.payments):
        fp["payments.amount"] = _r(np.nansum(np.asarray(st.payments["amount"], dtype=float)))
    caps = {k: v for k, v in st.caps.items() if k.startswith("has_")}
    fp["caps"] = dict(sorted(caps.items()))
    return fp


def compare(a: dict, b: dict) -> list[str]:
    return [k for k in a if a.get(k) != b.get(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transforms", nargs="*", default=list(TRANSFORMS))
    ap.add_argument("--shops", nargs="*")
    ap.add_argument("--no-e2e", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print per-shop ingest diffs")
    args = ap.parse_args()

    holdout = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    shops = args.shops or sorted(p.name for p in (TRAIN / "data").iterdir() if p.is_dir() and p.name not in holdout)
    if set(shops) & holdout:
        sys.exit("holdout shops are not allowed here (CLAUDE.md rule 2)")
    qs = [q for q in load_jsonl(TRAIN / "questions.jsonl") if q["business_id"] in set(shops)]
    golds = {g["id"]: g for g in load_jsonl(TRAIN / "answers.jsonl")}

    base_fp = {b: fingerprint(Store.build(TRAIN / "data" / b)) for b in shops}
    base_s = {}
    if not args.no_e2e:
        ctx = Context(root=TRAIN)
        for q in qs:
            try:
                base_s[q["id"]] = score(answer(q, ctx), golds[q["id"]])
            except Exception:
                base_s[q["id"]] = 0

    report = {"shops": shops, "transforms": {}}
    print(f"{'transform':12}{'ingest ok':>11}{'fields bad':>11}{'PA base':>9}{'PA synth':>9}{'ok->wrong':>10}{'ok->abst':>9}{'ok->ok':>8}")
    for tr in args.transforms:
        t0 = time.time()
        root = make(tr, shops)
        rec = {"ingest": {}, "e2e": {}}
        bad_fields = Counter()
        ing_ok = 0
        for b in shops:
            try:
                fp = fingerprint(Store.build(root / "data" / b))
                diff = compare(base_fp[b], fp)
            except Exception as e:  # noqa: BLE001
                diff = [f"ERROR {type(e).__name__}: {e}"]
                traceback.print_exc(limit=1, file=sys.stderr)
            rec["ingest"][b] = diff
            ing_ok += not diff
            if diff and args.verbose:
                print(f"  {tr} {b}: {', '.join(diff)[:300]}", flush=True)
            bad_fields.update(d.split(" ")[0] if d.startswith("ERROR") else d for d in diff)
        trans = Counter()
        if not args.no_e2e:
            ctx = Context(root=root)
            for q in qs:
                try:
                    s = score(answer(q, ctx), golds[q["id"]])
                except Exception:
                    s = 0
                trans[(base_s[q["id"]], s)] += 1
                rec["e2e"].setdefault(q["business_id"], Counter())[f"{base_s[q['id']]}>{s}"] += 1
        n = len(qs) or 1
        pa_b = sum(base_s.values()) / n if base_s else float("nan")
        pa_s = sum(s * c for (_, s), c in trans.items()) / n if trans else float("nan")
        rec["summary"] = {"ingest_ok": ing_ok, "shops": len(shops), "bad_fields": dict(bad_fields.most_common()),
                          "pa_base": pa_b, "pa_synth": pa_s, "transitions": {f"{a}>{b}": c for (a, b), c in trans.items()},
                          "seconds": round(time.time() - t0, 1)}
        report["transforms"][tr] = rec
        print(f"{tr:12}{ing_ok:>6}/{len(shops):<4}{sum(bad_fields.values()):>11}{pa_b:9.3f}{pa_s:9.3f}"
              f"{trans[(1, -1)]:10}{trans[(1, 0)]:9}{trans[(1, 1)]:8}   top bad: {', '.join(k for k, _ in bad_fields.most_common(4))}",
              flush=True)

    run_dir = ROOT / "eval" / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_synth"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf8")
    print(f"\nrun dir: {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
