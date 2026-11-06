"""Fault injection for the run-time invariants (DECISIONS D11).

PYTHONPATH=src .venv/Scripts/python eval/synth/faults.py [--faults ...] [--shops ...]   (foreground, a few shops per call: per-shop results are kept in out/faults_shops.jsonl and reused)

Each fault corrupts a dev shop's built Store the way a misread export would (dd/mm read as mm/dd, '1.250.000' read
as 1.25, status column not found, price column taken as the line amount, wrong order-id column, a table read twice).
For every dev question we compare the answer on the clean Store with the answer on the corrupted one, checks off and
on. Good invariants turn ok->wrong into ok->abstain; on clean Stores they must change nothing (column 'clean').
Dev shops only (CLAUDE.md rule 2). Writes eval/runs/<ts>_faults/report.json.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import copy
import json
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from datamate.ingest.checks import blocking, metrics, violations  # noqa: E402
from datamate.ingest.store import Store  # noqa: E402
from datamate.ingest.values import DONE  # noqa: E402
from datamate.pipeline import Context, analyse  # noqa: E402
from eval.scorer import load_jsonl, score  # noqa: E402

TRAIN = ROOT / "data" / "train" / "train"
ORDER_MONEY = ("subtotal", "order_subtotal", "order_discount", "order_total", "paid", "debt", "shipping_fee")
LINE_MONEY = ("amount", "unit_price", "line_discount", "cost_unit", "price_list")


def _clone(st: Store) -> Store:
    c = copy.copy(st)
    c.orders = st.orders.copy()
    c.lines = st.lines.copy()
    c.payments = st.payments.copy() if st.payments is not None else None
    c.extra = {k: v for k, v in st.extra.items() if not k.startswith("_")}
    c.caps = dict(st.caps)
    c.warnings = list(st.warnings)
    return c


def _resum(st: Store):
    """Order revenue is the sum of its line amounts (rulebook §1): recompute after touching lines."""
    if len(st.lines):
        s = pd.to_numeric(st.lines["amount"], errors="coerce").groupby(st.lines["order_id"]).sum()
        st.orders["subtotal"] = st.orders["order_id"].map(s)


def _swap(v):
    if isinstance(v, datetime):
        return v.replace(month=v.day, day=v.month) if v.day <= 12 else None
    if isinstance(v, date):
        return date(v.year, v.day, v.month) if v.day <= 12 else None
    return v


def f_day_month(st):
    for df in (st.orders, st.lines):
        for c in ("order_date", "order_dt"):
            if c in df:
                df[c] = df[c].map(_swap).astype(object)


def f_money_div1000(st):
    for df, cols in ((st.orders, ORDER_MONEY), (st.lines, LINE_MONEY)):
        for c in cols:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce") / 1000
    if st.payments is not None and "amount" in st.payments:
        st.payments["amount"] = pd.to_numeric(st.payments["amount"], errors="coerce") / 1000


def f_status_lost(st):
    st.orders["status"] = DONE


def f_amount_is_price(st):
    if len(st.lines) and "unit_price" in st.lines and st.lines["unit_price"].notna().any():
        st.lines["amount"] = st.lines["unit_price"]
        _resum(st)


def f_wrong_order_id(st):
    if len(st.lines):
        st.lines["order_id"] = [f"X{i}" for i in range(len(st.lines))]
        _resum(st)


def f_double_read(st):
    st.orders = pd.concat([st.orders, st.orders], ignore_index=True)
    st.lines = pd.concat([st.lines, st.lines], ignore_index=True)


FAULTS = {"day_month": f_day_month, "money_div1000": f_money_div1000, "status_lost": f_status_lost,
          "amount_is_price": f_amount_is_price, "wrong_order_id": f_wrong_order_id, "double_read": f_double_read}


def _run(q, ctx):
    """(op, row) with checks off; a crash is an abstain in production (run.py)."""
    try:
        it, _, row = analyse(q, ctx)
    except Exception:   # noqa: BLE001
        return None, None
    return (it.op if it is not None else None), row


PARTIAL = ROOT / "eval" / "synth" / "out" / "faults_shops.jsonl"


def _code_sig() -> str:
    """What the saved raw results depend on: answering code (checks off), the faults, and checks.metrics.
    Thresholds (config.Invariants), violations() and OP_AREAS are applied at aggregation, so tuning them reuses
    the saved results."""
    from datamate import config
    from datamate.ingest import checks
    h = hashlib.sha1()
    for f in sorted((ROOT / "src" / "datamate").rglob("*.py")):
        if f.name not in ("checks.py", "config.py"):
            h.update(f.read_bytes())
    for obj in (config.Rules, config.from_rulebook, checks.metrics, _clone, _resum, _swap, _run, _shop,
                *FAULTS.values()):
        h.update(inspect.getsource(obj).encode())
    return h.hexdigest()[:16]


def _shop(b, names, qs, golds, ctx) -> dict:
    """Raw results for one shop: per fault, the Store metrics and [qid, op, clean score, faulty score]."""
    base = Store.build(TRAIN / "data" / b)
    bq = [q for q in qs if q["business_id"] == b]
    ctx.stores[b] = base
    base_s = {}
    for q in bq:
        _, row = _run(q, ctx)
        base_s[q["id"]] = score(row, golds[q["id"]]) if row else 0
    res = {}
    for n in names:
        st = base if n == "clean" else _clone(base)
        if n != "clean":
            FAULTS[n](st)
        ctx.stores[b] = st
        rows = []
        for q in bq:
            op, row = _run(q, ctx)
            rows.append([q["id"], op, base_s[q["id"]], score(row, golds[q["id"]]) if row else 0])
        res[n] = {"metrics": metrics(st), "q": rows}
        ctx.stores.pop(b, None)
        del st
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", nargs="*", default=list(FAULTS))
    ap.add_argument("--shops", nargs="*")
    args = ap.parse_args()
    holdout = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    shops = args.shops or sorted(p.name for p in (TRAIN / "data").iterdir() if p.is_dir() and p.name not in holdout)
    if set(shops) & holdout:
        sys.exit("holdout shops are not allowed here (CLAUDE.md rule 2)")
    qs = [q for q in load_jsonl(TRAIN / "questions.jsonl") if q["business_id"] in set(shops)]
    golds = {g["id"]: g for g in load_jsonl(TRAIN / "answers.jsonl")}
    ctx = Context(root=TRAIN, checks=False)   # answers with checks off; blocking() applied below
    names = ["clean"] + list(args.faults)
    # per-shop results are appended as soon as a shop is done, so a killed run loses at most one shop;
    # a record is reused only if the code that produced it (checks, config, pipeline, this file) is unchanged
    sig = _code_sig()
    done = {}
    if PARTIAL.exists():
        for line in PARTIAL.read_text(encoding="utf8").splitlines():
            r = json.loads(line)
            if r["sig"] == sig and r["faults"] == names:
                done[r["shop"]] = r
    t0 = time.time()
    for b in shops:
        if b in done:
            print(f"  {b} reused", flush=True)
            continue
        rec = {"shop": b, "sig": sig, "faults": names, "res": _shop(b, names, qs, golds, ctx)}
        with open(PARTIAL, "a", encoding="utf8") as w:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done[b] = rec
        gc.collect()
        print(f"  {b} done ({time.time() - t0:.0f}s)", flush=True)
    tr = {n: {"off": Counter(), "on": Counter()} for n in names}
    fired = {n: Counter() for n in names}
    leak = {n: Counter() for n in names}
    for b in shops:
        for n in names:
            r = done[b]["res"][n]
            viol = violations(r["metrics"])
            fired[n].update({v.name for v in viol})
            for _qid, op, b0, s in r["q"]:
                blk = bool(blocking(op, viol))
                tr[n]["off"][(b0, s)] += 1
                tr[n]["on"][(b0, 0 if blk else s)] += 1
                if b0 == 1 and s == -1 and not blk:
                    leak[n][str(op)] += 1
    N = len(qs) or 1
    print(f"\n{'fault':16}{'PA off':>8}{'PA on':>8}{'ok>wrong off':>14}{'ok>wrong on':>13}{'ok>abst on':>12}   fired")
    rep = {}
    for n in names:
        pa = {k: sum(s * c for (_, s), c in tr[n][k].items()) / N for k in ("off", "on")}
        print(f"{n:16}{pa['off']:8.3f}{pa['on']:8.3f}{tr[n]['off'][(1, -1)]:14}{tr[n]['on'][(1, -1)]:13}"
              f"{tr[n]['on'][(1, 0)]:12}   {dict(fired[n].most_common())}")
        if leak[n]:
            print(f"{'':16}leaks: {dict(leak[n].most_common())}")
        rep[n] = {"leaks": dict(leak[n]),"pa": pa, "off": {f"{a}>{s}": c for (a, s), c in tr[n]["off"].items()},
                  "on": {f"{a}>{s}": c for (a, s), c in tr[n]["on"].items()}, "fired": dict(fired[n])}
    run_dir = ROOT / "eval" / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_faults"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps({"shops": shops, "faults": rep}, ensure_ascii=False, indent=1),
                                         encoding="utf8")
    print(f"\nrun dir: {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
