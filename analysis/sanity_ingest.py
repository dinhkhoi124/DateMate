"""Ingest sanity oracle: monthly revenue / order count questions (dev shops) vs gold answers.

Uses a deliberately tiny regex 'parser' so failures point at ingest, not at question understanding.
PYTHONPATH=src .venv/Scripts/python analysis/sanity_ingest.py [--show] [--holdout]  (holdout: counts only)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datamate.ingest.store import Store  # noqa: E402
from datamate.ingest.values import fold  # noqa: E402

TRAIN = ROOT / "data/train/train"
HOLDOUT = set(json.loads((ROOT / "eval/splits.json").read_text(encoding="utf8"))["holdout"])

REV = re.compile(r"^(?:cho toi biet |cho hoi |xem giup )?(?:tong )?doanh thu thang (\d{1,2})[/ ](?:nam )?(\d{4})(?: la)?(?: bao nhieu)?\??\.?$|"
                 r"^thang (\d{1,2})/(\d{4}) cua hang ban duoc tong cong bao nhieu tien\??$")
CNT = re.compile(r"^thang (\d{1,2})/(\d{4}) (?:cua hang )?(?:co )?(?:tong cong )?(?:ban duoc )?bao nhieu don(?: hang)?\??$")


def main():
    qs = [json.loads(l) for l in open(TRAIN / "questions.jsonl", encoding="utf8")]
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(TRAIN / "answers.jsonl", encoding="utf8")}
    stores, res = {}, defaultdict(lambda: [0, 0])
    show = "--show" in sys.argv
    for q in qs:
        b = q["business_id"]
        if (b in HOLDOUT) != ("--holdout" in sys.argv):
            continue
        n = fold(q["question"])
        m, kind = REV.match(n), "rev"
        if not m:
            m, kind = CNT.match(n), "cnt"
        if not m:
            continue
        g = [x for x in m.groups() if x]
        mo, yr = int(g[0]), int(g[1])
        if b not in stores:
            stores[b] = Store.build(TRAIN / "data" / b)
        st = stores[b]
        O = st.orders
        ok_orders = O[O["status"].isin(["done", "shipping"])]
        inm = ok_orders["order_date"].map(lambda d: d is not None and d.year == yr and d.month == mo)
        ids = set(ok_orders.loc[inm, "order_id"])
        if kind == "rev":
            val = round(float(ok_orders.loc[inm, "subtotal"].sum()))
        else:
            val = len(ids)
        gd = gold[q["id"]]
        hit = gd["type"] == "number" and abs(gd["answer"] - val) < 0.5
        res[(b, kind)][0 if hit else 1] += 1
        if show and not hit and b not in HOLDOUT:
            print(f"✗ {b} {kind} {mo}/{yr}: ours={val} gold={gd['answer']} ({gd['type']}) | {q['question']}")
    tot = [0, 0]
    for (b, kind), (ok, bad) in sorted(res.items()):
        tot[0] += ok
        tot[1] += bad
        print(f"{b} {kind}: {ok}/{ok + bad}")
    print(f"TOTAL {tot[0]}/{sum(tot)}")


if __name__ == "__main__":
    main()
