"""Rough keyword-based intent survey over train questions (Plan phase analysis).

Usage: .venv/Scripts/python analysis/intent_survey.py [--show INTENT] [--residual]
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from unidecode import unidecode

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train" / "train"

# common teencode / typo fixes applied after unaccenting
FIX = [
    (r"\bdthu\b", "doanh thu"), (r"\bdt\b", "doanh thu"), (r"\bsp\b", "san pham"),
    (r"\bbn\b|\bbnhiu\b|\bbao nhiu\b|\bnhiu\b|\bbnh\b", "bao nhieu"), (r"\bhnay\b|\bhom nay\b|\bbua nay\b", "hom nay"),
    (r"\bhqua\b|\bbua qua\b|\bhom qua\b", "hom qua"), (r"\bddown\b|\bdon\b", "don"), (r"\bkh\b", "khach"),
    (r"\bnowj\b|\bno\b", "no"), (r"\btrc\b|\btruoc\b", "truoc"), (r"\bdc\b|\bduoc\b|\bdduoc\b", "duoc"),
    (r"\bthang rồi\b", "thang truoc"),
]


def norm(s: str) -> str:
    s = unidecode(s.lower())
    s = re.sub(r"([a-z])\1{1,}", r"\1", s)  # nnhiu -> nhiu
    for a, b in FIX:
        s = re.sub(a, b, s)
    return re.sub(r"\s+", " ", s).strip()


# ordered: first match wins (more specific first)
RULES = [
    ("price_effect", r"tang gia|giam gia len|ngay ngay truoc"),
    ("avg_import_price", r"gia nhap binh quan|nhap vao gia binh quan|binh quan .*nhap"),
    ("import_total", r"phieu nhap|tien nhap|nhap hang|nha cung cap"),
    ("expiry", r"het han|han dung|lo gan"),
    ("stock_big_unit", r"chan.*le|nguyen.*le|thung.*le|hop.*le\b"),
    ("stock_value", r"gia tri .*ton|tien hang ton"),
    ("stock_out", r"het hang"),
    ("stock_qty", r"ton kho|trong kho|con bao nhieu|con may|con lai"),
    ("debt_aging", r"tuoi no|qua \d+ ngay|no lau"),
    ("debt_pay", r"tra bot no|tra no|thu no|thu duoc no"),
    ("debt_new", r"no them|mua chiu them"),
    ("debt_customer_count", r"so khach .*no"),
    ("debt_table", r"bang .*no|no nhieu nhat|cong no lon nhat"),
    ("debt_total", r"\bno\b|cong no"),
    ("cancel_rate", r"% don .*huy|ti le don huy|ty le don huy|huy .*phan tram|phan tram .*huy"),
    ("cancel_count", r"huy"),
    ("net_profit", r"lai rong|chi phi"),
    ("gross_profit_item", r"loi nhat|lai nhat|lai gop cao"),
    ("gross_profit", r"lai gop|tien lai|loi nhuan|lai bao nhieu"),
    ("shift", r"ca sang|ca chieu"),
    ("tet", r"\btet\b"),
    ("payment_method", r"tien mat|chuyen khoan|\bcod\b|ghi no|the\b|vi dien tu"),
    ("customer_group", r"nhom khach|khach si|khach le|khach vip|khach quen"),
    ("staff", r"nhan vien|\bnv\d"),
    ("dead_stock", r"khong ban (ra )?duoc|e\b|khong ban ra"),
    ("item_in_n_orders", r"mat hang .*don khac nhau|duoc mua o"),
    ("order_count_threshold", r"don hang .*(tro len|tren|duoi)|don .*tien hang tu"),
    ("top_customers", r"khach .*(nhieu tien|chi tieu|mua nhieu)|ai mua"),
    ("customer_count", r"bao nhieu khach|may khach|so khach"),
    ("category_share", r"ti trong|ty trong|chiem bao nhieu|dong gop|% tong doanh thu"),
    ("category_revenue", r"doanh thu nhom|nhom hang .*doanh thu"),
    ("best_seller", r"ban chay|ban nhieu nhat|hot nhat|best seller|ban duoc nhieu nhat|top \d+ (sp|san pham|mat hang|mon)"),
    ("best_day_month", r"ngay nao|thang nao"),
    ("size", r"size|\(m\)|\(l\)"),
    ("unit_price", r"gia ban|don gia|gia bao nhieu|ban .*gia"),
    ("increase_check", r"tang lien tuc|giam lien tuc"),
    ("growth", r"tang truong|thay doi|hon hay kem|tang hay giam|so voi|so cung ky|chenh lech"),
    ("aov", r"trung binh|binh quan"),
    ("order_count", r"\bdon\b|bill|hoa don"),
    ("item_qty", r"ban ra|ban duoc .*(cai|chai|hop|thung|cay|ly|goi|kg|bao|cuon|vien|lon|m\b)|so kg|bao nhieu (cai|chai|hop|thung|cay|ly|goi|bao|cuon|vien|lon)"),
    ("revenue", r"doanh thu|ban duoc .*tien|bao nhieu tien|thu ve|tien ban|tien hang"),
]
RULES = [(n, re.compile(p)) for n, p in RULES]


def classify(q: str) -> str:
    n = norm(q)
    for name, rx in RULES:
        if rx.search(n):
            return name
    return "OTHER"


def main():
    qs = [json.loads(l) for l in open(TRAIN / "questions.jsonl", encoding="utf8")]
    ans = {json.loads(l)["id"]: json.loads(l) for l in open(TRAIN / "answers.jsonl", encoding="utf8")}
    by = defaultdict(list)
    for x in qs:
        by[classify(x["question"])].append(x)
    show = sys.argv[sys.argv.index("--show") + 1] if "--show" in sys.argv else None
    if show:
        for x in by[show][:40]:
            a = ans[x["id"]]
            print(x["business_id"], a["type"][:5], "|", x["question"], "=>", json.dumps(a["answer"], ensure_ascii=False)[:80])
        return
    print(f"{'intent':24}{'n':>5}{'unans':>7}  types   shops")
    for name, items in sorted(by.items(), key=lambda kv: -len(kv[1])):
        types = Counter(ans[x["id"]]["type"] for x in items)
        un = types.pop("unanswerable", 0)
        shops = len({x["business_id"] for x in items})
        print(f"{name:24}{len(items):5}{un:7}  {dict(types)}  {shops}")
    print("total", len(qs), "intents", len(by))


if __name__ == "__main__":
    main()
