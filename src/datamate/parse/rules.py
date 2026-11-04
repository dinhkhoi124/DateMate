"""Rule parser: normalised Vietnamese question -> Intent. Ordered detectors, first match wins.

No shop names, ids or question ids appear here (CLAUDE.md rule 3): entities come from the shop catalog via link/.
"""
from __future__ import annotations

import re
from datetime import date

from datamate.execute.ops import channel_class
from datamate.link.catalog import Linker
from datamate.parse.intent import Intent
from datamate.timeparse.periods import Period

PAY_WORDS = [("cash", r"tien mat|tien tuoi"), ("transfer", r"chuyen khoan"), ("cod", r"\bcod\b"),
             ("ewallet", r"vi dien tu"), ("card", r"\bquet the\b|\bthe\b"), ("debt", r"ghi no")]
CHANNEL_WORDS = [("zalo", r"zalo(?: oa)?"), ("phone", r"dien thoai|hotline"), ("store", r"tai quay|tai cua hang|truc tiep"),
                 ("facebook", r"facebook|\bfb\b"), ("marketplace", r"shopee|lazada|tiki|san tmdt|san thuong mai")]
UNIT_WORDS = ("ly", "chai", "hop", "thung", "cay", "goi", "kg", "bao", "cuon", "vien", "lon", "cai", "dia", "phan", "to",
              "noi", "binh", "tuyp", "lo", "bo", "m3", "m2", "con", "vi", "ong", "mieng", "cuc", "loc", "tam", "suat",
              "chiec", "khay", "bich", "tui", "can", "met", "m", "vi", "xap", "lit", "hu", "ram", "tep", "doi", "chuc")
SUBJECTIVE = r"le phep|chu dao|quy men|lanh nghe|tay nghe|\bkhen\b|hai long|nhiet tinh|vui ve|dep trai|uy tin|tot bung|than thien|de thuong|cham chi"


def _num(s) -> float:
    return float(str(s).replace(",", "."))


class RuleParser:
    def __init__(self, lk: Linker, units: set[str] | None = None):
        self.lk = lk
        self.units = set(UNIT_WORDS) | set(units or ())

    # ------------------------------------------------------------------ entities / filters
    def entities(self, t: str, pre: str | None = None) -> dict:
        lk = self.lk
        ent = {}
        ph = lk.products.find(t)
        if pre is not None and pre != t:
            ph2 = lk.products.find(pre)
            if sum(h.end - h.start for h in ph2) > sum(h.end - h.start for h in ph):
                ph = ph2
                ent["products_from_pre"] = True
        ent["products"] = ph
        cats = []
        for m in list(re.finditer(r"nhom (?:hang |mat hang |san pham )?", t)) +                 list(re.finditer(r"nhom (?:hang |mat hang |san pham )?", pre or "")):
            tail = m.string[m.end():]
            for h in lk.categories.find(tail, limit=1):
                if h.start == 0:
                    cats.append(h.key)
                    break
        ent["category"] = cats[0] if cats else None
        m = re.search(r"hoat chat (.+)", t)
        if m:
            hs = lk.ingredients.find(m.group(1), limit=1)
            ent["ingredient"] = hs[0].key if hs and hs[0].start == 0 else ("?" + m.group(1)[:40])
        m = re.search(r"\bnv ?0*(\d+)\b", t)
        if m:
            hs = [h for h in lk.staff.find(t) if h.key]
            ent["staff"] = hs[0].key if hs else ("NV%03d" % int(m.group(1)))
        ch = [(m.start(), k) for k, p in CHANNEL_WORDS for m in re.finditer(p, t)]
        ent["channels"] = [k for _, k in sorted(ch)]
        pays = [(m.start(), k) for k, p in PAY_WORDS for m in re.finditer(p, t)]
        ent["pays"] = [k for _, k in sorted(pays)]
        g = re.search(r"khach (?:hang )?(?:thuoc )?(?:nhom )?(than thiet|si|vip|quen|le|moi)\b", t)
        if g and (("nhom" in t) or g.group(1) in ("than thiet", "si", "vip")):
            ent["group"] = g.group(1)
        if re.search(r"ca sang", t):
            ent.setdefault("shifts", []).append(("morning", t.index("ca sang")))
        if re.search(r"ca chieu", t):
            ent.setdefault("shifts", []).append(("afternoon", t.index("ca chieu")))
        m = re.search(r"size (m|l)\b", t)
        if m:
            ent["size"] = m.group(1)
        if re.search(r"(?:khong|chua) (?:ke don|can don)|khong can don bac si|tru thuoc ke don", t):
            ent["rx"] = False
        elif re.search(r"(?:phai )?ke don", t):
            ent["rx"] = True
        cu = lk.customers.find(t)
        ent["customers"] = cu
        return ent

    @staticmethod
    def product_of(ent) -> object:
        ph = ent["products"]
        if not ph:
            return None
        if any(h.ambiguous for h in ph):
            return "?ambiguous"
        keys = {h.key for h in ph}
        return ph[-1].key if len(keys) == 1 else "?multi"

    def base_filters(self, ent, t, product=True) -> dict:
        flt = {}
        if product and ent["products"]:
            flt["product"] = self.product_of(ent)
        for k in ("category", "ingredient", "staff", "group", "size", "rx"):
            if ent.get(k) is not None:
                flt[k] = ent[k]
        if len(ent["channels"]) == 1:
            flt["channel"] = ent["channels"][0]
        if len(ent["pays"]) == 1:
            flt["pay"] = ent["pays"][0]
        sh = ent.get("shifts", [])
        if len(sh) == 1:
            flt["shift"] = sh[0][0]
        return flt

    def unit_in(self, t) -> str | None:
        m = re.search(r"(?:bao nhieu|may|so|tinh theo|theo|nhieu) (" + "|".join(sorted(self.units, key=len, reverse=True)) + r")\b", t)
        return m.group(1) if m else None

    # ------------------------------------------------------------------ main
    def parse(self, t: str, periods: list[Period], as_of: date, pre: str | None = None) -> Intent | None:
        ent = self.entities(t, pre)
        I = lambda op, atype="number", **kw: Intent(op=op, periods=list(periods), atype=atype, **kw)  # noqa: E731
        # keywords are searched with product names masked out ('Lẩu gà lá é' must not trigger 'ế')
        tm = t
        if not ent.get("products_from_pre"):
            for h in sorted(ent["products"], key=lambda h: -h.start):
                tm = tm[:h.start] + " _sp_ " + tm[h.end:]
        has = lambda p: re.search(p, tm) is not None  # noqa: E731

        # ---- subjective / not in data (rulebook §7)
        if has(SUBJECTIVE):
            return I("unanswerable", note="subjective")
        if has(r"khach (?:hang )?(nam|nu)\b"):
            return I("need_cap", args={"cap": "has_gender"})
        if has(r"\bduong [a-z]") and has(r"khach"):
            return I("need_cap", args={"cap": "has_address"})
        if has(r"\bvi (loi|mua nham|giao tre|het hang|khach doi)|ly do"):
            return I("unanswerable", note="reason not recorded")

        # ---- price effect (rulebook §5 hiệu ứng tăng giá)
        if has(r"(tang|doi|len) gia") and has(r"(\d+) ngay"):
            n = int(re.search(r"(\d+) ngay", t).group(1))
            m = re.search(r"(?:len|gia) (\d+(?:\.\d+)?) ?d\b", t)
            newp = float(m.group(1)) if m else None
            if re.search(r"len (\d+(?:\.\d+)?) ?d", t):
                newp = float(re.search(r"len (\d+(?:\.\d+)?) ?d", t).group(1))
            day = [p for p in periods if p.kind == "day"]
            return I("price_effect", "percent", flt={"product": self.product_of(ent)},
                     args={"n": n, "new_price": newp, "day": day[0].start if day else None})

        # ---- imports (rulebook §5)
        if has(r"gia nhap binh quan|nhap vao gia binh quan|nhap .{0,90}binh quan|binh quan .{0,20}nhap"):
            return I("avg_import_price", flt={"product": self.product_of(ent)}, args={"unit": self.per_unit(t)})
        if has(r"nhap tru (so )?ban|chenh lech nhap"):
            return I("import_minus_sales", flt={"product": self.product_of(ent)})
        if has(r"nha cung cap"):
            return I("top_supplier", "string")
        if has(r"nhap hang|phieu nhap|tien nhap|hang nhap vao|tien hang nhap"):
            return I("import_total")

        # ---- expiry / lots (rulebook §6 nhà thuốc)
        if has(r"het han\b|han dung"):  # \b: 'het hang' (out of stock, §5) is not 'het han' (expiry, §6)
            if has(r"lo .{0,80}gan het han nhat|gan het han nhat"):
                return I("nearest_expiry", "date", flt={"product": self.product_of(ent)})
            m = re.search(r"(\d+) ngay toi", t)
            if has(r"(may|bao nhieu) lo"):
                return I("expiring_lots_count", args={"days": int(m.group(1)) if m else None})
            if has(r"tien von|gia tri"):
                return I("expiring_value", args={"days": int(m.group(1)) if m else None})
            return I("expiring_products", "list", args={"days": int(m.group(1)) if m else None})

        # ---- stock (rulebook §5)
        if has(r"het hang") and has(r"(may|bao nhieu|so) (mat hang|san pham|mon)"):
            return I("stock_out_count")
        if (has(r"gia von") and has(r"kho|ton")) or has(r"tien hang (nam )?trong kho|gia tri (hang )?ton"):
            return I("stock_value")
        if has(r"\b(chan|nguyen)\b") and has(r"\ble\b") and has(r"kho|ton|con"):
            return I("stock_big", "list", flt={"product": self.product_of(ent)}, args={"big": self.big_unit(t)})
        if has(r"ton kho|con ton|trong kho|con lai may|con lai bao nhieu|con bao nhieu|con may|ton cuoi") and ent["products"]:
            return I("stock_qty", flt={"product": self.product_of(ent)}, args={"unit": self.unit_in(t)})

        # ---- debt (rulebook §4)
        if has(r"\bno\b|cong no|mua chiu|con thieu|chua tra het|chua thu"):
            cust = self.customer_of(ent, t)
            if has(r"no them|mua chiu them"):
                return I("debt_customer_flow", "list", flt={"customer": cust})
            if has(r"tuoi no") and has(r"0-7|0 - 7|8-15|8 - 15"):
                return I("debt_aging", "list")
            m = re.search(r"(?:qua|hon|tren) (\d+) ngay", t)
            if m:
                return I("debt_total", args={"min_age": int(m.group(1))})
            m = re.search(r"(\d+) khach", t)
            if m and has(r"nhieu nhat|lon nhat|cao nhat"):
                return I("debt_table", "table", args={"k": int(m.group(1))})
            if has(r"(bao nhieu|may|so) khach"):
                return I("debt_customers")
            if has(r"tra no|thu no|tra bot no|thu hoi|khach tra|tra duoc"):
                return I("debt_collected")
            return I("debt_total")

        # ---- cancellations (rulebook §1)
        if has(r"\bhuy\b"):
            if has(r"%|phan tram|ti le"):
                return I("cancel_rate", "percent", flt=self.base_filters(ent, t))
            return I("cancel_count", flt=self.base_filters(ent, t))

        # ---- profit (rulebook §5)
        if has(r"lai rong|loi nhuan rong|tru (het )?chi phi"):
            return I("net_profit")
        if has(r"(loi|lai) nhat|lai gop cao nhat"):
            return I("top_products", "string", args={"by": "profit", "k": 1})
        if has(r"lai gop|loi gop|loi nhuan|tien lai"):
            return I("gross_profit", flt=self.base_filters(ent, t))

        # ---- deliveries (rulebook §6 VLXD)
        if has(r"dot (giao|hang)|dot giao"):
            return I("delivery_count")

        # ---- Tết (rulebook §2)
        if has(r"\btet\b"):
            m = re.search(r"(\d+) ngay truoc tet", t)
            if m or has(r"truoc tet"):
                return I("tet_vs_before", "percent", args={"n": int(m.group(1)) if m else None, "metric": self.metric(t)})
            return I("growth", "percent", args={"metric": self.metric(t), "compare": "explicit"})

        # ---- thresholds
        m = re.search(r"(?:it nhat|tu|tren|hon|qua|duoi) (\d+) don", t)
        if m and has(r"khach"):
            n, inc = int(m.group(1)), self.inclusive(t, m)
            if has(r"%|phan tram|ti le"):
                return I("customers_min_orders_pct", "percent", args={"n": n, "inclusive": inc})
            return I("customers_min_orders", args={"n": n, "inclusive": inc})
        if m and has(r"mat hang|san pham|\bmon\b"):
            return I("items_in_n_orders", args={"n": int(m.group(1)), "inclusive": self.inclusive(t, m)},
                     flt=self.base_filters(ent, t, product=False))
        m2 = re.search(r"(\d+) don (?:hang )?(?:khac nhau )?(?:tro len)", t) or re.search(r"(?:o|trong) (?:tu )?(\d+) don", t)
        if m2 and has(r"mat hang|san pham|\bmon\b") and has(r"xuat hien|co mat|duoc mua|mua o"):
            return I("items_in_n_orders", args={"n": int(m2.group(1)), "inclusive": True},
                     flt=self.base_filters(ent, t, product=False))
        m = re.search(r"(?:tu|tren|hon|it nhat|dat) (\d{4,}) ?(?:dong|d\b)", t) or re.search(r"(?:tu) (\d{4,})", t)
        if m and has(r"(bao nhieu|may|so|co nhieu) ngay"):
            return I("days_revenue_over", args={"x": float(m.group(1)), "inclusive": self.inclusive(t, m)},
                     flt=self.base_filters(ent, t))
        if m and has(r"don"):
            return I("orders_revenue_over", args={"x": float(m.group(1)), "inclusive": self.inclusive(t, m)},
                     flt=self.base_filters(ent, t))

        # ---- dead stock (rulebook §5)
        if has(r"\be\b|ban cham|khong ban (ra )?duoc|khong ai mua|ban khong duoc|e han"):
            return I("dead_count")

        # ---- unit price on a day (rulebook §5)
        if has(r"gia ban|don gia|gia bao nhieu|gia (la )?bao nhieu|ban .{0,60} gia") and ent["products"]:
            day = [p for p in periods if p.kind == "day"]
            return I("unit_price", flt={"product": self.product_of(ent)}, args={"day": day[0].start if day else None})

        metric = self.metric(t)
        # ---- trend checks
        if has(r"tang lien tuc|giam lien tuc|luon cao hon|co phai .{0,80}cao hon|thang sau .{0,30}cao hon"):
            return I("monotonic", "boolean", args={"metric": metric, "dir": "down" if has("giam lien tuc") else "up"},
                     flt=self.base_filters(ent, t))
        # ---- argmax over days / months
        if has(r"ngay nao"):
            return I("argmax_day", "date", args={"metric": metric}, flt=self.base_filters(ent, t))
        if has(r"thang nao"):
            return I("argmax_month", "date", args={"metric": metric}, flt=self.base_filters(ent, t))

        # ---- staff
        if has(r"nhan vien (nao|ban gioi)|nguoi ban .{0,30}(cao nhat|nhieu nhat)|ai la nguoi ban|nhan vien .{0,20}cao nhat"):
            return I("top_staff", "string", flt=self.base_filters(ent, t))

        # ---- rankings
        k = self.top_k(t)
        if (has(r"khach|\bai\b")) and has(r"(nhieu tien|chi tieu|chi nhieu|tong tien mua|mua nhieu|mua hang nhieu|nhieu don|nhieu lan|so don mua).{0,30}(nhat|cao nhat)|top \d+|\d+ khach"):
            by = "orders" if has(r"nhieu don|nhieu lan|so don") else "revenue"
            return I("top_customers", "list" if k else "string", args={"k": k or 1, "by": by},
                     flt=self.base_filters(ent, t, product=False))
        if has(r"ban chay|chay nhat|ban nhieu nhat|hot nhat|ban duoc nhieu nhat|best seller"):
            return I("top_products", "list" if k else "string", args={"k": k or 1, "by": "qty"},
                     flt=self.base_filters(ent, t, product=False))

        # ---- comparisons
        sh = ent.get("shifts", [])
        if len(sh) == 2:
            return I("shift_compare", "percent", args={"metric": metric}, flt=self.base_filters(ent, t))
        if has(r"tang truong|thay doi|hon hay kem|tang hay giam|tang/giam|chenh|nhieu hay it|so voi|so cung ky") and has(r"%|phan tram"):
            cmp = "prev"
            if has(r"cung ky nam|cung thang nam|nam ngoai|nam truoc") and not has(r"cung ky thang"):
                cmp = "yoy"
            if len([p for p in periods if p.kind in ("month", "quarter", "tet", "range", "week", "day")]) >= 2:
                cmp = "explicit"
            return I("growth", "percent", args={"metric": metric, "compare": cmp}, flt=self.base_filters(ent, t))

        # ---- multi-value lists (channels / payments / months)
        if len(set(ent["channels"])) >= 2 and metric == "revenue":
            return I("revenue_by", "list", args={"dim": "channel", "values": ent["channels"]}, flt=self.base_filters(ent, t))
        if len(set(ent["pays"])) >= 2 and metric == "revenue":
            return I("revenue_by", "list", args={"dim": "pay", "values": ent["pays"]}, flt=self.base_filters(ent, t))

        # ---- shares (rulebook §1 tỉ trọng: mẫu số là doanh thu cả shop)
        if has(r"chiem|ti trong|dong gop|bang .{0,30}(%|phan tram)|(%|phan tram) tong doanh thu|phan cua|ti le doanh thu|tren tong doanh thu"):
            return I("share", "percent", flt=self.base_filters(ent, t))

        # ---- averages
        if has(r"trung binh|binh quan") and has(r"\bdon\b"):
            return I("aov", flt=self.base_filters(ent, t))

        months = [p for p in periods if p.kind == "month"]
        if len(months) >= 2 and metric == "revenue":
            return I("revenue_series", "list", flt=self.base_filters(ent, t))

        # ---- quantities
        unit = self.unit_in(t)
        if unit and not has(r"bao nhieu tien|may tien|\btien\b|doanh thu"):
            flt = self.base_filters(ent, t)
            if flt.get("product") is None and not flt.get("size") and not flt.get("category"):
                flt["base_unit"] = unit
            return I("qty", flt=flt, args={"unit": unit})

        # ---- counts
        if metric == "orders":
            return I("order_count", flt=self.base_filters(ent, t))
        if metric == "revenue":
            return I("revenue", flt=self.base_filters(ent, t))
        return None

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def metric(t: str) -> str:
        if re.search(r"doanh thu|\btien\b|thu ve|thu duoc|ban duoc .{0,10}tien|ban dat", t):
            return "revenue"
        if re.search(r"(so|bao nhieu|may|tong cong) don|\bdon hang\b|so don|\bdon\b", t):
            return "orders"
        if re.search(r"ban duoc|ban ra|so luong", t):
            return "qty"
        return "revenue" if re.search(r"ban|thu", t) else "?"

    @staticmethod
    def top_k(t: str) -> int | None:
        m = re.search(r"top (\d+)|(?<![\d/])(\d{1,2}) (?:mat hang|san pham|mon|hang|sp|khach|nhan vien)\b", t)
        if m:
            return int(m.group(1) or m.group(2))
        return None

    @staticmethod
    def inclusive(t: str, m) -> bool:
        """rulebook §2: quá/hơn/trên N không gồm N; từ N trở lên, ít nhất N, ≥ N gồm N."""
        w = t[max(0, m.start() - 1):m.end() + 12]
        if re.search(r"\b(qua|hon|tren)\b", w) and not re.search(r"tro len", w):
            return False
        return True

    def per_unit(self, t: str) -> str | None:
        m = re.search(r"(?:mot|1|moi) (" + "|".join(sorted(self.units, key=len, reverse=True)) + r")\b", t)
        return m.group(1) if m else None

    def big_unit(self, t: str) -> str | None:
        m = re.search(r"(?:bao nhieu|may|duoc may|duoc bao nhieu) (" + "|".join(sorted(self.units, key=len, reverse=True)) +
                      r")(?: [a-z0-9 ,.%()/+-]{0,80}?)? (?:chan|nguyen)\b", t)
        return m.group(1) if m else None

    @staticmethod
    def customer_of(ent, t):
        cu = [h for h in ent["customers"]]
        if not cu:
            return None
        if any(h.ambiguous for h in cu):
            return "?ambiguous"
        return max(cu, key=lambda h: h.end - h.start).key
