"""Deterministic business ops (rulebook §1–§6). Every number in an answer is computed here.

Conventions: periods are inclusive dates; records after `as_of` have not happened yet (rulebook preamble), so every
flow window is clipped at as_of. Ops raise Unanswerable (deterministic: missing capability / period outside data)
or Abstain (we are not sure).
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from datamate.execute.facts import Facts, day64
from datamate.ingest.values import fold


class Unanswerable(Exception):
    pass


class Abstain(Exception):
    pass


# ------------------------------------------------------------------ rounding (rulebook §9)
def money(x) -> int:
    """rulebook §9: tiền làm tròn về đồng, nửa đồng làm tròn lên."""
    return int(math.floor(float(x) + 0.5 + 1e-9))


def r2(x) -> float:
    """rulebook §9: phần trăm / trung bình khác làm tròn 2 chữ số (half up)."""
    v = math.floor(abs(float(x)) * 100 + 0.5 + 1e-7) / 100
    return math.copysign(v, x) if v else 0.0


def pct_change(a, b) -> float:
    """rulebook §1: phần trăm thay đổi so với kỳ trước; giảm ra số âm."""
    if not b:
        raise Abstain("base is zero")
    return r2((a - b) / b * 100)


def pct(a, b) -> float:
    if not b:
        raise Abstain("denominator is zero")
    return r2(a / b * 100)


# ------------------------------------------------------------------ period guards
def clip(f: Facts, s: date, e: date, as_of: date):
    """Clip a flow window to as_of; unanswerable when it lies completely outside the data (rulebook §7)."""
    e = min(e, as_of)
    if f.dmin and e < f.dmin:
        raise Unanswerable("period before data")
    if f.dend and s > f.dend:
        raise Unanswerable("period after data")
    if s > e:
        raise Unanswerable("period in the future")
    return s, e


# ------------------------------------------------------------------ order / line selection
def order_filter(f: Facts, flt: dict) -> np.ndarray:
    """Order-level filters: customer group, channel, payment, staff, shift, customer."""
    O = f.O
    m = np.ones(len(O), dtype=bool)
    if flt.get("group"):
        if not f.caps.get("has_customer_group"):
            raise Unanswerable("no customer group")
        gm = f.customer_groups()
        want = fold(flt["group"])
        # rulebook §1: walk-in orders belong to no group, not even 'lẻ'
        m &= np.array([isinstance(k, str) and fold(gm.get(k, "")) == want for k in O["customer_id"]])
    if flt.get("channel"):
        if not f.caps.get("has_channel"):
            raise Unanswerable("no channel")
        m &= O["channel"].map(lambda c: channel_class(c) == flt["channel"]).values
    if flt.get("pay"):
        if not f.caps.get("has_payment_method"):
            raise Unanswerable("no payment method")
        m &= (O["pay"] == flt["pay"]).values
    if flt.get("staff"):
        if not f.caps.get("has_staff"):
            raise Unanswerable("no staff")
        m &= (O["staff_id"].map(lambda s: fold(s) if isinstance(s, str) else "") == fold(flt["staff"])).values
    if flt.get("shift"):
        if not f.caps.get("has_time"):
            raise Unanswerable("no order time")
        lo, hi = f.rules.morning if flt["shift"] == "morning" else f.rules.afternoon
        tod = O["tod"].values.astype(float)
        m &= (tod >= lo) & (tod <= hi)
    if flt.get("customer") is not None:
        m &= (O["ckey"] == flt["customer"]).values
    return m


def line_filter(f: Facts, flt: dict) -> np.ndarray | None:
    """Line-level filters: product, category, ingredient, rx, size. None when no line filter is set."""
    keys = ("product", "category", "ingredient", "rx", "size", "base_unit", "products")
    if not any(flt.get(k) is not None for k in keys):
        return None
    if not f.has_lines:
        raise Unanswerable("no line items")
    L = f.L
    m = np.ones(len(L), dtype=bool)
    if flt.get("product") is not None:
        m &= (L["product_key"] == flt["product"]).values
    if flt.get("products") is not None:
        m &= L["product_key"].isin(flt["products"]).values
    if flt.get("category") is not None:
        if not f.caps.get("has_category"):
            raise Unanswerable("no category")
        m &= (L["category"].map(lambda c: fold(c) if isinstance(c, str) else "") == fold(flt["category"])).values
    if flt.get("ingredient") is not None:
        # rulebook §6 nhà thuốc: hoạt chất ghi đúng như vậy; thuốc phối hợp là loại riêng
        ing = f.P["active_ingredient"].to_dict()
        m &= L["product_key"].map(lambda k: fold(ing.get(k) or "") == fold(flt["ingredient"])).values
    if flt.get("rx") is not None:
        if "is_rx" not in f.P or f.P["is_rx"].isna().all():
            raise Unanswerable("no rx flag")
        rx = f.P["is_rx"].to_dict()
        isrx = L["product_key"].map(lambda k: rx.get(k) is True).values
        m &= isrx if flt["rx"] else ~isrx
    if flt.get("size"):
        sz = flt["size"].lower()
        m &= L["product_name"].map(lambda n: isinstance(n, str) and n.strip().lower().endswith(f"({sz})")).values
    if flt.get("base_unit"):
        bu = fold(flt["base_unit"])
        m &= L["base_unit"].map(lambda u: fold(u) == bu if isinstance(u, str) else False).values
    return m


def channel_class(c) -> str | None:
    if not isinstance(c, str):
        return None
    c = fold(c).replace("_", " ")
    for k, pats in (("zalo", ("zalo",)), ("phone", ("phone", "dien thoai", "hotline", "goi")),
                    ("store", ("store", "quay", "offline", "pos", "cua hang", "truc tiep")),
                    ("facebook", ("facebook", "fb")), ("online", ("online", "web")),
                    ("marketplace", ("shopee", "lazada", "tiki", "tiktok", "san"))):
        if any(p in c for p in pats):
            return k
    return c


def revenue(f: Facts, s, e, flt: dict) -> float:
    """rulebook §1: doanh thu = Σ thành tiền dòng của đơn không huỷ, không trả; theo ngày đặt."""
    om = f.omask(s, e) & order_filter(f, flt)
    lm = line_filter(f, flt)
    if lm is None:
        return float(f.O["rev"].values[om].sum())
    ids = set(f.O["order_id"].values[om])
    L = f.L
    sel = lm & L["valid"].values & L["order_id"].isin(ids).values
    return float(L["amount"].values[sel].sum())


def order_count(f: Facts, s, e, flt: dict) -> int:
    """rulebook §1: số đơn chỉ tính đơn không huỷ, không trả."""
    om = f.omask(s, e) & order_filter(f, flt)
    lm = line_filter(f, flt)
    if lm is not None:
        ids = set(f.L["order_id"].values[lm])
        om &= f.O["order_id"].isin(ids).values
    return int(om.sum())


def cancel_count(f: Facts, s, e, flt: dict) -> int:
    """rulebook §1: đơn huỷ = trạng thái huỷ; đơn trả hàng không phải đơn huỷ."""
    m = f.omask(s, e, valid_only=False) & f.O["cancelled"].values & order_filter(f, flt)
    return int(m.sum())


def cancel_rate(f: Facts, s, e, flt: dict) -> float:
    """rulebook §1: % đơn huỷ trên mọi đơn đặt trong kỳ (mọi trạng thái)."""
    allm = f.omask(s, e, valid_only=False) & order_filter(f, flt)
    return pct(int((allm & f.O["cancelled"].values).sum()), int(allm.sum()))


def qty(f: Facts, s, e, flt: dict) -> float:
    """rulebook §3: số lượng quy về đơn vị cơ bản; bỏ đơn huỷ, trả."""
    if not f.has_lines:
        raise Unanswerable("no lines")
    lm = line_filter(f, flt)
    if lm is None:
        lm = ~f.L["is_fee"].values
    om = f.omask(s, e) & order_filter(f, flt)
    ids = set(f.O["order_id"].values[om])
    L = f.L
    sel = lm & L["valid"].values & L["order_id"].isin(ids).values
    return float(L["qty_base"].values[sel].sum())


def valid_lines(f: Facts, s, e, flt: dict) -> pd.DataFrame:
    if not f.has_lines:
        raise Unanswerable("no lines")
    lm = line_filter(f, flt)
    om = f.omask(s, e) & order_filter(f, flt)
    ids = set(f.O["order_id"].values[om])
    L = f.L
    sel = L["valid"].values & L["order_id"].isin(ids).values & ~L["is_fee"].values
    if lm is not None:
        sel &= lm
    return L[sel]


def top_products(f: Facts, s, e, flt: dict, by="qty") -> list:
    """rulebook §3: bán chạy theo SL quy về đơn vị cơ bản; Phí phục vụ không xếp hạng; hoà thì theo mã."""
    V = valid_lines(f, s, e, flt)
    if by == "qty":
        agg = V.groupby("product_key")["qty_base"].sum()
    elif by == "revenue":
        agg = V.groupby("product_key")["amount"].sum()
    elif by == "profit":
        if V["cost_unit"].isna().all():
            raise Unanswerable("no cost")
        agg = (V["amount"] - V["qty"] * V["cost_unit"]).groupby(V["product_key"]).sum()
    else:
        raise Abstain(by)
    items = [(k, float(v)) for k, v in agg.items() if v > 0]
    items.sort(key=lambda kv: (-round(kv[1], 6), f.sort_key_product(kv[0])))
    return items


def customers_agg(f: Facts, s, e, flt: dict, by="revenue") -> list:
    """rulebook §3: khách theo mã (hoặc tên đã chuẩn hoá); khách lẻ không xếp hạng."""
    if not f.caps.get("has_customers"):
        raise Unanswerable("no customers")
    om = f.omask(s, e) & order_filter(f, flt) & f.O["ckey"].notna().values
    D = f.O[om]
    if not len(D):
        return []
    agg = D.groupby("ckey")["rev"].sum() if by == "revenue" else D.groupby("ckey")["order_id"].nunique()
    items = [(k, float(v)) for k, v in agg.items()]
    items.sort(key=lambda kv: (-kv[1], str(kv[0])))
    return items


def staff_agg(f: Facts, s, e, flt: dict) -> list:
    if not f.caps.get("has_staff"):
        raise Unanswerable("no staff")
    om = f.omask(s, e) & order_filter(f, flt) & f.O["staff_id"].notna().values
    agg = f.O[om].groupby("staff_id")["rev"].sum()
    items = [(k, float(v)) for k, v in agg.items()]
    items.sort(key=lambda kv: (-kv[1], str(kv[0])))
    return items


def daily_revenue(f: Facts, s, e, flt: dict) -> pd.Series:
    om = f.omask(s, e) & order_filter(f, flt)
    lm = line_filter(f, flt)
    if lm is None:
        D = f.O[om]
        return D.groupby("d")["rev"].sum()
    ids = set(f.O["order_id"].values[om])
    L = f.L[lm & f.L["valid"].values & f.L["order_id"].isin(set(ids)).values]
    return L.groupby("d")["amount"].sum()


def dead_items(f: Facts, s, e) -> list:
    """rulebook §5: mặt hàng đã từng bán trước kỳ (đơn §1) nhưng trong kỳ không có đơn §1 nào mua."""
    L = f.L
    base = L["valid"].values & ~L["is_fee"].values & (L["qty_base"].values > 0)
    dd = L["d"].values.astype("datetime64[D]")
    before = set(L.loc[base & (dd < day64(s)), "product_key"])
    during = set(L.loc[base & (dd >= day64(s)) & (dd <= day64(e)), "product_key"])
    return sorted(before - during, key=f.sort_key_product)


def items_in_n_orders(f: Facts, s, e, n: int, flt: dict) -> int:
    V = valid_lines(f, s, e, flt)
    c = V[V["qty_base"] > 0].groupby("product_key")["order_id"].nunique()
    return int((c >= n).sum())


def unit_price_on(f: Facts, key, d: date) -> float:
    """rulebook §5: đơn giá trên dòng bán ngày D, cho một đơn vị cơ bản, trước chiết khấu; tính cả đơn huỷ/trả."""
    L = f.L
    dd = L["d"].values.astype("datetime64[D]")
    sel = (L["product_key"] == key).values & (dd == day64(d)) & (L["qty"].values > 0)
    V = L[sel]
    if not len(V):
        raise Abstain("no sale that day")
    fac = V["factor"].where(V["factor"].notna() & (V["factor"] > 0), 1.0)
    p = (V["unit_price"] / fac).round(4)
    vc = p.value_counts()
    if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
        raise Abstain("several prices")
    return float(vc.index[0])


def shift_of(text: str):
    return None
