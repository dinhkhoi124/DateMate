"""Imports, stock, prices (rulebook §5). Stock snapshot in the files is the stock at the end of the last data day
(README 'Tồn cuối ... cuối ngày cuối có dữ liệu'); stock at an earlier day D is rolled back from it with the imports
and sales recorded after D.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from datamate.execute.facts import Facts, day64
from datamate.execute.ops import Abstain, Unanswerable
from datamate.ingest.values import fold


def _factor(f: Facts, key, unit) -> float | None:
    """Units with no conversion row are the base unit, factor 1 (rulebook §10)."""
    if not isinstance(unit, str) or not unit:
        return 1.0
    fu = fold(unit)
    base = (getattr(f.st, "base_units", {}) or {}).get(key)
    if base is None and key in f.P.index:
        base = f.P.at[key, "base_unit"]
    if isinstance(base, str) and fold(base) == fu:
        return 1.0
    fac = (getattr(f.st, "factors", {}) or {}).get((key, fu))
    if fac:
        return float(fac)
    U = f.st.units
    if U is not None and len(U):
        hit = U[(U["product_key"] == key) & (U["unit"].map(lambda u: fold(u) if isinstance(u, str) else "") == fu)]
        if len(hit):
            return float(hit["factor"].iloc[0])
    return 1.0


def base_unit(f: Facts, key) -> str | None:
    b = (getattr(f.st, "base_units", {}) or {}).get(key)
    if b is None and key in f.P.index:
        b = f.P.at[key, "base_unit"]
    if not isinstance(b, str):
        S = f.st.stock
        if S is not None and len(S):
            hit = S.loc[S["product_key"] == key, "unit"].dropna()
            b = hit.iloc[0] if len(hit) else None
    return b if isinstance(b, str) else None


def import_lines(f: Facts) -> pd.DataFrame:
    """Import lines of non-cancelled receipts with day stamp and base quantity."""
    IL = f.st.extra.get("_il")
    if IL is not None:
        return IL
    IL = f.st.import_lines.copy()
    IL["d"] = np.array([], dtype="datetime64[D]") if not len(IL) else None
    H = f.st.imports
    canc = set(H.loc[H["cancelled"].astype(bool), "receipt_id"]) if H is not None and len(H) else set()
    if len(IL):
        if "import_status" in IL:
            from datamate.ingest.values import CANCELLED, parse_status
            canc |= set(IL.loc[IL["import_status"].map(lambda s: parse_status(s) == CANCELLED if isinstance(s, str) else False),
                               "receipt_id"])
        IL = IL[~IL["receipt_id"].isin(canc)].copy()
        IL["d"] = np.array([np.datetime64(v, "D") if v is not None and v == v else np.datetime64("NaT")
                            for v in IL["import_date"]], dtype="datetime64[D]")
        qb = pd.to_numeric(IL["qty_base"], errors="coerce")
        fac = [_factor(f, k, u) for k, u in zip(IL["product_key"], IL["unit"])]
        IL["qty_base"] = qb.where(qb.notna(), IL["qty"].astype(float) * np.array(fac, dtype=float))
        IL["amount"] = pd.to_numeric(IL["amount"], errors="coerce")
    f.st.extra["_il"] = IL
    return IL


def receipts(f: Facts) -> pd.DataFrame:
    H = f.st.imports
    if H is None or not len(H):
        IL = import_lines(f)
        if not len(IL):
            raise Unanswerable("no imports")
        H = IL.groupby("receipt_id").agg(import_date=("import_date", "first"), supplier=("supplier", "first"),
                                         supplier_id=("supplier_id", "first"), receipt_total=("amount", "sum")).reset_index()
        H["cancelled"] = False
    H = H[~H["cancelled"].astype(bool)].copy()
    H["d"] = np.array([np.datetime64(v, "D") if v is not None and v == v else np.datetime64("NaT")
                       for v in H["import_date"]], dtype="datetime64[D]")
    return H


def import_total(f: Facts, s: date, e: date) -> float:
    """rulebook §5: tổng tiền các phiếu nhập có ngày phiếu trong kỳ, bỏ phiếu huỷ (nhập đầu kỳ vẫn tính)."""
    if not f.caps.get("has_imports"):
        raise Unanswerable("no imports")
    H = receipts(f)
    m = (H["d"].values >= day64(s)) & (H["d"].values <= day64(e))
    return float(pd.to_numeric(H.loc[m, "receipt_total"], errors="coerce").sum())


def top_supplier(f: Facts, s: date, e: date) -> str:
    if not f.caps.get("has_imports"):
        raise Unanswerable("no imports")
    H = receipts(f)
    m = (H["d"].values >= day64(s)) & (H["d"].values <= day64(e))
    D = H[m].copy()
    D["sup"] = D["supplier"].where(D["supplier"].notna(), D.get("supplier_id"))
    if D["sup"].isna().all():
        raise Unanswerable("no supplier")
    agg = D.groupby("sup")["receipt_total"].sum().sort_values(ascending=False)
    if len(agg) > 1 and agg.iloc[0] == agg.iloc[1]:
        agg = agg[agg == agg.iloc[0]].sort_index()
    return str(agg.index[0])


def avg_import_price(f: Facts, key, s: date, e: date, unit: str | None) -> float:
    """rulebook §5: bình quân theo số lượng nhập (Σ tiền / Σ SL), theo đơn vị nhập, phiếu trong tháng, bỏ phiếu huỷ."""
    if not f.caps.get("has_imports"):
        raise Unanswerable("no imports")
    IL = import_lines(f)
    m = (IL["product_key"] == key).values & (IL["d"].values >= day64(s)) & (IL["d"].values <= day64(e))
    D = IL[m]
    if not len(IL) or IL["amount"].isna().all():
        raise Unanswerable("no import prices")
    if not len(D):
        raise Unanswerable("no import of this item in the period")
    units = {fold(u) for u in D["unit"].dropna()}
    if unit and units and units != {fold(unit)}:
        per_base = D["amount"].sum() / D["qty_base"].sum()
        return per_base * _factor(f, key, unit)
    return float(D["amount"].sum() / D["qty"].astype(float).sum())


def sold_base_after(f: Facts, d: date, end: date | None = None) -> pd.Series:
    """Base quantity that left stock on days in (d, end]: every non-cancelled order line (returns come back as
    negative lines)."""
    L = f.L
    dd = L["d"].values.astype("datetime64[D]")
    m = (dd > day64(d)) & (L["status"].values != "cancelled") & ~L["is_fee"].values
    if end is not None:
        m &= dd <= day64(end)
    return L[m].groupby("product_key")["qty_base"].sum()


def lot_stock_at(f: Facts, d: date) -> pd.Series:
    """Shops tracking lots (README: mỗi dòng nhập là một lô; hàng khách trả về lại lô đã bán): stock at end of d =
    Σ lot receipts up to d − Σ lot sales of non-cancelled orders up to d + their returned quantity."""
    LT, LS = f.st.lots, f.st.lot_sales
    H = f.st.imports
    rd = dict(zip(H["receipt_id"], H["import_date"])) if H is not None and len(H) else {}
    LT = LT.copy()
    LT["rdate"] = LT["receipt_id"].map(rd)
    inn = LT[LT["rdate"].map(lambda x: x is not None and x == x and x <= d)].groupby("product_key")["qty_base"].sum()
    lot2key = dict(zip(LT["lot"], LT["product_key"]))
    O = f.O.set_index("order_id")
    LS = LS.copy()
    LS["key"] = LS["lot"].map(lot2key)
    LS["od"] = LS["order_id"].map(O["order_date"])
    LS["st"] = LS["order_id"].map(O["status"])
    m = LS["od"].map(lambda x: x is not None and x == x and x <= d) & (LS["st"] != "cancelled")
    out_q = LS[m].groupby("key")["qty_base"].sum()
    ret_q = LS[m].groupby("key")["return_qty_base"].sum()
    return inn.sub(out_q, fill_value=0).add(ret_q, fill_value=0)


def stock_at(f: Facts, d: date) -> pd.Series:
    """Stock (base unit) per product at the end of day d."""
    S = f.st.stock
    if S is None or not len(S):
        LT, LS = f.st.lots, f.st.lot_sales
        if LT is not None and len(LT) and LS is not None and len(LS):
            return lot_stock_at(f, d)
        raise Unanswerable("no stock data")
    snap = S.groupby("product_key")["stock_qty"].sum()
    end = f.dend
    if d >= end:
        return snap
    IL = import_lines(f)
    imp = IL[(IL["d"].values > day64(d)) & (IL["d"].values <= day64(end))].groupby("product_key")["qty_base"].sum() \
        if len(IL) else pd.Series(dtype=float)
    sold = sold_base_after(f, d, end)
    out = snap.sub(imp, fill_value=0).add(sold, fill_value=0)
    return out.reindex(snap.index).fillna(snap)


def stock_qty(f: Facts, key, d: date) -> float:
    s = stock_at(f, d)
    if key not in s.index:
        raise Unanswerable("item not tracked in stock")
    return float(s[key])


def stock_value(f: Facts, d: date) -> float:
    """rulebook §5: tồn hết ngày × giá vốn một đơn vị cơ bản trong danh mục; chỉ hàng còn tồn."""
    s = stock_at(f, d)
    cost = pd.to_numeric(f.P["cost"], errors="coerce")
    if cost.isna().all():
        raise Unanswerable("no catalog cost")
    v = s[s > 0]
    c = cost.reindex(v.index)
    if c.isna().any():
        raise Abstain("missing cost for some items")
    return float((v * c).sum())


def qty_window(f: Facts, key, s: date, e: date) -> float:
    L = f.L
    dd = L["d"].values.astype("datetime64[D]")
    m = (L["product_key"] == key).values & L["valid"].values & (dd >= day64(s)) & (dd <= day64(e))
    return float(L.loc[m, "qty_base"].sum())


def price_increase_day(f: Facts, key, new_price: float | None) -> date | None:
    """rulebook §5: lần tăng giá = ngày giá mới cao hơn giá liền trước; cùng một giá nhiều lần thì lấy lần đầu."""
    PH = f.st.price_history
    if PH is None or not len(PH):
        return None
    D = PH[PH["product_key"] == key].sort_values("effective_date")
    prev = None
    for d, p in zip(D["effective_date"], D["new_price"]):
        if prev is not None and p > prev and (new_price is None or abs(p - new_price) < 0.5):
            return d
        prev = p
    return None
