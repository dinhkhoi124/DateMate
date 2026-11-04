"""Customer debt (rulebook §4)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from datamate.execute.facts import Facts
from datamate.execute.ops import Abstain, Unanswerable, channel_class
from datamate.ingest.values import COD, DEBT


def _orders(f: Facts) -> pd.DataFrame:
    """Orders that can carry debt: not cancelled; COD and marketplace orders are never customer debt (rulebook §4)."""
    D = f.st.extra.get("_debt_orders")
    if D is not None:
        return D
    O = f.O
    m = ~O["cancelled"].values & (O["pay"] != COD).values
    m &= ~O["channel"].map(lambda c: channel_class(c) == "marketplace").values
    D = O[m].copy()
    tot = pd.to_numeric(D["order_total"], errors="coerce")
    sub = pd.to_numeric(D["subtotal"], errors="coerce").fillna(0)
    disc = pd.to_numeric(D["order_discount"], errors="coerce").fillna(0)
    ship = pd.to_numeric(D["shipping_fee"], errors="coerce").fillna(0)
    D["total"] = tot.where(tot.notna(), sub - disc + ship)   # rulebook §4: tổng đơn đã trừ giảm giá cả đơn, gồm phí giao
    f.st.extra["_debt_orders"] = D
    return D


def _payments(f: Facts) -> pd.DataFrame:
    P = f.st.payments
    if P is None or not len(P):
        raise Unanswerable("no payment records")
    P = P.copy()
    P["pd"] = P["paid_at"].map(lambda v: v.date() if hasattr(v, "date") else v)
    P["amount"] = pd.to_numeric(P["amount"], errors="coerce").fillna(0)
    return P


def outstanding(f: Facts, x: date) -> pd.DataFrame:
    """Per order: unpaid part of the order total counting payments up to end of day x; only orders with debt > 0."""
    if not f.caps.get("has_debt"):
        raise Unanswerable("no debt data")
    D = _orders(f)
    D = D[D["order_date"].map(lambda d: d is not None and d == d and d <= x)]
    P = _payments(f)
    paid = P[P["pd"].map(lambda d: d is not None and d == d and d <= x)].groupby("order_id")["amount"].sum()
    D = D.assign(paid_x=D["order_id"].map(paid).fillna(0.0))
    D = D.assign(owe=(D["total"] - D["paid_x"]).round(6))
    D = D[D["owe"] > 0.5].copy()
    D["age"] = [(x - d).days for d in D["order_date"]]   # rulebook §4: tuổi nợ tính từ ngày đặt
    return D


def debt_by_customer(f: Facts, x: date) -> list:
    """rulebook §4: nợ theo khách (khách như §3); khách lẻ có trong tổng nhưng không có dòng."""
    D = outstanding(f, x)
    D = D[D["ckey"].notna()]
    agg = D.groupby("ckey")["owe"].sum()
    items = [(k, float(v)) for k, v in agg.items() if v > 0.5]
    items.sort(key=lambda kv: (-round(kv[1], 6), str(kv[0])))
    return items


def collected(f: Facts, s: date, e: date, ckey=None) -> float:
    """rulebook §4: trả nợ = khoản trả sau ngày mua có ngày trả trong kỳ, của mọi đơn ghi nợ không huỷ."""
    D = _orders(f)
    D = D[D["pay"] == DEBT]
    if ckey is not None:
        D = D[D["ckey"] == ckey]
    if not len(D):
        return 0.0
    od = dict(zip(D["order_id"], D["order_date"]))
    P = _payments(f)
    P = P[P["order_id"].isin(od)]
    m = np.array([(pd_ is not None and pd_ == pd_ and s <= pd_ <= e and pd_ > od[o])
                  for o, pd_ in zip(P["order_id"], P["pd"])], dtype=bool)
    return float(P.loc[m, "amount"].sum())


def new_debt(f: Facts, s: date, e: date, ckey=None) -> float:
    """rulebook §4: nợ thêm = phần tổng đơn chưa trả ngay trong ngày mua, của đơn ghi nợ đặt trong kỳ."""
    D = _orders(f)
    D = D[(D["pay"] == DEBT) & D["order_date"].map(lambda d: d is not None and d == d and s <= d <= e)]
    if ckey is not None:
        D = D[D["ckey"] == ckey]
    if not len(D):
        return 0.0
    od = dict(zip(D["order_id"], D["order_date"]))
    P = _payments(f)
    P = P[P["order_id"].isin(od)]
    mask = np.array([pd_ == od[o] for o, pd_ in zip(P["order_id"], P["pd"])], dtype=bool)
    same = P[mask].groupby("order_id")["amount"].sum() if len(P) else pd.Series(dtype=float)
    return float((D["total"] - D["order_id"].map(same).fillna(0.0)).clip(lower=0).sum())
