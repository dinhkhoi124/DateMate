"""Run-time invariants on a built Store (DECISIONS D11): is what ingest understood self-consistent?

metrics(store)    -> dict of deterministic numbers, computed from the Store only (no question, no gold)
violations(m, th) -> list of Violation(area, name, detail) given thresholds (config.Invariants)

An area is a part of the data an answer depends on. The pipeline abstains on a question whose op uses a violated
area; a violated 'orders' area abstains the whole shop. Metrics that cannot be computed (table absent) are None and
never violate: a missing capability is 'unanswerable' territory, handled elsewhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from datamate.config import INVARIANTS, Invariants
from datamate.ingest.values import CANCELLED, DONE, RETURNED, SHIPPING, fold


@dataclass(frozen=True)
class Violation:
    area: str
    name: str
    detail: str


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _day(v):
    if isinstance(v, datetime):
        return v.date()
    return v if isinstance(v, date) else None


def _frac(mask) -> float | None:
    n = len(mask)
    return None if n == 0 else round(float(np.asarray(mask, dtype=bool).mean()), 4)


def metrics(st) -> dict:
    m: dict = {}
    O, L = st.orders, st.lines
    if O is None or not len(O) or "order_date" not in O:
        m["orders.n"] = 0
        return m
    m["orders.n"] = len(O)
    days = [_day(v) for v in O["order_date"]]
    ok = [d for d in days if d is not None]
    m["orders.with_date"] = _frac([d is not None for d in days])
    m["orders.dup_id"] = _frac(O["order_id"].duplicated(keep=False)) if "order_id" in O else None
    end = st.spec.data_end if getattr(st, "spec", None) is not None else None
    m["orders.after_data_end"] = _frac([d > end for d in ok]) if end and ok else None
    if ok:
        dmin, dmax = min(ok), max(ok)
        m["orders.span_days"] = (dmax - dmin).days
        m["orders.year_min"], m["orders.year_max"] = dmin.year, dmax.year
        # day-of-month histogram: dd/mm read as mm/dd leaves days 13..31 empty
        dom = np.bincount([d.day for d in ok], minlength=32)[1:]
        m["orders.share_day_gt12"] = round(float(dom[12:].sum() / dom.sum()), 4)
    st_ = O["status"].astype(object)
    m["status.done"] = _frac(st_ == DONE)
    m["status.cancelled"] = _frac(st_ == CANCELLED)
    m["status.returned"] = _frac(st_ == RETURNED)
    m["status.shipping"] = _frac(st_ == SHIPPING)
    m["status.unknown_raw"] = sum(1 for w in st.warnings if w.startswith("unrecognized status"))
    # the README speaks of order statuses (cancel / return), so an all-completed shop means the status was not read
    raw = getattr(getattr(st, "spec", None), "raw", "") or ""
    m["status.readme_mentions"] = bool(re.search(r"trang thai|\bhuy\b|tra hang", fold(raw)))
    rev = _num(O["subtotal"])
    m["money.orders_with_rev"] = _frac(rev.notna() & (rev != 0))
    m["money.neg_rev"] = _frac(rev < 0)
    pos = rev[rev > 0]
    m["money.median_order"] = float(pos.median()) if len(pos) else None
    m["money.subtotal_mismatch"] = st.caps.get("check_subtotal_mismatch")
    if L is not None and len(L) and "order_id" in L:
        ids = set(O["order_id"].dropna())
        m["lines.n"] = len(L)
        m["lines.orphan"] = _frac(~L["order_id"].isin(ids))
        m["orders.without_lines"] = _frac(~O["order_id"].isin(set(L["order_id"].dropna())))
        q, p, a = _num(L.get("qty")), _num(L.get("unit_price")), _num(L.get("amount"))
        if q is not None and p is not None and a is not None:
            both = q.notna() & p.notna() & a.notna() & (q > 0) & (p > 0)
            if both.sum():
                gross = (q * p)[both]
                disc = _num(L.get("line_discount")).fillna(0)[both] if "line_discount" in L else 0
                # amount is qty x price, possibly after a line discount given as money or as a percentage
                aa = a[both]
                okm = ((aa - gross).abs() <= 1) | ((gross - disc - aa).abs() <= 1) | \
                    ((gross * (1 - disc / 100) - aa).abs() <= 1) | ((gross - disc * q[both] - aa).abs() <= 1)
                m["lines.amount_vs_qty_price"] = round(float(okm.mean()), 4)
                m["lines.amount_gt_gross"] = round(float((a[both] > gross * 1.01 + 1).mean()), 4)
        qb = _num(L.get("qty_base"))
        if q is not None and qb is not None:
            both = q.notna() & qb.notna() & (q > 0)
            m["lines.qty_base_lt_qty"] = _frac((qb[both] < q[both] - 1e-9).values) if both.sum() else None
        a = _num(L.get("amount"))
        if a is not None and len(a.dropna()):
            m["money.median_line"] = float(a[a > 0].median()) if (a > 0).any() else None
    if "customer_id" in O or "customer_name" in O:
        has = O["customer_id"].notna() | O["customer_name"].notna()
        m["customers.orders_with_customer"] = _frac(has)
    if "debt" in O and O["debt"].notna().any():
        d, t = _num(O["debt"]), _num(O["order_total"]) if "order_total" in O else None
        m["debt.negative"] = _frac((d < 0).values)
        if t is not None and t.notna().any():
            both = d.notna() & t.notna()
            m["debt.gt_total"] = _frac((d[both] > t[both] + 1).values) if both.sum() else None
    return m


def violations(m: dict, th: Invariants = INVARIANTS) -> list[Violation]:
    out: list[Violation] = []

    def chk(area, name, bad, detail):
        if bad:
            out.append(Violation(area, name, detail))

    g = m.get
    if not g("orders.n"):
        return out   # no sales table: the pipeline already abstains on the whole shop
    lo, hi = th.year_range
    chk("orders", "date_unread", g("orders.with_date") is not None and g("orders.with_date") < th.min_orders_with_date,
        f"with_date={g('orders.with_date')}")
    chk("orders", "dup_order_id", (g("orders.dup_id") or 0) > th.max_dup_order_id, f"dup_id={g('orders.dup_id')}")
    chk("orders", "after_data_end", (g("orders.after_data_end") or 0) > th.max_after_data_end,
        f"after_data_end={g('orders.after_data_end')}")
    chk("orders", "day_month_swapped", g("orders.share_day_gt12") is not None
        and g("orders.share_day_gt12") < th.min_share_day_gt12, f"share_day_gt12={g('orders.share_day_gt12')}")
    chk("orders", "year_out_of_range", g("orders.year_min") is not None
        and not (lo <= g("orders.year_min") and g("orders.year_max") <= hi),
        f"years={g('orders.year_min')}..{g('orders.year_max')}")
    chk("status", "status_distribution", (g("status.done") or 0) < th.min_done
        or (g("status.cancelled") or 0) > th.max_cancelled or (g("status.returned") or 0) > th.max_returned,
        f"done={g('status.done')} cancelled={g('status.cancelled')} returned={g('status.returned')}")
    chk("status", "status_unrecognized", (g("status.unknown_raw") or 0) > 0, "unrecognized raw status values")
    # rulebook: every order has one of four statuses; thousands of orders without a single cancel or return
    # means the status was not read (even when the README never mentions it)
    chk("status", "status_not_found", g("status.done") == 1.0
        and (g("status.readme_mentions") or g("orders.n") >= th.min_orders_status_evidence),
        f"every order reads as completed (n={g('orders.n')})")
    chk("money", "orders_without_revenue", g("money.orders_with_rev") is not None
        and g("money.orders_with_rev") < th.min_orders_with_rev, f"orders_with_rev={g('money.orders_with_rev')}")
    chk("money", "negative_revenue", (g("money.neg_rev") or 0) > th.max_neg_rev, f"neg_rev={g('money.neg_rev')}")
    mo, (a, b) = g("money.median_order"), th.order_money_range
    chk("money", "order_money_scale", mo is not None and not (a <= mo <= b), f"median_order={mo}")
    chk("money", "subtotal_mismatch", (g("money.subtotal_mismatch") or 0) > th.max_subtotal_mismatch,
        f"subtotal_mismatch={g('money.subtotal_mismatch')}")
    chk("lines", "orphan_lines", (g("lines.orphan") or 0) > th.max_orphan_lines, f"orphan={g('lines.orphan')}")
    chk("lines", "orders_without_lines", (g("orders.without_lines") or 0) > th.max_orders_without_lines,
        f"without_lines={g('orders.without_lines')}")
    chk("lines", "amount_not_qty_price", g("lines.amount_vs_qty_price") is not None
        and g("lines.amount_vs_qty_price") < th.min_amount_vs_qty_price,
        f"amount_vs_qty_price={g('lines.amount_vs_qty_price')}")
    chk("lines", "qty_base_lt_qty", (g("lines.qty_base_lt_qty") or 0) > th.max_qty_base_lt_qty,
        f"qty_base_lt_qty={g('lines.qty_base_lt_qty')}")
    chk("debt", "debt_negative", (g("debt.negative") or 0) > th.max_debt_negative, f"negative={g('debt.negative')}")
    chk("debt", "debt_gt_total", (g("debt.gt_total") or 0) > th.max_debt_gt_total, f"gt_total={g('debt.gt_total')}")
    # revenue is the sum of line amounts (rulebook §1): broken lines break money too
    if any(v.area == "lines" for v in out) and not any(v.area == "money" for v in out):
        out.append(Violation("money", "from_lines", "revenue is computed from the lines"))
    return out


# areas each op reads besides 'orders' (which every op reads); unknown ops get DEFAULT_AREAS
_S, _M, _L = "status", "money", "lines"
OP_AREAS: dict[str, frozenset] = {op: frozenset(a) for op, a in {
    "revenue": (_S, _M), "aov": (_S, _M), "growth": (_S, _M), "shift_compare": (_S, _M),
    "revenue_series": (_S, _M), "revenue_by": (_S, _M, _L), "argmax_day": (_S, _M), "argmax_month": (_S, _M),
    "monotonic": (_S, _M), "days_revenue_over": (_S, _M), "orders_revenue_over": (_S, _M), "top_staff": (_S, _M),
    "tet_vs_before": (_S, _M), "share": (_S, _M, _L),
    "order_count": (_S,), "cancel_count": (_S,), "cancel_rate": (_S,), "delivery_count": (_S,),
    "qty": (_S, _L), "items_in_n_orders": (_S, _L), "dead_count": (_S, _L), "unit_price": (_L, _M),
    "top_products": (_S, _M, _L), "price_effect": (_S, _M, _L), "gross_profit": (_S, _M, _L),
    "net_profit": (_S, _M, _L),
    "top_customers": (_S, _M, "customers"), "customers_min_orders": (_S, "customers"),
    "customers_min_orders_pct": (_S, "customers"),
    "import_minus_sales": (_S, _L, "imports"), "avg_import_price": ("imports",), "import_total": ("imports",),
    "top_supplier": ("imports",),
    "stock_qty": ("stock",), "stock_big": ("stock",), "stock_value": ("stock",), "stock_out_count": ("stock",),
    "debt_total": ("debt", _S, _M), "debt_customers": ("debt", _S, _M, "customers"),
    "debt_table": ("debt", _S, _M, "customers"),
    "debt_aging": ("debt", _S, _M), "debt_collected": ("debt", _S, _M), "debt_customer_flow": ("debt", _S, _M, "customers"),
    "nearest_expiry": ("lots",), "expiring_lots_count": ("lots",), "expiring_products": ("lots",),
}.items()}
DEFAULT_AREAS = frozenset((_S, _M, _L))


def blocking(op: str | None, viol: list[Violation]) -> list[Violation]:
    """Violations that make an answer to `op` untrustworthy."""
    areas = {"orders"} | (OP_AREAS.get(op, DEFAULT_AREAS) if op else DEFAULT_AREAS)
    return [v for v in viol if v.area in areas]
