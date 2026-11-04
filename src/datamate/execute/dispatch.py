"""Intent -> (answer type, value) using the deterministic ops. Unknown or unsupported -> Abstain."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from datamate.execute import ops
from datamate.execute.facts import Facts, day64
from datamate.execute.ops import Abstain, Unanswerable, money, pct, pct_change, r2
from datamate.parse.intent import Intent
from datamate.timeparse.periods import Period, previous_period, same_period_last_year

FLOW_KINDS = ("day", "range", "week", "last_n", "month", "mtd", "quarter", "ytd", "tet")


def _main_period(it: Intent, as_of: date) -> Period:
    ps = [p for p in it.periods if p.kind in FLOW_KINDS]
    if not ps:
        raise Abstain("no period")
    if len({(p.start, p.end) for p in ps}) > 1:
        raise Abstain("several periods")
    return ps[0]


def _check_filters(it: Intent):
    for k, v in it.flt.items():
        if isinstance(v, str) and v.startswith("?"):
            raise Abstain(f"unresolved {k}: {v}")


def _metric_value(f, metric, s, e, flt):
    if metric == "revenue":
        return ops.revenue(f, s, e, flt)
    if metric == "orders":
        return ops.order_count(f, s, e, flt)
    if metric == "qty":
        return ops.qty(f, s, e, flt)
    raise Abstain(f"metric {metric}")


def _out(metric, v):
    return ("number", money(v)) if metric == "revenue" else ("number", int(round(v)))


def run(it: Intent, f: Facts, as_of: date):
    _check_filters(it)
    op = it.op
    if op == "unanswerable":
        raise Unanswerable(it.note or "not in data")
    if op == "need_cap":
        if not f.caps.get(it.args["cap"]):
            raise Unanswerable(it.args["cap"])
        raise Abstain("capability present but op not implemented")
    h = HANDLERS.get(op)
    if h is None:
        raise Abstain(f"op {op} not implemented")
    return h(it, f, as_of)


# ------------------------------------------------------------------ handlers
def h_revenue(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    return "number", money(ops.revenue(f, s, e, it.flt))


def h_order_count(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    return "number", ops.order_count(f, s, e, it.flt)


def h_cancel_count(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    return "number", ops.cancel_count(f, s, e, it.flt)


def h_cancel_rate(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    return "percent", ops.cancel_rate(f, s, e, it.flt)


def h_qty(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    v = ops.qty(f, s, e, it.flt)
    return "number", int(round(v)) if abs(v - round(v)) < 1e-9 else r2(v)


def h_aov(it, f, as_of):
    """rulebook §3: giá trị trung bình một đơn = doanh thu / số đơn, làm tròn về đồng."""
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    n = ops.order_count(f, s, e, it.flt)
    if not n:
        raise Abstain("no orders")
    return "number", money(ops.revenue(f, s, e, it.flt) / n)


def h_share(it, f, as_of):
    """rulebook §1: tỉ trọng chia cho doanh thu cả shop cùng kỳ (tính cả đơn vãng lai)."""
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    if not it.flt:
        raise Abstain("share of what")
    return "percent", pct(ops.revenue(f, s, e, it.flt), ops.revenue(f, s, e, {}))


def _cmp_periods(it, as_of):
    ps = [p for p in it.periods if p.kind in FLOW_KINDS]
    cmp = it.args.get("compare")
    if cmp == "explicit":
        if len(ps) < 2:
            raise Abstain("need two periods")
        return ps[0], ps[1]
    if len({(p.start, p.end) for p in ps}) != 1:
        raise Abstain("ambiguous base period")
    p = ps[0]
    return (p, same_period_last_year(p)) if cmp == "yoy" else (p, previous_period(p, as_of))


def h_growth(it, f, as_of):
    """rulebook §1: % thay đổi so với kỳ trước; giảm ra số âm."""
    a, b = _cmp_periods(it, as_of)
    m = it.args.get("metric", "revenue")
    sa, ea = ops.clip(f, a.start, a.end, as_of)
    sb, eb = ops.clip(f, b.start, b.end, as_of)
    return "percent", pct_change(_metric_value(f, m, sa, ea, it.flt), _metric_value(f, m, sb, eb, it.flt))


def h_shift_compare(it, f, as_of):
    """rulebook §2: so ca chiều với ca sáng, mốc là ca sáng."""
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    m = it.args.get("metric", "revenue")
    base = {k: v for k, v in it.flt.items() if k != "shift"}
    a = _metric_value(f, m, s, e, {**base, "shift": "afternoon"})
    b = _metric_value(f, m, s, e, {**base, "shift": "morning"})
    return "percent", pct_change(a, b)


def h_revenue_series(it, f, as_of):
    ps = [p for p in it.periods if p.kind == "month"]
    out = []
    for p in ps:
        s, e = ops.clip(f, p.start, p.end, as_of)
        out.append(money(ops.revenue(f, s, e, it.flt)))
    return "list", out


def h_revenue_by(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    dim = it.args["dim"]
    base = {k: v for k, v in it.flt.items() if k != dim}
    out = []
    for v in dict.fromkeys(it.args["values"]):
        out.append(money(ops.revenue(f, s, e, {**base, dim: v})))
    return "list", out


def h_argmax_day(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    m = it.args.get("metric", "revenue")
    if m != "revenue":
        raise Abstain("argmax day metric")
    d = ops.daily_revenue(f, s, e, it.flt)
    if not len(d):
        raise Abstain("no sales")
    d = d.sort_index()
    best = d.max()
    day = d[d == best].index[0]
    return "date", str(np.datetime64(day, "D"))


def h_argmax_month(it, f, as_of):
    ps = list({p.label: p for p in it.periods if p.kind == "month"}.values())
    if len(ps) < 2:
        raise Abstain("months")
    m = it.args.get("metric", "revenue")
    vals = []
    for p in ps:
        s, e = ops.clip(f, p.start, p.end, as_of)
        vals.append((_metric_value(f, m, s, e, it.flt), p.label))
    best = max(v for v, _ in vals)
    return "date", next(lbl for v, lbl in vals if v == best)


def h_monotonic(it, f, as_of):
    ps = list({p.label: p for p in it.periods if p.kind == "month"}.values())
    if len(ps) < 2:
        raise Abstain("months")
    m = it.args.get("metric", "revenue")
    vals = []
    for p in ps:
        s, e = ops.clip(f, p.start, p.end, as_of)
        vals.append(_metric_value(f, m, s, e, it.flt))
    up = all(b > a for a, b in zip(vals, vals[1:]))
    down = all(b < a for a, b in zip(vals, vals[1:]))
    return "boolean", bool(up if it.args.get("dir") == "up" else down)


def h_top_products(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    items = ops.top_products(f, s, e, it.flt, by=it.args.get("by", "qty"))
    k = it.args.get("k", 1)
    if len(items) < k:
        raise Abstain("not enough items")
    names = [f.pname(key) for key, _ in items[:k]]
    return ("list", names) if it.atype == "list" else ("string", names[0])


def h_top_customers(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    items = ops.customers_agg(f, s, e, it.flt, by=it.args.get("by", "revenue"))
    k = it.args.get("k", 1)
    if len(items) < k:
        raise Abstain("not enough customers")
    names = [f.customer_display(key) for key, _ in items[:k]]
    return ("list", names) if it.atype == "list" else ("string", names[0])


def h_top_staff(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    items = ops.staff_agg(f, s, e, it.flt)
    if not items:
        raise Abstain("no staff sales")
    return "string", items[0][0]


def h_days_revenue_over(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    d = ops.daily_revenue(f, s, e, it.flt)
    x = it.args["x"]
    return "number", int(((d >= x) if it.args.get("inclusive", True) else (d > x)).sum())


def h_orders_revenue_over(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    om = f.omask(s, e) & ops.order_filter(f, it.flt)
    r = f.O["rev"].values[om]
    x = it.args["x"]
    return "number", int(((r >= x) if it.args.get("inclusive", True) else (r > x)).sum())


def h_items_in_n_orders(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    n = it.args["n"] if it.args.get("inclusive", True) else it.args["n"] + 1
    return "number", ops.items_in_n_orders(f, s, e, n, it.flt)


def _cust_orders(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    items = ops.customers_agg(f, s, e, it.flt, by="orders")
    n = it.args["n"] if it.args.get("inclusive", True) else it.args["n"] + 1
    return items, n


def h_customers_min_orders(it, f, as_of):
    items, n = _cust_orders(it, f, as_of)
    return "number", sum(1 for _, c in items if c >= n)


def h_customers_min_orders_pct(it, f, as_of):
    """rulebook §1: khách đã mua trong kỳ = có ít nhất một đơn §1."""
    items, n = _cust_orders(it, f, as_of)
    return "percent", pct(sum(1 for _, c in items if c >= n), len(items))


def h_dead_count(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    return "number", len(ops.dead_items(f, s, e))


def h_unit_price(it, f, as_of):
    d = it.args.get("day")
    if d is None:
        raise Abstain("no day")
    if f.dend and d > f.dend or f.dmin and d < f.dmin:
        raise Unanswerable("day outside data")
    return "number", money(ops.unit_price_on(f, it.flt["product"], d))


HANDLERS = {
    "revenue": h_revenue, "order_count": h_order_count, "cancel_count": h_cancel_count, "cancel_rate": h_cancel_rate,
    "qty": h_qty, "aov": h_aov, "share": h_share, "growth": h_growth, "shift_compare": h_shift_compare,
    "revenue_series": h_revenue_series, "revenue_by": h_revenue_by, "argmax_day": h_argmax_day,
    "argmax_month": h_argmax_month, "monotonic": h_monotonic, "top_products": h_top_products,
    "top_customers": h_top_customers, "top_staff": h_top_staff, "days_revenue_over": h_days_revenue_over,
    "orders_revenue_over": h_orders_revenue_over, "items_in_n_orders": h_items_in_n_orders,
    "customers_min_orders": h_customers_min_orders, "customers_min_orders_pct": h_customers_min_orders_pct,
    "dead_count": h_dead_count, "unit_price": h_unit_price,
}


# ------------------------------------------------------------------ imports / stock / prices (rulebook §5)
from datamate.execute import inventory as inv  # noqa: E402


def _asof_day(it, as_of) -> date:
    ps = [p for p in it.periods if p.kind in ("asof", "day")]
    d = ps[0].start if ps else as_of
    if d > as_of:
        raise Abstain("as-of after today")
    return d


def _need_product(it):
    k = it.flt.get("product")
    if k is None:
        raise Abstain("no product")
    return k


def h_price_effect(it, f, as_of):
    """rulebook §5 hiệu ứng tăng giá: N ngày kể từ D (gồm D) so với N ngày ngay trước D; bỏ đơn huỷ, trả."""
    key = _need_product(it)
    n = it.args["n"]
    d = it.args.get("day") or inv.price_increase_day(f, key, it.args.get("new_price"))
    if d is None:
        raise Abstain("no increase day")
    after = inv.qty_window(f, key, d, d + timedelta(days=n - 1))
    before = inv.qty_window(f, key, d - timedelta(days=n), d - timedelta(days=1))
    return "percent", pct_change(after, before)


def h_avg_import_price(it, f, as_of):
    key = _need_product(it)
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    return "number", money(inv.avg_import_price(f, key, s, e, it.args.get("unit")))


def h_import_total(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    return "number", money(inv.import_total(f, s, e))


def h_top_supplier(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    return "string", inv.top_supplier(f, s, e)


def h_import_minus_sales(it, f, as_of):
    """rulebook §5: phần bán là SL của đơn tính doanh thu, quy về cùng đơn vị với phần nhập."""
    key = _need_product(it)
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    if not f.caps.get("has_imports"):
        raise Unanswerable("no imports")
    IL = inv.import_lines(f)
    m = (IL["product_key"] == key).values & (IL["d"].values >= day64(s)) & (IL["d"].values <= day64(e))
    imp = float(IL.loc[m, "qty_base"].sum())
    sold = inv.qty_window(f, key, s, e)
    v = imp - sold
    return "number", int(round(v)) if abs(v - round(v)) < 1e-9 else r2(v)


def h_stock_qty(it, f, as_of):
    key = _need_product(it)
    d = _asof_day(it, as_of)
    v = inv.stock_qty(f, key, d)
    unit = it.args.get("unit")
    if unit:
        fac = inv._factor(f, key, unit)
        if fac and fac != 1:
            raise Abstain("stock asked in a big unit")
    return "number", int(round(v)) if abs(v - round(v)) < 1e-9 else r2(v)


def h_stock_big(it, f, as_of):
    """rulebook §5: tồn theo đơn vị lớn = [số đơn vị lớn nguyên, số đơn vị cơ bản lẻ]."""
    key = _need_product(it)
    d = _asof_day(it, as_of)
    v = inv.stock_qty(f, key, d)
    big = it.args.get("big")
    if not big:
        raise Abstain("big unit not found")
    fac = inv._factor(f, key, big)
    if not fac or fac == 1:
        raise Abstain("no factor for big unit")
    whole = int(v // fac)
    return "list", [whole, int(round(v - whole * fac))]


def h_stock_value(it, f, as_of):
    d = _asof_day(it, as_of)
    return "number", money(inv.stock_value(f, d))


def h_stock_out_count(it, f, as_of):
    """rulebook §5: mặt hàng mua về để bán có tồn hết ngày = 0."""
    d = _asof_day(it, as_of)
    s = inv.stock_at(f, d)
    return "number", int((s.round(6) == 0).sum())


HANDLERS.update({
    "price_effect": h_price_effect, "avg_import_price": h_avg_import_price, "import_total": h_import_total,
    "top_supplier": h_top_supplier, "import_minus_sales": h_import_minus_sales, "stock_qty": h_stock_qty,
    "stock_big": h_stock_big, "stock_value": h_stock_value, "stock_out_count": h_stock_out_count,
})


# ------------------------------------------------------------------ debt (rulebook §4)
from datamate.execute import debt as dbt  # noqa: E402


def h_debt_total(it, f, as_of):
    x = _asof_day(it, as_of)
    D = dbt.outstanding(f, x)
    if it.args.get("min_age") is not None:
        D = D[D["age"] > it.args["min_age"]]   # rulebook §2: quá/hơn N không gồm N
    return "number", money(D["owe"].sum())


def h_debt_customers(it, f, as_of):
    return "number", len(dbt.debt_by_customer(f, _asof_day(it, as_of)))


def h_debt_table(it, f, as_of):
    items = dbt.debt_by_customer(f, _asof_day(it, as_of))
    k = it.args.get("k", 3)
    if len(items) < k:
        raise Abstain("not enough debtors")
    rows = [[f.customer_display(c), money(v)] for c, v in items[:k]]
    return "table", {"columns": ["khach_hang", "so_tien_no"], "rows": rows}


def h_debt_aging(it, f, as_of):
    D = dbt.outstanding(f, _asof_day(it, as_of))
    a = D["age"]
    return "list", [money(D.loc[a <= 7, "owe"].sum()), money(D.loc[(a >= 8) & (a <= 15), "owe"].sum()),
                    money(D.loc[a > 15, "owe"].sum())]


def h_debt_collected(it, f, as_of):
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    if not f.caps.get("has_payments"):
        raise Unanswerable("no payments")
    return "number", money(dbt.collected(f, s, e))


def h_debt_customer_flow(it, f, as_of):
    # rulebook §7: shop không ghi công nợ -> unanswerable; checked before entity linking (no debtors to link to)
    if not f.caps.get("has_payments"):
        raise Unanswerable("no payments")
    c = it.flt.get("customer")
    if c is None:
        raise Abstain("no customer")
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    return "list", [money(dbt.new_debt(f, s, e, c)), money(dbt.collected(f, s, e, c))]


# ------------------------------------------------------------------ profit, deliveries, Tết, lots
def h_gross_profit(it, f, as_of):
    """rulebook §5: lãi gộp = doanh thu §1 − SL × giá vốn ghi trên dòng bán (một đơn vị bán)."""
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    return "number", money(_gross(f, s, e, it.flt))


def _gross(f, s, e, flt):
    if not f.caps.get("has_cost"):
        raise Unanswerable("no cost on sale lines")
    rev = ops.revenue(f, s, e, flt)
    om = f.omask(s, e) & ops.order_filter(f, flt)
    ids = set(f.O["order_id"].values[om])
    L = f.L
    sel = L["valid"].values & L["order_id"].isin(ids).values
    lm = ops.line_filter(f, flt)
    if lm is not None:
        sel &= lm
    V = L[sel]
    if V.loc[~V["is_fee"], "cost_unit"].isna().any():
        raise Abstain("missing cost on some lines")
    return rev - float((V["qty"] * V["cost_unit"].fillna(0)).sum())


def h_net_profit(it, f, as_of):
    """rulebook §5: lãi ròng tháng = lãi gộp tháng − mọi khoản trong sổ chi phí của tháng."""
    p = _main_period(it, as_of)
    if p.kind != "month":
        raise Abstain("net profit needs a month")
    s, e = ops.clip(f, p.start, p.end, as_of)
    E = f.st.expenses
    if E is None or not len(E):
        raise Unanswerable("no expense book")
    m = E["date"].map(lambda d: d is not None and d == d and s <= d <= e)
    return "number", money(_gross(f, s, e, {}) - float(E.loc[m, "amount"].sum()))


def h_delivery_count(it, f, as_of):
    """rulebook §6 VLXD: mỗi đợt của mỗi đơn có ngày giao trong kỳ là một đợt, kể cả đơn sau đó trả hàng."""
    p = _main_period(it, as_of)
    s, e = ops.clip(f, p.start, p.end, as_of)
    D = f.st.deliveries
    if D is None or not len(D):
        raise Unanswerable("no deliveries")
    D = D[D["delivery_date"].map(lambda d: d is not None and d == d and s <= d <= e)]
    key = D[["order_id", "delivery_no"]].astype(str).agg("|".join, axis=1) if D["delivery_no"].notna().any() else \
        D[["order_id", "delivery_date"]].astype(str).agg("|".join, axis=1)
    return "number", int(key.nunique())


def h_tet_vs_before(it, f, as_of):
    """rulebook §2: so với trước Tết = cùng số ngày ngay trước 23 tháng Chạp."""
    # 'N ngày trước Tết' mentions the same Tết again: one anchor period (rulebook §2)
    ps = list({(p.start, p.end): p for p in it.periods if p.kind == "tet"}.values())
    if len(ps) != 1:
        raise Abstain("tet period")
    t = ps[0]
    n = it.args.get("n") or t.days
    m = it.args.get("metric", "revenue")
    s, e = ops.clip(f, t.start, t.end, as_of)
    bs, be = ops.clip(f, t.start - timedelta(days=n), t.start - timedelta(days=1), as_of)
    return "percent", pct_change(_metric_value(f, m, s, e, it.flt), _metric_value(f, m, bs, be, it.flt))


def _lot_remaining(f, d):
    LT, LS = f.st.lots, f.st.lot_sales
    if LT is None or not len(LT) or "expiry" not in LT or LT["expiry"].isna().all():
        raise Unanswerable("no lots with expiry")
    H = f.st.imports
    rd = dict(zip(H["receipt_id"], H["import_date"])) if H is not None and len(H) else {}
    LT = LT.copy()
    LT["rdate"] = LT["receipt_id"].map(rd)
    LT = LT[LT["rdate"].map(lambda x: x is None or x != x or x <= d)]
    if LS is not None and len(LS):
        O = f.O.set_index("order_id")
        LS = LS.copy()
        LS["od"] = LS["order_id"].map(O["order_date"])
        LS["st"] = LS["order_id"].map(O["status"])
        m = LS["od"].map(lambda x: x is not None and x == x and x <= d) & (LS["st"] != "cancelled")
        net = LS[m].groupby("lot")["qty_base"].sum() - LS[m].groupby("lot")["return_qty_base"].sum().fillna(0)
        LT["remain"] = LT["qty_base"] - LT["lot"].map(net).fillna(0)
    else:
        raise Abstain("no lot sales")
    return LT[LT["remain"] > 0]


def h_nearest_expiry(it, f, as_of):
    """rulebook §6 nhà thuốc: lô còn tồn chưa quá hạn tại hôm nay, hạn sớm nhất."""
    key = _need_product(it)
    d = _asof_day(it, as_of)
    LT = _lot_remaining(f, d)
    LT = LT[(LT["product_key"] == key) & LT["expiry"].map(lambda x: x is not None and x == x and x >= d)]
    if not len(LT):
        raise Abstain("no live lot")
    return "date", min(LT["expiry"]).isoformat()


def _expiring(it, f, as_of):
    d = _asof_day(it, as_of)
    ps = [p for p in it.periods if p.kind == "range"]
    if ps:
        s, e = ps[0].start, ps[0].end
    elif it.args.get("days"):
        s, e = d, d + timedelta(days=it.args["days"])
    else:
        ds = [p for p in it.periods if p.kind in ("asof", "day") and p.start > d]
        if not ds:
            raise Abstain("expiry window")
        s, e = d, ds[-1].start
    LT = _lot_remaining(f, s)
    return LT[LT["expiry"].map(lambda x: x is not None and x == x and s <= x <= e)]


def h_expiring_lots_count(it, f, as_of):
    return "number", int(len(_expiring(it, f, as_of)))


def h_expiring_products(it, f, as_of):
    """rulebook §9: liệt kê hàng theo hạn dùng: hạn sớm nhất trước, cùng hạn theo mã."""
    LT = _expiring(it, f, as_of)
    g = LT.groupby("product_key")["expiry"].min()
    keys = sorted(g.index, key=lambda k: (g[k], f.sort_key_product(k)))
    return "list", [f.pname(k) for k in keys]


HANDLERS.update({
    "debt_total": h_debt_total, "debt_customers": h_debt_customers, "debt_table": h_debt_table,
    "debt_aging": h_debt_aging, "debt_collected": h_debt_collected, "debt_customer_flow": h_debt_customer_flow,
    "gross_profit": h_gross_profit, "net_profit": h_net_profit, "delivery_count": h_delivery_count,
    "tet_vs_before": h_tet_vs_before, "nearest_expiry": h_nearest_expiry,
    "expiring_lots_count": h_expiring_lots_count, "expiring_products": h_expiring_products,
})
