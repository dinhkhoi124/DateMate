"""Rule intent -> the reply the SQL agent should write (notes, TYPE, one query), for eval/build_fewshot_sql.py.

One template per executor op (execute/dispatch.py), written against the views and macros of agent/sqldb.py. A
template is only a CANDIDATE: build_fewshot_sql.py runs it and keeps the example only when the answer equals the
gold, so a template that reads a rule differently from the rulebook never reaches the few-shot bank. The match rate
per op is the check that the SQL layer carries the same semantics as the rule executor.

`CTX` is set per question by the caller: as_of (date), db (agent.sqldb.ShopDB), f (execute.facts.Facts).
"""
from __future__ import annotations

import dataclasses
from datetime import timedelta

from datamate.execute.dispatch import FLOW_KINDS, _cmp_periods
from datamate.ingest.values import fold

CTX: dict = {}

LINE_KEYS = ("product", "products", "category", "ingredient", "rx", "size", "base_unit")


class Skip(Exception):
    """No template for this intent (or the shop lacks the column): the question gives no example."""


def q_(s) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def d_(d) -> str:
    return f"DATE '{d.isoformat()}'"


def reply(atype: str, notes: list[str], sql: str) -> str:
    return "Notes:\n" + "\n".join(f"- {n}" for n in notes) + f"\nTYPE: {atype}\n```sql\n{sql}\n```"


# ------------------------------------------------------------------------------------------------- shop lookups

def cols(table: str) -> set:
    db = CTX["db"]
    cache = CTX.setdefault("_cols", {})
    key = (id(db), table)
    if key not in cache:
        cache[key] = {r[0] for r in db.con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [table]).fetchall()}
    return cache[key]


def need(table: str, *names: str):
    if not set(names) <= cols(table):
        raise Skip(f"{table} lacks {names}")


def exact(table: str, col: str, v, where: str = "") -> str:
    """The value of `col` as written in the data whose folded form equals fold(v) (filters copy data values)."""
    need(table, col)
    rows = CTX["db"].con.execute(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL {where} '
                                 f'ORDER BY 1').fetchall()
    hits = [r[0] for r in rows if isinstance(r[0], str) and fold(r[0]) == fold(str(v))]
    if len(hits) != 1:
        raise Skip(f"{col} value {v!r}")
    return hits[0]


# ------------------------------------------------------------------------------------------------- periods

def clip(p):
    """Flow periods never run past today (execute/ops.clip, agent/hints.period_hints)."""
    a = CTX["as_of"]
    return dataclasses.replace(p, end=a) if p.start <= a < p.end else p


def main_period(it):
    ps = [p for p in it.periods if p.kind in FLOW_KINDS]
    if not ps or len({(p.start, p.end) for p in ps}) > 1:
        raise Skip("period")
    return clip(ps[0])


def asof_day(it):
    """dispatch._asof_day: the day a stock / debt question is asked for."""
    ps = [p for p in it.periods if p.kind in ("asof", "day")]
    d = ps[0].start if ps else CTX["as_of"]
    if d > CTX["as_of"]:
        raise Skip("as-of after today")
    return d


def between(col: str, p) -> str:
    return f"{col} = {d_(p.start)}" if p.start == p.end else f"{col} BETWEEN {d_(p.start)} AND {d_(p.end)}"


def per(p) -> str:
    return f"kỳ: ngày {p.start}" if p.start == p.end else f"kỳ: {p.start} đến {p.end}"


# ------------------------------------------------------------------------------------------------- filters

def has_line_filter(it) -> bool:
    return any((it.flt or {}).get(k) is not None for k in LINE_KEYS)


def filters(it, table: str, skip: tuple = ()) -> tuple[list[str], list[str]]:
    """WHERE clauses + notes for the intent's filters on v_sales ('v_sales') or v_orders ('v_orders')."""
    w, n = [], []
    for k, v in (it.flt or {}).items():
        if v is None or k in skip:
            continue
        if k in LINE_KEYS and table != "v_sales":
            raise Skip(f"{k} on orders")
        if k == "product":
            w.append(f"product_key = {q_(v)}")
            n.append(f"lọc sản phẩm product_key = {q_(v)} (từ gợi ý)")
        elif k == "products":
            w.append("product_key IN (" + ", ".join(q_(x) for x in v) + ")")
            n.append("lọc các sản phẩm trong gợi ý")
        elif k == "category":
            x = exact(table, "category", v)
            w.append(f"category = {q_(x)}")
            n.append(f"lọc nhóm hàng category = {q_(x)}")
        elif k == "ingredient":
            x = exact(table, "active_ingredient", v)
            w.append(f"active_ingredient = {q_(x)}")
            n.append(f"lọc hoạt chất active_ingredient = {q_(x)} (thuốc phối hợp là loại riêng)")
        elif k == "rx":
            need(table, "is_rx")
            w.append("is_rx" if v else "NOT is_rx")
            n.append("thuốc kê đơn (is_rx)" if v else "thuốc không kê đơn (NOT is_rx)")
        elif k == "size":
            c = "product_name_raw" if "product_name_raw" in cols(table) else "product_name"
            w.append(f"lower(trim({c})) LIKE {q_('%(' + str(v).lower() + ')')}")
            n.append(f"size {v}: tên món kết thúc bằng ({v})")
        elif k == "base_unit":
            x = exact(table, "base_unit", v)
            w.append(f"base_unit = {q_(x)}")
            n.append(f"chỉ mặt hàng có đơn vị cơ sở {q_(x)}")
        elif k == "customer":
            w.append(f"customer_key = {q_(v)}")
            n.append(f"lọc khách customer_key = {q_(v)} (từ gợi ý)")
        elif k == "pay":
            need(table, "payment_method")
            w.append(f"payment_method = {q_(v)}")
            n.append(f"lọc thanh toán payment_method = {q_(v)}")
        elif k == "group":
            x = exact(table, "customer_group", v)
            w.append(f"customer_group = {q_(x)}")
            n.append(f"lọc nhóm khách customer_group = {q_(x)} (khách vãng lai không thuộc nhóm nào)")
        elif k == "channel":
            need(table, "channel_class")
            w.append(f"channel_class = {q_(v)}")
            n.append(f"lọc kênh channel_class = {q_(v)}")
        elif k == "staff":
            x = exact(table, "staff_id", v)
            w.append(f"staff_id = {q_(x)}")
            n.append(f"lọc nhân viên staff_id = {q_(x)}")
        elif k == "shift":
            need(table, "shift")
            w.append(f"shift = {q_(v)}")
            n.append(f"ca {'sáng' if v == 'morning' else 'chiều'}: shift = {q_(v)} (giờ ca theo rulebook §2)")
        else:
            raise Skip(f"filter {k}")
    return w, n


VALID = "chỉ đơn hợp lệ (is_valid, rulebook §1: bỏ đơn huỷ và đơn có trả hàng)"


def scope(it, p, table: str, skip: tuple = ()) -> tuple[str, list[str]]:
    """WHERE of valid orders/lines in period p with the intent's filters."""
    w, n = filters(it, table, skip)
    return " AND ".join([between("order_date", p), "is_valid"] + w), [per(p), VALID] + n


def metric(it, m: str, p, skip: tuple = ()) -> tuple[str, str]:
    """(scalar subquery, what it measures) for dispatch._metric_value: revenue | orders | qty."""
    line = has_line_filter(it)
    if m == "revenue":
        t, agg = ("v_sales", "SUM(amount)") if line else ("v_orders", "SUM(revenue)")
        what = "doanh thu"
    elif m == "orders":
        t, agg = ("v_sales", "COUNT(DISTINCT order_id)") if line else ("v_orders", "COUNT(*)")
        what = "số đơn"
    elif m == "qty":
        t, agg, what = "v_sales", "SUM(qty_base)", "số lượng (đơn vị cơ sở)"
    else:
        raise Skip(f"metric {m}")
    w, _ = scope(it, p, t, skip)
    if m == "qty" and not line:
        w += " AND NOT is_fee"
    return f"(SELECT {agg} FROM {t} WHERE {w})", what


def pct_change(a: str, b: str) -> str:
    return f"SELECT ROUND(100.0 * ({a} - {b}) / {b}, 2)"


# ------------------------------------------------------------------------------------------------- templates

def template(it) -> tuple[str, list[str], str]:
    """(TYPE, notes, SQL) for a rule intent; Skip when there is no template."""
    fn = TEMPLATES.get(it.op)
    if fn is None:
        raise Skip(it.op)
    return fn(it, it.args or {})


def t_revenue(it, a):
    p = main_period(it)
    if has_line_filter(it):
        w, n = scope(it, p, "v_sales")
        return "number", n + ["doanh thu = tổng amount của dòng bán (rulebook §1), làm tròn về đồng"], \
            f"SELECT ROUND(SUM(amount)) FROM v_sales WHERE {w}"
    w, n = scope(it, p, "v_orders")
    return "number", n + ["doanh thu = tổng revenue của đơn (rulebook §1), làm tròn về đồng"], \
        f"SELECT ROUND(SUM(revenue)) FROM v_orders WHERE {w}"


def t_order_count(it, a):
    p = main_period(it)
    if has_line_filter(it):
        w, n = scope(it, p, "v_sales")
        return "number", n + ["đếm số đơn khác nhau có dòng thoả điều kiện"], \
            f"SELECT COUNT(DISTINCT order_id) FROM v_sales WHERE {w}"
    w, n = scope(it, p, "v_orders")
    return "number", n + ["đếm số đơn hợp lệ"], f"SELECT COUNT(*) FROM v_orders WHERE {w}"


def _all_orders(it, p):
    w, n = filters(it, "v_orders")
    return " AND ".join([between("order_date", p)] + w), [per(p) + " (theo ngày đặt)"] + n


def t_cancel_count(it, a):
    w, n = _all_orders(it, main_period(it))
    return "number", n + ["đơn huỷ = status 'cancelled' (rulebook §1: trả hàng không phải huỷ)"], \
        f"SELECT COUNT(*) FROM v_orders WHERE {w} AND status = 'cancelled'"


def t_cancel_rate(it, a):
    w, n = _all_orders(it, main_period(it))
    return "percent", n + ["tỉ lệ huỷ = số đơn 'cancelled' / mọi đơn đặt trong kỳ (mọi trạng thái) (rulebook §1)"], \
        f"SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'cancelled') / COUNT(*), 2) FROM v_orders WHERE {w}"


def t_qty(it, a):
    p = main_period(it)
    w, n = scope(it, p, "v_sales")
    if not has_line_filter(it):
        w += " AND NOT is_fee"
    return "number", n + ["số lượng quy về đơn vị cơ sở: SUM(qty_base) (rulebook §3)"], \
        f"SELECT ROUND(SUM(qty_base), 2) FROM v_sales WHERE {w}"


def t_aov(it, a):
    p = main_period(it)
    rev, _ = metric(it, "revenue", p)
    cnt, _ = metric(it, "orders", p)
    _, n = scope(it, p, "v_sales" if has_line_filter(it) else "v_orders")
    if has_line_filter(it):
        return "number", n + ["giá trị trung bình đơn = doanh thu / số đơn, làm tròn về đồng (rulebook §3)"], \
            f"SELECT ROUND({rev} / {cnt})"
    w, _ = scope(it, p, "v_orders")
    return "number", n + ["giá trị trung bình đơn = doanh thu / số đơn, làm tròn về đồng (rulebook §3)"], \
        f"SELECT ROUND(SUM(revenue) / COUNT(*)) FROM v_orders WHERE {w}"


def t_share(it, a):
    p = main_period(it)
    if not any(v is not None for v in (it.flt or {}).values()):
        raise Skip("share of what")
    part, _ = metric(it, "revenue", p)
    _, n = scope(it, p, "v_sales" if has_line_filter(it) else "v_orders")
    whole = f"(SELECT SUM(revenue) FROM v_orders WHERE {between('order_date', p)} AND is_valid)"
    return "percent", n + ["tỉ trọng = doanh thu phần được lọc / doanh thu cả shop cùng kỳ (rulebook §1)"], \
        f"SELECT ROUND(100.0 * {part} / {whole}, 2)"


def t_growth(it, a):
    x, y = _cmp_periods(it, CTX["as_of"])
    x, y = clip(x), clip(y)
    m = a.get("metric", "revenue")
    A, what = metric(it, m, x)
    B, _ = metric(it, m, y)
    _, fn = filters(it, "v_sales" if has_line_filter(it) or m == "qty" else "v_orders")
    return "percent", [f"kỳ này: {x.start} đến {x.end}; kỳ so sánh: {y.start} đến {y.end}", VALID] + fn + [
        f"% thay đổi {what} = (kỳ này − kỳ so sánh) / kỳ so sánh × 100, giảm ra số âm (rulebook §1)"], pct_change(A, B)


def t_shift_compare(it, a):
    p = main_period(it)
    m = a.get("metric", "revenue")
    base = dataclasses.replace(it, flt={k: v for k, v in (it.flt or {}).items() if k != "shift"})
    A, what = metric(dataclasses.replace(base, flt={**base.flt, "shift": "afternoon"}), m, p)
    B, _ = metric(dataclasses.replace(base, flt={**base.flt, "shift": "morning"}), m, p)
    return "percent", [per(p), VALID, f"so {what} ca chiều (shift = 'afternoon') với ca sáng (shift = 'morning'), "
                       "mốc là ca sáng (rulebook §2)"], pct_change(A, B)


def t_tet_vs_before(it, a):
    ps = list({(p.start, p.end): p for p in it.periods if p.kind == "tet"}.values())
    if len(ps) != 1:
        raise Skip("tet period")
    t = ps[0]
    n = a.get("n") or t.days
    before = dataclasses.replace(t, start=t.start - timedelta(days=n), end=t.start - timedelta(days=1))
    m = a.get("metric", "revenue")
    A, what = metric(it, m, clip(t))
    B, _ = metric(it, m, before)
    return "percent", [f"dịp Tết: {t.start} đến {t.end} (từ gợi ý)",
                       f"trước Tết = {n} ngày ngay trước ngày bắt đầu: {before.start} đến {before.end} (rulebook §2)",
                       VALID, f"% thay đổi {what} so với trước Tết"], pct_change(A, B)


def _months(it):
    ps = list({p.label: p for p in it.periods if p.kind == "month"}.values())
    if len(ps) < 2:
        raise Skip("months")
    return [clip(p) for p in ps]


def _month_group(it, m: str, ps):
    line = has_line_filter(it)
    if m == "revenue":
        t, agg = ("v_sales", "SUM(amount)") if line else ("v_orders", "SUM(revenue)")
    elif m == "orders":
        t, agg = ("v_sales", "COUNT(DISTINCT order_id)") if line else ("v_orders", "COUNT(*)")
    elif m == "qty":
        t, agg = "v_sales", "SUM(qty_base)"
    else:
        raise Skip("metric")
    w, n = filters(it, t)
    if m == "qty" and not line:
        w.append("NOT is_fee")
    ps = sorted(ps, key=lambda p: p.start)
    contiguous = all(b.start == a.end + timedelta(days=1) for a, b in zip(ps, ps[1:]))
    rng = f"order_date BETWEEN {d_(ps[0].start)} AND {d_(ps[-1].end)}" if contiguous else \
        "(" + " OR ".join(between("order_date", p) for p in ps) + ")"
    return t, agg, " AND ".join([rng, "is_valid"] + w), ["các tháng: " + ", ".join(p.label for p in ps), VALID] + n


def t_argmax_month(it, a):
    t, agg, w, n = _month_group(it, a.get("metric", "revenue"), _months(it))
    return "date", n + ["gom theo tháng, lấy tháng cao nhất; trả 'YYYY-MM'"], \
        (f"SELECT strftime(order_date, '%Y-%m') AS thang FROM {t} WHERE {w} GROUP BY thang "
         f"ORDER BY {agg} DESC, thang LIMIT 1")


def t_revenue_series(it, a):
    ps = [clip(p) for p in it.periods if p.kind == "month"]
    if len(ps) < 2 or ps != sorted(ps, key=lambda p: p.start) or len({p.label for p in ps}) != len(ps):
        raise Skip("series order")
    t, agg, w, n = _month_group(it, "revenue", ps)
    return "list", n + ["doanh thu từng tháng theo thứ tự thời gian, làm tròn về đồng"], \
        f"SELECT ROUND({agg}) FROM {t} WHERE {w} GROUP BY strftime(order_date, '%Y-%m') ORDER BY strftime(order_date, '%Y-%m')"


def t_argmax_day(it, a):
    if a.get("metric", "revenue") != "revenue":
        raise Skip("metric")
    p = main_period(it)
    t, agg = ("v_sales", "SUM(amount)") if has_line_filter(it) else ("v_orders", "SUM(revenue)")
    w, n = scope(it, p, t)
    return "date", n + ["doanh thu theo ngày, lấy ngày cao nhất; hoà thì ngày sớm hơn"], \
        f"SELECT order_date FROM {t} WHERE {w} GROUP BY order_date ORDER BY {agg} DESC, order_date LIMIT 1"


def t_top_products(it, a):
    p = main_period(it)
    w, n = scope(it, p, "v_sales")
    by = a.get("by", "qty")
    agg = {"qty": "SUM(qty_base)", "revenue": "SUM(amount)", "profit": "SUM(amount - qty * cost_unit)"}.get(by)
    if agg is None:
        raise Skip("by")
    if by == "profit":
        need("v_sales", "cost_unit")
    crit = {"qty": "số lượng quy về đơn vị cơ sở (rulebook §3: bán chạy mặc định theo số lượng)",
            "revenue": "doanh thu", "profit": "lãi gộp = amount − qty × cost_unit (rulebook §5)"}[by]
    k = int(a.get("k", 1))
    return ("list" if it.atype == "list" else "string"), \
        n + [f"xếp hạng theo {crit}; bỏ phí dịch vụ (NOT is_fee); hoà thì theo mã tăng dần; lấy {k}"], \
        (f"SELECT product_name FROM v_sales WHERE {w} AND NOT is_fee GROUP BY product_key, product_name "
         f"HAVING {agg} > 0 ORDER BY {agg} DESC, product_key LIMIT {k}")


def t_top_customers(it, a):
    p = main_period(it)
    w, n = scope(it, p, "v_orders")
    agg = "SUM(revenue)" if a.get("by", "revenue") == "revenue" else "COUNT(*)"
    k = int(a.get("k", 1))
    return ("list" if it.atype == "list" else "string"), \
        n + ["gom theo customer_key, bỏ khách lẻ (customer_key IS NOT NULL) (rulebook §3); hoà thì theo mã"], \
        (f"SELECT customer_name FROM v_orders WHERE {w} AND customer_key IS NOT NULL "
         f"GROUP BY customer_key, customer_name ORDER BY {agg} DESC, customer_key LIMIT {k}")


def t_top_staff(it, a):
    need("v_orders", "staff_id")
    w, n = scope(it, main_period(it), "v_orders")
    return "string", n + ["doanh thu theo nhân viên (staff_id), lấy người cao nhất"], \
        (f"SELECT staff_id FROM v_orders WHERE {w} AND staff_id IS NOT NULL GROUP BY staff_id "
         "ORDER BY SUM(revenue) DESC, staff_id LIMIT 1")


def _cmp(a) -> str:
    return ">=" if a.get("inclusive", True) else ">"


def t_days_revenue_over(it, a):
    p = main_period(it)
    t, agg = ("v_sales", "SUM(amount)") if has_line_filter(it) else ("v_orders", "SUM(revenue)")
    w, n = scope(it, p, t)
    return "number", n + [f"tính tổng doanh thu từng ngày, đếm số ngày có tổng {_cmp(a)} {int(a['x'])} (rulebook §2: "
                          "'từ ... trở lên' gồm cả mốc, 'trên/hơn' không gồm)"], \
        (f"SELECT COUNT(*) FROM (SELECT order_date, {agg} AS r FROM {t} WHERE {w} GROUP BY order_date) "
         f"WHERE r {_cmp(a)} {int(a['x'])}")


def t_orders_revenue_over(it, a):
    w, n = scope(it, main_period(it), "v_orders")
    return "number", n + [f"đếm số đơn có doanh thu đơn {_cmp(a)} {int(a['x'])}"], \
        f"SELECT COUNT(*) FROM v_orders WHERE {w} AND revenue {_cmp(a)} {int(a['x'])}"


def _n(a) -> int:
    return int(a["n"]) if a.get("inclusive", True) else int(a["n"]) + 1


def t_items_in_n_orders(it, a):
    w, n = scope(it, main_period(it), "v_sales")
    return "number", n + [f"đếm mặt hàng (bỏ phí dịch vụ) xuất hiện trong ít nhất {_n(a)} đơn khác nhau"], \
        (f"SELECT COUNT(*) FROM (SELECT product_key FROM v_sales WHERE {w} AND NOT is_fee AND qty_base > 0 "
         f"GROUP BY product_key HAVING COUNT(DISTINCT order_id) >= {_n(a)})")


def t_customers_min_orders(it, a):
    w, n = scope(it, main_period(it), "v_orders")
    return "number", n + [f"đếm khách (customer_key, bỏ khách lẻ) có ít nhất {_n(a)} đơn trong kỳ"], \
        (f"SELECT COUNT(*) FROM (SELECT customer_key FROM v_orders WHERE {w} AND customer_key IS NOT NULL "
         f"GROUP BY customer_key HAVING COUNT(*) >= {_n(a)})")


def t_customers_min_orders_pct(it, a):
    w, n = scope(it, main_period(it), "v_orders")
    return "percent", n + [f"mẫu số = khách đã mua trong kỳ (có ít nhất một đơn hợp lệ, bỏ khách lẻ); tử số = khách có "
                           f"ít nhất {_n(a)} đơn (rulebook §1)"], \
        (f"SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE c >= {_n(a)}) / COUNT(*), 2) FROM (SELECT customer_key, "
         f"COUNT(*) AS c FROM v_orders WHERE {w} AND customer_key IS NOT NULL GROUP BY customer_key)")


def t_dead_count(it, a):
    p = main_period(it)
    base = "is_valid AND NOT is_fee AND qty_base > 0"
    return "number", [per(p), "hàng 'ế' = đã từng bán trước kỳ (đơn hợp lệ) nhưng trong kỳ không có đơn hợp lệ nào "
                      "mua (rulebook §5)"], \
        (f"SELECT COUNT(*) FROM (SELECT DISTINCT product_key FROM v_sales WHERE {base} AND order_date < {d_(p.start)}) "
         f"WHERE product_key NOT IN (SELECT product_key FROM v_sales WHERE {base} AND {between('order_date', p)})")


# ---- debt (rulebook §4)

def _debt():
    if not CTX["db"].has_debt:
        raise Skip("no debt macros")


def t_debt_customers(it, a):
    _debt()
    d = asof_day(it)
    return "number", [f"tính đến hết ngày {d}", "khách còn nợ = khách có đơn còn nợ (macro debt_at, rulebook §4); "
                      "khách lẻ không tính là một khách"], \
        f"SELECT COUNT(DISTINCT customer_key) FROM debt_at({d_(d)}) WHERE customer_key IS NOT NULL"


def t_debt_total(it, a):
    _debt()
    d = asof_day(it)
    age = a.get("min_age")
    if age is None:
        return "number", [f"tính đến hết ngày {d}", "tổng nợ = tổng phần còn nợ (owe) của các đơn (rulebook §4)"], \
            f"SELECT ROUND(SUM(owe)) FROM debt_at({d_(d)})"
    return "number", [f"tính đến hết ngày {d}", f"nợ quá {int(age)} ngày: age_days > {int(age)} (rulebook §2: 'quá/hơn N' "
                      "không gồm N; tuổi nợ tính từ ngày đặt)"], \
        f"SELECT ROUND(SUM(owe)) FROM debt_at({d_(d)}) WHERE age_days > {int(age)}"


def t_debt_table(it, a):
    _debt()
    d = asof_day(it)
    k = int(a.get("k", 3))
    return "table", [f"tính đến hết ngày {d}", f"nợ theo khách (bỏ khách lẻ), {k} khách nợ nhiều nhất; hoà thì theo "
                     "mã; cột: khach_hang, so_tien_no"], \
        (f"SELECT customer_name AS khach_hang, ROUND(SUM(owe)) AS so_tien_no FROM debt_at({d_(d)}) "
         f"WHERE customer_key IS NOT NULL GROUP BY customer_key, customer_name "
         f"ORDER BY SUM(owe) DESC, customer_key LIMIT {k}")


def t_debt_aging(it, a):
    _debt()
    d = asof_day(it)
    return "list", [f"tính đến hết ngày {d}", "ba nhóm tuổi nợ: 0–7 ngày, 8–15 ngày, trên 15 ngày (rulebook §4)"], \
        ("SELECT unnest([ROUND(COALESCE(SUM(owe) FILTER (WHERE age_days <= 7), 0)), "
         "ROUND(COALESCE(SUM(owe) FILTER (WHERE age_days BETWEEN 8 AND 15), 0)), "
         f"ROUND(COALESCE(SUM(owe) FILTER (WHERE age_days > 15), 0))]) FROM debt_at({d_(d)})")


def t_debt_collected(it, a):
    _debt()
    p = main_period(it)
    return "number", [per(p), "thu nợ = khoản trả sau ngày mua có ngày trả trong kỳ (macro debt_repaid, rulebook §4)"], \
        f"SELECT ROUND(SUM(amount)) FROM debt_repaid({d_(p.start)}, {d_(p.end)})"


def t_debt_customer_flow(it, a):
    _debt()
    c = (it.flt or {}).get("customer")
    if c is None:
        raise Skip("no customer")
    p = main_period(it)
    rng = f"{d_(p.start)}, {d_(p.end)}"
    return "list", [per(p), f"khách customer_key = {q_(c)} (từ gợi ý)",
                    "trả lời [nợ thêm, đã trả] trong kỳ: debt_new và debt_repaid (rulebook §4)"], \
        (f"SELECT unnest([(SELECT ROUND(COALESCE(SUM(new_debt), 0)) FROM debt_new({rng}) WHERE customer_key = {q_(c)}), "
         f"(SELECT ROUND(COALESCE(SUM(amount), 0)) FROM debt_repaid({rng}) WHERE customer_key = {q_(c)})])")


# ---- profit, deliveries (rulebook §5, §6)

def t_gross_profit(it, a):
    need("v_sales", "cost_unit")
    w, n = scope(it, main_period(it), "v_sales")
    return "number", n + ["lãi gộp = doanh thu − số lượng (đơn vị bán) × giá vốn một đơn vị bán trên dòng: "
                          "SUM(amount) − SUM(qty × cost_unit) (rulebook §5)"], \
        f"SELECT ROUND(SUM(amount) - SUM(qty * COALESCE(cost_unit, 0))) FROM v_sales WHERE {w}"


def t_net_profit(it, a):
    need("v_sales", "cost_unit")
    need("expenses", "date", "amount")
    p = main_period(it)
    if p.kind != "month":
        raise Skip("net profit needs a month")
    w, n = scope(it, p, "v_sales")
    return "number", n + ["lãi ròng tháng = lãi gộp tháng − mọi khoản trong sổ chi phí (expenses) của tháng "
                          "(rulebook §5)"], \
        (f"SELECT ROUND((SELECT SUM(amount) - SUM(qty * COALESCE(cost_unit, 0)) FROM v_sales WHERE {w}) - "
         f"(SELECT COALESCE(SUM(amount), 0) FROM expenses WHERE {between('date', p)}))")


def t_delivery_count(it, a):
    need("v_deliveries", "order_id", "delivery_date")
    p = main_period(it)
    return "number", [per(p) + " (theo ngày giao)", "mỗi đợt giao của mỗi đơn là một đợt (v_deliveries: một dòng một "
                      "đợt), kể cả đơn sau đó trả hàng (rulebook §6)"], \
        f"SELECT COUNT(*) FROM v_deliveries WHERE {between('delivery_date', p)}"


# ---- imports, stock, lots (rulebook §5, §6)

def _product(it):
    k = (it.flt or {}).get("product")
    if k is None:
        raise Skip("no product")
    return k


def t_import_total(it, a):
    need("v_receipts", "receipt_total")
    p = main_period(it)
    return "number", [per(p) + " (theo ngày phiếu nhập)", "tổng tiền các phiếu nhập không huỷ (v_receipts) "
                      "(rulebook §5)"], \
        f"SELECT ROUND(SUM(receipt_total)) FROM v_receipts WHERE {between('import_date', p)}"


def t_top_supplier(it, a):
    need("v_receipts", "supplier", "receipt_total")
    p = main_period(it)
    return "string", [per(p) + " (theo ngày phiếu nhập)", "nhà cung cấp có tổng tiền phiếu nhập (không huỷ) cao nhất"], \
        (f"SELECT supplier FROM v_receipts WHERE {between('import_date', p)} AND supplier IS NOT NULL "
         "GROUP BY supplier ORDER BY SUM(receipt_total) DESC, supplier LIMIT 1")


def t_avg_import_price(it, a):
    need("v_imports", "amount", "qty")
    k, p = _product(it), main_period(it)
    return "number", [per(p) + " (theo ngày phiếu nhập)", f"sản phẩm product_key = {q_(k)} (từ gợi ý)",
                      "giá nhập bình quân theo số lượng = Σ tiền / Σ số lượng (đơn vị nhập), bỏ phiếu huỷ "
                      "(rulebook §5)"], \
        (f"SELECT ROUND(SUM(amount) / SUM(qty)) FROM v_imports WHERE product_key = {q_(k)} "
         f"AND {between('import_date', p)}")


def t_import_minus_sales(it, a):
    need("v_imports", "qty_base")
    k, p = _product(it), main_period(it)
    return "number", [per(p), f"sản phẩm product_key = {q_(k)} (từ gợi ý)",
                      "nhập trừ bán, cùng đơn vị cơ sở: SUM(qty_base) nhập − SUM(qty_base) bán của đơn hợp lệ "
                      "(rulebook §5)"], \
        (f"SELECT (SELECT COALESCE(SUM(qty_base), 0) FROM v_imports WHERE product_key = {q_(k)} AND "
         f"{between('import_date', p)}) - (SELECT COALESCE(SUM(qty_base), 0) FROM v_sales WHERE product_key = {q_(k)} "
         f"AND is_valid AND {between('order_date', p)})")


def _stock():
    if not CTX["db"].has_stock:
        raise Skip("no stock macro")


def t_stock_qty(it, a):
    _stock()
    k, d = _product(it), asof_day(it)
    return "number", [f"tồn tại cuối ngày {d} (macro stock_at)", f"sản phẩm product_key = {q_(k)} (từ gợi ý)",
                      "tồn kho tính theo đơn vị cơ sở"], \
        f"SELECT ROUND(stock_qty, 2) FROM stock_at({d_(d)}) WHERE product_key = {q_(k)}"


def t_stock_big(it, a):
    _stock()
    k, d = _product(it), asof_day(it)
    if not a.get("big"):
        raise Skip("big unit")
    u = exact("units", "unit", a["big"], f"AND product_key = {q_(k)}")
    fac = f"(SELECT MAX(factor) FROM units WHERE product_key = {q_(k)} AND unit = {q_(u)})"
    return "list", [f"tồn tại cuối ngày {d} (macro stock_at)", f"sản phẩm product_key = {q_(k)} (từ gợi ý)",
                    f"tồn theo đơn vị lớn {q_(u)} = [số đơn vị lớn nguyên, số đơn vị cơ sở lẻ]; hệ số quy đổi lấy từ "
                    "bảng units (rulebook §5)"], \
        (f"SELECT unnest([floor(stock_qty / {fac}), round(stock_qty - floor(stock_qty / {fac}) * {fac})]) "
         f"FROM stock_at({d_(d)}) WHERE product_key = {q_(k)}")


def t_stock_value(it, a):
    _stock()
    d = asof_day(it)
    return "number", [f"tồn tại cuối ngày {d} (macro stock_at)", "giá trị tồn = tồn × giá vốn một đơn vị cơ sở trong "
                      "danh mục (cost), chỉ hàng còn tồn (rulebook §5)"], \
        f"SELECT ROUND(SUM(stock_qty * cost)) FROM stock_at({d_(d)}) WHERE stock_qty > 0"


def t_stock_out_count(it, a):
    _stock()
    d = asof_day(it)
    return "number", [f"tồn tại cuối ngày {d} (macro stock_at)", "hết hàng = tồn bằng 0 (rulebook §5; hết hàng khác "
                      "hết hạn)"], \
        f"SELECT COUNT(*) FROM stock_at({d_(d)}) WHERE round(stock_qty, 6) = 0"


def _lots():
    if not CTX["db"].has_lots:
        raise Skip("no lots macro")


def t_nearest_expiry(it, a):
    _lots()
    k, d = _product(it), asof_day(it)
    return "date", [f"lô còn tồn tại cuối ngày {d} (macro lots_at)", f"sản phẩm product_key = {q_(k)} (từ gợi ý)",
                    "chỉ lô chưa quá hạn (expiry >= hôm đó), lấy hạn sớm nhất (rulebook §6)"], \
        f"SELECT MIN(expiry) FROM lots_at({d_(d)}) WHERE product_key = {q_(k)} AND expiry >= {d_(d)}"


def _expiry_window(it, a):
    d = asof_day(it)
    ps = [p for p in it.periods if p.kind == "range"]
    if ps:
        return ps[0].start, ps[0].end
    if a.get("days"):
        return d, d + timedelta(days=int(a["days"]))
    ds = [p for p in it.periods if p.kind in ("asof", "day") and p.start > d]
    if not ds:
        raise Skip("expiry window")
    return d, ds[-1].start


def t_expiring_lots_count(it, a):
    _lots()
    s, e = _expiry_window(it, a)
    return "number", [f"lô còn tồn tại cuối ngày {s} (macro lots_at)", f"hạn dùng từ {s} đến hết {e} (gồm cả hai đầu)",
                      "đếm số lô"], \
        f"SELECT COUNT(*) FROM lots_at({d_(s)}) WHERE expiry BETWEEN {d_(s)} AND {d_(e)}"


def t_expiring_products(it, a):
    _lots()
    s, e = _expiry_window(it, a)
    return "list", [f"lô còn tồn tại cuối ngày {s} (macro lots_at)", f"hạn dùng từ {s} đến hết {e} (gồm cả hai đầu)",
                    "mỗi mặt hàng một dòng; xếp hạn sớm nhất trước, cùng hạn theo mã (rulebook §9)"], \
        (f"SELECT product_name FROM lots_at({d_(s)}) WHERE expiry BETWEEN {d_(s)} AND {d_(e)} "
         "GROUP BY product_key, product_name ORDER BY MIN(expiry), product_key")


def t_price_effect(it, a):
    k = _product(it)
    d, n = a.get("day"), int(a["n"])
    if d is None:
        raise Skip("increase day not in the question")
    def qty(s, e):
        return (f"(SELECT SUM(qty_base) FROM v_sales WHERE product_key = {q_(k)} AND is_valid AND "
                f"order_date BETWEEN {d_(s)} AND {d_(e)})")
    after, before = (d, d + timedelta(days=n - 1)), (d - timedelta(days=n), d - timedelta(days=1))
    return "percent", [f"sản phẩm product_key = {q_(k)} (từ gợi ý)",
                       f"{n} ngày kể từ ngày tăng giá (gồm ngày đó): {after[0]} đến {after[1]}; {n} ngày ngay trước: "
                       f"{before[0]} đến {before[1]} (rulebook §5)", "% thay đổi số lượng bán (đơn vị cơ sở), đơn hợp lệ"], \
        pct_change(qty(*after), qty(*before))


def t_unit_price(it, a):
    k, d = _product(it), a.get("day")
    if d is None:
        raise Skip("no day")
    need("v_sales", "unit_price", "qty", "qty_base")
    return "number", [f"ngày {d}", f"sản phẩm product_key = {q_(k)} (từ gợi ý)",
                      "đơn giá trên dòng bán của ngày đó cho MỘT đơn vị cơ sở, trước chiết khấu: unit_price / "
                      "(qty_base / qty); tính cả đơn huỷ/trả; lấy giá xuất hiện nhiều nhất (rulebook §5)"], \
        (f"SELECT ROUND(mode(round(unit_price * qty / qty_base, 4))) FROM v_sales WHERE product_key = {q_(k)} "
         f"AND order_date = {d_(d)} AND qty > 0")


def t_revenue_by(it, a):
    p = main_period(it)
    dim, vals = a.get("dim"), list(dict.fromkeys(a.get("values") or []))
    if not dim or len(vals) < 2:
        raise Skip("revenue_by")
    parts, notes = [], [per(p), VALID]
    for v in vals:
        sub = dataclasses.replace(it, flt={**{k: x for k, x in (it.flt or {}).items() if k != dim}, dim: v})
        m, _ = metric(sub, "revenue", p)
        parts.append(f"ROUND(COALESCE({m}, 0))")
        notes += filters(sub, "v_sales" if has_line_filter(sub) else "v_orders")[1][-1:]
    return "list", notes + ["doanh thu từng nhóm theo đúng thứ tự câu hỏi nêu"], \
        "SELECT unnest([" + ", ".join(parts) + "])"


def t_monotonic(it, a):
    ps = _months(it)
    m = a.get("metric", "revenue")
    vals = [metric(it, m, p)[0] for p in ps]       # in the order the question names the months
    cmp = ">" if a.get("dir") == "up" else "<"
    word = "tăng" if cmp == ">" else "giảm"
    conds = " AND ".join(f"COALESCE({b}, 0) {cmp} COALESCE({x}, 0)" for x, b in zip(vals, vals[1:]))
    return "boolean", ["các tháng theo thứ tự câu hỏi: " + ", ".join(p.label for p in ps), VALID,
                       f"true khi tháng sau luôn {word} hơn tháng trước"], f"SELECT {conds}"


TEMPLATES = {
    "unit_price": t_unit_price, "revenue_by": t_revenue_by, "monotonic": t_monotonic,
    "revenue": t_revenue, "order_count": t_order_count, "cancel_count": t_cancel_count, "cancel_rate": t_cancel_rate,
    "qty": t_qty, "aov": t_aov, "share": t_share, "growth": t_growth, "shift_compare": t_shift_compare,
    "tet_vs_before": t_tet_vs_before, "argmax_month": t_argmax_month, "revenue_series": t_revenue_series,
    "argmax_day": t_argmax_day, "top_products": t_top_products, "top_customers": t_top_customers,
    "top_staff": t_top_staff, "days_revenue_over": t_days_revenue_over, "orders_revenue_over": t_orders_revenue_over,
    "items_in_n_orders": t_items_in_n_orders, "customers_min_orders": t_customers_min_orders,
    "customers_min_orders_pct": t_customers_min_orders_pct, "dead_count": t_dead_count,
    "debt_customers": t_debt_customers, "debt_total": t_debt_total, "debt_table": t_debt_table,
    "debt_aging": t_debt_aging, "debt_collected": t_debt_collected, "debt_customer_flow": t_debt_customer_flow,
    "gross_profit": t_gross_profit, "net_profit": t_net_profit, "delivery_count": t_delivery_count,
    "import_total": t_import_total, "top_supplier": t_top_supplier, "avg_import_price": t_avg_import_price,
    "import_minus_sales": t_import_minus_sales, "stock_qty": t_stock_qty, "stock_big": t_stock_big,
    "stock_value": t_stock_value, "stock_out_count": t_stock_out_count, "nearest_expiry": t_nearest_expiry,
    "expiring_lots_count": t_expiring_lots_count, "expiring_products": t_expiring_products,
    "price_effect": t_price_effect,
}
