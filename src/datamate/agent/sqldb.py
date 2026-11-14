"""SQL data layer for the text-to-SQL agent (D26): one read-only DuckDB database per shop.

The model writes SQL; DuckDB computes (CLAUDE.md rule 4). Tables are the ingested Store frames (`agent/tables.py`,
same as the pandas agent's `T`) plus canonical views built deterministically from `execute/facts.py`:

- `v_orders`: one row per order (date, time, status, customer key, revenue of the order...).
- `v_sales`: one row per sold line, already joined with its order and product (qty, qty_base, amount, category...).
- debt macros `debt_at(day)`, `debt_new(a, b)`, `debt_repaid(a, b)` built from `execute/debt.py`'s definitions.
- `v_imports` / `v_receipts` (non-cancelled import lines and receipts) and macros `stock_at(day)`, `lots_at(day)`
  built from `execute/inventory.py` and `dispatch._lot_remaining`.

Views carry the TRAIN rulebook reading of each rule and say so in the schema (`is_valid` = status done/shipping...).
When the split's rulebook changes the rows a view relies on (D18 `RulebookDiff.blocked`), the view stays but the
schema carries a warning naming it, so the model applies the split rulebook from the prompt on the raw columns
(status, raw_status...) instead. Dropping the column made every train example unusable on such a split.

Safety: after loading, external access is disabled and the configuration locked; `query` runs one SELECT/WITH
statement only, with a wall-clock timeout (`con.interrupt()`). One thread: identical inputs give identical results.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

MAX_ROWS = 200          # result rows kept from a model query (lists/tables in the gold are short)


class QueryError(Exception):
    """The model's SQL was rejected or failed (message is shown to the model in the repair turn)."""


def _dateish(nn: pd.Series) -> str | None:
    """'date' | 'datetime' when the non-null values are dates, else None."""
    if pd.api.types.is_datetime64_any_dtype(nn):
        return "datetime"
    v = nn.iloc[0]
    if isinstance(v, datetime) or isinstance(v, pd.Timestamp):
        return "datetime"
    if isinstance(v, date):
        return "date"
    return None


def clean_frame(df: pd.DataFrame, keep_empty: bool = False) -> pd.DataFrame:
    """Plain DuckDB-friendly dtypes: date columns -> datetime64 (cast to DATE later), mixed objects -> str/None.
    Column order and row order are kept (deterministic). Empty columns are dropped unless `keep_empty`."""
    out = {}
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        if not len(nn):
            if keep_empty:                             # macros reference it: keep as an all-NULL text column
                out[c] = pd.Series([None] * len(s), index=s.index, dtype=object)
            continue                                   # all empty: dropped (frames_of does the same)
        kind = _dateish(nn)
        if kind:
            out[c] = pd.to_datetime(s.map(lambda v: v if v is None or v == v else None), errors="coerce")
            if kind == "date":
                out[c] = out[c].dt.normalize()
            continue
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            out[c] = s
            continue
        if nn.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            out[c] = s.map(lambda v: None if v is None or v != v else bool(v)).astype("boolean")
            continue
        out[c] = s.map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v)).astype(object)
    return pd.DataFrame(out, index=df.index).reset_index(drop=True)


def _date_cols(df: pd.DataFrame) -> set[str]:
    return {c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) and
            (df[c].dropna().dt.normalize() == df[c].dropna()).all()}


@dataclass
class ShopDB:
    con: object
    tables: dict = field(default_factory=dict)       # name -> (n rows, [(col, doc)])
    notes: list = field(default_factory=list)        # warnings shown under the schema (rulebook changed ...)
    doc: dict = field(default_factory=dict)          # column name -> meaning, for v_orders / v_sales
    macros: list = field(default_factory=list)       # one schema line per table macro
    has_valid: bool = False                          # True: the split keeps the train valid-order rule (§1)
    has_debt: bool = False
    has_stock: bool = False
    has_lots: bool = False
    data_end: object = None

    def query(self, sql: str, timeout_s: float = 10.0) -> tuple[list[str], list[list]]:
        """Run ONE read-only statement. Returns (columns, rows) with at most MAX_ROWS rows. Raises QueryError."""
        import duckdb  # noqa: PLC0415
        try:
            stmts = self.con.extract_statements(sql)
        except Exception as e:  # noqa: BLE001
            raise QueryError(f"parse error: {e}") from None
        if len(stmts) != 1:
            raise QueryError(f"exactly one statement expected, got {len(stmts)}")
        if stmts[0].type != duckdb.StatementType.SELECT:
            raise QueryError(f"only SELECT/WITH queries are allowed, got {stmts[0].type.name}")
        timer = threading.Timer(timeout_s, self.con.interrupt)
        timer.start()
        try:
            cur = self.con.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(MAX_ROWS + 1)
        except Exception as e:  # noqa: BLE001
            msg = str(e).split("\n")[0][:300]
            raise QueryError(f"{type(e).__name__}: {msg}") from None
        finally:
            timer.cancel()
        if len(rows) > MAX_ROWS:
            raise QueryError(f"query returned more than {MAX_ROWS} rows; aggregate or add LIMIT")
        return cols, [list(r) for r in rows]

    def close(self):
        try:
            self.con.close()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------------------------------------- build

def _add(con, db: ShopDB, name: str, df: pd.DataFrame, order: list[str], keep_empty: bool = False):
    df = clean_frame(df, keep_empty)
    if not len(df.columns):
        return
    dcols = _date_cols(df)
    sel = ", ".join(f'CAST("{c}" AS DATE) AS "{c}"' if c in dcols else f'"{c}"' for c in df.columns)
    keys = [c for c in order if c in df.columns]
    ob = (" ORDER BY " + ", ".join(f'"{c}" NULLS LAST' for c in keys)) if keys else ""
    con.register("_src", df)
    con.execute(f'CREATE TABLE "{name}" AS SELECT {sel} FROM _src{ob}')
    con.unregister("_src")
    db.tables[name] = len(df)


def _customer_names(f) -> dict:
    """customer_key -> display name (Facts.customer_display: customer table name, else most frequent spelling)."""
    if "_cust_names" not in f.st.extra:
        f.st.extra["_cust_names"] = {k: f.customer_display(k) for k in sorted(f.O["ckey"].dropna().unique(), key=str)}
    return f.st.extra["_cust_names"]


def _hm(sec: int) -> str:
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}"


def _order_extras(f) -> pd.DataFrame:
    """Order-level filter columns, indexed by order_id: customer_group (rulebook §1: walk-in orders belong to no
    group), shift (rulebook §2 hours, re-read from the split rulebook by D18), channel_class (ops.channel_class)."""
    if "_sql_order_extras" in f.st.extra:
        return f.st.extra["_sql_order_extras"]
    from datamate.execute.ops import channel_class  # noqa: PLC0415
    O = f.O
    x = pd.DataFrame(index=O["order_id"].values)
    gm = f.customer_groups() if f.caps.get("has_customer_group") else {}
    if gm:
        x["customer_group"] = [gm.get(k) if isinstance(k, str) else None for k in O["customer_id"]]
    tod = pd.to_numeric(O["tod"], errors="coerce").values
    if f.caps.get("has_time") and (tod == tod).any():
        (mlo, mhi), (alo, ahi) = f.rules.morning, f.rules.afternoon
        x["shift"] = np.where((tod >= mlo) & (tod <= mhi), "morning",
                              np.where((tod >= alo) & (tod <= ahi), "afternoon", None))
    if f.caps.get("has_channel") and O["channel"].notna().any():
        x["channel_class"] = [channel_class(c) for c in O["channel"]]
    f.st.extra["_sql_order_extras"] = x
    return x


def _orders_view(f) -> pd.DataFrame:
    """v_orders: one row per order. revenue = sum of positive line amounts (rulebook §1.6, Facts.rev)."""
    O = f.O
    v = pd.DataFrame({"order_id": O["order_id"], "order_date": O["order_date"]})
    if O["order_dt"].notna().any():
        v["order_time"] = O["order_dt"]
    v["status"] = O["status"]
    if "_raw_status" in O and O["_raw_status"].notna().any():
        v["raw_status"] = O["_raw_status"]
    v["is_valid"] = O["valid"].astype(bool)
    v["customer_key"] = O["ckey"]
    if O["ckey"].notna().any():
        v["customer_name"] = O["ckey"].map(_customer_names(f))
    ex = _order_extras(f)
    for c in ex.columns:
        v[c] = ex[c].values
    for c in ("customer_id", "is_walkin", "channel", "payment_method", "staff_id", "note"):
        if c in O and O[c].notna().any():
            v[c] = O[c]
    v["revenue"] = O["rev"]
    for c in ("order_discount", "shipping_fee", "order_total"):
        if c in O and pd.to_numeric(O[c], errors="coerce").notna().any():
            v[c] = pd.to_numeric(O[c], errors="coerce")
    return v


def _sales_view(f) -> pd.DataFrame:
    """v_sales: one row per order line joined with its order and product (rulebook §1 revenue = Σ amount)."""
    L, O = f.L, f.O.set_index("order_id")
    v = pd.DataFrame({"order_id": L["order_id"], "order_date": L["order_id"].map(O["order_date"])})
    if O["order_dt"].notna().any():
        v["order_time"] = L["order_id"].map(O["order_dt"])
    v["status"] = L["status"]
    if "_raw_status" in O and O["_raw_status"].notna().any():
        v["raw_status"] = L["order_id"].map(O["_raw_status"])
    v["is_valid"] = L["valid"].astype(bool)
    v["customer_key"] = L["order_id"].map(O["ckey"])
    if O["ckey"].notna().any():
        v["customer_name"] = v["customer_key"].map(_customer_names(f))
    ex = _order_extras(f)
    ex = ex[~ex.index.duplicated()]
    for c in ex.columns:
        v[c] = L["order_id"].map(ex[c])
    for c in ("channel", "payment_method", "staff_id"):
        if c in O and O[c].notna().any():
            v[c] = L["order_id"].map(O[c])
    v["product_key"] = L["product_key"]
    if "product_id" in L and L["product_id"].notna().any():
        v["product_id"] = L["product_id"]
    # one display name per product (Facts.pname, what the rule executor answers); the line's own spelling is kept
    names = {k: f.pname(k) for k in sorted(L["product_key"].dropna().unique(), key=str)}
    v["product_name"] = L["product_key"].map(names)
    if (L["product_name"].astype(object) != v["product_name"]).any():
        v["product_name_raw"] = L["product_name"]
    for c in ("category", "unit", "qty", "base_unit", "qty_base", "unit_price", "line_discount", "amount",
              "cost_unit", "is_fee"):
        if c in L and L[c].notna().any():
            v[c] = L[c]
    P = f.P
    if "is_rx" in P and P["is_rx"].notna().any():          # rulebook §6 nhà thuốc
        rx = P["is_rx"].to_dict()
        v["is_rx"] = [rx.get(k) is True for k in L["product_key"]]
    if "active_ingredient" in P and P["active_ingredient"].notna().any():
        v["active_ingredient"] = L["product_key"].map(P["active_ingredient"].to_dict())
    return v


def _debt_tables(f) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Orders that can carry debt and their payments, as execute/debt.py defines them (rulebook §4)."""
    from datamate.execute.debt import _orders  # noqa: PLC0415
    from datamate.ingest.values import DEBT  # noqa: PLC0415
    P = f.st.payments
    if not f.caps.get("has_debt") or P is None or not len(P):
        return None
    D = _orders(f)
    names = _customer_names(f)
    d = pd.DataFrame({"order_id": D["order_id"], "order_date": D["order_date"], "customer_key": D["ckey"],
                      "customer_name": D["ckey"].map(names), "is_debt_sale": (D["pay"] == DEBT).astype(bool),
                      "total": pd.to_numeric(D["total"], errors="coerce").fillna(0.0)})
    p = pd.DataFrame({"order_id": P["order_id"],
                      "paid_date": P["paid_at"].map(lambda v: v.date() if isinstance(v, datetime) else v),
                      "amount": pd.to_numeric(P["amount"], errors="coerce").fillna(0.0)})
    return d, p


# debt macros mirror execute/debt.py: outstanding() (§4 nợ còn tại cuối ngày x), new_debt(), collected()
_DEBT_MACROS = [
    """CREATE MACRO debt_at(x) AS TABLE
       SELECT o.order_id, o.order_date, o.customer_key, o.customer_name, o.total,
              coalesce(p.paid, 0) AS paid, o.total - coalesce(p.paid, 0) AS owe,
              date_diff('day', o.order_date, CAST(x AS DATE)) AS age_days
       FROM _debt_orders o
       LEFT JOIN (SELECT order_id, sum(amount) AS paid FROM _debt_payments
                  WHERE paid_date <= CAST(x AS DATE) GROUP BY order_id) p USING (order_id)
       WHERE o.order_date <= CAST(x AS DATE) AND o.total - coalesce(p.paid, 0) > 0.5
       ORDER BY o.order_id""",
    """CREATE MACRO debt_new(a, b) AS TABLE
       SELECT o.order_id, o.order_date, o.customer_key, o.customer_name,
              greatest(o.total - coalesce(p.same_day, 0), 0) AS new_debt
       FROM _debt_orders o
       LEFT JOIN (SELECT dp.order_id, sum(dp.amount) AS same_day FROM _debt_payments dp
                  JOIN _debt_orders od USING (order_id) WHERE dp.paid_date = od.order_date
                  GROUP BY dp.order_id) p USING (order_id)
       WHERE o.is_debt_sale AND o.order_date BETWEEN CAST(a AS DATE) AND CAST(b AS DATE)
       ORDER BY o.order_id""",
    """CREATE MACRO debt_repaid(a, b) AS TABLE
       SELECT p.order_id, o.customer_key, o.customer_name, p.paid_date, p.amount
       FROM _debt_payments p JOIN _debt_orders o USING (order_id)
       WHERE o.is_debt_sale AND p.paid_date BETWEEN CAST(a AS DATE) AND CAST(b AS DATE)
         AND p.paid_date > o.order_date
       ORDER BY p.order_id, p.paid_date""",
]


# ---------------------------------------------------------------------------------- imports, stock, lots (§5, §6)

def _inventory(con, db: ShopDB, f):
    """v_imports / v_receipts, _items and the macros stock_at(x), lots_at(x): the same definitions as
    execute/inventory.py (import_lines, receipts, stock_at, lot_stock_at) and dispatch._lot_remaining."""
    from datamate.execute import inventory as inv  # noqa: PLC0415
    st = f.st
    has_imp = False
    if f.caps.get("has_imports") and st.import_lines is not None:
        IL = inv.import_lines(f)          # rulebook §5: cancelled receipts dropped, qty_base filled
        if len(IL):
            H = inv.receipts(f)
            sup = H["supplier"].where(H["supplier"].notna(), H.get("supplier_id")) if "supplier" in H else None
            vi = pd.DataFrame({"receipt_id": IL["receipt_id"], "import_date": IL["import_date"]})
            if sup is not None and sup.notna().any():
                vi["supplier"] = IL["receipt_id"].map(dict(zip(H["receipt_id"], sup)))
            names = {k: f.pname(k) for k in sorted(IL["product_key"].dropna().unique(), key=str)}
            vi["product_key"] = IL["product_key"]
            vi["product_name"] = IL["product_key"].map(names)
            for c in ("unit", "qty", "qty_base", "price", "amount", "lot", "expiry"):
                if c in IL and IL[c].notna().any():
                    vi[c] = IL[c]
            _add(con, db, "v_imports", vi, ["import_date", "receipt_id", "product_key"])
            vr = pd.DataFrame({"receipt_id": H["receipt_id"], "import_date": H["import_date"]})
            if sup is not None and sup.notna().any():
                vr["supplier"] = sup
            vr["receipt_total"] = pd.to_numeric(H["receipt_total"], errors="coerce")
            _add(con, db, "v_receipts", vr, ["import_date", "receipt_id"])
            has_imp = "qty_base" in vi
    S, LT, LS = st.stock, st.lots, st.lot_sales
    has_snap = S is not None and len(S) and f.dend is not None
    has_lots = LT is not None and len(LT) and LS is not None and len(LS)
    if not has_snap and not has_lots:
        return
    keys = set(f.P.index) | (set(S["product_key"].dropna()) if has_snap else set()) | \
        (set(LT["product_key"].dropna()) if has_lots else set())
    keys = sorted(keys, key=str)
    cost = pd.to_numeric(f.P["cost"], errors="coerce").to_dict()
    items = pd.DataFrame({"product_key": keys, "product_name": [f.pname(k) for k in keys],
                          "base_unit": [inv.base_unit(f, k) for k in keys], "cost": [cost.get(k) for k in keys]})
    _add(con, db, "_items", items, ["product_key"], keep_empty=True)
    lots = None
    if has_lots:
        H = st.imports
        rd = dict(zip(H["receipt_id"], H["import_date"])) if H is not None and len(H) else {}
        lots = pd.DataFrame({"lot": LT["lot"], "product_key": LT["product_key"],
                             "expiry": LT["expiry"] if "expiry" in LT else None,
                             "qty_base": pd.to_numeric(LT["qty_base"], errors="coerce"),
                             "receipt_date": LT["receipt_id"].map(rd)})
        O = f.O.set_index("order_id")
        ls = pd.DataFrame({"lot": LS["lot"], "product_key": LS["lot"].map(dict(zip(LT["lot"], LT["product_key"]))),
                           "order_date": LS["order_id"].map(O["order_date"]), "status": LS["order_id"].map(O["status"]),
                           "qty_base": pd.to_numeric(LS["qty_base"], errors="coerce"),
                           "return_qty_base": pd.to_numeric(LS["return_qty_base"], errors="coerce")})
        _add(con, db, "_lots", lots, ["lot"], keep_empty=True)
        _add(con, db, "_lot_sales", ls, ["lot", "order_date"], keep_empty=True)
        con.execute("ALTER TABLE _lots ALTER receipt_date TYPE DATE")
        con.execute("ALTER TABLE _lots ALTER expiry TYPE DATE")
        con.execute("ALTER TABLE _lot_sales ALTER order_date TYPE DATE")
    X = "CAST(x AS DATE)"
    if has_snap:        # inventory.stock_at: snapshot at data end rolled back with imports and sales after x
        end = f"DATE '{f.dend.isoformat()}'"
        stock = S.groupby("product_key", sort=True)["stock_qty"].sum().reset_index()
        _add(con, db, "_stock", stock, ["product_key"], keep_empty=True)
        imp = (f"LEFT JOIN (SELECT product_key, sum(qty_base) AS q FROM v_imports WHERE import_date > {X} "
               f"AND import_date <= {end} GROUP BY product_key) i USING (product_key)") if has_imp else ""
        sold = (f"LEFT JOIN (SELECT product_key, sum(qty_base) AS q FROM v_sales WHERE order_date > {X} "
                f"AND order_date <= {end} AND status <> 'cancelled' AND NOT is_fee GROUP BY product_key) o "
                "USING (product_key)") if "v_sales" in db.tables else ""
        expr = "s.stock_qty" + (" - coalesce(i.q, 0)" if imp else "") + (" + coalesce(o.q, 0)" if sold else "")
        con.execute(f"""CREATE MACRO stock_at(x) AS TABLE
            SELECT s.product_key, it.product_name, it.base_unit,
                   CASE WHEN {X} >= {end} THEN s.stock_qty ELSE {expr} END AS stock_qty, it.cost
            FROM _stock s LEFT JOIN _items it USING (product_key) {imp} {sold}
            ORDER BY s.product_key""")
    else:               # inventory.lot_stock_at: lot receipts up to x - lot sales of non-cancelled orders + returns
        con.execute(f"""CREATE MACRO stock_at(x) AS TABLE
            SELECT t.product_key, it.product_name, it.base_unit, t.stock_qty, it.cost
            FROM (SELECT product_key, sum(q) AS stock_qty FROM (
                    SELECT product_key, coalesce(qty_base, 0) AS q FROM _lots WHERE receipt_date <= {X}
                    UNION ALL
                    SELECT product_key, coalesce(return_qty_base, 0) - coalesce(qty_base, 0) FROM _lot_sales
                    WHERE order_date <= {X} AND status <> 'cancelled') GROUP BY product_key) t
            LEFT JOIN _items it USING (product_key) WHERE t.product_key IS NOT NULL
            ORDER BY t.product_key""")
    db.has_stock = True
    db.macros.append("stock_at(DATE 'x') -> product_key, product_name, base_unit, stock_qty, cost: tồn kho (đơn vị "
                     "cơ sở) của từng mặt hàng tại cuối ngày x; cost = giá vốn một đơn vị cơ sở trong danh mục")
    if lots is not None and lots["expiry"].notna().any():   # dispatch._lot_remaining
        con.execute(f"""CREATE MACRO lots_at(x) AS TABLE
            SELECT l.lot, l.product_key, it.product_name, l.expiry, l.qty_base - coalesce(n.net, 0) AS remain
            FROM _lots l
            LEFT JOIN (SELECT lot, sum(coalesce(qty_base, 0)) - sum(coalesce(return_qty_base, 0)) AS net
                       FROM _lot_sales WHERE order_date <= {X} AND status <> 'cancelled' GROUP BY lot) n USING (lot)
            LEFT JOIN _items it USING (product_key)
            WHERE (l.receipt_date IS NULL OR l.receipt_date <= {X}) AND l.qty_base - coalesce(n.net, 0) > 0
            ORDER BY l.expiry, l.lot""")
        db.has_lots = True
        db.macros.append("lots_at(DATE 'x') -> lot, product_key, product_name, expiry, remain: các lô còn tồn "
                         "(remain > 0, đơn vị cơ sở) tại cuối ngày x, kèm hạn dùng")


def _deliveries_view(st) -> pd.DataFrame | None:
    """rulebook §6 VLXD: one row per delivery round, as execute/dispatch.h_delivery_count counts them: the key is
    (order_id, delivery_no), or (order_id, delivery_date) when the shop records no round number. The raw table has
    one row per delivered item, so COUNT(*) on it over-counts rounds."""
    D = st.deliveries
    if D is None or not len(D) or "order_id" not in D or "delivery_date" not in D:
        return None
    D = D[D["delivery_date"].map(lambda d: d is not None and d == d)].copy()
    if not len(D):
        return None
    by_no = "delivery_no" in D and D["delivery_no"].notna().any()
    D["_k"] = D["delivery_no"].astype(str) if by_no else D["delivery_date"].astype(str)
    agg = {"delivery_date": ("delivery_date", "min"), "n_items": ("order_id", "size")}
    if by_no:
        agg["delivery_no"] = ("delivery_no", "first")
    for c, how in (("qty_base", "sum"), ("plate", "first"), ("driver", "first")):
        if c in D and D[c].notna().any():
            agg[c] = (c, how)
    V = D.groupby(["order_id", "_k"], sort=True, dropna=False).agg(**agg).reset_index().drop(columns="_k")
    cols = ["order_id"] + [c for c in ("delivery_no", "delivery_date", "n_items", "qty_base", "plate", "driver")
                           if c in V]
    return V[cols]


_WARN = (  # (scope entries of rulebook.SCOPES, what the warning names)
    (("revenue", "order_count"), "đơn hợp lệ/doanh thu (§1): cột is_valid và revenue theo định nghĩa mặc định"),
    (("debt_total",), "công nợ (§4): macro debt_at, debt_new, debt_repaid theo định nghĩa mặc định"),
    (("stock_qty", "stock_value"), "tồn kho (§5): macro stock_at theo định nghĩa mặc định"),
    (("nearest_expiry", "expiring_products"), "lô/hạn dùng (§6): macro lots_at theo định nghĩa mặc định"),
)


# rulebook row (train id) -> view columns / macros / views computed with the TRAIN reading of that row. When a split's
# rulebook changes or drops one of these rows, a SQL sample that still reads the artifact answers with the old rule
# (bench B 14/11: 11/12 rule_exception cells stale) -> it does not vote (stale_artifacts). Rows that only steer how a
# query is written (§1.2, §2.x periods...) have no artifact: their text is quoted in the prompt only.
RULE_ARTIFACTS = {
    "§1.1": ("is_valid",),                      # rulebook §1.1: đơn tính doanh thu = không huỷ, không trả
    "§1.5": ("is_valid",),                      # rulebook §1.5: số đơn = đơn không huỷ, không trả
    "§1.6": ("revenue", "amount"),              # rulebook §1.6: doanh thu một đơn = tổng thành tiền dòng
    "§2.12": ("shift",),                        # rulebook §2.12: ca sáng / ca chiều theo giờ tạo đơn
    "§3.1": ("is_fee",),                        # rulebook §3.1: dòng Phí phục vụ không phải mặt hàng
    "§4.1": ("debt_at", "debt_new", "debt_repaid"),   # rulebook §4: công nợ
    "§4.2": ("debt_at", "debt_new", "debt_repaid"),
    "§4.4": ("debt_at", "debt_new", "debt_repaid"),
    "§4.5": ("debt_at",),
    "§4.6": ("debt_new", "debt_repaid"),
    "§5.4": ("stock_at",),                      # rulebook §5: tồn kho
    "§5.5": ("stock_at",),
    "§5.6": ("stock_at",),
    "§5.10": ("cost_unit",),                    # rulebook §5.10: giá vốn
    "§6[nha thuoc].2": ("lots_at",),            # rulebook §6 nhà thuốc: lô, hạn dùng
    "§6[nha thuoc].4": ("lots_at",),
    "§6[vat lieu xay dung].1": ("v_deliveries",),   # rulebook §6 VLXD: đợt giao
}


def stale_artifacts(sql: str | None, rows: list[str]) -> list[str]:
    """Artifacts built on the train reading of a rulebook row in `rows` (rows this question touches that the split
    changed or dropped) that the query reads. Non-empty -> the sample keeps the old rule and must not vote."""
    if not sql or not rows:
        return []
    names = sorted({a for r in rows for a in RULE_ARTIFACTS.get(r, ())})
    return [a for a in names if re.search(rf"\b{a}\b", sql, re.I)]


def build_db(st, rules, rb=None) -> ShopDB:
    """DuckDB database of one shop. `rb` = RulebookDiff of the split (None: train rulebook, nothing changed)."""
    import duckdb  # noqa: PLC0415

    from datamate.agent.tables import frames_of  # noqa: PLC0415
    from datamate.execute.facts import facts_of  # noqa: PLC0415

    con = duckdb.connect(":memory:")
    con.execute("SET threads=1")
    db = ShopDB(con)
    blocked = (rb.blocked if rb is not None else {}) or {}
    for name, df in frames_of(st).items():
        _add(con, db, name, df, ["order_id", "receipt_id", "product_key", "product_id", "customer_id", "date"])
    if st.orders is not None and len(st.orders) and "order_date" in st.orders:
        f = facts_of(st, rules)
        db.data_end = f.dend
        db.has_valid = not any(blocked.get(k) for k in _WARN[0][0])
        _add(con, db, "v_orders", _orders_view(f), ["order_date", "order_id"])
        if f.has_lines:
            _add(con, db, "v_sales", _sales_view(f), ["order_date", "order_id", "product_key"])
        dt = _debt_tables(f)
        if dt is not None:
            _add(con, db, "_debt_orders", dt[0], ["order_id"], keep_empty=True)
            _add(con, db, "_debt_payments", dt[1], ["order_id", "paid_date"], keep_empty=True)
            for m in _DEBT_MACROS:
                con.execute(m)
            db.has_debt = True
            db.macros += [
                "debt_at(DATE 'x') -> order_id, order_date, customer_key, customer_name, total, paid, owe, age_days: "
                "đơn còn nợ (owe > 0) tính đến hết ngày x; age_days = tuổi nợ tính từ ngày đặt",
                "debt_new(DATE 'a', DATE 'b') -> order_id, order_date, customer_key, customer_name, new_debt: nợ "
                "phát sinh của đơn ghi nợ đặt trong [a, b] (phần chưa trả trong ngày mua)",
                "debt_repaid(DATE 'a', DATE 'b') -> order_id, customer_key, customer_name, paid_date, amount: khoản "
                "trả nợ có ngày trả trong [a, b] và sau ngày mua"]
        _inventory(con, db, f)
        dv = _deliveries_view(st)
        if dv is not None:
            _add(con, db, "v_deliveries", dv, ["delivery_date", "order_id", "delivery_no"])
        (mlo, mhi), (alo, ahi) = rules.morning, rules.afternoon
        db.doc = {
            "revenue": "doanh thu của đơn = tổng thành tiền các dòng (sau giảm giá dòng; chưa trừ giảm giá cả đơn, "
                       "không gồm phí giao)",
            "amount": "thành tiền của dòng, sau giảm giá dòng",
            "qty": "số lượng theo đơn vị bán (unit)",
            "qty_base": "số lượng quy về đơn vị cơ sở (base_unit)",
            "is_valid": "true = đơn được tính bán hàng/doanh thu: status 'done' hoặc 'shipping' (không huỷ, không "
                        "có trả hàng kể cả trả một phần)",
            "status": "trạng thái đã chuẩn hoá từ raw_status",
            "customer_key": "khoá khách hàng; NULL = khách lẻ/vãng lai",
            "is_fee": "true = dòng phí dịch vụ, không phải mặt hàng",
            "cost_unit": "giá vốn của một đơn vị bán (unit) ghi trên dòng",
            "shift": f"ca theo giờ đặt: morning {_hm(mlo)}–{_hm(mhi)}, afternoon {_hm(alo)}–{_hm(ahi)}, NULL ngoài ca",
            "channel_class": "kênh bán đã gom nhóm từ channel",
            "customer_group": "nhóm khách; NULL với khách lẻ/vãng lai",
            "order_total": "tổng đơn phải trả (đã trừ giảm giá cả đơn, gồm phí giao)",
            "is_rx": "true = thuốc kê đơn; chỉ lọc cột này khi câu hỏi nói rõ kê đơn / không kê đơn",
            "active_ingredient": "hoạt chất của thuốc; lọc bằng đúng giá trị trong GỢI Ý",
            "n_items": "số dòng mặt hàng trong đợt giao",
        }
        have = (True, db.has_debt, db.has_stock, db.has_lots)
        for (keys, what), flag in zip(_WARN, have):
            if flag and any(blocked.get(k) for k in keys):
                db.notes.append(f"RULEBOOK của bộ dữ liệu này có thay đổi ở phần {what}. Đọc kỹ rulebook: nếu định "
                                "nghĩa khác thì tự tính từ các cột gốc (status, raw_status, bảng gốc) thay vì dùng sẵn.")
    con.execute("SET enable_external_access=false")
    con.execute("SET lock_configuration=true")
    return db


def schema_text(db: ShopDB, max_enum: int = 15) -> str:
    """Compact schema of every table/view: row count and one line per column (type, values or range, meaning)."""
    out = []
    if db.data_end is not None:
        out.append(f"Dữ liệu bán hàng có đến hết ngày {db.data_end}.")
    names = [r[0] for r in db.con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name").fetchall()]
    canon = [n for n in ("v_orders", "v_sales", "v_imports", "v_receipts", "v_deliveries") if n in names]
    # raw tables that a canonical view or macro already covers: queryable, listed by name only (prompt length:
    # rulebook + README + schema must stay well inside max_model_len with the repair turn)
    covered = {"orders": "v_orders", "lines": "v_sales", "import_lines": "v_imports", "imports": "v_receipts",
               "stock": "stock_at", "lots": "lots_at", "lot_sales": "lots_at", "deliveries": "v_deliveries"}
    hidden = [n for n in names if covered.get(n) in names or (covered.get(n) == "stock_at" and db.has_stock)
              or (covered.get(n) == "lots_at" and db.has_lots)]
    rest = [n for n in names if n not in canon and n not in hidden and not n.startswith("_")]
    documented = set()
    for name in canon + rest:
        n = db.con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        out.append(f"{name}  ({n} dòng)")
        for col, typ in db.con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name=? "
                "ORDER BY ordinal_position", [name]).fetchall():
            doc = ""
            if name in ("v_orders", "v_sales", "v_deliveries") and col in db.doc and col not in documented:
                doc, _ = f"  -- {db.doc[col]}", documented.add(col)
            out.append(f"  - {col} {typ}: {_col_doc(db, name, col, typ, max_enum)}{doc}")
    if hidden:
        out.append("Bảng gốc (đã có view/macro chuẩn ở trên, chỉ dùng khi thật cần cột gốc): " + ", ".join(hidden))
    if db.macros:
        out.append("MACRO (gọi như bảng: SELECT ... FROM tên_macro(DATE '2026-01-31'))")
        out += [f"  - {m}" for m in db.macros]
    if db.notes:
        out.append("LƯU Ý")
        out += [f"  - {n}" for n in db.notes]
    return "\n".join(out)


def _col_doc(db: ShopDB, table: str, col: str, typ: str, max_enum: int) -> str:
    q = f'"{col}"'
    if typ in ("DATE", "TIMESTAMP", "TIMESTAMP_NS") or typ.startswith("TIMESTAMP"):
        lo, hi = db.con.execute(f'SELECT min({q}), max({q}) FROM "{table}"').fetchone()
        return f"{lo} .. {hi}"
    if typ == "BOOLEAN":
        return "true/false"
    if typ in ("DOUBLE", "BIGINT", "INTEGER", "FLOAT", "HUGEINT") or typ.startswith("DECIMAL"):
        lo, hi = db.con.execute(f'SELECT min({q}), max({q}) FROM "{table}"').fetchone()
        return f"min {lo:g} max {hi:g}" if lo is not None else "empty"
    vals = db.con.execute(f'SELECT {q}, count(*) c FROM "{table}" WHERE {q} IS NOT NULL GROUP BY 1 '
                          f'ORDER BY c DESC, 1 LIMIT {max_enum + 1}').fetchall()
    if len(vals) <= max_enum:
        return "values: " + " | ".join(f"'{v}' ({c})" for v, c in vals)
    n = db.con.execute(f'SELECT count(DISTINCT {q}) FROM "{table}"').fetchone()[0]
    return f"{n} distinct, e.g. " + " | ".join(repr(str(v)[:40]) for v, _ in vals[:3])
