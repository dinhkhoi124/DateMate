"""SQL data layer for the text-to-SQL agent (D26): one read-only DuckDB database per shop.

The model writes SQL; DuckDB computes (CLAUDE.md rule 4). Tables are the ingested Store frames (`agent/tables.py`,
same as the pandas agent's `T`) plus canonical views built deterministically from `execute/facts.py`:

- `v_orders`: one row per order (date, time, status, customer key, revenue of the order...).
- `v_sales`: one row per sold line, already joined with its order and product (qty, qty_base, amount, category...).
- debt macros `debt_at(day)`, `debt_new(a, b)`, `debt_repaid(a, b)` built from `execute/debt.py`'s definitions.

Rules baked into a view are TRAIN rulebook rules, so each is added only while the split's rulebook keeps the rows it
relies on (D18 `RulebookDiff.blocked`): `is_valid` (rulebook §1) only when no revenue/valid-sales row changed, debt
macros (rulebook §4) only when no debt row changed. Otherwise the model applies the split rulebook from the prompt.

Safety: after loading, external access is disabled and the configuration locked; `query` runs one SELECT/WITH
statement only, with a wall-clock timeout (`con.interrupt()`). One thread: identical inputs give identical results.
"""
from __future__ import annotations

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
    notes: list = field(default_factory=list)        # which canonical objects exist and why (prompt + trace)
    has_valid: bool = False
    has_debt: bool = False

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


def _orders_view(f, with_valid: bool) -> pd.DataFrame:
    """v_orders: one row per order. revenue = sum of positive line amounts (rulebook §1.6, Facts.rev)."""
    O = f.O
    v = pd.DataFrame({"order_id": O["order_id"], "order_date": O["order_date"]})
    if O["order_dt"].notna().any():
        v["order_time"] = O["order_dt"]
    v["status"] = O["status"]
    if "_raw_status" in O and O["_raw_status"].notna().any():
        v["raw_status"] = O["_raw_status"]
    if with_valid:
        v["is_valid"] = O["valid"].astype(bool)
    v["customer_key"] = O["ckey"]
    if O["ckey"].notna().any():
        v["customer_name"] = O["ckey"].map(_customer_names(f))
    for c in ("customer_id", "is_walkin", "channel", "payment_method", "staff_id", "note"):
        if c in O and O[c].notna().any():
            v[c] = O[c]
    v["revenue"] = O["rev"]
    for c in ("order_discount", "shipping_fee", "order_total"):
        if c in O and pd.to_numeric(O[c], errors="coerce").notna().any():
            v[c] = pd.to_numeric(O[c], errors="coerce")
    return v


def _sales_view(f, with_valid: bool) -> pd.DataFrame:
    """v_sales: one row per order line joined with its order and product (rulebook §1 revenue = Σ amount)."""
    L, O = f.L, f.O.set_index("order_id")
    v = pd.DataFrame({"order_id": L["order_id"], "order_date": L["order_id"].map(O["order_date"])})
    if O["order_dt"].notna().any():
        v["order_time"] = L["order_id"].map(O["order_dt"])
    v["status"] = L["status"]
    if with_valid:
        v["is_valid"] = L["valid"].astype(bool)
    v["customer_key"] = L["order_id"].map(O["ckey"])
    if O["ckey"].notna().any():
        v["customer_name"] = v["customer_key"].map(_customer_names(f))
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


def build_db(st, rules, rb=None) -> ShopDB:
    """DuckDB database of one shop. `rb` = RulebookDiff of the split (None: train rulebook, nothing blocked)."""
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
        db.has_valid = not blocked.get("revenue") and not blocked.get("order_count")
        _add(con, db, "v_orders", _orders_view(f, db.has_valid), ["order_date", "order_id"])
        if f.has_lines:
            _add(con, db, "v_sales", _sales_view(f, db.has_valid), ["order_date", "order_id", "product_key"])
        if not db.has_valid:
            db.notes.append("is_valid omitted: this rulebook changes the valid-order rule; apply it from status")
        if not blocked.get("debt_total"):
            dt = _debt_tables(f)
            if dt is not None:
                _add(con, db, "_debt_orders", dt[0], ["order_id"], keep_empty=True)
                _add(con, db, "_debt_payments", dt[1], ["order_id", "paid_date"], keep_empty=True)
                for m in _DEBT_MACROS:
                    con.execute(m)
                db.has_debt = True
        else:
            db.notes.append("debt macros omitted: this rulebook changes the debt rules")
    con.execute("SET enable_external_access=false")
    con.execute("SET lock_configuration=true")
    return db


def schema_text(db: ShopDB, max_enum: int = 15) -> str:
    """Compact schema of every table/view: row count and one line per column (type, values or range)."""
    out = []
    names = [r[0] for r in db.con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name").fetchall()]
    canon = [n for n in ("v_orders", "v_sales") if n in names]
    rest = [n for n in names if n not in canon and not n.startswith("_")]
    for name in canon + rest:
        n = db.con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        out.append(f"{name}  ({n} dòng)")
        for col, typ in db.con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name=? "
                "ORDER BY ordinal_position", [name]).fetchall():
            out.append(f"  - {col} {typ}: {_col_doc(db, name, col, typ, max_enum)}")
    if db.has_debt:
        out.append("debt_at(x)  -> order_id, order_date, customer_key, customer_name, total, paid, owe, age_days: "
                   "đơn còn nợ (owe > 0) tính đến hết ngày x")
        out.append("debt_new(a, b) -> order_id, order_date, customer_key, customer_name, new_debt: nợ phát sinh "
                   "của đơn ghi nợ đặt trong [a, b] (phần chưa trả trong ngày mua)")
        out.append("debt_repaid(a, b) -> order_id, customer_key, customer_name, paid_date, amount: khoản trả nợ "
                   "có ngày trả trong [a, b] và sau ngày mua")
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
