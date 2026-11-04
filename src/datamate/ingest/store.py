"""Assemble classified tables into the canonical Store (docs/ARCHITECTURE.md §3).

Store.build(shop_dir) -> Store with pandas DataFrames:
  orders, lines, products, units, customers, payments, imports, import_lines, stock, price_history,
  lots, lot_sales, deliveries, debt_terms, expenses  + caps (capability profile) + warnings.
"""
from __future__ import annotations

import hashlib
import pickle
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from datamate.ingest.readme import ReadmeSpec, parse_readme
from datamate.ingest.reader import read_shop
from datamate.ingest.tables import Table, build_table
from datamate.ingest.values import (
    CANCELLED, DONE, RETURNED, SHIPPING, clean_text, fold, is_blank, is_walkin, parse_bool, parse_date,
    parse_datetime, parse_number, parse_payment, parse_qty, parse_status, parse_time,
)

STORE_VERSION = "s2"
CACHE_DIR = Path(__file__).resolve().parents[3] / "cache" / "store"

SUMMARY_ROW = r"^(tong|cong|total|sub ?total|sum)( (ngay|thang|tuan|nam|quy|ky|cong|so|tien|doanh thu)\b|\s*$|\s*[:\d])"

ORDER_LEVEL = ["order_date", "order_datetime", "order_time", "status", "customer_id", "customer_name", "note",
               "channel", "payment_method", "staff_id"]


def name_key(s) -> str | None:
    """Exact-ish product/customer name key: NFC, casefold, single spaces (rulebook §8: same name = same thing)."""
    t = clean_text(s)
    return unicodedata.normalize("NFC", t).casefold() if t else None


def _umap(s, fn):
    """Apply fn once per distinct value (dates/money repeat a lot -> big speedup)."""
    if s is None:
        return None
    try:
        d = {v: fn(v) for v in pd.unique(s) if v == v}
        nan_val = fn(None)
        return s.map(lambda v: d[v] if v == v else nan_val)
    except TypeError:  # unhashable cells
        return s.map(fn)


def _num(t: Table, role: str) -> pd.Series | None:
    s = t.col(role)
    return None if s is None else _umap(s, parse_number).astype(float)


def _txt(t: Table, role: str) -> pd.Series | None:
    s = t.col(role)
    return None if s is None else _umap(s, clean_text)


def _dates_with_year(s, years):
    if years is None:
        return _umap(s, parse_date)
    key = pd.Series([(v if v == v else None, None if y is None or y != y else int(y)) for v, y in zip(s, years)],
                    index=s.index)
    return _umap(key, lambda kv: parse_date(kv[0], kv[1]) if kv else None)


def _dates(t: Table, role: str) -> pd.Series | None:
    s = t.col(role)
    if s is None:
        return None
    return _dates_with_year(s, t.df["_sheet_year"] if "_sheet_year" in t.df else None)


def _dts(t: Table, role: str) -> pd.Series | None:
    s = t.col(role)
    return None if s is None else _umap(s, parse_datetime)


def _blank_to_none(s):
    return s.where(~_umap(s, is_blank).astype(bool), None)


@dataclass
class Store:
    business_id: str
    spec: ReadmeSpec
    orders: pd.DataFrame = None
    lines: pd.DataFrame = None
    products: pd.DataFrame = None
    units: pd.DataFrame = None
    customers: pd.DataFrame = None
    payments: pd.DataFrame = None
    imports: pd.DataFrame = None
    import_lines: pd.DataFrame = None
    stock: pd.DataFrame = None
    price_history: pd.DataFrame = None
    lots: pd.DataFrame = None
    lot_sales: pd.DataFrame = None
    deliveries: pd.DataFrame = None
    debt_terms: pd.DataFrame = None
    expenses: pd.DataFrame = None
    extra: dict = field(default_factory=dict)       # kind -> list[Table] not mapped into canonical frames
    caps: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    table_kinds: list = field(default_factory=list)  # (name, kind, roles) for debugging

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, shop_dir: Path | str, use_cache: bool = True) -> "Store":
        shop_dir = Path(shop_dir)
        ck = _cache_path(shop_dir)
        if use_cache and ck and ck.exists():
            with open(ck, "rb") as f:
                return pickle.load(f)
        spec = parse_readme(shop_dir / "README.md")
        tables = [build_table(t) for t in read_shop(shop_dir, spec)]
        st = cls(business_id=shop_dir.name, spec=spec)
        st.table_kinds = [(t.name, t.kind, dict(t.roles)) for t in tables]
        st._assemble(tables)
        if use_cache and ck:
            try:
                ck.parent.mkdir(parents=True, exist_ok=True)
                with open(ck, "wb") as f:
                    pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
            except OSError:
                pass
        return st

    def _assemble(self, tables: list[Table]):
        by: dict[str, list[Table]] = {}
        for t in tables:
            by.setdefault(t.kind, []).append(t)
        self.products = self._build_products(by)
        self.units = self._build_units(by)
        self.lines = self._build_lines(by.get("lines", []))
        self.orders = self._build_orders(by.get("orders", []))
        self.customers = self._build_customers(by.get("customers", []))
        self.payments = self._build_payments(by.get("payments", []))
        self.imports, self.import_lines = self._build_imports(by.get("imports", []), by.get("import_lines", []))
        self.stock = self._build_stock(by.get("stock", []))
        self.price_history = self._build_price_history(by.get("price_history", []))
        self.lots = self._build_lots(by.get("lots", []))
        self.lot_sales = self._build_lot_sales(by.get("lot_sales", []))
        self.deliveries = self._build_deliveries(by.get("deliveries", []))
        self.debt_terms = self._build_debt_terms(by.get("debt_terms", []))
        self.expenses = self._build_expenses(by.get("expenses", []))
        for k in ("returns", "stock_ledger", "staff", "suppliers", "branches", "unknown"):
            if k in by:
                self.extra[k] = by[k]
        self._finalize()

    # ------------------------------------------------------------------ products & units
    def _build_products(self, by) -> pd.DataFrame:
        rows = []
        for t in by.get("products", []):
            df = pd.DataFrame({
                "product_id": _txt(t, "product_id"), "product_name": _txt(t, "product_name"),
                "category": _txt(t, "category"), "base_unit": _txt(t, "base_unit"),
                "cost": _num(t, "cost"), "price_current": _num(t, "price_current"),
            }, index=t.df.index)
            rows.append(df)
        prod = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["product_id", "product_name", "category", "base_unit", "cost", "price_current"])
        prod = prod.dropna(subset=["product_id", "product_name"], how="all")
        prod = prod.assign(_k=prod["product_id"].fillna(prod["product_name"]))
        prod = prod.groupby("_k", sort=False).first().reset_index(drop=True)
        # drugs info
        for t in by.get("drugs", []):
            d = pd.DataFrame({"product_id": _txt(t, "product_id"), "product_name": _txt(t, "product_name"),
                              "active_ingredient": _txt(t, "active_ingredient"),
                              "is_rx": t.col("is_rx").map(parse_bool) if t.col("is_rx") is not None else None})
            key = "product_id" if d["product_id"].notna().any() else "product_name"
            prod = prod.merge(d[[key, "active_ingredient", "is_rx"]].drop_duplicates(key), on=key, how="left")
        return prod.reset_index(drop=True)

    def _build_units(self, by) -> pd.DataFrame:
        rows = []
        for t in by.get("units", []):
            rows.append(pd.DataFrame({
                "product_id": _txt(t, "product_id"), "product_name": _txt(t, "product_name"),
                "unit": _txt(t, "big_unit"), "factor": _num(t, "factor"), "base_unit": _txt(t, "base_unit"),
            }))
        for prod, big, fac, base in self.spec.unit_conversions:
            rows.append(pd.DataFrame([{"product_id": None, "product_name": prod, "unit": big, "factor": fac,
                                       "base_unit": base}]))
        u = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["product_id", "product_name", "unit", "factor", "base_unit"])
        return u.dropna(subset=["unit", "factor"])

    # ------------------------------------------------------------------ lines
    def _build_lines(self, tables: list[Table]) -> pd.DataFrame:
        frames = []
        for ti, t in enumerate(tables):
            df = t.df
            raw_cols = [c for c in df.columns if not c.startswith("_")]
            oid = _txt(t, "order_id")
            has_oid = oid is not None and oid.notna().any()
            if has_oid and oid.isna().any():  # continuation rows of an order group (rulebook §8)
                oid = oid.ffill()
            out = pd.DataFrame(index=df.index)
            out["order_id"] = oid if has_oid else [f"_R{ti}_{i}" for i in range(len(df))]
            out["_has_oid"] = has_oid
            for r in ORDER_LEVEL:
                c = t.roles.get(r)
                out["_raw_" + r] = df[c] if c else None
            if has_oid:  # forward-fill order-level cells within each order group (rulebook §8)
                cols = [c for c in ("_raw_" + r for r in ORDER_LEVEL) if out[c].notna().any()]
                for c in cols:
                    out[c] = _blank_to_none(out[c].astype(object))
                if cols:
                    out[cols] = out.groupby("order_id", sort=False)[cols].ffill()
                    out[cols] = out[cols].fillna(out.groupby("order_id", sort=False)[cols].bfill())
                # exact duplicate rows within ONE order: export error, count once (rulebook §8).
                # Signature uses the forward-filled order id: continuation rows of different orders may look alike.
                oc = t.roles.get("order_id")
                sig = df[[c for c in raw_cols if c != oc]].astype(str)
                sig.insert(0, "_oid", out["order_id"].astype(str).values)
                dup = sig.duplicated(keep="first")
                out = out[~dup.values]
                df = df[~dup.values]
            years = df["_sheet_year"] if "_sheet_year" in df else pd.Series([None] * len(df), index=df.index)
            out["order_date"] = _dates_with_year(out["_raw_order_date"], years if "_sheet_year" in df else None)
            out["order_dt"] = _umap(out["_raw_order_datetime"], parse_datetime)
            lt = _umap(out["_raw_order_time"], parse_time)
            out["line_time"] = lt.map(lambda t: t[0] * 3600 + t[1] * 60 + t[2] if t else np.nan).astype(float)
            out["line_no"] = _num(t, "line_no").loc[out.index] if t.col("line_no") is not None else np.nan
            out["product_id"] = _txt(t, "product_id").loc[out.index] if t.col("product_id") is not None else None
            out["product_name"] = _txt(t, "product_name").loc[out.index] if t.col("product_name") is not None else None
            qtys = _umap(df[t.roles["qty"]], parse_qty) if "qty" in t.roles else pd.Series([(None, None)] * len(df), index=df.index)
            out["qty"] = [q for q, _ in qtys]
            unit_cell = [u for _, u in qtys]
            out["unit"] = _txt(t, "unit").loc[out.index] if t.col("unit") is not None else unit_cell
            out["unit"] = out["unit"].where(out["unit"].notna(), pd.Series(unit_cell, index=out.index))
            out["qty_base"] = _num(t, "qty_base").loc[out.index] if t.col("qty_base") is not None else np.nan
            out["unit_price"] = _num(t, "unit_price").loc[out.index] if t.col("unit_price") is not None else np.nan
            out["line_discount"] = _num(t, "line_discount").loc[out.index] if t.col("line_discount") is not None else 0.0
            out["amount"] = _num(t, "amount").loc[out.index] if t.col("amount") is not None else np.nan
            out["cost_unit"] = _num(t, "cost_unit").loc[out.index] if t.col("cost_unit") is not None else np.nan
            out["price_list"] = _num(t, "price_current").loc[out.index] if t.col("price_current") is not None else np.nan
            out["_src"] = t.name
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["order_id", "product_key", "qty", "qty_base", "amount"])
        L = pd.concat(frames, ignore_index=True)
        # drop subtotal rows inside hand-kept books ('TỔNG NGÀY 01/02', 'Tổng tháng 1/2026') and empty rows
        pn = L["product_name"].map(lambda x: fold(x) if isinstance(x, str) else "")
        summary = pn.str.match(SUMMARY_ROW)
        empty = L["product_name"].isna() & L["product_id"].isna()
        if summary.any():
            self.warnings.append(f"dropped {int(summary.sum())} subtotal rows")
        L = L[~summary & ~empty].reset_index(drop=True)
        L["qty"] = L["qty"].astype(float)
        L["line_discount"] = L["line_discount"].fillna(0.0)
        miss_amt = L["amount"].isna() & L["qty"].notna() & L["unit_price"].notna()
        L.loc[miss_amt, "amount"] = L.loc[miss_amt, "qty"] * L.loc[miss_amt, "unit_price"] - L.loc[miss_amt, "line_discount"]
        miss_price = L["unit_price"].isna() & L["amount"].notna() & (L["qty"].fillna(0) != 0)
        L.loc[miss_price, "unit_price"] = L.loc[miss_price, "amount"] / L.loc[miss_price, "qty"]
        if L["qty"].isna().all() and L["qty_base"].notna().any():
            L["qty"] = L["qty_base"]
        return L

    # ------------------------------------------------------------------ orders
    def _build_orders(self, tables: list[Table]) -> pd.DataFrame:
        frames = []
        for t in tables:
            o = pd.DataFrame({"order_id": _txt(t, "order_id")}, index=t.df.index)
            years = t.df["_sheet_year"] if "_sheet_year" in t.df else None
            o["order_date"] = _dates(t, "order_date") if t.col("order_date") is not None else None
            o["order_dt"] = _dts(t, "order_datetime") if t.col("order_datetime") is not None else None
            for r in ("status", "customer_id", "customer_name", "channel", "payment_method", "staff_id", "note"):
                o["_raw_" + r] = t.col(r)
            for r in ("order_subtotal", "order_discount", "order_total", "paid", "debt", "shipping_fee"):
                o[r] = _num(t, r)
            o = o.dropna(subset=["order_id"])
            frames.append(o)
            del years
        O = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["order_id"])
        O = O.drop_duplicates("order_id", keep="first") if len(O) else O
        return O

    # ------------------------------------------------------------------ other tables
    def _build_customers(self, tables):
        frames = [pd.DataFrame({
            "customer_id": _txt(t, "customer_id"), "customer_name": _txt(t, "customer_name"), "phone": _txt(t, "phone"),
            "customer_group": _txt(t, "customer_group"), "address": _txt(t, "address"), "gender": _txt(t, "gender"),
            "registered_at": t.col("registered_at").map(parse_date) if t.col("registered_at") is not None else None,
        }) for t in tables]
        return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame(
            columns=["customer_id", "customer_name", "phone", "customer_group", "address", "gender", "registered_at"])

    def _build_payments(self, tables):
        frames = []
        for t in tables:
            p = pd.DataFrame({"order_id": _txt(t, "order_id"), "payment_id": _txt(t, "payment_id"),
                              "amount": _num(t, "pay_amount"),
                              "method": t.col("payment_method").map(parse_payment) if t.col("payment_method") is not None else None})
            p["paid_at"] = _dts(t, "paid_at") if t.col("paid_at") is not None else _dts(t, "order_datetime")
            frames.append(p)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["order_id", "payment_id", "amount", "method", "paid_at"])

    def _build_imports(self, heads, lines):
        H = []
        for t in heads:
            h = pd.DataFrame({"receipt_id": _txt(t, "receipt_id"), "import_date": _dates(t, "import_date"),
                              "supplier_id": _txt(t, "supplier_id"), "supplier": _txt(t, "supplier"),
                              "import_status": _txt(t, "import_status"), "receipt_total": _num(t, "receipt_total"),
                              "import_note": _txt(t, "import_note")})
            H.append(h)
        IL = []
        for t in lines:
            q = t.col("import_qty")
            qq = q.map(parse_qty) if q is not None else pd.Series([(None, None)] * len(t.df), index=t.df.index)
            il = pd.DataFrame({
                "receipt_id": _txt(t, "receipt_id"), "product_id": _txt(t, "product_id"),
                "product_name": _txt(t, "product_name"),
                "unit": _txt(t, "import_unit") if t.col("import_unit") is not None else _txt(t, "unit"),
                "qty": [a for a, _ in qq], "qty_base": _num(t, "import_qty_base"),
                "price": _num(t, "import_price"), "amount": _num(t, "import_amount") if t.col("import_amount") is not None else _num(t, "amount"),
                "lot": _txt(t, "lot"), "expiry": _dates(t, "expiry"),
                "import_date": _dates(t, "import_date"), "supplier": _txt(t, "supplier"),
                "supplier_id": _txt(t, "supplier_id"), "import_status": _txt(t, "import_status"),
                "import_note": _txt(t, "import_note") if t.col("import_note") is not None else _txt(t, "note"),
            })
            if t.col("receipt_id") is not None and il["receipt_id"].isna().any():
                il["receipt_id"] = il["receipt_id"].ffill()
            IL.append(il)
        IL = pd.concat(IL, ignore_index=True) if IL else pd.DataFrame(columns=[
            "receipt_id", "product_id", "product_name", "unit", "qty", "qty_base", "price", "amount", "lot", "expiry",
            "import_date", "supplier", "supplier_id", "import_status", "import_note"])
        IL["qty"] = IL["qty"].astype(float)
        m = IL["amount"].isna() & IL["qty"].notna() & IL["price"].notna()
        IL.loc[m, "amount"] = IL.loc[m, "qty"] * IL.loc[m, "price"]
        Hd = pd.concat(H, ignore_index=True) if H else pd.DataFrame(columns=[
            "receipt_id", "import_date", "supplier_id", "supplier", "import_status", "receipt_total", "import_note"])
        if len(IL) and IL["receipt_id"].notna().any():
            agg = IL.groupby("receipt_id", sort=False).agg(
                _date=("import_date", "first"), _sup=("supplier", "first"), _supid=("supplier_id", "first"),
                _status=("import_status", "first"), _note=("import_note", "first"),
                _lines_total=("amount", lambda s: s.sum(min_count=1))).reset_index()
            Hd = agg.merge(Hd, on="receipt_id", how="outer") if len(Hd) else agg.assign(
                import_date=None, supplier=None, supplier_id=None, import_status=None, receipt_total=np.nan, import_note=None)
            for a, b in (("import_date", "_date"), ("supplier", "_sup"), ("supplier_id", "_supid"),
                         ("import_status", "_status"), ("import_note", "_note")):
                Hd[a] = Hd[a].where(Hd[a].notna(), Hd[b]) if a in Hd else Hd[b]
            Hd["receipt_total"] = Hd["receipt_total"].where(Hd["receipt_total"].notna(), Hd["_lines_total"])
            Hd = Hd.drop(columns=[c for c in Hd.columns if c.startswith("_")])
            dmap = Hd.set_index("receipt_id")["import_date"].to_dict()
            IL["import_date"] = IL["import_date"].where(IL["import_date"].notna(), IL["receipt_id"].map(dmap))
        Hd["cancelled"] = Hd["import_status"].map(lambda s: parse_status(s) == CANCELLED if s else False)
        return Hd.reset_index(drop=True), IL

    def _build_stock(self, tables):
        frames = [pd.DataFrame({"product_id": _txt(t, "product_id"), "product_name": _txt(t, "product_name"),
                                "unit": _txt(t, "base_unit"), "stock_qty": _num(t, "stock_qty")}) for t in tables]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["product_id", "product_name", "unit", "stock_qty"])

    def _build_price_history(self, tables):
        frames = [pd.DataFrame({"product_id": _txt(t, "product_id"), "product_name": _txt(t, "product_name"),
                                "effective_date": _dates(t, "effective_date"), "new_price": _num(t, "new_price"),
                                "old_price": _num(t, "old_price")}) for t in tables]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["product_id", "product_name", "effective_date", "new_price", "old_price"])

    def _build_lots(self, tables):
        frames = [pd.DataFrame({"receipt_id": _txt(t, "receipt_id"), "product_id": _txt(t, "product_id"),
                                "product_name": _txt(t, "product_name"), "lot": _txt(t, "lot"),
                                "expiry": _dates(t, "expiry"),
                                "qty_base": _num(t, "lot_qty_base") if t.col("lot_qty_base") is not None else _num(t, "import_qty_base")})
                  for t in tables]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["receipt_id", "product_id", "product_name", "lot", "expiry", "qty_base"])

    def _build_lot_sales(self, tables):
        frames = []
        for t in tables:
            q = _num(t, "lot_qty_base")
            if q is None:
                q = _num(t, "qty_base")
            frames.append(pd.DataFrame({"order_id": _txt(t, "order_id"), "line_no": _num(t, "line_no"),
                                        "lot": _txt(t, "lot"), "qty_base": q,
                                        "return_qty_base": _num(t, "lot_return_qty_base")}))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["order_id", "line_no", "lot", "qty_base", "return_qty_base"])

    def _build_deliveries(self, tables):
        frames = [pd.DataFrame({"order_id": _txt(t, "order_id"), "delivery_date": _dates(t, "delivery_date"),
                                "delivery_no": _num(t, "delivery_no"), "product_name": _txt(t, "product_name"),
                                "qty_base": _num(t, "delivery_qty_base") if t.col("delivery_qty_base") is not None else _num(t, "delivery_qty"),
                                "plate": _txt(t, "plate"), "driver": _txt(t, "driver")}) for t in tables]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["order_id", "delivery_date", "delivery_no", "product_name", "qty_base", "plate", "driver"])

    def _build_debt_terms(self, tables):
        frames = [pd.DataFrame({"order_id": _txt(t, "order_id"), "debt_days": _num(t, "debt_days"),
                                "due_date": _dates(t, "due_date")}) for t in tables]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["order_id", "debt_days", "due_date"])

    def _build_expenses(self, tables):
        frames = []
        for t in tables:
            amt = _num(t, "expense_amount") if t.col("expense_amount") is not None else _num(t, "pay_amount")
            dcol = next((r for r in ("expense_date", "date", "order_date") if t.col(r) is not None), None)
            frames.append(pd.DataFrame({"date": _dates(t, dcol) if dcol else None, "amount": amt,
                                        "category": _txt(t, "expense_category"), "note": _txt(t, "note")}))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "amount", "category", "note"])

    # ------------------------------------------------------------------ finalize: keys, units, statuses
    def _finalize(self):
        P, L, O = self.products, self.lines, self.orders
        # product keys: id when available, else name key; map names -> ids via catalog
        id_by_name = {}
        if len(P):
            for pid, pn in zip(P["product_id"], P["product_name"]):
                if pid and pn:
                    id_by_name.setdefault(name_key(pn), pid)
            fold_map = {}
            for pid, pn in zip(P["product_id"], P["product_name"]):
                if pid and pn:
                    fold_map.setdefault(fold(pn), set()).add(pid)
        else:
            fold_map = {}

        def key_of(pid, pn):
            if pid:
                return pid
            k = name_key(pn)
            if k in id_by_name:
                return id_by_name[k]
            f = fold(pn) if isinstance(pn, str) and pn else None
            if f in fold_map and len(fold_map[f]) == 1:
                return next(iter(fold_map[f]))
            # rulebook §8: names that differ only in case/spacing/diacritics denote the same item
            return f or k

        def keyify(df):
            if df is None or not len(df):
                if df is not None:
                    df["product_key"] = pd.Series(dtype=object)
                return df
            pid = df["product_id"] if "product_id" in df else pd.Series([None] * len(df), index=df.index)
            pn = df["product_name"] if "product_name" in df else pd.Series([None] * len(df), index=df.index)
            df["product_key"] = [key_of(a, b) for a, b in zip(pid, pn)]
            return df

        for name in ("products", "lines", "units", "import_lines", "stock", "price_history", "lots", "deliveries"):
            keyify(getattr(self, name))
        if len(P):
            P["product_name"] = P["product_name"].fillna(P["product_key"])
        # fill product names / ids on lines from catalog
        if len(L) and len(P):
            nm = P.set_index("product_key")["product_name"].to_dict()
            L["product_name"] = L["product_name"].where(L["product_name"].notna(), L["product_key"].map(nm))
            pidm = {k: k for k in P["product_id"].dropna()}
            L["product_id"] = L["product_key"].map(pidm)

        self._resolve_qty_base()
        self._derive_products_from_lines()
        self._resolve_orders()
        self._caps_and_checks()

    def _base_unit_of(self) -> dict:
        bu = {}
        for df in (self.products, self.units, self.stock):
            if df is None or not len(df):
                continue
            col = "base_unit" if "base_unit" in df else ("unit" if df is self.stock else None)
            if col and col in df:
                for k, u in zip(df["product_key"], df[col]):
                    if k and u and k not in bu:
                        bu[k] = u
        return bu

    def _resolve_qty_base(self):
        L = self.lines
        if not len(L):
            return
        base = self._base_unit_of()
        fac = {}
        for k, u, f in zip(self.units["product_key"], self.units["unit"], self.units["factor"]):
            if k and u and f:
                fac[(k, fold(u))] = float(f)
        # learn factors from lines that carry both qty and qty_base
        both = L[L["qty"].notna() & L["qty_base"].notna() & (L["qty"] != 0) & L["unit"].notna()]
        for (k, u), g in both.groupby([both["product_key"], both["unit"].map(fold)]):
            r = (g["qty_base"] / g["qty"]).round(6)
            if r.nunique() == 1 and (k, u) not in fac:
                fac[(k, u)] = float(r.iloc[0])
        self.factors = fac
        self.base_units = base
        need = L["qty_base"].isna() & L["qty"].notna()
        unknown = 0
        vals = []
        for k, u, q in zip(L.loc[need, "product_key"], L.loc[need, "unit"], L.loc[need, "qty"]):
            fu = fold(u) or None
            if fu is None or (k in base and fold(base[k]) == fu):
                vals.append(q)
            elif (k, fu) in fac:
                vals.append(q * fac[(k, fu)])
            else:
                # unit not in any conversion table: it is the base unit (rulebook §10)
                vals.append(q)
                if k in base and fold(base[k]) != fu:
                    unknown += 1
        L.loc[need, "qty_base"] = vals
        L["factor"] = np.where(L["qty"].fillna(0) != 0, L["qty_base"] / L["qty"].replace(0, np.nan), np.nan)
        if k_base := {k for k in L["product_key"].dropna().unique() if k not in base}:
            # base unit for products only seen in lines: the unit used with factor 1
            for k in k_base:
                g = L[(L["product_key"] == k) & (L["factor"].round(6) == 1)]
                if len(g) and g["unit"].notna().any():
                    base[k] = g["unit"].dropna().mode().iloc[0]
        if unknown:
            self.warnings.append(f"{unknown} line(s) with non-base unit and no conversion factor (treated as factor 1)")

    def _derive_products_from_lines(self):
        """Shops without a catalog: products = distinct items seen in lines (first spelling kept)."""
        L, P = self.lines, self.products
        if not len(L):
            return
        known = set(P["product_key"].dropna()) if len(P) else set()
        extra = L[~L["product_key"].isin(known) & L["product_key"].notna()]
        if not len(extra):
            return
        d = extra.groupby("product_key", sort=False).agg(product_name=("product_name", "first")).reset_index()
        d["product_id"] = None
        d["base_unit"] = d["product_key"].map(self.base_units)
        d["_from_lines"] = True
        self.products = pd.concat([P, d], ignore_index=True)

    def _resolve_orders(self):
        L, O = self.lines, self.orders
        if not len(L) and not len(O):
            return
        # line-derived order facts
        if len(L):
            g = L.groupby("order_id", sort=False)
            stp = _umap(L["_raw_status"].astype(object), lambda x: None if is_blank(x) else parse_status(x))
            lo = g[["order_date", "order_dt", "_raw_status", "_raw_note", "_raw_customer_id", "_raw_customer_name",
                    "_raw_channel", "_raw_payment_method", "_raw_staff_id"]].first()
            lo.columns = ["l_date", "l_dt", "l_status_raw", "l_note_raw", "l_customer_id", "l_customer_name",
                          "l_channel", "l_payment", "l_staff"]
            oid = L["order_id"]
            lo["l_any_neg"] = (L["qty"] < 0).groupby(oid, sort=False).any()
            lo["l_any_returned"] = (stp == RETURNED).groupby(oid, sort=False).any()
            lo["l_any_cancelled"] = (stp == CANCELLED).groupby(oid, sort=False).any()
            lo["l_subtotal"] = L["amount"].where(L["amount"] > 0).groupby(oid, sort=False).sum(min_count=1)
            lo["l_time"] = L["line_time"].where(L["qty"].fillna(1) > 0).groupby(oid, sort=False).min()
            lo = lo.reset_index()
            if O is None or not len(O) or "order_id" not in O:
                O = pd.DataFrame({"order_id": lo["order_id"]})
                for c in ("order_date", "order_dt", "_raw_status", "_raw_customer_id", "_raw_customer_name",
                          "_raw_channel", "_raw_payment_method", "_raw_staff_id", "_raw_note",
                          "order_subtotal", "order_discount", "order_total", "paid", "debt", "shipping_fee"):
                    O[c] = None
            O = O.merge(lo, on="order_id", how="outer")
        else:
            for c in ("l_date", "l_dt", "l_status_raw", "l_note_raw", "l_customer_id", "l_customer_name", "l_channel",
                      "l_payment", "l_staff", "l_time", "l_subtotal"):
                O[c] = None
            O["l_any_neg"] = False
            O["l_any_returned"] = False
            O["l_any_cancelled"] = False

        def pick(a, b):
            return O[a].where(O[a].notna(), O[b]) if a in O else O[b]

        O["order_date"] = pick("order_date", "l_date")
        O["order_dt"] = pick("order_dt", "l_dt")
        # date from datetime when only datetime known; datetime from date + line time
        miss_d = O["order_date"].isna() & O["order_dt"].notna()
        O.loc[miss_d, "order_date"] = O.loc[miss_d, "order_dt"].map(lambda d: d.date())
        lt = pd.to_numeric(O["l_time"], errors="coerce")
        miss_t = O["order_dt"].isna() & O["order_date"].notna() & lt.notna()
        O.loc[miss_t, "order_dt"] = [datetime(d.year, d.month, d.day) + pd.Timedelta(seconds=float(t)).to_pytimedelta()
                                     for d, t in zip(O.loc[miss_t, "order_date"], lt[miss_t])]
        for a, b in (("_raw_customer_id", "l_customer_id"), ("_raw_customer_name", "l_customer_name"),
                     ("_raw_channel", "l_channel"), ("_raw_payment_method", "l_payment"), ("_raw_staff_id", "l_staff"),
                     ("_raw_status", "l_status_raw"), ("_raw_note", "l_note_raw")):
            O[a] = pick(a, b) if a in O else O[b]
        O["customer_id"] = _umap(O["_raw_customer_id"].astype(object), clean_text)
        O["customer_name"] = _umap(O["_raw_customer_name"].astype(object), clean_text)
        O["is_walkin"] = O["customer_id"].isna() & _umap(O["customer_name"].astype(object), is_walkin).astype(bool)
        O["channel"] = O["_raw_channel"].map(lambda x: fold(str(x)) if not is_blank(x) else None)
        O["payment_method"] = _umap(O["_raw_payment_method"].astype(object), parse_payment)
        O["staff_id"] = _umap(O["_raw_staff_id"].astype(object), clean_text)
        O["note"] = _umap(O["_raw_note"].astype(object), clean_text)

        # status (rulebook §1/§8): cancelled > returned (any evidence) > shipping > done
        st_raw = _umap(O["_raw_status"].astype(object), parse_status)
        st_note = _umap(O["note"].astype(object), parse_status)
        has_status_col = O["_raw_status"].notna().any()
        unknown_raw = O["_raw_status"].notna() & st_raw.isna()
        if unknown_raw.any():
            vals = sorted({str(v) for v in O.loc[unknown_raw, "_raw_status"].unique()})[:10]
            self.warnings.append(f"unrecognized status values: {vals}")
        base_st = st_raw if has_status_col else st_note
        returned_ev = (base_st == RETURNED) | O["l_any_neg"].fillna(False).astype(bool) | O["l_any_returned"].fillna(False).astype(bool)
        if self.lot_sales is not None and len(self.lot_sales) and self.lot_sales["return_qty_base"].notna().any():
            ret_ids = set(self.lot_sales.loc[self.lot_sales["return_qty_base"].fillna(0) > 0, "order_id"])
            returned_ev |= O["order_id"].isin(ret_ids)
        for t in self.extra.get("returns", []):
            if t.col("order_id") is not None:
                returned_ev |= O["order_id"].isin(set(t.col("order_id").map(clean_text).dropna()))
        cancelled = (base_st == CANCELLED) | O["l_any_cancelled"].fillna(False).astype(bool)
        status = np.where(cancelled, CANCELLED, np.where(returned_ev, RETURNED,
                          np.where(base_st == SHIPPING, SHIPPING, DONE)))
        O["status"] = status
        # rulebook §1: revenue of an order = sum of line amounts; shops with only an invoice list use the
        # order-level 'tiền hàng' (sum of line amounts before order discount, README vocabulary)
        O["subtotal"] = pd.to_numeric(O["l_subtotal"], errors="coerce")
        if "order_subtotal" in O:
            O["subtotal"] = O["subtotal"].where(O["subtotal"].notna(), pd.to_numeric(O["order_subtotal"], errors="coerce"))
        keep = ["order_id", "order_date", "order_dt", "status", "customer_id", "customer_name", "is_walkin", "channel",
                "payment_method", "staff_id", "note", "subtotal", "order_subtotal", "order_discount", "order_total",
                "paid", "debt", "shipping_fee", "_raw_status"]
        for c in keep:
            if c not in O:
                O[c] = None
        self.orders = O[keep].reset_index(drop=True)
        if len(L):
            sm = self.orders.set_index("order_id")
            L["status"] = L["order_id"].map(sm["status"])
            L["order_date"] = L["order_id"].map(sm["order_date"])

    def _caps_and_checks(self):
        O, L = self.orders, self.lines
        nn = lambda df, c: df is not None and len(df) and c in df and df[c].notna().any()  # noqa: E731
        dates = [d for d in (O["order_date"] if O is not None and len(O) else []) if d]
        self.caps = {
            "has_order_id": bool(len(L) and L["_has_oid"].any()) if len(L) else bool(len(O)),
            "has_lines": bool(len(L)), "has_customers": bool(nn(O, "customer_id") or nn(O, "customer_name")),
            "has_customer_id": bool(nn(O, "customer_id")), "has_customer_table": bool(len(self.customers)),
            "has_customer_group": bool(nn(self.customers, "customer_group")),
            "has_address": bool(nn(self.customers, "address")), "has_gender": bool(nn(self.customers, "gender")),
            "has_channel": bool(nn(O, "channel")), "has_payment_method": bool(nn(O, "payment_method")),
            "has_staff": bool(nn(O, "staff_id")), "has_time": bool(nn(O, "order_dt")),
            "has_payments": bool(len(self.payments)), "has_debt": bool(nn(O, "debt") or len(self.payments)),
            "has_cost": bool(nn(L, "cost_unit")), "has_stock": bool(len(self.stock)),
            "has_imports": bool(len(self.imports) or len(self.import_lines)),
            "has_import_amount": bool(nn(self.import_lines, "amount") or nn(self.imports, "receipt_total")),
            "has_lots": bool(len(self.lots) or nn(self.import_lines, "lot")),
            "has_expiry": bool(nn(self.lots, "expiry") or nn(self.import_lines, "expiry")),
            "has_price_history": bool(len(self.price_history)), "has_deliveries": bool(len(self.deliveries)),
            "has_debt_terms": bool(len(self.debt_terms)), "has_expenses": bool(len(self.expenses)),
            "has_category": bool(nn(self.products, "category")),
            "has_drugs": bool(nn(self.products, "active_ingredient")) if self.products is not None and "active_ingredient" in self.products else False,
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "data_end": self.spec.data_end or (max(dates) if dates else None),
        }
        # validation: line subtotal vs order subtotal from the order table
        if O is not None and len(O) and O["order_subtotal"].notna().any():
            m = O[O["order_subtotal"].notna() & O["subtotal"].notna() & (O["status"] == DONE)]
            if len(m):
                bad = (m["subtotal"] - m["order_subtotal"]).abs() > 0.5
                self.caps["check_subtotal_mismatch"] = round(float(bad.mean()), 4)
                if bad.mean() > 0.01:
                    self.warnings.append(f"line-sum != order subtotal for {bad.mean():.1%} of done orders")
        if O is not None and len(O):
            nod = float(O["order_date"].isna().mean())
            self.caps["check_orders_without_date"] = round(nod, 4)
            if nod > 0:
                self.warnings.append(f"{nod:.2%} orders without date")
            self.caps["status_counts"] = O["status"].value_counts().to_dict()


def _cache_path(shop_dir: Path) -> Path | None:
    try:
        sig = []
        for p in sorted(shop_dir.iterdir()):
            st = p.stat()
            sig.append(f"{p.name}|{st.st_size}|{st.st_mtime_ns}")
        h = hashlib.md5(("|".join(sig) + STORE_VERSION + _code_sig()).encode()).hexdigest()[:16]
        return CACHE_DIR / f"{shop_dir.name}_{h}.pkl"
    except OSError:
        return None


def _code_sig() -> str:
    """Invalidate store cache when ingest code changes."""
    here = Path(__file__).parent
    return "".join(str(p.stat().st_mtime_ns) for p in sorted(here.glob("*.py"))) + \
        str((here / "learned_names.json").stat().st_mtime_ns)


__all__ = ["Store", "name_key", "date"]
