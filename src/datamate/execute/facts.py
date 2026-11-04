"""Per-shop analytic views over the canonical Store, shared by all ops.

Facts(store) adds: numpy day stamps, revenue-valid flags (rulebook §1), customer keys (rulebook §3),
service-fee flags (rulebook §3), product attributes on lines. Built once per shop and cached on the Store.
"""
from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import pandas as pd

from datamate.config import DEFAULT, Rules
from datamate.ingest.store import Store
from datamate.ingest.values import CANCELLED, DONE, RETURNED, SHIPPING, fold


def day64(d) -> np.datetime64:
    return np.datetime64(d, "D")


def _to_day(s: pd.Series) -> np.ndarray:
    def conv(v):
        if v is None or v != v:
            return np.datetime64("NaT")
        if isinstance(v, datetime):
            v = v.date()
        return np.datetime64(v, "D")
    return np.array([conv(v) for v in s], dtype="datetime64[D]")


def cust_name_key(name) -> str | None:
    """rulebook §3: shop without customer ids -> same customer iff names differ only in spelling (case/space/accents)."""
    if not isinstance(name, str) or not name.strip():
        return None
    return re.sub(r"[^a-z0-9]+", "", fold(name)) or None


class Facts:
    def __init__(self, st: Store, rules: Rules = DEFAULT):
        self.st, self.rules = st, rules
        self.caps = st.caps
        O = st.orders.copy()
        O["d"] = _to_day(O["order_date"])
        O["valid"] = O["status"].isin([DONE, SHIPPING])      # rulebook §1: not cancelled, not returned
        O["cancelled"] = O["status"] == CANCELLED
        O["returned"] = O["status"] == RETURNED
        O["rev"] = pd.to_numeric(O["subtotal"], errors="coerce").fillna(0.0)
        O["tod"] = [(v.hour * 3600 + v.minute * 60 + v.second) if isinstance(v, datetime) else np.nan for v in O["order_dt"]]
        self.has_cid = bool(O["customer_id"].notna().any())
        if self.has_cid:   # rulebook §3: each customer id is one customer; orders without id are walk-in
            O["ckey"] = O["customer_id"].where(O["customer_id"].notna(), None)
        else:
            O["ckey"] = [None if w else cust_name_key(n) for n, w in zip(O["customer_name"], O["is_walkin"])]
        O["pay"] = O["payment_method"]
        self.O = O.reset_index(drop=True)

        P = st.products.copy()
        for c in ("category", "base_unit", "cost", "price_current", "is_rx", "active_ingredient", "product_id"):
            if c not in P:
                P[c] = None
        P["is_fee"] = P["product_name"].map(lambda n: fold(n) in rules.service_fee_names if isinstance(n, str) else False)
        self.P = P.drop_duplicates("product_key").set_index("product_key", drop=False)

        L = st.lines.copy()
        if len(L):
            om = self.O.set_index("order_id")
            L["d"] = _to_day(L["order_id"].map(om["order_date"]))
            L["valid"] = L["order_id"].map(om["valid"]).fillna(False).astype(bool)
            L["status"] = L["order_id"].map(om["status"])
            L["tod"] = L["order_id"].map(om["tod"])
            L["amount"] = pd.to_numeric(L["amount"], errors="coerce").fillna(0.0)
            L["qty_base"] = pd.to_numeric(L["qty_base"], errors="coerce").fillna(0.0)
            pf = self.P["is_fee"].to_dict()
            L["is_fee"] = L["product_key"].map(pf).fillna(False).astype(bool) | \
                L["product_name"].map(lambda n: fold(n) in rules.service_fee_names if isinstance(n, str) else False)
            L["category"] = L["product_key"].map(self.P["category"].to_dict())
            L["base_unit"] = L["product_key"].map(getattr(st, "base_units", {}) or {})
            bu2 = self.P["base_unit"].to_dict()
            L["base_unit"] = L["base_unit"].where(L["base_unit"].notna(), L["product_key"].map(bu2))
        self.L = L.reset_index(drop=True)
        self.has_lines = bool(len(L))
        d = self.O["d"][~np.isnat(self.O["d"].values.astype("datetime64[D]"))] if len(self.O) else []
        self.dmin = st.caps.get("date_min")
        self.dend = st.caps.get("data_end") or st.caps.get("date_max")
        del d

    # ------------------------------------------------------------------ masks
    def omask(self, start: date, end: date, valid_only=True, df=None):
        df = self.O if df is None else df
        dd = df["d"].values.astype("datetime64[D]")
        m = (dd >= day64(start)) & (dd <= day64(end))
        if valid_only:
            m &= df["valid"].values
        return m

    def lmask(self, start: date, end: date, valid_only=True):
        return self.omask(start, end, valid_only, self.L)

    # ------------------------------------------------------------------ catalog helpers
    def pname(self, key) -> str:
        if key in self.P.index:
            n = self.P.at[key, "product_name"]
            if isinstance(n, str):
                return n
        sub = self.L.loc[self.L["product_key"] == key, "product_name"].dropna()
        return sub.iloc[0] if len(sub) else str(key)

    def sort_key_product(self, key):
        """rulebook §9: ties by code ascending; no code -> by name."""
        pid = self.P.at[key, "product_id"] if key in self.P.index else None
        return (0, str(pid)) if isinstance(pid, str) and pid else (1, fold(self.pname(key)))

    def customer_display(self, ckey) -> str:
        O = self.O
        if self.has_cid:
            C = self.st.customers
            if C is not None and len(C):
                hit = C.loc[C["customer_id"] == ckey, "customer_name"].dropna()
                if len(hit):
                    return hit.iloc[0]
        names = O.loc[O["ckey"] == ckey, "customer_name"].dropna()
        return names.mode().iloc[0] if len(names) else str(ckey)

    def customer_groups(self) -> dict:
        C = self.st.customers
        if C is None or not len(C) or "customer_group" not in C:
            return {}
        return {k: g for k, g in zip(C["customer_id"], C["customer_group"]) if isinstance(k, str) and isinstance(g, str)}


def facts_of(st: Store, rules: Rules = DEFAULT) -> Facts:
    f = st.extra.get("_facts")
    if f is None or f.rules != rules:
        f = Facts(st, rules)
        st.extra["_facts"] = f
    return f
