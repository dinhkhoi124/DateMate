"""Entity linking: mentions in the normalised question -> shop catalog keys (rulebook §8: names that denote the
same thing are the same thing; file ids win). Exact token-boundary match in the folded token space, longest first;
a mention that maps to two different keys is ambiguous and is reported as such (caller abstains).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from datamate.execute.facts import Facts
from datamate.normalize.text import norm_name


@dataclass
class Hit:
    start: int
    end: int
    key: object
    text: str
    ambiguous: bool = False


class Dictionary:
    def __init__(self, pairs):
        self.map: dict[str, set] = {}
        for name, key in pairs:
            if not isinstance(name, str) or key is None or key != key:
                continue
            n = norm_name(name)
            if len(n) < 2:
                continue
            self.map.setdefault(n, set()).add(key)
        self.names = sorted(self.map, key=len, reverse=True)
        self._rx = {n: re.compile(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])") for n in self.names}

    def find(self, text: str, limit: int = 8) -> list[Hit]:
        taken = [False] * len(text)
        hits = []
        for n in self.names:
            if n not in text:
                continue
            for m in self._rx[n].finditer(text):
                a, b = m.span()
                if any(taken[a:b]):
                    continue
                for i in range(a, b):
                    taken[i] = True
                keys = self.map[n]
                hits.append(Hit(a, b, sorted(keys, key=str)[0], n, len(keys) > 1))
                if len(hits) >= limit:
                    break
        return sorted(hits, key=lambda h: h.start)

    def tokens(self) -> set:
        return {t for n in self.names for t in re.findall(r"[a-z]+", n)}


class Linker:
    def __init__(self, f: Facts):
        st = f.st
        pairs = []
        for df in (f.P, f.L, st.stock, st.import_lines, st.price_history, st.lots, st.units):
            if df is not None and len(df) and "product_key" in df:
                if "product_name" in df:
                    pairs += list(zip(df["product_name"], df["product_key"]))
                if "product_id" in df:
                    pairs += [(i, k) for i, k in zip(df["product_id"], df["product_key"]) if isinstance(i, str)]
        self.products = Dictionary(dict.fromkeys(pairs))
        self.categories = Dictionary((c, c) for c in f.P["category"].dropna().unique()) if "category" in f.P else Dictionary([])
        ing = f.P["active_ingredient"].dropna().unique() if "active_ingredient" in f.P else []
        self.ingredients = Dictionary((c, c) for c in ing)
        O = f.O
        cp = [(n, k) for n, k in zip(O["customer_name"], O["ckey"]) if k is not None]
        C = st.customers
        if f.has_cid and C is not None and len(C):
            cp += list(zip(C["customer_name"], C["customer_id"]))
        self.customers = Dictionary(dict.fromkeys(cp))
        sp = [(s, s) for s in O["staff_id"].dropna().unique()] if "staff_id" in O else []
        self.staff = Dictionary(sp)
        sup = []
        for df in (st.imports, st.import_lines):
            if df is not None and len(df):
                for c in ("supplier", "supplier_id"):
                    if c in df:
                        sup += [(s, s) for s in df[c].dropna().unique()]
        self.suppliers = Dictionary(sup)
        self.protect = frozenset(self.products.tokens() | self.customers.tokens() | self.categories.tokens())


def linker_of(f: Facts) -> Linker:
    lk = f.st.extra.get("_linker")
    if lk is None:
        lk = Linker(f)
        f.st.extra["_linker"] = lk
    return lk
