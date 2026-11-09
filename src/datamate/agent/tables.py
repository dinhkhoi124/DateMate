"""Shop tables `T` for model-written code and their schema text for the prompt (D20, ARCHITECTURE §8).

T exposes the ingested Store frames (column roles already mapped from the README, D05/D09) without internal
bookkeeping columns. `raw_status` keeps the status text exactly as written in the files, so a split rulebook that
defines statuses differently from train can still be applied by the model.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

FRAMES = ("orders", "lines", "products", "units", "customers", "payments", "imports", "import_lines", "stock",
          "price_history", "lots", "lot_sales", "deliveries", "debt_terms", "expenses")
_KEEP_PRIVATE = {"_raw_status": "raw_status"}
MAX_ENUM = 15


def frames_of(st) -> dict[str, pd.DataFrame]:
    T = {}
    for name in FRAMES:
        df = getattr(st, name, None)
        if df is None or not len(df):
            continue
        cols = [c for c in df.columns if not str(c).startswith("_") or c in _KEEP_PRIVATE]
        T[name] = df[cols].rename(columns=_KEEP_PRIVATE).reset_index(drop=True)
    return T


def _col_doc(s: pd.Series) -> str:
    kind = str(s.dtype)
    nn = s.dropna()
    if not len(nn):
        return f"{kind}, empty"
    if s.dtype == object or str(s.dtype).startswith(("string", "category", "bool")):
        try:
            vals = nn.astype(str).unique()
        except Exception:  # noqa: BLE001
            vals = []
        if 0 < len(vals) <= MAX_ENUM:
            return f"{kind}, values: " + " | ".join(sorted(vals)[:MAX_ENUM])
        return f"{kind}, e.g. " + " | ".join(str(v)[:40] for v in nn.head(3))
    if pd.api.types.is_datetime64_any_dtype(s) or isinstance(nn.iloc[0], date):
        return f"{kind}, {nn.min()} .. {nn.max()}"
    if pd.api.types.is_numeric_dtype(s):
        return f"{kind}, min {nn.min()} max {nn.max()}"
    return f"{kind}, e.g. {nn.iloc[0]!s:.40}"


def schema_text(T: dict[str, pd.DataFrame]) -> str:
    """Compact schema: table, row count, one line per column with dtype and values/examples/range."""
    out = []
    for name, df in T.items():
        out.append(f"T['{name}']  ({len(df)} dòng)")
        for c in df.columns:
            out.append(f"  - {c}: {_col_doc(df[c])}")
    return "\n".join(out)


def shop_tables(root: Path, biz: str, rulebook: Path | None = None):
    """(T, make_h): tables and a factory H(as_of, normalized question) for one shop (used inside the sandbox)."""
    from datamate.agent.helpers import Helpers  # noqa: PLC0415
    from datamate.execute.facts import facts_of  # noqa: PLC0415
    from datamate.ingest.store import Store  # noqa: PLC0415
    from datamate.link.catalog import linker_of  # noqa: PLC0415
    from datamate.rulebook import analyse  # noqa: PLC0415
    from datamate.timeparse.periods import LunarCalendar  # noqa: PLC0415

    st = Store.build(root / "data" / biz)
    rb = rulebook or root / "rulebook.md"
    rules = analyse(rb.read_text(encoding="utf8") if rb.exists() else None).rules
    cal = LunarCalendar.load(str(root / "calendar.csv"))
    cal_df = pd.read_csv(root / "calendar.csv", encoding="utf8") if (root / "calendar.csv").exists() else None
    lk = None
    try:
        if st.orders is not None and len(st.orders):
            lk = linker_of(facts_of(st, rules))
    except Exception:  # noqa: BLE001  (no catalog lookup is still usable)
        lk = None
    T = frames_of(st)

    def make_h(as_of: date, question_norm: str):
        return Helpers(as_of, question_norm, cal, rules, lk, cal_df)
    return T, make_h
