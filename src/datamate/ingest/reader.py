"""Read every data file of a shop into raw tables with detected headers.

- csv: header = first row; xlsx: header row detected among first rows (title rows, 2-row headers);
  sqlite: one table per sqlite table.
- Monthly sheets with identical headers are concatenated (column `_sheet_year`/`_sheet_month` kept for
  'ngày không ghi năm' rule). Files split as name_1/_2/_2025... are concatenated.
"""
from __future__ import annotations

import hashlib
import pickle
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from datamate.config import cache_root
from datamate.ingest.readme import SHEET_SEP, ReadmeSpec
from datamate.ingest.vocab import NAME_SYNONYMS, squash

def _cache_dir() -> Path:
    return cache_root() / "raw"


@dataclass
class RawTable:
    name: str                      # logical name, e.g. 'DanhSachHoaDon' or 'SoBanHang'
    df: pd.DataFrame               # header applied, all columns as read (object dtype)
    sources: list[str] = field(default_factory=list)
    doc: dict[str, str] = field(default_factory=dict)  # column -> README description (if documented)


_SPLIT_SUFFIX = re.compile(r"(_(\d{1,2}|\d{4}|T\d{1,2}_\d{4}|Thang\d{1,2}_\d{4}))+$", re.I)
_SHEET_PERIOD = [
    re.compile(r"^\s*T(?:hang|háng)?\s*[_\- ]?(\d{1,2})\s*[_\-./ ]\s*(\d{4})\s*$", re.I),
    re.compile(r"^\s*(\d{1,2})\s*[_\-./ ]\s*(\d{4})\s*$"),
    re.compile(r"^\s*Th[aá]ng\s*(\d{1,2})\s*[_\-./ ]?\s*(\d{4})\s*$", re.I),
    re.compile(r"^\s*(\d{4})\s*[_\-./ ]\s*(\d{1,2})\s*$"),
]


def sheet_period(name: str) -> tuple[int, int] | None:
    for i, rx in enumerate(_SHEET_PERIOD):
        m = rx.match(name)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            y, mth = (a, b) if i == 3 else (b, a)
            if 1 <= mth <= 12 and 2000 <= y <= 2100:
                return y, mth
    return None


def base_name(stem: str) -> str:
    return _SPLIT_SUFFIX.sub("", stem) or stem


def _known_names(spec: ReadmeSpec) -> set[str]:
    return set(NAME_SYNONYMS) | {squash(c.name) for c in spec.all_columns()}


def _detect_header(raw: pd.DataFrame, expected: set[str], known: set[str], max_rows: int = 15) -> int:
    best, best_score = 0, -1.0
    for i in range(min(max_rows, len(raw))):
        cells = [squash(str(x)) for x in raw.iloc[i].tolist() if pd.notna(x) and str(x).strip()]
        if not cells:
            continue
        score = sum(c in expected for c in cells) * 2 + sum(c in known for c in cells) + 0.01 * len(cells)
        if score > best_score:
            best, best_score = i, score
    return best


def _apply_header(raw: pd.DataFrame, h: int) -> pd.DataFrame:
    cols, seen = [], {}
    for j, x in enumerate(raw.iloc[h].tolist()):
        c = str(x).strip() if pd.notna(x) and str(x).strip() else f"_col{j}"
        if c in seen:
            seen[c] += 1
            c = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    df = raw.iloc[h + 1:].copy()
    df.columns = cols
    df = df.dropna(how="all")
    # drop repeated header rows (e.g. header repeated mid-sheet)
    if len(df):
        first = cols[0]
        df = df[df[first].astype(str).str.strip() != first]
    return df.reset_index(drop=True)


def _doc_for(spec: ReadmeSpec, *keys: str) -> dict[str, str]:
    for k in keys:
        cols = spec.columns_for(k)
        if cols:
            return {c.name: c.desc for c in cols}
    return {}


def _cache_key(path: Path) -> Path:
    st = path.stat()
    h = hashlib.md5(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|v4".encode()).hexdigest()[:16]
    return _cache_dir() / f"{path.stem}_{h}.pkl"


def _read_excel_all(path: Path) -> dict[str, pd.DataFrame]:
    ck = _cache_key(path)
    if ck.exists():
        with open(ck, "rb") as f:
            return pickle.load(f)
    try:  # calamine (Rust) ~8x faster than openpyxl, same values (D08)
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine="calamine")
    except Exception:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine="openpyxl")
    try:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        with open(ck, "wb") as f:
            pickle.dump(sheets, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:  # read-only FS in Docker: just skip cache
        pass
    return sheets


def fix_mojibake(name: str) -> str:
    """Zip-extracted names stored as CP437: 'Danh s├ích' -> 'Danh sách'."""
    try:
        fixed = name.encode("cp437").decode("utf8")
        return fixed if fixed != name else name
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def read_shop(shop_dir: Path, spec: ReadmeSpec) -> list[RawTable]:
    known = _known_names(spec)
    pieces: list[RawTable] = []
    files = sorted(p for p in shop_dir.iterdir() if p.is_file() and p.name != "README.md")
    for f in files:
        suf = f.suffix.lower()
        fname = fix_mojibake(f.name)
        stem = Path(fname).stem
        if suf == ".csv":
            df = pd.read_csv(f, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
            df = df.dropna(how="all").reset_index(drop=True)
            pieces.append(RawTable(base_name(stem), df, [fname], _doc_for(spec, fname)))
        elif suf in (".xlsx", ".xlsm", ".xls"):
            sheets = _read_excel_all(f)
            file_doc = _doc_for(spec, fname)
            for sname, raw in sheets.items():
                if raw.dropna(how="all").empty:
                    continue
                per = sheet_period(sname)
                doc = _doc_for(spec, f"{fname}{SHEET_SEP}{sname}") or (file_doc if (per or len(sheets) == 1) else (_doc_for(spec, sname) or file_doc))
                expected = {squash(c) for c in doc}
                h = _detect_header(raw, expected, known)
                df = _apply_header(raw, h)
                if per:
                    df["_sheet_year"], df["_sheet_month"] = per
                    name = base_name(stem)
                elif len(sheets) == 1:
                    name = base_name(stem)
                else:
                    name = sname
                pieces.append(RawTable(name, df, [f"{fname}:{sname}"], doc))
        elif suf in (".sqlite", ".db", ".sqlite3"):
            con = sqlite3.connect(f"file:{f.resolve().as_posix()}?mode=ro", uri=True)
            try:
                for (t,) in con.execute("select name from sqlite_master where type='table' order by name"):
                    df = pd.read_sql_query(f'select * from "{t}"', con)
                    pieces.append(RawTable(t, df, [f"{fname}:{t}"], _doc_for(spec, t)))
            finally:
                con.close()
    return _merge(pieces)


def _merge(pieces: list[RawTable]) -> list[RawTable]:
    """Concatenate pieces sharing a logical name and (mostly) the same columns."""
    out: dict[str, RawTable] = {}
    for p in pieces:
        key = p.name
        if key in out:
            cur = out[key]
            a = {c for c in cur.df.columns if not c.startswith("_")}
            b = {c for c in p.df.columns if not c.startswith("_")}
            if len(a & b) >= 0.6 * max(len(a), len(b)):
                cur.df = pd.concat([cur.df, p.df], ignore_index=True)
                cur.sources += p.sources
                cur.doc = cur.doc or p.doc
                continue
            key = f"{p.name}#{len(out)}"
        out[key] = RawTable(key, p.df, p.sources, p.doc)
    return list(out.values())
