"""Synthetic 'unseen' shops from train-dev shops (PLAN 04-06/11 step 1).

Every transform preserves meaning, so the dev gold answers stay valid; only the way the data is exported changes.
Transforms mimic what other shops do: odd column names, no column docs, title rows / column order / csv<->xlsx,
dates and money written as text, status kept in the notes column.

Output: eval/synth/out/<transform>/{rulebook.md, calendar.csv, data/<biz>/...}. data/ is only read (CLAUDE.md rule 7).
"""
from __future__ import annotations

import random
import re
import shutil
import sqlite3  # noqa: F401  (sqlite shops are skipped: none in dev)
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from datamate.ingest.reader import _detect_header, _doc_for, _known_names, _read_excel_all, fix_mojibake, sheet_period
from datamate.ingest import readme as _readme
from datamate.ingest.readme import parse_readme

SHEET_SEP = getattr(_readme, "SHEET_SEP", "::")
from datamate.ingest.values import fold
from datamate.ingest.vocab import squash

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data" / "train" / "train"
OUT = ROOT / "eval" / "synth" / "out"


@dataclass
class Sheet:
    file: str                 # output file name
    sheet: str | None         # xlsx sheet name (None for csv)
    top: list                 # rows above the header (titles), xlsx only
    header: list
    rows: list                # list of lists
    doc: dict = field(default_factory=dict)   # column -> README description


@dataclass
class Shop:
    biz: str
    readme: str
    sheets: list[Sheet]

    def files(self) -> dict[str, list[Sheet]]:
        out: dict[str, list[Sheet]] = {}
        for s in self.sheets:
            out.setdefault(s.file, []).append(s)
        return out


# ---------------------------------------------------------------------- load / save
def load(shop_dir: Path) -> Shop | None:
    spec = parse_readme(shop_dir / "README.md")
    known = _known_names(spec)
    sheets = []
    for f in sorted(p for p in shop_dir.iterdir() if p.is_file() and p.name != "README.md"):
        name, suf = fix_mojibake(f.name), f.suffix.lower()
        if suf == ".csv":
            df = pd.read_csv(f, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
            sheets.append(Sheet(name, None, [], list(df.columns), df.values.tolist(), _doc_for(spec, name)))
        elif suf in (".xlsx", ".xlsm", ".xls"):
            all_ = _read_excel_all(f)
            fdoc = _doc_for(spec, name)
            for sname, raw in all_.items():
                if raw.dropna(how="all").empty:
                    continue
                doc = _doc_for(spec, f"{name}{SHEET_SEP}{sname}") or (fdoc if (sheet_period(sname) or len(all_) == 1) else (_doc_for(spec, sname) or fdoc))
                h = _detect_header(raw, {squash(c) for c in doc}, known)
                g = raw.values.tolist()
                sheets.append(Sheet(Path(name).with_suffix(".xlsx").name, sname, g[:h], g[h], g[h + 1:], doc))
        else:
            return None   # sqlite / unknown formats: not generated
    return Shop(shop_dir.name, (shop_dir / "README.md").read_text(encoding="utf8"), sheets)


def _cell(v):
    return None if v is None or (isinstance(v, float) and v != v) else v


def save(shop: Shop, root: Path):
    d = root / "data" / shop.biz
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "README.md").write_text(shop.readme, encoding="utf8")
    for fname, sheets in shop.files().items():
        if fname.lower().endswith(".csv"):
            s = sheets[0]
            pd.DataFrame([[_cell(v) for v in r] for r in s.rows], columns=[str(h) for h in s.header]) \
                .to_csv(d / fname, index=False, encoding="utf8")
        else:
            with pd.ExcelWriter(d / fname, engine="openpyxl") as w:
                for s in sheets:
                    width = max(len(s.header), *(len(r) for r in s.top or [[]]))
                    pad = lambda r: [_cell(v) for v in r] + [None] * (width - len(r))  # noqa: E731
                    grid = [pad(r) for r in s.top] + [pad(s.header)] + [pad(r) for r in s.rows]
                    pd.DataFrame(grid).to_excel(w, sheet_name=(s.sheet or "Sheet1")[:31], header=False, index=False)
    for fn in ("rulebook.md", "calendar.csv"):
        if not (root / fn).exists():
            shutil.copy(TRAIN / fn, root / fn)


# ---------------------------------------------------------------------- README helpers
_BULLET = re.compile(r"^(-\s+\*\*)(.+?)(\*\*\s*:)", re.M)
_H2 = re.compile(r"^(##\s+.+)$", re.M)


def _col_sections(readme: str, fn):
    """Apply fn(body) to the body of every column section ('## Cột trong ...', '## Bảng ...')."""
    parts = _H2.split(readme)
    for i in range(1, len(parts), 2):
        t = parts[i].lower()
        if "cột trong" in t or t.startswith("## bảng"):
            parts[i + 1] = fn(parts[i + 1])
    return "".join(parts)


def _rename_bullets(readme: str, by_squash: dict[str, str]) -> str:
    def sub(m):
        name = m.group(2)
        pre, _, col = name.rpartition(".") if "." in name and not name.endswith(".") else ("", "", name)
        new = by_squash.get(squash(col), col)
        return f"{m.group(1)}{pre + '.' if pre else ''}{new}{m.group(3)}"
    return _col_sections(readme, lambda b: _BULLET.sub(sub, b))


# ---------------------------------------------------------------------- transforms
ABBR = [("so luong", "SL"), ("don gia", "DG"), ("thanh tien", "TT"), ("khach hang", "KH"), ("san pham", "SP"),
        ("hang hoa", "HH"), ("nhan vien", "NV"), ("trang thai", "TThai"), ("ghi chu", "GChu"), ("thanh toan", "TToan"),
        ("phuong thuc", "PT"), ("giam gia", "GG"), ("chiet khau", "CK"), ("don vi", "DVi"), ("nha cung cap", "NCC"),
        ("so dien thoai", "SDT"), ("ngay", "Ng"), ("tong", "Tg"), ("ma ", "Ma_"), ("ten ", "Ten_"), ("gia ", "Gia_")]


def odd_name(name: str) -> str:
    s = fold(str(name)) + " "
    for a, b in ABBR:
        s = s.replace(a, b + " ")
    toks = [t for t in re.split(r"[^A-Za-z0-9_]+", s) if t]
    return ".".join(t[:1].upper() + t[1:] for t in toks) or str(name)


def t_rename(shop: Shop, rng):
    """Odd abbreviated column names (like 'Slg', 'Đ.vị'); README bullets renamed to match, descriptions kept."""
    by_squash = {}
    for s in shop.sheets:
        for h in s.header:
            if isinstance(h, str) and h.strip() and squash(h) not in by_squash:
                by_squash[squash(h)] = odd_name(h)
    used: dict[str, int] = {}
    for k, v in list(by_squash.items()):
        used[v] = used.get(v, 0) + 1
        if used[v] > 1:
            by_squash[k] = f"{v}{used[v]}"
    for s in shop.sheets:
        s.header = [by_squash.get(squash(h), h) if isinstance(h, str) else h for h in s.header]
    shop.readme = _rename_bullets(shop.readme, by_squash)


def t_nodoc(shop: Shop, rng):
    """README without column descriptions (4/30 train shops are like this)."""
    shop.readme = _col_sections(shop.readme, lambda b: re.sub(r"^-\s+\*\*.+?\*\*\s*:.*\n?", "", b, flags=re.M))


def t_layout(shop: Shop, rng):
    """Shuffled column order, title rows above the header, csv <-> xlsx."""
    files = shop.files()
    for fname, sheets in files.items():
        for s in sheets:
            perm = list(range(len(s.header)))
            rng.shuffle(perm)
            s.header = [s.header[i] for i in perm]
            s.rows = [[(r[i] if i < len(r) else None) for i in perm] for r in s.rows]
        if fname.lower().endswith(".csv"):
            new = Path(fname).with_suffix(".xlsx").name
            for s in sheets:
                s.file, s.sheet = new, "Sheet1"
                s.top = [[f"BÁO CÁO {Path(fname).stem.upper()}"], [f"Xuất ngày {date(2026, 10, 1):%d/%m/%Y}"], []]
            shop.readme = shop.readme.replace(f"`{fname}`", f"`{new}`").replace("(csv)", "(xlsx)")
        elif len(sheets) == 1 and not sheet_period(sheets[0].sheet or ""):
            new = Path(fname).with_suffix(".csv").name
            sheets[0].file, sheets[0].sheet, sheets[0].top = new, None, []
            shop.readme = shop.readme.replace(f"`{fname}`", f"`{new}`")
        else:
            for s in sheets:
                s.top = [[f"SỔ {Path(fname).stem.upper()} - {s.sheet}"], []] + s.top


_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?$")
_MONEY = re.compile(r"tien|gia|tong|phi|giam|chiet khau|thanh toan|tra|no\b|von")
_NOT_MONEY = re.compile(r"\bma\b|ngay|ty le|ti le|phan tram|%|so luong|sl\b|ten")


def _dmy(v):
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.strftime("%d/%m/%Y %H:%M:%S") if (v.hour or v.minute or v.second) else v.strftime("%d/%m/%Y")
    if isinstance(v, date):
        return v.strftime("%d/%m/%Y")
    if isinstance(v, str) and (m := _ISO.match(v.strip())):
        y, mo, d, hh, mi, ss = m.groups()
        return f"{d}/{mo}/{y}" + (f" {hh}:{mi}:{ss or '00'}" if hh else "")
    return v


def _dots(v):
    try:
        x = float(str(v).replace(",", "")) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        return v
    if x != x or abs(x) < 1000 or x != int(x):
        return v
    return f"{int(x):,}".replace(",", ".")


def t_values(shop: Shop, rng):
    """Dates as dd/mm/yyyy text, money as '1.250.000' text."""
    for s in shop.sheets:
        money = []
        for h in s.header:
            k = fold(f"{h} {s.doc.get(h, '')}")
            money.append(bool(_MONEY.search(k)) and not _NOT_MONEY.search(fold(str(h))))
        s.rows = [[_dots(v) if (j < len(money) and money[j]) else _dmy(v) for j, v in enumerate(r)] for r in s.rows]


def t_status_note(shop: Shop, rng):
    """No status column: status written in the notes column (rulebook public §8 'Sổ không có cột trạng thái')."""
    removed = set()
    for s in shop.sheets:
        st = [j for j, h in enumerate(s.header) if isinstance(h, str) and
              (squash(h) in ("trangthai", "tinhtrang", "status") or "trang thai" in fold(s.doc.get(h, ""))[:40])]
        if len(st) != 1:
            continue
        j = st[0]
        removed.add(squash(s.header[j]))
        nt = [i for i, h in enumerate(s.header) if isinstance(h, str) and squash(h) in ("ghichu", "note", "notes")]
        rows = []
        for r in s.rows:
            r = list(r) + [None] * (len(s.header) - len(r))
            stv = r[j]
            if nt:
                cur = r[nt[0]]
                cur = "" if cur is None or (isinstance(cur, float) and cur != cur) else str(cur)
                stv_s = "" if stv is None or (isinstance(stv, float) and stv != stv) else str(stv)
                r[nt[0]] = "; ".join(x for x in (stv_s, cur) if x) or None
            else:
                r.append(stv)
            del r[j]
            rows.append(r)
        if not nt:
            s.header = s.header + ["Ghi chú"]
        del s.header[j]
        s.rows = rows
    if removed:
        def drop(body):
            body = "".join(ln for ln in body.splitlines(keepends=True)
                           if not ((m := _BULLET.match(ln)) and squash(m.group(2).rpartition(".")[2]) in removed))
            if "**Ghi chú**" not in body:
                body = body.rstrip("\n") + "\n- **Ghi chú**: ghi chú của đơn\n\n"
            return body
        shop.readme = _col_sections(shop.readme, drop)


TRANSFORMS = {
    "rename": [t_rename],
    "nodoc": [t_nodoc],
    "layout": [t_layout],
    "values": [t_values],
    "status_note": [t_status_note],
    "combo": [t_rename, t_values, t_layout],
}


def make(transform: str, shops: list[str], seed: int = 0) -> Path:
    root = OUT / transform
    root.mkdir(parents=True, exist_ok=True)
    for biz in shops:
        shop = load(TRAIN / "data" / biz)
        if shop is None:
            continue
        rng = random.Random(f"{seed}-{transform}-{biz}")
        for fn in TRANSFORMS[transform]:
            fn(shop, rng)
        save(shop, root)
    return root
