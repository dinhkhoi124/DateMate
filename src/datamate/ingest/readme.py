"""Parse a shop README.md into a structured spec.

README layout (BTC convention):
    # Dữ liệu <biz>
    <free text: export style, file splitting, sheet per month...>
    ## File
    - `name.xlsx` (xlsx)
    ## Cột trong `name.xlsx`          (may be absent: column names then inferred)
    - **Col name**: description
    ## Bảng ...                        (sqlite variant)
    ## Quy ước
    - Dữ liệu có đến hết dd/mm/yyyy.
    - **Topic**: note text, possibly with nested bullets (unit conversions "X: 1 hộp = 6 cái")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from datamate.ingest.values import parse_date

_H2 = re.compile(r"^##\s+(.+?)\s*$", re.M)
_FILE = re.compile(r"^-\s+`(.+?)`(?:\s*\((\w+)\))?", re.M)
_COL = re.compile(r"^-\s+\*\*(.+?)\*\*\s*:\s*(.+?)\s*$", re.M)
_SEC_FILE = re.compile(r"Cột trong\s+`(.+?)`", re.I)
_DATA_END = re.compile(r"[Dd]ữ liệu có (?:đến|tới) hết (?:ngày )?(\d{1,2}/\d{1,2}/\d{4})")
_DATA_START = re.compile(r"[Dd]ữ liệu (?:có )?từ (?:ngày )?(\d{1,2}/\d{1,2}/\d{4})")
_UNIT_CONV = re.compile(r"^\s*-\s+(.+?):\s*1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+?)\s*$", re.M)


@dataclass
class ColumnSpec:
    name: str
    desc: str


@dataclass
class ReadmeSpec:
    business_id: str
    intro: str = ""
    files: list[str] = field(default_factory=list)
    columns: dict[str, list[ColumnSpec]] = field(default_factory=dict)  # file or table -> columns
    notes: str = ""
    data_start: date | None = None
    data_end: date | None = None
    unit_conversions: list[tuple[str, str, float, str]] = field(default_factory=list)  # (product, big, factor, base)
    raw: str = ""

    def columns_for(self, file_or_table: str) -> list[ColumnSpec]:
        if file_or_table in self.columns:
            return self.columns[file_or_table]
        stem = Path(file_or_table).stem
        for k, v in self.columns.items():
            if Path(k).stem == stem:
                return v
        return []

    def all_columns(self) -> list[ColumnSpec]:
        return [c for cols in self.columns.values() for c in cols]


def parse_readme(path: Path | str) -> ReadmeSpec:
    path = Path(path)
    text = path.read_text(encoding="utf8")
    spec = ReadmeSpec(business_id=path.parent.name, raw=text)
    parts = _H2.split(text)
    spec.intro = re.sub(r"^#.*$", "", parts[0], flags=re.M).strip()
    for i in range(1, len(parts), 2):
        title, body = parts[i], parts[i + 1]
        tl = title.lower()
        if tl.startswith("file"):
            spec.files = [m.group(1) for m in _FILE.finditer(body)]
        elif _SEC_FILE.search(title):
            spec.columns[_SEC_FILE.search(title).group(1)] = [ColumnSpec(c.strip(), d.strip()) for c, d in _COL.findall(body)]
        elif tl.startswith("bảng") or tl.startswith("bang"):
            _parse_sqlite_section(title, body, spec)
        elif "quy ước" in tl or "quy uoc" in tl or "ghi chú" in tl:
            spec.notes += body.strip() + "\n"
        else:
            spec.notes += f"## {title}\n{body.strip()}\n"
    blob = spec.intro + "\n" + spec.notes
    if m := _DATA_END.search(blob):
        spec.data_end = parse_date(m.group(1))
    if m := _DATA_START.search(blob):
        spec.data_start = parse_date(m.group(1))
    for prod, big, fac, base in _UNIT_CONV.findall(blob):
        try:
            spec.unit_conversions.append((prod.strip(), big.strip(), float(fac.replace(",", ".")), base.strip()))
        except ValueError:
            pass
    return spec


def _parse_sqlite_section(title: str, body: str, spec: ReadmeSpec):
    """`## Bảng `a`, `b`` lists tables; columns may be given as `- **table.col**: desc`."""
    for c, d in _COL.findall(body):
        if "." in c:
            t, col = c.split(".", 1)
            spec.columns.setdefault(t.strip("` "), []).append(ColumnSpec(col.strip("` "), d.strip()))
        else:
            spec.columns.setdefault("_sqlite", []).append(ColumnSpec(c.strip(), d.strip()))
    spec.notes += f"## {title}\n{body.strip()}\n"
