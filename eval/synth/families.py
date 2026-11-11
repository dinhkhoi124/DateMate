"""Benchmark families G1-G6 for the agent-generalization benchmark (E0, docs/BENCHMARK.md).

Each family is a function `f(shop, c) -> bool` (True = the shop was changed) taking a `Ctx` with the side's
parameters: side A (Synth-Tune) and side B (Synth-Eval) use DIFFERENT vocabularies / formats / rulebook rows, so a fix
tuned on A cannot memorise B's transform (anti-leakage, BENCHMARK.md §3).

Label of a question in a family cell (BENCHMARK.md §2):
- same          : meaning kept, train gold stays valid                      (G1-G5 except drop_table, status_newword)
- unanswerable  : the data needed was removed; correct = unanswerable      (G5 drop_table)
- recomputed    : rulebook parameter changed; gold recomputed by the deterministic rule engine (G6 rule_param)
- changed       : rulebook exception added; new gold unknown -> only "stale" (old gold) answers are scored (G6 rule_exception)

data/ is only read (CLAUDE.md rule 7). Dev shops only (rule 2): enforced by eval/synth/bench.py.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from eval.synth.make import (_BULLET, _H2, _dmy, _dots, _MONEY, _NOT_MONEY, Sheet, Shop, fold,
                             odd_name, real_name, sheet_period, squash, t_layout, t_nodoc, _rename_with)

SIDES = ("A", "B")


@dataclass
class Ctx:
    side: str
    rng: random.Random
    kinds: list = field(default_factory=list)      # Store.table_kinds of the ORIGINAL shop: (name, kind, roles)
    cancelled_raw: set = field(default_factory=set)  # raw status texts ingest reads as cancelled (original shop)
    drop_kind: str | None = None                   # drop_table: group chosen by bench.py


# ---------------------------------------------------------------------- helpers
def sheet_roles(s: Sheet, kinds: list) -> tuple[str | None, dict]:
    """(kind, {role: column}) of the ingest table whose role columns all appear in this sheet's header."""
    hdr = {squash(h) for h in s.header if isinstance(h, str)}
    best = (None, {})
    for _, kind, roles in kinds:
        cols = {squash(c) for c in roles.values() if isinstance(c, str)}
        if cols and cols <= hdr and len(roles) > len(best[1]):
            best = (kind, roles)
    return best


def _col_idx(s: Sheet, col: str) -> int | None:
    for j, h in enumerate(s.header):
        if isinstance(h, str) and squash(h) == squash(col):
            return j
    return None


def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip())


def _replace_name(readme: str, old: str, new: str) -> str:
    """File / sheet name references in README (with or without extension, backticked or bare)."""
    for o, n in ((old, new), (Path(old).stem, Path(new).stem)):
        if o and o != n:
            readme = readme.replace(o, n)
    return readme


def _append_note(readme: str, col: str, note: str) -> str:
    """Append `note` to the first README bullet of column `col`; no such bullet -> a notes section at the end."""
    lines = readme.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        m = _BULLET.match(ln)
        if m:
            name = m.group(2).rpartition(".")[2] if "." in m.group(2) else m.group(2)
            if squash(name) == squash(str(col)):
                lines[i] = ln.rstrip("\n") + f" {note}\n"
                return "".join(lines)
    return readme.rstrip("\n") + f"\n\n## Ghi chú thêm\n\n- **{col}**: {note}\n"


# ---------------------------------------------------------------------- G1 renaming
def f_rename_cols(shop: Shop, c: Ctx) -> bool:
    """G1. A: real abbreviations (SL, ĐG, KH, PTTT). B: made-up names (Tg.Tien.Hang, TThai). README bullets renamed."""
    before = [list(s.header) for s in shop.sheets]
    _rename_with(shop, real_name if c.side == "A" else odd_name)
    return before != [list(s.header) for s in shop.sheets]


def f_rename_tables(shop: Shop, c: Ctx) -> bool:
    """G1. File names (and non-month sheet names) replaced by neutral ones. A: 'BangKe_01.xlsx' / 'Trang1'.
    B: 'export_a.xlsx' / 'tab_1'. README references renamed consistently."""
    files = list(shop.files().items())
    for i, (fname, sheets) in enumerate(files):
        suf = Path(fname).suffix
        new = f"BangKe_{i + 1:02d}{suf}" if c.side == "A" else f"export_{chr(97 + i)}{suf}"
        for s in sheets:
            s.file = new
        shop.readme = _replace_name(shop.readme, fname, new)
        for j, s in enumerate(sheets):
            if s.sheet and not sheet_period(s.sheet) and len(sheets) > 1:
                ns = f"Trang{j + 1}" if c.side == "A" else f"tab_{j + 1}"
                shop.readme = shop.readme.replace(f"`{s.sheet}`", f"`{ns}`")
                s.sheet = ns
    return bool(files)


# ---------------------------------------------------------------------- G2 layout
def f_layout(shop: Shop, c: Ctx) -> bool:
    """G2. A: shuffled columns + title rows + csv<->xlsx (make.t_layout). B: shuffled columns + one blank column +
    the header repeated every 400 rows + csv<->xlsx without titles."""
    if c.side == "A":
        t_layout(shop, c.rng)
        return True
    t_layout(shop, c.rng)
    for s in shop.sheets:
        s.top = []
        pos = c.rng.randrange(len(s.header) + 1)
        s.header = s.header[:pos] + [None] + s.header[pos:]
        rows = []
        for i, r in enumerate(s.rows):
            r = list(r) + [None] * (len(s.header) - 1 - len(r))
            if i and i % 400 == 0:
                rows.append(list(s.header))
            rows.append(r[:pos] + [None] + r[pos:])
        s.rows = rows
    return True


# ---------------------------------------------------------------------- G3 values
def _dmy_b(v):
    """d/m/yyyy without zero padding, time kept as H:MM."""
    s = _dmy(v)
    if isinstance(s, str) and re.match(r"^\d{2}/\d{2}/\d{4}", s):
        d, m, rest = s[:2], s[3:5], s[5:]
        rest = re.sub(r" (\d{2}):(\d{2}):\d{2}$", lambda mm: f" {int(mm.group(1))}:{mm.group(2)}", rest)
        return f"{int(d)}/{int(m)}{rest}"
    return s


def _money_b(v):
    s = _dots(v)
    return s.replace(".", ",") + "đ" if isinstance(s, str) and s != v else s


def f_values(shop: Shop, c: Ctx) -> bool:
    """G3. A: dates 'dd/mm/yyyy', money '1.250.000' (make.t_values). B: dates 'd/m/yyyy H:MM', money '1,250,000đ'."""
    datefn, moneyfn = (_dmy, _dots) if c.side == "A" else (_dmy_b, _money_b)
    for s in shop.sheets:
        money = []
        for h in s.header:
            k = fold(f"{h} {s.doc.get(h, '')}")
            money.append(bool(_MONEY.search(k)) and not _NOT_MONEY.search(fold(str(h))))
        s.rows = [[moneyfn(v) if (j < len(money) and money[j]) else datefn(v) for j, v in enumerate(r)] for r in s.rows]
    return True


# ---------------------------------------------------------------------- G4 metadata
def f_nodoc(shop: Shop, c: Ctx) -> bool:
    """G4. A: README without column descriptions (make.t_nodoc). B: README cut to its title and first paragraph."""
    if c.side == "A":
        before = shop.readme
        t_nodoc(shop, c.rng)
        return shop.readme != before
    head = _H2.split(shop.readme)[0].strip()
    shop.readme = head + "\n"
    return True


# ---------------------------------------------------------------------- G5 structure
def f_split_months(shop: Shop, c: Ctx) -> bool:
    """G5. The biggest single-sheet table with an order date becomes one xlsx with one sheet per month.
    A: sheets 'T6-2025'. B: sheets 'Thang 06.2025'."""
    cand = []
    for fname, sheets in shop.files().items():
        if len(sheets) != 1:
            continue
        s = sheets[0]
        kind, roles = sheet_roles(s, c.kinds)
        col = roles.get("order_date") or roles.get("order_datetime")
        j = _col_idx(s, col) if col else None
        if j is not None and kind in ("lines", "orders"):
            cand.append((len(s.rows), fname, s, j))
    if not cand:
        return False
    _, fname, s, j = max(cand, key=lambda x: (x[0], x[1]))
    raw = pd.Series([r[j] if j < len(r) else None for r in s.rows], dtype=object).astype(str)
    dayfirst = bool(raw.str.match(r"^\d{1,2}/\d{1,2}/\d{4}").mean() > 0.5)
    d = pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=dayfirst)
    if d.isna().mean() > 0.01 or d.dt.to_period("M").nunique() < 2:
        return False
    new = Path(fname).with_suffix(".xlsx").name
    out = []
    for per in sorted(d.dt.to_period("M").dropna().unique()):
        idx = [i for i, p in enumerate(d.dt.to_period("M")) if p == per]
        name = f"T{per.month}-{per.year}" if c.side == "A" else f"Thang {per.month:02d}.{per.year}"
        out.append(Sheet(new, name, [], list(s.header), [s.rows[i] for i in idx], s.doc, None))
    shop.sheets = [x for x in shop.sheets if x is not s] + out
    shop.readme = _replace_name(shop.readme, fname, new)
    shop.readme = shop.readme.rstrip("\n") + f"\n\nFile `{new}` chia mỗi tháng một sheet.\n"
    return True


def _status_col(s: Sheet, c: Ctx) -> int | None:
    kind, roles = sheet_roles(s, c.kinds)
    if kind not in ("lines", "orders") or not roles.get("status"):
        return None
    return _col_idx(s, roles["status"])


def f_status_codes(shop: Shop, c: Ctx) -> bool:
    """G5. Order status written as codes with a legend in README. A: '1', '2', ... B: 'TT01', 'TT02', ..."""
    changed = False
    for s in shop.sheets:
        j = _status_col(s, c)
        if j is None:
            continue
        vals = sorted({str(r[j]).strip() for r in s.rows if j < len(r) and not _empty(r[j])})
        if not vals:
            continue
        code = {v: (str(i + 1) if c.side == "A" else f"TT{i + 1:02d}") for i, v in enumerate(vals)}
        s.rows = [[(code.get(str(v).strip(), v) if (k == j and not _empty(v)) else v) for k, v in enumerate(r)]
                  for r in s.rows]
        legend = ", ".join(f"{code[v]} = {v}" for v in vals)
        shop.readme = _append_note(shop.readme, s.header[j], f"Ghi bằng mã: {legend}.")
        changed = True
    return changed


_NOISE = {
    "A": ("LichTrucNhanVien.xlsx", "Lịch trực", ["Ngày", "Nhân viên trực", "Ca trực"],
          lambda d, i: [d.strftime("%Y-%m-%d"), ["Lan", "Minh", "Hùng"][i % 3], ["Sáng", "Tối"][i % 2]],
          {"Ngày": "ngày trực", "Nhân viên trực": "tên người trực cửa hàng", "Ca trực": "ca trực trong ngày"}),
    "B": ("nhiet_do_kho.csv", None, ["Ngày đo", "Nhiệt độ (°C)", "Độ ẩm (%)"],
          lambda d, i: [d.strftime("%d/%m/%Y"), 24 + (i * 7) % 9, 55 + (i * 11) % 30],
          {"Ngày đo": "ngày ghi nhiệt kế trong kho", "Nhiệt độ (°C)": "nhiệt độ kho", "Độ ẩm (%)": "độ ẩm kho"}),
}


def f_noise_table(shop: Shop, c: Ctx) -> bool:
    """G5. An extra table unrelated to sales (A: staff rota, B: storeroom thermometer log) with its README section."""
    fname, sheet, header, row, doc = _NOISE[c.side]
    start = date(2025, 1, 1)
    rows = [row(start + timedelta(days=i), i) for i in range(120)]
    shop.sheets.append(Sheet(fname, sheet, [], header, rows, doc, None))
    bullets = "\n".join(f"- **{k}**: {v}" for k, v in doc.items())
    shop.readme = shop.readme.rstrip("\n") + f"\n\n## Cột trong `{fname}`\n\n{bullets}\n"
    return True


# groups of ingest kinds that can be removed, and the OP_AREAS area (ingest/checks.py) that needs them
DROP_GROUPS = {"imports": ("imports", "import_lines"), "stock": ("stock",), "lots": ("lots", "lot_sales")}


def drop_groups_present(kinds: list) -> list[str]:
    ks = {k for _, k, _ in kinds}
    return sorted(g for g, members in DROP_GROUPS.items() if ks & set(members))


def f_drop_table(shop: Shop, c: Ctx) -> bool:
    """G5 (label unanswerable). Every sheet of group `c.drop_kind` removed together with its README section:
    questions that need it can no longer be answered from the data."""
    members = set(DROP_GROUPS.get(c.drop_kind or "", ()))
    gone = [s for s in shop.sheets if sheet_roles(s, c.kinds)[0] in members]
    if not gone:
        return False
    shop.sheets = [s for s in shop.sheets if s not in gone]
    left = {s.file for s in shop.sheets}
    parts = _H2.split(shop.readme)
    for f in {s.file for s in gone} - left:
        for i in range(1, len(parts), 2):
            if f in parts[i] or Path(f).stem in parts[i]:
                parts[i], parts[i + 1] = "", ""
    shop.readme = "".join(parts)
    return True


# ---------------------------------------------------------------------- G6 semantics (data side)
_NEWWORD = {"A": "Bom hàng", "B": "Khách không nhận hàng"}


def f_status_newword(shop: Shop, c: Ctx) -> bool:
    """G6 (label same). Cancelled orders written with a word ingest has never seen; README explains it."""
    if not c.cancelled_raw:
        return False
    new = _NEWWORD[c.side]
    canc = {fold(x) for x in c.cancelled_raw}
    changed = False
    for s in shop.sheets:
        j = _status_col(s, c)
        if j is None:
            continue
        n0 = sum(1 for r in s.rows if j < len(r) and not _empty(r[j]) and fold(str(r[j]).strip()) in canc)
        if not n0:
            continue
        s.rows = [[(new if (k == j and not _empty(v) and fold(str(v).strip()) in canc) else v) for k, v in enumerate(r)]
                  for r in s.rows]
        shop.readme = _append_note(shop.readme, s.header[j], f"Giá trị '{new}' nghĩa là đơn đã huỷ.")
        changed = True
    return changed


# ---------------------------------------------------------------------- G6 semantics (rulebook side)
def _sub_row(md: str, rid: str, pairs: list[tuple[str, str]]) -> str:
    from datamate.rulebook import parse_rows   # noqa: PLC0415
    row = next(r for r in parse_rows(md) if r.rid == rid)
    lines = md.splitlines()
    ln = lines[row.line - 1]
    for a, b in pairs:
        if a not in ln:
            raise ValueError(f"rulebook {rid}: {a!r} not found")
        ln = ln.replace(a, b)
    lines[row.line - 1] = ln
    return "\n".join(lines) + "\n"


def rule_param(md: str, side: str) -> str:
    """G6 (label recomputed). Parameter rows the rule engine re-reads (rulebook.PARAMS), changed.
    A: shifts (§2.12). B: week start (§2.6) and Tết range (§2.11)."""
    if side == "A":
        return _sub_row(md, "§2.12", [("06:00–13:59", "07:00–12:59"), ("14:00–21:59", "13:00–20:59")])
    md = _sub_row(md, "§2.6", [("bắt đầu **thứ Hai**", "bắt đầu **Chủ nhật**"), ("từ thứ Hai", "từ Chủ nhật"),
                               ("thứ Hai đến Chủ nhật", "Chủ nhật đến thứ Bảy")])
    return _sub_row(md, "§2.11", [("**23 tháng Chạp**", "**25 tháng Chạp**"),
                                  ("**mùng 10 tháng Giêng**", "**mùng 7 tháng Giêng**"),
                                  ("trước 23 tháng Chạp", "trước 25 tháng Chạp")])


# rows changed by an exception sentence (our own wording); different rows per side
EXCEPTIONS = {
    "A": {"§1.5": "Ngoại lệ: số đơn đếm cả đơn đã huỷ và đơn trả hàng.",
          "§1.6": "Ngoại lệ: doanh thu một đơn tính cả phí giao hàng và thuế VAT."},
    "B": {"§3.3": "Ngoại lệ: bán chạy xếp theo số tiền bán được, không theo số lượng.",
          "§4.2": "Ngoại lệ: nợ một đơn không gồm phí giao hàng."},
}


def rule_exception(md: str, side: str) -> str:
    """G6 (label changed). An exception sentence appended to rows whose scope (rulebook.scope_of) is op-based."""
    from datamate.rulebook import parse_rows   # noqa: PLC0415
    lines = md.splitlines()
    rows = {r.rid: r for r in parse_rows(md)}
    for rid, sent in EXCEPTIONS[side].items():
        i = rows[rid].line - 1
        lines[i] = lines[i].rstrip().rstrip("|").rstrip() + " " + sent + " |"
    return "\n".join(lines) + "\n"


def exception_ops(side: str) -> set[str]:
    from datamate.rulebook import scope_of   # noqa: PLC0415
    out = set()
    for rid in EXCEPTIONS[side]:
        sc = scope_of(rid)
        out |= {s for s in (sc if isinstance(sc, list) else []) if not s.startswith(("period:", "["))}
    return out


# ---------------------------------------------------------------------- registry
# data families: (group, function, label). Rulebook families are handled in bench.py (no data copy).
DATA_FAMILIES = {
    "rename_cols": ("G1", f_rename_cols, "same"),
    "rename_tables": ("G1", f_rename_tables, "same"),
    "layout": ("G2", f_layout, "same"),
    "values": ("G3", f_values, "same"),
    "nodoc": ("G4", f_nodoc, "same"),
    "split_months": ("G5", f_split_months, "same"),
    "status_codes": ("G5", f_status_codes, "same"),
    "noise_table": ("G5", f_noise_table, "same"),
    "drop_table": ("G5", f_drop_table, "unanswerable"),
    "status_newword": ("G6", f_status_newword, "same"),
}
RULEBOOK_FAMILIES = {"rule_param": ("G6", rule_param, "recomputed"), "rule_exception": ("G6", rule_exception, "changed")}
TARGETED = ("drop_table", "rule_param", "rule_exception")   # only questions the change touches are assigned
