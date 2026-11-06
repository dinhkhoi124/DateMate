"""Time expressions -> date ranges (rulebook §2). Works on normalize()d text.

find_periods(text, as_of, cal) returns every period mention in reading order. The parser decides the role of each
(main period, comparison period, list of months...). All ranges are inclusive [start, end].
"""
from __future__ import annotations

import calendar as _cal
import csv
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from datamate.config import DEFAULT, Rules


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    kind: str            # day | range | week | last_n | month | mtd | quarter | ytd | tet | asof
    label: str = ""      # 'YYYY-MM' for months, 'YYYY-MM-DD' for days
    span: tuple = field(default=(0, 0), compare=False)

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def dates(self):
        d = self.start
        while d <= self.end:
            yield d
            d += timedelta(days=1)


# ------------------------------------------------------------------ calendar.csv (lunar dates)
class LunarCalendar:
    def __init__(self, rows: list[dict]):
        self.by_lunar = {}
        for r in rows:
            if str(r.get("lunar_leap", "")).lower() == "true":
                continue
            key = (int(r["lunar_year"]), int(r["lunar_month"]), int(r["lunar_day"]))
            self.by_lunar.setdefault(key, date.fromisoformat(r["date"]))

    @classmethod
    @lru_cache(maxsize=8)
    def load(cls, path: str) -> "LunarCalendar":
        p = Path(path)
        if not p.exists():
            return cls([])
        with open(p, encoding="utf8", newline="") as f:
            return cls(list(csv.DictReader(f)))

    def solar(self, ly: int, lm: int, ld: int) -> date | None:
        return self.by_lunar.get((ly, lm, ld))


def tet_period(year: int, cal: LunarCalendar, rules: Rules = DEFAULT) -> Period | None:
    """rulebook §2: Tết năm Y = 23 tháng Chạp (âm Y-1) .. hết mùng 10 tháng Giêng (âm Y)."""
    s = cal.solar(year - 1, rules.tet_start[1], rules.tet_start[0])
    e = cal.solar(year, rules.tet_end[1], rules.tet_end[0])
    if not s or not e:
        return None
    return Period(s, e, "tet", f"tet{year}")


# ------------------------------------------------------------------ helpers
def month_period(y: int, m: int, as_of: date | None = None, span=(0, 0)) -> Period:
    last = _cal.monthrange(y, m)[1]
    return Period(date(y, m, 1), date(y, m, last), "month", f"{y:04d}-{m:02d}", span)


def quarter_period(y: int, q: int, span=(0, 0)) -> Period:
    s = date(y, 3 * (q - 1) + 1, 1)
    e = month_period(y, 3 * q).end
    return Period(s, e, "quarter", f"{y}-Q{q}", span)


def _year_for(m: int, as_of: date) -> int:
    """rulebook §2: tháng M không ghi năm = năm nay nếu không sau hôm nay, ngược lại năm trước."""
    return as_of.year if date(as_of.year, m, 1) <= as_of else as_of.year - 1


def week_start(d: date, rules: Rules = DEFAULT) -> date:
    return d - timedelta(days=(d.weekday() - rules.week_start) % 7)


def previous_period(p: Period, as_of: date) -> Period:
    """rulebook §1/§2: 'so với kỳ trước'. Tháng này -> cùng số ngày của tháng trước."""
    if p.kind == "month":
        y, m = (p.start.year, p.start.month - 1) if p.start.month > 1 else (p.start.year - 1, 12)
        return month_period(y, m)
    if p.kind == "mtd":
        y, m = (p.start.year, p.start.month - 1) if p.start.month > 1 else (p.start.year - 1, 12)
        last = _cal.monthrange(y, m)[1]
        return Period(date(y, m, 1), date(y, m, min(p.end.day, last)), "mtd_prev", f"{y:04d}-{m:02d}")
    if p.kind == "quarter":
        q = (p.start.month - 1) // 3 + 1
        return quarter_period(p.start.year, q - 1) if q > 1 else quarter_period(p.start.year - 1, 4)
    n = p.days
    return Period(p.start - timedelta(days=n), p.start - timedelta(days=1), "range")


def same_period_last_year(p: Period) -> Period:
    if p.kind == "month":
        return month_period(p.start.year - 1, p.start.month)
    if p.kind == "quarter":
        return quarter_period(p.start.year - 1, (p.start.month - 1) // 3 + 1)

    def shift(d):
        try:
            return d.replace(year=d.year - 1)
        except ValueError:
            return d.replace(year=d.year - 1, day=28)
    return Period(shift(p.start), shift(p.end), p.kind)


# ------------------------------------------------------------------ patterns
D = r"(\d{1,2})/(\d{1,2})/(\d{4})"
_PATTERNS = [
    ("range", re.compile(rf"(?:tu |giai doan tu |trong giai doan tu )?(?:ngay )?{D} (?:den|toi) (?:het )?(?:ngay )?{D}")),
    ("month_range", re.compile(r"(?:tu )?thang (\d{1,2})/(\d{4}) (?:den|toi|qua) (?:het )?thang (\d{1,2})/(\d{4})")),
    ("months_ending", re.compile(r"(\d+) thang lien (?:nhau|tiep) ket thuc (?:o|vao|tai) thang (\d{1,2})/(\d{4})")),
    ("mtd", re.compile(r"tu dau thang nay (?:den|toi) (?:het )?(?:hom nay|nay)|thang nay")),
    ("ytd", re.compile(r"(?:nam nay,? )?(?:tinh )?tu (?:dau nam|ngay 1/1|1/1)(?: nam nay)? (?:den|toi) (?:het )?(?:hom nay|nay)"
                       r"(?:,? (?:doanh thu )?nam nay)?|tu dau nam(?: nay)?(?: toi nay)?")),
    ("last_n", re.compile(r"(\d+) ngay (?:gan nhat|gan day|qua|vua qua)")),
    ("this_week", re.compile(r"tuan nay")),
    ("last_week", re.compile(r"tuan (?:truoc|roi|vua roi)")),
    ("asof_day", re.compile(rf"(?:den|toi|tinh den|tinh toi) (?:het |cuoi )?ngay {D}|(?:cuoi|het) ngay {D}|ton kho ngay {D}")),
    ("asof_today", re.compile(r"(?:den|toi) (?:het |cuoi )?(?:ngay )?hom nay|cuoi (?:ngay )?hom nay|het hom nay|ton cuoi hom nay")),
    ("day", re.compile(rf"(?:ngay|hom) {D}|(?<![\d/]){D}")),
    ("today", re.compile(r"hom nay|ngay hom nay")),
    ("yesterday", re.compile(r"hom qua")),
    ("quarter", re.compile(r"quy (\d)(?:/| nam )(\d{4})|quy (\d)\b")),
    ("tet", re.compile(r"(?:(?:dip|mua) )?\btet\b(?: am lich)?(?: nam)?(?: (nay|truoc|ngoai|\d{4}))?")),
    ("month", re.compile(r"thang (\d{1,2})(?:/| nam )(\d{4})|thang (\d{1,2})\b(?![/\d])|(?<![\d/])(\d{1,2})/(\d{4})(?![\d/])")),
]


def find_periods(text: str, as_of: date, cal: LunarCalendar | None = None, rules: Rules = DEFAULT) -> list[Period]:
    taken = [False] * len(text)
    out: list[Period] = []

    def free(a, b):
        return not any(taken[a:b])

    for kind, rx in _PATTERNS:
        for m in rx.finditer(text):
            a, b = m.span()
            if not free(a, b):
                continue
            ps = _build(kind, m, as_of, cal, rules)
            if ps is None:
                continue
            for i in range(a, b):
                taken[i] = True
            for p in (ps if isinstance(ps, list) else [ps]):
                out.append(Period(p.start, p.end, p.kind, p.label, (a, b)))
    out.sort(key=lambda p: p.span[0])
    return out


def _d(dd, mm, yy):
    try:
        return date(int(yy), int(mm), int(dd))
    except ValueError:
        return None


def _build(kind, m, as_of, cal, rules):
    g = m.groups()
    if kind == "range":
        s, e = _d(*g[0:3]), _d(*g[3:6])
        return Period(s, e, "range") if s and e and s <= e else None
    if kind == "month_range":
        m1, y1, m2, y2 = map(int, g)
        out, y, mo = [], y1, m1
        while (y, mo) <= (y2, m2) and len(out) < 36:
            out.append(month_period(y, mo))
            y, mo = (y, mo + 1) if mo < 12 else (y + 1, 1)
        return [Period(p.start, p.end, "month", p.label) for p in out] if out else None
    if kind == "months_ending":
        n, mo, y = map(int, g)
        out = []
        for _ in range(n):
            out.append(month_period(y, mo))
            y, mo = (y, mo - 1) if mo > 1 else (y - 1, 12)
        return list(reversed(out))
    if kind == "mtd":
        return Period(date(as_of.year, as_of.month, 1), as_of, "mtd", f"{as_of.year:04d}-{as_of.month:02d}")
    if kind == "ytd":
        return Period(date(as_of.year, 1, 1), as_of, "ytd")
    if kind == "last_n":
        n = int(g[0])
        return Period(as_of - timedelta(days=n - 1), as_of, "last_n") if 0 < n < 4000 else None
    if kind == "this_week":
        return Period(week_start(as_of, rules), as_of, "week")
    if kind == "last_week":
        s = week_start(as_of, rules) - timedelta(days=7)
        return Period(s, s + timedelta(days=6), "week")
    if kind == "asof_day":
        dd = next(_d(*g[i:i + 3]) for i in (0, 3, 6) if g[i])
        return Period(dd, dd, "asof", dd.isoformat()) if dd else None
    if kind == "asof_today":
        return Period(as_of, as_of, "asof", as_of.isoformat())
    if kind == "day":
        dd = _d(*g[0:3]) if g[0] else _d(*g[3:6])
        return Period(dd, dd, "day", dd.isoformat()) if dd else None
    if kind == "today":
        return Period(as_of, as_of, "day", as_of.isoformat())
    if kind == "yesterday":
        y = as_of - timedelta(days=1)
        return Period(y, y, "day", y.isoformat())
    if kind == "quarter":
        if g[0]:
            q, y = int(g[0]), int(g[1])
        else:
            q = int(g[2])
            y = as_of.year if date(as_of.year, 3 * (q - 1) + 1, 1) <= as_of else as_of.year - 1
        return quarter_period(y, q) if 1 <= q <= 4 else None
    if kind == "tet":
        if cal is None:
            return None
        w = g[0]
        y = {"nay": as_of.year, None: as_of.year, "truoc": as_of.year - 1, "ngoai": as_of.year - 1}.get(w)
        if y is None:
            y = int(w)
        return tet_period(y, cal, rules)
    if kind == "month":
        if g[0]:
            mo, y = int(g[0]), int(g[1])
        elif g[2]:
            mo = int(g[2])
            if not 1 <= mo <= 12:
                return None
            y = _year_for(mo, as_of)
        else:
            mo, y = int(g[3]), int(g[4])
        return month_period(y, mo) if 1 <= mo <= 12 else None
    return None


__all__ = ["Period", "LunarCalendar", "find_periods", "previous_period", "same_period_last_year", "tet_period",
           "month_period", "quarter_period", "week_start"]
