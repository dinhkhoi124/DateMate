"""Deterministic helpers `H` handed to model-written code (D20, ARCHITECTURE §8).

Everything a small model tends to get wrong but our code already does right: periods from Vietnamese time phrases
(rulebook §2 incl. Tết via calendar.csv), entity lookup in the shop catalog, rulebook §9 rounding. The model decides
*what* to compute; these functions and pandas compute it (CLAUDE.md rule 4).
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd

from datamate.config import Rules
from datamate.timeparse.periods import (LunarCalendar, Period, find_periods, month_period, previous_period,
                                        quarter_period, same_period_last_year, tet_period, week_start)


def _p(p: Period | None) -> dict | None:
    if p is None:
        return None
    return {"start": p.start, "end": p.end, "kind": p.kind, "label": p.label}


def round_half_up(x: float, nd: int = 0) -> float:
    """rulebook §9: half rounds up (away from zero for negatives, like the gold)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return x
    q = 10 ** nd
    v = abs(float(x)) * q
    r = math.floor(v + 0.5 + 1e-9) / q
    return -r if x < 0 else r


class Helpers:
    """API documented to the model in agent/prompt.py:HELPER_DOC (keep both in sync)."""

    def __init__(self, as_of: date, question_norm: str, cal: LunarCalendar, rules: Rules, linker=None,
                 calendar_df: pd.DataFrame | None = None):
        self.as_of = as_of
        self._q = question_norm
        self._cal, self._rules, self._lk = cal, rules, linker
        self.calendar = calendar_df

    # ---- time (rulebook §2) -------------------------------------------------------------------------------------
    def periods(self, text: str | None = None) -> list[dict]:
        """Periods mentioned in `text` (default: this question), in order of appearance, relative to as_of."""
        from datamate.normalize.text import normalize  # noqa: PLC0415
        t = self._q if text is None else normalize(text)
        return [_p(p) for p in find_periods(t, self.as_of, self._cal, self._rules)]

    def month(self, y: int, m: int) -> dict:
        return _p(month_period(y, m))

    def quarter(self, y: int, q: int) -> dict:
        return _p(quarter_period(y, q))

    def tet(self, lunar_year: int) -> dict | None:
        return _p(tet_period(lunar_year, self._cal, self._rules))

    def week_start(self, d: date) -> date:
        return week_start(d, self._rules)

    def previous(self, p: dict) -> dict:
        """Period of the same length right before p (kỳ trước)."""
        return _p(previous_period(Period(p["start"], p["end"], p.get("kind", "range"), p.get("label", "")), self.as_of))

    def last_year(self, p: dict) -> dict:
        """Same period one year earlier (cùng kỳ năm trước)."""
        return _p(same_period_last_year(Period(p["start"], p["end"], p.get("kind", "range"), p.get("label", ""))))

    @staticmethod
    def in_period(s: pd.Series, p: dict) -> pd.Series:
        """Boolean mask: dates/datetimes of `s` fall inside p (inclusive, by calendar day)."""
        d = pd.to_datetime(s, errors="coerce").dt.normalize()
        return (d >= pd.Timestamp(p["start"])) & (d <= pd.Timestamp(p["end"]))

    @staticmethod
    def days(start: date, end: date) -> list[date]:
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]

    # ---- catalog lookup -----------------------------------------------------------------------------------------
    def _find(self, dic, text: str | None) -> list[dict]:
        if self._lk is None or dic is None:
            return []
        from datamate.normalize.text import normalize  # noqa: PLC0415
        t = self._q if text is None else normalize(text)
        return [{"key": h.key, "matched": h.text, "ambiguous": h.ambiguous} for h in dic.find(t)]

    def find_products(self, text: str | None = None) -> list[dict]:
        """Products named in text (default: the question). key = product_key column of T tables."""
        return self._find(getattr(self._lk, "products", None), text)

    def find_customers(self, text: str | None = None) -> list[dict]:
        return self._find(getattr(self._lk, "customers", None), text)

    def find_categories(self, text: str | None = None) -> list[dict]:
        return self._find(getattr(self._lk, "categories", None), text)

    # ---- rounding (rulebook §9) ---------------------------------------------------------------------------------
    @staticmethod
    def money(x) -> int:
        """VND: integer, half rounds up."""
        return int(round_half_up(float(x), 0))

    @staticmethod
    def r2(x) -> float:
        """Percent and other averages: 2 decimals, half rounds up."""
        return round_half_up(float(x), 2)

    @staticmethod
    def pct(part, whole) -> float | None:
        """100*part/whole rounded to 2 decimals; None when whole == 0."""
        whole = float(whole)
        return None if whole == 0 else round_half_up(100.0 * float(part) / whole, 2)
