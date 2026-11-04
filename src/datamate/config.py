"""Rule parameters (CLAUDE.md: parameters live in config, not scattered in code).

Defaults are the train rulebook values; `from_rulebook(text)` re-reads them from the split's rulebook.md at run time
(DECISIONS D07) so a private rulebook with other shift hours / Tết window is honoured.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from unidecode import unidecode


@dataclass(frozen=True)
class Rules:
    morning: tuple[int, int] = (6 * 3600, 13 * 3600 + 59 * 60 + 59)     # rulebook §2: ca sáng 06:00–13:59
    afternoon: tuple[int, int] = (14 * 3600, 21 * 3600 + 59 * 60 + 59)  # rulebook §2: ca chiều 14:00–21:59
    week_start: int = 0                    # rulebook §2: tuần bắt đầu thứ Hai (date.weekday() == 0)
    tet_start: tuple[int, int] = (23, 12)  # rulebook §2: 23 tháng Chạp (lunar day, lunar month) of lunar year Y-1
    tet_end: tuple[int, int] = (10, 1)     # rulebook §2: mùng 10 tháng Giêng of lunar year Y
    service_fee_names: tuple[str, ...] = ("phi phuc vu",)  # rulebook §3: Phí phục vụ is not an item


DEFAULT = Rules()


def _hms(h, m):
    return int(h) * 3600 + int(m) * 60


def from_rulebook(text: str | None) -> Rules:
    if not text:
        return DEFAULT
    t = unidecode(text).lower()
    r = DEFAULT
    m = re.search(r"ca sang (\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})", t)
    if m:
        r = replace(r, morning=(_hms(m[1], m[2]), _hms(m[3], m[4]) + 59))
    m = re.search(r"ca chieu (\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})", t)
    if m:
        r = replace(r, afternoon=(_hms(m[1], m[2]), _hms(m[3], m[4]) + 59))
    m = re.search(r"tu ngay duong cua \**(\d{1,2}) thang chap", t)
    if m:
        r = replace(r, tet_start=(int(m[1]), 12))
    m = re.search(r"mung (\d{1,2}) thang gieng", t)
    if m:
        r = replace(r, tet_end=(int(m[1]), 1))
    return r
