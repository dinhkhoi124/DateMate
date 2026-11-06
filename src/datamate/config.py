"""Rule parameters (CLAUDE.md: parameters live in config, not scattered in code).

Defaults are the train rulebook values; `from_rulebook(text)` re-reads them from the split's rulebook.md at run time
(DECISIONS D07) so a private rulebook with other shift hours / Tết window is honoured.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

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


@dataclass(frozen=True)
class Invariants:
    """Run-time ingest invariants (DECISIONS D11, ingest/checks.py). Wide, meaning-based bounds: every dev shop is
    far inside them (eval/synth/faults.py), a misread table is far outside."""
    min_orders_with_date: float = 0.98     # orders whose date could not be read
    max_dup_order_id: float = 0.01         # same order id twice: a table read twice or the wrong id column
    max_after_data_end: float = 0.01       # dates after the README 'Dữ liệu có đến hết ...'
    min_share_day_gt12: float = 0.20       # ~19/31 expected; dd/mm read as mm/dd leaves days 13..31 empty
    year_range: tuple[int, int] = (2000, 2100)
    min_done: float = 0.50                 # most orders are completed
    max_cancelled: float = 0.30
    max_returned: float = 0.30
    min_orders_status_evidence: int = 500   # this many orders, none cancelled/returned: status unread
    min_orders_with_rev: float = 0.90
    max_neg_rev: float = 0.05
    order_money_range: tuple[float, float] = (1_000.0, 1e9)   # median order in VND: '1.250.000' read as 1.25
    max_subtotal_mismatch: float = 0.05    # line sum vs the order table's own 'tiền hàng'
    max_orphan_lines: float = 0.05         # lines whose order id is not an order
    max_orders_without_lines: float = 0.05
    min_amount_vs_qty_price: float = 0.90  # line amount = qty x price (after line discount)
    max_qty_base_lt_qty: float = 0.05      # quantity in base unit smaller than in the line unit
    max_debt_negative: float = 0.01
    max_debt_gt_total: float = 0.01


INVARIANTS = Invariants()


def cache_root() -> Path:
    """Writable cache dir. SPEC §7 câu 2: in Docker /data is read-only and temp files go to /tmp, so predict.sh sets
    DATAMATE_CACHE=/tmp/datamate_cache. Default (dev): <repo>/cache."""
    env = os.environ.get("DATAMATE_CACHE")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "cache"
