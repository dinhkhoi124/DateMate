from datetime import date
from pathlib import Path

import pytest

from datamate.normalize.text import normalize
from datamate.timeparse.periods import (LunarCalendar, find_periods, previous_period, tet_period)

CAL_PATH = Path("data/train/train/calendar.csv")
CAL = LunarCalendar.load(str(CAL_PATH))  # empty calendar when data is missing
T = date(2026, 6, 24)  # Wednesday


def P(q, as_of=T):
    return [(p.kind, p.start.isoformat(), p.end.isoformat()) for p in find_periods(normalize(q), as_of, CAL)]


def test_basic_days():
    assert P("Doanh thu hôm nay được bao nhiêu?") == [("day", "2026-06-24", "2026-06-24")]
    assert P("xem giúp dthu hqua") == [("day", "2026-06-23", "2026-06-23")]


def test_week_rulebook_s2():
    # rulebook §2: tuần bắt đầu thứ Hai; tuần trước = cả tuần liền trước
    assert P("doanh thu tuần này") == [("week", "2026-06-22", "2026-06-24")]
    assert P("dthu tuần rồi") == [("week", "2026-06-15", "2026-06-21")]


def test_last_n_includes_today():
    assert P("7 ngày gần nhất doanh thu") == [("last_n", "2026-06-18", "2026-06-24")]


def test_month_quarter_ytd():
    assert P("Doanh thu tháng 3/2025") == [("month", "2025-03-01", "2025-03-31")]
    assert P("doanh thu tháng 2 năm 2024") == [("month", "2024-02-01", "2024-02-29")]
    assert P("quý 1/2026") == [("quarter", "2026-01-01", "2026-03-31")]
    assert P("Tính từ ngày 1/1 đến hết hôm nay, doanh thu năm nay") == [("ytd", "2026-01-01", "2026-06-24")]
    assert P("tháng này") == [("mtd", "2026-06-01", "2026-06-24")]


def test_month_without_year_rulebook_s2():
    # tháng M không ghi năm: năm nay nếu không sau hôm nay, ngược lại năm trước
    assert P("doanh thu tháng 8")[0][1] == "2025-08-01"
    assert P("doanh thu tháng 6")[0][1] == "2026-06-01"


def test_mtd_previous_same_days():
    p = find_periods(normalize("từ đầu tháng này đến hết hôm nay"), date(2026, 3, 31), CAL)[0]
    q = previous_period(p, date(2026, 3, 31))
    assert (q.start, q.end) == (date(2026, 2, 1), date(2026, 2, 28))


def test_range_and_month_lists():
    assert P("giai đoạn từ 05/07/2025 đến hết 11/07/2025") == [("range", "2025-07-05", "2025-07-11")]
    assert [x[1] for x in P("Doanh thu từng tháng của 3 tháng 11/2025, 12/2025, 1/2026")] == \
        ["2025-11-01", "2025-12-01", "2026-01-01"]
    assert [x[1] for x in P("Từ tháng 4/2025 đến tháng 6/2025, doanh thu mỗi tháng")] == \
        ["2025-04-01", "2025-05-01", "2025-06-01"]
    assert [x[1] for x in P("Doanh thu 3 tháng liền nhau kết thúc ở tháng 3/2026")] == \
        ["2026-01-01", "2026-02-01", "2026-03-01"]


def test_asof():
    assert P("Tính đến hết ngày 11/12/2025, có bao nhiêu khách còn nợ") == [("asof", "2025-12-11", "2025-12-11")]
    assert P("tồn kho X cuối ngày hôm nay") == [("asof", "2026-06-24", "2026-06-24")]


@pytest.mark.skipif(not CAL_PATH.exists(), reason="train data not available (calendar.csv)")
def test_tet_rulebook_s2():
    # Tết 2026: 23 tháng Chạp Ất Tỵ .. mùng 10 tháng Giêng Bính Ngọ
    p = tet_period(2026, CAL)
    assert p.start == date(2026, 2, 10) and p.end == date(2026, 2, 26)
    assert P("doanh thu tết năm nay so vs tết năm trc")[0][0] == "tet"
    assert len(P("doanh thu tết năm nay so vs tết năm trc")) == 2
