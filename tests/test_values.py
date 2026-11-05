from datetime import date, datetime

import pytest

from datamate.ingest.values import (
    is_walkin, parse_bool, parse_date, parse_datetime, parse_money, parse_payment, parse_qty, parse_status, parse_time,
)


@pytest.mark.parametrize("raw,val", [
    ("560,000", 560000), ("55.000đ", 55000), ("25k", 25000), ("-53.500đ", -53500), ("25,000.0000", 25000),
    ("6,719.0000", 6719), (42, 42), ("1270000", 1270000), ("100.000.000", 100000000), ("1,2 triệu", 1200000),
    ("0", 0), ("", None), (None, None), (float("nan"), None), ("abc", None), ("8,500", 8500), ("12.5", 12.5),
    ("3,185tr", 3185000), ("1tr250", 1250000), ("1tr5", 1500000), ("9tr", 9000000), ("2k5", 2500), ("25000k", 25000000),
    ("1.5tr", 1500000), ("396k", 396000), ("1 triệu 2", 1200000),
])
def test_money(raw, val):
    assert parse_money(raw) == val


@pytest.mark.parametrize("raw,val", [
    ("2 xe 4m3", (2, "xe 4m3")), ("1 bao 25kg", (1, "bao 25kg")), ("5 chai", (5, "chai")), (3, (3, None)),
    ("-1", (-1, None)), ("1 ly", (1, "ly")), ("", (None, None)),
])
def test_qty(raw, val):
    assert parse_qty(raw) == val


@pytest.mark.parametrize("raw,val", [
    (45717, date(2025, 3, 1)), ("45689", date(2025, 2, 1)), ("1 thg 6 2025", date(2025, 6, 1)),
    ("01/02/2025", date(2025, 2, 1)), ("2025-02-01 06:00:25", date(2025, 2, 1)), (datetime(2026, 1, 3), date(2026, 1, 3)),
    ("29/07/2025", date(2025, 7, 29)), ("bad", None),
])
def test_date(raw, val):
    assert parse_date(raw) == val


def test_date_without_year_uses_sheet_year():
    assert parse_date("05/03", default_year=2026) == date(2026, 3, 5)
    assert parse_date("05/03") is None


def test_datetime_and_time():
    assert parse_datetime("2025-02-01 06:00:25") == datetime(2025, 2, 1, 6, 0, 25)
    assert parse_time("2025-02-01 14:05:00") == (14, 5, 0)
    assert parse_time(0.5) == (12, 0, 0)


@pytest.mark.parametrize("raw,val", [
    ("Đã hoàn thành", "done"), ("completed", "done"), ("xong", "done"), ("đã lấy", "done"), ("Đã giao", "done"),
    ("Thành công", "done"), ("ok", "done"), ("Đã hủy", "cancelled"), ("khách hủy", "cancelled"), ("Huỷ đơn", "cancelled"),
    ("cancel", "cancelled"), ("Đã hoàn hàng", "returned"), ("Đổi trả", "returned"), ("hàng trả về", "returned"),
    ("returned", "returned"), ("Hoàn trả", "returned"), ("Đang giao hàng", "shipping"), ("Đang vận chuyển", "shipping"),
    ("shipping", "shipping"), ("đang gửi", "shipping"), (None, None),
])
def test_status(raw, val):
    assert parse_status(raw) == val


@pytest.mark.parametrize("raw,val", [
    ("cash", "cash"), ("Tiền mặt", "cash"), ("Chuyển khoản", "transfer"), ("transfer", "transfer"),
    ("ewallet", "ewallet"), ("Ví điện tử", "ewallet"), ("cod", "cod"), ("COD", "cod"), ("Ghi nợ", "debt"), ("debt", "debt"),
])
def test_payment(raw, val):
    assert parse_payment(raw) == val


def test_walkin_and_bool():
    for s in ["Khách lẻ", "lẻ", "kl", "khách lẻ", None, ""]:
        assert is_walkin(s)
    assert not is_walkin("Lê Văn A")
    assert parse_bool("TRUE") is True and parse_bool(0) is False and parse_bool("1") is True


@pytest.mark.parametrize("raw,val", [
    ("Hoàn tất", "done"), ("ĐÃ NHẬN HÀNG", "done"), ("Giao hàng thành công", "done"), ("Delivered", "done"),
    ("Đã hủy bỏ", "cancelled"), ("Giao không thành công", "cancelled"), ("Canceled", "cancelled"),
    ("Trả hàng/Hoàn tiền", "returned"), ("Đã nhận hàng trả", "returned"), ("Refunded", "returned"),
    ("Chờ lấy hàng", "shipping"), ("CHƯA GIAO", "shipping"), ("Chờ xác nhận", "shipping"), ("In transit", "shipping"),
    # negated or ambiguous: unknown, so the run-time invariant abstains instead of guessing
    ("Chưa hoàn thành", None), ("Chuyển hoàn", None),
])
def test_status_other_software_labels(raw, val):
    assert parse_status(raw) == val
