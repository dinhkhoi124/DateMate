"""Regression tests for core rule fixes, on dev shops only (never holdout)."""
from pathlib import Path

import pytest

from datamate.pipeline import Context, analyse

ROOT = Path("data/train/train")
pytestmark = pytest.mark.skipif(not (ROOT / "data").exists(), reason="train data not available")


@pytest.fixture(scope="module")
def ctx():
    return Context(root=ROOT, gate=False)  # parser/executor rules; the gate has its own tests


def ask(ctx, biz, q, as_of):
    return analyse({"id": "t", "business_id": biz, "question": q, "as_of_date": as_of}, ctx)


def test_tet_vs_days_before_single_anchor(ctx):
    # rulebook §2: so với trước Tết = cùng số ngày ngay trước 23 tháng Chạp (one Tết anchor)
    it, _, row = ask(ctx, "biz_106", "Dịp Tết âm lịch 2026 có doanh thu hơn hay kém 17 ngày trước Tết bao nhiêu %?",
                     "2026-09-14")
    assert it.op == "tet_vs_before" and row["type"] == "percent"


def test_out_of_stock_is_not_expiry(ctx):
    # rulebook §5: mặt hàng hết hàng cuối ngày = tồn hết ngày bằng 0 ('het hang' must not hit the expiry branch)
    it, _, row = ask(ctx, "biz_120", "Đến cuối ngày 12/06/2026, shop có mấy mặt hàng đang hết hàng?", "2026-06-12")
    assert it.op == "stock_out_count" and row["type"] == "number"


def test_customer_debt_flow_without_debt_data_is_unanswerable(ctx):
    # rulebook §7: shop không ghi loại thông tin câu hỏi cần -> không trả lời được
    it, _, row = ask(ctx, "biz_112", "tháng này, Bạch Hùng Ngân ghi nợ thêm bn tiền và đã trả nợ được bao nhiêu",
                     "2026-07-31")
    assert it.op == "debt_customer_flow" and row["type"] == "unanswerable"
