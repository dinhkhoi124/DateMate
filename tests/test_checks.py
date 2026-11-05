"""Run-time ingest invariants (DECISIONS D11): violations from metrics, and which ops they block."""
from datamate.config import INVARIANTS
from datamate.ingest.checks import Violation, blocking, violations

CLEAN = {
    "orders.n": 1000, "orders.with_date": 1.0, "orders.dup_id": 0.0, "orders.after_data_end": 0.0,
    "orders.share_day_gt12": 0.6, "orders.year_min": 2025, "orders.year_max": 2026,
    "status.done": 0.97, "status.cancelled": 0.02, "status.returned": 0.01, "status.unknown_raw": 0,
    "status.readme_mentions": True, "money.orders_with_rev": 1.0, "money.neg_rev": 0.0,
    "money.median_order": 65000.0, "lines.orphan": 0.0, "orders.without_lines": 0.0,
    "lines.amount_vs_qty_price": 0.99, "lines.qty_base_lt_qty": 0.0,
}


def names(m):
    return {v.name for v in violations({**CLEAN, **m})}


def test_clean_shop_has_no_violation():
    assert violations(CLEAN) == []


def test_no_sales_table_is_not_judged_here():
    assert violations({"orders.n": 0}) == []


def test_day_month_swapped():
    # dd/mm read as mm/dd: only days 1..12 survive
    assert "day_month_swapped" in names({"orders.share_day_gt12": 0.0})


def test_money_read_in_thousands():
    # '1.250.000' read as 1.25
    assert "order_money_scale" in names({"money.median_order": 65.0})


def test_status_not_found():
    assert "status_not_found" in names({"status.done": 1.0, "status.cancelled": 0.0, "status.returned": 0.0})
    # no README mention, but thousands of orders without one cancel/return: still unread
    assert "status_not_found" in names({"status.done": 1.0, "status.readme_mentions": False})
    # a small shop whose README never speaks of statuses: all-completed is plausible
    assert names({"status.done": 1.0, "status.readme_mentions": False, "orders.n": 100}) == set()


def test_broken_lines_also_break_money():
    v = violations({**CLEAN, "lines.amount_vs_qty_price": 0.3})
    assert {x.area for x in v} == {"lines", "money"}


def test_blocking_by_area():
    v = [Violation("debt", "debt_negative", "")]
    assert blocking("debt_total", v) and not blocking("revenue", v)
    assert blocking("revenue", [Violation("orders", "dup_order_id", "")])   # orders blocks every op
    assert blocking(None, [Violation("lines", "orphan_lines", "")])          # unknown op: conservative


def test_thresholds_are_config():
    assert 0 < INVARIANTS.min_share_day_gt12 < 0.5
