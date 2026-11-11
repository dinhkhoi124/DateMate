"""README status legends and the three-state status (known-valid / known-invalid / unknown). Toy shops only.

The phrasings here are deliberately NOT the ones eval/synth/families.py writes, so a parser that only matched the
benchmark's own sentences would fail these tests (BENCHMARK.md §7, anti-leakage)."""
import pandas as pd

from datamate.execute.facts import facts_of
from datamate.ingest.readme import status_legend
from datamate.ingest.store import Store
from datamate.ingest.values import CANCELLED, DONE, RETURNED, SHIPPING, UNKNOWN


def test_code_legend_varied_phrasing():
    m = status_legend("- **Tình trạng**: mã xử lý — A: hoàn thành; B: hủy bỏ; C: khách trả lại; D: đang giao")
    assert m == {"a": DONE, "b": CANCELLED, "c": RETURNED, "d": SHIPPING}
    m = status_legend("Cột Kết quả dùng số: 0 -> hoàn thành, 9 -> huỷ đơn")
    assert m == {"0": DONE, "9": CANCELLED}


def test_quoted_explanations():
    m = status_legend('Ghi chú: “Boom” được hiểu là khách huỷ; "Xả kho" tức là hoàn thành.\n'
                      "Đơn 'Đổi ý' (sau khi nhận) có nghĩa là trả hàng")
    assert m == {"boom": CANCELLED, "xa kho": DONE, "doi y": RETURNED}


def test_legend_ignores_non_status_lines_and_conflicts():
    # unit conversions and money notes are not status legends
    assert status_legend("- Bia: 1 thùng = 24 lon\n- **Giá**: 1 = một nghìn đồng") == {}
    # a value explained two ways is dropped (no guess)
    assert status_legend("Trạng thái: X = hoàn thành\nTrạng thái: X = đã huỷ") == {}
    # a value that already is a status word is left to the vocabulary
    assert status_legend("Trạng thái: 'Đã hủy' = hoàn thành") == {}


README = """# Dữ liệu biz_t

Một file phẳng.

## Cột trong `ban.csv`
- **Mã**: mã đơn
- **Ngày**: ngày bán
- **Hàng**: tên hàng
- **SL**: số lượng
- **Thành tiền**: tiền dòng
- **TT**: trạng thái đơn, ghi bằng ký hiệu: K1 = xong, K2 = hủy
"""


def _shop(tmp_path, statuses, readme=README):
    d = tmp_path / "biz_t"
    d.mkdir()
    (d / "README.md").write_text(readme, encoding="utf8")
    rows = [[f"D{i}", f"2025-06-{i + 1:02d}", "Bia", 1, 10000 * (i + 1), s] for i, s in enumerate(statuses)]
    pd.DataFrame(rows, columns=["Mã", "Ngày", "Hàng", "SL", "Thành tiền", "TT"]).to_csv(d / "ban.csv", index=False)
    return Store.build(d, use_cache=False)


def test_store_applies_readme_legend(tmp_path):
    st = _shop(tmp_path, ["K1", "K2", "K1"])
    assert list(st.orders.sort_values("order_id")["status"]) == [DONE, CANCELLED, DONE]
    assert not [w for w in st.warnings if "status" in w]


def test_unknown_status_is_not_a_valid_order(tmp_path):
    st = _shop(tmp_path, ["K1", "Zzz", "K2"])
    O = st.orders.sort_values("order_id")
    assert list(O["status"]) == [DONE, UNKNOWN, CANCELLED]
    assert any("unrecognized status" in w for w in st.warnings)
    f = facts_of(st)
    assert list(f.O.sort_values("order_id")["valid"]) == [True, False, False]   # unknown never counts as sales


def test_blank_status_still_done(tmp_path):
    st = _shop(tmp_path, ["K1", None, "K1"])
    assert list(st.orders.sort_values("order_id")["status"]) == [DONE, DONE, DONE]
