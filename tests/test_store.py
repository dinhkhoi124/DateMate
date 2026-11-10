"""Integration test of ingest on a tiny synthetic shop (no competition data)."""
from datetime import date

import pandas as pd

from datamate.ingest.store import Store

README = """# Dữ liệu biz_test

Sổ bán hàng ghi tay chuyển sang Excel, mỗi tháng một sheet.

## File
- `SoBanHang.xlsx` (xlsx)
- `DanhMuc.xlsx` (xlsx)

## Cột trong `SoBanHang.xlsx`
- **Mã HĐ**: mã đơn
- **Ngày**: ngày bán
- **Khách**: tên khách (Khách lẻ nếu không lưu)
- **Hàng**: tên hàng
- **SL**: số lượng
- **ĐVT**: đơn vị tính
- **Đơn giá**: đơn giá
- **Thành tiền**: thành tiền sau giảm
- **TT**: trạng thái đơn

## Cột trong `DanhMuc.xlsx`
- **Tên món**: tên hàng (file chỉ có mặt hàng có đơn vị lớn)
- **Đơn vị gốc**: đơn vị cơ bản
- **ĐV nhập**: đơn vị lớn (thùng, hộp, tấn...)
- **Hệ số**: hệ số: số đơn vị cơ bản trong 1 đơn vị lớn

## Quy ước
- Dữ liệu có đến hết 28/02/2026.
- **Trả hàng**: hàng trả ghi số lượng âm, cùng mã đơn với đơn đã bán.
"""

COLS = ["Mã HĐ", "Ngày", "Khách", "Hàng", "SL", "ĐVT", "Đơn giá", "Thành tiền", "TT"]


def _write_shop(tmp_path):
    shop = tmp_path / "biz_test"
    shop.mkdir()
    (shop / "README.md").write_text(README, encoding="utf8")
    t01 = [
        ["SỔ BÁN HÀNG THÁNG 1", None, None, None, None, None, None, None, None],  # title row
        COLS,
        ["HD1", "05/01", "Anh Ba", "Bia lon", "1 thùng", "thùng", "240.000đ", "240k", "xong"],
        [None, None, None, "Bia lon", 2, "lon", "10,000", "20,000", None],          # continuation row
        ["HD2", "05/01", "Khách lẻ", "Bia lon", 2, "lon", "10,000", "20,000", "xong"],
        ["HD2", "05/01", "Khách lẻ", "Bia lon", 2, "lon", "10,000", "20,000", "xong"],  # duplicate in same order
        [None, None, None, "Bia lon", 2, "lon", "10,000", "20,000", None],          # same cells, but HD2 continuation
        ["HD3", "06/01", "Chị Tư", "Gạo", 1, "bao", "1tr250", "1tr250", "đã hủy"],
        ["TỔNG NGÀY 06/01", None, None, None, None, None, None, "1.530.000", None],   # subtotal row
    ]
    t02 = [
        COLS,
        ["HD4", "01/02", "Anh Ba", "Bia lon", 3, "lon", "10,000", "30,000", "xong"],
        ["HD4", "03/02", "Anh Ba", "Bia lon", -1, "lon", "10,000", "-10,000", "Đổi trả"],
    ]
    with pd.ExcelWriter(shop / "SoBanHang.xlsx") as w:
        pd.DataFrame(t01).to_excel(w, sheet_name="Thang 01 2026", header=False, index=False)
        pd.DataFrame(t02).to_excel(w, sheet_name="Thang 02 2026", header=False, index=False)
    pd.DataFrame([["Tên món", "Đơn vị gốc", "ĐV nhập", "Hệ số"], ["Bia lon", "lon", "thùng", 24]]).to_excel(
        shop / "DanhMuc.xlsx", header=False, index=False)
    return shop


def test_store_rules(tmp_path):
    st = Store.build(_write_shop(tmp_path), use_cache=False)
    O = st.orders.set_index("order_id")
    L = st.lines
    assert st.caps["data_end"] == date(2026, 2, 28)
    # sheet year applied to 'dd/mm' dates
    assert O.loc["HD1", "order_date"] == date(2026, 1, 5)
    # subtotal row dropped, duplicate inside HD2 dropped once, continuation of HD2 kept
    assert not L["product_name"].astype(str).str.contains("TỔNG").any()
    assert (L["order_id"] == "HD2").sum() == 2
    assert (L["order_id"] == "HD1").sum() == 2
    # continuation row inherits order fields
    assert O.loc["HD1", "customer_name"] == "Anh Ba"
    # unit conversion: 1 thùng = 24 lon
    hd1 = L[L["order_id"] == "HD1"].sort_values("qty_base")
    assert hd1["qty_base"].tolist() == [2, 24]
    # money formats
    assert O.loc["HD1", "subtotal"] == 260000
    assert O.loc["HD3", "subtotal"] == 1250000
    # statuses: cancel, return via negative qty / status
    assert O.loc["HD3", "status"] == "cancelled"
    assert O.loc["HD4", "status"] == "returned"
    assert O.loc["HD1", "status"] == "done"
    assert bool(O.loc["HD2", "is_walkin"])


def _fresh_prune_state(monkeypatch, cache):
    import datamate.ingest.store as S
    monkeypatch.setenv("DATAMATE_CACHE", str(cache))
    monkeypatch.setattr(S, "_pruned", set())
    return S


def test_cache_prune_removes_only_other_generations(tmp_path, monkeypatch):
    """Stale entries (other ingest code, or the layout before generations) go; current ones and other files stay."""
    S = _fresh_prune_state(monkeypatch, tmp_path / "cache")
    d = tmp_path / "cache" / "store"
    d.mkdir(parents=True)
    gen = S._cache_gen()
    other = "0" * 8 if gen != "0" * 8 else "1" * 8
    h = "0123456789abcdef"
    keep = [f"biz_101_{gen}_{h}.pkl", f"dev_001_{gen}_{h}.pkl", f"biz_12345678_{gen}_{h}.pkl", "notes.txt",
            "biz_101.pkl"]
    drop = [f"biz_101_{other}_{h}.pkl", f"biz_101_{h}.pkl", f"dev_001_{h}.pkl"]
    for n in keep + drop:
        (d / n).write_bytes(b"x")
    assert S._prune_stale_cache(d) == len(drop)
    assert sorted(p.name for p in d.iterdir()) == sorted(keep)
    (d / drop[0]).write_bytes(b"x")
    assert S._prune_stale_cache(d) == 0          # once per process and directory


def test_cache_roundtrip_and_prune_on_build(tmp_path, monkeypatch):
    S = _fresh_prune_state(monkeypatch, tmp_path / "cache")
    shop = _write_shop(tmp_path)
    d = tmp_path / "cache" / "store"
    d.mkdir(parents=True)
    (d / "biz_test_0123456789abcdef.pkl").write_bytes(b"stale")
    st1 = Store.build(shop)
    files = [p.name for p in d.iterdir()]
    assert files == [S._cache_path(shop).name] and f"_{S._cache_gen()}_" in files[0]
    st2 = Store.build(shop)                       # served from cache
    pd.testing.assert_frame_equal(st1.orders, st2.orders)
    pd.testing.assert_frame_equal(st1.lines, st2.lines)
    (d / files[0]).write_bytes(b"half-written")   # unreadable entry: rebuild instead of crashing
    pd.testing.assert_frame_equal(Store.build(shop).orders, st1.orders)
