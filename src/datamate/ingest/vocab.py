"""Column role vocabulary.

Step 1: README description -> role (regex on folded description, first hit wins).
Step 2: if a column has no description, column *name* -> role via NAME_SYNONYMS (squashed fold) and
        names learned from documented READMEs (see learn_name_roles).
Generic roles ('name', 'qty', 'price', 'date', 'amount', 'code', 'cost') are specialised per table kind in tables.py.
"""
from __future__ import annotations

import re

from datamate.ingest.values import fold


def squash(s: str) -> str:
    """'Trị giá' / 'TriGia' / 'tri_gia' -> 'trigia'."""
    return re.sub(r"[^a-z0-9]", "", fold(s))


# (role, regex on fold(desc)) — specific before generic
DESC_RULES: list[tuple[str, str]] = [
    ("ignore", r"^cot trong|khong co du lieu"),
    # order lines
    ("qty_base", r"so luong da doi ra don vi co ban|so luong quy doi|quy ra dv le"),
    ("lot_return_qty_base", r"so luong khach tra lai"),
    ("lot_qty_base", r"so luong ban lay tu lo"),
    ("import_qty_base", r"so luong nhap \(don vi co ban\)"),
    ("import_qty", r"^so luong nhap"),
    ("line_no", r"so thu tu dong trong (don|hoa don)"),
    ("import_line_no", r"so thu tu dong trong phieu nhap"),
    ("amount", r"^thanh tien sau giam|^thanh tien cua dong|thanh tien \(da tru"),
    ("import_amount", r"tien nhap cua ca dong"),
    ("unit_price", r"^don gia$|^don gia ban|^gia ban (mot|1) don vi"),
    ("import_price", r"gia nhap mot don vi"),
    ("line_discount", r"giam gia (tren )?dong|^giam gia$"),
    ("cost_unit", r"gia von (mot|1) don vi ban|gia von luc ban"),
    # orders
    ("order_id", r"^ma don( hang)?$|^ma don \(|^ma hoa don"),
    ("order_datetime", r"thoi diem dat don|ngay gio dat"),
    ("order_time", r"^gio dat don"),
    ("order_date", r"^ngay ban|^ngay dat"),
    ("status", r"trang thai don"),
    ("import_status", r"trang thai phieu nhap"),
    ("order_subtotal", r"tong tien hang cua don|^tien hang"),
    ("order_total", r"^tong tien don|so khach phai tra"),
    ("paid", r"^da thanh toan|so khach da tra"),
    ("debt", r"^con no|so khach con no|chua tra"),
    ("channel", r"kenh ban"),
    ("payment_method", r"hinh thuc thanh toan"),
    ("staff_id", r"ma nhan vien|nhan vien ban"),
    ("shipping_fee", r"phi giao|phi ship|phi van chuyen"),
    ("order_discount", r"giam gia (cua |tren )?(ca )?don"),
    # customers
    ("customer_id", r"^ma khach"),
    ("customer_name", r"^ten khach"),
    ("phone", r"so dien thoai"),
    ("customer_group", r"nhom khach"),
    ("address", r"dia chi"),
    ("gender", r"gioi tinh"),
    ("registered_at", r"ngay tao khach|ngay dang ky"),
    # products
    ("product_id", r"^ma hang|^ma san pham|^ma thuoc"),
    ("product_name", r"^ten hang|^ten mat hang|^ten san pham|^ten mon"),
    ("category", r"nhom hang|danh muc"),
    ("base_unit", r"^don vi co ban|^don vi goc"),  # anchored: '(đơn vị cơ bản)' qualifies other roles
    ("big_unit", r"^don vi lon"),
    ("import_unit", r"^don vi nhap"),
    ("unit", r"^don vi tinh"),
    ("factor", r"he so|ti le quy doi"),
    ("price_current", r"gia ban hien tai|gia niem yet"),
    ("cost", r"^gia von"),
    ("stock_qty", r"^ton kho|^ton cuoi"),
    ("product_created_at", r"ngay tao mat hang"),
    ("active_ingredient", r"hoat chat"),
    ("is_rx", r"thuoc ke don|ke don"),
    # imports
    ("receipt_id", r"ma phieu nhap|ma don nhap"),
    ("import_date", r"^ngay nhap"),
    ("supplier_id", r"ma nha cung cap"),
    ("supplier", r"^nha cung cap"),
    ("receipt_total", r"tong tien phieu nhap|tong tien cua ca phieu"),
    ("lot", r"^so lo|^ma lo"),
    ("expiry", r"han su dung|han dung"),
    # payments
    ("payment_id", r"ma phieu thu"),
    ("paid_at", r"thoi diem thu|ngay thu"),
    ("pay_amount", r"^so tien"),
    # price history
    ("effective_date", r"ngay (bat dau )?ap dung"),
    ("new_price", r"gia ban ap dung|gia moi"),
    ("old_price", r"gia truoc lan doi|gia cu"),
    # deliveries / debt terms
    ("delivery_date", r"^ngay giao"),
    ("delivery_no", r"dot giao"),
    ("plate", r"bien so"),
    ("driver", r"tai xe|nguoi giao"),
    ("debt_days", r"so ngay duoc no"),
    ("due_date", r"han thanh toan"),
    # restaurant
    ("table_no", r"^ban$|^so ban|^phong/ban"),
    ("table_area", r"khu vuc"),
    ("guests", r"so khach cua don"),
    # generic (resolved by table kind)
    ("qty", r"^so luong"),
    ("name", r"^ten$|^ten "),
    ("note", r"^ghi chu"),
]
DESC_RULES_C = [(r, re.compile(p)) for r, p in DESC_RULES]


def role_from_desc(desc: str) -> str | None:
    d = fold(desc)
    for role, rx in DESC_RULES_C:
        if rx.search(d):
            return role
    return None


# Column-name synonyms for undocumented files (squashed fold -> role). Ambiguous names map to generic roles.
NAME_SYNONYMS: dict[str, str] = {squash(k): v for k, v in {
    # ids
    "Mã hóa đơn": "order_id", "Mã đơn hàng": "order_id", "Mã đơn": "order_id", "Ma don": "order_id",
    "Mã chứng từ": "order_id", "Mã giao dịch": "order_id", "Số HĐ": "order_id", "Số hóa đơn": "order_id",
    "MaDon": "order_id", "MaHoaDon": "order_id", "MaDonHang": "order_id", "Số chứng từ": "order_id",
    "Mã hàng": "product_id", "Mã hàng hóa": "product_id", "Mã SKU": "product_id", "SKU": "product_id", "ID SP": "product_id",
    "Mã HH": "product_id", "MaSP": "product_id", "MaHang": "product_id", "Mã sản phẩm": "product_id",
    "Mã KH": "customer_id", "Mã khách hàng": "customer_id", "MaKH": "customer_id", "Số KH": "customer_id",
    "Mã nhập hàng": "receipt_id", "Mã phiếu nhập": "receipt_id", "Số phiếu nhập": "receipt_id", "MaPhieuNhap": "receipt_id",
    "Mã NCC": "supplier_id", "MaNCC": "supplier_id", "Nhà cung cấp": "supplier", "Tên NCC": "supplier",
    "Mã phiếu": "payment_id", "Mã phiếu thu": "payment_id", "MaPhieuThu": "payment_id",
    "Lô": "lot", "Số lô": "lot", "Mã lô": "lot", "MaLo": "lot", "SoLo": "lot",
    # names
    "Tên hàng": "product_name", "Tên hàng hóa": "product_name", "Tên sản phẩm": "product_name", "Sản phẩm": "product_name",
    "Hàng hóa": "product_name", "Mặt hàng": "product_name", "TenSP": "product_name", "TenHang": "product_name",
    "Khách hàng": "customer_name", "Tên khách hàng": "customer_name", "Tên khách": "customer_name", "TenKH": "customer_name",
    "Điện thoại": "phone", "Số điện thoại": "phone", "SoDienThoai": "phone", "SDT": "phone",
    "Nhóm khách hàng": "customer_group", "Nhóm khách": "customer_group", "NhomKH": "customer_group",
    "Địa chỉ": "address", "DiaChi": "address", "Giới tính": "gender", "GioiTinh": "gender",
    "Nhóm hàng": "category", "Nhóm hàng(3 Cấp)": "category", "Nhóm hàng hóa": "category", "Danh mục": "category",
    "NhomHang": "category", "Loại sản phẩm": "category",
    # dates
    "Thời gian": "date", "Ngày": "date", "Ngày bán": "order_date", "Ngày đặt hàng": "order_date", "NgayBan": "order_date",
    "Thời gian tạo": "order_datetime", "NgayTao": "order_datetime", "ThoiGianTao": "order_datetime",
    "Ngày nhập": "import_date", "NgayNhap": "import_date", "Hạn sử dụng": "expiry", "HanSuDung": "expiry",
    "Thời gian thu": "paid_at", "Ngày thu": "paid_at", "Ngày áp dụng": "effective_date", "NgayApDung": "effective_date",
    "Ngày giao hàng": "delivery_date",
    # quantities / money
    "Số lượng": "qty", "SL": "qty", "SoLuong": "qty", "Số lượng quy đổi": "qty_base", "SL quy đổi": "qty_base",
    "SLQuyDoi": "qty_base", "Số lượng (ĐVT cơ bản)": "qty_base_generic", "Số lượng trả (ĐVT cơ bản)": "lot_return_qty_base",
    "Đơn giá": "price", "DonGia": "price", "Giá bán": "price", "GiaBan": "price", "Giá": "price",
    "Giảm giá": "discount", "GiamGia": "discount", "Chiết khấu": "discount", "ChietKhau": "discount", "Thành tiền": "amount", "ThanhTien": "amount", "TriGia": "amount",
    "Giá vốn": "cost", "GiaVon": "cost", "GiaVonLucBan": "cost_unit", "Giá vốn lúc bán": "cost_unit",
    "Tổng tiền hàng": "order_subtotal", "TongTienHang": "order_subtotal",
    "Tổng sau giảm giá": "order_total", "Tổng tiền": "order_total", "TongTien": "order_total", "Tổng đơn": "order_total",
    "TongThanhToan": "order_total", "Phải thu": "order_total", "Khách cần trả": "order_total",
    "Khách đã trả": "paid", "Đã trả": "paid", "DaTra": "paid", "Còn nợ": "debt", "ConNo": "debt", "Dư nợ": "debt",
    "Cần trả NCC": "receipt_total", "TongGiaTriNhap": "receipt_total",
    "Giá trị": "pay_amount", "Số tiền thu": "pay_amount", "SoTienThu": "pay_amount", "Số tiền": "pay_amount",
    "Tồn cuối kỳ": "stock_qty", "Tồn cuối": "stock_qty", "Tồn kho": "stock_qty", "TonCuoi": "stock_qty", "Ton kho": "stock_qty",
    "Tồn thực": "stock_qty",
    "Giá mới": "new_price", "Giá cũ": "old_price", "GiaBanLe": "new_price",
    # units
    "ĐVT": "unit", "Đơn vị tính": "unit", "DVT": "unit", "DonViTinh": "unit", "Đơn vị": "unit",
    "Quy đổi": "factor", "Hệ số": "factor", "Tỉ lệ quy đổi": "factor", "HeSo": "factor", "TyLeQuyDoi": "factor",
    "Đơn vị quy đổi": "big_unit", "Đơn vị cơ bản": "base_unit",
    # misc
    "Trạng thái": "status_generic", "TrangThai": "status_generic", "Kênh bán": "channel", "KenhBan": "channel",
    "Phương thức thanh toán": "payment_method", "Hình thức thanh toán": "payment_method", "HinhThucThanhToan": "payment_method",
    "PhuongThuc": "payment_method", "Phương thức": "payment_method",
    "Người bán": "staff_id", "MaNV": "staff_id", "Nhân viên": "staff_id",
    "Ngày sinh": "birthday", "Sinh nhật": "birthday", "Loại hàng hóa": "ignore", "Loại hình khách": "ignore",
    "Ghi chú": "note", "GhiChu": "note", "STT": "line_no", "Hoạt chất": "active_ingredient", "HoatChat": "active_ingredient",
    "Thuốc kê đơn": "is_rx", "Kê đơn": "is_rx", "BanTheoDon": "is_rx",
    "Lần giao": "delivery_no", "Biển số xe": "plate", "Người giao hàng": "driver",
    "Số ngày được nợ": "debt_days", "Hạn thanh toán": "due_date",
    "Phí giao hàng": "shipping_fee", "Phí ship": "shipping_fee", "PhiGiaoHang": "shipping_fee",
}.items()}


def learn_name_roles(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """From documented (column name, description) pairs, learn squashed-name -> role (majority vote)."""
    votes: dict[str, dict[str, int]] = {}
    for name, desc in pairs:
        r = role_from_desc(desc)
        if r and r != "ignore":
            votes.setdefault(squash(name), {}).setdefault(r, 0)
            votes[squash(name)][r] += 1
    return {k: max(v, key=v.get) for k, v in votes.items()}


# Abbreviations used in Vietnamese exports and hand-kept books ('SL', 'ĐG', 'Mã HĐ', 'PTTT'): token -> expansions,
# tried in this order. Only used when the name itself is unknown; an expansion must still hit a known name.
ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "sl": ("so luong",), "dg": ("don gia",), "tt": ("thanh tien", "thanh toan", "trang thai"),
    "ck": ("chiet khau",), "gg": ("giam gia",), "kh": ("khach hang",), "sp": ("san pham",), "hh": ("hang hoa",),
    "nv": ("nhan vien",), "ncc": ("nha cung cap",), "dvt": ("don vi tinh",), "sdt": ("so dien thoai",),
    "dt": ("dien thoai",), "pttt": ("phuong thuc thanh toan",), "httt": ("hinh thuc thanh toan",),
    "hd": ("hoa don",), "hsd": ("han su dung",),
}
_MAX_EXPANSIONS = 27


# roles that tables.py specialises by table kind; a specific role from an expansion beats them
GENERIC_ROLES = frozenset({"name", "qty", "price", "date", "amount", "discount", "cost", "unit", "status_generic",
                           "qty_base_generic"})


def _lookup(k: str, learned: dict[str, str] | None) -> str | None:
    if k in NAME_SYNONYMS:
        return NAME_SYNONYMS[k]
    if learned and k in learned:
        return learned[k]
    return None


def expansions(name: str) -> list[str]:
    """'SL quy đổi' -> ['so luong quy doi']; 'Phương thức TT' -> ['phuong thuc thanh tien', '... thanh toan', ...]."""
    toks = [t for t in re.split(r"[^a-z0-9]+", fold(name)) if t]
    if not any(t in ABBREVIATIONS for t in toks):
        return []
    out = [""]
    for t in toks:
        out = [f"{o} {e}".strip() for o in out for e in ABBREVIATIONS.get(t, (t,))][:_MAX_EXPANSIONS]
    return out


def role_from_name(name: str, learned: dict[str, str] | None = None) -> str | None:
    """Curated synonym > specific learned name > first abbreviation expansion (in ABBREVIATIONS order) > generic
    learned name. 'KH' learned as a generic 'tên' still reads as khách hàng; 'TT' stays thành tiền before thanh toán."""
    k = squash(name)
    if k in NAME_SYNONYMS:
        return NAME_SYNONYMS[k]
    lr = learned.get(k) if learned else None
    if lr is not None and lr not in GENERIC_ROLES:
        return lr
    for e in expansions(name):
        r = _lookup(squash(e), learned)
        if r is not None:
            return r
    return lr
