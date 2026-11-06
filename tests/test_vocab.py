"""Column-name vocabulary: abbreviations of undocumented columns (eval/synth 'rename')."""
import pytest

from datamate.ingest.vocab import expansions, role_from_name


@pytest.mark.parametrize("name,role", [
    ("SL", "qty"), ("SL quy đổi", "qty_base"), ("ĐG", "price"), ("GG", "discount"), ("CK", "discount"),
    ("Mã HĐ", "order_id"), ("Mã KH", "customer_id"), ("KH", "customer_name"), ("Tên HH", "product_name"),
    ("Mã SP", "product_id"), ("PTTT", "payment_method"), ("Phương thức TT", "payment_method"),
    ("HSD", "expiry"), ("SĐT", "phone"), ("Tổng sau GG", "order_total"), ("Tên NCC", "supplier"),
])
def test_abbreviated_names(name, role):
    assert role_from_name(name) == role


def test_ambiguous_abbreviation_tried_in_order():
    # 'TT': thành tiền, then thanh toán, then trạng thái; the first that is a known name wins
    assert expansions("Phương thức TT")[:2] == ["phuong thuc thanh tien", "phuong thuc thanh toan"]
    assert role_from_name("TT") == "amount"


def test_specific_expansion_beats_generic_learned_name():
    # a README somewhere documented 'KH' as 'tên ...' (generic 'name' -> product_name in an orders table)
    assert role_from_name("KH", {"kh": "name"}) == "customer_name"
    assert role_from_name("KH", {"kh": "customer_id"}) == "customer_id"   # a specific learned role is kept


def test_unknown_names_stay_unknown():
    assert expansions("Ghi chú") == []
    assert role_from_name("Ngày HĐ") is None


def test_ambiguous_abbreviation_order_beats_specificity():
    # 'thanh tiền' (generic amount) is tried before 'thanh toán' (specific paid)
    assert role_from_name("TT", {}) == "amount"
