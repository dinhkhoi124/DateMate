"""Assign column roles and classify each raw table into a kind (orders, lines, products, ...)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from datamate.ingest.infer import infer_role
from datamate.ingest.reader import RawTable
from datamate.ingest.values import parse_number
from datamate.ingest.vocab import role_from_desc, role_from_name, squash

_LEARNED = json.loads((Path(__file__).with_name("learned_names.json")).read_text(encoding="utf8"))

# table/file-name hints (squashed fold), checked in order
NAME_HINTS: list[tuple[str, str]] = [
    ("lot_sales", r"theolo|donlo|chitietdonlo|hoadonchitietlo|donhanglo|banlo"),
    ("lots", r"lohang|danhsachlo"),
    ("import_lines", r"chitietnhap|nhaphangchitiet|donnhaphangchitiet|chitietphieunhap"),
    ("imports", r"nhaphang|phieunhap|donnhap|sonhap"),
    ("payments", r"thanhtoan|phieuthu|soquy"),
    ("price_history", r"lichsugia|banggia"),
    ("returns", r"trahang|phieutra"),
    ("deliveries", r"giaohang|phieugiao|sogiao"),
    ("debt_terms", r"congno|hanno|sohanno"),
    ("drugs", r"thuoc"),
    ("units", r"quydoi|donvitinh|donvisp"),
    ("stock_ledger", r"thekho"),
    ("stock", r"tonkho"),
    ("expenses", r"chiphi"),
    ("customers", r"khachhang"),
    ("staff", r"nhanvien"),
    ("suppliers", r"nhacungcap"),
    ("branches", r"chinhanh"),
    ("lines", r"chitiet|sobanhang|banhang|sochitiet"),
    ("orders", r"hoadon|donhang"),
    ("products", r"sanpham|hanghoa|danhmuc|mathang"),
]


@dataclass
class Table:
    name: str
    kind: str
    df: object                              # pandas DataFrame (raw columns)
    roles: dict[str, str] = field(default_factory=dict)  # role -> column
    sources: list[str] = field(default_factory=list)

    def col(self, role: str):
        c = self.roles.get(role)
        return self.df[c] if c is not None else None

    def has(self, *roles: str) -> bool:
        return all(r in self.roles for r in roles)

    def any(self, *roles: str) -> bool:
        return any(r in self.roles for r in roles)


def _hint(name: str) -> str | None:
    s = squash(name)
    for kind, rx in NAME_HINTS:
        if re.search(rx, s):
            return kind
    return None


def assign_roles(t: RawTable) -> dict[str, str]:
    """column -> role: README description, then exact column name, then name keywords + value profile.

    Inferred roles come last in the dict so that build_table ('first column wins') never lets a guessed column
    take a role that a documented or known-name column already has.
    """
    sure: dict[str, str] = {}
    guessed: dict[str, str] = {}
    for c in t.df.columns:
        if c.startswith("_sheet_") or c.startswith("_col"):
            continue
        r = role_from_desc(t.doc[c]) if c in t.doc else None
        if r is None:
            r = role_from_name(c, _LEARNED)
        if r is not None:
            if r != "ignore":
                sure[c] = r
            continue
        if c in t.doc:   # documented but no rule matched: do not guess against the README
            continue
        r = infer_role(c, t.df[c])   # undocumented column (whole table or just this column)
        if r and r != "ignore":
            guessed[c] = r
    taken = set(sure.values())
    return {**sure, **{c: r for c, r in guessed.items() if r not in taken}}


SALES_SIDE = {"customer_id", "customer_name", "order_total", "paid", "debt", "payment_method", "channel", "staff_id"}
IMPORT_SIDE = {"supplier", "supplier_id", "import_price", "import_qty", "import_qty_base", "import_amount",
               "receipt_total", "import_date", "import_unit"}


def _sales_receipt(R: set, hint: str | None) -> bool:
    """A receipt-like id ('Số phiếu') in a table that only has the customer side of a sale is the order id."""
    return ("receipt_id" in R and "order_id" not in R and len(R & SALES_SIDE) >= 2 and not R & IMPORT_SIDE
            and hint not in ("imports", "import_lines", "lots"))


def classify(t: RawTable, colroles: dict[str, str]) -> str:
    R = set(colroles.values())
    prod = bool(R & {"product_id", "product_name"})
    hint = _hint(t.name)
    if _sales_receipt(R, hint):   # 'Số phiếu' of a sales book: customer/payment columns, no supplier side
        return "lines" if prod or R & {"qty", "qty_base", "unit_price"} else "orders"
    if "receipt_id" in R or hint in ("imports", "import_lines", "lots"):
        if "lot" in R and "expiry" in R and not R & {"import_price", "price", "import_amount"}:
            return "lots"
        if hint == "lots":
            return "lots"
        return "import_lines" if prod else "imports"
    if "lot" in R and "order_id" in R:
        return "lot_sales"
    if hint in ("returns", "stock_ledger", "staff", "suppliers", "branches", "expenses"):
        return hint
    if R & {"payment_id", "paid_at", "pay_amount"} and not prod:
        return "payments"
    # a price list has no sold quantities: 'Giá bán lẻ' next to qty + amount + product is a sales line's unit price
    sold = bool(R & {"qty", "qty_base"}) and bool(R & {"amount", "order_date", "date"})
    if R & {"effective_date", "old_price"} or ("new_price" in R and not (prod and sold)):
        return "price_history"
    if R & {"delivery_no", "delivery_date", "plate", "driver"}:
        return "deliveries"
    if R & {"debt_days", "due_date"}:
        return "debt_terms"
    if R & {"active_ingredient", "is_rx"}:
        return "drugs"
    if R & {"factor", "big_unit"}:
        return "units"
    if "stock_qty" in R:
        return "stock"
    if "order_id" in R:
        return "lines" if prod or R & {"qty", "qty_base", "unit_price"} else "orders"
    if prod and R & {"qty", "amount", "unit_price", "qty_base"} and R & {"order_date", "date", "order_datetime"}:
        return "lines"
    if hint == "customers" or (R & {"customer_id", "phone", "customer_group", "address"} and not prod):
        return "customers"
    if prod or hint == "products":
        return "products"
    return hint or "unknown"


# generic role -> specific role, per table kind
_RESOLVE = {
    "name": {"customers": "customer_name", "staff": "staff_name", "suppliers": "supplier", "*": "product_name"},
    "date": {"imports": "import_date", "import_lines": "import_date", "payments": "paid_at", "returns": "return_date",
             "deliveries": "delivery_date", "expenses": "expense_date", "*": "order_date"},
    "qty": {"import_lines": "import_qty", "deliveries": "delivery_qty", "lot_sales": "lot_qty_base",
            "lots": "lot_qty_base", "returns": "return_qty", "*": "qty"},
    "qty_base": {"deliveries": "delivery_qty_base", "lot_sales": "lot_qty_base", "lots": "lot_qty_base",
                 "import_lines": "import_qty_base", "returns": "return_qty_base", "*": "qty_base"},
    "qty_base_generic": {"lots": "lot_qty_base", "import_lines": "import_qty_base", "*": "lot_qty_base"},
    "price": {"import_lines": "import_price", "products": "price_current", "units": "unit_price", "*": "unit_price"},
    "new_price": {"lines": "unit_price", "*": "new_price"},
    "unit_price": {"import_lines": "import_price", "products": "price_current", "*": "unit_price"},
    "cost": {"lines": "cost_unit", "*": "cost"},
    "discount": {"orders": "order_discount", "*": "line_discount"},
    "line_discount": {"orders": "order_discount", "*": "line_discount"},
    "amount": {"import_lines": "import_amount", "expenses": "expense_amount", "returns": "return_amount", "*": "amount"},
    "pay_amount": {"expenses": "expense_amount", "*": "pay_amount"},
    "unit": {"products": "base_unit", "stock": "base_unit", "import_lines": "import_unit", "units": "big_unit",
             "*": "unit"},
    "status_generic": {"imports": "import_status", "import_lines": "import_status", "*": "status"},
    "order_date": {"imports": "import_date", "import_lines": "import_date", "*": "order_date"},
    "order_datetime": {"products": "product_created_at", "customers": "registered_at", "*": "order_datetime"},
    "note": {"imports": "import_note", "import_lines": "import_note", "*": "note"},
    "line_no": {"import_lines": "import_line_no", "*": "line_no"},
}


def build_table(t: RawTable) -> Table:
    colroles = assign_roles(t)
    kind = classify(t, colroles)
    roles: dict[str, str] = {}
    for col, r in colroles.items():
        spec = _RESOLVE.get(r)
        r2 = (spec.get(kind) or spec["*"]) if spec else r
        if r2 not in roles:  # first column wins
            roles[r2] = col
    # customers: a product-name-looking column ('Hạng' folds like 'Hàng') is the membership group
    if kind == "customers" and "product_name" in roles:
        roles.setdefault("customer_group", roles.pop("product_name"))
    # units table: a unit column next to an explicit base unit is the big unit
    if kind == "units" and "big_unit" not in roles and "unit" in roles:
        roles["big_unit"] = roles.pop("unit")
    # sales table without an order id column: a transaction id ('Mã GD', learned as a receipt id elsewhere) is it
    for alt in ("payment_id", "receipt_id"):
        if kind in ("lines", "orders") and "order_id" not in roles and alt in roles:
            roles["order_id"] = roles.pop(alt)
    if kind == "lines":
        _fix_line_money(t, roles)
    return Table(t.name, kind, t.df, roles, t.sources)


def _fix_line_money(t: RawTable, roles: dict[str, str], min_rows: int = 20):
    """Line table: a money column named like an order total ('Tổng tiền') that equals qty x unit price (- line
    discount) on the lines, and varies within multi-line orders, is the line amount (rulebook §1 thành tiền dòng)."""
    col = roles.get("order_total")
    if col is None or "amount" in roles or "qty" not in roles or "unit_price" not in roles:
        return
    df = t.df
    num = lambda c: df[c].map(parse_number)  # noqa: E731
    v, q, p = num(col), num(roles["qty"]), num(roles["unit_price"])
    disc = num(roles["line_discount"]).fillna(0.0) if "line_discount" in roles else 0.0
    ok = v.notna() & q.notna() & p.notna()
    if ok.sum() < min_rows:
        return
    exp = (q * p - disc)[ok]
    match = ((v[ok] - exp).abs() <= (exp.abs() * 0.005).clip(lower=1.0)).mean()
    if match < 0.9:
        return
    oc = roles.get("order_id")
    if oc is not None:   # an order total repeats on every line of the order
        g = pd.DataFrame({"o": df[oc].ffill(), "v": v})[ok]
        multi = g.groupby("o")["v"].agg(["size", "nunique"])
        multi = multi[multi["size"] >= 2]
        if len(multi) and (multi["nunique"] == 1).mean() > 0.5:
            return
    roles["amount"] = roles.pop("order_total")


def describe(tables: list[Table]) -> str:
    return "\n".join(f"  {t.kind:13} {t.name:32} {len(t.df):7}  {sorted(t.roles)}" for t in tables)
