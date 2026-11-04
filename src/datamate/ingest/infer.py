"""Fallback role inference for undocumented / oddly named columns, from name keywords + value profile.

Used only when neither the README description nor the exact-name vocab gives a role.
"""
from __future__ import annotations

import re

import pandas as pd

from datamate.ingest.values import fold, parse_datetime, parse_number

# value-pattern id detection: role, regex, min match fraction
ID_PATTERNS = [
    ("order_id", re.compile(r"^(ORD|HD|DH|INV|BILL)[-_]?\d{3,}$", re.I)),
    ("product_id", re.compile(r"^(SP|HH|MH|SKU|P)[-_]?\d{3,}$", re.I)),
    ("customer_id", re.compile(r"^(KH|CUS|C)[-_]?\d{3,}$", re.I)),
    ("receipt_id", re.compile(r"^(PN|NH|IMP)[-_]?\d{3,}$", re.I)),
    ("payment_id", re.compile(r"^(PAY|PT|TT)[-_]?\d{3,}$", re.I)),
    ("staff_id", re.compile(r"^(NV|EMP|ST)[-_]?\d{2,}$", re.I)),
    ("supplier_id", re.compile(r"^(NCC|SUP)[-_]?\d{2,}$", re.I)),
    ("lot", re.compile(r"^[A-Z]\d{6,8}$")),
]

# keyword rules on fold(name) with spaces; (role, regex, value_kind) value_kind in {num,date,text,any}
KEYWORDS: list[tuple[str, str, str]] = [
    ("expiry", r"\bhsd\b|han dung|han su dung|expir", "any"),
    ("lot_return_qty_base", r"\b(sl|so luong|slg) tra\b", "num"),
    ("qty_base", r"\b(sl|slg|so luong)\b.*\b(co ban|cb|quy|le|dvt co ban)\b", "num"),
    ("import_qty", r"\b(sl|so luong) nhap\b", "num"),
    ("qty", r"^(sl|slg|s\.l|so luong|qty|soluong)$|^(sl|slg) ", "num"),
    ("factor", r"^bang$|^(?!gia).*(he so|ty le|ti le|quy doi)", "num"),
    ("cost_unit", r"gia von luc ban|gia von dong", "num"),
    ("cost", r"gia von", "num"),
    ("order_subtotal", r"tien hang truoc giam|tong tien hang|tien hang", "num"),
    ("order_total", r"tong tien|tong bill|tong don|phai thu|can tra", "num"),
    ("paid", r"da thu|da tra|khach tra|da thanh toan", "num"),
    ("debt", r"con no|du no|con lai|chua tra", "num"),
    ("discount", r"khuyen mai|giam gia|chiet khau|^ck$|\bck\b|\bkm\b", "num"),
    ("amount", r"thanh tien|tri gia|amount|^tien$", "num"),
    ("price_current", r"gia niem yet|gia hien tai", "num"),
    ("import_price", r"gia nhap", "num"),
    ("price", r"don gia|gia ban|^gia\b|price", "num"),
    ("pay_amount", r"so tien|tien thu", "num"),
    ("order_datetime", r"gio tao|thoi gian tao|gio dat|thoi diem", "date"),
    ("paid_at", r"thoi gian thu|ngay thu|gio thu", "date"),
    ("import_date", r"ngay nhap", "date"),
    ("expense_date", r"ngay chi", "date"),
    ("date", r"ngay|thoi gian|date", "date"),
    ("staff_id", r"ma nv|nhan vien|nguoi ban|thu ngan", "any"),
    ("channel", r"nguon don|kenh|ban qua|channel", "text"),
    ("payment_method", r"phuong thuc|hinh thuc|pttt|httt|\btt\b", "text"),
    ("customer_group", r"nhom khach|hang thanh vien|^hang$|loai khach|nhom kh", "text"),
    ("customer_id", r"ma kh|ma khach|customer id|id kh", "any"),
    ("customer_name", r"ten kh|ten khach|^khach|customer", "text"),
    ("supplier", r"nha cung cap|ncc", "text"),
    ("unit", r"don vi|d\.?vi|dvt|unit", "text"),
    ("category", r"nhom hang|danh muc|phan loai|loai hang|category", "text"),
    ("product_name", r"chi tiet hang|ten hang|ten sp|san pham|product|^mon$|mat hang|hang hoa|^hang$", "text"),
    ("product_id", r"ma (sp|hang|mon|hh)|sku", "any"),
    ("expense_category", r"^muc$|khoan chi|loai chi|noi dung chi", "text"),
    ("note", r"ghi chu|mo ta|dien giai|note", "text"),
    ("status_generic", r"trang thai|tinh trang|status", "text"),
]
KEYWORDS_C = [(r, re.compile(p), k) for r, p, k in KEYWORDS]


def value_profile(s: pd.Series, n: int = 200) -> dict:
    v = s.dropna()
    v = v[v.astype(str).str.strip() != ""].head(n)
    if v.empty:
        return {"empty": True}
    strs = v.astype(str).str.strip()
    prof = {"empty": False}
    for role, rx in ID_PATTERNS:
        prof[role] = float(strs.map(lambda x: bool(rx.match(x))).mean())
    prof["num"] = float(v.map(lambda x: parse_number(x) is not None and not _looks_date(x)).mean())
    prof["date"] = float(v.map(_looks_date).mean())
    prof["date_loose"] = float(v.map(lambda x: parse_datetime(x) is not None).mean())  # incl. Excel serials
    prof["text"] = float(strs.map(lambda x: bool(re.search(r"[A-Za-zÀ-ỹ]{2,}", x))).mean())
    return prof


def _looks_date(x) -> bool:
    if hasattr(x, "year") and hasattr(x, "month"):
        return True
    s = str(x).strip()
    if re.fullmatch(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}.*|\d{4}-\d{2}-\d{2}.*|\d{1,2} thg \d{1,2} \d{4}", s):
        return parse_datetime(s) is not None
    return False


def infer_role(name: str, series: pd.Series) -> str | None:
    prof = value_profile(series)
    if prof.get("empty"):
        return None
    for role, _ in ID_PATTERNS:
        if prof[role] >= 0.9:
            return role
    n = fold(name)
    for role, rx, kind in KEYWORDS_C:
        if rx.search(n):
            if kind == "num" and prof["num"] < 0.8:
                continue
            if kind == "date" and prof["date_loose"] < 0.8:
                continue
            if kind == "text" and prof["text"] < 0.5:
                continue
            return role
    return None
