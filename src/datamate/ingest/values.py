"""Parsers for messy raw cell values: money, numbers, quantities, dates, statuses, payment methods.

All parsers are total: they return None for blanks/unparseable input instead of raising.
"""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime, timedelta

from unidecode import unidecode

_BLANK = {"", "nan", "none", "null", "nat", "-", "—"}


def is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return isinstance(x, str) and x.strip().lower() in _BLANK


def clean_text(x) -> str | None:
    if is_blank(x):
        return None
    s = unicodedata.normalize("NFC", str(x)).strip()
    return re.sub(r"\s+", " ", s)


def fold(s: str) -> str:
    """lowercase, no diacritics, single spaces — for matching labels, never for output. NaN/None -> ''."""
    if not isinstance(s, str):
        if is_blank(s):
            return ""
        s = str(s)
    return re.sub(r"\s+", " ", unidecode(unicodedata.normalize("NFC", s)).lower()).strip()


# ---------------------------------------------------------------- numbers / money
_NUM_TOKEN = re.compile(r"[-+]?\d[\d.,\s]*")


def parse_number(x) -> float | None:
    """'560,000' '55.000đ' '25k' '1,2 triệu' '25,000.0000' '-53.500đ' 42 -> float."""
    if is_blank(x):
        return None
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    s = fold(str(x))
    neg = s.startswith("-") or s.startswith("(")
    # spoken compound: '1tr250' = 1 250 000, '2k5' = 2 500, '1 trieu 2' = 1 200 000
    m = re.search(r"(\d+)\s*(tr|trieu|k|nghin|ngan)\s*(\d{1,3})(?!\d)\s*(d|vnd|dong)?\s*$", s)
    if m:
        unit = 1e6 if m.group(2) in ("tr", "trieu") else 1e3
        tail = m.group(3)
        v = int(m.group(1)) * unit + int(tail) * unit / (10 ** len(tail))
        return -v if neg else v
    mult = 1.0
    if re.search(r"\d\s*(k|nghin|ngan)\b", s):
        mult = 1e3
    elif re.search(r"\d\s*(tr|trieu)\b", s):
        mult = 1e6
    m = _NUM_TOKEN.search(s)
    if not m:
        return None
    t = m.group(0).replace(" ", "").lstrip("+-")
    if mult > 1 and t.count(",") + t.count(".") == 1:
        t = t.rstrip(".,").replace(",", ".")  # '3,185tr' / '1.5tr': separator before a scale word is decimal
    else:
        t = _normalize_separators(t)
    try:
        v = float(t) * mult
    except ValueError:
        return None
    return -v if neg and v > 0 else v


def _normalize_separators(t: str) -> str:
    """Decide which of '.' / ',' is the decimal separator."""
    t = t.rstrip(".,")
    if "," in t and "." in t:
        # the last one is decimal: 25,000.0000 -> 25000.0000 ; 1.234,5 -> 1234.5
        if t.rfind(".") > t.rfind(","):
            return t.replace(",", "")
        return t.replace(".", "").replace(",", ".")
    for sep in (",", "."):
        if sep in t:
            parts = t.split(sep)
            # thousands grouping: every group after the first has exactly 3 digits
            if len(parts) > 2 or (len(parts[-1]) == 3 and parts[0] not in ("", "0")):
                return t.replace(sep, "")
            return t.replace(sep, ".")
    return t


def parse_money(x) -> float | None:
    return parse_number(x)


_QTY = re.compile(r"^\s*([-+]?\d[\d.,]*)\s*(.*)$")


def parse_qty(x) -> tuple[float | None, str | None]:
    """'2 xe 4m3' -> (2, 'xe 4m3'); '1 bao 25kg' -> (1, 'bao 25kg'); 3 -> (3, None)."""
    if is_blank(x):
        return None, None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x), None
    s = clean_text(x)
    m = _QTY.match(s)
    if not m:
        return None, None
    raw = m.group(1)
    num = float(_normalize_separators(raw.lstrip("+-"))) * (-1 if raw.startswith("-") else 1)
    unit = m.group(2).strip() or None
    return num, unit


def parse_bool(x) -> bool | None:
    if is_blank(x):
        return None
    if isinstance(x, bool):
        return x
    s = fold(str(x))
    if s in {"true", "1", "1.0", "x", "co", "yes", "y", "dung", "ke don"}:
        return True
    if s in {"false", "0", "0.0", "khong", "no", "n", "sai"}:
        return False
    return None


# ---------------------------------------------------------------- dates
_EXCEL_EPOCH = datetime(1899, 12, 30)
_THG = re.compile(r"^(\d{1,2})\s*thg\s*(\d{1,2})\s*,?\s*(\d{4})")
_DMY = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?")
_YMD = re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?")
_DM = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})$")


def parse_datetime(x, default_year: int | None = None) -> datetime | None:
    """Excel serial 45717, '1 thg 6 2025', '01/02/2025', '2025-02-01 06:00:25', datetime, '01/02' (+default_year)."""
    if is_blank(x):
        return None
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime(x.year, x.month, x.day)
    if hasattr(x, "to_pydatetime"):
        return x.to_pydatetime()
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return _from_serial(float(x))
    s = str(x).strip()
    if re.fullmatch(r"\d{5}(\.\d+)?", s):
        return _from_serial(float(s))
    s_f = fold(s)
    m = _THG.match(s_f)
    if m:
        return _mk(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = _YMD.match(s)
    if m:
        return _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)), *_hms(m, 4))
    m = _DMY.match(s)
    if m:
        y = int(m.group(3))
        y = y + 2000 if y < 100 else y
        return _mk(y, int(m.group(2)), int(m.group(1)), *_hms(m, 4))
    m = _DM.match(s)
    if m and default_year:
        return _mk(default_year, int(m.group(2)), int(m.group(1)))
    return None


def parse_date(x, default_year: int | None = None) -> date | None:
    dt = parse_datetime(x, default_year)
    return dt.date() if dt else None


def parse_time(x) -> tuple[int, int, int] | None:
    """'06:00:25' / datetime / '2025-02-01 06:00:25' / Excel fraction -> (h, m, s)."""
    if is_blank(x):
        return None
    if isinstance(x, datetime):
        return x.hour, x.minute, x.second
    if hasattr(x, "hour") and hasattr(x, "minute"):
        return x.hour, x.minute, getattr(x, "second", 0)
    if isinstance(x, float) and 0 <= x < 1:
        sec = round(x * 86400)
        return sec // 3600, sec % 3600 // 60, sec % 60
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(x))
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return None


def _hms(m, i):
    return [int(g) if g else 0 for g in (m.group(i), m.group(i + 1), m.group(i + 2))]


def _from_serial(v: float) -> datetime | None:
    if 20000 < v < 80000:  # 1954..2119
        return _EXCEL_EPOCH + timedelta(days=v)
    return None


def _mk(y, mo, d, h=0, mi=0, s=0) -> datetime | None:
    try:
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


# ---------------------------------------------------------------- categorical
DONE, CANCELLED, RETURNED, SHIPPING = "done", "cancelled", "returned", "shipping"

# matched on fold()ed text; order matters (first hit wins)
_STATUS_RULES = [
    # 'nhận hàng trả' (shop received the returned goods) must not read as 'đã nhận hàng' (customer received them)
    (RETURNED, r"tra hang|tra lai|hoan hang|hoan tra|hoan tien|doi tra|hang tra ve|tra ve|nhan hang tra|return|refund"),
    # \b: 'van chuyen' contains 'huy'. 'Không giao được' = order never delivered -> cancelled (shops that use it have
    # no other cancel label; train cancel rates only match with this reading)
    (CANCELLED, r"\bhuy\b|cancel|void|khong giao duoc|giao khong thanh cong|khong thanh cong|giao that bai|bom hang"),
    # not finished yet: waiting for confirmation / pickup / packing counts as in progress (rulebook: 'đang giao')
    (SHIPPING, r"dang giao|dang van chuyen|dang gui|shipping|delivering|dang xu ly|dang giao dich|cho giao|pending"
               r"|cho xac nhan|da xac nhan|cho lay hang|dang chuan bi|dang dong goi|cho xu ly|chua giao|in transit"
               r"|processing"),
    (DONE, r"hoan thanh|hoan tat|xong|da lay|da giao|da nhan hang|thanh cong|completed|complete|done|^ok$|success"
           r"|da thanh toan|delivered|finished|fulfilled"),
]
_STATUS_RULES = [(k, re.compile(p)) for k, p in _STATUS_RULES]
_NEGATED = re.compile(r"\b(chua|khong)\b")


def parse_status(x) -> str | None:
    """Map raw order status text to one of done/cancelled/returned/shipping (rulebook §8).

    Unknown or ambiguous wording ('Chuyển hoàn') stays None: the Store keeps it as unrecognized and the run-time
    invariant (ingest/checks.py) abstains instead of guessing. A negated label ('chưa hoàn thành') is never done.
    """
    if is_blank(x):
        return None
    s = fold(str(x))
    for k, rx in _STATUS_RULES:
        if rx.search(s):
            if k == DONE and _NEGATED.search(s):
                return None
            return k
    return None


CASH, TRANSFER, EWALLET, COD, DEBT, CARD = "cash", "transfer", "ewallet", "cod", "debt", "card"
_PAY_RULES = [
    (COD, r"\bcod\b|thu ho|giao hang thu tien"),
    (DEBT, r"debt|ghi no|no\b|cong no|tra sau|mua chiu"),
    (EWALLET, r"ewallet|vi dien tu|vi\b|momo|zalopay|vnpay|shopeepay"),
    (TRANSFER, r"transfer|chuyen khoan|\bck\b|bank|ngan hang"),
    (CARD, r"\bcard\b|the\b|pos|visa|master"),
    (CASH, r"cash|tien mat|\btm\b"),
]
_PAY_RULES = [(k, re.compile(p)) for k, p in _PAY_RULES]


def parse_payment(x) -> str | None:
    if is_blank(x):
        return None
    s = fold(str(x))
    for k, rx in _PAY_RULES:
        if rx.search(s):
            return k
    return s


_WALKIN = re.compile(r"^(khach le|le|kl|khach vang lai|vang lai|khach|walk ?in|guest|khach hang le)$")


def is_walkin(name) -> bool:
    """'Khách lẻ' / 'lẻ' / 'kl' / blank -> walk-in customer (rulebook §1, §3)."""
    if is_blank(name):
        return True
    return bool(_WALKIN.match(fold(str(name))))
