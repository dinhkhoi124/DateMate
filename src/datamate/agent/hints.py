"""Question hints for the SQL agent (D26): facts our deterministic code already reads right, written into the prompt
so the model copies literals instead of computing them.

- periods: every time phrase -> inclusive date range, from timeparse (rulebook §2 of the split: week start, Tết
  window, shifts re-read by D18), so the model writes `order_date BETWEEN DATE 'a' AND DATE 'b'`.
- money: spoken amounts ('3tr7', '500k', '1,2 triệu') -> VND integer (ingest/values.parse_number).
- entities: products / customers / categories / active ingredients named in the question -> exact keys and names
  of the shop's data (link/catalog via agent/helpers), so filters never guess spellings.

Hints never carry a computed answer, only inputs (CLAUDE.md rule 4).
"""
from __future__ import annotations

import re
from datetime import date

from datamate.ingest.values import parse_number

_MONEY = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*(tr|trieu|triệu|k|nghin|nghìn|ngan|ngàn)\s*(\d{1,3})?(?![\w])",
                    re.I)


def money_hints(question: str) -> list[str]:
    """'3tr7' -> '3tr7 = 3700000 đồng' (rulebook §9: money in VND)."""
    out = []
    for m in _MONEY.finditer(question):
        v = parse_number(m.group(0))
        if v is not None and v >= 1000:
            out.append(f"'{m.group(0).strip()}' = {int(round(v))} đồng")
    return out


def period_hints(helpers) -> list[str]:
    """Each time phrase of the question as an inclusive date range (helpers.periods(): timeparse + split rules)."""
    from datamate.timeparse.periods import find_periods  # noqa: PLC0415
    t = helpers._q
    out = []
    for p in find_periods(t, helpers.as_of, helpers._cal, helpers._rules):
        a, b = p.span if p.span else (0, 0)
        name = f"'{t[a:b].strip()}'" if b > a else (p.label or p.kind)
        # a flow period never runs past today (execute/ops.clip): 'tuần này' on a Wednesday ends on that Wednesday
        end = min(p.end, helpers.as_of) if p.kind != "asof" and p.start <= helpers.as_of else p.end
        cut = " (đã cắt đến hôm nay)" if end != p.end else ""
        if p.start == end:
            out.append(f"{name} = ngày {p.start.isoformat()}")
        else:
            out.append(f"{name} = từ {p.start.isoformat()} đến hết {end.isoformat()} (gồm cả hai đầu){cut}")
    return out


def entity_hints(helpers) -> list[str]:
    """Catalog names found in the question with the exact key to filter on."""
    out = []
    for kind, col, fn in (("sản phẩm", "product_key", helpers.find_products),
                          ("khách hàng", "customer_key / customer_id", helpers.find_customers),
                          ("nhóm hàng", "category", helpers.find_categories),
                          ("hoạt chất", "active_ingredient", helpers.find_ingredients)):
        try:
            hits = fn()
        except Exception:  # noqa: BLE001  (a lookup failure only drops the hint)
            hits = []
        seen = set()
        for h in hits:
            if (h["matched"], h["key"]) in seen:
                continue
            seen.add((h["matched"], h["key"]))
            amb = " (mơ hồ: có nhiều khớp)" if h.get("ambiguous") else ""
            out.append(f"{kind} '{h['matched']}' -> {col} = '{h['key']}'{amb}")
    return out


def question_hints(question: str, helpers) -> list[str]:
    """All hints for one question, in a fixed order."""
    return period_hints(helpers) + money_hints(question) + entity_hints(helpers)


def as_of_line(as_of: date) -> str:
    return f"Hôm nay (as_of) = {as_of.isoformat()} ({['thứ Hai', 'thứ Ba', 'thứ Tư', 'thứ Năm', 'thứ Sáu', 'thứ Bảy', 'Chủ nhật'][as_of.weekday()]})"
