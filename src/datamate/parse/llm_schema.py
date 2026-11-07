"""LLM parser contract (DECISIONS D14): the model picks an op, filter mentions and a few small arguments; everything
else stays deterministic.

- Periods are never produced by the model: they come from timeparse (same list the rule parser gets), so dates and
  ranges cannot be hallucinated.
- Entity filters are mentions copied from the question; they are linked to catalog keys here with the same
  Dictionary the rule parser uses. A mention that does not link to exactly one key becomes '?…' and the executor
  abstains (dispatch._check_filters).
- `signature()` is the comparable form used by the ensemble: two intents agree iff their signatures are equal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from datamate.normalize.text import norm_name
from datamate.parse.intent import Intent
from datamate.timeparse.periods import Period

METRICS = ["revenue", "orders", "qty"]
CHANNELS = ["zalo", "phone", "store", "facebook", "marketplace"]
PAYS = ["cash", "transfer", "cod", "ewallet", "card", "debt"]
SHIFTS = ["morning", "afternoon"]
GROUPS = ["than thiet", "si", "vip", "quen", "le", "moi"]
CAPS = ["has_gender", "has_address"]


@dataclass(frozen=True)
class OpSpec:
    desc: str                    # one line, Vietnamese, shown in the prompt
    args: tuple[str, ...] = ()   # argument names the model may fill (see ARG_SCHEMA)
    atype: str = "number"


# Same op names as execute/dispatch.HANDLERS. Order = prompt order (roughly by frequency, rule parser order).
OPS: dict[str, OpSpec] = {
    "revenue": OpSpec("doanh thu (tiền bán được) trong kỳ"),
    "order_count": OpSpec("số đơn hàng trong kỳ"),
    "qty": OpSpec("số lượng hàng bán ra (ly, chai, kg…) trong kỳ", ("unit",)),
    "aov": OpSpec("giá trị trung bình một đơn"),
    "share": OpSpec("tỉ trọng % doanh thu của một nhóm/mặt hàng/kênh trên tổng doanh thu", atype="percent"),
    "growth": OpSpec("% tăng/giảm của một chỉ số so với kỳ khác (kỳ trước, cùng kỳ năm trước, hoặc hai kỳ nêu rõ)",
                     ("metric", "compare"), "percent"),
    "cancel_count": OpSpec("số đơn bị huỷ"),
    "cancel_rate": OpSpec("tỉ lệ % đơn bị huỷ", atype="percent"),
    "shift_compare": OpSpec("so sánh % ca chiều với ca sáng", ("metric",), "percent"),
    "argmax_day": OpSpec("ngày nào có chỉ số cao nhất", ("metric",), "date"),
    "argmax_month": OpSpec("tháng nào có chỉ số cao nhất", ("metric",), "date"),
    "monotonic": OpSpec("có/không: chỉ số tăng (hoặc giảm) liên tục qua các tháng nêu ra", ("metric", "dir"), "boolean"),
    "revenue_series": OpSpec("danh sách doanh thu từng tháng nêu ra", atype="list"),
    "revenue_by": OpSpec("danh sách doanh thu theo từng kênh hoặc từng hình thức thanh toán nêu ra",
                         ("dim", "values"), "list"),
    "top_products": OpSpec("mặt hàng bán chạy nhất / lãi nhất (top k)", ("k", "by_product")),
    "top_customers": OpSpec("khách mua nhiều tiền nhất / nhiều đơn nhất (top k)", ("k", "by_customer")),
    "top_staff": OpSpec("nhân viên bán được doanh thu cao nhất", atype="string"),
    "days_revenue_over": OpSpec("số ngày có doanh thu từ/trên X đồng", ("x", "inclusive")),
    "orders_revenue_over": OpSpec("số đơn có giá trị từ/trên X đồng", ("x", "inclusive")),
    "items_in_n_orders": OpSpec("số mặt hàng xuất hiện trong ít nhất N đơn", ("n", "inclusive")),
    "customers_min_orders": OpSpec("số khách mua ít nhất N đơn", ("n", "inclusive")),
    "customers_min_orders_pct": OpSpec("% khách mua ít nhất N đơn", ("n", "inclusive"), "percent"),
    "dead_count": OpSpec("số mặt hàng ế (không bán được) trong kỳ"),
    "unit_price": OpSpec("giá bán một mặt hàng vào một ngày"),
    "price_effect": OpSpec("hiệu ứng tăng giá: % thay đổi N ngày sau so với N ngày trước khi đổi giá",
                           ("n", "new_price"), "percent"),
    "avg_import_price": OpSpec("giá nhập bình quân của một mặt hàng", ("unit",)),
    "import_total": OpSpec("tổng tiền nhập hàng trong kỳ"),
    "top_supplier": OpSpec("nhà cung cấp nhập nhiều nhất", atype="string"),
    "import_minus_sales": OpSpec("số lượng nhập trừ số lượng bán của một mặt hàng"),
    "stock_qty": OpSpec("tồn kho của một mặt hàng", ("unit",)),
    "stock_big": OpSpec("tồn kho theo đơn vị lớn: [số đơn vị lớn nguyên, số lẻ]", ("big",), "list"),
    "stock_value": OpSpec("giá trị hàng tồn kho theo giá vốn"),
    "stock_out_count": OpSpec("số mặt hàng hết hàng"),
    "debt_total": OpSpec("tổng công nợ khách còn thiếu (có thể chỉ nợ quá N ngày)", ("min_age",)),
    "debt_customers": OpSpec("số khách đang nợ"),
    "debt_table": OpSpec("bảng k khách nợ nhiều nhất", ("k",), "table"),
    "debt_aging": OpSpec("công nợ theo nhóm tuổi nợ (0-7, 8-15… ngày)", atype="list"),
    "debt_collected": OpSpec("tiền nợ đã thu được trong kỳ"),
    "debt_customer_flow": OpSpec("diễn biến nợ của một khách (nợ thêm / trả)", atype="list"),
    "gross_profit": OpSpec("lãi gộp trong kỳ"),
    "net_profit": OpSpec("lãi ròng (lãi gộp trừ chi phí) trong kỳ"),
    "delivery_count": OpSpec("số đợt giao hàng trong kỳ"),
    "tet_vs_before": OpSpec("% thay đổi dịp Tết so với cùng số ngày ngay trước Tết", ("n", "metric"), "percent"),
    "nearest_expiry": OpSpec("hạn dùng gần nhất của lô còn tồn", atype="date"),
    "expiring_lots_count": OpSpec("số lô sắp hết hạn trong N ngày tới", ("days",)),
    "expiring_products": OpSpec("danh sách hàng sắp hết hạn trong N ngày tới", ("days",), "list"),
    "need_cap": OpSpec("câu hỏi cần dữ liệu giới tính hoặc địa chỉ khách", ("cap",)),
    "unanswerable": OpSpec("câu hỏi chủ quan hoặc hỏi điều dữ liệu không ghi (lý do, cảm nhận…)"),
}
NONE_OP = "none"   # the model's way to say "no op fits": always abstain

# rule-parser / executor argument name -> (json schema, model-facing name). by_* keep the enum per op small.
ARG_SCHEMA = {
    "unit": {"type": ["string", "null"]},
    "metric": {"enum": METRICS + [None]},
    "compare": {"enum": ["prev", "yoy", "explicit", None]},
    "dir": {"enum": ["up", "down", None]},
    "dim": {"enum": ["channel", "pay", None]},
    "values": {"type": ["array", "null"], "items": {"enum": CHANNELS + PAYS}},
    "k": {"type": ["integer", "null"]},
    "by_product": {"enum": ["qty", "profit", None]},
    "by_customer": {"enum": ["revenue", "orders", None]},
    "x": {"type": ["number", "null"]},
    "n": {"type": ["integer", "null"]},
    "inclusive": {"type": ["boolean", "null"]},
    "new_price": {"type": ["number", "null"]},
    "big": {"type": ["string", "null"]},
    "min_age": {"type": ["integer", "null"]},
    "days": {"type": ["integer", "null"]},
    "cap": {"enum": CAPS + [None]},
}
MENTION_FILTERS = ("product", "customer", "category", "ingredient", "staff")
FILTER_SCHEMA = {
    **{k: {"type": ["string", "null"]} for k in MENTION_FILTERS},
    "channel": {"enum": CHANNELS + [None]},
    "pay": {"enum": PAYS + [None]},
    "shift": {"enum": SHIFTS + [None]},
    "group": {"enum": GROUPS + [None]},
    "size": {"enum": ["m", "l", None]},
    "rx": {"type": ["boolean", "null"]},
}


def json_schema() -> dict:
    """Schema for guided decoding (vLLM / xgrammar). Flat and permissive on args; op-specific checks in to_intent."""
    return {
        "type": "object",
        "properties": {
            "op": {"enum": list(OPS) + [NONE_OP]},
            "filters": {"type": "object", "properties": FILTER_SCHEMA, "additionalProperties": False},
            "args": {"type": "object", "properties": ARG_SCHEMA, "additionalProperties": False},
        },
        "required": ["op", "filters", "args"],
        "additionalProperties": False,
    }


def _clean(d) -> dict:
    return {k: v for k, v in (d or {}).items() if v is not None and v != [] and v != ""}


def _link_one(dic, mention: str, kind: str):
    """Mention -> the single catalog key it names, else an unresolved marker (executor abstains)."""
    n = norm_name(mention)
    hits = dic.find(n, limit=8) if n else []
    if not hits:
        return "?unlinked " + kind
    if any(h.ambiguous for h in hits) or len({str(h.key) for h in hits}) > 1:
        return "?ambiguous " + kind
    return hits[0].key


def _day(periods: list[Period]):
    day = [p for p in periods if p.kind == "day"]
    return day[0].start if day else None


def to_intent(js: dict, periods: list[Period], lk) -> Intent | None:
    """Model JSON -> Intent, or None when the model said 'none' or produced something outside the contract."""
    if not isinstance(js, dict):
        return None
    op = js.get("op")
    spec = OPS.get(op)
    if spec is None:
        return None
    raw_f = _clean(js.get("filters"))
    raw_a = _clean(js.get("args"))
    flt = {}
    for k, v in raw_f.items():
        if k not in FILTER_SCHEMA:
            return None
        if k in MENTION_FILTERS:
            if not isinstance(v, str):
                return None
            dic = {"product": lk.products, "customer": lk.customers, "category": lk.categories,
                   "ingredient": lk.ingredients, "staff": lk.staff}[k]
            flt[k] = _link_one(dic, v, k)
        else:
            flt[k] = v
    args = {}
    for k, v in raw_a.items():
        if k not in spec.args:
            continue  # args outside the op's contract are ignored, not trusted
        args[{"by_product": "by", "by_customer": "by"}.get(k, k)] = v
    if "unit" in args:
        args["unit"] = norm_name(str(args["unit"])) or None
    if "big" in args:
        args["big"] = norm_name(str(args["big"])) or None
    atype = spec.atype
    # mirror the rule parser's conventions so that equal meaning gives equal Intent
    if op in ("top_products", "top_customers"):
        atype = "list" if args.get("k") else "string"
        args["k"] = args.get("k") or 1
        args.setdefault("by", "qty" if op == "top_products" else "revenue")
    if op == "qty":
        args.setdefault("unit", None)
        if flt.get("product") is None and not flt.get("size") and not flt.get("category") and args.get("unit"):
            flt["base_unit"] = args["unit"]
    if op in ("unit_price", "price_effect"):
        args["day"] = _day(periods)
    if op == "need_cap" and not args.get("cap"):
        return None
    if op == "revenue_by" and (not args.get("dim") or not args.get("values")):
        return None
    return Intent(op=op, periods=list(periods), flt=flt, args=args, atype=atype)


# --------------------------------------------------------------------------- comparison
_ALL_ARGS = {"by", "day"} | {a for s in OPS.values() for a in s.args}
_DEFAULTS = {"metric": "revenue", "inclusive": True}   # executor defaults: absent == default


def signature(it: Intent | None) -> tuple | None:
    """Comparable meaning of an intent: op, answer type, periods, resolved filters, contract args (defaults filled)."""
    if it is None:
        return None
    args = {k: v for k, v in it.args.items() if k in _ALL_ARGS and v is not None}
    for k, d in _DEFAULTS.items():
        if k in (OPS[it.op].args if it.op in OPS else ()):
            args.setdefault(k, d)
    flt = {k: v for k, v in it.flt.items() if v is not None}
    norm = lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)  # noqa: E731
    return (it.op, it.atype, tuple((p.start, p.end) for p in it.periods),
            tuple(sorted((k, norm(v)) for k, v in flt.items())), tuple(sorted((k, norm(v)) for k, v in args.items())))


# --------------------------------------------------------------------------- few-shot targets
def from_intent(it: Intent, names: dict | None = None) -> dict:
    """Intent (from the rule parser) -> model JSON, for few-shot examples. `names` maps filter -> display mention
    (catalog name of the linked key); filters without a name are dropped."""
    names = names or {}
    spec = OPS[it.op]
    filters = {}
    for k, v in it.flt.items():
        if k in MENTION_FILTERS:
            if k in names:
                filters[k] = names[k]
        elif k in FILTER_SCHEMA and v is not None:
            filters[k] = v
    args = {}
    for a in spec.args:
        src = "by" if a in ("by_product", "by_customer") else a
        v = it.args.get(src)
        if a == "k" and it.op in ("top_products", "top_customers") and it.atype == "string":
            v = None
        if v is not None:
            args[a] = v
    return {"op": it.op, "filters": filters, "args": args}
