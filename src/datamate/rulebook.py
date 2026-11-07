"""Run-time rulebook diff (DECISIONS D18, docs/RULEBOOK_PARAMS.md).

The split's rulebook.md is cut into rule rows (each table row; the prose of each section is one row) and compared with
`rulebook_index.json`, built from the TRAIN rulebook by analysis/build_rulebook_index.py (hashes and row ids only, no
rulebook text in the repo):

- same normalized text           -> unchanged rule
- same template (numbers and weekday names masked) and the row is a parameter row -> parameter re-read (shift hours,
  Tết window, week start), see `PARAMS`
- otherwise changed, or a train row missing -> the row's SCOPE is blocked: intents touching it abstain
- a row whose key the train rulebook does not have -> its phrases that occur nowhere in the train rulebook become
  keywords; a question containing one abstains
- global rows (§0, §6 intro, §7, §8, §9) are only reported: blocking them would mean abstaining on everything.

Everything is derived from the train rulebook and the split's own rulebook; nothing is tuned on public text.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

from unidecode import unidecode

from datamate.config import DEFAULT, Rules

INDEX_PATH = Path(__file__).with_name("rulebook_index.json")

# --------------------------------------------------------------------------- scopes (docs/RULEBOOK_PARAMS.md)
# A scope entry is an op name, "[filter]" (any intent with that filter set) or "period:<kind>" (timeparse kinds).
_VALID_SALES = ["revenue", "aov", "share", "growth", "revenue_series", "revenue_by", "argmax_day", "argmax_month",
                "monotonic", "top_products", "top_customers", "top_staff", "days_revenue_over", "orders_revenue_over",
                "qty", "order_count", "gross_profit", "net_profit", "items_in_n_orders", "customers_min_orders",
                "customers_min_orders_pct", "dead_count", "import_minus_sales", "price_effect", "tet_vs_before",
                "shift_compare"]
_REVENUE = ["revenue", "aov", "share", "growth", "revenue_series", "revenue_by", "argmax_day", "argmax_month",
            "monotonic", "top_customers", "top_staff", "days_revenue_over", "orders_revenue_over", "gross_profit",
            "net_profit", "tet_vs_before", "shift_compare"]
_DEBT = ["debt_total", "debt_customers", "debt_table", "debt_aging", "debt_collected", "debt_customer_flow"]
_STOCK = ["stock_qty", "stock_big", "stock_value", "stock_out_count"]
_THRESHOLD = ["days_revenue_over", "orders_revenue_over", "items_in_n_orders", "customers_min_orders",
              "customers_min_orders_pct", "debt_total"]
GLOBAL = "global"   # report only

SCOPES: dict[str, list[str] | str] = {
    # §1 orders and revenue
    "§1.1": _VALID_SALES + _STOCK,           # rulebook §1: đơn hợp lệ (không huỷ, không trả; trả một phần bỏ cả đơn)
    "§1.2": _REVENUE,                        # rulebook §1: "bán/thu được bao nhiêu tiền" = doanh thu
    "§1.3": _VALID_SALES + ["cancel_count", "cancel_rate"],  # rulebook §1: kỳ theo ngày đặt
    "§1.4": ["cancel_count", "cancel_rate"],  # rulebook §1: huỷ = trạng thái huỷ, trả ≠ huỷ
    "§1.5": ["order_count", "items_in_n_orders", "customers_min_orders", "customers_min_orders_pct", "top_customers"],
    "§1.6": _REVENUE,                        # rulebook §1: doanh thu đơn = Σ thành tiền dòng
    "§1.7": ["revenue_by", "share", "[pay]", "[channel]"],
    "§1.8": ["share", "[group]"],
    "§1.9": ["cancel_rate"],
    "§1.10": ["growth", "tet_vs_before", "price_effect", "shift_compare"],
    "§1.11": ["growth"],
    # §2 time (scoped by timeparse period kind)
    "§2.1": ["period:day"],
    "§2.2": ["period:asof"] + _STOCK + _DEBT,
    "§2.3": ["period:range"],
    "§2.4": ["period:last_n"],
    "§2.5": _THRESHOLD,
    "§2.6": ["period:week"],
    "§2.7": ["period:month"],
    "§2.8": ["period:mtd", "period:mtd_prev"],
    "§2.9": ["period:ytd"],
    "§2.10": ["period:month", "period:quarter"],
    "§2.11": ["period:tet", "tet_vs_before"],
    "§2.12": ["[shift]", "shift_compare"],
    # §3 ranking
    "§3.1": ["top_products", "qty", "items_in_n_orders", "dead_count"],
    "§3.2": ["[category]"],
    "§3.3": ["top_products"],
    "§3.4": ["qty"],
    "§3.5": ["top_products"],
    "§3.6": ["top_customers", "customers_min_orders", "customers_min_orders_pct", "debt_table", "debt_customers",
             "debt_customer_flow"],
    "§3.7": ["top_staff"],
    "§3.8": ["aov"],
    # §4 debt
    "§4.1": _DEBT, "§4.2": _DEBT, "§4.3": ["debt_total", "debt_table"], "§4.4": _DEBT,
    "§4.5": ["debt_aging", "debt_total"], "§4.6": ["debt_collected", "debt_customer_flow"],
    # §5 stock, imports, cost
    "§5.1": ["import_total", "top_supplier"],
    "§5.2": ["import_minus_sales"],
    "§5.3": ["avg_import_price"],
    "§5.4": ["stock_big"],
    "§5.5": ["stock_value"],
    "§5.6": ["stock_out_count"],
    "§5.7": ["dead_count"],
    "§5.8": ["unit_price"],
    "§5.9": ["price_effect"],
    "§5.10": ["gross_profit", "net_profit", "top_products"],
    "§5.11": ["gross_profit", "net_profit", "top_products"],
    "§5.12": ["net_profit"],
    # §6 industries
    "§6[ca phe].1": ["[size]"],
    "§6[nha thuoc].1": ["[rx]"],
    "§6[nha thuoc].2": ["expiring_lots_count", "expiring_products"],
    "§6[nha thuoc].3": ["[ingredient]"],
    "§6[nha thuoc].4": ["nearest_expiry"],
    "§6[vat lieu xay dung].1": ["delivery_count"],
    "§10.1": ["qty", "stock_qty", "stock_big", "import_minus_sales", "avg_import_price", "top_products"],
}
GLOBAL_SECTIONS = {"§7", "§8", "§9"}   # every row of these sections is report-only


def scope_of(rid: str) -> list[str] | str:
    """Prose rows (§m.pN) and every row of §7-§9 are global (report only); unknown ids block nothing."""
    if re.search(r"\.p\d+$", rid):
        return GLOBAL
    if rid in SCOPES:
        return SCOPES[rid]
    return GLOBAL if rid.split(".")[0].split("[")[0] in GLOBAL_SECTIONS else []


# --------------------------------------------------------------------------- parameter rows
_WD = ["thu hai", "thu ba", "thu tu", "thu nam", "thu sau", "thu bay", "chu nhat"]
_WD_RX = re.compile(r"\b(" + "|".join(_WD) + r")\b")


def _hms(h, m):
    return int(h) * 3600 + int(m) * 60


def _p_shifts(t: str, r: Rules) -> Rules | None:
    """rulebook §2: ca sáng / ca chiều theo giờ tạo đơn."""
    a = re.search(r"ca sang (\d{1,2}) (\d{2}) (\d{1,2}) (\d{2})", t)
    b = re.search(r"ca chieu (\d{1,2}) (\d{2}) (\d{1,2}) (\d{2})", t)
    if not (a and b):
        return None
    return replace(r, morning=(_hms(a[1], a[2]), _hms(a[3], a[4]) + 59),
                   afternoon=(_hms(b[1], b[2]), _hms(b[3], b[4]) + 59))


def _p_tet(t: str, r: Rules) -> Rules | None:
    """rulebook §2: Tết từ ngày dương của N tháng Chạp đến hết mùng M tháng Giêng."""
    a = re.search(r"(\d{1,2}) thang chap", t)
    b = re.search(r"mung (\d{1,2}) thang gieng", t)
    if not (a and b):
        return None
    return replace(r, tet_start=(int(a[1]), 12), tet_end=(int(b[1]), 1))


def _p_week(t: str, r: Rules) -> Rules | None:
    """rulebook §2: tuần bắt đầu <thứ>."""
    m = re.search(r"bat dau (" + "|".join(_WD) + r")\b", t)
    return replace(r, week_start=_WD.index(m[1])) if m else None


PARAMS = {"§2.6": _p_week, "§2.11": _p_tet, "§2.12": _p_shifts}


# --------------------------------------------------------------------------- parsing
def fold(s: str) -> str:
    """NFC -> ascii lower, markdown and punctuation to spaces, single spaces."""
    s = unidecode(unicodedata.normalize("NFC", s)).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def template(t: str) -> str:
    return _WD_RX.sub("@", re.sub(r"\d+", "#", t))


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf8")).hexdigest()[:16]


@dataclass
class Row:
    rid: str          # §m.k (train numbering) for indexed rows; for new rows the split's own position
    key: str          # match key: folded first cell, or "prose:<section title>"
    text: str         # folded row text
    phrases: list     # folded key + bold phrases (new-row keywords)
    line: int         # 1-based line in the file (tests mutate rows through it)


_HEAD = re.compile(r"^(#{1,3})\s+(.*)$")


def parse_rows(md: str) -> list[Row]:
    rows: list[Row] = []
    sec_no, sec_title, sub = "0", "", ""
    k = pn = 0
    prose: list[tuple[int, str]] = []

    def flush():
        nonlocal pn
        if prose:
            text = fold(" ".join(p for _, p in prose))
            if text:
                pn += 1
                rid = f"§{sec_no}" + (f"[{sub}]" if sub else "") + f".p{pn}"
                rows.append(Row(rid, f"prose:{sec_title}|{sub}|{pn}", text, [], prose[0][0]))
            prose.clear()

    lines = md.splitlines()
    for i, raw in enumerate(lines):
        s = raw.strip()
        m = _HEAD.match(s)
        if m:
            flush()
            level, title = len(m[1]), m[2]
            if level == 1:
                continue
            if level == 2:
                n = re.match(r"(\d+)\.?\s*(.*)", title)
                sec_no, sec_title = (n[1], fold(n[2])) if n else (sec_no + "+", fold(title))
                sub, k, pn = "", 0, 0
            else:
                sub, k, pn = fold(title), 0, 0
            continue
        if s.startswith("|"):
            flush()
            cells = [c.strip() for c in s.strip("|").split("|")]
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if re.fullmatch(r"[|\s:-]+", s) or re.fullmatch(r"\|?\s*:?-{3,}.*", nxt):
                continue   # separator line, or the header row above it
            k += 1
            rid = f"§{sec_no}" + (f"[{sub}]" if sub else "") + f".{k}"
            bold = re.findall(r"\*\*(.+?)\*\*", s)
            rows.append(Row(rid, fold(cells[0]), fold(" ".join(cells)),
                            [p for p in dict.fromkeys([fold(cells[0])] + [fold(b) for b in bold]) if p], i + 1))
            continue
        if s:
            prose.append((i + 1, s))
    flush()
    return rows


def build_index(md: str) -> dict:
    """Train rulebook -> {"rows": {key_hash: {id, text, tmpl}}, "phrases": [hashes of every folded n-gram]}."""
    rows = parse_rows(md)
    idx = {}
    for r in rows:
        kh = _h(r.key)
        if kh in idx:
            raise ValueError(f"duplicate rule key in the train rulebook: {r.rid} and {idx[kh]['id']}")
        idx[kh] = {"id": r.rid, "text": _h(r.text), "tmpl": _h(template(r.text))}
    toks = fold(md).split()
    grams = {_h(" ".join(toks[i:i + n])) for n in range(1, 5) for i in range(len(toks) - n + 1)}
    return {"rows": dict(sorted(idx.items())), "train_ngrams": sorted(grams)}


# --------------------------------------------------------------------------- diff
@dataclass
class RulebookDiff:
    rules: Rules = DEFAULT
    blocked: dict = field(default_factory=dict)      # scope entry -> sorted row ids
    changed: list = field(default_factory=list)      # row ids changed (any kind)
    missing: list = field(default_factory=list)      # train row ids absent here
    params: list = field(default_factory=list)       # row ids re-read as parameters
    reported: list = field(default_factory=list)     # global rows changed (log only)
    new_rows: int = 0
    unmapped_new_rows: int = 0                       # new rows without a train-unseen phrase
    keywords: list = field(default_factory=list)     # folded phrases from new rows

    def summary(self) -> str:
        return (f"rulebook: {len(self.changed)} changed, {len(self.missing)} missing, {len(self.params)} params "
                f"re-read, {len(self.reported)} global changed (log only), {self.new_rows} new rows "
                f"({len(self.keywords)} keywords, {self.unmapped_new_rows} unmapped), {len(self.blocked)} scopes blocked")

    def block_reason(self, it, folded_question: str) -> str | None:
        """Why this intent/question must abstain (None = fine)."""
        hits = set(self.blocked.get(it.op, []))
        for k, v in (it.flt or {}).items():
            if v is not None:
                hits.update(self.blocked.get(f"[{k}]", []))
        for p in it.periods or []:
            hits.update(self.blocked.get(f"period:{p.kind}", []))
        if hits:
            return "rulebook changed: " + ", ".join(sorted(hits))
        q = f" {folded_question} "
        if any(f" {kw} " in q for kw in self.keywords):
            return "rulebook new rule"
        return None


def _in_train(phrase: str, train_ngrams: set) -> bool:
    """Every window of up to 4 words of the phrase occurs in the train rulebook (index stores 1..4-gram hashes)."""
    t = phrase.split()
    n = min(4, len(t))
    return all(_h(" ".join(t[i:i + n])) in train_ngrams for i in range(len(t) - n + 1))


def load_index(path: Path = INDEX_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def analyse(md: str | None, index: dict | None = None) -> RulebookDiff:
    if not md:
        return RulebookDiff()
    index = index or load_index()
    known = index["rows"]
    train_ngrams = set(index["train_ngrams"])
    d = RulebookDiff()
    rules = DEFAULT
    seen = set()
    blocked: dict[str, set] = {}

    def block(rid):
        sc = scope_of(rid)
        if sc == GLOBAL:
            d.reported.append(rid)
            return
        for s in sc:
            blocked.setdefault(s, set()).add(rid)

    for r in parse_rows(md):
        ent = known.get(_h(r.key))
        if ent is None:
            if r.key.startswith("prose:"):
                d.reported.append(r.rid)   # new prose section: nothing to scope it to
                continue
            d.new_rows += 1
            kws = [p for p in r.phrases if len(p) >= 4 and not p.isdigit() and not _in_train(p, train_ngrams)]
            if kws:
                d.keywords.extend(kws)
            else:
                d.unmapped_new_rows += 1
            continue
        rid = ent["id"]
        seen.add(rid)
        if _h(r.text) == ent["text"]:
            continue
        d.changed.append(rid)
        parse = PARAMS.get(rid)
        if parse is not None and _h(template(r.text)) == ent["tmpl"]:
            new = parse(r.text, rules)
            if new is not None:
                rules = new
                d.params.append(rid)
                continue
        block(rid)
    for ent in known.values():
        if ent["id"] not in seen:
            d.missing.append(ent["id"])
            block(ent["id"])
    d.rules = rules
    d.blocked = {k: sorted(v) for k, v in sorted(blocked.items())}
    d.keywords = sorted(set(d.keywords))
    return d
