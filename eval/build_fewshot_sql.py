"""Build the SQL agent's few-shot bank (D26) from TRAIN DEV shops only, and check the SQL views on the way.

For every dev question the rule parser understands, a hand-written template turns the intent into the reply the
model should write (notes, TYPE, one SQL query over v_sales / v_orders / debt macros). The query runs on the shop's
DuckDB database; only replies whose answer matches the gold are kept (rejection sampling). The per-op match rate is
the check of the view layer: a template that matches the rule executor everywhere means the views carry the same
rulebook semantics.

Never reads holdout shops (CLAUDE.md rule 2) or any test split (rule 1). Output is gitignored (*.jsonl, D14).

    PYTHONPATH=src python eval/build_fewshot_sql.py [--out src/datamate/agent/fewshot_sql.jsonl] [--per-op 12]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from datamate.agent.answer import canon  # noqa: E402
from datamate.agent.hints import as_of_line, question_hints  # noqa: E402
from datamate.agent.sql_agent import SqlAgent  # noqa: E402
from datamate.config import AgentConfig  # noqa: E402
from datamate.agent.sql_prompt import to_row  # noqa: E402
from datamate.agent.sqldb import QueryError  # noqa: E402
from datamate.execute.dispatch import FLOW_KINDS, _cmp_periods  # noqa: E402
from datamate.pipeline import Context, analyse  # noqa: E402
from eval.scorer import load_jsonl, score  # noqa: E402

TRAIN = ROOT / "data" / "train" / "train"


def q_(s) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def d_(d) -> str:
    return f"DATE '{d.isoformat()}'"


class Skip(Exception):
    pass


AS_OF = {}   # filled per question: flow periods are clipped to today like execute/ops.clip and agent/hints.py


def _main(it):
    ps = [p for p in it.periods if p.kind in FLOW_KINDS]
    if not ps or len({(p.start, p.end) for p in ps}) > 1:
        raise Skip("period")
    p = ps[0]
    a = AS_OF.get("d")
    if a is not None and p.start <= a < p.end:
        p = dataclasses.replace(p, end=a)
    return p


def _where(it, p, table: str) -> tuple[str, list[str]]:
    """WHERE clause and notes for the intent's period + filters on v_sales ('s') or v_orders ('o')."""
    w = [f"order_date BETWEEN {d_(p.start)} AND {d_(p.end)}", "is_valid"]
    notes = [f"kỳ: {p.start} đến {p.end}", "chỉ đơn hợp lệ (is_valid, rulebook §1: bỏ đơn huỷ và đơn có trả hàng)"]
    for k, v in (it.flt or {}).items():
        if v is None:
            continue
        if k == "product":
            if table != "s":
                raise Skip("product on orders")
            w.append(f"product_key = {q_(v)}")
            notes.append(f"lọc sản phẩm product_key = {q_(v)}")
        elif k == "products":
            if table != "s":
                raise Skip("products on orders")
            w.append("product_key IN (" + ", ".join(q_(x) for x in v) + ")")
            notes.append("lọc các sản phẩm trong gợi ý")
        elif k == "category":
            if table != "s":
                raise Skip("category on orders")
            w.append(f"strip_accents(lower(category)) = strip_accents(lower({q_(v)}))")
            notes.append(f"lọc nhóm hàng {v}")
        elif k == "customer":
            w.append(f"customer_key = {q_(v)}")
            notes.append(f"lọc khách customer_key = {q_(v)}")
        elif k == "pay":
            w.append(f"payment_method = {q_(v)}")
            notes.append(f"lọc phương thức thanh toán payment_method = {q_(v)}")
        else:
            raise Skip(f"filter {k}")
    return " AND ".join(w), notes


def _line_filter(it) -> bool:
    return any((it.flt or {}).get(k) is not None for k in ("product", "products", "category"))


def template(it, as_of) -> tuple[str, list[str], str]:
    """(TYPE, notes, SQL) for a rule intent, or Skip."""
    op, a = it.op, it.args or {}
    if op == "revenue":
        p = _main(it)
        if _line_filter(it):
            w, n = _where(it, p, "s")
            return "number", n + ["doanh thu = tổng amount của dòng bán (rulebook §1), làm tròn về đồng"], \
                f"SELECT ROUND(SUM(amount)) FROM v_sales WHERE {w}"
        w, n = _where(it, p, "o")
        return "number", n + ["doanh thu = tổng revenue của đơn (rulebook §1), làm tròn về đồng"], \
            f"SELECT ROUND(SUM(revenue)) FROM v_orders WHERE {w}"
    if op == "share":
        p = _main(it)
        if not it.flt:
            raise Skip("share of what")
        src, col = ("v_sales", "amount") if _line_filter(it) else ("v_orders", "revenue")
        w, n = _where(it, p, "s" if src == "v_sales" else "o")
        return "percent", n + ["tỉ trọng = doanh thu phần được lọc / doanh thu cả shop cùng kỳ (rulebook §1)"], \
            (f"SELECT ROUND(100.0 * (SELECT SUM({col}) FROM {src} WHERE {w}) / "
             f"(SELECT SUM(revenue) FROM v_orders WHERE is_valid AND order_date BETWEEN {d_(p.start)} "
             f"AND {d_(p.end)}), 2)")
    if op == "order_count":
        p = _main(it)
        if _line_filter(it):
            w, n = _where(it, p, "s")
            return "number", n + ["đếm số đơn khác nhau có sản phẩm này"], \
                f"SELECT COUNT(DISTINCT order_id) FROM v_sales WHERE {w}"
        w, n = _where(it, p, "o")
        return "number", n + ["đếm số đơn hợp lệ"], f"SELECT COUNT(*) FROM v_orders WHERE {w}"
    if op in ("cancel_count", "cancel_rate"):
        p = _main(it)
        if it.flt and any(v is not None for v in it.flt.values()):
            raise Skip("filtered cancel")
        base = f"FROM v_orders WHERE order_date BETWEEN {d_(p.start)} AND {d_(p.end)}"
        n = [f"kỳ: {p.start} đến {p.end}", "đơn huỷ = status 'cancelled' (rulebook §1: trả hàng không phải huỷ)"]
        if op == "cancel_count":
            return "number", n, f"SELECT COUNT(*) {base} AND status = 'cancelled'"
        return "percent", n + ["tỉ lệ = số đơn huỷ / mọi đơn đặt trong kỳ (mọi trạng thái)"], \
            f"SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'cancelled') / COUNT(*), 2) {base}"
    if op == "qty":
        p = _main(it)
        w, n = _where(it, p, "s")
        if not _line_filter(it):
            w += " AND NOT is_fee"
        return "number", n + ["số lượng quy về đơn vị cơ sở: SUM(qty_base) (rulebook §3)"], \
            f"SELECT ROUND(SUM(qty_base), 2) FROM v_sales WHERE {w}"
    if op == "aov":
        p = _main(it)
        if _line_filter(it):
            raise Skip("aov filter")
        w, n = _where(it, p, "o")
        return "number", n + ["giá trị trung bình đơn = doanh thu / số đơn, làm tròn về đồng (rulebook §3)"], \
            f"SELECT ROUND(SUM(revenue) / COUNT(*)) FROM v_orders WHERE {w}"
    if op == "argmax_day" and a.get("metric", "revenue") == "revenue":
        p = _main(it)
        if _line_filter(it):
            w, n = _where(it, p, "s")
            src, m = "v_sales", "SUM(amount)"
        else:
            w, n = _where(it, p, "o")
            src, m = "v_orders", "SUM(revenue)"
        return "date", n + ["doanh thu theo ngày, lấy ngày cao nhất; hoà thì ngày sớm hơn"], \
            f"SELECT order_date FROM {src} WHERE {w} GROUP BY order_date ORDER BY {m} DESC, order_date LIMIT 1"
    if op == "top_products":
        p = _main(it)
        w, n = _where(it, p, "s")
        by = a.get("by", "qty")
        if by not in ("qty", "revenue"):
            raise Skip("by")
        m = "SUM(qty_base)" if by == "qty" else "SUM(amount)"
        k = int(a.get("k", 1))
        t = "list" if it.atype == "list" else "string"
        crit = "số lượng quy về đơn vị cơ sở (rulebook §3: bán chạy mặc định theo số lượng)" if by == "qty" \
            else "doanh thu"
        return t, n + [f"xếp hạng theo {crit}, bỏ phí dịch vụ; hoà thì theo mã tăng dần; lấy {k}"], \
            (f"SELECT product_name FROM v_sales WHERE {w} AND NOT is_fee GROUP BY product_key, product_name "
             f"HAVING {m} > 0 ORDER BY {m} DESC, product_key LIMIT {k}")
    if op == "top_customers":
        p = _main(it)
        w, n = _where(it, p, "o")
        by = a.get("by", "revenue")
        m = "SUM(revenue)" if by == "revenue" else "COUNT(DISTINCT order_id)"
        k = int(a.get("k", 1))
        t = "list" if it.atype == "list" else "string"
        return t, n + ["khách theo customer_key, bỏ khách lẻ (customer_key NULL) (rulebook §3); hoà theo mã"], \
            (f"SELECT customer_name FROM v_orders WHERE {w} AND customer_key IS NOT NULL "
             f"GROUP BY customer_key, customer_name ORDER BY {m} DESC, customer_key LIMIT {k}")
    if op == "days_revenue_over":
        p = _main(it)
        if _line_filter(it):
            raise Skip("days filter")
        w, n = _where(it, p, "o")
        cmp = ">=" if a.get("inclusive", True) else ">"
        return "number", n + [f"đếm số ngày có tổng doanh thu ngày {cmp} {int(a['x'])}"], \
            (f"SELECT COUNT(*) FROM (SELECT order_date, SUM(revenue) AS r FROM v_orders WHERE {w} "
             f"GROUP BY order_date) WHERE r {cmp} {int(a['x'])}")
    if op == "growth":
        if _line_filter(it):
            raise Skip("growth filter")
        x, y = _cmp_periods(it, as_of)
        m = a.get("metric", "revenue")
        agg = {"revenue": "SUM(revenue)", "orders": "COUNT(*)"}.get(m)
        if agg is None:
            raise Skip("growth metric")
        def part(pp):
            return f"(SELECT {agg} FROM v_orders WHERE is_valid AND order_date BETWEEN {d_(pp.start)} AND {d_(pp.end)})"
        return "percent", [f"kỳ này: {x.start} đến {x.end}; kỳ so sánh: {y.start} đến {y.end}",
                           "% thay đổi = (kỳ này - kỳ trước) / kỳ trước * 100, giảm ra số âm (rulebook §1)"], \
            f"SELECT ROUND(100.0 * ({part(x)} - {part(y)}) / {part(y)}, 2)"
    if op == "debt_customers":
        d = next((p.end for p in it.periods), as_of)
        return "number", [f"tính đến hết ngày {d}", "khách còn nợ = khách có đơn còn nợ (rulebook §4), bỏ khách lẻ"], \
            f"SELECT COUNT(DISTINCT customer_key) FROM debt_at({d_(d)}) WHERE customer_key IS NOT NULL"
    if op == "debt_total" and a.get("min_age") is None:
        d = next((p.end for p in it.periods), as_of)
        return "number", [f"tính đến hết ngày {d}", "tổng nợ = tổng phần còn nợ của các đơn (rulebook §4)"], \
            f"SELECT ROUND(SUM(owe)) FROM debt_at({d_(d)})"
    raise Skip(op)


def reply(atype: str, notes: list[str], sql: str) -> str:
    return "Notes:\n" + "\n".join(f"- {n}" for n in notes) + f"\nTYPE: {atype}\n```sql\n{sql}\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "src" / "datamate" / "agent" / "fewshot_sql.jsonl"))
    ap.add_argument("--per-op", type=int, default=12, help="examples kept per op (spread over shops)")
    ap.add_argument("--only", nargs="*", help="dev shops to use (default: all dev shops)")
    args = ap.parse_args()

    holdout = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    golds = {g["id"]: g for g in load_jsonl(TRAIN / "answers.jsonl")}
    qs = [q for q in load_jsonl(TRAIN / "questions.jsonl") if q["business_id"] not in holdout]
    if args.only:
        qs = [q for q in qs if q["business_id"] in set(args.only)]
    ctx = Context(root=TRAIN)
    agent = SqlAgent(None, AgentConfig(enabled=True, kind="sql"), TRAIN, fewshot=Path("/nonexistent"))
    stats = defaultdict(Counter)
    kept = defaultdict(list)
    t0 = time.time()
    by_shop = defaultdict(list)
    for q in qs:
        by_shop[q["business_id"]].append(q)
    for biz in sorted(by_shop):
        db, _, _, lk = agent._shop(biz, ctx.store(biz))     # same database the agent queries at run time
        for q in by_shop[biz]:
            it, _, _ = analyse(q, ctx)
            if it is None:
                stats["(none)"]["no_intent"] += 1
                continue
            as_of = date.fromisoformat(q["as_of_date"])
            AS_OF["d"] = as_of
            try:
                atype, notes, sql = template(it, as_of)
            except Skip:
                stats[it.op]["skip"] += 1
                continue
            except Exception:  # noqa: BLE001
                stats[it.op]["template_error"] += 1
                continue
            try:
                cols, rows = db.query(sql)
                row = canon(to_row(atype, cols, rows))
            except (QueryError, ValueError) as e:
                stats[it.op]["sql_error"] += 1
                if stats[it.op]["sql_error"] <= 2:
                    print(f"  {it.op} sql error: {str(e)[:160]}")
                continue
            s = score({"id": q["id"], **row}, golds[q["id"]])
            stats[it.op]["ok" if s == 1 else "wrong" if s == -1 else "abst"] += 1
            if s == 1:
                H = agent._helpers(q, lk)
                kept[it.op].append({"id": q["id"], "business_id": biz, "op": it.op, "question": q["question"],
                                    "context": [as_of_line(H.as_of)] + question_hints(q["question"], H),
                                    "reply": reply(atype, notes, sql)})
        agent.close()                                         # one shop at a time: free its database
        print(f"{biz} done ({time.time() - t0:.0f}s)", file=sys.stderr)

    print(f"\n{'op':22}{'ok':>6}{'wrong':>7}{'sqlerr':>7}{'skip':>6}")
    for op, c in sorted(stats.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"{op:22}{c['ok']:6}{c['wrong']:7}{c['sql_error']:7}{c['skip']:6}")
    tot = Counter()
    for c in stats.values():
        tot.update(c)
    print(f"{'TOTAL':22}{tot['ok']:6}{tot['wrong']:7}{tot['sql_error']:7}{tot['skip']:6}  no_intent={tot['no_intent']}")

    out = []
    for op in sorted(kept):          # spread over shops: round-robin by shop, deterministic order by id
        by = defaultdict(list)
        for r in sorted(kept[op], key=lambda r: r["id"]):
            by[r["business_id"]].append(r)
        picked, i = [], 0
        while len(picked) < args.per_op and any(i < len(v) for v in by.values()):
            for b in sorted(by):
                if i < len(by[b]) and len(picked) < args.per_op:
                    picked.append(by[b][i])
            i += 1
        out += picked
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(out)} examples ({len(kept)} ops) -> {args.out}")


if __name__ == "__main__":
    main()
