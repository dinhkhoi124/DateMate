"""Build the SQL agent's few-shot bank (D26) from TRAIN DEV shops only, and check the SQL views on the way.

For every dev question the rule parser understands, a hand-written template (eval/sql_templates.py) turns the
intent into the reply the model should write (notes, TYPE, one SQL query over the views and macros of
agent/sqldb.py). The query runs on the shop's DuckDB database; only replies whose answer matches the gold are kept
(rejection sampling). The per-op match rate is the check of the view layer: a template that matches the gold
everywhere means the views carry the same rulebook semantics as the rule executor.

Never reads holdout shops (CLAUDE.md rule 2) or any test split (rule 1). Output is gitignored (*.jsonl, D14).

    PYTHONPATH=src python eval/build_fewshot_sql.py [--out src/datamate/agent/fewshot_sql.jsonl] [--per-op 12]
"""
from __future__ import annotations

import argparse
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
from datamate.agent.sql_prompt import to_row  # noqa: E402
from datamate.agent.sqldb import QueryError  # noqa: E402
from datamate.config import AgentConfig  # noqa: E402
from datamate.execute.facts import facts_of  # noqa: E402
from datamate.pipeline import Context, analyse  # noqa: E402
from eval.scorer import load_jsonl, score  # noqa: E402
from eval.sql_templates import CTX, Skip, reply, template  # noqa: E402

TRAIN = ROOT / "data" / "train" / "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "src" / "datamate" / "agent" / "fewshot_sql.jsonl"))
    ap.add_argument("--per-op", type=int, default=12, help="examples kept per op (spread over shops)")
    ap.add_argument("--only", nargs="*", help="dev shops to use (default: all dev shops)")
    ap.add_argument("--show-wrong", type=int, default=0, help="print N wrong template answers per op (dev only)")
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
    shown = Counter()
    t0 = time.time()
    by_shop = defaultdict(list)
    for q in qs:
        by_shop[q["business_id"]].append(q)
    for biz in sorted(by_shop):
        st = ctx.store(biz)
        db, _, _, lk = agent._shop(biz, st)     # same database the agent queries at run time
        CTX.clear()
        CTX.update(db=db, f=facts_of(st, ctx.rules))
        for q in by_shop[biz]:
            it, _, _ = analyse(q, ctx)
            if it is None:
                stats["(none)"]["no_intent"] += 1
                continue
            CTX["as_of"] = date.fromisoformat(q["as_of_date"])
            try:
                atype, notes, sql = template(it)
            except Skip:
                stats[it.op]["skip"] += 1
                continue
            except Exception as e:  # noqa: BLE001
                stats[it.op]["template_error"] += 1
                if stats[it.op]["template_error"] <= 2:
                    print(f"  {it.op} template error: {type(e).__name__}: {str(e)[:160]}")
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
            if s == -1 and shown[it.op] < args.show_wrong:
                shown[it.op] += 1
                g = {k: v for k, v in golds[q["id"]].items() if k in ("type", "answer")}
                print(f"  WRONG {it.op} {biz} {q['as_of_date']}: got {row} gold {g}\n    {sql[:300]}")
            if s == 1:
                H = agent._helpers(q, lk)
                kept[it.op].append({"id": q["id"], "business_id": biz, "op": it.op, "question": q["question"],
                                    "context": [as_of_line(H.as_of)] + question_hints(q["question"], H),
                                    "reply": reply(atype, notes, sql)})
        agent.close()                                         # one shop at a time: free its database
        print(f"{biz} done ({time.time() - t0:.0f}s)", file=sys.stderr)

    print(f"\n{'op':26}{'ok':>6}{'wrong':>7}{'sqlerr':>7}{'tmplerr':>8}{'skip':>6}")
    for op, c in sorted(stats.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"{op:26}{c['ok']:6}{c['wrong']:7}{c['sql_error']:7}{c['template_error']:8}{c['skip']:6}")
    tot = Counter()
    for c in stats.values():
        tot.update(c)
    print(f"{'TOTAL':26}{tot['ok']:6}{tot['wrong']:7}{tot['sql_error']:7}{tot['template_error']:8}{tot['skip']:6}"
          f"  no_intent={tot['no_intent']}")

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
