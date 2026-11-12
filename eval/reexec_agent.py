"""Re-run the samples recorded in a GPU trace with the CURRENT code, on CPU (no model call).

Prompt changes need a GPU rerun, but sandbox / helper / table / SQL view fixes do not: this replays each recorded
sample again and writes a new trace with fresh `output`, which eval/calibrate_agent.py then scores.

- code-gen agent (D20, default): each recorded `code` through ShopSandbox.
- SQL agent (D26, --sql): each recorded reply (`thinking`, incl. its repair turn) through SqlAgent.rerun on a freshly
  built DuckDB database (current views, macros, result conversion).

    python eval/reexec_agent.py eval/runs/<run>/trace.jsonl [--sql] [--out new_trace.jsonl] [--data data/train/train]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from datamate.agent.sandbox import Job, ShopSandbox  # noqa: E402
from datamate.config import AgentConfig  # noqa: E402
from datamate.normalize.text import normalize  # noqa: E402
from eval.scorer import load_jsonl  # noqa: E402


def reexec_sql(root: Path, tr: list[dict], out: Path) -> int:
    from datamate.agent.sql_agent import SqlAgent  # noqa: PLC0415
    from datamate.ingest.store import Store  # noqa: PLC0415
    agent = SqlAgent(None, AgentConfig(enabled=True, kind="sql"), root)
    stores, changed = {}, 0
    with open(out, "w", encoding="utf8") as f:
        for t in tr:
            biz = t["business_id"]
            if biz not in stores:
                stores[biz] = Store.build(root / "data" / biz)
            db = agent._shop(biz, stores[biz])[0]
            runs = [agent.rerun(db, r) for r in (t.get("thinking") or [])]
            new = [{"status": st, "result": row if st == "ok" else err} for st, row, _, err in runs]
            changed += new != (t.get("output") or [])
            t["output"], t["code"] = new, [r[2] for r in runs]
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    agent.close()
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", help="default: <trace dir>/trace_reexec.jsonl")
    ap.add_argument("--data", default=str(ROOT / "data" / "train" / "train"))
    ap.add_argument("--sql", action="store_true", help="trace of the text-to-SQL agent (D26)")
    args = ap.parse_args()

    root = Path(args.data)
    qs = {q["id"]: q for q in load_jsonl(root / "questions.jsonl")}
    tr = load_jsonl(args.trace)
    out = Path(args.out) if args.out else Path(args.trace).with_name("trace_reexec.jsonl")
    t0 = time.time()
    if args.sql:
        changed = reexec_sql(root, tr, out)
        print(f"{len(tr)} questions re-executed in {time.time() - t0:.0f}s, {changed} changed -> {out}")
        return
    timeout = AgentConfig.from_env().timeout_s
    boxes: dict[str, ShopSandbox] = {}
    changed = 0
    try:
        with open(out, "w", encoding="utf8") as f:
            for t in tr:
                q = qs[t["id"]]
                sb = boxes.get(t["business_id"])
                if sb is None:
                    sb = boxes[t["business_id"]] = ShopSandbox(root, t["business_id"], root / "rulebook.md", timeout)
                new = []
                for code in t.get("code") or []:
                    if code is None:
                        new.append({"status": "no_code"})
                        continue
                    status, val = sb.run(Job(code, q["as_of_date"], normalize(q["question"]), q["question"]))
                    new.append({"status": status, "result": val})
                changed += new != (t.get("output") or [])
                t["output"] = new
                f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    finally:
        for sb in boxes.values():
            sb.close()
    print(f"{len(tr)} questions re-executed in {time.time() - t0:.0f}s, {changed} changed -> {out}")


if __name__ == "__main__":
    main()
