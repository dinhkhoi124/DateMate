"""Re-run the code samples recorded in a GPU trace with the CURRENT sandbox, tables and helpers (D20), on CPU.

Prompt changes need a GPU rerun, but sandbox / helper / table fixes do not: this replays each recorded `code`
through ShopSandbox again and writes a new trace with fresh `output`, which eval/calibrate_agent.py then scores.

    python eval/reexec_agent.py eval/runs/<run>/trace.jsonl [--out new_trace.jsonl] [--data data/train/train]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", help="default: <trace dir>/trace_reexec.jsonl")
    ap.add_argument("--data", default=str(ROOT / "data" / "train" / "train"))
    args = ap.parse_args()

    root = Path(args.data)
    qs = {q["id"]: q for q in load_jsonl(root / "questions.jsonl")}
    tr = load_jsonl(args.trace)
    out = Path(args.out) if args.out else Path(args.trace).with_name("trace_reexec.jsonl")
    timeout = AgentConfig.from_env().timeout_s
    boxes: dict[str, ShopSandbox] = {}
    t0, changed = time.time(), 0
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
