"""Build a submission from the per-question logs of a GPU agent run (D20), on CPU, without the model.

The Kaggle public notebook runs `datamate.run --agent codegen` in shards and ships each shard's `log.jsonl`
(Docker Guide fields: id, answer, thinking, code, inputs.rule_answer, output). This tool merges the shards,
optionally re-executes the recorded code with the CURRENT sandbox/helpers/tables (`--reexec`, like
eval/reexec_agent.py), re-votes with any policy/m and writes one row per question of `--data` (missing -> abstain).

Prints counts only (CLAUDE.md rule 1: public questions are never read for rules or prompts).

    python eval/resubmit_agent.py out/*/logs/log.jsonl --data data/public_test/public_test \
        --policy agree --m 2 --out eval/runs/public_v2/submission.jsonl [--reexec]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from datamate.agent.agent import combine  # noqa: E402
from eval.calibrate_agent import POLICIES, revote  # noqa: E402
from eval.scorer import load_jsonl  # noqa: E402


def reexec(recs: list[dict], qs: dict, root: Path) -> int:
    from datamate.agent.sandbox import Job, ShopSandbox  # noqa: PLC0415
    from datamate.config import AgentConfig  # noqa: PLC0415
    from datamate.normalize.text import normalize  # noqa: PLC0415
    timeout = AgentConfig.from_env().timeout_s
    boxes: dict[str, ShopSandbox] = {}
    changed = 0
    try:
        for r in recs:
            q = qs[r["id"]]
            sb = boxes.get(q["business_id"])
            if sb is None:
                sb = boxes[q["business_id"]] = ShopSandbox(root, q["business_id"], root / "rulebook.md", timeout)
            new = []
            for code in r.get("code") or []:
                if code is None:
                    new.append({"status": "no_code"})
                    continue
                status, val = sb.run(Job(code, q["as_of_date"], normalize(q["question"]), q["question"]))
                new.append({"status": status, "result": val})
            changed += new != (r.get("output") or [])
            r["output"] = new
    finally:
        for sb in boxes.values():
            sb.close()
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="log.jsonl of each shard (later files win on duplicate ids)")
    ap.add_argument("--data", required=True, help="dir with questions.jsonl (+ data/, rulebook.md for --reexec)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", choices=POLICIES, default="agree")
    ap.add_argument("--m", type=int, default=2, help="min votes for the code-gen answer")
    ap.add_argument("--unanswerable", action="store_true", help="let a code-gen 'unanswerable' majority through")
    ap.add_argument("--reexec", action="store_true", help="re-run recorded code with the current sandbox first")
    args = ap.parse_args()

    root = Path(args.data)
    qs = {q["id"]: q for q in load_jsonl(root / "questions.jsonl")}
    by_id: dict[str, dict] = {}
    for p in args.logs:
        for r in load_jsonl(p):
            if r.get("id") in qs:
                by_id[r["id"]] = r
    recs = [by_id[i] for i in sorted(by_id)]
    print(f"{len(qs)} questions, {len(recs)} with a log record, {len(qs) - len(recs)} missing -> abstain")
    if args.reexec:
        print(f"re-executed: {reexec(recs, qs, root)} questions changed")

    status = Counter(o.get("status") for r in recs for o in (r.get("output") or []))
    tot = sum(status.values()) or 1
    print("samples: " + "  ".join(f"{s}={n} ({100 * n / tot:.0f}%)" for s, n in status.most_common()))

    src, types = Counter(), Counter()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf8", newline="\n") as f:
        for qid in qs:   # submission keeps the order of questions.jsonl
            r = by_id.get(qid)
            if r is None:
                row, s = {"id": qid, "type": "abstain"}, "missing"
            else:
                rule = {"id": qid, **((r.get("inputs") or {}).get("rule_answer") or {"type": "abstain"})}
                row, s = combine(rule, revote(r.get("output") or [], args.m, args.unanswerable), args.policy)
            src[s] += 1
            types[row["type"]] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"policy={args.policy} m={args.m} unanswerable={'on' if args.unanswerable else 'off'}")
    print("source: " + "  ".join(f"{k}={v}" for k, v in src.most_common()))
    print("type:   " + "  ".join(f"{k}={v}" for k, v in types.most_common()))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
