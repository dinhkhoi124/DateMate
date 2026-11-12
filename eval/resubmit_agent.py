"""Build a submission from the per-question logs of a GPU agent run (D20), on CPU, without the model.

The Kaggle public notebook runs `datamate.run --agent codegen` in shards and ships each shard's `log.jsonl`
(Docker Guide fields: id, answer, thinking, code, inputs.rule_answer, output). This tool merges the shards,
optionally re-executes the recorded code with the CURRENT sandbox/helpers/tables (`--reexec`, like
eval/reexec_agent.py), re-votes and writes one row per question of `--data` (missing -> abstain).

D21 — official vs experiment are separate paths:
- **official** (default): policy/m/unanswerable come ONLY from the committed `AgentConfig()` defaults (chosen on
  train dev/holdout). No CLI override, env ignored. Writes `<out>.meta.json` (commit, config, counts) for the log.
- **--experiment**: `--policy/--m/--unanswerable` allowed, but only on a split that has `answers.jsonl` (train).
  Refuses public: the public score is never used to pick or tune a config.

- **--sql** (D26): logs of `datamate.run --agent sql`. The official threshold is ceil(tau x sql_samples) of the
  committed `AgentConfig()` and the post-vote sanity checks run as at inference; `--reexec` re-runs the recorded
  replies on a fresh DuckDB database (agent/sql_agent.SqlAgent.rerun).

Prints counts only (CLAUDE.md rule 1: public questions are never read for rules or prompts).

    python eval/resubmit_agent.py out/shard*/log.jsonl --data data/public_test/public_test \
        --out eval/runs/public_pub1/submission.jsonl --reexec
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from datamate.agent.agent import combine  # noqa: E402
from datamate.config import AgentConfig  # noqa: E402
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


def reexec_sql(recs: list[dict], qs: dict, root: Path) -> int:
    from datamate.agent.sql_agent import SqlAgent  # noqa: PLC0415
    from datamate.ingest.store import Store  # noqa: PLC0415
    agent = SqlAgent(None, AgentConfig(enabled=True, kind="sql"), root)
    stores, changed = {}, 0
    try:
        for r in recs:
            biz = qs[r["id"]]["business_id"]
            if biz not in stores:
                stores[biz] = Store.build(root / "data" / biz)
            db = agent._shop(biz, stores[biz])[0]
            new = [{"status": st, "result": row if st == "ok" else err}
                   for st, row, _, err in (agent.rerun(db, t) for t in (r.get("thinking") or []))]
            changed += new != (r.get("output") or [])
            r["output"] = new
    finally:
        agent.close()
    return changed


class SqlFinal:
    """Post-vote steps of agent/sql_agent.SqlAgent on recorded outputs: sanity checks, then 'unanswerable' when every
    time window of the question lies outside the shop's data (SqlAgent.outside_data). One Store per shop."""

    def __init__(self, root: Path):
        from datamate.agent.sql_agent import SqlAgent  # noqa: PLC0415
        self.root, self.stores = root, {}
        self.agent = SqlAgent(None, AgentConfig(enabled=True, kind="sql"), root)

    def store(self, biz: str):
        from datamate.ingest.store import Store  # noqa: PLC0415
        if biz not in self.stores:
            try:
                self.stores[biz] = Store.build(self.root / "data" / biz)
            except Exception:  # noqa: BLE001
                self.stores[biz] = None
        return self.stores[biz]

    def close(self):
        self.agent.close()


def sql_vote(r: dict, q: dict, m: int, allow_unans: bool, final: SqlFinal | None = None) -> dict | None:
    """revote() plus the SQL agent's post-vote steps (agent/sql_agent: sanity, outside_data)."""
    from datamate.agent.sql_agent import sanity  # noqa: PLC0415
    from datamate.normalize.text import normalize  # noqa: PLC0415
    row = revote(r.get("output") or [], m, allow_unans)
    if row is not None and sanity(row, normalize(q["question"])):
        row = None
    if row is None and final is not None:
        row, _ = final.agent.outside_data(q, final.store(q["business_id"]), None, {"output": r.get("output")})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="log.jsonl of each shard (later files win on duplicate ids)")
    ap.add_argument("--data", required=True, help="dir with questions.jsonl (+ data/, rulebook.md for --reexec)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiment", action="store_true", help="allow --policy/--m/--unanswerable (train only)")
    ap.add_argument("--policy", choices=POLICIES)
    ap.add_argument("--m", type=int, help="min votes for the code-gen answer")
    ap.add_argument("--unanswerable", action="store_true", help="let a code-gen 'unanswerable' majority through")
    ap.add_argument("--reexec", action="store_true", help="re-run recorded code with the current sandbox first")
    ap.add_argument("--sql", action="store_true", help="logs of the text-to-SQL agent (D26)")
    args = ap.parse_args()

    root = Path(args.data)
    official = AgentConfig()   # committed defaults, deliberately not from_env()
    if args.experiment:
        if not (root / "answers.jsonl").exists():
            ap.error("--experiment only on a split with answers.jsonl (train); public uses the official config (D21)")
        args.policy = args.policy or official.policy
        args.m = args.m or official.min_votes
    else:
        if args.policy or args.m or args.unanswerable:
            ap.error("official run: policy/m/unanswerable come from AgentConfig defaults (D21); use --experiment on train")
        args.policy, args.m, args.unanswerable = official.policy, official.min_votes, official.allow_unanswerable
        if args.sql:   # D26: answer when share >= tau of the k samples
            args.m = math.ceil(official.tau * official.sql_samples - 1e-9)
    qs = {q["id"]: q for q in load_jsonl(root / "questions.jsonl")}
    by_id: dict[str, dict] = {}
    for p in args.logs:
        for r in load_jsonl(p):
            if r.get("id") in qs:
                by_id[r["id"]] = r
    recs = [by_id[i] for i in sorted(by_id)]
    print(f"{len(qs)} questions, {len(recs)} with a log record, {len(qs) - len(recs)} missing -> abstain")
    if args.reexec:
        n = reexec_sql(recs, qs, root) if args.sql else reexec(recs, qs, root)
        print(f"re-executed: {n} questions changed")

    status = Counter(o.get("status") for r in recs for o in (r.get("output") or []))
    tot = sum(status.values()) or 1
    print("samples: " + "  ".join(f"{s}={n} ({100 * n / tot:.0f}%)" for s, n in status.most_common()))

    src, types = Counter(), Counter()
    final = SqlFinal(root) if args.sql else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf8", newline="\n") as f:
        for qid in qs:   # submission keeps the order of questions.jsonl
            r = by_id.get(qid)
            if r is None:
                row, s = {"id": qid, "type": "abstain"}, "missing"
            else:
                rule = {"id": qid, **((r.get("inputs") or {}).get("rule_answer") or {"type": "abstain"})}
                cg = sql_vote(r, qs[qid], args.m, args.unanswerable, final) if args.sql else \
                    revote(r.get("output") or [], args.m, args.unanswerable)
                row, s = combine(rule, cg, args.policy)
            src[s] += 1
            types[row["type"]] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if final is not None:
        final.close()
    mode = "experiment" if args.experiment else "official"
    print(f"{mode}{' sql' if args.sql else ''}: policy={args.policy} m={args.m} "
          f"unanswerable={'on' if args.unanswerable else 'off'}")
    print("source: " + "  ".join(f"{k}={v}" for k, v in src.most_common()))
    print("type:   " + "  ".join(f"{k}={v}" for k, v in types.most_common()))
    if not args.experiment:
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "eval"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = "unknown", True
        meta = {"mode": mode, "agent": "sql" if args.sql else "codegen", "m": args.m, "commit": commit,
                "dirty_src": dirty, "config": asdict(official), "reexec": args.reexec,
                "logs": [str(p) for p in args.logs], "source": dict(src), "type": dict(types)}
        Path(str(out) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf8")
        if dirty:
            print("WARNING: src/ or eval/ has uncommitted changes: commit before submitting (D21)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
