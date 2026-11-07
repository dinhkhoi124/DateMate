"""Run pipeline on a train split and score it.

PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split dev --by intent --errors 20
Writes eval/runs/<timestamp>_<split>/{submission.jsonl,report.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from datamate.pipeline import Context, abstain, answer_traced  # noqa: E402
from datamate.run import make_llm  # noqa: E402
from eval.scorer import evaluate, load_jsonl, score  # noqa: E402

TRAIN = ROOT / "data" / "train" / "train"


def intent_of(question: str) -> str:
    try:  # rough survey classifier until parse/ provides real intents
        from analysis.intent_survey import classify
        return classify(question)
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--by", default="intent", choices=["intent", "shop", "type", "source"])
    ap.add_argument("--rulebook", help="use this rulebook.md instead of the split's (synthetic-rulebook tests, D18)")
    ap.add_argument("--errors", type=int, default=0, help="print N wrong answers (never for holdout)")
    ap.add_argument("--only", nargs="*", help="restrict to business_ids (dev only)")
    ap.add_argument("--llm", choices=["off", "vllm", "replay"], help="LLM backend (default: $DATAMATE_LLM or off)")
    ap.add_argument("--llm-model", help="local weights dir")
    ap.add_argument("--llm-replay", help="replay/record jsonl")
    args = ap.parse_args()

    holdout = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    qs = load_jsonl(TRAIN / "questions.jsonl")
    golds = {g["id"]: g for g in load_jsonl(TRAIN / "answers.jsonl")}
    if args.split == "dev":
        qs = [q for q in qs if q["business_id"] not in holdout]
    elif args.split == "holdout":
        qs = [q for q in qs if q["business_id"] in holdout]
    if args.only:
        if args.split == "holdout":
            sys.exit("--only is not allowed on holdout")
        qs = [q for q in qs if q["business_id"] in set(args.only)]

    ctx = Context(root=TRAIN, llm=make_llm(args.llm, args.llm_model, args.llm_replay), rulebook=args.rulebook)
    preds, sources, t0 = [], {}, time.time()
    for q in qs:
        try:
            row, tr = answer_traced(q, ctx)
            preds.append(row)
            sources[q["id"]] = (tr.get("inputs") or {}).get("source", "-")
        except Exception as e:  # report, don't crash
            print(f"ERR {q['id']}: {e!r}", file=sys.stderr)
            preds.append(abstain(q["id"]))
    elapsed = time.time() - t0

    gl = [golds[q["id"]] for q in qs]
    summary = evaluate(preds, gl)
    summary["coverage"] = (summary["correct"] + summary["wrong"]) / summary["n"] if summary["n"] else 0
    summary["seconds"] = round(elapsed, 1)

    pm = {p["id"]: p for p in preds}
    groups = defaultdict(lambda: [0, 0, 0])  # correct, wrong, abstain
    rows = []
    for q in qs:
        g = golds[q["id"]]
        key = {"intent": lambda: intent_of(q["question"]), "shop": lambda: q["business_id"], "type": lambda: g["type"],
               "source": lambda: sources.get(q["id"], "-")}[args.by]()
        s = score(pm.get(q["id"]), g)
        groups[key][{1: 0, -1: 1, 0: 2}[s]] += 1
        if s == -1:
            rows.append((q, pm[q["id"]], g))

    run_dir = ROOT / "eval" / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "submission.jsonl", "w", encoding="utf8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (run_dir / "report.json").write_text(json.dumps({"summary": summary, "by": {k: v for k, v in groups.items()}},
                                                    ensure_ascii=False, indent=1), encoding="utf8")

    print(f"split={args.split} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))
    print(f"\n{'by ' + args.by:26}{'n':>5}{'ok':>5}{'wrong':>6}{'abst':>6}{'PA':>8}")
    for k, (c, w, a) in sorted(groups.items(), key=lambda kv: (-kv[1][1], -sum(kv[1]))):
        n = c + w + a
        print(f"{k:26}{n:5}{c:5}{w:6}{a:6}{(c - w) / n:8.3f}")
    if args.errors and args.split != "holdout":
        print(f"\n--- {min(args.errors, len(rows))} wrong answers ---")
        for q, p, g in rows[: args.errors]:
            print(f"{q['business_id']} {q['as_of_date']} | {q['question']}\n   pred={json.dumps(p, ensure_ascii=False)[:160]}\n   gold={json.dumps({'type': g['type'], 'answer': g['answer']}, ensure_ascii=False)[:160]}")
    if ctx.llm is not None:
        print(ctx.llm.stats_line())
    if ctx.judge is not None:
        print(ctx.judge.stats_line())
    print(f"\nrun dir: {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
