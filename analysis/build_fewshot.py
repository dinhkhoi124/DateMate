"""Few-shot examples for the LLM parser (D14) from train-dev only (CLAUDE.md rules 1-2: never public, never holdout).

PYTHONPATH=src .venv/Scripts/python analysis/build_fewshot.py [--per-op 2] [--only biz_101 ...]

1. Runs the rule pipeline on dev questions, keeps those scored correct (silver intent labels).
2. Round-trip check of the LLM contract on every such intent: Intent -> model JSON (catalog names as mentions)
   -> Intent must give the same signature. Failures are printed by op: they are ops the model could never express.
3. Picks up to --per-op examples per op (different shops first, deterministic by id hash) ->
   src/datamate/parse/fewshot.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from datamate.execute.facts import facts_of  # noqa: E402
from datamate.link.catalog import linker_of  # noqa: E402
from datamate.parse.llm import FEWSHOT_PATH  # noqa: E402
from datamate.parse.llm_schema import OPS, from_intent, signature, to_intent  # noqa: E402
from datamate.pipeline import Context, analyse  # noqa: E402
from eval.scorer import load_jsonl, score  # noqa: E402

TRAIN = ROOT / "data" / "train" / "train"


def mention_names(it, f) -> dict:
    names = {}
    for k, v in it.flt.items():
        if v is None or (isinstance(v, str) and v.startswith("?")):
            continue
        if k == "product":
            names[k] = f.pname(v)
        elif k == "customer":
            names[k] = f.customer_display(v)
        elif k in ("category", "ingredient", "staff"):
            names[k] = str(v)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-op", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=str(FEWSHOT_PATH))
    args = ap.parse_args()

    holdout = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    qs = [q for q in load_jsonl(TRAIN / "questions.jsonl") if q["business_id"] not in holdout]
    if args.only:
        qs = [q for q in qs if q["business_id"] in set(args.only)]
    golds = {g["id"]: g for g in load_jsonl(TRAIN / "answers.jsonl")}
    ctx = Context(root=TRAIN)

    ok, bad = Counter(), defaultdict(list)
    pool = defaultdict(list)   # op -> candidates
    for q in qs:
        it, _, row = analyse(q, ctx)
        if it is None or score(row, golds[q["id"]]) != 1 or it.op not in OPS:
            continue
        f = facts_of(ctx.store(q["business_id"]), ctx.rules)
        lk = linker_of(f)
        target = from_intent(it, mention_names(it, f))
        back = to_intent(target, it.periods, lk)
        if signature(back) != signature(it):
            bad[it.op].append((q["id"], target, signature(it), signature(back)))
            continue
        ok[it.op] += 1
        pool[it.op].append({"question": q["question"], "target": target, "shop": q["business_id"], "qid": q["id"]})

    n_ok, n_bad = sum(ok.values()), sum(len(v) for v in bad.values())
    print(f"round-trip: {n_ok}/{n_ok + n_bad} intents survive Intent -> JSON -> Intent")
    for op, rows in sorted(bad.items(), key=lambda kv: -len(kv[1])):
        print(f"  {op}: {len(rows)} fail, e.g. {rows[0][0]} {rows[0][1]}\n      rule {rows[0][2]}\n      back {rows[0][3]}")

    out = []
    for op in OPS:  # prompt order = catalog order
        cands = sorted(pool.get(op, []), key=lambda e: hashlib.sha256(e["qid"].encode()).hexdigest())
        seen, picked = set(), []
        for e in cands + cands:   # first pass: one per shop; second pass: fill up
            if len(picked) >= args.per_op:
                break
            if e in picked or (e["shop"] in seen and len(seen) < len({c["shop"] for c in cands})):
                continue
            picked.append(e)
            seen.add(e["shop"])
        out += picked
    missing = [op for op in OPS if op not in pool and op not in ("need_cap", "unanswerable")]
    with open(args.out, "w", encoding="utf8", newline="\n") as w:
        for e in out:
            w.write(json.dumps({"question": e["question"], "target": e["target"], "qid": e["qid"]},
                               ensure_ascii=False) + "\n")
    print(f"{len(out)} examples, {len({e['target']['op'] for e in out})} ops -> {args.out}")
    if missing:
        print("ops without a correct dev example:", ", ".join(missing))


if __name__ == "__main__":
    main()
