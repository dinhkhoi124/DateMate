"""Ensemble calibration from a GPU replay, on CPU (PLAN 08-10/11, DECISIONS D14/D15).

PYTHONPATH=src .venv/Scripts/python eval/calibrate.py --split dev --llm-model models/Qwen3-14B \
    --llm-replay <bundle>/replay/Qwen3-14B_dev_n1.jsonl [--only biz_101 ...]

One pass per question (rule parse, gate, LLM intents from the replay), then every ensemble policy is applied with
parse.ensemble.decide + pipeline.finish, so all policies are compared on exactly the same model outputs. The replay
must have been recorded with DATAMATE_LLM_CALL_ALL=1 (scripts/gpu/run_all.sh): a question missing from it counts as a
model failure (reported as `replay_miss`), never as a model answer.

Holdout: aggregate numbers only (CLAUDE.md rule 2), no per-question output. Results with an empty or fake replay
only check the plumbing; they are not LLM accuracy.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from datamate.config import LLMConfig  # noqa: E402
from datamate.parse.ensemble import decide  # noqa: E402
from datamate.parse.llm_schema import signature  # noqa: E402
from datamate.pipeline import Context, abstain, finish, parse_question  # noqa: E402
from datamate.run import make_llm  # noqa: E402
from eval.scorer import load_jsonl, score  # noqa: E402

TRAIN = ROOT / "data" / "train" / "train"


def policies(n_samples: int) -> dict[str, dict | None]:
    """name -> LLMConfig overrides (None = rule-only baseline, no model)."""
    k = max(1, n_samples)
    off = {"rescue_gate": False, "require_agreement": False, "llm_only": False}
    return {
        "rule": None,
        "rescue": {**off, "rescue_gate": True},
        "rescue+agree": {**off, "rescue_gate": True, "require_agreement": True},
        f"rescue+llm_only{k}": {**off, "rescue_gate": True, "llm_only": True, "llm_only_min_votes": k},
        f"rescue+agree+llm_only{k}": {"rescue_gate": True, "require_agreement": True, "llm_only": True,
                                      "llm_only_min_votes": k},
    }


def decided(p, cfg: LLMConfig | None):
    """Intent a policy answers with (None = abstain before execution)."""
    if p is None:
        return None
    if cfg is None:                                   # rule-only path of pipeline.analyse
        return p.it if p.it is not None and not p.unk else None
    return decide(p.it, not p.unk, p.llm_its, cfg).intent


def diagnostics(p, gold_ok_rule: bool | None) -> Counter:
    """Model behaviour on one question (counts only)."""
    c = Counter()
    if p is None:
        return c
    c["questions"] += 1
    if p.llm_its is None or not p.llm_its:
        c["replay_miss"] += 1
        return c
    valid = all(i is not None for i in p.llm_its)
    c["llm_valid"] += valid
    rs, ls = signature(p.it), [signature(i) for i in p.llm_its]
    agree = rs is not None and all(s == rs for s in ls)
    if p.it is not None and not p.unk:
        c["rule_answered"] += 1
        c["agree_on_rule_answered"] += agree
        if gold_ok_rule is True:
            c["agree_when_rule_correct"] += agree
            c["rule_correct"] += 1
        elif gold_ok_rule is False:
            c["disagree_when_rule_wrong"] += not agree
            c["rule_wrong"] += 1
    elif p.it is not None:
        c["gate_blocked"] += 1
        c["agree_on_gate_blocked"] += agree
    else:
        c["no_rule_parse"] += 1
        c["llm_valid_no_rule"] += valid
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--llm-model", required=True, help="weights dir name = replay key (e.g. models/Qwen3-14B)")
    ap.add_argument("--llm-replay", required=True)
    ap.add_argument("--only", nargs="*", help="dev shops only")
    ap.add_argument("--rulebook")
    args = ap.parse_args()

    holdout = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    qs = load_jsonl(TRAIN / "questions.jsonl")
    qs = [q for q in qs if (q["business_id"] in holdout) == (args.split == "holdout")]
    if args.only:
        if args.split == "holdout":
            sys.exit("--only is not allowed on holdout")
        qs = [q for q in qs if q["business_id"] in set(args.only)]
    golds = {g["id"]: g for g in load_jsonl(TRAIN / "answers.jsonl")}

    llm = make_llm("replay", args.llm_model, args.llm_replay)
    if not Path(args.llm_replay).exists():
        sys.exit(f"replay not found: {args.llm_replay}")
    llm.cfg = replace(llm.cfg, call_all=True)         # parse every question: the replay was recorded that way
    ctx = Context(root=TRAIN, llm=llm, rulebook=args.rulebook)
    pol = policies(llm.cfg.n_samples)
    cfgs = {k: (None if v is None else replace(llm.cfg, **v)) for k, v in pol.items()}

    t0 = time.time()
    res = {k: Counter() for k in pol}
    diag = Counter()
    for q in qs:
        g = golds[q["id"]]
        try:
            p = parse_question(q, ctx)
        except Exception as e:  # noqa: BLE001  (counted as abstain for every policy)
            print(f"ERR {q['id']}: {type(e).__name__}", file=sys.stderr)
            p = None
        cache = {}
        for name, cfg in cfgs.items():
            it = decided(p, cfg)
            if it is None:
                row = abstain(q["id"])
            else:
                key = it.key()
                if key not in cache:
                    try:
                        cache[key] = finish(q, ctx, p, copy.copy(it))[2]
                    except Exception:  # noqa: BLE001
                        cache[key] = abstain(q["id"])
                row = cache[key]
            res[name][{1: "correct", -1: "wrong", 0: "abstain"}[score(row, g)]] += 1
        rule_s = score(cache.get(p.it.key()), g) if p is not None and p.it is not None and p.it.key() in cache else None
        diag += diagnostics(p, None if rule_s in (None, 0) else rule_s == 1)

    n = len(qs) or 1
    print(f"split={args.split} n={len(qs)} model={llm.cfg.model_id} samples={llm.cfg.n_samples} "
          f"replay={Path(args.llm_replay).name} seconds={time.time() - t0:.0f}")
    print(f"\n{'policy':28}{'ok':>6}{'wrong':>7}{'abst':>6}{'PA':>8}{'prec':>8}")
    report = {"split": args.split, "n": len(qs), "model_id": llm.cfg.model_id, "policies": {}, "diagnostics": {}}
    for name, c in res.items():
        a = c["correct"] + c["wrong"]
        pa, prec = (c["correct"] - c["wrong"]) / n, (c["correct"] / a if a else 0.0)
        print(f"{name:28}{c['correct']:6}{c['wrong']:7}{c['abstain']:6}{pa:8.3f}{prec:8.3f}")
        report["policies"][name] = {**c, "PA": round(pa, 4), "precision": round(prec, 4)}
    print("\ndiagnostics: " + " ".join(f"{k}={v}" for k, v in sorted(diag.items())))
    if diag["questions"] and diag["replay_miss"] == diag["questions"]:
        print("WARNING: every question missed the replay (wrong model dir name / replay file / few-shot sha?): "
              "the table is rule-only, not an LLM result")
    report["diagnostics"] = dict(diag)
    run_dir = ROOT / "eval" / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_calibrate_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf8")
    print(f"\nrun dir: {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
