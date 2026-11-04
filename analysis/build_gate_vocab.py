"""Build the coverage-gate vocabulary (D10) from train-dev and estimate its cost with leave-one-industry-out.

PYTHONPATH=src .venv/Scripts/python analysis/build_gate_vocab.py [--min-shops 2] [--write]

Dev only: holdout shops are never read here (CLAUDE.md rule 2). Industry folds are inferred from question wording
(analysis only, nothing here reaches src/ except the vocabulary file).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from datamate.parse.gate import VOCAB_PATH  # noqa: E402
from datamate.pipeline import Context, abstain, answer  # noqa: E402
from eval.scorer import load_jsonl, score  # noqa: E402

TRAIN = ROOT / "data" / "train" / "train"
CACHE = ROOT / "analysis" / "out" / "gate_dev.json"
INDUSTRY_KW = {"pharmacy": ["thuốc", "hoạt chất", "kê đơn", "hạn dùng"], "cafe": ["món", "topping", "ly ", "trà sữa", "size"],
               "building": ["xi măng", "m3", "thép", "gạch", "cát", "đợt giao"]}


def industry(questions: list[str]) -> str:
    c = Counter(k for q in questions for k, ws in INDUSTRY_KW.items() if any(w in q.lower() for w in ws))
    if not c or c.most_common(1)[0][1] < 5:
        return "general"
    return c.most_common(1)[0][0]


def collect() -> list[dict]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf8"))
    holdout = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    qs = [q for q in load_jsonl(TRAIN / "questions.jsonl") if q["business_id"] not in holdout]
    golds = {g["id"]: g for g in load_jsonl(TRAIN / "answers.jsonl")}
    ctx = Context(root=TRAIN, gate=False, trace={})
    rows = []
    for q in qs:
        try:
            p = answer(q, ctx)
        except Exception:
            p = abstain(q["id"])
        rows.append({"id": q["id"], "biz": q["business_id"], "q": q["question"],
                     "words": ctx.trace.get(q["id"], []), "s": score(p, golds[q["id"]])})
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")
    return rows


def vocab_of(rows, min_shops: int) -> set[str]:
    shops = defaultdict(set)
    for r in rows:
        if r["s"] == 1:
            for w in r["words"]:
                shops[w].add(r["biz"])
    return {w for w, s in shops.items() if len(s) >= min_shops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-shops", type=int, default=2)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = collect()
    byshop = defaultdict(list)
    for r in rows:
        byshop[r["biz"]].append(r["q"])
    ind = {b: industry(qs) for b, qs in byshop.items()}
    folds = defaultdict(list)
    for b, i in sorted(ind.items()):
        folds[i].append(b)
    print("folds:", {k: len(v) for k, v in folds.items()})

    print(f"\n{'fold':10}{'min':>4}{'n':>6}{'ok':>6}{'wrong':>6}{'ok_g':>6}{'wr_g':>6}{'cov':>7}{'cov_g':>7}")
    for ms in (1, 2, 3):
        tot = Counter()
        for fold, shops in sorted(folds.items()):
            train = [r for r in rows if r["biz"] not in shops]
            test = [r for r in rows if r["biz"] in shops]
            v = vocab_of(train, ms)
            c = Counter()
            for r in test:
                passed = all(w in v for w in r["words"])
                c["n"] += 1
                c["ok"] += r["s"] == 1
                c["wrong"] += r["s"] == -1
                c["ok_g"] += r["s"] == 1 and passed
                c["wr_g"] += r["s"] == -1 and passed
            tot.update(c)
            n = c["n"]
            print(f"{fold:10}{ms:4}{n:6}{c['ok']:6}{c['wrong']:6}{c['ok_g']:6}{c['wr_g']:6}"
                  f"{(c['ok'] + c['wrong']) / n:7.3f}{(c['ok_g'] + c['wr_g']) / n:7.3f}")
        n = tot["n"]
        print(f"{'ALL':10}{ms:4}{n:6}{tot['ok']:6}{tot['wrong']:6}{tot['ok_g']:6}{tot['wr_g']:6}"
              f"{(tot['ok'] + tot['wrong']) / n:7.3f}{(tot['ok_g'] + tot['wr_g']) / n:7.3f}\n")

    if args.write:
        v = sorted(vocab_of(rows, args.min_shops))
        VOCAB_PATH.write_text(f"# built by analysis/build_gate_vocab.py --min-shops {args.min_shops} (train-dev only)\n"
                              + "\n".join(v) + "\n", encoding="utf8")
        print(f"wrote {len(v)} words -> {VOCAB_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
