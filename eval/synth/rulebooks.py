"""Synthetic rulebooks for the rule judge (D18 branch 3): made at run time from the train rulebook, no rulebook text
in the repo. The added sentences are our own.

PYTHONPATH=src .venv/Scripts/python eval/synth/rulebooks.py --out <dir>      # writes <variant>.md

Each variant touches the same rows (chosen for wide dev coverage):
- cosmetic : a neutral sentence appended -> every blocked question should be judged "same" (answer unchanged)
- semantic : an exception that changes the meaning -> blocked questions should NOT be judged "same"
Used as: eval/run_eval.py --split dev --rulebook <dir>/<variant>.md --llm ... (scripts/gpu/run_all.sh).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datamate.rulebook import parse_rows  # noqa: E402

TRAIN_RULEBOOK = ROOT / "data" / "train" / "train" / "rulebook.md"

NEUTRAL = "Ghi chú: luật này giữ nguyên như các kỳ trước."
EXCEPTIONS = {
    "§1.5": "Ngoại lệ: số đơn đếm cả đơn đã huỷ và đơn trả hàng.",
    "§1.6": "Ngoại lệ: doanh thu một đơn tính cả phí giao hàng và thuế VAT.",
    "§2.7": "Ngoại lệ: tháng M tính từ ngày 5 tháng M đến hết ngày 4 tháng sau.",
    "§3.3": "Ngoại lệ: bán chạy xếp theo số tiền bán được, không theo số lượng.",
    "§4.2": "Ngoại lệ: nợ một đơn không gồm phí giao hàng.",
}
EXPECTED = {"cosmetic": "same", "semantic": "different"}


def mutate_row(md: str, rid: str, f) -> str:
    """Apply f to the markdown line of row `rid` (located by the rulebook parser)."""
    row = next(r for r in parse_rows(md) if r.rid == rid)
    lines = md.splitlines()
    lines[row.line - 1] = f(lines[row.line - 1])
    return "\n".join(lines) + "\n"


def _append(sentence: str):
    return lambda s: s.rstrip().rstrip("|").rstrip() + " " + sentence + " |"


def variants(md: str) -> dict[str, str]:
    out = {}
    cos, sem = md, md
    for rid, exc in EXCEPTIONS.items():
        cos = mutate_row(cos, rid, _append(NEUTRAL))
        sem = mutate_row(sem, rid, _append(exc))
    out["cosmetic"], out["semantic"] = cos, sem
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, md in variants(TRAIN_RULEBOOK.read_text(encoding="utf8")).items():
        (out / f"{name}.md").write_text(md, encoding="utf8", newline="\n")
        print(f"{name}: rows {', '.join(EXCEPTIONS)} -> expected judge verdict '{EXPECTED[name]}'")


if __name__ == "__main__":
    main()
