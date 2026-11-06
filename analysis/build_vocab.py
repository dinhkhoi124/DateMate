"""Build the folded Vietnamese syllable vocabulary used by normalize/ to repair Telex typos.

Sources (train-only, never public, never holdout): rulebook, dev READMEs, and accented words of dev questions
(a word that carries a diacritic was typed with an IME, so its folded form is a real syllable).
Usage: .venv/Scripts/python analysis/build_vocab.py
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from unidecode import unidecode

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train" / "train"
OUT = ROOT / "src" / "datamate" / "normalize" / "vocab.txt"


def words(s):
    return re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", s.lower()))


def main():
    hold = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    c = Counter()
    for w in words((TRAIN / "rulebook.md").read_text(encoding="utf8")):
        c[unidecode(w)] += 5
    for d in sorted((TRAIN / "data").iterdir()):
        if d.name in hold:
            continue
        for w in words((d / "README.md").read_text(encoding="utf8")):
            if not w.isascii():
                c[unidecode(w)] += 1
    for line in open(TRAIN / "questions.jsonl", encoding="utf8"):
        q = json.loads(line)
        if q["business_id"] in hold:
            continue
        for w in words(q["question"]):
            if not w.isascii():
                c[unidecode(w)] += 1
    vocab = sorted(w for w, n in c.items() if n >= 2 and len(w) <= 7)
    OUT.write_text("\n".join(vocab) + "\n", encoding="utf8")
    print(len(vocab), "syllables ->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
