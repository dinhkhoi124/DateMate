"""Build src/datamate/rulebook_index.json from the TRAIN rulebook (DECISIONS D18).

PYTHONPATH=src .venv/Scripts/python analysis/build_rulebook_index.py [--check]

The index holds hashes and row ids only (no rulebook text: the repo is public, D15). Every indexed row id must have a
scope in datamate.rulebook.SCOPES or be global, so a new train row cannot silently block nothing.
--check: exit 1 if the committed index differs from the train rulebook (tests call this logic).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datamate.rulebook import GLOBAL, INDEX_PATH, SCOPES, build_index, scope_of  # noqa: E402

TRAIN_RULEBOOK = ROOT / "data" / "train" / "train" / "rulebook.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    idx = build_index(TRAIN_RULEBOOK.read_text(encoding="utf8"))
    ids = [e["id"] for e in idx["rows"].values()]
    unscoped = [i for i in ids if scope_of(i) != GLOBAL and not scope_of(i)]
    stale = sorted(set(SCOPES) - set(ids))
    print(f"{len(ids)} rows ({sum(scope_of(i) == GLOBAL for i in ids)} global), {len(idx['train_ngrams'])} n-grams")
    print("ids:", " ".join(sorted(ids, key=lambda s: (s.split('.')[0], s))))
    if unscoped or stale:
        sys.exit(f"rows without scope: {unscoped}; SCOPES ids not in rulebook: {stale}")
    text = json.dumps(idx, ensure_ascii=False, indent=0) + "\n"
    if args.check:
        same = INDEX_PATH.exists() and INDEX_PATH.read_text(encoding="utf8") == text
        print("index up to date" if same else "index STALE: rerun without --check")
        sys.exit(0 if same else 1)
    INDEX_PATH.write_text(text, encoding="utf8", newline="\n")
    print(f"-> {INDEX_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
