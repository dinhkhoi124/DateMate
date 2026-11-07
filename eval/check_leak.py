"""Leak check (D15): no tracked text file may contain a competition question verbatim. The repo is public.

.venv/Scripts/python eval/check_leak.py [files...]      # no files: every file `git ls-files` returns

Compares each file (NFC, lowercased, whitespace collapsed) against every question in data/**/questions.jsonl that is
at least MIN_LEN characters long. Prints only file and count, never the question text (CLAUDE.md rules 1-2: the
public and holdout questions are matched by code, nobody reads them). Without local data it skips (exit 0), so CI and
machines without data are not blocked; pre-commit on a machine with data is where it bites.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_LEN = 25
TEXT_EXT = {".py", ".md", ".txt", ".json", ".jsonl", ".yml", ".yaml", ".toml", ".sh", ".cfg", ".ini", ".csv"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip().lower()


def questions() -> set[str]:
    out = set()
    for p in sorted((ROOT / "data").glob("**/questions.jsonl")):
        with open(p, encoding="utf8") as f:
            for line in f:
                if line.strip():
                    q = norm(json.loads(line).get("question", ""))
                    if len(q) >= MIN_LEN:
                        out.add(q)
    return out


def main(argv: list[str]) -> int:
    qs = questions()
    if not qs:
        print("check_leak: no local competition data, skipped")
        return 0
    files = argv or subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                                   encoding="utf8").stdout.splitlines()
    bad = []
    for f in sorted(files):
        p = ROOT / f
        if p.suffix.lower() not in TEXT_EXT or not p.is_file() or p.stat().st_size > 5_000_000:
            continue
        try:
            text = norm(p.read_text(encoding="utf8"))
        except (UnicodeDecodeError, OSError):
            continue
        n = sum(1 for q in qs if q in text)
        if n:
            bad.append((f, n))
    for f, n in bad:
        print(f"{f}: {n} competition question(s) copied verbatim - reword them or keep the file off git")
    if not bad:
        print(f"check_leak: {len(files)} file(s) clean")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
