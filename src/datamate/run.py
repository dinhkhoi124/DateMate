"""Entrypoint: python -m datamate.run --data /data --out /result/submission.jsonl

Streams one line per question (flush after each). Any error or timeout -> abstain, never crash.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from datamate.pipeline import Context, abstain, answer


def find_root(data: Path) -> Path:
    """Locate the dir containing questions.jsonl (handles /data or /data/<split>/<split>)."""
    if (data / "questions.jsonl").exists():
        return data
    hits = sorted(data.rglob("questions.jsonl"))
    if not hits:
        raise FileNotFoundError(f"questions.jsonl not found under {data}")
    return hits[0].parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/result/submission.jsonl")
    ap.add_argument("--only", nargs="*", help="restrict to these business_ids (dev use)")
    args = ap.parse_args(argv)

    root = find_root(Path(args.data))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx = Context(root=root)
    with open(root / "questions.jsonl", encoding="utf8") as f:
        qs = [json.loads(l) for l in f if l.strip()]
    if args.only:
        qs = [q for q in qs if q["business_id"] in set(args.only)]

    t0 = time.time()
    with open(out, "w", encoding="utf8", newline="\n") as w:
        for q in qs:
            try:
                row = answer(q, ctx)
            except Exception:  # never crash the run
                traceback.print_exc(file=sys.stderr)
                row = abstain(q["id"])
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            w.flush()
    print(f"[datamate] {len(qs)} questions in {time.time() - t0:.1f}s -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
