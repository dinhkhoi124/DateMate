"""Ingest regression check on the original dev shops: Store fingerprints now vs a saved snapshot.

PYTHONPATH=src .venv/Scripts/python eval/synth/fp_snapshot.py save eval/synth/out/fp_base.json
PYTHONPATH=src .venv/Scripts/python eval/synth/fp_snapshot.py diff eval/synth/out/fp_base.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from datamate.ingest.store import Store  # noqa: E402
from eval.synth.make import TRAIN  # noqa: E402
from eval.synth.run import fingerprint  # noqa: E402


def current() -> dict:
    holdout = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    out = {}
    for d in sorted((TRAIN / "data").iterdir()):
        if d.is_dir() and d.name not in holdout:
            out[d.name] = json.loads(json.dumps(fingerprint(Store.build(d)), default=str))
    return out


def main():
    cmd, path = sys.argv[1], Path(sys.argv[2])
    cur = current()
    if cmd == "save":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf8")
        print(f"saved {len(cur)} shops -> {path}")
        return
    base = json.loads(path.read_text(encoding="utf8"))
    bad = 0
    for b, fp in base.items():
        diff = [k for k in fp if fp[k] != cur.get(b, {}).get(k)]
        if diff:
            bad += 1
            print(f"  {b}: " + ", ".join(f"{k}: {fp[k]} -> {cur.get(b, {}).get(k)}" for k in diff)[:600])
    print(f"ingest regression: {len(base) - bad}/{len(base)} dev shops unchanged")


if __name__ == "__main__":
    main()
