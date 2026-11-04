"""Build Store for every train shop and print a health report.

PYTHONPATH=src .venv/Scripts/python analysis/ingest_report.py [biz ...] [--no-cache]
Holdout shops: aggregate counts only (CLAUDE.md rule 2).
"""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datamate.ingest.store import Store  # noqa: E402

DATA = ROOT / "data" / "train" / "train" / "data"
HOLDOUT = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_cache = "--no-cache" not in sys.argv
    shops = args or sorted(p.name for p in DATA.iterdir())
    ok = 0
    t_all = time.time()
    for b in shops:
        t = time.time()
        try:
            st = Store.build(DATA / b, use_cache=use_cache)
        except Exception as e:
            print(f"✗ {b}: {type(e).__name__}: {e}")
            if b not in HOLDOUT:
                traceback.print_exc()
            continue
        ok += 1
        c = st.caps
        dt = time.time() - t
        if b in HOLDOUT:
            print(f"✓ {b} HOLDOUT orders={len(st.orders)} lines={len(st.lines)} warnings={len(st.warnings)} {dt:.1f}s")
            continue
        flags = [k[4:] for k, v in c.items() if k.startswith("has_") and v is True]
        print(f"✓ {b} {dt:.1f}s orders={len(st.orders)} lines={len(st.lines)} products={len(st.products)} "
              f"dates={c.get('date_min')}..{c.get('date_max')} end={c.get('data_end')} status={c.get('status_counts')} "
              f"mismatch={c.get('check_subtotal_mismatch')}")
        print(f"     caps: {' '.join(flags)}")
        for w in st.warnings:
            print(f"     ⚠ {w}")
    print(f"\n{ok}/{len(shops)} shops built in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
