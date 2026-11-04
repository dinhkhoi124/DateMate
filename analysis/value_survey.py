"""Collect distinct raw values of categorical columns (by README description) across dev shops.

.venv/Scripts/python analysis/value_survey.py "trạng thái đơn" "hình thức thanh toán" "kênh bán"
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train" / "train" / "data"
HOLDOUT = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
SEC = re.compile(r"^## Cột trong `(.+?)`", re.M)
COL = re.compile(r"^- \*\*(.+?)\*\*: (.+)$", re.M)


def readme_cols(text):
    out = {}
    parts = SEC.split(text)
    for i in range(1, len(parts), 2):
        out[parts[i]] = [(c, d) for c, d in COL.findall(parts[i + 1])]
    return out


def read_any(path: Path, col: str):
    vals = []
    if path.suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
        if col in df:
            vals += df[col].dropna().tolist()
    elif path.suffix == ".xlsx":
        for s, raw in pd.read_excel(path, sheet_name=None, header=None, dtype=str).items():
            for i in range(min(8, len(raw))):
                row = [str(x).strip() for x in raw.iloc[i].tolist()]
                if col in row:
                    j = row.index(col)
                    vals += raw.iloc[i + 1:, j].dropna().tolist()
                    break
    return vals


def main():
    wanted = sys.argv[1:] or ["trạng thái đơn"]
    for w in wanted:
        c = Counter()
        for shop in sorted(DATA.iterdir()):
            if shop.name in HOLDOUT:
                continue
            for fname, cols in readme_cols((shop / "README.md").read_text(encoding="utf8")).items():
                for col, desc in cols:
                    if desc.startswith(w):
                        f = shop / fname
                        if f.exists():
                            c.update(f"{shop.name}:{v}" if "--by-shop" in sys.argv else v for v in read_any(f, col))
        print(f"\n== {w}: {len(c)} distinct")
        for v, n in c.most_common(80):
            print(f"  {n:7} {v!r}")


if __name__ == "__main__":
    main()
