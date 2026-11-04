"""Dump structure of every data file for dev shops (never holdout).

.venv/Scripts/python analysis/file_survey.py [biz_id ...] > analysis/out/file_survey.txt
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train" / "train" / "data"
HOLDOUT = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])


def show(df: pd.DataFrame, n=3) -> str:
    return df.head(n).to_string(max_colwidth=28, max_cols=30)


def survey(shop: Path):
    print(f"\n######## {shop.name}")
    for f in sorted(shop.iterdir()):
        if f.name == "README.md":
            continue
        print(f"\n=== {f.name}")
        if f.suffix == ".csv":
            df = pd.read_csv(f, dtype=str, nrows=2000)
            print(f"rows~{sum(1 for _ in open(f, encoding='utf8')) - 1} cols={list(df.columns)}")
            print(show(df))
        elif f.suffix == ".xlsx":
            xl = pd.ExcelFile(f)
            names = xl.sheet_names
            print(f"sheets({len(names)}): {names[:6]}{' ...' if len(names) > 6 else ''}")
            for s in names[:2]:
                df = pd.read_excel(f, sheet_name=s, header=None, dtype=str, nrows=6)
                print(f"--- sheet {s!r} (raw first rows)")
                print(df.to_string(max_colwidth=22, max_cols=30))
        elif f.suffix == ".sqlite":
            con = sqlite3.connect(f)
            for (t,) in con.execute("select name from sqlite_master where type='table' order by name"):
                cols = [r[1] for r in con.execute(f"pragma table_info('{t}')")]
                n = con.execute(f"select count(*) from '{t}'").fetchone()[0]
                print(f"  {t} ({n}): {cols}")


def main():
    shops = sys.argv[1:] or sorted(p.name for p in DATA.iterdir())
    for s in shops:
        if s in HOLDOUT:
            continue
        survey(DATA / s)


if __name__ == "__main__":
    main()
