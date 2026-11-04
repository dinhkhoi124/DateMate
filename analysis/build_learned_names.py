"""Build src/datamate/ingest/learned_names.json: column-name -> role learned from DEV shop READMEs only.

.venv/Scripts/python analysis/build_learned_names.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from datamate.ingest.readme import parse_readme  # noqa: E402
from datamate.ingest.vocab import learn_name_roles  # noqa: E402

HOLDOUT = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
pairs = []
for p in sorted((ROOT / "data/train/train/data").glob("*/README.md")):
    if p.parent.name in HOLDOUT:
        continue
    pairs += [(c.name, c.desc) for c in parse_readme(p).all_columns()]
learned = dict(sorted(learn_name_roles(pairs).items()))
out = ROOT / "src/datamate/ingest/learned_names.json"
out.write_text(json.dumps(learned, ensure_ascii=False, indent=1), encoding="utf8")
print(f"{len(learned)} names -> {out.relative_to(ROOT)}")
