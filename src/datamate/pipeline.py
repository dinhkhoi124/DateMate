"""Question -> submission row. Stages are filled in as modules get built (see docs/ARCHITECTURE.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Context:
    """Per-split context: root dir holding questions.jsonl, rulebook.md, calendar.csv, data/<biz>/."""
    root: Path
    stores: dict = field(default_factory=dict)  # business_id -> Store (lazy, cached)


def abstain(qid: str) -> dict:
    return {"id": qid, "type": "abstain"}


def answer(q: dict, ctx: Context) -> dict:
    """Return one submission row for question `q`. Baseline: abstain everything."""
    return abstain(q["id"])
