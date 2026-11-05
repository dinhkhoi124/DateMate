"""Question -> submission row (docs/ARCHITECTURE.md §1): normalize -> timeparse -> parse -> link -> execute -> answer."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from datamate.config import DEFAULT, Rules, from_rulebook
from datamate.execute.dispatch import run
from datamate.execute.facts import facts_of
from datamate.execute.ops import Abstain, Unanswerable
from datamate.ingest.checks import blocking, metrics, violations
from datamate.ingest.store import Store
from datamate.link.catalog import linker_of
from datamate.normalize.text import normalize, normalize_pair
from datamate.parse.gate import load_vocab, residual, unknown
from datamate.parse.rules import RuleParser
from datamate.timeparse.periods import LunarCalendar, find_periods


@dataclass
class Context:
    """Per-split context: root dir holding questions.jsonl, rulebook.md, calendar.csv, data/<biz>/."""
    root: Path
    stores: dict = field(default_factory=dict)  # business_id -> Store (lazy, cached)
    rules: Rules = None
    cal: LunarCalendar = None
    debug: bool = False
    gate: bool = True                                 # D10 coverage gate; off only for analysis/build_gate_vocab.py
    checks: bool = True                               # D11 run-time ingest invariants; off only to measure them
    vocab: frozenset = None
    trace: dict = None                                # id -> residual words, filled when not None

    def __post_init__(self):
        if self.vocab is None:
            self.vocab = load_vocab()
        rb = self.root / "rulebook.md"
        self.rules = from_rulebook(rb.read_text(encoding="utf8")) if rb.exists() else DEFAULT
        self.cal = LunarCalendar.load(str(self.root / "calendar.csv"))

    def store(self, biz: str) -> Store:
        if biz not in self.stores:
            self.stores[biz] = Store.build(self.root / "data" / biz)
        return self.stores[biz]

    def violations(self, biz: str) -> list:
        """D11: invariant violations of the shop's Store (computed once per shop)."""
        st = self.store(biz)
        if "_violations" not in st.extra:
            st.extra["_violations"] = violations(metrics(st))
        return st.extra["_violations"]


def abstain(qid: str) -> dict:
    return {"id": qid, "type": "abstain"}


def unanswerable(qid: str, reason: str) -> dict:
    return {"id": qid, "type": "unanswerable", "reason": str(reason)}


def analyse(q: dict, ctx: Context):
    """Returns (intent, normalized text, row)."""
    st = ctx.store(q["business_id"])
    if st.orders is None or not len(st.orders) or "order_date" not in st.orders:
        # ingest found no sales table: data was not understood, so nothing is known about it (not 'unanswerable')
        return None, "", abstain(q["id"])
    f = facts_of(st, ctx.rules)
    lk = linker_of(f)
    as_of = date.fromisoformat(q["as_of_date"])
    pre, t = normalize_pair(q["question"], lk.protect)
    periods = find_periods(t, as_of, ctx.cal, ctx.rules)
    parser = st.extra.get("_parser")
    if parser is None:
        units = {normalize(u) for u in list(f.L["unit"].dropna().unique()) + list(st.units["unit"].dropna().unique())} \
            if f.has_lines else set()
        parser = st.extra["_parser"] = RuleParser(lk, {u for u in units if u and len(u) <= 6 and " " not in u})
    it = parser.parse(t, periods, as_of, pre=pre)
    words = residual(t, pre, lk, periods)
    if ctx.trace is not None:
        ctx.trace[q["id"]] = words
    if it is None:
        return None, t, abstain(q["id"])
    unk = unknown(words, ctx.vocab) if ctx.gate else []
    if unk:
        it.note = "gate: " + " ".join(unk)
        return it, t, abstain(q["id"])
    if ctx.checks:
        bad = blocking(it.op, ctx.violations(q["business_id"]))
        if bad:   # D11: the data this op reads failed a run-time invariant -> not enough evidence to answer
            it.note = "invariant: " + ", ".join(sorted({v.name for v in bad}))
            return it, t, abstain(q["id"])
    try:
        typ, val = run(it, f, as_of)
    except Unanswerable as e:
        return it, t, unanswerable(q["id"], str(e))
    except Abstain as e:
        it.note = f"abstain: {e}"
        return it, t, abstain(q["id"])
    return it, t, {"id": q["id"], "type": typ, "answer": val}


def answer(q: dict, ctx: Context) -> dict:
    it, t, row = analyse(q, ctx)
    if ctx.debug:
        print(q["id"], t, it, row, file=sys.stderr)
    return row
