"""Question -> submission row (docs/ARCHITECTURE.md §1): normalize -> timeparse -> parse -> link -> execute -> answer."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from datamate.config import Rules
from datamate.execute.dispatch import run
from datamate.execute.facts import facts_of
from datamate.execute.ops import Abstain, Unanswerable
from datamate.ingest.checks import blocking, metrics, violations
from datamate.ingest.store import Store
from datamate.link.catalog import linker_of
from datamate.normalize.text import normalize, normalize_pair
from datamate.parse.ensemble import decide, needs_llm
from datamate.parse.gate import load_vocab, residual, unknown
from datamate.parse.rules import RuleParser
from datamate.rulebook import RulebookDiff, fold
from datamate.rulebook import analyse as analyse_rulebook
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
    llm: object = None                                # parse.llm.LLMParser (D14); None = rule-only
    rulebook: Path = None                             # override <root>/rulebook.md (synthetic-rulebook tests)
    rb: RulebookDiff = None                           # D18 diff of the split's rulebook vs train
    judge: object = None                              # parse.rule_judge.RuleJudge (D18 branch 3), from `llm`

    def __post_init__(self):
        if self.vocab is None:
            self.vocab = load_vocab()
        rb = Path(self.rulebook) if self.rulebook else self.root / "rulebook.md"
        self.rb = analyse_rulebook(rb.read_text(encoding="utf8") if rb.exists() else None)
        self.rules = self.rb.rules
        if self.rb.changed or self.rb.missing or self.rb.new_rows:   # counts only, never rulebook text
            print(f"[datamate] {self.rb.summary()}", file=sys.stderr)
        self.cal = LunarCalendar.load(str(self.root / "calendar.csv"))
        if self.judge is None and self.llm is not None and self.llm.cfg.rule_judge:
            from datamate.parse.rule_judge import RuleJudge  # noqa: PLC0415
            self.judge = RuleJudge(self.llm.backend, self.llm.cfg)

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
    unk = unknown(words, ctx.vocab) if ctx.gate and it is not None else []
    llm_its = raw = None
    if ctx.llm is not None and needs_llm(it, not unk, ctx.llm.cfg):
        llm_its, raw = ctx.llm.parse(q["question"], periods, lk)
    if ctx.llm is not None:
        d = decide(it, not unk, llm_its, ctx.llm.cfg)
        shown = d.intent or it or next((i for i in llm_its or [] if i is not None), None)
        if shown is not None:
            shown.llm_raw, shown.source = raw, d.source
            if d.note:
                shown.note = d.note + (": " + " ".join(unk) if unk and d.intent is None else "")
        if d.intent is None:
            return shown, t, abstain(q["id"])
        it = d.intent
    else:
        if it is None:
            return None, t, abstain(q["id"])
        if unk:
            it.note = "gate: " + " ".join(unk)
            return it, t, abstain(q["id"])
    folded = fold(q["question"])
    why = ctx.rb.block_reason(it, folded)
    if why:   # D18: the split's rulebook changed a rule this intent relies on, or adds a rule it mentions
        ok = False   # fail closed: no judge, judge error or any non-'same' -> abstain, never the rule answer
        if ctx.judge is not None:   # D19: the model decides whether the split's rule computes what the executor does
            try:
                ok, raw = ctx.judge.judge(q["question"], it, ctx.rb.hits(it, folded), ctx.rb)
            except Exception as e:  # noqa: BLE001
                ok, raw = False, []
                why += f"; judge error {type(e).__name__}"
            it.llm_raw = (it.llm_raw or []) + raw
            why += "; judge " + ("same" if ok else "not same")
        it.note = why
        if not ok:
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


def trace_of(t: str, it, row: dict) -> dict:
    """Log fields besides id/answer/elapsed_ms (SPEC submission contract, Guide §3.3).
    thinking: raw model outputs (list, one per sample) when the LLM parser ran, else None (rule parser). The executor
    is a pre-written function, so the contract asks for code (which function), inputs and output."""
    inputs = {"normalized_question": t}
    if it is not None:
        inputs.update(op=it.op, atype=it.atype, filters=it.flt, args=it.args, source=it.source,
                      periods=[{"label": p.label, "kind": p.kind, "start": p.start.isoformat(),
                                "end": p.end.isoformat()} for p in it.periods])
        if it.note:
            inputs["note"] = it.note
    return {"thinking": it.llm_raw if it is not None else None,
            "code": f"datamate.execute.dispatch.run(op={it.op!r})" if it is not None else None,
            "inputs": inputs,
            "output": {k: v for k, v in row.items() if k != "id"}}


def answer_traced(q: dict, ctx: Context) -> tuple[dict, dict]:
    it, t, row = analyse(q, ctx)
    if ctx.debug:
        print(q["id"], t, it, row, file=sys.stderr)
    return row, trace_of(t, it, row)


def answer(q: dict, ctx: Context) -> dict:
    return answer_traced(q, ctx)[0]
