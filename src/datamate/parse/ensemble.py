"""Rule parser x LLM parser -> one intent or abstain (ARCHITECTURE §5 condition 1, DECISIONS D14).

Agreement is equality of llm_schema.signature (op, answer type, periods, linked filters, contract args).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from datamate.config import LLMConfig
from datamate.parse.intent import Intent
from datamate.parse.llm_schema import signature


@dataclass
class Decision:
    intent: Intent | None   # None -> abstain
    source: str             # rule | rule+llm | llm | none
    note: str = ""


def needs_llm(rule: Intent | None, gate_ok: bool, cfg: LLMConfig) -> bool:
    """Only call the model when its answer can change the outcome (or always, in record mode: D15)."""
    if cfg.call_all:
        return True
    if rule is not None and gate_ok:
        return cfg.require_agreement
    if rule is not None:
        return cfg.rescue_gate
    return cfg.llm_only


def decide(rule: Intent | None, gate_ok: bool, llm: list[Intent | None] | None, cfg: LLMConfig) -> Decision:
    """llm=None: model not called / not configured. llm=[]: model called but failed."""
    rs = signature(rule)
    ls = [signature(i) for i in llm] if llm is not None else []
    agree = bool(ls) and rs is not None and all(s == rs for s in ls)
    if rule is not None and gate_ok:
        if not cfg.require_agreement or llm is None or not llm or agree:
            return Decision(rule, "rule+llm" if agree else "rule")
        return Decision(None, "none", "llm disagrees")
    if rule is not None:  # coverage gate blocked an unfamiliar wording (D10)
        if cfg.rescue_gate and agree:
            return Decision(rule, "rule+llm", "gate rescued")
        return Decision(None, "none", "gate")
    if cfg.llm_only and ls:
        (top, votes), = Counter(ls).most_common(1)
        if top is not None and votes == len(ls) and votes >= cfg.llm_only_min_votes:
            return Decision(next(i for i, s in zip(llm, ls) if s == top), "llm", f"llm {votes}/{len(ls)}")
    return Decision(None, "none", "no parse")
