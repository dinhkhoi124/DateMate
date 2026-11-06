"""Intent schema shared by the rule parser and (later) the LLM parser (docs/ARCHITECTURE.md §4)."""
from __future__ import annotations

from dataclasses import dataclass, field

from datamate.timeparse.periods import Period


@dataclass
class Intent:
    op: str                                   # see execute/dispatch.py OPS
    periods: list[Period] = field(default_factory=list)
    flt: dict = field(default_factory=dict)   # product, category, ingredient, rx, size, base_unit, group, channel, pay, staff, shift, customer
    args: dict = field(default_factory=dict)  # k, n, threshold, unit, metric, compare, series...
    atype: str = "number"
    confidence: float = 1.0
    note: str = ""

    def key(self) -> tuple:
        """Comparable signature (for rule/LLM agreement)."""
        ps = tuple((p.start, p.end) for p in self.periods)
        return (self.op, ps, tuple(sorted((k, str(v)) for k, v in self.flt.items() if v is not None)),
                tuple(sorted((k, str(v)) for k, v in self.args.items())), self.atype)
