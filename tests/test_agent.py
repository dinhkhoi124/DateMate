"""Code-gen agent (D20): answer canon/vote, code extraction, sandbox, voting and rule x code-gen policy.
Model replaced by a fake backend; questions here are self-written (no competition text)."""
from pathlib import Path

import pytest

from datamate.agent.agent import CodeGenAgent, combine
from datamate.agent.answer import canon, vote_key
from datamate.agent.helpers import round_half_up
from datamate.agent.prompt import extract_code
from datamate.agent.sandbox import Job, ShopSandbox, exec_solve
from datamate.config import AgentConfig

ROOT = Path(__file__).resolve().parents[1] / "data" / "train" / "train"
SHOP = "biz_101"   # dev shop (never holdout, eval/splits.json)
needs_data = pytest.mark.skipif(not (ROOT / "data" / SHOP).exists(), reason="train data not present")

COUNT_ORDERS = """```python
def solve(T, H):
    return {"type": "number", "answer": len(T["orders"])}
```"""


class FakeBackend:
    name = "fake"

    def __init__(self, outputs):
        self.outputs = outputs

    def complete(self, prompts, n=1, schema=None, free=False, max_tokens=None):
        return [list(self.outputs) for _ in prompts]


def test_canon_types():
    assert canon({"type": "number", "answer": 12.0}) == {"type": "number", "answer": 12}
    assert canon({"type": "percent", "answer": 12.345}) == {"type": "percent", "answer": 12.35}
    assert canon({"type": "date", "answer": "2026-08"}) == {"type": "date", "answer": "2026-08"}
    assert canon({"type": "list", "answer": ("a", 2.50)}) == {"type": "list", "answer": ["a", 2.5]}
    assert canon({"type": "unanswerable"})["type"] == "unanswerable"
    for bad in ({"type": "number", "answer": float("nan")}, {"type": "boolean", "answer": 1}, {"answer": 3}, 7):
        with pytest.raises(ValueError):
            canon(bad)


def test_vote_key_matches_scorer_tolerance():
    a = {"type": "string", "answer": "Cà phê sữa"}
    b = {"type": "string", "answer": "cà phê  sữa."}
    assert vote_key(a) == vote_key(b)
    t1 = {"type": "table", "answer": {"columns": ["x", "y"], "rows": [["a", 1], ["b", 2]]}}
    t2 = {"type": "table", "answer": {"columns": ["x", "y"], "rows": [["b", 2], ["a", 1]]}}
    assert vote_key(t1) == vote_key(t2)                     # row order does not matter for tables
    l1, l2 = {"type": "list", "answer": ["a", "b"]}, {"type": "list", "answer": ["b", "a"]}
    assert vote_key(l1) != vote_key(l2)                     # but does for lists


def test_round_half_up():
    assert round_half_up(2.5) == 3 and round_half_up(12.345, 2) == 12.35 and round_half_up(-2.5) == -3


def test_extract_code():
    assert "def solve" in extract_code("<think>def solve(x): pass</think>\n" + COUNT_ORDERS)
    assert extract_code("no code here") is None


def test_exec_solve_blocks_imports_and_files():
    with pytest.raises(ImportError):
        exec_solve("import os\ndef solve(T, H):\n    return 1", {}, None)
    with pytest.raises(NameError):
        exec_solve("def solve(T, H):\n    return open('x')", {}, None)


def test_combine_policies():
    rule = {"id": "q", "type": "number", "answer": 5}
    same, other = {"type": "number", "answer": 5}, {"type": "number", "answer": 6}
    assert combine(rule, same, "agree") == (rule, "rule+codegen")
    assert combine(rule, other, "agree")[1] == "disagree"
    assert combine(rule, None, "agree") == (rule, "rule")
    ab = {"id": "q", "type": "abstain"}
    assert combine(ab, other, "agree") == ({"id": "q", **other}, "codegen")
    assert combine(rule, other, "rule_first") == (rule, "rule")
    assert combine(rule, None, "codegen")[0]["type"] == "abstain"


@needs_data
def test_sandbox_ok_error_timeout():
    sb = ShopSandbox(ROOT, SHOP, timeout_s=5)
    try:
        st, val = sb.run(Job(extract_code(COUNT_ORDERS), "2026-08-24", "x"))
        assert st == "ok" and val["type"] == "number" and val["answer"] > 0
        assert sb.run(Job("def solve(T, H):\n    return 1/0", "2026-08-24", "x"))[0] == "error"
        assert sb.run(Job("def solve(T, H):\n    while True: pass", "2026-08-24", "x"))[0] == "timeout"
        assert sb.run(Job(extract_code(COUNT_ORDERS), "2026-08-24", "x"))[0] == "ok"   # respawned after kill
    finally:
        sb.close()


@needs_data
def test_agent_votes():
    from datamate.ingest.store import Store
    q = {"id": "t1", "business_id": SHOP, "question": "shop có tổng cộng bao nhiêu đơn", "as_of_date": "2026-08-24"}
    st = Store.build(ROOT / "data" / SHOP)
    other = COUNT_ORDERS.replace('len(T["orders"])', "1")
    agent = CodeGenAgent(FakeBackend([COUNT_ORDERS] * 4 + [other]), AgentConfig(enabled=True), ROOT)
    try:
        (row, tr), = agent.solve_shop([q], st)
        assert row == {"type": "number", "answer": len(st.orders)} and tr["vote"] == "4/5"
        assert len(tr["thinking"]) == 5 and len(tr["code"]) == 5      # raw outputs logged (Docker Guide)
        agent.backend = FakeBackend([COUNT_ORDERS] * 3 + [other] * 2)
        (row, tr), = agent.solve_shop([q], st)
        assert row is None and tr["vote"] == "3/5"                    # below min_votes -> abstain
        p = agent.prompt(q, "README", "SCHEMA")
        assert "=== RULEBOOK ===" in p and p.rstrip().endswith(q["question"])
    finally:
        agent.close()
