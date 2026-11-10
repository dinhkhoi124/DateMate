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
    agent = CodeGenAgent(FakeBackend([COUNT_ORDERS] * 4 + [other]), AgentConfig(enabled=True, min_votes=4), ROOT)
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


def test_exec_solve_tolerates_import_of_params_and_strptime():
    # Kaggle run 09/11: 118/168 sample errors were `import H`; pd.to_datetime(format=) imports _strptime lazily
    code = ("import H\nimport pandas as pd\n"
            "def solve(T, H):\n"
            "    d = pd.to_datetime(pd.Series(['01/02/2025']), format='%d/%m/%Y')\n"
            "    return {'type': 'number', 'answer': int(d.dt.month[0]) + H}\n")
    assert exec_solve(code, {}, 1) == {"type": "number", "answer": 3}
    with pytest.raises(ImportError):
        exec_solve("from H import x\ndef solve(T, H):\n    return 1\n", {}, None)


def test_schema_lists_text_values_and_drops_empty_columns():
    import pandas as pd

    from datamate.agent.tables import schema_text
    df = pd.DataFrame({"status": pd.Series(["done", "cancelled", "done"], dtype="str"), "debt": [None] * 3})
    df = df[[c for c in df.columns if df[c].notna().any()]]
    s = schema_text({"orders": df})
    assert "'done' (2) | 'cancelled' (1)" in s and "debt" not in s


def test_helpers_find_products_raw_and_partial_names():
    # Kaggle v2: normalize() turns 'gas' into 'ga', so names passed by the model (or the question) missed the catalog
    from datetime import date

    from datamate.agent.helpers import Helpers
    from datamate.link.catalog import Dictionary

    class Lk:
        products = Dictionary([("Bộ van dây gas Namilux", "SP1"), ("Thép cuộn phi 6 Hòa Phát", "SP2"),
                               ("Thép cuộn phi 8 Hòa Phát", "SP3")])
    h = Helpers(date(2026, 1, 1), "so bo van day ga namilux", None, None, Lk(), None, "số bộ van dây gas Namilux")
    assert [r["key"] for r in h.find_products()] == ["SP1"]
    assert [r["key"] for r in h.find_products("Bộ van dây gas Namilux")] == ["SP1"]
    assert [r["key"] for r in h.find_products("thép cuộn phi 6")] == ["SP2"]
    hits = h.find_products("thép cuộn")
    assert [r["key"] for r in hits] == ["SP2", "SP3"] and all(r["ambiguous"] for r in hits)


def test_in_period_scalar_and_series():
    import pandas as pd
    from datetime import date
    from datamate.agent.helpers import Helpers
    p = {"start": date(2026, 7, 1), "end": date(2026, 7, 31)}
    assert Helpers.in_period(pd.Timestamp("2026-07-31 18:30"), p) is True     # .apply(lambda x: H.in_period(x, p))
    assert Helpers.in_period(date(2026, 8, 1), p) is False
    assert Helpers.in_period(None, p) is False
    s = pd.Series(pd.to_datetime(["2026-06-30", "2026-07-01", None]))
    assert Helpers.in_period(s, p).tolist() == [False, True, False]


def test_merge_coalesces_identical_overlap():
    import pandas as pd
    from datamate.agent.sandbox import coalesce_suffixes
    orders = pd.DataFrame({"order_id": [1, 2], "order_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                           "status": ["done", "cancelled"]})
    lines = pd.DataFrame({"order_id": [1, 1, 2], "order_date": pd.to_datetime(["2026-01-01"] * 2 + ["2026-01-02"]),
                          "status": ["ok", "ok", "ok"], "amount": [5, 7, 9]})
    m = coalesce_suffixes(lines.merge(orders, on="order_id"))
    assert "order_date" in m.columns and "order_date_x" not in m.columns   # same values -> one column
    assert {"status_x", "status_y"} <= set(m.columns)                       # differing values stay apart
    left = coalesce_suffixes(lines.merge(orders.iloc[:1], on="order_id", how="left"))
    assert left["order_date"].notna().all()                                 # NaN side filled from the other


@needs_data
def test_sandbox_merge_overlap_runs():
    code = """
def solve(T, H):
    m = T["lines"].merge(T["orders"], on="order_id")
    return {"type": "number", "answer": int(m["order_date"].notna().sum())}
"""
    sb = ShopSandbox(ROOT, SHOP, ROOT / "rulebook.md", 60)
    try:
        status, val = sb.run(Job(code, "2026-01-01", "x"))
    finally:
        sb.close()
    assert status == "ok", val


def test_resubmit_official_vs_experiment(tmp_path, monkeypatch):
    """D21: official run takes policy/m only from AgentConfig defaults; --experiment refuses a split without answers."""
    import sys
    from eval import resubmit_agent
    (tmp_path / "questions.jsonl").write_text('{"id": "a", "business_id": "b"}\n', encoding="utf8")
    log = tmp_path / "log.jsonl"
    log.write_text('{"id": "a", "inputs": {"rule_answer": {"type": "abstain"}}, "output": []}\n', encoding="utf8")
    base = ["resubmit_agent.py", str(log), "--data", str(tmp_path), "--out", str(tmp_path / "s.jsonl")]
    for extra in (["--m", "3"], ["--experiment"]):
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(SystemExit):
            resubmit_agent.main()
    monkeypatch.setattr(sys, "argv", base)
    resubmit_agent.main()
    assert (tmp_path / "s.jsonl").read_text(encoding="utf8").strip() == '{"id": "a", "type": "abstain"}'
    assert '"min_votes": 2' in (tmp_path / "s.jsonl.meta.json").read_text(encoding="utf8")
