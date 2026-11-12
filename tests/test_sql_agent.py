"""Text-to-SQL agent (D26): DuckDB data layer, reply parsing, result -> answer, voting by share, repair turn,
sanity checks, hints. Model replaced by fake backends; questions here are self-written (no competition text)."""
from datetime import date
from pathlib import Path

import pytest

from datamate.agent.hints import money_hints
from datamate.agent.sql_agent import FewShot, SqlAgent, sanity
from datamate.agent.sql_prompt import EmptyResult, parse_reply, to_row
from datamate.agent.sqldb import QueryError
from datamate.config import AgentConfig

ROOT = Path(__file__).resolve().parents[1] / "data" / "train" / "train"
SHOP = "biz_101"          # dev shop, no debt (never holdout, eval/splits.json)
DEBT_SHOP = "biz_111"     # dev shop with payments / debt
needs_data = pytest.mark.skipif(not (ROOT / "data" / SHOP).exists(), reason="train data not present")


def reply(atype, sql=None):
    return f" kỳ: ...\nTYPE: {atype}\n" + (f"```sql\n{sql}\n```" if sql else "")


class Fake:
    """Returns `outputs` for the first call (k samples) and `repairs` for a later call (repair turn)."""
    name = "fake"

    def __init__(self, outputs, repairs=None):
        self.outputs, self.repairs, self.calls = outputs, repairs or [], []

    def complete(self, prompts, n=1, schema=None, free=False, max_tokens=None):
        self.calls.append((len(prompts), n))
        if len(self.calls) == 1:
            return [list(self.outputs) for _ in prompts]
        return [list(self.repairs[:n]) for _ in prompts]


# ------------------------------------------------------------------------------------------------ pure functions

def test_parse_reply():
    assert parse_reply(reply("number", "SELECT 1;")) == ("number", "SELECT 1")
    assert parse_reply("<think>TYPE: list</think>\nTYPE: string\n```sql\nSELECT 'a'\n```") == ("string", "SELECT 'a'")
    assert parse_reply(" không có dữ liệu\nTYPE: unanswerable")[0] == "unanswerable"
    assert parse_reply("TYPE: number\nSELECT 2") == ("number", "SELECT 2")    # unfenced
    assert parse_reply("no type here") == (None, None)


def test_to_row_shapes():
    assert to_row("number", ["x"], [[1234.5]]) == {"type": "number", "answer": 1234.5}
    assert to_row("number", ["x"], [[1000.0]]) == {"type": "number", "answer": 1000}
    assert to_row("percent", ["x"], [[12.345]])["answer"] == 12.35            # rulebook §9: half up
    assert to_row("date", ["d"], [[date(2026, 3, 4)]]) == {"type": "date", "answer": "2026-03-04"}
    assert to_row("list", ["n"], [["a"], ["b"]]) == {"type": "list", "answer": ["a", "b"]}
    assert to_row("table", ["a", "b"], [["x", 2.0]])["answer"] == {"columns": ["a", "b"], "rows": [["x", 2]]}
    assert to_row("boolean", ["b"], [[True]])["answer"] is True
    with pytest.raises(EmptyResult):
        to_row("number", ["x"], [])
    with pytest.raises(EmptyResult):
        to_row("number", ["x"], [[None]])
    with pytest.raises(ValueError):
        to_row("string", ["x"], [["a"], ["b"]])


def test_money_hints():
    assert money_hints("đơn trên 3tr7 là bao nhiêu") == ["'3tr7' = 3700000 đồng"]
    assert money_hints("ngày bán từ 500k trở lên") == ["'500k' = 500000 đồng"]
    assert money_hints("tháng 5/2026") == []


def test_sanity():
    assert sanity({"type": "number", "answer": 2.5}, "co bao nhieu don hang") is not None
    assert sanity({"type": "number", "answer": 3}, "co bao nhieu don hang") is None
    assert sanity({"type": "number", "answer": -5}, "doanh thu thang 5") is not None
    assert sanity({"type": "percent", "answer": -12.5}, "tang giam bao nhieu phan tram") is None


def test_fewshot_excludes_own_shop(tmp_path):
    p = tmp_path / "fs.jsonl"
    p.write_text('{"id": "a", "business_id": "s1", "op": "revenue", "question": "doanh thu hôm qua", "reply": "x"}\n'
                 '{"id": "b", "business_id": "s2", "op": "revenue", "question": "doanh thu hôm nay", "reply": "y"}\n',
                 encoding="utf8")
    fs = FewShot(p)
    assert [r["id"] for r in fs.pick("doanh thu hôm qua", "s1", 4)] == ["b"]
    assert FewShot(tmp_path / "missing.jsonl").pick("x", "s1", 4) == []


def test_agent_config_env(monkeypatch):
    monkeypatch.setenv("DATAMATE_AGENT", "sql")
    monkeypatch.setenv("DATAMATE_AGENT_TAU", "0.625")
    c = AgentConfig.from_env()
    assert c.enabled and c.kind == "sql" and c.tau == 0.625 and c.sql_samples == 8
    monkeypatch.setenv("DATAMATE_AGENT", "codegen")
    assert AgentConfig.from_env().kind == "codegen"


# ------------------------------------------------------------------------------------------------ on train data

@pytest.fixture(scope="module")
def store():
    from datamate.ingest.store import Store
    return Store.build(ROOT / "data" / SHOP)


@needs_data
def test_db_views_and_lockdown(store):
    from datamate.agent.sqldb import build_db, schema_text
    from datamate.config import DEFAULT
    db = build_db(store, DEFAULT)
    assert {"v_orders", "v_sales", "orders", "lines"} <= set(db.tables)
    assert db.has_valid
    s = schema_text(db)
    assert "v_sales" in s and "is_valid" in s
    n = db.query("SELECT COUNT(*) FROM v_orders")[1][0][0]
    assert n == len(store.orders)
    for bad in ("COPY v_orders TO 'x.csv'", "CREATE TABLE z AS SELECT 1", "SELECT 1; SELECT 2",
                "SELECT * FROM read_csv('README.md')"):
        with pytest.raises(QueryError):
            db.query(bad)
    with pytest.raises(QueryError):
        db.query("SELECT * FROM v_sales")          # more than MAX_ROWS rows
    db.close()


@needs_data
def test_debt_macros_match_executor():
    """rulebook §4: debt_at / debt_new / debt_repaid give what execute/debt.py computes."""
    from datamate.agent.sqldb import build_db
    from datamate.config import DEFAULT
    from datamate.execute import debt
    from datamate.execute.facts import facts_of
    from datamate.ingest.store import Store
    st = Store.build(ROOT / "data" / DEBT_SHOP)
    db = build_db(st, DEFAULT)
    assert db.has_debt
    f = facts_of(st, DEFAULT)
    x, a = date(2026, 3, 31), date(2026, 1, 1)
    D = debt.outstanding(f, x)
    owe, n = db.query(f"SELECT SUM(owe), COUNT(DISTINCT customer_key) FROM debt_at(DATE '{x}')")[1][0]
    assert round(owe) == round(D["owe"].sum()) and n == D["ckey"].nunique()
    assert round(db.query(f"SELECT SUM(new_debt) FROM debt_new(DATE '{a}', DATE '{x}')")[1][0][0]) == \
        round(debt.new_debt(f, a, x))
    assert round(db.query(f"SELECT SUM(amount) FROM debt_repaid(DATE '{a}', DATE '{x}')")[1][0][0]) == \
        round(debt.collected(f, a, x))
    db.close()


def _agent(backend, **kw):
    cfg = AgentConfig(enabled=True, kind="sql", sql_samples=4, tau=0.5, fewshot_k=0, **kw)
    return SqlAgent(backend, cfg, ROOT)


Q = {"id": "t1", "business_id": SHOP, "question": "tháng 5/2026 có bao nhiêu đơn", "as_of_date": "2026-08-24"}
GOOD = "SELECT COUNT(*) FROM v_orders WHERE is_valid AND order_date BETWEEN DATE '2026-05-01' AND DATE '2026-05-31'"


@needs_data
def test_agent_votes_by_share(store):
    ag = _agent(Fake([reply("number", GOOD)] * 2 + [reply("number", "SELECT 7")] + ["garbage"]))
    (row, tr), = ag.solve_shop([Q], store)
    assert row["type"] == "number" and row["answer"] > 0
    assert tr["vote"] == "2/4"
    assert tr["code"][0] == GOOD and tr["output"][3]["status"] == "no_sql"
    ag.close()
    # 1 of 4 agree: share 0.25 < tau -> abstain
    ag = _agent(Fake([reply("number", GOOD), reply("number", "SELECT 7"), "x", "y"]))
    (row, tr), = ag.solve_shop([Q], store)
    assert row is None
    ag.close()


@needs_data
def test_agent_prompt_has_hints(store):
    ag = _agent(Fake([]))
    db, schema, readme, lk = ag._shop(SHOP, store)
    p = ag.prompt(Q, schema, readme, lk)
    assert "2026-05-01" in p and "2026-05-31" in p and "v_orders" in p
    ag.close()


@needs_data
def test_agent_repair_turn_and_empty(store):
    bad = reply("number", "SELECT COUNT(*) FROM no_such_table")
    ag = _agent(Fake([bad] * 4, repairs=[reply("number", GOOD)]))
    (row, tr), = ag.solve_shop([Q], store)
    assert ag.backend.calls == [(1, 4), (4, 1)]          # one batched repair call for the 4 failed samples
    assert row is not None and tr["vote"] == "4/4" and ag.repaired == 4
    ag.close()
    empty = reply("number", "SELECT SUM(revenue) FROM v_orders WHERE order_date = DATE '1990-01-01'")
    ag = _agent(Fake([empty] * 4))
    (row, tr), = ag.solve_shop([Q], store)
    assert row is None and {o["status"] for o in tr["output"]} == {"empty"}
    ag.close()


@needs_data
def test_agent_unanswerable_needs_flag(store):
    ag = _agent(Fake([reply("unanswerable")] * 4))
    (row, tr), = ag.solve_shop([Q], store)
    assert row is None and "unanswerable->abstain" in tr["vote"]
    ag.close()
