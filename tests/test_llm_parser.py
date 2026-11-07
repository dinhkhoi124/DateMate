"""D14 LLM parser contract + ensemble policy, with a fake backend (no GPU, no model)."""
import json
from datetime import date
from types import SimpleNamespace

from datamate.config import LLMConfig
from datamate.link.catalog import Dictionary
from datamate.parse.ensemble import decide, needs_llm
from datamate.parse.intent import Intent
from datamate.parse.llm import LLMParser, ReplayBackend, build_prompt, parse_json
from datamate.parse.llm_schema import OPS, from_intent, json_schema, signature, to_intent
from datamate.timeparse.periods import Period

MAY = [Period(date(2026, 5, 1), date(2026, 5, 31), "month", "2026-05")]


def _lk():
    e = Dictionary([])
    return SimpleNamespace(products=Dictionary([("Cà phê sữa đá", "P1"), ("Trà đào", "P2"), ("Trà Đào", "P3")]),
                           customers=Dictionary([("Chị Lan", "C1")]), staff=e, categories=e, ingredients=e,
                           suppliers=e)


class Fake:
    name = "fake"

    def __init__(self, outputs):
        self.outputs, self.calls = outputs, 0

    def complete(self, prompts, n=1):
        self.calls += 1
        return [list(self.outputs) for _ in prompts]


def test_schema_covers_every_op():
    from datamate.execute.dispatch import HANDLERS
    assert set(OPS) - {"need_cap", "unanswerable"} <= set(HANDLERS)
    enum = json_schema()["properties"]["op"]["enum"]
    assert "none" in enum and set(OPS) <= set(enum)


def test_mention_links_to_catalog_key_without_diacritics():
    it = to_intent({"op": "qty", "filters": {"product": "ca phe sua da"}, "args": {"unit": "ly"}}, MAY, _lk())
    assert it.flt == {"product": "P1"} and it.args == {"unit": "ly"} and it.periods == MAY


def test_ambiguous_or_unknown_mention_is_unresolved():
    lk = _lk()
    assert to_intent({"op": "revenue", "filters": {"product": "trà đào"}, "args": {}}, MAY, lk).flt["product"] \
        .startswith("?")  # two keys share the name: executor must abstain
    assert to_intent({"op": "revenue", "filters": {"product": "bánh mì"}, "args": {}}, MAY, lk).flt["product"] \
        .startswith("?")


def test_none_and_out_of_contract():
    lk = _lk()
    assert to_intent({"op": "none", "filters": {}, "args": {}}, MAY, lk) is None
    assert to_intent({"op": "revenue", "filters": {"colour": "red"}, "args": {}}, MAY, lk) is None
    assert to_intent({"op": "need_cap", "filters": {}, "args": {}}, MAY, lk) is None
    # args the op does not take are ignored, not trusted
    assert to_intent({"op": "revenue", "filters": {}, "args": {"k": 5}}, MAY, lk).args == {}


def test_round_trip_and_defaults():
    lk = _lk()
    rule = Intent(op="top_products", periods=MAY, args={"k": 1, "by": "qty"}, atype="string")
    js = from_intent(rule)
    assert js == {"op": "top_products", "filters": {}, "args": {"by_product": "qty"}}
    assert signature(to_intent(js, MAY, lk)) == signature(rule)
    # executor default metric=revenue: absent and explicit are the same meaning
    a = Intent(op="growth", periods=MAY, args={"compare": "prev"}, atype="percent")
    b = Intent(op="growth", periods=MAY, args={"compare": "prev", "metric": "revenue"}, atype="percent")
    assert signature(a) == signature(b)
    rule = Intent(op="qty", periods=MAY, flt={"product": "P1"}, args={"unit": "ly"})
    assert signature(to_intent(from_intent(rule, {"product": "Cà phê sữa đá"}), MAY, lk)) == signature(rule)


def test_parse_json_tolerates_reasoning_and_garbage():
    assert parse_json('<think>hmm {"x":1}</think> {"op": "revenue"}') == {"op": "revenue"}
    assert parse_json("no json") is None
    assert parse_json('{"op": ') is None


def test_prompt_has_contract_and_examples():
    p = build_prompt("Doanh thu tháng 5?", [{"question": "Tháng này bán bao nhiêu đơn?",
                                            "target": {"op": "order_count", "filters": {}, "args": {}}}])
    assert "order_count" in p and "KHÔNG tự tính số" in p and p.endswith("Câu hỏi: Doanh thu tháng 5?\nJSON:")


CFG = LLMConfig(backend="replay")
RULE = Intent(op="revenue", periods=MAY)
SAME = Intent(op="revenue", periods=MAY)
OTHER = Intent(op="order_count", periods=MAY)


def test_default_policy_keeps_rule_answers_and_rescues_gate():
    assert not needs_llm(RULE, True, CFG)                       # gate passed: no model call
    assert decide(RULE, True, None, CFG).intent is RULE
    assert needs_llm(RULE, False, CFG)
    assert decide(RULE, False, [SAME], CFG).note == "gate rescued"
    assert decide(RULE, False, [OTHER], CFG).intent is None
    assert decide(RULE, False, [SAME, OTHER], CFG).intent is None   # every sample must agree
    assert decide(RULE, False, [], CFG).intent is None              # model failed -> gate stands
    assert decide(None, True, [SAME], CFG).intent is None           # llm_only off by default


def test_strict_policies():
    strict = LLMConfig(backend="replay", require_agreement=True, llm_only=True, llm_only_min_votes=2)
    assert decide(RULE, True, [OTHER], strict).intent is None
    assert decide(RULE, True, [], strict).intent is RULE            # model failure never costs a rule answer
    assert decide(None, True, [SAME], strict).intent is None        # 1 vote < 2
    d = decide(None, True, [SAME, Intent(op="revenue", periods=MAY)], strict)
    assert d.intent is not None and d.source == "llm"


def test_parser_with_fake_backend_and_replay(tmp_path):
    out = json.dumps({"op": "qty", "filters": {"product": "Cà phê sữa đá"}, "args": {"unit": "ly"}})
    cfg = LLMConfig(backend="replay", model_id="fake", replay_path=str(tmp_path / "r.jsonl"))
    fake = Fake([out])
    rec = ReplayBackend(tmp_path / "r.jsonl", cfg, inner=fake)
    p = LLMParser(rec, cfg, examples=[])
    its, raw = p.parse("Tháng 5 bán bao nhiêu ly cà phê sữa đá?", MAY, _lk())
    assert raw == [out] and its[0].flt == {"product": "P1"}
    p.parse("Tháng 5 bán bao nhiêu ly cà phê sữa đá?", MAY, _lk())
    assert fake.calls == 1                                          # second call served from cache
    replay = LLMParser(ReplayBackend(tmp_path / "r.jsonl", cfg), cfg, examples=[])
    assert replay.parse("Tháng 5 bán bao nhiêu ly cà phê sữa đá?", MAY, _lk())[1] == [out]
    assert replay.parse("Câu khác?", MAY, _lk()) == ([], [])        # miss -> no answer, never a guess
    assert (p.calls, p.failed) == (2, 0) and replay.stats_line() == "llm_calls=2 llm_failed=1"


# --------------------------------------------------------------------------- pipeline, on a dev shop (never holdout)
def test_pipeline_gate_rescue_on_dev_shop():
    from pathlib import Path

    import pytest

    from datamate.pipeline import Context, analyse, trace_of
    root = Path("data/train/train")
    if not (root / "data").exists():
        pytest.skip("train data not available")
    q = {"id": "t", "business_id": "biz_106", "as_of_date": "2026-09-14"}
    base = analyse({**q, "question": "Doanh thu tháng 8/2026 là bao nhiêu?"}, Context(root=root))[2]
    assert base["type"] == "number"
    odd = {**q, "question": "Doanh thu tháng 8/2026 là bao nhiêu vậy xyzzyq?"}   # unknown word -> gate (D10)
    assert analyse(odd, Context(root=root))[2]["type"] == "abstain"
    cfg = LLMConfig(backend="replay")
    agree = LLMParser(Fake([json.dumps({"op": "revenue", "filters": {}, "args": {}})]), cfg, examples=[])
    it, t, row = analyse(odd, Context(root=root, llm=agree))
    assert row == {**base, "id": "t"} and it.source == "rule+llm"
    assert trace_of(t, it, row)["thinking"] == ['{"op": "revenue", "filters": {}, "args": {}}']
    differ = LLMParser(Fake([json.dumps({"op": "order_count", "filters": {}, "args": {}})]), cfg, examples=[])
    assert analyse(odd, Context(root=root, llm=differ))[2]["type"] == "abstain"


def test_call_all_records_every_question_without_changing_decisions(monkeypatch):
    # D15: record mode calls the model everywhere, the default policy still decides the same way
    rule = Intent(op="revenue", periods=MAY)
    rec = LLMConfig(call_all=True)
    assert needs_llm(rule, True, rec) and needs_llm(None, False, rec)
    assert not needs_llm(rule, True, LLMConfig())
    other = Intent(op="order_count", periods=MAY)
    assert decide(rule, True, [other], rec).intent is rule
    assert decide(None, False, [other], rec).intent is None


def test_gpu_env_overrides(monkeypatch):
    monkeypatch.setenv("DATAMATE_LLM_TP", "2")
    monkeypatch.setenv("DATAMATE_LLM_DTYPE", "float16")
    monkeypatch.setenv("DATAMATE_LLM_CALL_ALL", "1")
    c = LLMConfig.from_env()
    assert (c.tensor_parallel_size, c.dtype, c.call_all) == (2, "float16", True)
