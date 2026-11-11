"""eval/calibrate.py: policy replay on one parse pass. Fake models only: these check plumbing, not LLM accuracy."""
import json
import re
import sys
from pathlib import Path

import pytest

from datamate.config import LLMConfig
from datamate.parse.intent import Intent
from eval.calibrate import decided, policies

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train" / "train"
needs_data = pytest.mark.skipif(not (TRAIN / "questions.jsonl").exists(), reason="train data not available")


class P:  # minimal Parsed
    def __init__(self, it, unk, llm_its):
        self.it, self.unk, self.llm_its = it, unk, llm_its


def test_decided_rule_vs_rescue():
    it = Intent(op="revenue")
    cfgs = {k: (None if v is None else LLMConfig(**v)) for k, v in policies(1).items()}
    blocked = P(it, ["la"], [Intent(op="revenue")])
    assert decided(blocked, cfgs["rule"]) is None and decided(blocked, cfgs["rescue"]) is not None
    disagree = P(it, [], [Intent(op="order_count")])
    assert decided(disagree, cfgs["rule"]) is it and decided(disagree, cfgs["rescue+agree"]) is None
    assert decided(P(None, [], [Intent(op="revenue")]), cfgs["rescue"]) is None
    assert decided(P(None, [], [Intent(op="revenue")]), cfgs["rescue+llm_only1"]) is not None
    assert decided(None, cfgs["rescue"]) is None


def _run_main(monkeypatch, capsys, argv):
    import eval.calibrate as cal
    monkeypatch.setattr(sys, "argv", ["calibrate.py", *argv])
    cal.main()
    return capsys.readouterr().out


def _table(out):
    return {m[1]: (int(m[2]), int(m[3]), int(m[4])) for m in re.finditer(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s", out, re.M)}


@needs_data
def test_empty_replay_is_flagged_as_rule_only(tmp_path, monkeypatch, capsys):
    (tmp_path / "r.jsonl").write_text("", encoding="utf8")
    out = _run_main(monkeypatch, capsys, ["--llm-model", "models/fake", "--llm-replay", str(tmp_path / "r.jsonl"),
                                          "--only", "biz_101"])
    t = _table(out)
    assert len(set(t.values())) == 1 and "every question missed the replay" in out


@needs_data
def test_echo_model_agrees_with_rule_and_adds_no_wrong(tmp_path, monkeypatch, capsys):
    # fake model = the rule parser's own intent through the JSON contract (catalog names as mentions, exactly like the
    # few-shot targets): every rule-answered question must agree, and no policy may add a wrong answer
    from analysis.build_fewshot import mention_names
    from datamate.parse.llm import LLMParser, ReplayBackend
    from datamate.parse.llm_schema import from_intent
    from datamate.pipeline import Context, parse_question
    from eval.scorer import load_jsonl
    qs = [q for q in load_jsonl(TRAIN / "questions.jsonl") if q["business_id"] == "biz_101"]
    base = Context(root=TRAIN)
    by_q = {}
    for q in qs:
        p = parse_question(q, base)
        js = from_intent(p.it, mention_names(p.it, p.f)) if p and p.it else {"op": "none"}
        by_q[q["question"]] = json.dumps(js, ensure_ascii=False)

    class Echo:
        name = "echo"

        def complete(self, prompts, n=1, schema=None):
            return [[by_q[pr.rsplit("Câu hỏi: ", 1)[1].rsplit("\nJSON:", 1)[0]]] * n for pr in prompts]

    path = tmp_path / "echo.jsonl"
    cfg = LLMConfig(backend="replay", model_path="models/fake", replay_path=str(path), call_all=True)
    rec = Context(root=TRAIN, llm=LLMParser(ReplayBackend(path, cfg, Echo()), cfg))
    rec.stores = base.stores
    for q in qs:
        parse_question(q, rec)
    out = _run_main(monkeypatch, capsys, ["--llm-model", "models/fake", "--llm-replay", str(path), "--only", "biz_101"])
    t = _table(out)
    d = dict(kv.split("=") for kv in out.split("diagnostics: ")[1].split("\n")[0].split())
    assert d.get("replay_miss", "0") == "0" and d["agree_on_rule_answered"] == d["rule_answered"]
    assert t["rescue+agree"] == t["rule"] and all(w == 0 for _, w, _ in t.values())
