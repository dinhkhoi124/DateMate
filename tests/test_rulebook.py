"""D18 run-time rulebook diff. Part 1 uses a made-up mini rulebook (runs in CI); part 2 mutates the real train rulebook
read at test time (no rulebook text in this file) and checks the pipeline on one dev shop (never holdout)."""
import json
from datetime import date
from pathlib import Path

import pytest

from datamate.config import DEFAULT
from datamate.parse.intent import Intent
from datamate.rulebook import INDEX_PATH, analyse, build_index, fold, parse_rows
from datamate.timeparse.periods import Period

ROOT = Path(__file__).resolve().parents[1]
TRAIN_RB = ROOT / "data" / "train" / "train" / "rulebook.md"

# Written for this test (structure like the train rulebook, wording our own).
_T = ["| Mục {i} | nội dung thời gian số {i} |".format(i=i) for i in range(1, 13)]
_T[5] = "| Tuần | tuần bắt đầu thứ Hai, kết thúc Chủ nhật |"
_T[10] = "| Tết | từ 23 tháng Chạp đến hết mùng 10 tháng Giêng |"
_T[11] = "| Ca | ca sáng 06:00–13:59, ca chiều 14:00–21:59 |"
MINI = "\n".join([
    "# Rulebook thử", "", "Đoạn mở đầu tự viết.", "",
    "## 1. Đơn", "| Luật | Mặc định |", "|---|---|",
    "| Đơn hợp lệ | không huỷ |", "| Câu tiền | là doanh thu |", "| Kỳ | theo ngày đặt |",
    "| Đơn huỷ | trạng thái **huỷ** |", "",
    "## 2. Thời gian", "| Cách nói | Nghĩa |", "|---|---|", *_T, "",
    "## 9. Đáp án", "Mỗi dòng một câu.", "| type | Format |", "|---|---|", "| number | số |", "",
])
IDX = build_index(MINI)


def _sub(md, old, new):
    assert md.count(old) == 1, old
    return md.replace(old, new)


def test_mini_unchanged_is_empty():
    d = analyse(MINI, IDX)
    assert d.rules == DEFAULT and not (d.changed or d.missing or d.blocked or d.keywords or d.reported)


def test_param_rows_reread_when_only_numbers_change():
    # rulebook §2: ca sáng/chiều, Tết, đầu tuần are parameters
    md = _sub(MINI, "ca sáng 06:00–13:59, ca chiều 14:00–21:59", "ca sáng 07:00–12:59, ca chiều 13:00–20:59")
    md = _sub(md, "từ 23 tháng Chạp đến hết mùng 10", "từ 20 tháng Chạp đến hết mùng 7")
    md = _sub(md, "bắt đầu thứ Hai, kết thúc Chủ nhật", "bắt đầu Chủ nhật, kết thúc thứ Bảy")
    d = analyse(md, IDX)
    assert sorted(d.params) == ["§2.11", "§2.12", "§2.6"] and not d.blocked
    r = d.rules
    assert r.morning == (7 * 3600, 12 * 3600 + 59 * 60 + 59) and r.afternoon[0] == 13 * 3600
    assert r.tet_start == (20, 12) and r.tet_end == (7, 1) and r.week_start == 6


def test_param_row_with_other_wording_is_blocked_not_guessed():
    d = analyse(_sub(MINI, "ca chiều 14:00–21:59", "ca chiều 14:00–21:59, ca tối sau đó"), IDX)
    assert d.rules == DEFAULT and d.blocked == {"[shift]": ["§2.12"], "shift_compare": ["§2.12"]}


def test_definition_change_blocks_only_its_scope():
    # rulebook §1: đơn huỷ definition changed -> cancel ops abstain, revenue untouched
    d = analyse(_sub(MINI, "trạng thái **huỷ**", "trạng thái **huỷ** hoặc **hoàn**"), IDX)
    assert d.changed == ["§1.4"] and set(d.blocked) == {"cancel_count", "cancel_rate"}
    may = [Period(date(2026, 5, 1), date(2026, 5, 31), "month", "2026-05")]
    assert d.block_reason(Intent(op="cancel_rate", periods=may), "ti le huy") == "rulebook changed: §1.4"
    assert d.block_reason(Intent(op="revenue", periods=may), "doanh thu") is None


def test_period_kind_scope_and_missing_row():
    d = analyse(_sub(MINI, "| Mục 4 | nội dung thời gian số 4 |\n", ""), IDX)   # §2.4 (last_n) removed
    assert d.missing == ["§2.4"] and d.blocked == {"period:last_n": ["§2.4"]}
    p = Period(date(2026, 5, 1), date(2026, 5, 7), "last_n")
    assert d.block_reason(Intent(op="revenue", periods=[p]), "7 ngay") == "rulebook changed: §2.4"
    assert d.block_reason(Intent(op="revenue", periods=[Period(p.start, p.end, "month")]), "") is None


def test_new_row_keywords_are_train_unseen_phrases_only():
    md = _sub(MINI, "| Đơn huỷ | trạng thái **huỷ** |", "| Đơn huỷ | trạng thái **huỷ** |\n"
              "| Phí bảo hành | tính riêng, không vào **doanh thu**, gồm **gói gia hạn** |")
    d = analyse(md, IDX)
    assert d.new_rows == 1 and d.keywords == ["goi gia han", "phi bao hanh"]   # 'doanh thu' is a train phrase
    assert d.block_reason(Intent(op="revenue"), fold("Phí bảo hành tháng 5?")) == "rulebook new rule"
    assert d.block_reason(Intent(op="revenue"), fold("Doanh thu tháng 5?")) is None


def test_global_sections_are_reported_not_blocked():
    md = _sub(_sub(MINI, "| number | số |", "| number | số nguyên |"), "Đoạn mở đầu tự viết.", "Đoạn khác.")
    d = analyse(md, IDX)
    assert sorted(d.reported) == ["§0.p1", "§9.1"] and not d.blocked


def test_parse_rows_ids():
    ids = [r.rid for r in parse_rows(MINI)]
    assert ids[:2] == ["§0.p1", "§1.1"] and "§2.12" in ids and ids[-2:] == ["§9.p1", "§9.1"]


# --------------------------------------------------------------------------- real train rulebook
needs_data = pytest.mark.skipif(not TRAIN_RB.exists(), reason="train data not available")


@needs_data
def test_committed_index_matches_train_rulebook():
    idx = build_index(TRAIN_RB.read_text(encoding="utf8"))
    assert json.loads(INDEX_PATH.read_text(encoding="utf8")) == idx, "rerun analysis/build_rulebook_index.py"
    assert not analyse(TRAIN_RB.read_text(encoding="utf8")).changed


def _mutate(md: str, rid: str, f) -> str:
    """Apply f to the markdown line of row `rid` (located by the parser, so no rulebook text is needed here)."""
    row = next(r for r in parse_rows(md) if r.rid == rid)
    lines = md.splitlines()
    lines[row.line - 1] = f(lines[row.line - 1])
    return "\n".join(lines)


@needs_data
def test_train_shift_hours_reread():
    import re
    md = _mutate(TRAIN_RB.read_text(encoding="utf8"), "§2.12", lambda s: re.sub(r"06:00", "07:00", s, count=1))
    d = analyse(md)
    assert d.params == ["§2.12"] and d.rules.morning[0] == 7 * 3600 and not d.blocked


@needs_data
def test_pipeline_revenue_definition_change_on_dev_shop(tmp_path):
    # rulebook §1: doanh thu một đơn changed -> revenue-family ops abstain, every other answer unchanged
    from datamate.pipeline import Context, analyse as run_q
    from datamate.rulebook import SCOPES
    from eval.scorer import load_jsonl
    md = _mutate(TRAIN_RB.read_text(encoding="utf8"), "§1.6", lambda s: s.rstrip(" |") + " (đổi) |")
    (tmp_path / "rulebook.md").write_text(md, encoding="utf8")
    train = TRAIN_RB.parent
    qs = [q for q in load_jsonl(train / "questions.jsonl") if q["business_id"] == "biz_101"]
    base, mut = Context(root=train), Context(root=train, rulebook=tmp_path / "rulebook.md")
    mut.stores = base.stores
    n_blocked = 0
    for q in qs:
        it, _, row = run_q(q, base)
        it2, _, row2 = run_q(q, mut)
        if it is not None and it.op in SCOPES["§1.6"]:
            assert row2["type"] == "abstain" and it2.note == "rulebook changed: §1.6"
            n_blocked += 1
        else:
            assert row2 == row
    assert n_blocked > 0


# --------------------------------------------------------------------------- D19 rule judge (fake backend only)
class FakeJudge:
    """Judge prompts (schema given): `verdict(prompt) -> list of raw outputs`. Intent prompts: an unusable intent."""
    name = "fake"

    def __init__(self, verdict, raise_exc=False):
        self.verdict, self.raise_exc, self.prompts = verdict, raise_exc, []

    def complete(self, prompts, n=1, schema=None):
        if schema is None:
            return [["{}"] for _ in prompts]
        if self.raise_exc:
            raise RuntimeError("CUDA error")
        self.prompts += prompts
        return [self.verdict(p) for p in prompts]


def _always(*verdicts):
    return lambda p: [json.dumps({"verdict": v}) for v in verdicts]


def _cfg(n=1, **kw):
    from datamate.config import LLMConfig
    return LLMConfig(backend="replay", model_id="fake", n_samples=n, **kw)


def _two_changed():
    md = _sub(MINI, "trạng thái **huỷ**", "trạng thái **huỷ** hoặc **hoàn**")             # §1.4 cancel ops
    md = _sub(md, "| Kỳ | theo ngày đặt |", "| Kỳ | theo ngày giao |")                      # §1.3 cancel ops too
    return analyse(md, IDX)


def test_every_scoped_row_has_an_impl_description():
    from datamate.rulebook import GLOBAL, SCOPES
    from datamate.rulebook_impl import IMPL
    assert sorted(k for k, v in SCOPES.items() if v != GLOBAL) == sorted(IMPL)


def test_judge_default_off():
    from datamate.config import LLMConfig
    assert LLMConfig().rule_judge is False


@pytest.mark.parametrize("outputs, n, ok, outcome", [
    (['{"verdict": "same"}'], 1, True, "same"),
    (['{"verdict": "same"}'] * 2 + ['{"verdict": "different"}'], 3, False, "different"),
    (['{"verdict": "unsure"}'], 1, False, "unsure"),
    (['{"verdict": "same"}'] * 2, 3, False, "failed"),           # fewer samples than requested
    ([], 1, False, "failed"),                                    # backend error / replay miss
    (["not json"], 1, False, "invalid"),
    (['{"verdict": "SAME"}'], 1, False, "invalid"),              # outside the enum
    (['{"verdict": "different"} {"verdict": "same"}'], 1, False, "invalid"),
])
def test_judge_fails_closed(outputs, n, ok, outcome):
    from datamate.parse.rule_judge import RuleJudge
    d = analyse(_sub(MINI, "trạng thái **huỷ**", "trạng thái **huỷ** hoặc **hoàn**"), IDX)
    it = Intent(op="cancel_rate")
    j = RuleJudge(FakeJudge(lambda p: outputs), _cfg(n))
    assert j.judge("q", it, d.hits(it, "q"), d)[0] is ok and j.outcomes == {outcome: 1}


def test_judge_backend_exception_abstains():
    from datamate.parse.rule_judge import RuleJudge
    d = _two_changed()
    it = Intent(op="cancel_rate")
    j = RuleJudge(FakeJudge(_always("same"), raise_exc=True), _cfg(1))
    assert j.judge("q", it, d.hits(it, "q"), d) == (False, []) and j.outcomes == {"failed": 1}


def test_judge_one_prompt_per_row_and_one_unsafe_row_abstains():
    from datamate.parse.rule_judge import RuleJudge
    d = _two_changed()
    it = Intent(op="cancel_rate")
    assert d.hits(it, "q") == ["§1.3", "§1.4"]
    fake = FakeJudge(lambda p: ['{"verdict": "different"}'] if "ngày giao" in p else ['{"verdict": "same"}'])
    j = RuleJudge(fake, _cfg(1))
    ok, raw = j.judge("q", it, d.hits(it, "q"), d)
    assert not ok and len(fake.prompts) == 2 and len(raw) == 2 and j.outcomes == {"different": 1}
    assert RuleJudge(FakeJudge(_always("same")), _cfg(1)).judge("q", it, d.hits(it, "q"), d)[0] is True


def test_judge_missing_train_row_abstains_without_calling_the_model():
    from datamate.parse.rule_judge import RuleJudge
    d = analyse(_sub(MINI, "| Đơn huỷ | trạng thái **huỷ** |\n", ""), IDX)                # §1.4 dropped
    it = Intent(op="cancel_rate")
    fake = FakeJudge(_always("same"))
    j = RuleJudge(fake, _cfg(1))
    assert j.judge("q", it, d.hits(it, "q"), d) == (False, []) and not fake.prompts and j.outcomes == {"missing": 1}


def test_judge_prompt_carries_the_split_wording_read_at_run_time():
    from datamate.parse.rule_judge import judge_prompt
    md = _sub(MINI, "| Đơn huỷ | trạng thái **huỷ** |", "| Đơn huỷ | trạng thái **huỷ** hoặc **hoàn** | ghi chú cột ba |")
    md = _sub(md, "| Đơn hợp lệ | không huỷ |", "| Đơn hợp lệ | không huỷ |\n| Phí bảo hành | tính riêng |")
    d = analyse(md, IDX)
    it = Intent(op="cancel_rate")
    p = judge_prompt("q", it, "§1.4", d)
    assert "Đơn huỷ | trạng thái huỷ hoặc hoàn | ghi chú cột ba" in p      # every cell, as written in the split
    assert "Đơn huỷ là đơn có trạng thái huỷ" in p and "GIỐNG HỆT" in p     # executor meaning + strict criterion
    assert "Phí bảo hành" not in p                                         # one row per prompt
    hits = d.hits(it, fold("tỉ lệ huỷ phí bảo hành"))
    assert hits == ["new:§1.2", "§1.4"] and "Phí bảo hành | tính riêng" in judge_prompt("q", it, "new:§1.2", d)


@needs_data
@pytest.mark.parametrize("verdict", ["same", "different"])
def test_pipeline_judge_on_cosmetic_rulebook(tmp_path, verdict):
    # fake judge (no model): 'same' -> blocked answers identical to the train-rulebook run; otherwise abstain
    from datamate.parse.llm import LLMParser
    from datamate.pipeline import Context, analyse as run_q
    from eval.scorer import load_jsonl
    from eval.synth.rulebooks import variants
    (tmp_path / "rulebook.md").write_text(variants(TRAIN_RB.read_text(encoding="utf8"))["cosmetic"], encoding="utf8")
    train = TRAIN_RB.parent
    qs = [q for q in load_jsonl(train / "questions.jsonl") if q["business_id"] == "biz_101"]
    base = Context(root=train)
    mut = Context(root=train, rulebook=tmp_path / "rulebook.md",
                  llm=LLMParser(FakeJudge(_always(verdict)), _cfg(1, rule_judge=True), examples=[]))
    off = Context(root=train, rulebook=tmp_path / "rulebook.md",
                  llm=LLMParser(FakeJudge(_always("same")), _cfg(1), examples=[]))      # judge off by default
    mut.stores = off.stores = base.stores
    assert off.judge is None
    judged = 0
    for q in qs:
        _, _, row = run_q(q, base)
        it2, _, row2 = run_q(q, mut)
        _, _, row3 = run_q(q, off)
        if it2 is not None and "judge" in it2.note:
            judged += 1
            assert row2 == row if verdict == "same" else row2["type"] == "abstain"
            assert row3["type"] == "abstain"                                 # no judge -> no silent rule answer
        else:
            assert row2 == row == row3
    assert judged > 0 and mut.judge.calls == judged


@needs_data
def test_pipeline_judge_exception_is_abstain(tmp_path, monkeypatch):
    from datamate.parse.llm import LLMParser
    from datamate.parse.rule_judge import RuleJudge
    from datamate.pipeline import Context, analyse as run_q
    from eval.scorer import load_jsonl
    from eval.synth.rulebooks import variants
    (tmp_path / "rulebook.md").write_text(variants(TRAIN_RB.read_text(encoding="utf8"))["cosmetic"], encoding="utf8")
    train = TRAIN_RB.parent
    ctx = Context(root=train, rulebook=tmp_path / "rulebook.md",
                  llm=LLMParser(FakeJudge(_always("same")), _cfg(1, rule_judge=True), examples=[]))
    monkeypatch.setattr(RuleJudge, "judge", lambda *a, **k: 1 / 0)
    rows = [run_q(q, ctx) for q in load_jsonl(train / "questions.jsonl") if q["business_id"] == "biz_101"]
    errs = [(it, row) for it, _, row in rows if it is not None and "judge error" in it.note]
    assert errs and all(row["type"] == "abstain" for _, row in errs)
