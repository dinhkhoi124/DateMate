"""Rule judge (DECISIONS D18 branch 3, D19): may an intent blocked by a rulebook change still be answered?

The split's rulebook changed or added rule rows that the intent touches (rulebook.RulebookDiff.hits). The executor
implements the train meaning (rulebook_impl.IMPL, our own wording). For EACH touched row the model reads that meaning,
the split's wording of the row (read at run time, memory only) and the question, and answers whether the split's rule,
applied to this operation and this question, computes exactly what the executor computes: same | different | unsure.

Fail-closed: the intent is answered only if, for every touched row, all n_samples outputs came back, parsed, and say
"same". A missing train row (nothing to compare with), a backend error, a short or unparsable output, "different" or
"unsure" for any row -> abstain. There is no fallback to the rule answer. The model never computes a number
(CLAUDE.md rule 4): it only decides whether the deterministic answer still holds.
"""
from __future__ import annotations

import json
from collections import Counter

from datamate.config import LLMConfig
from datamate.parse.llm import parse_json
from datamate.parse.llm_schema import OPS
from datamate.rulebook import RulebookDiff
from datamate.rulebook_impl import IMPL

VERDICTS = ("same", "different", "unsure")
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)}},
          "required": ["verdict"], "additionalProperties": False}
# outcome of one judge call, worst first: the call's outcome is the worst over its rows and samples
OUTCOMES = ("missing", "failed", "invalid", "different", "unsure", "same")


def judge_prompt(question: str, it, rid: str, diff: RulebookDiff) -> str:
    """One touched row. `diff.texts[rid]` is the split's own wording, read from its rulebook.md at run time."""
    op = OPS.get(it.op)
    what = op.desc if op else it.op
    flt = {k: v for k, v in (it.flt or {}).items() if v is not None}
    periods = ", ".join(p.label or p.kind for p in it.periods or []) or "không"
    split = diff.texts[rid]
    if rid.startswith("new:"):
        rule = ("Hệ thống CHƯA cài luật này. Rulebook của bộ dữ liệu có thêm luật mới:\n"
                f"  {split}")
        same = ("same CHỈ KHI luật mới này không thay đổi bất kỳ bước nào của phép tính trên khi trả lời câu hỏi "
                "này (không thêm/bớt đơn, dòng, khoản tiền, điều kiện lọc, cách đếm, cách làm tròn).")
    else:
        rule = (f"Hệ thống đang tính: {IMPL.get(rid, '(không có mô tả)')}\n"
                f"Rulebook của bộ dữ liệu ghi: {split}")
        same = ("same CHỈ KHI theo rulebook của bộ dữ liệu, phép tính trên cho câu hỏi này được tính GIỐNG HỆT cách "
                "hệ thống đang tính: cùng tập đơn/dòng được tính, cùng khoản cộng trừ, cùng điều kiện, cùng cách đếm "
                "và làm tròn. Khác nhau chỉ ở cách diễn đạt thì là same.")
    return (
        "Hệ thống trả lời câu hỏi của chủ shop bằng một phép tính có sẵn, cài theo luật của bộ train. Một luật dưới "
        "đây trong rulebook của bộ dữ liệu đang hỏi khác bộ train.\n\n"
        f"Phép tính sẽ chạy: {what}. Bộ lọc: {json.dumps(flt, ensure_ascii=False)}. Kỳ: {periods}.\n\n"
        f"{rule}\n\n"
        f"Trả lời cho CÂU HỎI DƯỚI ĐÂY và PHÉP TÍNH TRÊN:\n- {same}\n"
        "- different: rulebook của bộ dữ liệu đổi bất kỳ phần nào phép tính trên dùng tới. KHÔNG được coi là same chỉ "
        "vì câu hỏi không nhắc tới phần bị đổi: phép tính vẫn áp dụng luật đó.\n"
        "- unsure: không chắc chắn hai cách tính giống hệt nhau.\n"
        "KHÔNG tự tính số.\n\n"
        f"Câu hỏi: {question}\nJSON:"
    )


def _verdict(text: str) -> str | None:
    js = parse_json(text)
    v = js.get("verdict") if js else None
    return v if v in VERDICTS else None


class RuleJudge:
    def __init__(self, backend, cfg: LLMConfig):
        self.backend, self.cfg = backend, cfg
        self.calls = 0
        self.outcomes: Counter = Counter()

    def judge(self, question: str, it, rids: list[str], diff: RulebookDiff) -> tuple[bool, list[str]]:
        """(answer allowed, raw outputs of every row in order). Raw outputs go to the log as `thinking`."""
        self.calls += 1
        if not rids or any(r not in diff.texts for r in rids):   # train row dropped: no definition to compare
            self.outcomes["missing"] += 1
            return False, []
        need = max(self.cfg.n_samples, self.cfg.judge_min_votes, 1)
        try:
            outs = self.backend.complete([judge_prompt(question, it, r, diff) for r in rids], self.cfg.n_samples,
                                         schema=SCHEMA)
        except Exception:  # noqa: BLE001  (fail closed)
            outs = []
        raw: list[str] = []
        worst = "same"
        if len(outs) != len(rids):
            worst = "failed"
        for o in outs:
            raw += o
            if len(o) < need:
                row = "failed"
            else:
                vs = [_verdict(x) for x in o]
                row = "invalid" if None in vs else min(vs, key=OUTCOMES.index)
            worst = min(worst, row, key=OUTCOMES.index)
        self.outcomes[worst] += 1
        return worst == "same", raw

    def stats_line(self) -> str:
        """judge_failed counts calls with no/short model output (backend error, replay miss): not a model verdict."""
        c = self.outcomes
        return (f"judge_calls={self.calls} judge_same={c['same']} judge_different={c['different']} "
                f"judge_unsure={c['unsure']} judge_invalid={c['invalid']} judge_missing_row={c['missing']} "
                f"judge_failed={c['failed']}")
