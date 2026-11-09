"""Code-gen agent (D20, ARCHITECTURE §8): prompt -> k code samples -> sandboxed runs -> majority vote.

Batched per shop: one backend call for all of a shop's questions (vLLM batches them and reuses the cached prefix),
then each sample runs in that shop's sandbox worker. Log fields follow the Docker Guide: thinking = raw model outputs
(list, one per sample), code = extracted code per sample, output = per-sample result.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from datamate.agent.answer import vote_key
from datamate.agent.prompt import build_prompt, extract_code
from datamate.agent.sandbox import Job, ShopSandbox
from datamate.agent.tables import frames_of, schema_text
from datamate.config import AgentConfig
from datamate.normalize.text import normalize


class CodeGenAgent:
    def __init__(self, backend, cfg: AgentConfig, root: Path, rulebook: Path | None = None):
        self.backend, self.cfg, self.root = backend, cfg, Path(root)
        self.rulebook_path = Path(rulebook) if rulebook else self.root / "rulebook.md"
        self.rulebook = self.rulebook_path.read_text(encoding="utf8") if self.rulebook_path.exists() else ""
        self.sandboxes: dict[str, ShopSandbox] = {}
        self.calls = self.failed = 0

    def stats_line(self) -> str:
        return f"agent_calls={self.calls} agent_failed={self.failed}"

    def _shop_text(self, biz: str, store) -> tuple[str, str]:
        rd = self.root / "data" / biz / "README.md"
        readme = rd.read_text(encoding="utf8") if rd.exists() else "(không có README)"
        return readme, schema_text(frames_of(store))

    def prompt(self, q: dict, readme: str, schema: str) -> str:
        return build_prompt(self.rulebook, readme, schema, q["question"], q["as_of_date"])

    def sandbox(self, biz: str) -> ShopSandbox:
        sb = self.sandboxes.get(biz)
        if sb is None:
            sb = self.sandboxes[biz] = ShopSandbox(self.root, biz, self.rulebook_path, self.cfg.timeout_s)
        return sb

    def solve_shop(self, qs: list[dict], store) -> list[tuple[dict | None, dict]]:
        """For questions of ONE shop: [(voted row without id or None, trace)]."""
        if not qs:
            return []
        readme, schema = self._shop_text(qs[0]["business_id"], store)
        prompts = [self.prompt(q, readme, schema) for q in qs]
        outs = self.backend.complete(prompts, self.cfg.n_samples, free=True, max_tokens=self.cfg.max_tokens)
        self.calls += len(qs)
        res = []
        for q, raw in zip(qs, outs):
            self.failed += not raw
            res.append(self._vote(q, raw))
        return res

    def _vote(self, q: dict, raw: list[str]) -> tuple[dict | None, dict]:
        sb = self.sandbox(q["business_id"])
        qn = normalize(q["question"])
        codes, results, keys, rows = [], [], [], {}
        for text in raw:
            code = extract_code(text)
            codes.append(code)
            if code is None:
                results.append({"status": "no_code"})
                continue
            status, val = sb.run(Job(code, q["as_of_date"], qn, q["question"]))
            results.append({"status": status, "result": val})
            if status == "ok":
                k = vote_key(val)
                keys.append(k)
                rows.setdefault(k, val)
        trace = {"thinking": list(raw), "code": codes, "output": results}
        if not keys:
            trace["vote"] = "no valid sample"
            return None, trace
        top, votes = sorted(Counter(keys).items(), key=lambda kv: (-kv[1], kv[0]))[0]
        trace["vote"] = f"{votes}/{len(raw)}"
        row = rows[top]
        if votes < self.cfg.min_votes or row["type"] == "abstain":
            return None, trace
        if row["type"] == "unanswerable" and not self.cfg.allow_unanswerable:
            trace["vote"] += " unanswerable->abstain"
            return None, trace
        return row, trace

    def close(self):
        for sb in self.sandboxes.values():
            sb.close()
        self.sandboxes.clear()


def combine(rule_row: dict, cg_row: dict | None, policy: str) -> tuple[dict, str]:
    """Final row (with id) and its source. rule_row always has 'id'; cg_row has no id (None = no majority)."""
    qid = rule_row["id"]
    rule_ans = rule_row.get("type") != "abstain"
    cg = {"id": qid, **cg_row} if cg_row is not None else None
    if policy == "codegen":
        return (cg, "codegen") if cg else ({"id": qid, "type": "abstain"}, "none")
    if policy == "rule_first":
        if rule_ans:
            return rule_row, "rule"
        return (cg, "codegen") if cg else (rule_row, "none")
    if policy == "agree":
        if rule_ans and cg is not None:
            same = vote_key({k: v for k, v in rule_row.items() if k != "id"}) == vote_key(cg_row)
            return (rule_row, "rule+codegen") if same else ({"id": qid, "type": "abstain"}, "disagree")
        if rule_ans:
            return rule_row, "rule"
        return (cg, "codegen") if cg else (rule_row, "none")
    print(f"[agent] unknown policy {policy!r}, abstaining", file=sys.stderr)
    return {"id": qid, "type": "abstain"}, "none"
