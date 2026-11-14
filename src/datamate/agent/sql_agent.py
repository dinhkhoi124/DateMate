"""Text-to-SQL agent (D26): prompt -> k samples (notes, TYPE, SQL) -> DuckDB -> repair turn -> vote by share.

Same interface as agent.CodeGenAgent (`solve_shop`, `cfg.policy`, `close`, `stats_line`) and the same trace fields
(thinking / code / output / vote), so pipeline.answer_many, eval/calibrate_agent.py and the Kaggle notebooks work
unchanged. A sample counts in the vote only when its query ran and returned a non-empty answer of its TYPE; a
question is answered only when the winning answer has share >= tau of the k samples (slide 33: share is over all
samples, failed ones included).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path

from datamate.agent.answer import canon, vote_key
from datamate.agent.hints import as_of_line, question_hints
from datamate.agent.sql_prompt import (EmptyResult, build_prompt, parse_reply, repair_prompt, rule_change_block,
                                       to_row)
from datamate.agent.sqldb import RULE_ARTIFACTS, QueryError, build_db, schema_text, stale_artifacts
from datamate.config import AgentConfig
from datamate.normalize.text import normalize

FEWSHOT_PATH = Path(__file__).with_name("fewshot_sql.jsonl")


class FewShot:
    """Train-dev examples {id, business_id, question, reply}; k most similar questions from OTHER shops."""

    def __init__(self, path: Path = FEWSHOT_PATH):
        self.rows = []
        if Path(path).exists():
            with open(path, encoding="utf8") as f:
                self.rows = [json.loads(l) for l in f if l.strip()]
        self._norm = [normalize(r["question"]) for r in self.rows]

    def pick(self, question: str, biz: str, k: int) -> list[dict]:
        if not self.rows or k <= 0:
            return []
        from rapidfuzz import fuzz  # noqa: PLC0415
        q = normalize(question)
        scored = [(-fuzz.token_set_ratio(q, n), r["id"], i) for i, (r, n) in enumerate(zip(self.rows, self._norm))
                  if r["business_id"] != biz]
        scored.sort()
        out, per_op = [], Counter()
        for _, _, i in scored:      # at most 2 examples of one op shape: avoid k copies of the same template
            op = self.rows[i].get("op") or self.rows[i]["id"]
            if per_op[op] >= 2:
                continue
            per_op[op] += 1
            out.append(self.rows[i])
            if len(out) == k:
                break
        return out


_COUNT_WORDS = re.compile(r"\b(bao nhieu|may|so)\s+(don|khach|ngay|lan|hoa don|mat hang|san pham|mon|loai)\b")


def sanity(row: dict, qn: str) -> str | None:
    """Reasons a voted answer is implausible (backlog A3/S7) -> abstain. Meaning-based, not tuned per shop."""
    t, a = row["type"], row.get("answer")
    if t == "percent" and isinstance(a, (int, float)) and abs(a) > 100000:
        return "percent out of range"
    if t == "number" and isinstance(a, (int, float)):
        if _COUNT_WORDS.search(qn) and "tien" not in qn and (a < 0 or a != int(a)):
            return "count is not a non-negative integer"
        if ("doanh thu" in qn or "bao nhieu tien" in qn) and a < 0:
            return "negative revenue"
    return None


def period_outside_data(q: dict, f, cal, rules) -> str | None:
    """Deterministic ground for 'unanswerable' (CLAUDE.md rule 5, rulebook §7): every time window of the question
    lies outside the shop's data, by the same guard as the rule executor (execute/ops.clip). None when the question
    names no window or one window overlaps the data."""
    from datamate.execute.ops import Unanswerable, clip  # noqa: PLC0415
    from datamate.timeparse.periods import find_periods  # noqa: PLC0415
    if f is None or f.dmin is None:
        return None
    as_of = date.fromisoformat(q["as_of_date"])
    ps = [p for p in find_periods(normalize(q["question"]), as_of, cal, rules) if p.kind != "asof"]
    why = None
    for p in ps:
        try:
            clip(f, p.start, p.end, as_of)
            return None
        except Unanswerable as e:
            why = str(e)
    return why


def no_concrete_sample(output: list[dict]) -> bool:
    """No sample ran to an answer other than 'unanswerable' (all empty / error / no SQL / model unanswerable)."""
    return not any(o.get("status") == "ok" and isinstance(o.get("result"), dict)
                   and o["result"].get("type") not in ("unanswerable", "abstain") for o in output or [])


class SqlAgent:
    def __init__(self, backend, cfg: AgentConfig, root: Path, rulebook: Path | None = None, fewshot: Path | None = None):
        from datamate.rulebook import analyse  # noqa: PLC0415
        from datamate.timeparse.periods import LunarCalendar  # noqa: PLC0415
        self.backend, self.cfg, self.root = backend, cfg, Path(root)
        rb = Path(rulebook) if rulebook else self.root / "rulebook.md"
        self.rulebook = rb.read_text(encoding="utf8") if rb.exists() else ""
        self.rb = analyse(self.rulebook or None)
        self.rules = self.rb.rules
        self.cal = LunarCalendar.load(str(self.root / "calendar.csv"))
        self.fewshot = FewShot(fewshot or FEWSHOT_PATH)
        self.dbs: dict = {}
        self.calls = self.failed = self.repaired = self.too_long = 0

    def stats_line(self) -> str:
        return (f"sql_agent calls={self.calls} failed={self.failed} repaired={self.repaired} "
                f"fewshot={len(self.fewshot.rows)} too_long={self.too_long}")

    # ---------------------------------------------------------------------------------------------------- per shop
    def _shop(self, biz: str, store):
        if biz not in self.dbs:
            from datamate.execute.facts import facts_of  # noqa: PLC0415
            from datamate.link.catalog import linker_of  # noqa: PLC0415
            db = build_db(store, self.rules, self.rb)
            lk = None
            try:
                if store.orders is not None and len(store.orders):
                    lk = linker_of(facts_of(store, self.rules))
            except Exception:  # noqa: BLE001
                lk = None
            rd = self.root / "data" / biz / "README.md"
            readme = rd.read_text(encoding="utf8") if rd.exists() else "(không có README)"
            self.dbs[biz] = (db, schema_text(db), readme, lk)
        return self.dbs[biz]

    def _helpers(self, q: dict, lk):
        from datamate.agent.helpers import Helpers  # noqa: PLC0415
        return Helpers(date.fromisoformat(q["as_of_date"]), normalize(q["question"]), self.cal, self.rules, lk,
                       None, q["question"])

    def touched_rows(self, q: dict) -> list[str]:
        """Rulebook rows this question touches that the split changed, dropped (train ids) or added ('new:' ids).
        `q['_rb_rows']` = the rule path's RulebookDiff.hits for its intent (pipeline.answer_many); no rule intent ->
        every changed/dropped scoped row plus the new rows whose keyword is in the question (fail closed). Train
        rulebook -> [] (prompt and vote unchanged)."""
        rb = self.rb
        if not (rb.changed or rb.missing or rb.new_rows):
            return []
        rows = q.get("_rb_rows")
        if rows is None:
            from datamate.rulebook import fold  # noqa: PLC0415
            qf = f" {fold(q['question'])} "
            rows = set().union(*rb.blocked.values()) if rb.blocked else set()
            for kw in rb.keywords:
                if f" {kw} " in qf:
                    rows |= rb.kw_rows[kw]
        return sorted(rows)

    def prompt(self, q: dict, schema: str, readme: str, lk) -> str:
        H = self._helpers(q, lk)
        ex = self.fewshot.pick(q["question"], q["business_id"], self.cfg.fewshot_k)
        rows = self.touched_rows(q)
        rc = rule_change_block([(r, self.rb.texts.get(r), list(RULE_ARTIFACTS.get(r, ()))) for r in rows])
        args = (self.rulebook, readme, schema, q["question"], as_of_line(H.as_of), question_hints(q["question"], H))
        p = build_prompt(*args, ex, rc)
        while len(p) > self.cfg.max_prompt_chars and ex:   # long README/schema: fewer examples, never a cut rulebook
            ex = ex[:-1]
            p = build_prompt(*args, ex, rc)
        return p

    def gate_stale(self, q: dict, runs: list) -> list:
        """Samples whose query reads an artifact built on the old reading of a changed rule row this question touches
        do not vote (status 'rule_stale'): they answer with the train rule (bench B 14/11, CLAUDE.md rule 5)."""
        rows = [r for r in self.touched_rows(q) if not r.startswith("new:")]
        if not rows:
            return runs
        out = []
        for st, row, sql, err in runs:
            arts = stale_artifacts(sql, rows) if st == "ok" else []
            out.append(("rule_stale", None, sql, f"reads {', '.join(arts)} (old rule)") if arts else (st, row, sql, err))
        return out

    def solve_shop(self, qs: list[dict], store) -> list[tuple[dict | None, dict]]:
        if not qs:
            return []
        t0 = time.perf_counter()
        db, schema, readme, lk = self._shop(qs[0]["business_id"], store)
        prompts = [self.prompt(q, schema, readme, lk) for q in qs]
        t1 = time.perf_counter()
        k = self.cfg.sql_samples
        # a prompt over budget even without examples is not sent: vLLM rejects an over-long prompt for the whole
        # batch, which would cost every question of the shop; that question alone abstains
        ok = [i for i, p in enumerate(prompts) if len(p) <= self.cfg.max_prompt_chars]
        got = self.backend.complete([prompts[i] for i in ok], k, free=True, max_tokens=self.cfg.max_tokens) if ok else []
        outs = [[] for _ in prompts]
        for i, o in zip(ok, got):
            outs[i] = list(o)
        self.too_long += len(prompts) - len(ok)
        t2 = time.perf_counter()
        self.calls += len(qs)
        # run every sample; collect failed ones for one batched repair turn per shop
        runs = []
        repairs = []      # (question index, sample index, repair prompt)
        for qi, (q, raw) in enumerate(zip(qs, outs)):
            self.failed += not raw
            res = [self._run(db, r) for r in raw]
            runs.append(res)
            for si, (r, (st, _, _, err)) in enumerate(zip(raw, res)):
                if st == "error" and err and self.cfg.repair_turns > 0:
                    repairs.append((qi, si, repair_prompt(prompts[qi], r, err)))
        t3 = time.perf_counter()
        if repairs:
            fixed = self.backend.complete([p for _, _, p in repairs], 1, free=True, max_tokens=self.cfg.max_tokens)
            for (qi, si, _), rr in zip(repairs, fixed):
                if rr:
                    new = self._run(db, rr[0])
                    if new[0] == "ok":
                        self.repaired += 1
                    outs[qi][si] = outs[qi][si] + "\n<<repair>>\n" + rr[0]
                    runs[qi][si] = new
        t4 = time.perf_counter()
        res = []
        for q, raw, rs in zip(qs, outs, runs):
            row, trace = self._vote(q, raw, self.gate_stale(q, rs), k)
            if row is None:
                row, trace = self.outside_data(q, store, row, trace)
            trace["timing"] = {"prompt_s": (t1 - t0) / len(qs), "gen_batch_s": t2 - t1, "batch_n": len(qs),
                               "exec_s": (t3 - t2) / len(qs), "repair_batch_s": t4 - t3}
            res.append((row, trace))
        return res

    def outside_data(self, q: dict, store, row: dict | None, trace: dict) -> tuple[dict | None, dict]:
        """An abstained question whose samples produced no concrete answer and whose every time window lies outside
        the data -> 'unanswerable' (deterministic, so also when the model's own 'unanswerable' is not trusted).
        Also used by the offline tools (eval/resubmit_agent.py --sql, eval/sql_report.py) on recorded outputs."""
        if row is not None or store is None or not no_concrete_sample(trace.get("output")):
            return row, trace
        from datamate.execute.facts import facts_of  # noqa: PLC0415
        try:
            f = facts_of(store, self.rules) if store.orders is not None and len(store.orders) else None
            why = period_outside_data(q, f, self.cal, self.rules)
        except Exception:  # noqa: BLE001  (the guard only ever turns an abstain into unanswerable)
            why = None
        if why is None:
            return row, trace
        trace["vote"] = f"{trace.get('vote', '')} {why}->unanswerable".strip()
        return {"type": "unanswerable", "reason": why}, trace

    def rerun(self, db, raw_reply: str) -> tuple[str, dict | None, str | None, str | None]:
        """Re-execute a recorded reply (eval/reexec_agent.py --sql): the original first, the repair if it fails."""
        first, _, fix = raw_reply.partition("\n<<repair>>\n")
        r = self._run(db, first)
        if r[0] == "error" and fix:
            r = self._run(db, fix)
        return r

    def _run(self, db, text: str) -> tuple[str, dict | None, str | None, str | None]:
        """(status, canonical row, sql, error) for one reply. status: ok | empty | error | no_sql."""
        atype, sql = parse_reply(text)
        if atype is None:
            return "no_sql", None, sql, "no TYPE line"
        if atype == "unanswerable":
            return "ok", canon(to_row("unanswerable", [], [])), None, None
        if not sql:
            return "no_sql", None, None, None
        try:
            cols, rows = db.query(sql, self.cfg.timeout_s)
            return "ok", canon(to_row(atype, cols, rows)), sql, None
        except EmptyResult as e:    # slide 33: empty answers never vote (abstain rather than answer 0)
            return "empty", None, sql, str(e)
        except QueryError as e:
            return "error", None, sql, str(e)
        except ValueError as e:     # wrong result shape for the TYPE: the model can fix it in the repair turn
            return "error", None, sql, f"result does not fit TYPE {atype}: {e}"

    def _vote(self, q: dict, raw: list[str], runs: list, k: int) -> tuple[dict | None, dict]:
        keys, rows = [], {}
        output = []
        for st, row, sql, err in runs:
            output.append({"status": st, "result": row if st == "ok" else err})
            if st == "ok":
                kk = vote_key(row)
                keys.append(kk)
                rows.setdefault(kk, row)
        trace = {"thinking": list(raw), "code": [r[2] for r in runs], "output": output}
        if not keys:
            trace["vote"] = "no valid sample"
            return None, trace
        top, votes = sorted(Counter(keys).items(), key=lambda kv: (-kv[1], kv[0]))[0]
        share = votes / max(k, 1)
        trace["vote"] = f"{votes}/{k}"
        row = rows[top]
        if share < self.cfg.tau or row["type"] == "abstain":
            return None, trace
        if row["type"] == "unanswerable" and not self.cfg.allow_unanswerable:
            trace["vote"] += " unanswerable->abstain"
            return None, trace
        if self.cfg.sanity:
            why = sanity(row, normalize(q["question"]))
            if why:
                trace["vote"] += f" sanity:{why}"
                return None, trace
        return row, trace

    def close(self):
        for db, *_ in self.dbs.values():
            db.close()
        self.dbs.clear()
