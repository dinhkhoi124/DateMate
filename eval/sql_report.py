"""Read a text-to-SQL agent run (D26) on CPU: where it is right/wrong, why samples fail, how long it takes.

Input is the trace.jsonl of `eval/run_eval.py --agent sql --trace` (Kaggle zip, or a reexec'd trace from
eval/reexec_agent.py --sql). Answers are re-voted with the committed AgentConfig (m = ceil(tau x k), sanity checks
on), so the numbers are those of the official config, not of a tuned one.

    python eval/sql_report.py eval/runs/kaggle_sql/<run>_dev/trace.jsonl
    python eval/sql_report.py <sql trace> --compare eval/runs/kaggle_qwen3_14b_v3/<run>_dev/trace.jsonl
    python eval/sql_report.py <holdout trace> --totals-only        # holdout: totals only (CLAUDE.md rule 2)

Sections: SQL alone and with each rule policy; PA by op (rule-parser intent); sample status and DuckDB error
classes; seconds per question (amortised over the shop batch, as in the Docker Guide budget); with --compare a
per-question cross table SQL x code-gen on the questions both traces have, and two fixed combinations of the two.
Train only: every choice made from this report stays on train (D21).
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from datamate.agent.agent import combine  # noqa: E402
from datamate.agent.answer import vote_key  # noqa: E402
from datamate.config import AgentConfig  # noqa: E402
from eval.calibrate_agent import POLICIES, revote  # noqa: E402
from eval.scorer import load_jsonl, score  # noqa: E402

_ERR = [   # (class, pattern on the error text) -- first match wins
    ("binder:column", re.compile(r"Binder Error.*(column|Referenced column|not found in FROM)", re.I | re.S)),
    ("binder:table", re.compile(r"Catalog Error|Table with name|does not exist", re.I)),
    ("binder:other", re.compile(r"Binder Error", re.I)),
    ("parser", re.compile(r"Parser Error|syntax error", re.I)),
    ("conversion", re.compile(r"Conversion Error|Could not convert|Invalid Input Error", re.I)),
    ("timeout", re.compile(r"timeout|interrupt", re.I)),
    ("not_select", re.compile(r"one SELECT|only SELECT|statement", re.I)),
    ("type_shape", re.compile(r"does not fit TYPE", re.I)),
]


def err_class(text) -> str:
    t = str(text or "")
    for name, rx in _ERR:
        if rx.search(t):
            return name
    return "other"


def sql_answer(t: dict, m: int, cfg: AgentConfig, final=None) -> dict | None:
    """Official SQL vote: share >= tau of k, unanswerable per config, then the agent's post-vote steps (sanity
    checks; 'unanswerable' when every time window lies outside the data, with `final` = resubmit_agent.SqlFinal)."""
    from datamate.agent.sql_agent import sanity  # noqa: PLC0415
    from datamate.normalize.text import normalize  # noqa: PLC0415
    row = revote(t.get("output") or [], m, cfg.allow_unanswerable)
    if row is not None and cfg.sanity and t.get("_q") and sanity(row, normalize(t["_q"]["question"])):
        row = None
    if row is None and final is not None and t.get("_q"):
        row, _ = final.agent.outside_data(t["_q"], final.store(t["business_id"]), None, {"output": t.get("output")})
    return row


def tally(scores) -> tuple[int, int, int]:
    c = Counter(scores)
    return c[1], c[-1], c[0]


def line(name: str, c: int, w: int, a: int, width: int = 22) -> str:
    n = c + w + a
    pa = (c - w) / n if n else 0.0
    prec = c / (c + w) if c + w else 0.0
    return f"{name:{width}}{n:5}{c:6}{w:6}{a:6}{pa:8.3f}{prec:8.3f}"


def head(width: int = 22) -> str:
    return f"{'':{width}}{'n':>5}{'ok':>6}{'wrong':>6}{'abst':>6}{'PA':>8}{'prec':>8}"


def seconds_per_question(tr: list[dict]) -> list[float]:
    """Amortised seconds: own rule + prompt + exec time, plus the shop batch's generation/repair time / batch size."""
    out = []
    for t in tr:
        tm = t.get("timing") or {}
        n = max(int(tm.get("batch_n") or 1), 1)
        s = sum(float(tm.get(k) or 0) for k in ("rule_s", "prompt_s", "exec_s"))
        s += (float(tm.get("gen_batch_s") or 0) + float(tm.get("repair_batch_s") or 0)) / n
        if tm.get("gen_batch_s") is not None:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--answers", default=str(ROOT / "data" / "train" / "train" / "answers.jsonl"))
    ap.add_argument("--data", default=str(ROOT / "data" / "train" / "train"),
                    help="split dir (questions.jsonl, data/, rulebook.md) for the out-of-data check")
    ap.add_argument("--compare", help="code-gen trace on the same questions (e.g. Kaggle v3 dev)")
    ap.add_argument("--totals-only", action="store_true", help="holdout: totals and timing only")
    ap.add_argument("--n-private", type=int, default=500, help="question count for the time estimate")
    ap.add_argument("--errors", type=int, default=6, help="example messages per error class (dev only)")
    args = ap.parse_args()

    cfg = AgentConfig()   # committed defaults (D21), deliberately not from_env()
    golds = {g["id"]: g for g in load_jsonl(args.answers)}
    qrec = {q["id"]: q for q in load_jsonl(Path(args.data) / "questions.jsonl")}
    tr = [t for t in load_jsonl(args.trace) if t["id"] in golds]
    for t in tr:
        t["_q"] = qrec.get(t["id"])
    k = max((len(t.get("output") or []) for t in tr), default=0)
    m = math.ceil(cfg.tau * (k or cfg.sql_samples) - 1e-9)
    print(f"{len(tr)} questions, k={k} samples, official m={m} (tau={cfg.tau}), sanity={'on' if cfg.sanity else 'off'}, "
          f"committed policy={cfg.policy}")

    # ---- answers: SQL alone, and each rule x SQL policy
    from eval.resubmit_agent import SqlFinal  # noqa: PLC0415
    final = SqlFinal(Path(args.data))
    rows = {}
    for t in tr:
        rule = {"id": t["id"], **((t.get("inputs") or {}).get("rule_answer") or {"type": "abstain"})}
        rows[t["id"]] = (rule, sql_answer(t, m, cfg, final))
    print("\n" + head())
    sc_sql = {i: score({"id": i, **s} if s else None, golds[i]) for i, (_, s) in rows.items()}
    print(line("SQL alone", *tally(sc_sql.values())))
    print(line("rule alone", *tally(score(r, golds[i]) for i, (r, _) in rows.items())))
    for pol in POLICIES:
        print(line(f"policy {pol}", *tally(score(combine(r, s, pol)[0], golds[i]) for i, (r, s) in rows.items())))
    print("(on public/private most rule answers are blocked by the rulebook diff (D18): 'SQL alone' is the number "
          "that carries over)")

    # ---- timing (contract budget: about 6 h incl. model load for the private set)
    sp = seconds_per_question(tr)
    if sp:
        med, p95 = statistics.median(sp), sorted(sp)[min(len(sp) - 1, int(0.95 * len(sp)))]
        tot = sum(sp)
        print(f"\nseconds/question (amortised): median {med:.1f}  p95 {p95:.1f}  mean {tot / len(sp):.1f}  "
              f"total {tot / 60:.0f} min for {len(sp)} q -> {args.n_private} q ~ {tot / len(sp) * args.n_private / 3600:.1f} h "
              f"on this GPU")

    # ---- sample status
    status, ecls, ex = Counter(), Counter(), defaultdict(list)
    repaired = 0
    for t in tr:
        for raw, o in zip(t.get("thinking") or [], t.get("output") or []):
            status[o.get("status")] += 1
            repaired += "\n<<repair>>\n" in (raw or "") and o.get("status") == "ok"
            if o.get("status") == "error":
                c = err_class(o.get("result"))
                ecls[c] += 1
                if len(ex[c]) < args.errors:
                    ex[c].append(re.sub(r"\s+", " ", str(o.get("result")))[:160])
    tot = sum(status.values()) or 1
    print("\nsamples: " + "  ".join(f"{s}={n} ({100 * n / tot:.0f}%)" for s, n in status.most_common())
          + f"  | ok after repair: {repaired}")
    votes = Counter(re.sub(r"\d+/\d+ ?", "", str((t.get("inputs") or {}).get("codegen_vote") or "none")) or "voted"
                    for t in tr)
    print("vote notes: " + "  ".join(f"{k}={v}" for k, v in votes.most_common(6)))
    if ecls:
        print("error classes: " + "  ".join(f"{c}={n}" for c, n in ecls.most_common()))
    if args.totals_only:
        return

    # ---- by op (rule-parser intent on the question; '-' = rule parser had no intent)
    by = defaultdict(list)
    for t in tr:
        by[(t.get("inputs") or {}).get("op") or "-"].append(sc_sql[t["id"]])
    print("\nSQL alone by op (worst first)\n" + head(26))
    for op, ss in sorted(by.items(), key=lambda kv: (-(kv[1].count(-1)), kv[1].count(1) - len(kv[1]), kv[0])):
        print(line(op, *tally(ss), width=26))
    by_shop = defaultdict(list)
    for t in tr:
        by_shop[t["business_id"]].append(sc_sql[t["id"]])
    print("\nSQL alone by shop: ok/wrong/abst  " + "  ".join(
        f"{b[4:]}:{c}/{w}/{a}" for b, (c, w, a) in ((b, tally(s)) for b, s in sorted(by_shop.items()))))
    for c, msgs in sorted(ex.items(), key=lambda kv: -ecls[kv[0]]):
        print(f"\n[{c}] {ecls[c]}")
        for msg in msgs:
            print("   ", msg)

    # ---- compare with a code-gen trace on the same questions
    if args.compare:
        cg_tr = {t["id"]: t for t in load_jsonl(args.compare) if t["id"] in golds}
        both = [i for i in rows if i in cg_tr]
        kc = max((len(cg_tr[i].get("output") or []) for i in both), default=0)
        mc = cfg.min_votes
        cg = {i: revote(cg_tr[i].get("output") or [], mc, cfg.allow_unanswerable) for i in both}
        sc_cg = {i: score({"id": i, **cg[i]} if cg[i] else None, golds[i]) for i in both}
        print(f"\n--compare: {len(both)} common questions; code-gen k={kc} m={mc}")
        print(head())
        print(line("SQL alone", *tally(sc_sql[i] for i in both)))
        print(line("code-gen alone", *tally(sc_cg[i] for i in both)))
        # SQL first, code-gen when SQL abstains
        fb = {i: (sc_sql[i] if rows[i][1] else sc_cg[i]) for i in both}
        print(line("SQL, else code-gen", *tally(fb.values())))
        # answer only when both answer the same
        agree = {}
        for i in both:
            s, c = rows[i][1], cg[i]
            agree[i] = sc_sql[i] if s and c and vote_key(s) == vote_key(c) else 0
        print(line("SQL == code-gen", *tally(agree.values())))
        cross = Counter((sc_sql[i], sc_cg[i]) for i in both)
        lab = {1: "ok", -1: "wrong", 0: "abst"}
        print("\ncross table (rows SQL, cols code-gen):")
        print(f"{'':10}" + "".join(f"{lab[c]:>8}" for c in (1, -1, 0)))
        for s in (1, -1, 0):
            print(f"{lab[s]:10}" + "".join(f"{cross[(s, c)]:8}" for c in (1, -1, 0)))


if __name__ == "__main__":
    main()
