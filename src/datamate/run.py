"""Entrypoint: python -m datamate.run --data /data --out /result/submission.jsonl

Streams one line per question (flush after each). Any error or timeout -> abstain, never crash.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from datamate.config import LLMConfig
from datamate.config import AgentConfig
from datamate.pipeline import Context, answer_many


def make_llm(backend: str | None = None, model: str | None = None, replay: str | None = None):
    """LLM parser from env (DATAMATE_LLM*) with CLI overrides; None when the backend is 'off' (D14)."""
    from datamate.parse.llm import LLMParser, make_backend
    kw = {k: v for k, v in (("backend", backend), ("model_path", model), ("replay_path", replay)) if v}
    cfg = LLMConfig.from_env(**kw)
    be = make_backend(cfg)
    return LLMParser(be, cfg) if be is not None else None


def make_agent(llm, root: Path, rulebook: Path | None = None, enabled: bool | None = None, kind: str | None = None):
    """Code-gen (D20) or text-to-SQL (D26) agent on the same backend as the LLM parser; None when off or no backend."""
    kw = {"enabled": enabled} if enabled is not None else {}
    if kind:
        kw["kind"] = kind
    cfg = AgentConfig.from_env(**kw)
    if not cfg.enabled or llm is None:
        return None
    if cfg.kind == "sql":
        from datamate.agent.sql_agent import SqlAgent  # noqa: PLC0415
        return SqlAgent(llm.backend, cfg, root, rulebook)
    from datamate.agent.agent import CodeGenAgent  # noqa: PLC0415
    return CodeGenAgent(llm.backend, cfg, root, rulebook)


def find_root(data: Path) -> Path:
    """Locate the dir containing questions.jsonl (handles /data or /data/<split>/<split>)."""
    if (data / "questions.jsonl").exists():
        return data
    hits = sorted(data.rglob("questions.jsonl"))
    if not hits:
        raise FileNotFoundError(f"questions.jsonl not found under {data}")
    return hits[0].parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/result/submission.jsonl")
    ap.add_argument("--log", default=None, help="per-question log jsonl (default: <out dir>/logs/log.jsonl)")
    ap.add_argument("--only", nargs="*", help="restrict to these business_ids (dev use)")
    ap.add_argument("--llm", choices=["off", "vllm", "replay"], help="LLM backend (default: $DATAMATE_LLM or off)")
    ap.add_argument("--llm-model", help="local weights dir (default: $DATAMATE_LLM_MODEL)")
    ap.add_argument("--llm-replay", help="replay/record jsonl (default: $DATAMATE_LLM_REPLAY)")
    ap.add_argument("--agent", choices=["off", "codegen", "sql"],
                    help="code-gen (D20) or text-to-SQL (D26) agent (default: $DATAMATE_AGENT or off)")
    args = ap.parse_args(argv)

    root = find_root(Path(args.data))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        llm = make_llm(args.llm, args.llm_model, args.llm_replay)
    except Exception:  # model failed to load (no GPU, OOM...): run the measured rule-only system, never crash (D14)
        print(f"[datamate] LLM disabled, rule-only run:\n{traceback.format_exc(limit=3)}", file=sys.stderr)
        llm = None
    agent = make_agent(llm, root, enabled=args.agent in ("codegen", "sql") if args.agent else None,
                       kind=args.agent if args.agent in ("codegen", "sql") else None)
    # D20: with the agent, the rule side stays pure rule (no D14 LLM parser/judge); the model serves the agent only
    ctx = Context(root=root, llm=None if agent else llm, agent=agent)
    with open(root / "questions.jsonl", encoding="utf8") as f:
        qs = [json.loads(l) for l in f if l.strip()]
    if args.only:
        qs = [q for q in qs if q["business_id"] in set(args.only)]

    log = Path(args.log) if args.log else out.parent / "logs" / "log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, "w", encoding="utf8", newline="\n") as w, open(log, "w", encoding="utf8", newline="\n") as lg:
        for q, row, trace in answer_many(qs, ctx):
            tq = trace.pop("_t0")   # question start (batched: when it entered its batch)
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            w.flush()
            # SPEC submission contract (Guide §3.3): one record per question in /result/logs/, always id, answer,
            # elapsed_ms (int ms from question start to answer), thinking; code/inputs/output when a function ran
            rec = {"id": q["id"], "answer": {k: v for k, v in row.items() if k != "id"},
                   "elapsed_ms": int(round((time.perf_counter() - tq) * 1000)), **trace}
            lg.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            lg.flush()
    print(f"[datamate] {len(qs)} questions in {time.time() - t0:.1f}s -> {out}", file=sys.stderr)
    if ctx.llm is not None:
        print(ctx.llm.stats_line(), file=sys.stderr)
    if ctx.judge is not None:
        print(ctx.judge.stats_line(), file=sys.stderr)
    if ctx.agent is not None:
        print(ctx.agent.stats_line(), file=sys.stderr)
        ctx.agent.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
