"""Rule parameters (CLAUDE.md: parameters live in config, not scattered in code).

Defaults are the train rulebook values; `from_rulebook(text)` re-reads them from the split's rulebook.md at run time
(DECISIONS D07) so a private rulebook with other shift hours / Tết window is honoured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rules:
    morning: tuple[int, int] = (6 * 3600, 13 * 3600 + 59 * 60 + 59)     # rulebook §2: ca sáng 06:00–13:59
    afternoon: tuple[int, int] = (14 * 3600, 21 * 3600 + 59 * 60 + 59)  # rulebook §2: ca chiều 14:00–21:59
    week_start: int = 0                    # rulebook §2: tuần bắt đầu thứ Hai (date.weekday() == 0)
    tet_start: tuple[int, int] = (23, 12)  # rulebook §2: 23 tháng Chạp (lunar day, lunar month) of lunar year Y-1
    tet_end: tuple[int, int] = (10, 1)     # rulebook §2: mùng 10 tháng Giêng of lunar year Y
    service_fee_names: tuple[str, ...] = ("phi phuc vu",)  # rulebook §3: Phí phục vụ is not an item


DEFAULT = Rules()


def from_rulebook(text: str | None) -> Rules:
    """Rules of a split's rulebook (D07, D18): parameters re-read only from rows that match the train rule template."""
    from datamate.rulebook import analyse  # noqa: PLC0415  (rulebook imports this module)
    return analyse(text).rules


@dataclass(frozen=True)
class Invariants:
    """Run-time ingest invariants (DECISIONS D11, ingest/checks.py). Wide, meaning-based bounds: every dev shop is
    far inside them (eval/synth/faults.py), a misread table is far outside."""
    min_orders_with_date: float = 0.98     # orders whose date could not be read
    max_dup_order_id: float = 0.01         # same order id twice: a table read twice or the wrong id column
    max_after_data_end: float = 0.01       # dates after the README 'Dữ liệu có đến hết ...'
    min_share_day_gt12: float = 0.20       # ~19/31 expected; dd/mm read as mm/dd leaves days 13..31 empty
    year_range: tuple[int, int] = (2000, 2100)
    min_done: float = 0.50                 # most orders are completed
    max_cancelled: float = 0.30
    max_returned: float = 0.30
    min_orders_status_evidence: int = 500   # this many orders, none cancelled/returned: status unread
    min_orders_with_rev: float = 0.90
    max_neg_rev: float = 0.05
    order_money_range: tuple[float, float] = (1_000.0, 1e9)   # median order in VND: '1.250.000' read as 1.25
    max_subtotal_mismatch: float = 0.05    # line sum vs the order table's own 'tiền hàng'
    max_orphan_lines: float = 0.05         # lines whose order id is not an order
    max_orders_without_lines: float = 0.05
    min_amount_vs_qty_price: float = 0.90  # line amount = qty x price (after line discount)
    max_qty_base_lt_qty: float = 0.05      # quantity in base unit smaller than in the line unit
    max_debt_negative: float = 0.01
    max_debt_gt_total: float = 0.01


INVARIANTS = Invariants()


@dataclass(frozen=True)
class LLMConfig:
    """LLM parser + ensemble (DECISIONS D14). backend 'off' = rule-only pipeline, unchanged behaviour.
    Env: DATAMATE_LLM (off|vllm|replay), DATAMATE_LLM_MODEL (local weights dir), DATAMATE_LLM_REPLAY (jsonl),
    DATAMATE_LLM_SAMPLES, DATAMATE_LLM_MODEL_ID, DATAMATE_LLM_CALL_ALL, DATAMATE_LLM_RULE_JUDGE (1 = on),
    DATAMATE_LLM_{DTYPE,TP,GPU_UTIL,MAX_LEN}."""
    backend: str = "off"
    model_path: str = ""
    model_id: str = ""                      # cache key part; defaults to the weights dir name
    dtype: str = "bfloat16"
    max_model_len: int = 12288                # prompt ~7k tokens (81 few-shot), prefix-cached
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    max_tokens: int = 256
    n_samples: int = 1                      # > 1: self-consistency votes, sampled at `temperature`
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 2026
    chat_template_kwargs: dict = None       # e.g. {"enable_thinking": False} for Qwen3
    replay_path: str = ""
    # ensemble policy (parse/ensemble.py); defaults keep every current rule answer and only add rescued ones
    rescue_gate: bool = True                # gate-blocked rule intent answered when every LLM sample agrees with it
    require_agreement: bool = False         # also abstain on gate-passed rule intents the LLM disagrees with
    llm_only: bool = False                  # answer when the rule parser has nothing and all samples agree
    llm_only_min_votes: int = 3
    # record mode (D15): call the model on every question so a replay covers every policy; decisions unchanged
    call_all: bool = False
    # rule judge (D18 branch 3, D19, parse/rule_judge.py): an intent blocked by a rulebook change is answered only when
    # every sample says the split's rule computes exactly what the executor computes. OFF until calibrated (D19):
    # DATAMATE_LLM_RULE_JUDGE=1 turns it on (scripts/gpu/run_all.sh records with it on)
    rule_judge: bool = False
    judge_min_votes: int = 1

    def __post_init__(self):
        if self.chat_template_kwargs is None:
            object.__setattr__(self, "chat_template_kwargs", {"enable_thinking": False})
        if not self.model_id and self.model_path:
            object.__setattr__(self, "model_id", Path(self.model_path).name)

    @classmethod
    def from_env(cls, **kw) -> "LLMConfig":
        env = {"backend": os.environ.get("DATAMATE_LLM", "off"), "model_path": os.environ.get("DATAMATE_LLM_MODEL", ""),
               "replay_path": os.environ.get("DATAMATE_LLM_REPLAY", "")}
        if os.environ.get("DATAMATE_LLM_SAMPLES"):
            env["n_samples"] = int(os.environ["DATAMATE_LLM_SAMPLES"])
        if os.environ.get("DATAMATE_LLM_MODEL_ID"):
            env["model_id"] = os.environ["DATAMATE_LLM_MODEL_ID"]
        # GPU-side knobs for the teammate machine (docs/GPU_RUNBOOK.md); none of them is part of the replay key
        for var, field, typ in (("DATAMATE_LLM_DTYPE", "dtype", str), ("DATAMATE_LLM_TP", "tensor_parallel_size", int),
                                ("DATAMATE_LLM_GPU_UTIL", "gpu_memory_utilization", float),
                                ("DATAMATE_LLM_MAX_LEN", "max_model_len", int)):
            if os.environ.get(var):
                env[field] = typ(os.environ[var])
        if os.environ.get("DATAMATE_LLM_CALL_ALL", "") not in ("", "0"):
            env["call_all"] = True
        if os.environ.get("DATAMATE_LLM_RULE_JUDGE", "") not in ("", "0"):
            env["rule_judge"] = True
        return cls(**{**env, **kw})


def cache_root() -> Path:
    """Writable cache dir. SPEC §7 câu 2: in Docker /data is read-only and temp files go to /tmp, so predict.sh sets
    DATAMATE_CACHE=/tmp/datamate_cache. Default (dev): <repo>/cache."""
    env = os.environ.get("DATAMATE_CACHE")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "cache"
