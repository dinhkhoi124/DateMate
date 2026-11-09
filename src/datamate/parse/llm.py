"""LLM parser (DECISIONS D14): question -> model JSON (guided by llm_schema.json_schema) -> Intent.

Backends are swappable so that everything except the model call runs and is tested on CPU:
- `VLLMBackend`: offline vLLM engine, guided JSON, temperature/seed from config (GPU, Docker image).
- `ReplayBackend`: answers from a jsonl cache recorded on a GPU machine (`record=True` wraps another backend and
  appends). Lets the ensemble policy be calibrated on CPU without re-running the model.
- any object with `complete(prompts, n) -> list[list[str]]` (tests use a fake).
No network access anywhere (CLAUDE.md rule 6): vLLM loads weights from a local path.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Protocol

from datamate.config import LLMConfig
from datamate.parse.intent import Intent
from datamate.parse.llm_schema import FILTER_SCHEMA, NONE_OP, OPS, json_schema, to_intent

FEWSHOT_PATH = Path(__file__).with_name("fewshot.jsonl")


class Backend(Protocol):
    name: str

    def complete(self, prompts: list[str], n: int = 1, schema: dict | None = None, free: bool = False,
                 max_tokens: int | None = None) -> list[list[str]]:
        """For each prompt, n raw completions (JSON strings, guided by `schema`, default the intent contract;
        `free=True`: unguided text, e.g. code for the code-gen agent, D20). A failed call returns [] for that prompt."""
        ...


def prompt_key(model: str, prompt: str, n: int, temperature: float, seed: int) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([model, prompt, n, temperature, seed], ensure_ascii=False).encode("utf8"))
    return h.hexdigest()


class ReplayBackend:
    """jsonl {"key", "outputs"}: replay a recorded run; with `inner`, record misses (append, flushed per prompt)."""

    def __init__(self, path: Path, cfg: LLMConfig, inner: Backend | None = None):
        self.path, self.cfg, self.inner = Path(path), cfg, inner
        self.name = f"replay:{cfg.model_id}"
        self.cache: dict[str, list[str]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.cache[r["key"]] = r["outputs"]
        self.misses = 0

    def complete(self, prompts: list[str], n: int = 1, schema: dict | None = None, free: bool = False,
                 max_tokens: int | None = None) -> list[list[str]]:
        """The key is the prompt (not the schema): every caller builds a distinct prompt (intent, judge, code-gen)."""
        c = self.cfg
        keys = [prompt_key(c.model_id, p, n, c.temperature, c.seed) for p in prompts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo and self.inner is not None:
            kw = {k: v for k, v in (("schema", schema), ("free", free), ("max_tokens", max_tokens)) if v}
            outs = self.inner.complete([prompts[i] for i in todo], n, **kw)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf8", newline="\n") as w:
                for i, o in zip(todo, outs):
                    if o:  # failed calls are not cached
                        self.cache[keys[i]] = o
                        w.write(json.dumps({"key": keys[i], "outputs": o}, ensure_ascii=False) + "\n")
        self.misses += sum(1 for k in keys if k not in self.cache)
        return [self.cache.get(k, []) for k in keys]


class VLLMBackend:
    """Offline vLLM engine with JSON-schema guided decoding. Imported lazily: the CPU dev env has no vllm."""

    def __init__(self, cfg: LLMConfig):
        from vllm import LLM  # noqa: PLC0415  (optional dependency, extra 'llm')
        self.cfg = cfg
        self.name = f"vllm:{cfg.model_id}"
        self.llm = LLM(model=cfg.model_path, dtype=cfg.dtype, seed=cfg.seed, max_model_len=cfg.max_model_len,
                       gpu_memory_utilization=cfg.gpu_memory_utilization, enable_prefix_caching=True,
                       tensor_parallel_size=cfg.tensor_parallel_size)
        self.tok = self.llm.get_tokenizer()

    def _sampling(self, n: int, schema: dict | None = None, free: bool = False, max_tokens: int | None = None):
        from vllm import SamplingParams  # noqa: PLC0415
        c = self.cfg
        if free:   # unguided (code-gen agent, D20)
            return SamplingParams(n=n, temperature=c.temperature if n > 1 else 0.0, top_p=c.top_p if n > 1 else 1.0,
                                  seed=c.seed, max_tokens=max_tokens or c.max_tokens)
        try:  # vLLM >= 0.10
            from vllm.sampling_params import StructuredOutputsParams  # noqa: PLC0415
            guided = {"structured_outputs": StructuredOutputsParams(json=schema or json_schema())}
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams  # noqa: PLC0415
            guided = {"guided_decoding": GuidedDecodingParams(json=schema or json_schema())}
        return SamplingParams(n=n, temperature=c.temperature if n > 1 else 0.0, top_p=c.top_p if n > 1 else 1.0,
                              seed=c.seed, max_tokens=max_tokens or c.max_tokens, **guided)

    def complete(self, prompts: list[str], n: int = 1, schema: dict | None = None, free: bool = False,
                 max_tokens: int | None = None) -> list[list[str]]:
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, **self.cfg.chat_template_kwargs)
                 for p in prompts]
        try:
            res = self.llm.generate(texts, self._sampling(n, schema, free, max_tokens), use_tqdm=False)
        except Exception:  # never crash a run; traceback only (no prompt text) so the GPU log can be shared
            import traceback  # noqa: PLC0415
            print(f"[llm] generate failed:\n{traceback.format_exc(limit=3)}", file=sys.stderr)
            return [[] for _ in prompts]
        return [[o.text for o in r.outputs] for r in res]


def make_backend(cfg: LLMConfig) -> Backend | None:
    if cfg.backend == "off":
        return None
    if cfg.backend == "vllm":
        inner = VLLMBackend(cfg)
        return ReplayBackend(cfg.replay_path, cfg, inner) if cfg.replay_path else inner
    if cfg.backend == "replay":
        return ReplayBackend(cfg.replay_path, cfg)
    raise ValueError(f"unknown LLM backend {cfg.backend!r}")


# --------------------------------------------------------------------------- prompt
def load_fewshot(path: Path = FEWSHOT_PATH) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _instructions() -> str:
    ops = "\n".join(f"- {k}: {s.desc}" + (f" (args: {', '.join(s.args)})" if s.args else "") for k, s in OPS.items())
    enums = "\n".join(f"- {k}: {', '.join(str(x) for x in v['enum'] if x is not None)}"
                      for k, v in FILTER_SCHEMA.items() if "enum" in v)
    return (
        "Bạn chuyển câu hỏi tiếng Việt của chủ shop nhỏ thành JSON mô tả phép tính. KHÔNG tự tính số.\n"
        "Thời gian (tháng, tuần, ngày, quý, Tết…) do hệ thống tự đọc từ câu hỏi: không ghi vào JSON.\n"
        f"Nếu không phép tính nào dưới đây đúng nghĩa câu hỏi, trả op = \"{NONE_OP}\".\n\n"
        f"Các op:\n{ops}\n\n"
        "filters: product, customer, category, ingredient, staff là tên chép nguyên từ câu hỏi (null nếu không có).\n"
        f"Các filter có giá trị cố định:\n{enums}\n"
        "args.metric: revenue (tiền), orders (số đơn), qty (số lượng hàng). args.compare: prev (kỳ liền trước), "
        "yoy (cùng kỳ năm trước), explicit (hai kỳ nêu rõ). args.inclusive: false khi 'quá/hơn/trên N', "
        "true khi 'từ N/ít nhất N/N trở lên'.\n"
    )


def build_prompt(question: str, examples: list[dict]) -> str:
    """Static prefix (instructions + few-shot) then the question: vLLM prefix caching makes the prefix free."""
    shots = "\n".join(f"Câu hỏi: {e['question']}\nJSON: {json.dumps(e['target'], ensure_ascii=False)}"
                      for e in examples)
    return f"{_instructions()}\nVí dụ:\n{shots}\n\nCâu hỏi: {question}\nJSON:"


def parse_json(text: str) -> dict | None:
    t = (text or "").strip()
    if "</think>" in t:  # reasoning models: keep only the answer
        t = t.split("</think>", 1)[1].strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < a:
        return None
    try:
        js = json.loads(t[a:b + 1])
    except json.JSONDecodeError:
        return None
    return js if isinstance(js, dict) else None


class LLMParser:
    def __init__(self, backend: Backend, cfg: LLMConfig, examples: list[dict] | None = None):
        self.backend, self.cfg = backend, cfg
        self.examples = load_fewshot() if examples is None else examples
        # backends swallow errors and return [] (never crash a run); count them so a GPU run that silently fell
        # back to rule-only shows up in the summary (docs/GPU_RUNBOOK.md)
        self.calls = self.failed = 0

    def prompt(self, question: str) -> str:
        return build_prompt(question, self.examples)

    def stats_line(self) -> str:
        return f"llm_calls={self.calls} llm_failed={self.failed}"

    def parse(self, question: str, periods, lk) -> tuple[list[Intent | None], list[str]]:
        """n samples -> (intents, raw outputs). An unparsable or out-of-contract sample gives None."""
        outs = self.backend.complete([self.prompt(question)], self.cfg.n_samples)[0]
        self.calls += 1
        self.failed += not outs
        its = [to_intent(parse_json(o), periods, lk) for o in outs]
        return its, outs
