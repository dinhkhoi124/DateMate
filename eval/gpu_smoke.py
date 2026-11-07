"""GPU smoke test for the LLM parser (D14/D15, docs/GPU_RUNBOOK.md step 4). Run on the GPU machine, not in CI.

PYTHONPATH=src python eval/gpu_smoke.py --model models/Qwen3-8B --out eval/runs/gpu/smoke_Qwen3-8B.json

Checks, with made-up questions only (no competition text):
1. versions (python, torch, CUDA, vllm, xgrammar) and which structured-output API this vLLM exposes;
2. the model loads with config.LLMConfig (+ DATAMATE_LLM_* env overrides) and how long that takes;
3. guided JSON: every output parses and its op is in the contract enum, for n=1 (temp 0) and n=3 (sampling);
4. determinism: the n=1 run repeated gives identical text.
Exit code 0 only if 3 and 4 hold.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datamate.config import LLMConfig  # noqa: E402
from datamate.parse.llm import FEWSHOT_PATH, LLMParser, VLLMBackend, parse_json  # noqa: E402
from datamate.parse.llm_schema import json_schema  # noqa: E402

# Written for this test, not taken from any split.
QUESTIONS = [
    "Doanh thu tháng 5 của tiệm được bao nhiêu tiền?",
    "Tuần trước có tổng cộng mấy đơn bị huỷ?",
    "Món nào bán chạy nhất trong quý này tính theo số lượng?",
    "Hôm nay trời có mưa không?",
]


def version(mod: str) -> str:
    try:
        return getattr(importlib.import_module(mod), "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return f"missing ({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local weights dir, e.g. models/Qwen3-8B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rep: dict = {"python": platform.python_version(), "torch": version("torch"), "vllm": version("vllm"),
                 "xgrammar": version("xgrammar"), "transformers": version("transformers")}
    try:
        import torch
        rep["cuda"] = torch.version.cuda
        rep["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:  # noqa: BLE001
        rep["cuda"] = f"error {e!r}"
    try:
        from vllm.sampling_params import StructuredOutputsParams  # noqa: F401
        rep["structured_api"] = "StructuredOutputsParams"
    except ImportError:
        rep["structured_api"] = "GuidedDecodingParams"
    rep["fewshot_present"] = FEWSHOT_PATH.exists()

    cfg = LLMConfig.from_env(backend="vllm", model_path=args.model)
    rep["config"] = {k: getattr(cfg, k) for k in ("model_id", "dtype", "max_model_len", "gpu_memory_utilization",
                                                  "tensor_parallel_size", "seed", "temperature")}
    t0 = time.time()
    be = VLLMBackend(cfg)
    rep["load_s"] = round(time.time() - t0, 1)
    parser = LLMParser(be, cfg)
    prompts = [parser.prompt(q) for q in QUESTIONS]
    rep["prompt_tokens"] = len(be.tok.encode(prompts[0]))
    ops = set(json_schema()["properties"]["op"]["enum"])

    ok = True
    for n in (1, 3):
        t0 = time.time()
        outs = be.complete(prompts, n)
        dt = time.time() - t0
        rows = []
        for q, o in zip(QUESTIONS, outs):
            js = [parse_json(x) for x in o]
            valid = len(o) == n and all(j is not None and j.get("op") in ops for j in js)
            ok &= valid
            rows.append({"question": q, "valid": valid, "outputs": o})
        rep[f"n{n}"] = {"seconds": round(dt, 2), "rows": rows}
    again = be.complete(prompts, 1)
    rep["deterministic_n1"] = again == [r["outputs"] for r in rep["n1"]["rows"]]
    ok &= rep["deterministic_n1"]
    rep["ok"] = ok

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("n1", "n3")}, ensure_ascii=False, indent=1))
    for n in (1, 3):
        print(f"n={n}: {rep[f'n{n}']['seconds']}s, valid {sum(r['valid'] for r in rep[f'n{n}']['rows'])}/{len(QUESTIONS)}")
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
