"""CPU offload knob (team's 8 GB RTX 4060 runs the 14B AWQ model; the organisers' 2x 5090 do not need it)."""
import sys
import types

from datamate.config import LLMConfig


def test_cpu_offload_from_env(monkeypatch):
    monkeypatch.delenv("DATAMATE_LLM_CPU_OFFLOAD_GB", raising=False)
    assert LLMConfig.from_env().cpu_offload_gb == 0.0
    monkeypatch.setenv("DATAMATE_LLM_CPU_OFFLOAD_GB", "6")
    assert LLMConfig.from_env().cpu_offload_gb == 6.0


def _engine_kwargs(monkeypatch, cfg):
    seen = {}

    class FakeLLM:
        def __init__(self, **kw):
            seen.update(kw)

        def get_tokenizer(self):
            return None

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM))
    from datamate.parse.llm import VLLMBackend
    VLLMBackend(cfg)
    return seen


def test_vllm_gets_cpu_offload_only_when_set(monkeypatch):
    assert "cpu_offload_gb" not in _engine_kwargs(monkeypatch, LLMConfig(backend="vllm", model_path="m"))
    kw = _engine_kwargs(monkeypatch, LLMConfig(backend="vllm", model_path="m", cpu_offload_gb=6.0))
    assert kw["cpu_offload_gb"] == 6.0


def _fake_vllm(monkeypatch, n_gpus):
    seen = {}

    class FakeLLM:
        def __init__(self, **kw):
            seen.update(kw)

        def get_tokenizer(self):
            return None

    utils = types.SimpleNamespace(cuda_device_count_stateless=lambda: n_gpus)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM, utils=utils))
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    return seen


def test_tensor_parallel_never_exceeds_visible_gpus(monkeypatch):
    """TP=2 on one visible GPU hung vLLM (Ray waits for the 2nd GPU); 2 GPUs keep TP=2 (organisers' machine)."""
    from datamate.parse.llm import VLLMBackend
    cfg = LLMConfig(backend="vllm", model_path="m", tensor_parallel_size=2)
    for n, want in ((1, 1), (2, 2), (4, 2), (0, 2)):   # 0 = count unknown -> as configured
        seen = _fake_vllm(monkeypatch, n)
        VLLMBackend(cfg)
        assert seen["tensor_parallel_size"] == want, (n, seen)
