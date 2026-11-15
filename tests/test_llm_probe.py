"""D29: engine probe ladder (parse/llm_probe.py). The GPU side is faked; the subprocess plumbing runs for real on CPU."""
import os
from pathlib import Path

import pytest

from datamate.config import ENGINE_PROFILES, LLMConfig
from datamate.parse import llm as llm_mod
from datamate.parse import llm_probe


def test_every_profile_avoids_torch_compile_and_uses_one_gpu():
    # rulebook-free D29 invariant: the 15/11 crash was in a torch.compile (Inductor/Triton) kernel under TP=2
    assert ENGINE_PROFILES
    for name, p in ENGINE_PROFILES.items():
        assert p["cfg"]["enforce_eager"] is True, name
        assert p["cfg"]["tensor_parallel_size"] == 1, name
        assert set(p["cfg"]) <= set(LLMConfig.__dataclass_fields__), name


def test_profile_names_keep_order_and_skip_unknown(capsys):
    cfg = LLMConfig(profiles=" v0-xformers-tp1, nope ,v1-eager-tp1")
    assert llm_probe.profile_names(cfg) == ["v0-xformers-tp1", "v1-eager-tp1"]
    assert "nope" in capsys.readouterr().err


def test_choose_profile_takes_the_first_that_passes(capsys):
    tried = []

    def probe(name, timeout):
        tried.append((name, timeout))
        return name == "v1-eager-tp1-awq-flashinfer", f"log of {name}\nCUDA error: an illegal memory access"

    cfg = LLMConfig(profiles="v1-eager-tp1,v1-eager-tp1-awq-flashinfer,v0-xformers-tp1", probe_timeout_s=7)
    assert llm_probe.choose_profile(cfg, probe) == "v1-eager-tp1-awq-flashinfer"
    assert tried == [("v1-eager-tp1", 7), ("v1-eager-tp1-awq-flashinfer", 7)]
    err = capsys.readouterr().err
    assert "v1-eager-tp1: FAILED" in err and "illegal memory access" in err


def test_choose_profile_none_when_all_fail():
    cfg = LLMConfig(profiles="v1-eager-tp1,v0-xformers-tp1")
    assert llm_probe.choose_profile(cfg, lambda n, t: (False, "boom")) is None


def test_apply_profile_sets_env_and_fields(monkeypatch):
    for k in ("VLLM_USE_V1", "VLLM_ATTENTION_BACKEND", "VLLM_USE_FLASHINFER_SAMPLER"):
        monkeypatch.delenv(k, raising=False)
    cfg = LLMConfig(backend="vllm", model_path="/m/Qwen3-14B-AWQ", tensor_parallel_size=2, profiles="v0-xformers-tp1")
    out = llm_probe.apply_profile(cfg, "v0-xformers-tp1")
    assert os.environ["VLLM_USE_V1"] == "0" and os.environ["VLLM_ATTENTION_BACKEND"] == "XFORMERS"
    assert (out.tensor_parallel_size, out.enforce_eager, out.quantization, out.prefix_caching) == (1, True, "awq", False)
    # the replay key parts never change with the engine variant
    assert (out.model_id, out.temperature, out.seed) == (cfg.model_id, cfg.temperature, cfg.seed)


class _FakeVLLM:
    def __init__(self, cfg):
        self.cfg = cfg


def test_make_backend_loads_the_chosen_profile(monkeypatch):
    monkeypatch.setattr(llm_mod, "VLLMBackend", _FakeVLLM)
    monkeypatch.setattr(llm_probe, "choose_profile", lambda cfg: "v1-eager-tp1")
    monkeypatch.setattr(llm_probe, "apply_profile",
                        lambda cfg, name: llm_probe.dataclasses.replace(cfg, **ENGINE_PROFILES[name]["cfg"]))
    be = llm_mod.make_backend(LLMConfig(backend="vllm", model_path="/m/x", tensor_parallel_size=2,
                                        profiles="v1-eager-tp1"))
    assert be.cfg.enforce_eager and be.cfg.tensor_parallel_size == 1


def test_make_backend_raises_when_no_profile_runs(monkeypatch):
    # run.py catches this and runs rule-only (the 15/11 behaviour, now after trying every variant)
    monkeypatch.setattr(llm_mod, "VLLMBackend", _FakeVLLM)
    monkeypatch.setattr(llm_probe, "choose_profile", lambda cfg: None)
    with pytest.raises(RuntimeError, match="no vLLM engine profile"):
        llm_mod.make_backend(LLMConfig(backend="vllm", model_path="/m/x", profiles="v1-eager-tp1"))


def test_make_backend_without_profiles_loads_once(monkeypatch):
    monkeypatch.setattr(llm_mod, "VLLMBackend", _FakeVLLM)
    monkeypatch.setattr(llm_probe, "choose_profile", lambda cfg: pytest.fail("no probing without profiles"))
    be = llm_mod.make_backend(LLMConfig(backend="vllm", model_path="/m/x", tensor_parallel_size=2))
    assert be.cfg.tensor_parallel_size == 2 and not be.cfg.enforce_eager


def test_run_probe_reports_a_failing_child(monkeypatch):
    # real subprocess: no vLLM on the CPU env, so the child fails fast and never prints the OK mark
    src = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.setenv("PYTHONPATH", src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("DATAMATE_LLM_MODEL", "/nonexistent")
    ok, text = llm_probe.run_probe("v1-eager-tp1", 120)
    assert not ok and llm_probe.OK_MARK not in text and "Traceback" in text


def test_probe_prompt_is_long_enough_to_exercise_prefill():
    long, short = llm_probe.probe_prompts()
    assert len(long) > 8000 and len(short) < 100
