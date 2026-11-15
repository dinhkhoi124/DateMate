"""Engine probe (D29): pick the first vLLM engine variant that really runs on this machine.

Each variant of `config.ENGINE_PROFILES` is tried in a fresh subprocess (`python -m datamate.parse.llm_probe <name>`):
load the model, then generate from a long prompt with n samples (prefill, prefix cache, sampling) and from a short one.
A CUDA fault kills only that subprocess; the main process never touched the GPU, so it can try the next variant and
finally load the winner itself. No variant passes -> the caller runs rule-only (run.py).
"""
from __future__ import annotations

import dataclasses
import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Callable

from datamate.config import ENGINE_PROFILES, LLMConfig

OK_MARK = "DATAMATE_PROBE_OK"


def profile_names(cfg: LLMConfig) -> list[str]:
    """Requested variants, in order; unknown names are reported and skipped."""
    names = [n.strip() for n in cfg.profiles.split(",") if n.strip()]
    for n in names:
        if n not in ENGINE_PROFILES:
            print(f"[llm] unknown engine profile {n!r}: skipped", file=sys.stderr)
    return [n for n in names if n in ENGINE_PROFILES]


def apply_profile(cfg: LLMConfig, name: str) -> LLMConfig:
    """Set the variant's vLLM env vars (process-wide) and return the config with its field overrides."""
    p = ENGINE_PROFILES[name]
    os.environ.update(p["env"])
    return dataclasses.replace(cfg, **p["cfg"])


def run_probe(name: str, timeout_s: float) -> tuple[bool, str]:
    """Probe one variant in its own process group; (passed, output). Passed = the child printed OK_MARK, even if its
    teardown then hung. The whole group is killed afterwards so no engine process keeps GPU memory."""
    with tempfile.TemporaryFile("w+", encoding="utf8", errors="replace") as out:
        kw = {"start_new_session": True} if os.name == "posix" else {}
        proc = subprocess.Popen([sys.executable, "-m", "datamate.parse.llm_probe", name], stdout=out,
                                stderr=subprocess.STDOUT, env=os.environ.copy(), **kw)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            print(f"[llm] probe {name}: no result after {timeout_s:.0f}s", file=sys.stderr)
        _kill_group(proc)
        out.seek(0)
        text = out.read()
    return OK_MARK in text, text


def _kill_group(proc: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    wait_gpu_free()


def gpu_used_mib() -> list[int] | None:
    """Memory used per visible GPU (nvidia-smi, no CUDA init in this process); None when it cannot be read."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        vals = [int(x) for x in r.stdout.split()]
        return vals if r.returncode == 0 and vals else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def wait_gpu_free(limit_mib: int = 2048, timeout_s: float = 90.0, read=gpu_used_mib, sleep=time.sleep) -> bool:
    """Wait until the killed engine's GPU memory is back (V1 refuses to start when free memory < its utilisation
    target). True = every GPU under `limit_mib`; unreadable -> a fixed 5 s pause, as before."""
    sleep(2)
    t0 = time.monotonic()
    while True:
        used = read()
        if used is None:
            sleep(5)
            return False
        if max(used) < limit_mib:
            return True
        if time.monotonic() - t0 > timeout_s:
            print(f"[llm] GPU memory still in use after {timeout_s:.0f}s: {used} MiB", file=sys.stderr)
            return False
        sleep(2)


def choose_profile(cfg: LLMConfig, probe: Callable[[str, float], tuple[bool, str]] = run_probe) -> str | None:
    """First variant whose probe passes; None when every one fails. A failed probe's log tail goes to stderr (it is
    the organisers' only view of why the model did not run)."""
    for name in profile_names(cfg):
        t = time.monotonic()
        ok, text = probe(name, cfg.probe_timeout_s)
        print(f"[llm] engine profile {name}: {'ok' if ok else 'FAILED'} ({time.monotonic() - t:.0f}s)", file=sys.stderr)
        if ok:
            return name
        tail = "\n".join(text.strip().splitlines()[-40:])
        print(f"[llm] probe {name} output (last 40 lines):\n{tail}", file=sys.stderr)
    return None


def probe_prompts() -> tuple[str, str]:
    """A ~3k-token table-like prompt (long prefill, shared prefix for n samples) and a short one."""
    rows = "\n".join(f"| {i} | sản phẩm {i % 37} | {i % 11} | {(i * 7919) % 100000} |" for i in range(300))
    long = ("Bảng bán hàng:\n| dòng | sản phẩm | số lượng | thành tiền |\n" + rows +
            "\n\nViết một câu SQL đếm số dòng của bảng sales. Chỉ trả về SQL.")
    return long, "Viết câu SQL trả về số 1. Chỉ trả về SQL."


def _main(name: str) -> int:
    from datamate.parse.llm import VLLMBackend  # noqa: PLC0415  (GPU-only dependency)
    cfg = apply_profile(LLMConfig.from_env(), name)
    be = VLLMBackend(cfg)
    long, short = probe_prompts()
    n = 4
    for prompts, k in (([long], n), ([long], n), ([short], 1)):   # second long call: prefix-cache hit path
        outs = be.complete(prompts, k, free=True, max_tokens=24)
        if len(outs) != 1 or len(outs[0]) != k or not any(t.strip() for t in outs[0]):
            print(f"[probe] {name}: bad output {outs!r}", flush=True)
            return 1
    print(f"{OK_MARK} {name}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    try:
        code = _main(sys.argv[1])
    except BaseException:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)   # skip engine teardown (it can hang after a CUDA fault): the parent kills the process group
