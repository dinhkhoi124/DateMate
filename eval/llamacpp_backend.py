"""Local llama.cpp backend for trying the agents on a personal GPU (dev only, never in the image).

Talks to a `llama-server` running on this machine (127.0.0.1) through its OpenAI-compatible chat endpoint, so a
laptop with a small GPU can run a GGUF model with part of the layers in system RAM. Lives in eval/, not src/: the
submitted image uses vLLM and src/ makes no network call (CLAUDE.md rule 6).

Numbers from this backend are NOT the submission config (other quantisation, other engine): use them to catch
format/SQL failure modes, never to choose tau, m, policy or prompts (D21: those are measured with vLLM on train).

    llama-server -m Qwen3-14B-Q4_K_M.gguf -ngl 24 -c 20480 --parallel 1 -fa on -ctk q8_0 -ctv q8_0 --jinja
    PYTHONPATH=src python eval/run_eval.py --split dev --agent sql --llamacpp http://127.0.0.1:8080         --llamacpp-workers 1 --per-shop 1 --trace

Context per slot = -c / --parallel must hold the longest prompt (~15k tokens: rulebook + README + schema) plus the
reply: one slot of 20480 (same as max_model_len of the image) fits an 8 GB laptop GPU with -ngl ~24.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class LlamaCppBackend:
    def __init__(self, url: str, cfg, workers: int = 2, timeout_s: float = 900.0):
        self.url, self.cfg, self.workers, self.timeout = url.rstrip("/"), cfg, workers, timeout_s
        self.name = f"llamacpp:{cfg.model_id}"
        if not url.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("llama.cpp backend only talks to a server on this machine")
        self._wait_ready()

    def _wait_ready(self, tries: int = 60):
        for _ in range(tries):
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise RuntimeError(f"llama-server not ready at {self.url}")

    def _one(self, prompt: str, seed: int, temperature: float, top_p: float, max_tokens: int) -> str:
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": temperature, "top_p": top_p,
                "seed": seed, "max_tokens": max_tokens, "cache_prompt": True,
                "chat_template_kwargs": dict(self.cfg.chat_template_kwargs or {})}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode("utf8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read().decode("utf8"))
            return out["choices"][0]["message"].get("content") or ""
        except Exception as e:  # noqa: BLE001  (a failed sample is dropped, never crashes a run)
            print(f"[llamacpp] request failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            return ""

    def complete(self, prompts: list[str], n: int = 1, schema: dict | None = None, free: bool = False,
                 max_tokens: int | None = None) -> list[list[str]]:
        """n samples per prompt with seeds seed..seed+n-1. Prompts go one after another (all samples of one prompt
        in parallel slots) so the long shared prefix stays in the server's prompt cache."""
        if not free:
            print("[llamacpp] guided JSON not supported here; only the agents (free text) use this backend",
                  file=sys.stderr)
            return [[] for _ in prompts]
        c = self.cfg
        temp, top_p = (c.temperature, c.top_p) if n > 1 else (0.0, 1.0)
        mt = max_tokens or c.max_tokens
        res = []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for i, p in enumerate(prompts):
                t = time.perf_counter()
                outs = list(ex.map(lambda s: self._one(p, c.seed + s, temp, top_p, mt), range(n)))
                res.append([o for o in outs if o])
                print(f"[llamacpp] prompt {i + 1}/{len(prompts)}: {n} samples in {time.perf_counter() - t:.0f}s",
                      file=sys.stderr)
        return res
