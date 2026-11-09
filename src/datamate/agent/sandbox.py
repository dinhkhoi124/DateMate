"""Run model-written `solve(T, H)` in a separate worker process with a timeout (D20, ARCHITECTURE §8).

One worker per shop holds the shop's tables (Store is cache-backed, so a restart is cheap). Code runs with a
restricted `__import__` (data/maths modules only) and without file/exec builtins; each call gets fresh copies of the
tables so one sample cannot leak state into the next. A timeout or crash kills the worker; it is respawned lazily.
Runs offline: nothing here touches the network (CLAUDE.md rule 6), and in Docker /data is mounted read-only anyway.
"""
from __future__ import annotations

import builtins
import multiprocessing as mp
import traceback
import types
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SAFE_MODULES = frozenset({"math", "datetime", "re", "collections", "itertools", "functools", "statistics",
                          "unicodedata", "decimal", "fractions", "calendar", "pandas", "numpy", "operator", "string",
                          # pure-computation stdlib; time/_strptime are imported lazily by datetime.strptime and
                          # pd.to_datetime(format=...) from the calling (sandboxed) frame
                          "time", "_strptime", "json", "typing", "dataclasses", "copy", "bisect", "heapq",
                          "numbers", "enum", "warnings", "zoneinfo", "dateutil", "pytz"})
_PARAM_NAMES = frozenset({"H", "T"})   # models often write `import H`; solve's parameters shadow it anyway
_BLOCKED = ("open", "exec", "eval", "compile", "input", "exit", "quit", "breakpoint", "help", "memoryview", "globals",
            "vars", "locals", "__import__")


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name in _PARAM_NAMES and not fromlist:
        return types.ModuleType(name)
    if level != 0 or name.split(".")[0] not in SAFE_MODULES:
        raise ImportError(f"import of {name!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def safe_builtins() -> dict:
    b = {k: v for k, v in vars(builtins).items() if k not in _BLOCKED}
    b["__import__"] = _guarded_import
    return b


def exec_solve(code: str, T: dict, H) -> object:
    """Execute `code` (must define solve(T, H)) in restricted globals and return solve's result."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    g = {"__builtins__": safe_builtins(), "pd": pd, "np": np, "date": date, "__name__": "solution"}
    exec(compile(code, "<solution>", "exec"), g)  # noqa: S102  (sandboxed model code is the point of this module)
    fn = g.get("solve")
    if not callable(fn):
        raise ValueError("code does not define solve(T, H)")
    return fn({k: v.copy() for k, v in T.items()}, H)


@dataclass
class Job:
    code: str
    as_of: str           # ISO date
    question_norm: str
    question: str = ""   # raw text, for catalog lookup (helpers.Helpers._find)


# --------------------------------------------------------------------------- worker side
def _load_shop(root: str, biz: str, rulebook: str | None):
    from datamate.agent.tables import shop_tables  # noqa: PLC0415
    return shop_tables(Path(root), biz, Path(rulebook) if rulebook else None)


def _worker(conn, root: str, biz: str, rulebook: str | None):
    from datamate.agent.answer import canon  # noqa: PLC0415
    try:
        T, make_h = _load_shop(root, biz, rulebook)
    except Exception:  # noqa: BLE001
        conn.send(("init_error", traceback.format_exc(limit=3)))
        return
    conn.send(("ready", None))
    while True:
        try:
            job = conn.recv()
        except EOFError:
            return
        if job is None:
            return
        try:
            res = exec_solve(job.code, T, make_h(date.fromisoformat(job.as_of), job.question_norm, job.question))
            conn.send(("ok", canon(res)))
        except Exception as e:  # noqa: BLE001  (model code: any error just discards this sample)
            conn.send(("error", f"{type(e).__name__}: {e}"[:300]))


# --------------------------------------------------------------------------- parent side
class ShopSandbox:
    """Worker process for one shop. run(job) -> ('ok', row) | ('error', msg) | ('timeout', None)."""

    def __init__(self, root: Path, biz: str, rulebook: Path | None = None, timeout_s: float = 20.0,
                 init_timeout_s: float = 600.0):
        self.args = (str(root), biz, str(rulebook) if rulebook else None)
        self.timeout_s, self.init_timeout_s = timeout_s, init_timeout_s
        self.proc = self.conn = None
        self.init_error = None

    def _start(self) -> bool:
        ctx = mp.get_context("spawn")      # same behaviour on Windows dev box and Linux image
        parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, *self.args), daemon=True)
        self.proc.start()
        self.conn = parent
        if not parent.poll(self.init_timeout_s):
            self.close()
            self.init_error = "worker init timeout"
            return False
        kind, msg = parent.recv()
        if kind != "ready":
            self.close()
            self.init_error = msg
            return False
        return True

    def run(self, job: Job) -> tuple[str, object]:
        if self.proc is None or not self.proc.is_alive():
            if self.init_error or not self._start():
                return "error", f"sandbox init failed: {self.init_error}"
        try:
            self.conn.send(job)
            if not self.conn.poll(self.timeout_s):
                self.close()          # kill runaway code; next run respawns
                return "timeout", None
            return self.conn.recv()
        except (EOFError, BrokenPipeError, OSError) as e:
            self.close()
            return "error", f"worker died: {type(e).__name__}"

    def close(self):
        if self.conn is not None:
            try:
                self.conn.send(None)
            except Exception:  # noqa: BLE001
                pass
        if self.proc is not None:
            self.proc.join(timeout=1)
            if self.proc.is_alive():
                self.proc.kill()
        self.proc = self.conn = None
