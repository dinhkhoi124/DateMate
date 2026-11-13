"""Agent-generalization benchmark on synthetic shops (E0, docs/BENCHMARK.md).

    PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py build                  # cells + variants (CPU, deterministic)
    PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py ingest                 # ingest fidelity per family (CPU)
    PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py run --arm base [--side A] [--agent codegen --llm vllm ...]
    PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py score <run_dir> [<run_dir> ...] [--baseline <run_dir> ...]

Cells = (question, family). Every selected question has an `original` cell and one family cell, so every family is
compared with the original on the SAME questions. Side A = Synth-Tune, side B = Synth-Eval (aggregates only).
Dev shops only (CLAUDE.md rule 2). data/ is only read (rule 7). bench_cells.json holds ids and labels only (no
question text, no answers: the repo is public, D14); recomputed golds stay in eval/synth/out (gitignored).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402

from datamate.ingest.checks import OP_AREAS  # noqa: E402
from datamate.ingest.store import Store  # noqa: E402
from eval.scorer import is_correct, load_jsonl, score  # noqa: E402
from eval.synth import families as F  # noqa: E402
from eval.synth.make import TRAIN, load, save  # noqa: E402

SPLITS = ROOT / "eval" / "synth" / "splits_gen.json"
CELLS = ROOT / "eval" / "synth" / "bench_cells.json"
BENCH = ROOT / "eval" / "synth" / "out" / "bench"
RECOMP = BENCH / "gold_recomputed.json"
DROP_AREA = {"imports": "imports", "stock": "stock", "lots": "lots"}
ABSTAINISH = (None, "abstain")


# ====================================================================== build
def gen_hash() -> str:
    """Generator identity: cells and variants are valid only for this generation code + split config (scoring code
    in this file is not part of it, so metrics can be added without unfreezing the benchmark)."""
    import inspect   # noqa: PLC0415
    h = hashlib.sha1()
    for f in (Path(F.__file__), ROOT / "eval" / "synth" / "make.py", SPLITS,
              ROOT / "src" / "datamate" / "ingest" / "reader.py", ROOT / "src" / "datamate" / "ingest" / "readme.py"):
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    for fn in (build, _strat_pick, _gen_shop, _cell):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()[:16]


def load_splits() -> dict:
    cfg = json.loads(SPLITS.read_text(encoding="utf8"))
    holdout = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    if (set(cfg["A"]) | set(cfg["B"])) & holdout or set(cfg["A"]) & set(cfg["B"]):
        sys.exit("splits_gen.json: holdout shop used, or a shop on both sides (CLAUDE.md rule 2, BENCHMARK.md §3)")
    return cfg


def rule_pass(qs: list[dict], root: Path, rulebook: Path | None = None) -> dict:
    """Deterministic rule engine (no LLM): qid -> {op, row}."""
    from datamate.pipeline import Context, answer_many   # noqa: PLC0415
    ctx = Context(root=root, rulebook=rulebook)
    out = {}
    for q, row, tr in answer_many(qs, ctx):
        out[q["id"]] = {"op": (tr.get("inputs") or {}).get("op"), "row": row}
    return out


def _strat_pick(qids: list[str], golds: dict, n: int, rng: random.Random) -> list[str]:
    """n questions spread over gold types (round robin over types, random inside a type)."""
    by = defaultdict(list)
    for q in qids:
        by[golds[q]["type"]].append(q)
    for v in by.values():
        rng.shuffle(v)
    out, types = [], sorted(by)
    while len(out) < n and any(by.values()):
        for t in types:
            if by[t] and len(out) < n:
                out.append(by[t].pop())
    return out


def _orig_meta(biz: str) -> tuple[list, set]:
    st = Store.build(TRAIN / "data" / biz)
    canc = set()
    for fr in (st.orders, st.lines):
        if fr is not None and "_raw_status" in fr and "status" in fr:
            canc |= {str(x) for x in fr.loc[fr["status"] == "cancelled", "_raw_status"].dropna().unique()}
    return st.table_kinds, canc


FP_DIR = BENCH / "fp"
STAMPS = BENCH / ".stamps"   # one per (side, shop): written only after ALL its variants are on disk


def _fp_path(key: str) -> Path:
    return FP_DIR / (key.replace("|", "__") + ".json")


def _atomic_write(path: Path, text: str):
    """Write-then-rename: an interrupted run never leaves a half-written file under the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf8", newline="\n")
    os.replace(tmp, path)


def _src_hash() -> str:
    """Rule engine identity (rule_pass.json is valid only for this src/datamate code and train rulebook)."""
    h = hashlib.sha1()
    for f in sorted((ROOT / "src" / "datamate").rglob("*.py")) + [TRAIN / "rulebook.md"]:
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def _variant_fp(path: Path, key: str) -> str:
    """Store fingerprint of a variant: saved in full (ingest reuses it) and returned as a short hash (freeze check)."""
    from eval.synth.run import fingerprint   # noqa: PLC0415
    fp = fingerprint(Store.build(path, use_cache=False))
    txt = json.dumps(fp, sort_keys=True, default=str)
    _atomic_write(_fp_path(key), txt)
    return hashlib.sha1(txt.encode()).hexdigest()[:12]


def _gen_shop(side: str, biz: str, seed: int, kinds: list, canc: set, drop_kind: str | None) -> dict:
    """Write every data-family variant of one shop: {family: (applicable, store fingerprint hash)}."""
    shop0 = load(TRAIN / "data" / biz)
    out = {}
    for fam, (_, fn, _) in F.DATA_FAMILIES.items():
        root = BENCH / side / fam
        shop = copy.deepcopy(shop0)
        c = F.Ctx(side, random.Random(f"{seed}-{side}-{fam}-{biz}"), kinds, canc,
                  drop_kind if fam == "drop_table" else None)
        ok = fn(shop, c) if (fam != "drop_table" or drop_kind) else False
        dst = root / "data" / biz
        fp = None
        if ok:
            save(shop, root)
            if fam == "drop_table":
                st = Store.build(dst, use_cache=False)
                ok = not [m for m in F.DROP_GROUPS[drop_kind] if getattr(st, m, None) is not None and len(getattr(st, m))]
            if ok:
                fp = _variant_fp(dst, f"{side}|{fam}|{biz}")
        if not ok and dst.exists():
            shutil.rmtree(dst)
        out[fam] = (ok, fp)
    return out


def _content_fp(path: Path) -> str | None:
    """Hash of a variant's FILE CONTENT, independent of our ingest code: the freeze check must not fail because ingest
    improved. csv/md: bytes. xlsx: every zip part except docProps/ (creation timestamps); the sheet XML written by
    openpyxl 3.1.5 (pinned in docker/requirements.txt) is deterministic. Another openpyxl version may write the same
    cells differently: then compare with `ingest`, not with this hash."""
    if not path.exists():
        return None
    h = hashlib.sha1()
    for f in sorted(p for p in path.iterdir() if p.is_file()):
        _hash_file(h, f)
    return h.hexdigest()[:16]


def _hash_file(h, f: Path) -> None:
    """One file's share of _content_fp (name, then content as described there)."""
    import zipfile   # noqa: PLC0415
    h.update(f.name.encode())
    if f.suffix.lower() in (".xlsx", ".xlsm"):
        with zipfile.ZipFile(f) as z:
            for n in sorted(z.namelist()):
                if not n.startswith("docProps/"):
                    h.update(n.encode())
                    h.update(z.read(n))
    else:
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))


def file_fps(path: Path) -> list[tuple[str, int, str]]:
    """(name, size, hash) of every file of a variant: tells which file differs (names and hashes only, no content)."""
    out = []
    for f in sorted(p for p in path.iterdir() if p.is_file()) if path.exists() else []:
        h = hashlib.sha1()
        try:
            _hash_file(h, f)
            fp = h.hexdigest()[:12]
        except Exception as e:   # noqa: BLE001  (a truncated xlsx is not a valid zip)
            fp = f"ERROR {type(e).__name__}"
        out.append((f.name, f.stat().st_size, fp))
    return out


def seal(args):
    """Add content hashes of the frozen variants to bench_cells.json meta (cells and gen_hash unchanged). Run once,
    on the machine that built the frozen variants; afterwards materialize/ingest check files, not Store output."""
    # check=False: on 2026-11-11 the seal came after an ingest-only change (readme.py status legend); the generation
    # functions and families/make/splits were verified unchanged vs the freeze commit eaead07 before sealing (D24)
    meta, cells = load_cells(check=False)
    if "content_fp" in meta and not args.force:
        sys.exit("already sealed")
    meta["content_fp"] = {k: _content_fp(BENCH / k.split("|")[0] / k.split("|")[1] / "data" / k.split("|")[2])
                          for k in sorted(meta["variant_fp"])}
    meta["rulebook_sha1"] = {f"{s}|{f}": hashlib.sha1((BENCH / s / "rulebooks" / f"{f}.md").read_bytes()).hexdigest()[:16]
                             for s in F.SIDES for f in F.RULEBOOK_FAMILIES}
    meta["sealed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write(CELLS, json.dumps({"meta": meta, "cells": cells}, ensure_ascii=False, indent=1))
    print(f"sealed {len(meta['content_fp'])} variants")


def _stamp_key(seed: int, drop_kind: str | None) -> str:
    """Resume key of one shop's variants: generation code only (families, make, _gen_shop), NOT ingest code, so an
    ingest fix does not force regenerating unchanged variants (content is still verified against the seal)."""
    import inspect   # noqa: PLC0415
    h = hashlib.sha1()
    for f in (Path(F.__file__), ROOT / "eval" / "synth" / "make.py"):
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    h.update(inspect.getsource(_gen_shop).encode())
    return f"{h.hexdigest()[:16]}|{seed}|{drop_kind}"


def _gen_job(job: tuple) -> tuple:
    """Process-pool worker: (side, biz, seed, drop_kind) -> (side, biz, {family: (applicable, fp)})."""
    side, biz, seed, drop_kind = job
    stamp = STAMPS / f"{side}__{biz}.json"
    key = _stamp_key(seed, drop_kind)
    if stamp.exists():   # resume: this shop finished in an earlier (interrupted) run with the same generator
        try:
            st = json.loads(stamp.read_text(encoding="utf8"))
            res = {f: tuple(v) for f, v in st["result"].items()}
            if st["key"] == key and all((BENCH / side / f / "data" / biz).exists() and _fp_path(f"{side}|{f}|{biz}").exists()
                                        for f, (ok, _) in res.items() if ok):
                return side, biz, res
        except (OSError, ValueError, KeyError):
            pass
        stamp.unlink(missing_ok=True)
    kinds, canc = _orig_meta(biz)
    res = _gen_shop(side, biz, seed, kinds, canc, drop_kind)
    _atomic_write(stamp, json.dumps({"key": key, "result": res}))
    return side, biz, res


def _gen_all(jobs: list, n_jobs: int) -> dict:
    """Variants of every (side, shop), in parallel (independent shops, deterministic per shop)."""
    from concurrent.futures import ProcessPoolExecutor   # noqa: PLC0415
    out, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=max(1, n_jobs)) as ex:
        for side, biz, res in ex.map(_gen_job, jobs):
            out[(side, biz)] = res
            print(f"  variants {side} {biz}: {sum(ok for ok, _ in res.values())}/{len(res)} families "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return out


def materialize(args):
    """Re-create the frozen variants and rulebooks from bench_cells.json (e.g. on Kaggle) without re-selecting
    questions; every variant is checked against its frozen Store fingerprint."""
    meta, cells = load_cells()
    md = (TRAIN / "rulebook.md").read_text(encoding="utf8")
    for side in F.SIDES:
        d = BENCH / side / "rulebooks"
        d.mkdir(parents=True, exist_ok=True)
        for fam, (_, fn, _) in F.RULEBOOK_FAMILIES.items():
            (d / f"{fam}.md").write_text(fn(md, side), encoding="utf8", newline="\n")
    bad = []
    jobs = [(side, biz, meta["seed"], meta["drop_kind"].get(f"{side}|{biz}"))
            for side, biz in sorted({(c["side"], c["biz"]) for c in cells})]
    for (side, biz), res in sorted(_gen_all(jobs, args.jobs).items()):
        for fam, (ok, fp) in res.items():
            key = f"{side}|{fam}|{biz}"
            want_c = meta.get("content_fp", {}).get(key)
            if want_c is not None or "content_fp" in meta:   # sealed: compare the FILES, independent of ingest code
                got_c = _content_fp(BENCH / side / fam / "data" / biz) if ok else None
                if got_c != want_c:
                    bad.append(key)
            elif fp != meta["variant_fp"].get(key):          # unsealed (old): Store fingerprint
                bad.append(key)
    print(f"materialized; {len(bad)} variants differ from the frozen build" + (f": {', '.join(bad)}" if bad else ""))
    if bad:
        sys.exit(1)


def regold(args):
    """Re-create gold_recomputed.json (gitignored, written only by `build`) on a machine that only ran
    `materialize`: the same rule passes as build (original rulebook vs the side's rule_param rulebook). Cells and
    sealed content are untouched; a recomputed cell without a gold here is left out of PA by `score`."""
    if RECOMP.exists() and not args.force:
        sys.exit(f"{RECOMP.relative_to(ROOT)} exists (use --force to rebuild it)")
    meta, cells = load_cells()
    cfg = load_splits()
    golds = {g["id"]: g for g in load_jsonl(TRAIN / "answers.jsonl")}
    by_shop = defaultdict(list)
    for q in load_jsonl(TRAIN / "questions.jsonl"):
        by_shop[q["business_id"]].append(q)
    want = {f"{c['side']}|{c['qid']}" for c in cells if c["label"] == "recomputed"}
    t0, recomputed = time.time(), {}
    for side in F.SIDES:
        rb = BENCH / side / "rulebooks" / "rule_param.md"
        if not rb.exists():
            sys.exit(f"{rb} missing: run `bench.py materialize` first")
        qs = [q for b in cfg[side] for q in by_shop[b] if f"{side}|{q['id']}" in want]
        orig, rp = rule_pass(qs, TRAIN), rule_pass(qs, TRAIN, rb)
        for q in qs:
            g, o, n = golds[q["id"]], orig[q["id"]]["row"], rp[q["id"]]["row"]
            if score(o, g) == 1 and n.get("type") not in ABSTAINISH and score(n, g) != 1:
                recomputed[f"{side}|{q['id']}"] = {"type": n["type"], "answer": n.get("answer")}
    _atomic_write(RECOMP, json.dumps(recomputed, ensure_ascii=False, indent=0))
    print(f"regold: {len(recomputed)}/{len(want)} recomputed cells have a gold ({time.time() - t0:.0f}s); "
          f"missing ones are not scored -> {RECOMP.relative_to(ROOT)}")


def build(args):
    if CELLS.exists() and not args.rebuild:
        sys.exit(f"{CELLS.name} exists (frozen benchmark): use `materialize`, or `build --rebuild` with a new D-entry")
    cfg = load_splits()
    seed, T = cfg["seed"], cfg["targeted_per_shop"]
    qs_all = load_jsonl(TRAIN / "questions.jsonl")
    golds = {g["id"]: g for g in load_jsonl(TRAIN / "answers.jsonl")}
    by_shop = defaultdict(list)
    for q in qs_all:
        by_shop[q["business_id"]].append(q)
    BENCH.mkdir(parents=True, exist_ok=True)
    md = (TRAIN / "rulebook.md").read_text(encoding="utf8")

    t0 = time.time()
    side_qs = {s: [q for b in cfg[s] for q in by_shop[b]] for s in F.SIDES}
    rb_paths = {}
    for side in F.SIDES:
        d = BENCH / side / "rulebooks"
        d.mkdir(parents=True, exist_ok=True)
        for fam, (_, fn, _) in F.RULEBOOK_FAMILIES.items():
            p = d / f"{fam}.md"
            p.write_text(fn(md, side), encoding="utf8", newline="\n")
            rb_paths[(side, fam)] = p
    rp_cache = BENCH / "rule_pass.json"   # rule engine output on train questions (gitignored)
    rp_key = f"{_src_hash()}|{','.join(cfg['A'] + cfg['B'])}"
    cached = None
    if rp_cache.exists():   # reused when made by the same rule engine on the same shops (resume after interruption)
        try:
            cached = json.loads(rp_cache.read_text(encoding="utf8"))
        except ValueError:
            cached = None
    if cached and (cached.get("key") == rp_key or (args.reuse_rule_pass and "key" not in cached)):
        orig, recomputed = cached["orig"], cached["recomputed"]
        print("rule passes reused from rule_pass.json", flush=True)
    else:
        orig = rule_pass(side_qs["A"] + side_qs["B"], TRAIN)
        recomputed = {}
        for side in F.SIDES:
            rp = rule_pass(side_qs[side], TRAIN, rb_paths[(side, "rule_param")])
            for q in side_qs[side]:
                g, o, n = golds[q["id"]], orig[q["id"]]["row"], rp[q["id"]]["row"]
                if score(o, g) == 1 and n.get("type") not in ABSTAINISH and score(n, g) != 1:
                    recomputed[f"{side}|{q['id']}"] = {"type": n["type"], "answer": n.get("answer")}
        _atomic_write(rp_cache, json.dumps({"key": rp_key, "orig": orig, "recomputed": recomputed},
                                           ensure_ascii=False, default=str))
    _atomic_write(RECOMP, json.dumps(recomputed, ensure_ascii=False, indent=0))
    print(f"rule passes: {len(orig)} questions, {len(recomputed)} recomputed golds ({time.time() - t0:.0f}s)", flush=True)

    cells, applicable, variant_fp, picked, drop_kinds = [], defaultdict(set), {}, {}, {}
    sel_of = {}
    for side in F.SIDES:   # phase 1: question selection (cheap, sequential)
        exc_ops = F.exception_ops(side)
        for biz in sorted(cfg[side]):
            qids = sorted(q["id"] for q in by_shop[biz])
            rng = random.Random(f"{seed}-{biz}-base")
            first = qids[:cfg["first_by_id"]]
            base = first + _strat_pick([q for q in qids if q not in first], golds, cfg["per_shop"] - len(first), rng)
            kinds, canc = _orig_meta(biz)
            # targeted pools (questions the change touches), outside the base sample
            taken = set(base)
            drop_kind, targeted = None, {}
            pools = {}
            for g in F.drop_groups_present(kinds):
                pools[g] = [q for q in qids if q not in taken and orig[q]["op"]
                            and DROP_AREA[g] in OP_AREAS.get(orig[q]["op"], ())]
            if pools:
                drop_kind = max(sorted(pools), key=lambda g: len(pools[g]))
            for fam, pool in (("drop_table", pools.get(drop_kind, [])),
                              ("rule_param", [q for q in qids if f"{side}|{q}" in recomputed]),
                              ("rule_exception", [q for q in qids if orig[q]["op"] in exc_ops])):
                pool = [q for q in pool if q not in taken]
                r2 = random.Random(f"{seed}-{biz}-{fam}")
                sel = sorted(r2.sample(pool, min(T, len(pool))))
                targeted[fam] = sel
                taken |= set(sel)
            drop_kinds[f"{side}|{biz}"] = drop_kind if targeted["drop_table"] else None
            sel_of[(side, biz)] = (base, targeted, rng)

    # phase 2: data variants of every shop, in parallel
    gen = _gen_all([(s, b, seed, drop_kinds[f"{s}|{b}"]) for s, b in sorted(sel_of)], args.jobs)

    for side in F.SIDES:   # phase 3: assignment (sequential: family counts are balanced across the side)
        fam_count = Counter()
        for biz in sorted(cfg[side]):
            base, targeted, rng = sel_of[(side, biz)]
            for fam, (ok, fp) in gen[(side, biz)].items():
                if ok:
                    applicable[(side, fam)].add(biz)
                    variant_fp[f"{side}|{fam}|{biz}"] = fp
            for fam in ("rule_param", "rule_exception"):
                applicable[(side, fam)].add(biz)

            # base questions -> the least used applicable non-targeted family of this side
            always = [f for f in F.DATA_FAMILIES if f not in F.TARGETED and biz in applicable[(side, f)]]
            assign = {}
            for q in sorted(base):
                cand = always[:]
                rng.shuffle(cand)
                f = min(cand, key=lambda f: fam_count[f])
                fam_count[f] += 1
                assign[q] = f
            for fam, sel in targeted.items():
                if biz in applicable[(side, fam)]:
                    for q in sel:
                        assign[q] = fam
            picked[biz] = assign
            for q, fam in sorted(assign.items()):
                cells.append(_cell(q, biz, side, "original", None))
                cells.append(_cell(q, biz, side, fam, rb_paths.get((side, fam))))
            print(f"  {side} {biz}: {len(assign)} questions "
                  f"({', '.join(f'{k}={v}' for k, v in sorted(Counter(assign.values()).items()))})", flush=True)

    meta = {"gen_hash": gen_hash(), "seed": seed, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_cells": len(cells), "variant_fp": variant_fp, "drop_kind": drop_kinds,
            "applicable": {f"{s}|{f}": sorted(b) for (s, f), b in sorted(applicable.items())}}
    _atomic_write(CELLS, json.dumps({"meta": meta, "cells": cells}, ensure_ascii=False, indent=1))
    _summary(cells)
    print(f"\n{CELLS.relative_to(ROOT)}: {len(cells)} cells, gen_hash {meta['gen_hash']} ({time.time() - t0:.0f}s)")


def _cell(qid, biz, side, fam, rulebook):
    if fam == "original":
        grp, label, root = "G0", "same", TRAIN
    elif fam in F.DATA_FAMILIES:
        grp, _, label = F.DATA_FAMILIES[fam]
        root = BENCH / side / fam
    else:
        grp, _, label = F.RULEBOOK_FAMILIES[fam]
        root = TRAIN
    rel = lambda p: p.relative_to(ROOT).as_posix() if p else None   # noqa: E731
    return {"key": f"{qid}|{fam}", "qid": qid, "biz": biz, "side": side, "family": fam, "group": grp,
            "label": label, "root": rel(root), "rulebook": rel(rulebook)}


def _summary(cells):
    c = Counter((x["side"], x["family"]) for x in cells)
    shops = defaultdict(set)
    for x in cells:
        shops[(x["side"], x["family"])].add(x["biz"])
    print(f"\n{'side':5}{'family':16}{'shops':>6}{'cells':>7}")
    for (s, f), n in sorted(c.items()):
        print(f"{s:5}{f:16}{len(shops[(s, f)]):6}{n:7}")


def load_cells(check: bool = True) -> tuple[dict, list]:
    """Frozen identity. Before sealing: the generator code hash. After `seal`: the cells + the CONTENT hash of every
    variant (BENCHMARK.md §3), so ingest/rule code may improve without unfreezing; generated files are verified by
    content in materialize / ingest / run instead."""
    d = json.loads(CELLS.read_text(encoding="utf8"))
    if check and d["meta"]["gen_hash"] != gen_hash():
        if "content_fp" not in d["meta"]:
            sys.exit(f"bench_cells.json was built by generator {d['meta']['gen_hash']}, current code is {gen_hash()}: "
                     "the benchmark is frozen (BENCHMARK.md §3); rebuild only with a new D-entry")
        print(f"[bench] generator code changed since the freeze ({d['meta']['gen_hash']} -> {gen_hash()}); "
              "variants are checked by sealed content", file=sys.stderr)
    return d["meta"], d["cells"]


def _fp_job(job: tuple) -> tuple:
    """(key, shop dir) -> (key, normalised Store fingerprint | 'ExceptionName')."""
    from eval.synth.run import fingerprint   # noqa: PLC0415
    key, path = job
    try:
        return key, json.loads(json.dumps(fingerprint(Store.build(path, use_cache=False)), sort_keys=True, default=str))
    except Exception as e:  # noqa: BLE001
        return key, type(e).__name__


def _parallel(fn, jobs: list, n_jobs: int) -> list:
    from concurrent.futures import ProcessPoolExecutor   # noqa: PLC0415
    if n_jobs <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(fn, jobs))


def verify_content(meta: dict, keys: list[str]) -> list[str]:
    """Sealed variants whose files on disk differ from the freeze (empty = OK)."""
    cf = meta.get("content_fp", {})
    return [k for k in keys if k in cf and _content_fp(BENCH / k.split("|")[0] / k.split("|")[1] / "data" / k.split("|")[2]) != cf[k]]


def _repair_name(name: str) -> str:
    """Undo what an upload/unzip may do to a Vietnamese file name: UTF-8 bytes read as cp437/latin-1 (zip without
    the UTF-8 flag), or NFD instead of NFC. Returns the name unchanged when nothing applies."""
    import unicodedata   # noqa: PLC0415
    for enc in ("cp437", "latin-1"):
        try:
            fixed = name.encode(enc).decode("utf8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if fixed != name:
            name = fixed
            break
    return unicodedata.normalize("NFC", name)


def verify_cmd(args):
    """Check the variants on disk against the sealed content hashes. Lists file names of a differing variant (names
    only, never content: side B stays totals-only). --fix-names first repairs mangled file names (_repair_name)."""
    meta, _ = load_cells()
    keys = sorted(k for k in meta.get("content_fp", {}) if not args.side or k.startswith(f"{args.side}|"))
    renamed = 0
    if args.fix_names:
        for k in keys:
            d = BENCH / k.split("|")[0] / k.split("|")[1] / "data" / k.split("|")[2]
            for f in sorted(d.iterdir()) if d.exists() else []:
                new = _repair_name(f.name)
                if new != f.name and not (d / new).exists():
                    f.rename(d / new)
                    renamed += 1
    bad = verify_content(meta, keys)
    print(f"{len(keys)} sealed variants, {len(bad)} differ, {renamed} file names repaired")
    for k in bad if not args.files else keys:
        d = BENCH / k.split("|")[0] / k.split("|")[1] / "data" / k.split("|")[2]
        print(f"  {k}{'' if d.exists() else ' <missing dir>'}")
        for name, size, fp in file_fps(d):
            print(f"      {name!r:32} {size:>10} {fp}")
    sys.exit(1 if bad else 0)


# ====================================================================== ingest fidelity
def ingest(args):
    """Store fingerprint of every variant vs the original (fraction of fingerprint fields equal). drop_table is
    excluded (its diff is the point). Also checks the variants on disk are the frozen ones (variant_fp)."""
    from eval.synth.run import compare   # noqa: PLC0415
    meta, _ = load_cells()
    keys = sorted(meta["variant_fp"])
    stale = verify_content(meta, keys) if args.verify else []
    todo = [k for k in keys if args.recompute or not _fp_path(k).exists()]
    fresh = dict(_parallel(_fp_job, [(k, BENCH / k.split("|")[0] / k.split("|")[1] / "data" / k.split("|")[2]) for k in todo]
                           + [(f"orig|{b}", TRAIN / "data" / b) for b in sorted({k.split("|")[2] for k in keys})],
                           args.jobs))
    base, rows = {}, defaultdict(list)
    for key in keys:
        side, fam, biz = key.split("|")
        base[biz] = fresh[f"orig|{biz}"]
        fp = fresh[key] if key in fresh else json.loads(_fp_path(key).read_text(encoding="utf8"))
        if isinstance(fp, str):
            rows[(side, fam)].append((biz, 0.0, [f"ERROR {fp}"]))
            continue
        if fam == "drop_table":
            continue
        diff = compare(base[biz], fp)
        rows[(side, fam)].append((biz, 1 - len(diff) / len(base[biz]), diff))
    print(f"{'side':5}{'family':16}{'shops':>6}{'exact':>7}{'fields ok':>11}   top mismatching fields")
    rep = {}
    for (side, fam), rs in sorted(rows.items()):
        bad = Counter(d.split(" ")[0] for _, _, ds in rs for d in ds)
        rep[f"{side}|{fam}"] = {"shops": len(rs), "exact": sum(1 for r in rs if not r[2]),
                                "fields_ok": float(np.mean([r[1] for r in rs])),
                                "diff": {b: ds for b, _, ds in rs if ds}}
        print(f"{side:5}{fam:16}{len(rs):6}{sum(1 for r in rs if not r[2]):7}{np.mean([r[1] for r in rs]):11.1%}   "
              f"{', '.join(k for k, _ in bad.most_common(4))}")
    out = ROOT / "eval" / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_bench_ingest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"gen_hash": meta["gen_hash"], "src_hash": _src_hash(),
                                                 "recompute": args.recompute, "families": rep},
                                                ensure_ascii=False, indent=1, default=str), encoding="utf8")
    print(f"\nreport: {out.relative_to(ROOT)}")
    if stale:
        print(f"WARNING {len(stale)} variant data dirs differ from the sealed content: {', '.join(stale[:6])}")


# ====================================================================== run
def run(args):
    from datamate.pipeline import Context, answer_many   # noqa: PLC0415
    from datamate.run import make_agent, make_llm   # noqa: PLC0415
    meta, cells = load_cells(check=not args.force)
    sel = [c for c in cells if (not args.side or c["side"] == args.side)
           and (not args.families or c["family"] in args.families) and (not args.shops or c["biz"] in args.shops)]
    qs = {q["id"]: q for q in load_jsonl(TRAIN / "questions.jsonl")}
    groups = defaultdict(list)
    for c in sel:
        groups[(c["root"], c["rulebook"])].append(c)
    # every variant checked BEFORE the model loads: a bad one used to stop the run hours in (Kaggle 13/11, 284 min)
    for root, _ in sorted(groups):
        if not (ROOT / root).exists():
            sys.exit(f"{root} missing: run `bench.py materialize` first")
    skipped = []
    if not args.force:
        bad = verify_content(meta, sorted({f"{c['side']}|{c['family']}|{c['biz']}" for c in sel
                                           if c["family"] in F.DATA_FAMILIES}))
        if bad and not args.skip_bad:
            sys.exit(f"{len(bad)} variants differ from the sealed benchmark: {', '.join(bad[:8])} "
                     f"(run materialize, `bench.py verify` for the files, or --skip-bad)")
        if bad:   # leave those cells out (scored over the cells run; meta.json lists them) rather than lose the run
            skipped = [c["key"] for c in sel if f"{c['side']}|{c['family']}|{c['biz']}" in bad]
            sel = [c for c in sel if f"{c['side']}|{c['family']}|{c['biz']}" not in bad]
            print(f"--skip-bad: {len(bad)} variants differ from the seal ({', '.join(bad)}): {len(skipped)} cells "
                  f"left out", flush=True)
            groups = defaultdict(list)
            for c in sel:
                groups[(c["root"], c["rulebook"])].append(c)
    run_dir = Path(args.out) if args.out else ROOT / "eval" / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_bench_{args.arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "?"
    llm = make_llm(args.llm, args.llm_model, args.llm_replay)
    info = {"arm": args.arm, "gen_hash": meta["gen_hash"], "commit": commit, "side": args.side, "families": args.families,
            "agent": args.agent, "llm_model": getattr(getattr(llm, "cfg", None), "model_id", None), "n_cells": len(sel),
            "skipped_cells": skipped,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    with open(run_dir / "trace.jsonl", "w", encoding="utf8", newline="\n") as f:
        for (root, rb), cs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
            rootp = ROOT / root
            kind = args.agent if args.agent in ("codegen", "sql") else None
            agent = make_agent(llm, rootp, ROOT / rb if rb else None,
                               enabled=kind is not None if args.agent else None, kind=kind)
            ctx = Context(root=rootp, llm=None if agent else llm, rulebook=ROOT / rb if rb else None, agent=agent)
            by_q = {c["qid"]: c for c in cs}
            for q, row, tr in answer_many([qs[c["qid"]] for c in cs], ctx):
                el = time.perf_counter() - tr.pop("_t0")
                c = by_q[q["id"]]
                inp = tr.get("inputs") or {}
                rec = {**c, "row": row, "elapsed_s": el, "source": inp.get("source"), "op": inp.get("op"),
                       "vote": inp.get("codegen_vote"), "rule_answer": inp.get("rule_answer"),
                       "output": tr.get("output") if isinstance(tr.get("output"), list) else None,
                       "code": tr.get("code") if isinstance(tr.get("code"), list) else None,
                       "thinking": tr.get("thinking") if isinstance(tr.get("thinking"), list) else None,
                       "timing": tr.get("timing"), "error": tr.get("error")}
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                f.flush()
            if agent is not None:
                agent.close()
            print(f"  {root} {Path(rb).name if rb else ''}: {len(cs)} cells ({time.time() - t0:.0f}s)", flush=True)
    info["seconds"] = round(time.time() - t0, 1)
    (run_dir / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"run dir: {run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}")


# ====================================================================== score
def gold_of(c: dict, golds: dict, recomp: dict) -> dict | None:
    """Gold the cell is scored against; None for 'changed' cells (scored as stale / not stale only)."""
    if c["label"] == "same":
        return golds[c["qid"]]
    if c["label"] == "unanswerable":
        return {"id": c["qid"], "type": "unanswerable", "answer": None}
    if c["label"] == "recomputed":
        g = recomp.get(f"{c['side']}|{c['qid']}")
        return {"id": c["qid"], **g} if g else None
    return None


def concrete(row: dict | None) -> bool:
    return row is not None and row.get("type") not in ("abstain", "unanswerable", None)


def same_answer(a: dict | None, b: dict | None) -> bool:
    """Semantic equality of two final answers under the scorer's typed match (rounding, case, aliases-free):
    abstain == abstain, unanswerable == unanswerable, concrete answers equal if one scores correct against the other."""
    ka = "abstain" if not a or a.get("type") in ABSTAINISH else a["type"]
    kb = "abstain" if not b or b.get("type") in ABSTAINISH else b["type"]
    if ka in ("abstain", "unanswerable") or kb in ("abstain", "unanswerable"):
        return ka == kb
    return is_correct(a, {"type": b["type"], "answer": b.get("answer")})


_NUM = (int, float)


def runtime_class(r: dict) -> str | None:
    """Sample failure class (BENCHMARK.md §4) or None when the sample ran."""
    st = r.get("status")
    if st == "ok":
        return None
    if st in ("timeout", "no_code", "empty", "no_sql"):   # empty / no_sql: text-to-SQL agent (D26)
        return st
    e = str(r.get("result") or "")
    if re.search(r"Binder Error|Catalog Error", e):          # DuckDB: column / table not in the views
        return "column"
    if re.search(r"Parser Error", e):
        return "syntax"
    if "does not fit TYPE" in e:
        return "type_shape"
    if re.search(r"Conversion Error|Invalid Input Error", e):
        return "conversion"
    last = e.strip().splitlines()[-1] if e.strip() else ""
    if "object has no attribute" in last and re.search(r"'Helpers'|module 'H'|'H'", last):
        return "helper_hallucinated"
    if last.startswith("KeyError") or "not in index" in last or "Column not found" in last:
        return "column"
    if last.startswith("TypeError") and re.search(r"Helpers\.|positional argument|unexpected keyword", last):
        return "helper_signature"
    if last.startswith("NameError"):
        return "name"
    m = re.match(r"(\w+)", last)
    return f"other:{m.group(1) if m else '?'}"


def semantic_class(pred: dict, gold: dict) -> str:
    """Wrong-but-ran sample: rough nature of the error."""
    if pred.get("type") == "unanswerable":
        return "false_unanswerable"
    if pred.get("type") != gold.get("type"):
        return "type"
    pa, ga = pred.get("answer"), gold.get("answer")
    if isinstance(pa, _NUM) and isinstance(ga, _NUM) and not isinstance(pa, bool):
        if ga == 0 or pa == 0:
            return "far"
        r = pa / ga
        if abs(r - 1) <= 0.05:
            return "near_miss"          # filter / rulebook detail (status, returns, discount, rounding)
        for x in (r, 1 / r):
            if x >= 1.9 and abs(x - round(x)) <= 0.02 * x:
                return "unit_factor"    # base unit vs pack, thousands, double counting
        return "far"
    if isinstance(pa, list) and isinstance(ga, list):
        return "order" if sorted(map(str, pa)) == sorted(map(str, ga)) else "set"
    return "value"


def _bootstrap(cells: list[dict], stat, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI by resampling SHOPS (cluster bootstrap)."""
    by = defaultdict(list)
    for c in cells:
        by[c["biz"]].append(c)
    shops = sorted(by)
    if len(shops) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(shops), len(shops))
        v = stat([c for i in pick for c in by[shops[i]]])
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def _pa(cs):
    return sum(c["s"] for c in cs) / len(cs) if cs else float("nan")


def _prec(cs):
    a = [c for c in cs if c["s"] != 0]
    return sum(c["s"] == 1 for c in a) / len(a) if a else float("nan")


def _cov(cs):
    return sum(c["s"] != 0 for c in cs) / len(cs) if cs else float("nan")


def _arec(cs):
    a = [c for c in cs if c["gold"]["type"] != "unanswerable"]
    return sum(c["s"] == 1 for c in a) / len(a) if a else float("nan")


def metrics(cs: list[dict], B: int) -> dict:
    """Main metrics over scorable cells (each has 's' and 'gold')."""
    n = len(cs)
    m = {"shops": len({c["biz"] for c in cs}), "n": n, "R": sum(c["s"] == 1 for c in cs),
         "W": sum(c["s"] == -1 for c in cs), "A": sum(c["s"] == 0 for c in cs)}
    m.update(PA=_pa(cs), precision=_prec(cs), coverage=_cov(cs), answerable_recall=_arec(cs))
    for k, fn in (("PA", _pa), ("precision", _prec), ("coverage", _cov), ("answerable_recall", _arec)):
        m[f"{k}_ci"] = _bootstrap(cs, fn, B) if B else (float("nan"),) * 2
    un = [c for c in cs if c["gold"]["type"] == "unanswerable"]
    m["n_unanswerable"] = len(un)
    m["forced_answer_rate"] = sum(concrete(c["row"]) for c in un) / len(un) if un else float("nan")
    m["consensus_wrong"] = sum(1 for c in cs if c["s"] == -1 and "codegen" in str(c.get("source"))) / n if n else float("nan")
    return m


def _fmt(v, ci=None, pct=False):
    if v != v:
        return "   -  "
    s = f"{100 * v:5.1f}" if pct else f"{v:6.3f}"
    if ci and ci[0] == ci[0]:
        s += f" [{100 * ci[0]:.0f},{100 * ci[1]:.0f}]" if pct else f" [{ci[0]:+.2f},{ci[1]:+.2f}]"
    return s


def _load_runs(dirs: list[str]) -> tuple[dict, list[dict]]:
    recs, infos = {}, []
    for d in dirs:
        d = Path(d)
        infos.append(json.loads((d / "meta.json").read_text(encoding="utf8")) if (d / "meta.json").exists() else {})
        for r in load_jsonl(d / "trace.jsonl"):
            recs[r["key"]] = r
    return recs, infos


def score_cmd(args):
    meta, cells = load_cells(check=False)
    golds = {g["id"]: g for g in load_jsonl(TRAIN / "answers.jsonl")}
    recomp = json.loads(RECOMP.read_text(encoding="utf8")) if RECOMP.exists() else {}
    recs, infos = _load_runs(args.runs)
    for i in infos:
        if i.get("gen_hash") and i["gen_hash"] != meta["gen_hash"]:
            print(f"WARNING run {i.get('arm')} used generator {i['gen_hash']}, cells are {meta['gen_hash']}")
    sc = scored(cells, recs, golds, recomp)
    report = {"meta": {"gen_hash": meta["gen_hash"], "runs": infos}, "sides": {}}
    for side in F.SIDES:
        rs = [c for c in sc if c["side"] == side]
        if not rs:
            continue
        print(f"\n=== side {side} ({'Synth-Tune' if side == 'A' else 'Synth-Eval: aggregates only'}) ===")
        report["sides"][side] = side_report(rs, args.bootstrap)
        if args.baseline:
            brecs, _ = _load_runs(args.baseline)
            report["sides"][side]["paired"] = paired(rs, scored(cells, brecs, golds, recomp), side, args.bootstrap)
        if args.errors and side == "A":
            _errors(rs, args.errors)
    out = Path(args.runs[0]) / ("report_vs_baseline.json" if args.baseline else "report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf8")
    print(f"\nreport: {out}")


def scored(cells, recs, golds, recomp) -> list[dict]:
    out = []
    for c in cells:
        r = recs.get(c["key"])
        if r is None:
            continue
        g = gold_of(c, golds, recomp)
        old = golds[c["qid"]]
        x = {**c, "row": r.get("row"), "source": r.get("source"), "output": r.get("output") or [],
             "timing": r.get("timing") or {}, "elapsed_s": r.get("elapsed_s"), "gold": g, "old_gold": old}
        if g is not None:
            x["s"] = score(r.get("row"), g)
        else:   # changed: stale = still the old (pre-change) answer
            x["stale"] = concrete(r.get("row")) and is_correct(r["row"], old)
        out.append(x)
    return out


def side_report(rs: list[dict], B: int) -> dict:
    rep = {}
    groups = defaultdict(list)
    for c in rs:
        if "s" in c:
            groups[("family", c["family"])].append(c)
            groups[("group", c["group"])].append(c)
            groups[("total", "variants" if c["family"] != "original" else "original")].append(c)
    print(f"{'':24}{'shops':>5}{'n':>5}{'R':>4}{'W':>4}{'A':>4}  {'PA [95% CI, shop bootstrap]':28}{'prec %':>15}"
          f"{'cov %':>15}{'ans.recall %':>16}{'forced %':>9}{'flip %':>8}")
    flips = flip_stats(rs)
    for key in sorted(groups, key=lambda k: ({"total": 0, "group": 1, "family": 2}[k[0]], k[1])):
        cs = groups[key]
        m = metrics(cs, B)
        fl = flips.get(key)
        m["flip_rate"] = fl["rate"] if fl else float("nan")
        m["flip_n"] = fl["n"] if fl else 0
        rep[f"{key[0]}:{key[1]}"] = m
        print(f"{key[0][:1]}:{key[1]:22}{m['shops']:5}{m['n']:5}{m['R']:4}{m['W']:4}{m['A']:4}  "
              f"{_fmt(m['PA'], m['PA_ci']):28}{_fmt(m['precision'], m['precision_ci'], True):>15}"
              f"{_fmt(m['coverage'], m['coverage_ci'], True):>15}{_fmt(m['answerable_recall'], m['answerable_recall_ci'], True):>16}"
              f"{_fmt(m['forced_answer_rate'], pct=True):>9}{_fmt(m['flip_rate'], pct=True):>8}")
    ch = [c for c in rs if c["label"] == "changed"]
    if ch:
        st = sum(c["stale"] for c in ch)
        ab = sum(not concrete(c["row"]) for c in ch)
        rep["changed"] = {"n": len(ch), "stale": st, "abstain_or_unans": ab, "other_answer": len(ch) - st - ab}
        print(f"rule_exception (label changed, not in PA): n={len(ch)} stale(old answer)={st} ({100 * st / len(ch):.0f}%) "
              f"abstain/unans={ab} other={len(ch) - st - ab}")
    rep["flip_transitions"] = {k: v for k, v in flips.get(("total", "variants"), {}).get("trans", {}).items()}
    rep["samples"] = sample_stats(rs)
    rep["runtime"] = runtime_stats(rs)
    return rep


def flip_stats(rs: list[dict]) -> dict:
    """Semantic flip: variant ('same' label) vs original cell of the same question, compared by same_answer()."""
    orig = {c["qid"]: c for c in rs if c["family"] == "original"}
    out = defaultdict(lambda: {"n": 0, "flip": 0, "trans": Counter()})
    for c in rs:
        if c["family"] == "original" or c["label"] != "same" or c["qid"] not in orig or "s" not in c:
            continue
        o = orig[c["qid"]]
        f = not same_answer(o["row"], c["row"])
        for key in (("family", c["family"]), ("group", c["group"]), ("total", "variants")):
            out[key]["n"] += 1
            out[key]["flip"] += f
            out[key]["trans"][f"{o['s']:+d}>{c['s']:+d}{'*' if f and o['s'] == c['s'] else ''}"] += 1
    for v in out.values():
        v["rate"] = v["flip"] / v["n"] if v["n"] else float("nan")
        v["trans"] = dict(sorted(v["trans"].items()))
    return dict(out)


def sample_stats(rs: list[dict]) -> dict:
    """Per-sample runtime / semantic error classes (code-gen runs only)."""
    by = defaultdict(lambda: {"samples": 0, "runtime": Counter(), "ok": 0, "ok_wrong": 0, "semantic": Counter()})
    for c in rs:
        g = c.get("gold")
        for key in ("all", c["group"]):
            for r in c["output"]:
                d = by[key]
                d["samples"] += 1
                rc = runtime_class(r)
                if rc:
                    d["runtime"][rc] += 1
                    continue
                d["ok"] += 1
                res = r.get("result")
                if g is not None and isinstance(res, dict) and res.get("type") not in ABSTAINISH and score(res, g) == -1:
                    d["ok_wrong"] += 1
                    d["semantic"][semantic_class(res, g)] += 1
    out = {}
    if by["all"]["samples"]:
        print(f"\nsamples{'':10}{'n':>6}{'runtime err %':>15}{'semantic err % (of ran)':>25}   classes")
    for k in sorted(by, key=lambda k: (k != "all", k)):
        d = by[k]
        n, rt = d["samples"], sum(d["runtime"].values())
        out[k] = {"samples": n, "runtime_error_rate": rt / n if n else None, "runtime": dict(d["runtime"]),
                  "semantic_error_rate": d["ok_wrong"] / d["ok"] if d["ok"] else None, "semantic": dict(d["semantic"])}
        if n:
            print(f"{k:17}{n:6}{100 * rt / n:15.1f}{(100 * d['ok_wrong'] / d['ok'] if d['ok'] else 0):25.1f}   "
                  f"rt: {', '.join(f'{a}={b}' for a, b in d['runtime'].most_common(5))} | "
                  f"sem: {', '.join(f'{a}={b}' for a, b in d['semantic'].most_common(5))}")
    return out


STAGES = ("rule_s", "prompt_s", "gen_s", "validate_s", "exec_s", "vote_s")


def runtime_stats(rs: list[dict]) -> dict:
    """End-to-end seconds per question: contract (from batch entry, Docker Guide) and amortised (own stages + a
    1/batch_n share of the batch generation). Median and p95 per stage."""
    rows = []
    for c in rs:
        t = dict(c["timing"])
        t["gen_s"] = t.get("gen_batch_s", 0.0) / max(t.get("batch_n", 1), 1)
        t["amortised_s"] = sum(t.get(s, 0.0) or 0.0 for s in STAGES)
        t["contract_s"] = c.get("elapsed_s") or 0.0
        rows.append(t)
    out = {}
    if not rows:
        return out
    print(f"\nruntime (s/question){'':4}{'median':>9}{'p95':>9}")
    for k in ("contract_s", "amortised_s", *STAGES):
        v = [r.get(k) or 0.0 for r in rows]
        out[k] = {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
        print(f"  {k:22}{out[k]['median']:9.3f}{out[k]['p95']:9.3f}")
    return out


def paired(rs: list[dict], base: list[dict], side: str, B: int) -> dict:
    """ΔPA arm − baseline on the cells BOTH runs answered (same questions, data, rulebook), shop bootstrap."""
    bm = {c["key"]: c for c in base if c["side"] == side and "s" in c}
    out = {}
    print(f"\npaired vs baseline{'':6}{'shops':>6}{'n':>5}{'ΔPA':>8}{'95% CI':>18}{'Δprec':>8}{'Δcov':>8}")
    groups = defaultdict(list)
    for c in rs:
        if "s" in c and c["key"] in bm:
            pair = {**c, "d": c["s"] - bm[c["key"]]["s"], "b": bm[c["key"]]}
            groups["total:variants" if c["family"] != "original" else "total:original"].append(pair)
            groups[f"group:{c['group']}"].append(pair)
    for k in sorted(groups):
        cs = groups[k]
        d = sum(c["d"] for c in cs) / len(cs)
        ci = _bootstrap(cs, lambda x: sum(c["d"] for c in x) / len(x) if x else float("nan"), B)
        dp = _prec(cs) - _prec([c["b"] for c in cs])
        dc = _cov(cs) - _cov([c["b"] for c in cs])
        out[k] = {"shops": len({c["biz"] for c in cs}), "n": len(cs), "dPA": d, "ci": ci, "dprec": dp, "dcov": dc}
        print(f"  {k:22}{out[k]['shops']:6}{len(cs):5}{d:+8.3f}  [{ci[0]:+.3f},{ci[1]:+.3f}]{dp:+8.3f}{dc:+8.3f}")
    return out


def _errors(rs: list[dict], n: int):
    qs = {q["id"]: q for q in load_jsonl(TRAIN / "questions.jsonl")}
    wrong = [c for c in rs if c.get("s") == -1][:n]
    print(f"\n--- {len(wrong)} wrong cells (side A only) ---")
    for c in wrong:
        print(f"{c['biz']} {c['family']:14} {qs[c['qid']]['question'][:110]}\n   pred={json.dumps(c['row'], ensure_ascii=False)[:120]}"
              f"\n   gold={json.dumps({k: c['gold'].get(k) for k in ('type', 'answer')}, ensure_ascii=False)[:120]}")


# ====================================================================== cli
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rebuild", action="store_true", help="re-select cells (unfreezes the benchmark: new D-entry)")
    b.add_argument("--reuse-rule-pass", action="store_true", help="reuse eval/synth/out/bench/rule_pass.json")
    b.add_argument("--jobs", type=int, default=3)
    se = sub.add_parser("seal")
    se.add_argument("--force", action="store_true")
    m = sub.add_parser("materialize")
    m.add_argument("--jobs", type=int, default=3)
    ig = sub.add_parser("ingest")
    ig.add_argument("--recompute", action="store_true", help="rebuild variant Stores instead of the saved fingerprints")
    ig.add_argument("--jobs", type=int, default=2)
    ig.add_argument("--verify", action="store_true", help="also check variant files against the sealed content (slow)")
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, help="name of the configuration under test (e.g. base, s1_schema)")
    r.add_argument("--side", choices=F.SIDES)
    r.add_argument("--families", nargs="*")
    r.add_argument("--shops", nargs="*")
    r.add_argument("--out")
    r.add_argument("--force", action="store_true", help="run although the generator changed (never for decisions)")
    r.add_argument("--skip-bad", action="store_true", help="leave out cells whose variant differs from the seal")
    r.add_argument("--llm", choices=["off", "vllm", "replay"])
    r.add_argument("--llm-model")
    r.add_argument("--llm-replay")
    r.add_argument("--agent", choices=["off", "codegen", "sql"])
    v = sub.add_parser("verify")
    v.add_argument("--side", choices=F.SIDES)
    v.add_argument("--fix-names", action="store_true", help="repair mangled file names (unzip codepage, NFD) first")
    v.add_argument("--files", action="store_true", help="list file name/size/hash of every variant, not only bad ones")
    rg = sub.add_parser("regold")
    rg.add_argument("--force", action="store_true", help="overwrite an existing gold_recomputed.json")
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+", help="run dirs (shards of one arm are merged)")
    s.add_argument("--baseline", nargs="*", help="run dirs of the baseline arm: paired ΔPA on common cells")
    s.add_argument("--bootstrap", type=int, default=2000)
    s.add_argument("--errors", type=int, default=0, help="list N wrong cells of side A (never side B)")
    args = ap.parse_args()
    {"build": build, "seal": seal, "materialize": materialize, "ingest": ingest, "run": run, "score": score_cmd,
     "regold": regold, "verify": verify_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
