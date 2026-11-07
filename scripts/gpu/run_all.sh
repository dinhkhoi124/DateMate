#!/usr/bin/env bash
# GPU run for the LLM parser (docs/GPU_RUNBOOK.md, DECISIONS D15). Linux or WSL2, from the repo root, venv active.
#
#   bash scripts/gpu/run_all.sh models/Qwen3-8B models/Qwen3-14B        # phase 1: n=1 for each model
#   SAMPLES=5 bash scripts/gpu/run_all.sh models/Qwen3-14B               # phase 2: only when the lead asks
#
# For each model: smoke test, then dev, holdout and public with DATAMATE_LLM_CALL_ALL=1 (model called on every
# question) recording replay jsonl. Everything goes to eval/runs/gpu/<run>/ (gitignored). At the end:
#   - summary.md       : numbers only (versions, timing, VRAM, PA tables). This file alone may go to GitHub.
#   - bundle_<run>.tgz : replay + submissions + logs. Contains model outputs on competition questions:
#                        team Drive only, NEVER GitHub.
set -uo pipefail

[ $# -ge 1 ] || { echo "usage: [SAMPLES=n] bash scripts/gpu/run_all.sh <model_dir> [<model_dir> ...]"; exit 2; }
[ -f data/train/train/questions.jsonl ] || { echo "missing data/train/train (see runbook step 2)"; exit 2; }

SAMPLES="${SAMPLES:-1}"
RUN="$(date +%Y%m%d_%H%M)_$(hostname -s)_n${SAMPLES}"
OUT="eval/runs/gpu/${RUN}"
mkdir -p "$OUT/replay" "$OUT/logs" "$OUT/public"
export PYTHONPATH=src PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DATAMATE_LLM=vllm DATAMATE_LLM_CALL_ALL=1 DATAMATE_LLM_SAMPLES="$SAMPLES"
PY="${PY:-python}"
SUM="$OUT/summary.md"

# few-shot must be byte-identical to the lead's, otherwise replay keys never match on the CPU side
[ -f src/datamate/parse/fewshot.jsonl ] || "$PY" analysis/build_fewshot.py > "$OUT/logs/build_fewshot.txt" 2>&1
FEWSHA=$(sha256sum src/datamate/parse/fewshot.jsonl | cut -d' ' -f1)

{
  echo "# GPU run ${RUN}"
  echo
  echo "- commit: \`$(git rev-parse --short HEAD)\` ($(git rev-parse --abbrev-ref HEAD))$(git diff --quiet || echo ', **dirty tree**')"
  echo "- samples per question: ${SAMPLES}"
  echo "- fewshot sha256: \`${FEWSHA}\`"
  echo "- GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | paste -sd ';')"
  echo "- extra env: $(env | grep -E '^DATAMATE_LLM_(DTYPE|TP|GPU_UTIL|MAX_LEN)=' | paste -sd ' ' || true)"
  echo "- packages: $("$PY" -m pip freeze 2>/dev/null | grep -iE '^(vllm|torch|xgrammar|transformers|flashinfer[a-z-]*)==' | paste -sd ' ')"
  echo
} > "$SUM"

step() {  # step <name> <logfile> <cmd...> : run, time, append a line to the summary
  local name="$1" log="$2"; shift 2
  local t0=$(date +%s)
  "$@" > "$log" 2>&1
  local rc=$? dt=$(( $(date +%s) - t0 ))
  echo "  - ${name}: exit ${rc}, ${dt}s" >> "$SUM"
  echo "[$(date +%H:%M:%S)] ${name}: exit ${rc}, ${dt}s"
  return $rc
}

for M in "$@"; do
  M="${M%/}"; ID="$(basename "$M")"
  [ -f "$M/config.json" ] || { echo "skip ${M}: no config.json"; echo "- ${ID}: **missing weights**" >> "$SUM"; continue; }
  echo "## ${ID}" >> "$SUM"
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 5 > "$OUT/logs/vram_${ID}.csv" 2>/dev/null &
  SMI=$!

  step "smoke" "$OUT/logs/smoke_${ID}.txt" "$PY" eval/gpu_smoke.py --model "$M" --out "$OUT/smoke_${ID}.json"
  tail -n 3 "$OUT/logs/smoke_${ID}.txt" | sed 's/^/    /' >> "$SUM"

  for SPLIT in dev holdout; do
    step "${SPLIT}" "$OUT/logs/${SPLIT}_${ID}.txt" "$PY" eval/run_eval.py --split "$SPLIT" --by source \
      --llm vllm --llm-model "$M" --llm-replay "$OUT/replay/${ID}_${SPLIT}_n${SAMPLES}.jsonl"
    { echo '    ```'; grep -E '^(split=|by source| *(rule|rule\+llm|llm|none|-) )' "$OUT/logs/${SPLIT}_${ID}.txt"; echo '    ```'; } >> "$SUM"
  done

  if [ -d data/public_test ]; then
    step "public" "$OUT/logs/public_${ID}.txt" "$PY" -m datamate.run --data data/public_test \
      --out "$OUT/public/${ID}/submission.jsonl" --llm vllm --llm-model "$M" \
      --llm-replay "$OUT/replay/${ID}_public_n${SAMPLES}.jsonl"
    "$PY" - "$OUT/public/${ID}/submission.jsonl" >> "$SUM" <<'EOF'
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf8") if l.strip()]
c = collections.Counter(r.get("type") for r in rows)
print(f"    public answers: {len(rows)} rows, types {dict(sorted(c.items(), key=str))}")
EOF
  fi
  kill "$SMI" 2>/dev/null
  PEAK=$(cut -d, -f2 "$OUT/logs/vram_${ID}.csv" | grep -oE '[0-9]+' | sort -n | tail -1)
  echo "  - peak VRAM (max over GPUs and 5s samples): ${PEAK:-?} MiB" >> "$SUM"
  echo >> "$SUM"
done

# smoke outputs are made-up questions: safe in the summary; replay/public/logs are not
tar -czf "eval/runs/gpu/bundle_${RUN}.tgz" -C eval/runs/gpu "${RUN}"
echo
echo "Done. Summary (may go to GitHub): ${SUM}"
echo "Bundle (team Drive ONLY, never GitHub): eval/runs/gpu/bundle_${RUN}.tgz"
