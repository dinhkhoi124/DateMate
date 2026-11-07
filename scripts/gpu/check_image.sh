#!/usr/bin/env bash
# Run a built image exactly like the organisers and check the result (PLAN Phase 4). GPU machine, repo root.
#
#   bash scripts/gpu/check_image.sh <image:tag> <split_dir> [<out_dir>]
#   e.g. bash scripts/gpu/check_image.sh zaic2026_datamate_x:final data/train/train
#
# Organisers' command (SPEC): --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash, /data read-only.
# On a 1-GPU machine set GPUS=0 TP=1 (the image defaults to TP=2 for the 2x 5090 grader; TP changes numerics a bit).
# Prints: wall time, submit_check, answer types, and (train split) PA. Outputs stay local: team Drive only (D15).
set -uo pipefail
IMG="${1:?image}"; SPLIT="$(cd "${2:?split dir}" && pwd)"; OUT="${3:-eval/runs/image_check/$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
GPUS="${GPUS:-0,1}"
EXTRA=()
[ -n "${TP:-}" ] && EXTRA+=(-e "DATAMATE_LLM_TP=${TP}")

t0=$(date +%s)
docker run --rm --gpus "\"device=${GPUS}\"" --network none --shm-size=16g --entrypoint bash "${EXTRA[@]}" \
  -v "${SPLIT}:/data:ro" -v "${OUT}:/result" "$IMG" /code/predict.sh > "$OUT/container.txt" 2>&1
rc=$?
dt=$(( $(date +%s) - t0 ))
echo "exit ${rc}, ${dt}s wall (includes model load)"
grep -E '^\[datamate\]|^llm_calls=|^judge_calls=|LLM disabled' "$OUT/container.txt" | tail -5
PY="${PY:-python3}"
"$PY" eval/submit_check.py "$OUT/submission.jsonl" | tail -3
if [ -f "${SPLIT}/answers.jsonl" ]; then
  "$PY" eval/scorer.py "$OUT/submission.jsonl" "${SPLIT}/answers.jsonl" | tail -3
fi
echo "outputs: $OUT (submission, logs/log.jsonl, container.txt) -> team Drive only"
