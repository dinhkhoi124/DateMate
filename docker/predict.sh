#!/usr/bin/env bash
# Organisers run `bash /code/predict.sh` with no arguments, --network none, --shm-size=16g (SPEC submission contract).
# Reads /data (read-only); streams /result/submission.jsonl and one log record per question to /result/logs/.
set -uo pipefail

export DATAMATE_CACHE="${DATAMATE_CACHE:-/tmp/datamate_cache}"   # temp files only under /tmp
export PYTHONPATH=/code/src
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p /result/logs "$DATAMATE_CACHE"
cd /code
exec python -m datamate.run --data /data --out /result/submission.jsonl --log /result/logs/log.jsonl
