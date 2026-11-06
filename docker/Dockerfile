# Rule-only fallback image (PLAN: "Dockerfile bản rule-only trước 10/11"). CPU only, no model weights.
# The LLM image will switch the base to vllm/vllm-openai:v0.10.1.1 (Guide recommendation); /code layout stays.
#
# Build (from repo root):  docker build --platform linux/amd64 -f docker/Dockerfile -t <tag> .
# Organisers' command:     docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
#                              -v <split>:/data:ro -v <out>:/result <tag> /code/predict.sh
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONHASHSEED=0 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/code/src DATAMATE_CACHE=/tmp/datamate_cache

WORKDIR /code
COPY docker/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Inference code
COPY src /code/src
COPY docker/predict.sh /code/predict.sh
COPY docker/README.md /code/README.md

# /code/training_code/: train code, train data, logs, data_manifest.json (SHA-256)
COPY pyproject.toml CLAUDE.md /code/training_code/
COPY src /code/training_code/src
COPY analysis /code/training_code/analysis
COPY eval/scorer.py eval/submit_check.py eval/run_eval.py eval/splits.json /code/training_code/eval/
COPY eval/synth /code/training_code/eval/synth
COPY tests /code/training_code/tests
COPY docs /code/training_code/docs
COPY docker /code/training_code/docker
COPY data/train/train /code/training_code/data/train/train

RUN sed -i 's/\r$//' /code/predict.sh && chmod +x /code/predict.sh \
 && python /code/training_code/docker/make_manifest.py /code/training_code /code/training_code/data_manifest.json

# Guide: organisers run with --entrypoint bash, so the image must not rely on its own entrypoint
ENTRYPOINT []
CMD ["bash", "/code/predict.sh"]
