# DataMate — Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của shop nhỏ: model sinh code pandas chạy trong sandbox, kiểm
chéo với parser luật + executor deterministic, không đủ căn cứ thì abstain. Có hai image dùng chung bố cục `/code` và `predict.sh`:

- **rule-only** (`docker/Dockerfile`): không dùng model.
- **LLM** (`docker/Dockerfile.llm`), base `vllm/vllm-openai:v0.10.1.1`. Model: **Qwen/Qwen3-14B-AWQ** (14.8B tham số,
  Apache-2.0, công bố năm 2025; lượng tử 4-bit AWQ), weights trong `/code/models/Qwen3-14B-AWQ`, chạy offline,
  tensor parallel trên 2 GPU, không thinking.
  - Model đọc rulebook của split (`/data/rulebook.md`, lúc chạy), README và schema của shop, rồi viết hàm
    `solve(T, H)` bằng pandas. Hàm chạy trong process sandbox có timeout. Mọi con số do code tính, không do model nói ra.
  - Mỗi câu lấy 5 mẫu; chỉ trả lời khi ≥ 2 mẫu ra cùng đáp án chuẩn hoá. Nếu parser luật cũng trả lời mà khác
    đáp án code thì abstain.
  - Không load được model thì chạy bản rule-only, có ghi lý do vào log.
  - `thinking` trong log là đầu ra thô của model (list theo mẫu), `code` là code từng mẫu, `output` là kết quả chạy.

## Chạy

```bash
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -v /path/to/split:/data:ro -v /path/to/output:/result \
  <image>:final /code/predict.sh
```

- Input: `/data` (read-only): `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<business_id>/`.
- Output: `/result/submission.jsonl`, ghi ngay sau mỗi câu.
- Log: `/result/logs/log.jsonl`, mỗi câu một dòng: `id`, `answer`, `elapsed_ms`, `thinking` (null: không có model),
  `code`, `inputs` (đáp án parser luật, nguồn của đáp án cuối, số phiếu), `output`.
- File tạm: `/tmp/datamate_cache`.
- Không có tham số chỉnh: luật nghiệp vụ đọc từ `/data/rulebook.md` lúc chạy, phần còn lại cố định trong code.

## Bố cục

- `/code/predict.sh`: entrypoint.
- `/code/src/datamate/`: code inference.
- `/code/training_code/`: code xây từ vựng/luật (`analysis/`, `eval/`, `tests/`), tài liệu và nhật ký thí nghiệm
  (`docs/`), dữ liệu train (`data/train/train/`), `data_manifest.json` (SHA-256 từng file).
