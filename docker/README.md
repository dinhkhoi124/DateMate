# DataMate — Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của shop nhỏ: parser luật + executor deterministic, không đủ căn
cứ thì abstain. Có hai image dùng chung bố cục `/code` và `predict.sh`:

- **rule-only** (`docker/Dockerfile`): không dùng model.
- **LLM** (`docker/Dockerfile.llm`), base `vllm/vllm-openai:v0.10.1.1`. Model: **Qwen3** (bản ghi trong
  `DATAMATE_LLM_MODEL`, weights trong `/code/models/`), ≤ 15B tham số, chạy offline, tensor parallel trên 2 GPU.
  - Model chỉ chuyển câu hỏi thành intent JSON (ép schema); mọi con số do executor tính.
  - Câu bị chặn vì rulebook của split khác train có thể được model xác nhận là không bị ảnh hưởng (rule judge, đang tắt).
  - Không load được model thì chạy bản rule-only, có ghi lý do vào log.
  - `thinking` trong log là đầu ra thô của model.

## Chạy

```bash
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -v /path/to/split:/data:ro -v /path/to/output:/result \
  <image>:final /code/predict.sh
```

- Input: `/data` (read-only): `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<business_id>/`.
- Output: `/result/submission.jsonl`, ghi ngay sau mỗi câu.
- Log: `/result/logs/log.jsonl`, mỗi câu một dòng: `id`, `answer`, `elapsed_ms`, `thinking` (null: không có model),
  `code` (hàm executor), `inputs` (intent đã parse), `output`.
- File tạm: `/tmp/datamate_cache`.
- Không có tham số chỉnh: luật nghiệp vụ đọc từ `/data/rulebook.md` lúc chạy, phần còn lại cố định trong code.

## Bố cục

- `/code/predict.sh`: entrypoint.
- `/code/src/datamate/`: code inference.
- `/code/training_code/`: code xây từ vựng/luật (`analysis/`, `eval/`, `tests/`), tài liệu và nhật ký thí nghiệm
  (`docs/`), dữ liệu train (`data/train/train/`), `data_manifest.json` (SHA-256 từng file).
