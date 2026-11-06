# DataMate — Zalo AI Challenge 2026 (image rule-only)

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của shop nhỏ. Bản này không dùng model: parser luật + executor
deterministic, không đủ căn cứ thì abstain.

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
