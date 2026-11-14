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
- Image LLM tự chuyển sang đáp án luật cho các shop còn lại sau 5 giờ 30 phút (`DATAMATE_DEADLINE_S=19800`), để không
  vượt giới hạn 6 giờ.

### Chạy trên một GPU 8 GB (máy của đội, RTX 4060 Laptop)

Máy chấm có 2× RTX 5090. Đội tự chạy vòng private trên một RTX 4060 8 GB. Image và model giữ nguyên, chỉ ghi đè
bốn biến môi trường:

| Biến | Giá trị | Lý do |
|---|---|---|
| `DATAMATE_LLM_TP` | `1` | Chỉ có một card |
| `DATAMATE_LLM_CPU_OFFLOAD_GB` | `8` | Để khoảng 8 GB trọng số trong RAM, VRAM còn chỗ cho KV cache |
| `DATAMATE_LLM_GPU_UTIL` | `0.85` | Màn hình laptop chiếm một phần VRAM |
| `DATAMATE_DEADLINE_S` | để trống | Lượt của đội không bị giới hạn 6 giờ |

```bash
docker run --rm --gpus all --network none --shm-size=16g --entrypoint bash \
  -e DATAMATE_LLM_TP=1 -e DATAMATE_LLM_CPU_OFFLOAD_GB=8 -e DATAMATE_LLM_GPU_UTIL=0.85 -e DATAMATE_DEADLINE_S= \
  -v /path/to/split:/data:ro -v /path/to/output:/result \
  <image>:final /code/predict.sh
```

Model, prompt, seed và số mẫu giống hệt lượt của BTC; chỉ cách đặt trọng số trên phần cứng là khác. Nếu vLLM báo
thiếu bộ nhớ cho KV cache, tăng `DATAMATE_LLM_CPU_OFFLOAD_GB` (ví dụ `9`) trước khi đổi gì khác.

## Bố cục

- `/code/predict.sh`: entrypoint.
- `/code/src/datamate/`: code inference.
- `/code/training_code/`: code xây từ vựng/luật (`analysis/`, `eval/`, `tests/`), tài liệu và nhật ký thí nghiệm
  (`docs/`), dữ liệu train (`data/train/train/`), `data_manifest.json` (SHA-256 từng file).
