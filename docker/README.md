# DataMate — Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của shop nhỏ. Model viết **SQL** trên các view DuckDB dựng sẵn
bằng code tất định. Mọi con số do DuckDB tính, không do model nói ra. Không đủ căn cứ thì abstain.

Có hai image dùng chung bố cục `/code` và `predict.sh`:

- **rule-only** (`docker/Dockerfile`): không dùng model. Parser luật cộng executor tất định.
- **LLM** (`docker/Dockerfile.llm`, image này), base `vllm/vllm-openai:v0.10.1.1`.
  - Model: **Qwen/Qwen3-14B-AWQ** (14.8B tham số, Apache-2.0, công bố năm 2025, lượng tử 4-bit AWQ).
  - Trọng số trong `/code/models/Qwen3-14B-AWQ`, chạy offline trên **một** GPU (engine chọn qua chuỗi dự phòng,
    mục 6), không thinking.

## Cách hoạt động (agent text-to-SQL, `DATAMATE_AGENT=sql`)

1. **Dữ liệu → DuckDB.** Mỗi shop được đọc bằng code tất định (nhận bảng, cột, trạng thái theo README của shop) và
   nạp vào một DuckDB trong bộ nhớ. Các view chuẩn tính sẵn theo rulebook:
   - `v_orders` (một dòng một đơn) và `v_sales` (một dòng một dòng bán);
   - macro nợ `debt_at`, `debt_new`, `debt_repaid`;
   - `stock_at`, `lots_at`, `v_imports`, `v_receipts`, `v_deliveries`.
2. **Prompt.** Gồm rulebook của split (`/data/rulebook.md`, đọc lúc chạy), README và schema của shop, và gợi ý do code
   tính (khoảng ngày của kỳ hỏi, khoá sản phẩm/khách). Thêm 4 ví dụ gần nhất từ các shop dev của **train**
   (`src/datamate/agent/fewshot_sql.jsonl`). Model trả lời theo khuôn: ghi chú, kiểu đáp án, rồi **một** câu SQL.
3. **Bỏ phiếu.** Mỗi câu lấy **4 mẫu** (nhiệt độ 0.7, seed 2026). SQL lỗi được sửa một lượt. Câu chỉ được trả lời khi
   có **≥ 2 mẫu** cùng ra một đáp án chuẩn hoá. Kết quả rỗng hoặc không đạt kiểm tra hợp lý thì abstain.
4. **Rulebook khác mặc định.** Prompt trích nguyên văn dòng luật mới ngay trước câu hỏi. Mẫu nào còn đọc cột hoặc
   macro tính theo luật cũ thì không được bỏ phiếu.
5. **Kết hợp với parser luật.** Khi parser luật cũng trả lời mà khác đáp án SQL thì abstain. `unanswerable` chỉ trả
   khi kỳ hỏi nằm ngoài khoảng dữ liệu (kiểm tra tất định).
6. **Nạp model qua chuỗi cấu hình dự phòng (D29).** Engine mặc định (2 GPU, torch.compile) gặp lỗi CUDA trên 5090
   ở Test Solution 15/11. Image thử lần lượt các biến thể trong `DATAMATE_LLM_PROFILES`, mỗi biến thể trong một tiến
   trình riêng:
   - `v1-eager-tp1`: 1 GPU, không compile, không CUDA graph;
   - `v1-eager-tp1-awq-flashinfer`: thêm kernel AWQ thường và attention FlashInfer;
   - `v0-xformers-tp1`: engine V0, xformers.

   Biến thể đầu tiên nạp được và sinh được thì dùng cho cả lượt. stderr ghi `[llm] engine profile <tên>: ok/FAILED`
   và đuôi log của biến thể hỏng.
7. **Lùi về rule-only.** Không biến thể nào chạy được thì cả lượt chạy rule-only, có ghi lý do ra stderr. Sau 5 giờ 30 phút
   (`DATAMATE_DEADLINE_S=19800`), các shop còn lại chỉ dùng đáp án luật, để không vượt giới hạn 6 giờ.

## Chạy

```bash
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -v /path/to/split:/data:ro -v /path/to/output:/result \
  <image>:final /code/predict.sh
```

- **Input:** `/data` (read-only): `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<business_id>/`.
- **Output:** `/result/submission.jsonl`. Câu hỏi được gom theo shop (mỗi shop một lượt gọi model), mỗi câu ghi ngay
  khi có đáp án.
- **Log:** `/result/logs/log.jsonl`, mỗi câu một dòng gồm:
  - `id`, `answer`, `elapsed_ms`;
  - `thinking`: đầu ra thô của model, list theo mẫu;
  - `code`: câu SQL của từng mẫu;
  - `output`: trạng thái và kết quả chạy từng mẫu (`ok`, `error`, `empty`, `no_sql`, `rule_stale`);
  - `inputs`: đáp án parser luật, nguồn đáp án cuối, số phiếu, các dòng rulebook liên quan.
- **File tạm:** `/tmp/datamate_cache`.
- **Không có tham số chỉnh.** Luật nghiệp vụ đọc từ `/data/rulebook.md` lúc chạy. Số mẫu và ngưỡng phiếu là mặc định
  đã commit (`AgentConfig`: 4 mẫu, ngưỡng 0.5), chọn chỉ trên train.

### Chạy trên phần cứng khác

Máy chấm có 2× RTX 5090. Mọi biến thể trong chuỗi dự phòng chỉ dùng **một** GPU, nên image chạy được nguyên trạng
trên một GPU ≥ 24 GB. Model, prompt, seed và số mẫu không đổi. Muốn thử một biến thể cụ thể thì đặt
`-e DATAMATE_LLM_PROFILES=<tên>`. GPU 8 GB không đủ cho model này. Image có biến
`DATAMATE_LLM_CPU_OFFLOAD_GB`, nhưng đã thử trên RTX 4060 8 GB qua Docker Desktop/WSL thì không chạy được:
- engine V1 cần pin memory, mà WSL không có;
- engine V0 không còn đủ VRAM cho KV cache.

Khi đó image tự chạy rule-only.

## Bố cục

- `/code/predict.sh`: entrypoint.
- `/code/src/datamate/`: code inference (`agent/sql_agent.py`, `agent/sqldb.py`, `agent/sql_prompt.py`).
- `/code/training_code/`:
  - code xây từ vựng, luật và kho few-shot (`analysis/`, `eval/`, `tests/`);
  - tài liệu và nhật ký thí nghiệm (`docs/`, xem `DECISIONS.md` D26–D28);
  - dữ liệu train (`data/train/train/`);
  - `data_manifest.json` (SHA-256 từng file).
