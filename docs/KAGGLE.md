# KAGGLE — chạy agent code-gen trên GPU Kaggle (D20)

Vòng lặp dev cho agent sinh code, không cần máy 5090 của thành viên. Notebook: `scripts/kaggle/datamate_dev.ipynb`.
Kết quả chỉ dùng để chọn model, sửa prompt/helper và calibrate. Ngưỡng cuối vẫn phải đo lại bằng chính image trên
5090 (D20 hệ quả).

## Chuẩn bị (một lần)

1. **Dataset private chứa train**: kaggle.com → Datasets → New Dataset → upload thư mục `data/train/train`
   (186 MB: `questions.jsonl`, `answers.jsonl`, `rulebook.md`, `calendar.csv`, `data/`). Đặt tên ví dụ
   `datamate-train`, **Visibility: Private**. Chỉ train, **không upload public_test**.
2. **Notebook**: Code → New Notebook → File → Import Notebook → chọn `scripts/kaggle/datamate_dev.ipynb`.
   - Settings: **Accelerator: GPU T4 x2**, **Internet: On**, notebook **Private**.
   - Add Input → dataset `datamate-train` ở bước 1.
3. Code được `git clone` từ GitHub nhánh `feat/llm-parser`: nhánh phải được push trước khi chạy.

## Chạy

1. Sửa cell cấu hình nếu cần (`MODEL_REPO`, `PER_SHOP`, `SAMPLES`). Lần đầu giữ `PER_SHOP = 6`, `RUN_SMOKE = True`.
2. Run All. Lần đầu mất thêm khoảng 5–10 phút tạo venv Python 3.12 và cài vLLM và khoảng 10 phút tải model 14B AWQ.
3. Tải `/kaggle/working/datamate_dev_<model>.zip` (Output → Download) và gửi lead qua Drive nhóm.
   **Không commit** file này: chứa câu hỏi train và output model (D15).

## Lead đọc kết quả trên CPU

```bash
unzip datamate_dev_<model>.zip -d eval/runs/kaggle_<model>
PYTHONPATH=src .venv/Scripts/python eval/calibrate_agent.py eval/runs/kaggle_<model>/<run>_dev/trace.jsonl
```

Trace có đủ code, đầu ra từng mẫu và đáp án rule, nên đổi m/policy không cần GPU. Sửa sandbox, helper hoặc bảng T
thì chạy lại code đã ghi trên CPU, rồi calibrate trên trace mới:

```bash
PYTHONPATH=src .venv/Scripts/python eval/reexec_agent.py eval/runs/kaggle_<model>/<run>_dev/trace.jsonl
PYTHONPATH=src .venv/Scripts/python eval/calibrate_agent.py eval/runs/kaggle_<model>/<run>_dev/trace_reexec.jsonl
```

Đổi prompt (kể cả schema hiển thị trong prompt) thì phải chạy lại GPU (khoá replay gồm prompt). Đọc dòng
**code-gen một mình** (policy `codegen`): trên public, rule bị D18 chặn gần hết nên đó mới là điểm thật của agent.

`THINKING = True` bật thinking mode của Qwen3 (đặt `MAX_TOKENS = 6144`). Khoá replay và tên zip có hậu tố `-think`.

## Chạy trên public (sinh submission)

Notebook: `scripts/kaggle/datamate_public.ipynb`. Public chỉ dùng để sinh submission (CLAUDE.md luật 1): không mở
`log.jsonl`/`replay.jsonl` để đọc câu hỏi hay output, chỉ xử lý bằng script đếm/gộp.

1. **Dataset private** `datamate-public`: upload thư mục `data/public_test/public_test` (105 MB), **Visibility: Private**.
2. Import notebook, GPU T4 x2, Internet On, Add Input → `datamate-public`. Nhánh `feat/llm-parser` phải đã push
   (cell clone kiểm `coalesce_suffixes`).
3. Ở mỗi lần chạy, đặt `SHARD = 0..NSHARDS-1` (mặc định `NSHARDS = 3`, khoảng 330 câu mỗi shard). Shop được chia
   cân theo số câu, tất định, nên mọi tài khoản tính ra cùng một cách chia. Ở dev, T4 chạy khoảng 83 giây/câu, nên
   mỗi shard mất khoảng 6–8 giờ, dưới giới hạn 12 giờ. Mỗi phút cell chạy chính in số câu đã xong kèm thời gian ước
   tính còn lại; ước tính chỉ nhảy sau mỗi shop vì câu của một shop được gửi model cùng một lượt.
4. Chạy bằng **Save Version → Save & Run All (Commit)**. Các shard chạy song song được trên tài khoản của từng thành
   viên.
5. Bị ngắt giữa chừng: Add Input → output của lần chạy đó (có `replay.jsonl`) rồi chạy lại cùng `SHARD`. Shop nào
   đã có trong replay sẽ không gọi GPU nữa. Notebook zip kết quả dở dang mỗi `ZIP_EVERY_MIN` phút.

Lead gộp các shard trên CPU. `--reexec` chạy lại code đã ghi bằng sandbox/helper hiện tại, nên các bản sửa CPU sau
lần chạy GPU vẫn áp dụng được:

```bash
for i in 0 1 2; do unzip -o datamate_public_*_shard${i}of3.zip -d eval/runs/public_pub1; done
PYTHONPATH=src .venv/Scripts/python eval/resubmit_agent.py eval/runs/public_pub1/shard*/log.jsonl \
    --data data/public_test/public_test --reexec --out eval/runs/public_pub1/submission.jsonl
```

Công cụ chỉ in số đếm (nguồn, type). Câu thiếu log được ghi abstain. **Luồng chính thức (D21)**: policy/m lấy từ
`AgentConfig` đã commit (chọn trên train), không nhận `--policy/--m`, và ghi `submission.jsonl.meta.json` (commit,
config, số đếm). Thử cấu hình khác chỉ qua `--experiment` trên train; công cụ từ chối `--experiment` trên public.
Mỗi lần chạy GPU trên public chỉ nộp **một** bản chính thức.

## Chọn model

Chạy cùng `PER_SHOP` cho từng ứng viên ≤ 15B (ngày public < 26/10/2026, license cho phép), so trong `calibrate_dev.txt`:
PA tốt nhất, precision, tỉ lệ `error`/`timeout`/`no_code`, thời gian. Ứng viên: `Qwen/Qwen3-14B-AWQ`,
`Qwen/Qwen3-8B-AWQ`, `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`.

## Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `No matching distribution found for vllm==0.10.1.1` (Requires-Python <3.13) | Kernel Kaggle là Python 3.13. Notebook cài vLLM vào venv Python 3.12 riêng (`uv`, `/tmp/venv`), mọi lệnh pipeline chạy bằng `{PY}`. Không `pip install vllm` vào kernel |
| vLLM import được nhưng không thấy CUDA / `driver too old` | Gửi output cell cài đặt cho lead (in python, torch, CUDA, capability). Không tự đổi bản vLLM: replay phải cùng bản với image (D17) |
| `Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers 5.x không chạy với vLLM 0.10.1.1. Notebook pin `TRANSFORMERS_VERSION = "4.55.4"`; xem dòng `transformers` trong output cell cài đặt |
| `PassManager::run failed` ở `prefix_prefill.py` (`compute-capability=75`) | Kernel Triton prefix-prefill của vLLM không biên dịch cho T4. Cell env tự đặt `DATAMATE_LLM_PREFIX_CACHE=0` (tắt prefix caching và chunked prefill) khi GPU < sm80. Chậm hơn vì mỗi câu prefill lại prompt |
| Lỗi engine V1 / attention trên T4 | Cell env tự đặt `VLLM_USE_V1=0` và `VLLM_ATTENTION_BACKEND=XFORMERS` khi GPU < sm80 |
| OOM khi load hoặc KV cache không đủ | Giảm `MAX_LEN` xuống 16384, hoặc dùng bản 8B |
| Quá chậm | Giảm `PER_SHOP` hoặc `SAMPLES`. Mỗi lần gọi `run_eval` load lại model, nên tắt `RUN_SMOKE` sau lần đầu |
| Hết quota GPU tuần (~30 giờ/tài khoản) | Thành viên khác chạy bằng tài khoản của mình, cùng notebook |
