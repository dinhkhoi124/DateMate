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
2. Run All. Lần đầu mất thêm khoảng 5–10 phút cài vLLM và khoảng 10 phút tải model 14B AWQ.
3. Tải `/kaggle/working/datamate_dev_<model>.zip` (Output → Download) và gửi lead qua Drive nhóm.
   **Không commit** file này: chứa câu hỏi train và output model (D15).

## Lead đọc kết quả trên CPU

```bash
unzip datamate_dev_<model>.zip -d eval/runs/kaggle_<model>
PYTHONPATH=src .venv/Scripts/python eval/calibrate_agent.py eval/runs/kaggle_<model>/<run>_dev/trace.jsonl
```

Trace có đủ code, đầu ra từng mẫu và đáp án rule, nên đổi m/policy không cần GPU. Đổi prompt hoặc helper thì
phải chạy lại GPU (khoá replay gồm prompt).

## Chọn model

Chạy cùng `PER_SHOP` cho từng ứng viên ≤ 15B (ngày public < 26/10/2026, license cho phép), so trong `calibrate_dev.txt`:
PA tốt nhất, precision, tỉ lệ `error`/`timeout`/`no_code`, thời gian. Ứng viên: `Qwen/Qwen3-14B-AWQ`,
`Qwen/Qwen3-8B-AWQ`, `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`.

## Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| vLLM 0.10.1.1 không import được / không thấy CUDA | Notebook tự cài `VLLM_FALLBACK` (0.8.5.post1). Vẫn lỗi: gửi output cell cài đặt cho lead |
| Lỗi attention backend hoặc FlashAttention trên T4 | Thêm `os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"` ở cell env |
| OOM khi load hoặc KV cache không đủ | Giảm `MAX_LEN` xuống 16384, hoặc dùng bản 8B |
| Quá chậm | Giảm `PER_SHOP` hoặc `SAMPLES`. Mỗi lần gọi `run_eval` load lại model, nên tắt `RUN_SMOKE` sau lần đầu |
| Hết quota GPU tuần (~30 giờ/tài khoản) | Thành viên khác chạy bằng tài khoản của mình, cùng notebook |
