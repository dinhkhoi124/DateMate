# WORKFLOW — Phối hợp GPU & nhóm

## 1. Phân vai máy

| Việc | Ở đâu | Vì sao |
|---|---|---|
| Ingest, executor, rule parser, eval, unit test | **Local (CPU)** | Không cần GPU, iterate nhanh nhất |
| LLM parse câu hỏi, ghi replay | **Máy GPU của thành viên** theo [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md) (D15) | Có sẵn, không tốn tiền, chạy một lệnh. Kaggle (2× T4) là dự phòng |
| Thử nhanh một lần | Colab (1× T4) | Dự phòng, dễ bị ngắt kết nối |
| Test Docker trên phần cứng thật | **Thuê 1× RTX 5090** (vast.ai/RunPod, khoảng 2–3 giờ quanh 13/11) | Kaggle/Colab không chạy được Docker và không có 5090 |

**Nguyên tắc chính: GPU chỉ sinh replay, mọi thứ khác chạy local.** Máy GPU chạy `scripts/gpu/run_all.sh`, ghi đầu ra model cho mọi câu dev/holdout/public vào replay jsonl. Bundle lên **Drive nhóm** (không bao giờ lên GitHub), local chạy `--llm replay` để eval và calibrate. Chỉ chạy lại GPU khi đổi prompt, few-shot hoặc model.

### Giới hạn của T4 cần nhớ
- T4 không hỗ trợ BF16/FP8, nên chạy FP16 hoặc AWQ/GPTQ 4-bit.
- Model 14B dùng bản AWQ 4-bit (khoảng 9GB, vừa 1 T4). Model 8B FP16 thì chia ra 2 T4 (`tensor_parallel_size=2`).
- Tốc độ T4 chậm hơn 5090 khoảng 4–6 lần. Thời gian đo trên Kaggle chỉ để ước lượng, số thật phải đo trên 5090.

## 2. Quy trình Kaggle (dự phòng, chưa làm)

> Đường chính đã chuyển sang máy GPU thành viên (D15, `GPU_RUNBOOK.md`). Chỉ dùng Kaggle nếu máy đó không chạy được. Nếu dùng: dataset phải để **Private**, notebook gọi `scripts/gpu/run_all.sh` thay cho các script dưới đây.

**Thiết lập một lần:**
1. Kaggle → Datasets → New → upload `data/train.zip` dưới dạng **Private**, đặt tên `datamate-train`. **Không upload public test.**
2. Settings → Phone verification (bắt buộc để dùng GPU và Internet).
3. Nếu repo GitHub để private: Add-ons → Secrets → thêm `GITHUB_TOKEN` (fine-grained, chỉ quyền read repo).

**Mỗi lần chạy:**
1. Mở notebook `notebooks/kaggle_llm.ipynb`: Accelerator **GPU T4 x2**, Internet **On**, Add data `datamate-train`.
2. Ô đầu tiên chạy `git clone` repo và `pip install`. Ô tiếp theo chạy:
   ```
   python scripts/remote/llm_parse.py --model <hf_id> --split dev --out /kaggle/working/out
   ```
3. Bấm **Save Version → Save & Run All** để chạy nền, không cần mở trình duyệt.
4. Khi xong, vào tab Output → tải `out.zip` → giải nén vào `eval/remote/<tên-run>/` ở local, rồi ghi tên run vào PR.
5. Chạy `eval/run_eval.py --intents eval/remote/<tên-run>/intents.jsonl`, phân tích lỗi rồi sửa prompt hoặc parser. Lặp lại.

## 3. Phân công nhóm

> **Từ 12/11 phân vai theo `docs/HANDOVER.md` mục 2**: Khôi là PM (quyết định cấu hình, image, nộp bài; duyệt PR),
> Người A làm kỹ thuật, Người B vận hành máy 5090. Bảng dưới là phân vai cũ, chỉ để tham khảo.

| Người | Vai trò | Việc cụ thể |
|---|---|---|
| **Khôi** (lead) | AI/kỹ thuật | Duyệt thiết kế, chạy Kaggle, review code, quyết định và ghi vào `DECISIONS.md` |
| **TV2** | QA dữ liệu & luật | Lấy các câu bị sai trong `/eval dev`, **tính tay trên Excel** để xác nhận đáp án đúng và luật áp dụng, ghi vào `docs/qa/<id>.md` **chỉ gồm id câu, luật áp dụng và cách tính, không chép câu hỏi hay đáp án** (repo public, D15). Đọc kỹ rulebook để tìm các tình huống dễ hiểu sai |
| **TV3** | Ngôn ngữ & BTC | (1) Viết biến thể câu hỏi tiếng Việt cho mỗi intent: teencode, gõ Telex sai, không dấu, cách nói Bắc/Nam, tự viết, không chép câu thi, lưu trên Drive nhóm (repo chặn mọi `.jsonl`, D15). (2) Theo dõi forum và portal, gửi câu hỏi cho BTC (`SPEC.md` §7), cập nhật timeline. (3) Đặt lịch thuê GPU 5090 |
| Cả nhóm | Review | Review PR về docs/checklist, test lệnh trong README trên máy mình |

TV2 và TV3 không cần biết AI. Chỉ cần dùng Excel, GitHub web và đọc tiếng Việt cẩn thận.

## 4. Git, review và CI

**Luồng làm việc (GitHub flow):**
1. `git checkout main && git pull && git checkout -b <loại>/<việc>`. Loại: `feat/` tính năng, `fix/` sửa lỗi, `qa/` dữ liệu kiểm tra, `docs/`, `chore/` công cụ.
2. Commit nhỏ, message dạng mệnh lệnh, nói rõ thay đổi (ví dụ `Ingest: read status from Ghi chú column`).
3. Push và mở PR vào `main`, điền đủ template (kết quả đo + checklist). Mọi PR cần **CI xanh** (GitHub bắt buộc) và **ít nhất 1 người khác review** (quy ước nhóm, GitHub không chặn). PM duyệt mọi PR vào `main`. Không tự merge PR khi chưa ai xem.
4. Merge bằng **Squash and merge**, rồi xoá branch.

**Tự động hoá:**

| Lớp | Công cụ | Kiểm gì |
|---|---|---|
| Trên máy, trước mỗi commit | `pre-commit` (`.pre-commit-config.yaml`) | ruff, file > 1MB, conflict marker, private key, **chặn commit dữ liệu thi** |
| Trên GitHub, mỗi PR | GitHub Actions (`.github/workflows/ci.yml`) | ruff, pytest (test cần data tự skip), không có data/token/file lớn trong repo, **build được image Docker** và import được trong container `--network none` |
| Bảo vệ `main` | Ruleset `protect-main` trên GitHub | Bắt buộc qua PR, 4 check CI phải xanh, chỉ cho squash merge, cấm force-push và cấm xoá nhánh. Không bắt buộc approval: review theo quy ước nhóm |
| Sau khi merge | Tag phiên bản `v0.x` | Mỗi lần nộp public gắn tag tương ứng, ghi vào `EXPERIMENTS.md` |

**CI không chạy được eval độ chính xác** vì dữ liệu thi không được đưa lên GitHub. Người mở PR tự chạy `eval/run_eval.py --split dev` và dán số vào PR.

**Không làm CD đẩy image lên registry:** image chứa dữ liệu train, nên không được đẩy lên registry public (GHCR, Docker Hub). Image nộp bài được build trên máy lead hoặc máy thuê GPU.

**Quy định:**
- `main` luôn chạy được.
- Không commit `data/`, weights, `eval/runs/`, token, mọi file `.jsonl` (few-shot, replay, submission, log) và không chép nguyên văn câu hỏi thi vào code, test hay docs (D15; pre-commit `no-verbatim-questions` kiểm). Kết quả tóm tắt ghi vào `docs/EXPERIMENTS.md`.
- Không đọc câu hỏi public và không mở dữ liệu hay câu hỏi của 6 shop holdout. Đây là luật tự giác, và mỗi PR phải xác nhận trong checklist.
