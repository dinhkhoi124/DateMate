# WORKFLOW — Phối hợp GPU & nhóm

## 1. Phân vai máy

| Việc | Ở đâu | Vì sao |
|---|---|---|
| Ingest, executor, rule parser, eval, unit test | **Local (CPU)** | Không cần GPU, iterate nhanh nhất |
| LLM parse câu hỏi, sinh intent | **Kaggle** (2× T4 16GB, khoảng 30 giờ GPU/tuần) | Miễn phí, có chế độ chạy nền "Save & Run All" |
| Thử nhanh một lần | Colab (1× T4) | Dự phòng, dễ bị ngắt kết nối |
| Test Docker trên phần cứng thật | **Thuê 1× RTX 5090** (vast.ai/RunPod, khoảng 2–3 giờ quanh 13/11) | Kaggle/Colab không chạy được Docker và không có 5090 |

**Nguyên tắc chính: GPU chỉ sinh intent cache, mọi thứ khác chạy local.** Trên Kaggle, LLM parse toàn bộ câu train rồi xuất `intents_<model>_<split>.jsonl`. Local đọc file cache này để chạy executor và eval mà không cần GPU. Chỉ phải chạy lại Kaggle khi đổi prompt hoặc model.

### Giới hạn của T4 cần nhớ
- T4 không hỗ trợ BF16/FP8, nên chạy FP16 hoặc AWQ/GPTQ 4-bit.
- Model 14B dùng bản AWQ 4-bit (khoảng 9GB, vừa 1 T4). Model 8B FP16 thì chia ra 2 T4 (`tensor_parallel_size=2`).
- Tốc độ T4 chậm hơn 5090 khoảng 4–6 lần. Thời gian đo trên Kaggle chỉ để ước lượng, số thật phải đo trên 5090.

## 2. Quy trình Kaggle (bạn chạy, tôi soạn script)

**Thiết lập một lần:**
1. Kaggle → Datasets → New → upload `data/train.zip` dưới dạng **Private**, đặt tên `datamate-train`. **Không upload public test.**
2. Settings → Phone verification (bắt buộc để dùng GPU và Internet).
3. Nếu repo GitHub để private: Add-ons → Secrets → thêm `GITHUB_TOKEN` (fine-grained, chỉ quyền read repo).

**Mỗi lần chạy:**
1. Mở notebook `notebooks/kaggle_llm.ipynb` (tôi soạn sẵn): Accelerator **GPU T4 x2**, Internet **On**, Add data `datamate-train`.
2. Ô đầu tiên chạy `git clone` repo và `pip install`. Ô tiếp theo chạy:
   ```
   python scripts/remote/llm_parse.py --model <hf_id> --split dev --out /kaggle/working/out
   ```
3. Bấm **Save Version → Save & Run All** để chạy nền, không cần mở trình duyệt.
4. Khi xong, vào tab Output → tải `out.zip` → giải nén vào `eval/remote/<tên-run>/` ở local, rồi báo tôi tên run.
5. Tôi chạy `eval/run_eval.py --intents eval/remote/<tên-run>/intents.jsonl`, phân tích lỗi rồi sửa prompt hoặc parser. Lặp lại.

## 3. Phân công nhóm

| Người | Vai trò | Việc cụ thể |
|---|---|---|
| **Khôi** (lead) | AI/kỹ thuật | Duyệt thiết kế, chạy Kaggle, review code, quyết định và ghi vào `DECISIONS.md` |
| **TV2** | QA dữ liệu & luật | Lấy các câu bị sai trong `/eval dev`, **tính tay trên Excel** để xác nhận đáp án đúng và luật áp dụng, ghi vào `docs/qa/<id>.md`. Đọc kỹ rulebook để tìm các tình huống dễ hiểu sai |
| **TV3** | Ngôn ngữ & BTC | (1) Viết biến thể câu hỏi tiếng Việt cho mỗi intent: teencode, gõ Telex sai, không dấu, cách nói Bắc/Nam, vào `tests/data/paraphrases.jsonl`. (2) Theo dõi forum và portal, gửi câu hỏi cho BTC (`SPEC.md` §7), cập nhật timeline. (3) Đặt lịch thuê GPU 5090 |
| Cả nhóm | Review | Review PR về docs/checklist, test lệnh trong README trên máy mình |

TV2 và TV3 không cần biết AI. Chỉ cần dùng Excel, GitHub web và đọc tiếng Việt cẩn thận.

## 4. Git
- `main` luôn chạy được. Làm việc trên branch `feat/<việc>` rồi mở PR. Commit nhỏ, message rõ ràng.
- Không commit `data/`, weights, `eval/runs/`. Kết quả tóm tắt ghi vào `docs/EXPERIMENTS.md`.
- Repo phải public trước hạn (quy định 8.2). Trước khi public, kiểm tra lại không có dữ liệu thi và token trong lịch sử commit.
