# HANDOVER — Bàn giao DataMate cho nhóm (11/11/2026)

Từ 12/11 Khôi (lead cũ) **không tham gia nữa**. Hai thành viên còn lại nhận toàn bộ: cải tiến, nộp public, nộp image
final (16/11) và chạy private (20–23/11). File này là điểm bắt đầu. Đọc hết **một lần** trước khi làm gì khác.

Trong các tài liệu cũ (`GPU_RUNBOOK.md`, `KAGGLE.md`, `WORKFLOW.md`…), chữ **"lead" nay có nghĩa là Người A** ở mục 2.

---

## 1. Hiện trạng (chốt 11/11)

| Hạng mục | Kết quả | Ở đâu |
|---|---|---|
| Pipeline rule-only (không model) | dev **0.985**, holdout **0.859** | `src/datamate/`, chạy CPU |
| **Image rule-only** | **Test Solution #1: Test passed** (10/11, 11 phút). MD5 `8cf7dec3b6e6383a339921657ab68d8a`, tar 308 MB. **Chưa bấm nộp final** | Drive nhóm |
| Public v0.7 (rule-only) | **0.114** | `docs/EXPERIMENTS.md` bảng "Nộp public" |
| Agent code-gen (Qwen3-14B-AWQ, D20/D22) | code-gen một mình: dev 0.653, holdout **0.444** (precision 0.776). Policy `agree` m=2 trên holdout 0.708 (72 câu) | `src/datamate/agent/` |
| Public v2 (agent, Kaggle) | **0.117** | như trên |
| **Image LLM** (`docker/Dockerfile.llm`) | **Chưa build trên GPU.** Mới kiểm build không weights trên CPU | `docs/GPU_RUNBOOK.md` mục 6b |
| Test | 203 passed; ruff sạch | `pytest tests -q` |
| Benchmark shop tổng hợp E0 (D24) | **Đóng băng** 534 cell; đã đo ingest (status_codes/status_newword hỏng 24/24 shop). Chưa chạy agent (E2 cần GPU, sau việc nộp image) | `docs/BENCHMARK.md` |

**Đánh giá thẳng:** agent chưa tổng quát tốt sang shop lạ (dev 0.653 → holdout 0.444), và image LLM chưa từng chạy
trên 5090. Image rule-only là bản **đã chắc chắn chạy được**. Đó là phương án an toàn cho final.

---

## 2. Phân vai

| Người | Phụ trách | Cần |
|---|---|---|
| **Người A — kỹ thuật** (người biết Python nhiều hơn) | Sửa code, chạy eval/Kaggle, calibrate, merge PR, ghi `DECISIONS.md`/`EXPERIMENTS.md`, quyết định image final | Laptop CPU, tài khoản GitHub (admin repo), tài khoản Kaggle |
| **Người B — vận hành & nộp bài** | Máy 5090 (build/kiểm image, chạy private), Drive, MD5, form nộp, theo dõi forum/BTC, đặt GPU dự phòng | Máy 5090 (hoặc GPU thuê), tài khoản portal cuộc thi, Drive nhóm, tài khoản Kaggle |

Hai người **review PR của nhau** (luật nhóm: mỗi PR có người thứ hai xem).

---

## 3. Khôi làm trước khi rời (checklist bàn giao quyền)

- [ ] Merge nhánh `fix/agent-runtime-errors` vào `main` (PR, CI xanh). Sau đó **mọi việc làm từ `main`**
- [ ] GitHub: mời hai người làm collaborator, đưa Người A quyền **Admin** (ruleset `protect-main`, Settings)
- [ ] Sửa `.github/CODEOWNERS`: thay `@dinhkhoi124` bằng tài khoản GitHub của Người A
- [ ] Drive nhóm: chuyển quyền **Owner** thư mục `DataMate/` cho Người B. Kiểm thư mục có: `data/train.zip`,
      `data/public_test.zip`, tar image rule-only (MD5 `8cf7dec3…`), các zip Kaggle (`datamate_dev_*`, `datamate_public_*`)
- [ ] Kaggle: share hai dataset private `datamate-train`, `datamate-public` cho tài khoản Kaggle của hai người
      (hoặc để họ tự tạo lại theo `docs/KAGGLE.md`)
- [ ] Portal cuộc thi: xác nhận ai trong đội có quyền **nộp public, Test Solution và form "Solution Submission"**.
      Nếu chỉ tài khoản Khôi nộp được thì đổi người đại diện hoặc đưa Người B quyền nộp
- [ ] Nhắn nhóm: link file này, và thông tin liên lạc nếu có câu hỏi khẩn

---

## 4. Lịch còn lại và việc cụ thể

Hạn cứng: **Test Solution đóng 15/11 23:59 (1 lượt/ngày)** · **Nộp image final 16/11 23:59 (tối đa 2 lần, BTC xét lần
cuối)** · **Private: 20/11 09:00 → 23/11 23:59, cần GPU** · Public: tối đa 5 lần/ngày, đến 16/11.

### 12/11 — Cài đặt + thử image LLM (Người B, máy 5090)
1. Cả hai: cài repo theo `README.md` mục "Cài đặt", chạy `pytest tests -q` (phải xanh) và
   `eval/run_eval.py --split dev` (phải ra **0.985**). Ra số khác thì dừng, xem lại dữ liệu/phiên bản thư viện.
2. Người B: làm `docs/GPU_RUNBOOK.md` **mục 2, 3, 6b**: tải `Qwen/Qwen3-14B-AWQ`, build `Dockerfile.llm`, chạy
   `scripts/gpu/check_image.sh <tag> data/train/train` **hai lần**. Ghi lại: exit code, tổng thời gian, giây/câu,
   PA train của mỗi lần, độ lệch hai lần.
3. Gửi các số đó cho Người A (chỉ số, không gửi nội dung câu hỏi).

### 13–14/11 — Quyết định image final (Người A)
Image LLM **chỉ được chọn** khi đạt đủ cả ba cổng (D23, luật đề):
- Hai lần chạy trên train lệch **≤ 0.03** (mục tiêu ≤ 0.01).
- Thời gian ước tính cho private **≤ 6 giờ**, tính cả load model. Docker Guide ước ~500 câu, tức trung bình
  **≤ ~40 giây/câu**. Trên Kaggle T4 agent chạy 83 giây/câu, nên phải đo thật trên 5090.
- PA train của image khớp bản local (kiểm lỗi đóng gói), và **Test Solution passed**.

Không đạt cổng nào thì **final = image rule-only** (đã passed). Không được chọn image bằng cách so điểm public
(CLAUDE.md luật 8, D21).

Nếu image LLM đạt: Người B upload tar lên Drive (30–40 GB, mất nhiều giờ), tính MD5, nộp **Test Solution #2** trước
**15/11 23:59**.

### 13–15/11 — Cải thiện điểm (Người A, song song, chỉ khi còn thời gian)
Chỉ làm trên **train** (dev để sửa, holdout chỉ xem số tổng). Mỗi thay đổi: chạy dev, ghi một dòng `EXPERIMENTS.md`.
Theo thứ tự lợi/công:
1. **Lỗi runtime còn lại của agent** (EXPERIMENTS 11/11): `find_units` bịa tên helper (2), `KeyError` cột (4),
   ưu tiên toán tử `&` (3). Sửa helper/sandbox rồi kiểm bằng `eval/reexec_agent.py` + `eval/calibrate_agent.py` trên
   trace Kaggle cũ, **không cần GPU** (`docs/KAGGLE.md` mục "Lead đọc kết quả trên CPU").
2. Đổi prompt hay model **cần chạy lại Kaggle** (T4 ×2, ~83 giây/câu, 30 giờ GPU/tuần/tài khoản). Ứng viên model:
   `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`, `Qwen/Qwen3-8B-AWQ` (`docs/KAGGLE.md` mục "Chọn model").
3. Mọi thay đổi code sau khi image đã build thì **phải build lại image và kiểm lại cổng**. Từ 15/11 chỉ sửa lỗi chặn.

### 16/11 — Nộp final (Người B, Người A xác nhận)
1. Chốt commit trên `main`, gắn tag `vfinal`.
2. Tar image trên Drive **public link**, MD5 khớp, thử tải về từ một máy khác.
3. Nộp form "Solution Submission" **trước 23:59** (để dư vài giờ). Model type: **"Based on small models (<=15B)"**.
4. Ghi vào `EXPERIMENTS.md` và tick `PLAN.md`. Giữ nguyên tar và MD5: private **bắt buộc chạy đúng image này**.

### 20–23/11 — Private (Người B)
1. **Trước 16/11 phải chắc có GPU** cho 20–23/11 (máy 5090 thành viên; dự phòng thuê 2× 5090). Image rule-only chạy
   được trên CPU nếu mất GPU hoàn toàn, nhưng chỉ khi image final chính là rule-only.
2. 20/11 sau 09:00: tải private → `docker load` đúng tar đã nộp → chạy đúng lệnh BTC (`docker/README.md` mục "Chạy")
   → `python eval/submit_check.py submission.jsonl` → upload trước 23/11 23:59.
3. Giữ nguyên `logs/log.jsonl` và `container.txt`: BTC có thể hỏi nếu điểm chạy lại lệch > 0.03.

---

## 5. Luật không được vi phạm (bị loại nếu sai)

Bản đầy đủ: `CLAUDE.md` và `docs/SPEC.md` mục 4–5. Tóm tắt:
- **Không đọc câu hỏi public**, không mở dữ liệu/câu hỏi 6 shop holdout (`eval/splits.json`).
- **Không dùng điểm public để chọn cấu hình** (prompt, model, m, policy, luật). Public chỉ để nộp và theo dõi.
- Không hardcode theo `id` câu hay `business_id`. Không human-in-the-loop lúc chạy private.
- Model ≤ 15B, weights public trước 26/10/2026. Image chạy `--network none`.
- Repo GitHub là **public**: không commit `data/`, file `.jsonl`, weights, tar, token. Pre-commit và CI chặn sẵn,
  đừng lách.

---

## 6. Lệnh hay dùng

```bash
PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split dev                       # thước đo hằng ngày
PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split dev --by intent --errors 30
PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split holdout                   # chỉ khi chốt
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python eval/submit_check.py <submission.jsonl>
```

Trên Linux/WSL thay `.venv/Scripts/python` bằng `.venv/bin/python`. Laptop RAM thấp: thêm `--only <biz_dev>` để chạy
một shop dev; cache Store ở `cache/` có thể xoá khi đầy ổ (tự build lại).

| Việc | Tài liệu |
|---|---|
| Đề, format đáp án, luật nộp | `docs/SPEC.md` |
| Kiến trúc, agent code-gen | `docs/ARCHITECTURE.md` (mục 8) |
| Chạy Kaggle, gộp shard public | `docs/KAGGLE.md` |
| Máy 5090: cài, build image, kiểm image | `docs/GPU_RUNBOOK.md` |
| Image, lệnh BTC | `docker/README.md` |
| Vì sao làm như vậy | `docs/DECISIONS.md` (D20–D23 quan trọng nhất) |
| Số liệu mọi phiên bản | `docs/EXPERIMENTS.md` |

---

## 7. Rủi ro lớn nhất và cách xử lý

| Rủi ro | Xử lý |
|---|---|
| Image LLM không ổn định hoặc quá 6 giờ | Final = image rule-only (đã passed). Không cố đến phút chót |
| Upload 30–40 GB lên Drive không kịp | Bắt đầu upload ngay khi image LLM qua cổng, muộn nhất 15/11 |
| Không có GPU 20–23/11 | Đặt máy trước 16/11, có phương án thuê dự phòng |
| Sửa code phút chót làm hỏng image | Sau 15/11 không đổi `src/` trừ lỗi chặn; mọi thay đổi qua PR + CI xanh |
| Thắc mắc về luật | Hỏi BTC qua forum (ghi câu trả lời vào `SPEC.md` mục 7), không đoán |
