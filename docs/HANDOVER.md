# HANDOVER — Bàn giao DataMate cho nhóm (11/11/2026)

Từ 12/11 Khôi chuyển sang vai trò **PM**: không trực tiếp làm kỹ thuật hằng ngày, nhưng **giữ quyền quyết định** cấu
hình cuối, build image, Test Solution và nộp public. Hai thành viên nhận phần thực thi: cải tiến, chạy eval/Kaggle,
build và kiểm image, chạy private (20–23/11). File này là điểm bắt đầu. Đọc hết **một lần** trước khi làm gì khác.

Trong các tài liệu cũ (`GPU_RUNBOOK.md`, `KAGGLE.md`, `WORKFLOW.md`…), chữ **"lead"** nay có nghĩa là **PM (Khôi)**
khi nói về quyết định (chọn cấu hình, chọn image, bấm nộp), và **Người A** khi nói về việc kỹ thuật (sửa code, chạy
eval, đọc kết quả).

---

## 0. Bắt đầu nhanh (làm theo thứ tự, mỗi người một lần, khoảng 30–45 phút)

Tên đội: **AIO_Zalo**. Tag image bắt buộc (viết thường): **`zaic2026_datamate_aio_zalo:final`**.

### 0.1. Cả hai người

1. **Lấy code** (nhánh `main`, luôn làm việc từ đây):
   ```bash
   git clone https://github.com/dinhkhoi124/DateMate.git
   cd DateMate
   git log --oneline -1          # phải thấy commit mới nhất của main
   ```
   Đã clone từ trước thì: `git checkout main && git pull`.
2. **Cài Python 3.11 và thư viện** (không cần GPU):
   ```bash
   # Windows (Git Bash)
   py -3.11 -m venv .venv
   .venv/Scripts/python -m pip install -e ".[dev]"
   .venv/Scripts/pre-commit install          # bắt buộc: chặn commit nhầm dữ liệu thi
   # Linux / WSL: thay `py -3.11` bằng `python3.11`, `.venv/Scripts/` bằng `.venv/bin/`
   ```
3. **Lấy dữ liệu train từ Drive nhóm** (`DataMate/data/train.zip`) và giải nén **vào trong repo**:
   ```bash
   mkdir -p data/train
   unzip train.zip -d data/train     # phải có data/train/train/questions.jsonl và data/train/train/data/biz_101 …
   ```
   **Chưa giải nén `public_test.zip`** trừ khi PM yêu cầu sinh file nộp (chỉ Người B). `data/` không bao giờ được commit.
4. **Kiểm cài đặt đúng.** Hai số dưới phải khớp **chính xác**:
   ```bash
   .venv/Scripts/python -m pytest tests -q
   #   → 210 passed   (chưa giải nén data: 192 passed, 18 skipped → quay lại bước 3)
   PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split dev
   #   → split=dev n=2447 correct=2410 wrong=0 abstain=37 PA=0.9849   (lần đầu 5–8 phút)
   ```
   Ra số khác thì **dừng lại**, chụp màn hình gửi nhóm, đừng sửa code để "cho khớp".
5. **Đọc luật** (10 phút): `CLAUDE.md` (8 luật) và mục 5 của file này. Ba điều dễ phạm nhất:
   - không mở `data/public_test/**/questions.jsonl`, không xem log hay cấu trúc dữ liệu public (D23);
   - không mở dữ liệu/câu hỏi 6 shop holdout (`biz_103, 110, 117, 123, 125, 128`);
   - không chọn cấu hình bằng điểm public (D21).
6. **Kiểm tài khoản portal** (mục 3b): mở được các mục nộp không, **không bấm nộp**. Báo kết quả vào nhóm.

### 0.2. Người A (kỹ thuật) — việc đầu tiên

- Đọc `docs/ARCHITECTURE.md` mục 8 (agent), `docs/DECISIONS.md` D20–D25, `docs/BENCHMARK.md` mục 7.
- Chạy thử benchmark ingest (CPU, không cần GPU):
  ```bash
  PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py materialize --jobs 2   # 40–50 phút, chạy lại được nếu bị ngắt
  PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py ingest --jobs 2        # in bảng giống BENCHMARK mục 7
  ```
  `materialize` phải in `0 variants differ`. Máy ≤ 16 GB RAM: giữ `--jobs 2` (hoặc 1).
- Việc tiếp theo theo mục 4 (13–15/11). Mỗi thay đổi `src/`: chạy dev, ghi `EXPERIMENTS.md`, mở PR.

### 0.3. Người B (vận hành, máy 5090) — việc đầu tiên

- Đọc `docs/GPU_RUNBOOK.md` mục 1–3 và 6b, `docker/README.md`.
- Khi PM nhắn: build image LLM từ commit PM chỉ định (hiện là `97aee0b`) theo mục 4 ngày 12/11, chạy
  `check_image.sh` **hai lần**, gửi số liệu. **Không upload, không nộp gì** khi PM chưa chọn.
- Kiểm có sẵn trên Drive tar rule-only `zaic2026_datamate_aio_zalo_final.tar` (MD5 `fb43b6cae2c3eb5132d58077bbc8bf58`): đây là bản final mặc định.

### 0.4. Khi bị kẹt

| Triệu chứng | Cách xử lý |
|---|---|
| `py -3.11` không có | Cài Python 3.11 từ python.org (tick "Add to PATH"); kiểm bằng `py -0` |
| `pytest` ra số khác 210 | 192 passed + 18 skipped = chưa thấy data: kiểm đã giải nén đúng chỗ (`data/train/train/questions.jsonl`) |
| dev ra số khác 2410/0/37 | Kiểm đang ở `main` mới nhất, đã `pip install -e ".[dev]"` lại sau khi pull |
| Hết RAM, máy treo, tiến trình bị kill | Bỏ bớt `--jobs`; chạy một shop: `run_eval.py --split dev --only biz_101` |
| Đầy ổ đĩa | Xoá `cache/` (tự build lại) và `eval/synth/out/` (dựng lại bằng `materialize`) |
| Git Bash: `docker run ... /code/predict.sh` báo không thấy file | Thêm `MSYS_NO_PATHCONV=1` trước lệnh `docker` |
| `git commit` bị pre-commit chặn | Đọc dòng báo lỗi: thường là file `data/`, `.jsonl` hay file > 1 MB. Bỏ file đó khỏi commit, **đừng** `--no-verify` |
| Không chắc một việc có phạm luật không | Hỏi PM trước khi làm |

---

## 1. Hiện trạng (chốt 11/11)

| Hạng mục | Kết quả | Ở đâu |
|---|---|---|
| Pipeline rule-only (không model) | dev **0.985**, holdout **0.859** | `src/datamate/`, chạy CPU |
| **Image rule-only** (final mặc định) | Image `sha256:2b324d85…`, code tag `ts1-ruleonly` (`3c5848c`). **Tar dùng để nộp: `zaic2026_datamate_aio_zalo_final.tar`, tag `zaic2026_datamate_aio_zalo:final`, MD5 `fb43b6cae2c3eb5132d58077bbc8bf58`**, 308 MB (11/11: chỉ gắn lại tag tên đội, image không đổi; đã chạy lệnh BTC trên 1 shop dev: 40/40, `submit_check` PASS). **Cần Test Solution cho tar này** (12/11). Tar cũ tag `zaic2026_datamate_test:final` (MD5 `8cf7dec3…`) đã Test passed 10/11 nhưng **sai tên đội, không dùng để nộp** | Drive nhóm; code: `git checkout ts1-ruleonly` |
| Public v0.7 (rule-only) | **0.114** | `docs/EXPERIMENTS.md` bảng "Nộp public" |
| Agent code-gen (Qwen3-14B-AWQ, D20/D22) | code-gen một mình: dev 0.653, holdout **0.444** (precision 0.776). Policy `agree` m=2 trên holdout 0.708 (72 câu) | `src/datamate/agent/` |
| Public v2 (agent, Kaggle) | **0.117** | như trên |
| **Image LLM** (`docker/Dockerfile.llm`) | **Chưa build có weights, chưa chạy trên GPU.** 11/11: build không weights từ `main` `97aee0b` trên CPU OK, chạy lệnh BTC trên 1 shop dev (không GPU → tự chuyển rule-only) 40/40. Commit build chính thức: `97aee0b` (PM chốt) | `docs/GPU_RUNBOOK.md` mục 6b |
| Test | 210 passed; ruff sạch | `pytest tests -q` |
| Benchmark shop tổng hợp E0 (D24) | **Đóng băng và niêm phong** 534 cell (tái lập 0/205 khác). Ingest sau D25: status_newword 100%, status_codes 10/16 A, 5/8 B; layout B và rename_tables A mới phân tích, chưa sửa. **Chưa chạy agent (E2), chỉ chạy sau Test Solution** | `docs/BENCHMARK.md` |
| Ingest trạng thái (D25) | Trạng thái lạ → `unknown` (không hợp lệ); đọc chú giải trạng thái trong README. Dev, holdout không đổi. **Chưa có trong image rule-only**; PM quyết có đưa vào image LLM không | `docs/DECISIONS.md` D25 |

**Đánh giá thẳng:** agent chưa tổng quát tốt sang shop lạ (dev 0.653 → holdout 0.444), và image LLM chưa từng chạy
trên 5090. Image rule-only là bản **đã chắc chắn chạy được**. Đó là phương án an toàn cho final.

---

## 2. Phân vai

| Người | Phụ trách | Cần |
|---|---|---|
| **PM — Khôi** | **Quyết định cuối**: cấu hình nộp (AgentConfig, policy, model), image final, thời điểm build image, nộp Test Solution, nộp public, nộp form "Solution Submission". Duyệt PR vào `main`. Phân công E2 sau khi Test Solution xong | Quyền admin GitHub, tài khoản portal (nộp bài), Owner Drive |
| **Người A — kỹ thuật** (người biết Python nhiều hơn) | Sửa code, chạy eval/Kaggle, calibrate, ghi `DECISIONS.md`/`EXPERIMENTS.md`. **Chuẩn bị số liệu và đề xuất** cho PM (không tự chốt cấu hình hay image) | Laptop CPU, tài khoản GitHub (write), tài khoản Kaggle |
| **Người B — vận hành** | Máy 5090: build/kiểm image **khi PM yêu cầu**, chạy private; Drive, MD5, chuẩn bị file nộp; theo dõi forum/BTC; đặt GPU dự phòng | Máy 5090 (hoặc GPU thuê), Drive nhóm (Editor), tài khoản Kaggle |

Hai người **review PR của nhau**; PR vào `main` cần PM duyệt. Mọi việc **bấm nộp** (public, Test Solution, Solution
Submission) do PM làm hoặc PM xác nhận bằng tin nhắn trước.

### Mặc định khi PM vắng

Các mặc định này áp dụng khi PM không trả lời được. Không ai được tự chọn cấu hình khác thay cho PM.

- **Image final:** nếu đến **16/11 18:00** vẫn chưa có quyết định của PM, final = **image rule-only đã qua Test
  Solution**: tar `zaic2026_datamate_aio_zalo_final.tar`, tag `zaic2026_datamate_aio_zalo:final`, MD5 `fb43b6cae2c3eb5132d58077bbc8bf58` (build từ tag `ts1-ruleonly` = `3c5848c`, trên Drive).
  Không nộp image LLM, kể cả khi nó đã qua cổng, nếu PM chưa chọn. Không nộp tar cũ MD5 `8cf7dec3…` (sai tên đội).
- **Người nộp form "Solution Submission":** nếu PM vắng lúc 16/11 18:00, Người B nộp image rule-only ở trên bằng tài
  khoản đã kiểm nộp được (mục 3b), trước 23:59, rồi nhắn nhóm link Drive và MD5 đã nộp.
- **Public:** khi PM vắng thì không nộp public. Public chỉ để theo dõi (D21), bỏ một lượt không ảnh hưởng điểm.
- **Private 20–23/11:** Người B chạy **đúng image đã nộp**, kiểm `submit_check`, và upload trước 23/11 23:59 nếu PM
  không xác nhận được trước 23/11 18:00.
- **Code:** khi PM vắng thì không merge vào `main` và không build image mới.

---

## 3. Checklist bàn giao (PM làm)

### 3a. Nhánh code
- [x] `fix/agent-runtime-errors` merge vào `main` qua PR #8 (11/11, `5e21a8e`), tag ở PR #9. CI `main` xanh. Mọi
      nhánh cũ khác đã có nội dung trong `main` (trừ nhánh Dependabot, để sau 16/11). Chưa xoá nhánh nào.
      **Từ giờ mọi việc làm từ `main`**
- [ ] Thay đổi ingest ngày 11/11 (chú giải trạng thái từ README, trạng thái `unknown`) làm đổi hành vi trên shop có
      trạng thái lạ. Dev không đổi (2410/0/37). PM quyết có đưa vào image LLM hay không. Image rule-only đã qua Test
      Solution build từ tag `ts1-ruleonly` (`3c5848c`) và **không** có thay đổi này

### Tag trên GitHub (không xoá)
- `ts1-ruleonly` → `3c5848c`: mã nguồn của image rule-only đã qua Test Solution #1, tức phương án final mặc định.
  BTC hỏi tái lập image này thì dùng tag này.
- `archive/pre-squash-pr8` → `02bd1c2`: toàn bộ lịch sử trước khi PR #8 được squash vào `main` (`5e21a8e`). Các hash
  dẫn trong `DECISIONS.md`, `EXPERIMENTS.md`, `PLAN.md` (ví dụ `eaead07`, `3994246`, `fae7b33`) chỉ có trong lịch sử
  này. Xem bằng `git log archive/pre-squash-pr8`. Không xoá tag này, và không xoá nhánh `fix/agent-runtime-errors`.

### 3b. Quyền truy cập
| Hệ thống | Việc cần làm | PM giữ | Người A | Người B |
|---|---|---|---|---|
| **GitHub** repo | Mời collaborator; ruleset `protect-main` giữ nguyên; `.github/CODEOWNERS` thêm Người A cạnh `@dinhkhoi124` (PM vẫn duyệt) | Admin | Write | Write |
| **Drive** thư mục `DataMate/` | Kiểm có: `data/train.zip`, `data/public_test.zip`, **tar rule-only `zaic2026_datamate_aio_zalo_final.tar` + file `.md5` (MD5 `fb43b6cae2c3eb5132d58077bbc8bf58`)**, zip Kaggle (`datamate_dev_*`, `datamate_public_*`). Tar image mới upload vào đây | Owner | Viewer | Editor |
| **Kaggle** | Share dataset private `datamate-train`, `datamate-public` (hoặc tạo lại theo `docs/KAGGLE.md`). Mỗi tài khoản 30 giờ GPU/tuần: ghi ai dùng bao nhiêu | Owner dataset | Collaborator | Collaborator |
| **Portal cuộc thi** | Xác nhận tài khoản nào nộp được public, Test Solution, "Solution Submission" và upload private. **PM nộp**; nếu 20–23/11 PM vắng thì Người B upload private theo tin nhắn xác nhận của PM | Người nộp | – | Dự phòng upload private |
| **Máy 5090 / GPU thuê** | Đặt lịch 12/11 (build image LLM) và 20–23/11 (private); phương án thuê dự phòng | Duyệt chi phí | – | Quản lý |
| **Liên lạc** | Nhóm chat; PM trả lời câu hỏi chặn trong ngày | ✓ | ✓ | ✓ |

- [ ] **Kiểm tài khoản portal của từng thành viên** (Người A, Người B): đăng nhập, vào DataMate, xem có thấy và dùng
      được các mục **nộp public**, **Test Solution**, form **"Solution Submission"** và **upload private** không.
      Chỉ mở form để kiểm, **không bấm nộp**. Ghi kết quả (ai nộp được mục nào) vào nhóm chat. Nếu chỉ tài khoản PM
      nộp được thì hỏi BTC cách thêm người nộp, trước 14/11, vì mục "Mặc định khi PM vắng" cần Người B nộp được
- [ ] **PM upload lên Drive** tar rule-only gắn tag mới: `zaic2026_datamate_aio_zalo_final.tar` và `zaic2026_datamate_aio_zalo_final.tar.md5` (máy PM, thư mục
      `zaic_submit/`), bật link công khai, rồi tải thử về một máy khác và kiểm MD5 `fb43b6cae2c3eb5132d58077bbc8bf58`
- [ ] Nhắn nhóm: link file này và lịch mục 4

---

## 4. Lịch còn lại và việc cụ thể

Hạn cứng: **Test Solution đóng 15/11 23:59 (1 lượt/ngày)** · **Nộp image final 16/11 23:59 (tối đa 2 lần, BTC xét lần
cuối)** · **Private: 20/11 09:00 → 23/11 23:59, cần GPU** · Public: tối đa 5 lần/ngày, đến 16/11.

### 12/11 — Cài đặt + Test Solution rule-only + thử image LLM
1. Cả hai: làm mục 0.1 (cài đặt, kiểm 210 test và dev 2410/0/37).
2. **PM:** upload tar rule-only gắn tag mới lên Drive và nộp **Test Solution** cho nó (lượt của ngày 12/11).
   Ghi kết quả vào `PLAN.md`. Image không đổi, chỉ đổi tên tag, nên kỳ vọng Test passed như 10/11.
3. **Người B (máy 5090):** build image LLM từ commit **`97aee0b`** (PM đã chốt):
   ```bash
   git fetch && git checkout 97aee0b
   hf download Qwen/Qwen3-14B-AWQ --local-dir models/Qwen3-14B-AWQ
   docker build --platform linux/amd64 -f docker/Dockerfile.llm \
     --build-context weights=models/Qwen3-14B-AWQ --build-arg MODEL=Qwen3-14B-AWQ \
     -t zaic2026_datamate_aio_zalo:final .
   bash scripts/gpu/check_image.sh zaic2026_datamate_aio_zalo:final data/train/train      # chạy HAI lần
   ```
   Chú ý: image LLM và image rule-only **cùng tên tag**. Trên máy 5090 chỉ build image LLM; tar rule-only để nguyên
   trên Drive, không `docker load` nó lên cùng máy lúc đang kiểm image LLM.
4. Người B gửi PM và Người A: exit code, tổng thời gian, giây/câu, PA train của mỗi lần, độ lệch hai lần (chỉ số,
   không gửi nội dung câu hỏi). **Không upload, không nộp gì.**

### 13–14/11 — Quyết định image final (PM quyết; Người A chuẩn bị số liệu)
Image LLM **chỉ được chọn** khi đạt đủ cả ba cổng (D23, luật đề):
- Hai lần chạy trên train lệch **≤ 0.03** (mục tiêu ≤ 0.01).
- Thời gian ước tính cho private **≤ 6 giờ**, tính cả load model. Docker Guide ước ~500 câu, tức trung bình
  **≤ ~40 giây/câu**. Trên Kaggle T4 agent chạy 83 giây/câu, nên phải đo thật trên 5090.
- PA train của image khớp bản local (kiểm lỗi đóng gói), và **Test Solution passed**.

Không đạt cổng nào thì **final = image rule-only** (tar `zaic2026_datamate_aio_zalo_final.tar`, MD5 `fb43b6cae2c3eb5132d58077bbc8bf58`). Không được chọn image bằng cách so điểm public
(CLAUDE.md luật 8, D21).

Nếu image LLM đạt và PM chọn: Người B upload tar lên Drive (30–40 GB, mất nhiều giờ), tính MD5; **PM nộp Test
Solution #2** trước **15/11 23:59**. **Ưu tiên tuyệt đối**: không việc nghiên cứu nào (kể cả benchmark E2) được chiếm
GPU hay thời gian cần cho việc build, kiểm và nộp Test Solution.

### 13–15/11 — Cải thiện điểm (Người A, song song, chỉ khi còn thời gian)

**Benchmark E2 (agent trên shop tổng hợp, `docs/BENCHMARK.md`) chưa chạy.** Chỉ chạy **sau khi Test Solution đã
xong** và còn đủ GPU (ước khoảng 12 giờ Kaggle T4). Người chạy do PM phân công sau Test Solution. Phần CPU (ingest
trên synthetic: `bench.py ingest`) chạy được bất cứ lúc nào.

Chỉ làm trên **train** (dev để sửa, holdout chỉ xem số tổng). Mỗi thay đổi: chạy dev, ghi một dòng `EXPERIMENTS.md`.
Theo thứ tự lợi/công:
1. **Lỗi runtime còn lại của agent** (EXPERIMENTS 11/11): `find_units` bịa tên helper (2), `KeyError` cột (4),
   ưu tiên toán tử `&` (3). Sửa helper/sandbox rồi kiểm bằng `eval/reexec_agent.py` + `eval/calibrate_agent.py` trên
   trace Kaggle cũ, **không cần GPU** (`docs/KAGGLE.md` mục "Lead đọc kết quả trên CPU").
2. Đổi prompt hay model **cần chạy lại Kaggle** (T4 ×2, ~83 giây/câu, 30 giờ GPU/tuần/tài khoản). Ứng viên model:
   `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`, `Qwen/Qwen3-8B-AWQ` (`docs/KAGGLE.md` mục "Chọn model").
3. Mọi thay đổi code sau khi image đã build thì **phải build lại image và kiểm lại cổng**. Từ 15/11 chỉ sửa lỗi chặn.

### 16/11 — Nộp final (PM nộp; Người B chuẩn bị tar/MD5, Người A xác nhận số liệu)
1. PM chốt commit trên `main`, gắn tag `vfinal`. Nếu final là image rule-only thì commit là tag `ts1-ruleonly`,
   tar `zaic2026_datamate_aio_zalo_final.tar`, MD5 `fb43b6cae2c3eb5132d58077bbc8bf58`.
2. Tar image trên Drive **public link**, MD5 khớp, thử tải về từ một máy khác.
3. PM nộp form "Solution Submission" **trước 23:59** (để dư vài giờ). Model type: **"Based on small models (<=15B)"**.
4. Ghi vào `EXPERIMENTS.md` và tick `PLAN.md`. Giữ nguyên tar và MD5: private **bắt buộc chạy đúng image này**.

### 20–23/11 — Private (Người B chạy; PM upload hoặc xác nhận trước khi upload)
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
| Image LLM không ổn định hoặc quá 6 giờ | Final = image rule-only (tar `zaic2026_datamate_aio_zalo_final.tar`). Không cố đến phút chót |
| Nộp nhầm tar sai tên đội | Chỉ nộp tar có tag `zaic2026_datamate_aio_zalo:final`. Kiểm bằng `tar -xOf <tar> manifest.json` phải thấy `"RepoTags":["zaic2026_datamate_aio_zalo:final"]` |
| Upload 30–40 GB lên Drive không kịp | Bắt đầu upload ngay khi image LLM qua cổng, muộn nhất 15/11 |
| Không có GPU 20–23/11 | Đặt máy trước 16/11, có phương án thuê dự phòng |
| Sửa code phút chót làm hỏng image | Sau 15/11 không đổi `src/` trừ lỗi chặn; mọi thay đổi qua PR + CI xanh |
| Thắc mắc về luật | Hỏi BTC qua forum (ghi câu trả lời vào `SPEC.md` mục 7), không đoán |
