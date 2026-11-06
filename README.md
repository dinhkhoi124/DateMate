# DataMate — Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng (csv / xlsx / sqlite) của các shop nhỏ, track **DataMate**.

Điểm chấm là **PA = (Đúng − Sai) / N**. Trả lời sai bị trừ 1, còn abstain (từ chối trả lời) được 0. Vì vậy hệ thống
ưu tiên **không trả lời sai**: không đủ căn cứ thì abstain.

Nguyên tắc thiết kế: **LLM dịch, code tính.** Câu hỏi được chuyển thành intent có cấu trúc, rồi executor
deterministic tính trên dữ liệu đã chuẩn hoá theo đúng rulebook. Chi tiết ở [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Trạng thái hiện tại (v0.4, rule-only, chưa có LLM):**

| Tập | PA | Đúng / Sai / Abstain | Precision |
|---|---|---|---|
| dev (24 shop train) | 0.985 | 2410 / 0 / 37 | 100% |
| holdout (6 shop train giữ lại) | 0.859 | 485 / 10 / 58 | 98.0% |
| public (đã nộp, bản v0.3) | 0.076 | – | ~62% |

Khoảng cách giữa holdout và public là vấn đề chính đang giải quyết: shop lạ, ngành lạ, luật mới trong rulebook. Xem
[`docs/PLAN.md`](docs/PLAN.md) và [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## ⚠️ Luật bắt buộc (đọc trước khi code)

Bản đầy đủ ở [`CLAUDE.md`](CLAUDE.md). Vi phạm có thể làm cả đội bị loại hoặc làm số đo mất ý nghĩa.

1. **Không đọc, không dùng `data/public_test/**/questions.jsonl`** để viết luật, prompt hay few-shot. Public chỉ
   dùng để sinh file nộp. Xem cấu trúc **dữ liệu** của public (file, cột) để cải thiện bộ đọc thì được.
2. **Không mở dữ liệu và câu hỏi của 6 shop holdout** (`eval/splits.json` → `holdout`: `biz_103, biz_110, biz_117,
   biz_123, biz_125, biz_128`). Chỉ chạy eval tổng hợp trên chúng. `--errors` và `--only` chỉ dùng cho shop dev.
3. **Không hardcode** theo `id`, `business_id` hay tên shop. Xử lý đặc thù nào cũng phải suy ra được từ README của
   shop, từ dữ liệu hoặc từ rulebook.
4. **Mọi con số do code deterministic tính** (`src/datamate/execute/`). LLM chỉ sinh intent hoặc code.
5. **Không chắc thì abstain.** Chỉ trả `unanswerable` khi có căn cứ deterministic.
6. Model ≤ 15B tham số, chạy offline. Không gọi mạng trong `src/`.
7. `data/` là read-only và **không commit lên git** (đã có trong `.gitignore`).

---

## Cài đặt

Yêu cầu **Python 3.11**. Không cần GPU cho phần rule-only.

```bash
git clone https://github.com/dinhkhoi124/DateMate.git
cd DateMate
```

**Windows (Git Bash):**
```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

**Linux / macOS:**
```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

> Các lệnh phía dưới viết theo Windows (`.venv/Scripts/python`). Trên Linux/macOS thay bằng `.venv/bin/python`.

### Dữ liệu

Dữ liệu thi **không nằm trong repo**. Tải `train.zip` (và `public_test.zip` nếu cần nộp) từ portal Zalo AI Challenge
(tab Data), hoặc xin file từ lead, rồi giải nén để được cấu trúc sau:

```
data/
├── train/train/                 ← giải nén train.zip vào data/train/
│   ├── questions.jsonl
│   ├── answers.jsonl
│   ├── rulebook.md
│   ├── calendar.csv
│   └── data/biz_101 … biz_130/  (csv, xlsx, sqlite + README.md mỗi shop)
└── public_test/public_test/     ← giải nén public_test.zip vào data/public_test/
```

```bash
mkdir -p data/train data/public_test
unzip train.zip -d data/train
unzip public_test.zip -d data/public_test
```

### Kiểm tra cài đặt

```bash
.venv/Scripts/python -m pytest tests -q
# Có data: 142 passed. Chưa có data: 138 passed, 4 skipped.
```

---

## Lệnh thường dùng

```bash
# Eval trên dev (thước đo hằng ngày, khoảng 3–4 phút lần đầu, sau đó nhanh hơn nhờ cache)
PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split dev
PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split dev --by intent --errors 30   # xem câu sai
PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split dev --only biz_101           # một shop dev

# Holdout: chỉ để regression check khi chốt milestone, KHÔNG debug trên đó
PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split holdout

# Proxy shop lạ: biến đổi shop dev (đổi tên cột, gộp file, đổi định dạng…) rồi đo lại
PYTHONPATH=src .venv/Scripts/python eval/synth/run.py --transforms rename --verbose

# Sinh file nộp + kiểm tra định dạng
PYTHONPATH=src .venv/Scripts/python -m datamate.run --data data/public_test/public_test --out eval/runs/<tên>/submission.jsonl
.venv/Scripts/python eval/submit_check.py eval/runs/<tên>/submission.jsonl

# Chấm một file submission với đáp án
.venv/Scripts/python eval/scorer.py <submission.jsonl> data/train/train/answers.jsonl
```

Cache (Store đã ingest) nằm ở `cache/`, tự build lại khi code ingest đổi. Kết quả chạy để trong `eval/runs/`
(không commit).

---

## Cấu trúc repo

```
src/datamate/          code inference (đây là phần đóng gói vào Docker)
├── ingest/            đọc csv/xlsx/sqlite, map cột theo README của shop → Store chuẩn; kiểm tra bất biến
├── normalize/         Telex, teencode, bỏ dấu
├── timeparse/         "tuần trước", "quý 2", "Tết năm nay"… → khoảng ngày (rulebook §2, calendar.csv)
├── parse/             rule parser → Intent; gate (từ lạ → abstain)
├── link/              khớp tên hàng/khách/nhóm trong câu với catalog
├── execute/           các phép tính deterministic theo rulebook
├── answer/            định dạng đáp án đúng type
├── pipeline.py        ghép các bước; run.py là entrypoint
└── config.py          tham số luật và ngưỡng (không rải hằng số trong code)
eval/                  scorer, run_eval, submit_check, splits.json, synth/ (proxy shop lạ + fault injection)
analysis/              script khảo sát dữ liệu và build từ vựng (learned_names.json, gate_vocab.txt)
tests/                 unit test, mỗi luật rulebook có test riêng
docker/                Dockerfile, predict.sh, manifest (xem docker/README.md)
docs/                  SPEC, ARCHITECTURE, PLAN, DECISIONS, EXPERIMENTS, WORKFLOW
thele/                 thể lệ cuộc thi
```

| Tài liệu | Đọc khi |
|---|---|
| [`docs/SPEC.md`](docs/SPEC.md) | Cần biết đề, định dạng đáp án, luật nộp bài, trả lời của BTC |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Cần hiểu pipeline và module |
| [`docs/PLAN.md`](docs/PLAN.md) | Muốn biết việc đang làm và việc còn mở (checklist theo ngày) |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Muốn biết vì sao làm như vậy (D01–D12) |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Muốn xem số liệu từng phiên bản và các lần nộp public |
| [`docs/WORKFLOW.md`](docs/WORKFLOW.md) | Cần biết phân vai, cách dùng GPU (Kaggle / thuê 5090) |

---

## Quy trình làm việc nhóm

1. **Luôn tạo branch từ `main` mới nhất**: `git checkout main && git pull && git checkout -b feat/<việc>`
   (ví dụ `feat/llm-parser`, `fix/ingest-sqlite`, `qa/paraphrases`).
2. Commit nhỏ, message rõ ràng. Không commit `data/`, `cache/`, `eval/runs/`, weights, token.
3. **Trước khi mở PR:**
   - `pytest tests -q` phải xanh.
   - Nếu đổi code trong `src/`: chạy `eval/run_eval.py --split dev`. **Số câu sai không được tăng.** PA dev giảm thì
     ghi rõ lý do trong PR.
   - Ghi một dòng vào `docs/EXPERIMENTS.md` (version, thay đổi, PA, đúng/sai/abstain).
   - Thêm luật rulebook mới thì cần một hàm có docstring trích mục rulebook (`# rulebook §x: …`) và một test trong `tests/`.
   - Quyết định kiến trúc hoặc chọn model thì ghi vào `docs/DECISIONS.md` (bối cảnh → lựa chọn → lý do → hệ quả).
4. Mở PR vào `main`, nhờ lead review. **`main` phải luôn chạy được** vì image Docker build từ đây.
5. Tick việc đã xong trong `docs/PLAN.md`.

**Quy ước code:** Python 3.11; đọc file luôn dùng `encoding="utf8"`; chuẩn hoá Unicode NFC; tên trả về lấy nguyên từ
dữ liệu; sắp xếp ổn định (tie-break theo mã) để kết quả deterministic; thêm dependency thì ghi vào cả `pyproject.toml`
và `DECISIONS.md`.

### Dùng Claude Code (tuỳ chọn)

Repo có sẵn `CLAUDE.md` và các lệnh tắt trong `.claude/commands/`: `/eval [dev|holdout|all]` và
`/submit-check <file>`. `.claude/settings.json` chặn sẵn việc đọc câu hỏi public và việc ghi vào `data/`.

---

## Docker

Image bản rule-only (CPU, chưa có model) đã build và chạy đúng lệnh của BTC: kết quả giống hệt bản local trên 3000 câu train.

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t zaic2026_datamate_<team>:final .
docker run --rm --network none --shm-size=16g --entrypoint bash \
  -v "$(pwd)/data/train/train:/data:ro" -v "$(pwd)/eval/runs/docker:/result" \
  zaic2026_datamate_<team>:final /code/predict.sh
```

Khi chấm, BTC thêm `--gpus`. Xem chi tiết ở [`docker/README.md`](docker/README.md) và SPEC mục *Submission contract*.
**Hạn chót nộp image: 16/11/2026 23:59.**
