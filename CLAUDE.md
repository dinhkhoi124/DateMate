# CLAUDE.md — DataMate (Zalo AI Challenge 2026)

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của shop nhỏ. Chấm bằng **PA = (Đúng − Sai)/N**: sai bị −1, abstain được 0.
Đọc thêm khi cần: `docs/SPEC.md` (đề và luật), `docs/ARCHITECTURE.md` (thiết kế), `docs/PLAN.md` (việc hôm nay), `docs/DECISIONS.md`, `docs/EXPERIMENTS.md`.

## Luật bất di bất dịch
1. **Không đọc, không dùng `data/public_test/**/questions.jsonl` để viết luật, prompt hay few-shot.** Public chỉ dùng để sinh submission khi nộp. Không có private.
2. **Không mở dữ liệu và câu hỏi của shop holdout** (`eval/splits.json` → `holdout`) khi debug. Chỉ chạy eval tổng hợp trên chúng.
3. **Không hardcode** theo `id`, `business_id` hay tên shop cụ thể. Mọi xử lý đặc thù phải suy ra từ README, dữ liệu hoặc rulebook.
4. **Mọi phép tính số làm bằng code deterministic** (`src/datamate/execute/`). LLM chỉ sinh intent hoặc code. Không để LLM tự ra con số.
5. **Không chắc thì abstain.** `unanswerable` chỉ trả khi có căn cứ deterministic (capability thiếu hoặc kỳ hỏi ngoài dữ liệu).
6. Model ≤ 15B tổng tham số, chạy offline. Không gọi mạng trong `src/`.
7. `data/` là read-only. Không ghi vào đó.
8. **Không dùng điểm public để chọn hoặc tối ưu cấu hình** (policy, m, prompt, luật). Mọi tuning chạy trên train; submission chính thức chỉ sinh bằng cấu hình đã commit (D21).

## Lệnh
```bash
# chạy Python (Windows, Git Bash)
PYTHONPATH=src .venv/Scripts/python -m datamate.run --data data/train/train --out eval/runs/latest/submission.jsonl
PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split dev        # dev | holdout | all
PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split dev --by intent --errors 30
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python eval/scorer.py <submission.jsonl> [answers.jsonl]
```
Lệnh tắt: `/eval [dev|holdout|all]`, `/submit-check <submission.jsonl>`.

## Quy ước code
- Python 3.11, package `src/datamate/`. Module: `ingest, normalize, timeparse, parse, link, execute, answer`. Entrypoint là `datamate.run`.
- Mỗi luật rulebook là một hàm có docstring trích mục rulebook (ví dụ `# rulebook §1: đơn trả một phần bỏ cả đơn`) và có test trong `tests/`.
- Tham số luật (ca sáng/chiều, đầu tuần, khoảng ngày của Tết) đặt trong config, không rải trong code.
- Đọc file luôn dùng `encoding="utf8"`. Chuỗi Unicode chuẩn hoá NFC. Tên trả về lấy nguyên từ dữ liệu.
- Sắp xếp ổn định (tie-break theo mã), không phụ thuộc thứ tự dict/set, để kết quả deterministic.
- Không thêm dependency khi chưa ghi vào `pyproject.toml` và `DECISIONS.md`.

## Quy trình làm việc
- Bắt đầu phiên: đọc mục ngày hiện tại trong `docs/PLAN.md`.
- Sau mỗi thay đổi đáng kể: chạy `/eval dev`, rồi ghi một dòng vào `docs/EXPERIMENTS.md`. Chạy `holdout` khi chốt một milestone.
- Quyết định kiến trúc hoặc chọn model: ghi vào `docs/DECISIONS.md` (bối cảnh → lựa chọn → lý do → hệ quả).
- Tick checklist trong `docs/PLAN.md` khi xong việc.
- Trả lời người dùng bằng tiếng Việt.
