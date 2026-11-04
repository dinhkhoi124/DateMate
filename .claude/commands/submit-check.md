---
description: Kiểm tra file submission.jsonl hợp lệ trước khi nộp (format, đủ id, type, tuân thủ luật)
argument-hint: "<path/to/submission.jsonl> [questions.jsonl]"
---

Kiểm tra submission `$ARGUMENTS`:

1. Chạy `PYTHONPATH=src .venv/Scripts/python eval/submit_check.py $ARGUMENTS`. Nếu script chưa có, viết nó theo các điều kiện ở bước 2 trước khi chạy.
2. Các điều kiện phải đạt (theo `docs/SPEC.md` §3):
   - Mỗi dòng là JSON hợp lệ, có `id` và `type`. Không có id trùng. Mọi id đều thuộc `questions.jsonl`. Báo số câu thiếu (sẽ tính là abstain).
   - `type` ∈ {number, percent, date, boolean, string, list, table, unanswerable, abstain}.
   - number/percent là JSON number (không phải chuỗi, không phải bool). Tiền là số nguyên. Percent đã làm tròn 2 chữ số.
   - date khớp `^\d{4}-\d{2}(-\d{2})?$`. boolean là `true`/`false`. string không rỗng. list là mảng.
   - table có `columns` và `rows`, các ô số là number.
   - unanswerable: nếu có `reason` thì phải là chuỗi.
   - Không có NaN/Infinity. File mã hoá UTF-8 không BOM.
3. Thống kê phân bố type và tỉ lệ abstain. Cảnh báo nếu tỉ lệ `unanswerable` vượt quá 5%, vì train chỉ có 2.3%.
4. Kết luận **PASS / FAIL** kèm danh sách lỗi. Không tự nộp bài.
