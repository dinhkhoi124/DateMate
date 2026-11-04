---
description: Chạy pipeline trên split train (dev|holdout|all), chấm PA, báo cáo lỗi
argument-hint: "[dev|holdout|all] [--by intent|shop|type]"
---

Chạy eval cho split `$ARGUMENTS` (mặc định `dev` nếu trống):

1. Chạy `PYTHONPATH=src .venv/Scripts/python eval/run_eval.py --split <split> --by intent --errors 20` (thêm các tham số khác trong `$ARGUMENTS` nếu có).
2. Báo cáo ngắn gọn bằng tiếng Việt:
   - PA, precision, coverage, số đúng/sai/abstain.
   - So với dòng gần nhất cùng split trong `docs/EXPERIMENTS.md`: tăng hay giảm.
   - Top intent gây **sai** (−1). Các lỗi sai quan trọng hơn abstain.
3. Với split `holdout`: **không mở dữ liệu hay câu hỏi của shop holdout để debug**, chỉ đọc số tổng hợp theo intent.
4. Hỏi người dùng có muốn ghi kết quả vào `docs/EXPERIMENTS.md` không. Nếu đồng ý, thêm một dòng theo đúng format bảng.
