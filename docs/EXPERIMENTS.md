# EXPERIMENTS — Bảng kết quả

Quy ước:
- Mỗi lần chạy `/eval` sinh ra một dòng. `version` = `v<major>.<minor>` kèm commit hoặc mô tả ngắn.
- **PA holdout là chỉ số chính.** Dev dùng để debug, public chỉ ghi khi đã nộp.
- Log chi tiết lưu ở `eval/runs/<timestamp>/` (gitignore), bảng này chỉ ghi kết quả tóm tắt.

## Kết quả

| Ngày | Version | Thay đổi | PA dev | PA holdout | Prec holdout | Coverage holdout | PA public | Thời gian | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | v0.0 | baseline toàn abstain | 0.000 | 0.000 | – | 0% | – | – | kiểm tra harness |

## Theo intent (bản mới nhất)

Dán bảng breakdown của `eval/run_eval.py --by intent` cho holdout vào đây khi có thay đổi lớn.

## Nộp public

| Ngày | Lần | Version | PA public | Ghi chú |
|---|---|---|---|---|
