# EXPERIMENTS — Bảng kết quả

Quy ước:
- Mỗi lần chạy `/eval` sinh ra một dòng. `version` = `v<major>.<minor>` kèm commit hoặc mô tả ngắn.
- **PA holdout là chỉ số chính.** Dev dùng để debug, public chỉ ghi khi đã nộp.
- Log chi tiết lưu ở `eval/runs/<timestamp>/` (gitignore), bảng này chỉ ghi kết quả tóm tắt.

## Kết quả

| Ngày | Version | Thay đổi | PA dev | PA holdout | Prec holdout | Coverage holdout | PA public | Thời gian | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | v0.0 | baseline toàn abstain | 0.000 | 0.000 | – | 0% | – | – | kiểm tra harness |
| 2026-11-04 | v0.1-ingest | Ingest 30 shop → Canonical Store | – | – | – | – | – | 113s nguội / 30 shop | Oracle doanh thu tháng (`analysis/sanity_ingest.py`): **dev 68/68, holdout 8/8** |
| 2026-11-04 | v0.2-core | normalize + timeparse + link + execute + rule parser (chưa có LLM) | **1.000** (2447/2447, 0 sai) | **0.872** (502/20/31) | 96.2% | 94.4% | – | ~210s dev, 41s holdout (cache nóng) | Dev đầu 0.835 → 1.000. Holdout lệch dev 0.13 → parser/ingest overfit dev. Sai nhiều: revenue 4, item_qty 3, shift 2, expiry 2, gross_profit 2. Precision < 97% |

## Theo intent (bản mới nhất)

Dán bảng breakdown của `eval/run_eval.py --by intent` cho holdout vào đây khi có thay đổi lớn.

## Nộp public

| Ngày | Lần | Version | PA public | Ghi chú |
|---|---|---|---|---|
