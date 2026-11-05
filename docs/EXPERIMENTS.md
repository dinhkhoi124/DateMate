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
| 2026-11-04 | v0.3-gate | Chốt chặn phủ từ D10 (min-shops 2, 206 từ) | – | **0.859** (485/10/58) | 98.0% | 89.5% | – | 59s holdout | Sai 20 → 10. LOIO dev coverage 0.889. Public: abstain 69.0%, unanswerable 1.7%, submit_check PASS (`eval/runs/public_v0.3`). Còn sai: revenue 3, shift 2 |
| 2026-11-05 | v0.3 + ingest (0b04c7a) + eval/synth fix | Proxy shop lạ đủ 6 transform × 24 shop dev, có e2e (2447 câu, PA gốc 0.985) | 0.985 | – | – | – | – | ~3–6 phút / 8 shop / transform | Ingest/PA synth: rename 21/24 0.773 (209 đúng→sai, 101 →abstain; hỏng 106/109/124), nodoc 24/24 0.985, layout 24/24 0.985, values 24/24 0.985, status_note 24/24 0.985 (trước 23/24 0.933: README synth vẫn ghi Ghi chú 'cột trống' → lỗi transform, không phải ingest), combo 20/24 0.695 (rename + biz_105 with_time/rev_by_month) |

## Theo intent (bản mới nhất)

Dán bảng breakdown của `eval/run_eval.py --by intent` cho holdout vào đây khi có thay đổi lớn.

## Nộp public

| Ngày | Lần | Version | PA public | Ghi chú |
|---|---|---|---|---|
| 2026-11-04 | 1 | v0.2-core (+ fix NaN date_min) | **0.066** | Abstain 32.2% (214 câu do ingest không nhận bảng đơn), unanswerable 4.0%. Holdout 0.872 → public 0.066: ước ~372 đúng / ~306 sai trên 678 câu trả lời (precision ~55%). Rulebook public thêm ngành mới (mẹ & bé, sửa xe, điện thoại) và ~10 luật mới → parser trả lời tự tin sai trên câu lạ |
| 2026-11-04 | 2 | v0.3-gate (6e5f1a6) | **0.076** | 310 câu trả lời → ước ~193 đúng / ~117 sai (precision ~62%). Gate bỏ 368 câu, trong đó ~179 đúng / ~189 sai → **từ lạ gần như không phân biệt được đúng/sai trên public**. Phần lớn câu sai nằm ở câu diễn đạt quen nhưng tính sai → nghi ingest/ngữ nghĩa trên shop lạ |
