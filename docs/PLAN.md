# PLAN — Roadmap & checklist

Freeze Docker: **16/11 23:59** · Mốc an toàn: **12/11 phải có image chạy được** · Chi tiết thiết kế: `ARCHITECTURE.md` · Đề bài: `SPEC.md`

## Mục tiêu đo được

| Chỉ số | Cách đo | Ngưỡng |
|---|---|---|
| PA holdout | 6 shop train giữ lại, không nhìn khi viết code | ≥ 0.90 |
| Precision (câu đã trả lời) | đúng / (đúng + sai) | ≥ 97% |
| Thời gian | 1000 câu trên 1× GPU 32GB, tính cả load model | ≤ 2 giờ |
| Ổn định | chạy 2 lần | lệch PA ≤ 0.005 |

**Chia tập:** holdout gồm `biz_103, biz_110, biz_117, biz_123, biz_125, biz_128`, mỗi shop một kiểu xuất khác nhau. Danh sách chính thức nằm ở `eval/splits.json`. Phần còn lại là dev.

## Checklist theo ngày

### Phase 1 — Plan & Design (04/11) ✅
- [x] Đọc thể lệ, rulebook. Khám phá data và intent (`analysis/intent_survey.py`)
- [x] SPEC, ARCHITECTURE, PLAN, DECISIONS, CLAUDE.md, cấu hình `.claude/`
- [x] venv `.venv` (Python 3.11) và skeleton `src/datamate/`

### Phase 2 — Build (05–12/11)
**05/11 — Harness**
- [x] `eval/scorer.py` (exact typed match §9)
- [x] `eval/run_eval.py`: chạy pipeline trên split, báo cáo PA/precision theo intent × shop, ghi log lỗi
- [x] Unit test cho scorer (`tests/test_scorer.py`, 7 test)
- [x] `src/datamate/run.py` chạy được end-to-end (baseline toàn abstain, PA = 0), `eval/submit_check.py` báo PASS

**05–06/11 — Ingest**
- [x] Từ điển `mô tả README → field chuẩn` (`ingest/vocab.py`), học tên cột (`learned_names.json`), suy luận dự phòng theo tên + giá trị (`ingest/infer.py`)
- [x] Reader: csv, xlsx nhiều sheet/tháng, dò header, sqlite, gộp `_1/_2/_năm`, sửa tên file CP437, calamine + cache
- [x] Forward-fill nhóm dòng, dedupe dòng trùng trong đơn, bỏ dòng tổng phụ, trả hàng số âm/lô, trạng thái trong Ghi chú
- [x] Capability profile + validation. **Load được 30/30 shop train (113s nguội)**, doanh thu tháng khớp 68/68 dev, 8/8 holdout
- [ ] Còn mở: xác minh `Không giao được` / `Đang xử lý`, shop không mã đơn (107/113/126: mỗi dòng = 1 đơn?), sqlite `the_kho` và `tra_hang`

**06–09/11 — Core**
- [ ] `normalize/`: Telex, teencode, bỏ dấu
- [ ] `timeparse/`: mọi mục rulebook §2, kể cả Tết theo `calendar.csv`
- [ ] `link/`: fuzzy match catalog
- [ ] `execute/`: viết op theo thứ tự tần suất (doanh thu → số đơn → SL hàng → tăng trưởng → huỷ → argmax ngày → AOV → tỉ trọng → ca → top khách → bán chạy → giá → tồn → nhập → nợ → lãi → ngành)
- [ ] `parse/rules.py`: rule parser. **Mục tiêu PA dev ≥ 0.85**

**08–10/11 — LLM** (cần GPU)
- [ ] Chốt model ≤ 15B, ghi vào `DECISIONS.md`
- [ ] vLLM + guided JSON, prompt có few-shot từ dev
- [ ] Ensemble + Confidence Gate, calibrate. **Mục tiêu PA holdout ≥ 0.85, precision ≥ 97%**

**10–12/11 — Mở rộng**
- [ ] Fallback code-gen cho câu OOD
- [ ] Luật ngành bổ sung, đọc rulebook lúc chạy
- [ ] Dữ liệu tự sinh: paraphrase có lỗi gõ, shop tổng hợp đổi schema

### Phase 3 — Test (song song, chốt 15/11)
- [ ] Test độ bền: shop và rulebook tổng hợp
- [ ] Test thời gian và độ lệch giữa 2 lần chạy
- [ ] `/submit-check` xanh

### Phase 4 — Deploy (12–16/11)
- [ ] Dockerfile (CUDA ≥ 12.8, weights đóng gói sẵn), chạy `--network none`
- [ ] Nộp public (≤ 5 lần/ngày), ghi kết quả vào `EXPERIMENTS.md`
- [ ] **Nộp image cuối trước 16/11 23:59**

### Phase 5 — Maintain (20–23/11)
- [ ] Theo dõi kết quả private
- [ ] Public code (quy định 8.2), viết writeup

## Rủi ro

| Rủi ro | Phương án |
|---|---|
| Private có ngành/luật/kiểu file mới | Map README theo mô tả, đọc rulebook lúc chạy, nhánh code-gen dự phòng, abstain mặc định |
| Máy dev không có GPU | Executor và rule parser làm trên CPU. Thuê GPU cho phần LLM và Docker |
| Kết quả không deterministic | temp 0, seed cố định, pin phiên bản, sắp xếp ổn định |
| Chạy chậm | Cache Store, batch LLM, timeout mỗi câu (quá giờ thì abstain) |
| Hiểu sai luật | Mỗi luật có test riêng từ câu train. Báo cáo lỗi theo intent |
