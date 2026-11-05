# PLAN — Roadmap & checklist

Freeze Docker: **16/11 23:59** · Mốc an toàn: **12/11 phải có image chạy được** · Chi tiết thiết kế: `ARCHITECTURE.md` · Đề bài: `SPEC.md`

## Mục tiêu đo được

| Chỉ số | Cách đo | Ngưỡng |
|---|---|---|
| **PA proxy shop lạ** (từ 04/11) | shop tổng hợp từ train, đổi schema/cách ghi (`eval/synth/`). **Thước đo chính để ra quyết định** | tăng đều; precision ≥ 90% |
| **Ingest shop lạ** | đo riêng: Store dựng từ shop tổng hợp so với Store gốc (đơn, dòng, tiền, trạng thái, ngày) | ≥ 95% số chỉ số khớp |
| PA holdout | 6 shop train giữ lại, không nhìn khi viết code. Chỉ dùng làm **regression check**, không tối ưu thêm | không giảm > 0.02 |
| Precision (câu đã trả lời) | đúng / (đúng + sai) | ≥ 97% holdout |
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
- [x] `normalize/`: Telex, teencode, bỏ dấu
- [x] `timeparse/`: mọi mục rulebook §2, kể cả Tết theo `calendar.csv`
- [x] `link/`: fuzzy match catalog
- [x] `execute/`: viết op theo thứ tự tần suất (doanh thu → số đơn → SL hàng → tăng trưởng → huỷ → argmax ngày → AOV → tỉ trọng → ca → top khách → bán chạy → giá → tồn → nhập → nợ → lãi → ngành)
- [x] `parse/rules.py`: rule parser. **Mục tiêu PA dev ≥ 0.85**

**04–06/11 — Thứ tự lại sau public 0.066 / 0.076 (D10, EXPERIMENTS)**
Holdout 0.86 nhưng public 0.08: holdout cùng ngành với dev nên không đo được tổng quát hoá. Câu sai trên public phần lớn ở câu diễn đạt quen, nên nghi ingest hoặc ngữ nghĩa trên shop lạ. **Đánh giá ingest và parser tách riêng.** Không tối ưu thêm precision holdout.
1. - [x] Thước đo proxy shop lạ (`eval/synth/`): biến đổi shop dev (đổi tên cột, gộp/tách file, trạng thái vào Ghi chú, đổi đơn vị tiền, bỏ mã, đổi định dạng ngày). Hai chỉ số: **ingest** (so Store) và **end-to-end** (đáp án dev đúng vẫn phải đúng)
   - Lệnh: `eval/synth/run.py --transforms <t> [--shops ...] [--no-e2e] --verbose` (chạy foreground từng transform: máy thiếu RAM). Đủ 24 shop một lần quá 10 phút → chạy 3 nhóm 8 shop. Shop sinh ra được cache (`out/<t>/.stamps`, khoá theo hash của make.py + reader/readme); lần đầu `make()` chậm (openpyxl không có lxml), file không đổi thì copy nguyên bản; regression shop gốc: `eval/synth/fp_snapshot.py diff eval/synth/out/fp_base.json`
   - Lần đo đầu (05/11, đủ 24 shop): rename 19/24 (PA 0.985→0.690, 275 đúng→sai), nodoc 17/24 (0.716), status_note 23/24 (0.933), layout/values 24/24
2. - [ ] Ingest bền + bất biến lúc chạy. Số chốt 05/11 tối (đủ 24 shop + e2e, PA gốc 0.985): **rename, nodoc, layout, values, status_note, combo đều 24/24, PA 0.985, 0 câu mất**; rename_hard 21/24 (chỉ ingest). Trước khi sửa viết tắt: rename 21/24 (0.773), combo 20/24 (0.695)
   - [x] README theo sheet; suy vai trò cột thiếu mô tả; 'Tổng tiền' dòng = SL×giá; bảng bán ≠ bảng giá; 'Số phiếu' bán ≠ phiếu nhập; không có bảng đơn thì abstain
   - [x] Chạy lại đủ 6 transform × 24 shop có e2e để chốt số (EXPERIMENTS 05/11)
   - [x] status_note biz_120: lỗi transform: README synth vẫn ghi Ghi chú là 'cột trống, không có dữ liệu' nên ingest bỏ qua cột đúng theo README. Sửa `t_status_note`: chỉ sửa mục README của bảng bị bỏ cột trạng thái, mô tả Ghi chú thành 'ghi chú'
   - [x] combo biz_105: `with_time`/`rev_by_month` lệch chỉ khi dùng tên tự chế; combo với rename thực tế 24/24
   - [x] Tách `rename` (viết tắt thực tế: SL, ĐG, TT, CK, KH, ĐVT, PTTT, HĐ, HSD…) và `rename_hard` (tên tự chế cũ). Ingest: `vocab.ABBREVIATIONS` mở rộng viết tắt khi tên lạ (nghĩa mơ hồ thử theo thứ tự: TT = thành tiền > thanh toán > trạng thái); tên học được chung chung ('KH' → name) nhường cho bản mở rộng cụ thể. Shop không mô tả 106/109/124: rename 0/3 → 3/3
   - [ ] (thấp) rename_hard 106/109/124 (`TThai`, `GChu`, `Tg.Tien.Hang`): tên không có thật, chỉ sửa nếu public cho thấy kiểu tên này
   - [ ] Bất biến lúc chạy (đối soát dòng/đơn, phân bố trạng thái) → abstain cả shop
   - [x] Sinh lại public (`eval/runs/public_v0.4`, 57d2d2f): vẫn **107 câu** abstain do không có bảng đơn, cùng 2 shop như v0.2/v0.3 (số 214 cũ đếm trùng). Loại câu trả lời y hệt v0.3 (abstain 690), 21 câu đổi giá trị. Các sửa ingest từ proxy chưa chạm tới 2 shop này
   - (gốc) Ingest bền + bất biến lúc chạy (đối soát dòng/đơn, phân bố trạng thái, khoảng ngày); shop không qua thì abstain. Sửa tổng quát lỗi không nhận ra bảng đơn hàng
3. - [ ] Luật đọc từ rulebook lúc chạy (xác minh D03 trên forum trước)
4. - [ ] SLM parser + ensemble gate (mục LLM bên dưới), thay gate từ lạ
5. - [ ] Code-gen fallback để cuối
- [ ] Song song: Dockerfile bản rule-only trước 10/11 (dự phòng)

**08–10/11 — LLM** (cần GPU)
- [ ] Chốt model ≤ 15B, ghi vào `DECISIONS.md`
- [ ] vLLM + guided JSON, prompt có few-shot từ dev
- [ ] Ensemble + Confidence Gate, calibrate. **Mục tiêu PA holdout ≥ 0.85, precision ≥ 97%**

**10–12/11 — Mở rộng**
- [ ] Fallback code-gen cho câu OOD
- [ ] Luật ngành bổ sung, đọc rulebook lúc chạy
- [ ] Dữ liệu tự sinh: paraphrase có lỗi gõ (shop tổng hợp đổi schema: kéo lên bước 1 ở trên)

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
