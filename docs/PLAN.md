# PLAN — Roadmap & checklist

Freeze Docker: **16/11 23:59** · Mốc an toàn: **12/11 phải có image chạy được** · **Private: đội tự chạy 20/11 09:00 → 23/11 23:59, cần GPU** (D16) · Chi tiết thiết kế: `ARCHITECTURE.md` · Đề bài: `SPEC.md`

## Mục tiêu đo được

| Chỉ số | Cách đo | Ngưỡng |
|---|---|---|
| **PA public** (từ 07/11, D16) | nộp phiên bản hệ thống hoàn chỉnh mỗi milestone (≤ 2 lượt/ngày), loại model "<=15B". **Thước đo khả thi chính**. Không nộp bản chỉ khác ở một nhóm câu nhỏ (Rule 6) | tăng qua mỗi milestone; precision ước tính ≥ 90% |
| **PA proxy shop lạ** (từ 04/11) | shop tổng hợp từ train, đổi schema/cách ghi (`eval/synth/`). Thước đo để ra quyết định hằng ngày (không tốn lượt nộp) | tăng đều; precision ≥ 90% |
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
   - [x] Bất biến lúc chạy (D11, `ingest/checks.py`, ngưỡng `config.Invariants`): vi phạm gắn vùng dữ liệu, op chạm vùng đó thì abstain. Đo bằng `eval/synth/faults.py` (6 lỗi giả lập × 24 shop): ok→wrong 7345 → **0**, shop sạch không đổi; dev 0.985, holdout 0.859 không đổi. Public v0.5: 2/14 shop `status_unrecognized` → 100 câu thành abstain
   - [x] Từ vựng trạng thái tổng quát + transform `status_words` (nhãn phần mềm khác + nhiễu hoa/thường/bỏ dấu; có nhãn mơ hồ 'Chuyển hoàn' phải abstain). Trước: 24/24 shop bị chặn (0 sai). Sau: ingest 22/24, PA 0.985 → 0.900, **0 sai**; 2 shop còn lại là nhãn mơ hồ, abstain đúng thiết kế. Public v0.6: vẫn 2 shop `status_unrecognized` (không đọc giá trị ô, D11)
   - [ ] (giữ nguyên, có lý do) `status_unrecognized` chặn cả vùng khi có giá trị lạ: chấm exact match, một đơn huỷ đọc thành hoàn thành đã làm sai số đếm/doanh thu; nới theo tỉ lệ sẽ sinh câu sai
   - [x] Docker: cache Store/raw ghi vào `$DATAMATE_CACHE` (`config.cache_root()`, `predict.sh` đặt `/tmp/datamate_cache`; SPEC §7 câu 2); log `/result/logs/log.jsonl` mỗi câu {id, answer, elapsed_ms, thinking=null, code, inputs, output} stream cùng submission (SPEC §7 câu 3, D12). Dev không đổi (2410/0)
   - [x] Sinh lại public (`eval/runs/public_v0.4`, 57d2d2f): vẫn **107 câu** abstain do không có bảng đơn, cùng 2 shop như v0.2/v0.3 (số 214 cũ đếm trùng). Loại câu trả lời y hệt v0.3 (abstain 690), 21 câu đổi giá trị. Các sửa ingest từ proxy chưa chạm tới 2 shop này
   - (gốc) Ingest bền + bất biến lúc chạy (đối soát dòng/đơn, phân bố trạng thái, khoảng ngày); shop không qua thì abstain. Sửa tổng quát lỗi không nhận ra bảng đơn hàng
3. - [ ] **Ưu tiên ngang LLM (D16): đọc rulebook lúc chạy.** Đề yêu cầu dùng rulebook của đúng split (SPEC §7.1). Không xem rulebook public để định hướng code
   - [x] Liệt kê tham số luật đang đặt cứng từ rulebook train, mỗi luật gắn mục rulebook và op chịu ảnh hưởng: `docs/RULEBOOK_PARAMS.md` (07/11). Đọc lúc chạy mới 4 tham số (ca, Tết); P✗: đầu tuần, phí phục vụ, size, quá/từ N, N ngày, tuổi nợ, làm tròn. Đề xuất đối chiếu rulebook theo dòng (hash), luật định nghĩa đổi thì abstain op liên quan: chờ duyệt
   - [x] (07/11, D18) `datamate/rulebook.py`: tách rulebook thành 75 dòng luật, so hash với `rulebook_index.json` (sinh từ train bằng `analysis/build_rulebook_index.py`, không chứa chữ). Tham số (ca, Tết, đầu tuần) đọc lại khi chỉ đổi số; định nghĩa đổi/mất → chặn phạm vi (op/filter/loại kỳ); dòng mới → từ khoá chưa có trong train. Log chỉ ghi số đếm
   - [x] Rulebook tổng hợp trên dev: train → giống hệt (dev 2410/0/37, holdout 485/10/58 byte-identical); S1 đổi ca + đầu tuần → 113 câu đổi đáp án, đúng 68 câu kỳ tuần + 45 câu ca, 0 câu khác; S2 đổi §1.1 → 0 sai, 2005 abstain. 11 test mới (166 pass)
   - [ ] **Public v0.8 với D18: 995/1000 abstain** (3 dòng đổi, 3 dòng mất, 25 dòng mới, 42 phạm vi bị chặn). Chờ quyết định có nộp/nới không (xem EXPERIMENTS 07/11)
   - [ ] Nộp public bản có đọc rulebook (milestone)
4. - [ ] SLM parser + ensemble gate (mục LLM bên dưới), thay gate từ lạ. Code CPU xong 07/11 (D14), còn chạy model
5. - [ ] Code-gen fallback để cuối
- [x] Song song: Dockerfile bản rule-only trước 10/11 (dự phòng)
   - [x] `docker/Dockerfile` (python:3.11-slim, deps pin `docker/requirements.txt`), `docker/predict.sh` → `/code/predict.sh`, `/code/training_code/data_manifest.json` (`docker/make_manifest.py`), `.dockerignore`. Image chứa code + `data/train/train`. 06/11: 142 test pass; giả lập `predict.sh` local (`DATAMATE_CACHE` → thư mục tạm) ghi đúng submission + `logs/log.jsonl`, cache không rơi vào repo
   - [x] Build + chạy thử `--network none --shm-size=16g -v <split>:/data:ro -v <out>:/result` (06/11, sau khi sửa PATH WSL): image 1.08GB, train 3000 câu trong 386s (CPU, gồm ingest nguội), submission container **giống hệt local 3000/3000**, `submit_check` PASS, log 3000 dòng. Chưa chạy với `--gpus` (máy không có GPU NVIDIA)

**07/11 — Baseline public (D16)**
- [ ] Sinh lại public từ HEAD (LLM tắt), so với `public_v0.6`, `/submit-check`, nộp với loại "<=15B", ghi `EXPERIMENTS.md`

**08–10/11 — LLM** (cần GPU)
- [x] (07/11, CPU) Phần không cần GPU (D14): contract `parse/llm_schema.py` (op = handler executor + `none`, kỳ lấy từ timeparse, mention link bằng Dictionary), backend `parse/llm.py` (vLLM offline guided JSON / replay jsonl ghi-đọc), ensemble `parse/ensemble.py` + `config.LLMConfig`, cờ `--llm/--llm-model/--llm-replay` cho `datamate.run` và `eval/run_eval.py` (`--by source`). Round-trip Intent → JSON → Intent **2410/2410** câu dev đúng. Backend replay rỗng: dev 2410/0/37 không đổi. 153 test pass
- [x] Few-shot từ dev (`analysis/build_fewshot.py`): 81 câu, 41 op, prompt ~16.7k ký tự (~7k token). 6 op chưa có câu dev đúng: import_minus_sales, stock_out_count, net_profit, delivery_count, nearest_expiry, expiring_lots_count
- [x] (07/11) Quy trình GPU qua máy thành viên (D15): `docs/GPU_RUNBOOK.md`, `scripts/gpu/run_all.sh`, `eval/gpu_smoke.py`, chế độ ghi `DATAMATE_LLM_CALL_ALL`. Chặn lộ dữ liệu: gitignore/pre-commit/CI chặn mọi `.jsonl`, hook `eval/check_leak.py`, viết lại 5 câu test trùng train. Rà runbook lần 2: dòng 5090/FP8, AWQ fp16, RAM WSL, pin deps, đếm lỗi model trong summary
- [ ] Thành viên GPU chạy runbook (hạn 10/11): smoke + dev + holdout + public cho Qwen3-8B và 14B, bundle lên Drive, PR `docs/gpu_runs/`. Kiểm tra guided JSON trên bản vLLM đó, bản đó thành bản pin
- [ ] (CPU) Chốt model ≤ 15B từ replay: độ khớp signature với rule trên dev, thời gian/câu, VRAM; ghi `DECISIONS.md`. 14B BF16 cần TP=2 hoặc FP8 trên 5090 (D15)
- [x] (CPU, 07/11) Thêm cờ `--llm/--llm-model/--llm-replay` cho `eval/synth/run.py` (câu synth = câu dev nên replay dev dùng được; in `llm_calls/llm_failed` = replay miss). Replay rỗng: rename 2 shop giống hệt rule-only
- [ ] (CPU, khi có replay) Đo proxy synth bằng replay dev: `eval/synth/run.py --llm replay --llm-model models/<M> --llm-replay <dev replay>`
- [ ] Calibrate ensemble trên replay (CPU): `rescue_gate` (mặc định bật), `require_agreement`, `llm_only` + `n_samples`. **Mục tiêu PA holdout ≥ 0.85, precision ≥ 97%**
- [ ] Image LLM: base CUDA ≥ 12.8 + vLLM pin, weights trong image, so container với local như bản rule-only

**11/11 — Go/no-go 1 (D16)**
- [ ] So PA public các milestone (baseline, rulebook, LLM). Ngưỡng chốt sau khi có baseline. Không đạt thì dừng LLM, dồn sức cho đọc rulebook và ingest, nộp image rule-only

**10–12/11 — Mở rộng**
- [ ] Fallback code-gen cho câu OOD (Rule 5 cho phép; ngân sách ~40s/câu). Chỉ làm sau khi rulebook + LLM xong
- [ ] Dữ liệu tự sinh: paraphrase có lỗi gõ (shop tổng hợp đổi schema: kéo lên bước 1 ở trên)

### Phase 3 — Test (song song, chốt 15/11)
- [ ] Test độ bền: shop và rulebook tổng hợp
- [ ] Test thời gian và độ lệch giữa 2 lần chạy
- [ ] `/submit-check` xanh

### Phase 4 — Deploy (12–16/11)
- [ ] Dockerfile (CUDA ≥ 12.8, weights đóng gói sẵn), chạy `--network none`
- [ ] Nộp public (≤ 5 lần/ngày), model type **"Based on small models (<=15B)"** (D15), ghi kết quả vào `EXPERIMENTS.md`
- [ ] Base image theo Docker Guide (`vllm/vllm-openai:v0.10.1.1` hoặc bản khác chạy được trên 5090), tag `zaic2026_datamate_<team>:final`, `ENTRYPOINT []`
- [ ] README bài nộp khai báo model (Rule 2)
- [ ] Upload image lên **Google Drive public** + checksum, thử tải về từ máy khác (image ~30–40 GB: upload trước 15/11)
- [ ] Chạy thử quy trình private từ đầu tới cuối trên public: tải image từ Drive → `docker load` → lệnh BTC → `/submit-check` → so với local
- [ ] **Nộp image cuối trước 16/11 23:59** qua form "Solution Submission"

### Phase 5 — Private (20–23/11)
- [ ] **Đặt GPU trước 16/11** cho 20–23/11 (máy thành viên hoặc thuê 2× 5090): chạy đúng image đã nộp
- [ ] 20/11 09:00 tải private → chạy image → `/submit-check` → upload `submission.jsonl` trước 23/11 23:59. Giữ log để giải trình nếu lệch > 0.03
- [ ] Theo dõi bảng tạm. Top 2 công bố 02/12
- [ ] Public code (quy định 8.2), viết writeup

## Rủi ro

| Rủi ro | Phương án |
|---|---|
| Private có ngành/luật/kiểu file mới | Map README theo mô tả, đọc rulebook lúc chạy, nhánh code-gen dự phòng, abstain mặc định |
| Máy dev không có GPU | Executor và rule parser làm trên CPU. GPU thành viên cho LLM (D15), thuê 5090 cho Docker |
| Không có GPU trong 20–23/11 | Đặt trước 16/11, có phương án thuê dự phòng. Image rule-only chạy được trên CPU làm phương án cuối |
| Rulebook private khác train | Đọc rulebook lúc chạy (bước 3), test bằng rulebook tổng hợp |
| Kết quả không deterministic | temp 0, seed cố định, pin phiên bản, sắp xếp ổn định |
| Chạy chậm | Cache Store, batch LLM, timeout mỗi câu (quá giờ thì abstain) |
| Hiểu sai luật | Mỗi luật có test riêng từ câu train. Báo cáo lỗi theo intent |
