# BENCHMARK — Agent Generalization trên shop tổng hợp (E0, D24)

> **Trạng thái 11/11: bản nháp, chưa chạy.** Code và test đã có, nhưng `bench_cells.json` chưa build và mục D24 chưa
> ghi vào `DECISIONS.md`. Muốn dùng: ghi D24, chạy `bench.py build` một lần (chỉ shop dev), commit `bench_cells.json`.

Mục đích: đo agent code-gen trên shop có schema lạ **mà không dùng public** (D23) và **không tuning trên holdout**.
Code: `eval/synth/families.py` (biến đổi), `eval/synth/bench.py` (build / materialize / ingest / run / score).
Cấu hình chia tập: `eval/synth/splits_gen.json`. Bộ cell đã đóng băng: `eval/synth/bench_cells.json` (chỉ id + nhãn).

## 1. Đơn vị đo

**Cell = (câu hỏi train, biến thể).** Mỗi câu được chọn có **hai** cell: `original` (dữ liệu và rulebook gốc) và
**một** cell biến thể. Vì vậy mọi nhóm biến đổi đều so được với bản gốc trên **cùng câu hỏi**. Mọi so sánh giữa hai
cấu hình (arm) là so sánh **cặp**: cùng cell, cùng dữ liệu, cùng rulebook, cùng model và cùng tham số sinh.

Chọn câu theo từng shop (seed cố định):
- **8 câu nền/shop**: 3 câu đầu theo id (trùng 72 câu dev Kaggle v3, có thể dùng lại trace) + 5 câu rải đều theo
  kiểu đáp án. Mỗi câu nền được gán cho nhóm biến đổi "luôn áp dụng" đang ít cell nhất của side đó.
- **Câu nhắm đích (≤ 2/shop/nhóm)** cho ba nhóm chỉ tác động lên một số câu: `drop_table`, `rule_param`, `rule_exception`.

## 2. Nhóm biến đổi và nhãn

| Nhóm | Family | Side A (Synth-Tune) | Side B (Synth-Eval) | Nhãn |
|---|---|---|---|---|
| G1 đổi tên | `rename_cols` | viết tắt thật (SL, ĐG, KH, PTTT) | tên tự chế (Tg.Tien.Hang, TThai) | same |
| G1 | `rename_tables` | `BangKe_01.xlsx`, sheet `Trang1` | `export_a.xlsx`, sheet `tab_1` | same |
| G2 thứ tự/layout | `layout` | đảo cột + dòng tiêu đề + csv↔xlsx | đảo cột + cột trống + header lặp mỗi 400 dòng + csv↔xlsx | same |
| G3 định dạng | `values` | `dd/mm/yyyy`, `1.250.000` | `d/m/yyyy H:MM`, `1,250,000đ` | same |
| G4 thiếu metadata | `nodoc` | README bỏ mô tả cột | README chỉ còn tiêu đề + đoạn đầu | same |
| G5 cấu trúc | `split_months` | bảng đơn thành mỗi tháng một sheet `T6-2025` | sheet `Thang 06.2025` | same |
| G5 | `status_codes` | trạng thái ghi bằng mã `1,2,3` + chú giải README | mã `TT01…` | same |
| G5 | `noise_table` | thêm bảng lịch trực nhân viên | thêm bảng nhiệt độ kho | same |
| G5 | `drop_table` | bỏ hẳn nhóm bảng (nhập hàng / tồn kho / lô) mà câu cần | như A | **unanswerable** |
| G6 ngữ nghĩa | `status_newword` | đơn huỷ ghi "Bom hàng" + README giải thích | "Khách không nhận hàng" | same |
| G6 | `rule_param` | rulebook §2.12 ca sáng 07:00–12:59, ca chiều 13:00–20:59 | §2.6 tuần bắt đầu Chủ nhật + §2.11 Tết 25 Chạp → mùng 7 | **recomputed** |
| G6 | `rule_exception` | ngoại lệ ở §1.5 (số đơn), §1.6 (doanh thu) | ngoại lệ ở §3.3 (bán chạy), §4.2 (nợ) | **changed** |

- **same**: biến đổi giữ nghĩa, gold train vẫn đúng.
- **unanswerable**: gold là `unanswerable`. Câu được chọn là câu mà intent (rule parser trên bản gốc) cần nhóm bảng
  đã bỏ (`ingest/checks.py::OP_AREAS`). Build kiểm thêm: Store dựng từ biến thể không còn frame của nhóm đó.
- **recomputed**: gold tính lại bằng rule engine deterministic với rulebook đã đổi (`rulebook.PARAMS`). Chỉ lấy câu mà
  rule engine trả **đúng** trên bản gốc (đáng tin) và đáp án **đổi** khi luật đổi. Gold lưu ở
  `eval/synth/out/bench/gold_recomputed.json` (gitignored).
- **changed**: chưa có cách tính gold mới, nên **không đưa vào PA**. Chỉ đo `stale` = trả đúng đáp án cũ (bỏ qua luật
  mới, chắc chắn sai), `abstain/unans` và `other`.

## 3. Chia tập chống leakage

- **Đơn vị chia là shop gốc.** Mọi biến thể của một shop nằm cùng một side. A = 16 shop, B = 8 shop, không giao nhau,
  không có shop holdout (build tự kiểm và dừng nếu sai).
- **B lấy một shop cho mỗi kiểu xuất.** `biz_105` (sổ ghi tay, mỗi tháng một sheet) không có kiểu tương ứng bên A
  (leave-style-out). Sáu shop có câu dev bị xem ngày 11/11 (101, 106, 118, 119, 121, 124) nằm ở A.
- **Tham số biến đổi khác nhau giữa A và B** (bảng §2): từ điển đổi tên, định dạng, mã trạng thái, từ mới, hàng
  rulebook bị đổi. Bản sửa học thuộc từ điển của A sẽ không có lợi trên B.
- **Quyền xem:** A được đọc trace và từng câu sai (`score --errors`). B chỉ có số tổng (`--errors` không in B), giống
  luật holdout. Holdout không tạo biến thể và chỉ dùng cho regression khi chốt (E6). Public không dùng (D23).
- **Đóng băng:** `gen_hash` gồm code sinh biến thể, `splits_gen.json` và reader/readme của ingest. `run` từ chối chạy
  khi hash đổi. `build` từ chối ghi đè `bench_cells.json`; `build --rebuild` chỉ dùng kèm một mục DECISIONS mới.
  `materialize` dựng lại biến thể (ví dụ trên Kaggle) và kiểm từng biến thể bằng fingerprint Store đã đóng băng.
- **Giới hạn đã biết:** 24 shop dev đã được dùng để phát triển ingest/rule trước E0, nên B **không** phải shop lạ thật.
  B chỉ bảo vệ việc tuning từ E0 trở đi, và các biến đổi của B chưa từng được nhìn thấy.

## 4. Định nghĩa metric (`bench.py score`)

Trên cell có gold (same / unanswerable / recomputed), với R = đúng, W = sai, A = abstain, N = R + W + A:

| Metric | Định nghĩa |
|---|---|
| **PA** | (R − W) / N. So arm với baseline luôn bằng **ΔPA cặp** trên các cell cả hai lần chạy cùng có |
| **Precision** | R / (R + W). **Luôn báo kèm coverage** |
| **Coverage** | (R + W) / N |
| **Answerable recall** | R / số cell có gold ≠ unanswerable. Chặn việc tăng precision bằng cách abstain bừa |
| **Forced-answer rate** | Trong các cell gold = `unanswerable` (gồm `drop_table` và câu unanswerable của train), tỉ lệ cell mà hệ thống vẫn trả **một đáp án cụ thể** (kiểu khác `abstain` và `unanswerable`). Trả `unanswerable` = đúng, abstain = không phạt, trả số/chuỗi = cố trả lời |
| **Stale rate** (changed) | Tỉ lệ cell `rule_exception` mà đáp án vẫn khớp gold cũ (theo scorer) |
| **Flip rate** | Chỉ xét cell nhãn `same`, ghép với cell `original` của cùng câu. **Flip** = hai đáp án cuối **khác nhau về ngữ nghĩa** theo scorer: abstain ≠ trả lời, unanswerable ≠ abstain, hai đáp án cụ thể bằng nhau nếu một bên chấm đúng khi lấy bên kia làm gold (làm tròn, hoa thường, khoảng trắng). Code hay output khác hình thức nhưng cùng đáp án **không** tính là flip. Báo kèm bảng chuyển trạng thái điểm `orig>variant` (`*` = cùng điểm nhưng khác đáp án, ví dụ hai đáp án sai khác nhau) |
| **Runtime error** | % mẫu code-gen không chạy được: `column` (KeyError, cột không có), `helper_hallucinated` (gọi H.* không tồn tại), `helper_signature` (gọi helper sai tham số), `name`, `timeout`, `no_code`, `other:<Exception>` |
| **Semantic error** | % mẫu **chạy được** nhưng sai: `near_miss` (lệch ≤ 5%: luật lọc trạng thái, trả hàng, giảm giá), `unit_factor` (lệch bội số nguyên ≥ 2: đơn vị cơ sở/hộp, đếm trùng), `far`, `type`, `set`/`order` (list), `value`, `false_unanswerable` |
| **Consensus wrong** | % cell trả lời sai mà nguồn là code-gen (các mẫu đã đồng thuận đủ m phiếu) |
| **Runtime** | Giây/câu, **median và p95**. `contract_s`: từ lúc câu vào batch tới khi có đáp án (định nghĩa Docker Guide). `amortised_s` = rule + prompt + gen (thời gian sinh của batch chia đều cho số câu) + validate (trích code) + exec (sandbox, tổng k mẫu) + vote. Báo từng giai đoạn |
| **Ingest fidelity** | (`bench.py ingest`) % trường fingerprint Store khớp bản gốc, theo family. Đo riêng với agent |

**Khoảng tin cậy:** 95%, **bootstrap theo shop** (lấy lại shop có hoàn lại, 2000 lần, seed cố định), cho PA,
precision, coverage, answerable recall và ΔPA cặp. Mỗi dòng báo cáo luôn ghi **số shop và số cell**.

**Báo cáo trade-off trung thực:** không đạt một mục tiêu bằng cách giảm coverage. Mọi kết quả báo đủ bộ
PA, precision, coverage và answerable recall. Một thay đổi tăng precision nhưng giảm answerable recall phải ghi rõ.

## 5. Luật nhận một thay đổi (đăng ký trước E2)

Trên **side B**, so cặp với baseline:
- ΔPA(variants) ≥ +0.03 và cận dưới CI 95% > −0.01.
- Precision không giảm. Answerable recall không giảm quá 0.02 (nếu giảm hơn thì phải giải thích bằng forced-answer
  rate hoặc stale rate giảm).
- ΔPA(original) không âm quá 0.02 (không đổi tổng quát hoá lấy điểm bản gốc).
- Runtime p95 không vượt ngân sách (≤ 30 s/câu trên 2× 5090).
- Holdout (E6) không giảm quá 0.02.

Mục tiêu định hướng (không phải cam kết): precision B ≥ 0.85, forced-answer ≤ 20%, flip ≤ 5%, lỗi runtime/mẫu ≤ 3%.

## 6. Lệnh

```bash
PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py materialize         # dựng lại biến thể đã đóng băng
PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py ingest
PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py run --arm rule_only --agent off            # CPU, kiểm đường đi
PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py run --arm base --side A --agent codegen --llm vllm --llm-model <dir>
PYTHONPATH=src .venv/Scripts/python eval/synth/bench.py score eval/runs/<A> eval/runs/<B> [--baseline <dirs>] [--errors 30]
```
Có thể chạy theo shard bằng `--side/--families/--shops`. `score` gộp nhiều thư mục run của cùng một arm.
