# RESEARCH BACKLOG — Tổng quát hoá agent sinh code trên shop lạ (12/11/2026)

Kế hoạch nghiên cứu lập ngày 11/11 (E0–E6, giải pháp S1–S8). File này ghi **việc nào đã xong, việc nào còn**, cần gì
để làm, và làm xong thì đo bằng gì. **Người làm do PM chỉ định**: không tự nhận việc, không tự đổi `AgentConfig`.

Đọc trước: `docs/BENCHMARK.md` (cách đo, luật nhận §5), `docs/DECISIONS.md` D20–D25, `HANDOVER.md` mục 4–5.

---

## 1. Vì sao có backlog này

- Code-gen một mình: dev **0.653** → holdout **0.444** → public ước precision **~60%** (v2 0.117). Agent trả lời
  nhiều nhưng sai gần ngang đúng trên shop lạ.
- Phân tích 360 mẫu dev (Kaggle v3, reexec `fae7b33`, 11/11):
  - 6% mẫu lỗi runtime: `KeyError` cột 6, helper bịa hoặc gọi sai 4 (`H.find_units` ×2), `&` thiếu ngoặc 3,
    timeout 4…
  - 27% mẫu chạy được nhưng **sai ngữ nghĩa**.
  - 9 câu sai (m=2) **đều là lỗi ngữ nghĩa**: 7 câu lệch 0.2–4% (luật lọc huỷ/trả, cột tiền), 1 câu đếm dòng thay
    vì đếm đợt giao, 1 câu sai đơn vị cơ sở/hộp. Không câu nào sai do tên cột.
  - Lỗi runtime chỉ làm câu bị **abstain** (0 điểm). Thứ làm mất điểm là **đồng thuận sai** (≥ m mẫu cùng ra một
    đáp án sai).
- Agent chỉ thấy các frame chuẩn của ingest (`agent/tables.py`). Cột/bảng không map được bị bỏ, nên lỗi ingest đi
  thẳng vào agent (BENCHMARK §7).

## 2. Ràng buộc thời gian (quan trọng)

- Image LLM build từ `97aee0b` (12/11). **Mọi thay đổi `src/` sau đó đều phải build lại image, chạy lại
  `check_image.sh` hai lần và nộp lại Test Solution.** Test Solution chỉ có 1 lượt/ngày và **đóng 15/11 23:59**.
- Private bắt buộc chạy đúng Docker đã nộp ngày 16/11. **Việc làm sau 16/11 không đổi được điểm cuộc thi**, chỉ có
  giá trị học hỏi và portfolio.
- Vì vậy chia hai nhóm: **Nhóm A** = có thể vào bài nộp, nhưng phải xong và được PM duyệt **trước 14/11 12:00**;
  **Nhóm B** = sau Test Solution hoặc sau 16/11.
- Không việc nào trong file này được chiếm GPU hay thời gian cần cho việc build, kiểm và nộp image (HANDOVER mục 4).

## 3. Trạng thái

| Bước | Nội dung | Trạng thái |
|---|---|---|
| **E0** | Benchmark synthetic 534 cell, chia A/B theo shop, metric, luật nhận | ✅ Xong, đóng băng và niêm phong (D24) |
| **E1** | Phân loại lỗi codegen | 🟡 Một phần: bộ phân loại tự động có trong `bench.py score`; đã phân tích tay 72 câu dev (mục 1); ingest synthetic đã đo trước/sau D25. Chưa chạy trên trace agent của benchmark (cần E2) |
| Ingest trạng thái | Chú giải README, trạng thái `unknown` | ✅ Xong (D25) |
| Ingest `layout` B | Header lặp không bị bỏ khi cột đầu là cột trống | ⏳ Đã tìm nguyên nhân + đề xuất (BENCHMARK §7), chưa sửa |
| Ingest `rename_tables` A | Bảng chi phí bị đọc thành phiếu thu khi tên file trung tính | ⏳ Đã tìm nguyên nhân + đề xuất, chưa sửa |
| **E2** | Baseline agent trên benchmark | ❌ Chưa chạy (GPU) |
| **E3** | S3, S6, S7 trên CPU | ❌ |
| **E4** | S1, S2 (prompt/schema mới, GPU) | ❌ |
| **E5** | S4, S5, S8 (GPU) | ❌ |
| **E6** | Chốt cấu hình, holdout, D-entry | ❌ |
| **D26** | Agent text-to-SQL trên view DuckDB (hướng của baseline BTC). Gồm A3 (kiểm tra hợp lý sau vote) cho đường SQL | 🟡 Code + test CPU xong (nhánh `feat/sql-agent`); chờ Kaggle dev/holdout |

## 4. Nhóm A — có thể vào bài nộp (CPU, rủi ro thấp, hạn 14/11 12:00)

Chỉ làm khi PM đồng ý build lại image LLM. Nếu final là image rule-only thì nhóm A **không ảnh hưởng bài nộp**,
vì image rule-only không chạy agent.

Cách đo chung cho nhóm A, **không cần GPU**: chạy lại code mà model đã sinh với helper/sandbox mới, rồi vote lại.

```bash
# trace dev Kaggle v3: giải nén zip datamate_dev_* trên Drive nhóm (DataMate/) vào eval/runs/kaggle_qwen3_14b_v3/
PYTHONPATH=src .venv/Scripts/python eval/reexec_agent.py eval/runs/kaggle_qwen3_14b_v3/<run>_dev/trace.jsonl
PYTHONPATH=src .venv/Scripts/python eval/calibrate_agent.py eval/runs/kaggle_qwen3_14b_v3/<run>_dev/trace_reexec.jsonl
```

Luật nhận cho nhóm A (dev, 72 câu): **số câu sai không tăng**, PA code-gen một mình (m=2) không giảm so với **0.653
(56/9/7)**, rule-only dev giữ **2410/0/37**. Trace holdout (`ho1`) chỉ được xem **số tổng** (`--totals-only`).

| Mã | Việc | Đích | Gợi ý cách làm | Công |
|---|---|---|---|---|
| **A1** | Lỗi runtime còn lại | `KeyError` cột 4, `find_units` bịa 2, `&` thiếu ngoặc 3 | Xem lỗi bằng `calibrate_agent.py --errors 20`. `&`: lỗi logic của model, sandbox không nên vá; chỉ để S7 bắt. `KeyError`: kiểm tên cột trong schema prompt có khớp `T` không | 0.5 ngày |
| **A2 = S3** | Helper bịa | gọi `H.<tên không có>` | `Helpers.__getattr__` ném lỗi kèm danh sách helper có thật (giúp S4 sau này); thêm alias cho tên hay bị bịa (`find_units` → API đơn vị có sẵn) nếu nghĩa rõ ràng. Test trong `tests/test_agent.py` | 0.5 ngày |
| **A3 = S7** | Kiểm tra hợp lý trước khi trả lời | giảm **đồng thuận sai** | Sau bỏ phiếu, abstain nếu: percent ngoài [0, 100]; đếm (số đơn, số khách…) không phải số nguyên ≥ 0; tiền âm khi câu hỏi doanh thu; kiểu đáp án khác `atype` mà rule parser đoán. Luật đặt trong config, có test. **Bật/tắt bằng cờ mới phải được PM duyệt** (đổi hành vi bài nộp) | 1 ngày |

Không đưa vào nhóm A: bất cứ việc gì đổi prompt (cần GPU để đo) hoặc đổi `AgentConfig` (k, m, policy) mà không có
số trên train và PM duyệt (D21).

## 5. Nhóm B — sau Test Solution / sau 16/11

Thứ tự đề xuất theo lợi/công. Mọi so sánh dùng **benchmark đóng băng**: so cặp với baseline E2, luật nhận BENCHMARK
§5, báo đủ PA, precision, coverage, answerable recall, kèm số shop/cell và CI bootstrap theo shop. Side B chỉ xem số
tổng; mọi phân tích từng câu làm trên side A.

| Mã | Việc | Cần | Ghi chú |
|---|---|---|---|
| **E2** | Baseline agent trên benchmark | GPU, khoảng 12 giờ Kaggle T4 (khoảng 530 câu × 83 s) | `bench.py materialize` → `bench.py run --arm base --agent codegen --llm vllm ...` theo shard (`--side/--families/--shops`) → `bench.py score`. Cần thêm notebook Kaggle cho benchmark (chưa có; dựa trên `scripts/kaggle/datamate_dev.ipynb`) |
| **S6** | Bộ chấm tự tin có hiệu chỉnh | trace E2 (CPU sau đó) | Đặc trưng: độ chênh phiếu, đồng thuận với rule parser, tỉ lệ mẫu lỗi, ingest có cảnh báo (`checks.py`), có cột `unknown`. Hiệu chỉnh trên side A, đánh giá trên side B. Đòn bẩy lớn nhất khi precision chỉ khoảng 60% |
| **I1** | Ingest `layout` B | CPU | So cả dòng với header (≥ 60% ô trùng) thay vì chỉ cột đầu (`reader._apply_header`). Đo: `bench.py ingest`, dev, holdout |
| **I2** | Ingest `rename_tables` A | CPU | Mô tả README có trọng số hơn tên file khi phân loại bảng; điểm sát nhau thì `unknown` thay vì gộp nhầm |
| **S1** | Schema card | GPU (prompt mới) | Mỗi cột: vai trò, kiểu, ví dụ, mô tả README của cột gốc, ánh xạ tên gốc → tên chuẩn |
| **S2** | Mở schema | GPU | Cột chưa map và bảng `extra` hiện trong `T` (`T['raw:<tên>']`). Đo forced-answer rate trên `drop_table` |
| **S4** | Sửa lỗi bằng traceback | GPU | Thử lại một lần khi mẫu lỗi runtime, đưa traceback vào prompt. Theo dõi precision |
| **S5** | Helper ngữ nghĩa | GPU | `H.valid_orders`, `H.sales_lines` theo rulebook của split. **Phải đo riêng trên G6 (`rule_param`, `rule_exception`)**: helper có thể "nướng" sẵn luật train |
| **S8** | Đa dạng mẫu | GPU | k mẫu chia cho 2 cách diễn đạt prompt để lỗi bớt tương quan |
| **Prompt `unknown`** | Agent xử lý trạng thái `unknown` (D25) | GPU | Hiện agent lọc `status != 'cancelled'` sẽ tính nhầm đơn `unknown` |
| **E6** | Chốt | – | Ablation leave-one-out từ bản bật hết, chạy holdout một lần, ghi D-entry |

## 6. Quy trình cho mỗi việc

1. PM chỉ định người làm và nhóm (A hay B).
2. Tạo nhánh từ `main`: `git checkout main && git pull && git checkout -b fix/<việc>`.
3. Làm trên train (dev hoặc side A của benchmark). Không xem public, không mở holdout, không xem chi tiết side B.
4. Đo theo bảng ở trên. Ghi một dòng `docs/EXPERIMENTS.md`; quyết định thiết kế thì ghi `docs/DECISIONS.md`.
5. Mở PR. CI xanh, người kia review, **PM duyệt và merge**. Cập nhật cột trạng thái trong file này.
6. Thay đổi muốn vào image: báo PM **trước 14/11 12:00** để kịp build, kiểm và nộp Test Solution trước 15/11.
