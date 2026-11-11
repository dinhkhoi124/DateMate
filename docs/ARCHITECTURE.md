# ARCHITECTURE — DataMate agent

Nguyên tắc: **LLM dịch, code tính, cổng tự tin quyết.** LLM nhỏ (≤ 15B) không tin cậy khi tính số chính xác, nhưng đủ tốt để chuyển câu hỏi thành intent có cấu trúc. Mọi phép tính đều chạy bằng code deterministic, cài đúng rulebook.

## 1. Pipeline

```
question + as_of_date + business_id
        │
 [1] normalize   bỏ Telex/teencode, bỏ dấu song song (giữ bản gốc để lấy tên)
        │
 [2] parse       (a) rule parser (regex/template)                      ┐ ensemble
        │        (b) LLM parser → Intent JSON (grammar-constrained, temp 0)  ┘
        │
 [3] link        tên hàng/khách/nhóm/NV → mã trong catalog shop (fuzzy, bỏ dấu)
        │
 [4] timeparse   "tuần rồi", "tháng này vs cùng kỳ", "Tết 2026", "quý Q" → [start, end]
        │
 [5] execute     op deterministic trên Canonical Store, theo rulebook
        │
 [6] answer      format đúng type, làm tròn, thứ tự, tie-break theo mã
        │
 [7] gate        answer | unanswerable | abstain
                 ▲
 [0] ingest      (1 lần/shop) README parser + reader csv/xlsx/sqlite → Canonical Store
                 + Capability profile + Validation

 Fallback (OOD): LLM viết pandas trên Canonical Store, có rulebook trong context
                 → chỉ trả lời khi k lần chạy cho cùng kết quả, ngược lại abstain
```

## 2. Module (`src/datamate/`)

| Module | Trách nhiệm | Input → Output |
|---|---|---|
| `ingest/` | Đọc mọi file của shop, map cột theo mô tả trong README, chuẩn hoá về schema chung, profile capability, validation | thư mục shop → `Store` |
| `normalize/` | Chuẩn hoá văn bản: Telex (`tieenf` → `tiền`), teencode (`dthu`, `bn`, `hqua`), gõ lặp chữ, bỏ dấu | str → str (+ bản bỏ dấu) |
| `timeparse/` | Biến cách nói thời gian thành khoảng ngày theo `as_of_date` và `calendar.csv` | text, as_of → `Period(start, end, kind)` |
| `parse/` | Định nghĩa Intent schema, rule parser, LLM parser, ensemble | text → `Intent` + điểm tin cậy |
| `link/` | Khớp thực thể trong câu hỏi với catalog của shop | mention, Store → mã + điểm khớp |
| `execute/` | Thư viện op theo rulebook: revenue, orders, ranking, debt, stock, import, cost, industry ops | Intent, Store → giá trị thô |
| `answer/` | Formatter theo type và Confidence Gate | giá trị, Intent → dòng submission |
| `run.py` | Entrypoint: đọc `/data`, ghi stream `/result/submission.jsonl`, có timeout mỗi câu | — |

## 3. Canonical Store (schema chuẩn)

| Bảng | Cột chính |
|---|---|
| `orders` | order_id, ordered_at (datetime), customer_id, customer_name, status ∈ {done, cancelled, returned, shipping}, channel, payment_method, staff_id, order_discount, shipping_fee, order_total, paid, debt |
| `order_lines` | order_id, line_no, product_id, product_name, qty, unit, qty_base, unit_price, line_discount, amount, cost_per_unit |
| `products` | product_id, name, category, base_unit, cost, price, is_rx, active_ingredient, size, brand… |
| `units` | product_id, unit, factor (về đơn vị cơ bản) |
| `customers` | customer_id, name, phone, group, address, gender, registered_at |
| `payments` | order_id, paid_at, amount, method |
| `imports` / `import_lines` | receipt_id, date, supplier, status, total / product_id, qty, unit, price, amount, lot, expiry |
| `stock`, `stock_ledger` | tồn theo snapshot / theo phát sinh |
| `price_history` | product_id, effective_date, price |
| `expenses`, `deliveries`, `lots`, `industry_*` | sổ chi phí, đợt giao, lô, sửa xe / bảo hành / thu máy cũ / trả góp |

**Capability profile:** `{has_stock, has_imports, has_lots, has_customers, has_debt, has_cost, has_channel, has_gender, has_address, date_min, date_max, …}`. Đây là cơ sở để trả `unanswerable` deterministic.

**Validation:** kiểm tra Σ thành tiền dòng = tiền hàng của đơn, `qty_base = qty × factor`, ngày nằm trong khoảng hợp lý. Shop nào lệch thì cảnh báo và hạ độ tin cậy.

## 4. Intent schema
```json
{"metric": "revenue|order_count|qty|aov|cancel_rate|debt|stock|...",
 "op": "value|rank|compare|growth|share|argmax|list|check",
 "period": {"kind": "day|range|week|month|quarter|ytd|last_n|tet|...", "args": {}},
 "compare_period": null,
 "filters": {"product": null, "category": null, "customer": null, "staff": null,
             "payment": null, "channel": null, "shift": null, "size": null, "unit": null},
 "rank": {"k": null, "by": null, "order": "desc"},
 "threshold": null,
 "answer_type": "number|percent|date|string|list|table|boolean"}
```

## 5. Confidence Gate
Chỉ trả lời khi **tất cả** điều kiện đúng:
1. Rule parser và LLM parser cho cùng intent, hoặc LLM self-consistency đạt k/k.
2. Executor hỗ trợ intent đó.
3. Entity link khớp cao và không mơ hồ.
4. Validation của shop đạt.

Trả `unanswerable` khi capability thiếu, kỳ hỏi nằm hoàn toàn ngoài khoảng dữ liệu, hoặc câu hỏi chủ quan/cần kiến thức ngoài. Các trường hợp còn lại thì `abstain`. Ngưỡng được calibrate trên holdout.

## 6. Model & serving
- Ứng viên: dòng Qwen3 8B/14B hoặc Qwen2.5-Coder-14B (≤ 15B, Apache-2.0). Phải xác minh ngày public và license, ghi vào `DECISIONS.md`.
- Serving bằng vLLM với guided JSON (xgrammar), temperature 0, seed cố định. Bản 14B BF16 khoảng 28GB vừa 1 GPU 5090.
- Chỉ dùng 1 model, không dùng embedding/reranker; entity linking làm bằng rapidfuzz.

## 7. Docker
- Base: CUDA ≥ 12.8 + vLLM, đóng gói sẵn weights, cache sẵn `pip`. Chạy `--network none`.
- Entrypoint `python -m datamate.run --data /data --out /result/submission.jsonl`.
  - Ingest từng shop rồi cache.
  - Parse theo batch.
  - Ghi từng dòng và gọi `flush()`.
  - Câu nào lỗi hoặc quá thời gian thì ghi `abstain`.
- Deterministic: cố định phiên bản thư viện, sắp xếp ổn định, không phụ thuộc thứ tự hash.

## 8. Agent v2: sinh code là đường chính (D20, 09/11)

```
câu hỏi + as_of_date + business_id
  │
  ├─ [ingest] Store của shop (giữ nguyên tầng 0)
  ├─ [context] rulebook của split (nguyên văn, đọc lúc chạy; prefix cache)
  │            + README shop + schema bảng (cột, kiểu, vài dòng mẫu)
  │            + mô tả API helper
  ├─ [LLM ≤15B] sinh `def solve(T, H):` (pandas) → trả {"type", "answer"}
  ├─ [sandbox] chạy trong process riêng, timeout, chỉ đọc bảng; lỗi → mẫu đó bỏ
  ├─ [vote] k mẫu; trả lời khi ≥ m mẫu cùng đáp án đã chuẩn hoá §9, ngược lại abstain
  ├─ [rule] đường nhanh: rule parser + executor khi dòng luật liên quan không đổi (D18) và qua gate
  │         rule ≠ code-gen → abstain (chính sách calibrate)
  └─ [log] thinking (output thô của model, list theo mẫu), code, output (bắt buộc theo Docker Guide)
```

- **Helper `H`** (code deterministic, test sẵn): `H.period(text)` → (start, end) theo `timeparse` + `calendar.csv`;
  `H.link(kind, mention)` → mã/tên trong catalog; `H.money/H.pct` làm tròn §9; `H.as_of`.
- **Bảng `T`**: DataFrame của Store chuẩn (orders, order_lines, products…) kèm bảng thô theo file khi Store thiếu cột.
- Model không tự ra số: mọi con số là kết quả chạy code (CLAUDE.md luật 4).
