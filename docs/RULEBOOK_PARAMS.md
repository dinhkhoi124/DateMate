# RULEBOOK_PARAMS: luật nào đang đặt cứng, luật nào đọc lúc chạy (D16 bước 3)

Bảng kê từ **rulebook train** (`data/train/train/rulebook.md`, 142 dòng, ~55 dòng luật). Chỉ diễn giải và trích số
mục, không chép nguyên văn (repo public, D15). Không dùng rulebook public để soạn bảng này (SPEC §7.1).

Mã dòng `§m.k` = mục m, dòng thứ k trong bảng của mục đó (`§0` là đoạn mở đầu).

**Loại:**

- **P (tham số):** giá trị đọc được thành số hoặc tên.
  - **P✓**: đã đọc lúc chạy (`config.from_rulebook`).
  - **P✗**: đang dùng mặc định train.
- **D (định nghĩa):** công tắc ngữ nghĩa nằm trong executor hoặc ingest. Đổi chữ thì không đọc thành tham số được.
- **F (định dạng):** cách ghi đáp án, chấm điểm.

## 1. Tham số

| Mã | Nội dung (diễn giải) | Loại | Code | Op chịu ảnh hưởng |
|---|---|---|---|---|
| §2.12 | ca sáng 06:00–13:59, ca chiều 14:00–21:59; mốc so là ca sáng | P✓ | `Rules.morning/afternoon`, `ops.py:92` | shift_compare, mọi op có filter ca |
| §2.11 | Tết: 23 tháng Chạp → hết mùng 10 Giêng; "trước Tết" cùng số ngày | P✓ | `Rules.tet_start/tet_end`, `periods.tet_period` | tet_vs_before, mọi op có kỳ Tết |
| §2.6 | tuần bắt đầu thứ Hai; tuần này / tuần trước | **P✗** | `Rules.week_start`, `periods.week_start` | mọi op có kỳ tuần |
| §3.1 | dòng "Phí phục vụ" không phải mặt hàng nhưng vẫn tính doanh thu | **P✗** | `Rules.service_fee_names`, `facts.py:63,77` | top_products, qty, items_in_n_orders, dead_count |
| §6 cà phê | size theo hậu tố `(M)`/`(L)`, không phân biệt hoa thường | **P✗** (đặt cứng) | filter size | qty, revenue, share có filter size |
| §2.5 | quá/hơn/trên/dưới N không gồm N; "từ N trở lên" gồm N | **P✗** | `timeparse`, parser `inclusive` | days_revenue_over, orders_revenue_over, items_in_n_orders, customers_min_orders(_pct), debt_total quá N ngày |
| §2.4 | N ngày gần nhất gồm hôm nay | **P✗** | `timeparse` | mọi op có kỳ "N ngày" |
| §4.5 | tuổi nợ từ ngày đặt; đặt hôm nay = 0 ngày | **P✗** | executor nợ | debt_aging, debt_total quá N ngày |
| §1.9, §9 | phần trăm, trung bình làm tròn 2 chữ số; tiền làm tròn về đồng (nửa lên) | **P✗** | `answer/` | mọi op ra percent hoặc trung bình |

## 2. Định nghĩa (công tắc trong executor/ingest)

| Mã | Nội dung (diễn giải) | Op chịu ảnh hưởng |
|---|---|---|
| §0 | "hôm nay" = as_of; bản ghi sau hôm nay coi như chưa xảy ra; trạng thái huỷ/trả lấy theo đơn | mọi op |
| §1.1 | đơn tính doanh thu: không huỷ, không trả (trả một phần bỏ cả đơn); đang giao vẫn tính | revenue, aov, share, growth, revenue_series, revenue_by, argmax_*, monotonic, top_products, top_customers, top_staff, days_revenue_over, orders_revenue_over, qty, order_count, gross_profit |
| §1.2 | "bán/thu được bao nhiêu tiền" = doanh thu | parser (từ vựng) |
| §1.3 | đơn thuộc kỳ theo ngày đặt | mọi op theo kỳ |
| §1.4 | huỷ = trạng thái huỷ; trả hàng không phải huỷ | cancel_count, cancel_rate |
| §1.5 | số đơn chỉ tính đơn hợp lệ §1.1; "khách đã mua" | order_count, items_in_n_orders, customers_min_orders(_pct), top_customers (theo số đơn) |
| §1.6 | doanh thu đơn = Σ thành tiền dòng (đã trừ CK dòng), không trừ CK đơn, không VAT, không phí giao | họ revenue, aov, orders_revenue_over |
| §1.7 | theo hình thức thanh toán ghi trên đơn; ghi nợ không vào tiền mặt/CK; COD riêng | revenue_by, share (kênh/thanh toán) |
| §1.8 | nhóm khách theo cột nhóm; vãng lai không thuộc nhóm; mẫu tỉ trọng = cả shop | share, revenue lọc nhóm khách |
| §1.9 | tỉ lệ huỷ: mẫu = mọi đơn đặt trong kỳ, mọi trạng thái | cancel_rate |
| §1.10 | % thay đổi so với kỳ trước, giảm ra âm | growth, tet_vs_before, price_effect, shift_compare |
| §1.11 | chênh lệch = kỳ hỏi − kỳ so | growth (dạng hiệu) |
| §2.1–2.3, 2.7–2.10 | ngày X; đến hết ngày X (tính từ đầu dữ liệu); X–Y gồm hai đầu; tháng; tháng này (so cùng số ngày); từ đầu năm; tháng/quý không ghi năm | `timeparse`: mọi op theo kỳ; stock_*, debt_* (đến hết ngày) |
| §3.2 | nhóm hàng đúng tên, không gộp tên gần | filter category |
| §3.3–3.4 | bán chạy theo SL quy đơn vị cơ bản; "bán được N <đơn vị>" chỉ cộng hàng có đơn vị cơ bản đó | top_products, qty |
| §3.5 | lời nhất theo tổng lãi gộp | top_products (lãi) |
| §3.6 | xác định khách theo mã; không mã thì theo tên (chỉ khác cách viết = một khách); khách lẻ không tính | top_customers, customers_min_orders(_pct), debt_table, debt_customers |
| §3.7 | nhân viên giỏi nhất theo doanh thu | top_staff |
| §3.8 | AOV làm tròn về đồng | aov |
| §4.1–4.4 | nợ: đơn không huỷ (trả hàng vẫn tính); nợ = tổng đơn (sau CK đơn, gồm phí giao) − đã trả đến ngày X; khách lẻ vào tổng, không vào bảng; COD/sàn không phải nợ | debt_total, debt_customers, debt_table, debt_aging |
| §4.6 | nợ thêm / thu nợ trong kỳ | debt_collected, debt_customer_flow |
| §5.1 | tiền nhập theo ngày phiếu, bỏ phiếu huỷ, nhập đầu kỳ là phiếu thường | import_total, top_supplier |
| §5.2 | nhập trừ bán: phần bán theo §1.1, cùng đơn vị | import_minus_sales |
| §5.3 | giá nhập bình quân theo SL nhập | avg_import_price |
| §5.4 | tồn đơn vị lớn = [nguyên, lẻ đơn vị cơ bản] | stock_big |
| §5.5 | giá trị tồn = tồn hết hôm nay × giá vốn trong danh mục | stock_value |
| §5.6 | hết hàng: hàng mua về bán, tồn cuối ngày = 0 | stock_out_count |
| §5.7 | ế: từng bán trước kỳ, trong kỳ không bán | dead_count |
| §5.8 | giá bán ngày D: đơn giá dòng / đơn vị cơ bản, trước CK, gồm đơn huỷ/trả | unit_price |
| §5.9 | hiệu ứng tăng giá: N ngày từ D so N ngày trước D | price_effect |
| §5.10–5.11 | lãi gộp = doanh thu − SL × giá vốn ghi trên dòng bán | gross_profit, top_products (lãi) |
| §5.12 | lãi ròng tháng = lãi gộp − mọi khoản chi phí | net_profit |
| §6 nhà thuốc | kê đơn theo danh mục, tỉ lệ theo doanh thu; hết hạn trong N ngày; hoạt chất ghi đúng như vậy; lô gần hết hạn | share (kê đơn), expiring_lots_count, expiring_products, revenue (hoạt chất), nearest_expiry |
| §6 VLXD | đợt giao: mỗi đợt của mỗi đơn có ngày giao trong kỳ | delivery_count |
| §7 | không trả lời được: kỳ ngoài dữ liệu, shop không ghi loại thông tin | mọi op (`unanswerable`) |
| §8 | điền xuôi nhóm dòng, 4 trạng thái, dòng trùng cùng đơn, ngày không năm | ingest |
| §10 | đơn vị cơ bản / đơn vị lớn, hệ số đọc ở bảng quy đổi | qty, stock_*, import_* |
| §9 | format từng type, thứ tự list, tie-break theo mã | F: `answer/` |

## 3. Đề xuất thiết kế (chưa làm, chờ duyệt)

Tham số P✗ thì thêm vào `from_rulebook` như P✓, có test bằng rulebook tổng hợp. Phần lớn rủi ro lại nằm ở loại D:
rulebook split khác đổi một định nghĩa thì executor vẫn tính theo train, ra **câu sai tự tin** (−1). Đề xuất
**đối chiếu rulebook theo dòng** lúc chạy:

1. Tách rulebook của split thành dòng luật theo cấu trúc train (mục `##`, bảng `| Luật | Mặc định |`).
2. So từng dòng với dòng train cùng khoá. Lưu **hash** của chữ đã chuẩn hoá (bỏ dấu, thường, gộp khoảng trắng), không
   lưu chữ, sinh bằng script từ rulebook train.
   - **Trùng hash:** luật không đổi, chạy như hiện tại.
   - **Khác hash, dòng loại P:** parse tham số. Parse được thì áp dụng, không được thì như loại D.
   - **Khác hash, dòng loại D, hoặc dòng train bị mất:** các op ở cột phải của bảng trên → **abstain** cho cả split, ghi
     lý do vào log.
   - **Dòng mới** (khoá không có trong train), nhất là ngành mới ở §6: lấy từ khoá của dòng đó. Câu hỏi chứa từ khoá
     thì abstain (sau này chuyển sang nhánh LLM/code-gen có rulebook trong context).
3. Đo bằng rulebook tổng hợp trên shop dev:
   - đổi tham số (ca, đầu tuần, Tết): đáp án tính lại phải đúng;
   - đổi một định nghĩa: chỉ các op liên quan abstain, op khác không đổi;
   - thêm một ngành giả: câu chứa từ khoá abstain.
   - Rulebook train nguyên bản: dev 2410/0/37 và holdout không đổi.

Lý do: hợp precision-first (D11, D16 mục 4). Hoàn toàn suy từ rulebook train và rulebook của split lúc chạy, không
nhìn rulebook public. Đổi chữ vô hại (sửa chính tả) chỉ làm abstain thêm, không làm sai.
