# SPEC — DataMate (Zalo AI Challenge 2026)

> Nguồn: `thele/rule-game.md` (thể lệ), `data/*/rulebook.md` (luật nghiệp vụ). Mục nào chưa xác minh trên forum/portal thì đánh dấu ❓.
> Cập nhật lần cuối: 2026-11-04

## 1. Bài toán
Xây AI agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của doanh nghiệp nhỏ. Agent phải hiểu câu hỏi, tìm đúng bảng và cột, kết hợp nhiều nguồn, tính toán rồi trả kết quả. **Biết từ chối khi dữ liệu không đủ, không được đoán.**

Track: **DataMate — Talking To Your Data Through Agentic AI**. Chỉ hạng mục **SLM (open-source ≤ 15B)** được xếp hạng.

## 2. Data

| Split | Quy mô | Đáp án | Vị trí |
|---|---|---|---|
| Train | 3.000 câu, 30 shop `biz_101`–`biz_130` | ✅ `answers.jsonl` | `data/train/train/` |
| Public test | 1.000 câu, 14 shop `dev_001`–`dev_014` | ❌ | `data/public_test/public_test/` |
| Private test | ra ngày 20/11, shop mới | ❌ | `/data` trong Docker |

Mỗi split có:
- `questions.jsonl`: `{id, business_id, question, as_of_date}`. `as_of_date` là "hôm nay" của câu hỏi.
- `rulebook.md`: luật nghiệp vụ mặc định. **Mỗi split có một bản khác nhau**: public có thêm luật so với train, nên private cũng có thể có luật mới.
- `calendar.csv`: `date, weekday, lunar_day, lunar_month, lunar_year, lunar_leap, holiday` (2024–2027), dùng cho Tết và Rằm.
- `data/<business_id>/`: các file `.csv`, `.xlsx` (nhiều sheet), `.sqlite`, kèm `README.md` mô tả cột theo dạng `**Tên cột**: nghĩa`.

**Đặc điểm train (đã đo):**
- Phân bố kiểu đáp án: number 1884, percent 578, list 193, date 145, string 88, unanswerable 70, boolean 23, table 19.
- Có khoảng 40–60 intent sinh từ template. Câu hỏi có teencode, không dấu, lỗi gõ Telex.
- 167 câu có `alias` (mã hàng, mã khách, cách viết khác của tên).

## 3. Đánh giá

**PA = (Đúng − Sai) / N.** Đúng được +1, sai bị −1, abstain hoặc bỏ trống được 0.
- Trả `unanswerable` được chấm đúng/sai như một đáp án, **không phải abstain**.
- Chấm bằng exact typed match, không dùng LLM-judge. Chỉ nên trả lời khi P(đúng) > 0.5, vì kỳ vọng điểm là 2p − 1.
- Mốc tham khảo: baseline trả lời 41/1000 câu đạt 0.019. **≥ 17 đội đạt PA = 1.0 trên public**, nên public đã bão hoà và thứ hạng nằm ở private.

### Format đáp án (rulebook §9)
Mỗi dòng submission: `{"id": "...", "type": "...", "answer": ...}`.

| type | Format | Ví dụ đúng |
|---|---|---|
| number | JSON number, tiền là số nguyên VND | `4820000`, `18.6` |
| percent | số phần trăm, không có ký hiệu % | `12.4` |
| date | `"YYYY-MM-DD"`. Câu hỏi theo tháng thì `"YYYY-MM"` | `"2026-08-15"` |
| boolean | `true` / `false` | `true` |
| string | tên đúng như trong dữ liệu, đủ dấu | `"Sting dâu 330ml"` |
| list | mảng có thứ tự, một phần tử vẫn là mảng | `["Cà phê sữa"]` |
| table | `{"columns": [...], "rows": [[...]]}` | cột đầu là tên đối tượng |
| unanswerable | `{"type": "unanswerable", "reason": "..."}` | `reason` phải là chuỗi |
| abstain | `{"type": "abstain"}` hoặc không ghi dòng | |

**Quy tắc so sánh:**
- Số khớp tuyệt đối: tiền đúng từng đồng, % đúng 2 chữ số thập phân.
- Chuỗi không phân biệt hoa thường, khoảng trắng, dấu câu. Chấp nhận mọi cách viết có trong file, hoặc mã.
- List phải đúng cả tập lẫn thứ tự. Bằng nhau thì xếp theo mã tăng dần.
- Table: thứ tự dòng không quan trọng, cột so theo tên rồi đến vị trí.

**Làm tròn:** tiền làm tròn về đồng (nửa đồng làm tròn lên). Phần trăm và trung bình khác làm tròn 2 chữ số thập phân.

## 4. Ràng buộc

| Hạng mục | Ràng buộc |
|---|---|
| Model | Tổng ≤ 15B tham số, tính cả model phụ (embedding, reranker). MoE tính tổng tham số. Được quantize |
| Weights | Public trước 01/10/2026, license hợp lệ |
| Phần cứng chấm | 2× RTX 5090 32GB, 512GB RAM, CUDA ≥ 12.8 |
| Thời gian | Private run ≤ 6 giờ, tính cả load model |
| Mạng | Offline hoàn toàn (`--network none`) |
| Cấm | Human-in-the-loop, hardcode đáp án theo ID, train hoặc tune trên public/private test |
| Được dùng | Tập train, dữ liệu tự sinh, tài nguyên public hợp lệ |
| Code | Phải public qua forum chính thức (quy định 8.2) |

**Quy ước nội bộ:** không đọc câu hỏi public để viết luật hay prompt. Public chỉ dùng để nộp kiểm tra.

## 5. Nộp bài

- **Public:** nộp file `submission.jsonl`, tối đa 5 lần/ngày.
- **Private:** nộp Docker image, tối đa 2 lần, chỉ tính lần cuối.
  - Container đọc `/data` (read-only) và ghi `/result/submission.jsonl` theo kiểu **stream từng câu**.
  - BTC chạy lại image; điểm chạy lại chỉ được lệch **≤ 0.03** so với điểm bạn upload.
  - ❓ Cấu trúc `/data` lúc chấm: giả định giống thư mục split (`questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<biz>/`). Cần xác minh.

## 6. Timeline (GMT+7)

| Mốc | Thời điểm |
|---|---|
| Khởi tranh | 26/10/2026 |
| **Docker freeze** | **16/11, 23:59** |
| Private test release | 20/11 (09:00 hoặc 11:00, nguồn ghi chưa thống nhất) |
| Kết quả private | 23/11, 23:59 |

## 7. Câu hỏi mở ❓
1. Có được cài các luật mới lấy từ rulebook public vào code không? Đây là ngữ cảnh, không phải đáp án.
2. Cấu trúc chính xác của `/data` khi chấm private.
3. Số câu của private test, để tính ngân sách thời gian cho mỗi câu.
