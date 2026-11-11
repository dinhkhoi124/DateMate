# SPEC — DataMate (Zalo AI Challenge 2026)

> Nguồn: `thele/rule-game.md` (thể lệ), `data/*/rulebook.md` (luật nghiệp vụ). Mục nào chưa xác minh trên forum/portal thì đánh dấu ❓.
> Cập nhật lần cuối: 2026-11-07, đối chiếu portal `challenge.zalo.ai/portal/datamate` (Overview: Description, Evaluation, Rule, Submission Guidance; tab Data cần đăng nhập, chưa đối chiếu lại).

## 1. Bài toán

Xây AI agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của doanh nghiệp nhỏ. Agent phải hiểu câu hỏi, tìm đúng bảng và cột, kết hợp nhiều nguồn, tính toán rồi trả kết quả. **Biết từ chối khi dữ liệu không đủ, không được đoán.**

Track: **DataMate — Talking To Your Data Through Agentic AI**. Chỉ hạng mục **SLM (open-source ≤ 15B)** được xếp hạng.

## 2. Data

| Split        | Quy mô                                 | Đáp án             | Vị trí                          |
| ------------ | -------------------------------------- | ------------------ | ------------------------------- |
| Train        | 3.000 câu, 30 shop `biz_101`–`biz_130` | ✅ `answers.jsonl` | `data/train/train/`             |
| Public test  | 1.000 câu, 14 shop `dev_001`–`dev_014` | ❌                 | `data/public_test/public_test/` |
| Private test | ra ngày 20/11, shop mới                | ❌                 | `/data` trong Docker            |

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
- Mốc tham khảo: baseline trả lời 41/1000 câu đạt 0.019. **≥ 17 đội đạt PA = 1.0 trên public** (hạng SLM, nhiều đội từ 02/11). Loại model trên leaderboard do đội tự khai và public không được kiểm chứng (portal có disclaimer), nên con số này chỉ cho biết public giải trọn được, không cho biết cách làm. Thứ hạng thật nằm ở private.
- Hoà điểm: xét tổng thời gian inference khi BTC chạy lại, rồi tới bài nộp sớm hơn.

### Format đáp án (rulebook §9)

Mỗi dòng submission: `{"id": "...", "type": "...", "answer": ...}`.

| type         | Format                                             | Ví dụ đúng               |
| ------------ | -------------------------------------------------- | ------------------------ |
| number       | JSON number, tiền là số nguyên VND                 | `4820000`, `18.6`        |
| percent      | số phần trăm, không có ký hiệu %                   | `12.4`                   |
| date         | `"YYYY-MM-DD"`. Câu hỏi theo tháng thì `"YYYY-MM"` | `"2026-08-15"`           |
| boolean      | `true` / `false`                                   | `true`                   |
| string       | tên đúng như trong dữ liệu, đủ dấu                 | `"Sting dâu 330ml"`      |
| list         | mảng có thứ tự, một phần tử vẫn là mảng            | `["Cà phê sữa"]`         |
| table        | `{"columns": [...], "rows": [[...]]}`              | cột đầu là tên đối tượng |
| unanswerable | `{"type": "unanswerable", "reason": "..."}`        | `reason` phải là chuỗi   |
| abstain      | `{"type": "abstain"}` hoặc không ghi dòng          |                          |

**Quy tắc so sánh:**

- Số khớp tuyệt đối: tiền đúng từng đồng, % đúng 2 chữ số thập phân.
- Chuỗi không phân biệt hoa thường, khoảng trắng, dấu câu. Chấp nhận mọi cách viết có trong file, hoặc mã.
- List phải đúng cả tập lẫn thứ tự. Bằng nhau thì xếp theo mã tăng dần.
- Table: thứ tự dòng không quan trọng, cột so theo tên rồi đến vị trí.

**Làm tròn:** tiền làm tròn về đồng (nửa đồng làm tròn lên). Phần trăm và trung bình khác làm tròn 2 chữ số thập phân.

## 4. Ràng buộc

| Hạng mục       | Ràng buộc                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------- |
| Model          | Tổng ≤ 15B tham số, tính cả model phụ (embedding, reranker). MoE tính tổng tham số. Được quantize |
| Weights        | Public trước **26/10/2026**, license cho phép dùng, được fine-tune. Phải khai báo model trong README bài nộp (Rule 2) |
| Phần cứng chấm | 2× RTX 5090 32GB, 512GB RAM, CUDA ≥ 12.8                                                          |
| Thời gian      | Private run ≤ 6 giờ, tính cả load model                                                           |
| Mạng           | Offline hoàn toàn (`--network none`)                                                              |
| Cấm            | Human-in-the-loop; hardcode đáp án hoặc logic theo id câu/business (Rule 4); dùng public/private test để train, **thu thập đáp án câu test bằng bất kỳ cách nào** (Rule 6) |
| Chạy code      | Agent được sinh và chạy code trong container: "expected but not required" (Rule 5). `/data` read-only, file tạm ghi chỗ khác |
| Được dùng      | Tập train, dữ liệu tự sinh, tài nguyên public hợp lệ                                              |
| Code           | Phải public qua forum chính thức (quy định 8.2)                                                   |

**Quy ước nội bộ (D23):** không đọc câu hỏi, rulebook, cấu trúc file hay log của public để viết luật, prompt hay sửa code. Public chỉ dùng để sinh submission (`/submit-check` kiểm format) và nhận điểm. Được đóng gói rulebook train trong Docker để phát hiện luật đổi lúc chạy (FAQ).

## 5. Nộp bài

- **Public (26/10 → 16/11):** nộp file `submission.jsonl`, tối đa 5 lần/ngày. Điểm chỉ để tham khảo trong lúc phát triển. Bảng xếp hạng chỉ tính cột SLM; chọn loại model **"Based on small models (<=15B)"** khi nộp.
- **Nộp giải pháp (hạn 16/11 23:59):** gói Docker gồm code train, code inference và dữ liệu, gửi qua form **"Solution Submission"** trên portal bằng **link Google Drive public kèm checksum**. Tối đa 2 lần, BTC chỉ xét lần cuối.
- **Vòng private (20/11 09:00 → 23/11 23:59): đội tự chạy bản đã đóng băng trên tập private** rồi upload `submission.jsonl`. File upload chỉ quyết định bảng tạm và đội nào được chạy lại. **Điểm cuối là điểm BTC chạy lại image** (Rule 8). Cần có GPU trong 4 ngày này.
- **Yêu cầu với image** (chi tiết ở mục Submission contract bên dưới):
  - Container đọc `/data` (read-only) và ghi `/result/submission.jsonl` theo kiểu **stream từng câu**.
  - BTC chạy lại image; điểm chạy lại chỉ được lệch **≤ 0.03** so với điểm bạn upload.
  - Cấu trúc `/data` lúc chấm: giống thư mục split, đã xác minh theo Docker Guide §3.1 (xem §7 câu 2).
  - FAQ (D23): BTC chạy lại trên **2× 5090**, chạy lần 2 nếu lần 1 không khớp, không sửa code; batch hay tuần tự do đội quyết. Không chạy thử trước khi nhận Docker, chỉ có Test Solution (1 lượt/ngày, đóng 15/11 23:59). Private phải chạy bằng Docker cuối cùng đã nộp. Hoà điểm xét thời gian inference.
  - Lệch quá 0.03: BTC xem log trong image, có thể chạy thêm lần nữa và hỏi đội. Không giải thích được thì loại. Chạy quá 6 giờ thì chỉ chấm các câu đã ghi, phần còn lại tính là abstain.

## 6. Timeline (GMT+7)

| Mốc                  | Thời điểm                                           |
| -------------------- | --------------------------------------------------- |
| Khởi tranh           | 26/10/2026                                          |
| **Docker freeze**    | **16/11, 23:59**                                    |
| Private test release | 20/11, 09:00 (Submission Guidance)                  |
| Hạn upload private   | 23/11, 23:59 (đội tự chạy, upload submission)       |
| Công bố top 2        | 02/12                                               |

## 7. Câu hỏi mở ❓

1. Có được cài các luật mới lấy từ rulebook public vào code không? Đây là ngữ cảnh, không phải đáp án.
   Trả lời: Không nên. Luật của đề yêu cầu agent đọc rulebook lúc chạy. Tab Data ghi:

“The rulebook may differ between splits so your agent must read and apply the rulebook of the split it is answering, not a copy of the training rulebook.”

2. Cấu trúc chính xác của `/data` khi chấm private.
   Theo Docker Guide §3.1:
   /data/ ← read-only
   ├── data/<business_id>/
   │ ├── *.csv | *.xlsx | *.sqlite
   │ └── README.md
   ├── rulebook.md ← riêng cho từng split
   ├── calendar.csv ← lịch âm–dương
   └── questions.jsonl ← id, business_id, question, as_of_date

Không có answers.jsonl.
File tạm phải ghi vào /tmp.
BTC chạy bash /code/predict.sh không truyền tham số, kèm các cờ --network none và --shm-size=16g, trên 2 GPU.

3. Số câu của private test, để tính ngân sách thời gian cho mỗi câu.
   Trả lời: Portal không ghi trực tiếp, nhưng Docker Guide §10 cho công thức ước lượng:

“model loading + (average time per question … × 500)”

→ Private test có khoảng 500 câu.

Tổng thời gian 6 giờ = 21.600 giây
Load model ~2 phút (BTC đo với Qwen3-14B)
Trung bình mỗi câu ≈ 43 giây
Mục tiêu an toàn ≤ 25–30 giây/câu, chừa dư địa cho câu khó và retry

Lưu ý: hoà điểm thì xếp hạng theo tổng thời gian chạy, nên càng nhanh càng có lợi.

Những điểm mới quan trọng khác
Định dạng đã chốt:
percent: -12.45 (đơn vị %, không phải -0.1245)
date: "2026-08-15"
table: {"columns": [...], "rows": [[...]]}
unanswerable: có thể kèm trường "reason"
Log bắt buộc cho từng câu: id, answer, elapsed_ms, thinking. Trường thinking phải là reasoning gốc, không tóm tắt. Nếu agent chạy code thì log thêm code và output.
Mốc tuổi model là 26/10/2026, không phải 01/10 như nguồn cộng đồng đã ghi.
Image Docker phải chứa cả code train, dữ liệu train và manifest checksum, dung lượng tối đa 100 GB khi chưa nén.
Với câu hỏi xếp hạng: hoà thì sắp theo mã tăng dần. Câu nào hoà ngay tại ranh giới top-k đã bị BTC loại khỏi bộ test.

4. ~~"Có được xem cấu trúc dữ liệu public để cải thiện bộ đọc không" Câu trả lời: chắn chắn có nhé.~~
   **Thay bằng FAQ BTC (10/11, D23):** "Tập public test không được sử dụng để train model và không được sử dụng trong quá trình inference. Kể cả việc xử lý tập public test để tạo ra thông tin train khác." Public chỉ để sinh submission và nhận điểm.

---

Submission contract
Hạng mục Yêu cầu chính thức Nguồn
Input /data (read-only): questions.jsonl, rulebook.md, calendar.csv, data/<business_id>/ Guide §3.1, Rule 5
File tạm Ghi ở chỗ khác, ví dụ /tmp Guide §2
Output /result/submission.jsonl. Ghi ngay sau mỗi câu. Tự tạo thư mục nếu chưa có Guide §3.2, §5
Log ✅ Bắt buộc. Ghi vào /result/logs/, mỗi câu một record, định dạng tuỳ ý Guide §3.3
Entrypoint /code/predict.sh. BTC chạy bash /code/predict.sh, không truyền tham số Guide §4, §5
Các trường log bắt buộc
Trường Khi nào
id, answer Luôn có
elapsed_ms Luôn có. Số nguyên mili-giây từ lúc câu bắt đầu đến khi có đáp án. Nếu chạy theo batch thì tính từ lúc câu vào batch
thinking Luôn có. Reasoning gốc, không tóm tắt, không cắt. Nếu gọi model nhiều lần thì là list theo thứ tự. Ghi null nếu không có reasoning
code, output Khi có chạy code. Với hàm viết sẵn thì ghi thêm inputs
Tài nguyên và thời gian
Hạng mục Giới hạn
GPU 2 × RTX 5090, 32 GB mỗi card (tổng 64 GB)
RAM 512 GB
CPU ❓ Không được nêu
Shared memory --shm-size=16g
Mạng Tắt hoàn toàn (--network none)
Thời gian 6 giờ cho toàn bộ private test, tính cả load model. Hết giờ thì container bị dừng, câu chưa ghi tính là abstain
Kích thước image ≤ 100 GB khi chưa nén (file .tar từ docker save)
Model Tổng ≤ 15B tham số cho mọi model dùng lúc inference
Yêu cầu Docker
Hạng mục Yêu cầu
Base image Khuyến nghị vllm/vllm-openai:v0.10.1.1. Dùng image khác thì phải hỗ trợ RTX 5090 (compute capability 12.0, CUDA ≥ 12.8)
Platform linux/amd64
Tag zaic2026_datamate_<team_name>:final (tên đội viết thường)
ENTRYPOINT Đặt ENTRYPOINT []. BTC chạy với --entrypoint bash, nên ENTRYPOINT riêng của image sẽ không được dùng
Biến môi trường HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1. Nếu cặp GPU không có P2P thì thêm NCCL_P2P_DISABLE=1
Cấu hình Model, seed, temperature, batch size… phải cố định sẵn trong code. BTC không chỉnh
Nội dung image Code inference, weights, predict.sh, README.md và /code/training_code/ (code train, dữ liệu train, logs, data_manifest.json kèm SHA-256)

Lệnh chấm của BTC (dùng nguyên lệnh này để tự kiểm tra):

bash
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g \
 --entrypoint bash \
 -v /path/to/public*test:/data:ro \
 -v /path/to/output:/result \
 zaic2026_datamate_<team_name>:final /code/predict.sh
