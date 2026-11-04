1. Xác định cuộc thi

Cuộc thi Zalo AI Challenge 2026, mùa thứ 8, chủ đề “Advancing AI Agents for Real Life”
Khởi tranh 26/10/2026
Hai bảng LifeMate (voice agent) và DataMate (data agent)
Giải thưởng Tổng 12.000 USD. Mỗi đội Nhất nhận 3.500 USD, mỗi đội Nhì nhận 2.500 USD
Tên đầy đủ của track ⚠️ DataMate: Talking To Your Data Through Agentic AI

2. Description: bài toán

Bài toán là xây dựng AI agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng của doanh nghiệp nhỏ (cửa hàng, trường học, phòng khám...). Agent phải:

Hiểu câu hỏi, tìm đúng bảng và cột, kết hợp nhiều nguồn, tính toán rồi trả kết quả.
Xử lý dữ liệu rời rạc, không chuẩn hoá.
Biết từ chối khi dữ liệu không đủ, không được đoán. Đây là điểm cốt lõi của đề.

3. Data ⚠️

```
Split	Quy mô	Có đáp án?
Train	3.000 câu, 30 business	✅ có answers.jsonl
Public test	1.000 câu, 14 business	❌
Private test	ra mắt ngày 20/11	❌
Input: questions.jsonl, mỗi dòng gồm question, business_id, as_of_date.
Dữ liệu mỗi business: các file .csv, .xlsx (nhiều sheet), .sqlite, kèm README.md mô tả schema.

```

Ngữ cảnh chung:
rulebook.md chứa quy tắc nghiệp vụ, ví dụ “bán chạy” xếp theo số lượng, doanh thu không tính VAT và phí giao, đơn huỷ xử lý riêng. Rulebook của train và public test có thể khác nhau.
calendar.csv là lịch âm, dùng cho các mốc như Tết, Rằm.
Độ khó ẩn: câu hỏi dùng thời gian tương đối tiếng Việt như “tuần rồi”, “cùng kỳ năm ngoái”, “quý này”. Các mốc này phải tính theo as_of_date.

4. Evaluation ⚠️

Penalized Accuracy: PA = (Đúng − Sai) / N

Kết quả Điểm
Đúng +1
Sai −1
Abstain hoặc bỏ trống 0

Trả unanswerable được chấm đúng/sai như một đáp án, không phải abstain
Đáp án được so khớp chính xác theo kiểu (exact typed match), không dùng LLM-judge. Có 9 kiểu: number, percent, string, date, boolean, list, table, unanswerable, abstain.
Chỉ nên trả lời khi xác suất đúng trên 50%, vì kỳ vọng điểm là 2p − 1.

Một mốc tham khảo: một đội nộp baseline trả lời 41/1000 câu, abstain 959 câu, được public score 0.019.

5. Rules ⚠️
   Hạng mục Ràng buộc
   Model Tổng ≤ 15B tham số, tính cả model phụ. MoE tính tổng tham số. Được phép quantize
   Weights Phải public trước 01/10/2026, license hợp lệ
   Phần cứng chấm 2× RTX 5090 (32GB), 512GB RAM, CUDA ≥ 12.8
   Thời gian Private run tối đa 6 giờ, tính cả thời gian load model
   Mạng Offline hoàn toàn (--network none)
   Cấm Human-in-the-loop, hardcode đáp án theo ID, train trên public hoặc private test
   Được dùng Tập train, dữ liệu tự sinh, tài nguyên public hợp lệ
   Code Phải public qua forum chính thức (quy định 8.2)

6. Submit Guide ⚠️
   Giai đoạn public: nộp file submission.jsonl, tối đa 5 lần/ngày.
   Giai đoạn private: nộp Docker image, tối đa 2 lần, chỉ lần cuối được tính. Container đọc /data (read-only) và ghi /result/submission.jsonl theo kiểu stream từng câu. BTC sẽ chạy lại Docker, và điểm chạy lại chỉ được lệch tối đa 0.03 so với điểm bạn upload.

7. Timeline ⚠️
   Mốc Thời điểm (GMT+7)
   Khởi tranh 26/10
   Docker freeze 16/11, 23:59, còn khoảng 12 ngày
   Private test release 20/11. Giờ chưa thống nhất: portal và PDF ghi 09:00, trang chủ ghi 11:00
   Kết quả private 23/11, 23:59
   Nhận định nhanh trước khi lên kế hoạch
   Đây là bài toán calibration + Text-to-SQL/Pandas, không chỉ là QA. Một câu trả lời sai còn tệ hơn bỏ trống.
   Khó nhất nằm ở ngữ nghĩa nghiệp vụ (rulebook), thời gian tương đối và lịch âm, và chuẩn hoá đáp án đúng kiểu. Năng lực LLM không phải nút thắt chính.
   Thời gian rất gấp. Môi trường Docker offline, model ≤15B trên 2×5090 phải chạy được sớm, không để sát hạn freeze.

---

1. Dùng LLM API hay model offline?

✅ Trên portal: Leaderboard chia làm 3 hạng mục:

Hạng mục Định nghĩa Có được xếp hạng?
SLM Model open-source ≤ 15B ✅ Chỉ hạng mục này được xếp hạng
LLM Model open-source > 15B Chỉ để tham khảo
APIs/Others Claude, Gemini, OpenAI… Chỉ để tham khảo

Portal cũng ghi rõ: lời giải hợp lệ phải tuân thủ luật của track và của cuộc thi.

→ Kết luận: muốn có giải thì phải dùng model open-source ≤15B, chạy offline. Bạn vẫn có thể nộp kết quả từ API để đo trần (upper bound), nhưng kết quả đó không được tính.

⚠️ Thêm từ rulebook: tổng ≤15B tính cả các model phụ (embedding, reranker…). MoE tính theo tổng tham số. Weights phải public trước 01/10/2026. Được phép quantize.

2. Giới hạn thời gian / chi phí / phần cứng ⚠️

Phần cứng chấm 2× RTX 5090 (32GB), 512GB RAM, CUDA ≥ 12.8
Thời gian Private run ≤ 6 giờ, tính cả thời gian load model
Mạng Offline, Docker chạy với --network none
Chi phí Không có giới hạn chi phí, vì đằng nào cũng không được dùng API

3. Nộp gì? ⚠️
   Public test: nộp file submission.jsonl, tối đa 5 lần/ngày.
   Private test: nộp Docker image. BTC tự chạy image trên máy của họ: đọc dữ liệu từ /data (read-only), ghi kết quả ra /result/submission.jsonl. Được nộp tối đa 2 lần, chỉ lần cuối được tính. Điểm khi BTC chạy lại chỉ được lệch tối đa 0.03 so với điểm bạn upload.
   Code phải công bố qua forum chính thức (quy định 8.2).

4. Có được xem dữ liệu test trước không? ⚠️
   Public test (1.000 câu): có, bạn tải về để chạy, nhưng không có đáp án. Cấm dùng tập này để train, tune prompt hay thu thập đáp án.
   Private test: không xem trước. Dữ liệu ra mắt ngày 20/11, sau khi Docker đã bị đóng băng (freeze) ngày 16/11. Pipeline phải tự chạy, không có người can thiệp (human-in-the-loop) và không hardcode đáp án theo ID.
   🚩 Phát hiện quan trọng từ leaderboard

Ít nhất 17 đội đã đạt PA = 1.0 ở hạng mục SLM trên public test, chỉ sau khoảng 2 ngày. Nghĩa là:

Public test đã bão hoà, không còn phân biệt được các đội.
Thứ hạng thực sự sẽ được quyết định ở private test, gồm cả business mới chưa từng thấy. Nên ưu tiên khả năng tổng quát hoá và độ ổn định của Docker, không nên tối ưu cho public test.
