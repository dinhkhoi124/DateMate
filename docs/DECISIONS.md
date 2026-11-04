# DECISIONS — Nhật ký quyết định

Mỗi quyết định ghi: **bối cảnh → lựa chọn → lý do → hệ quả**. Quyết định mới thêm lên đầu. Đổi quyết định cũ thì ghi entry mới, đánh dấu entry cũ là `Superseded by Dxx`.

---

### D10 — Chốt chặn phủ từ (coverage gate) trước khi trả lời (2026-11-04)
- **Bối cảnh:** lần nộp public đầu được PA 0.066 (ước ~372 đúng / ~306 sai, precision ~55%), trong khi holdout là 0.872. Public có ngành mới (mẹ & bé, sửa xe, điện thoại) và ~10 định nghĩa mới trong rulebook. Parser luật gán câu lạ vào op gần nhất rồi trả lời tự tin nhưng sai.
- **Lựa chọn:** `parse/gate.py` che thực thể catalog, mốc thời gian và số. Nếu còn từ nào ngoài `parse/gate_vocab.txt` thì abstain. Từ vựng gồm từ của các câu dev đã trả lời đúng, mỗi từ phải xuất hiện ở ≥ 2 shop (`analysis/build_gate_vocab.py --min-shops 2`, 206 từ, chỉ dùng dev).
- **Lý do:**
  - Leave-one-industry-out trên dev: gặp ngành chưa thấy thì chốt chặn bỏ ~11% câu đúng (coverage 0.889).
  - Holdout: số câu sai 20 → 10, precision 96.2% → 98.0%, PA 0.872 → 0.859.
  - Đánh đổi có lợi khi precision thấp; public và private là ngành lạ.
- **Hệ quả:**
  - Tỉ lệ abstain trên public 32% → 69%.
  - Thêm cách diễn đạt mới thì phải build lại từ vựng.
  - Khi có LLM parser: câu bị chốt chặn chặn sẽ chuyển sang LLM thay vì abstain.

### D09 — Ingest 3 tầng gán vai trò + các luật làm sạch (2026-11-04)
- **Bối cảnh:**
  - 4/30 shop không mô tả cột. biz_124 đặt tên cột cố tình lạ (`Order ID`, `Slg`, `Đ.vị`, `Bằng`, `Hạng`).
  - Sổ tay có dòng tổng phụ. Tiền ghi dạng `3,185tr`, `1tr250`.
- **Lựa chọn:**
  - Gán vai trò theo thứ tự: (1) mô tả README → vai trò; (2) tên cột → vai trò (từ điển tay + `learned_names.json` học từ README dev); (3) suy luận từ từ khoá tên cột + profile giá trị (mẫu mã `ORD/SP/KH/PN`, ngày, số).
  - Loại bảng xác định từ tập vai trò kết hợp gợi ý tên file. Vai trò chung (`name`, `qty`, `price`…) được cụ thể hoá theo loại bảng.
- **Luật làm sạch:**
  - Bỏ dòng trùng theo mã đơn **sau khi** điền xuôi.
  - Bỏ dòng tổng phụ (`TỔNG NGÀY…`).
  - Trạng thái ưu tiên huỷ > trả hàng (mọi bằng chứng) > đang giao > hoàn thành.
  - Shop chỉ có hoá đơn thì lấy tiền hàng từ cấp đơn.
- **Lý do:** đã đo được doanh thu tháng khớp tuyệt đối 68/68 dev, 8/8 holdout.
- **Hệ quả:** tầng (3) là heuristic. Dự định dùng LLM làm schema mapper dự phòng cho shop lạ ở private.

### D08 — Đọc Excel bằng calamine, fallback openpyxl (2026-11-04)
- **Bối cảnh:** đọc toàn bộ 30 shop train bằng openpyxl mất 587 giây, trong khi private có ngân sách 6 giờ gồm cả LLM.
- **Lựa chọn:** dùng `python-calamine` (Rust) qua `pd.read_excel(engine="calamine")`. Lỗi thì fallback sang openpyxl. Cache pickle theo file vào `cache/raw/`.
- **Lý do:** nhanh hơn 8,5 lần trên cùng file (2,5s so với 21,5s), giá trị giống hệt.
- **Hệ quả:** thêm dependency `python-calamine`. Cache có thể ghi được trong container (không ghi vào `/data`).

### D07 — Đọc rulebook lúc chạy, chỉ hardcode luật của train (2026-11-04)
- **Bối cảnh:** rulebook mỗi split khác nhau (public đã thêm luật ngành). Việc hardcode luật chỉ có trong rulebook public là vùng xám (thể lệ 4.2).
- **Lựa chọn:**
  - ✅ Đọc `rulebook.md` trong `/data` lúc chạy, đưa vào prompt hoặc parser. Đây là cách an toàn nhất.
  - ✅ Hardcode các luật nghiệp vụ chung học từ rulebook train.
  - ⚠️ Chưa hardcode luật chỉ xuất hiện trong rulebook public, đợi BTC trả lời.
- **Lý do:** đọc lúc chạy là cách duy nhất vừa an toàn về luật vừa tổng quát được, vì private có thể có luật mới.
- **Hệ quả:**
  - Executor tham số hoá luật theo rulebook đọc được, ví dụ giờ ca, khoảng Tết.
  - Câu hỏi chạm tới luật lạ đi vào nhánh fallback, có rulebook trong context.
  - Cần ghi trên forum câu hỏi cho BTC (`SPEC.md` §7.1).

### D06 — Holdout 6 shop theo kiểu xuất (2026-11-04)
- **Bối cảnh:** private gồm shop chưa từng thấy. Đo trên chính những shop đã dùng để viết code sẽ cho kết quả lạc quan quá mức.
- **Lựa chọn:** giữ lại `biz_103, biz_110, biz_117, biz_123, biz_125, biz_128`, không mở dữ liệu các shop này khi viết code.
- **Lý do:** mỗi shop đại diện một kiểu xuất khác nhau (Sapo-like, CSV hoá đơn, sổ tay VLXD, POS sqlite, CSV phẳng, KiotViet-like).
- **Hệ quả:** PA holdout là chỉ số chính trong `EXPERIMENTS.md`.

### D05 — Map schema theo mô tả README, không theo tên cột (2026-11-04)
- **Bối cảnh:** 244 tên cột khác nhau, nhưng chỉ có khoảng 120 mô tả trong README, theo một từ vựng có kiểm soát.
- **Lựa chọn:** từ điển `mô tả → field chuẩn`, khớp mềm (fuzzy). LLM chỉ dùng khi không khớp.
- **Lý do:** deterministic, tổng quát hoá được sang shop mới miễn là BTC giữ phong cách viết README.
- **Hệ quả:** cần có test với README tổng hợp có mô tả biến thể.

### D04 — LLM dịch, code tính (2026-11-04)
- **Bối cảnh:** chấm exact match tới từng đồng. Model ≤ 15B tính toán không ổn định.
- **Lựa chọn:** LLM chỉ sinh Intent JSON, còn executor deterministic tính kết quả. Có nhánh code-gen dự phòng cho câu OOD, kèm abstain nghiêm ngặt.
- **Lý do:** dễ kiểm thử từng luật, ổn định khi chạy lại (yêu cầu lệch ≤ 0.03).
- **Hệ quả:** coverage bị giới hạn bởi tập intent đã cài, cần nhánh dự phòng cho câu mới.

### D03 — Không dùng public test để phát triển (2026-11-04)
- **Bối cảnh:** thể lệ cấm train hoặc tune trên public/private. Public đã bão hoà (≥ 17 đội đạt 1.0).
- **Lựa chọn:** chỉ đọc train. Public chỉ dùng để nộp kiểm tra định kỳ.
- **Hệ quả:** ❓ việc dùng luật ngành từ rulebook public cần xác minh trên forum (`SPEC.md` §7).

### D02 — Chỉ hạng mục SLM, model open-source ≤ 15B offline (2026-11-04)
- **Lý do:** chỉ hạng mục SLM được xếp hạng. Kết quả dùng API chỉ để tham khảo.
- **Hệ quả:** cần GPU để phát triển phần LLM, vì máy dev không có GPU NVIDIA.

### D01 — Python 3.11, src layout (2026-11-04)
- **Lý do:** tương thích vLLM/CUDA image. `src/datamate` tách package khỏi script.
- **Hệ quả:** chạy bằng `.venv/Scripts/python` với `PYTHONPATH=src` (hoặc `pip install -e .`).
