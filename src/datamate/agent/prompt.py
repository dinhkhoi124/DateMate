"""Code-gen prompt (D20). Order = split-wide prefix (instructions, helper API, rulebook) -> shop part (README,
schema) -> question, so vLLM prefix caching reuses the long parts across a shop's questions.

No few-shot and no question text from any test split here (CLAUDE.md rule 1). The rulebook text is the one of the
split being answered, read at run time (SPEC §7.1); it is never written to logs by this module.
"""
from __future__ import annotations

import re

INSTRUCTIONS = """Bạn là trợ lý phân tích dữ liệu bán hàng cho chủ shop nhỏ. Nhiệm vụ: viết MỘT hàm Python
`solve(T, H)` dùng pandas để trả lời chính xác câu hỏi ở cuối, áp dụng ĐÚNG luật nghiệp vụ trong RULEBOOK bên dưới
(rulebook này là nguồn sự thật; nếu khác thói quen thông thường thì theo rulebook).

Quy tắc bắt buộc:
- Chỉ dùng dữ liệu trong T (dict tên bảng -> DataFrame) và các hàm trong H. Không đọc file, không tự bịa số.
- `solve` trả về dict {"type": ..., "answer": ...} với type thuộc: number, percent, date, string, list, table, boolean.
  * number: tiền là số nguyên VND (dùng H.money), số khác làm tròn 2 chữ số (H.r2).
  * percent: số phần trăm, 2 chữ số thập phân, không có ký hiệu % (dùng H.pct(phần, tổng)).
  * date: "YYYY-MM-DD"; nếu câu hỏi theo tháng thì "YYYY-MM".
  * string: tên đúng như trong dữ liệu. list: danh sách có thứ tự (hoà thì xếp theo mã tăng dần).
  * table: {"columns": [...], "rows": [[...], ...]}, cột đầu là tên đối tượng.
- Nếu dữ liệu không có thông tin cần thiết, hoặc kỳ hỏi nằm ngoài khoảng dữ liệu, trả
  {"type": "unanswerable", "reason": "<lý do ngắn>"}. Nếu không chắc cách hiểu câu hỏi, trả {"type": "abstain"}.
- "Hôm nay" là H.as_of. Mọi cách nói thời gian (tháng này, tuần trước, quý 2, Tết…) lấy bằng H.periods()
  thay vì tự tính, trừ khi rulebook định nghĩa khác.
- Chỉ import: pandas, numpy, math, datetime, re, collections, itertools, statistics, unicodedata, calendar.

Lưu ý khi viết code:
- T và H là tham số của solve: KHÔNG `import H` hay `import T`. pd, np, date đã có sẵn.
- Đầu hàm solve, viết vài dòng comment nêu mục rulebook áp dụng (đơn nào được tính, tính tiền theo cột nào,
  đơn vị, kỳ thời gian) rồi mới viết code theo đúng các mục đó.
- So sánh cột chữ (status, payment_method, category, unit…) phải dùng ĐÚNG giá trị liệt kê ở "values" trong schema,
  không tự đoán cách viết. raw_status là chữ gốc trong file; ánh xạ sang status đã làm sẵn.
- Cột date (datetime.date) so trực tiếp với period["start"], period["end"]; lọc kỳ nên dùng H.in_period(series, p).
  Không dùng between(..., inclusive=True) (pandas 3 chỉ nhận "both", "neither", "left", "right").
- Tìm sản phẩm/khách/nhóm hàng trong câu hỏi: gọi H.find_products(), H.find_customers(), H.find_categories()
  KHÔNG truyền đối số. Lọc bảng theo cột product_key / customer_id / category bằng key trả về.
- Mọi giá trị tiền, kể cả trung bình tiền mỗi đơn hay giá bình quân, trả qua H.money.
- Số lượng theo đơn vị cơ sở là qty_base (= qty × factor). Câu hỏi theo đơn vị khác (thùng, hộp, vỉ…) thì đổi
  bằng factor trong T['units'] của đúng sản phẩm.
- Nếu sau khi lọc không còn dòng nào mà câu hỏi kỳ vọng có dữ liệu, trả {"type": "abstain"} thay vì 0.

Trả lời bằng đúng một khối ```python ... ``` chứa hàm solve (có thể kèm hàm phụ). Không giải thích ngoài khối code."""

HELPER_DOC = """API của H (đã cài đúng rulebook mục thời gian và làm tròn):
- H.as_of: datetime.date, ngày "hôm nay" của câu hỏi.
- H.periods(text=None) -> list[{"start": date, "end": date, "kind": str, "label": str}]: các kỳ thời gian nhắc trong
  câu hỏi (hoặc trong text), theo thứ tự xuất hiện. kind: day|range|week|last_n|month|mtd|quarter|ytd|tet|asof.
- H.month(y, m), H.quarter(y, q), H.tet(lunar_year) -> period dict; H.week_start(d) -> date.
- H.previous(p): kỳ liền trước cùng độ dài; H.last_year(p): cùng kỳ năm trước.
- H.in_period(series, p) -> mask bool (so theo ngày, gồm cả hai đầu).
- H.find_products(text=None), H.find_customers(text=None), H.find_categories(text=None)
  -> list[{"key", "matched", "ambiguous"}]: thực thể nhắc trong câu hỏi; key khớp cột product_key / customer_id /
  category của T. Dùng các hàm này thay vì so chuỗi tay (câu hỏi có thể không dấu, viết tắt, sai chính tả).
- H.money(x) -> int; H.r2(x) -> float; H.pct(part, whole) -> float | None.
- H.calendar: DataFrame lịch (date, weekday, lunar_day, lunar_month, lunar_year, lunar_leap, holiday)."""


def build_prompt(rulebook: str, readme: str, schema: str, question: str, as_of: str) -> str:
    return (f"{INSTRUCTIONS}\n\n{HELPER_DOC}\n\n=== RULEBOOK ===\n{rulebook.strip()}\n\n"
            f"=== README CỦA SHOP (mô tả cột gốc) ===\n{readme.strip()}\n\n"
            f"=== CÁC BẢNG T (đã chuẩn hoá từ file gốc) ===\n{schema}\n\n"
            f"=== CÂU HỎI ===\nHôm nay: {as_of}\nCâu hỏi: {question}\n")


_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    """Last fenced code block that defines solve (reasoning text before </think> is ignored)."""
    t = text or ""
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    blocks = [b for b in _BLOCK.findall(t) if "def solve" in b]
    if blocks:
        return blocks[-1]
    if "def solve" in t:              # unfenced answer
        return t[t.index("def solve"):]
    return None
