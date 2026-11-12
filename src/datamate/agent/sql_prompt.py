"""Text-to-SQL prompt (D26) and the model-output -> answer conversion.

Prompt order = split-wide prefix (instructions, rulebook) -> shop part (README, schema) -> per-question part
(train examples, hints, question), so vLLM prefix caching reuses the long parts across a shop's questions.

Examples come only from train dev shops (agent/fewshot_sql.jsonl, gitignored, built by eval/build_fewshot_sql.py);
never from holdout or any test split (CLAUDE.md rules 1-2). The model writes notes, an answer TYPE and ONE SQL query;
DuckDB computes the value (CLAUDE.md rule 4).
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime

from datamate.agent.helpers import round_half_up

TYPES = ("number", "percent", "date", "string", "list", "table", "boolean", "unanswerable")

INSTRUCTIONS = """Bạn là chuyên viên phân tích dữ liệu bán hàng cho chủ shop nhỏ. Nhiệm vụ: viết MỘT câu truy vấn
SQL (DuckDB) trả lời chính xác câu hỏi ở cuối, áp dụng ĐÚNG luật nghiệp vụ trong RULEBOOK (rulebook là nguồn sự
thật; khác thói quen thông thường thì theo rulebook).

Định dạng trả lời (bắt buộc, không viết gì khác):
Notes:
- kỳ thời gian: ... (chép khoảng ngày từ GỢI Ý)
- đơn/dòng nào được tính, theo mục rulebook nào; cột tiền/số lượng nào; đơn vị
- kiểu đáp án và cách xếp hạng (nếu có)
TYPE: <number|percent|date|string|list|table|boolean|unanswerable>
```sql
SELECT ...
```

Quy tắc SQL:
- Chỉ một câu SELECT (được dùng WITH). Không CREATE/INSERT/COPY/PRAGMA. Chỉ dùng các bảng, view, macro có trong
  SCHEMA, đúng tên cột đã liệt kê.
- Ưu tiên view chuẩn: v_sales (một dòng một dòng bán, đã ghép đơn và sản phẩm), v_orders (một dòng một đơn).
  Nếu có cột is_valid thì is_valid = true nghĩa là đơn được tính doanh thu/bán hàng theo rulebook (đã bỏ đơn huỷ,
  đơn trả kể cả trả một phần). Không có is_valid thì tự lọc theo status và rulebook.
- So cột chữ (status, category, payment_method, unit...) bằng ĐÚNG giá trị liệt kê trong SCHEMA.
- Kỳ thời gian: chép đúng khoảng ngày trong GỢI Ý, viết `order_date BETWEEN DATE 'YYYY-MM-DD' AND DATE 'YYYY-MM-DD'`.
  Không tự tính lại tuần, tháng, Tết.
- Lọc sản phẩm/khách/nhóm hàng bằng khoá trong GỢI Ý (product_key, customer_key, category), không tự đoán chuỗi.
- Công nợ: dùng macro debt_at(DATE 'x'), debt_new(DATE 'a', DATE 'b'), debt_repaid(DATE 'a', DATE 'b') nếu có.
- Tồn kho, lô, hạn dùng: macro stock_at(DATE 'x') và lots_at(DATE 'x') nếu có. Nhập hàng: v_imports (dòng nhập),
  v_receipts (phiếu nhập), đã bỏ phiếu huỷ.
- Lọc theo ca, nhóm khách, kênh: cột shift, customer_group, channel_class (đã gom nhóm sẵn).
- Đáp án list gồm nhiều số tính riêng: SELECT unnest([biểu thức 1, biểu thức 2, ...]).
- Đọc phần LƯU Ý cuối SCHEMA: nếu rulebook của bộ dữ liệu này đổi định nghĩa thì áp dụng rulebook, không dùng sẵn
  cột/macro được nêu.
- Kết quả: number/percent/date/string/boolean -> đúng 1 dòng 1 cột. list -> 1 cột, mỗi dòng một phần tử, có
  ORDER BY. table -> các cột cần hiển thị, cột đầu là tên đối tượng.
- Tiền: số nguyên VND, viết ROUND(x) (kể cả trung bình). percent: ROUND(100.0 * phần / tổng, 2), không có ký hiệu %.
  Số khác (trung bình số lượng...) ROUND(x, 2).
- date: trả ngày (DATE). Câu hỏi theo tháng thì trả strftime(..., '%Y-%m').
- Tên trả về (string, list) lấy cột tên (product_name, customer_name...) đúng như dữ liệu.
- Xếp hạng: ORDER BY chỉ số DESC, hoà thì theo mã tăng dần (product_id hoặc product_key), rồi LIMIT k.
  Xếp hạng sản phẩm bỏ dòng phí dịch vụ (NOT is_fee nếu có cột này). Xếp hạng khách bỏ khách lẻ
  (customer_key IS NOT NULL), gom theo customer_key.
- Số lượng bán theo đơn vị cơ sở là SUM(qty_base); hỏi theo đơn vị khác thì chia cho factor trong bảng units.
  Hỏi MỘT đối tượng ("nào nhiều nhất") -> string; hỏi nhiều ("top 3", "những", "liệt kê") -> list.
- Dữ liệu không có thông tin cần thiết (cột/bảng không tồn tại) hoặc kỳ hỏi nằm ngoài khoảng dữ liệu:
  TYPE: unanswerable và không viết SQL."""


def build_prompt(rulebook: str, readme: str, schema: str, question: str, as_of_line: str, hints: list[str],
                 examples: list[dict]) -> str:
    ex = ""
    if examples:
        ex = "=== VÍ DỤ (shop khác, chỉ để tham khảo cách viết) ===\n" + "\n\n".join(
            "Gợi ý: " + "; ".join(e.get("context") or []) + f"\nCâu hỏi: {e['question']}\n{e['reply'].strip()}"
            for e in examples) + "\n\n"
    hint = "\n".join(f"- {h}" for h in hints) if hints else "- (không có)"
    return (f"{INSTRUCTIONS}\n\n=== RULEBOOK ===\n{rulebook.strip()}\n\n"
            f"=== README CỦA SHOP ===\n{readme.strip()}\n\n"
            f"=== SCHEMA (DuckDB) ===\n{schema}\n\n"
            f"{ex}"
            f"=== GỢI Ý (đã tính sẵn bằng code, dùng đúng các giá trị này) ===\n- {as_of_line}\n{hint}\n\n"
            f"=== CÂU HỎI ===\n{question}\n\nNotes:\n-")


# ------------------------------------------------------------------------------------------- parse model output

_SQL = re.compile(r"```(?:sql|SQL)?\s*\n(.*?)```", re.S)
_TYPE = re.compile(r"^\s*TYPE\s*:\s*([A-Za-z]+)", re.M)


def parse_reply(text: str) -> tuple[str | None, str | None]:
    """(answer type, SQL) from a model reply. Reasoning before </think> is ignored. The prompt ends with 'Notes:\\n-'
    so the reply continues the notes; the last TYPE line and the last sql block win."""
    t = text or ""
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    types = [m.lower() for m in _TYPE.findall(t)]
    atype = types[-1] if types and types[-1] in TYPES else None
    blocks = [b.strip() for b in _SQL.findall(t) if b.strip()]
    sql = blocks[-1] if blocks else None
    if sql is None and atype not in (None, "unanswerable"):   # unfenced query after the TYPE line
        tail = t[t.rfind("TYPE"):] if "TYPE" in t else t
        m = re.search(r"((?:WITH|SELECT)\b.*)", tail, re.S | re.I)
        sql = m.group(1).strip().rstrip("`").strip() if m else None
    if sql:
        sql = sql.strip().rstrip(";").strip()
    return atype, sql


class EmptyResult(ValueError):
    """The query ran but returned nothing usable (no row / NULL): never answered as 0 (abstain)."""


def _cell(v):
    if isinstance(v, datetime):
        return v.date().isoformat() if (v.hour, v.minute, v.second) == (0, 0, 0) else v.isoformat(sep=" ")
    if isinstance(v, date):
        return v.isoformat()
    if hasattr(v, "is_finite") or type(v).__name__ == "Decimal":
        v = float(v)
    if isinstance(v, float):
        if math.isnan(v):
            return None
        v = round_half_up(v, 2)
        return int(v) if v == int(v) else v
    return v


def to_row(atype: str, cols: list[str], rows: list[list]) -> dict:
    """SQL result -> {'type', 'answer'} for agent/answer.canon. Raises EmptyResult / ValueError on a bad shape."""
    if atype == "unanswerable":
        return {"type": "unanswerable", "reason": "model: data does not contain this information"}
    if not rows:
        raise EmptyResult("no rows")
    if atype in ("number", "percent", "date", "string", "boolean"):
        if len(rows) != 1:
            raise ValueError(f"{atype} answer needs exactly one row, got {len(rows)}")
        v = rows[0][0]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            raise EmptyResult("NULL value")
        if atype in ("number", "percent"):
            if isinstance(v, bool):
                raise ValueError("bool is not a number")
            return {"type": atype, "answer": _cell(float(v))}
        if atype == "date":
            return {"type": "date", "answer": _cell(v) if not isinstance(v, str) else v.strip()}
        if atype == "boolean":
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v in (0, 1):
                v = bool(v)
            if not isinstance(v, bool):
                raise ValueError("boolean answer must be true/false")
            return {"type": "boolean", "answer": v}
        return {"type": "string", "answer": str(_cell(v))}
    if atype == "list":
        vals = [_cell(r[0]) for r in rows]
        if any(v is None for v in vals):
            raise EmptyResult("NULL in list")
        return {"type": "list", "answer": vals}
    if atype == "table":
        return {"type": "table", "answer": {"columns": list(cols), "rows": [[_cell(x) for x in r] for r in rows]}}
    raise ValueError(f"unknown type {atype!r}")


def repair_prompt(prompt: str, reply: str, error: str) -> str:
    """One repair turn (slide 32): the same prompt, the failed reply and DuckDB's message."""
    return (f"{prompt}{reply.rstrip()}\n\n=== LỖI KHI CHẠY SQL ===\n{error}\n\n"
            "Sửa lại. Trả lời đúng định dạng (Notes, TYPE, một khối ```sql```).\n\nNotes:\n-")
