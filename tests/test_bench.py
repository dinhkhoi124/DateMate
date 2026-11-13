"""eval/synth/families.py + bench.py (E0, docs/BENCHMARK.md): transforms keep meaning / change it as labelled,
A and B use different parameters, metrics follow the definitions of BENCHMARK.md §4. Toy shops only."""
import json
import random
from pathlib import Path

import pytest

from datamate.config import from_rulebook
from datamate.ingest.reader import sheet_period
from eval.synth import bench as BM
from eval.synth import families as F
from eval.synth.make import Sheet, Shop

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train" / "train"
needs_data = pytest.mark.skipif(not (TRAIN / "rulebook.md").exists(), reason="train data not available")

KINDS = [("ban_hang", "lines", {"order_id": "Mã đơn", "order_date": "Ngày", "status": "Trạng thái",
                                "amount": "Thành tiền"}),
         ("ton_kho", "stock", {"product_name": "Tên hàng", "stock_qty": "Tồn"})]
README = ("# Dữ liệu biz_x\n\nMột file phẳng.\n\n## Cột trong `ban_hang.csv`\n\n- **Mã đơn**: mã đơn hàng\n"
          "- **Ngày**: ngày bán\n- **Trạng thái**: trạng thái đơn\n- **Thành tiền**: tiền dòng hàng\n\n"
          "## Cột trong `ton_kho.csv`\n\n- **Tên hàng**: tên\n- **Tồn**: tồn cuối\n")


def shop():
    rows = [["D1", "2025-06-03", "Xong", "120000"], ["D2", "2025-06-20", "Đã hủy", "50000"],
            ["D3", "2025-07-01", "Xong", "75000"], ["D4", "2025-07-09", "Đổi trả", "30000"]]
    return Shop("biz_x", README, [Sheet("ban_hang.csv", None, [], ["Mã đơn", "Ngày", "Trạng thái", "Thành tiền"], rows),
                                  Sheet("ton_kho.csv", None, [], ["Tên hàng", "Tồn"], [["Bia", "10"]])])


def ctx(side, **kw):
    return F.Ctx(side, random.Random(0), KINDS, **kw)


def test_rename_cols_sides_differ_and_readme_follows():
    a, b = shop(), shop()
    assert F.f_rename_cols(a, ctx("A")) and F.f_rename_cols(b, ctx("B"))
    assert a.sheets[0].header != b.sheets[0].header
    for s in (a, b):
        for h in s.sheets[0].header:
            assert f"**{h}**" in s.readme   # bullets renamed with the columns
    assert a.sheets[0].rows == shop().sheets[0].rows   # values untouched


def test_rename_tables_updates_readme():
    s = shop()
    F.f_rename_tables(s, ctx("B"))
    assert {x.file for x in s.sheets} == {"export_a.csv", "export_b.csv"}
    assert "`export_a.csv`" in s.readme and "ban_hang" not in s.readme


def test_status_codes_legend():
    a, b = shop(), shop()
    F.f_status_codes(a, ctx("A"))
    F.f_status_codes(b, ctx("B"))
    assert [r[2] for r in a.sheets[0].rows] == ["1", "2", "1", "3"]   # code-point order: Xong, Đã hủy, Đổi trả
    assert all(r[2].startswith("TT") for r in b.sheets[0].rows)
    assert "2 = Đã hủy" in a.readme and "TT01 = Xong" in b.readme


def test_split_months_keeps_rows_and_names_are_periods():
    for side, name in (("A", "T6-2025"), ("B", "Thang 06.2025")):
        s = shop()
        assert F.f_split_months(s, ctx(side))
        months = [x for x in s.sheets if x.file == "ban_hang.xlsx"]
        assert [x.sheet for x in months][0] == name
        assert all(sheet_period(x.sheet) for x in months)
        assert sum(len(x.rows) for x in months) == 4
        assert "`ban_hang.xlsx`" in s.readme


def test_status_newword_only_cancelled():
    s = shop()
    assert F.f_status_newword(s, ctx("A", cancelled_raw={"Đã hủy"}))
    assert [r[2] for r in s.sheets[0].rows] == ["Xong", "Bom hàng", "Xong", "Đổi trả"]
    assert "'Bom hàng' nghĩa là đơn đã huỷ" in s.readme
    assert not F.f_status_newword(shop(), ctx("A"))   # nothing known as cancelled: not applicable


def test_drop_table_removes_sheet_and_section():
    s = shop()
    assert F.f_drop_table(s, ctx("A", drop_kind="stock"))
    assert [x.file for x in s.sheets] == ["ban_hang.csv"]
    assert "ton_kho" not in s.readme and "ban_hang.csv" in s.readme


def test_noise_table_and_layout_and_values_apply():
    for f in (F.f_noise_table, F.f_layout, F.f_values, F.f_nodoc):
        for side in F.SIDES:
            assert f(shop(), ctx(side))
    s = shop()
    F.f_values(s, ctx("B"))
    assert s.sheets[0].rows[0][1] == "3/6/2025" and s.sheets[0].rows[0][3] == "120,000đ"


@needs_data
def test_rule_param_is_read_by_the_rule_engine():
    md = (TRAIN / "rulebook.md").read_text(encoding="utf8")
    base = from_rulebook(md)
    a, b = from_rulebook(F.rule_param(md, "A")), from_rulebook(F.rule_param(md, "B"))
    assert a.morning == (7 * 3600, 12 * 3600 + 59 * 60 + 59) and a.week_start == base.week_start
    assert b.week_start == 6 and b.tet_start == (25, 12) and b.tet_end == (7, 1) and b.morning == base.morning


@needs_data
def test_rule_exception_sides_use_different_rows():
    md = (TRAIN / "rulebook.md").read_text(encoding="utf8")
    for side in F.SIDES:
        out = F.rule_exception(md, side)
        assert all(sent in out for sent in F.EXCEPTIONS[side].values())
    assert not set(F.EXCEPTIONS["A"]) & set(F.EXCEPTIONS["B"])
    assert "revenue" in F.exception_ops("A") and "top_products" in F.exception_ops("B")


def test_same_answer_is_semantic():
    n = lambda v: {"type": "number", "answer": v}   # noqa: E731
    assert BM.same_answer(n(12), n(12.0))
    assert not BM.same_answer(n(12), n(13))
    assert BM.same_answer(None, {"type": "abstain"})
    assert not BM.same_answer({"type": "unanswerable"}, None)
    assert BM.same_answer({"type": "string", "answer": "Cà phê sữa"}, {"type": "string", "answer": "cà phê  sữa"})


def test_error_classes():
    assert BM.runtime_class({"status": "error", "result": "Traceback\nKeyError: 'order_id'"}) == "column"
    assert BM.runtime_class({"status": "error",
                             "result": "AttributeError: 'Helpers' object has no attribute 'find_units'"}) == "helper_hallucinated"
    assert BM.runtime_class({"status": "error",
                             "result": "TypeError: Helpers.in_period() takes 2 positional arguments but 3 were given"}) == "helper_signature"
    assert BM.runtime_class({"status": "timeout"}) == "timeout"
    assert BM.runtime_class({"status": "ok"}) is None
    # text-to-SQL agent (D26): DuckDB errors and its own sample statuses
    assert BM.runtime_class({"status": "error", "result": 'BinderException: Binder Error: Referenced column "rev" '
                                                         "not found in FROM clause!"}) == "column"
    assert BM.runtime_class({"status": "error",
                             "result": "CatalogException: Catalog Error: Table with name x does not exist!"}) == "column"
    assert BM.runtime_class({"status": "error", "result": 'ParserException: Parser Error: syntax error at "FROM"'}) == "syntax"
    assert BM.runtime_class({"status": "error", "result": "result does not fit TYPE number: 2 columns"}) == "type_shape"
    assert BM.runtime_class({"status": "empty", "result": "empty result"}) == "empty"
    assert BM.runtime_class({"status": "no_sql"}) == "no_sql"
    g = {"type": "number", "answer": 290}
    assert BM.semantic_class({"type": "number", "answer": 291}, g) == "near_miss"
    assert BM.semantic_class({"type": "number", "answer": 580}, g) == "unit_factor"
    assert BM.semantic_class({"type": "number", "answer": 664}, g) == "far"
    assert BM.semantic_class({"type": "unanswerable"}, g) == "false_unanswerable"


def _c(biz, s, gtype="number", row=None):
    return {"biz": biz, "s": s, "gold": {"type": gtype}, "row": row or {"type": "number", "answer": 1}, "source": "codegen"}


def test_metrics_definitions():
    cs = [_c("a", 1), _c("a", -1), _c("b", 0, row={"type": "abstain"}),
          _c("b", -1, "unanswerable"), _c("c", 1, "unanswerable", {"type": "unanswerable"})]
    m = BM.metrics(cs, B=200)
    assert (m["R"], m["W"], m["A"], m["n"], m["shops"]) == (2, 2, 1, 5, 3)
    assert m["PA"] == 0 and m["precision"] == 0.5 and m["coverage"] == 0.8
    assert m["answerable_recall"] == pytest.approx(1 / 3)          # 1 right of 3 answerable
    assert m["forced_answer_rate"] == 0.5                          # 1 of 2 unanswerable got a concrete answer
    lo, hi = m["PA_ci"]
    assert lo <= m["PA"] <= hi
    assert BM.metrics(cs, B=200)["PA_ci"] == m["PA_ci"]            # deterministic bootstrap


CELLS = ROOT / "eval" / "synth" / "bench_cells.json"


@pytest.mark.skipif(not CELLS.exists(), reason="benchmark not built")
def test_frozen_cells_respect_splits():
    d = json.loads(CELLS.read_text(encoding="utf8"))
    holdout = set(json.loads((ROOT / "eval" / "splits.json").read_text(encoding="utf8"))["holdout"])
    cfg = json.loads((ROOT / "eval" / "synth" / "splits_gen.json").read_text(encoding="utf8"))
    keys = {c["key"] for c in d["cells"]}
    for c in d["cells"]:
        assert c["biz"] not in holdout and c["biz"] in cfg[c["side"]]
        assert f"{c['qid']}|original" in keys                       # every family cell is paired with the original
        assert "question" not in c and "answer" not in c            # ids and labels only (public repo, D14)


def test_content_fp_ignores_xlsx_timestamps_and_sees_cell_changes(tmp_path):
    import time
    from eval.synth.make import _write_xlsx
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "README.md").write_text("# x\n", encoding="utf8")
    s = [Sheet("t.xlsx", "S", [], ["Mã", "Tiền"], [["D1", 1000], ["D2", 2000]])]
    _write_xlsx(a / "t.xlsx", s)
    time.sleep(1.1)                                   # docProps/core.xml timestamps differ
    _write_xlsx(b / "t.xlsx", s)
    assert BM._content_fp(a) == BM._content_fp(b)
    s[0].rows[1][1] = 2001
    _write_xlsx(b / "t.xlsx", s)
    assert BM._content_fp(a) != BM._content_fp(b)


@needs_data
def test_run_passes_the_agent_kind(monkeypatch, tmp_path):
    """`run --agent sql` builds the text-to-SQL agent (kind='sql'), `--agent codegen` the code-gen one."""
    import datamate.run as R
    seen = []
    monkeypatch.setattr(R, "make_agent", lambda llm, root, rb=None, enabled=None, kind=None: seen.append((enabled, kind)))
    monkeypatch.setattr(R, "make_llm", lambda *a: None)
    cell = {"qid": "q1", "biz": "biz_x", "side": "A", "family": "original", "label": "same", "root": "data/train/train",
            "rulebook": None}
    monkeypatch.setattr(BM, "load_cells", lambda check=True: ({"gen_hash": "x"}, [cell]))
    monkeypatch.setattr(BM, "load_jsonl", lambda p: [{"id": "q1", "business_id": "biz_x", "question": "?"}])
    monkeypatch.setattr("datamate.pipeline.answer_many", lambda qs, ctx: iter(()))
    for agent, want in (("sql", (True, "sql")), ("codegen", (True, "codegen")), ("off", (False, None))):
        seen.clear()
        args = type("A", (), {"side": None, "families": None, "shops": None, "out": str(tmp_path / agent),
                              "force": True, "llm": None, "llm_model": None, "llm_replay": None, "agent": agent,
                              "arm": "t"})()
        BM.run(args)
        assert seen == [want]


def test_repair_name_undoes_unzip_codepage_and_nfd():
    import unicodedata
    good = "Bảng giá tháng.xlsx"
    assert BM._repair_name(good) == good
    assert BM._repair_name("Bán.csv") == "Bán.csv"   # latin-1 encodable but valid: unchanged
    assert BM._repair_name(good.encode("utf8").decode("cp437")) == good
    assert BM._repair_name(good.encode("utf8").decode("latin-1")) == good
    assert BM._repair_name(unicodedata.normalize("NFD", good)) == good


def test_run_skip_bad_leaves_out_only_the_differing_variant(monkeypatch, tmp_path):
    """A variant that differs from the seal stops `run` up front; with --skip-bad only its cells are left out."""
    import datamate.run as R
    monkeypatch.setattr(R, "make_agent", lambda *a, **k: None)
    monkeypatch.setattr(R, "make_llm", lambda *a: None)
    root = str(tmp_path / "layout")
    (tmp_path / "layout").mkdir()
    cells = [{"key": f"q{i}|layout", "qid": f"q{i}", "biz": biz, "side": "B", "family": "layout", "label": "same",
              "root": root, "rulebook": rb} for i, biz, rb in ((1, "biz_1", None), (2, "biz_2", None), (3, "biz_2", "x.md"))]
    monkeypatch.setattr(BM, "load_cells", lambda check=True: ({"gen_hash": "x"}, cells))
    monkeypatch.setattr(BM, "load_jsonl", lambda p: [{"id": c["qid"], "business_id": c["biz"]} for c in cells])
    monkeypatch.setattr(BM, "verify_content", lambda meta, keys: [k for k in keys if k.endswith("biz_1")])
    asked = []
    monkeypatch.setattr("datamate.pipeline.answer_many", lambda qs, ctx: asked.extend(q["id"] for q in qs) or iter(()))
    args = type("A", (), {"side": None, "families": None, "shops": None, "out": str(tmp_path / "run"), "force": False,
                          "skip_bad": False, "llm": None, "llm_model": None, "llm_replay": None, "agent": "off",
                          "arm": "t"})()
    with pytest.raises(SystemExit):
        BM.run(args)
    assert asked == []
    args.skip_bad = True
    BM.run(args)
    assert asked == ["q2", "q3"]   # groups with and without a rulebook (sorting them once raised TypeError)
    assert json.loads((tmp_path / "run" / "meta.json").read_text(encoding="utf8"))["skipped_cells"] == ["q1|layout"]


def test_content_fp_file_order_is_case_insensitive(tmp_path):
    """The seal was hashed on Windows (case-insensitive order); Linux must hash the files in the same order."""
    import hashlib
    for n in ("thanh_toan.csv", "README.md", "a.csv", "Z.csv"):
        (tmp_path / n).write_text(n, encoding="utf8")
    assert [p.name for p in BM._files(tmp_path)] == ["a.csv", "README.md", "thanh_toan.csv", "Z.csv"]
    h = hashlib.sha1()
    for n in ("a.csv", "README.md", "thanh_toan.csv", "Z.csv"):
        h.update(n.encode())
        h.update(n.encode())
    assert BM._content_fp(tmp_path) == h.hexdigest()[:16]
