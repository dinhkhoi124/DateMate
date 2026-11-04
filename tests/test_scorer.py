from eval.scorer import evaluate, score


def g(t, a, alias=None):
    return {"id": "q", "type": t, "answer": a, "alias": alias or []}


def p(t, a=None):
    r = {"id": "q", "type": t}
    if a is not None:
        r["answer"] = a
    return r


def test_number_exact_and_format():
    assert score(p("number", 4820000), g("number", 4820000)) == 1
    assert score(p("number", 4820000.0000001), g("number", 4820000)) == 1
    assert score(p("number", 4820001), g("number", 4820000)) == -1
    assert score(p("number", "4820000"), g("number", 4820000)) == -1  # string is wrong format
    assert score(p("number", True), g("number", 1)) == -1


def test_percent_two_decimals():
    assert score(p("percent", 12.4), g("percent", 12.4)) == 1
    assert score(p("percent", 0.124), g("percent", 12.4)) == -1


def test_type_mismatch_and_abstain():
    assert score(p("percent", 12.4), g("number", 12.4)) == -1
    assert score(p("abstain"), g("number", 1)) == 0
    assert score(None, g("number", 1)) == 0
    assert score(p("unanswerable"), g("unanswerable", None)) == 1
    assert score(p("unanswerable"), g("number", 1)) == -1


def test_string_case_punct_alias_diacritics():
    gold = g("string", "Costar Omega 3 1000mg lọ 100 viên", ["SP0076", "Costar Omega 3 1000 mg lọ 100 viên"])
    assert score(p("string", "costar omega 3 1000mg  lọ 100 viên"), gold) == 1
    assert score(p("string", "SP0076"), gold) == 1
    assert score(p("string", "Costar Omega 3 1000mg lo 100 vien"), gold) == -1  # wrong diacritics
    assert score(p("string", ["Costar Omega 3 1000mg lọ 100 viên"]), gold) == -1


def test_list_order_and_alias():
    gold = g("list", ["Cà phê sữa (M)", "Cà phê đen (M)"], [["SP0003"], ["SP0001"]])
    assert score(p("list", ["SP0003", "cà phê đen (m)"]), gold) == 1
    assert score(p("list", ["Cà phê đen (M)", "Cà phê sữa (M)"]), gold) == -1
    assert score(p("list", [3, 10]), g("list", [3, 10])) == 1


def test_table_rows_unordered_columns_by_name():
    gold = g("table", {"columns": ["khach_hang", "so_tien_no"], "rows": [["A", 10], ["B", 5]]}, [["KH1"], ["KH2"]])
    assert score(p("table", {"columns": ["khach_hang", "so_tien_no"], "rows": [["KH2", 5], ["A", 10]]}), gold) == 1
    assert score(p("table", {"columns": ["so_tien_no", "khach_hang"], "rows": [[10, "A"], [5, "B"]]}), gold) == 1
    assert score(p("table", {"columns": ["khach_hang", "so_tien_no"], "rows": [["A", 10], ["B", 6]]}), gold) == -1


def test_evaluate_pa():
    golds = [{"id": str(i), "type": "number", "answer": i, "alias": []} for i in range(4)]
    preds = [{"id": "0", "type": "number", "answer": 0}, {"id": "1", "type": "number", "answer": 9},
             {"id": "2", "type": "abstain"}]
    r = evaluate(preds, golds)
    assert (r["correct"], r["wrong"], r["abstain"]) == (1, 1, 2) and r["PA"] == 0.0
