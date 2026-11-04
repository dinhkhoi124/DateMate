"""D10 coverage gate: unfamiliar wording -> abstain; entities, time and numbers never count as unfamiliar."""
from datetime import date
from types import SimpleNamespace

from datamate.config import DEFAULT
from datamate.link.catalog import Dictionary
from datamate.parse.gate import residual, unknown
from datamate.timeparse.periods import LunarCalendar, find_periods


def _lk():
    e = Dictionary([])
    return SimpleNamespace(products=Dictionary([("Cà phê sữa đá", "P1")]), customers=Dictionary([("Chị Lan", "C1")]),
                           staff=e, categories=e, ingredients=e, suppliers=e)


def _words(t):
    cal = LunarCalendar.load("data/train/train/calendar.csv")
    return residual(t, None, _lk(), find_periods(t, date(2026, 5, 20), cal, DEFAULT))


def test_entities_time_numbers_masked():
    w = _words("thang 3 nam 2026 ca phe sua da ban duoc bao nhieu ly")
    assert "phe" not in w and "da" not in w and "nam" not in w
    assert {"ban", "duoc", "bao", "nhieu", "ly"} <= set(w)


def test_unknown_word_flagged():
    vocab = frozenset({"khach", "nao", "mua", "nhieu", "tien", "nhat", "thang", "nay"})
    assert unknown(_words("khach nao mua nhieu tien nhat thang nay"), vocab) == []
    assert unknown(_words("khach nao sop nhat thang nay"), vocab) == ["sop"]


def test_no_vocab_means_no_gate():
    assert unknown(["bat", "ky"], None) == []
