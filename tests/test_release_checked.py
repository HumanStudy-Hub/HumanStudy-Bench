"""Release rules of scripts/research/release_checked.py (owner 2026-11-24): short labels ship, operator notes are
dropped, undeclared cited sources become researcher-supplied. Invented data only."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("hsenv")
pytest.importorskip("pypdfium2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.research import release_checked as rc  # noqa: E402


def test_word_count_of_a_label():
    assert rc.nwords("Strongly agree") == 2
    assert rc.nwords("850 to 950 boys") == 4
    assert rc.nwords("one two three four five six") == 6
    assert rc.nwords(None) > rc.SHORT_WORDS


def test_slim_drops_operator_notes_and_declares_cited_sources():
    p = {"id": "x", "exclusions_note": "a private note", "deviations": ["d"],
         "meta": {"citation": "C", "conflicts": ["c"], "paper_inconsistencies": [{"code": "q"}],
                  "sources": {"paper": {"what": "Paper", "note": "n", "sha256": "ab", "distribution": {"tier": "user"}},
                              "root": "stage1/x"}},
         "data": {"q": {"text": "Café", "cite": {"label": "verbatim", "source": "qsf"}}}}
    out = rc.slim(p)
    assert "exclusions_note" not in out and "deviations" not in out
    assert set(out["meta"]) == {"citation", "sources"}
    assert out["meta"]["sources"]["paper"] == {"what": "Paper", "sha256": "ab", "distribution": {"tier": "user"}}
    assert "root" not in out["meta"]["sources"]
    assert out["meta"]["sources"]["qsf"]["distribution"] == {"tier": "user"}
    assert out["data"]["q"]["text"] == "Café"   # NFC


def test_short_held_text_ships_and_long_stays_held():
    private = {"data": {"a": {"text": "Strongly agree"}, "b": {"text": "a much longer sentence of the source text"}}}
    projected = {"data": {"a": {"text": "░" * 14}, "b": {"text": "░" * 41}},
                 "distribution": {"held": {"h1": {"path": "/data/a/text", "kind": "text"},
                                           "h2": {"path": "/data/b/text", "kind": "text"}}}}
    assert rc.unhold_short(projected, private) == 1
    assert projected["data"]["a"]["text"] == "Strongly agree"
    assert set(projected["distribution"]["held"]) == {"h2"}
    assert set(projected["data"]["b"]["text"]) == {"░"}


def test_long_uncited_screen_text_still_blocks():
    private = {"data": {"a": "Set 1", "b": "a sentence that is clearly longer than five words"}}
    report = {"errors": [], "uncited_source_text": [{"path": "/data/a", "source": "paper"},
                                                    {"path": "/data/b", "source": "paper"}]}
    out = rc.blocking_errors(report, {"data": dict(private["data"]), "distribution": {"held": {}}}, private)
    assert len(out) == 1 and "/data/b" in out[0] and "/data/a" not in out[0]
