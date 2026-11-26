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


def test_slim_drops_machine_paths_and_review_records():
    p = {"id": "x", "meta": {"sources": {"paper": {"what": "/Users/someone/work/x/paper.pdf (sha256 ab)"}}},
         "run_excluded": [{"arm": "a", "why": "not printed", "decision": "converter 2026-11-08: rule 2"}],
         "coding": {"q": {"rule": "r", "decision": "coordinator decision", "heldout_probes": "review/h.json"}}}
    out = rc.slim(p)
    assert out["meta"]["sources"]["paper"]["what"] == "paper.pdf (sha256 ab)"
    assert out["run_excluded"] == [{"arm": "a", "why": "not printed"}]
    assert out["coding"] == {"q": {"rule": "r"}}
    assert rc.local_paths({"a.json": b'{"p": "/Users/someone/x"}', "b.json": b"{}"}) == ["a.json"]


def test_maintainer_prose_loses_internal_references_but_not_participant_text():
    p = {"id": "x", "meta": {"sources": {"paper": {"what": "text layer (sourcetext/paper.txt)",
                                                   "licence": "CC BY (coordinator instruction, fidelity review 1)"}}},
         "stages": {"s": {"text": "Talk to the coordinator.",
                          "presentation_cite": {"note": "page break (skip logic; LESSONS 38; flow_cites.x)"}}},
         "analysis": {"unsupported": [{"why": "kept in review/m2-faithful-variant.json, so not run"}]}}
    out = rc.slim(p)
    assert out["meta"]["sources"]["paper"]["what"] == "text layer (paper.txt)"
    assert out["meta"]["sources"]["paper"]["licence"] == "CC BY (maintainer instruction, fidelity review 1)"
    assert out["stages"]["s"]["text"] == "Talk to the coordinator."   # participant wording untouched
    assert out["stages"]["s"]["presentation_cite"]["note"] == "page break (skip logic; flow_cites.x)"
    assert out["analysis"]["unsupported"][0]["why"] == "kept in the maintainers' review record, so not run"


def test_withheld_reasons_become_readable():
    p = {"run_excluded": [{"arm": "a", "why": "withheld:4be71abac526d72a"}, {"arm": "b", "why": "not printed"}],
         "omitted": [{"what": "withheld:7b5d", "why": "withheld:72678f4b"}, {"what": "debriefing", "why": "not reported"}],
         "data": {"q": {"cite": {"note": "withheld:aa11"}}}}
    assert rc.readable_reasons(p) == 3
    assert not p["run_excluded"][0]["why"].startswith("withheld:") and p["run_excluded"][1]["why"] == "not printed"
    assert not p["omitted"][0]["what"].startswith("withheld:") and p["omitted"][1]["what"] == "debriefing"
    assert p["data"]["q"]["cite"]["note"] == "withheld:aa11"   # a cite locator keeps its token
