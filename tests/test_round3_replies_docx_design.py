"""Round-3 build fixes: amounts, number words and printed typos in replies; Word-export text; design disclosures.

Option sets are taken from the round-3 packages (Biziou & van Pol 2015 white lies, Pennycook et al. 2015
pseudo-profound bullshit, Xiao et al. 2021 status quo bias).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

from package_fixture import REPLIES, write_runnable

spec = importlib.util.spec_from_file_location("replies_round3", REPLIES)
replies = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replies
spec.loader.exec_module(replies)
Question, parse, ask, render_questions = replies.Question, replies.parse, replies.ask, replies.render_questions

from agent_pipeline import design_checks, docx_text, estimate_run, smoke_test  # noqa: E402
from agent_pipeline.check_sources import file_text  # noqa: E402

BIZIOU_DG = [f"{n}c" for n in range(0, 21, 2)]          # "1. 0c ... 6. 10c ... 10. 18c 11. 20c"
BIZIOU_PD = ["Tranfer the 10c", "Do not trasfer the 10c"]  # printed typos, kept verbatim
PENNYCOOK = ["Not at all profound", "somewhat profound", "fairly profound", "definitely profound", "very profound"]
XIAO_QUIZ4 = ["70% for auto safety and 30% for highway safety", "60% for auto safety and 40% for highway safety",
              "30% for auto safety and 70% for highway safety", "50% for auto safety and 50% for highway safety"]


def read(reply: str, question: Question):
    answer = parse(reply, question)
    return answer.get("label") if answer["status"] == "ok" else answer["status"]


# --------------------------------------------------------------------------- H3: amounts

@pytest.mark.parametrize("reply,expected", [
    ("10", "ambiguous"),           # option 10 (18c) or the amount 10c: never silently 18c
    ("4", "ambiguous"),            # option 4 (6c) or 4c
    ("10 because it is fair", "ambiguous"),
    ("I give 10", "ambiguous"),
    ("ten", "ambiguous"),
    ("0", "0c"), ("20", "20c"),   # amounts that are no option number
    ("1", "0c"), ("2", "2c"),     # an option number that is no other option's amount (or the same option's)
    ("10c", "10c"), ("10 cents", "10c"), ("$0.10", "10c"), ("I will transfer 10c", "10c"),
    ("6. 10c", "10c"), ("10. 18c", "18c"), ("4. 4c", "ambiguous"),
    ("I give 10c of my 20c", "ambiguous"), ("15c", "unparsed"), ("Not 10c", "ambiguous"),
])
def test_amount_list_reads_amounts_and_never_guesses(reply: str, expected: str) -> None:
    assert read(reply, Question("dg", options=BIZIOU_DG)) == expected


def test_amount_list_asks_for_the_number_and_the_option() -> None:
    q = Question("dg", options=BIZIOU_DG)
    assert q.codes == [str(n) for n in range(1, 12)] and q.amount_clash()
    rendered = render_questions([q])
    assert "6. 10c" in rendered and "followed by the option as printed" in rendered
    # No clash, no change: positions 1-3 are no amount on this list.
    plain = Question("p", options=["8%", "65%", "70%"])
    assert not plain.amount_clash() and render_questions([plain]).endswith("Reply with the number of the option you choose.")
    assert read("65", plain) == "65%" and read("2", plain) == "65%"


def test_ambiguous_reply_is_asked_again_with_the_reason() -> None:
    prompts = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "10" if len(prompts) == 1 else "6. 10c"

    q = Question("dg", options=BIZIOU_DG)
    page = ask(llm, render_questions([q]), [q])
    assert page.ok and page.value("dg") == "10c" and len(prompts) == 2
    assert "could be read as more than one option" in prompts[1]


def test_g7_percent_and_amount_leading_labels_are_still_not_codes() -> None:
    q = Question("quiz", options=XIAO_QUIZ4)
    assert q.codes == ["1", "2", "3", "4"] and not q.amount_clash()
    assert read("4", q) == XIAO_QUIZ4[3] and read("70", q) == "unparsed"
    scale = Question("s", options=["1 - Not at all", "2", "3", "4", "5 - Very much"])
    assert scale.codes == ["1", "2", "3", "4", "5"] and read("3", scale) == "3"
    hours = Question("h", options=["30 hours", "90 hours"])
    assert hours.codes == ["1", "2"] and read("90", hours) == "90 hours" and read("1", hours) == "30 hours"
    mixed = Question("spend", options=["Nothing", "US$90", "30 hours", "US$30", "90 hours"])
    assert read("90", mixed) == "ambiguous" and read("$90", mixed) == "US$90" and read("90 hours", mixed) == "90 hours"


# --------------------------------------------------------------------------- H3: number words

@pytest.mark.parametrize("reply,expected", [
    ("Three", "fairly profound"), ("three.", "fairly profound"), ("Answer: Five", "very profound"),
    ("Three (fairly profound)", "fairly profound"), ("3", "fairly profound"),
    ("One of those", "unparsed"),               # a number word in prose is not an answer
    ("Three (or maybe four)", "ambiguous"),
    ("Three, maybe four", "unparsed"),
])
def test_number_words_on_a_scale(reply: str, expected: str) -> None:
    assert read(reply, Question("rate", options=PENNYCOOK)) == expected


def test_number_words_stay_words_when_a_label_uses_them() -> None:
    q = Question("f", options=["None", "One", "Two or more"])
    assert read("One", q) == "One" and read("Two", q) == "unparsed"   # never "Two" -> option 2 ("One")
    assert parse("Five", Question("n", kind="number", minimum=0, maximum=10))["value"] == 5
    assert parse("five or six", Question("n", kind="number", minimum=0, maximum=10))["status"] == "unparsed"


# --------------------------------------------------------------------------- H3: printed typos

@pytest.mark.parametrize("reply,expected", [
    ("Transfer the 10c", "Tranfer the 10c"), ("Do not transfer the 10c", "Do not trasfer the 10c"),
    ("1. Transfer the 10c", "Tranfer the 10c"), ("2. Transfer the 10c", "ambiguous"),
    ("Do transfer the 10c", "unparsed"), ("Transfer the 20c", "unparsed"),   # numbers must be identical
    ("Transfer", "unparsed"),   # a shortened label needs an alias
])
def test_correct_spelling_of_a_printed_typo(reply: str, expected: str) -> None:
    assert read(reply, Question("pd", options=BIZIOU_PD)) == expected


def test_near_matches_never_change_meaning_or_guess() -> None:
    # Xiao phase 2 printed "Increase" where phase 1 had "Decrease": a different word, not a typo.
    q = Question("sq", options=["Increase auto program by 20% and lower highway program by like amount",
                                "Retain the auto program at 50%"])
    assert read("Decrease auto program by 20% and lower highway program by like amount", q) == "unparsed"
    assert read("atypical outcome", Question("t", options=["typical outcome", "rare outcome"])) == "unparsed"
    assert read("likely", Question("l", options=["unlikely", "certain"])) == "unparsed"
    # One letter from two labels: not read.
    assert read("Transfer it", Question("two", options=["Tranfer it", "Trasnfer it"])) == "ambiguous" or \
        read("Transfer it", Question("two", options=["Tranfer it", "Trasnfer it"])) == "unparsed"
    assert read("Transfer it", Question("two", options=["Tranfer it", "Trasnfer it"])) != "Tranfer it"
    # Short words are not corrected ("lose"/"lost").
    assert read("lost", Question("w", options=["lose", "win"])) == "unparsed"


def test_aliases_are_read_never_shown_and_never_shared() -> None:
    q = Question("pd", options=BIZIOU_PD, aliases={"Tranfer the 10c": ["Transfer"], 1: ["Do not transfer"]})
    assert read("Transfer", q) == BIZIOU_PD[0] and read("Do not transfer", q) == BIZIOU_PD[1]
    assert read("I choose Transfer", q) == BIZIOU_PD[0] and read("2. Transfer", q) == "ambiguous"
    assert "Transfer\n" not in render_questions([q]) + "\n" and "Do not transfer" not in render_questions([q])
    with pytest.raises(ValueError):
        Question("pd", options=BIZIOU_PD, aliases={0: ["Do not trasfer the 10c"]})
    with pytest.raises(ValueError):
        Question("pd", options=BIZIOU_PD, aliases={"Cooperate": ["x"]})


def test_stub_tries_amounts_number_words_and_aliases() -> None:
    stub = replies.StubParticipant(3, noise=0, use_keys=False)
    questions = [Question("dg", options=BIZIOU_DG), Question("rate", options=PENNYCOOK),
                 Question("pd", options=BIZIOU_PD, aliases={0: ["Transfer"], 1: ["Do not transfer"]})]
    for q in questions:
        for _ in range(6):
            assert parse(stub.answer_for(render_questions([q]), [q]), q)["status"] == "ok"
    assert stub.probed == {"dg", "rate", "pd"} and stub.misreads == []
    assert "dg" in stub.ambiguous       # a bare option number on this list is asked again, and reported


def test_sessions_for_and_paper_cell_sizes(tmp_path: Path) -> None:
    assert replies.sessions_for(4, "a") == 4
    assert replies.sessions_for({"a": 357, "b": 257}, "b") == 257 and replies.sessions_for({"a": 1}, "b") == 0
    root = tmp_path / "package" / "paper"
    write_runnable(root)
    task = json.loads((root / "task/task.json").read_text())
    task["conditions"] = [{"id": "control", "paper_n": 357}, {"id": "treatment", "paper_n": 257}]
    (root / "task/task.json").write_text(json.dumps(task))
    assert estimate_run.total_sessions(root, "paper") == 614
    assert estimate_run.total_sessions(root, "control=357,treatment=257") == 614
    report = smoke_test.run(root)    # the fixture adapter does range(n): a dict n raises
    assert any("n may be a dict of cell sizes" in p for p in report["problems"]), report["problems"]
    adapter = (root / "task/adapter.py").read_text().replace(
        "from replies import Question, ask, render_questions, smoke_main",
        "from replies import Question, ask, render_questions, sessions_for, smoke_main").replace(
        "for index in range(n):", "for index in range(sessions_for(n, arm)):")
    (root / "task/adapter.py").write_text(adapter)
    report = smoke_test.run(root)
    assert report["ok"], report["problems"]


def test_playground_reads_per_condition_sizes(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "playground"))
    import run_package  # noqa: E402
    (tmp_path / "task").mkdir()
    (tmp_path / "task/task.json").write_text(json.dumps({"conditions": [{"id": "a", "paper_n": 3}, {"id": "b"}]}))
    assert run_package.per_condition_sizes(tmp_path, {"paperCellSizes": True}) == {"a": 3}
    assert run_package.per_condition_sizes(tmp_path, {"participantsPerCondition": {"a": 2, "b": 0}}) == {"a": 2}
    assert run_package.per_condition_sizes(tmp_path, {"participantsPerScenario": 8}) is None


# --------------------------------------------------------------------------- H4: Word exports

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _docx(path: Path, body: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", f"<w:document {W}><w:body>{body}</w:body></w:document>")
    return path


def _p(*runs: str, ppr: str = "") -> str:
    return f"<w:p>{ppr}" + "".join(runs) + "</w:p>"


def _r(text: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>'


BR = "<w:r><w:br/></w:r>"
QUALTRICS_PAGE_BREAK = ('<w:tbl><w:tr><w:tc><w:p><w:pPr><w:rPr><w:color w:val="cccccc"/></w:rPr></w:pPr>'
                        '<w:r><w:t>Page Break</w:t></w:r></w:p></w:tc><w:tc><w:p/></w:tc></w:tr></w:tbl>')


def test_docx_reader_keeps_breaks_and_ends_pages_at_page_break_markers(tmp_path: Path) -> None:
    body = (_p(_r("Start of Block: Scenario 1")) + _p(_r("sc1 Scenario 1"), BR, BR, _r("In this study we ask."))
            + _p(_r("1) Improving automobile safety"), BR, _r("2) Improving highways"))
            + _p(_r("Which car was available?"), ppr='<w:pPr><w:tabs><w:tab w:val="left" w:pos="720"/></w:tabs></w:pPr>')
            + QUALTRICS_PAGE_BREAK + _p(_r("Q &amp; A"), "<w:r><w:br w:type=\"page\"/></w:r>", _r("Last page"))
            + _p(_r("Thanks"), ppr="<w:pPr><w:pageBreakBefore/></w:pPr>"))
    record = docx_text.read(_docx(tmp_path / "s.docx", body))
    assert record["pages"] == [["sc1 Scenario 1", "In this study we ask.", "1) Improving automobile safety",
                                "2) Improving highways", "Which car was available?"], ["Q & A"], ["Last page"],
                               ["Thanks"]]
    assert record["export_lines"] == ["Start of Block: Scenario 1"]
    text = file_text(tmp_path / "s.docx")
    assert "Page Break" not in text and "Scenario 1 In this study" in text and "Scenario 1In" not in text


def _visible_package(tmp_path: Path, screen: str, docx_body: str | None = None) -> tuple[Path, Path]:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    for folder in ("materials", "source"):
        (root / folder).mkdir(parents=True)
    (root / "materials/materials.json").write_text(json.dumps({"screens": [{"id": "s1", "text": screen,
                                                                             "note": "Page Break line dropped"}]}))
    sources = []
    if docx_body is not None:
        (job / "input/external").mkdir(parents=True)
        _docx(job / "input/external/survey.docx", docx_body)
        sources = [{"id": "osf-docx", "path": "input/external/survey.docx"}]
    (root / "source/paper_metadata.json").write_text(json.dumps({"external_sources": sources}))
    return root, job


def test_export_markup_and_lost_breaks_in_participant_text_fail(tmp_path: Path) -> None:
    root, job = _visible_package(tmp_path / "a", "Which car was available? Page Break")
    problems = docx_text.check(root, job)
    assert any("'Page Break'" in p for p in problems), problems
    probe = {"runs": [{"calls": [{"prompt": "Start of Block: Intro\nWelcome"}]}]}
    root, job = _visible_package(tmp_path / "b", "Welcome")
    assert any("start of block:" in p.lower() for p in docx_text.check(root, job, probe))
    body = _p(_r("Scenario 1"), BR, BR, _r("In this study we are interested"))
    root, job = _visible_package(tmp_path / "c", "Scenario 1In this study we are interested", body)
    assert any("runs two lines" in p and "Scenario 1In this" in p for p in docx_text.check(root, job))
    root, job = _visible_package(tmp_path / "d", "Scenario 1\n\nIn this study we are interested", body)
    assert docx_text.check(root, job) == []   # the note mentioning "Page Break" is not participant text


# --------------------------------------------------------------------------- H5: design and reply checks

def _design_package(tmp_path: Path, *, task: dict, audit: list | None = None, reply: str = "",
                    issues: list | None = None, metadata: dict | None = None) -> tuple[Path, Path]:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    for folder in ("task", "audit", "source"):
        (root / folder).mkdir(parents=True)
    (job / "input").mkdir(parents=True)
    (job / "input/paper.pdf").write_bytes(b"%PDF-1.4")
    (root / "task/task.json").write_text(json.dumps(task))
    (root / "audit/missing_information.json").write_text(json.dumps({"entries": audit or []}))
    (root / "study.json").write_text(json.dumps({"program": {"issues": issues or []}}))
    (root / "source/paper_metadata.json").write_text(json.dumps(metadata or {}))
    if reply:
        (root / "studio-reply.md").write_text(reply)
    return root, job


REPORTED = {"layout": "one statement per screen", "origin": "reported", "evidence": "p. 4"}


def test_licence_question_is_a_hub_policy_item(tmp_path: Path) -> None:
    reply = ("Ready to run: Study 1\nNeeds you: two questions\nLicence: CC BY 3.0 (printed on page 1)\n\n"
             "1. Can you upload the consent screen? If not, it stays out.\n"
             "2. May the item lists from the journal supplement be published? Until then it stays private.\n")
    root, job = _design_package(tmp_path, task={"conditions": [], "presentation": REPORTED}, reply=reply)
    problems = design_checks.check(root, job)
    assert len(problems) == 1 and "Hub policy item" in problems[0] and "May the item lists" in problems[0]


def test_blocked_condition_names_what_is_missing_and_what_it_blocks(tmp_path: Path) -> None:
    vague = {"id": "study3", "status": "blocked", "blocked_reason": "The instruction and scale are not printed."}
    root, job = _design_package(tmp_path / "a", task={"conditions": [vague], "presentation": REPORTED})
    assert any("study3" in p and "blocked_missing" in p for p in design_checks.check(root, job))
    exact = dict(vague, blocked_reason="The Study 3 rating instruction is not printed.",
                 blocked_missing=["Study 3 rating instruction"],
                 blocked_results=["bullshit sensitivity (profound minus mundane ratings)"])
    root, job = _design_package(tmp_path / "b", task={"conditions": [exact], "presentation": REPORTED})
    assert design_checks.check(root, job) == []


def test_layout_the_sources_do_not_state_is_a_departure(tmp_path: Path) -> None:
    derived = {"layout": "one statement per screen", "origin": "derived"}
    root, job = _design_package(tmp_path / "a", task={"conditions": []})
    assert any("needs 'presentation'" in p for p in design_checks.check(root, job))
    root, job = _design_package(tmp_path / "b", task={"conditions": [], "presentation": derived})
    assert any("presentation.layout" in p for p in design_checks.check(root, job))
    audit = [{"study": "1", "field": "presentation.layout", "reason": "not stated", "impact": "small",
              "suggested_action": "confirm"}]
    root, job = _design_package(tmp_path / "c", task={"conditions": [], "presentation": derived}, audit=audit)
    assert design_checks.check(root, job) == []


def test_pending_materials_need_the_minimum_search_list(tmp_path: Path) -> None:
    issues = [{"id": "materials-request", "severity": "decision", "title": "Materials"}]
    searches = [{"where": "journal supplementary material", "url": "https://www.cambridge.org/core/x"},
                {"where": "OSF", "url": "https://api.osf.io/v2/nodes/?filter[title]=white%20lies"}]
    metadata = {"journal": "Judgment and Decision Making", "material_searches": searches,
                "researcher_questions": [{"issue": "materials-request", "sources_checked": []}]}
    root, job = _design_package(tmp_path / "a", task={"conditions": [], "presentation": REPORTED},
                                issues=issues, metadata=metadata)
    problem = next(p for p in design_checks.check(root, job) if "material_searches" in p)
    for kind in ("preprint", "Unpaywall or Europe PMC", "OSF search by author", "journal.sjdm.org"):
        assert kind in problem
    searches += [{"where": "OSF Preprints", "url": "https://api.osf.io/v2/preprints/?filter[title]=white"},
                 {"where": "Europe PMC", "url": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI"},
                 {"where": "OSF users (author search)", "url": "https://api.osf.io/v2/users/?filter[full_name]=Biziou"},
                 {"where": "JDM original site", "url": "https://journal.sjdm.org/15/15415a/"}]
    root, job = _design_package(tmp_path / "b", task={"conditions": [], "presentation": REPORTED},
                                issues=issues, metadata=metadata)
    assert design_checks.check(root, job) == []


def test_questions_are_checked_against_the_paper_first(tmp_path: Path) -> None:
    issues = [{"id": "dv-units", "severity": "decision", "title": "Are the coefficients in cents?"}]
    checked = {"paths": ["input/external/si.pdf"], "answers": False, "found": "the SI does not say"}
    metadata = {"researcher_questions": [{"issue": "dv-units", "sources_checked": [checked]}]}
    root, job = _design_package(tmp_path / "a", task={"conditions": [], "presentation": REPORTED},
                                issues=issues, metadata=metadata)
    assert any("'dv-units'" in p and "arithmetic" in p for p in design_checks.check(root, job))
    metadata["researcher_questions"][0]["sources_checked"].append(
        {"paths": ["input/paper.pdf"], "answers": False, "found": "the paper does not state the unit"})
    root, job = _design_package(tmp_path / "b", task={"conditions": [], "presentation": REPORTED},
                                issues=issues, metadata=metadata)
    assert design_checks.check(root, job) == []
