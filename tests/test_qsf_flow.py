"""Tests for the Qualtrics survey-flow reader (agent_pipeline/qsf_flow.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_pipeline import qsf_flow
from agent_pipeline.repair_guard import compare_packages


def mc(qid: str, text: str, choices: list[str], order=None, **extra) -> dict:
    payload = {
        "QuestionID": qid, "DataExportTag": qid.lower(), "QuestionType": "MC", "Selector": "SAVR",
        "SubSelector": "TX", "QuestionText": text,
        "Choices": {str(i + 1): {"Display": c} for i, c in enumerate(choices)},
        "ChoiceOrder": order or [str(i + 1) for i in range(len(choices))],
        "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
    }
    payload.update(extra)
    return payload


def db(qid: str, text: str, **extra) -> dict:
    payload = {"QuestionID": qid, "DataExportTag": qid.lower(), "QuestionType": "DB", "Selector": "TB",
               "QuestionText": text, "Validation": {"Settings": {"Type": "None"}}}
    payload.update(extra)
    return payload


def selected(qid: str, choice: str, conj: str | None = None) -> dict:
    expr = {"LogicType": "Question", "QuestionID": qid, "Operator": "Selected",
            "LeftOperand": f"q://{qid}/SelectableChoice/{choice}", "Type": "Expression",
            "Description": "<span>ignored</span>"}
    if conj:
        expr["Conjuction"] = conj
    return expr


def block(bid: str, description: str, elements: list, kind: str = "Standard", options=None) -> dict:
    return {"ID": bid, "Type": kind, "Description": description, "Options": options or {},
            "BlockElements": [{"Type": "Page Break"} if e == "|" else {"Type": "Question", "QuestionID": e}
                              for e in elements]}


def survey(questions: list[dict], blocks: list[dict], flow: list[dict] | None, blocks_as_dict: bool = True) -> dict:
    elements = [{"Element": "SQ", "PrimaryAttribute": q["QuestionID"], "Payload": q} for q in questions]
    payload = {str(i): b for i, b in enumerate(blocks)} if blocks_as_dict else blocks
    elements.append({"Element": "BL", "Payload": payload})
    if flow is not None:
        elements.append({"Element": "FL", "Payload": {"Type": "Root", "FlowID": "FL_1", "Flow": flow}})
    elements.append({"Element": "SO", "Payload": {"BackButton": "false"}})
    return {"SurveyEntry": {"SurveyID": "SV_1", "SurveyName": "Fixture"}, "SurveyElements": elements}


CORRECT = {"Settings": {"ForceResponse": "ON", "ForceResponseType": "ON", "Type": "CustomValidation",
                        "CustomValidation": {"Logic": {"0": {"0": selected("QID3", "2"), "Type": "If"},
                                                       "Type": "BooleanExpression"},
                                             "Message": {"messageID": "MS_1",
                                                         "description": "The answer is incorrect. Please choose again."}}}}


def fixture() -> dict:
    questions = [
        db("QID1", "<p>Please read <strong>carefully</strong>.</p><p>A <u>new</u> page&nbsp;follows.</p>"),
        db("QID2", "Scenario A: the operation <b>succeeded</b>."),
        mc("QID3", "Who decided?", ["Physician", "Patient"], Validation=CORRECT,
           Randomization={"Advanced": None, "Type": "All", "TotalRandSubset": ""}),
        mc("QID4", "Have you seen this before?", ["Yes", "No"], order=["2", "1"],
           Validation={"Settings": {"ForceResponse": "ON", "ForceResponseType": "ON", "Type": "None"}}),
        {"QuestionID": "QID5", "DataExportTag": "where", "QuestionType": "TE", "Selector": "SL",
         "QuestionText": "Where?", "Choices": {"1": {"Display": "Click to write Choice 1"}},
         "Validation": {"Settings": {"ForceResponse": "ON", "ForceResponseType": "RequestResponse", "Type": "None"}},
         "DisplayLogic": {"0": {"0": selected("QID4", "1"), "Type": "If"}, "Type": "BooleanExpression",
                          "inPage": True}},
        db("QID6", "Scenario B: the operation failed."),
        mc("QID7", "Unused screener: how serious were you in answering?", ["1 (Not at all)", "5 (Very much)"]),
        db("QID8", "An orphan question that belongs to no block at all."),
        db("QID9", "Old instructions kept in a block the flow never uses."),
        {"QuestionID": "QID10", "DataExportTag": "timer", "QuestionType": "Timing", "Selector": "PageTimer",
         "QuestionText": "Timing"},
    ]
    blocks = [
        block("BL_trash", "Trash / Unused Questions", ["QID7"], kind="Trash"),
        block("BL_intro", "Intro", ["QID1", "QID10"], kind="Default"),
        block("BL_a", "Condition A", ["QID2", "QID3", "|", "QID4", "QID5"]),
        block("BL_b", "Condition B", ["QID6"]),
        block("BL_old", "Old", ["QID9"]),
    ]
    flow = [
        {"Type": "EmbeddedData", "FlowID": "FL_2",
         "EmbeddedData": [{"Field": "PROLIFIC_PID", "Type": "Recipient"}]},
        {"Type": "Block", "ID": "BL_intro", "FlowID": "FL_3"},
        {"Type": "Branch", "FlowID": "FL_4", "Description": "Bots",
         "BranchLogic": {"0": {"0": {"LogicType": "EmbeddedField", "LeftOperand": "Q_RecaptchaScore",
                                     "Operator": "LessThanOrEqual", "RightOperand": "0.5", "Type": "Expression"},
                               "Type": "If"}, "Type": "BooleanExpression"},
         "Flow": [{"Type": "EndSurvey", "FlowID": "FL_5"}]},
        {"Type": "BlockRandomizer", "FlowID": "FL_6", "SubSet": 1, "EvenPresentation": True, "Flow": [
            {"Type": "Group", "FlowID": "FL_7", "Description": "Arm A", "Flow": [
                {"Type": "EmbeddedData", "FlowID": "FL_8",
                 "EmbeddedData": [{"Field": "arm", "Type": "Custom", "Value": "A"}]},
                {"Type": "Standard", "ID": "BL_a", "FlowID": "FL_9"}]},
            {"Type": "Standard", "ID": "BL_b", "FlowID": "FL_10"}]},
    ]
    return survey(questions, blocks, flow)


def test_flow_order_randomizer_branch_and_embedded_data() -> None:
    report = qsf_flow.read_survey(fixture())
    flow = report["flow"]
    assert [n["type"] for n in flow] == ["embedded_data", "block", "branch", "randomizer"]
    assert flow[0]["fields"][0]["value"] is None and "panel" in flow[0]["fields"][0]["set_by"]
    assert flow[2]["condition"] == "embedded data Q_RecaptchaScore ≤ 0.5"
    assert flow[2]["elements"][0]["type"] == "end_survey"
    rand = flow[3]
    assert (rand["present"], rand["of"], rand["evenly"]) == (1, 2, True)
    assert "evenly presented" in rand["summary"]
    group = rand["elements"][0]
    assert group["type"] == "group" and group["elements"][0]["fields"][0] == {
        "field": "arm", "value": "A", "set_by": "the survey flow"}
    assert report["block_order"] == ["BL_intro", "BL_a", "BL_b"]
    assert report["survey"]["options"]["back_button"] is False


def test_unreachable_blocks_and_questions() -> None:
    report = qsf_flow.read_survey(fixture())
    blocks = {b["id"]: b["reason"] for b in report["not_reachable"]["blocks"]}
    assert set(blocks) == {"BL_trash", "BL_old"}
    assert "Trash" in blocks["BL_trash"] and "not in the survey flow" in blocks["BL_old"]
    questions = {q["qid"]: q["reason"] for q in report["not_reachable"]["questions"]}
    assert set(questions) == {"QID7", "QID8", "QID9"}
    assert "no block" in questions["QID8"]
    shown = {q["qid"] for b in report["blocks"].values() for q in b["questions"]}
    assert "QID7" not in shown and {"QID1", "QID3", "QID5"} <= shown


def test_question_details() -> None:
    report = qsf_flow.read_survey(fixture())
    a = report["blocks"]["BL_a"]
    assert [p["questions"] for p in a["pages"]] == [["QID2", "QID3"], ["QID4", "QID5"]]
    q = {x["qid"]: x for b in report["blocks"].values() for x in b["questions"]}
    assert q["QID1"]["text"] == "Please read **carefully**.\n\nA <u>new</u> page follows." or \
        q["QID1"]["text"] == "Please read **carefully**.\nA <u>new</u> page follows."
    assert " ".join(q["QID1"]["text_plain"].split()) == "Please read carefully. A new page follows."
    check = q["QID3"]
    assert check["validation"]["response"] == "forced"
    assert check["validation"]["correct_choices"] == [{"id": "2", "text": "Patient"}]
    assert check["validation"]["message_description"] == "The answer is incorrect. Please choose again."
    assert check["choice_randomisation"]["type"] == "all"
    assert [c["text"] for c in q["QID4"]["choices"]] == ["No", "Yes"]  # ChoiceOrder is the display order
    assert "display_logic" not in q["QID4"]
    where = q["QID5"]
    assert where["validation"]["response"] == "requested"
    assert where["display_logic"]["in_page"] is True
    assert where["display_logic"]["depends_on_questions"] == ["QID4"]
    assert '"Yes" is selected' in where["display_logic"]["condition"]
    assert "choices" not in where  # text-entry placeholders are not options
    assert q["QID10"]["participant_visible"] is False


def test_advanced_block_randomisation_and_undisplayed() -> None:
    questions = [db("QID1", "Intro text here."), mc("QID2", "Choice question?", ["a b", "c d"]),
                 mc("QID3", "Preference question?", ["e f", "g h"]), db("QID4", "Final question text."),
                 db("QID5", "Never displayed question text.")]
    options = {"RandomizeQuestions": "Advanced", "Randomization": {"Advanced": {
        "FixedOrder": ["QID1", "{~Randomized~}", "{~Randomized~}", "QID4"], "RandomizeAll": ["QID2", "QID3"],
        "RandomSubSet": [], "Undisplayed": ["QID5"], "TotalRandSubset": 0, "QuestionsPerPage": "3"},
        "EvenPresentation": False}}
    data = survey(questions, [block("BL_1", "S1", ["QID1", "QID2", "QID3", "|", "QID4", "QID5"], options=options)],
                  [{"Type": "Standard", "ID": "BL_1", "FlowID": "FL_2"}], blocks_as_dict=False)
    report = qsf_flow.read_survey(data)
    b = report["blocks"]["BL_1"]
    assert b["question_randomisation"]["randomize_all"] == ["QID2", "QID3"]
    assert [p["questions"] for p in b["pages"]] == [["QID1", "QID2", "QID3"], ["QID4"]]
    assert b["pages"][0]["randomised"] == ["QID2", "QID3"]
    assert [q["qid"] for q in report["not_reachable"]["questions"]] == ["QID5"]
    assert "undisplayed" in report["not_reachable"]["questions"][0]["reason"]


def test_trash_block_named_in_flow_and_unknown_elements() -> None:
    questions = [db("QID1", "Trash text that is never shown.")]
    flow = [{"Type": "Block", "ID": "BL_t", "FlowID": "FL_2"},
            {"Type": "WebService", "FlowID": "FL_3"}, {"Type": "Block", "ID": "BL_missing", "FlowID": "FL_4"}]
    report = qsf_flow.read_survey(survey(questions, [block("BL_t", "Trash / Unused Questions", ["QID1"], "Trash")], flow))
    assert report["blocks"] == {}
    assert report["not_reachable"]["questions"][0]["qid"] == "QID1"
    assert any("WebService" in n for n in report["notes"])
    assert any("BL_missing" in n for n in report["notes"])


def test_rejects_files_that_are_not_survey_exports(tmp_path: Path) -> None:
    data = tmp_path / "responses.qsf"
    data.write_text("ResponseId,Q1\nR_1,3\n")
    with pytest.raises(qsf_flow.NotASurveyExport):
        qsf_flow.load_qsf(data)
    (tmp_path / "list.qsf").write_text("[1, 2]")
    with pytest.raises(qsf_flow.NotASurveyExport):
        qsf_flow.load_qsf(tmp_path / "list.qsf")
    assert qsf_flow.main([str(data)]) == 2


def test_cli_writes_json_and_outline(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "survey.qsf"
    path.write_text(json.dumps(fixture()))
    out = tmp_path / "work" / "survey.flow.json"
    assert qsf_flow.main([str(path), "--json", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["block_order"] == ["BL_intro", "BL_a", "BL_b"]
    text = capsys.readouterr().out
    assert "correct answer required: Patient" in text
    assert "Never shown to participants" in text and "Trash / Unused Questions" in text
    assert "options (base order, shuffled per participant): Physician | Patient" in text


# --------------------------------------------------------------------------- package check

def make_package(tmp_path: Path, materials: dict, fields: list | None = None) -> tuple[Path, Path]:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    (root / "source").mkdir(parents=True)
    (root / "materials").mkdir()
    (job / "input/external").mkdir(parents=True)
    (job / "input/external/survey.qsf").write_text(json.dumps(fixture()))
    (root / "source/paper_metadata.json").write_text(json.dumps({"external_sources": [
        {"id": "osf-qsf", "path": "input/external/survey.qsf"}]}))
    (root / "materials/materials.json").write_text(json.dumps(materials))
    nodes = [{"id": "m", "kind": "material", "fields": fields or []}]
    (root / "study.json").write_text(json.dumps({"program": {"nodes": nodes}}))
    return job, root


def test_package_check_passes_reachable_text(tmp_path: Path) -> None:
    job, root = make_package(tmp_path, {"sourceId": "osf-qsf", "screens": [
        {"id": "s1", "text": "Scenario A: the operation succeeded.", "options": ["Physician", "Patient"]},
        {"id": "s2", "text": "Who decided?"}],
        "notShown": [{"qid": "QID7", "text": "Unused screener: how serious were you in answering?"}]})
    assert qsf_flow.package_problems(root, job) == []


def test_package_check_fails_trash_text_in_materials(tmp_path: Path) -> None:
    job, root = make_package(tmp_path, {"screens": [
        {"id": "s1", "sourceId": "osf-qsf", "text": "Scenario A: the operation succeeded."},
        {"id": "s2", "source": {"path": "input/external/survey.qsf"},
         "text": "Unused screener: how serious were you in answering?", "options": ["1 (Not at all)"]},
        {"id": "s3", "sourceId": "paper", "text": "Old instructions kept in a block the flow never uses."}]})
    problems = qsf_flow.package_problems(root, job)
    assert len(problems) == 1
    assert "screens[1]" in problems[0] and "QID7" in problems[0] and "Trash" in problems[0]


def test_package_check_fails_unreachable_program_field(tmp_path: Path) -> None:
    field = {"id": "f", "value": "Old instructions kept in a block the flow never uses.", "origin": "verbatim",
             "extensions": {"material": {"classification": "needs-original", "status": "found-open",
                                         "sourceIds": ["osf-qsf"]}}}
    job, root = make_package(tmp_path, {}, [field])
    problems = qsf_flow.package_problems(root, job)
    assert len(problems) == 1 and problems[0].startswith("m.f uses survey text from QID9")


def test_package_check_skips_exports_without_a_flow(tmp_path: Path) -> None:
    job, root = make_package(tmp_path, {"sourceId": "osf-qsf", "text": "Unused screener: how serious were you"})
    (job / "input/external/survey.qsf").write_text(json.dumps(
        {"SurveyElements": [{"Payload": {"QuestionText": "Unused screener: how serious were you"}}]}))
    assert qsf_flow.package_problems(root, job) == []


def test_text_quirks_trash_name_autocomplete_and_response_types() -> None:
    questions = [
        db("QID1", "If\nyou were a <em>student</em>,<br>read this."),
        {"QuestionID": "QID2", "DataExportTag": "country", "QuestionType": "TE", "Selector": "SL",
         "QuestionText": "Which country?", "SearchSource": {"AllowFreeResponse": "false"},
         "Validation": {"Settings": {"ForceResponse": "OFF", "ForceResponseType": "RequestResponse",
                                     "Type": "None"}}},
        mc("QID3", "Asked but optional to answer here?", ["a b", "c d"],
           Validation={"Settings": {"ForceResponse": "ON", "ForceResponseType": "RequestResponse", "Type": "None"}}),
    ]
    blocks = [block("BL_1", "Trash sorting task", ["QID1", "QID2", "QID3"])]
    report = qsf_flow.read_survey(survey(questions, blocks, [{"Type": "Standard", "ID": "BL_1", "FlowID": "FL_2"}]))
    assert report["block_order"] == ["BL_1"]  # a block named "Trash ..." is not the Trash block
    q = {x["qid"]: x for x in report["blocks"]["BL_1"]["questions"]}
    assert q["QID1"]["text"] == "If you were a *student*,\nread this."
    assert q["QID2"]["autocomplete"]["free_response"] is False
    assert q["QID2"]["validation"]["response"] == "optional"
    assert q["QID3"]["validation"]["response"] == "requested"


def test_malformed_exports_do_not_crash() -> None:
    data = survey([db("QID1", "Some text here.")], [{"ID": "BL_1", "Type": "Standard", "BlockElements": None,
                                                     "Options": {"RandomizeQuestions": "Advanced",
                                                                 "Randomization": []}}],
                  [{"Type": "Standard", "ID": "BL_1", "FlowID": "FL_2"}])
    report = qsf_flow.read_survey(data)
    assert report["blocks"]["BL_1"]["questions"] == []
    data["SurveyElements"] = [e if e.get("Element") != "FL" else {"Element": "FL", "Payload": None}
                              for e in data["SurveyElements"]]
    assert qsf_flow.read_survey(data)["flow"] == []


def test_package_check_ignores_duplicates_short_strings_metadata_and_markers(tmp_path: Path) -> None:
    data = fixture()
    # A Trash copy of reachable text, as Qualtrics keeps when a question is duplicated.
    data["SurveyElements"].insert(0, {"Element": "SQ", "Payload": db("QID11", "Scenario A: the operation succeeded.")})
    for element in data["SurveyElements"]:
        if element["Element"] == "BL":
            element["Payload"]["0"]["BlockElements"].append({"Type": "Question", "QuestionID": "QID11"})
    job, root = make_package(tmp_path, {"sourceId": "osf-qsf", "note": "Unused screener: how serious were you in answering",
                                        "screens": [
        {"id": "s1", "text": "Please read this first. Scenario A: the operation succeeded."},
        {"id": "s2", "text": "Who decided?", "options": ["1 (Not at all)"]},
        {"id": "s4", "qsfReachability": "not-shown", "text": "Unused screener: how serious were you in answering?"},
        {"id": "s5", "image": {"path": "materials/pic.png"}, "text": "Scenario A: the operation succeeded."}]})
    (job / "input/external/survey.qsf").write_text(json.dumps(data))
    assert qsf_flow.package_problems(root, job) == []


def test_package_check_keeps_qsf_scope_past_unrelated_paths_and_scans_adapter(tmp_path: Path) -> None:
    job, root = make_package(tmp_path, {"sourceId": "osf-qsf", "screens": [
        {"id": "s1", "path": "materials/pic.png", "text": "Unused screener: how serious were you in answering?"}]})
    (root / "task").mkdir()
    (root / "task/adapter.py").write_text('SCREEN = "Old instructions kept in a block the flow never uses."\n')
    problems = qsf_flow.package_problems(root, job)
    assert any(p.startswith("materials/materials.json screens[0]") and "QID7" in p for p in problems)
    assert any(p.startswith("task/adapter.py line 1") and "QID9" in p for p in problems)


def test_repair_guard_allows_removing_or_marking_never_shown_items(tmp_path: Path) -> None:
    trash = {"id": "s2", "origin": "verbatim", "sourceId": "osf-qsf",
             "text": "Unused screener: how serious were you in answering?"}
    live = {"id": "s1", "origin": "verbatim", "sourceId": "osf-qsf", "text": "Scenario A: the operation succeeded."}
    job, root = make_package(tmp_path, {"screens": [live, trash]})
    removable = qsf_flow.never_shown_matcher(root, job)
    pre = {"materials/materials.json": {"screens": [live, trash]}}
    assert not compare_packages(pre, {"materials/materials.json": {"screens": [live]}})["passed"]
    assert compare_packages(pre, {"materials/materials.json": {"screens": [live]}}, removable)["passed"]
    marked = dict(trash, qsfReachability="not-shown")
    assert compare_packages(pre, {"materials/materials.json": {"screens": [live, marked]}}, removable)["passed"]
    reworded = dict(marked, text="A different sentence entirely, now.")
    assert not compare_packages(pre, {"materials/materials.json": {"screens": [live, reworded]}}, removable)["passed"]
    # A reachable verbatim item still may not be removed.
    assert not compare_packages(pre, {"materials/materials.json": {"screens": [trash]}}, removable)["passed"]


def test_package_check_fails_marked_items_the_run_still_uses(tmp_path: Path) -> None:
    trash = "Unused screener: how serious were you in answering?"
    job, root = make_package(tmp_path, {"sourceId": "osf-qsf", "screens": [
        {"id": "s.trash", "qsfReachability": "not-shown", "text": trash},
        {"id": "s.status", "status": "unused", "text": trash}]})
    (root / "task").mkdir()
    (root / "task/adapter.py").write_text('def run(ask):\n    ask("s.trash")\n')
    problems = qsf_flow.package_problems(root, job)
    assert any("'s.trash'" in p and "still" in p for p in problems)
    assert any("screens[1]" in p and "QID7" in p for p in problems)  # only the explicit marker exempts text


def test_repair_guard_keeps_nested_and_mixed_items(tmp_path: Path) -> None:
    trash = "Unused screener: how serious were you in answering?"
    live = "Scenario A: the operation succeeded."
    job, root = make_package(tmp_path, {})
    removable = qsf_flow.never_shown_matcher(root, job)
    parent = {"id": "p", "origin": "verbatim", "text": trash, "children": [
        {"id": "c1", "origin": "verbatim", "text": live},
        {"id": "c2", "origin": "missing", "state": "missing", "value": None}]}
    pre = {"materials/materials.json": {"screens": [parent]}}
    result = compare_packages(pre, {"materials/materials.json": {"screens": []}}, removable)
    assert not result["passed"]
    assert any("c1" in v for v in result["violations"]) and any("c2" in v for v in result["violations"])
    mixed = {"id": "m", "origin": "verbatim", "text": live, "options": [trash]}
    pre = {"materials/materials.json": {"screens": [mixed]}}
    assert not compare_packages(pre, {"materials/materials.json": {"screens": []}}, removable)["passed"]


def test_mixed_items_and_question_id_selectors(tmp_path: Path) -> None:
    trash = "Unused screener: how serious were you in answering?"
    live = "Scenario A: the operation succeeded."
    mixed = {"id": "m", "origin": "verbatim", "state": "confirmed", "sourceId": "osf-qsf", "text": live,
             "options": [trash]}
    job, root = make_package(tmp_path, {"screens": [mixed]})
    assert qsf_flow.package_problems(root, job)
    marked = dict(mixed, qsfReachability="mixed", state="check")
    (root / "materials/materials.json").write_text(json.dumps({"screens": [marked]}))
    assert qsf_flow.package_problems(root, job) == []
    removable = qsf_flow.never_shown_matcher(root, job)
    pre = {"materials/materials.json": {"screens": [mixed]}}
    assert compare_packages(pre, {"materials/materials.json": {"screens": [marked]}}, removable)["passed"]
    assert not compare_packages(pre, {"materials/materials.json": {"screens": [dict(marked, state="confirmed")]}},
                                removable)["passed"]
    # A not-shown item selected by its question id is still caught.
    item = {"id": "w1", "qsfId": "QID7", "qsfReachability": "not-shown", "sourceId": "osf-qsf", "text": trash}
    (root / "materials/materials.json").write_text(json.dumps({"screens": [item]}))
    (root / "task").mkdir()
    (root / "task/adapter.py").write_text('WRAPUP = ["QID7"]\n')
    assert any("'QID7'" in p for p in qsf_flow.package_problems(root, job))
