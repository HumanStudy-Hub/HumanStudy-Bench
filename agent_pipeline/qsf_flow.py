#!/usr/bin/env python3
"""Read a Qualtrics survey export (.qsf) in the order participants saw it.

A .qsf file is the survey's design, not its responses. It lists every question
ever written for the survey, including ones in the "Trash / Unused Questions"
block and blocks that are not in the survey flow. Participants only saw the
blocks the survey flow reaches, in the flow's order, with its randomisers,
branches and each question's display logic, validation and answer order. This
reader turns that structure into a deterministic record so a build follows the
survey instead of guessing from the raw JSON.

    python agent_pipeline/qsf_flow.py input/external/survey.qsf            # outline
    python agent_pipeline/qsf_flow.py input/external/survey.qsf --json out.json

Output (``read_survey``):

- ``flow``: the survey flow in order. Elements are ``block`` (a reference to a
  block, by id), ``randomizer`` (``present`` of ``of`` elements, ``evenly``),
  ``branch`` (``condition`` in plain words, then its elements), ``group``,
  ``embedded_data`` (fields and the values the flow sets), ``end_survey`` and
  anything else Qualtrics allows, kept with its type and a note.
- ``blocks``: each reachable block, in first-appearance order: question order,
  pages (page breaks, or questions per page under advanced randomisation), the
  block's question randomisation, and each question with its text, choices in
  display order, answer-order randomisation, display logic, skip logic and
  validation (forced or requested response, correct-answer custom validation
  with its message, content checks).
- ``not_reachable``: blocks and questions participants never saw (the Trash
  block, blocks missing from the flow, questions in no block, questions an
  advanced randomisation never displays), with the reason.
- ``notes``: anything the reader could not interpret fully.

Text: HTML becomes plain text with line breaks kept. Emphasis is kept as
markers: ``**bold**``, ``*italic*`` and ``<u>underlined</u>``. ``text_plain``
has no markers.

Limits: the reader reports logic; it does not evaluate it. Display logic and
branch conditions are rendered from the export's structured fields (falling back
to Qualtrics's own description) and a person or agent decides what they imply.
JavaScript attached to a question (which can hide, time or auto-advance it),
loop & merge, quotas, web services, authenticators and end-of-survey messages
(stored in the account's message library, not in the export) are reported but
not interpreted. Survey-level options such as the back button are copied from
the export. The reader never opens response data: it accepts only a Qualtrics
survey export (a JSON object with ``SurveyEntry`` and ``SurveyElements``).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

READER_VERSION = 1
PIPE_RE = re.compile(r"\$\{[^}]+\}")
HIDDEN_TYPES = {"Timing", "Meta", "Captcha"}  # recorded silently, or not a question participants answer
OPERATORS = {
    "Selected": "is selected", "NotSelected": "is not selected", "EqualTo": "=", "NotEqualTo": "≠",
    "GreaterThan": ">", "GreaterThanOrEqual": "≥", "LessThan": "<", "LessThanOrEqual": "≤",
    "Contains": "contains", "DoesNotContain": "does not contain", "Displayed": "was displayed",
    "NotDisplayed": "was not displayed", "Empty": "is empty", "NotEmpty": "is not empty",
    "MatchesRegex": "matches", "StartsWith": "starts with", "EndsWith": "ends with",
}
SELECTORS = {
    "SAVR": "single answer, vertical", "SAHR": "single answer, horizontal", "SACOL": "single answer, columns",
    "MAVR": "multiple answers, vertical", "MAHR": "multiple answers, horizontal", "MACOL": "multiple answers, columns",
    "DL": "dropdown list", "SB": "select box", "MSB": "multi-select box", "TB": "text / graphic",
    "SL": "single line", "ML": "multiple lines", "ESTB": "essay box", "FORM": "form", "PW": "password",
    "Likert": "Likert matrix", "Bipolar": "bipolar matrix", "TE": "text entry matrix", "HSLIDER": "horizontal slider",
}
TYPES = {"MC": "multiple choice", "DB": "text / graphic", "TE": "text entry", "Matrix": "matrix",
         "Slider": "slider", "RO": "rank order", "CS": "constant sum", "SBS": "side by side",
         "Timing": "page timer", "Meta": "browser meta info", "PGR": "pick, group and rank",
         "HL": "heat map", "HotSpot": "hot spot", "Draw": "signature", "FileUpload": "file upload",
         "Captcha": "captcha", "NPS": "net promoter score"}


class NotASurveyExport(ValueError):
    pass


# --------------------------------------------------------------------------- text

def html_to_text(value: Any, markers: bool = True) -> str:
    """Survey HTML as plain text, keeping paragraphs and (optionally) emphasis."""
    if not isinstance(value, str):
        return ""
    text = re.sub(r"(?is)<(script|style)\b.*?</\1\s*>", "", value)
    text = re.sub(r"\s+", " ", text)  # a line break in the HTML source renders as a space
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</?(p|div|li|ul|ol|h[1-6]|tr|table|blockquote)\b[^>]*>", "\n", text)
    if markers:
        text = re.sub(r"(?i)<(strong|b)\b[^>]*>", "**", text)
        text = re.sub(r"(?i)</(strong|b)\s*>", "**", text)
        text = re.sub(r"(?i)<(em|i)\b[^>]*>", "*", text)
        text = re.sub(r"(?i)</(em|i)\s*>", "*", text)
        text = re.sub(r"(?i)<u\b[^>]*>", "\x01", text)
        text = re.sub(r"(?i)</u\s*>", "\x02", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ").replace("\r", "")
    text = text.replace("\x01", "<u>").replace("\x02", "</u>")
    if markers:
        # Empty or doubled markers left by adjacent tags (<b></b>, <strong><b>).
        text = re.sub(r"\*\*\s*\*\*", "", text)
        text = text.replace("<u></u>", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _short(text: str, n: int = 90) -> str:
    one = re.sub(r"\s+", " ", text).strip()
    return one if len(one) <= n else one[: n - 1].rstrip() + "…"


# --------------------------------------------------------------------------- loading

def load_qsf(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except ValueError as exc:
        raise NotASurveyExport(f"{path} is not JSON, so it is not a Qualtrics survey export") from exc
    if not isinstance(data, dict) or not isinstance(data.get("SurveyElements"), list):
        raise NotASurveyExport(f"{path} has no SurveyElements, so it is not a Qualtrics survey export (.qsf)")
    return data


def _elements(data: dict, kind: str) -> list[dict]:
    return [e for e in data.get("SurveyElements", []) if isinstance(e, dict) and e.get("Element") == kind]


def _blocks(data: dict) -> dict[str, dict]:
    blocks: dict[str, dict] = {}
    for element in _elements(data, "BL"):
        payload = element.get("Payload")
        items = payload.values() if isinstance(payload, dict) else payload if isinstance(payload, list) else []
        for block in items:
            if isinstance(block, dict) and isinstance(block.get("ID"), str):
                blocks[block["ID"]] = block
    return blocks


def _questions(data: dict) -> dict[str, dict]:
    questions: dict[str, dict] = {}
    for element in _elements(data, "SQ"):
        payload = element.get("Payload")
        if isinstance(payload, dict):
            qid = payload.get("QuestionID") or element.get("PrimaryAttribute")
            if isinstance(qid, str):
                questions[qid] = payload
    return questions


def _is_trash(block: dict) -> bool:
    if block.get("Type"):
        return str(block["Type"]).lower() == "trash"
    return str(block.get("Description", "")).strip().lower() == "trash / unused questions"


# --------------------------------------------------------------------------- logic

def _choice_text(questions: dict[str, dict], qid: str, choice: str) -> str:
    q = questions.get(qid, {})
    choices = q.get("Choices") if isinstance(q.get("Choices"), dict) else {}
    item = choices.get(str(choice))
    if isinstance(item, dict):
        return html_to_text(item.get("Display"), markers=False)
    return f"choice {choice}"


def _answer_text(questions: dict[str, dict], qid: str, answer: str) -> str:
    q = questions.get(qid, {})
    answers = q.get("Answers") if isinstance(q.get("Answers"), dict) else {}
    item = answers.get(str(answer))
    return html_to_text(item.get("Display"), markers=False) if isinstance(item, dict) else f"column {answer}"


def _operand(locator: Any, questions: dict[str, dict]) -> str:
    """A q://QID/... locator as words."""
    if not isinstance(locator, str):
        return str(locator)
    match = re.match(r"q://(QID\d+)/([A-Za-z]+)(?:/([^/]+))?(?:/([^/]+))?", locator)
    if not match:
        return locator
    qid, what, first, second = match.groups()
    if what == "SelectableChoice" and first:
        return f'"{_choice_text(questions, qid, first)}"'
    if what == "SelectableAnswer" and first and second:
        return f'row "{_choice_text(questions, qid, first)}" → "{_answer_text(questions, qid, second)}"'
    if what in ("ChoiceTextEntryValue", "ChoiceNumericEntryValue") and first:
        return f'the text typed for "{_choice_text(questions, qid, first)}"'
    if what in ("ChoiceTextEntryValue", "QuestionText"):
        return "the answer typed"
    return f"{what} {first or ''} {second or ''}".strip()


def _expression(expr: dict, questions: dict[str, dict]) -> str:
    kind = expr.get("LogicType")
    operator = OPERATORS.get(str(expr.get("Operator")), str(expr.get("Operator", "")))
    right = expr.get("RightOperand")
    right_text = "" if right in (None, "") else f" {str(right).strip()}"
    if kind == "Question" and isinstance(expr.get("QuestionID"), str):
        qid = expr["QuestionID"]
        q = questions.get(qid, {})
        label = f'{qid} ({q.get("DataExportTag", "")}: "{_short(html_to_text(q.get("QuestionText"), False), 60)}")'
        left = _operand(expr.get("LeftOperand") or expr.get("ChoiceLocator"), questions)
        return f"{label} {left} {operator}{right_text}".strip()
    if kind == "EmbeddedField":
        return f"embedded data {expr.get('LeftOperand')} {operator}{right_text}".strip()
    description = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", str(expr.get("Description") or "")))).strip()
    description = re.sub(r"^(If|And|Or)\s+", "", description)
    return description or json.dumps({k: v for k, v in expr.items() if k != "Description"}, ensure_ascii=False)


def render_logic(logic: Any, questions: dict[str, dict]) -> str:
    """A Qualtrics BooleanExpression (display, branch, skip or validation logic) in words."""
    if not isinstance(logic, dict):
        return ""
    groups = [(k, v) for k, v in logic.items() if str(k).isdigit() and isinstance(v, dict)]
    rendered: list[str] = []
    for index, (_, group) in enumerate(sorted(groups, key=lambda kv: int(kv[0]))):
        parts: list[str] = []
        exprs = [(k, v) for k, v in group.items() if str(k).isdigit() and isinstance(v, dict)]
        for position, (_, expr) in enumerate(sorted(exprs, key=lambda kv: int(kv[0]))):
            text = _expression(expr, questions)
            if position:
                text = f"{str(expr.get('Conjuction') or expr.get('Conjunction') or 'And').upper()} {text}"
            parts.append(text)
        body = " ".join(parts)
        if index:
            joiner = group.get("Conjuction") or group.get("Conjunction") or group.get("Type")
            joiner = joiner if str(joiner).lower() in ("and", "or") else "Or"
            body = f"{str(joiner).upper()} ({body})"
        elif len(groups) > 1:
            body = f"({body})"
        rendered.append(body)
    return " ".join(rendered)


def _logic_expressions(logic: Any) -> Iterable[dict]:
    if isinstance(logic, dict):
        for key, value in logic.items():
            if str(key).isdigit() and isinstance(value, dict):
                if "LogicType" in value or "Operator" in value:
                    yield value
                else:
                    yield from _logic_expressions(value)


def _logic_record(logic: Any, questions: dict[str, dict]) -> dict | None:
    if not isinstance(logic, dict) or not any(True for _ in _logic_expressions(logic)):
        return None
    record: dict[str, Any] = {"condition": render_logic(logic, questions)}
    if "inPage" in logic:
        record["in_page"] = bool(logic.get("inPage"))
    depends = sorted({e["QuestionID"] for e in _logic_expressions(logic) if isinstance(e.get("QuestionID"), str)})
    fields = sorted({str(e.get("LeftOperand")) for e in _logic_expressions(logic) if e.get("LogicType") == "EmbeddedField"})
    if depends:
        record["depends_on_questions"] = depends
    if fields:
        record["depends_on_embedded_data"] = fields
    return record


# --------------------------------------------------------------------------- questions

def _randomisation(q: dict) -> dict | None:
    rand = q.get("Randomization")
    if isinstance(rand, str):
        rand = {"Type": rand}
    if not isinstance(rand, dict):
        return None
    kind = str(rand.get("Type", "None"))
    if kind in ("None", "", "false"):
        return None
    if kind == "All":
        return {"type": "all", "summary": "answer options shown in a new random order for each participant"}
    if kind == "RandomSubset":
        return {"type": "subset", "show": rand.get("TotalRandSubset"),
                "summary": f"a random {rand.get('TotalRandSubset')} of the answer options are shown, in random order"}
    if kind == "Advanced":
        advanced = rand.get("Advanced") if isinstance(rand.get("Advanced"), dict) else {}
        return {"type": "advanced", "fixed_order": advanced.get("FixedOrder"),
                "randomize_all": advanced.get("RandomizeAll"), "random_subset": advanced.get("RandomSubSet"),
                "undisplayed": advanced.get("Undisplayed"), "total_random_subset": advanced.get("TotalRandSubset"),
                "summary": "advanced answer-order randomisation: options in fixed_order stay; '{~Randomized~}' "
                           "slots are filled at random from randomize_all; undisplayed options are never shown"}
    if kind == "Flip" or rand.get("Type") == "ReverseOrder":
        return {"type": "flip", "summary": "answer options shown in order or reversed, at random"}
    return {"type": kind, "summary": f"answer-order randomisation of type {kind!r} (not interpreted)"}


def _validation(q: dict, questions: dict[str, dict]) -> dict:
    settings = q.get("Validation", {}).get("Settings", {}) if isinstance(q.get("Validation"), dict) else {}
    settings = settings if isinstance(settings, dict) else {}
    force = str(settings.get("ForceResponse", "OFF"))
    force_type = str(settings.get("ForceResponseType", ""))
    # ForceResponse decides; ForceResponseType only says which kind when it is on
    # (an "OFF" question can keep a remembered type from an earlier setting).
    if force == "RequestResponse" or (force == "ON" and force_type == "RequestResponse"):
        response = "requested"
    elif force == "ON":
        response = "forced"
    else:
        response = "optional"
    record: dict[str, Any] = {"response": response}
    if response == "requested":
        record["response_note"] = "the participant is asked to answer but may continue without answering"
    kind = settings.get("Type")
    if kind and kind != "None":
        record["type"] = kind
    if kind == "CustomValidation":
        custom = settings.get("CustomValidation", {}) if isinstance(settings.get("CustomValidation"), dict) else {}
        logic = custom.get("Logic")
        record["must_satisfy"] = render_logic(logic, questions)
        message = custom.get("Message")
        if isinstance(message, dict):
            record["message_description"] = html_to_text(message.get("description"), markers=False) or None
            record["message_note"] = ("the export stores only the description of a message in the account's "
                                      "message library; it may be the shown text (\"The answer is incorrect. "
                                      "Please choose again.\") or just the message's name (\"Require "
                                      "response\"). The shown text is not in the export")
        own = [e for e in _logic_expressions(logic) if e.get("QuestionID") == q.get("QuestionID")
               and e.get("Operator") == "Selected"]
        locators = [str(e.get("LeftOperand") or e.get("ChoiceLocator") or "") for e in own]
        choice_ids = [m.group(1) for loc in locators for m in [re.search(r"/SelectableChoice/([^/]+)$", loc)] if m]
        conjunctions = {str(e.get("Conjuction") or e.get("Conjunction") or "").lower()
                        for e in list(_logic_expressions(logic))[1:]}
        if choice_ids and len(choice_ids) == len(list(_logic_expressions(logic))) and conjunctions <= {"", "or"}:
            record["correct_choices"] = [{"id": c, "text": _choice_text(questions, q["QuestionID"], c)}
                                         for c in choice_ids]
            record["summary"] = ("only the correct answer is accepted (" + " or ".join(
                f'"{c["text"]}"' for c in record["correct_choices"]) + "); any other answer is refused"
                + (f' (message description: "{record["message_description"]}")' if record.get("message_description") else "")
                + " and the participant must answer again")
        else:
            record["summary"] = ("the answer must satisfy the condition in must_satisfy, otherwise it is refused"
                                 + (f' with the message "{record["message_description"]}"' if record.get("message_description") else ""))
    elif kind == "ContentType":
        content = settings.get("ContentType")
        record["content"] = content
        detail = settings.get(str(content)) if isinstance(settings.get(str(content)), dict) else None
        if detail:
            record["limits"] = detail
    elif kind in ("MinChoices", "MaxChoices", "MinChar", "MaxChar", "ChoiceRange", "TotalChars"):
        record["limits"] = {k: v for k, v in settings.items() if k in
                            ("MinChoices", "MaxChoices", "MinChars", "MaxChars", "TotalChars")}
    return record


def _choices(q: dict, questions: dict[str, dict], key: str, order_key: str) -> list[dict]:
    choices = q.get(key) if isinstance(q.get(key), dict) else {}
    order = q.get(order_key) if isinstance(q.get(order_key), list) else list(choices)
    recodes = q.get("RecodeValues") if key == "Choices" and isinstance(q.get("RecodeValues"), dict) else {}
    out: list[dict] = []
    seen: set[str] = set()
    for cid in [str(c) for c in order] + [c for c in map(str, choices) if c not in map(str, order)]:
        if cid in seen or cid not in choices:
            continue
        seen.add(cid)
        item = choices[cid] if isinstance(choices[cid], dict) else {"Display": str(choices[cid])}
        if not html_to_text(item.get("Display"), markers=False) and str(item.get("TextEntry", "")).lower() != "true":
            continue
        record: dict[str, Any] = {"id": cid, "text": html_to_text(item.get("Display")),
                                  "text_plain": html_to_text(item.get("Display"), markers=False)}
        if cid in recodes:
            record["recode"] = recodes[cid]
        if str(item.get("TextEntry", "")).lower() == "true":
            record["text_entry"] = True
        if str(item.get("ExclusiveAnswer", "")).lower() == "true":
            record["exclusive"] = True
        logic = _logic_record(item.get("DisplayLogic"), questions)
        if logic:
            record["display_logic"] = logic
        out.append(record)
    return out


def _question(q: dict, questions: dict[str, dict], notes: list[str]) -> dict:
    qid = str(q.get("QuestionID"))
    qtype, selector, sub = q.get("QuestionType"), q.get("Selector"), q.get("SubSelector")
    raw = q.get("QuestionText", "")
    record: dict[str, Any] = {
        "qid": qid, "tag": q.get("DataExportTag"),
        "type": "/".join(str(x) for x in (qtype, selector, sub) if x),
        "kind": ", ".join(dict.fromkeys(x for x in (TYPES.get(str(qtype), str(qtype)),
                                                     SELECTORS.get(str(selector), "")) if x)),
        "participant_visible": qtype not in HIDDEN_TYPES,
        "text": html_to_text(raw), "text_plain": html_to_text(raw, markers=False),
    }
    choices = _choices(q, questions, "Choices", "ChoiceOrder")
    if (qtype == "TE" and selector != "FORM") or qtype in HIDDEN_TYPES:
        choices = []  # leftover placeholders ("Click to write Choice 1") that a text box never shows
    if choices:
        record["choices" if qtype != "Matrix" else "rows"] = choices
        record["choice_order_note"] = "listed in display order (the export's ChoiceOrder)"
    answers = _choices(q, questions, "Answers", "AnswerOrder")
    if answers:
        record["columns"] = answers
    if isinstance(q.get("ColumnLabels"), list) and q["ColumnLabels"]:
        record["column_labels"] = [html_to_text(c.get("Display") if isinstance(c, dict) else c) for c in q["ColumnLabels"]]
    if qtype == "Slider" and isinstance(q.get("Configuration"), dict):
        config = q["Configuration"]
        record["scale"] = {k: config.get(k) for k in ("CSSliderMin", "CSSliderMax", "GridLines", "NumDecimals",
                                                      "SnapToGrid", "ShowValue") if k in config}
        if isinstance(q.get("Labels"), dict):
            record["scale"]["labels"] = [html_to_text(v.get("Display") if isinstance(v, dict) else v)
                                         for _, v in sorted(q["Labels"].items(),
                                                           key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0)]
    rand = _randomisation(q)
    if rand:
        record["choice_randomisation"] = rand
    logic = _logic_record(q.get("DisplayLogic") or q.get("InPageDisplayLogic"), questions)
    if logic:
        record["display_logic"] = logic
    skips = q.get("SkipLogic")
    skips = list(skips.values()) if isinstance(skips, dict) else skips if isinstance(skips, list) else []
    if skips:
        record["skip_logic"] = [{
            "when": f'{_operand(s.get("ChoiceLocator") or s.get("Locator"), questions)} '
                    f'{OPERATORS.get(str(s.get("Condition")), str(s.get("Condition", "")))}'.strip(),
            "skip_to": s.get("SkipToDescription") or s.get("SkipToDestination")} for s in skips if isinstance(s, dict)]
    record["validation"] = _validation(q, questions)
    defaults = q.get("DefaultChoices")
    if defaults not in (None, False, "", [], {}):
        record["default_answer"] = defaults
    pipes = sorted(set(PIPE_RE.findall(str(raw)) + [p for c in choices for p in PIPE_RE.findall(c["text"])]))
    if pipes:
        record["piped_text"] = pipes
    if q.get("QuestionJS"):
        record["has_javascript"] = True
        notes.append(f"{qid} has JavaScript, which can hide, time or auto-advance it; read it before relying on this record")
    if isinstance(q.get("DynamicChoices"), dict):
        record["carry_forward"] = {"from": q["DynamicChoices"].get("Locator"),
                                   "note": "the options shown are carried forward from another question's answers"}
    if isinstance(q.get("SearchSource"), dict) and q["SearchSource"]:
        source = q["SearchSource"]
        record["autocomplete"] = {"free_response": str(source.get("AllowFreeResponse", "")).lower() == "true",
                                  "note": "the text box suggests entries from a Qualtrics list that is not in the "
                                          "export; label that list missing (a list elsewhere in the file, e.g. in "
                                          "the Trash block, is not this list)"}
    return record


# --------------------------------------------------------------------------- blocks and flow

def _block_record(block: dict, questions: dict[str, dict], notes: list[str]) -> tuple[dict, list[str]]:
    """The block as shown, and the question ids an advanced randomisation never displays."""
    options = block.get("Options") if isinstance(block.get("Options"), dict) else {}
    elements = [e for e in block.get("BlockElements") or [] if isinstance(e, dict)]
    order = [e["QuestionID"] for e in elements if e.get("Type") == "Question" and isinstance(e.get("QuestionID"), str)]
    pages: list[list[str]] = [[]]
    for element in elements:
        if element.get("Type") == "Page Break":
            if pages[-1]:
                pages.append([])
        elif element.get("Type") == "Question" and isinstance(element.get("QuestionID"), str):
            pages[-1].append(element["QuestionID"])
    pages = [p for p in pages if p]
    record: dict[str, Any] = {"id": block.get("ID"), "description": block.get("Description"),
                              "type": block.get("Type")}
    undisplayed: list[str] = []
    shuffled: list[str] = []
    mode = str(options.get("RandomizeQuestions", "false"))
    if mode == "Advanced":
        randomization = options.get("Randomization") if isinstance(options.get("Randomization"), dict) else {}
        advanced = randomization.get("Advanced") if isinstance(randomization.get("Advanced"), dict) else {}
        undisplayed = [q for q in advanced.get("Undisplayed") or [] if isinstance(q, str)]
        per_page = int(str(advanced.get("QuestionsPerPage") or 0) or 0) if str(advanced.get("QuestionsPerPage") or 0).isdigit() else 0
        record["question_randomisation"] = {
            "type": "advanced", "fixed_order": advanced.get("FixedOrder"),
            "randomize_all": advanced.get("RandomizeAll"), "random_subset": advanced.get("RandomSubSet"),
            "total_random_subset": advanced.get("TotalRandSubset"), "undisplayed": undisplayed,
            "questions_per_page": per_page,
            "even_presentation": bool(randomization.get("EvenPresentation")),
            "summary": "questions in fixed_order keep their place; each '{~Randomized~}' slot is filled with a "
                       "question from randomize_all in a new random order per participant"
                       + ("; random_subset questions are drawn at random" if advanced.get("RandomSubSet") else "")
                       + ("; undisplayed questions are never shown" if undisplayed else "")}
        fixed = advanced.get("FixedOrder") if isinstance(advanced.get("FixedOrder"), list) else []
        pool = iter([q for q in advanced.get("RandomizeAll") or [] if isinstance(q, str)])
        subset = [q for q in advanced.get("RandomSubSet") or [] if isinstance(q, str)]
        slots = []
        for slot in fixed:
            if slot == "{~Randomized~}":
                slot = next(pool, None)
            elif slot == "{~SubSet~}":
                slot = subset.pop(0) if subset else None  # drawn at random from random_subset
            if isinstance(slot, str) and slot not in undisplayed:
                slots.append(slot)
        shuffled = [q for q in advanced.get("RandomizeAll") or [] if isinstance(q, str)]
        shuffled += [q for q in advanced.get("RandomSubSet") or [] if isinstance(q, str)]
        record["question_randomisation"]["note"] = ("pages list the randomised questions in their base order; "
                                                    "their order among the randomised slots differs per participant")
        if per_page > 0 and slots:
            pages = [slots[i:i + per_page] for i in range(0, len(slots), per_page)]
            record["paging_note"] = (f"advanced randomisation sets {per_page} questions per page; Qualtrics uses that "
                                     "instead of the block's page breaks")
        else:
            record["paging_note"] = ("pages follow the block's page breaks; check them, because Qualtrics may lay out "
                                     "a randomised block differently")
    elif mode not in ("false", "", "None"):
        record["question_randomisation"] = {"type": mode, "options": {k: v for k, v in options.items()
                                                                      if "andom" in k},
                                            "summary": f"question randomisation {mode!r}: questions shown in random "
                                                       "order (see options for subset sizes)"}
        notes.append(f"block {block.get('ID')} uses question randomisation {mode!r}; its exact paging is not interpreted")
    looping = options.get("Looping")
    if looping not in (None, "None", "", False):
        record["loop_and_merge"] = {"looping": looping, "options": options.get("LoopingOptions")}
        notes.append(f"block {block.get('ID')} uses loop & merge ({looping}); repeats are not expanded here")
    record["question_order"] = order
    record["pages"] = []
    for i, page in enumerate(pages):
        entry: dict[str, Any] = {"page": i + 1, "questions": [q for q in page if q not in undisplayed]}
        if any(q in shuffled for q in entry["questions"]):
            entry["randomised"] = [q for q in entry["questions"] if q in shuffled]
        record["pages"].append(entry)
    record["questions"] = []
    for qid in order:
        if qid in undisplayed:
            continue
        if qid not in questions:
            notes.append(f"block {block.get('ID')} lists {qid}, which the export does not contain")
            continue
        record["questions"].append(_question(questions[qid], questions, notes))
    return record, undisplayed


def _embedded(element: dict) -> list[dict]:
    fields = []
    for field in element.get("EmbeddedData", []) if isinstance(element.get("EmbeddedData"), list) else []:
        if not isinstance(field, dict):
            continue
        source = field.get("Type")
        value = field.get("Value")
        fields.append({"field": field.get("Field") or field.get("Description"),
                       "value": value if source == "Custom" else None,
                       "set_by": "the survey flow" if source == "Custom" else
                                 "the panel, URL or Qualtrics (not set by the flow)" if source == "Recipient" else source})
    return fields


def _flow(elements: Any, questions: dict[str, dict], blocks: dict[str, dict], seen: list[str], notes: list[str]) -> list[dict]:
    out: list[dict] = []
    for element in elements if isinstance(elements, list) else []:
        if not isinstance(element, dict):
            continue
        kind = element.get("Type")
        flow_id = element.get("FlowID")
        if kind in ("Block", "Standard", "Default"):
            bid = element.get("ID")
            block = blocks.get(str(bid))
            node: dict[str, Any] = {"type": "block", "flow_id": flow_id, "block_id": bid,
                                    "description": block.get("Description") if block else None}
            if block is None:
                node["note"] = "this block is not in the export"
                notes.append(f"flow {flow_id} shows block {bid}, which the export does not contain")
            elif _is_trash(block):
                node["note"] = "the Trash block is never shown, even when the flow names it"
            elif bid not in seen:
                seen.append(str(bid))
            if element.get("Autofill"):
                node["autofill"] = element.get("Autofill")
            out.append(node)
        elif kind == "BlockRandomizer" or kind == "Randomizer":
            children = _flow(element.get("Flow"), questions, blocks, seen, notes)
            subset = element.get("SubSet")
            try:
                present = int(subset)
            except (TypeError, ValueError):
                present = len(children)
            even = bool(element.get("EvenPresentation"))
            if present >= len(children):
                summary = f"shows all {len(children)} elements below, in a random order for each participant"
            else:
                summary = (f"shows {present} of the {len(children)} elements below, chosen at random"
                           + (" (evenly presented: Qualtrics balances how often each is shown)" if even else
                              " (not evenly presented: a plain random draw)"))
            out.append({"type": "randomizer", "flow_id": flow_id, "present": present, "of": len(children),
                        "evenly": even, "summary": summary, "elements": children})
        elif kind == "Branch":
            logic = element.get("BranchLogic")
            out.append({"type": "branch", "flow_id": flow_id, "description": element.get("Description"),
                        "condition": render_logic(logic, questions),
                        "summary": "the elements below run only for participants who meet the condition",
                        "elements": _flow(element.get("Flow"), questions, blocks, seen, notes)})
        elif kind == "Group":
            out.append({"type": "group", "flow_id": flow_id, "description": element.get("Description"),
                        "elements": _flow(element.get("Flow"), questions, blocks, seen, notes)})
        elif kind == "EmbeddedData":
            out.append({"type": "embedded_data", "flow_id": flow_id, "fields": _embedded(element)})
        elif kind == "EndSurvey":
            options = element.get("Options") if isinstance(element.get("Options"), dict) else {}
            out.append({"type": "end_survey", "flow_id": flow_id, "termination": options.get("SurveyTermination"),
                        "message_id": options.get("EOSMessage"),
                        "note": "ends the survey here; a custom end message is in the account's message library, "
                                "not in the export"})
        else:
            node = {"type": str(kind), "flow_id": flow_id,
                    "note": f"flow element {kind!r} is not interpreted by this reader; read it in the export"}
            if "Flow" in element:
                node["elements"] = _flow(element.get("Flow"), questions, blocks, seen, notes)
            notes.append(f"flow {flow_id}: element type {kind!r} is reported but not interpreted")
            out.append(node)
    return out


def read_survey(data: dict) -> dict:
    notes: list[str] = []
    questions = _questions(data)
    blocks = _blocks(data)
    flows = _elements(data, "FL")
    entry = data.get("SurveyEntry") if isinstance(data.get("SurveyEntry"), dict) else {}
    options = next((e.get("Payload") for e in _elements(data, "SO") if isinstance(e.get("Payload"), dict)), {})
    entry = entry if isinstance(entry, dict) else {}
    seen: list[str] = []
    if not flows:
        notes.append("the export has no survey flow (FL element), so nothing can be confirmed as shown")
        flow: list[dict] = []
    else:
        payload = flows[0].get("Payload") if isinstance(flows[0].get("Payload"), dict) else {}
        flow = _flow(payload.get("Flow"), questions, blocks, seen, notes)
    reachable_blocks: dict[str, dict] = {}
    undisplayed: dict[str, str] = {}
    for bid in seen:
        record, hidden = _block_record(blocks[bid], questions, notes)
        reachable_blocks[bid] = record
        for qid in hidden:
            undisplayed[qid] = bid
    shown = {q["qid"] for b in reachable_blocks.values() for q in b["questions"]}
    block_of = {qid: bid for bid, b in blocks.items() for e in b.get("BlockElements") or []
                if isinstance(e, dict) for qid in [e.get("QuestionID")] if isinstance(qid, str)}
    not_blocks = []
    for bid, block in blocks.items():
        if bid in reachable_blocks:
            continue
        reason = ("the Trash / Unused Questions block: never shown to participants" if _is_trash(block)
                  else "not in the survey flow: never shown to participants")
        not_blocks.append({"id": bid, "description": block.get("Description"), "type": block.get("Type"),
                           "questions": [e.get("QuestionID") for e in block.get("BlockElements") or []
                                         if isinstance(e, dict) and e.get("QuestionID")], "reason": reason})
    not_questions = []
    for qid, q in questions.items():
        if qid in shown:
            continue
        bid = block_of.get(qid)
        if qid in undisplayed:
            reason = f"set as undisplayed by the advanced randomisation of block {undisplayed[qid]}"
        elif bid is None:
            reason = "in no block: never shown"
        elif _is_trash(blocks[bid]):
            reason = "in the Trash / Unused Questions block: never shown"
        else:
            reason = f"in block {bid} ({blocks[bid].get('Description')}), which the survey flow never shows"
        not_questions.append({"qid": qid, "tag": q.get("DataExportTag"), "block": bid, "reason": reason,
                              "text_plain": html_to_text(q.get("QuestionText"), markers=False),
                              "choices": [c["text_plain"] for c in _choices(q, questions, "Choices", "ChoiceOrder")],
                              "columns": [c["text_plain"] for c in _choices(q, questions, "Answers", "AnswerOrder")]})
    return {
        "reader": "agent_pipeline/qsf_flow.py", "reader_version": READER_VERSION,
        "survey": {"id": entry.get("SurveyID"), "name": entry.get("SurveyName"), "language": entry.get("SurveyLanguage"),
                   "options": {"back_button": str(options.get("BackButton", "")).lower() == "true",
                               "next_button": html_to_text(options.get("NextButton"), False) or None,
                               "previous_button": html_to_text(options.get("PreviousButton"), False) or None,
                               "progress_bar": options.get("ProgressBarDisplay")}},
        "flow": flow,
        "block_order": list(reachable_blocks),
        "blocks": reachable_blocks,
        "not_reachable": {"blocks": not_blocks, "questions": not_questions},
        "notes": sorted(set(notes)),
    }


def read_qsf(path: Path) -> dict:
    return read_survey(load_qsf(path))


def correct_answer_checks(report: dict) -> list[dict]:
    """Every question participants saw that accepts only a correct answer (custom validation).

    This holds whether or not the response is forced: Qualtrics refuses any other
    answer, and an empty one, until the participant chooses a correct option. A
    package asks each of these with the correct options as its ``key``.
    """
    checks = []
    for block in (report.get("blocks") or {}).values():
        for question in block.get("questions") or []:
            correct = (question.get("validation") or {}).get("correct_choices")
            if not correct:
                continue
            by_id = {c.get("id"): c for c in question.get("choices") or [] if isinstance(c, dict)}
            checks.append({
                "qid": question.get("qid"), "tag": question.get("tag"), "text": question.get("text_plain") or "",
                "options": [c.get("text_plain", "") for c in question.get("choices") or [] if isinstance(c, dict)],
                "correct": [by_id.get(c.get("id"), {}).get("text_plain") or html_to_text(c.get("text"), markers=False)
                            for c in correct],
                "response": (question.get("validation") or {}).get("response"),
            })
    return checks


# --------------------------------------------------------------------------- reachable text (validator)

def _record_texts(record: dict) -> list[str]:
    texts = [record.get("text_plain") or ""]
    for key in ("choices", "rows", "columns"):
        for item in record.get(key, []) or []:
            texts.append(item.get("text_plain", "") if isinstance(item, dict) else str(item))
    texts += [html_to_text(t, False) for t in record.get("column_labels", []) or []]
    labels = (record.get("scale") or {}).get("labels") if isinstance(record.get("scale"), dict) else None
    texts += [html_to_text(t, False) for t in labels or []]
    message = (record.get("validation") or {}).get("message_description")
    if message:
        texts.append(message)
    return [t for t in texts if t]


def text_sets(report: dict) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Texts participants could see, and (text, qid, reason) for texts they never saw."""
    reachable = [t for b in report["blocks"].values() for q in b["questions"] for t in _record_texts(q)]
    unreachable: list[tuple[str, str, str]] = []
    for q in report["not_reachable"]["questions"]:
        for text in [q.get("text_plain") or ""] + list(q.get("choices", [])) + list(q.get("columns", [])):
            if text:
                unreachable.append((text, q["qid"] + (f" ({q['tag']})" if q.get("tag") else ""), q["reason"]))
    return reachable, unreachable


# Package keys under which a build may list what it left out, by id (never as participant material).
EXCLUDED_KEY_RE = re.compile(r"(?i)(not[_ -]?(shown|reachable|transcribed|used|in[_ -]?flow)|unused|excluded|"
                             r"unreachable|trash|omitted)")
# An item can say so itself: {"qsfReachability": "not-shown"} (or "shown": false).
MARKER_KEY = "qsfReachability"
MARKER_VALUE = "not-shown"
ITEM_LABEL_KEYS = ("origin", "evidence_label", "evidenceLabel", "label")
ITEM_LABELS = ("verbatim", "reported", "derived", "missing", "researcher", "implementation", "unknown")
# Keys that describe an item rather than hold what participants read.
METADATA_KEYS = frozenset((
    "id", "ids", "what", "note", "notes", "reason", "rationale", "derivation", "comment",
    "comments", "why", "kind", "type", "tag", "qsfTag", "qsfId", "qid", "study", "studyIds", "arm",
    "evidence", "evidenceIds", "locator", "sourceId", "sourceIds", "source", "sourceFile", "path", "file",
    "version", "versionUsed", "status", "origin", "state", "evidenceLabel", "label_source", "sha256"))
SOURCE_ID_KEYS = ("sourceId", "sourceIds", "source")
SOURCE_PATH_KEYS = ("sourceFile", "path", "file", "source")
MIN_WORDS = 5  # shorter strings ("Not at all", "Very good") are too common to attribute to one question


def _squash(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text)).replace("*", "")
    text = (text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-").replace("\xa0", " "))
    return re.sub(r"\s+", "", text).casefold()


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


class _Reach:
    """What one survey's flow shows, for asking whether a package string was ever shown."""

    def __init__(self, report: dict):
        reachable, unreachable = text_sets(report)
        self.reachable = "\x00".join(_squash(t) for t in reachable)
        # Unused copies of text the flow also shows are not evidence of anything.
        self.unreachable = [(_squash(t), qid, reason) for t, qid, reason in unreachable
                            if _words(t) >= MIN_WORDS and _squash(t) not in self.reachable]

    def shown(self, text: str) -> bool:
        """Text of at least MIN_WORDS words that a question participants saw carries."""
        return _words(text) >= MIN_WORDS and _squash(text) in self.reachable

    def never_shown(self, text: str) -> tuple[str, str] | None:
        """(qid, reason) when text is survey wording only an unreachable question carries."""
        pieces = [text] + [line for line in re.split(r"\n+", text) if line.strip() != text.strip()]
        for piece in pieces:
            if _words(piece) < MIN_WORDS:
                continue
            needle = _squash(piece)
            if not needle or needle in self.reachable:
                continue
            for hay, qid, reason in self.unreachable:
                if needle in hay or (len(hay) >= 25 and hay in needle):
                    return qid, reason
        return None


MIXED_VALUE = "mixed"  # holds shown text and never-shown text; a person checks it (state "check")
SELECTOR_KEYS = ("id", "qsfId", "qid")  # export tags are not unique (Trash copies reuse them)


def _marked_not_shown(node: dict) -> bool:
    return node.get(MARKER_KEY) == MARKER_VALUE


def _marked_mixed(node: dict) -> bool:
    return node.get(MARKER_KEY) == MIXED_VALUE and node.get("state") == "check"


def _is_item(node: Any) -> bool:
    """An object the repair guard tracks on its own (it carries an evidence label or a state)."""
    return isinstance(node, dict) and (any(node.get(k) in ITEM_LABELS for k in ITEM_LABEL_KEYS)
                                       or node.get("state") in ("confirmed", "check", "missing", "decision"))


def _own_strings(item: dict) -> list[str]:
    """Participant strings of one item, not counting nested items (the guard judges those separately)."""
    out: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key in METADATA_KEYS or EXCLUDED_KEY_RE.search(str(key)) or key == MARKER_KEY or _is_item(child):
                    continue
                walk(child)
        elif isinstance(value, list):
            for child in value:
                if not _is_item(child):
                    walk(child)

    walk({k: v for k, v in item.items()})
    return out


def _ids_marked_not_shown(materials: Any) -> list[str]:
    ids: list[str] = []

    def walk(value: Any, excluded: bool) -> None:
        if isinstance(value, dict):
            if excluded or _marked_not_shown(value):
                ids.extend(value[k] for k in SELECTOR_KEYS if isinstance(value.get(k), str) and value[k])
            for key, child in value.items():
                walk(child, excluded or bool(EXCLUDED_KEY_RE.search(str(key))))
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, str) and excluded:
                    ids.append(child)
                else:
                    walk(child, excluded)

    walk(materials, False)
    return ids


def _run_references(root: Path) -> set[str]:
    """Every string the run uses to pick material: adapter literals and task.json / program step strings."""
    refs: set[str] = set()
    import ast
    try:
        tree = ast.parse((root / "task/adapter.py").read_text(encoding="utf-8"))
        refs |= {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    except (OSError, SyntaxError, ValueError):
        pass

    def walk(value: Any) -> None:
        if isinstance(value, str):
            refs.add(value)
        elif isinstance(value, dict):
            for key, child in value.items():
                if not EXCLUDED_KEY_RE.search(str(key)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for name, pick in (("task/task.json", lambda d: d), ("study.json", lambda d: (d.get("program") or {}).get("steps")
                                                          if isinstance(d, dict) else None)):
        try:
            walk(pick(json.loads((root / name).read_text(encoding="utf-8"))))
        except (OSError, ValueError, AttributeError):
            pass
    return refs


def _text_strings(value: Any, path: str) -> Iterable[tuple[str, str]]:
    """Strings participants would read: skips metadata keys, left-out lists and items marked not shown."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        if _marked_not_shown(value) or _marked_mixed(value):
            return
        for key, item in value.items():
            if not EXCLUDED_KEY_RE.search(str(key)) and key not in METADATA_KEYS:
                yield from _text_strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _text_strings(item, f"{path}[{index}]")


def _refs(node: dict, known_paths: set[str]) -> set[str]:
    """The sources an object declares. Ids always count; a path counts only when it is a recorded source."""
    refs: set[str] = set()
    for key in set(SOURCE_ID_KEYS) | set(SOURCE_PATH_KEYS):
        value = node.get(key)
        for item in value if isinstance(value, list) else [value]:
            candidates: list[tuple[str, str]] = []
            if isinstance(item, str):
                candidates = [(key, item)]
            elif isinstance(item, dict):
                candidates = [(k, v) for k, v in item.items() if k in ("id", "sourceId", "path") and isinstance(v, str)]
            for k, v in candidates:
                if k in SOURCE_ID_KEYS or k in ("id",) or v in known_paths:
                    refs.add(v)
    return refs


def _qsf_scoped(value: Any, path: str, keys: set[str], known_paths: set[str], inherited: bool
                ) -> Iterable[tuple[str, str]]:
    """Participant strings in value whose nearest declared source is one of keys (a .qsf source id or path)."""
    if isinstance(value, dict):
        if _marked_not_shown(value) or _marked_mixed(value):
            return
        refs = _refs(value, known_paths)
        scoped = bool(refs & keys) if refs else inherited
        for key, item in value.items():
            if EXCLUDED_KEY_RE.search(str(key)) or key in METADATA_KEYS:
                continue
            child = f"{path}.{key}" if path else str(key)
            if isinstance(item, str):
                if scoped:
                    yield child, item
            else:
                yield from _qsf_scoped(item, child, keys, known_paths, scoped)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, str):
                if inherited:
                    yield f"{path}[{index}]", item
            else:
                yield from _qsf_scoped(item, f"{path}[{index}]", keys, known_paths, inherited)


def _reaches(root: Path, job: Path) -> tuple[dict[str, _Reach], dict[str, set[str]], set[str]]:
    """Per .qsf source with a survey flow: its reach, the keys that name it; and every recorded source path."""
    try:
        metadata = json.loads((root / "source/paper_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}, set()
    sources = metadata.get("external_sources") if isinstance(metadata, dict) else None
    sources = [s for s in sources if isinstance(s, dict)] if isinstance(sources, list) else []
    known_paths = {s["path"] for s in sources if isinstance(s.get("path"), str)}
    reach: dict[str, _Reach] = {}
    keys_of: dict[str, set[str]] = {}
    for source in sources:
        path = source.get("path")
        if not isinstance(path, str) or not path.lower().endswith(".qsf"):
            continue
        try:
            data = load_qsf(job / path)
            if not _elements(data, "FL"):
                continue  # without a survey flow nothing can be judged unreachable
            survey = read_survey(data)
        except (OSError, NotASurveyExport):
            continue
        except Exception:  # an export this reader cannot follow is skipped, never a build failure
            continue
        sid = str(source.get("id") or path)
        reach[sid] = _Reach(survey)
        keys_of[sid] = {sid, path}
    return reach, keys_of, known_paths


def _adapter_strings(root: Path) -> Iterable[tuple[int, str]]:
    import ast
    try:
        tree = ast.parse((root / "task/adapter.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _words(node.value) >= MIN_WORDS:
            yield node.lineno, node.value


def _mixed_items(value: Any, path: str = "materials/materials.json ") -> list[tuple[str, dict]]:
    found: list[tuple[str, dict]] = []
    if isinstance(value, dict):
        if _marked_mixed(value):
            found.append((path.strip(), value))
        for key, child in value.items():
            found += _mixed_items(child, f"{path}{'.' if path.strip() else ''}{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found += _mixed_items(child, f"{path.rstrip()}[{index}]")
    return found


def _mixed_fields(program: Any) -> list[tuple[str, str, dict]]:
    nodes = program.get("nodes") if isinstance(program, dict) else None
    return [(str(n.get("id")), str(f.get("id")), f) for n in nodes if isinstance(n, dict)
            for f in n.get("fields") or [] if isinstance(f, dict) and _marked_mixed(f)] if isinstance(nodes, list) else []


def package_problems(root: Path, job: Path | None = None) -> list[str]:
    """Participant-facing text a package copies from a .qsf must be text the survey flow shows.

    Checked, for each recorded .qsf source that has a survey flow:
    - program fields classified found-open / found-private from that source;
    - materials/materials.json strings whose declared source (sourceId, source, or a recorded
      source path, inherited from the enclosing object) is that source;
    - string literals in task/adapter.py.
    A string of at least MIN_WORDS words fails when only a question participants never saw
    carries it (the Trash block, a block missing from the flow). Metadata keys (description,
    note, ...), lists of what was left out (notShown, notTranscribed, unused, excluded, ...)
    and objects marked {"qsfReachability": "not-shown"} are not participant text, but the adapter,
    task.json and program steps must not refer to a marked or left-out item's id.
    """
    job = job if job is not None else root.parent.parent
    reach, keys_of, known_paths = _reaches(root, job)
    if not reach:
        return []
    hits: dict[tuple[str, str], list] = {}

    def report(where: str, text: str, sids: Iterable[str]) -> None:
        for sid in sids:
            hit = reach[sid].never_shown(text)
            if hit:
                item = re.match(r"^(.*?\[\d+\])", where)
                hits.setdefault((item.group(1) if item else where, hit[0]), [hit[1], text, 0])[2] += 1
                return

    try:
        study = json.loads((root / "study.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        study = {}
    program = study.get("program") if isinstance(study, dict) else None
    nodes = program.get("nodes") if isinstance(program, dict) else None
    for node in nodes if isinstance(nodes, list) else []:
        fields = node.get("fields") if isinstance(node, dict) else None
        for field in fields if isinstance(fields, list) else []:
            if not isinstance(field, dict) or not isinstance(field.get("extensions"), dict):
                continue
            material = field["extensions"].get("material")
            if not isinstance(material, dict) or material.get("status") not in ("found-open", "found-private"):
                continue
            ids = material.get("sourceIds") if isinstance(material.get("sourceIds"), list) else []
            sids = [i for i in ids if isinstance(i, str) and i in reach]
            for _, text in _text_strings(field.get("value"), ""):
                report(f"{node.get('id')}.{field.get('id')}", text, sids)
    try:
        materials = json.loads((root / "materials/materials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        materials = None
    if materials is not None:
        for sid, keys in keys_of.items():
            for where, text in _qsf_scoped(materials, "", keys, known_paths, False):
                report(f"materials/materials.json {where}", text, [sid])
    for line, text in _adapter_strings(root):
        report(f"task/adapter.py line {line}", text, list(reach))
    problems = []
    for where, item in _mixed_items(materials) + [(f"{n}.{f}", v) for n, f, v in _mixed_fields(program)]:
        own = _own_strings({k: v for k, v in item.items() if k not in (MARKER_KEY, "state")})
        if not any(r.shown(text) for text in own for r in reach.values()):
            problems.append(f"{where} is marked \"qsfReachability\": \"mixed\" but holds no text participants "
                            "saw; an item with only never-shown text is removed or marked \"not-shown\"")
    if materials is not None:
        used = _run_references(root)
        for item_id in dict.fromkeys(_ids_marked_not_shown(materials)):
            if item_id in used:
                problems.append(f"materials/materials.json item {item_id!r} (its id or question id) is marked as "
                                "never shown (or listed "
                                "as left out), but task/adapter.py, task/task.json or the program steps still "
                                "use it; the run must not show it")
    for (where, qid), (reason, text, count) in hits.items():
        more = f" and {count - 1} more of its strings" if count > 1 else ""
        problems.append(f"{where} uses survey text from {qid}, which participants never saw ({reason}; "
                        f"see agent_pipeline/qsf_flow.py's not_reachable list): {_short(text, 70)!r}{more}. "
                        "Take it out of the participant material, or move it under notShown or mark it "
                        "\"qsfReachability\": \"not-shown\" if it must stay on record; if the paper says it "
                        "was shown, ask the researcher about the difference")
    return problems


def never_shown_matcher(root: Path, job: Path | None = None):
    """For the repair guard: ``matches(before, after)`` is true when a verbatim package item holds
    survey text participants never saw and the repair removed it (``after`` None) or only marked it
    {"qsfReachability": "not-shown"}. That is the repair the survey flow check asks for, not a lost
    transcription. Returns None when the package has no .qsf source with a survey flow.
    """
    job = job if job is not None else root.parent.parent
    reach, _, _ = _reaches(root, job)
    if not reach:
        return None

    def shown_text(item: dict) -> dict:
        return {k: v for k, v in item.items() if k not in (MARKER_KEY, "state")}

    def matches(before: dict, after: dict | None) -> bool:
        own = _own_strings(shown_text(before))
        if not any(r.never_shown(text) for text in own for r in reach.values()):
            return False
        mixed = any(r.shown(text) for text in own for r in reach.values())
        if after is None:
            return not mixed  # an item that also holds text participants saw is never deleted
        if shown_text(after) != shown_text(before):
            return False  # nothing but the marker (and, for a mixed item, state check) may change
        if mixed:
            return _marked_mixed(after)
        return _marked_not_shown(after) and after.get("state") == before.get("state")

    return matches


# --------------------------------------------------------------------------- outline

def outline(report: dict) -> str:
    lines = [f"Survey {report['survey'].get('name')!r} ({report['survey'].get('id')}): flow in order",
             f"Back button: {'yes' if report['survey']['options']['back_button'] else 'no'}", ""]
    blocks = report["blocks"]

    def walk(nodes: list[dict], depth: int) -> None:
        pad = "  " * depth
        for node in nodes:
            kind = node["type"]
            if kind == "block":
                block = blocks.get(node["block_id"])
                lines.append(f"{pad}- Block {node['description']!r} [{node['block_id']}]"
                             + (f" — {node['note']}" if node.get("note") else ""))
                if block and node.get("block_id") and not node.get("_printed"):
                    rand = block.get("question_randomisation")
                    if rand:
                        lines.append(f"{pad}    question order: {rand['summary']}")
                        if rand.get("fixed_order"):
                            lines.append(f"{pad}    fixed_order {rand['fixed_order']}, randomize_all {rand.get('randomize_all')}")
                    by_id = {q["qid"]: q for q in block["questions"]}
                    for page in block["pages"]:
                        lines.append(f"{pad}    page {page['page']}:" + (
                            f" (order of {', '.join(page['randomised'])} is random per participant)"
                            if page.get("randomised") else ""))
                        for qid in page["questions"]:
                            q = by_id.get(qid)
                            if not q:
                                continue
                            flags = []
                            v = q["validation"]
                            if v["response"] != "optional":
                                flags.append(f"{v['response']} response")
                            if v.get("correct_choices"):
                                flags.append("correct answer required: " + "; ".join(c["text"] for c in v["correct_choices"])
                                             + (f" (message description: {v['message_description']!r})" if v.get("message_description") else ""))
                            elif v.get("type"):
                                flags.append(f"validation {v['type']}")
                            if q.get("choice_randomisation"):
                                flags.append("answer order randomised")
                            if q.get("display_logic"):
                                flags.append(f"shown only if {q['display_logic']['condition']}")
                            if q.get("skip_logic"):
                                flags.append("skip logic")
                            if q.get("autocomplete"):
                                flags.append("autocomplete from a list not in the export")
                            if q.get("carry_forward"):
                                flags.append("options carried forward from another question")
                            if not q["participant_visible"]:
                                flags.append("not visible (recorded silently)")
                            lines.append(f"{pad}      {qid} {q.get('tag')!s} [{q['kind']}]: {_short(q['text_plain'])}")
                            for key, label in (("choices", "options"), ("rows", "rows"), ("columns", "columns")):
                                if q.get(key):
                                    order = ("base order, shuffled per participant" if key != "columns"
                                             and q.get("choice_randomisation") else "display order")
                                    lines.append(f"{pad}        {label} ({order}): "
                                                 + " | ".join(_short(c["text_plain"], 40) for c in q[key]))
                            if q.get("column_labels"):
                                lines.append(f"{pad}        column labels: " + " | ".join(q["column_labels"]))
                            if flags:
                                lines.append(f"{pad}        " + "; ".join(flags))
            elif kind == "randomizer":
                lines.append(f"{pad}- Randomizer [{node['flow_id']}]: {node['summary']}")
                walk(node["elements"], depth + 1)
            elif kind == "branch":
                lines.append(f"{pad}- Branch [{node['flow_id']}] if {node['condition']}:")
                walk(node["elements"], depth + 1)
            elif kind == "group":
                lines.append(f"{pad}- Group {node.get('description')!r} [{node['flow_id']}]")
                walk(node["elements"], depth + 1)
            elif kind == "embedded_data":
                lines.append(f"{pad}- Embedded data: " + ", ".join(
                    f"{f['field']} = {f['value']!r}" if f["value"] is not None else f"{f['field']} (from {f['set_by']})"
                    for f in node["fields"]))
            elif kind == "end_survey":
                lines.append(f"{pad}- End of survey")
            else:
                lines.append(f"{pad}- {kind} [{node.get('flow_id')}]: {node.get('note')}")
                walk(node.get("elements", []), depth + 1)

    walk(report["flow"], 0)
    lines += ["", "Never shown to participants (do not use as participant material):"]
    for block in report["not_reachable"]["blocks"]:
        lines.append(f"- Block {block['description']!r} [{block['id']}]: {block['reason']} "
                     f"({len(block['questions'])} questions)")
    loose = [q for q in report["not_reachable"]["questions"] if not q.get("block")
             or q["block"] not in {b["id"] for b in report["not_reachable"]["blocks"]}]
    for q in loose:
        lines.append(f"- {q['qid']} {q.get('tag')}: {q['reason']}")
    if report["notes"]:
        lines += ["", "Notes:"] + [f"- {n}" for n in report["notes"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a Qualtrics .qsf survey export in the order participants saw it.")
    parser.add_argument("qsf", type=Path)
    parser.add_argument("--json", type=Path, help="also write the full record as JSON to this path")
    args = parser.parse_args(argv)
    try:
        report = read_qsf(args.qsf)
    except NotASurveyExport as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(outline(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
