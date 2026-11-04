"""Reading a participant model's replies, shared by every HumanStudy package.

The build pipeline copies this file unchanged into each package as
``task/replies.py``; the package check fails if the copy differs. Adapters ask
every question through :func:`ask`, so all packages read replies the same way:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from replies import Question, ask, render_questions

    q = Question("pd-decision", text="Which option do you choose?", options=["Cooperate", "Defect"])
    page = ask(llm, screen_text + "\\n\\n" + render_questions([q]), [q])
    if page.ok:
        choice = page.answers["pd-decision"]["label"]

What a reply may look like, for options ``1. Be nice`` / ``2. Be fair``:
``1``, ``1.``, ``1)``, ``1. Be nice``, ``Be nice``, ``be NICE``, ``Answer: 1``,
``I choose Be nice``, ``I pick 1 because ...``. A reply naming two different
options (``1 or 2``, ``1. Be fair``) is ambiguous and is not read. A label inside a
longer sentence (``I would not be nice``) is not read either: negations make
that guess unsafe. A reply that cannot be read is
asked again once (``retries``); if it still cannot be read the answer is
recorded as unparsed, never guessed.

Unparsed answers are counted, not hidden: :class:`Page` lists them, and a
session ends as ``unparsed`` only when a *required* answer could not be read.
An optional question (``required=False``, e.g. a wrap-up comment) that cannot be
read is recorded and the participant stays in the data.

The answer-aware smoke test (:func:`smoke_test`) lives here too, so a package
downloaded from the Hub can run the same check offline:
``python task/adapter.py --smoke-test`` calls ``replies.smoke_main(<package>)``.
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

VERSION = "2"

RETRY_NOTE = "Your previous reply could not be read{which}. Reply again in the format asked, and nothing else."


# --------------------------------------------------------------------------- questions

class Question:
    """One question a participant answers.

    kind      "choice" (options), "number" (a numeric entry) or "text" (free text).
    options   the option labels, exactly as participants see them (choice only).
    codes     what a participant types to pick each option. Default: the number
              each label starts with when the labels are a numbered scale (a 1-7
              scale printed as "1 (strongly disagree)", "2", ...: every label
              starts with a different number, set off from its words or running
              in steps of one), else 1, 2, 3, ..., printed before each label. A
              percentage or an amount ("70% for auto safety", "200 people",
              "30 hours") is never a code. Pass survey codes ("30", "1", "4") or
              letters ("A", "B") only when the screen shows those.
    key       the answer the survey requires to continue: a comprehension or
              attention check, consent, a qualification question. A code or label,
              or a list of accepted ones. For ``multiple=True``, the list of codes
              that together make the correct set. The smoke test answers keyed
              questions correctly; adapters still decide what a wrong answer does.
    required  False for answers the analysis does not use (wrap-up comments):
              when unreadable they are recorded, and the participant stays in.
    multiple  several options may be ticked (a "select all that apply" item).
    minimum, maximum, integer   the accepted range of a numeric entry.
    """

    def __init__(self, id: str, text: str = "", options: Optional[Sequence[str]] = None,
                 kind: Optional[str] = None, codes: Optional[Sequence[Any]] = None, key: Any = None,
                 required: bool = True, multiple: bool = False, minimum: Optional[float] = None,
                 maximum: Optional[float] = None, integer: bool = True) -> None:
        self.id = str(id)
        self.text = text
        self.kind = kind or ("choice" if options else "text")
        if self.kind not in ("choice", "number", "text"):
            raise ValueError(f"question {id}: kind must be choice, number or text")
        self.options = [str(option) for option in (options or [])]
        if self.kind == "choice" and not self.options:
            raise ValueError(f"question {id}: a choice question needs options")
        self.codes = [str(code) for code in codes] if codes is not None else default_codes(self.options)
        # Positions (1, 2, 3, ...) are always printed before the label, even when a label starts with a number.
        self.numbered = codes is None and self.codes == [str(n) for n in range(1, len(self.options) + 1)] \
            and self.codes != _leading_numbers(self.options)
        if len(self.codes) != len(self.options) or len({c.lower() for c in self.codes}) != len(self.codes):
            raise ValueError(f"question {id}: codes must be distinct and one per option")
        self.key = key
        self.required = required
        self.multiple = multiple
        self.minimum = minimum
        self.maximum = maximum
        self.integer = integer

    def option_line(self, index: int) -> str:
        code, label = self.codes[index], self.options[index]
        if not self.numbered and _starts_with_code(label, code):
            return label
        return f"{code}. {label}"

    def key_indices(self) -> List[int]:
        """Indices of the options the key names (empty when there is no key)."""
        if self.key is None or self.kind != "choice":
            return []
        keys = self.key if isinstance(self.key, (list, tuple, set, frozenset)) else [self.key]
        found = []
        for key in keys:
            text = str(key).strip().lower()
            for index, (code, label) in enumerate(zip(self.codes, self.options)):
                if text == code.lower() or _norm(text) == _norm(label):
                    found.append(index)
                    break
            else:
                raise ValueError(f"question {self.id}: key {key!r} is not one of its options")
        return found


LEADING_RE = re.compile(r"\s*([-+]?\d+)(?!\d)(?![.,]\d)(.*)$", re.S)
# What follows a scale point's number: nothing, a line break, or a separator ("1 (low)", "1 = low", "1 - low", "1: low").
SET_OFF_RE = re.compile(r"\s*$|[ \t]*\n|\s*[(\[=:)\]/|]|\s*\.(?!\d)|\s+[-\u2013\u2014]\s|\s*[-\u2013\u2014]\s")
AMOUNT_RE = re.compile(r"(?i)\s*(%|percent\b|per\s?cent\b|pct\b|[$\u20ac\u00a3\u00a5]|(us)?d(ollars?)?\b|cents?\b|"
                       r"euros?\b|pounds?\b|yen\b|usd\b|eur\b|gbp\b)")


def _leading_numbers(options: Sequence[str]) -> Optional[List[str]]:
    """The distinct number every label starts with, when all do; else None."""
    matches = [LEADING_RE.match(str(option)) for option in options]
    if not options or not all(matches):
        return None
    numbers = [m.group(1).lstrip("+") for m in matches if m]
    return numbers if len(set(numbers)) == len(numbers) else None


def default_codes(options: Sequence[str]) -> List[str]:
    """The numbers of a printed scale, else positions 1, 2, 3, ...

    A label's leading number is its code only when every label starts with a
    different number, none of them is a percentage or an amount, and the numbers
    are set off from the words ("1 (strongly disagree)", "0\nNeutral") or run in
    steps of one (a "1 Strongly disagree" ... "7 Strongly agree" scale). Otherwise
    "70% for auto safety" or "200 people will be saved" would become codes 70 and
    200 that no participant model types.
    """
    numbers = _leading_numbers(options)
    if numbers is not None:
        rests = [LEADING_RE.match(str(option)).group(2) for option in options]  # type: ignore[union-attr]
        values = [int(number) for number in numbers]
        steps = {b - a for a, b in zip(values, values[1:])}
        if not any(AMOUNT_RE.match(rest) for rest in rests) and (
                all(SET_OFF_RE.match(rest) for rest in rests) or (len(values) > 1 and steps in ({1}, {-1}))):
            return numbers
    return [str(number) for number in range(1, len(options) + 1)]


def _starts_with_code(label: str, code: str) -> bool:
    """The label already prints its code first ("1 (low)"), so it is shown alone. "70%" is not code 70."""
    match = re.match(rf"\s*\+?{re.escape(code)}(?![\w.,]\d)(?=\b|\s|$|[(\[:.)-])", label)
    return bool(match) and not re.match(rf"\s*\+?{re.escape(code)}\w", label) \
        and not AMOUNT_RE.match(label[match.end():])  # type: ignore[union-attr]


def _norm(text: str) -> str:
    text = str(text).lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[\s ]+", " ", text)
    return text.strip().strip(" .,;:!?\"'`*_()[]")


# --------------------------------------------------------------------------- rendering

def render_options(question: Question) -> str:
    return "\n".join(question.option_line(index) for index in range(len(question.options)))


def instruction(question: Question) -> str:
    if question.kind == "choice":
        what = "letter" if all(re.fullmatch(r"[A-Za-z]", code) for code in question.codes) else "number"
        if question.multiple:
            return f"Reply with the {what} of every option that applies, separated by commas."
        return f"Reply with the {what} of the option you choose."
    if question.kind == "number":
        span = ""
        if question.minimum is not None and question.maximum is not None:
            span = f" from {_fmt(question.minimum)} to {_fmt(question.maximum)}"
        return f"Reply with a {'whole ' if question.integer else ''}number{span}."
    return "Reply in your own words."


def render_questions(questions: Sequence[Question]) -> str:
    """The questions, their options and how to reply, as participants should see them.

    One question: its text, options and a one-line reply instruction. Several
    questions on one screen: numbered Q1, Q2, ... and the reply asks for one line
    per question (``Q1: <answer>``). Survey ids never appear in the prompt.
    """
    if len(questions) == 1:
        question = questions[0]
        parts = [question.text] if question.text else []
        if question.kind == "choice":
            parts.append(render_options(question))
        parts.append(instruction(question))
        return "\n".join(parts)
    blocks = []
    for number, question in enumerate(questions, 1):
        lines = [f"Q{number}. {question.text}".rstrip()]
        if question.kind == "choice":
            lines += ["   " + line for line in render_options(question).split("\n")]
        lines.append("   " + instruction(question))
        blocks.append("\n".join(lines))
    blocks.append("Reply with one line per question, in the form Q1: <your answer>, Q2: <your answer>, and so on.")
    return "\n\n".join(blocks)


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


# --------------------------------------------------------------------------- parsing

LEAD_RE = re.compile(r"(?i)^\s*(?:(?:my\s+)?(?:final\s+)?(?:answer|choice|response|selection|option)\s*(?:is)?\s*[:=-]?\s*"
                     r"|i\s+(?:would\s+)?(?:choose|pick|select|go\s+with)\s*(?:option\s*)?:?\s*)")


def _clean(reply: str) -> str:
    text = str(reply or "").strip()
    text = re.sub(r"[*_`]+", "", text)              # markdown emphasis
    text = re.sub(r"(?is)^<answer>(.*)</answer>$", r"\1", text).strip()
    text = text.strip().strip("\"'")
    return text


def _result(question: Question, status: str, raw: Any, **values: Any) -> Dict[str, Any]:
    result = {"id": question.id, "status": status, "raw": raw if isinstance(raw, str) else str(raw)}
    result.update(values)
    return result


NEGATION_RE = re.compile(r"(?i)\b(not|never|refuse[sd]?|won't|wouldn't|don't|doesn't|didn't|no)\b|n't\b")
RANGE_RE = re.compile(r"\d\s*(?:-|\u2013|\u2014|to)\s*[-+]?\d")


def _hedged(text: str) -> bool:
    """A negation ("not 1") or a range ("3-4") makes a code in prose unsafe to read."""
    return bool(NEGATION_RE.search(text) or RANGE_RE.search(text))


def _standalone_codes(text: str, codes: Sequence[str]) -> set:
    """Codes written on their own in a sentence. Letters count only as "option B", "(B)" or "B)"."""
    if all(re.fullmatch(r"[-+]?\d+", code) for code in codes):
        found = {m.group(1).lstrip("+") for m in re.finditer(r"(?<![\w$.,%-])([-+]?\d+)(?![\w%]|[.,]\d)", text)}
    else:
        found = {m.group(1).lower() for m in re.finditer(
            r"(?i)(?:\b(?:option|choice|answer)\s*[:#]?\s*\(?|\()([a-z0-9]{1,4})(?:\)|(?!\w))", text)}
        found |= {m.group(1).lower() for m in re.finditer(r"(?<![\w(])([A-Za-z0-9]{1,4})\)", text)}
    return {code for code in found if code in codes}


def _choice_index(text: str, question: Question) -> tuple:
    """(index, status) for a single option in ``text``; status "ok", "ambiguous" or "unparsed"."""
    codes = [code.lower() for code in question.codes]
    labels = [_norm(label) for label in question.options]
    body = LEAD_RE.sub("", _clean(text), count=1).strip()
    norm = _norm(body)
    if not norm:
        return None, "unparsed"

    def label_in(index: int, haystack: str) -> bool:
        return bool(labels[index]) and re.search(rf"(?<!\w){re.escape(labels[index])}(?!\w)", haystack) is not None

    # 1. The whole reply is a code ("1", "1.", "(1)", "B)") or a label ("be NICE").
    bare = norm.strip("()[]{}")
    found = {i for i, label in enumerate(labels) if label == norm}
    if bare in codes:
        found.add(codes.index(bare))
    if found:
        return (found.pop(), "ok") if len(found) == 1 else (None, "ambiguous")
    # 2. A code, then text: "1. Be nice", "2) Be fair", "B - Defect", "1 because it is kind".
    match = re.match(r"^[(\[]?([-+]?\w{1,4})(?:[)\].:=]+\s*|\s*[-–—]\s+|\s+)(.*)$", body, re.S)
    if match and match.group(1).lower().lstrip("+") in codes:
        index = codes.index(match.group(1).lower().lstrip("+"))
        rest = _norm(match.group(2))
        if not rest or rest == labels[index] or labels[index].startswith(rest) \
                or _norm(question.option_line(index)) == norm:
            return index, "ok"
        named = [i for i in range(len(labels)) if label_in(i, rest)]
        named = [i for i in named if not any(j != i and labels[i] in labels[j] for j in named)]  # longest label wins
        if named and named != [index]:
            return None, "ambiguous"     # "1. Be fair", "1. Don't hand over": the number and the words disagree
        if named:
            return index, "ok"
        if _standalone_codes(rest, codes) - {codes[index]} or _hedged(rest):
            return None, "ambiguous"     # "1 or 2", "1 is not what I'd pick"
        return index, "ok"
    # 3. One code on its own in a sentence: "I pick 2 because ...".
    found_codes = _standalone_codes(body, codes)
    if found_codes and _hedged(body):
        return None, "ambiguous"         # "Not 1", "somewhere around 3-4"
    if len(found_codes) == 1:
        return codes.index(found_codes.pop()), "ok"
    return (None, "ambiguous") if found_codes else (None, "unparsed")


def parse_choice(reply: str, question: Question) -> Dict[str, Any]:
    """Read one option (or, for ``multiple``, a set of options) from a reply."""
    if question.multiple:
        text = LEAD_RE.sub("", _clean(reply), count=1)
        exact = _norm(text) in [_norm(label) for label in question.options]  # a label with a comma in it
        if not exact and NEGATION_RE.search(text):
            return _result(question, "ambiguous", reply)   # "1 and not 2"
        parts = [text] if exact else [p for p in re.split(r"\s*(?:,|;|/|\band\b|\n|&)\s*", text) if p.strip()]
        indices: List[int] = []
        for part in parts:
            index, status = _choice_index(part, question)
            if index is None:
                return _result(question, status, reply)
            if index not in indices:
                indices.append(index)
        if not indices:
            return _result(question, "unparsed", reply)
        indices.sort()
        return _result(question, "ok", reply, indices=indices, codes=[question.codes[i] for i in indices],
                       labels=[question.options[i] for i in indices])
    index, status = _choice_index(reply, question)
    if index is None:
        return _result(question, status, reply)
    return _result(question, "ok", reply, index=index, code=question.codes[index], label=question.options[index])


def parse_number(reply: str, question: Question) -> Dict[str, Any]:
    text = LEAD_RE.sub("", _clean(reply), count=1).replace("−", "-")
    numbers = re.findall(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?![\w.]\d)", text.replace(",000", "000"))
    values = {float(n.replace(",", ".")) for n in numbers}
    if not values:
        return _result(question, "unparsed", reply)
    if NEGATION_RE.search(text) or (len(numbers) > 1 and RANGE_RE.search(text)):
        return _result(question, "ambiguous", reply)
    if len(values) > 1:
        return _result(question, "ambiguous", reply)
    value = values.pop()
    if question.integer and not value.is_integer():
        return _result(question, "unparsed", reply)
    if (question.minimum is not None and value < question.minimum) or \
            (question.maximum is not None and value > question.maximum):
        return _result(question, "out_of_range", reply, value=value)
    return _result(question, "ok", reply, value=int(value) if question.integer else value)


def parse_text(reply: str, question: Question) -> Dict[str, Any]:
    text = _clean(reply)
    text = LEAD_RE.sub("", text, count=1).strip() if re.match(r"(?i)\s*answer\s*:", text) else text
    if not text:
        return _result(question, "unparsed", reply)
    return _result(question, "ok", reply, value=text[:4000])


def parse(reply: str, question: Question) -> Dict[str, Any]:
    """Read one question's answer from a reply. ``status`` is "ok" or why it was not read."""
    if question.kind == "choice":
        return parse_choice(reply, question)
    if question.kind == "number":
        return parse_number(reply, question)
    return parse_text(reply, question)


def split_reply(reply: str, questions: Sequence[Question]) -> Dict[str, str]:
    """The part of a multi-question reply that answers each question (Q1: ..., or a JSON object)."""
    text = _clean(reply)
    parts: Dict[str, str] = {}
    blob = re.search(r"\{.*\}", text, re.S)
    if blob:
        try:
            data = json.loads(blob.group(0))
        except ValueError:
            data = None
        if isinstance(data, dict):
            for number, question in enumerate(questions, 1):
                for key in (f"Q{number}", f"q{number}", str(number), question.id):
                    if key in data:
                        value = data[key]
                        parts[question.id] = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
                        break
            if parts:
                return parts
    pattern = re.compile(r"(?im)^\s*(?:q(?:uestion)?\s*)(\d+)\s*[:.)=-]\s*")
    marks = list(pattern.finditer(text))
    for position, mark in enumerate(marks):
        number = int(mark.group(1))
        end = marks[position + 1].start() if position + 1 < len(marks) else len(text)
        if 1 <= number <= len(questions) and questions[number - 1].id not in parts:
            parts[questions[number - 1].id] = text[mark.end():end].strip()
    return parts


# --------------------------------------------------------------------------- asking

class Page:
    """The answers to one prompt: ``answers[id]`` is the parse result of each question."""

    def __init__(self, questions: Sequence[Question]) -> None:
        self.questions = list(questions)
        self.answers: Dict[str, Dict[str, Any]] = {}
        self.replies: List[str] = []

    @property
    def ok(self) -> bool:
        """Every required answer was read."""
        return all(self.answers.get(q.id, {}).get("status") == "ok" for q in self.questions if q.required)

    @property
    def unparsed(self) -> List[Dict[str, Any]]:
        """Each answer that could not be read, required or not, with the replies tried."""
        return [{"question": q.id, "required": q.required, "status": self.answers.get(q.id, {}).get("status", "unparsed"),
                 "replies": list(self.replies)}
                for q in self.questions if self.answers.get(q.id, {}).get("status") != "ok"]

    def value(self, question_id: str) -> Any:
        """The label (choice), labels (multiple), number or text of an answer; None when unread."""
        answer = self.answers.get(question_id, {})
        if answer.get("status") != "ok":
            return None
        for key in ("label", "labels", "value"):
            if key in answer:
                return answer[key]
        return None


def ask(llm: Callable[[str], str], prompt: str, questions: Sequence[Question], retries: int = 1,
        retry_note: str = RETRY_NOTE) -> Page:
    """Ask one prompt and read every question's answer from the reply.

    ``prompt`` is the full text the participant sees, ending with
    ``render_questions(questions)`` (or the same questions in the package's own
    words). An unreadable required answer is asked again up to ``retries`` times;
    answers already read are kept. The model's replies are in ``page.replies``.
    """
    if isinstance(questions, Question):
        questions = [questions]
    page = Page(questions)
    current = prompt
    for attempt in range(retries + 1):
        final = attempt == retries
        stub = getattr(llm, "answer_for", None)
        reply = stub(current, page.questions, attempt=attempt, final=final) if callable(stub) else llm(current)
        reply = reply if isinstance(reply, str) else str(reply)
        page.replies.append(reply)
        pieces = {questions[0].id: reply} if len(questions) == 1 else split_reply(reply, questions)
        for question in questions:
            if page.answers.get(question.id, {}).get("status") == "ok":
                continue
            piece = pieces.get(question.id)
            page.answers[question.id] = parse(piece, question) if piece is not None else \
                _result(question, "unparsed", "")
        if page.ok:
            break
        missing = [number for number, q in enumerate(questions, 1)
                   if q.required and page.answers[q.id]["status"] != "ok"]
        which = "" if len(questions) == 1 else " for " + ", ".join(f"Q{n}" for n in missing)
        current = prompt + "\n\n" + retry_note.format(which=which)
    return page


# --------------------------------------------------------------------------- smoke test

class StubParticipant:
    """An offline stand-in model for the smoke test only. It is not a participant.

    It answers the way a model reading the prompt would: it finds each option in
    the prompt text and replies with the number (or letter) printed before it,
    or, when the prompt prints the options without numbers, with the option's
    position, as a model told "reply with the number" does. It never reads
    ``Question.codes`` to write a reply. Each reply is checked against
    :func:`parse`: when the number shown on the screen is read as another option,
    or not read at all, the question is recorded in ``misreads``, because a real
    model's answer would be lost or misrecorded the same way.

    With ``use_keys=True`` it answers keyed questions (comprehension, attention,
    consent, qualification) correctly, so the smoke test can tell whether every
    gate is passable; with ``use_keys=False`` it is blind to keys and picks
    options at random. It varies the reply format ("2", "2.", "2. Label",
    "label", "Answer: 2") and sometimes sends an unreadable first reply so the
    retry path runs. Calls that bypass :func:`ask` are counted in
    ``direct_calls`` and answered with a random number shown in the prompt.
    """

    STYLES = ("{code}", "{code}.", "{code}. {label}", "{label}", "{upper}", "Answer: {code}", "I choose {label}.")

    def __init__(self, seed: int = 0, noise: float = 0.1, use_keys: bool = True) -> None:
        self.rng = random.Random(seed)
        self.noise = noise
        self.use_keys = use_keys
        self.calls = 0
        self.direct_calls = 0
        self.prompts: List[str] = []
        self.misreads: List[Dict[str, Any]] = []
        self.unshown: List[str] = []
        self.ambiguous: List[str] = []
        self.asked: Dict[str, Question] = {}

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.direct_calls += 1
        self.prompts.append(prompt)
        shown = re.findall(r"(?m)^\s*\(?([0-9]{1,3}|[A-H])[.):=]\s+\S", prompt)
        return self.rng.choice(shown) if shown else "1"

    # -- reading the prompt

    @staticmethod
    def _region(prompt: str, number: int, total: int) -> str:
        """The part of the prompt that shows question ``number`` of ``total`` (Q1., Q2., ...)."""
        if total == 1:
            return prompt
        starts = list(re.finditer(rf"(?m)^[ \t]*Q{number}[.:)]\s", prompt))
        if not starts:
            return prompt
        begin = starts[-1].start()
        after = re.compile(rf"(?m)^[ \t]*Q{number + 1}[.:)]\s").search(prompt, starts[-1].end())
        return prompt[begin:after.start() if after else len(prompt)]

    @staticmethod
    def _line_pattern(label: str) -> "re.Pattern[str]":
        words = [re.escape(word) for word in str(label).split()]
        body = r"\s+".join(words) if words else re.escape(str(label))
        selector = r"(?:\(?(?P<sel>[-+]?\d{1,4}|[A-Za-z])\s*[.):=\]]\s*|(?P<dash>[-+]?\d{1,4}|[A-Za-z])\s+[-–—]\s+)"
        return re.compile(rf"(?im)^[ \t>*-]*{selector}?[*_]*{body}[*_]*[ \t]*$")

    def _shown(self, region: str, question: Question) -> List[Optional[str]]:
        """Per option, what a model would type to pick it, read from the prompt; None when not shown.

        The options are looked for as the list they were rendered in (in order,
        the last such list in ``region``), so a label printed twice ("3" on both
        sides of a bipolar scale) gets the number printed before that copy.
        """
        patterns = [self._line_pattern(label) for label in question.options]
        letters = all(re.fullmatch(r"[A-Za-z]", code) for code in question.codes)

        def typed(match: Any, index: int) -> Optional[str]:
            if match is None:
                return None
            selector = match.group("sel") or match.group("dash")
            if selector:
                return selector
            leading = LEADING_RE.match(question.options[index])
            if leading and not AMOUNT_RE.match(leading.group(2)):
                return leading.group(1).lstrip("+")  # "1 (strongly disagree)": the label's own number
            return chr(ord("A") + index) if letters else str(index + 1)  # unnumbered: its position

        for first in reversed(list(patterns[0].finditer(region))):
            found, position = [first], first.end()
            for pattern in patterns[1:]:
                match = pattern.search(region, position)
                if match is None:
                    break
                found.append(match)
                position = match.end()
            if len(found) == len(patterns):
                return [typed(match, index) for index, match in enumerate(found)]
        lasts = [list(pattern.finditer(region)) for pattern in patterns]
        return [typed(matches[-1], index) if matches else None for index, matches in enumerate(lasts)]

    # -- answering

    def _one(self, question: Question, region: str) -> str:
        if question.kind == "text":
            return "No comment."
        if question.kind == "number":
            if question.key is not None and self.use_keys:
                key = question.key
                return str(key[0] if isinstance(key, (list, tuple)) and key else key)
            low = question.minimum if question.minimum is not None else 0
            high = question.maximum if question.maximum is not None else max(low, 10)
            value = self.rng.randint(int(low), int(high)) if question.integer else round(self.rng.uniform(low, high), 2)
            return str(value)
        keyed = question.key_indices() if self.use_keys else []
        shown = self._shown(region, question)
        if question.multiple:
            indices = keyed or sorted(self.rng.sample(range(len(question.options)),
                                                      self.rng.randint(1, len(question.options))))
            if all(shown[i] is not None for i in indices):
                reply = ", ".join(str(shown[i]) for i in indices)
                self._check(question, reply, indices)
                return reply
            self.unshown.append(question.id)
            return ", ".join(question.codes[i] for i in indices)
        index = keyed[0] if keyed else self.rng.randrange(len(question.options))
        label = question.options[index]
        typed = shown[index]
        if typed is None:
            # Not printed as a line of its own: a model would type the words it was asked about.
            self.unshown.append(question.id)
            typed = question.codes[index]
        elif not self._check(question, typed, [index]) and question.id not in self.ambiguous:
            return typed  # what a model would send; the adapter must cope with it as it would with a real reply
        replies = [style.format(code=typed, label=label, upper=label.upper()) for style in self.STYLES]
        # Only formats that name this option unambiguously ("3" on a scale that prints 3 twice does not).
        readable = [reply for reply in replies if parse_choice(reply, question).get("index") == index]
        return self.rng.choice(readable) if readable else typed

    def _check(self, question: Question, reply: str, indices: List[int]) -> bool:
        """True when ``reply`` (what the screen tells a model to type) is read as the options meant."""
        answer = parse_choice(reply, question)
        got = answer.get("indices", [answer["index"]] if "index" in answer else None)
        if answer.get("status") == "ok" and sorted(got or []) == sorted(indices):
            return True
        if answer.get("status") == "ambiguous":
            # A number that is also another option's label ("3" on both sides of a bipolar scale): not
            # guessed, asked again. Reported, but a reply naming the label too ("2. 3") is read.
            if question.id not in self.ambiguous:
                self.ambiguous.append(question.id)
            return False
        if not any(m["question"] == question.id for m in self.misreads):
            self.misreads.append({
                "question": question.id, "option": " | ".join(question.options[i] for i in indices),
                "reply": reply, "read_as": (" | ".join(question.options[i] for i in got) if got else answer.get("status")),
                "codes": list(question.codes)})
        return False

    def answer_for(self, prompt: str, questions: Sequence[Question], attempt: int = 0, final: bool = True) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        for question in questions:
            self.asked.setdefault(question.id, question)
        if not final and self.rng.random() < self.noise:
            return "Hmm, I am not sure what to say here."
        if len(questions) == 1:
            return self._one(questions[0], prompt)
        return "\n".join(f"Q{number}: {self._one(question, self._region(prompt, number, len(questions)))}"
                         for number, question in enumerate(questions, 1))


def _asked_summary(question: Question) -> Dict[str, Any]:
    try:
        key = [question.options[i] for i in question.key_indices()] if question.kind == "choice" else question.key
    except ValueError:
        key = question.key
    return {"id": question.id, "kind": question.kind, "text": question.text[:600], "options": list(question.options),
            "key": key, "required": question.required}


COMPLETED = ("completed", "complete")  # "complete": packages built before the contract named it


def _arm_of(session: Dict[str, Any]) -> List[str]:
    arms: List[str] = []
    for key in ("arm", "condition"):
        if isinstance(session.get(key), str):
            arms.append(session[key])
    if isinstance(session.get("arms"), list):
        arms += [a for a in session["arms"] if isinstance(a, str)]
    return arms


def runnable_arms(task: Dict[str, Any]) -> tuple:
    """(runnable arm ids, blocked arm ids) from task/task.json ``conditions``."""
    conditions = task.get("conditions") or []
    if isinstance(conditions, dict):
        conditions = [dict(value, id=key) if isinstance(value, dict) else {"id": key} for key, value in conditions.items()]
    blocked_ids = set()
    for item in task.get("blocked_arms") or []:
        blocked_ids.add(item.get("id") if isinstance(item, dict) else item)
    runnable, blocked = [], []
    for condition in conditions:
        if not isinstance(condition, dict) or not isinstance(condition.get("id"), str):
            continue
        if condition.get("status") == "blocked" or condition["id"] in blocked_ids:
            blocked.append(condition["id"])
        else:
            runnable.append(condition["id"])
    return runnable, blocked


def smoke_test(run_sessions: Callable, evaluate: Callable, arms: Sequence[str], blocked: Sequence[str] = (),
               seed: int = 0, sizes: Iterable[int] = (2, 8, 32)) -> Dict[str, Any]:
    """Run every arm with :class:`StubParticipant` and check it reaches ``evaluate``.

    Two passes, both replying from the prompt text as a model would:

    - a key-blind reader (``report["reader"]``): every reply it types from what
      the screen prints must be read as the option it meant;
    - a keyed pass (``report["arms"]``): it answers every keyed check correctly,
      so each runnable arm must have at least one completed session.

    Passes when both hold, no model call bypasses :func:`ask`, and ``evaluate``
    returns a non-empty result for the sessions. Sessions are reported by status
    exactly as the adapter labelled them, so an excluded or unparsed session is
    never counted as completed. ``report["asked"]`` lists every question asked
    (id, text, options, key), for checks against the survey's own answer rules.
    """
    stub = StubParticipant(seed)
    report: Dict[str, Any] = {"ok": False, "arms": {}, "blocked_arms": list(blocked), "problems": []}
    every: List[Dict[str, Any]] = []
    seen = set()
    for arm in arms:
        sessions: List[Dict[str, Any]] = []
        calls_before = stub.calls
        for size in sizes:
            try:
                batch = run_sessions(stub, seed, size, arms=[arm])
            except Exception as exc:  # the adapter's own error, reported as-is
                report["problems"].append(f"run_sessions(arms=[{arm!r}], n={size}) raised "
                                          f"{type(exc).__name__}: {exc}")
                break
            sessions = [s for s in batch if isinstance(s, dict) and arm in _arm_of(s)]
            for session in batch:
                if isinstance(session, dict) and id(session) not in seen:
                    seen.add(id(session))
                    every.append(session)
            if any(s.get("status") in COMPLETED for s in sessions):
                break
        statuses: Dict[str, int] = {}
        for session in sessions:
            status = str(session.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
        completed = sum(n for status, n in statuses.items() if status in COMPLETED)
        report["arms"][arm] = {"sessions": len(sessions), "completed": completed, "statuses": statuses,
                               "model_calls": stub.calls - calls_before}
        if sessions and stub.calls == calls_before:
            report["problems"].append(f"arm {arm}: sessions were returned without asking the model once; every "
                                      "participant's answers must come from llm")
        if completed:
            try:
                result = evaluate([s for s in sessions if s.get("status") in COMPLETED])
                if not isinstance(result, dict) or not result:
                    report["problems"].append(f"arm {arm}: evaluate() on this arm's sessions returned nothing")
            except Exception as exc:
                report["problems"].append(f"arm {arm}: evaluate() on this arm's sessions alone raised "
                                          f"{type(exc).__name__}: {exc} (the playground can run any subset of arms)")
        if not sessions:
            report["problems"].append(f"arm {arm}: run_sessions returned no session labelled with this arm")
        elif not completed:
            why = ("a stub that answers every keyed check correctly" if not stub.direct_calls else
                   "the stub, which cannot see answer keys when the adapter bypasses replies.ask()")
            report["problems"].append(f"arm {arm}: no session completed with {why} ({statuses}); "
                                      "evaluate() would get nothing for this arm")
    reader = StubParticipant(seed + 1, use_keys=False)
    report["reader"] = {}
    for arm in arms:
        try:
            batch = run_sessions(reader, seed + 1, 4, arms=[arm])
        except Exception as exc:
            report["problems"].append(f"run_sessions(arms=[{arm!r}], n=4) raised {type(exc).__name__}: {exc} with a "
                                      "participant that does not know the correct answers to the checks")
            continue
        statuses = {}
        for session in batch if isinstance(batch, list) else []:
            if isinstance(session, dict) and arm in _arm_of(session):
                statuses[str(session.get("status"))] = statuses.get(str(session.get("status")), 0) + 1
        report["reader"][arm] = statuses
    seen_misreads = set()
    for item in stub.misreads + reader.misreads:
        if item["question"] in seen_misreads:
            continue
        seen_misreads.add(item["question"])
        report["problems"].append(
            f"question {item['question']}: the prompt tells a model to type {item['reply']!r} for "
            f"{item['option'][:80]!r}, but replies.ask reads {item['reply']!r} as {str(item['read_as'])[:80]!r} "
            f"(the question's codes are {', '.join(item['codes'])}). A participant model's answer would be lost or "
            "misrecorded. Ask with replies.render_questions (it prints each option's number), and pass codes= only "
            "when the screen prints those codes")
    asked = dict(reader.asked)
    asked.update(stub.asked)
    report["asked"] = [_asked_summary(question) for question in asked.values()]
    report["unshown_options"] = sorted(set(stub.unshown + reader.unshown))[:20]
    report["ambiguous_numbers"] = sorted(set(stub.ambiguous + reader.ambiguous))[:20]
    report["model_calls"] = stub.calls + reader.calls
    stub.direct_calls += reader.direct_calls
    report["direct_model_calls"] = stub.direct_calls
    if stub.direct_calls:
        report["problems"].append(f"{stub.direct_calls} of {report['model_calls']} model calls did not go through "
                                  "replies.ask(); every question must be asked with it")
    try:
        result = evaluate(every)
        if not isinstance(result, dict) or not result:
            report["problems"].append(f"evaluate() returned {type(result).__name__} {str(result)[:80]!r}, "
                                      "not a non-empty dict")
        else:
            json.dumps(result, default=str)
            report["evaluate_keys"] = sorted(map(str, result))[:20]
    except Exception as exc:
        report["problems"].append(f"evaluate() raised {type(exc).__name__}: {exc}\n"
                                  + "".join(traceback.format_exc(limit=3)))
    report["ok"] = not report["problems"]
    return report


def _import(path: Path, name: str) -> Any:
    for directory in (path.parent, path.parent.parent):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def smoke_package(root: Path, seed: int = 0) -> Dict[str, Any]:
    """Load a package's adapter and evaluation and run :func:`smoke_test` on it."""
    root = Path(root).resolve()
    sys.dont_write_bytecode = True  # a smoke test leaves no __pycache__ in the package
    task = json.loads((root / "task" / "task.json").read_text(encoding="utf-8"))
    arms, blocked = runnable_arms(task)
    adapter = _import(root / "task" / "adapter.py", "package_adapter")
    evaluation = _import(root / "evaluation" / "evaluation.py", "package_evaluation")
    if not arms:
        return {"ok": False, "arms": {}, "blocked_arms": blocked,
                "problems": ["task/task.json lists no runnable conditions (each needs an id)"]}
    return smoke_test(adapter.run_sessions, evaluation.evaluate, arms, blocked, seed=seed)


def smoke_main(package_root: Any = None) -> int:
    """``python task/adapter.py --smoke-test``: print the report; exit 1 when it fails."""
    root = Path(package_root) if package_root else Path(__file__).resolve().parents[1]
    if root.is_file():
        root = root.resolve().parents[1]
    report = smoke_package(root)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(smoke_main(sys.argv[1] if len(sys.argv) > 1 else None))
