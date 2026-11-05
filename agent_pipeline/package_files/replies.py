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
``I choose Be nice``, ``I pick 1 because ...``, ``One`` (a number word, when no
label contains it). A reply naming two different options (``1 or 2``,
``1. Be fair``) is ambiguous and is not read.

Amounts. When an option's whole label is an amount (``0c``, ``$0.10``, ``30 hours``,
``70%``), a reply may name the amount (``10c``, ``$0.10``, ``I give 10 cents``) and is
read as that option. Named currencies are kept apart (``€10`` is not ``$10``); cents
(``c``, ``cents``) are hundredths of whichever currency the list uses. In "I give 3"
the number is the amount given, never option 3; "Option 12" is an option number only. A bare number is read by the option number printed before
it *and* by the amounts: on ``1. 0c 2. 2c ... 6. 10c ... 10. 18c 11. 20c`` the reply
``10`` names option 10 (18c) and the amount 10c (option 6), so it is ambiguous and
asked again; ``0``, ``20`` and ``6`` name one option each and are read. Such a list
asks for the number followed by the option as printed (``6. 10c``).

Printed typos. Labels keep the source's spelling (``Tranfer the 10c``). A reply
that spells exactly one label correctly (``Transfer the 10c``) is read as it: word
for word, at most two words differing by one dropped, added or swapped letter,
same first letter, words of five or more letters, numbers identical. An adapter
can also pass ``aliases`` (other spellings the reader accepts for a label, never
shown). A reply that comes this close to two labels is not read. A label inside a
longer sentence (``I would not be nice``) is not read either: negations make
that guess unsafe. A reply that cannot be read is
asked again once (``retries``); if it still cannot be read the answer is
recorded as unparsed, never guessed.

Unparsed answers are counted, not hidden: :class:`Page` lists them, and a
session ends as ``unparsed`` only when a *required* answer could not be read.
An optional question (``required=False``, e.g. a wrap-up comment) that cannot be
read is recorded and the participant stays in the data. An optional answer the
analysis uses (a contribution in round 7) passes ``retry=True``: it is asked
again once like a required one, and stays unparsed (never 0, never guessed) if
the second reply cannot be read either.

Several numbers in one reply. "I keep 5 and contribute 15" to "How many points do
you contribute?" is read as 15: the number that follows the question's own verb
(contribute, give, send, keep, assign, ...). Two numbers after that verb, or a
negation, are not read.

Several questions in one reply. Besides ``Q1: ...`` lines and a JSON object, a
reply may key its answers by the word that tells the questions apart in their
text (``A: 2, B: 0, C: 3`` for "Points for participant A/B/C", or
``contribute: 5, keep: 15``), or, when every question asks for a number, list the
numbers in order (``2, 0, 3``).

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

VERSION = "4"

RETRY_NOTE = "Your previous reply could not be read{which}. Reply again in the format asked, and nothing else."
AMBIGUOUS_NOTE = ("Your previous reply could be read as more than one option{which}. Reply with the number (or letter) "
                  "of the option you choose followed by the option as it is printed, and nothing else.")


# --------------------------------------------------------------------------- questions

class Question:
    """One question a participant answers.

    kind      "choice" (options), "number" (a numeric entry) or "text" (free text).
    options   the option labels, exactly as participants see them (choice only).
    codes     what a participant types to pick each option. Default: the number
              each label starts with when the labels are a numbered scale (a 1-7
              scale printed as "1 (strongly disagree)", "2", ...: every label
              starts with a different number, set off from its words, or bare
              numbers between labelled ends), else 1, 2, 3, ..., printed before each label. A
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
    retry     ask again once when this answer cannot be read. Default: only when
              ``required``. Pass ``retry=True`` for an optional answer the analysis
              uses (a per-round decision), so one unreadable reply is asked again
              before it is recorded as unparsed.
    aliases   other spellings the reader accepts for an option, never shown: a dict
              from the option's label (or its index) to a list of strings, for a
              label printed with a typo ({"Tranfer the 10c": ["Transfer", ...]}).
              An alias may not be another option's label or alias.
    """

    def __init__(self, id: str, text: str = "", options: Optional[Sequence[str]] = None,
                 kind: Optional[str] = None, codes: Optional[Sequence[Any]] = None, key: Any = None,
                 required: bool = True, multiple: bool = False, minimum: Optional[float] = None,
                 maximum: Optional[float] = None, integer: bool = True,
                 aliases: Optional[Dict[Any, Sequence[str]]] = None, retry: Optional[bool] = None) -> None:
        self.id = str(id)
        self.text = text
        self.kind = kind or ("choice" if options else "text")
        if self.kind not in ("choice", "number", "text"):
            raise ValueError(f"question {id}: kind must be choice, number or text")
        self.options = [str(option) for option in (options or [])]
        if self.kind == "choice" and not self.options:
            raise ValueError(f"question {id}: a choice question needs options")
        self.codes = [str(code) for code in codes] if codes is not None else default_codes(self.options)
        # Positions (1, 2, 3, ...) chosen by default are printed before every label, even one starting with a number;
        # a printed scale's own numbers are its codes and are shown as printed.
        self.numbered = codes is None and self.codes == [str(n) for n in range(1, len(self.options) + 1)] \
            and self.codes != _leading_numbers(self.options)
        if len(self.codes) != len(self.options) or len({c.lower() for c in self.codes}) != len(self.codes):
            raise ValueError(f"question {id}: codes must be distinct and one per option")
        self.key = key
        self.required = required
        self.retry = required if retry is None else bool(retry)
        self.multiple = multiple
        self.minimum = minimum
        self.maximum = maximum
        self.integer = integer
        self.amounts = [_amount(option) for option in self.options]
        self.aliases: List[List[str]] = [[] for _ in self.options]
        for which, spellings in (aliases or {}).items():
            index = which if isinstance(which, int) else (self.options.index(which) if which in self.options else -1)
            if not 0 <= index < len(self.options):
                raise ValueError(f"question {id}: alias target {which!r} is not one of its options")
            self.aliases[index] += [_norm(spelling) for spelling in ([spellings] if isinstance(spellings, str)
                                                                      else spellings) if _norm(spelling)]
        seen: Dict[str, int] = {_norm(label): index for index, label in enumerate(self.options)}
        for index, spellings in enumerate(self.aliases):
            for spelling in spellings:
                if seen.setdefault(spelling, index) != index:
                    raise ValueError(f"question {id}: alias {spelling!r} also names another option")

    def amount_clash(self) -> bool:
        """A printed option number is also the amount of another option ("10. 18c" and "6. 10c")."""
        return any(amount is not None and amount[0] in self.codes and self.codes.index(amount[0]) != index
                   for index, amount in enumerate(self.amounts))

    def spellings(self, index: int) -> List[str]:
        """The normalised label and the aliases of one option."""
        return [_norm(self.options[index])] + self.aliases[index]

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
    steps of one with bare numbers between labelled ends (a "1 Strongly disagree",
    "2", ..., "7 Strongly agree" scale). Otherwise "0 times", "1 time", or
    "70% for auto safety" or "200 people will be saved" would become codes 70 and
    200 that no participant model types.
    """
    numbers = _leading_numbers(options)
    if numbers is not None:
        rests = [LEADING_RE.match(str(option)).group(2) for option in options]  # type: ignore[union-attr]
        values = [int(number) for number in numbers]
        steps = {b - a for a, b in zip(values, values[1:])}
        if not any(AMOUNT_RE.match(rest) for rest in rests) and (
                all(SET_OFF_RE.match(rest) for rest in rests)
                or (len(values) > 1 and steps in ({1}, {-1}) and any(not rest.strip() for rest in rests))):
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


CURRENCY = r"(?:(?:US|CA|AU|NZ|HK|S)?\$|USD|[€£¥]|EUR|GBP|JPY)"
UNIT = (r"(?:%|percent|per\s?cent|pct|¢|c|cents?|p|pence|dollars?|euros?|pounds?|yen|usd|eur|gbp|points?|"
        r"pts?|tokens?|coins?|credits?|ecus?|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|"
        r"years?|people|persons?|lives|times?|items?|units?)")
# A number: thousands set off by commas ("1,000"), a decimal point, a leading point (".5"). A comma is never a
# decimal mark here: "1,5" would be read as one or as fifteen.
NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
AMOUNT_ONLY_RE = re.compile(rf"(?i)^\s*(?P<pre>{CURRENCY})?\s*(?P<num>{NUMBER})\s*(?P<unit>{UNIT})?\s*$")
AMOUNT_IN_TEXT_RE = re.compile(rf"(?i)(?<![\w.,])(?P<pre>{CURRENCY})?\s*(?P<num>{NUMBER})"
                               rf"(?:\s*(?P<unit>{UNIT}))?(?![\w%¢]|[.,]\d)")
# What an amount counts. Cents are hundredths of whatever currency the list uses ("10c" is "$0.10" or "0.10 EUR");
# pence are hundredths of a pound. Other currencies are kept apart ("$10" is not "10 EUR").
UNIT_FAMILY = {"%": "%", "percent": "%", "per cent": "%", "pct": "%",
               "c": "cent", "¢": "cent", "cent": "cent", "p": "pence", "pence": "pence",
               "dollar": "usd", "usd": "usd", "euro": "eur", "eur": "eur", "pound": "gbp", "gbp": "gbp",
               "yen": "jpy", "jpy": "jpy", "pt": "point", "hr": "hour", "min": "minute", "sec": "second",
               "person": "people", "lives": "people"}
CURRENCY_FAMILY = {"$": "usd", "usd": "usd", "€": "eur", "eur": "eur", "£": "gbp", "gbp": "gbp",
                   "¥": "jpy", "jpy": "jpy"}
MONEY = ("usd", "eur", "gbp", "jpy")


def _unit_family(pre: Optional[str], unit: Optional[str]) -> Optional[str]:
    """What an amount counts ("usd", "cent", "%", "hour"); None for a bare number or a mismatch ("$10 hours")."""
    currency = None
    if pre:
        key = pre.lower()
        currency = CURRENCY_FAMILY.get(key) or CURRENCY_FAMILY.get(key[-1], "usd")
    if not unit:
        return currency
    word = re.sub(r"\s+", " ", unit.lower())
    if word not in UNIT_FAMILY and word.endswith("s") and len(word) > 2:
        word = word[:-1]                      # "hours" -> "hour", "points" -> "point"
    family = UNIT_FAMILY.get(word, word)
    if currency is None:
        return family
    if family in ("cent", "pence"):
        return {"cent": currency, "pence": "gbp"}[family] + "/100"
    return currency if family == currency else None


def _numeric(number: str) -> float:
    return float(number.replace(",", "").replace("−", "-").lstrip("+"))


def _amount_of(match: "re.Match[str]") -> Optional[tuple]:
    """(printed number, family, value in the family's base unit) of an amount match; None without a unit."""
    family = _unit_family(match.group("pre"), match.group("unit"))
    if family is None:
        return None
    number = _numeric(match.group("num"))
    value = number
    if family.endswith("/100"):
        family, value = family[:-4], number / 100
    elif family in ("cent", "pence"):
        family, value = ("gbp" if family == "pence" else "money"), number / 100
    return (_number_text(number), family, round(value, 6))


def _number_text(number: Any) -> str:
    """"10", "10.0", "10.00" and "1,0" -> "10"; "0.10" and ".1" -> "0.1"."""
    value = _numeric(number) if isinstance(number, str) else float(number)
    return str(int(value)) if value.is_integer() else repr(value)


def _same_amount(a: tuple, b: tuple) -> bool:
    """One amount: equal values in one family; "money" (cents of an unnamed currency) matches any currency."""
    if round(a[2], 6) != round(b[2], 6):
        return False
    return a[1] == b[1] or ("money" in (a[1], b[1]) and {a[1], b[1]} - {"money"} <= set(MONEY))


def _amount(label: str) -> Optional[tuple]:
    """The amount a label consists of ("0c", "$0.10", "30 hours", "70%"), else None."""
    match = AMOUNT_ONLY_RE.match(str(label).replace("−", "-"))
    return _amount_of(match) if match else None


NUMBER_WORDS = {word: str(number) for number, word in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
NUMBER_WORD_RE = re.compile(r"(?i)\b(" + "|".join(NUMBER_WORDS) + r")\b")
# A number word read as a number: the whole reply, or set off before the rest ("Three (fairly profound)").
LEADING_WORD_RE = re.compile(r"(?i)^(" + "|".join(NUMBER_WORDS) + r")(?=\s*$|\s*[.):=]|\s*\(|\s+[-–—]\s)")
# Labels that count something ("Once", "Never", "1 child", "Two or more"): a number word there is a quantity, not a
# position, so it is never read as an option number.
COUNT_WORDS_RE = re.compile(r"(?i)\d|\b(once|twice|thrice|never|none|nothing|times?|more|fewer|less|"
                            r"several|many|few|couple|dozens?|half|rarely|seldom|sometimes|occasionally|often|"
                            r"frequently|usually|always|daily|weekly|monthly|yearly|hourly|per|every|"
                            + "|".join(NUMBER_WORDS) + r")\b")


def _number_words_allowed(question: "Question") -> bool:
    """Number words are option numbers on a scale whose labels are its numbers, or whose words count nothing."""
    printed = _leading_numbers(question.options)
    if printed is not None and printed == question.codes:
        return True               # "1 (strongly disagree)" ... "7": the words name the printed numbers
    return not any(COUNT_WORDS_RE.search(label) for label in question.options) and \
        not any(COUNT_WORDS_RE.search(alias) for aliases in question.aliases for alias in aliases)


def _number_words(body: str) -> str:
    """``body`` with its number words as digits, when it starts with one ("Three", "Three (fairly profound)")."""
    if not LEADING_WORD_RE.match(body):
        return body
    return NUMBER_WORD_RE.sub(lambda m: NUMBER_WORDS[m.group(1).lower()], body)


def _one_edit(a: str, b: str) -> bool:
    """``a`` and ``b`` differ by one dropped or added letter, not its first: "tranfer"/"transfer".

    Swapped and replaced letters are not typos here: "casual"/"causal", "trail"/"trial" and
    "increase"/"decrease" are different words.
    """
    if abs(len(a) - len(b)) != 1:
        return False
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return i > 0 and short[i:] == long_[i + 1:]


def _near(reply: str, label: str) -> bool:
    """The reply is the label with at most two of its words spelt differently by one letter (a printed typo)."""
    got, want = reply.split(" "), label.split(" ")
    if len(got) != len(want) or got == want:
        return False
    pairs = [(g, w) for g, w in zip(got, want) if g != w]
    return len(pairs) <= 2 and all(
        g.isalpha() and w.isalpha() and min(len(g), len(w)) >= 5 and _one_edit(g, w) for g, w in pairs)


# --------------------------------------------------------------------------- rendering

def render_options(question: Question) -> str:
    return "\n".join(question.option_line(index) for index in range(len(question.options)))


def instruction(question: Question) -> str:
    if question.kind == "choice":
        what = "letter" if all(re.fullmatch(r"[A-Za-z]", code) for code in question.codes) else "number"
        if question.amount_clash():
            # "10" could be option 10 or the amount 10c: ask for both, so the reply names one option.
            if question.multiple:
                return (f"Reply with the {what} of every option that applies, each followed by the option as "
                        "printed, separated by commas.")
            return f"Reply with the {what} of the option you choose, followed by the option as printed."
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


def _amount_options(question: Question, text: str) -> tuple:
    """(options, unknown) for the amounts written in ``text``: "10c", "$0.10", "10 cents", or a bare "10".

    A written unit names the options with that amount; ``unknown`` is True for an
    amount of the list's kind that the list does not offer ("15c" on 0c/10c/20c,
    "1 day" on a list of hours). A bare number names the options whose printed
    amount has that number and, when it is a code, the option it numbers. Only
    options whose whole label is an amount take part.
    """
    found: set = set()
    if not any(question.amounts):
        return found, False
    codes = [code.lower() for code in question.codes]
    for match in AMOUNT_IN_TEXT_RE.finditer(text):
        amount = _amount_of(match)
        number = _number_text(match.group("num"))
        if amount is not None:
            hits = {i for i, own in enumerate(question.amounts) if own and _same_amount(own, amount)}
            if not hits and (match.group("unit") or match.group("pre")):
                return found, True
            found |= hits
        else:
            found |= {i for i, own in enumerate(question.amounts) if own and own[0] == number}
            if number in codes:
                found.add(codes.index(number))
    return found, False


AMOUNT_PART = rf"(?:{CURRENCY})?\s*{NUMBER}(?:\s*{UNIT})?"
# The only sentences an amount is read from: "I give 10c", "I'd transfer $0.10 to the other participant".
# "I keep 20c", "more than 10c", "20c for me" and "200 die" are not read.
AMOUNT_SENTENCE_RE = re.compile(
    rf"(?i)^(?:i(?:\s+will|\s+would|\s+shall|'d|’d|'ll|’ll)?\s+(?:give|transfer|send|donate|offer|"
    rf"allocate|pay|contribute)\s+(?:the\s+other\s+(?:participant|person|player)\s+)?)?(?:{AMOUNT_PART})"
    rf"(?:\s+to\s+(?:the\s+other\s+(?:participant|person|player)|them|my\s+partner))?\s*[.!]?$")


def _label_words(label: str) -> str:
    """A scale label without its number: "7 (strongly agree)" -> "strongly agree"."""
    match = LEADING_RE.match(label)
    words = match.group(2) if match else label
    return _norm(re.sub(r"^\s*[(\[=:)\]/|.\-–—]*\s*", "", words)).strip("()[] ")


def _choice_index(text: str, question: Question) -> tuple:
    """(index, status) for a single option in ``text``; status "ok", "ambiguous" or "unparsed"."""
    codes = [code.lower() for code in question.codes]
    labels = [_norm(label) for label in question.options]
    spellings = [question.spellings(i) for i in range(len(labels))]
    explicit = re.match(r"(?i)^\s*(?:(?:my\s+)?(?:answer|choice)\s*(?:is)?\s*[:=-]?\s*)?(?:option|number|no\.)\s*#?\s*"
                        r"([-+]?\d+)\s*[.!]?\s*$", _clean(text))
    if explicit:  # "Option 12": an option number, never an amount; none beyond the list
        number = explicit.group(1).lstrip("+")
        return (codes.index(number), "ok") if number in codes else (None, "unparsed")
    body = LEAD_RE.sub("", _clean(text), count=1).strip().replace("\u2212", "-")
    body = re.sub(r"(?<![\w.])\.(\d)", r"0.\1", body)          # ".5%" is 0.5%, not 5%
    if all(re.fullmatch(r"[-+]?\d+", code) for code in codes) and _number_words_allowed(question):
        body = _number_words(body)
    norm = _norm(body)
    if not norm:
        return None, "unparsed"

    def label_in(index: int, haystack: str) -> bool:
        return any(spelling and re.search(rf"(?<!\w){re.escape(spelling)}(?!\w)", haystack) is not None
                   for spelling in spellings[index])

    def is_label(index: int, words: str) -> bool:
        return words in spellings[index]

    def near_label(words: str) -> Optional[int]:
        """The one option whose label the words spell with a printed typo corrected; None if not exactly one."""
        close = {i for i in range(len(labels)) for spelling in spellings[i] if _near(words, spelling)}
        return close.pop() if len(close) == 1 else None

    def amount_clash(number: str, index: int) -> bool:
        """The code ``number`` (option ``index``) is also the printed amount of another option."""
        return any(own and own[0] == _number_text(number) and i != index for i, own in enumerate(question.amounts)) \
            if re.fullmatch(r"[-+]?\d+", number) else False

    # 1. The whole reply is a code ("1", "1.", "(1)", "B)"), a label ("be NICE") or an amount ("10c", "0").
    bare = norm.strip("()[]{}")
    found = {i for i in range(len(labels)) if is_label(i, norm)}
    if bare in codes:
        found.add(codes.index(bare))
    whole = re.sub(r"[\s.!;:]+$", "", body).strip("()[] ")
    if AMOUNT_IN_TEXT_RE.fullmatch(whole):
        amounts, unknown = _amount_options(question, whole)
        if unknown and not found:
            return None, "unparsed"
        found |= amounts
    if found:
        return (found.pop(), "ok") if len(found) == 1 else (None, "ambiguous")
    close = near_label(norm)
    if close is not None:
        return close, "ok"
    # 1b. An amount first ("1 day", "15c, I think"): on a list of amounts it is read as one, never as a code.
    lead = AMOUNT_IN_TEXT_RE.match(body)
    if lead and any(question.amounts) and _amount_of(lead) is not None:
        return None, "unparsed" if not _amount_options(question, lead.group(0))[0] else "ambiguous"
    # 2. A code, then text: "1. Be nice", "2) Be fair", "B - Defect", "1 because it is kind".
    match = re.match(r"^[(\[]?([-+]?\w{1,4})(?:[)\].:=]+\s*|\s*[-–—]\s+|\s+)(.*)$", body, re.S)
    if match and match.group(1).lower().lstrip("+") in codes:
        token = match.group(1).lower().lstrip("+")
        index = codes.index(token)
        rest = _norm(match.group(2))
        if rest and (is_label(index, rest) or _norm(question.option_line(index)) == norm or near_label(rest) == index):
            return index, "ok"
        named = [i for i in range(len(labels)) if label_in(i, rest)]
        named = [i for i in named if not any(j != i and labels[i] in labels[j] for j in named)]  # longest label wins
        close = near_label(rest)
        if close is not None and close not in named:
            named.append(close)              # "2. Transfer the 10c" names option 1 with its typo corrected
        words = rest.strip("()[] ")
        anchors = [i for i in range(len(labels)) if words and _label_words(question.options[i]) == words]
        printed = [(LEADING_RE.match(question.options[i]) or [None, None])[1] for i in anchors]
        if anchors and (index not in anchors or any(n is not None and n.lstrip("+") != token for n in printed)):
            return None, "ambiguous"     # "1 (strongly agree)", "3 children": the number and the words disagree
        if named and named != [index]:
            return None, "ambiguous"     # "1. Be fair", "1. Don't hand over": the number and the words disagree
        if named or anchors:
            return index, "ok"
        if amount_clash(token, index):
            return None, "ambiguous"     # "10 because it is fair": option 10 or the amount 10c
        if rest and re.fullmatch(r"\d+", token) and any(
                j != index and (LEADING_RE.match(label) or [None, None])[1] == token
                for j, label in enumerate(question.options)):
            return None, "ambiguous"     # "3 children" on "2 children / 3 or more children": a count, not a code
        if not rest or any(spelling.startswith(rest) for spelling in spellings[index]):
            return index, "ok"
        if _standalone_codes(rest, codes) - {codes[index]} or _hedged(rest) or \
                _amount_options(question, rest)[0] - {index}:
            return None, "ambiguous"     # "1 or 2", "1 is not what I'd pick", "1, that is 10c"
        return index, "ok"
    # 3. One option on its own in a sentence: "I pick 2 because ...", "I will transfer 10c".
    if AMOUNT_SENTENCE_RE.match(body) and any(question.amounts):
        # "I give 3": the number of cents given, never option 3; read only when the list offers that amount.
        found_amount = AMOUNT_IN_TEXT_RE.search(body)
        number = _number_text(found_amount.group("num")) if found_amount else None
        amount = _amount_of(found_amount) if found_amount else None
        hits = {i for i, own in enumerate(question.amounts) if own and (
            _same_amount(own, amount) if amount is not None else own[0] == number)}
        if amount is None and number in codes and hits and hits != {codes.index(number)}:
            return None, "ambiguous"     # "I give 10" on "6. 10c ... 10. 18c": still asked again
        return (hits.pop(), "ok") if len(hits) == 1 else (None, "ambiguous" if hits else "unparsed")
    found_codes = {codes.index(code) for code in _standalone_codes(body, codes)}
    amounts, unknown = set(), False
    named_options = found_codes | amounts
    if (named_options or unknown) and _hedged(body):
        return None, "ambiguous"         # "Not 1", "somewhere around 3-4"
    if len(named_options) == 1 and not unknown:
        index = named_options.pop()
        if amounts or not any(amount_clash(code, index) for code in _standalone_codes(body, codes)):
            return index, "ok"
        return None, "ambiguous"
    return (None, "ambiguous") if named_options or unknown else (None, "unparsed")


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


# Verbs a numeric question asks with ("How many points do you contribute?"): in a reply holding several numbers,
# the one after the question's own verb is its answer ("I keep 5 and contribute 15" -> 15).
ACTION_VERBS = ("contribut", "keep", "kept", "give", "gave", "send", "sent", "transfer", "invest", "offer", "allocat",
                "assign", "deduct", "donat", "pay", "paid", "return", "bid", "spend", "spent", "put", "take", "took",
                "guess", "estimat", "demand", "request", "withdraw", "withdrew", "claim", "accept", "reject", "share",
                "split", "rate", "rating", "choose", "chose", "pick", "select", "enter", "report", "save", "wager", "bet")
# A verb is the stem with a verb ending only: "contribute", "contributed", "contribution", never "between" or "payoff".
VERB_RE = re.compile(r"(?i)\b(" + "|".join(ACTION_VERBS) + r")(?:e|es|ed|ing|s|d|ion|ions|ted|ting|ped|ping|ten|"
                     r"n|al)?\b")
# Numbers that name a round or an item, not an amount ("Round 3: I contribute 8").
COUNTER_RE = re.compile(r"(?i)\b(?:round|period|trial|stage|part|step|block|question|item|q|no\.?|number|#)\s*#?\s*$")
NUMBER_IN_TEXT = r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?![\w.]\d)"
FILLER = r"(?:(?:about|around|exactly|only|just|all|the|my|of|a|an|total|sum|amount|points?|tokens?|coins?|units?|" \
         r"dollars?|cents?|euros?|money|mus?|ecus?|credits?|is|be|will|would|to|it|into|in)\s+){0,4}"


def _asking(text: str) -> str:
    """The sentence that asks: the last one ending in "?", else the last sentence ("Your contribution:")."""
    sentences = [s for s in re.split(r"(?<=[.?!:])\s+|\n+", text or "") if s.strip()]
    asked = [s for s in sentences if s.rstrip().endswith("?")]
    return (asked or sentences or [""])[-1]


def _asked_verbs(question: Question) -> set:
    """The verbs of the sentence that asks ("Each token you keep earns 1 point. How much do you contribute?" asks
    contribute). A sentence naming both sides of a split ("how much do you keep and how much do you give?") asks
    both, so no single number is taken from a reply."""
    return {m.group(1).lower() for m in VERB_RE.finditer(_asking(question.text or ""))}


def _tied(text: str) -> List[tuple]:
    """(verb, number, position of the number) for every number written right after an action verb."""
    out = []
    for m in VERB_RE.finditer(text):
        after = re.match(r"[\s:=]*" + FILLER + r"(" + NUMBER_IN_TEXT[len(r"(?<![\w.])"):] + r")",
                         text[m.end():], re.I)
        if after:
            out.append((m.group(1).lower(), after.group(1), m.end() + after.start(1)))
    return out


# Someone other than the participant: "others contribute 15 on average", "the group gave 12 last round".
OTHERS_RE = re.compile(r"(?i)\b(?:others?|they|their|them|he|she|his|her|people|everyone|everybody|someone|somebody|"
                       r"members?|participants?|players?|partners?|group|average|typically|usually|last|previous|"
                       r"you)\b")
CLAUSE_RE = re.compile(r"(?i)[.;:!?,\n]|\b(?:and|but|so|while|whereas|since|because|as|though|although)\b")


def _by_others(text: str, start: int) -> bool:
    """The clause before a verb names someone else, or the past ("others contribute", "last round I gave")."""
    clause = CLAUSE_RE.split(text[:start])[-1] if text[:start] else ""
    return bool(OTHERS_RE.search(clause))


def _by_others_only(text: str, question: Question) -> bool:
    """Every number tied to the question's verb, or every number of the reply, is someone else's ("Others give 15
    on average."): never read as this participant's answer."""
    tied = _tied(text)
    verbs = _asked_verbs(question)
    asked = [position for verb, _, position in tied if verb in verbs]
    if asked and all(_by_others(text, position) for position in asked):
        return True
    positions = {position for _, _, position in tied}
    numbers = [m.start() for m in re.finditer(NUMBER_IN_TEXT, text)]
    return bool(numbers) and all(n in positions and _by_others(text, n) for n in numbers)


def _verb_number(text: str, question: Question) -> Optional[str]:
    """The one number written right after the question's own verb ("I keep 5 and contribute 15" -> "15"), when
    every other number of the reply is accounted for: tied to another verb, or a round or item counter ("Round 3").
    A number left over ("assign 2 to member 1 and 5 to member 3", "6 for me and 4 for them") leaves it unread."""
    verbs = _asked_verbs(question)
    if not verbs:
        return None
    tied = _tied(text)
    # Someone else's number is accounted for but never the answer ("Others give 5 on average; I contribute 8").
    mine = {number for verb, number, position in tied if verb in verbs and not _by_others(text, position)}
    if len(mine) != 1:
        return None
    accounted = {position for _, _, position in tied}
    for m in re.finditer(NUMBER_IN_TEXT, text):
        if m.start() not in accounted and not COUNTER_RE.search(text[:m.start()]):
            return None
    return mine.pop()


KEEP_VERBS = ("keep", "kept", "save")
PASS_VERBS = ("contribut", "give", "gave", "send", "sent", "transfer", "invest", "offer", "allocat", "donat", "pay",
              "paid", "return", "spend", "spent", "put", "share")


def _other_side(text: str, question: Question) -> bool:
    """The reply ties its number to the opposite of what is asked: "I keep 5" to "How much do you contribute?"
    (5 kept is 15 contributed), or "I give 5" to "How much do you keep?". Asked again, never read."""
    asked = _asked_verbs(question)
    said = {verb for verb, _, _ in _tied(text)}
    if not asked or said & asked:
        return False
    return bool((asked & set(PASS_VERBS) and said & set(KEEP_VERBS)) or
                (asked & set(KEEP_VERBS) and said & set(PASS_VERBS)))


def parse_number(reply: str, question: Question) -> Dict[str, Any]:
    text = LEAD_RE.sub("", _clean(reply), count=1).replace("−", "-")
    if re.fullmatch(r"(?i)\s*(" + "|".join(NUMBER_WORDS) + r")\s*[.!]?\s*", text):
        text = NUMBER_WORDS[text.strip(" .!").lower()]      # "Five"
    numbers = re.findall(NUMBER_IN_TEXT, text.replace(",000", "000"))
    values = {float(n.replace(",", ".")) for n in numbers}
    if not values:
        return _result(question, "unparsed", reply)
    if NEGATION_RE.search(text) or (len(numbers) > 1 and RANGE_RE.search(text)):
        return _result(question, "ambiguous", reply)
    if _other_side(text, question) or _by_others_only(text, question):
        return _result(question, "ambiguous", reply)
    if len(values) > 1:
        tied = _verb_number(text.replace(",000", "000"), question)
        if tied is None:
            return _result(question, "ambiguous", reply)
        values = {float(tied.replace(",", "."))}
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
            labels = question_labels(questions)
            for question in questions:
                hits = [k for k in data if isinstance(k, str) and k.lower() in labels.get(question.id, ())]
                if len(hits) == 1:
                    value = data[hits[0]]
                    parts[question.id] = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
            if parts:
                return parts
    pattern = re.compile(r"(?im)^\s*(?:q(?:uestion)?\s*)(\d+)\s*[:.)=-]\s*")
    marks = list(pattern.finditer(text))
    for position, mark in enumerate(marks):
        number = int(mark.group(1))
        end = marks[position + 1].start() if position + 1 < len(marks) else len(text)
        if 1 <= number <= len(questions) and questions[number - 1].id not in parts:
            parts[questions[number - 1].id] = text[mark.end():end].strip()
    if not parts:
        parts = _labelled(text, questions) or _in_order(text, questions)
    return parts


def question_labels(questions: Sequence[Question]) -> Dict[str, tuple]:
    """Per question, the words of its text that no other question on the screen has (lower case): "a" for
    "Points for participant A" next to B and C; "contribute" next to "keep". A reply may key answers by them."""
    words = [{w.lower() for w in re.findall(r"[A-Za-z0-9]+", question.text or "")} for question in questions]
    stem = lambda w: w[:6] if len(w) > 6 else w      # "contribution" and "contribute" name the same thing
    out = {}
    for index, question in enumerate(questions):
        others = set().union(*(w for j, w in enumerate(words) if j != index)) if len(words) > 1 else set()
        stems = {stem(w) for w in others}
        out[question.id] = tuple(sorted(w for w in words[index] - others if stem(w) not in stems))
    return out


def _labelled(text: str, questions: Sequence[Question]) -> Dict[str, str]:
    """Answers keyed by each question's own label ("A: 2, B: 0, C: 3", "Participant B: 5"): a label followed by
    ":" or "=", its answer up to the next label or the end of the line (a comma inside an answer stays in it). A question whose label is written
    twice with different answers is left out (unparsed, asked again)."""
    labels = question_labels(questions)
    marks = []
    for question in questions:
        for label in labels.get(question.id, ()):
            for m in re.finditer(rf"(?i)(?<![\w]){re.escape(label)}\s*[:=]\s*", text):
                marks.append((m.start(), m.end(), question.id))
    marks.sort()
    found: Dict[str, set] = {}
    for position, (_, end, qid) in enumerate(marks):
        stop = marks[position + 1][0] if position + 1 < len(marks) else len(text)
        value = text[end:stop].split("\n", 1)[0].strip().rstrip(",;").strip()
        if value:
            found.setdefault(qid, set()).add(value)
    return {qid: values.pop() for qid, values in found.items() if len(values) == 1}


def _in_order(text: str, questions: Sequence[Question]) -> Dict[str, str]:
    """"2, 0, 3" for three number questions: one bare number per question, in the order asked."""
    if not questions or any(q.kind != "number" for q in questions) or re.search(r"\d,\d{3}(?!\d)", text):
        return {}                      # "1,000" is a thousand, not two answers
    pieces = [p.strip() for p in re.split(r"[,;\n]", text.strip().rstrip(".")) if p.strip()]
    if len(pieces) != len(questions) or not all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", p) for p in pieces):
        return {}
    return {q.id: p for q, p in zip(questions, pieces)}


# --------------------------------------------------------------------------- run size

def sessions_for(n: Any, arm: str) -> int:
    """How many sessions ``run_sessions(llm, seed, n, arms)`` runs for ``arm``.

    ``n`` is one number for every condition, or a dict from condition id to its
    number, so a run can use the paper's own cell sizes ({"awl_dg": 357, ...}).
    A condition a dict leaves out gets none.
    """
    if isinstance(n, dict):
        return int(n.get(arm, 0) or 0)
    return int(n)


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
    words). An unreadable required answer (or one with ``retry=True``) is asked
    again up to ``retries`` times; answers already read are kept. An answer that
    still cannot be read stays unparsed: the adapter records it, never a default. The model's replies are in ``page.replies``.
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
        missing = [number for number, q in enumerate(questions, 1)
                   if getattr(q, "retry", q.required) and page.answers[q.id]["status"] != "ok"]
        if not missing:
            break
        which = "" if len(questions) == 1 else " for " + ", ".join(f"Q{n}" for n in missing)
        note = retry_note
        if retry_note == RETRY_NOTE and all(page.answers[q.id]["status"] == "ambiguous" and q.kind == "choice" for q in questions
                                            if getattr(q, "retry", q.required) and page.answers[q.id]["status"] != "ok"):
            note = AMBIGUOUS_NOTE  # "10" on an amount list: say why, so the next reply names one option
        current = prompt + "\n\n" + note.format(which=which)
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

    Once per question it also tries the other ways a model names an option: the
    number as a word ("Three"), an amount label's amount ("10c", "10"), and the
    adapter's ``aliases``. Each must be read as that option or asked again; one
    read as a different option is a misread.

    Numbers and several questions are answered the ways models write them too: a
    number in a sentence with the question's verb ("I keep 5 and contribute 15"),
    and a screen of questions keyed by their labels ("A: 2, B: 0, C: 3") or as
    ``Q1:`` lines. A reply the reader takes as another number is a misread.
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
        self.probed: set = set()

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
            return self._number_reply(question, value)
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
        if typed is not None and question.id not in self.probed and not getattr(question, "legacy", False):
            self.probed.add(question.id)
            self._probe(question, index, typed)
        replies = [style.format(code=typed, label=label, upper=label.upper()) for style in self.STYLES]
        if not getattr(question, "legacy", False):
            replies += self._variants(question, index, typed)
        # Only formats that name this option unambiguously ("3" on a scale that prints 3 twice does not).
        readable = [reply for reply in replies if parse_choice(reply, question).get("index") == index]
        return self.rng.choice(readable) if readable else typed

    def _number_reply(self, question: Question, value: Any) -> str:
        """The number alone, or in a sentence with the question's verb next to another number ("I keep 5 and
        contribute 15") when the reader takes that sentence back as the number meant."""
        verb = VERB_RE.search(_asking(question.text or ""))
        if verb is None or getattr(question, "legacy", False) or self.rng.random() < 0.5:
            return str(value)
        word = verb.group(0).lower()
        other = "keep" if not word.startswith("keep") else "give"
        reply = f"I {other} {self.rng.randint(1, 99) + 0.0:g} and {word} {value}."
        answer = parse_number(reply, question)
        if answer.get("status") == "ok" and answer.get("value") != value \
                and not any(m["question"] == question.id for m in self.misreads):
            # A wrong number read with confidence is the worst outcome: always reported.
            self.misreads.append({"question": question.id, "option": str(value), "reply": reply,
                                  "read_as": str(answer.get("value")), "codes": []})
        # Only a sentence the reader takes back as this number: a reply it leaves ambiguous is the reader's
        # caution, not the adapter's fault.
        return reply if answer.get("status") == "ok" and answer.get("value") == value else str(value)

    def _labelled(self, questions: Sequence[Question], answers: List[str]) -> Optional[str]:
        """The answers keyed by each question's label ("A: 2, B: 0"), when every question has one and the reader
        gives each answer back to its own question; else None."""
        if any(getattr(q, "legacy", False) or "\n" in a or "," in a for q, a in zip(questions, answers)):
            return None
        labels = question_labels(questions)
        keys = [min(labels.get(q.id) or ("",), key=len) for q in questions]
        if not all(keys):
            return None
        reply = ", ".join(f"{key.upper() if len(key) == 1 else key}: {answer}" for key, answer in zip(keys, answers))
        return reply if split_reply(reply, questions) == {q.id: a for q, a in zip(questions, answers)} else None

    @staticmethod
    def _variants(question: Question, index: int, typed: str) -> List[str]:
        """Other ways a model names the option: its number as a word, its amount, a spelling the adapter accepts."""
        variants = []
        words = {number: word for word, number in NUMBER_WORDS.items()}
        if typed in words:
            variants += [words[typed].capitalize(), words[typed]]
        amount = question.amounts[index]
        if amount is not None:
            variants += [question.options[index].strip(), amount[0], f"I choose {question.options[index].strip()}."]
        variants += question.aliases[index]
        return variants

    def _probe(self, question: Question, index: int, typed: str) -> None:
        """Each variant is read as this option or asked again; one read as another option is a misread."""
        for reply in self._variants(question, index, typed):
            answer = parse_choice(reply, question)
            if answer.get("status") == "ok" and answer.get("index") != index \
                    and not any(m["question"] == question.id for m in self.misreads):
                self.misreads.append({"question": question.id, "option": question.options[index], "reply": reply,
                                      "read_as": question.options[answer["index"]], "codes": list(question.codes)})
            elif answer.get("status") != "ok" and (reply in question.aliases[index]
                                                   or reply == question.options[index].strip()) \
                    and not any(m["question"] == question.id for m in self.misreads):
                self.misreads.append({"question": question.id, "option": question.options[index], "reply": reply,
                                      "read_as": answer.get("status"), "codes": list(question.codes)})

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

    @staticmethod
    def _current(question: Any) -> Question:
        """A question built by an older copy of this file, as this version's Question."""
        if isinstance(question, Question) and hasattr(question, "aliases"):
            return question
        kind = getattr(question, "kind", "choice")
        current = Question(question.id, getattr(question, "text", ""), getattr(question, "options", None) or None,
                           kind=kind, codes=list(question.codes) if kind == "choice" else None,
                           key=getattr(question, "key", None), required=getattr(question, "required", True),
                           multiple=getattr(question, "multiple", False), minimum=getattr(question, "minimum", None),
                           maximum=getattr(question, "maximum", None), integer=getattr(question, "integer", True))
        current.legacy = True  # read by an older copy: send only the reply forms every version reads
        return current

    def answer_for(self, prompt: str, questions: Sequence[Question], attempt: int = 0, final: bool = True) -> str:
        questions = [self._current(question) for question in questions]
        self.calls += 1
        self.prompts.append(prompt)
        for question in questions:
            self.asked.setdefault(question.id, question)
        if not final and self.rng.random() < self.noise:
            return "Hmm, I am not sure what to say here."
        if len(questions) == 1:
            return self._one(questions[0], prompt)
        answers = [self._one(question, self._region(prompt, number, len(questions)))
                   for number, question in enumerate(questions, 1)]
        keyed = self._labelled(questions, answers) if self.rng.random() < 0.5 else None
        return keyed or "\n".join(f"Q{number}: {answer}" for number, answer in enumerate(answers, 1))


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
            f"question {item['question']}: a model reading the screen may type {item['reply']!r} for "
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
    report = smoke_test(adapter.run_sessions, evaluation.evaluate, arms, blocked, seed=seed)
    sized = [c["id"] for c in task.get("conditions") or [] if isinstance(c, dict) and c.get("id") in arms
             and isinstance(c.get("paper_n"), int)]
    if sized:
        # The paper's cell sizes: run_sessions takes n as {condition id: sessions} too.
        arm = sized[0]
        try:
            batch = adapter.run_sessions(StubParticipant(seed + 2), seed + 2, {arm: 1}, arms=[arm])
            if not any(isinstance(s, dict) and arm in _arm_of(s) for s in batch or []):
                report["problems"].append(f"run_sessions(llm, seed, {{{arm!r}: 1}}, arms=[{arm!r}]) returned no "
                                          "session for that condition; read n with replies.sessions_for(n, arm)")
        except Exception as exc:
            report["problems"].append(f"run_sessions(llm, seed, {{{arm!r}: 1}}, arms=[{arm!r}]) raised "
                                      f"{type(exc).__name__}: {exc}. task/task.json gives paper_n per condition, "
                                      "so n may be a dict of cell sizes: read it with replies.sessions_for(n, arm)")
        report["ok"] = not report["problems"]
    return report


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
