"""Are the paper's results recorded, beyond its tables?

Called by ``package_checks.check_package``. Contract: ``agent_pipeline/CLAUDE.md``,
"Human results to compare against". Builds recorded the printed tables and dropped
results that the paper prints only in its text. This module reads the PDF text and
checks three things against the result nodes of ``study.json.program``:

1. **Text statistics with their p values.** Every exact p value ("p = 0.048"), every
   coefficient, SE, mean, SD, alpha and test statistic, "26% vs 18%" comparisons and
   the sample's make-up (sex, age) in the paper's own results. Fails when fewer than
   half of the paper's exact p values are recorded, and when a results passage that
   prints several of these numbers has none of them recorded (a whole subsection
   dropped while the tables were recorded).
2. **The abstract's results.** Every finding the abstract states ("We found three
   major results: (i) ..., (ii) ..., (iii) ...") is listed in
   ``source/paper_metadata.json`` ``abstract_results`` with the result fields that
   record it, or with the reason it holds no statistic.
3. **Supplement tables** (``paper_tables.check_supplement_tables``) are checked there.

A number the check reads as a result but that is not one (a payoff, a power
analysis, a different paper's result) is cleared by naming its passage in
``source/paper_metadata.json`` ``results_not_recorded``: ``[{"page": 4, "quote":
"<exact text of the passage>", "reason": "..."}]``. The quote must occur in the
paper; every number inside it is then not counted.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

# p values: exact ones only ("p = .048", "p = 0.0008"); a bound such as "p < .001" says little.
P_EXACT_RE = re.compile(r"(?<![\w.])p(?:'s)?\s*=\s*(0?\.\d+|1\.0+|\d\.\d+\s*[x×]\s*10\s*-\s*\d+)")
NUMBER = r"-?(?:\d+\.\d+|\.\d+|\d+)"
STATISTIC_RES = [
    # coefficients and effect sizes: "coeff = 0.25", "b = -0.18", "OR = 4.30", "d = 0.42", "h = 0.71"
    ("estimate", re.compile(r"(?<![\w.])(?:coeff(?:icient)?|b|B|β|beta|estimate|OR|odds ratio|RR|d|g|h|r|ρ|rho|"
                            r"φ|phi|w|V|R2|R²|η2|η²|ηp2|ηp²|η\s*p\s*2)\s*=\s*(" + NUMBER + r")")),
    ("test", re.compile(r"(?<![\w.])(?:t|z|U|W|H|Z)\s*(?:\(\s*\d+(?:\.\d+)?\s*\))?\s*=\s*(" + NUMBER + r")")),
    ("test", re.compile(r"(?<![\w.])F\s*\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)\s*=\s*(" + NUMBER + r")")),
    ("test", re.compile(r"(?:χ|χ2|χ²|chi-square|chi2|X2)\s*(?:2|²)?\s*\([^)=]{0,30}\)\s*=\s*(" + NUMBER + r")")),
    ("descriptive", re.compile(r"(?<![\w.])(?:M|SD|SE|S\.E\.|SEM|Mdn|mean|median|α|alpha)\s*=\s*(" + NUMBER + r")")),
    ("ci", re.compile(r"CI\s*[=:]?\s*[\[(]\s*(" + NUMBER + r")\s*[,;]\s*(" + NUMBER + r")\s*[\])]")),
    # "26% vs 18%", "45% versus 30%"; "12 out of the 16 options"; "85 (45.5%)"
    ("pair", re.compile(r"(\d+(?:\.\d+)?)\s?%\s*(?:vs\.?|versus|compared (?:with|to))\s*(\d+(?:\.\d+)?)\s?%")),
    ("pair", re.compile(r"(?<![\w.])(\d+)\s+out\s+of\s+(?:the\s+|all\s+)?(\d+)(?![\w.%])")),
    ("pair", re.compile(r"(?<![\w.])(\d+)\s*\((\d+(?:\.\d+)?)\s?%\)")),
]
# The sample's make-up: "58 male / 222 female", "164 (52.7%) females", "59% males";
# "mean age = 33.83", "Mage = 38.05, SD = 12.38", "M = 20.9 years".
SEX_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?%?\s*(?:\(\s*(\d+(?:\.\d+)?)\s?%\s*\)\s*)?(?:were\s+|are\s+)?"
                    r"(?:male|female|men|women)s?\b")
AGE_RE = re.compile(r"(?:(?i:mean|average)\s+age|\bM\s?age|\bage\s*\(?\s*M|\bMage)\s*(?:=|of|was|:)?\s*(\d+(?:\.\d+)?)"
                    r"(?:\s*(?:years|yrs)?\s*[,;(]?\s*(?:SD|s\.d\.)\s*=\s*(\d+(?:\.\d+)?))?")
# Numbers just after these words are another study's result, a power analysis or a design value.
NOT_OWN_RE = re.compile(r"(?i)power|sensitivity|pilot|g\*power|meta-analy|previous (stud|research|work)|"
                        r"prior (stud|research|work)|et al\.?,? \(?\d{4}|\(\d{4}[a-z]?\)[^.]{0,60}(found|report|show)|"
                        r"\b(found|reported|showed) (that )?[^.]{0,40}\(\d{4}")
CITATION_RE = re.compile(r"\(\s*[A-Z][A-Za-z'’\-]+(?: et al\.?| (?:and|&) [A-Z][A-Za-z'’\-]+)?,? \d{4}[a-z]?\s*\)|"
                         r"[A-Z][A-Za-z'’\-]+(?: et al\.?| (?:and|&) [A-Z][A-Za-z'’\-]+)? \(\d{4}[a-z]?\)")
# Where the paper's own empirical part begins.
METHOD_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:\d+(?:\.\d+)*\.?[ \t]+)?(?:[A-Z][a-z]+[ \t]+){0,3}(?:methods?|design|procedures?|participants|"
    r"materials|(?:experiment|study)[ \t]+(?:\d|one|I\b)|results)\b[^\n]{0,50}$")
REFERENCES_RE = re.compile(r"(?im)^\s*(references|bibliography|literature cited)\s*$")
ABSTRACT_RE = re.compile(r"(?im)^\s*(abstract|summary)\b[.:]?\s*")
ABSTRACT_END_RE = re.compile(r"(?im)^\s*(keywords?|key words|(?:1\.?\s+)?introduction|significance|"
                             r"public significance|jel)\b")
FINDING_RE = re.compile(
    r"(?i)\b(we (?:also )?(?:find|found|show|showed|observe[ds]?|demonstrate[ds]?|replicat\w+)|results? "
    r"(?:show|showed|indicate[ds]?|reveal(?:ed)?|suggest(?:ed)?|support(?:ed)?)|(?:more|less) likely|"
    r"(?:positively|negatively) (?:correlated|related|associated)|correlat\w+ with|significant(?:ly)? (?:\w+ )?(?:more|less|higher|lower|greater|smaller|larger|different|"
    r"associated|correlated|related|effects?|differences?|predict\w*|increase\w*|decrease\w*|interaction)|"
    r"(?:higher|lower|greater|larger|smaller) than|increase[ds]?|decrease[ds]?|predict(?:s|ed)?|"
    r"(?:was|were) (?:not )?(?:replicated|found|observed)|(?:associated|related) (?:with|to)|"
    r"(?:less|more) evident|fail(?:ed)? to (?:find|replicate|observe)|results? (?:also )?(?:suggest|support)|"
    r"(?:found|find) (?:strong |weak |no |little |some )?(?:empirical )?(?:support|evidence)|"
    r"no (?:significant |reliable )?(?:effect|difference|evidence|association))\b")
ENUMERATION_RE = re.compile(r"\((?:i{1,3}|iv|v|vi{1,3}|[1-9]|[a-f])\)\s+")
STOP = set("""a an the and or but nor of to in on at for with by from as is are was were be been being that this
these those than then it its we our us they their there here which who whom whose what when where how not no
both either neither each every all any some such more less most least very also only even just so too can could
may might will would shall should do does did done have has had having into onto over under between among about
across after before during while if whether because since though although via per vs versus i ii iii iv v vi""".split())
PASSAGE_MIN = 4  # numbers in a results passage before "none recorded" fails
P_SHARE = 0.5


def _norm(text: str) -> str:
    text = (text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-").replace("−", "-").replace(" ", " ")
            .replace(" ", " ").replace(" ", " "))
    return re.sub(r"[ \t]+", " ", text)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pdf_pages(pdf: Path) -> list[str]:
    """Plain text (reading order, not layout) of each page."""
    if not pdf.is_file() or not shutil.which("pdftotext"):
        return []
    try:
        done = subprocess.run(["pdftotext", "-q", str(pdf), "-"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return []
    if done.returncode != 0:
        return []
    pages = done.stdout.split("\f")
    return pages[:-1] if pages and not pages[-1].strip() else pages


# --------------------------------------------------------------------------
# What the paper prints


def _dehyphen(text: str) -> str:
    return re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)


def passages(pages: list[str]) -> list[dict]:
    """Paragraph-sized passages of the paper's own empirical part (after the first
    methods/results heading, before the references), with their page."""
    out: list[dict] = []
    started = False
    stopped = False
    for index, raw in enumerate(pages):
        text = _norm(_dehyphen(raw))
        if stopped:
            break
        refs = REFERENCES_RE.search(text) if index >= len(pages) * 0.4 else None
        if refs:
            text, stopped = text[:refs.start()], True
        if not started:
            # The whole page of the first methods heading: the text layer may put the
            # heading after the paragraphs it heads.
            if index == 0 or METHOD_HEADING_RE.search(text) is None:
                continue
            started = True
        for block in re.split(r"\n\s*\n", text):
            block = re.sub(r"\s*\n\s*", " ", block).strip()
            if block:
                out.append({"page": index + 1, "text": block})
    if not started:  # no heading found: every page after the first
        for index, raw in enumerate(pages[1:], start=1):
            for block in re.split(r"\n\s*\n", _norm(_dehyphen(raw))):
                block = re.sub(r"\s*\n\s*", " ", block).strip()
                if block:
                    out.append({"page": index + 1, "text": block})
    return out


def _own(text: str, start: int) -> bool:
    """A number is the paper's own result unless the words just before it cite
    another study or a power analysis."""
    before = text[max(0, start - 160):start]
    # Only the sentence the number is in ("... reported by Erat and Gneezy (2012). In our
    # sample, men ... (26% vs 18%)" is this paper's result).
    sentences = list(re.finditer(r"(?<!\bal)(?<!\be\.g)(?<!\bi\.e)\.\s+(?=[A-Z(])", before))
    if sentences:
        before = before[sentences[-1].end():]
    return not NOT_OWN_RE.search(before) and not CITATION_RE.search(before[-60:])


def statistics_in(text: str) -> list[dict]:
    """Every result number in a passage: kind, value as printed, offset."""
    found: list[dict] = []
    seen: set[int] = set()
    for match in P_EXACT_RE.finditer(text):
        if _own(text, match.start()) and "x" not in match.group(1) and "×" not in match.group(1):
            found.append({"kind": "p", "value": match.group(1), "at": match.start(1)})
            seen.add(match.start(1))
    for kind, pattern in [("sex", SEX_RE), ("age", AGE_RE)] + STATISTIC_RES:
        for match in pattern.finditer(text):
            if not _own(text, match.start()):
                continue
            # A coding ("gender (1 = female, 0 = male)") is not the sample's make-up.
            if kind == "sex" and (match.group(1) in ("0", "1") or re.search(
                    r"(?i)(cod(ed|ing)|dummy|[=:]|\b[01]\s*[=:5]?)\s*$", text[max(0, match.start() - 14):match.start()])):
                continue
            values = [(match.group(g), match.start(g)) for g in range(1, (pattern.groups or 0) + 1)
                      if match.group(g) is not None]
            values = [(value, start) for value, start in values if start not in seen]
            if not values:
                continue
            group_id = f"{kind}@{match.start()}" if len(values) > 1 else None
            for value, start in values:
                seen.add(start)
                found.append({"kind": kind, "value": value, "at": start, "group": group_id})
    return sorted(found, key=lambda item: item["at"])


# --------------------------------------------------------------------------
# What the package records


def recorded_keys(program: Any) -> set[str]:
    """Every number in the program's result nodes (values and non-evaluation
    extensions), as absolute values at 0-4 decimals (and as a share or percentage)."""
    keys: set[str] = set()
    if not isinstance(program, dict):
        return keys
    for node in program.get("nodes") or []:
        if not isinstance(node, dict) or node.get("kind") != "result":
            continue
        for field in node.get("fields") or []:
            if not isinstance(field, dict):
                continue
            extensions = {k: v for k, v in (field.get("extensions") or {}).items() if k != "evaluation"}
            for text in _strings([field.get("value"), extensions]):
                for token in re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+)", _norm(text)):
                    _add(keys, float(token))
            for number in _numbers([field.get("value"), extensions]):
                _add(keys, number)
    return keys


def _strings(value: Any) -> Iterable[str]:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            yield item


def _numbers(value: Any) -> Iterable[float]:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            yield float(item)


def _add(keys: set[str], number: float) -> None:
    if number != number or number in (float("inf"), float("-inf")):
        return
    for places in range(0, 5):
        keys.add(f"{abs(number):.{places}f}")
    if 0 < abs(number) <= 1:  # a share recorded as 0.26 is the paper's 26%
        for places in range(0, 3):
            keys.add(f"{abs(number) * 100:.{places}f}")
    if 1 < abs(number) <= 100:  # a percentage recorded as 26 is the paper's .26
        for places in range(0, 5):
            keys.add(f"{abs(number) / 100:.{places}f}")


P_KEY_RE = re.compile(r"(?i)^p$|^p[_\- ]|[_\- ]p$|[_\- ]p[_\- ]|p_?val|pvalue|^p\d$|p_printed")
P_TEXT_RE = re.compile(r"(?<![\w.])p(?:'s)?\s*(?:=|<|>|≤|≥|<=|>=)\s*(0?\.\d+|1(?:\.0+)?)")


def recorded_p_keys(program: Any) -> set[str]:
    """The p values recorded in result nodes: numbers under a p key ("p", "p_value",
    "p_printed") and numbers after "p =" / "p <" in recorded strings."""
    keys: set[str] = set()
    if not isinstance(program, dict):
        return keys
    stack: list[tuple[str, Any]] = []
    for node in program.get("nodes") or []:
        if isinstance(node, dict) and node.get("kind") == "result":
            for field in node.get("fields") or []:
                if isinstance(field, dict):
                    extensions = {k: v for k, v in (field.get("extensions") or {}).items() if k != "evaluation"}
                    stack += [(str(field.get("id", "")), field.get("value")), ("", extensions)]
    while stack:
        key, item = stack.pop()
        if isinstance(item, dict):
            stack.extend((str(k), v) for k, v in item.items())
        elif isinstance(item, list):
            stack.extend((key, v) for v in item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            if P_KEY_RE.search(key):
                _add(keys, float(item))
        elif isinstance(item, str):
            if P_KEY_RE.search(key):
                for token in re.findall(r"(?:\d+\.\d+|\.\d+|\d+)", item):
                    _add(keys, float(token))
            for match in P_TEXT_RE.finditer(_norm(item)):
                _add(keys, float(match.group(1)))
    return keys


SEX_WORDS_RE = re.compile(r"(?i)sex|gender|male|female|\bm[ae]n\b|wom[ae]n|girls?|boys?")
AGE_WORDS_RE = re.compile(r"(?i)(?<![a-z])age(?![a-z])|_age|age_|\bage\b|years? old")


def recorded_sample_keys(program: Any) -> dict[str, set[str]]:
    """Numbers recorded about the sample's sex and age, in result or participants nodes:
    under a key, field id or label that names it, or in a string that does."""
    found: dict[str, set[str]] = {"sex": set(), "age": set()}
    if not isinstance(program, dict):
        return found
    stack: list[tuple[str, Any]] = []
    for node in program.get("nodes") or []:
        if isinstance(node, dict) and node.get("kind") in ("result", "participants"):
            for field in node.get("fields") or []:
                if isinstance(field, dict):
                    context = f"{field.get('id', '')} {field.get('label', '')}"
                    stack.append((context, field.get("value")))
    while stack:
        context, item = stack.pop()
        if isinstance(item, dict):
            stack.extend((f"{context} {k}", v) for k, v in item.items())
        elif isinstance(item, list):
            stack.extend((context, v) for v in item)
        elif isinstance(item, (int, float, str)) and not isinstance(item, bool):
            text = item if isinstance(item, str) else ""
            numbers = [float(item)] if not isinstance(item, str) else \
                [float(t) for t in re.findall(r"(?:\d+\.\d+|\.\d+|\d+)", _norm(item))]
            for kind, words in (("sex", SEX_WORDS_RE), ("age", AGE_WORDS_RE)):
                if words.search(context) or words.search(text):
                    for number in numbers:
                        _add(found[kind], number)
    return found


def is_recorded(value: str, keys: set[str]) -> bool:
    value = value.strip()
    places = len(value.split(".", 1)[1]) if "." in value else 0
    try:
        return f"{abs(float(value)):.{places}f}" in keys
    except ValueError:
        return True


def _skipped_spans(root: Path, pages: list[str]) -> list[tuple[int, str]]:
    """Passages the build says hold no result of this paper (results_not_recorded),
    whose quotes occur in the paper."""
    metadata = _load(root / "source/paper_metadata.json")
    items = metadata.get("results_not_recorded") if isinstance(metadata, dict) else None
    spans: list[tuple[int, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        quote = _flat(str(item.get("quote", "")))
        reason = str(item.get("reason", ""))
        if len(quote) < 12 or len(re.findall(r"\w+", reason)) < 3:
            continue
        page = item.get("page")
        spans.append((page if isinstance(page, int) else 0, quote))
    return spans


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", _norm(_dehyphen(text))).strip().casefold()


def skip_problems(root: Path, pages: list[str]) -> list[str]:
    metadata = _load(root / "source/paper_metadata.json")
    items = metadata.get("results_not_recorded") if isinstance(metadata, dict) else None
    if items is None:
        return []
    if not isinstance(items, list):
        return ["human results: source/paper_metadata.json results_not_recorded must be a list of {\"page\", \"quote\", \"reason\"}"]
    whole = _flat(" ".join(pages))
    problems = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            problems.append(f"human results: results_not_recorded[{index}] must be an object with page, quote and reason")
            continue
        quote = _flat(str(item.get("quote", "")))
        if len(quote) < 12 or quote not in whole:
            problems.append(f"human results: results_not_recorded[{index}]: the quote must be the passage's exact text in the "
                            "paper (at least a few words)")
        if len(re.findall(r"\w+", str(item.get("reason", "")))) < 3:
            problems.append(f"human results: results_not_recorded[{index}]: say in 'reason' why the passage holds no result of "
                            "this paper (a payoff or design value, a power analysis, another study's result)")
    return problems


def _in_skipped(passage: dict, offset: int, length: int, spans: list[tuple[int, str]]) -> bool:
    if not spans:
        return False
    flat = _flat(passage["text"])
    # Offsets in the normalised passage map closely onto the flattened text.
    window = _flat(passage["text"][max(0, offset - 200):offset + length + 200])
    for page, quote in spans:
        if page and page != passage["page"]:
            continue
        if quote in flat and (quote in window or _flat(passage["text"][offset:offset + length]) in quote):
            return True
    return False


# --------------------------------------------------------------------------
# 1. Text statistics


def recorded_groups(program: Any) -> list[set[str]]:
    """The numbers recorded together: each object's own numbers (and the numbers in its
    strings), and each string's numbers. "26% vs 18%" or "164 (52.7%)" is recorded when
    both numbers sit in one of these."""
    groups: list[set[str]] = []
    if not isinstance(program, dict):
        return groups
    stack: list[Any] = []
    for node in program.get("nodes") or []:
        if isinstance(node, dict) and node.get("kind") == "result":
            for field in node.get("fields") or []:
                if isinstance(field, dict):
                    extensions = {k: v for k, v in (field.get("extensions") or {}).items() if k != "evaluation"}
                    stack += [field.get("value"), extensions]
    while stack:
        item = stack.pop()
        children = list(item.values()) if isinstance(item, dict) else item if isinstance(item, list) else None
        if children is None:
            if isinstance(item, str):
                group: set[str] = set()
                for token in re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+)", _norm(item)):
                    _add(group, float(token))
                groups.append(group)
            continue
        group = set()
        for child in children:
            if isinstance(child, (int, float)) and not isinstance(child, bool):
                _add(group, float(child))
            elif isinstance(child, str):
                for token in re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+)", _norm(child)):
                    _add(group, float(token))
            elif isinstance(child, (dict, list)):
                stack.append(child)
        if isinstance(item, list):  # a list of numbers ([2.64, 13.15]) belongs with its object
            groups.append(group)
        else:
            groups.append(group)
    return groups


def text_coverage(root: Path, pages: list[str], program: Any) -> dict:
    """The paper's text statistics and which are recorded."""
    keys = recorded_keys(program)
    p_keys = recorded_p_keys(program)
    groups = recorded_groups(program)
    sample_keys = recorded_sample_keys(program)
    spans = _skipped_spans(root, pages)
    report: dict = {"p_values": [], "pairs": [], "sample": [], "passages": []}
    for passage in passages(pages):
        # A line of a table in the text layer ("53 (66%) 27 (34%)") is the table check's.
        if len(re.findall(r"[A-Za-z]{3,}", passage["text"])) < 5:
            continue
        stats = [s for s in statistics_in(passage["text"])
                 if not _in_skipped(passage, s["at"], len(s["value"]), spans) and _nonzero(s["value"])]
        if not stats:
            continue
        for stat in stats:
            stat["recorded"] = is_recorded(stat["value"], p_keys if stat["kind"] == "p" else
                                           sample_keys.get(stat["kind"], keys))
        # Numbers printed together ("26% vs 18%", "164 (52.7%) females", "12 out of 16") are one
        # result: recorded when one recorded object or string holds them all.
        by_group: dict[str, list[dict]] = {}
        for stat in stats:
            if stat.get("group"):
                by_group.setdefault(stat["group"], []).append(stat)
        for members in by_group.values():
            if members[0]["kind"] in sample_keys:  # sex and age: each number on its own, but only as sample data
                continue
            together = any(all(is_recorded(m["value"], group) for m in members) for group in groups)
            for member in members:
                member["recorded"] = together
        # A statistic and the p value printed right after it ("coeff = 0.25, p = 0.25") are one
        # result: recorded only when both are.
        for first, second in zip(stats, stats[1:]):
            if first["kind"] not in ("p", "sex", "age", "pair") and second["kind"] == "p" \
                    and second["at"] - first["at"] <= 40 and not second["recorded"]:
                first["recorded"] = False
        for stat in stats:
            stat["page"] = passage["page"]
        report["p_values"] += [s for s in stats if s["kind"] == "p"]
        report["pairs"] += _one_per_group([s for s in stats if s["kind"] == "pair"])
        report["sample"] += _one_per_group([s for s in stats if s["kind"] in ("sex", "age")])
        report["passages"].append({"page": passage["page"], "text": passage["text"], "stats": stats})
    return report


def _one_per_group(stats: list[dict]) -> list[dict]:
    seen: set = set()
    out = []
    for stat in stats:
        key = stat.get("group") or id(stat)
        if key not in seen:
            seen.add(key)
            out.append(stat)
    return out


def _context(report: dict, stat: dict) -> str:
    """The printed words around a statistic, for a message."""
    for passage in report["passages"]:
        if stat in passage["stats"]:
            text = passage["text"]
            return "\"" + _snippet(text[max(0, stat["at"] - 30):stat["at"] + 40], 70).strip() + "\""
    return stat["value"]


def _nonzero(value: str) -> bool:
    try:
        return float(value) != 0
    except ValueError:
        return False


def _snippet(text: str, limit: int = 70) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def text_problems(root: Path, pages: list[str], program: Any) -> list[str]:
    report = text_coverage(root, pages, program)
    problems: list[str] = []
    p_values = report["p_values"]
    absent = [p for p in p_values if not p["recorded"]]
    if len(p_values) >= 6 and len(absent) > len(p_values) * P_SHARE:
        examples = ", ".join(f"p = {p['value']} (page {p['page']})" for p in absent[:6])
        problems.append(
            f"human results: only {len(p_values) - len(absent)} of the {len(p_values)} exact p values printed in "
            f"the paper's text are recorded in study.json.program result nodes (missing e.g. {examples}). Record "
            "each text statistic with its p value (coefficient, test, effect size, SE), not only the tables.")
    pairs = report["pairs"]
    absent = [p for p in pairs if not p["recorded"]]
    if len(pairs) >= 3 and len(absent) > len(pairs) * P_SHARE:
        examples = "; ".join(f"page {p['page']}: {_context(report, p)}" for p in absent[:5])
        problems.append(
            f"human results: only {len(pairs) - len(absent)} of the {len(pairs)} counts and percentage comparisons "
            f"printed in the paper's text (\"26% vs 18%\", \"12 out of 16\", \"85 (45.5%)\") are recorded together "
            f"in a result node (missing e.g. {examples}). Record each with both numbers in one result field.")
    sample = report["sample"]
    for kind, words in (("sex", "sex or gender"), ("age", "age")):
        printed = [s for s in sample if s["kind"] == kind]
        if printed and not any(s["recorded"] for s in printed):
            examples = "; ".join(f"page {s['page']}: {_context(report, s)}" for s in printed[:3])
            problems.append(
                f"human results: the paper reports its sample's {words} ({examples}) but no result or participants node records it. Record the sample's make-up (counts, percentages, mean and SD) in a result field with "
                "not_ready when the replay does not ask it.")
    dropped = []
    for passage in report["passages"]:
        stats = passage["stats"]
        tests = [s for s in stats if s["kind"] in ("p", "estimate", "test", "comparison", "ci")]
        if len(stats) >= PASSAGE_MIN and tests and not any(s["recorded"] for s in stats):
            values = ", ".join(s["value"] for s in stats[:5])
            dropped.append(f"page {passage['page']} \"{_snippet(passage['text'])}\" ({len(stats)} numbers, "
                           f"e.g. {values})")
    if dropped:
        problems.append(
            "human results: these results passages print numbers none of which is recorded in study.json.program "
            "result nodes: " + "; ".join(dropped[:6]) + (f"; and {len(dropped) - 6} more" if len(dropped) > 6 else "")
            + ". Record each result with its statistics, p values, SEs and n (with not_ready when the replay cannot "
            "compute it). When a passage holds no result of this paper (a payoff, a power analysis, another "
            "study's result), list it in source/paper_metadata.json results_not_recorded as {\"page\": <page>, "
            "\"quote\": \"<its exact text>\", \"reason\": \"...\"}.")
    return problems


# --------------------------------------------------------------------------
# 2. The abstract's results


def abstract_text(pages: list[str]) -> str:
    """The abstract: after an 'Abstract' heading on the first two pages, up to
    the keywords or the introduction. Empty when it cannot be found."""
    for raw in pages[:2]:
        text = _norm(_dehyphen(raw))
        start = ABSTRACT_RE.search(text)
        if start is None:
            continue
        rest = text[start.end():]
        end = ABSTRACT_END_RE.search(rest)
        body = rest[:end.start()] if end else rest[:2500]
        body = re.sub(r"\s*\n\s*", " ", body).strip()
        if len(body.split()) >= 40:
            return body[:4000]
    # No heading: the longest paragraph of the first page before its keywords or introduction.
    if pages:
        text = _norm(_dehyphen(pages[0]))
        end = ABSTRACT_END_RE.search(text)
        if end is not None:
            blocks = [re.sub(r"\s*\n\s*", " ", b).strip() for b in re.split(r"\n\s*\n", text[:end.start()])]
            longest = max(blocks, key=lambda b: len(b.split()), default="")
            if len(longest.split()) >= 80:
                return longest[:4000]
    return ""


def abstract_findings(abstract: str) -> list[str]:
    """The findings the abstract states: each sentence with a result cue, split at
    an enumeration ("(i) ..., (ii) ...")."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", abstract)
    findings: list[str] = []
    for sentence in sentences:
        parts = [p for p in ENUMERATION_RE.split(sentence) if p.strip()]
        if len(parts) > 1:
            # The lead-in ("We found three major results:") is not a finding of its own.
            parts = parts[1:] if not FINDING_RE.search(parts[0]) or parts[0].rstrip().endswith(":") else parts
        for part in parts:
            part = part.strip(" ;,")
            if FINDING_RE.search(part) and len(_words(part)) >= 3:
                findings.append(part)
    return findings


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z'\-]+", text.casefold()) if w not in STOP and len(w) > 2}


def _field_text(program: dict, name: str) -> str | None:
    """Words a result field carries: node and field ids and labels, value strings and
    keys, and the quotes of the evidence it cites."""
    node_id, _, field_id = name.partition(".")
    evidence = {e.get("id"): e for e in program.get("evidence") or [] if isinstance(e, dict)}
    for node in program.get("nodes") or []:
        if not isinstance(node, dict) or node.get("id") != node_id:
            continue
        for field in node.get("fields") or []:
            if isinstance(field, dict) and field.get("id") == field_id:
                parts = [str(node.get("id")), str(node.get("label", "")), str(field.get("id")),
                         str(field.get("label", "")), json.dumps(field.get("value"), ensure_ascii=False),
                         json.dumps(field.get("extensions") or {}, ensure_ascii=False)]
                for eid in field.get("evidenceIds") or field.get("evidence_ids") or []:
                    parts.append(str((evidence.get(eid) or {}).get("quote", "")))
                return re.sub(r"[_\-.]", " ", " ".join(parts))
    return None


def abstract_problems(root: Path, pages: list[str], program: Any) -> list[str]:
    abstract = abstract_text(pages)
    findings = abstract_findings(abstract) if abstract else []
    if not findings or not isinstance(program, dict):
        return []
    metadata = _load(root / "source/paper_metadata.json")
    claims = metadata.get("abstract_results") if isinstance(metadata, dict) else None
    claims = [c for c in claims if isinstance(c, dict)] if isinstance(claims, list) else []
    flat_abstract = _flat(abstract)
    problems: list[str] = []
    uncovered: list[str] = []
    # Words every finding shares (the paper's topic) do not tell findings apart.
    common = set.intersection(*[_words(f) for f in findings]) if len(findings) > 1 else set()
    for finding in findings:
        words = _words(finding)
        best = None
        for claim in claims:
            text = str(claim.get("claim", ""))
            if _flat(text) and _flat(text) in flat_abstract and len(words & _words(text)) >= 0.6 * len(words):
                best = claim
                break
        if best is None:
            uncovered.append(_snippet(finding, 90))
            continue
        targets = best.get("recorded_in")
        targets = [targets] if isinstance(targets, str) else targets if isinstance(targets, list) else []
        reason = str(best.get("not_recorded", ""))
        if not targets:
            if len(re.findall(r"\w+", reason)) < 4:
                problems.append(f"human results: abstract_results \"{_snippet(finding, 60)}\": give recorded_in (the result fields "
                                "that hold it) or not_recorded with the reason it has no statistic")
            continue
        texts = [_field_text(program, str(t)) for t in targets]
        missing = [str(t) for t, text in zip(targets, texts) if text is None]
        if missing:
            problems.append(f"human results: abstract_results \"{_snippet(finding, 60)}\": recorded_in names fields that do not "
                            f"exist ({', '.join(missing[:3])}); use <result node id>.<field id>")
            continue
        distinctive = (words - common) or words
        joined = _words(" ".join(t or "" for t in texts))
        if not distinctive & joined:
            problems.append(f"human results: abstract_results \"{_snippet(finding, 60)}\": none of the fields in recorded_in "
                            f"({', '.join(map(str, targets[:3]))}) is about it (no word of "
                            f"'{', '.join(sorted(distinctive)[:5])}' in their ids, labels, values or cited quotes)")
    if uncovered:
        problems.insert(0, "human results: the abstract states results the package does not map to result nodes: "
                        + "; ".join(f"\"{u}\"" for u in uncovered[:5]) + ". Record each result the abstract names "
                        "(with not_ready when the replay cannot compute it) and list every abstract finding in "
                        "source/paper_metadata.json abstract_results as {\"claim\": \"<the abstract's words>\", "
                        "\"recorded_in\": [\"<result node id>.<field id>\"]} or {\"claim\": ..., \"not_recorded\": "
                        "\"<why it holds no statistic>\"}.")
    return problems


# --------------------------------------------------------------------------


def _job_purpose(job: Path) -> str | None:
    request = _load(job / "studio_request.json")
    return request.get("purpose") if isinstance(request, dict) else None


def check(root: Path, job: Path, pages: list[str] | None = None) -> list[str]:
    """Text statistics and abstract results; an empty list passes."""
    if _job_purpose(job) == "accepted-model-sync":
        return []
    study = _load(root / "study.json")
    program = study.get("program") if isinstance(study, dict) else None
    if not isinstance(program, dict):
        return []
    pages = pdf_pages(job / "input/paper.pdf") if pages is None else pages
    if not pages:
        return []
    return skip_problems(root, pages) + text_problems(root, pages, program) + abstract_problems(root, pages, program)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:  # print what the check reads from a PDF
        found = pdf_pages(Path(sys.argv[1]))
        print(json.dumps({"abstract_findings": abstract_findings(abstract_text(found)),
                          "passages": [{"page": p["page"], "text": _snippet(p["text"], 100),
                                        "stats": [(s["kind"], s["value"]) for s in statistics_in(p["text"])]}
                                       for p in passages(found) if statistics_in(p["text"])]},
                         indent=1, ensure_ascii=False))
    elif len(sys.argv) == 3:
        found = check(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
        print("\n".join(f"- {p}" for p in found) or "no problems")
    else:
        raise SystemExit("usage: results_coverage.py <paper.pdf> | <paper-folder> <job-dir>")
