#!/usr/bin/env python3
"""The paper's tables and the numbers its text layer gets wrong.

    python agent_pipeline/paper_tables.py <job>/input/paper.pdf [--render <job>/work/pages]

Prints, as JSON:

- ``tables``: every numbered table of this paper ("Table 3", not a supplementary
  "Table 3S" or "Table S3"), found from its caption or, when the caption is not in
  the text layer, from the text that mentions it ("In Table 1 and 2, we ..."). Each
  says on which page it is captioned or mentioned and whether the text layer holds
  its rows. A table without a text layer (drawn as vector graphics or an image) can
  only be read from the rendered page: ``--render`` writes those pages as PNG files
  (pdftoppm) so they can be opened and read.
- ``sign_loss``: numbers whose minus sign the plain text layer drops. A minus at the
  end of a line ("95% CI [-" / "0.16, 0.54]") is taken for a hyphen and removed when
  the line is joined, so the plain text reads "[0.16, 0.54]"; a true minus (U+2212)
  split from its digits ("− 0.011") is easy to miss as well. Check each on the rendered
  page and record the signed value.

The package check (``validate_package.py``) uses the same functions: a table without a
text layer must be listed in ``source/paper_metadata.json`` ``paper_tables`` and, when
it holds results, recorded cell by cell; a number listed here must be recorded with its
sign, or its evidence must say the rendered page was checked.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# A caption: "Table 3", "Table 3." or "Table 3: ..." or "TABLE 3 Means ...", never a sentence such as
# "Table 3 presents the means" (a mention that starts a line).
# Also "Table 1 | Design", "Table 1 – Means", "Table\t1".
CAPTION_RE = re.compile(r"(?:^\s*|\s{2,})(?:Table|TABLE)[ \t]+(\d{1,2})(?![\dSs])"
                        r"(?:\s*$|\s{3,}\S[^\n]*$|\s*[.:|\u2013\u2014-]\s*(\S[^\n]*)?$|\s{1,2}([A-Z(][^\n]*)$)")
# A mention of a supplementary or appendix table ("Supplementary Table 3", "Extended Data Table 1",
# "Table 8 in S1 Appendix"): not a table of this PDF's main text.
SUPPLEMENT_BEFORE_RE = re.compile(r"(?i)(supplement\w*|supplemen-|extended data|\bSI|\bS\d+|appendix|online)\s*$")
SUPPLEMENT_AFTER_RE = re.compile(r"(?i)^\s*(of|in)\s+(the\s+)?(S\d+\s+)?(online\s+)?(appendix|supplement\w*|SI\b|"
                                 r"electronic supplementary)")
RESULT_WORDS_RE = re.compile(r"(?i)result|statistic|descriptive|mean|summar|effect|test|estimate|frequenc|"
                             r"proportion|percent|odds|correlat|regression|anova|chi|choice|rate|compar")
# "Table 1", "Tables 1 and 2", "Table 3 and 4", "Tables 2-4"; never "Table 2S" or "Table S2".
MENTION_RE = re.compile(r"\b(?:Table|TABLE)s?\s+(\d{1,2})(?![\dSs])((?:\s*(?:,|and|&|to|-|–)\s*\d{1,2}(?![\dSs]))*)")
ELSEWHERE_RE = re.compile(r"(?i)\b(et al\.?|\(\d{4}\)|\d{4}[a-z]?\)|'s|’s)\s*$")
NUMBER_RE = r"(\d+(?:\.\d+)?|\.\d+)"
NUMERIC_CELL_RE = re.compile(r"^[\[(]?-?(?:\d+(?:\.\d+)?|\.\d+)%?[\])]?[,;*†‡]*$")


def _run(command: list[str]) -> str:
    if not shutil.which(command[0]):
        return ""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def page_texts(pdf: Path, layout: bool) -> list[str]:
    """The text layer of each page (pdftotext pages are separated by form feeds)."""
    text = _run(["pdftotext", "-q"] + (["-layout"] if layout else []) + [str(pdf), "-"])
    pages = text.split("\f")
    return pages[:-1] if pages and not pages[-1].strip() else pages


def _references_start(pages: list[str]) -> int:
    """Index of the page where the reference list begins (tables after it still count)."""
    for index in range(len(pages) - 1, -1, -1):
        if re.search(r"(?im)^\s*(references|bibliography|literature cited)\s*$", pages[index]) \
                and index >= len(pages) * 0.4:
            return index
    return len(pages)


def _table_row(line: str) -> bool:
    """A line of a table's body: two or more numeric cells, and numbers outweigh words.
    Prose of a neighbouring column (a cell of four or more words) is not part of it."""
    cells = [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]
    cells = [c for c in cells if len(re.findall(r"[A-Za-z]{2,}", c)) < 4]
    tokens = [t.strip(",;*†‡") for c in cells for t in c.replace("–", "-").replace("−", "-").split()]
    numeric = [t for t in tokens if t and NUMERIC_CELL_RE.match(t)]
    words = [t for t in tokens if re.search(r"[A-Za-z]{3,}", t)]
    return len(numeric) >= 2 and len(numeric) >= len(words)


def _expand(first: str, rest: str) -> list[str]:
    numbers = [int(first)]
    for joiner, value in re.findall(r"(,|and|&|to|-|–)\s*(\d{1,2})", rest):
        value = int(value)
        if joiner in ("to", "-", "–") and numbers and numbers[-1] < value <= numbers[-1] + 12:
            numbers += list(range(numbers[-1] + 1, value + 1))
        else:
            numbers.append(value)
    return [str(n) for n in numbers]


def table_inventory(pdf: Path) -> list[dict]:
    """Every numbered table: caption page and text, pages that mention it, and how many
    table rows the text layer holds next to its caption (0 when it has no text layer)."""
    layout = page_texts(pdf, layout=True)
    if not layout:
        return []
    end = _references_start(layout)
    tables: dict[str, dict] = {}

    def entry(number: str) -> dict:
        return tables.setdefault(number, {"table": number, "caption_page": None, "caption": "",
                                          "mentioned_on": [], "text_rows": 0, "mention_context": ""})

    for index, page in enumerate(layout):
        lines = page.splitlines()
        for at, line in enumerate(lines):
            caption = CAPTION_RE.search(line)
            if caption:
                table = entry(caption.group(1))
                # In a two-column page only the caption's column holds the table.
                column = caption.start(0) + len(caption.group(0)) - len(caption.group(0).lstrip())
                head = re.match(r"(?:Table|TABLE)[ \t]+\d{1,2}", line[column:])
                after = line[column + (head.end() if head else 0):]
                if len(after) - len(after.lstrip(" .:|\u2013\u2014-")) >= 8 or not after.strip(" .:|\u2013\u2014-"):
                    # The title is on the next line, in the caption's column (the rest is another column).
                    nxt = lines[at + 1][max(0, column - 2):] if at + 1 < len(lines) else ""
                    title = nxt.strip()
                else:
                    title = after.strip(" .:|\u2013\u2014-")
                title = re.split(r"\s{4,}", title)[0][:120]
                window = lines[max(0, at - 45):at] + lines[at + 1:at + 46]
                rows = sum(1 for row in window if _table_row(row[max(0, column - 4):]))
                # A sentence that starts a line ("Table 4. For men, ...") also looks like a caption:
                # keep the candidate with the most table rows next to it.
                if table["caption_page"] is None or rows > table["text_rows"]:
                    table["caption_page"], table["caption"], table["text_rows"] = index + 1, title, rows
            if index >= end:
                continue
            previous = lines[at - 1].rstrip() if at else ""
            for segment in re.split(r"\s{2,}", line):
                for match in MENTION_RE.finditer(segment):
                    before = segment[:match.start()].rstrip()
                    # "Supplemen-" / "tary Table 3" across a line break; a mention that starts a line.
                    if re.fullmatch(r"(?i)(tary|tal)?", before) and re.search(r"(?i)supplemen-?$|supplementary$|"
                                                                            r"extended data$|appendix$", previous):
                        continue
                    if ELSEWHERE_RE.search(before) or SUPPLEMENT_BEFORE_RE.search(before) \
                            or SUPPLEMENT_AFTER_RE.match(segment[match.end():]) or \
                            re.match(r"\s*(of|in|from)\s+[A-Z][a-z]+(\s+et al|\s*\(|\s+and\s+[A-Z])", segment[match.end():]):
                        continue
                    for number in _expand(match.group(1), match.group(2) or ""):
                        table = entry(number)
                        if index + 1 not in table["mentioned_on"]:
                            table["mentioned_on"].append(index + 1)
                        table["mention_context"] += " " + segment[max(0, match.start() - 80):match.end() + 120]
    out = []
    for number in sorted(tables, key=int):
        table = tables[number]
        table["text_layer"] = table["caption_page"] is not None and table["text_rows"] >= 3
        table["needs_page_image"] = needs_page_image(table)
        out.append(table)
    return out


RESULT_CAPTION_RE = re.compile(
    r"(?i)(result|mean|descriptive|statistic|effect|correlat|regression|model|analys|frequenc|proportion|"
    r"percent|anova|test|estimate|coefficient|rating|sample|outcome|choice|score|response|comparison|summary)")
NON_RESULT_CAPTION_RE = re.compile(
    r"(?i)(power|sensitivity|taxonomy|methodolog|pre-?registration|procedure|material|stimul|wording|"
    r"instruction|hypothes[ie]s(?! and a comparison)|design\b|previous|prior\b|literature|meta-analy|deviation from|"
    r"changes to|analysis plan)")


def needs_page_image(table: dict) -> bool:
    """A table whose numbers the text layer cannot give: mentioned but never captioned in
    the text, or captioned as results with no table rows next to the caption."""
    if table.get("text_layer"):
        return False
    if table.get("caption_page") is None:
        # Only mentioned (Xiao: "In Table 1 and 2, we summarized ... the results"): a table of
        # results when the sentences that mention it say so.
        return bool(RESULT_WORDS_RE.search(table.get("mention_context") or ""))
    caption = table.get("caption") or ""
    if NON_RESULT_CAPTION_RE.search(caption) and not re.search(r"(?i)result|effect|mean", caption):
        return False
    return bool(RESULT_CAPTION_RE.search(caption))


def pages_to_render(table: dict) -> list[int]:
    if table.get("caption_page"):
        return [table["caption_page"]]
    pages: list[int] = []
    for page in table.get("mentioned_on") or []:
        pages += [page, page + 1]
    return sorted(set(pages))


def render_pages(pdf: Path, pages: list[int], out_dir: Path, dpi: int = 150) -> list[str]:
    """Render pages to PNG with pdftoppm; returns the files written."""
    if not shutil.which("pdftoppm"):
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for page in sorted(set(pages)):
        prefix = out_dir / f"page-{page:02d}"
        subprocess.run(["pdftoppm", "-q", "-r", str(dpi), "-png", "-singlefile", "-f", str(page), "-l", str(page),
                        str(pdf), str(prefix)], capture_output=True, timeout=120)
        if prefix.with_suffix(".png").is_file():
            written.append(str(prefix.with_suffix(".png")))
    return written


def _squash(text: str) -> str:
    text = (text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            .replace(" ", " ").replace(" ", " ").replace(" ", " "))
    return re.sub(r"\s+", " ", text)


SPACE = r"[ \t   ]"
# A minus ending a line in the layout text, not after a letter or digit (so not "self-" or "1-").
LINE_END_MINUS_RE = re.compile(r"(?:^|(?<=[\[(=,:;<>])|(?<=\s))\s*[-−]$")
# A true minus split from its digits where a number starts ("= − 0.011", "[− 0.25"),
# never "n − 1" (a subtraction).
SPLIT_MINUS_RE = re.compile(r"((?:^|[=(\[,:])" + SPACE + r"*)−" + SPACE + r"+" + NUMBER_RE, re.MULTILINE)


def sign_loss(pdf: Path) -> list[dict]:
    """Numbers printed with a minus that the plain text layer reads as positive.

    Each item: page, the value as the plain text reads it, the signed value, and the
    context. Found two ways: a minus that ends a line in the layout text (not after a
    letter, so not a hyphenated word) whose number the plain text joins without it,
    and a U+2212 minus separated from its digits where a number starts."""
    layout = page_texts(pdf, layout=True)
    plain = page_texts(pdf, layout=False)
    found: list[dict] = []
    seen: set[tuple] = set()
    end = _references_start(layout)
    for index, page in enumerate(layout[:end + 1]):
        raw = plain[index] if index < len(plain) else ""
        text = _squash(raw)
        lines = page.splitlines()
        if index == end:  # the reference list's own page: stop at its heading
            heading = next((i for i, l in enumerate(lines)
                            if re.match(r"(?i)\s*(references|bibliography|literature cited)\s*$", l)), len(lines))
            lines = lines[:heading]
        for at, line in enumerate(lines):
            # Column segments with their start column ("text  text" stays one segment).
            for segment_match in re.finditer(r"\S+(?: \S+)*", line.rstrip()):
                segment, column = segment_match.group(0), segment_match.start()
                match = LINE_END_MINUS_RE.search(segment)
                if not match:
                    continue
                tail = _squash(segment[:match.start()]).rstrip()[-28:]
                # Not a DOI, page range or identifier split at a line end ("1099-" / "0984").
                if len(re.sub(r"\W", "", tail)) < 3 or re.search(r"/|\bdoi\b|\d$", tail, re.I):
                    continue
                # The number that continues this column on the next non-empty line.
                following = next((l for l in lines[at + 1:at + 3] if l[max(0, column - 3):].strip()), "")
                start_of_next = following[max(0, column - 3):].lstrip()
                number = re.match(NUMBER_RE + r"(?![\d])", start_of_next)
                if not number:
                    continue
                value = number.group(1)
                # Confirm that the plain text joined them without the minus.
                if not re.search(re.escape(tail) + r" ?" + re.escape(value) + r"(?![\d])", text):
                    continue
                key = (index + 1, value)
                if key not in seen:
                    seen.add(key)
                    found.append({"page": index + 1, "reads": value, "printed": "-" + value,
                                  "context": f"{tail}-{value}", "how": "minus at a line end"})
        for hit in SPLIT_MINUS_RE.finditer(raw):
            value = hit.group(2)
            key = (index + 1, value)
            if key not in seen:
                seen.add(key)
                start = max(0, hit.start() - 24)
                found.append({"page": index + 1, "reads": value, "printed": "-" + value,
                              "context": _squash(raw[start:hit.end()]).strip(), "how": "minus split from its digits"})
    return found


# --------------------------------------------------------------------------
# Package checks (validate_package.py)


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _program(root: Path) -> dict:
    study = _load(root / "study.json")
    program = study.get("program") if isinstance(study, dict) else None
    return program if isinstance(program, dict) else {}


def _result_fields(program: dict) -> dict[str, dict]:
    out = {}
    for node in program.get("nodes") or []:
        if isinstance(node, dict) and node.get("kind") == "result":
            for field in node.get("fields") or []:
                if isinstance(field, dict):
                    out[f"{node.get('id')}.{field.get('id')}"] = field
    return out


def _numbers(value) -> list[float]:
    out: list[float] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, bool) or item is None:
            continue
        elif isinstance(item, (int, float)):
            out.append(float(item))
        elif isinstance(item, str):
            out += [float(t) for t in re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+)", item.replace("−", "-"))]
    return out


def _job_purpose(job: Path):
    request = _load(job / "studio_request.json")
    return request.get("purpose") if isinstance(request, dict) else None


RENDER_HINT = ("run `python agent_pipeline/paper_tables.py <job>/input/paper.pdf --render <job>/work/pages` and "
               "open the page images")


def check_tables(root: Path, job: Path) -> list[str]:
    """Tables the text layer cannot give must be read from the rendered page and recorded."""
    pdf = job / "input/paper.pdf"
    if _job_purpose(job) == "accepted-model-sync" or not pdf.is_file():
        return []
    needs = [t for t in table_inventory(pdf) if t["needs_page_image"]]
    if not needs:
        return []
    metadata = _load(root / "source/paper_metadata.json")
    declared_list = metadata.get("paper_tables") if isinstance(metadata, dict) else None
    declared = {}
    for item in declared_list if isinstance(declared_list, list) else []:
        if isinstance(item, dict) and str(item.get("table", "")).strip():
            declared[str(item["table"]).strip().removeprefix("Table ").strip()] = item
    fields = _result_fields(_program(root))
    undeclared, problems = [], []
    for table in needs:
        where = (f"page {table['caption_page']}" if table["caption_page"]
                 else "mentioned on page " + ", ".join(map(str, table["mentioned_on"])))
        item = declared.get(table["table"])
        if item is None:
            undeclared.append(f"Table {table['table']} ({where})")
            continue
        kind = item.get("kind")
        label = f"paper_tables Table {table['table']}"
        if kind == "result":
            if not re.search(r"(?i)page|image|render", str(item.get("read_from", ""))):
                problems.append(f"{label}: the text layer has no rows for this table; read it from the rendered "
                                "page and set read_from to \"rendered page\"")
            target = fields.get(str(item.get("recorded_in", "")))
            if target is None:
                problems.append(f"{label}: recorded_in must name the result field (<node id>.<field id>) that "
                                "holds the table cell by cell")
            elif not isinstance(target.get("value"), (dict, list)) or len(_numbers(target.get("value"))) < 6:
                problems.append(f"{label}: {item.get('recorded_in')} must hold the table as rows and columns of "
                                "numbers, every cell")
        elif kind in ("other", "not-in-paper"):
            if len(re.findall(r"\w+", str(item.get("what", "")))) < 3:
                problems.append(f"{label}: say in 'what' what the table holds (e.g. 'design of the four "
                                "conditions, no results') or where it is ('in the supplement, not in the PDF')")
        else:
            problems.append(f"{label}: kind must be 'result', 'other' or 'not-in-paper'")
    if undeclared:
        problems.insert(0, "paper tables: the PDF's text layer holds no rows for " + "; ".join(undeclared) +
                        " (drawn as graphics or an image), so the package check cannot read them. " + RENDER_HINT +
                        ", then list every such table in source/paper_metadata.json paper_tables as {\"table\": \"1\", "
                        "\"page\": 10, \"kind\": \"result\" | \"other\" | \"not-in-paper\", \"what\": \"...\", "
                        "\"read_from\": \"rendered page\", \"recorded_in\": \"<result node id>.<field id>\"}, and record "
                        "each result table cell by cell (contract 'Human results to compare against').")
    return problems


CI_KEY_RE = re.compile(r"(?i)(^|[_\-.])(ci|ci\d{2}|conf(idence)?[_\- ]?int(erval)?)($|[_\-.])")
# Effect sizes a bare "ci" key in the same object most likely belongs to (not "value" or a mean).
ESTIMATE_KEYS = ("estimate", "d", "g", "h", "r", "f", "or", "rr", "b", "beta", "mean_diff", "mdiff",
                 "cohens_d", "cohens_h", "cohens_f", "hedges_g", "odds_ratio", "eta2", "eta_squared", "phi", "w", "v")
RATIO_KEYS = ("or", "rr", "odds_ratio", "risk_ratio", "hr")


def _ci_pair(value) -> tuple[float, float] | None:
    if isinstance(value, list) and len(value) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                           for x in value):
        return float(value[0]), float(value[1])
    if isinstance(value, dict):
        lo = next((value[k] for k in ("lower", "lo", "low", "ll", "lcl") if k in value), None)
        hi = next((value[k] for k in ("upper", "hi", "high", "ul", "ucl") if k in value), None)
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return float(lo), float(hi)
    return None


def _outside(estimate: float, lo: float, hi: float, ratio: bool = False) -> bool:
    slack = 0.01 * abs(hi - lo) + 0.006
    if ratio and estimate > 0:
        import math  # a ratio may sit next to a CI on the log scale
        if lo - slack <= math.log(estimate) <= hi + slack:
            return False
    return estimate < lo - slack or estimate > hi + slack


def _impossible(name: str, value, problems: list[str], depth: int = 0) -> None:
    if depth > 8:
        return
    if isinstance(value, dict):
        cis = {k: _ci_pair(v) for k, v in value.items() if CI_KEY_RE.search(str(k)) and _ci_pair(v)}
        for key, (lo, hi) in cis.items():
            if lo > hi + 1e-9:
                problems.append(f"{name}.{key} = [{lo:g}, {hi:g}]: the lower limit is above the upper one")
                continue
            prefix = re.split(CI_KEY_RE, str(key))[0].rstrip("_-.")
            candidates = [k for k, v in value.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                          and (k == prefix if prefix else str(k).lower() in ESTIMATE_KEYS)]
            if len(candidates) == 1 and _outside(float(value[candidates[0]]), lo, hi,
                                                 ratio=str(candidates[0]).lower() in RATIO_KEYS):
                problems.append(f"{name}: {candidates[0]} = {value[candidates[0]]:g} lies outside its CI "
                                f"[{lo:g}, {hi:g}]")
        for key, item in value.items():
            if isinstance(item, (dict, list)) and key not in cis:
                _impossible(f"{name}.{key}", item, problems, depth + 1)
            if re.fullmatch(r"(?i)p(_value)?|pval", str(key)) and isinstance(item, (int, float)) \
                    and not isinstance(item, bool) and not 0 <= item <= 1:
                problems.append(f"{name}.{key} = {item:g} is not a probability")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, (dict, list)):
                _impossible(f"{name}[{index}]", item, problems, depth + 1)


def _signed_recorded(fields: dict[str, dict]) -> list[float]:
    numbers: list[float] = []
    for field in fields.values():
        extensions = {k: v for k, v in (field.get("extensions") or {}).items() if k != "evaluation"}
        numbers += _numbers(field.get("value")) + _numbers(extensions)
    return numbers


def _same(a: float, text: str) -> bool:
    places = len(text.split(".", 1)[1]) if "." in text else 0
    return round(a, places) == round(float(text), places)


def check_signs(root: Path, job: Path) -> list[str]:
    """Recorded human numbers whose printed minus the text layer lost, and impossible values."""
    program = _program(root)
    fields = _result_fields(program)
    if not fields:
        return []
    problems: list[str] = []
    impossible: list[str] = []
    for name, field in fields.items():
        _impossible(name, {"value": field.get("value")}, impossible)
        base = re.sub(r"(?i)[_\-.]?ci\d{0,2}$", "", name)
        pair = _ci_pair(field.get("value"))
        sibling = fields.get(base) if base != name else None
        if pair and sibling is not None and isinstance(sibling.get("value"), (int, float)) \
                and not isinstance(sibling.get("value"), bool) and pair[0] <= pair[1] \
                and _outside(float(sibling["value"]), *pair):
            impossible.append(f"{base} = {sibling['value']:g} lies outside its CI {name} [{pair[0]:g}, {pair[1]:g}]")
        elif pair and pair[0] > pair[1] + 1e-9:
            impossible.append(f"{name} = [{pair[0]:g}, {pair[1]:g}]: the lower limit is above the upper one")
    if impossible:
        problems.append("human results: impossible values (" + "; ".join(dict.fromkeys(impossible[:5]).keys())
                        + "). A lost minus sign usually causes this: check the number on the rendered page "
                        "(" + RENDER_HINT + ") and record it as printed.")
    pdf = job / "input/paper.pdf"
    if not pdf.is_file() or _job_purpose(job) == "accepted-model-sync":
        return problems
    recorded = _signed_recorded(fields)
    checked = {(e.get("locator") or {}).get("page") for e in program.get("evidence") or []
               if isinstance(e, dict) and str(e.get("note", "")).strip().casefold() == "rendered page checked"}
    # The same attestation in source/paper_metadata.json, which a repair round may add.
    metadata = _load(root / "source/paper_metadata.json")
    for item in (metadata.get("sign_checks") if isinstance(metadata, dict) else None) or []:
        if isinstance(item, dict) and str(item.get("note", "")).strip().casefold() == "rendered page checked":
            checked.add(item.get("page"))
    lost = []
    for spot in sign_loss(pdf):
        reads = spot["reads"]
        if float(reads) == 0 or spot["page"] in checked:
            continue
        positive = any(x > 0 and _same(x, reads) for x in recorded)
        negative = any(x < 0 and _same(-x, reads) for x in recorded)
        if positive and not negative:
            lost.append(f"{spot['printed']} on page {spot['page']} (\"{spot['context']}\")")
    if lost:
        problems.append("human results: the PDF prints " + "; ".join(lost[:6]) + " with a minus sign that its plain "
                        "text layer drops (a minus at a line end is read as a hyphen), and the package records the "
                        "number only without its sign. " + RENDER_HINT + " and record the value with the sign the "
                        "page prints. When the page really prints it without a minus, add {\"page\": <page>, "
                        "\"value\": <value>, \"note\": \"rendered page checked\"} to source/paper_metadata.json "
                        "sign_checks (contract 'Text-layer numeric sign loss').")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--render", type=Path, help="write the pages of tables without a text layer (and of "
                        "numbers with a lost minus) as PNG files here")
    args = parser.parse_args()
    if not args.pdf.is_file():
        sys.exit(f"{args.pdf} not found")
    tables = table_inventory(args.pdf)
    signs = sign_loss(args.pdf)
    report: dict = {"tables": tables, "sign_loss": signs}
    if args.render:
        pages = [p for t in tables if needs_page_image(t) for p in pages_to_render(t)] + [s["page"] for s in signs]
        report["rendered"] = render_pages(args.pdf, pages, args.render)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
