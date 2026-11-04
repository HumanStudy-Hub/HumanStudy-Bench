"""Package checks for how the build used what it downloaded.

Called by ``validate_package.py`` after ``check_sources.py`` (which checks that each
downloaded file is recorded, hashed and licensed). Contract: ``agent_pipeline/CLAUDE.md``,
"The authors' analysis code", "Survey versions" and "Researcher reply" → Questions.

1. Analysis code. The authors' public analysis code (R, R Markdown, Python, Stata,
   SPSS syntax, notebooks; never data) settles what the paper leaves unsaid: recodes,
   exclusions, the exact test and its options. When a listing the build read names
   such code, it is downloaded. Every downloaded analysis file is recorded in
   ``source/paper_metadata.json`` ``analysis_code``: what was taken from it (or why a
   part does not apply), with its line numbers and the code quoted. Each comment or
   prose line of the code that reports a coding error, a recode, an exclusion or a
   removal must be covered by such a record, so a correction in the authors' own code
   cannot be missed silently.
2. Survey versions. When two or more survey files were downloaded (a Qualtrics export
   and its Word copy, two phases, a pilot), ``survey_versions`` says for each whether it
   matches the paper (scenario count, wording, response codes) and what it was compared
   on; a version that does not match the paper is never used for participant text, and
   one study's text taken from two files says which part came from which.
3. Researcher questions. Before a question goes to the researcher, every downloaded file
   is checked for its answer: ``researcher_questions`` lists, per question issue, each
   downloaded file and what it says. A question a file answers is not asked: the decision
   is taken and recorded in ``decisions_from_sources`` with the file and quote.

Packages that downloaded nothing pass 2 and 3 unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from agent_pipeline.check_sources import file_text, normalise, quote_in_text
    from agent_pipeline.network_policy import CODE_SUFFIXES, looks_like_participant_data
except ModuleNotFoundError:
    from check_sources import file_text, normalise, quote_in_text
    from network_policy import CODE_SUFFIXES, looks_like_participant_data

ANALYSIS_SUFFIXES = tuple(s for s in CODE_SUFFIXES if s != ".js")
# A .py or .m file is analysis code only when it runs statistics (a PsychoPy script is material).
STATS_CALL_RE = re.compile(
    r"(?i)\b(t\.test|aov|anova|lm|glm|lmer|glmer|prop\.test|chisq\.test|wilcox\.test|cor\.test|TOSTER|tsum_TOST|"
    r"esc_\w+|ttest\w*|scipy\.stats|statsmodels|pingouin|regress|xtreg|UNIANOVA|T-TEST|ONEWAY|fitlm|anovan)\s*[(.]?")
SIGNAL_RE = re.compile(
    r"(?i)\b(coding errors?|errors?|mistakes?|wrong(ly)?|incorrect(ly)?|mis-?coded|re-?cod(e|ed|es|ing)|corrected|"
    r"correction|exclud(e|ed|es|ing)|exclusions?|remov(e|ed|es|ing)|drop(ped|s|ping)?|attention[- ]checks?|"
    r"reverse[- ]?(cod\w*|scor\w*)|outliers?)\b")
RECODE_CALL_RE = re.compile(r"\b(recode|fct_recode|mapvalues|RECODE)\s*\(|^\s*RECODE\b")
# Stata and SPSS: "recode part 1=0 if ...", "drop if ...", "keep if ...", "SELECT IF".
STATA_RECODE_RE = re.compile(r"(?i)^\s*(recode\s+\w|drop\s+if\b|keep\s+if\b|select\s+if\b|filter\s+by\b)")
NOT_SIGNAL_RE = re.compile(r"(?i)standard errors?|type (i+|1|2) errors?|error bars?|errors? terms?|"
                           r"remove (all )?(objects|everything|the workspace)|rm\(list|clear (all|the) (memory|workspace)|"
                           r"marginal error|mean squared? error")
CODE_LISTED_RE = re.compile(
    r"(?i:\b(analysis|analyses|analytic)\s+(script|code|syntax|file|folder)s?\b|\b(R|Stata|SPSS|Python)\s+"
    r"(script|code|syntax|markdown)s?\b|\bdata and code\b)"
    r"|(?<![\w/.-])(?!fetch_public|qsf_flow|paper_tables|estimate_run|validate_package)[\w-]+"
    r"\.(R|Rmd|qmd|py|ipynb|do|sps|sas)(?![\w.-])")
NOT_FETCHED_OK_RE = re.compile(r"(?i)refused|archive|participant data|not public|not found|no (analysis )?code|"
                               r"login|private|not this (paper|study)|different study|other study")
NOT_THIS_PAPER_RE = re.compile(r"(?i)not this (paper|study)|different (paper|study)|other (paper|study)|pilot|"
                               r"earlier version|unrelated")
USES = ("carried over", "not applicable")
SURVEY_NAME_RE = re.compile(r"(?i)survey|questionnaire|qualtrics|instrument|materials?|stimul")
STRONG_SURVEY_NAME_RE = re.compile(r"(?i)survey|questionnaire|qualtrics")
NOT_SURVEY_NAME_RE = re.compile(r"(?i)prereg|pre-reg|registration|analysis|readme|codebook|preprint|manuscript|"
                                r"supplement|consent[-_ ]form")
MATCHES = ("yes", "no", "partly")


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _metadata(root: Path) -> dict:
    data = _load(root / "source/paper_metadata.json")
    return data if isinstance(data, dict) else {}


def _program(root: Path) -> dict:
    study = _load(root / "study.json")
    program = study.get("program") if isinstance(study, dict) else None
    return program if isinstance(program, dict) else {}


def _downloads(job: Path) -> list[str]:
    log = _load(job / "input/external/fetch_log.json")
    paths = [e.get("path") for e in log if isinstance(e, dict)] if isinstance(log, list) else []
    return [p for p in dict.fromkeys(paths) if isinstance(p, str) and not looks_like_participant_data(p)]


def _list(value: Any) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _job_purpose(job: Path) -> str | None:
    request = _load(job / "studio_request.json")
    return request.get("purpose") if isinstance(request, dict) else None


# --------------------------------------------------------------------------
# 1. The authors' analysis code


def _code_lines(path: Path) -> list[str]:
    """The file's lines; a notebook's code and markdown cells joined in order."""
    if path.suffix.lower() == ".ipynb":
        data = _load(path)
        cells = data.get("cells") if isinstance(data, dict) else None
        lines: list[str] = []
        for cell in cells if isinstance(cells, list) else []:
            source = cell.get("source") if isinstance(cell, dict) else None
            text = "".join(source) if isinstance(source, list) else str(source or "")
            lines += text.splitlines()
        return lines
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _code_squash(text: str) -> str:
    """Code compared as written: no HTML unescaping or tag stripping (``x < 5 %>%`` is code),
    only typographic quotes and whitespace evened out."""
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", "", text).casefold()


def is_analysis_code(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in ANALYSIS_SUFFIXES:
        return False
    if suffix in (".py", ".m"):
        return bool(STATS_CALL_RE.search("\n".join(_code_lines(path))))
    return True


def signal_lines(path: Path) -> list[tuple[int, str]]:
    """(line number, text) of comment and prose lines that report a recode, a coding error,
    an exclusion or a removal, and of recode calls."""
    suffix = path.suffix.lower()
    out: list[tuple[int, str]] = []
    in_chunk = False
    for number, line in enumerate(_code_lines(path), 1):
        stripped = line.strip()
        if suffix in (".rmd", ".qmd") and stripped.startswith("```"):
            in_chunk = not in_chunk
            continue
        if suffix in (".rmd", ".qmd") and not in_chunk and re.match(r"#{1,6}\s", stripped):
            continue  # a section heading ("## 6. Data Analysis (After Exclusion)")
        prose = suffix in (".rmd", ".qmd") and not in_chunk and stripped and not stripped.startswith(("<!--", "---"))
        comment = bool(re.match(r"(#|//|\*|/\*|%)", stripped)) or bool(re.search(r"\s#\s", line)) and \
            suffix in (".r", ".rmd", ".qmd", ".py", ".jl", ".ipynb")
        if suffix == ".ipynb" and re.match(r"#{1,6}\s", stripped) and not re.match(r"#\s", stripped):
            continue  # a markdown heading of a notebook
        if (prose or comment) and SIGNAL_RE.search(NOT_SIGNAL_RE.sub(" ", stripped)):
            out.append((number, stripped[:160]))
        elif RECODE_CALL_RE.search(line) or suffix in (".do", ".ado", ".sps") and STATA_RECODE_RE.search(line):
            out.append((number, stripped[:160]))
    return out


def _line_range(value: Any) -> tuple[int, int] | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value, value
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return min(value), max(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*(?:[-–]\s*(\d+))?\s*", value)
        if match:
            first, last = int(match.group(1)), int(match.group(2) or match.group(1))
            return min(first, last), max(first, last)
    return None


def check_analysis_code(root: Path, job: Path) -> list[str]:
    if _job_purpose(job) == "accepted-model-sync":
        return []
    metadata = _metadata(root)
    sources = {s.get("id"): s for s in _list(metadata.get("external_sources")) if isinstance(s.get("id"), str)}
    by_path = {s.get("path"): s for s in sources.values() if isinstance(s.get("path"), str)}
    unused = {u.get("path"): u for u in _list(metadata.get("unused_downloads")) if isinstance(u.get("path"), str)}
    records = _list(metadata.get("analysis_code"))
    problems: list[str] = []
    code_files = [p for p in _downloads(job) if (job / p).is_file() and is_analysis_code(job / p)]

    if not code_files:
        searches = " ".join(f"{s.get('query', '')} {s.get('result', '')}" for s in _list(metadata.get("material_searches")))
        listed = CODE_LISTED_RE.search(searches)
        excused = [r for r in records if str(r.get("source", "")).strip().lower() in ("none", "not fetched")
                   and NOT_FETCHED_OK_RE.search(str(r.get("reason", "")))]
        if listed and not excused:
            problems.append(
                f"analysis code: the materials search found the authors' analysis code (\"{listed.group(0)}\" in "
                "source/paper_metadata.json material_searches) but no analysis file was downloaded. Download the "
                "code files (never data) with agent_pipeline/fetch_public.py and read them for recodes, exclusions "
                "and the exact tests (contract 'The authors' analysis code'). Only when the code cannot be fetched "
                "(the guard refused it, it is only inside an archive with participant data) add "
                "{\"source\": \"not fetched\", \"reason\": \"...\"} to analysis_code.")
        return problems

    for path in code_files:
        if path in unused and path not in by_path:
            if not NOT_THIS_PAPER_RE.search(str(unused[path].get("reason", ""))):
                problems.append(f"analysis code: {path} was downloaded but listed as unused. The authors' analysis "
                                "code is read for its recodes, exclusions and test options; record what was taken "
                                "from it in source/paper_metadata.json analysis_code")
            continue
        source = by_path.get(path)
        if source is None:
            continue  # check_sources reports a download that is not recorded at all
        mine = [r for r in records if r.get("source") in (source.get("id"), path)]
        if not mine:
            problems.append(
                f"analysis code: nothing is recorded from {path} (source {source.get('id')}). Add "
                "source/paper_metadata.json analysis_code entries {\"source\": \"<external_sources id>\", \"lines\": "
                "\"79-90\", \"quote\": \"<a line of the code, exactly>\", \"what\": \"<the recode, exclusion or test "
                "option>\", \"use\": \"carried over\", \"where\": \"<where the package applies it>\"} (or use \"not "
                "applicable\" with a reason) for each recode, exclusion and test specification it holds.")
            continue
        lines = _code_lines(job / path)
        covered: list[tuple[int, int]] = []
        for index, record in enumerate(mine):
            label = f"analysis_code entry {index + 1} for {path}"
            span = _line_range(record.get("lines"))
            if span is None or span[0] < 1 or span[1] > len(lines):
                problems.append(f"{label}: lines must be a line number or range in the file (1-{len(lines)})")
                continue
            covered.append(span)
            quote = record.get("quote")
            window = "\n".join(lines[max(0, span[0] - 6):span[1] + 5])
            if not isinstance(quote, str) or not quote.strip():
                problems.append(f"{label}: quote the code line(s) it is about")
            elif _code_squash(quote) not in _code_squash(window):
                problems.append(f"{label}: the quote does not occur at lines {span[0]}-{span[1]} of {path}")
            if len(re.findall(r"\w+", str(record.get("what", "")))) < 3:
                problems.append(f"{label}: say in 'what' what the code does there (a recode, an exclusion, a test "
                                "and its options)")
            use = str(record.get("use", "")).strip().lower()
            if use not in USES:
                problems.append(f"{label}: use must be 'carried over' or 'not applicable'")
            elif use == "carried over" and len(str(record.get("where", "")).split()) < 2 \
                    and not re.search(r"[/.]", str(record.get("where", ""))):
                problems.append(f"{label}: say in 'where' where the package applies it (evaluation, adapter, "
                                "materials)")
            elif use == "not applicable" and len(re.findall(r"\w+", str(record.get("reason", "")))) < 4:
                problems.append(f"{label}: say in 'reason' why it does not apply to the replay")
        missed = [(n, text) for n, text in signal_lines(job / path)
                  if not any(lo - 3 <= n <= hi + 3 for lo, hi in covered)]
        if missed:
            problems.append(
                f"analysis code: {path} reports recodes, coding errors or exclusions that no analysis_code entry "
                "covers: " + "; ".join(f"line {n}: \"{text[:90]}\"" for n, text in missed[:6])
                + (f" and {len(missed) - 6} more" if len(missed) > 6 else "")
                + ". Read each: carry a recode or exclusion over to the evaluation (or adapter) and record it, or "
                "record why it does not apply to the replay.")
    return problems


# --------------------------------------------------------------------------
# 2. Survey versions


def survey_files(paths: list[str]) -> list[str]:
    stems = {Path(p).stem.lower() for p in paths if Path(p).suffix.lower() == ".qsf"}

    def stem(p: str) -> str:
        return re.sub(r"-\d+$", "", Path(p).stem.lower())

    out = []
    for path in paths:
        suffix = Path(path).suffix.lower()
        name = Path(path).name
        if suffix == ".qsf":
            out.append(path)
        elif suffix in (".docx", ".doc", ".pdf", ".odt", ".txt", ".html", ".htm") and (
                stem(path) in stems or STRONG_SURVEY_NAME_RE.search(name) and not NOT_SURVEY_NAME_RE.search(
                    STRONG_SURVEY_NAME_RE.sub("", name))):
            out.append(path)
    return out


def _field_sources(program: dict, sources: dict) -> list[tuple[str, list[str], str]]:
    """(field, cited source paths, study ids) of each participant-material field copied from a source."""
    out = []
    for node in program.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        studies = " ".join(str(s) for s in node.get("studyIds") or [])
        for field in node.get("fields") or []:
            material = ((field or {}).get("extensions") or {}).get("material") if isinstance(field, dict) else None
            if not isinstance(material, dict) or material.get("status") not in ("found-open", "found-private"):
                continue
            paths = [sources[i].get("path") for i in material.get("sourceIds") or [] if i in sources]
            out.append((f"{node.get('id')}.{field.get('id')}", [p for p in paths if isinstance(p, str)], studies))
    return out


def check_survey_versions(root: Path, job: Path) -> list[str]:
    if _job_purpose(job) == "accepted-model-sync":
        return []
    surveys = survey_files(_downloads(job))
    if len(surveys) < 2:
        return []
    metadata = _metadata(root)
    sources = {s.get("id"): s for s in _list(metadata.get("external_sources")) if isinstance(s.get("id"), str)}
    entries: dict[str, list[dict]] = {}
    for entry in _list(metadata.get("survey_versions")):
        path = entry.get("path") or (sources.get(entry.get("source")) or {}).get("path")
        if isinstance(path, str):
            entries.setdefault(path, []).append(entry)
    problems: list[str] = []
    absent = [p for p in surveys if p not in entries]
    if absent:
        problems.append(
            "survey versions: " + str(len(surveys)) + " survey files were downloaded (" + ", ".join(surveys)
            + ") but source/paper_metadata.json survey_versions does not compare " + ", ".join(absent)
            + " with the paper. For each file add {\"path\": \"input/external/<file>\", \"study\": \"<study id>\", "
            "\"matches_paper\": \"yes\" | \"no\" | \"partly\", \"compared\": \"<what was compared: number and order of "
            "scenarios, wording of the items the paper prints, response codes against option text>\", \"used_for\": "
            "\"<which screens it supplies, or 'not used'>\"}. Build each study from the version that matches the "
            "paper; never mix versions silently (contract 'Survey versions').")
    for path, items in entries.items():
        for item in items:
            if item.get("matches_paper") not in MATCHES:
                problems.append(f"survey_versions {path}: matches_paper must be 'yes', 'no' or 'partly'")
            if len(re.findall(r"\w+", str(item.get("compared", "")))) < 6:
                problems.append(f"survey_versions {path}: say in 'compared' what was compared with the paper")
    program = _program(root)
    per_study: dict[str, set[str]] = {}
    for field, paths, studies in _field_sources(program, sources):
        for path in paths:
            if path not in entries:
                continue
            per_study.setdefault(studies, set()).add(path)
            bad = [e for e in entries[path] if e.get("matches_paper") == "no"
                   and (not e.get("study") or str(e.get("study")) in studies.split() or not studies)]
            if bad:
                problems.append(f"survey versions: {field} copies participant text from {path}, which "
                                "survey_versions says does not match the paper; use the version that matches")
    for studies, paths in per_study.items():
        if len(paths) >= 2:
            unexplained = [p for p in paths if not any(len(str(e.get("used_for", "")).split()) >= 3
                                                        for e in entries.get(p, []))]
            if unexplained:
                problems.append(
                    "survey versions: one study (" + (studies or "the package") + ") takes participant text from "
                    + " and ".join(sorted(paths)) + "; say in each file's survey_versions used_for which screens it "
                    "supplies and why they are the same version participants saw")
    return problems


# --------------------------------------------------------------------------
# 3. Researcher questions answered by the sources


def question_issues(program: dict) -> list[dict]:
    """Issues that put a question to the researcher. The per-item "Needs the original ..." issues
    point to the single materials request, which is checked instead."""
    return [i for i in program.get("issues") or [] if isinstance(i, dict)
            and (i.get("severity") in ("decision", "blocking") or str(i.get("title", "")).strip().endswith("?"))
            and not (i.get("id") != "materials-request"
                     and str(i.get("title", "")).strip().lower().startswith("needs the original"))]


def check_questions(root: Path, job: Path) -> list[str]:
    if _job_purpose(job) == "accepted-model-sync":
        return []
    downloads = _downloads(job)
    if not downloads:
        return []
    metadata = _metadata(root)
    program = _program(root)
    records: dict[str, dict] = {}
    for record in _list(metadata.get("researcher_questions")):
        if isinstance(record.get("issue"), str):
            records[record["issue"]] = record
    problems: list[str] = []
    unchecked, answered = [], []
    for issue in question_issues(program):
        record = records.get(issue.get("id"))
        checked = _list(record.get("sources_checked")) if record else []
        covered = set()
        for item in checked:
            for path in ([item.get("path")] if isinstance(item.get("path"), str) else []) + \
                    [p for p in item.get("paths") or [] if isinstance(p, str)]:
                covered.add(path)
            if len(re.findall(r"\w+", str(item.get("found", "")))) < 3:
                unchecked.append(f"'{issue.get('id')}' (say in 'found' what each file says on this point)")
            if item.get("answers") is True:
                answered.append(f"'{issue.get('id')}' ({item.get('path') or ', '.join(item.get('paths') or [])})")
        missing = [p for p in downloads if p not in covered]
        if missing:
            unchecked.append(f"'{issue.get('id')}'" + ("" if record else " (no record)")
                             + (f" (not checked: {', '.join(missing[:4])}{' and more' if len(missing) > 4 else ''})"
                                if record else ""))
    if unchecked:
        problems.append(
            "researcher questions: before asking the researcher, check every downloaded file for the answer and "
            "record it in source/paper_metadata.json researcher_questions: {\"issue\": \"<issue id>\", "
            "\"sources_checked\": [{\"paths\": [\"input/external/<file>\", ...], \"answers\": false, \"found\": "
            "\"<what the file says on this point>\"}]}. Missing for " + ", ".join(unchecked[:6])
            + ". A question a downloaded file answers is not asked (contract 'Researcher reply').")
    if answered:
        problems.append(
            "researcher questions: a downloaded file answers " + ", ".join(answered[:4]) + ". Take the decision from "
            "the source instead of asking: record it in source/paper_metadata.json decisions_from_sources "
            "({\"what\", \"path\", \"locator\", \"quote\"}), apply it, remove the question issue, and list it in "
            "studio-reply.md as a decision taken from the sources.")
    for index, decision in enumerate(_list(metadata.get("decisions_from_sources"))):
        path, quote = decision.get("path"), decision.get("quote")
        label = f"decisions_from_sources[{index}]"
        if not isinstance(path, str) or path not in downloads and path != "input/paper.pdf":
            problems.append(f"{label}: path must be a downloaded file (input/external/...) or input/paper.pdf")
            continue
        if not isinstance(quote, str) or not quote.strip() or len(re.findall(r"\w+", str(decision.get("what", "")))) < 3:
            problems.append(f"{label}: give 'what' was decided and the 'quote' that settles it")
            continue
        text = file_text(job / path)
        if text is not None and not quote_in_text(quote, text):
            problems.append(f"{label}: the quote does not occur in {path}")
    return problems


def check(root: Path, job: Path) -> list[str]:
    problems: list[str] = []
    for function in (check_analysis_code, check_survey_versions, check_questions):
        try:
            problems += function(root, job)
        except Exception as exc:  # noqa: BLE001 - a malformed record is reported, not a crash
            problems.append(f"{function.__name__} could not run ({type(exc).__name__}: {exc}); fix the malformed "
                            "record in source/paper_metadata.json")
    return problems
