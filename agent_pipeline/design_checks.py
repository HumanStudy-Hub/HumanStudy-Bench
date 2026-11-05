"""Package checks for what the researcher is asked and what the replay discloses.

Called by ``package_checks.check_package``. Contract: ``agent_pipeline/CLAUDE.md``,
sections "Materials the paper does not print", "Record the design you built" and
"Researcher reply".

1. Licence doubts are a Hub policy item, never one of the researcher's questions:
   the researcher does not hold the rights to a third-party file.
2. A blocked condition names exactly what is missing (``blocked_missing``) and the
   results it blocks (``blocked_results``), besides its ``blocked_reason``.
3. ``task/task.json`` ``presentation`` records how items were paged (one per screen,
   a list page) and whether the paper says so; a layout the paper does not state is
   a departure in ``audit/missing_information.json`` (``field`` "presentation...").
4. When a material is still pending, the search covered the minimum public list:
   the journal's site (and its original site when the journal changed publisher), a
   preprint server, an open-copy index (Unpaywall or Europe PMC) and an OSF search
   by author (users, not titles).
5. A question put to the researcher was first checked against the paper itself
   (``researcher_questions[].sources_checked`` lists ``input/paper.pdf``): its text,
   its arithmetic, and which model (pooled or per condition) a number comes from.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# A licence question names rights or publication *and* a file or material it is about. "Should participants be
# licensed drivers?" or "Can the donations be made public to the other players?" are study questions.
# A licence question asks about rights to a file: "licence", "copyright", "permission to publish", "rights
# holder", "may we publish/share/redistribute ...". "Should the scale run left to right?" or "the order published
# in the appendix" are study questions.
LICENCE_RE = re.compile(
    r"(?i)\b(licen[cs]e[sd]?\b|licen[cs]ing|copyright\w*|permission to (use|publish|share|redistribute|reproduce)|"
    r"rights?[- ]holders?|rights? to (use|publish|share|redistribute|reproduce)|"
    r"(may|can|could) (we|i|you|the (package|hub|study))( also)? (publish|share|redistribute|reproduce)|"
    r"\bbe (published|redistributed|reproduced|shared publicly)\b)")
LICENCE_OBJECT_RE = re.compile(r"(?i)\b(supplement\w*|files?|materials?|item lists?|items|package|survey|"
                               r"questionnaires?|stimul\w*|instruments?|wording|texts?|scales?|cop(y|ies)|appendix|"
                               r"figures?|images?|data ?sets?|exports?|documents?|pdf)\b")
# Journals that moved publisher; their original site may still hold supplements the new one lacks.
LEGACY_SITES = {
    "judgment and decision making": ("sjdm.org", "journal.sjdm.org (the journal's original site, before Cambridge)"),
}
SEARCH_KINDS = {
    "journal": (re.compile(r"(?i)journal|publisher|supplement|cambridge\.org|springer|wiley|sciencedirect|"
                           r"tandfonline|sagepub|frontiersin|plos|apa\.org|ucpress|collabra"),
                "the journal's page for the paper and its supplementary material"),
    "preprint": (re.compile(r"(?i)preprint|psyarxiv|arxiv|biorxiv|medrxiv|ssrn|researchsquare"),
                 "preprint servers by title (PsyArXiv / OSF Preprints, arXiv, SSRN)"),
    "open copy": (re.compile(r"(?i)unpaywall|europepmc|europe pmc|ebi\.ac\.uk"),
                  "an open-copy index for the DOI (Unpaywall or Europe PMC)"),
    "osf author": (re.compile(r"(?i)api\.osf\.io/v2/users|osf\.io/search\?\S*type=user|\bOSF users?\b|"
                              r"\b(users?|authors?)\s+search|search\w*\s+(by|for)\s+(the\s+)?(authors?|users?)\b"),
                   "an OSF search by author (https://api.osf.io/v2/users/?filter[full_name]=<surname>, then "
                   "each match's /nodes/), not only by title"),
}


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _job_purpose(job: Path) -> str | None:
    request = _load(job / "studio_request.json")
    return request.get("purpose") if isinstance(request, dict) else None


def _reply_questions(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[?])\s+|(?<=？)|\n", text)
            if s.strip().rstrip("\"'”’)").endswith(("?", "？"))]


def check_licence_questions(root: Path) -> list[str]:
    reply = root / "studio-reply.md"
    if not reply.is_file():
        return []
    asked = [q for q in _reply_questions(reply.read_text(encoding="utf-8", errors="replace")) if LICENCE_RE.search(q) and LICENCE_OBJECT_RE.search(q)]
    if not asked:
        return []
    return [f"studio-reply.md asks the researcher about a licence ('{asked[0][:120]}'). The researcher does not hold "
            "the rights to the authors' or publisher's files: record the doubt as a Hub policy item "
            "(source/paper_metadata.json licence_policy, with the file and what its licence evidence says), keep "
            "use 'researcher-own-use-only', and state it on the Licence line, not as a question."]


def _conditions(task: Any) -> list[dict]:
    conditions = task.get("conditions") if isinstance(task, dict) else None
    if isinstance(conditions, dict):
        conditions = [dict(v, id=k) if isinstance(v, dict) else {"id": k} for k, v in conditions.items()]
    return [c for c in conditions or [] if isinstance(c, dict)]


def check_blocked(task: Any) -> list[str]:
    arms = {}
    for item in (task.get("blocked_arms") or [] if isinstance(task, dict) else []):
        arms[item.get("id") if isinstance(item, dict) else item] = item if isinstance(item, dict) else {}
    vague = []
    for condition in _conditions(task):
        if condition.get("status") != "blocked" and condition.get("id") not in arms:
            continue
        condition = {**arms.get(condition.get("id"), {}), **{k: v for k, v in condition.items() if v}}
        missing = condition.get("blocked_missing")
        results = condition.get("blocked_results")
        if not (isinstance(missing, list) and missing and all(isinstance(m, str) and m.strip() for m in missing)) \
                or not (isinstance(results, list) and results) \
                or not str(condition.get("blocked_reason") or "").strip():
            vague.append(str(condition.get("id")))
    if not vague:
        return []
    return ["task/task.json: blocked condition(s) " + ", ".join(vague[:6]) + " must state exactly what is missing "
            "and what it blocks: 'blocked_reason' (one sentence), 'blocked_missing' (each missing item by its 'what', "
            "only what no source prints: when the paper prints the scale anchors, the scale is not missing) and "
            "'blocked_results' (the result fields or headline findings that cannot be computed without it)."]


def check_presentation(task: Any, audit: Any) -> list[str]:
    if not isinstance(task, dict):
        return []
    presentation = task.get("presentation")
    if not isinstance(presentation, dict) or not str(presentation.get("layout") or "").strip() \
            or presentation.get("origin") not in ("verbatim", "reported", "derived"):
        return ["task/task.json needs 'presentation': {\"layout\": \"<how items were paged, e.g. one statement per "
                "screen, or all items on one page>\", \"origin\": \"reported\" | \"verbatim\" | \"derived\", "
                "\"evidence\": \"<the passage, when reported>\"}. When the paper or survey file does not state the "
                "layout, origin is 'derived' and the choice is a departure in audit/missing_information.json."]
    if presentation["origin"] != "derived":
        return []
    entries = audit.get("entries") if isinstance(audit, dict) else audit
    fields = [str(e.get("field") or "") for e in entries or [] if isinstance(e, dict)]
    if any(re.match(r"(?i)\s*presentation\b", field) for field in fields):
        return []
    return ["task/task.json presentation.origin is 'derived' (the sources do not say how items were paged), but "
            "audit/missing_information.json has no departure for it: add an entry with field "
            "'presentation.layout' giving the choice made, why, and its likely effect on the findings."]


def check_searches(root: Path) -> list[str]:
    study = _load(root / "study.json")
    program = study.get("program") if isinstance(study, dict) else None
    issues = program.get("issues") if isinstance(program, dict) else None
    if not any(isinstance(i, dict) and i.get("id") == "materials-request" for i in issues or []):
        return []
    metadata = _load(root / "source" / "paper_metadata.json")
    if not isinstance(metadata, dict):
        return []
    searches = [s for s in metadata.get("material_searches") or [] if isinstance(s, dict)]
    blobs = [" ".join(str(s.get(k) or "") for k in ("where", "query", "url")) for s in searches]
    missing = [what for pattern, what in SEARCH_KINDS.values() if not any(pattern.search(b) for b in blobs)]
    names = [metadata.get(key) for key in ("journal", "venue", "container_title", "journal_title")]
    for nested in (metadata.get("paper"), metadata.get("bibliographic"), metadata.get("citation")):
        if isinstance(nested, dict):
            names += [nested.get(key) for key in ("journal", "venue", "container_title", "journal_title")]
    journal = " ".join(str(name) for name in names if isinstance(name, str)).casefold()
    for name, (host, what) in LEGACY_SITES.items():
        if name in journal and not any(host in b for b in blobs):
            missing.append(what)
    if not missing:
        return []
    return ["source/paper_metadata.json material_searches: materials are still pending, but the search skipped "
            + "; ".join(missing) + ". Search them (and record each, including those that find nothing) before "
            "asking the researcher for the materials."]


def check_paper_checked(root: Path, job: Path) -> list[str]:
    try:
        from agent_pipeline.source_use import question_issues
    except ModuleNotFoundError:
        from source_use import question_issues  # type: ignore[no-redef]
    if not (job / "input" / "paper.pdf").is_file():
        return []
    study = _load(root / "study.json")
    program = study.get("program") if isinstance(study, dict) else None
    if not isinstance(program, dict):
        return []
    metadata = _load(root / "source" / "paper_metadata.json")
    records = {r.get("issue"): r for r in (metadata or {}).get("researcher_questions") or [] if isinstance(r, dict)} \
        if isinstance(metadata, dict) else {}
    unchecked = []
    for issue in question_issues(program):
        if issue.get("id") == "materials-request":
            continue
        record = records.get(issue.get("id")) or {}
        paths = set()
        for item in record.get("sources_checked") or []:
            if isinstance(item, dict):
                paths |= {p for p in [item.get("path")] + list(item.get("paths") or []) if isinstance(p, str)}
        if "input/paper.pdf" not in paths:
            unchecked.append(str(issue.get("id")))
    if not unchecked:
        return []
    return ["researcher questions " + ", ".join(f"'{i}'" for i in unchecked[:6]) + " were not checked against the "
            "paper itself: add {\"paths\": [\"input/paper.pdf\"], \"answers\": false, \"found\": \"...\"} to "
            "sources_checked. Many questions are settled by the paper's own text, by arithmetic on its numbers "
            "(units: does 4.4 cents equal the printed 22%?) or by noticing that a text figure comes from a "
            "different model (pooled, not per condition); then record a decision with the quote instead of asking."]


def check(root: Path, job: Path) -> list[str]:
    if _job_purpose(job) == "accepted-model-sync":
        return []
    task = _load(root / "task" / "task.json")
    audit = _load(root / "audit" / "missing_information.json")
    problems: list[str] = []
    problems += check_licence_questions(root)
    problems += check_blocked(task)
    problems += check_presentation(task, audit)
    problems += check_searches(root)
    problems += check_paper_checked(root, job)
    return problems
