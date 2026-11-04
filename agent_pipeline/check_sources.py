"""Package check for public materials found outside the paper.

Called by ``validate_package.py``. It checks the records the build agent writes
when it searches for materials the paper does not print (contract:
``agent_pipeline/CLAUDE.md``, "Materials the paper does not print"):

- every ``source/paper_metadata.json`` → ``external_sources`` entry names its URL,
  how it is tied to this paper, its version, licence and use, says it is not
  participant data, and points to a downloaded file whose SHA-256 matches;
- a source without an open licence is marked for the researcher's own use only,
  and so is the package (``redistribution``);
- every program evidence record that cites a URL or a downloaded file cites a
  recorded source, and a quote from a downloaded file really occurs in that file;
- every program field classified in ``extensions.material``:
  ``schema-enough`` holds structure filled from a cited description (``derived``
  with its derivation, or ``implementation``), never wording that no cited quote
  prints; ``needs-original`` is either copied from a recorded source (``found-open``
  / ``found-private``) or left ``pending`` (state ``missing``, no value, a search
  record, and named in the single ``materials-request`` issue);
- a ``missing`` material field is classified, so a gap cannot skip the search;
- no participant-data file was downloaded.

Packages without external sources or material classifications pass unchanged,
except that missing material fields must be classified (not checked for an
accepted-model-sync job, which copies an accepted program exactly).
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Iterable

try:
    from agent_pipeline.network_policy import looks_like_participant_data
except ModuleNotFoundError:
    from network_policy import looks_like_participant_data

OPEN_LICENCE_RE = re.compile(
    r"(?i)\b(cc[- ]?by|cc0|cc[- ]?zero|creative commons|public domain|mit|apache|bsd|gpl|lgpl|mpl|"
    r"isc|unlicense|odc[- ]?by|odbl|pddl|open government licence)\b")
NOT_OPEN_RE = re.compile(r"(?i)\b(nc|nd|non[- ]?commercial|no[- ]?deriv|all rights reserved|none|unknown|not stated|no licen[cs]e)\b")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
USES = ("open", "researcher-own-use-only")
CLASSIFICATIONS = ("schema-enough", "needs-original")
NEEDS_ORIGINAL_STATUSES = ("found-open", "found-private", "pending")
REQUEST_ISSUE = "materials-request"
PROSE_WORDS = 8
TEXT_SUFFIXES = (".txt", ".md", ".html", ".htm", ".xml", ".csv", ".tsv", ".rmd", ".tex", ".js", ".py", ".r")
JSON_SUFFIXES = (".json", ".qsf")


def licence_is_open(licence: str) -> bool:
    return bool(OPEN_LICENCE_RE.search(licence)) and not NOT_OPEN_RE.search(licence)


def normalise(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (text.replace("’", "'").replace("‘", "'").replace("“", '"')
            .replace("”", '"').replace("–", "-").replace("—", "-").replace(" ", " "))
    return re.sub(r"\s+", " ", text).strip()


def _json_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)


def file_text(path: Path) -> str | None:
    """Plain text of a downloaded file, or None when it cannot be read here."""
    suffix = path.suffix.lower()
    try:
        if suffix in JSON_SUFFIXES:
            return normalise("\n".join(_json_strings(json.loads(path.read_text(encoding="utf-8", errors="replace")))))
        if suffix in TEXT_SUFFIXES:
            return normalise(path.read_text(encoding="utf-8", errors="replace"))
        if suffix in (".docx", ".pptx", ".odt"):
            with zipfile.ZipFile(path) as archive:
                parts = [archive.read(name).decode("utf-8", "replace") for name in archive.namelist()
                         if name.endswith(".xml") and ("word/" in name or "ppt/slides/" in name or name == "content.xml")]
            joined = "\n".join(parts)
            joined = re.sub(r"</w:p>|</a:p>|</text:p>", "\n", joined)
            return normalise(joined)
        if suffix == ".pdf" and shutil.which("pdftotext"):
            result = subprocess.run(["pdftotext", "-q", str(path), "-"], capture_output=True, text=True, timeout=120)
            return normalise(re.sub(r"-\n(?=[a-z])", "", result.stdout)) if result.returncode == 0 else None
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.SubprocessError):
        return None
    return None


def quote_in_text(quote: str, text: str) -> bool:
    """Every part of the quote (split at an ellipsis) occurs in the text, in order."""
    segments = [normalise(part) for part in re.split(r"\.\.\.|…|\[\.\.\.\]", quote)]
    segments = [segment for segment in segments if segment]
    if not segments:
        return True
    position = 0
    for segment in segments:
        found = text.find(segment, position)
        if found < 0:
            found = text.casefold().find(segment.casefold(), position)
        if found < 0:
            return False
        position = found + len(segment)
    return True


def _strings(value: Any) -> Iterable[str]:
    yield from _json_strings(value)


def _is_prose(text: str) -> bool:
    return len(re.findall(r"[A-Za-zÀ-ɏ一-鿿]+", text)) >= PROSE_WORDS


def _job_purpose(job: Path) -> str | None:
    try:
        request = json.loads((job / "studio_request.json").read_text(encoding="utf-8"))
        return request.get("purpose") if isinstance(request, dict) else None
    except (OSError, ValueError):
        return None


def check_package_sources(root: Path, job: Path | None = None) -> list[str]:
    """Return every problem found; an empty list means the package passes."""
    job = job if job is not None else root.parent.parent
    problems: list[str] = []
    try:
        metadata = json.loads((root / "source/paper_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    sources = metadata.get("external_sources", [])
    if not isinstance(sources, list):
        problems.append("source/paper_metadata.json external_sources must be a list")
        sources = []
    log_path = job / "input/external/fetch_log.json"
    try:
        fetch_log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else []
    except ValueError:
        fetch_log = []
    logged = {item.get("path"): item for item in fetch_log if isinstance(item, dict)}

    by_id: dict[str, dict] = {}
    by_url: dict[str, dict] = {}
    by_path: dict[str, dict] = {}
    texts: dict[str, str | None] = {}
    private = False
    for index, source in enumerate(sources):
        label = f"external_sources[{index}]"
        if not isinstance(source, dict):
            problems.append(f"{label} must be an object with url, path, sha256, version, tie, licence and use")
            continue
        sid = source.get("id")
        if not isinstance(sid, str) or not sid.strip():
            problems.append(f"{label} needs an id")
        else:
            by_id[sid] = source
        url = source.get("url")
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            problems.append(f"{label} needs the http(s) url it was downloaded from")
        for key in ("version", "tie", "licence"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                problems.append(f"{label} needs {key}" + (
                    " (how this source is tied to this paper: linked from the paper or its project, or a DOI/title/author match)"
                    if key == "tie" else ""))
        if not isinstance(source.get("retrieved"), str) or not DATE_RE.match(source["retrieved"]):
            problems.append(f"{label} needs retrieved as YYYY-MM-DD")
        use = source.get("use")
        if use not in USES:
            problems.append(f"{label} use must be 'open' or 'researcher-own-use-only'")
        elif use == "open" and not licence_is_open(str(source.get("licence", ""))):
            problems.append(f"{label} states no open licence ({source.get('licence')!r}); its use must be 'researcher-own-use-only'")
        if use == "researcher-own-use-only":
            private = True
        if source.get("participant_data") is not False:
            problems.append(f"{label} must state participant_data: false; participant data are never downloaded")
        path = source.get("path")
        sha = source.get("sha256")
        if not isinstance(path, str) or not path.startswith("input/external/") or ".." in Path(path).parts:
            problems.append(f"{label} needs path: the downloaded file under input/external/ (use agent_pipeline/fetch_public.py)")
        elif not isinstance(sha, str) or not SHA_RE.match(sha):
            problems.append(f"{label} needs the file's sha256")
        else:
            file = job / path
            if not file.is_file():
                problems.append(f"{label} cites {path}, which was not downloaded")
            elif hashlib.sha256(file.read_bytes()).hexdigest() != sha:
                problems.append(f"{label} sha256 does not match {path}")
            else:
                entry = logged.get(path)
                if entry is None:
                    problems.append(f"{label} cites {path}, which agent_pipeline/fetch_public.py did not download")
                elif url not in (entry.get("url"), entry.get("final_url")):
                    problems.append(f"{label} url does not match the URL {path} was downloaded from")
                by_path[path] = source
                texts[path] = file_text(file)
            if looks_like_participant_data(path):
                problems.append(f"{label} {path} looks like participant data, which is never downloaded")
        for key in ("url", "final_url"):
            if isinstance(source.get(key), str):
                by_url[source[key]] = source
        if isinstance(path, str) and isinstance(logged.get(path), dict) and logged[path].get("final_url"):
            by_url.setdefault(logged[path]["final_url"], source)
    if private and metadata.get("redistribution") != "researcher-own-use-only":
        problems.append("source/paper_metadata.json redistribution must be 'researcher-own-use-only' "
                        "because a source without an open licence is used")
    if sources and not metadata.get("material_searches"):
        problems.append("source/paper_metadata.json material_searches must record where materials were searched")

    for entry in fetch_log:
        if isinstance(entry, dict) and looks_like_participant_data(str(entry.get("path", ""))):
            problems.append(f"{entry.get('path')} looks like participant data, which is never downloaded")

    try:
        study = json.loads((root / "study.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return problems
    program = study.get("program") if isinstance(study, dict) else None
    if not isinstance(program, dict):
        return problems
    evidence = {e.get("id"): e for e in program.get("evidence", []) if isinstance(e, dict)}
    paper_urls = {str(metadata.get("doi", "")).strip()} - {""}

    def source_of(record: dict) -> dict | None:
        if record.get("path") in by_path:
            return by_path[record["path"]]
        if record.get("url") in by_url:
            return by_url[record["url"]]
        return None

    for eid, record in evidence.items():
        cites_external = bool(record.get("url")) or str(record.get("path", "")).startswith("input/")
        if not cites_external or record.get("url") in paper_urls:
            continue
        source = source_of(record)
        if source is None:
            problems.append(f"evidence {eid} cites {record.get('url') or record.get('path')}, "
                            "which is not a downloaded source in external_sources")
            continue
        quote = record.get("quote")
        text = texts.get(source.get("path"))
        if isinstance(quote, str) and quote.strip() and text is not None and not quote_in_text(quote, text):
            problems.append(f"evidence {eid}: the quote does not occur in {source.get('path')}")

    purpose = _job_purpose(job)
    issues = [i for i in program.get("issues", []) if isinstance(i, dict)]
    request_issue = next((i for i in issues if i.get("id") == REQUEST_ISSUE), None)
    pending: list[str] = []
    for node in program.get("nodes", []):
        if not isinstance(node, dict):
            continue
        for field in node.get("fields", []):
            if not isinstance(field, dict):
                continue
            where = f"{node.get('id')}.{field.get('id')}"
            material = (field.get("extensions") or {}).get("material")
            if material is None:
                if (node.get("kind") == "material" and field.get("state") == "missing"
                        and purpose != "accepted-model-sync"):
                    problems.append(f"{where} is a missing material: classify it in extensions.material "
                                    "(schema-enough or needs-original) after searching public sources")
                continue
            if not isinstance(material, dict):
                problems.append(f"{where} extensions.material must be an object")
                continue
            kind = material.get("classification")
            if kind not in CLASSIFICATIONS:
                problems.append(f"{where} extensions.material.classification must be schema-enough or needs-original")
                continue
            if not isinstance(material.get("what"), str) or not material["what"].strip():
                problems.append(f"{where} extensions.material.what must name the material for the researcher")
            cited = [evidence[e] for e in field.get("evidenceIds", []) if e in evidence]
            if kind == "schema-enough":
                if field.get("origin") not in ("derived", "implementation"):
                    problems.append(f"{where} is schema-enough, so its origin is derived or implementation, "
                                    "never verbatim, reported or unknown")
                if field.get("state") == "missing":
                    problems.append(f"{where} is schema-enough, so it is filled and does not stay missing")
                if not cited:
                    problems.append(f"{where} is schema-enough and must cite the passage it is rebuilt from")
                quotes = [normalise(str(e.get("quote", ""))) for e in cited]
                for text in _strings(field.get("value")):
                    if _is_prose(text) and not any(normalise(text) in quote for quote in quotes):
                        problems.append(f"{where} is schema-enough but holds wording no cited quote prints "
                                        f"({text[:60]!r}); text participants read is needs-original")
                        break
                continue
            status = material.get("status")
            if status not in NEEDS_ORIGINAL_STATUSES:
                problems.append(f"{where} needs-original status must be found-open, found-private or pending")
                continue
            if status == "pending":
                pending.append(str(material.get("what", "")))
                if field.get("state") != "missing" or field.get("origin") != "unknown":
                    problems.append(f"{where} is pending, so it stays state missing with origin unknown")
                if any(_is_prose(text) for text in _strings(field.get("value"))):
                    problems.append(f"{where} is pending but holds text; needs-original wording is never invented")
                if not isinstance(material.get("search"), list) or not material["search"]:
                    problems.append(f"{where} is pending and must list where it was searched (extensions.material.search)")
                continue
            ids = material.get("sourceIds")
            if not isinstance(ids, list) or not ids or any(i not in by_id for i in ids):
                problems.append(f"{where} is {status} and must list sourceIds of external_sources entries")
                continue
            expected = "open" if status == "found-open" else "researcher-own-use-only"
            if any(by_id[i].get("use") != expected for i in ids):
                problems.append(f"{where} is {status}, so its sources must have use {expected!r}")
            if not any(source_of(e) in [by_id[i] for i in ids] for e in cited):
                problems.append(f"{where} is {status} but cites no evidence record from its sources")
            # The value itself, not only the cited quote, must be text the source prints.
            source_texts = [texts.get(by_id[i].get("path")) for i in ids]
            quotes = [normalise(str(e.get("quote", ""))) for e in cited]
            for text in _strings(field.get("value")):
                if not _is_prose(text):
                    continue
                readable = [t for t in source_texts if t is not None]
                in_source = any(quote_in_text(text, t) for t in readable)
                in_quote = any(normalise(text) in quote for quote in quotes)
                if not (in_source or (not readable and in_quote)):
                    problems.append(f"{where} is {status} but its text does not occur in its source "
                                    f"({text[:60]!r})")
                    break
            if field.get("state") == "missing":
                problems.append(f"{where} was found, so it is not missing")
    if pending:
        if request_issue is None:
            problems.append(f"materials still needed ({'; '.join(pending)}) must be listed in one issue "
                            f"with id '{REQUEST_ISSUE}'")
        else:
            body = " ".join(str(request_issue.get(k, "")) for k in ("title", "reason", "suggestedAction"))
            for what in pending:
                if what and normalise(what).casefold() not in normalise(body).casefold():
                    problems.append(f"the '{REQUEST_ISSUE}' issue must name {what!r}")
    return problems
