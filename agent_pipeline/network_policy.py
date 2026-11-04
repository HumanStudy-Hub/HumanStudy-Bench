#!/usr/bin/env python3
"""Network rules for the build agent's public-materials search.

The build agent may look for public materials of the paper it is building
(OSF, the journal's supplementary material, the authors' repositories,
preprints). This module holds the rules that keep that search narrow:

- ``allowed_hosts(job)``: the hosts the agent may contact. It is a fixed list of
  research hosts plus every host the paper itself prints and every host the
  researcher wrote in the job (the OSF URL field or the Studio note).
- ``looks_like_participant_data(name)``: file names the agent must not download,
  because they hold participant responses rather than materials.
- ``main()``: a Claude Code ``PreToolUse`` hook. It refuses a ``WebFetch`` of a
  host outside the list, and a shell command that names such a host in a network
  call, sends data (POST, upload, form), or pipes downloaded content into a shell.

The hook is a guard against mistakes and against instructions planted in fetched
pages. It is not a sandbox: a command that builds a URL at run time is not seen.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

# Research infrastructure the search needs regardless of what the paper prints.
DEFAULT_HOSTS = (
    # Open Science Framework (projects, API, file storage, preprints)
    "osf.io", "psyarxiv.com", "files.osf.io", "files.us.osf.io",
    # DOI resolution and bibliographic lookup
    "doi.org", "dx.doi.org", "crossref.org", "unpaywall.org", "europepmc.org", "ebi.ac.uk",
    "ncbi.nlm.nih.gov", "openalex.org", "semanticscholar.org", "datacite.org",
    # Code and data repositories
    "github.com", "api.github.com", "raw.githubusercontent.com", "codeload.github.com",
    "objects.githubusercontent.com", "gitlab.com", "zenodo.org", "figshare.com",
    "datadryad.org", "dataverse.harvard.edu", "researchbox.org", "aspredicted.org",
    # Preprint servers
    "arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "researchsquare.com",
    "peercommunityin.org",
    # Experiment hosting with open materials
    "pavlovia.org", "gorilla.sc", "jspsych.org", "cognition.run",
    # Publishers' article and supplementary-material pages
    "royalsocietypublishing.org", "journals.plos.org", "plos.org", "frontiersin.org",
    "springer.com", "nature.com", "wiley.com", "tandfonline.com", "sagepub.com",
    "sciencedirect.com", "elsevier.com", "elifesciences.org", "pnas.org", "apa.org",
    "cambridge.org", "oup.com", "mdpi.com", "science.org", "cell.com", "biomedcentral.com",
    "collabra.org", "online.ucpress.edu", "journalofcognition.org", "psychopen.eu",
)

URL_RE = re.compile(r"https?://([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+)", re.I)
# Bare hosts printed without a scheme, e.g. "osf.io/gf8rc" or "www.example.org/materials".
BARE_RE = re.compile(r"(?<![@\w.-])((?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:io|org|com|edu|net|ac\.uk|eu|de|nl|sc|ch|ca|au))/", re.I)

# Tabular or statistics-package formats that hold responses, never materials.
DATA_SUFFIXES = (".sav", ".zsav", ".por", ".dta", ".rds", ".rdata", ".rda", ".feather",
                 ".parquet", ".sas7bdat", ".xpt", ".omv", ".jasp", ".mat", ".h5", ".sqlite", ".db")
TABLE_SUFFIXES = (".csv", ".tsv", ".xlsx", ".xls", ".ods", ".txt", ".json", ".zip")
DATA_NAME_RE = re.compile(
    r"(?i)(^|[^a-z])(data|dataset|datafile|raw|responses?|participants?|subjects?|demographics?|"
    r"sessions?|trials?[-_ ]?level|results?[-_ ]?export|qualtrics[-_ ]?export|prolific|mturk|export)([^a-z]|$)")

NETWORK_COMMAND_RE = re.compile(
    r"(?i)\b(curl|wget|aria2c|httpie|https?\s|git\s+(clone|fetch|pull|ls-remote|submodule)|svn|"
    r"urllib|urlopen|requests\.|http\.client|httpx|aiohttp|fetch\(|scp|rsync|ssh|ftp|nc|ncat|telnet|"
    r"fetch_public\.py)")
SEND_RE = re.compile(
    r"(?i)(\s-d\s|\s-d['\"]|--data|--form|\s-F\s|--upload-file|\s-T\s|--json\s|"
    r"-X\s*(POST|PUT|PATCH|DELETE)|--request\s+(POST|PUT|PATCH|DELETE)|"
    r"--post-(data|file)|--method[= ](POST|PUT|PATCH|DELETE)|method=['\"](POST|PUT|PATCH|DELETE))")
SECRET_RE = re.compile(r"(?i)\$\{?[A-Z0-9_]*(TOKEN|KEY|SECRET|PASSWORD|AUTH|ANTHROPIC|OPENROUTER)[A-Z0-9_]*\}?|"
                       r"\b(printenv|env)\b\s*(\||$)|/proc/self/environ|os\.environ")
PIPE_TO_SHELL_RE = re.compile(r"(?i)(curl|wget)[^|;&]*\|\s*(sudo\s+)?(ba|z|da)?sh\b|"
                              r"(curl|wget)[^|;&]*\|\s*(python3?|node|perl|ruby)\s*(-\s*)?($|[;&|)])|"
                              r"(ba|z)?sh\s+<\(\s*(curl|wget)|\b(source|eval)\b[^;&|]*(curl|wget)")


def host_of(url: str) -> str:
    try:
        return (urlsplit(url.strip()).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def hosts_in_text(text: str) -> set[str]:
    hosts = {match.group(1).lower().rstrip(".") for match in URL_RE.finditer(text)}
    hosts |= {match.group(1).lower() for match in BARE_RE.finditer(text)}
    return {host[4:] if host.startswith("www.") else host for host in hosts}


def host_allowed(host: str, hosts: Iterable[str]) -> bool:
    host = host.lower().rstrip(".")
    if not host:
        return False
    return any(host == entry or host.endswith("." + entry) for entry in hosts)


def _paper_text(paper: Path) -> str:
    if not paper.is_file() or shutil.which("pdftotext") is None:
        return ""
    try:
        result = subprocess.run(["pdftotext", "-q", str(paper), "-"], capture_output=True,
                                text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return ""
    # Hyphenated line breaks split printed URLs; joining them recovers the host.
    return re.sub(r"-\n", "-", result.stdout or "")


def _researcher_text(job: Path) -> str:
    parts: list[str] = []
    try:
        data = json.loads((job / "job.json").read_text(encoding="utf-8"))
        for key in ("osfUrl", "materialsUrl", "notes", "researcherNote"):
            if isinstance(data.get(key), str):
                parts.append(data[key])
    except (OSError, ValueError):
        pass
    try:
        request = json.loads((job / "studio_request.json").read_text(encoding="utf-8"))
        if isinstance(request.get("message"), str):
            parts.append(request["message"])
        document = request.get("document") if isinstance(request.get("document"), dict) else {}
        for conversation in document.get("conversations") or []:
            if not isinstance(conversation, dict):
                continue
            for message in conversation.get("messages") or [conversation]:
                if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("text"), str):
                    parts.append(message["text"])
    except (OSError, ValueError, AttributeError):
        pass
    return "\n".join(parts)


def allowed_hosts(job: Path) -> list[str]:
    """The fixed research hosts plus every host the paper or the researcher names."""
    hosts = set(DEFAULT_HOSTS)
    hosts |= hosts_in_text(_paper_text(job / "input" / "paper.pdf"))
    hosts |= hosts_in_text(_researcher_text(job))
    return sorted(host for host in hosts if "." in host)


def looks_like_participant_data(name: str) -> bool:
    lowered = name.lower().split("?")[0]
    base = lowered.rsplit("/", 1)[-1]
    if base.endswith(DATA_SUFFIXES):
        return True
    return base.endswith(TABLE_SUFFIXES) and bool(DATA_NAME_RE.search(base))


def check_tool_call(tool: str, tool_input: dict, hosts: Iterable[str]) -> str | None:
    """Return a refusal reason, or None when the call may proceed."""
    hosts = list(hosts)
    if tool == "WebFetch":
        url = str(tool_input.get("url", ""))
        host = host_of(url)
        if not host_allowed(host, hosts):
            return (f"{host or url!r} is not on this build's list of research hosts. Only hosts the "
                    "paper prints, hosts the researcher named, and research repositories may be contacted.")
        return None
    if tool != "Bash":
        return None
    command = str(tool_input.get("command", ""))
    if not NETWORK_COMMAND_RE.search(command):
        return None
    if PIPE_TO_SHELL_RE.search(command):
        return "Downloaded content must never be run: do not pipe a download into a shell or interpreter."
    if SECRET_RE.search(command):
        return "A network command may not read environment variables or credentials."
    if SEND_RE.search(command):
        return "The materials search only reads public pages: sending data (POST, upload, form) is not allowed."
    blocked = sorted({m.group(1).lower() for m in URL_RE.finditer(command)
                      if not host_allowed(m.group(1), hosts)})
    if blocked:
        return (f"{', '.join(blocked)} is not on this build's list of research hosts. Only hosts the paper "
                "prints, hosts the researcher named, and research repositories may be contacted.")
    return None


def main() -> None:
    """Claude Code PreToolUse hook: exit 2 with a reason to refuse a call."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts-file", required=True, type=Path)
    parser.add_argument("--log", type=Path, help="append each refusal here for reviewers")
    args = parser.parse_args()
    try:
        event = json.load(sys.stdin)
        hosts = json.loads(args.hosts_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"network guard could not read its input: {exc}", file=sys.stderr)
        sys.exit(2)
    reason = check_tool_call(str(event.get("tool_name", "")), event.get("tool_input") or {}, hosts)
    if reason:
        if args.log is not None:
            try:
                args.log.parent.mkdir(parents=True, exist_ok=True)
                with args.log.open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps({"tool": event.get("tool_name"), "input": event.get("tool_input"),
                                           "refused": reason}, ensure_ascii=False)[:4000] + "\n")
            except OSError:
                pass
        print(reason, file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
