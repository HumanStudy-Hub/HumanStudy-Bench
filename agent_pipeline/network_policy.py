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

URL_RE = re.compile(r"https?://[^\s'\"<>()\[\]{}|;`]+", re.I)
# Bare hosts written without a scheme, e.g. "osf.io/gf8rc" or "www.example.org/materials".
BARE_RE = re.compile(r"(?<![@\w.:/-])((?:[a-z0-9-]+\.)+[a-z]{2,24})/", re.I)
# Storage hosts that research repositories redirect file downloads to. They are
# reached only through a redirect from a listed host, never named directly.
REDIRECT_HOSTS = ("oraclecloud.com", "googleapis.com", "amazonaws.com", "githubusercontent.com",
                  "figstatic.com", "blob.core.windows.net", "cloudfront.net")
DEFAULT_HOSTS = DEFAULT_HOSTS + ("github.io", "gitlab.io")

# Statistics-package and serialised formats that hold responses, never materials.
DATA_SUFFIXES = (".sav", ".zsav", ".por", ".dta", ".rds", ".rdata", ".rda", ".feather", ".parquet",
                 ".sas7bdat", ".xpt", ".omv", ".jasp", ".mat", ".h5", ".hdf5", ".sqlite", ".db",
                 ".psydat", ".pkl", ".pickle", ".npy", ".npz", ".edf", ".bdf", ".fif", ".log", ".iqdat")
# Tables: refused unless the name says they are materials (stimuli, items, ...).
TABLE_SUFFIXES = (".csv", ".tsv", ".xlsx", ".xls", ".xlsm", ".ods", ".jsonl", ".dat")
MATERIAL_NAME_RE = re.compile(
    r"(?i)(stimul|item|material|questionnaire|survey|instruction|scale|vignette|scenario|codebook|"
    r"word[-_ ]?list|wordlist|lists?([^a-z]|$)|conditions?([^a-z]|$)|design|counterbalanc|trial[-_ ]?list|"
    r"sequence|schedule|lexicon|norms)")
DATA_NAME_RE = re.compile(
    r"(?i)(^|[^a-z])(data|dataset|datafile|raw|clean(ed)?|processed|responses?|participants?|subjects?|"
    r"results?|export|prolific|mturk|sona|answers?|scores?|P\d+|sub[-_]?\d+|s\d{2,})([^a-z]|$)")
# Opaque downloads whose file name is not in the URL.
DOWNLOAD_URL_RE = re.compile(
    r"(?i)(osf\.io/download/|files\.(us\.)?osf\.io/|osf\.io/[a-z0-9]{5}/download|/ndownloader/|"
    r"zenodo\.org/(api/)?records?/[^ ]*/files/|raw\.githubusercontent\.com/|codeload\.github\.com/|"
    r"/archive/|/releases/download/|dataverse[^ ]*/api/access/|datadryad\.org/[^ ]*/download)")

ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz", ".7z", ".rar")
NETWORK_PROGRAMS = {"curl", "wget", "aria2c", "http", "https", "httpie", "xh", "lynx", "links", "w3m"}
FORBIDDEN_PROGRAMS = {"nslookup", "dig", "host", "nc", "ncat", "netcat", "telnet", "ssh", "scp", "sftp",
                      "rsync", "ftp", "socat", "openssl"}
GIT_NETWORK = {"clone", "fetch", "pull", "ls-remote", "submodule", "remote", "push", "archive"}
INTERPRETERS = {"python", "python3", "node", "perl", "ruby", "php", "deno", "bun"}
SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
NETWORK_API_RE = re.compile(r"(?i)(urllib|urlopen|requests\.|http\.client|httpx|aiohttp|\bfetch\(|socket|"
                            r"XMLHttpRequest|Net::HTTP|open-uri|file_get_contents\(\s*['\"]https?)")
CURL_FORBIDDEN_FLAGS = (
    ("-d", "--data", "--data-raw", "--data-binary", "--data-urlencode", "--data-ascii", "-F", "--form",
     "--form-string", "-T", "--upload-file", "--json", "--post-data", "--post-file", "--body-data",
     "--body-file"), "sending data")
CURL_ROUTE_FLAGS = ("--connect-to", "--resolve", "-x", "--proxy", "--preproxy", "--socks5", "--socks5-hostname",
                    "-K", "--config", "--unix-socket", "--abstract-unix-socket", "--doh-url", "-e", "--execute",
                    "-i", "--input-file")
CURL_OUTPUT_FLAGS = ("-o", "--output", "-O", "--remote-name", "--remote-name-all", "-P", "--directory-prefix",
                     "--output-document", "-J", "--remote-header-name", "--output-dir", "-r", "--recursive",
                     "-m", "--mirror")
SECRET_NAME_RE = re.compile(
    r"(?i)\$\{?[A-Z0-9_]*(TOKEN|SECRET|PASSW|API_KEY|APIKEY|ANTHROPIC|OPENROUTER|CREDENTIAL)[A-Z0-9_]*\}?|"
    r"\b(printenv|declare\s+-x|export\s+-p)\b|(^|[;&|]\s*)env\s*($|[|;&>])|/proc/[^ ]*/environ|"
    r"os\.environ|os\.getenv|process\.env|\bENV\[|getenv\(|\.credentials|\.netrc|\.claude")
SECRET_ENV = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "PIPELINE_PROGRESS_TOKEN",
              "HUMANSTUDY_PIPELINE_TOKEN", "GITHUB_TOKEN")
SEGMENT_SPLIT_RE = re.compile(r"\|\||&&|[|;&\n]")


def host_of(url: str) -> str:
    try:
        return (urlsplit(url.strip()).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def hosts_in_text(text: str) -> set[str]:
    hosts = {host_of(match.group(0)) for match in URL_RE.finditer(text)}
    hosts |= {match.group(1).lower() for match in BARE_RE.finditer(text)}
    hosts = {host[4:] if host.startswith("www.") else host for host in hosts}
    return {host for host in hosts if "." in host and not re.fullmatch(r"[\d.]+", host)}


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
    """True for a file name that marks participant data rather than materials."""
    base = name.lower().split("?")[0].split("#")[0].rstrip("/").rsplit("/", 1)[-1]
    if base.endswith(DATA_SUFFIXES):
        return True
    if base.endswith(TABLE_SUFFIXES):
        return not MATERIAL_NAME_RE.search(base) or bool(re.search(r"(?i)(^|[^a-z])(raw|responses?|export)([^a-z]|$)", base))
    if base.endswith((".txt", ".json")):
        return bool(DATA_NAME_RE.search(base)) and not MATERIAL_NAME_RE.search(base)
    return False


def _secret_values() -> list[str]:
    import os
    import base64

    values = []
    for key in SECRET_ENV:
        value = os.environ.get(key, "")
        if len(value) >= 12:
            values += [value, value[:24], base64.b64encode(value.encode()).decode()[:24]]
    return values


def redact(text: str) -> str:
    for value in _secret_values():
        text = text.replace(value, "[redacted]")
    return text


def _tokens(segment: str) -> list[str]:
    import shlex

    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _program(tokens: list[str]) -> tuple[str, list[str]]:
    index = 0
    while index < len(tokens) and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[index])
                                   or tokens[index] in ("sudo", "time", "nohup", "command", "exec", "xargs",
                                                        "timeout", "env", "(", "{", "!")):
        index += 1
        if index < len(tokens) and tokens[index - 1] == "timeout" and re.match(r"^\d", tokens[index]):
            index += 1
    if index >= len(tokens):
        return "", []
    return tokens[index].rsplit("/", 1)[-1].lower(), tokens[index + 1:]


def _url_host(token: str) -> str:
    token = token.strip("'\"")
    if "://" not in token:
        token = "https://" + token
    return host_of(token)


def _looks_like_target(token: str) -> bool:
    if "://" in token:
        return True
    return bool(re.match(r"^(https?://|ftp://)?([A-Za-z0-9-]+\.)+[A-Za-z]{2,}([:/?#]|$)", token.strip("'\"")))


HOST_REFUSAL = ("{} is not on this build's list of research hosts. Only hosts the paper prints, hosts the "
                "researcher named, and research repositories may be contacted.")
DOWNLOAD_REFUSAL = ("Download files only with `python agent_pipeline/fetch_public.py --job <job> <url>`: it records "
                    "the hash and refuses participant data. curl/wget/Python may only read pages and API listings "
                    "to the terminal.")


SEND_REFUSAL = "The materials search only reads public pages: sending data (POST, upload, form) is not allowed."


def _check_fetcher(program: str, args: list[str], segment: str, hosts: list[str]) -> str | None:
    """curl, wget and similar: read-only, to the terminal, listed hosts, no file downloads."""
    send_flags, _ = CURL_FORBIDDEN_FLAGS
    for position, flag in enumerate(args):
        name = flag.split("=", 1)[0]
        if name in ("-X", "--request", "--method"):
            method = flag.split("=", 1)[1] if "=" in flag else (args[position + 1] if position + 1 < len(args) else "")
            if method.upper() not in ("GET", "HEAD"):
                return SEND_REFUSAL
        elif name in send_flags:
            return SEND_REFUSAL
        elif name in CURL_ROUTE_FLAGS:
            return f"`{name}` is not allowed in the materials search."
        elif name in CURL_OUTPUT_FLAGS:
            return DOWNLOAD_REFUSAL
        elif re.fullmatch(r"-[A-Za-z]{2,}", flag):  # bundled short flags such as -sLo
            if set(flag[1:]) & set("dFT"):
                return SEND_REFUSAL
            if set(flag[1:]) & set("oOPJrmKx"):
                return DOWNLOAD_REFUSAL
    if re.search(r"(^|\s)\d?>{1,2}\s*[^&\s]", segment):
        return DOWNLOAD_REFUSAL
    for arg in args:
        if arg.startswith("-") or not _looks_like_target(arg):
            continue
        host = _url_host(arg)
        if not host_allowed(host, hosts):
            return HOST_REFUSAL.format(host)
        if DOWNLOAD_URL_RE.search(arg) or looks_like_participant_data(arg):
            return DOWNLOAD_REFUSAL
    return None


def _read_script(path: str, cwd: str | None) -> str:
    try:
        target = Path(path) if Path(path).is_absolute() or not cwd else Path(cwd) / path
        if target.is_file() and target.stat().st_size <= 2_000_000:
            return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return ""


def _check_code(code: str, hosts: list[str]) -> str | None:
    """Code that makes network calls may only name listed hosts and never downloads files."""
    if not NETWORK_API_RE.search(code):
        return None
    if SECRET_NAME_RE.search(code):
        return "Code that makes network calls may not read environment variables or credentials."
    for match in URL_RE.finditer(code):
        host = host_of(match.group(0))
        if not host_allowed(host, hosts):
            return HOST_REFUSAL.format(host)
        if DOWNLOAD_URL_RE.search(match.group(0)) or looks_like_participant_data(match.group(0)):
            return DOWNLOAD_REFUSAL
    return None


def check_tool_call(tool: str, tool_input: dict, hosts: Iterable[str], cwd: str | None = None) -> str | None:
    """Return a refusal reason, or None when the call may proceed."""
    hosts = list(hosts)
    serialised = json.dumps(tool_input, ensure_ascii=False)
    if any(value in serialised for value in _secret_values()):
        return "A tool call may not contain credentials."
    if tool == "WebFetch":
        url = str(tool_input.get("url", ""))
        host = host_of(url)
        if not host_allowed(host, hosts):
            return HOST_REFUSAL.format(host or repr(url))
        if looks_like_participant_data(url):
            return "Participant data are never fetched."
        return None
    if tool != "Bash":
        return None
    command = str(tool_input.get("command", ""))
    if SECRET_NAME_RE.search(command):
        return "Commands may not read environment variables, credentials or configuration secrets."
    if re.search(r"(?i)<\(\s*(curl|wget)|\$\(\s*(curl|wget)|`\s*(curl|wget)", command):
        return "Downloaded content must never be run or substituted into a command."
    if re.search(r"(?i)\b(curl|wget)\b[^\n]*\|\s*tee\b", command):
        return DOWNLOAD_REFUSAL
    segments = [segment.strip() for segment in SEGMENT_SPLIT_RE.split(command)]
    previous_network = False
    for segment in segments:
        if not segment:
            continue
        tokens = _tokens(segment)
        program, args = _program(tokens)
        if previous_network and (program in SHELLS or program.startswith(tuple(INTERPRETERS))
                                 and (not args or args[0] in ("-", "-i", "/dev/stdin"))):
            return "Downloaded content must never be run: do not pipe a download into a shell or interpreter."
        if program in SHELLS and "-c" not in args and any(a.endswith((".sh",)) and "external" in a for a in args):
            return "Downloaded content must never be run."
        if program in FORBIDDEN_PROGRAMS:
            return f"`{program}` is not available to the materials search."
        if program in ("pip", "pip3", "npm", "npx", "yarn", "pnpm", "gem", "cargo", "go", "apt", "apt-get", "brew"):
            if any(_looks_like_target(a) or a.startswith(("git+", "http")) for a in args):
                return "Packages may only be installed from their default registry."
            previous_network = False
            continue
        if program == "git" and args and args[0] in GIT_NETWORK:
            for arg in args[1:]:
                if _looks_like_target(arg) or "@" in arg and ":" in arg:
                    host = _url_host(arg.split("@")[-1].replace(":", "/", 1)) if "://" not in arg else _url_host(arg)
                    if not host_allowed(host, hosts):
                        return HOST_REFUSAL.format(host)
            if args[0] in ("push", "remote"):
                return "The materials search never pushes or adds remotes."
            previous_network = True
            continue
        if program in NETWORK_PROGRAMS:
            previous_network = True
            reason = _check_fetcher(program, args, segment, hosts)
            if reason:
                return reason
            continue
        previous_network = False
        if program.startswith(tuple(INTERPRETERS)) and args and not args[0].startswith("-"):
            if args[0].endswith("fetch_public.py"):
                for arg in args[1:]:
                    if _looks_like_target(arg) and not host_allowed(_url_host(arg), hosts):
                        return HOST_REFUSAL.format(_url_host(arg))
                continue
            reason = _check_code(_read_script(args[0], cwd), hosts)
            if reason:
                return reason
    # Inline code (python -c, heredocs): any URL in a command that makes network calls.
    return _check_code(command, hosts)


def main() -> None:
    """Claude Code PreToolUse hook: exit 2 with a reason to refuse a call."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts-file", type=Path)
    parser.add_argument("--hosts", help="comma-separated hosts (preferred: not writable by the agent)")
    parser.add_argument("--log", type=Path, help="append each refusal here for reviewers")
    args = parser.parse_args()
    try:
        event = json.load(sys.stdin)
        hosts = args.hosts.split(",") if args.hosts else json.loads(args.hosts_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, AttributeError) as exc:
        print(f"The network guard could not read its host list ({exc}); network calls are refused. "
              "Report this in studio-reply.md and build from the paper and uploaded materials.", file=sys.stderr)
        sys.exit(2)
    reason = check_tool_call(str(event.get("tool_name", "")), event.get("tool_input") or {}, hosts,
                             event.get("cwd"))
    if reason:
        if args.log is not None:
            try:
                args.log.parent.mkdir(parents=True, exist_ok=True)
                with args.log.open("a", encoding="utf-8") as sink:
                    sink.write(redact(json.dumps({"tool": event.get("tool_name"), "input": event.get("tool_input"),
                                                  "refused": reason}, ensure_ascii=False))[:4000] + "\n")
            except OSError:
                pass
        print(reason, file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
