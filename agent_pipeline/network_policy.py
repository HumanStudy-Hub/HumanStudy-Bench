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
import os
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
    "collabra.org", "online.ucpress.edu", "journalofcognition.org", "psychopen.eu", "sjdm.org",
    # Supplementary-material file hosts of large publishers
    "els-cdn.com", "silverchair-cdn.com", "silverchair.com", "static-content.springer.com",
)

URL_RE = re.compile(r"https?://[^\s'\"<>()\[\]{}|;`]+", re.I)
# Bare hosts written without a scheme, e.g. "osf.io/gf8rc" or "www.example.org/materials".
BARE_RE = re.compile(r"(?<![@\w.:/-])((?:[a-z0-9-]+\.)+[a-z]{2,24})/", re.I)
# Storage hosts that research repositories redirect file downloads to. They are
# reached only through a redirect from a listed host, never named directly.
# jbaron.org now serves the Judgment and Decision Making archive: every
# journal.sjdm.org article and supplement link (which Cambridge's pages still
# print) redirects there permanently.
REDIRECT_HOSTS = ("oraclecloud.com", "googleapis.com", "amazonaws.com", "githubusercontent.com",
                  "figstatic.com", "blob.core.windows.net", "cloudfront.net", "jbaron.org")
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
    r"(?i)([?&]action=download|[?&]download=|osf\.io/download/|files\.(us\.)?osf\.io/|osf\.io/[a-z0-9]+/download|/ndownloader/|"
    r"zenodo\.org/(api/)?records?/[^ ]*/files/|raw\.githubusercontent\.com/|codeload\.github\.com/|"
    r"/archive/|/releases/download/|dataverse[^ ]*/api/access/|datadryad\.org/[^ ]*/download)")

# The authors' analysis code (scripts, notebooks, syntax files) is read for its recodes,
# exclusions and test options; it is never participant data, whatever its name says.
CODE_SUFFIXES = (".r", ".rmd", ".qmd", ".py", ".ipynb", ".do", ".ado", ".sps", ".sas", ".m", ".jl", ".stan", ".js")
CODE_NAME_RE = re.compile(r"(?i)(syntax|script)")
# Words that mark a table of responses whatever else the name says.
RESPONSE_NAME_RE = re.compile(r"(?i)(^|[^a-z])(raw|responses?|participants?|subjects?|export|answers?|scores?|"
                              r"data(set)?)([^a-z]|$)")

ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz", ".7z", ".rar")
FETCHERS = {"curl", "wget", "aria2c", "http", "https", "httpie", "xh", "lynx", "links", "w3m"}
FORBIDDEN_PROGRAMS = {"nslookup", "dig", "host", "nc", "ncat", "netcat", "telnet", "ssh", "scp", "sftp",
                      "rsync", "ftp", "socat", "openssl", "aria2c", "lynx", "links", "w3m", "gh", "svn", "hg"}
INTERPRETERS = ("python", "node", "perl", "ruby", "php", "deno", "bun")
SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish", "source", "."}
INSTALLERS = {"pip", "pip3", "npm", "npx", "yarn", "pnpm", "gem", "cargo", "apt", "apt-get", "brew", "uv"}
NETWORK_API_RE = re.compile(r"(?i)(urllib|urlopen|requests\.|http\.client|httpx|aiohttp|\bfetch\(|socket|"
                            r"XMLHttpRequest|Net::HTTP|open-uri|LWP::|file_get_contents|https?\.get\(|curl_)")
# Per-program flags. Values are (flags that send data, flags that reroute or read
# config, flags that write files, short letters of each kind in bundles like -sLo).
CURL_FLAGS = (
    {"-d", "--data", "--data-raw", "--data-binary", "--data-urlencode", "--data-ascii", "-F", "--form",
     "--form-string", "-T", "--upload-file", "--json", "--url-query"},
    {"--connect-to", "--resolve", "-x", "--proxy", "--preproxy", "--socks4", "--socks4a", "--socks5",
     "--socks5-hostname", "-K", "--config", "--unix-socket", "--abstract-unix-socket", "--doh-url", "-:", "--next"},
    {"-o", "--output", "-O", "--remote-name", "--remote-name-all", "--output-dir", "--create-dirs"},
    ("dFT", "xK", "oO"),
)
WGET_FLAGS = (
    {"--post-data", "--post-file", "--body-data", "--body-file", "--method"},
    {"-e", "--execute", "-i", "--input-file", "--config", "--proxy", "-B", "--base"},
    {"-O", "--output-document", "-P", "--directory-prefix", "-r", "--recursive", "-m", "--mirror", "-N",
     "--timestamping", "-x", "--force-directories"},
    ("", "eiB", "OPrmNx"),
)
SECRET_PATH_RE = re.compile(r"/proc/[^\s]*/(environ|cmdline)|\.netrc\b|\.git-credentials|\.claude/\.credentials|\.claude\.json")
# Git internals and the pipeline's private copy: planting a hook, a URL rewrite
# or an edited guard there would run outside this guard.
PROTECTED_RE = re.compile(r"(^|[/\s'\"])\.git(/|\s|$|['\"])|\bgit\b[^|;&\n]*(\sconfig\b|\s-c\s)|core\.hooksPath|"
                          r"humanstudy-network-|GIT_CONFIG|insteadOf")
SECRET_ENV = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "PIPELINE_PROGRESS_TOKEN",
              "HUMANSTUDY_PIPELINE_TOKEN", "GITHUB_TOKEN")
GUARDED_DIRS = ("agent_pipeline", "contracts")
# The repository the agent works in. The hook may run from a copy of this module
# (so the agent cannot edit it); --repo then names the original.
REPO_ROOT = Path(__file__).resolve().parents[1]


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
    if base.endswith(CODE_SUFFIXES):
        return False
    if base.endswith(TABLE_SUFFIXES):
        return not MATERIAL_NAME_RE.search(base) or bool(re.search(r"(?i)(^|[^a-z])(raw|responses?|export)([^a-z]|$)", base))
    if base.endswith((".txt", ".json")):
        # "spss_syntax.txt" is code; "analysis_data.txt" and "analysis_results.json" stay refused.
        code = base.endswith(".txt") and CODE_NAME_RE.search(base) and not RESPONSE_NAME_RE.search(base)
        return bool(DATA_NAME_RE.search(base)) and not MATERIAL_NAME_RE.search(base) and not code
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


def _strip_heredocs(command: str) -> tuple[str, list[tuple[str, str]]]:
    """Remove heredoc bodies; return the command and (opening line, body) pairs."""
    lines = command.split("\n")
    kept: list[str] = []
    bodies: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        kept.append(line)
        index += 1
        for match in re.finditer(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", line):
            body: list[str] = []
            while index < len(lines) and lines[index].strip() != match.group(2):
                body.append(lines[index])
                index += 1
            index += 1  # the delimiter line
            bodies.append((line, "\n".join(body)))
    return "\n".join(kept), bodies


def _segments(command: str) -> list[str]:
    """Split at unquoted |, ;, &, && and newlines."""
    return [segment for segment, _ in _pipeline_segments(command)]


def _pipeline_segments(command: str) -> list[tuple[str, bool]]:
    """Segments with a flag: True when the segment's output is piped into the next one.

    Splits at unquoted, unescaped |, ;, &, &&, || and newlines, except the & and |
    of a redirection (2>&1, &>file, >|file). Comments (# at the start of a word)
    are dropped, and $'...' quoting honours its backslash escapes, so neither can
    hide the next command from the checks.
    """
    segments, current, quote, escaped, ansi = [], [], "", False, False
    pipes: list[bool] = []
    operator = False  # the last character kept was an unquoted, unescaped < or >
    index = 0
    while index < len(command):
        char = command[index]
        index += 1
        if escaped:
            current.append(char)
            escaped, operator = False, False
            continue
        if char == "\\" and (quote != "'" or ansi):
            escaped = True
            current.append(char)
            continue
        if quote:
            if char == quote:
                quote, ansi = "", False
            current.append(char)
            continue
        if char in "'\"":
            ansi = char == "'" and bool(current) and current[-1] == "$"
            quote, operator = char, False
            current.append(char)
            continue
        if char == "#" and (not current or current[-1] in " \t"):
            while index < len(command) and command[index] != "\n":
                index += 1  # a comment runs to the end of the line
            continue
        if char == "&" and (operator or command[index:index + 1] == ">"):
            current.append(char)  # part of a redirection: 2>&1, >&2, &>file
            operator = False
            continue
        if char == "|" and operator and current[-1] == ">":
            current.append(char)  # >| file
            operator = False
            continue
        if char in "|;&\n":
            pipe = char == "|"
            if pipe and command[index:index + 1] == "|":
                pipe = False  # || runs the next command; it is not a pipe
                index += 1
            elif char == "&" and command[index:index + 1] == "&":
                index += 1
            if "".join(current).strip():
                segments.append("".join(current))
                pipes.append(pipe)
            elif pipe and pipes:
                pipes[-1] = True
            current, operator = [], False
            continue
        current.append(char)
        operator = char in "<>"
    if "".join(current).strip():
        segments.append("".join(current))
        pipes.append(False)
    return [(segment.strip(), pipe) for segment, pipe in zip(segments, pipes)]


REDIRECT_OP_RE = re.compile(r"&>>?|>>|>\||>&|<&|<<<|<<|<>|>|<")


def _redirections(segment: str) -> tuple[str, list[tuple[str, str]]]:
    """Remove unquoted shell redirections; return the rest and (operator, target) pairs.

    ``2>&1``, ``2>/dev/null``, ``> out.json`` and ``&>log`` are redirections, not
    arguments: left in, they were read as URLs ("'2>' is not a research host").
    The operator keeps its fd number (``2>``). Quoted text (``grep '>' x``,
    ``python3 -c "print('>')"``) and process substitution (``<(...)``) are left alone.
    """
    out: list[str] = []
    found: list[tuple[str, str]] = []
    quote, escaped, index, ansi = "", False, 0, False
    while index < len(segment):
        char = segment[index]
        if escaped or (char == "\\" and (quote != "'" or ansi)):
            escaped = not escaped
            out.append(char)
            index += 1
            continue
        if quote or char in "'\"":
            if not quote:
                ansi = char == "'" and bool(out) and out[-1] == "$"
            quote = "" if char == quote else (quote or char)
            out.append(char)
            index += 1
            continue
        match = REDIRECT_OP_RE.match(segment, index) if char in "<>&" else None
        if match is None or segment[match.end():match.end() + 1] == "(" or (char == "&" and not match.group(0).startswith("&>")):
            out.append(char)
            index += 1
            continue
        operator = match.group(0)
        digits = re.search(r"(^|[\s(])(\d+)$", "".join(out))
        if digits:  # 2> : the fd number is part of the operator, not an argument
            del out[len(out) - len(digits.group(2)):]
            operator = digits.group(2) + operator
        index = match.end()
        while index < len(segment) and segment[index] in " \t":
            index += 1
        target: list[str] = []
        tquote = ""
        while index < len(segment):
            tchar = segment[index]
            if tquote:
                if tchar == tquote:
                    tquote = ""
                else:
                    target.append(tchar)
            elif tchar in "'\"":
                tquote = tchar
            elif tchar in " \t|;&<>()":
                break
            else:
                target.append(tchar)
            index += 1
        found.append((operator, "".join(target)))
        out.append(" ")
    return "".join(out), found


def _writes_file(redirections: list[tuple[str, str]]) -> bool:
    """True when stdout ends up in a file, following fd copies in order (2>out 1>&2).

    Stderr to a file and /dev/null are fine; ``1<>out`` and ``01>out`` are stdout.
    """
    table: dict[int, str | None] = {}
    for operator, target in redirections:
        digits, op = re.match(r"(\d*)(.*)", operator).groups()
        if op in ("<<", "<<<"):
            continue
        if op.startswith("&>"):
            table[1] = table[2] = target
            continue
        fd = int(digits) if digits else (0 if op in ("<", "<&", "<>") else 1)
        if op in (">&", "<&") and re.fullmatch(r"\d+", target):
            table[fd] = table.get(int(target))
        elif op in (">&", "<&") and target == "-":
            table[fd] = None
        elif op == "<":
            table[fd] = None  # reading a file is not writing one
        else:
            table[fd] = target  # >, >>, >|, <>, >&file
    return table.get(1) not in (None, "/dev/null")


VAR_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
# Ways to set a variable that _shell_values does not follow; with any of them the value is unknown.
DYNAMIC_VARS_RE = re.compile(r"\b(eval|read|mapfile|readarray|let|getopts|source|declare\s+-n|typeset)\b|"
                             r"printf\s+-v|\+=|\$\{[^}]*[^A-Za-z0-9_}][^}]*\}|\$\{[!#]")


def _shell_values(command: str) -> dict[str, list[str]]:
    """Literal values a command gives its variables: ``u=https://...`` and ``for u in a b; do``."""
    values: dict[str, list[str]] = {}
    for segment in _segments(command):
        tokens = _tokens(_redirections(segment)[0])
        if len(tokens) >= 3 and tokens[0] == "for" and tokens[2] == "in":
            values.setdefault(tokens[1], []).extend(t for t in tokens[3:] if t not in ("do", ";"))
            continue
        for token in tokens:
            match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)", token)
            if not match:
                break
            values.setdefault(match.group(1), []).append(match.group(2))
        if tokens and tokens[0] in ("export", "readonly", "declare", "local"):
            for token in tokens[1:]:
                match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", token)
                if match:
                    values.setdefault(match.group(1), []).append(match.group(2))
    return values


def _expand(arg: str, values: dict[str, list[str]], depth: int = 0) -> list[str] | None:
    """Every literal value of an argument that uses variables, or None when one is not set literally."""
    if "$" not in arg:
        return [arg]
    if depth > 3 or "$(" in arg or "`" in arg:
        return None
    if "${" in arg and not re.search(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", arg):
        return None
    match = VAR_RE.search(arg)
    if match is None or "$" in VAR_RE.sub("", arg):
        return None  # $(...), $1, $@ or another form whose value is not written in the command
    options = values.get(match.group(1) or match.group(2))
    if not options or any("$(" in option or "`" in option for option in options):
        return None
    results: list[str] = []
    for option in options:
        expanded = _expand(arg[:match.start()] + option + arg[match.end():], values, depth + 1)
        if expanded is None:
            return None
        results.extend(expanded)
    return results


def _tokens(segment: str) -> list[str]:
    import shlex

    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


# Wrappers whose options can start another program, change its directory or environment (env -C, env -S,
# sudo -E, exec -a): with options the guard cannot tell what runs, so such a command is refused.
FLAGGED_WRAPPERS = ("sudo", "env", "exec", "nice", "ionice", "stdbuf", "unbuffer", "time", "command", "builtin",
                    "nohup", "watch", "busybox", "eval")
WRAPPER_REFUSAL = ("Run the program directly, not through `{}` with options: the guard cannot tell which program "
                   "that starts or where.")


def _program(tokens: list[str]) -> tuple[str, list[str]]:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in FLAGGED_WRAPPERS and index + 1 < len(tokens) and tokens[index + 1].startswith("-") \
                and not (token == "nice" and re.fullmatch(r"-n|-\d+|--adjustment=\S+", tokens[index + 1])):
            return "\0wrapper:" + token, tokens[index + 1:]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token) or token in (
                "sudo", "time", "nohup", "command", "exec", "xargs", "env", "(", "{", "!", "then", "do", "else",
                "stdbuf", "watch", "busybox", "unbuffer", "eval", "builtin", "ionice"):
            index += 1
        elif token == "timeout" or token == "nice":
            index += 1
            while index < len(tokens) and (tokens[index].startswith("-") or re.match(r"^\d", tokens[index])):
                index += 1
        else:
            break
    if index >= len(tokens):
        return "", []
    return tokens[index].lstrip("({").rsplit("/", 1)[-1].lower(), tokens[index + 1:]


def host_of_any(url: str) -> str:
    """Host of a URL, or "" when parsers could disagree about it (backslashes, spaces)."""
    if "\\" in url or re.search(r"\s", url.strip()):
        return ""
    return host_of(url)


def _urls_in(arg: str) -> list[str]:
    """URLs named by one argument: ``https://...``, ``--url=https://...``, ``host.tld/path``."""
    value = arg.split("=", 1)[1] if arg.startswith("-") and "=" in arg else arg
    value = value.strip("'\"")
    match = re.search(r"[A-Za-z][A-Za-z0-9+.-]*://\S*", value)
    if match:
        return [match.group(0)]
    if not arg.startswith("-") and re.match(r"^([A-Za-z0-9-]+\.)+[A-Za-z]{2,}([:/?#]|$)", value):
        return ["https://" + value]
    return []


HOST_REFUSAL = ("{} is not on this build's list of research hosts. Only hosts the paper prints, hosts the "
                "researcher named, and research repositories may be contacted.")
DOWNLOAD_REFUSAL = ("Download files only with `python agent_pipeline/fetch_public.py --job <job> <url>`: it records "
                    "the hash and refuses participant data. curl and wget may only print pages and API listings "
                    "to the terminal (or a pipe).")
SEND_REFUSAL = "The materials search only reads public pages: sending data (POST, upload, form) is not allowed."
FILE_OUTPUT_REFUSAL = ("Refused only because curl or wget would write to a file (-o, -O, > file, tee); the host itself "
                       "is reachable. Run the same command again printing to the terminal or piping into "
                       "`python3 -c` or `jq`. Download a material file with "
                       "`python agent_pipeline/fetch_public.py --job <job> <url>`.")
VARIABLE_REFUSAL = ("The guard cannot see which URL `{}` holds, so fetch_public.py was not run. Write each URL "
                    "literally in the command (a `for u in <url> <url>; do ... \"$u\"; done` loop over literal "
                    "URLs is fine).")


def _check_url(url: str, hosts: list[str], download_ok: bool = False) -> str | None:
    host = host_of_any(url)
    if not host_allowed(host, hosts):
        return HOST_REFUSAL.format(host or repr(url[:80]))
    if not download_ok and (DOWNLOAD_URL_RE.search(url) or looks_like_participant_data(url)):
        return DOWNLOAD_REFUSAL
    return None


def _wget_to_stdout(args: list[str]) -> bool:
    for position, arg in enumerate(args):
        following = args[position + 1] if position + 1 < len(args) else ""
        if arg in ("--spider", "--output-document=-", "-O-") or (arg in ("-O", "--output-document") and following == "-"):
            return True
        if re.fullmatch(r"-[A-Za-z]*O-?", arg) and (arg.endswith("-") or following == "-"):
            return True
    return False


def _check_fetcher(program: str, args: list[str], redirections: list[tuple[str, str]], hosts: list[str]) -> str | None:
    """curl, wget and similar: read-only, to the terminal, listed hosts, no file downloads."""
    send, route, output, (send_short, route_short, output_short) = CURL_FLAGS if program != "wget" else WGET_FLAGS
    for position, flag in enumerate(args):
        name = flag.split("=", 1)[0]
        value = flag.split("=", 1)[1] if "=" in flag else (args[position + 1] if position + 1 < len(args) else "")
        if name in ("-X", "--request", "--method"):
            if value.upper() not in ("GET", "HEAD"):
                return SEND_REFUSAL
        elif name in send:
            return SEND_REFUSAL
        elif name in route:
            return f"`{name}` is not allowed in the materials search."
        elif name in output:
            if value not in ("/dev/null", "-"):
                return FILE_OUTPUT_REFUSAL
        elif re.fullmatch(r"-[A-Za-z]\S+", flag) and flag[:2] in output and not flag[2:].isalpha():
            if flag[2:] not in ("/dev/null", "-"):  # -o/tmp/x attached value
                return FILE_OUTPUT_REFUSAL
        elif re.fullmatch(r"-[A-Za-z]{2,}", flag):  # bundled short flags such as -sLo
            letters = set(flag[1:])
            if letters & set(send_short):
                return SEND_REFUSAL
            if letters & set(route_short):
                return f"`{flag}` is not allowed in the materials search."
            if letters & set(output_short) and value not in ("/dev/null", "-"):
                return FILE_OUTPUT_REFUSAL
    # Output redirection to a file (stderr redirection and /dev/null are fine).
    if _writes_file(redirections):
        return FILE_OUTPUT_REFUSAL
    if program == "wget" and not _wget_to_stdout(args):
        return FILE_OUTPUT_REFUSAL  # wget saves to a file unless told -O - (or --spider)
    head_only = any(a in ("-I", "--head", "--spider") or re.fullmatch(r"-[A-Za-z]*I[A-Za-z]*", a) for a in args)
    for arg in args:
        for url in _urls_in(arg):
            reason = _check_url(url, hosts, download_ok=head_only)
            if reason:
                return reason
    return None


def _read_script(path: str, cwd: str | None) -> str:
    try:
        target = _resolve(path, cwd)
        if target is not None and target.is_file() and target.stat().st_size <= 2_000_000:
            return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return ""


def _check_code(code: str, hosts: list[str]) -> str | None:
    """Code that makes network calls may only name listed hosts and never downloads files."""
    if not NETWORK_API_RE.search(code):
        return None
    if re.search(r"""(['"]wb['"]|write_bytes|urlretrieve|shutil\.copyfileobj|iter_content)""", code):
        return DOWNLOAD_REFUSAL
    for match in URL_RE.finditer(code):
        reason = _check_url(match.group(0).rstrip(".,"), hosts)
        if reason:
            return reason
    return None


# The working directory after a `cd` the guard cannot follow (a variable, a subshell, `cd ... ||`).
UNKNOWN_CWD = "\0unknown-cwd"
# `cd <literal dir> &&` at the very start of a command: the rest runs there, or not at all.
LEADING_CD_RE = re.compile(r"""^\s*(?:cd|pushd)[ \t]+('[^'\n]*'|"[^"$`\\\n]*"|[^\s;&|<>()$`\\'"#*?\[\]{}!-][^\s;&|<>()$`\\'"*?\[\]{}!]*)[ \t]*&&""")
CD_PROGRAMS = ("cd", "pushd", "popd")
# External, read-only tools that cannot move the shell (no function may shadow them: a function
# definition anywhere in the command makes the directory unknown).
KEEPS_DIR = {"tail", "head", "grep", "sha256sum", "shasum", "md5sum", "ls", "cat", "jq", "echo", "printf", "wc",
             "unzip", "file", "sort", "uniq", "true", "stat", "du"}
# Anything that can move the shell elsewhere without a plain `cd` segment: a cd inside another command,
# functions, eval, traps, sourced files, BASH_ENV, conditions.
DIR_CHANGE_RE = re.compile(r"(?<![\w./-])(cd|pushd|popd|eval|trap|source|function|if|while|until|case|select|"
                           r"coproc|shopt|alias|enable)(?![\w.-])|\w\s*\(\s*\)|BASH_ENV|\bENV=|"
                           r"(^|[;&|({\n])[ \t]*\.[ \t]|\$'|\bexec\b")


def _leading_cd(command: str, cwd: str | None) -> tuple[str | None, int]:
    """The directory a command's leading ``cd <dir> && ...`` chain moves to, and how many segments it spans.

    Only a literal directory at the start of the command, joined by ``&&``, is
    followed: the commands after it run there or not at all. A ``cd`` anywhere
    else (after ``;`` or ``||``, in a subshell, through a variable or CDPATH) is
    not followed, and later relative paths are then treated as unknown.
    """
    if "CDPATH" in command:
        return cwd, 0
    rest, count, current = command, 0, cwd
    while True:
        match = LEADING_CD_RE.match(rest)
        if not match:
            return current, count
        target = match.group(1)
        if target[:1] in "'\"":
            target = target[1:-1]
        if target.startswith("~"):
            # bash expands only an unquoted ~ (and ~+, ~- to $PWD, $OLDPWD): not followed.
            return UNKNOWN_CWD, count + 1
        if not Path(target).is_absolute():
            if current in (None, UNKNOWN_CWD):
                return UNKNOWN_CWD, count + 1
            target = str(Path(current) / target)
        current = os.path.normpath(target)
        rest = rest[match.end():]
        count += 1


def _resolve(path: str, cwd: str | None) -> Path | None:
    """``path`` as the shell sees it from ``cwd``; None when the directory is unknown or the path uses ~."""
    if path.startswith("~"):
        return None  # quoted or not, ~ forms are not resolved here (bash leaves a quoted ~ as a directory name)
    if Path(path).is_absolute():
        return Path(path)
    if cwd == UNKNOWN_CWD:
        return None
    return Path(cwd) / path if cwd else Path(path)


def _is_fetch_helper(path: str, cwd: str | None) -> bool:
    own = REPO_ROOT / "agent_pipeline" / "fetch_public.py"
    try:
        target = _resolve(path, cwd)
        return target is not None and target.resolve() == own.resolve()
    except OSError:
        return False


def _helper_refusal(script: str, cwd: str | None) -> str:
    where = "an unknown directory (after a `cd` the guard cannot follow)" if cwd == UNKNOWN_CWD else (cwd or "here")
    helper = REPO_ROOT / "agent_pipeline" / "fetch_public.py"
    return (f"`{script}` run from {where} is not the pipeline's fetch_public.py, so nothing was downloaded. Run "
            f"`python {helper} --job <job> <url>` (the absolute path works from any directory), or start the "
            f"command with `cd {REPO_ROOT} && python agent_pipeline/fetch_public.py ...`.")


def _script_arg(args: list[str], program: str = "python") -> str:
    """The script an interpreter runs: the first argument that is not an option (or, for python, its value).

    Option values are skipped only for python (``-X opt``, ``-W arg``); perl, ruby and node read the
    same letters differently (ruby ``-X dir`` changes directory), so for them it is the first
    non-option argument, as before.
    """
    if not program.startswith("python"):
        return next((a for a in args if not a.startswith("-")), "")
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-X", "-W", "--check-hash-based-pycs"):
            index += 2  # python -X opt / -W arg
            continue
        if arg == "-m" or (arg.startswith("-m") and len(arg) > 2 and not arg.startswith("--")):
            return ""   # a module, not a script file
        if arg.startswith("-"):
            index += 1
            continue
        return arg
    return ""


def _check_bash(command: str, hosts: list[str], cwd: str | None, depth: int = 0) -> str | None:
    if depth > 3:
        return "Nested shells are not allowed in the materials search."
    if SECRET_PATH_RE.search(command) or re.search(r"(^|[|;&(\s])ps\s+-?[A-Za-z]*e", command):
        return "Commands may not read credentials or another process's environment."
    if PROTECTED_RE.search(command):
        return "Git internals, git configuration and the pipeline's private copy are off limits to the build agent."
    if re.search(r"\b(find\b[^|;&]*-exec(dir)?|xargs)\b[^|;&]*\b(curl|wget)\b", command):
        return "Run curl or wget directly, not through find -exec or xargs."
    if re.search(r"(?i)(<\(|\$\(|`)\s*(curl|wget)\b", command):
        return "Downloaded content must never be run or substituted into a command."
    stripped, heredocs = _strip_heredocs(command)
    session_cwd = cwd
    after_cd, leading = _leading_cd(stripped, cwd)
    rest = stripped
    for _ in range(leading):
        rest = LEADING_CD_RE.sub("", rest, count=1)
    moves = bool(DIR_CHANGE_RE.search(rest))
    for opening, body in heredocs:
        parts = _segments(opening)
        reader = next((part for part in parts if "<<" in part), parts[0] if parts else "")
        program, _ = _program(_tokens(_redirections(reader)[0]))
        if program in SHELLS or program.startswith(INTERPRETERS):
            body_cwd = UNKNOWN_CWD if leading or moves else session_cwd
            reason = _check_code(body, hosts) or (_check_bash(body, hosts, body_cwd) if program in SHELLS else None)
            if reason:
                return reason
    if re.search(r"HUMANSTUDY_ALLOWED_HOSTS_FILE\s*=|export\s+HUMANSTUDY_ALLOWED_HOSTS_FILE", stripped):
        return "The research-host list is fixed for this build and cannot be replaced."
    values = _shell_values(stripped)
    previous_fetch = False
    fetch_pipeline = False  # inside a pipeline that a curl or wget feeds
    # The directory is followed segment by segment: a leading `cd <dir> &&` sets it; from the first segment
    # that could move the shell (a cd the guard does not follow, a function, eval, a program name built from
    # variables) it is unknown. In a loop, any such segment makes the whole command unknown. After a leading
    # cd, only commands that cannot move the shell (the helper, assignments, read-only tools) keep it known.
    loop = bool(re.search(r"(?<![\w.-])(for|while|until|select)(?![\w.-])", rest))
    next_cwd = UNKNOWN_CWD if moves and loop else after_cd
    for position_in_command, (segment, pipes_on) in enumerate(_pipeline_segments(stripped)):
        clean, redirections = _redirections(segment)
        program, args = _program(_tokens(clean))
        if program.startswith("\0wrapper:"):
            return WRAPPER_REFUSAL.format(program.split(":", 1)[1])
        first = (_tokens(clean) or [""])[0]
        if position_in_command >= leading and (
                program in CD_PROGRAMS or DIR_CHANGE_RE.search(segment)
                or (re.search(r"[$`]", first) and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", first))
                or (program and re.search(r"[$`]", program))):
            next_cwd = UNKNOWN_CWD  # from here on the shell may be elsewhere
        cwd = next_cwd  # the directory this segment runs in
        assignment_only = bool(_tokens(clean)) and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t) for t in _tokens(clean))
        keeps_dir = assignment_only or program in KEEPS_DIR or first in ("for", "done") or (
            program.startswith("python") and bool(_script_arg(args, program))
            and _is_fetch_helper(_script_arg(args, program), cwd))
        if leading and position_in_command >= leading and not keeps_dir:
            next_cwd = UNKNOWN_CWD  # after another command the shell may have moved
        if previous_fetch and (program in SHELLS or program.startswith(INTERPRETERS)
                               and (not args or args[0] in ("-", "-i", "/dev/stdin"))):
            return "Downloaded content must never be run: do not pipe a download into a shell or interpreter."
        if fetch_pipeline and (program == "tee" or not program.startswith(INTERPRETERS) and _writes_file(redirections)):
            return FILE_OUTPUT_REFUSAL
        reads_input = any("<" in operator for operator, _ in redirections)
        if program in SHELLS and (args or reads_input) and "-c" not in args and re.search(r"\b(curl|wget)\b", stripped):
            return "Downloaded content must never be run."
        for operator, text in redirections:  # here-strings are code for the program that reads them
            if operator.endswith("<<<") and (program in SHELLS or program.startswith(INTERPRETERS)):
                reason = _check_code(text, hosts) or (_check_bash(text, hosts, cwd, depth + 1)
                                                      if program in SHELLS else None)
                if reason:
                    return reason
        previous_fetch = False
        fetch_pipeline = fetch_pipeline and pipes_on
        closer = _tokens(clean)[:1]
        if closer and (closer[0] in ("}", "done", "fi", "esac") or closer[0].startswith(")")) \
                and _writes_file(redirections) and re.search(r"(?i)\b(curl|wget)\b", stripped):
            return FILE_OUTPUT_REFUSAL  # { curl ...; } > file
        if program in FORBIDDEN_PROGRAMS:
            return f"`{program}` is not available to the materials search."
        if program in SHELLS and "-c" in args:
            position = args.index("-c")
            reason = _check_bash(args[position + 1] if position + 1 < len(args) else "", hosts, cwd, depth + 1)
            if reason:
                return reason
        if program in INSTALLERS:
            if any("://" in a or a.startswith(("git+", "git@")) for a in args):
                return "Packages may only be installed from their default registry."
        elif program == "git":
            rest = list(args)
            while rest and rest[0].startswith("-"):
                rest = rest[2:] if rest[0] in ("-C", "-c", "--git-dir", "--work-tree") else rest[1:]
            sub = rest[0] if rest else ""
            if sub in ("clone", "fetch", "pull", "submodule", "archive", "ls-remote", "push", "remote"):
                return ("Repositories are not cloned: list files with the GitHub API "
                        "(api.github.com/repos/<owner>/<repo>/contents/...) and download each material file with "
                        "fetch_public.py, so participant data are never pulled in.")
            if any("http." in a and "proxy" in a for a in args):
                return "Proxy settings are not allowed in the materials search."
        elif program in FETCHERS:
            previous_fetch = True
            fetch_pipeline = pipes_on
            reason = _check_fetcher(program, args, redirections, hosts)
            if reason:
                return reason
        elif program.startswith(INTERPRETERS):
            script = _script_arg(args, program)
            if "-c" in args or "-e" in args:
                flag = "-c" if "-c" in args else "-e"
                position = args.index(flag)
                reason = _check_code(args[position + 1] if position + 1 < len(args) else "", hosts)
            elif script and Path(script).name == "fetch_public.py" and not (
                    program.startswith("python") and _is_fetch_helper(script, cwd)):
                hosts_named = [host_of_any(url) for arg in args for url in (_urls_in(arg) if "://" in arg else [])]
                reason = next((HOST_REFUSAL.format(h or "an unreadable URL") for h in hosts_named
                               if not host_allowed(h, hosts)), None) or _helper_refusal(script, cwd)
            elif "://" in script:
                reason = f"`{script[:80]}` is not a script file: give the interpreter the script, then the URL."
            elif script and _is_fetch_helper(script, cwd) and any(
                    len(a.split("=")[0]) > 2 and "--scrub".startswith(a.split("=")[0]) for a in args):
                reason = "`fetch_public.py --scrub` is run by the pipeline before a commit, not by the build agent."
            elif script and program.startswith("python") and _is_fetch_helper(script, cwd):
                position = args.index(script)
                rest = args[position + 1:]
                skip = {i + 1 for i, a in enumerate(rest) if a in ("--job", "--name")}
                urls: list[str] = []
                reason = None
                for i, arg in enumerate(rest):
                    if i in skip or arg.startswith("-"):
                        continue
                    if "$" in arg and DYNAMIC_VARS_RE.search(stripped):
                        reason = VARIABLE_REFUSAL.format(arg[:80])
                        break
                    expanded = _expand(arg, values)
                    if expanded is None:
                        reason = VARIABLE_REFUSAL.format(arg[:80])
                        break
                    urls.extend(expanded)
                reason = reason or next((r for r in (_check_url(u, hosts, download_ok=True) for u in urls) if r), None)
            else:
                # Every directory the script may be read from: where the guard believes the shell is, where a
                # leading cd went, and the session's own directory (a cd it cannot follow never hides a script).
                places = [c for c in dict.fromkeys((cwd, after_cd, session_cwd)) if c != UNKNOWN_CWD]
                # Every non-option argument may be the script (perl -mstrict x.pl, python -mrunpy x.py).
                candidates = list(dict.fromkeys([script] + [a for a in args if not a.startswith("-")]))[:10]
                code = "\n".join(_read_script(c, place) for c in candidates if c for place in places)
                reason = _check_code(code, hosts) if code.strip() else None
                # URLs handed to any other script are checked like a fetcher's.
                for arg in args:
                    for url in _urls_in(arg) if "://" in arg else []:
                        reason = reason or _check_url(url, hosts)
            if reason:
                return reason
    # Backstop: a command that names a fetcher may not name an unlisted host anywhere.
    if re.search(r"(?i)\b(curl|wget)\b", stripped):
        for match in URL_RE.finditer(stripped):
            if not host_allowed(host_of_any(match.group(0)), hosts):
                return HOST_REFUSAL.format(host_of_any(match.group(0)) or repr(match.group(0)[:80]))
    return None


def check_tool_call(tool: str, tool_input: dict, hosts: Iterable[str], cwd: str | None = None) -> str | None:
    """Return a refusal reason, or None when the call may proceed."""
    hosts = list(hosts)
    serialised = json.dumps(tool_input, ensure_ascii=False)
    if any(value in serialised for value in _secret_values()):
        return "A tool call may not contain credentials."
    if tool == "WebFetch":
        return _check_url(str(tool_input.get("url", "")), hosts)
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        repo = REPO_ROOT
        try:
            resolved = (Path(cwd or ".") / target).resolve() if not Path(target).is_absolute() else Path(target).resolve()
        except OSError:
            return None
        if any(resolved.is_relative_to(repo / folder) for folder in GUARDED_DIRS):
            return "The pipeline's own code and contracts are read-only for the build agent."
        if ".git" in resolved.parts or "humanstudy-network-" in str(resolved):
            return "Git internals and the pipeline's private copy are off limits to the build agent."
        return None
    if tool != "Bash":
        return None
    return _check_bash(str(tool_input.get("command", "")), hosts, cwd)


def main() -> None:
    """Claude Code PreToolUse hook: exit 2 with a reason to refuse a call."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts-file", type=Path)
    parser.add_argument("--hosts", help="comma-separated hosts (preferred: not writable by the agent)")
    parser.add_argument("--log", type=Path, help="append each refusal here for reviewers")
    parser.add_argument("--repo", type=Path, help="the repository the agent works in")
    args = parser.parse_args()
    global REPO_ROOT
    if args.repo is not None:
        REPO_ROOT = args.repo.resolve()
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
