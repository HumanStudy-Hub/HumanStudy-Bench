#!/usr/bin/env python3
"""Download one public material file for a build and record where it came from.

    python agent_pipeline/fetch_public.py --job <job-dir> <url> [--name file.ext]

The file is saved under ``<job>/input/external/`` (outside the package, so it is
not shipped) and a record is appended to ``<job>/input/external/fetch_log.json``:
the URL asked for, the final URL after redirects, the saved path, SHA-256, size,
content type and date. ``source/paper_metadata.json`` → ``external_sources``
entries cite that path and hash, and the package check compares them with the
file on disk, so a recorded source is always a file that was really downloaded.

Refused: a host outside this build's research-host list, a file whose name or
format marks it as participant data, and a file over the size limit.

    python agent_pipeline/fetch_public.py --job <job-dir> --licence <OSF URL>

reads a licence without downloading anything: for an OSF project, component,
registration or file it asks the OSF API for the file's name and SHA-256 and the
licence of its project and of each parent project, prints them and appends them
to ``<job>/input/external/licence_log.json``.

``final_url`` is recorded without the query of a signed storage link
(``X-Amz-Signature=...``): those links expire within minutes and are
credentials while they last, so they are never written to the job folder.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import socket
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    from agent_pipeline.network_policy import (
        ARCHIVE_SUFFIXES, REDIRECT_HOSTS, allowed_hosts, host_allowed, host_of, looks_like_participant_data,
    )
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from network_policy import (
        ARCHIVE_SUFFIXES, REDIRECT_HOSTS, allowed_hosts, host_allowed, host_of, looks_like_participant_data,
    )

MAX_FILE = 40 * 1024 * 1024
MAX_TOTAL = 100 * 1024 * 1024
CHUNK = 1024 * 1024
USER_AGENT = "HumanStudy-Hub materials search (+https://github.com/HumanStudy-Hub/HumanStudy-Bench)"


def _hosts(job: Path) -> list[str]:
    path = os.environ.get("HUMANSTUDY_ALLOWED_HOSTS_FILE")
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return allowed_hosts(job)


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", unquote(name)).strip("._") or "download"
    return name[:120]


def _filename(response, url: str) -> str:
    disposition = response.headers.get("Content-Disposition") or ""
    match = re.search(r"filename\*=(?:UTF-8'')?\"?([^\";]+)", disposition, re.I) or \
        re.search(r"filename=\"?([^\";]+)", disposition, re.I)
    if match:
        return match.group(1)
    return urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1] or "download"


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only over https to a listed host or a repository's file storage."""

    def __init__(self, hosts: list[str]):
        super().__init__()
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("a redirect left https")
        if not host_allowed(host_of(newurl), list(self.hosts) + list(REDIRECT_HOSTS)):
            raise ValueError(f"a redirect led to {host_of(newurl)}, which is not a research host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _archive_has_data(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if looks_like_participant_data(name):
                    return name
    except zipfile.BadZipFile:
        return None
    return None


def _notebook_shows_data(path: Path) -> bool:
    """A notebook whose stored outputs print tables (``df.head()``, a printed data frame)
    may show participant responses; its code is fetched only without outputs."""
    try:
        cells = json.loads(path.read_text(encoding="utf-8", errors="replace")).get("cells") or []
    except (OSError, ValueError, AttributeError):
        return False
    for cell in cells if isinstance(cells, list) else []:
        for output in (cell.get("outputs") or []) if isinstance(cell, dict) else []:
            if not isinstance(output, dict):
                continue
            data = output.get("data") if isinstance(output.get("data"), dict) else {}
            html = "".join(data.get("text/html") or []) if isinstance(data.get("text/html"), list) else str(data.get("text/html") or "")
            texts = [output.get("text"), data.get("text/plain")]
            plain = "".join("".join(t) if isinstance(t, list) else str(t or "") for t in texts)
            if "<table" in html.lower() or len([l for l in plain.splitlines() if len(l.split()) >= 3]) >= 5:
                return True
    return False


# A storage link is a short-lived credential when its query carries a provider's signature: AWS S3 and
# S3-compatible stores (OSF's Oracle storage), Google Cloud Storage, CloudFront, Azure SAS. An OSF
# view-only key is a private-access credential too.
SIGNED_URL_RE = re.compile(r"(?i)https?://[^\s\"'<>?)\]\\]+\?[^\s\"'<>)\]\\]*")
SCRUB_SUFFIXES = (".json", ".jsonl", ".md", ".txt", ".log", ".html", ".yml", ".yaml")


def _is_signed(query: str) -> bool:
    names = {name.lower() for name, _ in
             (part.split("=", 1) if "=" in part else (part, "") for part in re.split(r"&(?:amp;)?|;", query))}
    return bool(
        names & {"x-amz-signature", "x-amz-credential", "x-amz-security-token", "x-goog-signature",
                 "x-goog-credential", "view_only"}
        or ("signature" in names and names & {"expires", "key-pair-id", "googleaccessid", "awsaccesskeyid", "policy"})
        or ("sig" in names and names & {"sv", "se", "sp", "sr"}))


def public_url(url: str) -> str:
    """``url`` without the query of a signed storage link; any other URL unchanged."""
    parts = urlsplit(url)
    if parts.query and _is_signed(parts.query):
        return parts._replace(query="", fragment="").geturl()
    return url


def strip_signed_urls(text: str) -> str:
    """Every signed storage link in ``text`` with its query removed."""
    return SIGNED_URL_RE.sub(lambda match: public_url(match.group(0)), text)


def scrub_signed_urls(job: Path) -> list[str]:
    """Remove signed storage links from every text file of a job folder before it is committed.

    Downloaded files under ``input/external/`` are left byte for byte as fetched
    (their hashes are checked); only the logs this helper writes there are
    scrubbed. Returns the paths that changed, relative to ``job``.
    """
    changed = []
    base = job.resolve()
    external = (job / "input" / "external").resolve()
    for path in sorted(job.rglob("*")):
        if not path.is_file() or path.is_symlink() or ".git" in path.relative_to(job).parts \
                or not path.resolve().is_relative_to(base):
            continue
        if path.suffix.lower() not in SCRUB_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
            continue
        if path.resolve().is_relative_to(external) and path.name not in ("fetch_log.json", "licence_log.json"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        clean = strip_signed_urls(text)
        if clean != text:
            path.write_text(clean, encoding="utf-8")
            changed.append(path.relative_to(job).as_posix())
    return changed


def fetch(job: Path, url: str, name: str | None = None) -> dict:
    if urlsplit(url).scheme != "https":
        raise ValueError("only https URLs are fetched")
    hosts = _hosts(job)
    if not host_allowed(host_of(url), hosts):
        raise ValueError(f"{host_of(url)} is not on this build's list of research hosts")
    if looks_like_participant_data(url):
        raise ValueError("this URL names a participant-data file; materials searches never download participant data")
    out_dir = job / "input" / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "fetch_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else []
    used = sum(int(item.get("bytes", 0)) for item in log)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(_CheckedRedirects(hosts))
    with opener.open(request, timeout=120) as response:
        final_url = response.geturl()
        # The server's own file name is always checked, whatever name the file is saved under.
        served = _filename(response, final_url)
        for candidate in (served, name or ""):
            if candidate and looks_like_participant_data(candidate):
                raise ValueError(f"{candidate} looks like participant data; materials searches never download participant data")
        filename = _safe_name(name or served)
        if Path(filename).suffix.lower() != Path(_safe_name(served)).suffix.lower() and Path(served).suffix:
            raise ValueError("--name must keep the server's file extension")
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_FILE:
            raise ValueError(f"file exceeds the {MAX_FILE // (1024 * 1024)} MiB limit")
        digest = hashlib.sha256()
        target = out_dir / filename
        stem, suffix, index = target.stem, target.suffix, 2
        while target.exists():
            target = out_dir / f"{stem}-{index}{suffix}"
            index += 1
        size = 0
        try:
            with target.open("xb") as sink:
                while chunk := response.read(CHUNK):
                    size += len(chunk)
                    if size > MAX_FILE or used + size > MAX_TOTAL:
                        raise ValueError("download exceeds the per-file or per-build size limit")
                    digest.update(chunk)
                    sink.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        content_type = response.headers.get("Content-Type", "")
    if target.suffix.lower() in ARCHIVE_SUFFIXES or zipfile.is_zipfile(target) and target.suffix.lower() not in (
            ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".psyexp"):
        member = _archive_has_data(target) if zipfile.is_zipfile(target) else "an archive that cannot be listed"
        if member:
            target.unlink(missing_ok=True)
            raise ValueError(f"the archive contains {member}, which looks like participant data; download the "
                             "material files one by one instead")
    if target.suffix.lower() == ".ipynb" and _notebook_shows_data(target):
        target.unlink(missing_ok=True)
        raise ValueError("this notebook stores cell outputs that print tables, which can show participant data; "
                         "fetch the same analysis as a script (.R, .Rmd, .py) or a notebook saved without outputs")
    record = {
        "url": url,
        "final_url": public_url(final_url),
        "path": target.relative_to(job).as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": size,
        "content_type": content_type,
        "retrieved": dt.date.today().isoformat(),
    }
    log.append(record)
    log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return record


# OSF's licence names and their SPDX ids.
OSF_LICENCES = {
    "cc-by attribution 4.0 international": "CC-BY-4.0",
    "cc-by attribution-sharealike 4.0 international": "CC-BY-SA-4.0",
    "cc-by attribution-noderivatives 4.0 international": "CC-BY-ND-4.0",
    "cc-by attribution-noncommercial 4.0 international": "CC-BY-NC-4.0",
    "cc-by attribution-noncommercial-sharealike 4.0 international": "CC-BY-NC-SA-4.0",
    "cc-by attribution-noncommercial-noderivatives 4.0 international": "CC-BY-NC-ND-4.0",
    "cc0 1.0 universal": "CC0-1.0",
    "mit license": "MIT",
    "apache license 2.0": "Apache-2.0",
    "gnu general public license (gpl) 3.0": "GPL-3.0",
    "gnu general public license (gpl) 2.0": "GPL-2.0",
    "mozilla public license 2.0": "MPL-2.0",
    "no license": "none stated",
}
OSF_URL_RE = re.compile(r"(?i)^https://(?:www\.|api\.)?osf\.io/(?:v2/(?:nodes|registrations|files|guids)/|download/)?"
                        r"([a-z0-9]{5,24})(?=/|\?|$)")
OSF_STORAGE_FILE_RE = re.compile(r"(?i)^https://(?:www\.)?osf\.io/[a-z0-9]{5}/files/osfstorage/([a-z0-9]{24})(?=/|\?|$)")


def _api_json(job: Path, url: str) -> dict | None:
    """A small JSON document from a listed host; None when it is not there (HTTP 404/410)."""
    hosts = _hosts(job)
    if urlsplit(url).scheme != "https" or not host_allowed(host_of(url), hosts):
        raise ValueError(f"{host_of(url)} is not on this build's list of research hosts")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.build_opener(_CheckedRedirects(hosts)).open(request, timeout=60) as response:
            data = response.read(4 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410):
            return None
        raise ValueError(f"{url} answered HTTP {exc.code}") from exc
    if len(data) > 4 * 1024 * 1024:
        raise ValueError(f"{url} is not a small API record")
    return json.loads(data.decode("utf-8"))


def _osf_node(job: Path, kind: str, node_id: str) -> dict | None:
    api = f"https://api.osf.io/v2/{kind}/{node_id}/?embed=license"
    record = _api_json(job, api)
    data = record.get("data") if isinstance(record, dict) else None
    if not isinstance(data, dict):
        return None
    attributes = data.get("attributes") or {}
    embedded = ((data.get("embeds") or {}).get("license") or {}).get("data") or {}
    name = (embedded.get("attributes") or {}).get("name") if isinstance(embedded, dict) else None
    relationships = data.get("relationships") or {}
    parent = ((relationships.get("parent") or {}).get("data") or {})
    source = ((relationships.get("registered_from") or {}).get("data") or {})
    holders = (attributes.get("node_license") or {}) if isinstance(attributes.get("node_license"), dict) else {}
    return {
        "id": node_id, "type": kind, "title": attributes.get("title"), "api": api,
        "licence_name": name or "none stated",
        "licence_explicitly_none": str(name or "").strip().lower() == "no license",
        "licence": OSF_LICENCES.get(str(name or "no license").strip().lower(), name),
        "copyright_holders": [h for h in holders.get("copyright_holders") or [] if h], "year": holders.get("year"),
        "parent": parent.get("id") if isinstance(parent, dict) else None,
        # A registration is a frozen copy of a project; that project's own licence is checked separately.
        "registered_from": source.get("id") if isinstance(source, dict) else None,
    }


def osf_licence(job: Path, url: str) -> dict:
    """The licence of an OSF file's project (or of a project or registration), read from the API.

    Nothing is downloaded. A file's record carries its name and the SHA-256 OSF
    lists for it, so a copy of the same file elsewhere can be matched by hash.
    The licence is the nearest one stated on the item or its parent projects.
    The record is appended to ``input/external/licence_log.json``.
    """
    match = OSF_STORAGE_FILE_RE.match(url.strip()) or OSF_URL_RE.match(url.strip())
    if not match:
        raise ValueError("--licence takes an OSF project, registration or file URL (https://osf.io/<id>/)")
    osf_id = match.group(1)
    record: dict = {"url": url, "osf_id": osf_id, "checked": dt.date.today().isoformat()}
    file_record = _api_json(job, f"https://api.osf.io/v2/files/{osf_id}/")
    data = file_record.get("data") if isinstance(file_record, dict) else None
    chain: list[dict] = []
    if isinstance(data, dict) and data.get("type") == "files":
        attributes = data.get("attributes") or {}
        hashes = (attributes.get("extra") or {}).get("hashes") or {}
        record["file"] = {"name": attributes.get("name"), "sha256": hashes.get("sha256"),
                          "guid": attributes.get("guid"), "path": attributes.get("materialized_path")}
        target = (((data.get("relationships") or {}).get("target") or {}).get("data") or {})
        kind, node_id = target.get("type"), target.get("id")
    else:
        kind, node_id = None, osf_id
    for candidate in ([kind] if kind else ["nodes", "registrations"]):
        node = _osf_node(job, candidate, node_id) if node_id else None
        if node:
            chain.append(node)
            break
    if not chain:
        raise ValueError(f"OSF has no public project, registration or file {osf_id}")
    # Up the parents until a licence is stated; an explicit "No license" is a statement too and stops the walk.
    while chain[-1].get("parent") and chain[-1]["licence"] == "none stated" \
            and not chain[-1]["licence_explicitly_none"] and len(chain) < 6:
        parent = _osf_node(job, chain[-1]["type"], chain[-1]["parent"])
        if not parent:
            break
        chain.append(parent)
    record["projects"] = chain
    source = chain[0].get("registered_from")
    if source:
        # Recorded for the reader; a registration does not take its project's licence automatically.
        record["registered_from"] = _osf_node(job, "nodes", source)
    stated = next((node for node in chain if node["licence"] != "none stated"), None)
    record["licence"] = stated["licence"] if stated else "none stated"
    record["licence_from"] = (f"OSF {stated['type'][:-1]} {stated['id']} ('{stated['title']}'): "
                              f"License {stated['licence_name']}") if stated else \
        "no licence on the OSF item or its parent projects"
    out_dir = job / "input" / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "licence_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else []
    log.append(record)
    log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return record


def remote_copy(job: Path, url: str, attempts: int = 3) -> bytes:
    """The bytes now served at ``url`` (not saved), retried on network errors."""
    import time

    hosts = _hosts(job)
    if urlsplit(url).scheme != "https" or not host_allowed(host_of(url), hosts):
        raise ValueError(f"{url} is not an https URL on this build's research hosts")
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            chunks, size = [], 0
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.build_opener(_CheckedRedirects(hosts)).open(request, timeout=120) as response:
                while chunk := response.read(CHUNK):
                    size += len(chunk)
                    if size > MAX_FILE:
                        raise ValueError("file exceeds the size limit")
                    chunks.append(chunk)
            return b"".join(chunks)
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                # Not found or refused: the recorded URL does not serve this file.
                raise ValueError(f"{url} answered HTTP {exc.code}; the recorded source is not there") from exc
            last = exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.gaierror):
                raise ValueError(f"{host_of(url)} does not exist; the recorded source is not there") from exc
            last = exc
        except OSError as exc:
            last = exc
        time.sleep(2 * (attempt + 1) if attempt + 1 < attempts else 0)
    raise ConnectionError(f"could not re-download {url}: {last}")


def _external_sources(job: Path) -> list[dict]:
    package = job / "package"
    roots = [path for path in package.iterdir() if path.is_dir()] if package.is_dir() else []
    if len(roots) != 1:
        return []
    try:
        sources = json.loads((roots[0] / "source/paper_metadata.json").read_text(encoding="utf-8")).get("external_sources")
    except (OSError, ValueError, AttributeError):
        return []
    return [s for s in sources if isinstance(s, dict)] if isinstance(sources, list) else []


def restore_downloads(job: Path) -> list[str]:
    """Download again the recorded sources of a copied package (a refinement job
    starts with the package but without the previous job's input/external/)."""
    notes = []
    for source in _external_sources(job):
        path, url, sha = source.get("path"), source.get("url"), source.get("sha256")
        if not (isinstance(path, str) and path.startswith("input/external/") and isinstance(url, str)):
            continue
        if (job / path).is_file():
            continue
        try:
            record = fetch(job, url, Path(path).name)
        except (OSError, ValueError) as exc:
            notes.append(f"could not restore {path}: {exc}")
            continue
        if record["path"] != path or record["sha256"] != sha:
            notes.append(f"{url} has changed since it was recorded ({path}); the build must re-check it")
    return notes


def verify_downloads(job: Path, attempts: int = 3) -> tuple[list[str], list[str]]:
    """Download every recorded source again; return (problems, warnings).

    A file the agent wrote or edited itself does not match what its URL serves.
    Files must match byte for byte; a web page (HTML) changes between requests,
    so for a page every quote the program cites from it must still occur in the
    page's text. A source that cannot be reached is a warning, not a failure."""
    try:
        from agent_pipeline.check_sources import file_text, quote_in_text, normalise
    except ModuleNotFoundError:
        from check_sources import file_text, quote_in_text, normalise
    problems, warnings = [], []
    quotes = _quotes_by_source(job)
    for source in _external_sources(job):
        url, sha, path = source.get("url"), source.get("sha256"), source.get("path")
        if not (isinstance(url, str) and isinstance(sha, str) and isinstance(path, str)):
            continue
        try:
            data = remote_copy(job, url, attempts)
        except ConnectionError as exc:
            warnings.append(str(exc))
            continue
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if hashlib.sha256(data).hexdigest() == sha:
            continue
        if Path(path).suffix.lower() in (".html", ".htm") or data.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html")):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                page = Path(tmp) / "page.html"
                page.write_bytes(data)
                text = file_text(page) or ""
            missing = [q for q in quotes.get(path, []) if not quote_in_text(q, text)]
            if missing:
                problems.append(f"{url} no longer shows text cited from {path}: {normalise(missing[0])[:80]!r}")
            continue
        problems.append(f"{url} does not serve the file recorded as {path} (sha256 differs)")
    return problems, warnings


def _quotes_by_source(job: Path) -> dict[str, list[str]]:
    package = job / "package"
    roots = [p for p in package.iterdir() if p.is_dir()] if package.is_dir() else []
    try:
        program = json.loads((roots[0] / "study.json").read_text(encoding="utf-8"))["program"]
    except (IndexError, OSError, ValueError, KeyError, TypeError):
        return {}
    quotes: dict[str, list[str]] = {}
    for record in program.get("evidence", []) if isinstance(program, dict) else []:
        if isinstance(record, dict) and isinstance(record.get("path"), str) and isinstance(record.get("quote"), str):
            quotes.setdefault(record["path"], []).append(record["quote"])
    return quotes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--name", help="file name to save as (default: the server's name)")
    parser.add_argument("--licence", "--license", dest="licence", metavar="OSF_URL",
                        help="read the licence of an OSF project, registration or file (nothing is downloaded)")
    parser.add_argument("--scrub", action="store_true",
                        help="remove signed storage links from the job folder's text files (before a commit)")
    parser.add_argument("url", nargs="?")
    args = parser.parse_args()
    if args.scrub:
        for path in scrub_signed_urls(args.job.resolve()):
            print(f"removed signed links from {path}")
        return
    if bool(args.licence) == bool(args.url):
        parser.error("give one URL to download, or --licence <OSF URL>")
    try:
        record = osf_licence(args.job.resolve(), args.licence) if args.licence else \
            fetch(args.job.resolve(), args.url, args.name)
    except (ValueError, OSError) as exc:
        print(f"Not fetched: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
