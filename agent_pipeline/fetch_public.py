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
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
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
    record = {
        "url": url,
        "final_url": final_url,
        "path": target.relative_to(job).as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": size,
        "content_type": content_type,
        "retrieved": dt.date.today().isoformat(),
    }
    log.append(record)
    log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return record


def remote_sha256(job: Path, url: str, attempts: int = 3) -> str:
    """SHA-256 of the file now served at ``url``, without saving it."""
    hosts = _hosts(job)
    if urlsplit(url).scheme != "https" or not host_allowed(host_of(url), hosts):
        raise ValueError(f"{url} is not an https URL on this build's research hosts")
    last: Exception | None = None
    for _ in range(attempts):
        try:
            digest, size = hashlib.sha256(), 0
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.build_opener(_CheckedRedirects(hosts)).open(request, timeout=120) as response:
                while chunk := response.read(CHUNK):
                    size += len(chunk)
                    if size > MAX_FILE:
                        raise ValueError("file exceeds the size limit")
                    digest.update(chunk)
            return digest.hexdigest()
        except (OSError, ValueError) as exc:
            last = exc
    raise ValueError(f"could not re-download {url}: {last}")


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


def verify_downloads(job: Path) -> list[str]:
    """Problems found by downloading every recorded source again: a file the agent
    wrote itself, or edited after download, does not match what the URL serves."""
    problems = []
    for source in _external_sources(job):
        url, sha = source.get("url"), source.get("sha256")
        if not (isinstance(url, str) and isinstance(sha, str)):
            continue
        try:
            if remote_sha256(job, url) != sha:
                problems.append(f"{url} does not serve the file recorded as {source.get('path')} (sha256 differs)")
        except ValueError as exc:
            problems.append(str(exc))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--name", help="file name to save as (default: the server's name)")
    parser.add_argument("url")
    args = parser.parse_args()
    try:
        record = fetch(args.job.resolve(), args.url, args.name)
    except (ValueError, OSError) as exc:
        print(f"Not fetched: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
