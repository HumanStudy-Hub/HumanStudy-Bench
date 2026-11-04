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
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    from agent_pipeline.network_policy import allowed_hosts, host_allowed, host_of, looks_like_participant_data
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from network_policy import allowed_hosts, host_allowed, host_of, looks_like_participant_data

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
    with urllib.request.urlopen(request, timeout=120) as response:
        final_url = response.geturl()
        filename = _safe_name(name or _filename(response, final_url))
        if looks_like_participant_data(filename):
            raise ValueError(f"{filename} looks like participant data; materials searches never download participant data")
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
