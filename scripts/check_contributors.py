#!/usr/bin/env python3
"""Contributor attribution for submitted studies (one rule for both doors, V080-PLATFORM §2.1 / B2).

    python scripts/check_contributors.py --author <github-login> <study-dir> [<study-dir> ...]

A legacy study (`studies/study_NNN`) keeps its contributors in index.json; a protocol study keeps them in
study.json. Each directory must list a contributor whose `github` is the PR author ('login' or
'https://github.com/login'). A directory with neither file is skipped, as before. The maintainer bypass
(admin/maintain permission) stays in the workflow, which never calls this script for maintainers.

This is the inline check that validate-studies.yml carried, moved here unchanged except that study.json
is read when there is no index.json. Exit 1 when any directory fails.
"""
import argparse
import json
import sys
from pathlib import Path


def contributors_file(study_dir: Path):
    for name in ("index.json", "study.json"):
        if (study_dir / name).is_file():
            return study_dir / name
    return None


def verdict(path: Path, author: str) -> str:
    """MATCH, MISMATCH or MISSING, exactly as the former inline check."""
    return verdict_from(json.loads(path.read_text(encoding="utf-8")), author)


def verdict_from(data: dict, author: str) -> str:
    contributors = data.get("contributors", [])
    if not contributors:
        return "MISSING"
    # Match 'username' or 'https://github.com/username'
    for c in contributors:
        gh = (c.get("github", "") if isinstance(c, dict) else "").lower()
        if gh.rstrip("/").split("/")[-1] == author.lower():
            return "MATCH"
    return "MISMATCH"


def with_contributor(raw: bytes, login: str, name: str = "") -> bytes:
    """study.json bytes with the PR author appended to contributors when absent (community proposals)."""
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("contributors", []), list):
        return raw   # not a study card; the release checks report the malformed study.json
    if verdict_from(data, login) == "MATCH":
        return raw
    data.setdefault("contributors", []).append({"name": name or login, "github": login})
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--author", required=True)
    ap.add_argument("dirs", nargs="*")
    a = ap.parse_args(argv)
    failed = False
    for d in a.dirs:
        study_dir = Path(d)
        path = contributors_file(study_dir)
        if path is None:
            continue
        study_id = study_dir.name
        if verdict(path, a.author) in ("MISSING", "MISMATCH"):
            print(f"::error file={path}::Contributor check failed for {study_id}. Your GitHub username is "
                  f"'{a.author}'. Please add the following to your {path.name}:%0A%0A\"contributors\": [%0A  "
                  f"{{ \"name\": \"Your Name\", \"github\": \"{a.author}\" }}%0A]")
            failed = True
        else:
            print(f"{study_id}: Contributor @{a.author} verified.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
