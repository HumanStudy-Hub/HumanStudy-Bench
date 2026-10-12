#!/usr/bin/env python3
"""Decide whether a pull request that touches community/ follows the contributor rules.

Usage (CI):
  python scripts/check_community_pr.py --author <login> --base origin/main

Reads ``git diff --name-status <base>...HEAD -- community/``. Prints one line per contribution
folder to validate on stdout-file ``--folders-out`` and, when ``GITHUB_OUTPUT`` is set, writes
``add_only=true|false``. Exit status 1 means the PR must not merge as it stands.

Rules (docs: community/README.md):
  * A new contribution folder: the PR author is ``contributor.github`` (case-insensitive), or the
    author is humanstudy-bot and ``contributor.hub.signature`` is present. The signature itself
    is verified server-side; CI only checks that it exists.
  * Changing or deleting files in an existing folder: only its owner, humanstudy-bot or a
    maintainer. Everyone else proposes a correction that the owner reviews.
  * Files outside contribution folders (README, shared paper.json edits): bot or maintainer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.community import BOT_LOGIN, MAINTAINERS, github_matches, is_maintainer  # noqa: E402


def parse_name_status(text: str) -> list:
    """Turn ``git diff --name-status`` output into (status letter, path) pairs.

    A rename is reported as a delete of the old path plus an add of the new one.
    """
    changes = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0]
        if status in "RC" and len(parts) >= 3:
            if status == "R":
                changes.append(("D", parts[1]))
            changes.append(("A", parts[2]))
        else:
            changes.append((status, parts[-1]))
    return changes


def _classify(path: str) -> tuple:
    """('contribution', paper_dir, folder) | ('paper', paper_dir, None) | ('other', None, None)."""
    parts = PurePosixPath(path).parts
    if len(parts) >= 4 and parts[0] == "community":
        return ("contribution", parts[1], parts[2])
    if len(parts) == 3 and parts[0] == "community" and parts[2] == "paper.json":
        return ("paper", parts[1], None)
    return ("other", None, None)


def check_changes(
    changes: list,
    author: str,
    read_head: Callable[[str], Optional[str]],
    read_base: Callable[[str], Optional[str]],
) -> tuple:
    """Return (errors, folders_to_validate, add_only)."""
    errors: list = []
    folders: dict = {}
    privileged = author.lower() == BOT_LOGIN or is_maintainer(author)
    add_only = bool(changes)
    maintainers = ", ".join(MAINTAINERS)

    for status, path in changes:
        kind, paper_dir, folder = _classify(path)
        if status != "A":
            add_only = False
        if kind == "contribution":
            folders.setdefault((paper_dir, folder), []).append((status, path))
        elif kind == "paper":
            if status != "A":
                add_only = False
                if not privileged:
                    errors.append(
                        f"{path}: the paper details are shared by everyone who contributed to "
                        f"this paper, so only a maintainer ({maintainers}) can change them. "
                        "Open an issue describing the correction."
                    )
        else:
            add_only = False
            if not privileged:
                errors.append(
                    f"{path}: community pull requests may only add contribution folders. "
                    f"Changes elsewhere are reviewed by a maintainer ({maintainers})."
                )

    for (paper_dir, folder), items in sorted(folders.items()):
        base_path = f"community/{paper_dir}/{folder}/CONTRIBUTION.json"
        base_source = read_base(base_path)
        is_new = base_source is None
        # An existing folder is judged by its owner at the base commit, so a PR cannot hand
        # itself ownership by rewriting CONTRIBUTION.json.
        source = read_head(base_path) if is_new else base_source
        try:
            meta = json.loads(source) if source else None
        except json.JSONDecodeError:
            meta = None
        contributor = (meta or {}).get("contributor") if isinstance(meta, dict) else None
        if not isinstance(contributor, dict):
            errors.append(
                f"community/{paper_dir}/{folder}: CONTRIBUTION.json is missing or unreadable, "
                "so the owner cannot be checked."
            )
            continue
        owner = contributor.get("github")
        has_signature = bool((contributor.get("hub") or {}).get("signature"))
        if is_new:
            if author.lower() == BOT_LOGIN:
                if not has_signature:
                    errors.append(
                        f"community/{paper_dir}/{folder}: submissions made through HumanStudy "
                        "Hub must carry a signed Hub id in CONTRIBUTION.json "
                        "(contributor.hub.signature)."
                    )
            elif not (is_maintainer(author) or github_matches(owner, author)):
                errors.append(
                    f"community/{paper_dir}/{folder}: your GitHub username is '{author}', but "
                    f"CONTRIBUTION.json credits '{owner or 'no GitHub account'}'. Set "
                    f'"contributor": {{ "name": "Your Name", "github": "{author}" }}, or publish '
                    "from the HumanStudy app, which fills this in for you."
                )
        else:
            add_only = False
            if not (privileged or github_matches(owner, author)):
                errors.append(
                    f"community/{paper_dir}/{folder} belongs to {owner or 'another contributor'}. "
                    "Corrections to someone else's work are reviewed by its owner, so this "
                    "pull request needs their approval; CI will not accept it on its own."
                )
    to_validate = [
        f"community/{p}/{f}"
        for (p, f), items in sorted(folders.items())
        if any(s != "D" for s, _ in items)
    ]
    return errors, to_validate, add_only and not errors


def _git(*args: str) -> Optional[str]:
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT)
    return result.stdout if result.returncode == 0 else None


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--author", required=True)
    parser.add_argument("--base", required=True, help="base ref, for example origin/main")
    parser.add_argument("--folders-out", type=Path, help="write folders to validate here")
    args = parser.parse_args(argv)

    diff = _git("diff", "--name-status", "--no-renames", f"{args.base}...HEAD", "--", "community/")
    if diff is None:
        print(f"Could not compare against {args.base}.", file=sys.stderr)
        return 2
    changes = parse_name_status(diff)

    def read_head(path: str) -> Optional[str]:
        file = REPO_ROOT / path
        return file.read_text(encoding="utf-8") if file.is_file() else None

    errors, folders, add_only = check_changes(
        changes, args.author, read_head, lambda p: _git("show", f"{args.base}:{p}")
    )
    if args.folders_out:
        args.folders_out.write_text("\n".join(folders) + ("\n" if folders else ""))
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"add_only={'true' if add_only else 'false'}\n")
    for message in errors:
        print(f"::error::{message}")
    if errors:
        return 1
    print(f"Contributor rules satisfied for @{args.author}; add-only: {add_only}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
