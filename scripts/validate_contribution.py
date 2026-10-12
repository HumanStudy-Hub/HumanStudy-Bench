#!/usr/bin/env python3
"""Validate one community contribution folder.

Usage (from repo root):
  python scripts/validate_contribution.py community/<paper-key>/<slug>--<contributor-id>

Messages are written for people who have never used GitHub: they say what is wrong and what to
do, not which line of Python failed. Exit status is 0 when the folder is acceptable, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator  # noqa: E402

from agent_pipeline.human_program import validate_human_program  # noqa: E402
from agent_pipeline.validate_package import validate_package  # noqa: E402
from scripts.community import (  # noqa: E402
    MAX_FILE_BYTES,
    paper_key_to_dir,
    parse_contribution_dir,
)

CONTRACTS = REPO_ROOT / "contracts"
ALLOWED_TOP_LEVEL = {"CONTRIBUTION.json", "program.json", "package", "artifacts"}


def _load_schema(name: str) -> dict:
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


def _read_json(path: Path, label: str, errors: list) -> Optional[Any]:
    if not path.is_file():
        errors.append(f"{label} is missing.")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{label} is not valid JSON ({exc}). Re-export it from the app.")
        return None


def schema_errors(value: Any, schema_name: str, label: str) -> list:
    """Plain-language messages for every way ``value`` breaks the named schema."""
    validator = Draft202012Validator(_load_schema(schema_name))
    messages = []
    for err in sorted(validator.iter_errors(value), key=lambda e: list(map(str, e.path))):
        where = " > ".join(str(p) for p in err.path)
        field = f"{label} field '{where}'" if where else label
        if err.validator == "required":
            missing = err.message.split("'")[1] if "'" in err.message else "a required field"
            messages.append(
                f"{label}: '{missing}' is missing" + (f" in '{where}'." if where else ".")
            )
        elif err.validator == "additionalProperties":
            messages.append(f"{label}: unexpected entry ({err.message}).")
        else:
            messages.append(f"{field} is not valid: {err.message}")
    return messages


def check_files(folder: Path, includes_pdf: bool, errors: list) -> None:
    """No stray top-level files, no links out of the folder, no huge files, no unflagged PDFs."""
    for entry in sorted(folder.iterdir()):
        if entry.name not in ALLOWED_TOP_LEVEL:
            errors.append(
                f"'{entry.name}' does not belong in a contribution. Only program.json, "
                "CONTRIBUTION.json, package/ and artifacts/ are published."
            )
    root = folder.resolve()
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder).as_posix()
        if path.is_symlink():
            errors.append(f"'{rel}' is a link. Links are not allowed; include the real file.")
            continue
        resolved = path.resolve()
        if root != resolved and root not in resolved.parents:
            errors.append(f"'{rel}' points outside the contribution folder.")
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            size = path.stat().st_size / 1024 / 1024
            errors.append(f"'{rel}' is {size:.0f} MB; the limit is 25 MB per file.")
        if path.suffix.lower() == ".pdf" and not includes_pdf:
            errors.append(
                f"'{rel}' is a PDF, but this contribution is not marked as including the paper. "
                "Remove it, or tick 'Include the paper PDF' if you hold the rights."
            )


def validate_contribution(folder: Path) -> list:
    """Return a list of problems; an empty list means the contribution is acceptable."""
    errors: list = []
    folder = Path(folder)
    if not folder.is_dir():
        return [f"Folder '{folder}' does not exist."]
    parsed = parse_contribution_dir(folder.name)
    if not parsed:
        errors.append(
            f"Folder name '{folder.name}' must look like <slug>--<contributor-id>, "
            "for example 'knobe-2003--gh-xuanl17' (add -v2, -v3 for updates)."
        )

    contribution = _read_json(folder / "CONTRIBUTION.json", "CONTRIBUTION.json", errors)
    includes_pdf = False
    if isinstance(contribution, dict):
        msgs = schema_errors(contribution, "contribution.schema.json", "CONTRIBUTION.json")
        errors.extend(msgs)
        includes_pdf = contribution.get("includesPdf") is True
        if not msgs:
            _check_identity(folder, parsed, contribution, errors)
    elif contribution is not None:
        errors.append("CONTRIBUTION.json must be a JSON object.")

    check_files(folder, includes_pdf, errors)

    program = _read_json(folder / "program.json", "program.json", errors)
    if program is not None:
        try:
            validate_human_program(program)
        except ValueError as exc:
            errors.append(f"program.json is not a valid Study document: {exc}")
            program = None

    paper = _read_json(folder.parent / "paper.json", "paper.json (in the paper folder)", errors)
    if isinstance(paper, dict):
        errors.extend(schema_errors(paper, "paper.schema.json", "paper.json"))
        if paper.get("paperKey") and paper_key_to_dir(str(paper["paperKey"])) != folder.parent.name:
            errors.append(
                f"paper.json is for '{paper['paperKey']}' but sits in folder "
                f"'{folder.parent.name}'."
            )
        if isinstance(contribution, dict) and paper.get("paperKey") != contribution.get("paperKey"):
            errors.append("paper.json and CONTRIBUTION.json name different papers (paperKey).")

    package = folder / "package"
    if package.exists():
        _check_package(package, contribution, program, errors)
    elif isinstance(contribution, dict):
        _check_claims(folder, contribution, errors)
    return errors


def _check_identity(folder: Path, parsed: Optional[dict], contribution: dict, errors: list) -> None:
    if paper_key_to_dir(contribution["paperKey"]) != folder.parent.name:
        errors.append(
            f"CONTRIBUTION.json is for paper '{contribution['paperKey']}' but the folder is "
            f"under '{folder.parent.name}'."
        )
    if parsed and parsed["slug"] != contribution["slug"]:
        errors.append(
            f"CONTRIBUTION.json says slug '{contribution['slug']}' but the folder name starts "
            f"with '{parsed['slug']}'."
        )
    superseded = contribution.get("supersedes")
    if superseded and not (folder.parent / superseded).is_dir():
        errors.append(f"'supersedes' names '{superseded}', which does not exist for this paper.")
    if contribution["contributor"].get("github") is None and not contribution["contributor"].get(
        "hub"
    ):
        errors.append(
            "CONTRIBUTION.json needs a GitHub username or a HumanStudy Hub id so the work can be "
            "credited to someone."
        )


def _check_claims(folder: Path, contribution: dict, errors: list) -> None:
    checks = contribution.get("checks")
    if not isinstance(checks, dict):
        return
    if checks.get("package") is True:
        errors.append(
            "CONTRIBUTION.json says the package passed its checks, but there is no package/ folder."
        )
    sims = folder / "artifacts" / "simulations"
    if checks.get("simulation") is True and not (sims.is_dir() and any(sims.iterdir())):
        errors.append(
            "CONTRIBUTION.json says a simulation passed, but artifacts/simulations/ is empty."
        )


def _check_package(package: Path, contribution: Any, program: Any, errors: list) -> None:
    slug = contribution.get("slug") if isinstance(contribution, dict) else None
    inside = [p for p in package.iterdir() if p.is_dir()]
    if slug and [p.name for p in inside] != [slug]:
        errors.append(f"package/ must contain exactly one folder, named '{slug}'.")
        return
    try:
        root = validate_package(package)
    except SystemExit as exc:
        errors.append(f"The runnable package did not pass its checks: {exc.code}")
        return
    except Exception as exc:  # noqa: BLE001 - surface anything a bad package can raise
        errors.append(f"The runnable package could not be checked: {exc}")
        return
    if program is not None:
        study = json.loads((root / "study.json").read_text(encoding="utf-8"))
        if study.get("program") != program:
            errors.append(
                "package/study.json holds a different Study document than program.json. "
                "Publish again from the app so both are written from the same version."
            )


def main(argv: Optional[list] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: validate_contribution.py <contribution-dir>", file=sys.stderr)
        return 2
    folder = Path(args[0])
    errors = validate_contribution(folder)
    if errors:
        print(f"This contribution cannot be accepted yet ({folder.name}):", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print(f"Contribution OK: {folder.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
