#!/usr/bin/env python3
"""Publish a validated Claude Code result to the private job branch."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

try:
    from agent_pipeline.fetch_public import scrub_signed_urls, strip_signed_urls, verify_downloads
    from agent_pipeline.run_agent import studio_complete, validate
    from agent_pipeline.studio_output import studio_mode, validate_build_sidecars, validate_discussion
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from fetch_public import scrub_signed_urls, strip_signed_urls, verify_downloads
    from run_agent import studio_complete, validate
    from studio_output import studio_mode, validate_build_sidecars, validate_discussion


def remove_bytecode(folder: Path) -> None:
    """Python's __pycache__ folders are build leftovers, not part of a package."""
    for cache in sorted(folder.rglob("__pycache__"), reverse=True):
        if cache.is_dir() and not cache.is_symlink():
            shutil.rmtree(cache, ignore_errors=True)
    for stray in folder.rglob("*.py[co]"):
        if stray.is_file():
            stray.unlink()


def unverified_note(warnings: list[str]) -> str:
    """The sentence added to the review message when some sources could not be re-checked."""
    if not warnings:
        return ""
    urls = [match.group(0).rstrip(".,;)") for w in warnings if (match := re.search(r"https://\S+", w))]
    shown = ", ".join(dict.fromkeys(urls[:3])) + (" and others" if len(urls) > 3 else "")
    many = len(warnings) != 1
    where = f" ({shown})" if shown else ""
    return (f". {len(warnings)} public source{'s' if many else ''} could not be re-checked from the build "
            f"server{where}; compare the downloaded cop{'ies' if many else 'y'} with the original before "
            f"relying on {'them' if many else 'it'}")


def finalize(job: Path, validator: Path) -> None:
    request_path = job / "studio_request.json"
    request = json.loads(request_path.read_text()) if request_path.is_file() else None
    mode = studio_mode(request) if request is not None else "build"
    request_id = request.get("requestId") if request is not None else None
    if not studio_complete(job, request_id):
        raise ValueError("Studio completion marker is missing or does not match this request")

    # Signed storage links expire in minutes but are credentials while they last; none is committed or zipped.
    scrub_signed_urls(job)
    job_path = job / "job.json"
    data = json.loads(job_path.read_text())
    if mode == "discuss":
        valid, detail = validate_discussion(job, request_id)
        if not valid:
            raise ValueError(f"Invalid Studio discussion response: {detail}")
        data.update({
            "status": "complete",
            "currentStage": 1,
            "packageReady": False,
            "message": "Studio discussion is complete",
        })
    else:
        valid, detail = validate(validator, job / "package")
        if not valid:
            raise ValueError(f"Invalid study package: {detail}")
        # Each recorded external source must be what its URL serves, not a file
        # written or edited in the job directory.
        notes: list[str] = []
        mismatches, warnings = verify_downloads(job, notes=notes)
        notes, warnings = [strip_signed_urls(n) for n in notes], [strip_signed_urls(w) for w in warnings]
        mismatches = [strip_signed_urls(m) for m in mismatches]
        for note in notes:
            print(note)
        for warning in warnings:
            print(f"::warning::{warning}")
        if mismatches:
            raise ValueError("External sources could not be verified: " + "; ".join(mismatches))
        if request is not None:
            accepted_model = None
            if request.get("purpose") == "accepted-model-sync":
                document = request.get("document")
                if not isinstance(document, dict) or document.get("model") is None:
                    raise ValueError("accepted-model-sync requires document.model")
                accepted_model = document["model"]
            valid, detail = validate_build_sidecars(job / "package", accepted_model)
            if not valid:
                raise ValueError(f"Invalid Studio build sidecars: {detail}")
        (job / "output").mkdir(exist_ok=True)
        remove_bytecode(job / "package")
        shutil.make_archive(str(job / "output" / "study"), "zip", job / "package")
        data.update({
            "status": "review",
            "currentStage": 1,
            "packageReady": True,
            "message": "Study package is ready for researcher review" + unverified_note(warnings),
        })
        # A source the build server could not reach is unverified, not disproved: the
        # package is published and the review message says which copy to check
        # (sourceWarnings keeps the full lines for the job record).
        if warnings:
            data["sourceWarnings"] = warnings
        else:
            data.pop("sourceWarnings", None)
    job_path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    finalize(Path(sys.argv[1]), Path(sys.argv[2]))
