#!/usr/bin/env python3
"""Publish a validated Claude Code result to the private job branch."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

try:
    from agent_pipeline.run_agent import studio_complete, validate
    from agent_pipeline.studio_output import studio_mode, validate_build_sidecars, validate_discussion
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from run_agent import studio_complete, validate
    from studio_output import studio_mode, validate_build_sidecars, validate_discussion


def finalize(job: Path, validator: Path) -> None:
    request_path = job / "studio_request.json"
    request = json.loads(request_path.read_text()) if request_path.is_file() else None
    mode = studio_mode(request) if request is not None else "build"
    request_id = request.get("requestId") if request is not None else None
    if not studio_complete(job, request_id):
        raise ValueError("Studio completion marker is missing or does not match this request")

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
        shutil.make_archive(str(job / "output" / "study"), "zip", job / "package")
        data.update({
            "status": "review",
            "currentStage": 1,
            "packageReady": True,
            "message": "Study package is ready for researcher review",
        })
    job_path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    finalize(Path(sys.argv[1]), Path(sys.argv[2]))
