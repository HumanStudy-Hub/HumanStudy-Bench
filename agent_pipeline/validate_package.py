#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path


try:
    from agent_pipeline.human_program import validate_human_program
    from agent_pipeline.check_sources import check_package_sources
except ModuleNotFoundError:
    from human_program import validate_human_program
    from check_sources import check_package_sources

REQUIRED = (
    "study.json",
    "source/paper_metadata.json",
    "source/evidence.json",
    "materials/materials.json",
    "task/task.json",
    "task/adapter.py",
    "evaluation/evaluation.py",
    "audit/missing_information.json",
)


def main() -> None:
    package = Path(sys.argv[1]).resolve()
    roots = [path for path in package.iterdir() if path.is_dir()] if package.exists() else []
    if len(roots) != 1:
        raise SystemExit("package must contain exactly one paper folder")
    root = roots[0]
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise SystemExit("missing required package files: " + ", ".join(missing))
    for path in root.rglob("*.json"):
        try:
            json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid JSON in {path.relative_to(root)}: {exc}") from exc
    study = json.loads((root / "study.json").read_text())
    if "program" in study:
        program = validate_human_program(study["program"])
        task = json.loads((root / "task/task.json").read_text())
        expected = {
            "programId": program["id"],
            "stepIds": [s["id"] for s in program["steps"]],
            "recordNodeIds": [n["id"] for n in program["nodes"] if n["kind"] in ("record", "variable")],
            "analysisNodeIds": [n["id"] for n in program["nodes"] if n["kind"] == "analysis"],
        }
        if task.get("programBindings") != expected:
            raise SystemExit("task programBindings do not match the canonical Human Program")
    # Public materials found outside the paper: provenance, licence, no participant
    # data, and no invented wording passed off as rebuilt structure.
    try:
        problems = check_package_sources(root, package.parent)
    except Exception as exc:  # A malformed record must be reported, not crash the repair loop.
        problems = [f"source/paper_metadata.json or the program could not be checked "
                    f"({type(exc).__name__}: {exc}); fix the malformed external source or material record"]
    if problems:
        raise SystemExit("external materials check failed:\n- " + "\n- ".join(problems))
    result = subprocess.run(
        [sys.executable, str(root / "task/adapter.py"), "--smoke-test"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode != 0:
        raise SystemExit("adapter smoke test failed:\n" + result.stdout + result.stderr)
    # The pipeline's own smoke test: an answer-aware stub must take every arm to evaluate().
    smoke = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "smoke_test.py"), str(root)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if smoke.returncode != 0:
        try:
            report = json.loads(smoke.stdout)
            detail = "\n- ".join(report.get("problems") or ["no detail"])
            arms = "\n".join(f"  {arm}: {c.get('statuses')}" for arm, c in report.get("arms", {}).items())
        except ValueError:
            detail, arms = (smoke.stdout + smoke.stderr)[-4000:], ""
        raise SystemExit("answer-aware smoke test failed (python agent_pipeline/smoke_test.py <package>):\n- "
                         + detail + ("\nsessions by arm:\n" + arms if arms else ""))

    def _defines(path: Path, name: str) -> bool:
        return bool(re.search(rf"^\s*def\s+{name}\s*\(", path.read_text(encoding="utf-8"), re.MULTILINE))

    if not _defines(root / "task/adapter.py", "run_sessions"):
        raise SystemExit("task/adapter.py must expose run_sessions(agent_fn, seed) so the playground can run it.")
    if not _defines(root / "evaluation/evaluation.py", "evaluate"):
        raise SystemExit("evaluation/evaluation.py must expose evaluate(sessions) so the playground can score it.")

    print(f"Validated agent package: {root.name}")


if __name__ == "__main__":
    main()
