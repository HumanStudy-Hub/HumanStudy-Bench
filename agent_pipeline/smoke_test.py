#!/usr/bin/env python3
"""Answer-aware smoke test of a built package.

    python agent_pipeline/smoke_test.py package/<paper-slug>

Runs every runnable arm in ``task/task.json`` through the package's
``run_sessions`` with :class:`replies.StubParticipant`, an offline stand-in that
answers comprehension, attention, consent and qualification checks correctly
(from the ``key`` each question carries) and picks random valid answers
otherwise. It passes only when every arm has at least one completed session,
every model call goes through ``replies.ask``, and ``evaluate`` returns a
non-empty result for the sessions. A constant "1" stub could pass the old check
while every session was excluded; this one cannot.

The package must carry ``task/replies.py`` unchanged from
``agent_pipeline/package_files/replies.py`` and its adapter must import it.
Prints a JSON report (per-arm sessions by status) and exits 1 on any problem.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

CANONICAL = Path(__file__).resolve().parent / "package_files" / "replies.py"


def _canonical():
    spec = importlib.util.spec_from_file_location("humanstudy_replies_canonical", CANONICAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def package_root(path: Path) -> Path:
    path = Path(path).resolve()
    if (path / "task" / "adapter.py").is_file():
        return path
    folders = [child for child in path.iterdir() if child.is_dir()] if path.is_dir() else []
    if len(folders) == 1:
        return folders[0]
    raise SystemExit(f"{path} is not a package folder")


def copy_problems(root: Path) -> list[str]:
    problems = []
    copy = root / "task" / "replies.py"
    want = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
    if not copy.is_file():
        problems.append("task/replies.py is missing: copy agent_pipeline/package_files/replies.py there unchanged")
    elif hashlib.sha256(copy.read_bytes()).hexdigest() != want:
        problems.append("task/replies.py differs from agent_pipeline/package_files/replies.py: copy it again, unchanged")
    adapter = (root / "task" / "adapter.py").read_text(encoding="utf-8", errors="replace")
    if not re.search(r"(?m)^\s*(from\s+replies\s+import\b|import\s+replies\b)", adapter):
        problems.append("task/adapter.py does not import replies: ask every question with replies.ask()")
    return problems


def run(path: Path, seed: int = 0) -> dict:
    root = package_root(path)
    problems = copy_problems(root)
    replies = _canonical()
    previous = os.getcwd()
    sys.dont_write_bytecode = True
    os.chdir(root)
    try:
        report = replies.smoke_package(root, seed=seed)
    except Exception as exc:  # an import error in the package is a finding, not a crash
        report = {"ok": False, "arms": {}, "problems": [f"the package could not be loaded: {type(exc).__name__}: {exc}"]}
    finally:
        os.chdir(previous)
    report["problems"] = problems + list(report.get("problems", []))
    report["ok"] = not report["problems"]
    return report


def summary(report: dict) -> str:
    """Plain-language lines for the build log and the repair prompt."""
    lines = []
    for arm, counts in report.get("arms", {}).items():
        statuses = ", ".join(f"{n} {status}" for status, n in sorted(counts.get("statuses", {}).items())) or "none"
        lines.append(f"{arm}: {counts.get('sessions', 0)} sessions ({statuses})")
    return "\n".join(lines + [f"problem: {problem}" for problem in report.get("problems", [])])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    report = run(Path(sys.argv[1]))
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
