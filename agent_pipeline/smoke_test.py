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

Both stub passes reply from the rendered prompt, as a model would (the number
printed before an option, or its position when none is printed), never from the
codes a question carries, so a screen whose numbers the reader cannot read back
fails here rather than in a paid run. And every question a recorded Qualtrics
survey accepts only with a correct answer (custom validation, forced or not) must
be asked with a ``key``: the survey's own rule, not the package's choice.
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


def _plain(text: object) -> str:
    text = re.sub(r"<[^>]+>|[*_]", " ", str(text or "")).replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip().strip(" .,;:!?").casefold()


def survey_checks(root: Path, job: Path) -> list[dict]:
    """Correct-answer checks of every recorded .qsf source, as agent_pipeline/qsf_flow.py reads them."""
    try:
        from agent_pipeline.qsf_flow import correct_answer_checks, read_qsf
    except ModuleNotFoundError:
        from qsf_flow import correct_answer_checks, read_qsf
    try:
        metadata = json.loads((root / "source/paper_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    sources = metadata.get("external_sources") if isinstance(metadata, dict) else None
    checks = []
    for source in sources if isinstance(sources, list) else []:
        path = source.get("path") if isinstance(source, dict) else None
        if not isinstance(path, str) or not path.lower().endswith(".qsf") or not (job / path).is_file():
            continue
        try:
            report = read_qsf(job / path)
        except Exception:  # an export the reader cannot follow is not a smoke-test failure
            continue
        checks += [dict(check, source=path) for check in correct_answer_checks(report)]
    return checks


def unkeyed_checks(checks: list[dict], asked: list[dict]) -> list[str]:
    """Asked questions that are a survey's correct-answer check but carry no key (or a key it refuses)."""
    problems = []
    for question in asked:
        if question.get("kind") != "choice":
            continue
        qid = str(question.get("id", ""))
        options = {_plain(o) for o in question.get("options") or []}
        text = _plain(question.get("text"))
        matched = []
        for check in checks:
            by_id = qid in (check.get("qid"), check.get("tag")) or bool(
                check.get("qid") and re.search(rf"(?<![A-Za-z0-9]){re.escape(str(check['qid']))}(?![0-9])", qid))
            same_options = len(options) > 1 and options == {_plain(o) for o in check.get("options") or []}
            wording = _plain(check.get("text"))
            same_text = not text or not wording or wording in text or text in wording
            if by_id or (same_options and same_text):
                matched.append((by_id, check))
        if not matched:
            continue
        # The survey question with this id, when the package kept the id; else every question it reproduces.
        matched = [c for strong, c in matched if strong] or [c for _, c in matched]
        names = ", ".join(dict.fromkeys(f"{c['qid']}" + (f" ({c['tag']})" if c.get("tag") else "") for c in matched))
        correct = list(dict.fromkeys(answer for c in matched for answer in c.get("correct") or []))
        key = question.get("key")
        if key in (None, [], ""):
            problems.append(
                f"question {qid} is the survey's comprehension or attention check {names}: the survey "
                f"(source {matched[0]['source']}) accepts only {' or '.join(repr(a) for a in correct[:4])}, even "
                f"though its response is {matched[0].get('response') or 'not forced'}. Ask it with key= set to "
                "the correct option and apply the same rule as the other checks")
            continue
        keys = key if isinstance(key, list) else [key]
        if correct and not {_plain(k) for k in keys} <= {_plain(a) for a in correct}:
            problems.append(f"question {qid}: its key {keys!r} is not an answer the survey accepts for {names} "
                            f"({' or '.join(repr(a) for a in correct[:4])})")
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
    try:
        key_problems = unkeyed_checks(survey_checks(root, root.parent.parent), report.get("asked") or [])
    except Exception as exc:  # the check reads the package's own survey files; a crash is reported, not raised
        key_problems = [f"the survey's correct-answer checks could not be compared ({type(exc).__name__}: {exc})"]
    report["problems"] = problems + list(report.get("problems", [])) + key_problems
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
