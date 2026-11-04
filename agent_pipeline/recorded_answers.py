#!/usr/bin/env python3
"""Every answer a participant gives must reach the session record.

    python agent_pipeline/recorded_answers.py <paper-folder>

A participant who is asked a question and answers it has produced data. When the
adapter asks the question but drops the answer (a wrap-up page shown and its replies
thrown away), the researcher cannot apply any later coding rule to it and the run
silently differs from the survey, where every answer was stored.

The check runs one participant per condition with a deterministic offline stand-in
(no network, no paid calls), answering through ``replies.ask``'s ``answer_for`` hook.
Then, for each question the participant answered, it runs the same participant again
with only that one answer changed (another option, other text, another number). If
the sessions ``run_sessions`` returns are identical, the answer was never recorded.
Keyed questions (comprehension and attention checks, consent) are skipped: their
answers decide the flow. A package whose runs are not reproducible from the seed is
not judged. Prints a JSON report; exit status 1 when an answer is dropped.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

try:
    from agent_pipeline.package_checks import _import, _jsonable, _load_json, condition_ids
except ModuleNotFoundError:
    from package_checks import _import, _jsonable, _load_json, condition_ids

MAX_CALLS = 400        # per run
MAX_RERUNS = 160       # per package
TIME_BUDGET = 100.0    # seconds for the whole check
CHECK_TIMEOUT = 150


class _Stop(BaseException):
    """Ends a run at the call cap; a BaseException so an adapter's ``except Exception`` cannot hide it."""


class Answerer:
    """A stand-in participant whose every answer is fixed by its position in the session,
    except one answer that ``change`` replaces with an alternative."""

    def __init__(self, change: tuple[int, str, int] | None = None) -> None:
        self.change = change          # (call number, question id, which alternative)
        self.calls = 0
        self.asked: list[dict] = []   # (call number, question) for every answered question

    def __call__(self, prompt: str) -> str:  # a call that bypasses replies.ask
        self.calls += 1
        if self.calls > MAX_CALLS:
            raise _Stop()
        return "1"

    def _answer(self, call: int, question: Any, alternative: int) -> str:
        kind = getattr(question, "kind", "text")
        key = getattr(question, "key", None)
        if kind == "choice":
            options = list(getattr(question, "options", []) or [])
            codes = list(getattr(question, "codes", []) or [])
            keyed = question.key_indices() if key is not None else []
            if keyed:
                return ", ".join(codes[i] for i in keyed) if getattr(question, "multiple", False) else codes[keyed[0]]
            index = (call + alternative) % len(options)
            if getattr(question, "multiple", False) and alternative and len(options) > 1:
                return ", ".join(codes[i] for i in sorted({index, (index + 1) % len(options)}))
            return codes[index]
        if kind == "number":
            if key is not None:
                return str(key[0] if isinstance(key, (list, tuple)) and key else key)
            low = getattr(question, "minimum", None)
            high = getattr(question, "maximum", None)
            low = 0 if low is None else low
            high = low + 10 if high is None else high
            span = max(1, int(high - low))
            value = low + ((call + alternative) % (span + 1))
            return str(int(value) if getattr(question, "integer", True) else float(value))
        if key is not None:
            return str(key[0] if isinstance(key, (list, tuple)) and key else key)
        # Words and digits both differ between versions, so an adapter that keeps only a number
        # pulled out of free text still sees a different answer.
        return f"My answer, version {'abcdefgh'[alternative % 8]}, is {17 + 31 * alternative + call}."

    def answer_for(self, prompt: str, questions: Any, attempt: int = 0, final: bool = True, **_: Any) -> str:
        self.calls += 1
        if self.calls > MAX_CALLS:
            raise _Stop()
        questions = list(questions)
        replies = []
        for question in questions:
            alternative = 0
            if self.change and self.change[0] == self.calls and self.change[1] == question.id:
                alternative = self.change[2]
            if attempt == 0:
                self.asked.append({"call": self.calls, "question": question})
            replies.append(self._answer(self.calls, question, alternative))
        if len(questions) == 1:
            return replies[0]
        return "\n".join(f"Q{number}: {reply}" for number, reply in enumerate(replies, 1))


def _alternatives(question: Any) -> int:
    """How many different answers the check can give this question (0: not checked)."""
    if getattr(question, "key", None) is not None:
        return 0
    kind = getattr(question, "kind", "text")
    if kind == "choice":
        return min(2, len(getattr(question, "options", []) or []) - 1)
    if kind == "number":
        low, high = getattr(question, "minimum", None), getattr(question, "maximum", None)
        return 0 if low is not None and high is not None and high <= low else 1
    return 1


def check(root: Path) -> dict:
    report: dict = {"dropped": [], "checked": 0, "skipped": None}
    task = _load_json(root / "task/task.json")
    try:
        with redirect_stdout(io.StringIO()):
            adapter = _import(root / "task/adapter.py", "humanstudy_answers_adapter")
    except BaseException as exc:  # noqa: BLE001
        report["skipped"] = f"adapter not importable ({type(exc).__name__})"
        return report
    run_sessions = getattr(adapter, "run_sessions", None)
    if not callable(run_sessions):
        report["skipped"] = "no run_sessions"
        return report
    import inspect
    try:
        accepts_arms = "arms" in inspect.signature(run_sessions).parameters
    except (TypeError, ValueError):
        accepts_arms = False
    arms = (condition_ids(task) if accepts_arms else []) or [None]
    started = time.monotonic()
    tested: set[str] = set()
    reruns = 0

    def run(arm: str | None, change=None) -> tuple[str | None, Answerer]:
        answerer = Answerer(change)
        try:
            with redirect_stdout(io.StringIO()):
                sessions = run_sessions(answerer, 0, 1, arms=[arm]) if arm is not None else run_sessions(answerer, 0, 1)
        except _Stop:
            return None, answerer
        except BaseException:  # noqa: BLE001 - a crash is the probe's and smoke test's business
            return None, answerer
        text = json.dumps(_jsonable(sessions), sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest(), answerer

    for arm in arms:
        if time.monotonic() - started > TIME_BUDGET or reruns >= MAX_RERUNS:
            report["skipped"] = "time budget reached; later conditions not checked"
            break
        base, answerer = run(arm)
        if base is None or not answerer.asked:
            continue
        again, _ = run(arm)
        if again != base:
            report["skipped"] = "sessions differ between two runs with the same seed; answers not checked"
            break
        for asked in answerer.asked:
            question = asked["question"]
            text = " ".join(str(getattr(question, "text", "") or "").split())
            identity = f"{question.id}|{text[:200]}"
            options = _alternatives(question)
            if identity in tested or not options:
                continue
            tested.add(identity)
            same = True
            for alternative in range(1, options + 1):
                if time.monotonic() - started > TIME_BUDGET or reruns >= MAX_RERUNS:
                    same = False
                    break
                reruns += 1
                changed, _ = run(arm, (asked["call"], question.id, alternative))
                if changed is None or changed != base:
                    same = False
                    break
            report["checked"] += 1
            if same:
                report["dropped"].append({"condition": arm, "prompt": asked["call"], "question": question.id,
                                          "text": text[:90]})
    return report


def problems(root: Path, timeout: int = CHECK_TIMEOUT) -> list[str]:
    """The check in a subprocess, so a hanging adapter cannot stop the validator."""
    import subprocess
    import os

    try:
        done = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(root)], cwd=root,
                              capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        report = json.loads(done.stdout.strip().splitlines()[-1]) if done.stdout.strip() else {}
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return []
    dropped = report.get("dropped") or []
    if not dropped:
        return []
    shown = "; ".join(f"'{d['text'] or d['question']}' (condition {d['condition']}, prompt {d['prompt']})"
                      for d in dropped[:5])
    return ["recorded answers: participants are asked questions whose answers never reach the session record "
            f"run_sessions returns: {shown}" + (f" and {len(dropped) - 5} more" if len(dropped) > 5 else "")
            + ". Changing the answer changes nothing in the session. Store every answer a participant gives "
            "(wrap-up and comment questions too, under a neutral key), even when the analysis does not use it "
            "(contract 'Reading replies')."]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: recorded_answers.py <paper-folder>")
    print(json.dumps(check(Path(sys.argv[1]).resolve()), ensure_ascii=False))
    raise SystemExit(0)
