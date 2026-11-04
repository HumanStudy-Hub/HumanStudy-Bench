"""A smallest runnable package for tests that need the package check to pass.

Two arms, one comprehension check with a key, one decision and one optional
comment, asked through the shared ``task/replies.py``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLIES = ROOT / "agent_pipeline" / "package_files" / "replies.py"

ADAPTER = '''import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replies import Question, ask, render_questions, smoke_main  # noqa: E402

ARMS = ["control", "treatment"]


def run_sessions(llm, seed, n, arms=None):
    sessions = []
    for arm in [a for a in ARMS if arms is None or a in arms]:
        for index in range(n):
            check = Question("check", text="What is 2 + 2?", options=["3", "4"], key="4")
            page = ask(llm, render_questions([check]), [check])
            session = {"arm": arm, "session_id": f"{arm}-{index}", "status": "completed", "unparsed": []}
            if not page.ok:
                session.update(status="unparsed", unparsed=page.unparsed)
            elif page.value("check") != "4":
                session.update(status="excluded", exclusion_reason="failed the check")
            else:
                questions = [Question("choice", text="Pick one.", options=["Left", "Right"]),
                             Question("comment", text="Any comments?", kind="text", required=False)]
                page = ask(llm, render_questions(questions), questions)
                session["unparsed"] = page.unparsed
                if not page.ok:
                    session["status"] = "unparsed"
                else:
                    session["choice"] = page.value("choice")
                    session["comment"] = page.value("comment")
            sessions.append(session)
    return sessions


if __name__ == "__main__":
    raise SystemExit(smoke_main(Path(__file__)) if "--smoke-test" in sys.argv else 1)
'''

EVALUATION = '''def evaluate(sessions):
    by_arm = {}
    for session in sessions:
        counts = by_arm.setdefault(session["arm"], {})
        counts[session["status"]] = counts.get(session["status"], 0) + 1
    return {"status_by_arm": by_arm}
'''


def write_runnable(root: Path) -> None:
    """Write a runnable adapter, evaluation, replies copy and task conditions into ``root``."""
    (root / "task").mkdir(parents=True, exist_ok=True)
    (root / "evaluation").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPLIES, root / "task" / "replies.py")
    (root / "task" / "adapter.py").write_text(ADAPTER)
    (root / "evaluation" / "evaluation.py").write_text(EVALUATION)
    task_path = root / "task" / "task.json"
    try:
        task = json.loads(task_path.read_text())
    except (OSError, ValueError):
        task = {}
    task = task if isinstance(task, dict) else {}
    task["conditions"] = [{"id": "control"}, {"id": "treatment"}]
    task_path.write_text(json.dumps(task, indent=2) + "\n")
