"""Shared reply reading (task/replies.py), the answer-aware smoke test, and build hygiene."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from package_fixture import ADAPTER, REPLIES, write_runnable

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replies_under_test", REPLIES)
replies = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replies
spec.loader.exec_module(replies)
Question, ask, parse = replies.Question, replies.ask, replies.parse

from agent_pipeline import smoke_test  # noqa: E402
from agent_pipeline.build_prompt import scrub_signed_links  # noqa: E402
from agent_pipeline.check_sources import check_package_sources  # noqa: E402
from agent_pipeline.finalize_job import finalize  # noqa: E402

NICE = Question("tog", options=["Be nice", "Be fair"])


# --------------------------------------------------------------------------- parsing

@pytest.mark.parametrize("reply,code", [
    ("1", "1"), ("1.", "1"), ("1)", "1"), ("(1)", "1"), ("1. Be nice", "1"), ("Be nice", "1"), ("be NICE", "1"),
    ("**2**", "2"), ("Answer: 2", "2"), ("My answer is 2.", "2"), ("I choose Be fair.", "2"),
    ("I pick 2 because it is fair", "2"), ("2 - Be fair", "2"), ("Option 2: Be fair", "2"), ("  2\n", "2"),
])
def test_readable_replies(reply: str, code: str) -> None:
    answer = parse(reply, NICE)
    assert answer["status"] == "ok" and answer["code"] == code, answer


@pytest.mark.parametrize("reply,status", [
    ("1 or 2", "ambiguous"), ("1. Be fair", "ambiguous"), ("I would pick 1, maybe 2", "ambiguous"),
    ("", "unparsed"), ("I am not sure", "unparsed"), ("3", "unparsed"),
    ("I would not be nice", "unparsed"),  # a label inside a sentence is not guessed (negation)
])
def test_unreadable_replies_are_not_guessed(reply: str, status: str) -> None:
    assert parse(reply, NICE)["status"] == status


def test_label_inside_a_longer_label_and_curly_quotes() -> None:
    q = Question("pd", options=["Hand over", "Don’t hand over"])
    assert parse("Don't hand over", q)["code"] == "2"
    assert parse("2. Don't hand over", q)["code"] == "2"
    assert parse("1. Don't hand over", q)["status"] == "ambiguous"
    assert parse("hand over", q)["code"] == "1"


def test_printed_scales_letters_numbers_and_sets() -> None:
    scale = Question("s", options=["-3 (incorrect)", "-2", "-1", "0", "1 (correct)", "2", "3 (correct, very)"])
    assert scale.codes == ["-3", "-2", "-1", "0", "1", "2", "3"]
    assert parse("1", scale)["label"] == "1 (correct)"
    assert parse("-3", scale)["index"] == 0
    assert "1. -3" not in replies.render_options(scale)
    give = Question("give", options=[str(n) for n in range(11)])
    assert parse("I give 5", give)["label"] == "5"
    letters = Question("l", options=["Cooperate", "Defect"], codes=["A", "B"])
    assert [parse(r, letters).get("code") for r in ("A", "b", "B)", "option A", "Defect")] == ["A", "B", "B", "A", "B"]
    assert parse("I think a lot", letters)["status"] == "unparsed"  # the word "a" is not option A
    tick = Question("m", options=["Cinema", "Farmer", "US600"], multiple=True)
    assert parse("1, 3", tick)["codes"] == ["1", "3"]
    assert parse("Cinema and US600", tick)["codes"] == ["1", "3"]
    assert parse("1, 7", tick)["status"] == "unparsed"
    number = Question("n", kind="number", minimum=0, maximum=10)
    assert parse("I give 5 tokens", number)["value"] == 5
    assert parse("11", number)["status"] == "out_of_range"
    assert parse("3 or 4", number)["status"] == "ambiguous"


def test_duplicate_scale_labels_are_ambiguous_by_label_but_codes_work() -> None:
    bipolar = Question("pref", options=["4 (Rock)", "3", "2", "1", "0 (Indifferent)", "1", "2", "3", "4 (Theatre)"],
                       codes=[str(n) for n in range(1, 10)])
    assert parse("3", bipolar)["status"] == "ambiguous"     # code 3, or the label 3 on either side?
    assert parse("9", bipolar)["label"] == "4 (Theatre)"
    assert parse("5. 0 (Indifferent)", bipolar)["index"] == 4


def test_several_questions_on_one_screen() -> None:
    qs = [Question("a", text="Who decided?", options=["Physician", "Patient"]),
          Question("b", text="Rating", options=["1", "2", "3"]),
          Question("c", text="Comments?", kind="text", required=False)]
    text = replies.render_questions(qs)
    assert "Q1. Who decided?" in text and "Q1: <your answer>" in text and "QID" not in text
    page = ask(lambda prompt: "Q1: Patient\nQ2: 3\nQ3: none", text, qs)
    assert page.ok and page.value("a") == "Patient" and page.value("b") == "3" and page.value("c") == "none"
    page = ask(lambda prompt: '{"Q1": "1", "Q2": "2"}', text, qs)
    assert page.ok and page.value("a") == "Physician"
    assert [u["question"] for u in page.unparsed] == ["c"] and page.unparsed[0]["required"] is False


def test_optional_answer_never_drops_a_participant_and_retry_keeps_read_answers() -> None:
    qs = [Question("rating", text="Rate", options=["1", "2", "3"]),
          Question("wrap", text="What was this study about?", options=["A", "B"], codes=["x", "y"], required=False)]
    calls = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return "Q1: 2\nQ2: no idea"
    page = ask(llm, "screen", qs)
    assert page.ok and len(calls) == 1 and page.value("rating") == "2"
    assert page.unparsed == [{"question": "wrap", "required": False, "status": "unparsed", "replies": ["Q1: 2\nQ2: no idea"]}]

    answers = iter(["hmm", "Q1: 3"])
    page = ask(lambda prompt: next(answers), "screen", qs[:1] + [Question("r2", options=["1", "2"])], retries=1)
    assert not page.ok and page.answers["rating"]["status"] == "ok"
    replies_seen = iter(["hmm", "2"])
    page = ask(lambda prompt: next(replies_seen), "screen", [qs[0]])
    assert page.ok and page.value("rating") == "2" and len(page.replies) == 2


def test_key_must_be_an_option() -> None:
    with pytest.raises(ValueError):
        Question("k", options=["Yes", "No"], key="Maybe").key_indices()


# --------------------------------------------------------------------------- smoke test

def make_package(tmp_path: Path, adapter: str | None = None, evaluation: str | None = None) -> Path:
    root = tmp_path / "package" / "paper"
    write_runnable(root)
    if adapter is not None:
        (root / "task/adapter.py").write_text(adapter)
    if evaluation is not None:
        (root / "evaluation/evaluation.py").write_text(evaluation)
    return root


def test_runnable_package_passes(tmp_path: Path) -> None:
    report = smoke_test.run(make_package(tmp_path))
    assert report["ok"], report["problems"]
    assert set(report["arms"]) == {"control", "treatment"}
    assert all(arm["completed"] >= 1 for arm in report["arms"].values())
    assert report["direct_model_calls"] == 0


def test_constant_stub_adapter_fails(tmp_path: Path) -> None:
    # The round-1 pattern: an own parser, every check failed, and nothing reached evaluate().
    adapter = (
        "def run_sessions(llm, seed, n, arms=None):\n"
        "    out = []\n"
        "    for arm in arms or ['control', 'treatment']:\n"
        "        for i in range(n):\n"
        "            reply = llm('What is 2 + 2?\\n1. 3\\n2. 4')\n"
        "            out.append({'arm': arm, 'status': 'completed' if reply.strip() == '2' and False else 'excluded'})\n"
        "    return out\n")
    report = smoke_test.run(make_package(tmp_path, adapter=adapter))
    assert not report["ok"]
    text = "\n".join(report["problems"])
    assert "replies.ask()" in text and "no session completed" in text
    assert report["arms"]["control"]["statuses"] == {"excluded": 32}  # excluded is never counted as completed


def test_wrong_key_in_adapter_logic_is_caught(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    adapter = ADAPTER.replace('elif page.value("check") != "4":', 'elif page.value("check") != "3":')
    (root / "task/adapter.py").write_text(adapter)
    report = smoke_test.run(root)
    assert not report["ok"] and "no session completed" in "\n".join(report["problems"])


@pytest.mark.parametrize("evaluation,message", [
    ("def evaluate(sessions):\n    return {}\n", "non-empty dict"),
    ("def evaluate(sessions):\n    raise KeyError('dg_give')\n", "evaluate() raised KeyError"),
])
def test_evaluate_must_return_results(tmp_path: Path, evaluation: str, message: str) -> None:
    report = smoke_test.run(make_package(tmp_path, evaluation=evaluation))
    assert not report["ok"] and message in "\n".join(report["problems"])


def test_edited_replies_copy_and_missing_arm_fail(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    (root / "task/replies.py").write_text(REPLIES.read_text() + "\n# tweaked\n")
    report = smoke_test.run(root)
    assert any("differs" in p for p in report["problems"])
    task = json.loads((root / "task/task.json").read_text())
    task["conditions"].append({"id": "extra"})
    (root / "task/task.json").write_text(json.dumps(task))
    shutil.copyfile(REPLIES, root / "task/replies.py")
    report = smoke_test.run(root)
    assert any("arm extra" in p for p in report["problems"])
    task["conditions"][-1]["status"] = "blocked"
    (root / "task/task.json").write_text(json.dumps(task))
    assert smoke_test.run(root)["ok"]


def test_adapter_smoke_flag_and_validator_run_the_same_check(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    result = subprocess.run([sys.executable, str(root / "task/adapter.py"), "--smoke-test"], cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["ok"] is True


# --------------------------------------------------------------------------- build hygiene (F8)

def test_signed_links_are_removed_from_job_json(tmp_path: Path) -> None:
    (tmp_path / "job.json").write_text(json.dumps({
        "id": "j", "paperUrl": "https://store.example/sign/paper.pdf?token=abc",
        "openMaterialsUrl": "https://store.example/sign/m.zip?token=def", "osfUrl": "https://osf.io/abcde/"}))
    scrub_signed_links(tmp_path)
    data = json.loads((tmp_path / "job.json").read_text())
    assert "paperUrl" not in data and "openMaterialsUrl" not in data and data["osfUrl"] == "https://osf.io/abcde/"


def test_workflow_drops_paper_link_and_bytecode_before_committing() -> None:
    workflow = (ROOT / ".github/workflows/run-humanstudy-pipeline.yml").read_text()
    assert 'data.pop("paperUrl", None)' in workflow
    assert "__pycache__" in workflow


def test_study_zip_has_no_bytecode(tmp_path: Path) -> None:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    (root / "task/__pycache__").mkdir(parents=True)
    (root / "task/__pycache__/adapter.cpython-311.pyc").write_bytes(b"\0")
    (root / "task/stray.pyc").write_bytes(b"\0")
    (root / "study.json").write_text("{}\n")
    (job / "job.json").write_text(json.dumps({"status": "running"}))
    validator = tmp_path / "validator.py"
    validator.write_text("raise SystemExit(0)\n")
    finalize(job, validator)
    names = zipfile.ZipFile(job / "output/study.zip").namelist()
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")], names
    assert not (root / "task/__pycache__").exists()


def test_every_downloaded_file_must_be_recorded(tmp_path: Path) -> None:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    (root / "source").mkdir(parents=True)
    (root / "source/paper_metadata.json").write_text(json.dumps({"external_sources": [], "material_searches": []}))
    (job / "input/external").mkdir(parents=True)
    (job / "input/external/osf-print-version.pdf").write_bytes(b"%PDF-1.4")
    (job / "input/external/fetch_log.json").write_text(json.dumps([{
        "url": "https://osf.io/download/abcde/", "final_url": "https://osf.io/download/abcde/",
        "path": "input/external/osf-print-version.pdf", "sha256": "0" * 64}]))
    problems = check_package_sources(root, job)
    assert any("osf-print-version.pdf was downloaded" in p for p in problems), problems


# --------------------------------------------------------------------------- review round 1

@pytest.mark.parametrize("reply", ["Not 1", "I would not choose 1", "I refuse to pick 1", "Not option 1",
                                   "1 is not what I'd pick", "somewhere around 1-2"])
def test_negations_and_ranges_are_never_read_as_an_option(reply: str) -> None:
    assert parse(reply, NICE)["status"] == "ambiguous"


def test_negations_in_numbers_scales_and_sets() -> None:
    assert parse("Not 50", Question("n", kind="number", minimum=0, maximum=100))["status"] == "ambiguous"
    scale = Question("s", options=[str(n) for n in range(1, 8)])
    assert parse("3-4", scale)["status"] == "ambiguous"
    assert parse("1 and not 2", Question("m", options=["Red", "Green"], multiple=True))["status"] == "ambiguous"
    yes_no = Question("c", options=["Yes", "No"])
    assert parse("No", yes_no)["code"] == "2" and parse("2. No", yes_no)["code"] == "2"


def test_scripted_sessions_and_arm_blind_evaluation_fail(tmp_path: Path) -> None:
    scripted = ("def run_sessions(llm, seed, n, arms=None):\n"
                "    return [{'arm': a, 'status': 'completed'} for a in (arms or ['control', 'treatment'])"
                " for _ in range(n)]\n")
    report = smoke_test.run(make_package(tmp_path, adapter=scripted))
    assert any("without asking the model" in p for p in report["problems"])
    one_arm = ("def evaluate(sessions):\n"
               "    rows = [s for s in sessions if s['arm'] == 'control']\n"
               "    return {'n': len(rows), 'mean': 1 / len(rows)}\n")
    report = smoke_test.run(make_package(tmp_path / "b", evaluation=one_arm))
    assert any("arm treatment: evaluate() on this arm's sessions alone raised" in p for p in report["problems"])


def test_stub_answers_a_numeric_key_list() -> None:
    stub = replies.StubParticipant(0)
    q = Question("give", kind="number", minimum=0, maximum=10, key=[5, 6])
    assert parse(stub.answer_for("x", [q]), q)["value"] == 5


def test_unused_downloads_are_listed_with_a_reason(tmp_path: Path) -> None:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    (root / "source").mkdir(parents=True)
    (job / "input/external").mkdir(parents=True)
    (job / "input/external/fetch_log.json").write_text(json.dumps([{"path": "input/external/pilot.pdf"}]))
    meta = {"external_sources": [], "material_searches": [],
            "unused_downloads": [{"path": "input/external/pilot.pdf", "reason": "pilot, not the final survey"}]}
    (root / "source/paper_metadata.json").write_text(json.dumps(meta))
    assert not [p for p in check_package_sources(root, job) if "pilot.pdf" in p or "unused_downloads" in p]
    meta["unused_downloads"][0]["reason"] = ""
    (root / "source/paper_metadata.json").write_text(json.dumps(meta))
    assert any("unused_downloads[0]" in p for p in check_package_sources(root, job))
