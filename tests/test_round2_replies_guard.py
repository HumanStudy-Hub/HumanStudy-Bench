"""Round-2 build fixes: option numbers, prompt-reading smoke stub, guard cwd, survey keys, signed links, licences.

Option sets are taken from the round-2 packages' own survey files (Xiao 2021 status quo bias, Aiyer 2023 outcome
bias) and the round-1 Petrov 2023 sunk-cost survey.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from package_fixture import REPLIES, write_runnable

spec = importlib.util.spec_from_file_location("replies_round2", REPLIES)
replies = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replies
spec.loader.exec_module(replies)
Question, parse, render_options = replies.Question, replies.parse, replies.render_options

from agent_pipeline import fetch_public, network_policy, smoke_test  # noqa: E402
from agent_pipeline.check_sources import check_package_sources  # noqa: E402
from agent_pipeline.qsf_flow import correct_answer_checks, read_survey  # noqa: E402

XIAO_QUIZ4 = ["70% for auto safety and 30% for highway safety", "60% for auto safety and 40% for highway safety",
              "30% for auto safety and 70% for highway safety", "50% for auto safety and 50% for highway safety"]
XIAO_SQ = ["Allocate 70% to auto safety and 30% to highway safety", "Allocate 60% to auto safety and 40% to highway safety"]
XIAO_SATISFIED = ["0\nExtremely unsatisfied", "1", "2", "3\nNeutral", "4", "5", "6\nVery satisfied"]
XIAO_LIKERT = ["1 - Not at all", "2", "3", "4", "5 - Very much"]
AIYER_PERCENT = ["8%", "65%", "70%"]
AIYER_SCALE = ["3 (clearly correct, and the opposite decision would be inexcusable)",
               "2 (correct, all things considered)", "1 (correct, but the opposite would be reasonable too)",
               "0 (the decision and its opposite are equally good)", "-1 (incorrect, but not unreasonable)",
               "-2 (incorrect, all things considered)", "-3 (incorrect and inexcusable)"]
AIYER_RESPONSIBILITY = ["1\n(no responsibility)", "2\n(very low level of responsibility)", "3\n(low level)",
                        "4\n(moderate level)", "5\n(high level)", "6\n(very high level)", "7\n(full responsibility)"]
PETROV_SPEND = ["Nothing", "US$90", "30 hours", "US$30", "90 hours"]
PETROV_HOURS = ["30 hours", "90 hours"]
PETROV_BIPOLAR = ["4", "3", "2", "1", "0", "1", "2", "3"]


# --------------------------------------------------------------------------- G7: option numbers


@pytest.mark.parametrize("options,codes", [
    (XIAO_QUIZ4, ["1", "2", "3", "4"]),          # percentages are never codes
    (XIAO_SQ, ["1", "2"]),
    (AIYER_PERCENT, ["1", "2", "3"]),
    (PETROV_HOURS, ["1", "2"]),                   # amounts are never codes
    (["10 hours", "30 hours", "50 hours"], ["1", "2", "3"]),
    (["200 people will be saved", "600 people will die"], ["1", "2"]),
    (PETROV_SPEND, ["1", "2", "3", "4", "5"]),
    (PETROV_BIPOLAR, [str(n) for n in range(1, 9)]),  # a number printed twice cannot be a code
    (XIAO_SATISFIED, ["0", "1", "2", "3", "4", "5", "6"]),  # printed scales keep their own numbers
    (XIAO_LIKERT, ["1", "2", "3", "4", "5"]),
    (AIYER_SCALE, ["3", "2", "1", "0", "-1", "-2", "-3"]),
    (AIYER_RESPONSIBILITY, [str(n) for n in range(1, 8)]),
    (["1 Strongly disagree", "2", "3", "4 Strongly agree"], ["1", "2", "3", "4"]),
    ([str(n) for n in range(11)], [str(n) for n in range(11)]),
])
def test_default_codes(options: list, codes: list) -> None:
    assert Question("q", options=options).codes == codes


@pytest.mark.parametrize("options", [XIAO_QUIZ4, XIAO_SQ, AIYER_PERCENT, PETROV_HOURS, PETROV_SPEND, PETROV_BIPOLAR,
                                     XIAO_SATISFIED, XIAO_LIKERT, AIYER_SCALE, AIYER_RESPONSIBILITY])
def test_every_option_shows_the_number_that_selects_it(options: list) -> None:
    q = Question("q", options=options)
    for index, code in enumerate(q.codes):
        line = q.option_line(index)
        assert line.startswith(code) and not line[len(code):len(code) + 1].isdigit(), (line, code)
        assert parse(line, q)["index"] == index
        if options is not PETROV_BIPOLAR:  # there "3" is option 3 and the label of options 2 and 8: asked again
            assert parse(code, q)["index"] == index
            assert parse(f"Answer: {code}", q)["index"] == index


def test_xiao_quiz4_reply_with_the_number_is_read() -> None:
    q = Question("QID80", text="How is the budget currently allocated?", options=XIAO_QUIZ4)
    text = replies.render_questions([q])
    assert "1. 70% for auto safety" in text and "Reply with the number of the option you choose." in text
    for reply in ("1", "1.", "Answer: 1", "1. 70% for auto safety and 30% for highway safety",
                  "70% for auto safety and 30% for highway safety"):
        assert parse(reply, q)["index"] == 0, reply
    assert parse("70", q)["status"] == "unparsed"  # 70 is not an option number any more


def test_explicit_codes_are_printed_even_when_a_label_starts_with_an_amount() -> None:
    q = Question("q", options=XIAO_QUIZ4, codes=["70", "60", "30", "50"])
    assert render_options(q).split("\n")[0] == "70. 70% for auto safety and 30% for highway safety"
    assert parse("70", q)["index"] == 0
    survey_codes = Question("s", options=["1 (low)", "2", "3 (high)"], codes=["1", "2", "3"])
    assert render_options(survey_codes) == "1 (low)\n2\n3 (high)"


# --------------------------------------------------------------------------- G7: the stub reads the prompt

UNNUMBERED_ADAPTER = '''import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replies import Question, ask, smoke_main  # noqa: E402

OPTIONS = {options!r}


def run_sessions(llm, seed, n, arms=None):
    sessions = []
    for arm in [a for a in ["control", "treatment"] if arms is None or a in arms]:
        for index in range(n):
            q = Question("budget", text="How is the budget allocated?", options=OPTIONS, codes={codes!r})
            screen = q.text + "\\n" + "\\n".join(OPTIONS) + "\\nReply with the number of the option you choose."
            page = ask(llm, screen, [q])
            sessions.append({{"arm": arm, "status": "completed" if page.ok else "unparsed",
                              "answer": page.value("budget")}})
    return sessions


if __name__ == "__main__":
    raise SystemExit(smoke_main(Path(__file__)) if "--smoke-test" in sys.argv else 1)
'''


def _package(tmp_path: Path, adapter: str | None = None) -> Path:
    root = tmp_path / "job" / "package" / "paper"
    write_runnable(root)
    if adapter is not None:
        (root / "task" / "adapter.py").write_text(adapter)
    return root


def test_stub_fails_options_printed_without_their_numbers(tmp_path: Path) -> None:
    # The round-2 Xiao screen: labels printed as they are, codes 70/60/30/50 taken from the labels.
    adapter = UNNUMBERED_ADAPTER.format(options=XIAO_QUIZ4, codes=["70", "60", "30", "50"])
    report = smoke_test.run(_package(tmp_path, adapter))
    assert not report["ok"]
    problem = next(p for p in report["problems"] if p.startswith("question budget"))
    assert "type '" in problem and "unparsed" in problem and "70, 60, 30, 50" in problem


def test_stub_passes_the_same_screen_once_numbers_are_printed(tmp_path: Path) -> None:
    adapter = UNNUMBERED_ADAPTER.format(options=XIAO_QUIZ4, codes=None).replace(
        '"\\n".join(OPTIONS)', 'replies_render(q)').replace(
        "from replies import Question, ask, smoke_main", "from replies import Question, ask, smoke_main, "
        "render_options as replies_render")
    report = smoke_test.run(_package(tmp_path, adapter))
    assert report["ok"], report["problems"]
    assert set(report["reader"]) == {"control", "treatment"}


def test_stub_replies_with_what_the_screen_prints_not_with_codes() -> None:
    stub = replies.StubParticipant(seed=3, noise=0.0, use_keys=False)
    letters = Question("l", text="Which?", options=["Cooperate", "Defect"], codes=["A", "B"])
    prompt = "Which?\nA. Cooperate\nB. Defect\nReply with the letter of the option you choose."
    reply = stub.answer_for(prompt, [letters])
    assert parse(reply, letters)["status"] == "ok" and not stub.misreads
    # A bipolar scale printing "3" twice: the number before the chosen copy is typed; a bare number that is
    # also a label is ambiguous (asked again, reported), never a misread, and the stub's own reply is readable.
    bipolar = Question("b", options=PETROV_BIPOLAR)
    prompt = replies.render_questions([bipolar])
    for seed in range(20):
        stub = replies.StubParticipant(seed=seed, noise=0.0, use_keys=False)
        reply = stub.answer_for(prompt, [bipolar])
        assert parse(reply, bipolar)["status"] == "ok" and not stub.misreads, reply
    stubs = [replies.StubParticipant(seed=s, noise=0.0, use_keys=False) for s in range(20)]
    for stub in stubs:
        stub.answer_for(prompt, [bipolar])
    assert any(stub.ambiguous == ["b"] for stub in stubs)


def test_stub_uses_keys_only_in_the_keyed_pass() -> None:
    check = Question("check", text="What is 2 + 2?", options=["3", "4"], key="4")
    prompt = replies.render_questions([check])
    keyed = {replies.StubParticipant(seed=s, noise=0.0).answer_for(prompt, [check]) for s in range(10)}
    blind = {parse(replies.StubParticipant(seed=s, noise=0.0, use_keys=False).answer_for(prompt, [check]),
                   check)["label"] for s in range(30)}
    assert all(parse(r, check)["label"] == "4" for r in keyed)
    assert blind == {"3", "4"}


def test_stub_reads_each_question_block_of_a_multi_question_screen() -> None:
    qs = [Question("a", text="First?", options=["Yes", "No"]), Question("b", text="Second?", options=["No", "Yes"])]
    prompt = replies.render_questions(qs)
    for seed in range(20):
        stub = replies.StubParticipant(seed=seed, noise=0.0, use_keys=False)
        page = replies.ask(stub, prompt, qs)
        assert page.ok and not stub.misreads


# --------------------------------------------------------------------------- G8: guard and inline cd


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "HumanStudy-Bench"
    (root / "agent_pipeline").mkdir(parents=True)
    (root / "agent_pipeline" / "fetch_public.py").write_text("# helper\n")
    monkeypatch.setattr(network_policy, "REPO_ROOT", root)
    return root


HOSTS = ["osf.io", "files.osf.io", "api.osf.io"]


def _guard(command: str, cwd: str) -> str | None:
    return network_policy.check_tool_call("Bash", {"command": command}, HOSTS, cwd=cwd)


def test_aiyer_run_commands_after_an_inline_cd(repo: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    job = f"{repo}/job-storage/jobs/studio-55ec"
    # Run 41784656634: the session had cd'd to /tmp/work; these two were refused and the licence never checked.
    allowed = [
        f"cd {repo} && J={job} && python agent_pipeline/fetch_public.py --job $J "
        "https://osf.io/download/5bead3eab378ff0018a947dc/ --name survey-knjhu.qsf 2>&1 | tail -3; "
        f"sha256sum $J/input/external/survey-knjhu.qsf",
        f"cd {repo} && python agent_pipeline/fetch_public.py --job {job} --name survey-knjhu.qsf "
        '"https://osf.io/download/5bead3eab378ff0018a947dc/"',
        f"cd '{repo}' && python3 agent_pipeline/fetch_public.py --job {job} https://osf.io/download/vfw38/",
        f"cd {repo}/agent_pipeline && python3 fetch_public.py --job {job} https://osf.io/download/vfw38/",
        f"cd {repo} && cd agent_pipeline && python3 ./fetch_public.py --job {job} https://osf.io/download/vfw38/",
        f"python {repo}/agent_pipeline/fetch_public.py --job {job} https://osf.io/download/vfw38/",
        f"python agent_pipeline/fetch_public.py --job {job} --licence https://osf.io/knjhu/",
    ]
    for command in allowed[:-1]:
        assert _guard(command, str(work)) is None, command
    assert _guard(allowed[-1], str(repo)) is None
    # The first refused command had no cd: the relative path really did not name the helper.
    first = (f"J={job}; python agent_pipeline/fetch_public.py --job $J "
             "https://osf.io/download/5bead3eab378ff0018a947dc/ --name survey-knjhu.qsf 2>&1 | tail -3")
    reason = _guard(first, str(work))
    assert reason and str(repo / "agent_pipeline" / "fetch_public.py") in reason and "absolute path" in reason


def test_guard_does_not_follow_a_cd_it_cannot_verify(repo: Path, tmp_path: Path) -> None:
    evil = tmp_path / "evil"
    (evil / "agent_pipeline").mkdir(parents=True)
    (evil / "agent_pipeline" / "fetch_public.py").write_text("import urllib.request\n")
    url = "https://osf.io/download/vfw38/"
    refused = [
        f"cd {evil} && python agent_pipeline/fetch_public.py --job j {url}",           # not the helper
        f"cd {repo} && cd {evil} && python agent_pipeline/fetch_public.py --job j {url}",
        f"cd {evil} || cd {repo} && python agent_pipeline/fetch_public.py --job j {url}",
        f"cd {repo}; cd {evil}; python agent_pipeline/fetch_public.py --job j {url}",
        f"(cd {repo}) && python agent_pipeline/fetch_public.py --job j {url}",          # a subshell's cd
        f"cd {repo} | cat; python agent_pipeline/fetch_public.py --job j {url}",
        f"R={repo}; cd $R && python agent_pipeline/fetch_public.py --job j {url}",     # a variable
        f"CDPATH={repo} cd agent_pipeline && python fetch_public.py --job j {url}",
        f"cd {repo} && popd && python agent_pipeline/fetch_public.py --job j {url}",
        f"cd {repo} && curl -sL {url} -o x.qsf",                                       # cd never excuses curl
        f"cd {repo} && python agent_pipeline/fetch_public.py --job j https://evil.example.com/x.qsf",
    ]
    for command in refused:
        assert _guard(command, str(evil)) is not None, command
    # From the session's own directory, a later cd makes the relative path unknown.
    assert _guard(f"cd /tmp; python agent_pipeline/fetch_public.py --job j {url}", str(repo)) is not None
    assert _guard(f"python agent_pipeline/fetch_public.py --job j {url}", str(repo)) is None


def test_guard_reads_scripts_where_the_inline_cd_put_them(repo: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "get.py").write_text("import urllib.request\nurllib.request.urlopen('https://evil.example.com/')\n")
    assert _guard(f"cd {work} && python get.py", str(repo)) is not None
    assert _guard(f"cd {work} && python3 -c 'print(1)'", str(repo)) is None


# --------------------------------------------------------------------------- G9: survey checks need keys

def _mc(qid: str, text: str, choices: list, correct: str | None, force: str) -> dict:
    validation = {"Settings": {"ForceResponse": force, "Type": "None"}}
    if correct is not None:
        index = str(choices.index(correct) + 1)
        validation = {"Settings": {"ForceResponse": force, "ForceResponseType": "ON", "Type": "CustomValidation",
                                   "CustomValidation": {"Logic": {"0": {"0": {
                                       "LogicType": "Question", "QuestionID": qid, "Operator": "Selected",
                                       "LeftOperand": f"q://{qid}/SelectableChoice/{index}", "Type": "Expression"},
                                       "Type": "If"}, "Type": "BooleanExpression"},
                                       "Message": {"description": "Wrong quiz answer"}}}}
    return {"QuestionID": qid, "DataExportTag": f"tag-{qid}", "QuestionType": "MC", "Selector": "SAVR",
            "QuestionText": text, "Choices": {str(i + 1): {"Display": c} for i, c in enumerate(choices)},
            "ChoiceOrder": [str(i + 1) for i in range(len(choices))], "Validation": validation}


def _qsf(questions: list) -> dict:
    elements = [{"Element": "SQ", "PrimaryAttribute": q["QuestionID"], "Payload": q} for q in questions]
    elements.append({"Element": "BL", "Payload": {"0": {"ID": "BL_1", "Type": "Standard", "Description": "Quiz",
                                                        "BlockElements": [{"Type": "Question", "QuestionID": q["QuestionID"]}
                                                                          for q in questions]}}})
    elements.append({"Element": "FL", "Payload": {"Type": "Root", "FlowID": "FL_1",
                                                  "Flow": [{"Type": "Block", "ID": "BL_1", "FlowID": "FL_2"}]}})
    return {"SurveyEntry": {"SurveyID": "SV_1", "SurveyName": "Fixture"}, "SurveyElements": elements}


XIAO_QSF = _qsf([
    _mc("QID80", "How is the budget currently allocated by the commission?", XIAO_QUIZ4, XIAO_QUIZ4[0], "OFF"),
    _mc("QID83", "How is the budget currently allocated by the commission?", XIAO_QUIZ4, XIAO_QUIZ4[2], "ON"),
    _mc("QID56", "Which allocation do you choose?", XIAO_SQ, None, "ON"),
])


def test_correct_answer_checks_include_checks_that_are_not_forced() -> None:
    checks = correct_answer_checks(read_survey(XIAO_QSF))
    assert [(c["qid"], c["response"], c["correct"]) for c in checks] == [
        ("QID80", "optional", [XIAO_QUIZ4[0]]), ("QID83", "forced", [XIAO_QUIZ4[2]])]


def _asked(qid: str, options: list, key=None, text="How is the budget currently allocated by the commission?"):
    return {"id": qid, "kind": "choice", "text": text, "options": options, "key": key}


def test_unkeyed_survey_check_is_a_problem() -> None:
    checks = [dict(c, source="input/external/s.qsf") for c in correct_answer_checks(read_survey(XIAO_QSF))]
    problems = smoke_test.unkeyed_checks(checks, [_asked("QID80", XIAO_QUIZ4), _asked("QID83", XIAO_QUIZ4, [XIAO_QUIZ4[2]]),
                                                  _asked("QID56", XIAO_SQ, text="Which allocation do you choose?")])
    assert len(problems) == 1 and problems[0].startswith("question QID80") and "optional" in problems[0]
    # Matched by wording and options when the package renamed the question.
    problems = smoke_test.unkeyed_checks(checks, [_asked("sc1-cond2-quiz4", XIAO_QUIZ4)])
    assert len(problems) == 1 and "QID80" in problems[0]
    # A key the survey does not accept.
    problems = smoke_test.unkeyed_checks(checks, [_asked("QID83", XIAO_QUIZ4, [XIAO_QUIZ4[1]])])
    assert problems and "not an answer the survey accepts" in problems[0]
    assert smoke_test.unkeyed_checks(checks, [_asked("QID80", XIAO_QUIZ4, [XIAO_QUIZ4[0]])]) == []


def test_smoke_test_reports_an_unkeyed_survey_check(tmp_path: Path) -> None:
    root = _package(tmp_path)
    job = root.parent.parent
    (job / "input/external").mkdir(parents=True)
    survey = _qsf([_mc("QID9", "What is 2 + 2?", ["3", "4"], "4", "OFF")])
    (job / "input/external/s.qsf").write_text(json.dumps(survey))
    (root / "source").mkdir()
    (root / "source/paper_metadata.json").write_text(json.dumps({"external_sources": [{"path": "input/external/s.qsf"}]}))
    assert smoke_test.run(root)["ok"]  # the fixture's check carries key="4"
    adapter = (root / "task/adapter.py").read_text().replace(', key="4")', ")")
    (root / "task/adapter.py").write_text(adapter.replace('page.value("check") != "4"', "False"))
    report = smoke_test.run(root)
    assert not report["ok"] and any(p.startswith("question check is the survey's") for p in report["problems"])


# --------------------------------------------------------------------------- G10: signed links

SIGNED = ("https://idvyorynw6xy.compat.objectstorage.us-ashburn-1.oraclecloud.com/cos-prod-osf-1-files/abc?"
          "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=K%2F20261103&X-Amz-Expires=60&X-Amz-Signature=deadbeef")


def test_public_url_and_strip() -> None:
    assert fetch_public.public_url(SIGNED) == SIGNED.split("?")[0]
    assert fetch_public.public_url("https://osf.io/download/3x2bv/?version=2") == "https://osf.io/download/3x2bv/?version=2"
    text = json.dumps({"final_url": SIGNED, "note": f"see ({SIGNED})"})
    clean = fetch_public.strip_signed_urls(text)
    assert "X-Amz" not in clean and json.loads(clean)["note"].endswith("abc)")


def test_scrub_job_folder_keeps_downloads_byte_for_byte(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "input/external").mkdir(parents=True)
    (job / "package/p/source").mkdir(parents=True)
    (job / "logs").mkdir()
    (job / "input/external/fetch_log.json").write_text(json.dumps([{"final_url": SIGNED}]))
    (job / "input/external/survey.txt").write_text(SIGNED)  # a downloaded file: never edited
    (job / "package/p/source/paper_metadata.json").write_text(json.dumps({"external_sources": [{"final_url": SIGNED}]}))
    (job / "logs/agent.log").write_text(f"Location: {SIGNED}\n")
    changed = fetch_public.scrub_signed_urls(job)
    assert sorted(changed) == ["input/external/fetch_log.json", "logs/agent.log", "package/p/source/paper_metadata.json"]
    assert (job / "input/external/survey.txt").read_text() == SIGNED
    assert "X-Amz" not in (job / "logs/agent.log").read_text()


def test_fetch_records_final_url_without_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = tmp_path / "job"
    job.mkdir()
    monkeypatch.setattr(fetch_public, "_hosts", lambda job: ["osf.io"])

    class Response:
        headers = {"Content-Disposition": 'attachment; filename="survey.qsf"', "Content-Type": "application/json"}

        def __init__(self):
            self.chunks = [b"{}"]

        def geturl(self):
            return SIGNED

        def read(self, size=-1):
            return self.chunks.pop() if self.chunks else b""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Opener:
        def open(self, request, timeout=None):
            return Response()

    monkeypatch.setattr(fetch_public.urllib.request, "build_opener", lambda *handlers: Opener())
    record = fetch_public.fetch(job, "https://osf.io/download/3x2bv/")
    assert record["final_url"] == SIGNED.split("?")[0]
    assert "X-Amz" not in (job / "input/external/fetch_log.json").read_text()


def _sourced_package(tmp_path: Path, licence: str = "none stated", use: str = "researcher-own-use-only") -> tuple:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    (root / "source").mkdir(parents=True)
    (job / "input/external").mkdir(parents=True)
    data = b'{"SurveyEntry": {}}'
    (job / "input/external/survey.qsf").write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    url = "https://osf.io/download/vfw38/"
    (job / "input/external/fetch_log.json").write_text(json.dumps([
        {"url": url, "final_url": SIGNED.split("?")[0], "path": "input/external/survey.qsf", "sha256": sha}]))
    meta = {"redistribution": "researcher-own-use-only",
            "material_searches": [{"where": "OSF", "query": "czha8", "url": "https://osf.io/czha8/", "date": "2026-11-03",
                                   "result": "survey"}],
            "external_sources": [{"id": "s", "url": url, "final_url": SIGNED.split("?")[0],
                                  "path": "input/external/survey.qsf", "sha256": sha, "retrieved": "2026-11-03",
                                  "version": "registration copy", "tie": "linked from the paper", "licence": licence,
                                  "licence_evidence": "OSF registration czha8: no licence field", "use": use,
                                  "participant_data": False}]}
    (root / "source/paper_metadata.json").write_text(json.dumps(meta))
    return job, root, sha


def test_validator_refuses_signed_links_in_the_package(tmp_path: Path) -> None:
    job, root, _ = _sourced_package(tmp_path)
    assert not [p for p in check_package_sources(root, job) if "signed storage link" in p]
    meta = json.loads((root / "source/paper_metadata.json").read_text())
    meta["external_sources"][0]["final_url"] = SIGNED
    (root / "source/paper_metadata.json").write_text(json.dumps(meta))
    assert any("source/paper_metadata.json holds a signed storage link" in p for p in check_package_sources(root, job))


def test_unlisted_download_is_a_problem(tmp_path: Path) -> None:
    job, root, _ = _sourced_package(tmp_path)
    log = json.loads((job / "input/external/fetch_log.json").read_text())
    log.append({"url": "https://osf.io/download/jrbvd/", "path": "input/external/README.txt", "sha256": "0" * 64})
    (job / "input/external/fetch_log.json").write_text(json.dumps(log))
    assert any("README.txt was downloaded" in p for p in check_package_sources(root, job))


# --------------------------------------------------------------------------- G8: licence from OSF metadata

def _osf_api(monkeypatch: pytest.MonkeyPatch, sha: str) -> None:
    records = {
        "https://api.osf.io/v2/files/5bead3eab378ff0018a947dc/": {"data": {
            "type": "files", "attributes": {"name": "outcome_bias.qsf", "guid": "u9dbp", "materialized_path": "/p.qsf",
                                            "extra": {"hashes": {"sha256": sha}}},
            "relationships": {"target": {"data": {"type": "nodes", "id": "knjhu"}}}}},
        "https://api.osf.io/v2/nodes/knjhu/?embed=license": {"data": {
            "attributes": {"title": "Outcome bias replication", "node_license": {"copyright_holders": [""]}},
            "embeds": {"license": {"data": {"attributes": {"name": "CC-By Attribution 4.0 International"}}}},
            "relationships": {}}},
        "https://api.osf.io/v2/files/czha8/": None,
        "https://api.osf.io/v2/nodes/czha8/?embed=license": None,
        "https://api.osf.io/v2/registrations/czha8/?embed=license": {"data": {
            "attributes": {"title": "Outcome bias (registration)"}, "embeds": {"license": {"errors": []}},
            "relationships": {"registered_from": {"data": {"id": "knjhu"}}}}},
    }
    monkeypatch.setattr(fetch_public, "_api_json", lambda job, url: records[url])


def test_licence_lookup_records_hash_and_licence_without_download(tmp_path: Path, monkeypatch) -> None:
    job, root, sha = _sourced_package(tmp_path)
    _osf_api(monkeypatch, sha)
    record = fetch_public.osf_licence(job, "https://osf.io/download/5bead3eab378ff0018a947dc/")
    assert record["licence"] == "CC-BY-4.0" and record["file"]["sha256"] == sha
    assert "knjhu" in record["licence_from"]
    registration = fetch_public.osf_licence(job, "https://osf.io/czha8/")
    assert registration["licence"] == "none stated" and registration["projects"][0]["registered_from"] == "knjhu"
    log = json.loads((job / "input/external/licence_log.json").read_text())
    assert len(log) == 2
    assert sorted(p.name for p in (job / "input/external").iterdir()) == ["fetch_log.json", "licence_log.json",
                                                                         "survey.qsf"]


def test_identical_file_in_an_open_project_carries_its_licence(tmp_path: Path, monkeypatch) -> None:
    job, root, sha = _sourced_package(tmp_path)
    assert not [p for p in check_package_sources(root, job) if "same SHA-256" in p]
    _osf_api(monkeypatch, sha)
    fetch_public.osf_licence(job, "https://osf.io/download/5bead3eab378ff0018a947dc/")
    problems = [p for p in check_package_sources(root, job) if "same SHA-256" in p]
    assert problems and "CC-BY-4.0" in problems[0] and "knjhu" in problems[0]
    meta = json.loads((root / "source/paper_metadata.json").read_text())
    meta["external_sources"][0].update(licence="CC-BY-4.0", use="open")
    meta.pop("redistribution")
    (root / "source/paper_metadata.json").write_text(json.dumps(meta))
    assert not [p for p in check_package_sources(root, job) if "same SHA-256" in p or "redistribution" in p]
