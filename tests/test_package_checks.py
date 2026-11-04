"""Package checks: participant memory, leaks, human results, analyses, researcher reply."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_pipeline import package_checks as pc
from agent_pipeline.estimate_run import estimate

ROOT = Path(__file__).resolve().parents[1]

SCENARIO = ("A 55-year-old man had a heart condition and had to stop working because of chest pain. "
            "A bypass operation would relieve his pain and increase his life expectancy.")
OUTCOME = "The operation was carried out and the patient recovered fully after a few weeks at home."
RATING = "Please evaluate the decision that was made about going ahead with the operation today."
IMPORTANCE = "Do you think outcomes should be taken into account when evaluating the quality of decisions?"

ADAPTER_HEAD = '''
import json, sys
SCREENS = [@SCREENS@]
ARMS = ["physician_success", "patient_failure"]

def smoke_llm(prompt):
    return "ANSWER: 1"
'''

MEMORYLESS = ADAPTER_HEAD + '''
def run_sessions(llm, seed, n, arms=None):
    out = []
    for arm in (arms or ARMS):
        for i in range(n):
            answers = [llm("You have only seen the screens above.\\nSCREEN:\\n" + s) for s in SCREENS]
            out.append({"arm": arm, "condition": arm, "answers": answers})
    return out
'''

WITH_MEMORY = ADAPTER_HEAD + '''
def run_sessions(llm, seed, n, arms=None):
    out = []
    for arm in (arms or ARMS):
        for i in range(n):
            history, answers = [], []
            for s in SCREENS:
                prompt = "Earlier screens and your answers:\\n" + "\\n".join(history) + "\\nCurrent screen:\\n" + s + "\\nq1:"
                reply = llm(prompt)
                history += [s, "Your answer: " + reply]
                answers.append(reply)
            out.append({"arm": arm, "condition": arm, "answers": answers})
    return out
'''

LEAKY = WITH_MEMORY.replace('"\\nCurrent screen:\\n" + s + "\\nq1:"', '"\\nCurrent screen [" + arm + "]:\\n" + s + "\\nQID14:"')

EVALUATION = '''
def evaluate(sessions):
    return {"outcome_effect": {"status": "not_ready", "reason": "fewer than two participants per condition"},
            "descriptives": {"n": len(sessions)}}
'''


def make_package(tmp_path: Path, adapter: str, *, task_extra: dict | None = None, results: list | None = None,
                 evaluation: str = EVALUATION, reply: str | None = None) -> tuple[Path, Path]:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    for folder in ("task", "evaluation", "materials", "source", "audit"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    screens = ", ".join(repr(text) for text in (SCENARIO + " " + OUTCOME, RATING, IMPORTANCE))
    (root / "task/adapter.py").write_text(adapter.replace("@SCREENS@", screens))
    (root / "evaluation/evaluation.py").write_text(evaluation)
    task = {"conditions": [{"id": "physician_success"}, {"id": "patient_failure"}]}
    task.update(task_extra or {})
    (root / "task/task.json").write_text(json.dumps(task))
    (root / "materials/materials.json").write_text(json.dumps({"screens": [
        {"id": "s1", "qsfId": "QID3", "text": SCENARIO + " " + OUTCOME}, {"id": "s2", "qsfId": "QID11", "text": RATING},
        {"id": "s3", "qsfId": "QID14", "text": IMPORTANCE}]}))
    nodes = [{"id": "res-1", "kind": "result", "fields": results or []}]
    (root / "study.json").write_text(json.dumps({"program": {"nodes": nodes}}))
    if reply is not None:
        (root / "studio-reply.md").write_text(reply)
    return root, job


def probe_of(root: Path) -> dict:
    result = pc.run_probe(root)
    assert result is not None and not result["errors"], result
    return result


# --- memory and leaks ------------------------------------------------------

def test_memoryless_prompts_fail(tmp_path: Path) -> None:
    root, _ = make_package(tmp_path, MEMORYLESS)
    problems = pc.check_memory(root, probe_of(root))
    assert any("carry none of what that participant already saw" in p for p in problems), problems
    assert any("say earlier screens or answers are shown but contain none" in p for p in problems), problems


def test_full_transcript_passes(tmp_path: Path) -> None:
    root, _ = make_package(tmp_path, WITH_MEMORY)
    result = probe_of(root)
    assert result["stub"] == "smoke_llm"
    assert pc.check_memory(root, result) == []


def test_dropping_the_screen_just_answered_fails(tmp_path: Path) -> None:
    # Keeps the scenario but forgets the rating screen it just answered.
    partial = WITH_MEMORY.replace('history += [s, "Your answer: " + reply]', 'history = [SCREENS[0]]')
    root, _ = make_package(tmp_path, partial)
    problems = pc.check_memory(root, probe_of(root))
    assert any(p.startswith("participant memory") for p in problems), problems
    # With a scenario that differs by condition, it is reported as the dropped screen.
    per_condition = partial.replace("for s in SCREENS:", "for s in [OUTCOMES[arm]] + SCREENS[1:]:").replace(
        "history = [SCREENS[0]]", "history = [OUTCOMES[arm]]").replace(
        "def smoke_llm", "OUTCOMES = {'physician_success': 'The physician decided to operate and the operation "
        "succeeded without any complication at all.', 'patient_failure': 'The patient decided to have the "
        "operation and the operation failed, and he died soon after.'}\ndef smoke_llm")
    root, _ = make_package(tmp_path / "b", per_condition)
    (root / "materials/materials.json").write_text(json.dumps({"screens": [
        {"text": "The physician decided to operate and the operation succeeded without any complication at all."},
        {"text": "The patient decided to have the operation and the operation failed, and he died soon after."},
        {"text": RATING}, {"text": IMPORTANCE}]}))
    problems = pc.check_memory(root, probe_of(root))
    assert any("drop the screen the participant has just answered" in p for p in problems), problems


def test_condition_and_question_ids_in_prompts_fail(tmp_path: Path) -> None:
    root, _ = make_package(tmp_path, LEAKY)
    problems = pc.check_memory(root, probe_of(root))
    leak = next(p for p in problems if "internal identifiers" in p)
    assert "physician_success" in leak and "QID14" in leak


def test_declared_narrow_memory_needs_reason_and_skips_carry_check(tmp_path: Path) -> None:
    root, _ = make_package(tmp_path, MEMORYLESS, task_extra={"memory": {"scope": "trial"}})
    problems = pc.check_memory(root, probe_of(root))
    assert any("gives no reason" in p for p in problems)
    assert not any("carry none" in p for p in problems)
    root, _ = make_package(tmp_path / "b", MEMORYLESS.replace("You have only seen the screens above.", "Screen."),
                           task_extra={"memory": {"scope": "separate-days", "reason": "paper p.3: sessions a week apart"}})
    assert pc.check_memory(root, probe_of(root)) == []


def test_interleaved_pair_with_declared_agents_passes(tmp_path: Path) -> None:
    pair = ADAPTER_HEAD + '''
def run_sessions(llm, seed, n, arms=None):
    out = []
    for arm in (arms or ARMS):
        hist = {"A": [], "B": []}
        intro = {"A": "You are Player A in a two-person game played for real money today.",
                 "B": "You are Player B in a two-person game played for real money this afternoon."}
        for s in SCREENS:
            for role in ("A", "B"):
                reply = llm("\\n".join([intro[role]] + hist[role] + [s]))
                hist[role] += [s, reply]
        out.append({"arm": arm, "condition": arm})
    return out
'''
    root, _ = make_package(tmp_path / "b", pair, task_extra={"memory": {"scope": "session", "agents_per_session": 2}})
    assert pc.check_memory(root, probe_of(root)) == []


# --- human results ---------------------------------------------------------

LAYOUT_TABLE = """
Some prose about the study design and the participants who took part in the survey.

                              SUCCESS                 FAILURE
                         M       SD      M       SD      M      SD      M      SD
        Evaluation       1.81    0.84    1.76    0.79    0.45   1.55    0.90   1.36
        Importance       4.49    0.82    4.40    0.77    4.13   1.04    4.28   0.91

Table 5 Means and standard deviations for each measured variable for all conditions.

Table 3 Power analysis.
        Effect           0.21    0.35    0.50    0.80    0.95   0.99    0.12   0.33
"""


def test_paper_tables_find_result_tables_only() -> None:
    tables = pc.paper_tables(LAYOUT_TABLE)
    assert [t["number"] for t in tables] == ["5"]
    assert "1.81" in tables[0]["values"] and len(tables[0]["values"]) == 16


def test_missing_table_cells_fail_and_recorded_table_passes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: LAYOUT_TABLE if layout else "")
    root, job = make_package(tmp_path, WITH_MEMORY, results=[
        {"id": "anova", "value": "F(1, 688) = 152.37"}])
    problems = pc.check_results(root, job)
    assert problems and "Table 5" in problems[0]
    cells = [[1.81, 0.84, 1.76, 0.79, 0.45, 1.55, 0.90, 1.36], [4.49, 0.82, 4.40, 0.77, 4.13, 1.04, 4.28, 0.91]]
    root, job = make_package(tmp_path / "b", WITH_MEMORY, results=[
        {"id": "table-5", "value": {"table": "Table 5", "rows": [{"label": "Evaluation", "cells": cells[0]},
                                                                  {"label": "Importance", "cells": cells[1]}]}}])
    assert pc.check_results(root, job) == []


PROSE_STATS = " ".join(
    f"Study {i}: t({100 + i}) = {2 + i / 10:.2f}, p = .01, d = {0.3 + i / 100:.2f}, 95% CI [{0.1 + i / 100:.2f}, "
    f"{0.5 + i / 100:.2f}]." for i in range(6))


def test_reported_statistics_coverage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: "" if layout else PROSE_STATS)
    stats = pc.paper_statistics(PROSE_STATS)
    assert len(stats) == 24
    root, job = make_package(tmp_path, WITH_MEMORY, results=[{"id": "s0", "value": "t(100) = 2.00, d = 0.30"}])
    assert "test statistics" in pc.check_results(root, job)[0]
    root, job = make_package(tmp_path / "b", WITH_MEMORY, results=[{"id": "all", "value": PROSE_STATS}])
    assert pc.check_results(root, job) == []


def test_prose_only_paper_with_few_statistics_passes(tmp_path: Path, monkeypatch) -> None:
    text = "Participants preferred the theatre (t(98) = 2.10, d = .42). No tables are printed."
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: text)
    root, job = make_package(tmp_path, WITH_MEMORY, results=[])
    assert pc.check_results(root, job) == []


def test_accepted_model_sync_skips_result_records(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: LAYOUT_TABLE if layout else PROSE_STATS)
    root, job = make_package(tmp_path, WITH_MEMORY)
    (job / "studio_request.json").write_text(json.dumps({"purpose": "accepted-model-sync"}))
    assert pc.check_results(root, job) == []


# --- analyses --------------------------------------------------------------

def test_result_fields_need_an_evaluation_link(tmp_path: Path) -> None:
    root, job = make_package(tmp_path, WITH_MEMORY, results=[
        {"id": "result_kind", "value": "reported"},
        {"id": "anova", "value": "F(1, 688) = 152.37"},
        {"id": "desc", "value": "n = 692", "extensions": {"evaluation": {"output": "descriptives.n"}}},
        {"id": "ttest", "value": "t(281.42) = 7.20", "extensions": {"evaluation": {"output": "welch.t"}}},
        {"id": "effect", "value": "d = 0.77", "extensions": {"evaluation": {"output": "outcome_effect"}}},
    ])
    problems = pc.check_analyses(root, job, probe_of(root))
    assert any("res-1.anova" in p and "do not say which evaluate() output" in p for p in problems), problems
    assert any("res-1.ttest -> welch.t" in p for p in problems), problems
    assert not any("res-1.desc" in p or "res-1.effect" in p for p in problems)


def test_missing_library_is_not_a_reason(tmp_path: Path) -> None:
    evaluation = EVALUATION.replace("fewer than two participants per condition", "needs scipy, not installed")
    root, job = make_package(tmp_path, WITH_MEMORY, evaluation=evaluation, results=[
        {"id": "anova", "value": "F = 4.73", "extensions": {"evaluation": {"not_ready": "statsmodels unavailable"}}}])
    problems = pc.check_analyses(root, job, probe_of(root))
    assert any("blames a missing library" in p for p in problems), problems


def test_evaluate_crash_is_reported(tmp_path: Path) -> None:
    root, job = make_package(tmp_path, WITH_MEMORY, evaluation="def evaluate(sessions):\n    return 1 / 0\n")
    assert any("ZeroDivisionError" in p for p in pc.check_analyses(root, job, probe_of(root)))


def test_reply_cannot_claim_unimplemented_analyses(tmp_path: Path) -> None:
    root, job = make_package(tmp_path, WITH_MEMORY, reply=(
        "The evaluation computes the 2 x 2 ANOVA and the interaction.\n"
        "Mediation is not computed yet.\n"))
    problems = pc.check_analyses(root, job, probe_of(root))
    claim = next(p for p in problems if "does not implement" in p)
    assert "ANOVA" in claim and "mediation" not in claim
    implemented = EVALUATION + "\nimport statsmodels.formula.api as smf\ndef _anova(df):\n    return smf.ols('y ~ a * b', df)\n"
    root, job = make_package(tmp_path / "b", WITH_MEMORY, evaluation=implemented,
                             reply="The evaluation computes the 2 x 2 ANOVA.\n")
    assert pc.check_analyses(root, job, probe_of(root)) == []


# --- researcher reply ------------------------------------------------------

GOOD_REPLY = """Ready to run: Studies 1 and 2, all four conditions.
Needs you: one question about the exclusion rule below.
Licence: CC BY 4.0 (printed on page 1 of the paper); the OSF project is also CC BY 4.0.

## Question
1. The paper excludes people who guessed the hypothesis but gives no coding rule; the authors' public script does not apply it either. Shall we leave it out (our default)?

Run size: 692 participants, about 3,460 model calls, about $4 at $1 / $5 per million tokens.

## Not computed yet
The mediation analysis.
"""


def test_good_reply_passes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: "This work is licensed under the Creative "
                        "Commons Attribution 4.0 International License (CC-BY 4.0).")
    root, job = make_package(tmp_path, WITH_MEMORY, reply=GOOD_REPLY)
    assert pc.check_reply(root, job) == []


def test_reply_problems_are_reported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: "This article is distributed under the terms of "
                        "the Creative Commons Attribution 3.0 License.")
    reply = GOOD_REPLY.replace("Licence: CC BY 4.0 (printed on page 1 of the paper); the OSF project is also CC BY 4.0.",
                               "Licence: Crossref lists CC BY-NC-ND 3.0, which is not open.")
    reply += ("\nThe arms use QID14; the smoke test is plumbing only and `materials-request` lists them. "
              "Nothing is missing. Is A right? Is B right? Is C right?\n")
    reply = reply.replace("Run size: 692 participants, about 3,460 model calls, about $4", "It takes a while")
    root, job = make_package(tmp_path, WITH_MEMORY, reply=reply)
    text = "\n".join(pc.check_reply(root, job))
    assert "(CC BY 3.0)" in text
    assert "asks 4 questions" in text
    assert "QID" in text and "'arm'" in text and "plumbing" in text and "materials-request" in text
    assert "overclaims" in text
    assert "Run size:" in text


def test_reply_must_open_with_three_lines(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: "")
    root, job = make_package(tmp_path, WITH_MEMORY, reply="# Studio reply\n\n" + GOOD_REPLY)
    assert any("must open with three plain lines" in p for p in pc.check_reply(root, job))


def test_paper_licence_reading() -> None:
    assert pc._line_mentions_licence("Licence: CC BY 4.0", "CC BY 4.0")
    assert pc._line_mentions_licence("Licence: Creative Commons Attribution 4.0 International", "CC BY 4.0")
    assert not pc._line_mentions_licence("Licence: CC BY-NC-ND 3.0 (Crossref)", "CC BY 3.0")


# --- run-size estimate and the validator entry point ------------------------

def test_estimate_scales_stand_in_run() -> None:
    probe = {"runs": [{"calls": [{"prompt": "x" * 400}, {"prompt": "x" * 400}], "sessions": 1},
                      {"calls": [{"prompt": "x" * 400}] * 4, "sessions": 2}]}
    result = estimate(probe, 100, price_in=1.0, price_out=5.0)
    assert result["model_calls"] == 200 and result["input_tokens"] == 20_000
    assert result["cost_usd"] == pytest.approx(0.02 + 200 * 150 * 5 / 1e6, abs=0.01)


def test_checks_fail_a_memoryless_package(tmp_path: Path) -> None:
    root, job = make_package(tmp_path, MEMORYLESS + "\nif __name__ == '__main__':\n    sys.exit(0)\n")
    (root / "study.json").write_text("{}\n")
    for name in ("source/paper_metadata.json", "source/evidence.json", "audit/missing_information.json"):
        (root / name).write_text("{}\n")
    # validate_package.py runs these checks after the answer-aware smoke test; run them directly here.
    result = subprocess.run([sys.executable, str(ROOT / "agent_pipeline/package_checks.py"), str(root), str(job)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "participant memory" in result.stdout
    assert "check_package(root, package.parent)" in (ROOT / "agent_pipeline/validate_package.py").read_text()


def test_licence_first_mention_must_be_the_papers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: "This work is licensed under a Creative Commons "
                        "Attribution 3.0 License.")
    root, job = make_package(tmp_path, WITH_MEMORY, reply="# Reply\n\nThe Crossref record lists CC BY-NC-ND 3.0, "
                             "which is not open. The paper prints CC BY 3.0.\n")
    assert any("(CC BY 3.0)" in p for p in pc.check_reply(root, job))


def test_t_test_variant_must_match_the_paper(tmp_path: Path, monkeypatch) -> None:
    welch_only = EVALUATION + "\ndef welch_t(x, y):\n    return 0\n"
    pooled = EVALUATION + "\nfrom scipy import stats\ndef t(x, y):\n    return stats.ttest_ind(x, y)\n"
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: "t(183) = 6.6, p < .001 and t(192) = 2.05, p = .04")
    root, job = make_package(tmp_path, WITH_MEMORY, evaluation=welch_only)
    assert any("pooled-variance Student" in p for p in pc.check_analyses(root, job, probe_of(root)))
    root, job = make_package(tmp_path / "b", WITH_MEMORY, evaluation=pooled)
    assert not any("Student" in p for p in pc.check_analyses(root, job, probe_of(root)))
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: "t(665.68) = -10.16 and t(281.42) = 7.20")
    assert any("Welch t-tests" in p for p in pc.check_analyses(root, job, probe_of(root)))


def test_article_licence_not_data_or_figure_licence(tmp_path: Path, monkeypatch) -> None:
    text = ("Figure 1 photo: Wikimedia, CC BY-SA 3.0. Data and materials are on OSF under CC0. "
            "This is an open-access article distributed under the terms of the Creative Commons Attribution "
            "4.0 International License (CC-BY 4.0).")
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: text)
    assert pc.paper_licence(tmp_path) == "CC BY 4.0"
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: "Data are on OSF under CC0. © 2020 The Authors.")
    assert pc.paper_licence(tmp_path) is None


def test_reply_in_another_language_and_paper_words(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: "Participants kept the ticket stub in an arms race.")
    reply = ("Ready to run: 研究一的四个条件。\nNeeds you: 一个问题。\nLicence: CC BY 4.0（论文第 1 页）。\n"
             "1. 是否排除猜出假设的人？\n2. 是否保留？\n3. 是否继续？\n4. 还有吗？\n"
             "Run size: 692 名参与者，约 3,460 次模型调用，约 $4。\n参与者保留了 ticket stub，arms race 是论文用语。\n")
    root, job = make_package(tmp_path, WITH_MEMORY, reply=reply)
    problems = pc.check_reply(root, job)
    assert len(problems) == 1 and "asks 4 questions" in problems[0], problems


def test_library_excuse_is_narrow(tmp_path: Path) -> None:
    assert pc.LIBRARY_EXCUSE_RE.search("needs numpy and scipy in this runtime")
    assert pc.LIBRARY_EXCUSE_RE.search("scipy is not installed here")
    assert not pc.LIBRARY_EXCUSE_RE.search("statsmodels MixedLM needs at least 20 sessions")
    assert not pc.LIBRARY_EXCUSE_RE.search("scipy chi2_contingency requires expected counts of 5")
    assert not pc.LIBRARY_EXCUSE_RE.search("Welch via scipy without equal variances")


def test_table_must_be_recorded_as_a_table(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: LAYOUT_TABLE if layout else "")
    pasted = "1.81 0.84 1.76 0.79 0.45 1.55 0.90 1.36 4.49 0.82 4.40 0.77 4.13 1.04 4.28 0.91"
    root, job = make_package(tmp_path, WITH_MEMORY, results=[{"id": "table-5", "value": pasted}])
    assert any("Table 5" in p for p in pc.check_results(root, job))


def test_unresolved_result_needs_no_link(tmp_path: Path) -> None:
    root, job = make_package(tmp_path, WITH_MEMORY, results=[
        {"id": "s2", "value": "Study 2 results not reported", "origin": "missing", "state": "missing"}])
    assert pc.check_analyses(root, job, probe_of(root)) == []


def test_adapter_output_and_failing_conditions_cannot_hide_prompts(tmp_path: Path) -> None:
    noisy = MEMORYLESS.replace("def run_sessions(llm, seed, n, arms=None):",
                               "import os\nos.write(1, b'noise')\ndef run_sessions(llm, seed, n, arms=None):\n"
                               "    os.write(1, b'more noise')")
    root, job = make_package(tmp_path, noisy)
    assert any(p.startswith("participant memory") for p in pc.check_package(root, job))
    failing = MEMORYLESS.replace("for arm in (arms or ARMS):", "for arm in (arms or ARMS):\n        raise KeyError(arm)")
    root, job = make_package(tmp_path / "b", failing)
    assert any("run_sessions raised" in p and "KeyError" in p for p in pc.check_package(root, job))


def test_long_full_memory_session_is_checked_quickly(tmp_path: Path) -> None:
    import time
    long_adapter = WITH_MEMORY.replace("for s in SCREENS:", "for s in [f'{t} (screen {k}).' for k in range(300) for t in SCREENS]:")
    root, job = make_package(tmp_path, long_adapter)
    (root / "materials/materials.json").write_text(json.dumps({"screens": [
        {"text": f"{t} (screen {k})."} for k in range(300) for t in (SCENARIO + " " + OUTCOME, RATING, IMPORTANCE)]}))
    result = probe_of(root)
    start = time.monotonic()
    assert pc.check_memory(root, result) == []
    assert time.monotonic() - start < 30


def test_repair_guard_allows_evaluation_links() -> None:
    from agent_pipeline.repair_guard import compare_packages
    field = {"id": "anova", "value": "F(1, 688) = 152.37", "origin": "verbatim", "state": "confirmed",
             "evidenceIds": ["ev-1"]}
    linked = dict(field, extensions={"evaluation": {"output": "anova.F"}})
    study = lambda f: {"study.json": {"program": {"nodes": [{"id": "r", "kind": "result", "fields": [f]}],
                                                  "evidence": [{"id": "ev-1", "quote": "F(1, 688) = 152.37"}]}}}
    verdict = compare_packages(study(field), study(linked))
    assert not verdict.get("violations"), verdict


def test_history_lines_with_answers_count_as_memory(tmp_path: Path) -> None:
    wrapped = WITH_MEMORY.replace('history += [s, "Your answer: " + reply]',
                                  'history.append("Screen: " + s + " (you answered " + reply + ")")')
    root, _ = make_package(tmp_path, wrapped)
    assert pc.check_memory(root, probe_of(root)) == []


def test_repeating_a_shared_intro_is_not_memory(tmp_path: Path) -> None:
    intro = MEMORYLESS.replace('"You have only seen the screens above.\\nSCREEN:\\n" + s',
                               'SCREENS[0] + "\\nSCREEN:\\n" + s').replace(
        "for s in SCREENS]", "for s in SCREENS + [f'How sure are you about screen number {k} of this survey, from one to seven?' "
        "for k in range(3)]]")
    root, _ = make_package(tmp_path, intro, task_extra={"memory": {"scope": "session", "agents_per_session": 2}})
    assert any("carry none" in p for p in pc.check_memory(root, probe_of(root)))


def test_quoted_item_is_not_a_question(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pc, "paper_text", lambda job, layout: "")
    reply = GOOD_REPLY + '\nThe last screen asks "Do you think outcomes matter?" exactly as printed.\n'
    root, job = make_package(tmp_path, WITH_MEMORY, reply=reply)
    assert not any("questions" in p for p in pc.check_reply(root, job))
    assert not pc.LIBRARY_EXCUSE_RE.search("requires scipy chi2_contingency expected counts of 5")


def test_probe_uses_the_answer_aware_participant(tmp_path: Path) -> None:
    adapter = ADAPTER_HEAD.replace("import json, sys", "import json, sys\nimport replies") + '''
def run_sessions(llm, seed, n, arms=None):
    out = []
    for arm in (arms or ARMS):
        history = []
        for s in SCREENS:
            prompt = "\\n".join(history + [s])
            reply = llm.answer_for(prompt, ["q1"], attempt=0, final=True)
            if reply != "keyed answer":
                raise ValueError("comprehension failed")
            history += [s, reply]
        out.append({"arm": arm, "condition": arm})
    return out
'''
    root, _ = make_package(tmp_path, adapter)
    (root / "task/replies.py").write_text(
        "class StubParticipant:\n    def __init__(self, seed=0):\n        pass\n"
        "    def __call__(self, prompt):\n        return '1'\n"
        "    def answer_for(self, prompt, questions, attempt=0, final=True):\n        return 'keyed answer'\n")
    result = probe_of(root)
    assert result["stub"] == "replies.StubParticipant"
    assert all(len(run["calls"]) == 3 and not run["error"] for run in result["runs"]), result["runs"]
