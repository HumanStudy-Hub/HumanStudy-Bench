"""Analysis code, survey versions, researcher questions, image tables, lost minus signs and
recorded answers (agent_pipeline/source_use.py, paper_tables.py, recorded_answers.py)."""
import hashlib
import json
from pathlib import Path

import pytest

from agent_pipeline import paper_tables as pt
from agent_pipeline import recorded_answers as ra
from agent_pipeline import source_use as su
from agent_pipeline.check_sources import file_text
from agent_pipeline.network_policy import looks_like_participant_data
from package_fixture import write_runnable

RMD = """---
title: analysis
---

Load the data.

```{r}
# Remove the first row as it contains useless information
df <- df[-1, ]
```

```{r}
# There were coding errors in Status quo B. These errors were corrected using the following chunk.
df$B <- recode(df$B, `3` = 4, `4` = 3)
prop.test(x = c(10, 20), n = c(50, 50), correct = FALSE)
```

## 6. Data Analysis (After Exclusion)
"""


def _job(tmp_path: Path, files: dict[str, str], metadata: dict, issues: list | None = None) -> tuple[Path, Path]:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    (root / "source").mkdir(parents=True)
    external = job / "input" / "external"
    external.mkdir(parents=True)
    log = []
    for name, text in files.items():
        (external / name).write_text(text)
        log.append({"url": f"https://osf.io/download/{name}/", "path": f"input/external/{name}",
                    "sha256": hashlib.sha256(text.encode()).hexdigest()})
    (external / "fetch_log.json").write_text(json.dumps(log))
    (root / "source/paper_metadata.json").write_text(json.dumps(metadata))
    program = {"nodes": [], "evidence": [], "issues": issues or []}
    (root / "study.json").write_text(json.dumps({"program": program}))
    return job, root


def _source(sid: str, name: str) -> dict:
    return {"id": sid, "path": f"input/external/{name}", "url": f"https://osf.io/download/{name}/"}


# -------------------------------------------------------------------------- code is not data

@pytest.mark.parametrize("name,data", [
    ("analysis.R", False), ("Analysis_2nd_Run.Rmd", False), ("clean_data.Rmd", False), ("data_prep.sps", False),
    ("study1.do", False), ("notebook.ipynb", False), ("spss_syntax.txt", False), ("analysis_data.txt", True), ("scores.py", False),
    ("data.csv", True), ("results.json", True), ("analysis.sav", True), ("raw_data.txt", True),
])
def test_analysis_code_is_fetched_and_data_still_refused(name, data):
    assert looks_like_participant_data(name) is data


def test_code_files_are_readable_for_quotes(tmp_path):
    notebook = tmp_path / "a.ipynb"
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["t.test(x, y, var.equal = FALSE)\n"],
                                               "outputs": [{"text": "participant 17 said"}]}]}))
    assert "var.equal = FALSE" in file_text(notebook) and "participant 17" not in file_text(notebook)
    syntax = tmp_path / "a.sps"
    syntax.write_text("T-TEST GROUPS=cond(1 2) /VARIABLES=score.\n")
    assert "T-TEST" in file_text(syntax)


# -------------------------------------------------------------------------- 1. analysis code

def test_signal_lines_find_recodes_and_skip_headings(tmp_path):
    path = tmp_path / "a.Rmd"
    path.write_text(RMD)
    lines = dict(su.signal_lines(path))
    assert any("coding errors" in text for text in lines.values())
    assert any("recode(" in text for text in lines.values())
    assert not any("Data Analysis" in text for text in lines.values())
    assert not any("correct = FALSE" in text for text in lines.values())


def test_downloaded_code_without_a_record_fails(tmp_path):
    job, root = _job(tmp_path, {"a.Rmd": RMD}, {"external_sources": [_source("code", "a.Rmd")]})
    problems = su.check_analysis_code(root, job)
    assert problems and "nothing is recorded from input/external/a.Rmd" in problems[0]


def test_uncovered_recode_fails_and_covered_passes(tmp_path):
    record = {"source": "code", "lines": "13-14", "quote": "df$B <- recode(df$B, `3` = 4, `4` = 3)",
              "what": "framing B options 3 and 4 are recoded", "use": "carried over",
              "where": "evaluation/evaluation.py RECODE"}
    metadata = {"external_sources": [_source("code", "a.Rmd")], "analysis_code": [record]}
    job, root = _job(tmp_path, {"a.Rmd": RMD}, metadata)
    problems = su.check_analysis_code(root, job)
    assert len(problems) == 1 and "line 8" in problems[0] and "useless" in problems[0]
    metadata["analysis_code"].append({"source": "code", "lines": 8, "quote": "# Remove the first row",
                                      "what": "drops the export's second header row", "use": "not applicable",
                                      "reason": "the replay has no export header rows"})
    (root / "source/paper_metadata.json").write_text(json.dumps(metadata))
    assert su.check_analysis_code(root, job) == []


def test_code_record_quote_must_be_at_its_lines(tmp_path):
    record = {"source": "code", "lines": "1-3", "quote": "df$B <- recode(df$B, `3` = 4, `4` = 3)",
              "what": "framing B options are recoded", "use": "carried over", "where": "evaluation recode step"}
    job, root = _job(tmp_path, {"a.Rmd": RMD}, {"external_sources": [_source("code", "a.Rmd")],
                                                "analysis_code": [record]})
    assert any("does not occur at lines 1-3" in p for p in su.check_analysis_code(root, job))


def test_listed_but_not_downloaded_code_fails(tmp_path):
    metadata = {"material_searches": [{"query": "Data and code folder, Analysis subfolder",
                                       "result": "analysis scripts listed; not downloaded (not needed for the survey wording)"}]}
    job, root = _job(tmp_path, {}, metadata)
    assert any("no analysis file was downloaded" in p for p in su.check_analysis_code(root, job))
    metadata["analysis_code"] = [{"source": "not fetched", "reason": "not needed"}]
    (root / "source/paper_metadata.json").write_text(json.dumps(metadata))
    assert su.check_analysis_code(root, job)
    metadata["analysis_code"] = [{"source": "not fetched", "reason": "only inside a zip with participant data"}]
    (root / "source/paper_metadata.json").write_text(json.dumps(metadata))
    assert su.check_analysis_code(root, job) == []


def test_psychopy_script_is_not_analysis_code(tmp_path):
    task = tmp_path / "experiment.py"
    task.write_text("from psychopy import visual\n# remove the fixation cross\n")
    assert not su.is_analysis_code(task)
    analysis = tmp_path / "analysis.py"
    analysis.write_text("from scipy import stats\nstats.ttest_ind(a, b, equal_var=False)\n")
    assert su.is_analysis_code(analysis)


# -------------------------------------------------------------------------- 2. survey versions

def test_two_survey_files_need_a_comparison(tmp_path):
    files = {"Survey_v3.qsf": "{}", "Survey_v3.docx": "x", "prereg-design.docx": "x"}
    job, root = _job(tmp_path, files, {})
    assert su.survey_files(su._downloads(job)) == ["input/external/Survey_v3.qsf", "input/external/Survey_v3.docx"]
    problems = su.check_survey_versions(root, job)
    assert problems and "survey_versions does not compare" in problems[0]


def test_text_from_a_version_that_does_not_match_fails(tmp_path):
    files = {"Survey_v3.qsf": "{}", "Survey_v3.docx": "x"}
    versions = [
        {"path": "input/external/Survey_v3.docx", "matches_paper": "yes", "used_for": "every screen",
         "compared": "two scenarios in fixed order as the paper says"},
        {"path": "input/external/Survey_v3.qsf", "matches_paper": "no", "used_for": "not used",
         "compared": "four scenarios in random order, not as the paper says"},
    ]
    metadata = {"external_sources": [_source("qsf", "Survey_v3.qsf"), _source("doc", "Survey_v3.docx")],
                "survey_versions": versions}
    job, root = _job(tmp_path, files, metadata)
    assert su.check_survey_versions(root, job) == []
    node = {"id": "mat-s1", "kind": "material", "studyIds": ["phase1"], "fields": [
        {"id": "text", "extensions": {"material": {"status": "found-open", "sourceIds": ["qsf"]}}},
        {"id": "intro", "extensions": {"material": {"status": "found-open", "sourceIds": ["doc"]}}}]}
    (root / "study.json").write_text(json.dumps({"program": {"nodes": [node], "issues": []}}))
    problems = su.check_survey_versions(root, job)
    assert any("mat-s1.text copies participant text from input/external/Survey_v3.qsf" in p for p in problems)


# -------------------------------------------------------------------------- 3. researcher questions

def test_question_needs_sources_checked_and_is_not_asked_when_answered(tmp_path):
    issue = {"id": "version", "title": "Two or four scenarios?", "severity": "decision"}
    item = {"id": "need-consent", "title": "Needs the original consent screen", "severity": "decision"}
    job, root = _job(tmp_path, {"Survey.qsf": "{}", "Survey.docx": "x"}, {}, issues=[issue, item])
    problems = su.check_questions(root, job)
    assert len(problems) == 1 and "'version' (no record)" in problems[0] and "need-consent" not in problems[0]
    record = {"issue": "version", "sources_checked": [{"paths": ["input/external/Survey.qsf"], "answers": False,
                                                       "found": "four scenarios"}]}
    (root / "source/paper_metadata.json").write_text(json.dumps({"researcher_questions": [record]}))
    assert "not checked: input/external/Survey.docx" in su.check_questions(root, job)[0]
    record["sources_checked"].append({"path": "input/external/Survey.docx", "answers": True, "found": "two scenarios"})
    (root / "source/paper_metadata.json").write_text(json.dumps({"researcher_questions": [record]}))
    assert any("a downloaded file answers 'version'" in p for p in su.check_questions(root, job))


def test_decision_from_sources_must_quote_the_file(tmp_path):

    job, root = _job(tmp_path, {"Survey.txt": "Participants saw two different scenarios."}, {})
    metadata = {"decisions_from_sources": [{"what": "phase 1 shows two scenarios", "path": "input/external/Survey.txt",
                                            "quote": "three different scenarios"}]}
    (root / "source/paper_metadata.json").write_text(json.dumps(metadata))
    assert any("quote does not occur" in p for p in su.check_questions(root, job))
    metadata["decisions_from_sources"][0]["quote"] = "two different scenarios"
    (root / "source/paper_metadata.json").write_text(json.dumps(metadata))
    assert su.check_questions(root, job) == []


# -------------------------------------------------------------------------- tables without a text layer

LAYOUT_P9 = ("   In Table 1 and 2, we summarized the results         1.46, 95% CI [0.72, 2.97], Cohen's h = 0.19, 95% CI [-\n"
             "   of the first phase. See Table 12S for more.         0.16, 0.54]. The other three differences were\n")
PLAIN_P9 = ("In Table 1 and 2, we summarized the results of the first phase. See Table 12S for more.\n"
            "1.46, 95% CI [0.72, 2.97], Cohen's h = 0.19, 95% CI [0.16, 0.54]. The other three differences were\n")
TEXT_TABLE = ("Table 5 Means and standard deviations for each condition.\n"
              "              Evaluation      1.81    0.84     1.76    0.79\n"
              "              Responsibility  5.88    1.02     6.13    0.97\n"
              "              Norms           4.03    0.68     4.06    0.62\n")


@pytest.fixture
def fake_pdf(tmp_path, monkeypatch):
    pages = {True: [LAYOUT_P9, TEXT_TABLE], False: [PLAIN_P9, TEXT_TABLE]}
    monkeypatch.setattr(pt, "page_texts", lambda pdf, layout: pages[layout])
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    (job / "input/paper.pdf").write_bytes(b"%PDF-1.4")
    root = job / "package" / "paper"
    (root / "source").mkdir(parents=True)
    return job, root


def test_inventory_finds_mentioned_tables_without_text_and_skips_supplement(fake_pdf):
    job, _ = fake_pdf
    tables = {t["table"]: t for t in pt.table_inventory(job / "input/paper.pdf")}
    assert set(tables) == {"1", "2", "5"}
    assert tables["1"]["needs_page_image"] and tables["1"]["mentioned_on"] == [1]
    assert tables["5"]["text_layer"] and not tables["5"]["needs_page_image"]


def test_caption_sentence_is_not_a_caption():
    assert not pt.CAPTION_RE.search("   Table 3 presents the means and standard deviations")
    assert pt.CAPTION_RE.search("Table 3. Means and standard deviations")
    assert pt.CAPTION_RE.search("                  Table 1                         half (68 women and 53 men)")


def _write_program(root: Path, fields: list, evidence: list | None = None, metadata: dict | None = None):
    node = {"id": "res", "kind": "result", "fields": fields}
    (root / "study.json").write_text(json.dumps({"program": {"nodes": [node], "evidence": evidence or []}}))
    (root / "source/paper_metadata.json").write_text(json.dumps(metadata or {}))


def test_image_tables_must_be_listed_and_recorded(fake_pdf):
    job, root = fake_pdf
    _write_program(root, [])
    problems = pt.check_tables(root, job)
    assert problems and "Table 1 (mentioned on page 1); Table 2 (mentioned on page 1)" in problems[0]
    table = {"rows": [[18, 62, 0.29], [22, 60, 0.37], [2.47, 1.37, 4.44]]}
    metadata = {"paper_tables": [
        {"table": "1", "kind": "result", "read_from": "rendered page", "recorded_in": "res.table1"},
        {"table": "2", "kind": "result", "read_from": "text", "recorded_in": "res.missing"}]}
    _write_program(root, [{"id": "table1", "value": table}], metadata=metadata)
    problems = pt.check_tables(root, job)
    assert len(problems) == 2 and all("Table 2" in p for p in problems)
    metadata["paper_tables"][1] = {"table": "2", "kind": "other", "what": "the design of the five framings"}
    _write_program(root, [{"id": "table1", "value": table}], metadata=metadata)
    assert pt.check_tables(root, job) == []


# -------------------------------------------------------------------------- lost minus signs

def test_sign_loss_finds_the_line_end_minus(fake_pdf):
    job, _ = fake_pdf
    spots = pt.sign_loss(job / "input/paper.pdf")
    assert [(s["page"], s["printed"]) for s in spots] == [(1, "-0.16")]


def test_split_true_minus_is_found_but_not_a_subtraction(tmp_path, monkeypatch):
    minus = "−"
    plain = f"the effect was b = {minus} 0.011, and n {minus} 1 degrees of freedom"
    monkeypatch.setattr(pt, "page_texts", lambda pdf, layout: [plain])
    spots = pt.sign_loss(tmp_path / "x.pdf")
    assert [s["printed"] for s in spots] == ["-0.011"]


def test_recorded_without_its_minus_fails_until_signed(fake_pdf):
    job, root = fake_pdf
    _write_program(root, [{"id": "h", "value": 0.19}, {"id": "h-ci", "value": [0.16, 0.54]}])
    assert any("prints -0.16 on page 1" in p for p in pt.check_signs(root, job))
    _write_program(root, [{"id": "h", "value": 0.19}, {"id": "h-ci", "value": [-0.16, 0.54]}])
    assert pt.check_signs(root, job) == []
    evidence = [{"id": "ev", "locator": {"page": 1}, "quote": "x", "note": "rendered page checked"}]
    _write_program(root, [{"id": "h-ci", "value": [0.16, 0.54]}], evidence=evidence)
    assert pt.check_signs(root, job) == []


def test_impossible_values_fail(fake_pdf):
    job, root = fake_pdf
    _write_program(root, [{"id": "a", "value": {"d": 0.64, "ci95": [0.71, 1.08]}},
                          {"id": "b", "value": {"cohens_f": 0.08, "cohens_f_ci90": [0.15, 0.02]}},
                          {"id": "c", "value": {"t": 2.1, "p": 1.3}},
                          {"id": "h", "value": 0.5}, {"id": "h_ci", "value": [0.1, 0.3]},
                          {"id": "ok", "value": {"mean_diff": 0.75, "cohens_d": 0.64, "ci95": [0.21, 1.08]}}])
    problem = next(p for p in pt.check_signs(root, job) if "impossible values" in p)
    assert "d = 0.64 lies outside" in problem and "lower limit is above" in problem
    assert "not a probability" in problem and "h = 0.5 lies outside its CI" in problem
    assert "ok" not in problem


# -------------------------------------------------------------------------- recorded answers

def test_dropped_answer_is_found(tmp_path):
    root = tmp_path / "paper"
    write_runnable(root)
    adapter = root / "task/adapter.py"
    adapter.write_text(adapter.read_text().replace('                    session["comment"] = page.value("comment")\n', ""))
    report = ra.check(root)
    assert [d["question"] for d in report["dropped"]] == ["comment"]
    assert ra.problems(root) and "Any comments?" in ra.problems(root)[0]


def test_recorded_answers_pass_and_keyed_checks_are_skipped(tmp_path):
    root = tmp_path / "paper"
    write_runnable(root)
    report = ra.check(root)
    assert report["dropped"] == [] and report["checked"] == 2
    assert ra.problems(root) == []


def test_nondeterministic_package_is_not_judged(tmp_path):
    root = tmp_path / "paper"
    write_runnable(root)
    adapter = root / "task/adapter.py"
    text = adapter.read_text().replace('session = {"arm": arm,', 'import uuid\n            session = {"run": str(uuid.uuid4()), "arm": arm,')
    adapter.write_text(text.replace('                    session["comment"] = page.value("comment")\n', ""))
    report = ra.check(root)
    assert report["dropped"] == [] and "differ" in report["skipped"]


# -------------------------------------------------------------------------- review follow-ups

def test_two_column_sign_loss_only_flags_the_continued_number(tmp_path, monkeypatch):
    LAYOUT = ("left column text here and more words      h = 0.19, 95% CI [-\n"
              "more left column words go on here          0.16, 0.54]. Another result\n"
              "\n"
              "still the left column of the page          had h = 0.33, 95% CI [0.30, 0.94].\n")
    plain = ("left column text here and more words more left column words go on here "
             "still the left column of the page\nh = 0.19, 95% CI [0.16, 0.54]. Another result "
             "had h = 0.33, 95% CI [0.30, 0.94].\n")
    monkeypatch.setattr(pt, "page_texts", lambda pdf, layout: [LAYOUT] if layout else [plain])
    assert [s["printed"] for s in pt.sign_loss(tmp_path / "x.pdf")] == ["-0.16"]


def test_supplementary_tables_and_caption_styles(tmp_path, monkeypatch):
    page = ("We report it in Supplementary Table 3 and in Extended Data Table 4, and Table 8 in S1 Appendix.\n"
            "The results are in Table 1 of the main text.\n"
            " Table 1. Mean ratings by condition\n"
            "   Self    4.68   7.07\n   Other   5.10   6.80\n   Total   5.33   7.04\n"
            "Table 2 | Design of the study\n")
    second = "Table 5 – Means and standard deviations\n"
    monkeypatch.setattr(pt, "page_texts", lambda pdf, layout: [page, second])
    tables = {t["table"]: t for t in pt.table_inventory(tmp_path / "x.pdf")}
    assert set(tables) == {"1", "2", "5"}
    assert tables["1"]["caption_page"] == 1 and tables["1"]["text_layer"]
    assert tables["2"]["caption"].startswith("Design") and not tables["2"]["needs_page_image"]
    assert tables["5"]["caption"].startswith("Means") and tables["5"]["needs_page_image"]


def test_mentioned_table_without_result_words_is_not_demanded(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "page_texts", lambda pdf, layout: ["The stimuli are listed in Table 2.\n"])
    assert [t["needs_page_image"] for t in pt.table_inventory(tmp_path / "x.pdf")] == [False]


def test_sign_check_in_metadata_clears_a_checked_number(fake_pdf):
    job, root = fake_pdf
    _write_program(root, [{"id": "h-ci", "value": [0.16, 0.54]}],
                   metadata={"sign_checks": [{"page": 1, "value": 0.16, "note": "rendered page checked"}]})
    assert pt.check_signs(root, job) == []


def test_ratio_next_to_log_ci_and_mean_next_to_difference_ci_pass(fake_pdf):
    job, root = fake_pdf
    _write_program(root, [{"id": "a", "value": {"or": 2.47, "ci": [0.31, 1.49]}},
                          {"id": "b", "value": {"value": 4.5, "ci": [0.1, 0.9]}},
                          {"id": "c", "value": {"city": [5, 2]}}])
    assert pt.check_signs(root, job) == []


def test_notebook_with_printed_tables_is_refused(tmp_path):
    from agent_pipeline.fetch_public import _notebook_shows_data
    shows = tmp_path / "a.ipynb"
    rows = "   id  age  answer\n" + "".join(f"{i}  {20 + i}  {i % 7}\n" for i in range(6))
    shows.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["df.head()"],
                                            "outputs": [{"data": {"text/plain": [rows]}}]}]}))
    assert _notebook_shows_data(shows)
    clean = tmp_path / "b.ipynb"
    clean.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["t.test(x)"], "outputs": []}]}))
    assert not _notebook_shows_data(clean)


def test_stata_recode_and_noise(tmp_path):
    do = tmp_path / "a.do"
    do.write_text('recode part 1=0 if ParticipantNumber=="14"\n* standard errors clustered\nregress y x\n')
    assert [n for n, _ in su.signal_lines(do)] == [1]
    metadata = {"material_searches": [{"query": "ran agent_pipeline/fetch_public.py", "result": "cran.r-project.org"}]}
    assert not su.CODE_LISTED_RE.search(" ".join(m["query"] + " " + m["result"] for m in metadata["material_searches"]))
    assert su.CODE_LISTED_RE.search("listing shows analysis.R and data.csv")
