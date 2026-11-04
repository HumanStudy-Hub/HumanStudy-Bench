"""Round 3: results printed in the text, the abstract's results, supplement tables,
and evaluation links that hold the statistic they claim."""
import json
from pathlib import Path

from agent_pipeline import evaluation_links as el
from agent_pipeline import paper_tables as pt
from agent_pipeline import results_coverage as rc
from agent_pipeline.repair_guard import compare_packages, source_quote_matcher

ABSTRACT_PAGE = """Judgment and Decision Making
Does telling white lies signal pro-social preferences?
Abstract
The opportunity to tell a white lie generates a moral conflict between two opposite moral dictates, one
pushing towards telling the truth always and the other pushing towards helping others. Here we study how
people resolve this moral conflict. To answer this question, we conducted a two-stage experiment. We found
three major results: (i) both altruism and cooperation are positively correlated with aversion to telling a
Pareto white lie; (ii) both altruism and cooperation are negatively correlated with aversion to telling an
altruistic white lie; (iii) men are more likely than women to tell an altruistic white lie, but not to tell
a Pareto white lie.
Keywords: lying-aversion, white lies.

1 Introduction
Moral philosophers have long argued about white lies (Plato, 1997).
"""

RESULTS_PAGE = """2 Method and results
A total of 1212 subjects (59% males, mean age = 33.83) passed the comprehension questions.

Dictators on average transferred 22% of their endowment. Linear regression shows that women donated
more than men (coeff = 1.74, p = 0.0008), that older people donated slightly more (coeff = 0.03,
p = 0.048) and that education has no significant effect (coeff = 0.04, p = 0.78).

Erat and Gneezy (2012) found that 43% of people lie, t(80) = 2.10, p = 0.031. In our sample, men are
slightly more likely than women to tell an altruistic white lie (26% vs 18%). The logit regression shows
the difference (gender: coeff = 0.50, p = 0.02; age: coeff = -0.01, p = 0.85; education: coeff = -0.02,
p = 0.73).

A power analysis showed that N = 300 gives power .80 for d = 0.35, p = 0.05 two-sided.

Payoffs: option A pays 10 cents and option B pays 30 cents to the other.
"""

PAGES = [ABSTRACT_PAGE, RESULTS_PAGE]


def package(tmp_path: Path, fields: list, metadata: dict | None = None, participants: list | None = None,
            evidence: list | None = None) -> tuple[Path, Path]:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    (root / "source").mkdir(parents=True)
    nodes = [{"id": "res", "kind": "result", "fields": fields}]
    if participants is not None:
        nodes.append({"id": "sample", "kind": "participants", "fields": participants})
    (root / "study.json").write_text(json.dumps({"program": {"nodes": nodes, "evidence": evidence or []}}))
    (root / "source/paper_metadata.json").write_text(json.dumps(metadata or {}))
    return root, job


ALL_TEXT_RESULTS = [
    {"id": "sex_model", "value": {"sex": {"coeff": 1.74, "p": 0.0008}, "age": {"coeff": 0.03, "p": 0.048},
                                  "education": {"coeff": 0.04, "p": 0.78}}},
    {"id": "gender_lies", "value": {"men": 26, "women": 18, "gender": {"coeff": 0.5, "p": 0.02},
                                    "age": {"coeff": -0.01, "p": 0.85}, "education": {"coeff": -0.02, "p": 0.73}}},
    {"id": "dg_share", "value": 0.22},
]
SAMPLE = [{"id": "demographics", "value": {"male_share": 0.59, "mean_age": 33.83}}]
CLAIMS = {"abstract_results": [
    {"claim": "both altruism and cooperation are positively correlated with aversion to telling a Pareto white lie",
     "not_recorded": "the per-treatment regressions are in the tables, recorded elsewhere"},
    {"claim": "both altruism and cooperation are negatively correlated with aversion to telling an altruistic "
              "white lie", "not_recorded": "the per-treatment regressions are in the tables, recorded elsewhere"},
    {"claim": "men are more likely than women to tell an altruistic white lie",
     "recorded_in": ["res.gender_lies"]}]}


# --- what the text prints -----------------------------------------------------

def test_text_statistics_skip_other_studies_and_power_analyses() -> None:
    passages = rc.passages(PAGES)
    found = [(s["kind"], s["value"]) for p in passages for s in rc.statistics_in(p["text"])]
    assert ("p", "0.048") in found and ("estimate", "1.74") in found
    assert ("pair", "26") in found and ("pair", "18") in found  # the next sentence is this paper's own
    assert ("sex", "59") in found and ("age", "33.83") in found
    assert ("p", "0.031") not in found and ("test", "2.10") not in found  # Erat and Gneezy (2012)
    assert ("estimate", "0.35") not in found  # power analysis


def test_sex_coding_is_not_sample_make_up() -> None:
    assert not [s for s in rc.statistics_in("gender (1 = female, 0 = male) and age were coded")
                if s["kind"] == "sex"]
    assert [s["value"] for s in rc.statistics_in("(58 male, 222 female, M age = 20.9)") if s["kind"] == "sex"] \
        == ["58", "222"]


def test_unrecorded_text_results_fail(tmp_path: Path) -> None:
    root, job = package(tmp_path, [{"id": "dg_share", "value": 0.22}], CLAIMS)
    problems = rc.check(root, job, pages=PAGES)
    text = " ".join(problems)
    assert "exact p values" in text and "p = 0.048" in text
    assert "results passages print numbers none of which is recorded" in text
    assert "sex or gender" in text and "age" in text
    assert all(p.startswith("human results") for p in problems)


def test_recorded_text_results_pass(tmp_path: Path) -> None:
    root, job = package(tmp_path, ALL_TEXT_RESULTS, CLAIMS, participants=SAMPLE)
    assert rc.check(root, job, pages=PAGES) == []


def test_a_coefficient_without_its_p_value_is_not_recorded(tmp_path: Path) -> None:
    fields = json.loads(json.dumps(ALL_TEXT_RESULTS))
    for name in ("sex", "age", "education"):
        fields[0]["value"][name].pop("p")
    for name in ("gender", "age", "education"):
        fields[1]["value"][name].pop("p")
    root, job = package(tmp_path, fields, CLAIMS, participants=SAMPLE)
    report = rc.text_coverage(root, PAGES, json.loads((root / "study.json").read_text())["program"])
    coefficients = [s for p in report["passages"] for s in p["stats"] if s["value"] == "1.74"]
    assert coefficients and not coefficients[0]["recorded"]
    assert any("exact p values" in p for p in rc.check(root, job, pages=PAGES))


def test_pairs_need_both_numbers_together(tmp_path: Path) -> None:
    fields = json.loads(json.dumps(ALL_TEXT_RESULTS))
    fields[1]["value"].pop("women")
    fields.append({"id": "other", "value": 18})  # 18 somewhere else is not the comparison
    root, job = package(tmp_path, fields, CLAIMS, participants=SAMPLE)
    report = rc.text_coverage(root, PAGES, json.loads((root / "study.json").read_text())["program"])
    assert [s["recorded"] for s in report["pairs"]] == [False]


def test_results_not_recorded_clears_a_passage_and_needs_the_exact_text(tmp_path: Path) -> None:
    payoff_page = RESULTS_PAGE + "\nThe exchange rate was t = 4.5 points, SE = 0.25 cents, b = 0.75 and F(1, 2) = 3.5.\n"
    root, job = package(tmp_path, ALL_TEXT_RESULTS, CLAIMS, participants=SAMPLE)
    assert any("results passages" in p for p in rc.check(root, job, pages=[ABSTRACT_PAGE, payoff_page]))
    skip = {"page": 2, "quote": "The exchange rate was t = 4.5 points", "reason": "a design value, not a result"}
    root, job = package(tmp_path / "b", ALL_TEXT_RESULTS, {**CLAIMS, "results_not_recorded": [skip]},
                        participants=SAMPLE)
    assert rc.check(root, job, pages=[ABSTRACT_PAGE, payoff_page]) == []
    bad = {"page": 2, "quote": "a passage the paper does not print", "reason": "x"}
    root, job = package(tmp_path / "c", ALL_TEXT_RESULTS, {**CLAIMS, "results_not_recorded": [bad]},
                        participants=SAMPLE)
    problems = rc.check(root, job, pages=PAGES)
    assert any("exact text" in p for p in problems) and any("say in 'reason'" in p for p in problems)


# --- the abstract ---------------------------------------------------------------

def test_abstract_findings_are_split_at_the_enumeration() -> None:
    findings = rc.abstract_findings(rc.abstract_text(PAGES))
    assert len(findings) == 3
    assert findings[2].startswith("men are more likely than women")


def test_every_abstract_finding_needs_a_record(tmp_path: Path) -> None:
    root, job = package(tmp_path, ALL_TEXT_RESULTS, {"abstract_results": CLAIMS["abstract_results"][:2]},
                        participants=SAMPLE)
    problems = rc.check(root, job, pages=PAGES)
    assert len(problems) == 1 and "men are more likely than women" in problems[0]


def test_abstract_finding_mapped_to_an_unrelated_field_fails(tmp_path: Path) -> None:
    claims = json.loads(json.dumps(CLAIMS))
    claims["abstract_results"][2]["recorded_in"] = ["res.dg_share"]
    root, job = package(tmp_path, ALL_TEXT_RESULTS, claims, participants=SAMPLE)
    problems = rc.check(root, job, pages=PAGES)
    assert len(problems) == 1 and "is about it" in problems[0]
    claims["abstract_results"][2]["recorded_in"] = ["res.missing"]
    root, job = package(tmp_path / "b", ALL_TEXT_RESULTS, claims, participants=SAMPLE)
    assert "do not exist" in rc.check(root, job, pages=PAGES)[0]


def test_abstract_finding_matched_through_cited_quote(tmp_path: Path) -> None:
    fields = json.loads(json.dumps(ALL_TEXT_RESULTS))
    fields[1]["id"] = "logit_awl"
    fields[1]["evidenceIds"] = ["ev-gender"]
    claims = json.loads(json.dumps(CLAIMS))
    claims["abstract_results"][2]["recorded_in"] = ["res.logit_awl"]
    evidence = [{"id": "ev-gender", "quote": "men are slightly more likely than women to tell an altruistic white lie"}]
    root, job = package(tmp_path, fields, claims, participants=SAMPLE, evidence=evidence)
    assert rc.check(root, job, pages=PAGES) == []


def test_paper_without_an_abstract_heading_uses_its_first_long_paragraph() -> None:
    long = " ".join(["Status quo bias refers to the preference for a previously chosen option."] * 8)
    page = ("Meta-Psychology, 2021\n\n" + long + " We found strong empirical support for the status quo bias in "
            "three scenarios.\n\nKeywords: status quo bias\n")
    assert rc.abstract_findings(rc.abstract_text([page])) == [
        "We found strong empirical support for the status quo bias in three scenarios."]


def test_accepted_model_sync_and_missing_pdf_skip(tmp_path: Path) -> None:
    root, job = package(tmp_path, [], {})
    (job / "studio_request.json").write_text(json.dumps({"purpose": "accepted-model-sync"}))
    assert rc.check(root, job, pages=PAGES) == []
    root, job = package(tmp_path / "b", [], {})
    assert rc.check(root, job) == []  # no input/paper.pdf


# --- supplement tables ----------------------------------------------------------

SUPPLEMENT_LAYOUT = """Table S1.
Descriptive statistics for the Receptivity Scale (Study 1). N = 258.

  Item                                         Mean     SD     Skew     ITC
  Hidden meaning transforms beauty.            2.99    1.08    -0.09    .454
  Good health imparts reality.                 2.51    1.01     0.37    .437
  Wholeness quiets infinite phenomena.         2.71    1.09     0.16    .552
"""


def supplement_job(tmp_path: Path, monkeypatch, tables: list | None, field_value) -> tuple[Path, Path]:
    root, job = package(tmp_path, [{"id": "s1", "value": field_value}], {"paper_tables": tables or []})
    (job / "input/external").mkdir(parents=True)
    (job / "input/external/supp.pdf").write_bytes(b"%PDF-1.4")
    (job / "input/external/fetch_log.json").write_text(json.dumps([{"path": "input/external/supp.pdf"}]))
    monkeypatch.setattr(pt, "page_texts", lambda pdf, layout: [SUPPLEMENT_LAYOUT])
    return root, job


def test_fetched_supplement_tables_must_be_listed_and_recorded(tmp_path: Path, monkeypatch) -> None:
    root, job = supplement_job(tmp_path, monkeypatch, None, 0)
    problems = pt.check_supplement_tables(root, job)
    assert problems and "Table S1" in problems[0] and "input/external/supp.pdf" in problems[0]
    entry = {"table": "S1", "source": "input/external/supp.pdf", "kind": "result", "recorded_in": "res.s1"}
    root, job = supplement_job(tmp_path / "b", monkeypatch, [entry], {"rows": [[2.99, 1.08]]})
    assert "holds only" in pt.check_supplement_tables(root, job)[0]
    cells = {"rows": [[2.99, 1.08, -0.09, 0.454], [2.51, 1.01, 0.37, 0.437], [2.71, 1.09, 0.16, 0.552]]}
    root, job = supplement_job(tmp_path / "c", monkeypatch, [entry], cells)
    assert pt.check_supplement_tables(root, job) == []
    other = {"table": "S1", "kind": "not-in-paper", "what": "in the supplement"}
    root, job = supplement_job(tmp_path / "d", monkeypatch, [other], 0)
    assert "kind must be 'result'" in pt.check_supplement_tables(root, job)[0]


# --- evaluation links -----------------------------------------------------------

def linked(field_id: str, value, output: str | None = None, **link) -> tuple[str, dict]:
    evaluation = dict(link)
    if output is not None:
        evaluation["output"] = output
    return f"res.{field_id}", {"id": field_id, "value": value, "extensions": {"evaluation": evaluation}}


PAIRED_T = {"study2": {"participant": {"n": 30, "mean_a": 2.4, "mean_b": 2.8, "t": 3.1, "df": 29, "p": 0.004},
                       "pending": {"status": "not_ready", "reason": "fewer than two participants"}}}


def test_correlation_linked_to_a_paired_t_fails() -> None:
    problems = el.check([linked("scales_correlation", 0.88, "study2.participant")], PAIRED_T, None)
    assert len(problems) == 1 and "records r" in problems[0] and "holds no r" in problems[0]
    assert el.check([linked("participant_test", {"t": 10.6, "df": 187}, "study2.participant")], PAIRED_T, None) == []


def test_not_ready_output_needs_a_declaration_that_holds_the_kind() -> None:
    field = linked("alpha", {"BSR": 0.93, "Chopra": 0.89}, "study2.pending")
    assert "Declare it" in el.check([field], PAIRED_T, None)[0]
    assert "holds no alpha" in el.check([field], PAIRED_T, {"study2.pending": "t df p"})[0]
    assert el.check([field], PAIRED_T, {"study2.*": {"kinds": ["alpha"]}}) == []


def test_declaration_contradicted_by_the_computed_output_fails() -> None:
    problems = el.check([linked("item_test", {"t": 3.98}, "study2.participant")], PAIRED_T,
                        {"study2.participant": "t r"})
    assert any("declares r" in p for p in problems)


def test_share_linked_to_cents_fails() -> None:
    output = {"descriptives": {"dg_mean_cents": 4.4, "dg_mean_share": 0.22}}
    field = linked("dg_mean_share_of_endowment", 0.22, "descriptives.dg_mean_cents")
    assert "which is in money" in el.check([field], output, None)[0]
    assert el.check([linked("dg_mean_share_of_endowment", 0.22, "descriptives.dg_mean_share")], output, None) == []


def test_two_values_linked_to_one_fail_and_cells_pass() -> None:
    output = {"lie": {"AWL": {"n": 40, "lie_share": 0.2}, "PWL": {"n": 40, "lie_share": 0.8}}}
    value = {"AWL_lie_share": 0.23, "PWL_lie_share": 0.83}
    assert "records 2 values of share" in el.check([linked("rates", value, "lie.AWL")], output, None)[0]
    cells = {"AWL_lie_share": "lie.AWL", "PWL_lie_share": {"output": "lie.PWL"}}
    assert el.check([linked("rates", value, cells=cells)], output, None) == []
    assert "not a path" in el.check([linked("rates", value, cells={"XYZ": "lie.AWL"})], output, None)[0]


TABLE = {"title": "Table 1", "columns": ["I", "IV"],
         "rows": {"AWL": {"I": [-1.14, 0.61]}, "PWL": {"IV": [1.43, 0.68]}, "N": {"I": 357, "IV": 340}}}
MODELS = {"dg": {"AWL": {"n": 357, "coef": -1.1, "se": 0.6, "p": 0.06}, "PWL": {"n": 340, "coef": 1.4, "se": 0.7}}}


def test_table_linked_to_one_model_fails_and_cell_links_pass() -> None:
    assert "links a table of 3 rows" in el.check([linked("table", TABLE, "dg.AWL")], MODELS, None)[0]
    cells = {"rows.AWL.I": "dg.AWL", "rows.PWL.IV": "dg.PWL"}
    assert el.check([linked("table", TABLE, not_ready="columns II, III need the demographics",
                            cells=cells)], MODELS, None) == []
    rows_output = {"exp": {"table": {"rows": [{"chi2": 1.0}, {"chi2": 2.0}, {"chi2": 3.0}]}}}
    table = {"rows": [{"chi2": 22.1}, {"chi2": 9.5}, {"chi2": 4.4}]}
    assert el.check([linked("table", table, "exp.table")], rows_output, None) == []


def row(option: str, nsq: int) -> dict:
    return {"option": option, "status_quo": {"chosen": 18, "n": 63}, "neutral": {"chosen": 8, "n": 65},
            "non_status_quo": {"chosen": 20, "n": nsq}, "chi2": 11.8, "or": 3.4, "h": 0.46}


def test_a_row_with_a_smaller_n_needs_the_exclusion() -> None:
    table = {"rows": [row("70A30H", 188), row("30A70H", 188), row("60A40H", 188), row("50A50H", 125),
                      row("Red", 188)]}
    declarations = {"exp2.table3": {"kinds": "chi2 p or h n", "rows": True}}
    output = {"exp2": {"table3": {"rows": [{"status": "not_ready", "reason": "too few"}] * 5}}}
    problems = el.check([linked("table", table, "exp2.table3")], output, declarations)
    assert len(problems) == 1 and "row 50A50H" in problems[0] and "(other rows 188)" in problems[0]
    note = "framing A had no 50/50 option in phase 2; evaluate() leaves it out of this row"
    assert el.check([linked("table", table, "exp2.table3", exclusion=note)], output, declarations) == []
    # Rows of different participants (no common n) are not judged.
    varied = {"rows": [row("a", 188), row("b", 150), row("c", 120), row("d", 90)]}
    for item, n in zip(varied["rows"], (63, 50, 40, 30)):
        item["status_quo"]["n"] = n
    assert el.check([linked("table", varied, "exp2.table3")], output, declarations) == []


def test_summary_starts_with_analyses() -> None:
    assert el.summary(["x"])[0].startswith("analyses:")
    assert el.summary([]) == []


# --- repair guard: verified quotes ---------------------------------------------------

def test_repair_may_add_evidence_quoted_word_for_word(tmp_path: Path, monkeypatch) -> None:
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    (job / "input/paper.pdf").write_bytes(b"%PDF")
    from agent_pipeline import check_sources
    monkeypatch.setattr(check_sources, "file_text", lambda path: check_sources.normalise(RESULTS_PAGE))
    verified = source_quote_matcher(job)
    record = {"id": "ev-gender", "path": "input/paper.pdf", "locator": {"page": 2},
              "quote": "men are slightly more likely than women to tell an altruistic white lie (26% vs 18%)"}
    pre = {"study.json": {"program": {"evidence": [{"id": "ev-1", "quote": "A total of 1212 subjects"}], "nodes": []}}}
    field = {"id": "gender", "value": {"men": 26, "women": 18}, "origin": "reported", "state": "confirmed",
             "evidenceIds": ["ev-gender"]}
    post = {"study.json": {"program": {"evidence": pre["study.json"]["program"]["evidence"] + [record],
                                       "nodes": [{"id": "res", "kind": "result", "fields": [field]}]}}}
    assert compare_packages(pre, post, None, verified)["passed"]
    assert not compare_packages(pre, post)["passed"]  # no verifier: refused as before
    invented = dict(record, quote="men are much more likely than women to lie (40% vs 10%)")
    post["study.json"]["program"]["evidence"][-1] = invented
    assert not compare_packages(pre, post, None, verified)["passed"]
    stitched = dict(record, quote="men are slightly more likely ... (26% vs 18%)")
    post["study.json"]["program"]["evidence"][-1] = stitched
    assert not compare_packages(pre, post, None, verified)["passed"]
    outside = dict(record, path="../elsewhere.pdf")
    post["study.json"]["program"]["evidence"][-1] = outside
    assert not compare_packages(pre, post, None, verified)["passed"]
