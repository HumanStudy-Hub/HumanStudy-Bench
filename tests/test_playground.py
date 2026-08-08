import json
from pathlib import Path

import pytest

from playground.analysis import agent_effect_from_t, build_analysis, human_effect_from_reported, normalise_tests
from playground.default_charts import build_charts
from playground.personas import PersonaError, describe_mix, normalise_group, sample_profiles
from playground.run_playground import render_prompt
from playground.settings import trial_limits
from playground.study_loader import StudyNotRunnable, study_dir
from playground.validate_charts import ChartError, validate


def write_ground_truth(tmp_path: Path) -> Path:
    study = tmp_path / "study_900"
    (study / "source").mkdir(parents=True)
    (study / "source" / "ground_truth.json").write_text(json.dumps({
        "studies": [{
            "study_id": "Study 1",
            "findings": [{
                "finding_id": "F1",
                "main_hypothesis": "Choosers estimate their own choice as more common.",
                "statistical_tests": [{
                    "reported_statistics": "F(1, 312) = 49.1, p < .001",
                    "expected_direction": "positive",
                }],
            }],
        }],
    }))
    return study


def test_human_effect_is_converted_from_the_reported_statistic() -> None:
    assert human_effect_from_reported("F(1, 312) = 49.1, p < .001") == pytest.approx(0.7934, abs=1e-3)
    assert human_effect_from_reported("t(78) = 2.66, p = .01") == pytest.approx(0.6023, abs=1e-3)
    # A multi-numerator F has no single-d equivalent and must not be guessed at.
    assert human_effect_from_reported("F(3, 120) = 4.2") is None
    assert human_effect_from_reported("chi2(1) = 6.1") is None


def test_agent_effect_is_derived_from_the_t_statistic() -> None:
    assert agent_effect_from_t(2.0, 30, 30) == pytest.approx(0.5164, abs=1e-3)
    assert agent_effect_from_t(None, 30, 30) is None


def test_raw_evaluator_output_is_normalised_against_the_paper(tmp_path: Path) -> None:
    study = write_ground_truth(tmp_path)
    evaluation = {"test_results": [{
        "study_id": "Study 1",
        "finding_id": "F1",
        "scenario": "supermarket_story",
        "t_stat": 2.4,
        "p_value": 0.02,
        "n_agent_1": 30,
        "n_agent_2": 30,
        "significant": True,
        "direction_match": True,
        "human_significant": True,
        "replication": True,
    }]}

    rows = normalise_tests(evaluation, study)

    assert len(rows) == 1
    row = rows[0]
    assert row["human_effect"] == pytest.approx(0.7934, abs=1e-3)
    assert row["agent_effect"] == pytest.approx(0.6197, abs=1e-3)
    assert row["reported_statistics"] == "F(1, 312) = 49.1, p < .001"
    assert row["replicated"] is True


def test_enriched_evaluator_output_keeps_its_own_effect_sizes(tmp_path: Path) -> None:
    study = write_ground_truth(tmp_path)
    evaluation = {"test_results": [{
        "study_id": "Study 1",
        "finding_id": "F1",
        "human_effect_d": 0.9,
        "agent_effect_d": 0.2,
        "p_value_agent": 0.4,
        "p_value_human": 0.001,
        "is_significant_agent": False,
        "is_significant_human": True,
        "direction_match": True,
    }]}

    row = normalise_tests(evaluation, study)[0]

    assert row["human_effect"] == 0.9
    assert row["agent_effect"] == 0.2
    # Not significant for the agent, so the published finding was not reproduced.
    assert row["replicated"] is False


def test_analysis_summarises_replication(tmp_path: Path) -> None:
    study = write_ground_truth(tmp_path)
    evaluation = {"test_results": [
        {"study_id": "Study 1", "finding_id": "F1", "human_effect_d": 0.8, "agent_effect_d": 0.6, "is_significant_agent": True, "is_significant_human": True, "direction_match": True},
        {"study_id": "Study 1", "finding_id": "F1", "human_effect_d": 0.8, "agent_effect_d": -0.4, "is_significant_agent": True, "is_significant_human": True, "direction_match": False},
    ]}

    analysis = build_analysis(evaluation, study, {"id": "r1", "studyId": "study_900"}, {"participants": 20})

    assert analysis["summary"]["replicatedTests"] == 1
    assert analysis["summary"]["scoredTests"] == 2
    assert analysis["summary"]["replicationRate"] == pytest.approx(0.5)
    assert analysis["summary"]["directionMatchRate"] == pytest.approx(0.5)
    assert analysis["participants"] == 20


def test_default_charts_are_valid_and_skip_what_the_data_cannot_support(tmp_path: Path) -> None:
    study = write_ground_truth(tmp_path)
    evaluation = {"test_results": [
        {"study_id": "Study 1", "finding_id": "F1", "human_effect_d": 0.8, "agent_effect_d": 0.6, "is_significant_agent": True, "is_significant_human": True, "direction_match": True},
    ]}
    analysis = build_analysis(evaluation, study, {"id": "r1"}, {})

    charts = build_charts(analysis)

    assert validate(charts) == ["effect-scatter", "effect-bars", "replication-breakdown"]

    # The agent produced no measurable effect here. The scatter needs both sides
    # so it is dropped rather than plotted against an invented zero; the bar
    # chart still shows the human effect the agent failed to produce.
    one_sided = build_analysis({"test_results": [{"study_id": "Study 1", "finding_id": "F1", "direction_match": False}]}, study, {"id": "r1"}, {})
    assert [chart["id"] for chart in build_charts(one_sided)["charts"]] == ["effect-bars", "replication-breakdown"]

    # With nothing measured on either side there is no chart to draw at all.
    nothing = build_analysis({"test_results": []}, study, {"id": "r1"}, {})
    assert build_charts(nothing)["charts"] == []


def test_charts_from_the_agent_must_be_plain_plottable_data() -> None:
    def chart(trace: dict) -> dict:
        return {"charts": [{"id": "a", "title": "t", "description": "d", "plotly": {"data": [trace]}}]}

    assert validate(chart({"type": "bar", "x": ["a"], "y": [1]})) == ["a"]
    with pytest.raises(ChartError):
        validate(chart({"type": "pie", "values": [1, 2]}))
    with pytest.raises(ChartError):
        validate(chart({"type": "bar", "x": ["a"], "y": [1], "meta": "<img onerror=alert(1)>"}))
    with pytest.raises(ChartError):
        validate(chart({"type": "bar"}))
    with pytest.raises(ChartError):
        validate({"charts": []})


def test_run_budget_depends_on_whose_key_pays() -> None:
    shared_total, shared_per_scenario = trial_limits(has_own_key=False)
    own_total, own_per_scenario = trial_limits(has_own_key=True)
    assert own_total > shared_total
    assert own_per_scenario > shared_per_scenario


def test_study_ids_cannot_escape_the_studies_directory(tmp_path: Path) -> None:
    with pytest.raises(StudyNotRunnable):
        study_dir(tmp_path, "../secrets")
    with pytest.raises(StudyNotRunnable):
        study_dir(tmp_path, "study_999")


def test_persona_shares_split_participants_by_largest_remainder() -> None:
    group = {"segments": [
        {"id": "a", "label": "A", "share": 30},
        {"id": "b", "label": "B", "share": 70},
    ]}
    mix = {entry["id"]: entry["count"] for entry in describe_mix(group, 10)}
    assert mix == {"a": 3, "b": 7}
    # Shares are relative, so 30/70 and 0.3/0.7 describe the same population.
    assert describe_mix({"segments": [
        {"id": "a", "label": "A", "share": 0.3},
        {"id": "b", "label": "B", "share": 0.7},
    ]}, 10) == describe_mix(group, 10)


def test_every_segment_keeps_at_least_one_participant() -> None:
    group = {"segments": [
        {"id": "a", "label": "A", "share": 0.98},
        {"id": "b", "label": "B", "share": 0.02},
    ]}
    counts = [entry["count"] for entry in describe_mix(group, 10)]
    assert counts == [9, 1]
    assert sum(counts) == 10


def test_personas_are_sampled_within_their_ranges_and_are_reproducible() -> None:
    group = {"segments": [
        {"id": "nurses", "label": "Nurses", "share": 0.5, "age": {"min": 30, "max": 40}, "gender": "female", "persona": "You are a nurse."},
        {"id": "students", "label": "Students", "share": 0.5, "age": {"min": 18, "max": 22}},
    ]}

    profiles = sample_profiles(group, 20, seed=11)

    assert len(profiles) == 20
    assert [profile["participant_id"] for profile in profiles] == list(range(20))
    nurses = [profile for profile in profiles if profile["persona_segment"] == "nurses"]
    assert len(nurses) == 10
    assert all(30 <= profile["age"] <= 40 for profile in nurses)
    assert all(profile["gender"] == "female" for profile in nurses)
    assert all(profile["persona"] == "You are a nurse." for profile in nurses)
    assert all(18 <= profile["age"] <= 22 for profile in profiles if profile["persona_segment"] == "students")
    # A saved group plus a seed reproduces the same participants exactly.
    assert sample_profiles(group, 20, seed=11) == profiles
    assert sample_profiles(group, 20, seed=12) != profiles


def test_a_group_fits_a_run_of_any_size() -> None:
    group = {"segments": [
        {"id": "a", "label": "A", "share": 0.25},
        {"id": "b", "label": "B", "share": 0.75},
    ]}
    for total in (2, 7, 28, 600):
        assert len(sample_profiles(group, total, seed=3)) == total


def test_unusable_persona_groups_are_rejected() -> None:
    with pytest.raises(PersonaError):
        normalise_group({"segments": []})
    with pytest.raises(PersonaError):
        normalise_group({"segments": [{"label": "A", "share": 0}]})
    with pytest.raises(PersonaError):
        normalise_group({"segments": [{"label": "A", "age": {"min": 2, "max": 5}}]})
    with pytest.raises(PersonaError):
        normalise_group({"segments": [{"id": "a", "label": "A"}, {"id": "a", "label": "B"}]})
    with pytest.raises(PersonaError):
        normalise_group({"segments": [{"label": "A", "gender": {"female": 0, "male": 0}}]})


def test_age_and_gender_accept_a_single_value_or_a_spread() -> None:
    group = normalise_group({"segments": [{"label": "A", "age": 34, "gender": "female"}]})
    assert group["segments"][0]["age"] == {"min": 34, "max": 34}
    assert group["segments"][0]["gender"] == {"female": 1.0}
    weighted = normalise_group({"segments": [{"label": "A", "gender": {"female": 80, "male": 20}}]})
    assert weighted["segments"][0]["gender"] == {"female": 0.8, "male": 0.2}


def test_a_custom_prompt_is_filled_in_per_agent() -> None:
    profile = {"age": 34, "persona": "You are a nurse.", "gender": "female"}
    assert render_prompt("You are {{age}}. {{persona}}", profile) == "You are 34. You are a nurse."
    # A placeholder with nothing behind it leaves no stray gap.
    assert render_prompt("Age {{age}}. {{missing}} Done.", profile) == "Age 34. Done."
