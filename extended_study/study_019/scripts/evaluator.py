"""Fail-closed environment evaluator for the two social-influence tasks."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


STUDY_1_ID = "study_1_urn_scenarios"
STUDY_2_ID = "study_2_medical_authority_scenarios"
EXPECTED_COUNTS = {STUDY_1_ID: 24, STUDY_2_ID: 40}
GROUND_TRUTH_PATH = Path(__file__).parents[1] / "source" / "ground_truth.json"
GROUND_TRUTH = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def _reported_findings(section: str) -> Dict[str, float]:
    return {
        str(finding["metric"]): float(finding["reported_value"])
        for finding in GROUND_TRUTH[section]["findings"]
        if finding.get("reported_value") is not None
    }


REPORTED = {
    STUDY_1_ID: _reported_findings("study_1"),
    STUDY_2_ID: _reported_findings("study_2"),
}
STUDY_1_PROBABILITY_JUDGMENT = GROUND_TRUTH["study_1"][
    "probability_judgment_by_bayesian_probability"
]
STUDY_2_AUTHORITY = GROUND_TRUTH["study_2"]["authority_condition_means"]
BEHAVIORAL_VALIDATION = GROUND_TRUTH["validation_criteria"][
    "behavioral_validation"
]
MINIMUM_PARTICIPANTS_PER_STUDY = int(
    BEHAVIORAL_VALIDATION["minimum_participants_per_sub_study"]
)
BROAD_CHOICE_MAE_TOLERANCE = float(
    BEHAVIORAL_VALIDATION["broad_choice_mae_tolerance"]
)
STUDY_1_PROBABILITY_JUDGMENT_MAE_TOLERANCE = float(
    BEHAVIORAL_VALIDATION["study_1_probability_judgment_mae_tolerance"]
)
STUDY_2_AUTHORITY_CHOICE_MAE_TOLERANCE = float(
    BEHAVIORAL_VALIDATION["study_2_authority_choice_mae_tolerance"]
)
STUDY_2_AUTHORITY_PROBABILITY_JUDGMENT_MAE_TOLERANCE = float(
    BEHAVIORAL_VALIDATION[
        "study_2_authority_probability_judgment_mae_tolerance"
    ]
)


def _load_material_index() -> Dict[str, Dict[str, Dict[str, Any]]]:
    path = Path(__file__).parents[1] / "source" / "materials" / "scenarios.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        STUDY_1_ID: {
            scenario["scenario_id"]: scenario
            for scenario in payload["study_1"]["scenarios"]
        },
        STUDY_2_ID: {
            scenario["scenario_id"]: scenario
            for scenario in payload["study_2"]["scenarios"]
        },
    }


def _flatten(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        response
        for participant in results.get("individual_data", [])
        for response in participant.get("responses", [])
    ]


def _requested_sub_studies(
    results: Dict[str, Any],
) -> Tuple[Tuple[str, ...], bool]:
    requested = results.get("selected_sub_studies")
    if requested is None:
        return tuple(EXPECTED_COUNTS), True
    if not isinstance(requested, list) or not requested:
        return tuple(EXPECTED_COUNTS), False

    requested_set = {str(value) for value in requested}
    if not requested_set.issubset(set(EXPECTED_COUNTS)):
        return tuple(EXPECTED_COUNTS), False
    return (
        tuple(
            sub_study_id
            for sub_study_id in EXPECTED_COUNTS
            if sub_study_id in requested_set
        ),
        True,
    )


def _test(test_id: str, passed: bool, **details: Any) -> Dict[str, Any]:
    return {"test_id": test_id, "passed": bool(passed), **details}


def _mean_rate(
    rows: Sequence[Dict[str, Any]],
    predicate: Any,
) -> Optional[float]:
    return mean(bool(predicate(row)) for row in rows) if rows else None


def _prompt_contains_visible_evidence(
    sub_study_id: str,
    prompt: str,
    source_signature: Dict[str, Any],
) -> bool:
    lower = prompt.lower()
    if sub_study_id == STUDY_1_ID:
        return (
            str(source_signature["private_ball"]).lower() in lower
            and all(
                f"predicted urn {choice.lower()}" in lower
                for choice in source_signature["previous_decisions"]
            )
        )
    return (
        str(source_signature["private_symptom"]).lower() in lower
        and all(
            str(item["role"]).lower() in lower
            and str(item["diagnosis"]).lower() in lower
            for item in source_signature["previous_diagnoses"]
        )
    )


def _execution_checks(
    results: Dict[str, Any],
    responses: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    participants = results.get("individual_data", [])
    material_index = _load_material_index()
    tests: List[Dict[str, Any]] = []
    selected_sub_studies, scope_definition_valid = _requested_sub_studies(results)

    assignment_counts = {
        sub_study_id: sum(
            participant.get("sub_study_id") == sub_study_id
            for participant in participants
        )
        for sub_study_id in EXPECTED_COUNTS
    }
    unselected = set(EXPECTED_COUNTS) - set(selected_sub_studies)
    if len(selected_sub_studies) == 1:
        assignments_valid = (
            scope_definition_valid
            and bool(participants)
            and assignment_counts[selected_sub_studies[0]] == len(participants)
            and all(assignment_counts[sub_study_id] == 0 for sub_study_id in unselected)
        )
    else:
        assignments_valid = (
            scope_definition_valid
            and len(participants) >= 2
            and len(participants) % 2 == 0
            and assignment_counts[STUDY_1_ID] == assignment_counts[STUDY_2_ID]
        )
    tests.append(
        _test(
            "selected_sub_study_assignment",
            assignments_valid,
            requested=results.get("selected_sub_studies"),
            participants=len(participants),
            assignment_counts=assignment_counts,
        )
    )

    schedule_valid = bool(participants)
    schedule_details: Dict[str, Any] = {}
    source_order_seen = 0
    for participant in participants:
        sub_study_id = participant.get("sub_study_id")
        participant_responses = participant.get("responses", [])
        expected_count = EXPECTED_COUNTS.get(sub_study_id)
        ids = [
            response.get("trial_info", {}).get("scenario_id")
            for response in participant_responses
        ]
        presented = [
            response.get("trial_info", {}).get("presented_trial_number")
            for response in participant_responses
        ]
        expected_ids = set(material_index.get(sub_study_id, {}))
        valid = (
            expected_count is not None
            and len(participant_responses) == expected_count
            and len(set(ids)) == expected_count
            and set(ids) == expected_ids
            and presented == list(range(1, expected_count + 1))
        )
        if ids == sorted(ids):
            source_order_seen += 1
        if not valid:
            schedule_valid = False
        schedule_details[str(participant.get("participant_id"))] = {
            "sub_study_id": sub_study_id,
            "responses": len(participant_responses),
            "unique_scenarios": len(set(ids)),
            "valid": valid,
        }
    tests.append(
        _test(
            "complete_source_scenario_set_once_per_participant",
            schedule_valid,
            details=schedule_details,
        )
    )
    tests.append(
        _test(
            "participant_level_scenario_order_randomized",
            bool(participants)
            and source_order_seen < len(participants)
            and all(
                response.get("trial_info", {}).get("scenario_order_randomized") is True
                for response in responses
            ),
            participants_in_source_order=source_order_seen,
        )
    )

    source_valid = bool(responses)
    source_failures: List[str] = []
    for response in responses:
        info = response.get("trial_info", {})
        sub_study_id = info.get("sub_study_id")
        scenario_id = info.get("scenario_id")
        expected = material_index.get(sub_study_id, {}).get(scenario_id)
        if not expected or not (
            info.get("raw_variable_id") == expected["raw_variable_id"]
            and info.get("article_scenario_id") == expected["article_scenario_id"]
            and info.get("material_fingerprint") == expected["material_fingerprint"]
            and info.get("source_signature") == expected["source_signature"]
            and info.get("source_material_verified") is True
        ):
            source_valid = False
            source_failures.append(str(scenario_id))
    tests.append(
        _test(
            "stored_trials_match_compiled_source_material",
            source_valid,
            failures=source_failures[:20],
        )
    )

    valid_responses = bool(responses)
    for response in responses:
        info = response.get("trial_info", {})
        sub_study_id = info.get("sub_study_id")
        confidence = info.get("confidence")
        choice = response.get("response")
        allowed = (
            {"A", "B"}
            if sub_study_id == STUDY_1_ID
            else {"appendicitis", "sigmoid diverticulitis"}
        )
        if (
            choice not in allowed
            or not isinstance(confidence, int)
            or not 50 <= confidence <= 100
        ):
            valid_responses = False
    tests.append(
        _test(
            "valid_binary_decisions_and_confidence_range",
            valid_responses,
            responses=len(responses),
        )
    )

    instruction_timing = bool(participants)
    for participant in participants:
        flags = [
            response.get("trial_info", {}).get("initial_instructions_shown") is True
            for response in participant.get("responses", [])
        ]
        if not flags or flags[0] is not True or any(flags[1:]):
            instruction_timing = False
    tests.append(
        _test(
            "instructions_shown_once_before_first_scenario",
            instruction_timing,
        )
    )

    visible_evidence = bool(responses)
    no_hidden_labels = bool(responses)
    forbidden = (
        "posterior probability",
        "bayesian choice",
        "reported value",
        "raw variable",
        "human raw data",
        "material fingerprint",
    )
    for response in responses:
        info = response.get("trial_info", {})
        prompt = str(info.get("agent_visible_prompt", ""))
        if not _prompt_contains_visible_evidence(
            str(info.get("sub_study_id")),
            prompt,
            info.get("source_signature", {}),
        ):
            visible_evidence = False
        if any(token in prompt.lower() for token in forbidden):
            no_hidden_labels = False
    tests.append(
        _test(
            "all_participant_facing_evidence_present",
            visible_evidence,
        )
    )
    tests.append(
        _test(
            "source_answers_and_outcomes_absent_from_prompts",
            no_hidden_labels,
        )
    )

    no_feedback = bool(responses) and all(
        response.get("trial_info", {}).get("answer_revealed_before_response") is False
        and response.get("trial_info", {}).get("feedback_provided") is False
        for response in responses
    )
    tests.append(
        _test(
            "no_trial_level_answer_feedback",
            no_feedback,
        )
    )

    execution_score = sum(test["passed"] for test in tests) / len(tests)
    return execution_score, tests


def _posterior_group(response: Dict[str, Any]) -> str:
    confidence = float(response["trial_info"]["bayesian_confidence"])
    return f"posterior_{confidence:.2f}"


def _probability_judgment_for_target(row: Dict[str, Any]) -> float:
    confidence = row["trial_info"]["confidence"] / 100.0
    bayesian_choice = row["trial_info"]["bayesian_choice"]
    if bayesian_choice is None:
        return confidence
    return (
        confidence
        if row["response"] == bayesian_choice
        else 1.0 - confidence
    )


def _mean_probability_judgment(
    rows: Sequence[Dict[str, Any]],
) -> Optional[float]:
    return (
        mean(_probability_judgment_for_target(row) for row in rows)
        if rows
        else None
    )


def _broad_observed(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    non_ties = [
        row for row in rows if row["trial_info"]["bayesian_choice"] is not None
    ]
    private_supported = [
        row
        for row in non_ties
        if row["trial_info"]["bayesian_choice"]
        == row["trial_info"]["private_information_favors"]
    ]
    cascades = [row for row in rows if row["trial_info"]["cascade_scenario"]]
    ties = [row for row in rows if row["trial_info"]["indifference_scenario"]]
    observed: Dict[str, Optional[float]] = {
        "bayesian_choice_rate_non_indifference": _mean_rate(
            non_ties,
            lambda row: row["response"] == row["trial_info"]["bayesian_choice"],
        ),
        "bayesian_choice_rate_when_private_supported": _mean_rate(
            private_supported,
            lambda row: row["response"] == row["trial_info"]["bayesian_choice"],
        ),
        "cascade_choice_rate": _mean_rate(
            cascades,
            lambda row: row["response"] == row["trial_info"]["bayesian_choice"],
        ),
        "private_choice_rate_at_bayesian_indifference": _mean_rate(
            ties,
            lambda row: row["response"]
            == row["trial_info"]["private_information_favors"],
        ),
    }
    director_rows = [
        row
        for row in rows
        if row["trial_info"].get("medical_director_diagnosis") is not None
    ]
    if director_rows:
        observed["medical_director_alignment_rate"] = _mean_rate(
            director_rows,
            lambda row: row["response"]
            == row["trial_info"]["medical_director_diagnosis"],
        )
    return observed


def _study_1_probability_judgment_observed(
    rows: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    return {
        group: _mean_probability_judgment(
            [row for row in rows if _posterior_group(row) == group]
        )
        for group in STUDY_1_PROBABILITY_JUDGMENT
    }


def _study_2_authority_observed(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    observed: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for group, reported_metrics in STUDY_2_AUTHORITY.items():
        observed[group] = {
            "private_choice_rate": {},
            "probability_judgment": {},
        }
        for metric, condition_values in reported_metrics.items():
            for condition in condition_values:
                cell_rows = [
                    row
                    for row in rows
                    if _posterior_group(row) == group
                    and row["trial_info"]["authority_condition"] == condition
                ]
                if metric == "private_choice_rate":
                    value = _mean_rate(
                        cell_rows,
                        lambda row: row["response"]
                        == row["trial_info"]["private_information_favors"],
                    )
                else:
                    value = _mean_probability_judgment(cell_rows)
                observed[group][metric][condition] = value
    return observed


def _alignment_test(
    test_id: str,
    comparisons: List[Dict[str, Any]],
    *,
    tolerance: float,
) -> Dict[str, Any]:
    missing = [
        comparison["metric"]
        for comparison in comparisons
        if comparison["observed"] is None
    ]
    errors = [
        abs(float(comparison["observed"]) - float(comparison["reported"]))
        for comparison in comparisons
        if comparison["observed"] is not None
    ]
    mae = mean(errors) if errors else None
    return _test(
        test_id,
        not missing and mae is not None and mae <= tolerance,
        mean_absolute_error=mae,
        tolerance=tolerance,
        missing=missing,
        comparisons=comparisons,
    )


def _behavioral_diagnostics(
    participants: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    selected_sub_studies: Sequence[str],
) -> Tuple[
    Dict[str, Any],
    Optional[float],
    bool,
    bool,
    List[Dict[str, Any]],
]:
    by_study = {
        sub_study_id: [
            response
            for response in responses
            if response.get("trial_info", {}).get("sub_study_id") == sub_study_id
        ]
        for sub_study_id in EXPECTED_COUNTS
    }
    participant_counts = {
        sub_study_id: sum(
            participant.get("sub_study_id") == sub_study_id
            for participant in participants
        )
        for sub_study_id in EXPECTED_COUNTS
    }
    sample_test = _test(
        "minimum_original_behavioral_sample",
        all(
            participant_counts[sub_study_id] >= MINIMUM_PARTICIPANTS_PER_STUDY
            for sub_study_id in selected_sub_studies
        ),
        required_per_sub_study=MINIMUM_PARTICIPANTS_PER_STUDY,
        required_sub_studies=list(selected_sub_studies),
        observed=participant_counts,
    )

    broad_observed = {
        sub_study_id: _broad_observed(rows)
        for sub_study_id, rows in by_study.items()
        if sub_study_id in selected_sub_studies
    }
    broad_comparisons = [
        {
            "metric": f"{sub_study_id}.{metric}",
            "observed": broad_observed[sub_study_id].get(metric),
            "reported": reported,
        }
        for sub_study_id, reported_metrics in REPORTED.items()
        if sub_study_id in selected_sub_studies
        for metric, reported in reported_metrics.items()
    ]
    broad_test = _alignment_test(
        "broad_choice_alignment",
        broad_comparisons,
        tolerance=BROAD_CHOICE_MAE_TOLERANCE,
    )

    alignment_tests = [broad_test]
    diagnostics: Dict[str, Any] = {
        "broad_choice": {
            sub_study_id: {
                "observed": broad_observed[sub_study_id],
                "reported": REPORTED[sub_study_id],
            }
            for sub_study_id in selected_sub_studies
        },
    }

    if STUDY_1_ID in selected_sub_studies:
        study_1_probability_judgment = _study_1_probability_judgment_observed(
            by_study[STUDY_1_ID]
        )
        study_1_probability_judgment_test = _alignment_test(
            "study_1_probability_judgment_alignment",
            [
                {
                    "metric": group,
                    "observed": study_1_probability_judgment.get(group),
                    "reported": reported,
                }
                for group, reported in STUDY_1_PROBABILITY_JUDGMENT.items()
            ],
            tolerance=STUDY_1_PROBABILITY_JUDGMENT_MAE_TOLERANCE,
        )
        alignment_tests.append(study_1_probability_judgment_test)
        diagnostics["study_1_probability_judgment"] = {
            "observed": study_1_probability_judgment,
            "reported": STUDY_1_PROBABILITY_JUDGMENT,
        }

    if STUDY_2_ID in selected_sub_studies:
        authority_observed = _study_2_authority_observed(by_study[STUDY_2_ID])
        authority_choice_comparisons: List[Dict[str, Any]] = []
        authority_probability_judgment_comparisons: List[Dict[str, Any]] = []
        for group, metrics in STUDY_2_AUTHORITY.items():
            for metric, conditions in metrics.items():
                target = (
                    authority_choice_comparisons
                    if metric == "private_choice_rate"
                    else authority_probability_judgment_comparisons
                )
                for condition, reported in conditions.items():
                    target.append(
                        {
                            "metric": f"{group}.{metric}.{condition}",
                            "observed": authority_observed[group][metric].get(
                                condition
                            ),
                            "reported": reported,
                        }
                    )
        authority_choice_test = _alignment_test(
            "study_2_authority_choice_alignment",
            authority_choice_comparisons,
            tolerance=STUDY_2_AUTHORITY_CHOICE_MAE_TOLERANCE,
        )
        authority_probability_judgment_test = _alignment_test(
            "study_2_authority_probability_judgment_alignment",
            authority_probability_judgment_comparisons,
            tolerance=STUDY_2_AUTHORITY_PROBABILITY_JUDGMENT_MAE_TOLERANCE,
        )
        alignment_tests.extend(
            [
                authority_choice_test,
                authority_probability_judgment_test,
            ]
        )
        diagnostics["study_2_authority_conditions"] = {
            "observed": authority_observed,
            "reported": STUDY_2_AUTHORITY,
        }

    all_errors = [
        abs(float(comparison["observed"]) - float(comparison["reported"]))
        for test in alignment_tests
        for comparison in test["comparisons"]
        if comparison["observed"] is not None
    ]
    missing_any = any(test["missing"] for test in alignment_tests)
    alignment_score = (
        max(0.0, 1.0 - mean(all_errors))
        if all_errors and not missing_any
        else None
    )
    behavioral_evaluable = bool(sample_test["passed"])
    behavioral_passed = behavioral_evaluable and all(
        test["passed"] for test in alignment_tests
    )
    return (
        diagnostics,
        alignment_score,
        behavioral_evaluable,
        behavioral_passed,
        [sample_test, *alignment_tests],
    )


def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """Separate environment integrity from source-grounded behavioral replication."""

    participants = results.get("individual_data", [])
    responses = _flatten(results)
    selected_sub_studies, _ = _requested_sub_studies(results)
    execution_score, tests = _execution_checks(results, responses)
    (
        diagnostics,
        behavioral_alignment,
        behavioral_evaluable,
        behavioral_passed,
        behavioral_tests,
    ) = _behavioral_diagnostics(
        participants,
        responses,
        selected_sub_studies,
    )
    environment_passed = bool(tests) and all(test["passed"] for test in tests)
    passed = environment_passed and behavioral_passed
    reported_behavioral_alignment = (
        behavioral_alignment if behavioral_evaluable else None
    )
    total_score = (
        0.4 * execution_score + 0.6 * reported_behavioral_alignment
        if reported_behavioral_alignment is not None
        else 0.4 * execution_score
    )
    return {
        "total_score": total_score,
        "passed": passed,
        "environment_passed": environment_passed,
        "execution_score": execution_score,
        "evaluated_sub_studies": list(selected_sub_studies),
        "behavioral_evaluable": behavioral_evaluable,
        "behavioral_passed": behavioral_passed,
        "behavioral_alignment_score": reported_behavioral_alignment,
        "provisional_behavioral_alignment_score": (
            behavioral_alignment if not behavioral_evaluable else None
        ),
        "behavioral_diagnostics": diagnostics,
        "environment_test_results": tests,
        "behavioral_test_results": behavioral_tests,
        "test_results": [*tests, *behavioral_tests],
        "scoring_note": (
            "environment_passed checks source-grounded execution and information "
            "visibility. passed additionally requires the original per-study sample "
            "size and source-grounded choice, confidence, and authority-cell alignment."
        ),
    }
