"""Fail-closed environment evaluator for the two social-influence tasks."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple


STUDY_1_ID = "study_1_urn_scenarios"
STUDY_2_ID = "study_2_medical_authority_scenarios"
EXPECTED_COUNTS = {STUDY_1_ID: 24, STUDY_2_ID: 40}
REPORTED = {
    STUDY_1_ID: {
        "bayesian_choice_rate_non_indifference": 0.869,
        "cascade_choice_rate": 0.755,
        "private_choice_rate_at_indifference": 0.799,
    },
    STUDY_2_ID: {
        "bayesian_choice_rate_non_indifference": 0.920,
        "cascade_choice_rate": 0.821,
        "private_choice_rate_at_indifference": 0.497,
        "medical_director_alignment_rate": 0.7489,
    },
}


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

    assignment_counts = {
        sub_study_id: sum(
            participant.get("sub_study_id") == sub_study_id
            for participant in participants
        )
        for sub_study_id in EXPECTED_COUNTS
    }
    assignments_valid = (
        len(participants) >= 2
        and len(participants) % 2 == 0
        and assignment_counts[STUDY_1_ID] == assignment_counts[STUDY_2_ID]
    )
    tests.append(
        _test(
            "balanced_sub_study_assignment",
            assignments_valid,
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


def _behavioral_diagnostics(
    responses: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[float]]:
    by_study = {
        sub_study_id: [
            response
            for response in responses
            if response.get("trial_info", {}).get("sub_study_id") == sub_study_id
        ]
        for sub_study_id in EXPECTED_COUNTS
    }
    diagnostics: Dict[str, Any] = {}
    alignment_values: List[float] = []
    for sub_study_id, rows in by_study.items():
        non_ties = [
            row for row in rows if row["trial_info"]["bayesian_choice"] is not None
        ]
        cascades = [row for row in rows if row["trial_info"]["cascade_scenario"]]
        ties = [row for row in rows if row["trial_info"]["indifference_scenario"]]
        observed: Dict[str, Optional[float]] = {
            "bayesian_choice_rate_non_indifference": _mean_rate(
                non_ties,
                lambda row: row["response"] == row["trial_info"]["bayesian_choice"],
            ),
            "cascade_choice_rate": _mean_rate(
                cascades,
                lambda row: row["response"] == row["trial_info"]["bayesian_choice"],
            ),
            "private_choice_rate_at_indifference": _mean_rate(
                ties,
                lambda row: row["response"]
                == row["trial_info"]["private_information_favors"],
            ),
        }
        if sub_study_id == STUDY_2_ID:
            director_rows = [
                row
                for row in rows
                if row["trial_info"]["medical_director_diagnosis"] is not None
            ]
            observed["medical_director_alignment_rate"] = _mean_rate(
                director_rows,
                lambda row: row["response"]
                == row["trial_info"]["medical_director_diagnosis"],
            )
        metric_alignment: Dict[str, Optional[float]] = {}
        for metric, reported in REPORTED[sub_study_id].items():
            value = observed.get(metric)
            score = (
                max(0.0, 1.0 - abs(float(value) - reported) / 0.35)
                if value is not None
                else None
            )
            metric_alignment[metric] = score
            if score is not None:
                alignment_values.append(score)
        diagnostics[sub_study_id] = {
            "observed": observed,
            "reported": REPORTED[sub_study_id],
            "alignment": metric_alignment,
            "diagnostic_only": True,
        }
    return diagnostics, mean(alignment_values) if alignment_values else None


def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """Require environment integrity and keep behavioral matching diagnostic."""

    responses = _flatten(results)
    execution_score, tests = _execution_checks(results, responses)
    diagnostics, behavioral_alignment = _behavioral_diagnostics(responses)
    all_required_passed = bool(tests) and all(test["passed"] for test in tests)
    return {
        "total_score": execution_score,
        "passed": all_required_passed,
        "execution_score": execution_score,
        "behavioral_alignment_score": behavioral_alignment,
        "behavioral_diagnostics": diagnostics,
        "test_results": tests,
        "scoring_note": (
            "Pass/fail evaluates source-grounded scenario execution and information "
            "visibility. Behavioral agreement with the human sample is diagnostic."
        ),
    }
