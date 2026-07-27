"""Evaluation for the continuous Dates Task 3B/3C runtime."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


REPORTED_NO_FEEDBACK_AGREEING_RATE = 0.51
REPORTED_FEEDBACK_AGREEING_RATE = 0.17
TOTAL_TRIAL_SLOTS = 52
ATTENTION_CHECK_GLOBAL_INDICES = [16, 36]
NO_FEEDBACK_ID = "dates_task_3b_no_feedback"
FEEDBACK_ID = "dates_task_3c_feedback"
SUPPORTED_SUB_STUDIES = (NO_FEEDBACK_ID, FEEDBACK_ID)


def _flatten(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        response
        for participant in results.get("individual_data", [])
        for response in participant.get("responses", [])
    ]


def _scope_test(
    results: Dict[str, Any],
    participants: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    requested = results.get("selected_sub_studies")
    participant_observed = {
        str(participant.get("sub_study_id"))
        for participant in participants
        if participant.get("sub_study_id") is not None
    }
    response_observed = {
        str(response.get("trial_info", {}).get("sub_study_id"))
        for response in responses
        if response.get("trial_info", {}).get("sub_study_id") is not None
    }
    recorded = results.get("executed_sub_studies")
    recorded_observed = (
        {str(value) for value in recorded}
        if isinstance(recorded, list)
        else None
    )
    observed = participant_observed | response_observed
    recognized = observed.issubset(set(SUPPORTED_SUB_STUDIES))
    internally_consistent = (
        bool(participants)
        and bool(responses)
        and bool(observed)
        and participant_observed == response_observed
        and (
            recorded_observed is None
            or recorded_observed == observed
        )
    )
    if requested is None:
        valid = internally_consistent and recognized
    elif not isinstance(requested, list) or not requested:
        valid = False
    else:
        normalized = set(str(value) for value in requested)
        valid = (
            internally_consistent
            and normalized.issubset(set(SUPPORTED_SUB_STUDIES))
            and observed == normalized
        )
    return {
        "test_id": "selected_sub_study_scope",
        "passed": valid,
        "requested": requested,
        "observed": sorted(observed),
        "participant_observed": sorted(participant_observed),
        "response_observed": sorted(response_observed),
        "recorded_observed": (
            sorted(recorded_observed)
            if recorded_observed is not None
            else None
        ),
    }


def _execution_checks(
    results: Dict[str, Any], responses: List[Dict[str, Any]]
) -> Tuple[float, List[Dict[str, Any]]]:
    participants = results.get("individual_data", [])
    tests: List[Dict[str, Any]] = []
    tests.append(_scope_test(results, participants, responses))

    valid_schedules = bool(participants)
    schedule_details: Dict[str, Any] = {}
    for participant in participants:
        participant_responses = participant.get("responses", [])
        observed_indices = [
            response.get("trial_info", {}).get("global_trial_index")
            for response in participant_responses
        ]
        terminated = bool(participant.get("terminated_early"))
        if terminated:
            last = participant_responses[-1] if participant_responses else {}
            last_info = last.get("trial_info", {})
            expected_indices = list(range(int(last_info.get("global_trial_index", -1)) + 1))
            valid = (
                last_info.get("attention_check") is True
                and last_info.get("attention_check_passed") is False
                and last_info.get("global_trial_index") in ATTENTION_CHECK_GLOBAL_INDICES
                and observed_indices == expected_indices
            )
        else:
            valid = (
                len(participant_responses) == TOTAL_TRIAL_SLOTS
                and observed_indices == list(range(TOTAL_TRIAL_SLOTS))
            )
        schedule_details[str(participant.get("participant_id"))] = {
            "responses": len(participant_responses),
            "terminated_early": terminated,
            "valid": valid,
        }
        if not valid:
            valid_schedules = False
    tests.append(
        {
            "test_id": "complete_schedule_or_valid_attention_termination",
            "passed": valid_schedules,
            "participants": len(participants),
            "responses": len(responses),
            "details": schedule_details,
        }
    )

    valid_estimates = bool(responses) and all(
        1890 <= int(response.get("response", -1)) <= 2010
        and response.get("trial_info", {}).get("initial_width") in {7, 13, 21}
        and response.get("trial_info", {}).get("final_width") in {7, 13, 21}
        for response in responses
    )
    tests.append(
        {
            "test_id": "valid_years_and_marker_widths",
            "passed": valid_estimates,
        }
    )

    practice_complete = bool(participants)
    for participant in participants:
        participant_responses = participant.get("responses", [])
        unaided = [
            response
            for response in participant_responses
            if response.get("trial_info", {}).get("phase") == "unaided_practice"
        ]
        advised = [
            response
            for response in participant_responses
            if response.get("trial_info", {}).get("phase") == "practice_advisor"
        ]
        if (
            len(unaided) != 10
            or len(advised) != 2
            or any(
                response.get("trial_info", {}).get(
                    "correct_year_revealed_after_response"
                )
                is not True
                for response in unaided + advised
            )
            or any(
                response.get("trial_info", {}).get("advisor_display_name")
                != "Practice advisor"
                or response.get("trial_info", {}).get("advice_mode")
                != "practice_correct"
                or response.get("trial_info", {}).get("advice_width") != 8
                for response in advised
            )
        ):
            practice_complete = False
    tests.append(
        {
            "test_id": "original_ten_plus_two_practice_sequence",
            "passed": practice_complete,
        }
    )

    attention_checks_complete = bool(participants)
    for participant in participants:
        checks = [
            response
            for response in participant.get("responses", [])
            if response.get("trial_info", {}).get("attention_check") is True
        ]
        terminated = bool(participant.get("terminated_early"))
        if terminated:
            expected_count = (
                1
                if checks
                and checks[-1].get("trial_info", {}).get("global_trial_index") == 16
                else 2
            )
            valid = (
                len(checks) == expected_count
                and checks[-1].get("trial_info", {}).get("attention_check_passed")
                is False
            )
        else:
            valid = (
                [
                    response.get("trial_info", {}).get("global_trial_index")
                    for response in checks
                ]
                == ATTENTION_CHECK_GLOBAL_INDICES
                and all(
                    response.get("trial_info", {}).get("attention_check_passed")
                    is True
                    for response in checks
                )
            )
        for response in checks:
            info = response.get("trial_info", {})
            prompt = info.get("agent_visible_prompts", {}).get("initial", "")
            valid = (
                valid
                and info.get("attention_check_required_width") == 7
                and info.get("attention_check_target_words") in prompt
                and "smallest marker" in prompt
            )
        if not valid:
            attention_checks_complete = False
    tests.append(
        {
            "test_id": "attention_checks_at_original_slots_and_fail_closed",
            "passed": attention_checks_complete,
        }
    )

    ten_choices = bool(participants) and all(
        sum(
            response.get("trial_info", {}).get("advisor_choice") in {"A", "B"}
            for response in participant.get("responses", [])
        )
        == (0 if participant.get("terminated_early") else 10)
        for participant in participants
    )
    tests.append(
        {
            "test_id": "ten_advisor_choices_per_participant",
            "passed": ten_choices,
        }
    )

    core_responses = [
        response
        for response in responses
        if response.get("trial_info", {}).get("analysis_included") is True
    ]
    no_answer_leak = bool(core_responses) and all(
        response.get("trial_info", {}).get("answer_revealed_before_final") is False
        for response in core_responses
    )
    tests.append(
        {
            "test_id": "correct_year_hidden_until_final_response",
            "passed": no_answer_leak,
        }
    )

    advised_core_responses = [
        response
        for response in core_responses
        if response.get("trial_info", {}).get("advisor_type") in {"accurate", "agreeing"}
    ]
    anonymous = bool(advised_core_responses) and all(
        str(response.get("trial_info", {}).get("advisor_display_name", "")).startswith(
            "Advisor #"
        )
        for response in advised_core_responses
    )
    tests.append(
        {
            "test_id": "advisor_policy_hidden_behind_anonymous_identity",
            "passed": anonymous,
        }
    )
    return sum(test["passed"] for test in tests) / len(tests), tests


def _condition_pick_rate(
    participants: List[Dict[str, Any]], feedback: bool
) -> Optional[float]:
    rates: List[float] = []
    for participant in participants:
        if bool(participant.get("profile", {}).get("feedback_condition")) != feedback:
            continue
        choices = [
            response
            for response in participant.get("responses", [])
            if response.get("trial_info", {}).get("advisor_choice") in {"A", "B"}
        ]
        if choices:
            rates.append(
                mean(
                    response["trial_info"].get("advisor_type") == "agreeing"
                    for response in choices
                )
            )
    return mean(rates) if rates else None


def _rate_alignment(observed: Optional[float], reported: float) -> float:
    if observed is None:
        return 0.0
    return max(0.0, 1.0 - abs(observed - reported) / 0.5)


def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """Score runtime integrity separately from behavioral alignment."""

    participants = results.get("individual_data", [])
    responses = _flatten(results)
    execution_score, tests = _execution_checks(results, responses)
    environment_passed = bool(tests) and all(test["passed"] for test in tests)

    no_feedback_rate = _condition_pick_rate(participants, False)
    feedback_rate = _condition_pick_rate(participants, True)
    no_feedback_alignment = _rate_alignment(
        no_feedback_rate, REPORTED_NO_FEEDBACK_AGREEING_RATE
    )
    feedback_alignment = _rate_alignment(
        feedback_rate, REPORTED_FEEDBACK_AGREEING_RATE
    )
    choice_scores = [
        score
        for score, value in (
            (no_feedback_alignment, no_feedback_rate),
            (feedback_alignment, feedback_rate),
        )
        if value is not None
    ]
    choice_alignment = mean(choice_scores) if choice_scores else 0.0
    tests.append(
        {
            "test_id": "feedback_choice_alignment",
            "passed": choice_alignment >= 0.7,
            "observed_no_feedback_agreeing_rate": no_feedback_rate,
            "reported_no_feedback_agreeing_rate": REPORTED_NO_FEEDBACK_AGREEING_RATE,
            "observed_feedback_agreeing_rate": feedback_rate,
            "reported_feedback_agreeing_rate": REPORTED_FEEDBACK_AGREEING_RATE,
            "score": choice_alignment,
        }
    )

    reductions = {"accurate": [], "agreeing": []}
    for response in responses:
        info = response.get("trial_info", {})
        if not str(info.get("block", "")).startswith("familiarisation"):
            continue
        advisor_type = info.get("advisor_type")
        if advisor_type in reductions:
            reductions[advisor_type].append(float(info.get("error_reduction", 0)))
    accurate_reduction = mean(reductions["accurate"]) if reductions["accurate"] else None
    agreeing_reduction = mean(reductions["agreeing"]) if reductions["agreeing"] else None
    advice_use_alignment = (
        1.0
        if accurate_reduction is not None
        and agreeing_reduction is not None
        and accurate_reduction > agreeing_reduction
        else 0.0
    )
    tests.append(
        {
            "test_id": "accurate_advice_reduces_more_error",
            "passed": advice_use_alignment == 1.0,
            "observed_accurate_reduction": accurate_reduction,
            "observed_agreeing_reduction": agreeing_reduction,
            "reported_accurate_reduction": 9.67,
            "reported_agreeing_reduction": 1.46,
            "score": advice_use_alignment,
        }
    )

    total_score = (
        0.4 * execution_score
        + 0.35 * choice_alignment
        + 0.25 * advice_use_alignment
    )
    return {
        "total_score": total_score,
        "passed": environment_passed and total_score >= 0.7,
        "environment_passed": environment_passed,
        "execution_score": execution_score,
        "evaluated_sub_studies": results.get("executed_sub_studies"),
        "behavioral_alignment_score": (
            0.35 * choice_alignment + 0.25 * advice_use_alignment
        )
        / 0.6,
        "test_results": tests,
    }
