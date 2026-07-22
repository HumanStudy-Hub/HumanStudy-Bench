"""Evaluation for the continuous Dates Task 3B/3C runtime."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


REPORTED_NO_FEEDBACK_AGREEING_RATE = 0.51
REPORTED_FEEDBACK_AGREEING_RATE = 0.17


def _flatten(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        response
        for participant in results.get("individual_data", [])
        for response in participant.get("responses", [])
    ]


def _execution_checks(
    results: Dict[str, Any], responses: List[Dict[str, Any]]
) -> Tuple[float, List[Dict[str, Any]]]:
    participants = results.get("individual_data", [])
    tests: List[Dict[str, Any]] = []

    complete = bool(participants) and all(
        len(participant.get("responses", [])) == 40 for participant in participants
    )
    tests.append(
        {
            "test_id": "complete_core_blocks",
            "passed": complete,
            "participants": len(participants),
            "responses": len(responses),
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

    ten_choices = bool(participants) and all(
        sum(
            response.get("trial_info", {}).get("advisor_choice") in {"A", "B"}
            for response in participant.get("responses", [])
        )
        == 10
        for participant in participants
    )
    tests.append(
        {
            "test_id": "ten_advisor_choices_per_participant",
            "passed": ten_choices,
        }
    )

    no_answer_leak = bool(responses) and all(
        response.get("trial_info", {}).get("answer_revealed_before_final") is False
        for response in responses
    )
    tests.append(
        {
            "test_id": "correct_year_hidden_until_final_response",
            "passed": no_answer_leak,
        }
    )

    anonymous = bool(responses) and all(
        str(response.get("trial_info", {}).get("advisor_display_name", "")).startswith(
            "Advisor #"
        )
        for response in responses
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
        "passed": total_score >= 0.7,
        "execution_score": execution_score,
        "behavioral_alignment_score": (
            0.35 * choice_alignment + 0.25 * advice_use_alignment
        )
        / 0.6,
        "test_results": tests,
    }
