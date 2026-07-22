"""Evaluation helpers for the Anderson-Holt information-cascade adapter."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple


REPORTED_CASCADE_RATE = 41.0 / 56.0


def _flatten_responses(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        response
        for participant in results.get("individual_data", [])
        for response in participant.get("responses", [])
    ]


def _execution_checks(responses: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    tests: List[Dict[str, Any]] = []
    valid_choices = all(response.get("response") in {"A", "B"} for response in responses)
    tests.append(
        {
            "test_id": "valid_choices",
            "passed": bool(responses) and valid_choices,
            "observed": len(responses),
        }
    )

    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for response in responses:
        info = response.get("trial_info", {})
        grouped[(int(info.get("session_index", -1)), int(info.get("period_number", -1)))].append(
            response
        )

    complete_periods = bool(grouped)
    for period_responses in grouped.values():
        positions = sorted(
            int(response.get("trial_info", {}).get("decision_position", -1))
            for response in period_responses
        )
        if positions != [1, 2, 3, 4, 5, 6]:
            complete_periods = False
            break
    tests.append(
        {
            "test_id": "complete_sequential_periods",
            "passed": complete_periods,
            "observed_periods": len(grouped),
        }
    )

    history_consistent = True
    for response in responses:
        info = response.get("trial_info", {})
        position = int(info.get("decision_position", -1))
        history = info.get("prior_decisions", [])
        if len(history) != position - 1 or any(choice not in {"A", "B"} for choice in history):
            history_consistent = False
            break
    tests.append(
        {
            "test_id": "public_history_matches_position",
            "passed": history_consistent,
        }
    )

    private_information_complete = all(
        response.get("trial_info", {}).get("private_signal") in {"L", "D"}
        and response.get("correct_answer") in {"A", "B"}
        for response in responses
    )
    tests.append(
        {
            "test_id": "private_signal_and_post_period_reveal_recorded",
            "passed": bool(responses) and private_information_complete,
        }
    )

    score = sum(1.0 for test in tests if test["passed"]) / len(tests)
    return score, tests


def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """Separate runtime integrity from behavioral alignment with the paper."""

    responses = _flatten_responses(results)
    execution_score, tests = _execution_checks(responses)
    opportunities = [
        response
        for response in responses
        if response.get("trial_info", {}).get("cascade_opportunity") is True
    ]
    followed = [
        response
        for response in opportunities
        if response.get("response")
        == response.get("trial_info", {}).get("cascade_direction_before_choice")
    ]
    observed_rate = len(followed) / len(opportunities) if opportunities else None
    if observed_rate is None:
        alignment_score = 0.0
    else:
        alignment_score = max(0.0, 1.0 - abs(observed_rate - REPORTED_CASCADE_RATE) / 0.5)

    total_score = 0.4 * execution_score + 0.6 * alignment_score
    tests.append(
        {
            "test_id": "cascade_following_alignment",
            "passed": alignment_score >= 0.7,
            "observed_rate": observed_rate,
            "reported_rate": REPORTED_CASCADE_RATE,
            "opportunities": len(opportunities),
            "followed": len(followed),
            "score": alignment_score,
            "scope_warning": (
                "The reported 41/56 rate pools all six symmetric sessions, including "
                "public-draw variants; this runtime implements the evidence-complete baseline."
            ),
        }
    )

    return {
        "total_score": total_score,
        "passed": total_score >= 0.7,
        "execution_score": execution_score,
        "behavioral_alignment_score": alignment_score,
        "test_results": tests,
    }
