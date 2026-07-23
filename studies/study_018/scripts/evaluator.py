"""Evaluation for the disparate social-information runtime."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


REPORTED_ADJUSTMENT = {
    "LN": 0.415,
    "HN": 0.290,
    "HF": 0.279,
    "HC": 0.365,
}


def _flatten(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        response
        for participant in results.get("individual_data", [])
        for response in participant.get("responses", [])
    ]


def _mean_or_none(values: List[float]) -> Optional[float]:
    return mean(values) if values else None


def _execution_checks(
    results: Dict[str, Any],
    responses: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    participants = results.get("individual_data", [])
    tests: List[Dict[str, Any]] = []

    complete = bool(participants) and all(
        len(participant.get("responses", [])) == 35 for participant in participants
    )
    tests.append(
        {
            "test_id": "complete_included_blocks",
            "passed": complete,
            "participants": len(participants),
            "responses": len(responses),
        }
    )

    block_counts_valid = bool(participants) and all(
        Counter(
            response.get("trial_info", {}).get("block")
            for response in participant.get("responses", [])
        )
        == Counter({"main_task": 30, "four_peer_control": 5})
        for participant in participants
    )
    tests.append(
        {
            "test_id": "thirty_main_and_five_control_rounds",
            "passed": block_counts_valid,
        }
    )

    condition_counts_valid = bool(participants) and all(
        Counter(
            response.get("trial_info", {}).get("condition")
            for response in participant.get("responses", [])
            if response.get("trial_info", {}).get("block") == "main_task"
        )
        == Counter({"LN": 5, "HN": 5, "HF": 5, "HC": 5, "filler": 10})
        for participant in participants
    )
    tests.append(
        {
            "test_id": "published_main_condition_schedule",
            "passed": condition_counts_valid,
        }
    )

    valid_estimates = bool(responses) and all(
        isinstance(response.get("response"), int)
        and 1 <= int(response["response"]) <= 150
        and all(
            1 <= int(peer) <= 150
            for peer in response.get("trial_info", {}).get("peer_estimates", [])
        )
        for response in responses
    )
    tests.append(
        {
            "test_id": "all_estimates_in_slider_domain",
            "passed": valid_estimates,
        }
    )

    source_grounded = bool(responses) and all(
        response.get("trial_info", {}).get("peer_lookup_verified") is True
        for response in responses
    )
    tests.append(
        {
            "test_id": "published_peer_lookup_used",
            "passed": source_grounded,
        }
    )

    no_answer_leak = bool(responses) and all(
        response.get("trial_info", {}).get("answer_revealed_before_response") is False
        for response in responses
    )
    tests.append(
        {
            "test_id": "true_counts_hidden_until_response",
            "passed": no_answer_leak,
        }
    )

    main_visual_only = bool(responses) and all(
        (
            response.get("trial_info", {}).get("stimulus_presented") is True
            if response.get("trial_info", {}).get("block") == "main_task"
            else response.get("trial_info", {}).get("stimulus_presented") is False
        )
        for response in responses
    )
    tests.append(
        {
            "test_id": "visual_stimuli_confined_to_main_first_estimates",
            "passed": main_visual_only,
        }
    )
    return sum(test["passed"] for test in tests) / len(tests), tests


def _condition_adjustments(
    responses: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    values: Dict[str, List[float]] = {condition: [] for condition in REPORTED_ADJUSTMENT}
    for response in responses:
        info = response.get("trial_info", {})
        condition = info.get("condition")
        adjustment = info.get("social_information_use")
        if (
            info.get("block") == "main_task"
            and condition in values
            and isinstance(adjustment, (int, float))
        ):
            values[condition].append(float(adjustment))
    return {condition: _mean_or_none(items) for condition, items in values.items()}


def _strategy_rates(
    responses: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Optional[float]]]:
    counts = {
        condition: Counter()
        for condition in REPORTED_ADJUSTMENT
    }
    totals = Counter()
    for response in responses:
        info = response.get("trial_info", {})
        condition = info.get("condition")
        if info.get("block") != "main_task" or condition not in counts:
            continue
        counts[condition][info.get("strategy")] += 1
        totals[condition] += 1
    return {
        condition: {
            strategy: (
                counts[condition][strategy] / totals[condition]
                if totals[condition]
                else None
            )
            for strategy in ("keep", "adopt_nearest", "compromise", "other")
        }
        for condition in REPORTED_ADJUSTMENT
    }


def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """Score runtime integrity separately from behavioral alignment."""

    responses = _flatten(results)
    execution_score, tests = _execution_checks(results, responses)

    observed = _condition_adjustments(responses)
    available = [
        condition
        for condition, value in observed.items()
        if value is not None
    ]
    value_alignment = (
        mean(
            max(
                0.0,
                1.0
                - abs(float(observed[condition]) - REPORTED_ADJUSTMENT[condition])
                / 0.40,
            )
            for condition in available
        )
        if available
        else 0.0
    )
    directional_alignment = (
        1.0
        if len(available) == 4
        and float(observed["LN"]) > float(observed["HC"])
        and float(observed["HC"]) > float(observed["HN"])
        and float(observed["HC"]) > float(observed["HF"])
        else 0.0
    )
    adjustment_alignment = 0.75 * value_alignment + 0.25 * directional_alignment
    tests.append(
        {
            "test_id": "condition_adjustment_alignment",
            "passed": adjustment_alignment >= 0.7,
            "observed": observed,
            "reported": REPORTED_ADJUSTMENT,
            "value_alignment": value_alignment,
            "directional_alignment": directional_alignment,
            "score": adjustment_alignment,
        }
    )

    strategy_rates = _strategy_rates(responses)
    strategy_alignment = (
        1.0
        if all(
            strategy_rates[condition]["compromise"] is not None
            for condition in REPORTED_ADJUSTMENT
        )
        and float(strategy_rates["LN"]["compromise"])
        > float(strategy_rates["HN"]["compromise"])
        and float(strategy_rates["HC"]["compromise"])
        > float(strategy_rates["HF"]["compromise"])
        else 0.0
    )
    tests.append(
        {
            "test_id": "strategy_direction_alignment",
            "passed": strategy_alignment == 1.0,
            "observed": strategy_rates,
            "reported_direction": (
                "Compromise is more frequent in LN than HN and in HC than HF."
            ),
            "score": strategy_alignment,
        }
    )

    main_deviation = [
        abs(
            float(response["response"])
            - float(response["trial_info"]["peer_mean"])
        )
        for response in responses
        if response.get("trial_info", {}).get("block") == "main_task"
    ]
    control_deviation = [
        float(response["trial_info"]["absolute_deviation_from_peer_mean"])
        for response in responses
        if response.get("trial_info", {}).get("block") == "four_peer_control"
    ]
    observed_main_deviation = _mean_or_none(main_deviation)
    observed_control_deviation = _mean_or_none(control_deviation)
    control_alignment = (
        1.0
        if observed_main_deviation is not None
        and observed_control_deviation is not None
        and observed_control_deviation < observed_main_deviation
        else 0.0
    )
    tests.append(
        {
            "test_id": "four_peer_control_closer_to_arithmetic_mean",
            "passed": control_alignment == 1.0,
            "main_mean_absolute_deviation": observed_main_deviation,
            "control_mean_absolute_deviation": observed_control_deviation,
            "score": control_alignment,
        }
    )

    behavioral_score = (
        0.65 * adjustment_alignment
        + 0.175 * strategy_alignment
        + 0.175 * control_alignment
    )
    total_score = 0.45 * execution_score + 0.55 * behavioral_score
    return {
        "total_score": total_score,
        "passed": total_score >= 0.7,
        "execution_score": execution_score,
        "behavioral_alignment_score": behavioral_score,
        "test_results": tests,
    }
