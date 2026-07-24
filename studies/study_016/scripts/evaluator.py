"""Environment-integrity evaluator for the Anderson-Holt runtime."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


PARTICIPANTS_PER_SESSION = 6
PERIODS_PER_SESSION = 15
REPORTED_SYMMETRIC_CASCADE_RATE = 41.0 / 56.0
REPORTED_ASYMMETRIC_CASCADE_RATE = 46.0 / 66.0
EXPECTED_SESSION_SCHEDULE: Tuple[Tuple[str, int], ...] = (
    ("symmetric_baseline", 1),
    ("symmetric_baseline", 2),
    ("symmetric_baseline", 3),
    ("symmetric_public_draw_after_position_4", 4),
    ("symmetric_public_draw_after_position_4", 5),
    ("asymmetric_baseline", 7),
    ("asymmetric_baseline", 8),
    ("asymmetric_baseline", 9),
    ("asymmetric_baseline", 10),
    ("asymmetric_baseline", 11),
    ("asymmetric_baseline", 12),
)
EXPECTED_URNS = {
    "symmetric_baseline": {
        "A": {"light": 2, "dark": 1},
        "B": {"light": 1, "dark": 2},
    },
    "symmetric_public_draw_after_position_4": {
        "A": {"light": 2, "dark": 1},
        "B": {"light": 1, "dark": 2},
    },
    "asymmetric_baseline": {
        "A": {"light": 6, "dark": 1},
        "B": {"light": 5, "dark": 2},
    },
}


def _flatten_responses(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        response
        for participant in results.get("individual_data", [])
        for response in participant.get("responses", [])
    ]


def _test(test_id: str, passed: bool, **details: Any) -> Dict[str, Any]:
    return {"test_id": test_id, "passed": bool(passed), **details}


def _group_responses(
    responses: Sequence[Dict[str, Any]],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for response in responses:
        info = response.get("trial_info", {})
        grouped[(int(info.get("session_index", -1)), int(info.get("period_number", -1)))].append(
            response
        )
    for period_responses in grouped.values():
        period_responses.sort(
            key=lambda row: int(row.get("trial_info", {}).get("decision_position", -1))
        )
    return grouped


def _practice_is_complete(participants: Sequence[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    sessions: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for participant in participants:
        sessions[int(participant.get("session_index", -1))].append(participant)

    observed_counts: Dict[str, int] = {}
    complete = bool(sessions)
    for session_index, rows in sessions.items():
        practices = [
            row.get("profile", {}).get("practice_periods", [])
            for row in rows
        ]
        reference = practices[0] if practices else []
        observed_counts[str(session_index)] = len(reference)
        if len(rows) != PARTICIPANTS_PER_SESSION or not reference:
            complete = False
            continue
        if any(practice != reference for practice in practices[1:]):
            complete = False
        if {period.get("true_urn") for period in reference} != {"A", "B"}:
            complete = False
        for period in reference:
            if (
                period.get("urn_selection_visible") is not True
                or period.get("draws_public") is not True
                or period.get("decisions_required") is not False
                or period.get("paid") is not False
                or len(period.get("draws", [])) != PARTICIPANTS_PER_SESSION
                or any(draw.get("signal") not in {"L", "D"} for draw in period.get("draws", []))
            ):
                complete = False
    return complete, {"practice_periods_by_session": observed_counts}


def _execution_checks(
    results: Dict[str, Any],
    responses: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    tests: List[Dict[str, Any]] = []
    participants = results.get("individual_data", [])

    valid_choices = bool(responses) and all(
        response.get("response") in {"A", "B"} for response in responses
    )
    tests.append(_test("valid_choices", valid_choices, observed=len(responses)))

    grouped = _group_responses(responses)
    session_periods: Dict[int, set[int]] = defaultdict(set)
    complete_periods = bool(grouped)
    for (session_index, period_number), period_responses in grouped.items():
        session_periods[session_index].add(period_number)
        positions = [
            int(response.get("trial_info", {}).get("decision_position", -1))
            for response in period_responses
        ]
        if positions != [1, 2, 3, 4, 5, 6]:
            complete_periods = False
    complete_sessions = complete_periods and all(
        periods == set(range(1, PERIODS_PER_SESSION + 1))
        for periods in session_periods.values()
    )
    tests.append(
        _test(
            "complete_six_person_fifteen_period_sessions",
            complete_sessions,
            observed_sessions=len(session_periods),
            observed_periods=len(grouped),
        )
    )

    history_consistent = bool(grouped)
    for period_responses in grouped.values():
        announced: List[str] = []
        for response in period_responses:
            history = response.get("trial_info", {}).get("prior_decisions", [])
            if history != announced:
                history_consistent = False
            announced.append(response.get("response"))
    tests.append(_test("public_history_matches_announced_order", history_consistent))

    decision_fields_complete = bool(responses)
    for response in responses:
        info = response.get("trial_info", {})
        die_roll = info.get("die_roll_hidden_during_period")
        expected_urn = (
            "A" if isinstance(die_roll, int) and 1 <= die_roll <= 3
            else "B" if isinstance(die_roll, int) and 4 <= die_roll <= 6
            else None
        )
        if not (
            info.get("private_signal") in {"L", "D"}
            and response.get("correct_answer") in {"A", "B"}
            and info.get("true_urn_revealed_after_period") == response.get("correct_answer")
            and expected_urn == response.get("correct_answer")
            and isinstance(response.get("prompt"), str)
            and bool(response.get("prompt"))
        ):
            decision_fields_complete = False
    tests.append(
        _test(
            "private_signal_and_post_period_reveal_recorded",
            decision_fields_complete,
        )
    )

    payoff_consistent = bool(responses) and all(
        response.get("earnings")
        == (2 if response.get("response") == response.get("correct_answer") else 0)
        for response in responses
    )
    tests.append(_test("payoff_matches_revealed_urn", payoff_consistent))

    initial_instruction_count: Dict[str, int] = {}
    instructions_complete = bool(participants)
    for participant in participants:
        participant_responses = participant.get("responses", [])
        flags = [
            response.get("trial_info", {}).get("initial_instructions_shown") is True
            for response in participant_responses
        ]
        initial_instruction_count[str(participant.get("participant_id"))] = sum(flags)
        if not flags or flags[0] is not True or any(flags[1:]):
            instructions_complete = False
    tests.append(
        _test(
            "instructions_and_practice_shown_before_first_paid_choice",
            instructions_complete,
            initial_context_count_by_participant=initial_instruction_count,
        )
    )

    practice_complete, practice_details = _practice_is_complete(participants)
    tests.append(
        _test(
            "practice_covers_both_urns_without_decisions_or_pay",
            practice_complete,
            **practice_details,
        )
    )

    observed_sessions = sorted(session_periods)
    schedule_complete = observed_sessions == list(range(len(observed_sessions)))
    treatment_details: Dict[str, Any] = {}
    for session_index in observed_sessions:
        if session_index >= len(EXPECTED_SESSION_SCHEDULE):
            schedule_complete = False
            continue
        expected_treatment, expected_published_number = EXPECTED_SESSION_SCHEDULE[session_index]
        period_responses = [
            response
            for (candidate_session, _), rows in grouped.items()
            if candidate_session == session_index
            for response in rows
        ]
        observed_treatments = {
            response.get("trial_info", {}).get("sub_study_id")
            for response in period_responses
        }
        observed_published_numbers = {
            response.get("trial_info", {}).get("published_session_number")
            for response in period_responses
        }
        urns_match = all(
            response.get("trial_info", {}).get("urns") == EXPECTED_URNS[expected_treatment]
            for response in period_responses
        )
        treatment_details[str(session_index)] = {
            "expected_treatment": expected_treatment,
            "observed_treatments": sorted(str(value) for value in observed_treatments),
            "expected_published_session_number": expected_published_number,
            "observed_published_session_numbers": sorted(
                int(value) for value in observed_published_numbers if value is not None
            ),
            "urns_match": urns_match,
        }
        if (
            observed_treatments != {expected_treatment}
            or observed_published_numbers != {expected_published_number}
            or not urns_match
        ):
            schedule_complete = False
    tests.append(
        _test(
            "published_treatment_schedule_and_urns_match",
            bool(observed_sessions) and schedule_complete,
            sessions=treatment_details,
        )
    )

    public_draw_timing = bool(grouped)
    for (session_index, _), period_responses in grouped.items():
        if session_index < 0 or session_index >= len(EXPECTED_SESSION_SCHEDULE):
            public_draw_timing = False
            continue
        treatment = EXPECTED_SESSION_SCHEDULE[session_index][0]
        visible_after_four: List[str] = []
        for response in period_responses:
            info = response.get("trial_info", {})
            position = int(info.get("decision_position", -1))
            draws = info.get("public_draws_visible", [])
            if treatment == "symmetric_public_draw_after_position_4":
                if position <= 4 and draws:
                    public_draw_timing = False
                if position == 5:
                    visible_after_four = list(draws)
                    if len(visible_after_four) != 2:
                        public_draw_timing = False
                if position == 6 and list(draws) != visible_after_four:
                    public_draw_timing = False
            elif draws:
                public_draw_timing = False
    tests.append(
        _test(
            "public_draws_visible_only_to_positions_five_and_six",
            public_draw_timing,
        )
    )

    score = sum(1.0 for test in tests if test["passed"]) / len(tests)
    return score, tests


def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """Fail closed on environment integrity; report behavior as diagnostics."""

    responses = _flatten_responses(results)
    execution_score, tests = _execution_checks(results, responses)

    by_treatment: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for response in responses:
        by_treatment[str(response.get("trial_info", {}).get("sub_study_id", "unknown"))].append(
            response
        )

    behavioral_diagnostics: Dict[str, Any] = {}
    reported_rates = {
        "symmetric_baseline": REPORTED_SYMMETRIC_CASCADE_RATE,
        "symmetric_public_draw_after_position_4": REPORTED_SYMMETRIC_CASCADE_RATE,
        "asymmetric_baseline": REPORTED_ASYMMETRIC_CASCADE_RATE,
    }
    for treatment, treatment_responses in sorted(by_treatment.items()):
        opportunities = [
            response
            for response in treatment_responses
            if response.get("trial_info", {}).get("cascade_opportunity") is True
        ]
        followed = [
            response
            for response in opportunities
            if response.get("response")
            == response.get("trial_info", {}).get("cascade_direction_before_choice")
        ]
        behavioral_diagnostics[treatment] = {
            "cascade_opportunities": len(opportunities),
            "cascade_followed": len(followed),
            "observed_rate": len(followed) / len(opportunities) if opportunities else None,
            "reported_rate": reported_rates.get(treatment),
            "diagnostic_only": True,
        }

    all_required_passed = bool(tests) and all(test["passed"] for test in tests)
    return {
        "total_score": execution_score,
        "passed": all_required_passed,
        "execution_score": execution_score,
        "behavioral_alignment_score": None,
        "behavioral_diagnostics": behavioral_diagnostics,
        "test_results": tests,
        "scoring_note": (
            "Pass/fail evaluates environment, materials, treatment schedule, and "
            "information timing. Agent choices are not required to reproduce the "
            "paper's participant-level outcomes."
        ),
    }
