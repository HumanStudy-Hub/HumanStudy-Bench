"""Stateful runtime for Jaquiery and Yeung's continuous Dates Task 3B/3C."""

from __future__ import annotations

import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder


YEAR_MIN = 1890
YEAR_MAX = 2010
PARTICIPANT_WIDTHS = (7, 13, 21)
MARKER_POINTS = {7: 25, 13: 10, 21: 5}
ADVISOR_WIDTH = 6
ADVICE_SD = 5
REFLECTED_CONTROL_PROBABILITY = 0.135
FAMILIARISATION_TRIALS = 15
CHOICE_TRIALS = 10
CORE_TRIALS = 2 * FAMILIARISATION_TRIALS + CHOICE_TRIALS


def marker_contains(center: int, width: int, target: int) -> bool:
    """Return whether an integer target is covered by a centered marker."""

    half = width // 2
    return center - half <= target <= center + half


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return mean(values) if values else None


class DatesAdvisorPromptBuilder(PromptBuilder):
    """Build prompts while keeping advisor policies and correct years hidden."""

    def build_system_prompt(self, participant_profile: Dict[str, Any] = None) -> str:
        del participant_profile
        return (
            "Act as one participant in a general-knowledge advice experiment. "
            "Use only your own historical knowledge and information revealed during "
            "the task. Do not look up answers. The two anonymous advisors may behave "
            "differently; learn from your own experience with them. Follow every "
            "requested response format exactly."
        )

    def build_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        return self.build_initial_prompt(
            event_prompt=str(trial_data["event_prompt"]),
            trial_number=int(trial_data.get("trial_number", 1)),
            block_label=str(trial_data.get("block_label", "task")),
            pending_feedback=trial_data.get("pending_feedback"),
        )

    def build_initial_prompt(
        self,
        *,
        event_prompt: str,
        trial_number: int,
        block_label: str,
        pending_feedback: Optional[str] = None,
    ) -> str:
        feedback = f"Feedback from the previous trial: {pending_feedback}\n\n" if pending_feedback else ""
        return (
            f"{feedback}"
            f"Historical dates task, core trial {trial_number} of {CORE_TRIALS} "
            f"({block_label}).\n"
            f"Event: {event_prompt}\n\n"
            f"Estimate the year from {YEAR_MIN} through {YEAR_MAX}. Choose a marker "
            "width of 7, 13, or 21 years. If the marker covers the correct year it "
            "earns 25, 10, or 5 points respectively.\n"
            "Output exactly: YEAR=<1890-2010>; WIDTH=<7|13|21>"
        )

    def build_choice_prompt(
        self,
        *,
        advisor_a_name: str,
        advisor_b_name: str,
    ) -> str:
        return (
            "Choose which advisor you want to consult for this trial, based on your "
            "experience so far.\n"
            f"A: {advisor_a_name}\n"
            f"B: {advisor_b_name}\n"
            "Output exactly: ADVISOR=A or ADVISOR=B"
        )

    def build_final_prompt(
        self,
        *,
        event_prompt: str,
        initial_year: int,
        initial_width: int,
        advisor_name: str,
        advice_year: int,
        advice_width: int,
    ) -> str:
        half = advice_width // 2
        return (
            f"Event: {event_prompt}\n"
            f"Your initial marker: center {initial_year}, width {initial_width}.\n"
            f"{advisor_name}'s marker: center {advice_year}, spanning approximately "
            f"{advice_year - half} through {advice_year + half}.\n\n"
            "Give your final estimate. You may keep or change your year and marker width.\n"
            "Output exactly: YEAR=<1890-2010>; WIDTH=<7|13|21>"
        )


class StudyStudy017Config(BaseStudyConfig):
    """Independent participants with stateful, multi-turn trials."""

    prompt_builder_class = DatesAdvisorPromptBuilder
    REQUIRES_GROUP_TRIALS = True
    MATERIAL_ID = "dates_task_3b_3c"

    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        participant_count = self.get_n_participants() if n_trials is None else int(n_trials)
        if participant_count < 1:
            raise ValueError("study_017 requires at least one participant")
        material = self.load_material(self.MATERIAL_ID)
        return [
            {
                "trial_number": participant_index + 1,
                "participant_index": participant_index,
                "study_type": "stateful_judge_advisor_dates_task",
                "sub_study_id": self._condition_assignment(participant_index, participant_count)[
                    "sub_study_id"
                ],
                "condition_assignment": self._condition_assignment(
                    participant_index, participant_count
                ),
                "instructions": material["instructions"],
            }
            for participant_index in range(participant_count)
        ]

    @staticmethod
    def _condition_assignment(participant_index: int, participant_count: int) -> Dict[str, Any]:
        if participant_count == 60:
            feedback = participant_index >= 29
            within_condition_index = participant_index if not feedback else participant_index - 29
            accurate_first = within_condition_index % 2 == 0
        else:
            cell = participant_index % 4
            feedback = cell >= 2
            accurate_first = cell % 2 == 0
        return {
            "feedback": feedback,
            "advisor_order": "accurate_first" if accurate_first else "agreeing_first",
            "sub_study_id": (
                "dates_task_3c_feedback" if feedback else "dates_task_3b_no_feedback"
            ),
        }

    @staticmethod
    def parse_estimate(response_text: str) -> Optional[Tuple[int, int]]:
        if not response_text:
            return None
        year_match = re.search(r"\bYEAR\s*[:=]\s*(\d{4})\b", response_text, re.IGNORECASE)
        width_match = re.search(r"\bWIDTH\s*[:=]\s*(7|13|21)\b", response_text, re.IGNORECASE)
        if not year_match or not width_match:
            return None
        year, width = int(year_match.group(1)), int(width_match.group(1))
        if not YEAR_MIN <= year <= YEAR_MAX:
            return None
        return year, width

    @staticmethod
    def parse_advisor_choice(response_text: str) -> Optional[str]:
        if not response_text:
            return None
        match = re.search(r"\bADVISOR\s*[:=]\s*([AB])\b", response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        compact = response_text.strip().upper().rstrip(".")
        return compact if compact in {"A", "B"} else None

    @staticmethod
    def _merge_usage(total: Dict[str, Any], new_usage: Dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = int(total.get(key, 0) or 0) + int(new_usage.get(key, 0) or 0)
        total["cost"] = float(total.get("cost", 0.0) or 0.0) + float(
            new_usage.get("cost", 0.0) or 0.0
        )

    @staticmethod
    def _sample_advice(
        rng: random.Random,
        *,
        advisor_type: str,
        initial_year: int,
        correct_year: int,
    ) -> Tuple[int, str]:
        reflected = rng.random() < REFLECTED_CONTROL_PROBABILITY
        if reflected:
            base = 2 * correct_year - initial_year
            advice_mode = "reflected_control"
        elif advisor_type == "accurate":
            base = correct_year
            advice_mode = "accurate"
        elif advisor_type == "agreeing":
            base = initial_year
            advice_mode = "agreeing"
        else:
            raise ValueError(f"unknown advisor type: {advisor_type}")

        lower = YEAR_MIN + ADVISOR_WIDTH // 2
        upper = YEAR_MAX - ADVISOR_WIDTH // 2
        for _ in range(100):
            candidate = round(rng.gauss(base, ADVICE_SD))
            if lower <= candidate <= upper and candidate != initial_year:
                return candidate, advice_mode
        fallback = min(max(round(base), lower), upper)
        if fallback == initial_year:
            fallback = fallback + 1 if fallback < upper else fallback - 1
        return fallback, advice_mode

    @staticmethod
    def _mock_estimate(rng: random.Random, correct_year: int) -> Tuple[int, int]:
        year = min(max(round(rng.gauss(correct_year, 19)), YEAR_MIN), YEAR_MAX)
        error = abs(year - correct_year)
        width = 7 if error <= 11 else 13 if error <= 17 else 21
        return year, width

    @staticmethod
    def _mock_final_estimate(
        rng: random.Random,
        *,
        initial_year: int,
        initial_width: int,
        advice_year: int,
        advice_mode: str,
        correct_year: int,
    ) -> Tuple[int, int]:
        weight = (
            0.65
            if advice_mode == "accurate"
            else 0.45
            if advice_mode == "reflected_control"
            else 0.2
        )
        year = round(
            initial_year
            + weight * (advice_year - initial_year)
            + rng.gauss(0, 1.5)
        )
        year = min(max(year, YEAR_MIN), YEAR_MAX)
        error = abs(year - correct_year)
        width = 7 if error <= 11 else 13 if error <= 17 else max(initial_width, 21)
        return year, width

    def _call_estimate(
        self,
        participant: Any,
        prompt: str,
        usage: Dict[str, Any],
    ) -> Tuple[int, int, str]:
        result = participant.continue_conversation(prompt, max_tokens=48)
        self._merge_usage(usage, result.get("usage", {}))
        response_text = str(result.get("response_text", ""))
        parsed = self.parse_estimate(response_text)
        if parsed is None:
            repair = participant.continue_conversation(
                "Your previous response was invalid. Output exactly "
                "YEAR=<1890-2010>; WIDTH=<7|13|21>.",
                max_tokens=32,
            )
            self._merge_usage(usage, repair.get("usage", {}))
            response_text = str(repair.get("response_text", ""))
            parsed = self.parse_estimate(response_text)
        if parsed is None:
            raise ValueError("participant returned no valid year and marker width")
        return parsed[0], parsed[1], response_text

    def _call_choice(
        self,
        participant: Any,
        prompt: str,
        usage: Dict[str, Any],
    ) -> Tuple[str, str]:
        result = participant.continue_conversation(prompt, max_tokens=24)
        self._merge_usage(usage, result.get("usage", {}))
        response_text = str(result.get("response_text", ""))
        choice = self.parse_advisor_choice(response_text)
        if choice is None:
            repair = participant.continue_conversation(
                "Your previous response was invalid. Output exactly ADVISOR=A or ADVISOR=B.",
                max_tokens=16,
            )
            self._merge_usage(usage, repair.get("usage", {}))
            response_text = str(repair.get("response_text", ""))
            choice = self.parse_advisor_choice(response_text)
        if choice is None:
            raise ValueError("participant returned no valid advisor choice")
        return choice, response_text

    def _run_one_participant(
        self,
        trial: Dict[str, Any],
        *,
        participant: Optional[Any],
        profile: Dict[str, Any],
        prompt_builder: DatesAdvisorPromptBuilder,
        base_seed: int,
        question_bank: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        participant_id = int(trial["participant_index"])
        condition = dict(trial["condition_assignment"])
        rng = random.Random(base_seed + 100_003 * participant_id)
        questions = list(question_bank)
        rng.shuffle(questions)
        questions = questions[:CORE_TRIALS]

        advisor_numbers = rng.sample(range(10, 60), 2)
        advisor_types = ["accurate", "agreeing"]
        rng.shuffle(advisor_types)
        letter_to_type = {"A": advisor_types[0], "B": advisor_types[1]}
        type_to_letter = {value: key for key, value in letter_to_type.items()}
        letter_to_name = {
            "A": f"Advisor #{advisor_numbers[0]}",
            "B": f"Advisor #{advisor_numbers[1]}",
        }
        type_to_name = {
            advisor_type: letter_to_name[type_to_letter[advisor_type]]
            for advisor_type in ("accurate", "agreeing")
        }
        advisor_order = (
            ["accurate", "agreeing"]
            if condition["advisor_order"] == "accurate_first"
            else ["agreeing", "accurate"]
        )

        if participant is not None:
            participant.start_conversation()

        responses: List[Dict[str, Any]] = []
        pending_feedback: Optional[str] = None
        no_feedback_preference = rng.choice(["accurate", "agreeing"])

        for trial_index, question in enumerate(questions):
            trial_number = trial_index + 1
            if trial_index < FAMILIARISATION_TRIALS:
                block_label = "familiarisation block 1"
                assigned_advisor = advisor_order[0]
            elif trial_index < 2 * FAMILIARISATION_TRIALS:
                block_label = "familiarisation block 2"
                assigned_advisor = advisor_order[1]
            else:
                block_label = "advisor-choice block"
                assigned_advisor = None

            initial_prompt = prompt_builder.build_initial_prompt(
                event_prompt=question["prompt"],
                trial_number=trial_number,
                block_label=block_label,
                pending_feedback=pending_feedback,
            )
            pending_feedback = None
            usage: Dict[str, Any] = {}
            if participant is not None:
                initial_year, initial_width, initial_response_text = self._call_estimate(
                    participant, initial_prompt, usage
                )
            else:
                initial_year, initial_width = self._mock_estimate(
                    rng, int(question["target_year"])
                )
                initial_response_text = (
                    f"YEAR={initial_year}; WIDTH={initial_width}"
                )

            choice_letter: Optional[str] = None
            choice_response_text: Optional[str] = None
            choice_prompt: Optional[str] = None
            if assigned_advisor is None:
                choice_prompt = prompt_builder.build_choice_prompt(
                    advisor_a_name=letter_to_name["A"],
                    advisor_b_name=letter_to_name["B"],
                )
                if participant is not None:
                    choice_letter, choice_response_text = self._call_choice(
                        participant, choice_prompt, usage
                    )
                    assigned_advisor = letter_to_type[choice_letter]
                else:
                    if condition["feedback"]:
                        assigned_advisor = (
                            "accurate" if rng.random() < 0.83 else "agreeing"
                        )
                    else:
                        assigned_advisor = no_feedback_preference
                    choice_letter = type_to_letter[assigned_advisor]
                    choice_response_text = f"ADVISOR={choice_letter}"

            advice_year, advice_mode = self._sample_advice(
                rng,
                advisor_type=str(assigned_advisor),
                initial_year=initial_year,
                correct_year=int(question["target_year"]),
            )
            advisor_name = type_to_name[str(assigned_advisor)]
            final_prompt = prompt_builder.build_final_prompt(
                event_prompt=question["prompt"],
                initial_year=initial_year,
                initial_width=initial_width,
                advisor_name=advisor_name,
                advice_year=advice_year,
                advice_width=ADVISOR_WIDTH,
            )
            if participant is not None:
                final_year, final_width, final_response_text = self._call_estimate(
                    participant, final_prompt, usage
                )
            else:
                final_year, final_width = self._mock_final_estimate(
                    rng,
                    initial_year=initial_year,
                    initial_width=initial_width,
                    advice_year=advice_year,
                    advice_mode=advice_mode,
                    correct_year=int(question["target_year"]),
                )
                final_response_text = f"YEAR={final_year}; WIDTH={final_width}"

            correct_year = int(question["target_year"])
            initial_error = abs(initial_year - correct_year)
            final_error = abs(final_year - correct_year)
            if condition["feedback"]:
                pending_feedback = (
                    f"The correct year was {correct_year}. Your final marker "
                    f"{'covered' if marker_contains(final_year, final_width, correct_year) else 'did not cover'} "
                    "the answer."
                )

            responses.append(
                {
                    "participant_id": participant_id,
                    "trial_number": trial_number,
                    "response": final_year,
                    "response_text": final_response_text,
                    "raw_response_text": final_response_text,
                    "usage": usage,
                    "correct_answer": correct_year,
                    "is_correct": marker_contains(final_year, final_width, correct_year),
                    "trial_info": {
                        "study_type": "stateful_judge_advisor_dates_task",
                        "sub_study_id": condition["sub_study_id"],
                        "feedback_condition": condition["feedback"],
                        "advisor_order": condition["advisor_order"],
                        "block": block_label,
                        "event_id": question["id"],
                        "event_prompt": question["prompt"],
                        "topic": question.get("topic"),
                        "initial_year": initial_year,
                        "initial_width": initial_width,
                        "initial_error": initial_error,
                        "initial_response_text": initial_response_text,
                        "advisor_choice": choice_letter,
                        "advisor_choice_response_text": choice_response_text,
                        "advisor_display_name": advisor_name,
                        "advisor_type": assigned_advisor,
                        "advice_mode": advice_mode,
                        "advice_year": advice_year,
                        "advice_width": ADVISOR_WIDTH,
                        "final_year": final_year,
                        "final_width": final_width,
                        "final_error": final_error,
                        "error_reduction": initial_error - final_error,
                        "points": (
                            MARKER_POINTS[final_width]
                            if marker_contains(final_year, final_width, correct_year)
                            else 0
                        ),
                        "answer_revealed_before_final": False,
                        "correct_year_revealed_after_final": condition["feedback"],
                        "agent_visible_prompts": {
                            "initial": initial_prompt,
                            "choice": choice_prompt,
                            "final": final_prompt,
                        },
                    },
                }
            )

        if participant is not None:
            participant.clear_conversation()
        return {
            "participant_id": participant_id,
            "profile": {
                **profile,
                "feedback_condition": condition["feedback"],
                "advisor_order": condition["advisor_order"],
            },
            "sub_study_id": condition["sub_study_id"],
            "advisor_identity_map": {
                letter: {
                    "display_name": letter_to_name[letter],
                    "advisor_type": letter_to_type[letter],
                }
                for letter in ("A", "B")
            },
            "responses": responses,
        }

    def run_group_experiment(
        self,
        trials: List[Dict[str, Any]],
        instructions: str,
        participant_pool_kwargs: Dict[str, Any],
        prompt_builder: Optional[Any] = None,
    ) -> Dict[str, Any]:
        del instructions
        builder = prompt_builder or self.prompt_builder
        if not isinstance(builder, DatesAdvisorPromptBuilder):
            raise TypeError("study_017 requires DatesAdvisorPromptBuilder")

        question_bank_data = self.load_material("question_bank")
        question_bank = question_bank_data.get("questions", [])
        if len(question_bank) < CORE_TRIALS:
            raise ValueError("study_017 question bank must contain at least 40 questions")

        use_real_llm = bool(participant_pool_kwargs.get("use_real_llm", False))
        participants: Dict[int, Any] = {}
        if use_real_llm:
            from src.agents.llm_participant_agent import ParticipantPool

            pool_kwargs = dict(participant_pool_kwargs)
            pool_kwargs["n_participants"] = len(trials)
            pool = ParticipantPool(**pool_kwargs)
            participants = {
                int(trial["participant_index"]): participant
                for trial, participant in zip(trials, pool.participants)
            }
            profiles = {
                int(trial["participant_index"]): dict(participant.profile)
                for trial, participant in zip(trials, pool.participants)
            }
        else:
            supplied_profiles = participant_pool_kwargs.get("profiles")
            profiles = {
                int(trial["participant_index"]): (
                    dict(supplied_profiles[index])
                    if supplied_profiles is not None and index < len(supplied_profiles)
                    else {"participant_id": int(trial["participant_index"])}
                )
                for index, trial in enumerate(trials)
            }

        base_seed = int(participant_pool_kwargs.get("random_seed", 42) or 42)
        configured_workers = participant_pool_kwargs.get("num_workers")
        workers = (
            int(configured_workers)
            if configured_workers is not None
            else min(8, len(trials))
            if use_real_llm
            else 1
        )
        results: List[Dict[str, Any]] = []
        if use_real_llm and workers > 1 and len(trials) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(trials))) as executor:
                futures = {
                    executor.submit(
                        self._run_one_participant,
                        trial,
                        participant=participants[int(trial["participant_index"])],
                        profile=profiles[int(trial["participant_index"])],
                        prompt_builder=builder,
                        base_seed=base_seed,
                        question_bank=question_bank,
                    ): int(trial["participant_index"])
                    for trial in trials
                }
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for trial in trials:
                participant_id = int(trial["participant_index"])
                results.append(
                    self._run_one_participant(
                        trial,
                        participant=participants.get(participant_id),
                        profile=profiles[participant_id],
                        prompt_builder=builder,
                        base_seed=base_seed,
                        question_bank=question_bank,
                    )
                )

        results.sort(key=lambda row: int(row["participant_id"]))
        return {"individual_data": results}

    def aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        participants = raw_results.get("individual_data", [])
        responses = [
            response
            for participant in participants
            for response in participant.get("responses", [])
        ]
        valid = [
            response
            for response in responses
            if isinstance(response.get("response"), int)
        ]
        choice_trials = [
            response
            for response in valid
            if response.get("trial_info", {}).get("advisor_choice") in {"A", "B"}
        ]
        familiarisation = [
            response
            for response in valid
            if str(response.get("trial_info", {}).get("block", "")).startswith(
                "familiarisation"
            )
        ]

        def choice_rate(feedback: bool) -> Optional[float]:
            participant_rates: List[float] = []
            for participant in participants:
                if bool(participant.get("profile", {}).get("feedback_condition")) != feedback:
                    continue
                choices = [
                    response
                    for response in participant.get("responses", [])
                    if response.get("trial_info", {}).get("advisor_choice") in {"A", "B"}
                ]
                if choices:
                    participant_rates.append(
                        mean(
                            1.0
                            if response["trial_info"]["advisor_type"] == "agreeing"
                            else 0.0
                            for response in choices
                        )
                    )
            return _mean_or_none(participant_rates)

        reductions = {"accurate": [], "agreeing": []}
        for response in familiarisation:
            info = response.get("trial_info", {})
            advisor_type = info.get("advisor_type")
            if advisor_type in reductions:
                reductions[advisor_type].append(float(info["error_reduction"]))

        initial_errors = [
            float(response["trial_info"]["initial_error"])
            for response in familiarisation
        ]
        final_errors = [
            float(response["trial_info"]["final_error"])
            for response in familiarisation
        ]
        return {
            "descriptive_statistics": {
                "participants": len(participants),
                "core_trials": len(valid),
                "choice_trials": len(choice_trials),
                "familiarisation_trials": len(familiarisation),
                "parse_failures": len(responses) - len(valid),
                "mean_initial_absolute_error_years": _mean_or_none(initial_errors),
                "mean_final_absolute_error_years": _mean_or_none(final_errors),
                "mean_error_reduction_accurate_years": _mean_or_none(
                    reductions["accurate"]
                ),
                "mean_error_reduction_agreeing_years": _mean_or_none(
                    reductions["agreeing"]
                ),
                "agreeing_advisor_pick_rate_no_feedback": choice_rate(False),
                "agreeing_advisor_pick_rate_feedback": choice_rate(True),
                "reflected_control_share": (
                    sum(
                        response.get("trial_info", {}).get("advice_mode")
                        == "reflected_control"
                        for response in valid
                    )
                    / len(valid)
                    if valid
                    else None
                ),
                "mean_points": _mean_or_none(
                    [float(response["trial_info"]["points"]) for response in valid]
                ),
            },
            "inferential_statistics": {
                "reported_agreeing_pick_rate_no_feedback": 0.51,
                "reported_agreeing_pick_rate_feedback": 0.17,
                "reported_initial_error_years": 15.84,
                "reported_final_error_years": 10.28,
                "reported_error_reduction_accurate_years": 9.67,
                "reported_error_reduction_agreeing_years": 1.46,
            },
        }
