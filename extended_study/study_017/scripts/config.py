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
UNAIDED_PRACTICE_TRIALS = 10
PRACTICE_ADVISOR_TRIALS = 2
PRACTICE_ADVISOR_WIDTH = 8
PRACTICE_ADVISOR_VARIATION = 3
FAMILIARISATION_TRIALS = 15
CHOICE_TRIALS = 10
CORE_SLOTS = 2 * FAMILIARISATION_TRIALS + CHOICE_TRIALS
ATTENTION_CHECK_GLOBAL_INDICES = (16, 36)
ATTENTION_CHECK_CORE_INDICES = tuple(
    index - UNAIDED_PRACTICE_TRIALS - PRACTICE_ADVISOR_TRIALS
    for index in ATTENTION_CHECK_GLOBAL_INDICES
)
TOTAL_TRIAL_SLOTS = UNAIDED_PRACTICE_TRIALS + PRACTICE_ADVISOR_TRIALS + CORE_SLOTS
HISTORICAL_QUESTION_TRIALS = TOTAL_TRIAL_SLOTS - len(ATTENTION_CHECK_GLOBAL_INDICES)
NO_FEEDBACK_ID = "dates_task_3b_no_feedback"
FEEDBACK_ID = "dates_task_3c_feedback"
ORIGINAL_CONDITION_COUNTS = {
    NO_FEEDBACK_ID: 29,
    FEEDBACK_ID: 31,
}


def marker_contains(center: int, width: int, target: int) -> bool:
    """Return whether an integer target is covered by a centered marker."""

    half = width // 2
    return center - half <= target <= center + half


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return mean(values) if values else None


def number_to_digit_words(value: int) -> str:
    """Match the original interface's digit-by-digit attention-check wording."""

    words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    return " ".join(words[digit] for digit in str(value))


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
        phase_instructions: Optional[str] = None,
    ) -> str:
        feedback = f"Feedback from the previous trial: {pending_feedback}\n\n" if pending_feedback else ""
        transition = f"{phase_instructions}\n\n" if phase_instructions else ""
        return (
            f"{feedback}"
            f"{transition}"
            f"Historical dates task, trial {trial_number} of {TOTAL_TRIAL_SLOTS} "
            f"({block_label}).\n"
            f"Event: {event_prompt}\n\n"
            f"Estimate the year from {YEAR_MIN} through {YEAR_MAX}. Choose a marker "
            "width of 7, 13, or 21 years. If the marker covers the correct year it "
            "earns 25, 10, or 5 points respectively.\n"
            "Output exactly: YEAR=<1890-2010>; WIDTH=<7|13|21>"
        )

    def build_attention_check_prompt(
        self,
        *,
        target_year_words: str,
        trial_number: int,
        pending_feedback: Optional[str] = None,
    ) -> str:
        feedback = f"Feedback from the previous trial: {pending_feedback}\n\n" if pending_feedback else ""
        return (
            f"{feedback}"
            f"Historical dates task, trial {trial_number} of {TOTAL_TRIAL_SLOTS}.\n"
            "For this question use the smallest marker to cover the year "
            f"{target_year_words}.\n\n"
            "Output exactly: YEAR=<1890-2010>; WIDTH=7"
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
    SUPPORTED_SUB_STUDIES = (NO_FEEDBACK_ID, FEEDBACK_ID)

    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        if n_trials is None and len(self.selected_sub_studies) == 1:
            participant_count = ORIGINAL_CONDITION_COUNTS[
                self.selected_sub_studies[0]
            ]
        else:
            participant_count = (
                self.get_n_participants() if n_trials is None else int(n_trials)
            )
        if participant_count < 1:
            raise ValueError("study_017 requires at least one participant")
        material = self.load_material(self.MATERIAL_ID)
        trials = []
        for participant_index in range(participant_count):
            condition = self._condition_assignment(
                participant_index,
                participant_count,
            )
            trials.append({
                "trial_number": participant_index + 1,
                "participant_index": participant_index,
                "study_type": "stateful_judge_advisor_dates_task",
                "sub_study_id": condition["sub_study_id"],
                "condition_assignment": condition,
                "instructions": material["instructions"],
            })
        return trials

    def _condition_assignment(
        self,
        participant_index: int,
        participant_count: int,
    ) -> Dict[str, Any]:
        if len(self.selected_sub_studies) == 1:
            feedback = self.selected_sub_studies[0] == FEEDBACK_ID
            accurate_first = participant_index % 2 == 0
        elif participant_count == 60:
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
            "sub_study_id": FEEDBACK_ID if feedback else NO_FEEDBACK_ID,
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
    def _sample_practice_advice(
        rng: random.Random,
        correct_year: int,
    ) -> Tuple[int, str]:
        """Reproduce the correct practice advisor's width and variation."""

        lower = YEAR_MIN + PRACTICE_ADVISOR_WIDTH // 2
        upper = YEAR_MAX - PRACTICE_ADVISOR_WIDTH // 2
        advice_year = correct_year + rng.randint(
            -PRACTICE_ADVISOR_VARIATION,
            PRACTICE_ADVISOR_VARIATION,
        )
        return min(max(advice_year, lower), upper), "practice_correct"

    @staticmethod
    def attention_check_passed(
        estimate_year: int,
        marker_width: int,
        target_year: int,
    ) -> bool:
        return marker_width == min(PARTICIPANT_WIDTHS) and marker_contains(
            estimate_year,
            marker_width,
            target_year,
        )

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
        questions = questions[:HISTORICAL_QUESTION_TRIALS]
        question_index = 0

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
        terminated_early = False
        termination_reason: Optional[str] = None
        attention_checks: List[Dict[str, Any]] = []

        def run_historical_question(
            *,
            question: Dict[str, Any],
            global_trial_index: int,
            block_label: str,
            phase: str,
            advisor_mode: str,
            assigned_advisor: Optional[str],
            feedback_after_response: bool,
            phase_instructions: Optional[str] = None,
            core_slot_index: Optional[int] = None,
        ) -> None:
            nonlocal pending_feedback, no_feedback_preference

            trial_number = global_trial_index + 1
            initial_prompt = prompt_builder.build_initial_prompt(
                event_prompt=question["prompt"],
                trial_number=trial_number,
                block_label=block_label,
                pending_feedback=pending_feedback,
                phase_instructions=phase_instructions,
            )
            pending_feedback = None
            usage: Dict[str, Any] = {}
            correct_year = int(question["target_year"])
            if participant is not None:
                initial_year, initial_width, initial_response_text = self._call_estimate(
                    participant, initial_prompt, usage
                )
            else:
                initial_year, initial_width = self._mock_estimate(rng, correct_year)
                initial_response_text = f"YEAR={initial_year}; WIDTH={initial_width}"

            choice_letter: Optional[str] = None
            choice_response_text: Optional[str] = None
            choice_prompt: Optional[str] = None
            advisor_name: Optional[str] = None
            advice_year: Optional[int] = None
            advice_width: Optional[int] = None
            advice_mode: Optional[str] = None
            final_prompt: Optional[str] = None

            if advisor_mode == "choice":
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

            if advisor_mode == "none":
                final_year = initial_year
                final_width = initial_width
                final_response_text = initial_response_text
            else:
                if advisor_mode == "practice":
                    assigned_advisor = "practice"
                    advisor_name = "Practice advisor"
                    advice_year, advice_mode = self._sample_practice_advice(
                        rng, correct_year
                    )
                    advice_width = PRACTICE_ADVISOR_WIDTH
                else:
                    if assigned_advisor not in {"accurate", "agreeing"}:
                        raise ValueError("main advised trial has no valid assigned advisor")
                    advisor_name = type_to_name[assigned_advisor]
                    advice_year, advice_mode = self._sample_advice(
                        rng,
                        advisor_type=assigned_advisor,
                        initial_year=initial_year,
                        correct_year=correct_year,
                    )
                    advice_width = ADVISOR_WIDTH

                final_prompt = prompt_builder.build_final_prompt(
                    event_prompt=question["prompt"],
                    initial_year=initial_year,
                    initial_width=initial_width,
                    advisor_name=advisor_name,
                    advice_year=advice_year,
                    advice_width=advice_width,
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
                        advice_mode=(
                            "accurate" if advice_mode == "practice_correct" else advice_mode
                        ),
                        correct_year=correct_year,
                    )
                    final_response_text = f"YEAR={final_year}; WIDTH={final_width}"

            initial_error = abs(initial_year - correct_year)
            final_error = abs(final_year - correct_year)
            is_correct = marker_contains(final_year, final_width, correct_year)
            feedback_text: Optional[str] = None
            if feedback_after_response:
                feedback_text = (
                    f"The correct year was {correct_year}. Your "
                    f"{'final ' if advisor_mode != 'none' else ''}marker "
                    f"{'covered' if is_correct else 'did not cover'} the answer."
                )
                pending_feedback = feedback_text

            responses.append(
                {
                    "participant_id": participant_id,
                    "trial_number": trial_number,
                    "response": final_year,
                    "response_text": final_response_text,
                    "raw_response_text": final_response_text,
                    "usage": usage,
                    "correct_answer": correct_year,
                    "is_correct": is_correct,
                    "trial_info": {
                        "study_type": "stateful_judge_advisor_dates_task",
                        "sub_study_id": condition["sub_study_id"],
                        "feedback_condition": condition["feedback"],
                        "advisor_order": condition["advisor_order"],
                        "phase": phase,
                        "block": block_label,
                        "global_trial_index": global_trial_index,
                        "core_slot_index": core_slot_index,
                        "analysis_included": phase == "core",
                        "attention_check": False,
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
                        "advice_width": advice_width,
                        "final_year": final_year,
                        "final_width": final_width,
                        "final_error": final_error,
                        "error_reduction": initial_error - final_error,
                        "points": MARKER_POINTS[final_width] if is_correct else 0,
                        "answer_revealed_before_final": False,
                        "correct_year_revealed_after_response": feedback_after_response,
                        "correct_year_revealed_after_final": feedback_after_response,
                        "feedback_text": feedback_text,
                        "agent_visible_prompts": {
                            "initial": initial_prompt,
                            "choice": choice_prompt,
                            "final": final_prompt,
                        },
                    },
                }
            )

        for practice_index in range(UNAIDED_PRACTICE_TRIALS):
            question = questions[question_index]
            question_index += 1
            phase_instructions = None
            if practice_index == 0:
                phase_instructions = (
                    f"{trial['instructions']}\n\n"
                    "Practice: first complete ten questions without advice. "
                    "The correct year and whether your marker covered it are shown after "
                    "each practice response."
                )
            run_historical_question(
                question=question,
                global_trial_index=practice_index,
                block_label="unaided practice",
                phase="unaided_practice",
                advisor_mode="none",
                assigned_advisor=None,
                feedback_after_response=True,
                phase_instructions=phase_instructions,
            )

        for practice_advisor_index in range(PRACTICE_ADVISOR_TRIALS):
            question = questions[question_index]
            question_index += 1
            run_historical_question(
                question=question,
                global_trial_index=UNAIDED_PRACTICE_TRIALS + practice_advisor_index,
                block_label="practice with advice",
                phase="practice_advisor",
                advisor_mode="practice",
                assigned_advisor="practice",
                feedback_after_response=True,
                phase_instructions=(
                    "Practice with advice: give an initial answer, inspect the Practice "
                    "advisor's timeline marker, and then give a final answer. Feedback "
                    "follows each of these two trials."
                    if practice_advisor_index == 0
                    else None
                ),
            )

        for core_slot_index in range(CORE_SLOTS):
            global_trial_index = (
                UNAIDED_PRACTICE_TRIALS
                + PRACTICE_ADVISOR_TRIALS
                + core_slot_index
            )
            if core_slot_index in ATTENTION_CHECK_CORE_INDICES:
                target_year = rng.randrange(YEAR_MIN, YEAR_MAX)
                attention_prompt = prompt_builder.build_attention_check_prompt(
                    target_year_words=number_to_digit_words(target_year),
                    trial_number=global_trial_index + 1,
                    pending_feedback=pending_feedback,
                )
                pending_feedback = None
                usage: Dict[str, Any] = {}
                if participant is not None:
                    estimate_year, marker_width, response_text = self._call_estimate(
                        participant, attention_prompt, usage
                    )
                else:
                    estimate_year, marker_width = target_year, min(PARTICIPANT_WIDTHS)
                    response_text = f"YEAR={estimate_year}; WIDTH={marker_width}"
                passed = self.attention_check_passed(
                    estimate_year, marker_width, target_year
                )
                attention_record = {
                    "participant_id": participant_id,
                    "trial_number": global_trial_index + 1,
                    "response": estimate_year,
                    "response_text": response_text,
                    "raw_response_text": response_text,
                    "usage": usage,
                    "correct_answer": target_year,
                    "is_correct": passed,
                    "trial_info": {
                        "study_type": "stateful_judge_advisor_dates_task",
                        "sub_study_id": condition["sub_study_id"],
                        "feedback_condition": condition["feedback"],
                        "advisor_order": condition["advisor_order"],
                        "phase": "attention_check",
                        "block": "attention check",
                        "global_trial_index": global_trial_index,
                        "core_slot_index": core_slot_index,
                        "analysis_included": False,
                        "attention_check": True,
                        "attention_check_target_year": target_year,
                        "attention_check_target_words": number_to_digit_words(target_year),
                        "attention_check_required_width": min(PARTICIPANT_WIDTHS),
                        "attention_check_passed": passed,
                        "initial_year": estimate_year,
                        "initial_width": marker_width,
                        "final_year": estimate_year,
                        "final_width": marker_width,
                        "advisor_choice": None,
                        "advisor_display_name": None,
                        "advisor_type": None,
                        "target_explicitly_provided": True,
                        "answer_revealed_before_final": True,
                        "correct_year_revealed_after_response": False,
                        "correct_year_revealed_after_final": False,
                        "agent_visible_prompts": {
                            "initial": attention_prompt,
                            "choice": None,
                            "final": None,
                        },
                    },
                }
                responses.append(attention_record)
                attention_checks.append(
                    {
                        "global_trial_index": global_trial_index,
                        "core_slot_index": core_slot_index,
                        "target_year": target_year,
                        "estimate_year": estimate_year,
                        "marker_width": marker_width,
                        "passed": passed,
                    }
                )
                if not passed:
                    terminated_early = True
                    termination_reason = (
                        f"failed_attention_check_at_global_index_{global_trial_index}"
                    )
                    break
                continue

            question = questions[question_index]
            question_index += 1
            if core_slot_index < FAMILIARISATION_TRIALS:
                block_label = "familiarisation block 1"
                advisor_mode = "fixed"
                assigned_advisor = advisor_order[0]
            elif core_slot_index < 2 * FAMILIARISATION_TRIALS:
                block_label = "familiarisation block 2"
                advisor_mode = "fixed"
                assigned_advisor = advisor_order[1]
            else:
                block_label = "advisor-choice block"
                advisor_mode = "choice"
                assigned_advisor = None

            run_historical_question(
                question=question,
                global_trial_index=global_trial_index,
                block_label=block_label,
                phase="core",
                advisor_mode=advisor_mode,
                assigned_advisor=assigned_advisor,
                feedback_after_response=condition["feedback"],
                phase_instructions=(
                    "Main experiment: the two anonymous advisors may behave differently. "
                    "Learn how useful each is and use their advice accordingly."
                    if core_slot_index == 0
                    else None
                ),
                core_slot_index=core_slot_index,
            )

        if participant is not None:
            participant.clear_conversation()
        return {
            "participant_id": participant_id,
            "profile": {
                **profile,
                "feedback_condition": condition["feedback"],
                "advisor_order": condition["advisor_order"],
                "completed_unaided_practice_trials": UNAIDED_PRACTICE_TRIALS,
                "completed_practice_advisor_trials": PRACTICE_ADVISOR_TRIALS,
                "attention_checks": attention_checks,
                "terminated_early": terminated_early,
                "termination_reason": termination_reason,
            },
            "sub_study_id": condition["sub_study_id"],
            "terminated_early": terminated_early,
            "termination_reason": termination_reason,
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
        if len(question_bank) < HISTORICAL_QUESTION_TRIALS:
            raise ValueError("study_017 question bank must contain at least 50 questions")

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
        core_responses = [
            response
            for response in valid
            if response.get("trial_info", {}).get("analysis_included") is True
        ]
        choice_trials = [
            response
            for response in core_responses
            if response.get("trial_info", {}).get("advisor_choice") in {"A", "B"}
        ]
        familiarisation = [
            response
            for response in core_responses
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
                "completed_participants": sum(
                    not participant.get("terminated_early", False)
                    for participant in participants
                ),
                "attention_check_terminations": sum(
                    participant.get("terminated_early", False)
                    for participant in participants
                ),
                "total_trial_slots_recorded": len(valid),
                "core_historical_trials": len(core_responses),
                "choice_trials": len(choice_trials),
                "familiarisation_trials": len(familiarisation),
                "unaided_practice_trials": sum(
                    response.get("trial_info", {}).get("phase") == "unaided_practice"
                    for response in valid
                ),
                "practice_advisor_trials": sum(
                    response.get("trial_info", {}).get("phase") == "practice_advisor"
                    for response in valid
                ),
                "attention_checks": sum(
                    response.get("trial_info", {}).get("attention_check") is True
                    for response in valid
                ),
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
                        for response in core_responses
                    )
                    / len(core_responses)
                    if core_responses
                    else None
                ),
                "mean_points": _mean_or_none(
                    [
                        float(response["trial_info"]["points"])
                        for response in core_responses
                    ]
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
