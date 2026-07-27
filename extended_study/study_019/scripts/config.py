"""Runtime for Schöbel, Rieskamp, and Huber's two scenario experiments."""

from __future__ import annotations

import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder


STUDY_1_ID = "study_1_urn_scenarios"
STUDY_2_ID = "study_2_medical_authority_scenarios"
STUDY_1_SCENARIOS = 24
STUDY_2_SCENARIOS = 40
CONFIDENCE_MIN = 50
CONFIDENCE_MAX = 100


class SequentialSocialInfluencePromptBuilder(PromptBuilder):
    """Expose scenario evidence without source answers or reported outcomes."""

    def build_system_prompt(self, participant_profile: Dict[str, Any] = None) -> str:
        del participant_profile
        return (
            "Act as one participant in a social decision-making experiment. "
            "Use only the information presented in the task and do not use external "
            "tools or lookup. Treat each scenario as a new case, preserve the stated "
            "order of earlier decisions, and follow the requested output format."
        )

    def build_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        sub_study_id = str(trial_data["sub_study_id"])
        if sub_study_id == STUDY_1_ID:
            return self.build_urn_prompt(
                presented_trial_number=int(trial_data["presented_trial_number"]),
                previous_decisions=trial_data["previous_decisions"],
                private_ball=str(trial_data["private_ball"]),
                initial_instructions=trial_data.get("initial_instructions"),
            )
        if sub_study_id == STUDY_2_ID:
            return self.build_medical_prompt(
                presented_trial_number=int(trial_data["presented_trial_number"]),
                previous_diagnoses=trial_data["previous_diagnoses"],
                private_symptom=str(trial_data["private_symptom"]),
                initial_instructions=trial_data.get("initial_instructions"),
            )
        raise ValueError(f"unknown study_019 sub-study: {sub_study_id}")

    @staticmethod
    def _instructions(initial_instructions: Optional[str]) -> str:
        if not initial_instructions:
            return ""
        return f"Instructions:\n{initial_instructions}\n\n"

    def build_urn_prompt(
        self,
        *,
        presented_trial_number: int,
        previous_decisions: Sequence[str],
        private_ball: str,
        initial_instructions: Optional[str] = None,
    ) -> str:
        history = "\n".join(
            f"{position}. Person {position} predicted Urn {choice}."
            for position, choice in enumerate(previous_decisions, start=1)
        )
        return (
            f"{self._instructions(initial_instructions)}"
            f"Scenario {presented_trial_number} of {STUDY_1_SCENARIOS}.\n"
            "Earlier public predictions, in order:\n"
            f"{history}\n\n"
            f"Your private ball is {private_ball}.\n\n"
            "Which urn was most likely selected, and how confident are you that your "
            "prediction is correct?\n"
            "Output exactly: CHOICE=<A|B>; CONFIDENCE=<50-100>"
        )

    def build_medical_prompt(
        self,
        *,
        presented_trial_number: int,
        previous_diagnoses: Sequence[Dict[str, Any]],
        private_symptom: str,
        initial_instructions: Optional[str] = None,
    ) -> str:
        history = "\n".join(
            f"{item['position']}. The {item['role']} diagnosed {item['diagnosis']}."
            for item in previous_diagnoses
        )
        return (
            f"{self._instructions(initial_instructions)}"
            f"Scenario {presented_trial_number} of {STUDY_2_SCENARIOS}.\n"
            "Patient-record diagnoses, in order:\n"
            f"{history}\n\n"
            f"The patient has this symptom: {private_symptom}.\n\n"
            "Which disease is more likely, and how confident are you that your "
            "diagnosis is correct?\n"
            "Output exactly: DIAGNOSIS=<APPENDICITIS|SIGMOID_DIVERTICULITIS>; "
            "CONFIDENCE=<50-100>"
        )


class StudyStudy019Config(BaseStudyConfig):
    """Independent participants completing a full randomized scenario set."""

    prompt_builder_class = SequentialSocialInfluencePromptBuilder
    REQUIRES_GROUP_TRIALS = True
    MATERIAL_ID = "scenarios"
    SUPPORTED_SUB_STUDIES = (STUDY_1_ID, STUDY_2_ID)

    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        selected = self.selected_sub_studies or self.SUPPORTED_SUB_STUDIES
        if n_trials is not None:
            participant_count = int(n_trials)
        elif len(selected) == 1:
            participant_count = 40
        else:
            participant_count = self.get_n_participants()
        minimum = 1 if len(selected) == 1 else 2
        if participant_count < minimum:
            raise ValueError(
                f"study_019 selected scope requires at least {minimum} participant(s)"
            )
        if len(selected) > 1 and participant_count % 2:
            raise ValueError(
                "study_019 participant count must be even so both studies are represented"
            )
        return [
            {
                "trial_number": participant_index + 1,
                "participant_index": participant_index,
                "study_type": "randomized_social_influence_scenario_tasks",
                "sub_study_id": selected[participant_index % len(selected)],
            }
            for participant_index in range(participant_count)
        ]

    @staticmethod
    def parse_urn_response(response_text: str) -> Optional[Tuple[str, int]]:
        if not response_text:
            return None
        choice_match = re.search(
            r"\bCHOICE\s*[:=]\s*([AB])\b",
            response_text,
            re.IGNORECASE,
        )
        confidence_match = re.search(
            r"\bCONFIDENCE\s*[:=]\s*(\d{1,3})\b",
            response_text,
            re.IGNORECASE,
        )
        if not choice_match or not confidence_match:
            return None
        confidence = int(confidence_match.group(1))
        if not CONFIDENCE_MIN <= confidence <= CONFIDENCE_MAX:
            return None
        return choice_match.group(1).upper(), confidence

    @staticmethod
    def parse_medical_response(response_text: str) -> Optional[Tuple[str, int]]:
        if not response_text:
            return None
        diagnosis_match = re.search(
            r"\bDIAGNOSIS\s*[:=]\s*"
            r"(APPENDICITIS|SIGMOID(?:[_\s-]+)DIVERTICULITIS)\b",
            response_text,
            re.IGNORECASE,
        )
        confidence_match = re.search(
            r"\bCONFIDENCE\s*[:=]\s*(\d{1,3})\b",
            response_text,
            re.IGNORECASE,
        )
        if not diagnosis_match or not confidence_match:
            return None
        raw_diagnosis = diagnosis_match.group(1).upper()
        diagnosis = (
            "appendicitis"
            if raw_diagnosis == "APPENDICITIS"
            else "sigmoid diverticulitis"
        )
        confidence = int(confidence_match.group(1))
        if not CONFIDENCE_MIN <= confidence <= CONFIDENCE_MAX:
            return None
        return diagnosis, confidence

    @staticmethod
    def _merge_usage(total: Dict[str, Any], new_usage: Dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = int(total.get(key, 0) or 0) + int(new_usage.get(key, 0) or 0)
        total["cost"] = float(total.get("cost", 0.0) or 0.0) + float(
            new_usage.get("cost", 0.0) or 0.0
        )

    def _call_response(
        self,
        participant: Any,
        *,
        sub_study_id: str,
        prompt: str,
        usage: Dict[str, Any],
    ) -> Tuple[str, int, str]:
        parser = (
            self.parse_urn_response
            if sub_study_id == STUDY_1_ID
            else self.parse_medical_response
        )
        result = participant.continue_conversation(prompt, max_tokens=64)
        self._merge_usage(usage, result.get("usage", {}))
        response_text = str(result.get("response_text", ""))
        parsed = parser(response_text)
        if parsed is None:
            repair_prompt = (
                "Your previous response was invalid. Output exactly "
                "CHOICE=<A|B>; CONFIDENCE=<50-100>."
                if sub_study_id == STUDY_1_ID
                else "Your previous response was invalid. Output exactly "
                "DIAGNOSIS=<APPENDICITIS|SIGMOID_DIVERTICULITIS>; "
                "CONFIDENCE=<50-100>."
            )
            repair = participant.continue_conversation(repair_prompt, max_tokens=48)
            self._merge_usage(usage, repair.get("usage", {}))
            response_text = str(repair.get("response_text", ""))
            parsed = parser(response_text)
        if parsed is None:
            raise ValueError("participant returned no valid decision and confidence")
        return parsed[0], parsed[1], response_text

    @staticmethod
    def _mock_response(
        rng: random.Random,
        *,
        sub_study_id: str,
        scenario: Dict[str, Any],
    ) -> Tuple[str, int, str]:
        human = scenario["human_raw_data"]
        choose_option_1 = rng.random() < float(human["option_1_rate"])
        if sub_study_id == STUDY_1_ID:
            choice = "A" if choose_option_1 else "B"
            prefix = f"CHOICE={choice}"
        else:
            choice = (
                "appendicitis"
                if choose_option_1
                else "sigmoid diverticulitis"
            )
            encoded = (
                "APPENDICITIS"
                if choice == "appendicitis"
                else "SIGMOID_DIVERTICULITIS"
            )
            prefix = f"DIAGNOSIS={encoded}"
        confidence = round(
            rng.gauss(float(human["mean_confidence"]), 4.0)
        )
        confidence = min(max(confidence, CONFIDENCE_MIN), CONFIDENCE_MAX)
        return choice, confidence, f"{prefix}; CONFIDENCE={confidence}"

    @staticmethod
    def _probability_for_choice(
        sub_study_id: str,
        scenario: Dict[str, Any],
        choice: str,
    ) -> float:
        if sub_study_id == STUDY_1_ID:
            probability_option_1 = float(scenario["posterior_probability_urn_a"])
            return probability_option_1 if choice == "A" else 1.0 - probability_option_1
        probability_option_1 = float(
            scenario["posterior_probability_appendicitis"]
        )
        return (
            probability_option_1
            if choice == "appendicitis"
            else 1.0 - probability_option_1
        )

    def _run_one_participant(
        self,
        trial: Dict[str, Any],
        *,
        participant: Optional[Any],
        profile: Dict[str, Any],
        prompt_builder: SequentialSocialInfluencePromptBuilder,
        base_seed: int,
        material: Dict[str, Any],
    ) -> Dict[str, Any]:
        participant_id = int(trial["participant_index"])
        sub_study_id = str(trial["sub_study_id"])
        material_key = "study_1" if sub_study_id == STUDY_1_ID else "study_2"
        sub_material = material[material_key]
        scenarios = [dict(scenario) for scenario in sub_material["scenarios"]]
        rng = random.Random(base_seed + 100_003 * participant_id)
        rng.shuffle(scenarios)

        if participant is not None:
            participant.start_conversation()

        responses: List[Dict[str, Any]] = []
        for presented_index, scenario in enumerate(scenarios, start=1):
            prompt_data = {
                **scenario["source_signature"],
                "sub_study_id": sub_study_id,
                "presented_trial_number": presented_index,
                "initial_instructions": (
                    sub_material["instructions"] if presented_index == 1 else None
                ),
            }
            prompt = prompt_builder.build_trial_prompt(prompt_data)
            usage: Dict[str, Any] = {}
            if participant is not None:
                choice, confidence, response_text = self._call_response(
                    participant,
                    sub_study_id=sub_study_id,
                    prompt=prompt,
                    usage=usage,
                )
            else:
                choice, confidence, response_text = self._mock_response(
                    rng,
                    sub_study_id=sub_study_id,
                    scenario=scenario,
                )

            bayesian_choice = scenario["bayesian_choice"]
            probability_for_choice = self._probability_for_choice(
                sub_study_id,
                scenario,
                choice,
            )
            is_correct = (
                choice == bayesian_choice if bayesian_choice is not None else None
            )
            responses.append(
                {
                    "participant_id": participant_id,
                    "trial_number": presented_index,
                    "response": choice,
                    "response_text": response_text,
                    "raw_response_text": response_text,
                    "usage": usage,
                    "correct_answer": bayesian_choice,
                    "is_correct": is_correct,
                    "trial_info": {
                        "study_type": "randomized_social_influence_scenario_tasks",
                        "sub_study_id": sub_study_id,
                        "presented_trial_number": presented_index,
                        "scenario_id": scenario["scenario_id"],
                        "raw_variable_id": scenario["raw_variable_id"],
                        "article_scenario_id": scenario["article_scenario_id"],
                        "material_fingerprint": scenario["material_fingerprint"],
                        "private_information_favors": scenario[
                            "private_information_favors"
                        ],
                        "bayesian_choice": bayesian_choice,
                        "bayesian_confidence": scenario["bayesian_confidence"],
                        "bayesian_probability_for_choice": probability_for_choice,
                        "confidence": confidence,
                        "confidence_absolute_error": abs(
                            confidence / 100.0 - probability_for_choice
                        ),
                        "decision_bonus_eligible": is_correct,
                        "confidence_bonus_eligible": (
                            abs(confidence / 100.0 - probability_for_choice) <= 0.05
                        ),
                        "cascade_scenario": scenario["cascade_scenario"],
                        "indifference_scenario": scenario["indifference_scenario"],
                        "authority_condition": scenario.get("authority_condition"),
                        "medical_director_diagnosis": scenario.get(
                            "medical_director_diagnosis"
                        ),
                        "posterior_probability_urn_a": scenario.get(
                            "posterior_probability_urn_a"
                        ),
                        "posterior_probability_appendicitis": scenario.get(
                            "posterior_probability_appendicitis"
                        ),
                        "source_signature": scenario["source_signature"],
                        "source_material_verified": True,
                        "scenario_order_randomized": True,
                        "answer_revealed_before_response": False,
                        "feedback_provided": False,
                        "initial_instructions_shown": presented_index == 1,
                        "agent_visible_prompt": prompt,
                    },
                }
            )

        if participant is not None:
            participant.clear_conversation()
        return {
            "participant_id": participant_id,
            "profile": {
                **profile,
                "sub_study_id": sub_study_id,
                "scenario_count": len(scenarios),
            },
            "sub_study_id": sub_study_id,
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
        if not isinstance(builder, SequentialSocialInfluencePromptBuilder):
            raise TypeError(
                "study_019 requires SequentialSocialInfluencePromptBuilder"
            )
        material = self.load_material(self.MATERIAL_ID)
        if len(material["study_1"]["scenarios"]) != STUDY_1_SCENARIOS:
            raise ValueError("study_019 Study 1 material must contain 24 scenarios")
        if len(material["study_2"]["scenarios"]) != STUDY_2_SCENARIOS:
            raise ValueError("study_019 Study 2 material must contain 40 scenarios")

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
                        material=material,
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
                        material=material,
                    )
                )
        results.sort(key=lambda row: int(row["participant_id"]))
        return {"individual_data": results}

    @staticmethod
    def _rate(rows: Sequence[Dict[str, Any]], predicate: Any) -> Optional[float]:
        return mean(bool(predicate(row)) for row in rows) if rows else None

    @staticmethod
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

    @classmethod
    def _mean_probability_judgment(
        cls,
        rows: Sequence[Dict[str, Any]],
    ) -> Optional[float]:
        return (
            mean(cls._probability_judgment_for_target(row) for row in rows)
            if rows
            else None
        )

    @staticmethod
    def _posterior_group(row: Dict[str, Any]) -> str:
        confidence = float(row["trial_info"]["bayesian_confidence"])
        return f"posterior_{confidence:.2f}"

    def aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        participants = raw_results.get("individual_data", [])
        responses = [
            response
            for participant in participants
            for response in participant.get("responses", [])
        ]
        by_study: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for response in responses:
            by_study[response["trial_info"]["sub_study_id"]].append(response)

        diagnostics: Dict[str, Any] = {}
        for sub_study_id, rows in by_study.items():
            non_ties = [
                row for row in rows if row["trial_info"]["bayesian_choice"] is not None
            ]
            ties = [
                row for row in rows if row["trial_info"]["indifference_scenario"]
            ]
            cascades = [
                row for row in rows if row["trial_info"]["cascade_scenario"]
            ]
            private_supported = [
                row
                for row in non_ties
                if row["trial_info"]["bayesian_choice"]
                == row["trial_info"]["private_information_favors"]
            ]
            study_stats: Dict[str, Any] = {
                "bayesian_choice_rate_non_indifference": self._rate(
                    non_ties,
                    lambda row: row["response"]
                    == row["trial_info"]["bayesian_choice"],
                ),
                "bayesian_choice_rate_when_private_supported": self._rate(
                    private_supported,
                    lambda row: row["response"]
                    == row["trial_info"]["bayesian_choice"],
                ),
                "private_choice_rate_at_bayesian_indifference": self._rate(
                    ties,
                    lambda row: row["response"]
                    == row["trial_info"]["private_information_favors"],
                ),
                "cascade_choice_rate": self._rate(
                    cascades,
                    lambda row: row["response"]
                    == row["trial_info"]["bayesian_choice"],
                ),
                "mean_confidence": (
                    mean(row["trial_info"]["confidence"] for row in rows)
                    if rows
                    else None
                ),
            }
            posterior_groups = sorted(
                {self._posterior_group(row) for row in rows}
            )
            study_stats["probability_judgment_by_bayesian_probability"] = {
                group: self._mean_probability_judgment(
                    [
                        row
                        for row in rows
                        if self._posterior_group(row) == group
                    ]
                )
                for group in posterior_groups
            }
            if sub_study_id == STUDY_2_ID:
                director_rows = [
                    row
                    for row in rows
                    if row["trial_info"]["medical_director_diagnosis"] is not None
                ]
                study_stats["medical_director_alignment_rate"] = self._rate(
                    director_rows,
                    lambda row: row["response"]
                    == row["trial_info"]["medical_director_diagnosis"],
                )
                authority_cells: Dict[str, Dict[str, Dict[str, Any]]] = {}
                for group in posterior_groups:
                    group_rows = [
                        row
                        for row in rows
                        if self._posterior_group(row) == group
                    ]
                    conditions = sorted(
                        {
                            str(row["trial_info"]["authority_condition"])
                            for row in group_rows
                        }
                    )
                    authority_cells[group] = {}
                    for condition in conditions:
                        cell_rows = [
                            row
                            for row in group_rows
                            if row["trial_info"]["authority_condition"] == condition
                        ]
                        authority_cells[group][condition] = {
                            "responses": len(cell_rows),
                            "private_choice_rate": self._rate(
                                cell_rows,
                                lambda row: row["response"]
                                == row["trial_info"]["private_information_favors"],
                            ),
                            "probability_judgment": (
                                self._mean_probability_judgment(cell_rows)
                            ),
                        }
                study_stats["authority_condition_cells"] = authority_cells
            diagnostics[sub_study_id] = study_stats

        return {
            **raw_results,
            "descriptive_statistics": {
                "participants": len(participants),
                "participants_by_sub_study": {
                    sub_study_id: sum(
                        participant.get("sub_study_id") == sub_study_id
                        for participant in participants
                    )
                    for sub_study_id in (STUDY_1_ID, STUDY_2_ID)
                },
                "responses": len(responses),
                "responses_by_sub_study": {
                    sub_study_id: len(by_study.get(sub_study_id, []))
                    for sub_study_id in (STUDY_1_ID, STUDY_2_ID)
                },
                "parse_failures": sum(
                    not isinstance(row.get("trial_info", {}).get("confidence"), int)
                    for row in responses
                ),
            },
            "behavioral_diagnostics": diagnostics,
        }
