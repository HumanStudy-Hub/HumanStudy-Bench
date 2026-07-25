"""Runtime for Molleman et al.'s disparate social-information experiment."""

from __future__ import annotations

import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, pvariance
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder


ESTIMATE_MIN = 1
ESTIMATE_MAX = 150
ONE_PEER_ROUNDS = 5
MAIN_ROUNDS = 30
FOUR_PEER_ROUNDS = 5
CORE_RESPONSES_PER_PARTICIPANT = (
    ONE_PEER_ROUNDS + MAIN_ROUNDS + FOUR_PEER_ROUNDS
)
COMPREHENSION_MAX_ATTEMPTS = 3
STUDY_TYPE = "multimodal_social_information_revision"
CONDITION_NAMES = {
    "LN": "low variance, no skew",
    "HN": "high variance, no skew",
    "HF": "high variance, peers clustered far from the initial estimate",
    "HC": "high variance, peers clustered close to the initial estimate",
    "filler": "filler distribution",
}
REPORTED_MEAN_ADJUSTMENT = {
    "LN": 0.415,
    "HN": 0.290,
    "HF": 0.279,
    "HC": 0.365,
}
REPORTED_KEEP_PROBABILITY = {
    "LN": 0.07,
    "HN": 0.16,
    "HF": 0.16,
    "HC": 0.06,
    "filler": 0.10,
}
REPORTED_ADOPT_NEAREST_PROBABILITY = {
    "LN": 0.02,
    "HN": 0.07,
    "HF": 0.09,
    "HC": 0.06,
    "filler": 0.05,
}


def clamp_estimate(value: float) -> int:
    return min(max(round(value), ESTIMATE_MIN), ESTIMATE_MAX)


def distribution_skew(values: Sequence[float]) -> float:
    center = mean(values)
    second = mean((value - center) ** 2 for value in values)
    if second == 0:
        return 0.0
    third = mean((value - center) ** 3 for value in values)
    return third / (second**1.5)


def classify_adjustment(initial: int, final: int, peers: Sequence[int]) -> str:
    """Apply the strategy definitions used for Supplementary Table S2."""

    nearest = min(peers, key=lambda peer: (abs(peer - initial), peer))
    if final == initial:
        return "keep"
    if final == nearest:
        return "adopt_nearest"
    if min(initial, nearest) < final < max(initial, nearest):
        return "compromise"
    return "other"


def social_information_use(
    initial: int,
    final: int,
    peers: Sequence[int],
) -> Optional[float]:
    peer_mean = mean(peers)
    denominator = peer_mean - initial
    if denominator == 0:
        return None
    return (final - initial) / denominator


class DisparateSocialInformationPromptBuilder(PromptBuilder):
    """Build the visual first estimate and social revision prompts."""

    _PLURALS = {
        "ant": "ants",
        "bee": "bees",
        "flamingo": "flamingos",
        "crane": "cranes",
        "cricket": "crickets",
    }

    def build_system_prompt(self, participant_profile: Dict[str, Any] = None) -> str:
        del participant_profile
        return (
            "Act as one participant in a visual estimation experiment. "
            "Use only the image and numerical information presented in the task. "
            "Do not use tools, code, image metadata, file names, or external lookup. "
            "Respond independently and follow the requested format exactly."
        )

    def build_trial_prompt(self, trial_data: Dict[str, Any]) -> Any:
        return self.build_initial_prompt(
            round_number=int(trial_data["round"]),
            species=str(trial_data["species"]),
            image_path=Path(trial_data["image_path"]),
        )

    def build_initial_prompt(
        self,
        *,
        round_number: int,
        species: str,
        image_path: Path,
    ) -> List[Dict[str, Any]]:
        plural = self._PLURALS[species]
        return [
            {
                "type": "text",
                "text": (
                    f"Round {round_number}, part A. Observe the image once. Estimate how many "
                    f"{plural} it contains. The allowed range is 1 through 150. "
                    "After this response the image will be removed. "
                    "Output exactly: ESTIMATE=<1-150>"
                ),
            },
            {
                "type": "image",
                "image": str(image_path),
                "mime": "image/png",
            },
        ]

    def build_social_prompt(
        self,
        *,
        round_number: int,
        species: str,
        initial_estimate: int,
        peer_estimates: Sequence[int],
    ) -> str:
        plural = self._PLURALS[species]
        peers = ", ".join(str(value) for value in peer_estimates)
        peer_count = len(peer_estimates)
        peer_label = "One previous participant estimated" if peer_count == 1 else (
            f"{peer_count} previous participants estimated"
        )
        return (
            f"Round {round_number}, part B. The image of the {plural} is no longer "
            f"visible. Your part A estimate was {initial_estimate}. {peer_label}: "
            f"{peers}. Give your second estimate from 1 "
            "through 150. You may keep or revise your estimate. "
            "Output exactly: ESTIMATE=<1-150>"
        )

    def build_control_prompt(
        self,
        *,
        round_number: int,
        peer_estimates: Sequence[int],
    ) -> str:
        peers = ", ".join(str(value) for value in peer_estimates)
        return (
            f"Four-estimate control round {round_number}. No animal image and no "
            "personal first estimate are available. Four previous participants "
            f"estimated: {peers}. Integrate this information and give one estimate "
            "from 1 through 150. Output exactly: ESTIMATE=<1-150>"
        )

    def build_comprehension_prompt(
        self,
        *,
        block_label: str,
        instructions: str,
        statements: Sequence[str],
    ) -> str:
        numbered = "\n".join(
            f"{index}. {statement}"
            for index, statement in enumerate(statements, start=1)
        )
        return (
            f"Instructions for the {block_label} block:\n{instructions}\n\n"
            "Check your understanding. Mark each statement C if it is correct "
            "according to the instructions or I if it is incorrect.\n"
            f"{numbered}\n\n"
            f"Output exactly: ANSWERS=<{len(statements)} comma-separated C or I values>"
        )


class StudyStudy018Config(BaseStudyConfig):
    """Independent participants complete all three counterbalanced task blocks."""

    prompt_builder_class = DisparateSocialInformationPromptBuilder
    REQUIRES_GROUP_TRIALS = True
    MATERIAL_ID = "disparate_social_information"
    LOOKUP_ID = "peer_lookup"

    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        participant_count = self.get_n_participants() if n_trials is None else int(n_trials)
        if participant_count < 1:
            raise ValueError("study_018 requires at least one participant")
        material = self.load_material(self.MATERIAL_ID)
        return [
            {
                "trial_number": participant_index + 1,
                "participant_index": participant_index,
                "study_type": STUDY_TYPE,
                "sub_study_id": "complete_three_block_session",
                "instructions": material["instructions"],
                "block_instructions": material["block_instructions"],
                "comprehension_checks": material["comprehension_checks"],
            }
            for participant_index in range(participant_count)
        ]

    @staticmethod
    def parse_estimate(response_text: str) -> Optional[int]:
        if not response_text:
            return None
        match = re.search(
            r"\bESTIMATE\s*[:=]\s*(\d{1,3})\b",
            str(response_text),
            re.IGNORECASE,
        )
        if match is None:
            compact = str(response_text).strip()
            match = re.fullmatch(r"(\d{1,3})", compact)
        if match is None:
            return None
        value = int(match.group(1))
        return value if ESTIMATE_MIN <= value <= ESTIMATE_MAX else None

    @staticmethod
    def parse_comprehension_answers(
        response_text: str,
        *,
        expected_count: int,
    ) -> Optional[List[bool]]:
        if not response_text:
            return None
        match = re.search(
            r"\bANSWERS\s*[:=]\s*([A-Za-z,\s]+)",
            str(response_text),
            re.IGNORECASE,
        )
        if match is None:
            return None
        tokens = [
            token.strip().lower()
            for token in match.group(1).split(",")
            if token.strip()
        ]
        if len(tokens) != expected_count:
            return None
        mapping = {
            "c": True,
            "correct": True,
            "true": True,
            "t": True,
            "i": False,
            "incorrect": False,
            "false": False,
            "f": False,
        }
        if any(token not in mapping for token in tokens):
            return None
        return [mapping[token] for token in tokens]

    @staticmethod
    def _merge_usage(total: Dict[str, Any], new_usage: Dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = int(total.get(key, 0) or 0) + int(new_usage.get(key, 0) or 0)
        total["cost"] = float(total.get("cost", 0.0) or 0.0) + float(
            new_usage.get("cost", 0.0) or 0.0
        )

    def _call_estimate(
        self,
        participant: Any,
        prompt: Any,
        usage: Dict[str, Any],
    ) -> Tuple[int, str]:
        result = participant.continue_conversation(prompt, max_tokens=128)
        self._merge_usage(usage, result.get("usage", {}))
        response_text = str(result.get("response_text", ""))
        estimate = self.parse_estimate(response_text)
        if estimate is None:
            repair = participant.continue_conversation(
                "Your previous response was invalid. Output exactly "
                "ESTIMATE=<one integer from 1 through 150>.",
                max_tokens=64,
            )
            self._merge_usage(usage, repair.get("usage", {}))
            response_text = str(repair.get("response_text", ""))
            estimate = self.parse_estimate(response_text)
        if estimate is None:
            raise ValueError("participant returned no valid estimate from 1 through 150")
        return estimate, response_text

    def _run_comprehension_check(
        self,
        *,
        participant: Optional[Any],
        prompt_builder: DisparateSocialInformationPromptBuilder,
        block_name: str,
        instructions: str,
        quiz: Dict[str, Any],
    ) -> Dict[str, Any]:
        statements = [str(item["statement"]) for item in quiz["items"]]
        expected = [bool(item["correct"]) for item in quiz["items"]]
        prompt = prompt_builder.build_comprehension_prompt(
            block_label=block_name.replace("_", " "),
            instructions=instructions,
            statements=statements,
        )
        usage: Dict[str, Any] = {}
        raw_responses: List[str] = []

        if participant is None:
            parsed = list(expected)
            raw_responses.append(
                "ANSWERS="
                + ",".join("C" if answer else "I" for answer in parsed)
            )
        else:
            participant.start_conversation()
            parsed = None
            current_prompt = prompt
            for attempt in range(1, COMPREHENSION_MAX_ATTEMPTS + 1):
                result = participant.continue_conversation(
                    current_prompt,
                    max_tokens=64,
                )
                self._merge_usage(usage, result.get("usage", {}))
                response_text = str(result.get("response_text", ""))
                raw_responses.append(response_text)
                parsed = self.parse_comprehension_answers(
                    response_text,
                    expected_count=len(expected),
                )
                if parsed == expected:
                    break
                if parsed is None:
                    feedback = "Your response did not match the required format."
                else:
                    wrong = [
                        str(index)
                        for index, (answer, correct) in enumerate(
                            zip(parsed, expected),
                            start=1,
                        )
                        if answer != correct
                    ]
                    feedback = (
                        "Your answers to statement(s) "
                        + ", ".join(wrong)
                        + " were incorrect."
                    )
                current_prompt = (
                    f"{feedback} Re-read the instructions and answer all "
                    f"{len(expected)} statements. Output exactly: "
                    "ANSWERS=<comma-separated C or I values>"
                )
            participant.clear_conversation()

        return {
            "block": block_name,
            "source_stages": list(quiz.get("source_stages", [])),
            "passed": parsed == expected,
            "attempts": len(raw_responses),
            "max_attempts": COMPREHENSION_MAX_ATTEMPTS,
            "answers": parsed,
            "raw_responses": raw_responses,
            "usage": usage,
        }

    @staticmethod
    def _peer_estimates(
        lookup_block: Dict[str, Any],
        *,
        round_index: int,
        initial_estimate: int,
    ) -> List[int]:
        if not ESTIMATE_MIN <= initial_estimate <= ESTIMATE_MAX:
            raise ValueError("initial estimate cannot index the peer lookup")
        peers = [
            int(lookup_block[peer_name][round_index][initial_estimate - 1])
            for peer_name in ("p1", "p2", "p3")
        ]
        if any(not ESTIMATE_MIN <= peer <= ESTIMATE_MAX for peer in peers):
            raise ValueError("peer lookup returned an out-of-range estimate")
        return peers

    @staticmethod
    def _sample_one_peer(
        lookup_block: Dict[str, Any],
        *,
        round_index: int,
        initial_estimate: int,
        rng: random.Random,
    ) -> Tuple[int, Dict[str, Any]]:
        if not ESTIMATE_MIN <= initial_estimate <= ESTIMATE_MAX:
            raise ValueError("initial estimate cannot select one-peer information")
        true_count = int(lookup_block["rounds"][round_index]["true_count"])
        if initial_estimate < true_count:
            direction = "higher"
            multiplier = 1.2
        elif initial_estimate > true_count:
            direction = "lower"
            multiplier = 0.8
        elif rng.random() < 0.5:
            direction = "lower"
            multiplier = 0.8
        else:
            direction = "higher"
            multiplier = 1.2

        target = initial_estimate * multiplier
        pool = [int(value) for value in lookup_block["peer_pools"][round_index]]
        minimum_distance = float("inf")
        candidates: List[int] = []
        for value in pool:
            distance = abs(value - target)
            if distance < minimum_distance:
                minimum_distance = distance
                candidates = [value]
            if distance == minimum_distance:
                # The source uses consecutive `if` statements, so the first
                # value establishing the final minimum is inserted twice.
                candidates.append(value)
        peer = int(rng.choice(candidates))
        if not ESTIMATE_MIN <= peer <= ESTIMATE_MAX:
            raise ValueError("one-peer source selection returned an invalid estimate")
        return peer, {
            "target": target,
            "direction": direction,
            "target_multiplier": multiplier,
            "candidate_peers": candidates,
        }

    @staticmethod
    def _mock_initial(rng: random.Random, true_count: int) -> int:
        return clamp_estimate(rng.gauss(true_count, 19.0))

    @staticmethod
    def _mock_social_revision(
        rng: random.Random,
        *,
        condition: str,
        initial: int,
        peers: Sequence[int],
    ) -> int:
        keep_probability = REPORTED_KEEP_PROBABILITY.get(condition, 0.10)
        adopt_probability = REPORTED_ADOPT_NEAREST_PROBABILITY.get(condition, 0.05)
        draw = rng.random()
        if draw < keep_probability:
            return initial

        nearest = min(peers, key=lambda peer: (abs(peer - initial), peer))
        if draw < keep_probability + adopt_probability:
            return nearest

        target = REPORTED_MEAN_ADJUSTMENT.get(condition, 0.33)
        noisy_weight = max(-0.05, min(1.15, rng.gauss(target, 0.08)))
        return clamp_estimate(initial + noisy_weight * (mean(peers) - initial))

    @staticmethod
    def _mock_control_estimate(rng: random.Random, peers: Sequence[int]) -> int:
        distances = [
            abs(peer - mean(peers)) + 1.0
            for peer in peers
        ]
        weights = [1.0 / distance for distance in distances]
        weighted = sum(peer * weight for peer, weight in zip(peers, weights)) / sum(weights)
        return clamp_estimate(rng.gauss(weighted, 1.5))

    @staticmethod
    def _score_estimate(estimate: int, true_count: int) -> int:
        return max(0, 100 - 5 * abs(estimate - true_count))

    @staticmethod
    def _prompt_text(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        return "\n".join(
            str(part.get("text", ""))
            for part in prompt
            if isinstance(part, dict) and part.get("type") == "text"
        )

    def _run_main_round(
        self,
        *,
        participant: Optional[Any],
        prompt_builder: DisparateSocialInformationPromptBuilder,
        rng: random.Random,
        participant_id: int,
        global_trial_number: int,
        round_data: Dict[str, Any],
        main_lookup: Dict[str, Any],
    ) -> Dict[str, Any]:
        round_number = int(round_data["round"])
        species = str(round_data["species"])
        true_count = int(round_data["true_count"])
        condition = str(round_data["condition"])
        image_path = (
            self.source_path / "stimuli" / f"round_{round_number:02d}_{species}.png"
        )
        if not image_path.exists():
            raise FileNotFoundError(f"stimulus image not found: {image_path}")

        initial_prompt = prompt_builder.build_initial_prompt(
            round_number=round_number,
            species=species,
            image_path=image_path,
        )
        usage: Dict[str, Any] = {}
        if participant is not None:
            participant.start_conversation()
            initial, initial_response_text = self._call_estimate(
                participant,
                initial_prompt,
                usage,
            )
            participant.clear_conversation()
        else:
            initial = self._mock_initial(rng, true_count)
            initial_response_text = f"ESTIMATE={initial}"

        peers = self._peer_estimates(
            main_lookup,
            round_index=round_number - 1,
            initial_estimate=initial,
        )
        social_prompt = prompt_builder.build_social_prompt(
            round_number=round_number,
            species=species,
            initial_estimate=initial,
            peer_estimates=peers,
        )
        if participant is not None:
            participant.start_conversation()
            final, final_response_text = self._call_estimate(
                participant,
                social_prompt,
                usage,
            )
            participant.clear_conversation()
        else:
            final = self._mock_social_revision(
                rng,
                condition=condition,
                initial=initial,
                peers=peers,
            )
            final_response_text = f"ESTIMATE={final}"

        peer_mean = mean(peers)
        return {
            "participant_id": participant_id,
            "trial_number": global_trial_number,
            "response": final,
            "response_text": final_response_text,
            "raw_response_text": final_response_text,
            "usage": usage,
            "correct_answer": true_count,
            "is_correct": final == true_count,
            "trial_info": {
                "study_type": STUDY_TYPE,
                "sub_study_id": "main_task",
                "block": "main_task",
                "main_round": round_number,
                "species": species,
                "stimulus_file": image_path.name,
                "stimulus_presented": True,
                "stimulus_seconds_in_human_interface": 6,
                "visual_exposure_emulation": (
                    "single multimodal first-estimate call; image absent from revision"
                ),
                "condition": condition,
                "condition_description": CONDITION_NAMES[condition],
                "treatment_code": int(round_data["treatment_code"]),
                "initial_estimate": initial,
                "initial_response_text": initial_response_text,
                "peer_estimates": peers,
                "peer_mean": peer_mean,
                "peer_variance": pvariance(peers),
                "peer_skewness": distribution_skew(peers),
                "final_estimate": final,
                "social_information_use": social_information_use(initial, final, peers),
                "strategy": classify_adjustment(initial, final, peers),
                "initial_absolute_error": abs(initial - true_count),
                "final_absolute_error": abs(final - true_count),
                "points": self._score_estimate(final, true_count),
                "answer_revealed_before_response": False,
                "peer_lookup_verified": True,
                "agent_visible_prompts": {
                    "initial": self._prompt_text(initial_prompt),
                    "social_revision": social_prompt,
                },
            },
        }

    def _run_one_peer_round(
        self,
        *,
        participant: Optional[Any],
        prompt_builder: DisparateSocialInformationPromptBuilder,
        rng: random.Random,
        participant_id: int,
        global_trial_number: int,
        round_data: Dict[str, Any],
        one_peer_lookup: Dict[str, Any],
    ) -> Dict[str, Any]:
        round_number = int(round_data["round"])
        species = str(round_data["species"])
        true_count = int(round_data["true_count"])
        image_path = (
            self.source_path
            / "stimuli"
            / f"one_peer_round_{round_number:02d}_{species}.png"
        )
        if not image_path.exists():
            raise FileNotFoundError(f"one-peer stimulus image not found: {image_path}")

        initial_prompt = prompt_builder.build_initial_prompt(
            round_number=round_number,
            species=species,
            image_path=image_path,
        )
        usage: Dict[str, Any] = {}
        if participant is not None:
            participant.start_conversation()
            initial, initial_response_text = self._call_estimate(
                participant,
                initial_prompt,
                usage,
            )
            participant.clear_conversation()
        else:
            initial = self._mock_initial(rng, true_count)
            initial_response_text = f"ESTIMATE={initial}"

        peer, selection = self._sample_one_peer(
            one_peer_lookup,
            round_index=round_number - 1,
            initial_estimate=initial,
            rng=rng,
        )
        peers = [peer]
        social_prompt = prompt_builder.build_social_prompt(
            round_number=round_number,
            species=species,
            initial_estimate=initial,
            peer_estimates=peers,
        )
        if participant is not None:
            participant.start_conversation()
            final, final_response_text = self._call_estimate(
                participant,
                social_prompt,
                usage,
            )
            participant.clear_conversation()
        else:
            final = self._mock_social_revision(
                rng,
                condition="one_peer_control",
                initial=initial,
                peers=peers,
            )
            final_response_text = f"ESTIMATE={final}"

        return {
            "participant_id": participant_id,
            "trial_number": global_trial_number,
            "response": final,
            "response_text": final_response_text,
            "raw_response_text": final_response_text,
            "usage": usage,
            "correct_answer": true_count,
            "is_correct": final == true_count,
            "trial_info": {
                "study_type": STUDY_TYPE,
                "sub_study_id": "one_peer_control",
                "block": "one_peer_control",
                "one_peer_round": round_number,
                "species": species,
                "stimulus_file": image_path.name,
                "stimulus_presented": True,
                "stimulus_seconds_in_human_interface": 6,
                "visual_exposure_emulation": (
                    "single multimodal first-estimate call; image absent from revision"
                ),
                "condition": "one_peer_control",
                "condition_description": (
                    "one previous participant estimate selected by the published rule"
                ),
                "treatment_code": 1,
                "initial_estimate": initial,
                "initial_response_text": initial_response_text,
                "peer_estimates": peers,
                "peer_mean": float(peer),
                "peer_variance": 0.0,
                "peer_skewness": 0.0,
                "one_peer_selection": selection,
                "final_estimate": final,
                "social_information_use": social_information_use(
                    initial,
                    final,
                    peers,
                ),
                "strategy": classify_adjustment(initial, final, peers),
                "initial_absolute_error": abs(initial - true_count),
                "final_absolute_error": abs(final - true_count),
                "points": self._score_estimate(final, true_count),
                "answer_revealed_before_response": False,
                "peer_lookup_verified": True,
                "agent_visible_prompts": {
                    "initial": self._prompt_text(initial_prompt),
                    "social_revision": social_prompt,
                },
            },
        }

    def _run_control_round(
        self,
        *,
        participant: Optional[Any],
        prompt_builder: DisparateSocialInformationPromptBuilder,
        rng: random.Random,
        participant_id: int,
        global_trial_number: int,
        round_index: int,
        control_lookup: Dict[str, Any],
        main_rounds: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        round_data = control_lookup["rounds"][round_index]
        valid_anchors = [
            int(value)
            for value in control_lookup["anchor_pools"][round_index]
            if ESTIMATE_MIN <= int(value) <= ESTIMATE_MAX
        ]
        if not valid_anchors:
            raise ValueError(f"control round {round_index + 1} has no valid anchors")
        anchor = rng.choice(valid_anchors)
        three_peers = self._peer_estimates(
            control_lookup,
            round_index=round_index,
            initial_estimate=anchor,
        )
        peers = [anchor, *three_peers]
        prompt = prompt_builder.build_control_prompt(
            round_number=round_index + 1,
            peer_estimates=peers,
        )
        usage: Dict[str, Any] = {}
        if participant is not None:
            participant.start_conversation()
            final, response_text = self._call_estimate(participant, prompt, usage)
            participant.clear_conversation()
        else:
            final = self._mock_control_estimate(rng, peers)
            response_text = f"ESTIMATE={final}"

        emulated_main_round = int(round_data["emulated_main_round"])
        true_count = int(main_rounds[emulated_main_round - 1]["true_count"])
        peer_mean = mean(peers)
        return {
            "participant_id": participant_id,
            "trial_number": global_trial_number,
            "response": final,
            "response_text": response_text,
            "raw_response_text": response_text,
            "usage": usage,
            "correct_answer": true_count,
            "is_correct": final == true_count,
            "trial_info": {
                "study_type": STUDY_TYPE,
                "sub_study_id": "four_peer_control",
                "block": "four_peer_control",
                "control_round": round_index + 1,
                "emulated_main_round": emulated_main_round,
                "condition": str(round_data["condition"]),
                "stimulus_presented": False,
                "personal_first_estimate_presented": False,
                "peer_estimates": peers,
                "peer_mean": peer_mean,
                "peer_variance": pvariance(peers),
                "peer_skewness": distribution_skew(peers),
                "final_estimate": final,
                "absolute_deviation_from_peer_mean": abs(final - peer_mean),
                "final_absolute_error": abs(final - true_count),
                "points": self._score_estimate(final, true_count),
                "answer_revealed_before_response": False,
                "peer_lookup_verified": True,
                "invalid_zero_anchor_rejected": True,
                "agent_visible_prompts": {
                    "control": prompt,
                },
            },
        }

    def _run_one_participant(
        self,
        trial: Dict[str, Any],
        *,
        participant: Optional[Any],
        profile: Dict[str, Any],
        prompt_builder: DisparateSocialInformationPromptBuilder,
        base_seed: int,
        lookup: Dict[str, Any],
    ) -> Dict[str, Any]:
        participant_id = int(trial["participant_index"])
        rng = random.Random(base_seed + 100_003 * participant_id)
        main_rounds = lookup["main_task"]["rounds"]
        order_code = participant_id % 6 + 1
        block_order = list(lookup["block_orders"]["orders"][str(order_code)])

        responses: List[Dict[str, Any]] = []
        comprehension_checks: List[Dict[str, Any]] = []
        terminated_early = False
        termination_reason: Optional[str] = None
        for block in block_order:
            check = self._run_comprehension_check(
                participant=participant,
                prompt_builder=prompt_builder,
                block_name=block,
                instructions=str(trial["block_instructions"][block]),
                quiz=trial["comprehension_checks"][block],
            )
            comprehension_checks.append(check)
            if not check["passed"]:
                terminated_early = True
                termination_reason = f"failed_comprehension_check:{block}"
                break

            if block == "main_task":
                for round_data in main_rounds:
                    responses.append(
                        self._run_main_round(
                            participant=participant,
                            prompt_builder=prompt_builder,
                            rng=rng,
                            participant_id=participant_id,
                            global_trial_number=len(responses) + 1,
                            round_data=round_data,
                            main_lookup=lookup["main_task"],
                        )
                    )
            elif block == "one_peer_control":
                for round_data in lookup["one_peer_control"]["rounds"]:
                    responses.append(
                        self._run_one_peer_round(
                            participant=participant,
                            prompt_builder=prompt_builder,
                            rng=rng,
                            participant_id=participant_id,
                            global_trial_number=len(responses) + 1,
                            round_data=round_data,
                            one_peer_lookup=lookup["one_peer_control"],
                        )
                    )
            elif block == "four_peer_control":
                for round_index in range(FOUR_PEER_ROUNDS):
                    responses.append(
                        self._run_control_round(
                            participant=participant,
                            prompt_builder=prompt_builder,
                            rng=rng,
                            participant_id=participant_id,
                            global_trial_number=len(responses) + 1,
                            round_index=round_index,
                            control_lookup=lookup["four_peer_control"],
                            main_rounds=main_rounds,
                        )
                    )
            else:
                raise ValueError(f"unknown study_018 block: {block}")

        if not terminated_early and len(responses) != CORE_RESPONSES_PER_PARTICIPANT:
            raise RuntimeError(
                "study_018 completed without the expected 40 behavioral responses"
            )
        if participant is not None:
            participant.clear_conversation()

        by_block = {
            block: [
                response
                for response in responses
                if response["trial_info"]["block"] == block
            ]
            for block in block_order
        }
        payment: Optional[Dict[str, Any]]
        if terminated_early:
            payment = None
        else:
            paid_responses = {
                block: rng.choice(block_responses)
                for block, block_responses in by_block.items()
            }
            paid_points = sum(
                int(response["trial_info"]["points"])
                for response in paid_responses.values()
            )
            payment = {
                "participation_fee_usd": 4.50,
                "paid_rounds": {
                    "one_peer_control": int(
                        paid_responses["one_peer_control"]["trial_info"][
                            "one_peer_round"
                        ]
                    ),
                    "main_task": int(
                        paid_responses["main_task"]["trial_info"]["main_round"]
                    ),
                    "four_peer_control": int(
                        paid_responses["four_peer_control"]["trial_info"][
                            "control_round"
                        ]
                    ),
                },
                "bonus_points": paid_points,
                "bonus_usd": paid_points / 100.0,
                "note": "One round from each of the three original task blocks.",
            }
        return {
            "participant_id": participant_id,
            "profile": {
                **profile,
                "block_order_code": order_code,
                "block_order": block_order,
            },
            "sub_study_id": "complete_three_block_session",
            "comprehension_checks": comprehension_checks,
            "terminated_early": terminated_early,
            "termination_reason": termination_reason,
            "payment": payment,
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
        if not isinstance(builder, DisparateSocialInformationPromptBuilder):
            raise TypeError(
                "study_018 requires DisparateSocialInformationPromptBuilder"
            )
        lookup = self.load_material(self.LOOKUP_ID)
        if len(lookup["main_task"]["rounds"]) != MAIN_ROUNDS:
            raise ValueError("study_018 peer lookup must define exactly 30 main rounds")
        if len(lookup["one_peer_control"]["rounds"]) != ONE_PEER_ROUNDS:
            raise ValueError("study_018 lookup must define exactly five one-peer rounds")
        if len(lookup["four_peer_control"]["rounds"]) != FOUR_PEER_ROUNDS:
            raise ValueError("study_018 lookup must define exactly five four-peer rounds")
        published_orders = {
            tuple(order)
            for order in lookup["block_orders"]["orders"].values()
        }
        if len(published_orders) != 6 or any(
            set(order)
            != {"one_peer_control", "main_task", "four_peer_control"}
            for order in published_orders
        ):
            raise ValueError("study_018 lookup must define all six three-block orders")

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
                        lookup=lookup,
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
                        lookup=lookup,
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
            and ESTIMATE_MIN <= int(response["response"]) <= ESTIMATE_MAX
        ]
        main = [
            response
            for response in valid
            if response.get("trial_info", {}).get("block") == "main_task"
        ]
        one_peer = [
            response
            for response in valid
            if response.get("trial_info", {}).get("block") == "one_peer_control"
        ]
        control = [
            response
            for response in valid
            if response.get("trial_info", {}).get("block") == "four_peer_control"
        ]

        condition_adjustments: Dict[str, List[float]] = {
            condition: [] for condition in REPORTED_MEAN_ADJUSTMENT
        }
        condition_strategies: Dict[str, Dict[str, int]] = {
            condition: {
                "keep": 0,
                "adopt_nearest": 0,
                "compromise": 0,
                "other": 0,
            }
            for condition in REPORTED_MEAN_ADJUSTMENT
        }
        condition_totals = {condition: 0 for condition in REPORTED_MEAN_ADJUSTMENT}
        for response in main:
            info = response["trial_info"]
            condition = info["condition"]
            if condition not in condition_adjustments:
                continue
            adjustment = info.get("social_information_use")
            if isinstance(adjustment, (int, float)) and math.isfinite(float(adjustment)):
                condition_adjustments[condition].append(float(adjustment))
            strategy = info.get("strategy")
            if strategy in condition_strategies[condition]:
                condition_strategies[condition][strategy] += 1
            condition_totals[condition] += 1

        strategy_frequencies = {
            condition: {
                strategy: (
                    count / condition_totals[condition]
                    if condition_totals[condition]
                    else None
                )
                for strategy, count in frequencies.items()
            }
            for condition, frequencies in condition_strategies.items()
        }
        mean_adjustment = {
            condition: (
                mean(values)
                if values
                else None
            )
            for condition, values in condition_adjustments.items()
        }

        return {
            "descriptive_statistics": {
                "participants": len(participants),
                "responses": len(valid),
                "complete_participants": sum(
                    not participant.get("terminated_early", False)
                    for participant in participants
                ),
                "terminated_participants": sum(
                    participant.get("terminated_early", False)
                    for participant in participants
                ),
                "main_task_responses": len(main),
                "one_peer_control_responses": len(one_peer),
                "four_peer_control_responses": len(control),
                "comprehension_checks_passed": sum(
                    check.get("passed") is True
                    for participant in participants
                    for check in participant.get("comprehension_checks", [])
                ),
                "parse_failures": len(responses) - len(valid),
                "mean_initial_absolute_error": (
                    mean(
                        float(response["trial_info"]["initial_absolute_error"])
                        for response in main
                    )
                    if main
                    else None
                ),
                "mean_final_absolute_error": (
                    mean(
                        float(response["trial_info"]["final_absolute_error"])
                        for response in main
                    )
                    if main
                    else None
                ),
                "mean_social_information_use_by_condition": mean_adjustment,
                "strategy_frequencies_by_condition": strategy_frequencies,
                "mean_control_deviation_from_peer_mean": (
                    mean(
                        float(
                            response["trial_info"][
                                "absolute_deviation_from_peer_mean"
                            ]
                        )
                        for response in control
                    )
                    if control
                    else None
                ),
                "mean_main_deviation_from_peer_mean": (
                    mean(
                        abs(
                            float(response["response"])
                            - float(response["trial_info"]["peer_mean"])
                        )
                        for response in main
                    )
                    if main
                    else None
                ),
            },
            "inferential_statistics": {
                "reported_mean_social_information_use": REPORTED_MEAN_ADJUSTMENT,
                "reported_repeatability": {
                    "R": 0.790,
                    "credible_interval_95": [0.697, 0.839],
                },
            },
        }
