"""Runtime adapter for Anderson and Holt's symmetric information-cascade experiment."""

from __future__ import annotations

import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder


PARTICIPANTS_PER_SESSION = 6
PERIODS_PER_SESSION = 15
CORRECT_PAYOFF_DOLLARS = 2


def infer_public_evidence(prior_decisions: Sequence[str]) -> Tuple[int, Optional[str]]:
    """Infer the zero-error public signal balance and active cascade direction.

    Before a cascade, an announced decision is treated as revealing a signal. Once
    the public balance reaches two, decisions that continue the cascade add no new
    information. A contrary decision is treated as revealing a contrary signal and
    can break the cascade, matching the interpretation used in the paper.
    """

    balance = 0
    cascade_direction: Optional[str] = None
    for decision in prior_decisions:
        if decision not in {"A", "B"}:
            continue
        value = 1 if decision == "A" else -1
        if cascade_direction is not None:
            if decision == cascade_direction:
                continue
            balance += value
            cascade_direction = "A" if balance >= 2 else "B" if balance <= -2 else None
            continue

        balance += value
        cascade_direction = "A" if balance >= 2 else "B" if balance <= -2 else None

    return balance, cascade_direction


def bayesian_choice(private_signal: str, prior_decisions: Sequence[str]) -> str:
    """Return the optimal choice in the paper's symmetric zero-error model."""

    balance, cascade_direction = infer_public_evidence(prior_decisions)
    if cascade_direction is not None:
        return cascade_direction

    private_value = 1 if private_signal == "L" else -1
    posterior_balance = balance + private_value
    if posterior_balance > 0:
        return "A"
    if posterior_balance < 0:
        return "B"
    # The paper assumes that a participant follows the private signal at a tie.
    return "A" if private_signal == "L" else "B"


class InformationCascadePromptBuilder(PromptBuilder):
    """Build one decision prompt without exposing the urn or other private draws."""

    def build_system_prompt(self, participant_profile: Dict[str, Any] = None) -> str:
        del participant_profile
        return (
            "Act as one participant in an economics experiment. Make each decision "
            "only from the rules, your private draw, and decisions publicly announced "
            "before your turn. Never assume access to the true urn or other private draws."
        )

    def build_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        return self.build_decision_prompt(
            period_number=int(trial_data["period_number"]),
            decision_position=int(trial_data["decision_position"]),
            private_signal=str(trial_data["private_signal"]),
            prior_decisions=list(trial_data.get("prior_decisions", [])),
            cumulative_earnings=float(trial_data.get("cumulative_earnings", 0.0)),
            previous_period=trial_data.get("previous_period"),
        )

    def build_decision_prompt(
        self,
        *,
        period_number: int,
        decision_position: int,
        private_signal: str,
        prior_decisions: Sequence[str],
        cumulative_earnings: float,
        previous_period: Optional[Dict[str, Any]],
    ) -> str:
        history = ", ".join(prior_decisions) if prior_decisions else "None"
        previous = ""
        if previous_period:
            previous = (
                "Previous period feedback:\n"
                f"- Public decision sequence: {', '.join(previous_period['decisions'])}\n"
                f"- Urn used: {previous_period['true_urn']}\n"
                f"- Your decision: {previous_period['own_choice']}\n"
                f"- Your earnings: ${previous_period['earnings']}\n\n"
            )

        return (
            "Decision experiment rules:\n"
            "- Urn A and urn B are equally likely to be selected.\n"
            "- Urn A contains two light balls and one dark ball.\n"
            "- Urn B contains one light ball and two dark balls.\n"
            "- Every participant receives one private draw, with replacement.\n"
            "- Decisions are made in a newly randomized order in every period.\n"
            "- You see earlier public decisions, but never anyone else's private draw.\n"
            "- A correct prediction earns $2; an incorrect prediction earns $0.\n\n"
            f"{previous}"
            f"Period: {period_number} of {PERIODS_PER_SESSION}\n"
            f"Your position in this period: {decision_position} of {PARTICIPANTS_PER_SESSION}\n"
            f"Your private draw: {'Light' if private_signal == 'L' else 'Dark'}\n"
            f"Earlier public decisions in this period: {history}\n"
            f"Your cumulative earnings before this decision: ${cumulative_earnings:g}\n\n"
            "Which urn do you predict was selected?\n"
            "Output exactly one line: CHOICE=A or CHOICE=B"
        )


class StudyStudy016Config(BaseStudyConfig):
    """Six-person, sequential, repeated-session adapter for the symmetric baseline."""

    prompt_builder_class = InformationCascadePromptBuilder
    REQUIRES_GROUP_TRIALS = True
    SUPPORTED_TREATMENT = "symmetric_baseline"

    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        participant_count = 18 if n_trials is None else int(n_trials)
        if participant_count < PARTICIPANTS_PER_SESSION:
            raise ValueError(
                f"study_016 needs at least {PARTICIPANTS_PER_SESSION} participants for one session"
            )
        if participant_count % PARTICIPANTS_PER_SESSION:
            raise ValueError(
                "study_016 participant count must be a multiple of 6 so no partial session is run"
            )

        material = self.load_material(self.SUPPORTED_TREATMENT)
        trials: List[Dict[str, Any]] = []
        for participant_index in range(participant_count):
            trials.append(
                {
                    "trial_number": participant_index + 1,
                    "participant_index": participant_index,
                    "session_index": participant_index // PARTICIPANTS_PER_SESSION,
                    "seat_index": participant_index % PARTICIPANTS_PER_SESSION,
                    "sub_study_id": self.SUPPORTED_TREATMENT,
                    "study_type": "sequential_information_cascade",
                    "instructions": material["instructions"],
                    "items": material["items"],
                }
            )
        return trials

    @staticmethod
    def parse_choice(response_text: str) -> Optional[str]:
        if not response_text:
            return None
        explicit = re.search(r"\bCHOICE\s*[:=]\s*([AB])\b", response_text, re.IGNORECASE)
        if explicit:
            return explicit.group(1).upper()
        compact = response_text.strip().upper().rstrip(".")
        if compact in {"A", "B", "URN A", "URN B"}:
            return compact[-1]
        return None

    @staticmethod
    def _draw_signal(rng: random.Random, true_urn: str) -> str:
        probability_light = 2.0 / 3.0 if true_urn == "A" else 1.0 / 3.0
        return "L" if rng.random() < probability_light else "D"

    @staticmethod
    def _merge_usage(total: Dict[str, Any], new_usage: Dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = int(total.get(key, 0) or 0) + int(new_usage.get(key, 0) or 0)
        total["cost"] = float(total.get("cost", 0.0) or 0.0) + float(
            new_usage.get("cost", 0.0) or 0.0
        )

    def _run_one_session(
        self,
        session_trials: List[Dict[str, Any]],
        participant_pool_kwargs: Dict[str, Any],
        prompt_builder: InformationCascadePromptBuilder,
        base_seed: int,
        all_profiles: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        session_index = int(session_trials[0]["session_index"])
        rng = random.Random(base_seed + 100_003 * session_index)
        use_real_llm = bool(participant_pool_kwargs.get("use_real_llm", False))
        global_ids = [int(trial["participant_index"]) for trial in session_trials]

        participants: Dict[int, Any] = {}
        profiles: Dict[int, Dict[str, Any]] = {}
        if use_real_llm:
            from src.agents.llm_participant_agent import ParticipantPool

            pool_kwargs = dict(participant_pool_kwargs)
            pool_kwargs.pop("study_specification", None)
            pool_kwargs["n_participants"] = PARTICIPANTS_PER_SESSION
            if all_profiles is not None:
                pool_kwargs["profiles"] = [all_profiles[participant_id] for participant_id in global_ids]
            else:
                pool_kwargs["profiles"] = None
            pool = ParticipantPool(study_specification=self.specification, **pool_kwargs)
            for global_id, participant in zip(global_ids, pool.participants):
                participant.participant_id = global_id
                participant.profile["participant_id"] = global_id
                participant.profile["session_index"] = session_index
                participant.start_conversation()
                participants[global_id] = participant
                profiles[global_id] = dict(participant.profile)
        else:
            for global_id in global_ids:
                profile = all_profiles[global_id] if all_profiles is not None else {}
                profiles[global_id] = {**profile, "participant_id": global_id, "session_index": session_index}

        responses: Dict[int, List[Dict[str, Any]]] = {participant_id: [] for participant_id in global_ids}
        cumulative_earnings = {participant_id: 0 for participant_id in global_ids}
        previous_period: Dict[int, Optional[Dict[str, Any]]] = {
            participant_id: None for participant_id in global_ids
        }
        period_summaries: List[Dict[str, Any]] = []

        for period_number in range(1, PERIODS_PER_SESSION + 1):
            true_urn = rng.choice(["A", "B"])
            signals = {
                participant_id: self._draw_signal(rng, true_urn) for participant_id in global_ids
            }
            decision_order = list(global_ids)
            rng.shuffle(decision_order)
            prior_decisions: List[str] = []
            period_records: List[Dict[str, Any]] = []

            for decision_position, participant_id in enumerate(decision_order, start=1):
                private_signal = signals[participant_id]
                public_balance, cascade_direction = infer_public_evidence(prior_decisions)
                expected_choice = bayesian_choice(private_signal, prior_decisions)
                cascade_opportunity = (
                    cascade_direction is not None
                    and expected_choice == cascade_direction
                    and ((private_signal == "L" and cascade_direction == "B")
                         or (private_signal == "D" and cascade_direction == "A"))
                )
                decision_context = {
                    "period_number": period_number,
                    "decision_position": decision_position,
                    "private_signal": private_signal,
                    "prior_decisions": list(prior_decisions),
                    "cumulative_earnings": cumulative_earnings[participant_id],
                    "previous_period": previous_period[participant_id],
                }
                prompt = prompt_builder.build_trial_prompt(decision_context)

                usage: Dict[str, Any] = {}
                if use_real_llm:
                    result = participants[participant_id].continue_conversation(prompt, max_tokens=64)
                    response_text = str(result.get("response_text", ""))
                    self._merge_usage(usage, result.get("usage", {}))
                    choice = self.parse_choice(response_text)
                    if choice is None:
                        repair = participants[participant_id].continue_conversation(
                            "Your previous response was invalid. Output exactly CHOICE=A or CHOICE=B.",
                            max_tokens=16,
                        )
                        self._merge_usage(usage, repair.get("usage", {}))
                        response_text = str(repair.get("response_text", ""))
                        choice = self.parse_choice(response_text)
                    if choice is None:
                        raise ValueError(
                            f"participant {participant_id} returned no valid urn choice in "
                            f"session {session_index}, period {period_number}"
                        )
                else:
                    choice = expected_choice
                    response_text = f"CHOICE={choice}"

                prior_snapshot = list(prior_decisions)
                prior_decisions.append(choice)
                record = {
                    "participant_id": participant_id,
                    "trial_number": period_number,
                    "response": choice,
                    "response_text": response_text,
                    "raw_response_text": response_text,
                    "usage": usage,
                    "trial_info": {
                        "study_type": "sequential_information_cascade",
                        "sub_study_id": self.SUPPORTED_TREATMENT,
                        "session_index": session_index,
                        "period_number": period_number,
                        "decision_position": decision_position,
                        "private_signal": private_signal,
                        "prior_decisions": prior_snapshot,
                        "public_evidence_balance": public_balance,
                        "cascade_direction_before_choice": cascade_direction,
                        "cascade_opportunity": cascade_opportunity,
                        "bayesian_choice": expected_choice,
                    },
                }
                responses[participant_id].append(record)
                period_records.append(record)

            public_sequence = [record["response"] for record in period_records]
            for record in period_records:
                participant_id = int(record["participant_id"])
                is_correct = record["response"] == true_urn
                earnings = CORRECT_PAYOFF_DOLLARS if is_correct else 0
                cumulative_earnings[participant_id] += earnings
                record["correct_answer"] = true_urn
                record["is_correct"] = is_correct
                record["earnings"] = earnings
                record["trial_info"]["true_urn_revealed_after_period"] = true_urn
                previous_period[participant_id] = {
                    "decisions": public_sequence,
                    "true_urn": true_urn,
                    "own_choice": record["response"],
                    "earnings": earnings,
                }

            period_summaries.append(
                {
                    "session_index": session_index,
                    "period_number": period_number,
                    "true_urn": true_urn,
                    "decision_order": decision_order,
                    "private_signals": signals,
                    "public_decisions": public_sequence,
                }
            )

        for participant in participants.values():
            participant.clear_conversation()

        individual_data = [
            {
                "participant_id": participant_id,
                "profile": profiles[participant_id],
                "session_index": session_index,
                "total_earnings": cumulative_earnings[participant_id],
                "responses": responses[participant_id],
            }
            for participant_id in global_ids
        ]
        return {"individual_data": individual_data, "periods": period_summaries}

    def run_group_experiment(
        self,
        trials: List[Dict[str, Any]],
        instructions: str,
        participant_pool_kwargs: Dict[str, Any],
        prompt_builder: Optional[Any] = None,
    ) -> Dict[str, Any]:
        del instructions
        if len(trials) < PARTICIPANTS_PER_SESSION or len(trials) % PARTICIPANTS_PER_SESSION:
            raise ValueError("study_016 requires complete six-participant sessions")
        builder = prompt_builder or self.prompt_builder
        if not isinstance(builder, InformationCascadePromptBuilder):
            raise TypeError("study_016 requires InformationCascadePromptBuilder")

        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for trial in trials:
            grouped[int(trial["session_index"])].append(trial)
        if any(len(session_trials) != PARTICIPANTS_PER_SESSION for session_trials in grouped.values()):
            raise ValueError("each study_016 session must contain exactly six participants")

        base_seed = int(participant_pool_kwargs.get("random_seed", 42) or 42)
        all_profiles = participant_pool_kwargs.get("profiles")
        if all_profiles is not None and len(all_profiles) < len(trials):
            raise ValueError("profiles list is shorter than the requested participant count")

        session_results: List[Dict[str, Any]] = []
        use_real_llm = bool(participant_pool_kwargs.get("use_real_llm", False))
        workers = int(participant_pool_kwargs.get("num_workers") or 1)
        session_items = sorted(grouped.items())

        if use_real_llm and workers > 1 and len(session_items) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(session_items))) as executor:
                futures = {
                    executor.submit(
                        self._run_one_session,
                        session_trials,
                        participant_pool_kwargs,
                        builder,
                        base_seed,
                        all_profiles,
                    ): session_index
                    for session_index, session_trials in session_items
                }
                for future in as_completed(futures):
                    session_results.append(future.result())
        else:
            for _, session_trials in session_items:
                session_results.append(
                    self._run_one_session(
                        session_trials,
                        participant_pool_kwargs,
                        builder,
                        base_seed,
                        all_profiles,
                    )
                )

        individual_data = [
            participant
            for result in session_results
            for participant in result["individual_data"]
        ]
        periods = [period for result in session_results for period in result["periods"]]
        individual_data.sort(key=lambda row: int(row["participant_id"]))
        periods.sort(key=lambda row: (int(row["session_index"]), int(row["period_number"])))
        return {"individual_data": individual_data, "session_periods": periods}

    def aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        responses = [
            response
            for participant in raw_results.get("individual_data", [])
            for response in participant.get("responses", [])
        ]
        valid = [response for response in responses if response.get("response") in {"A", "B"}]
        correct = [response for response in valid if response.get("is_correct") is True]
        bayesian = [
            response
            for response in valid
            if response.get("response") == response.get("trial_info", {}).get("bayesian_choice")
        ]
        opportunities = [
            response
            for response in valid
            if response.get("trial_info", {}).get("cascade_opportunity") is True
        ]
        followed = [
            response
            for response in opportunities
            if response.get("response")
            == response.get("trial_info", {}).get("cascade_direction_before_choice")
        ]
        private_matches = [
            response
            for response in valid
            if response.get("response")
            == ("A" if response.get("trial_info", {}).get("private_signal") == "L" else "B")
        ]
        cascade_rate = len(followed) / len(opportunities) if opportunities else None
        reported_rate = 41.0 / 56.0

        def ratio(numerator: int, denominator: int) -> Optional[float]:
            return numerator / denominator if denominator else None

        return {
            "descriptive_statistics": {
                "participants": len(raw_results.get("individual_data", [])),
                "sessions": len({p.get("session_index") for p in raw_results.get("individual_data", [])}),
                "periods": len(raw_results.get("session_periods", [])),
                "decisions": len(valid),
                "parse_failures": len(responses) - len(valid),
                "decision_accuracy": ratio(len(correct), len(valid)),
                "bayesian_agreement_rate": ratio(len(bayesian), len(valid)),
                "private_signal_agreement_rate": ratio(len(private_matches), len(valid)),
                "cascade_opportunities": len(opportunities),
                "cascade_followed": len(followed),
                "cascade_following_rate": cascade_rate,
                "mean_earnings": ratio(
                    sum(float(row.get("total_earnings", 0)) for row in raw_results.get("individual_data", [])),
                    len(raw_results.get("individual_data", [])),
                ),
            },
            "inferential_statistics": {
                "reported_symmetric_cascade_rate": reported_rate,
                "absolute_rate_difference": (
                    abs(cascade_rate - reported_rate) if cascade_rate is not None else None
                ),
                "scope_note": (
                    "The paper's 41/56 benchmark pools six symmetric sessions, including "
                    "public-draw variants. This adapter implements the three-session baseline "
                    "whose complete participant instructions are public."
                ),
            },
        }
