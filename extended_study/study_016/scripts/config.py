"""Runtime adapter for Anderson and Holt's information-cascade experiment."""

from __future__ import annotations

import math
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

# The article reports three baseline symmetric sessions, two fully described
# public-draw sessions, one under-described public-draw session, and six
# asymmetric sessions. The under-described session is intentionally omitted.
IMPLEMENTED_SESSION_SCHEDULE: Tuple[Tuple[str, int], ...] = (
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
SYMMETRIC_URNS: Dict[str, Dict[str, int]] = {
    "A": {"light": 2, "dark": 1},
    "B": {"light": 1, "dark": 2},
}


def infer_public_evidence(prior_decisions: Sequence[str]) -> Tuple[int, Optional[str]]:
    """Infer the symmetric zero-error signal balance and active cascade."""

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
    """Return the paper's symmetric zero-error benchmark choice.

    This public helper is retained for backwards-compatible analysis. The
    runtime uses the probability-based functions below so the same mechanism
    also supports public draws and asymmetric urns.
    """

    balance, cascade_direction = infer_public_evidence(prior_decisions)
    if cascade_direction is not None:
        return cascade_direction

    private_value = 1 if private_signal == "L" else -1
    posterior_balance = balance + private_value
    if posterior_balance > 0:
        return "A"
    if posterior_balance < 0:
        return "B"
    return "A" if private_signal == "L" else "B"


def _signal_probability(
    urns: Dict[str, Dict[str, int]],
    urn: str,
    signal: str,
) -> float:
    color = "light" if signal == "L" else "dark"
    composition = urns[urn]
    return float(composition[color]) / float(composition["light"] + composition["dark"])


def _posterior_after_signal(
    prior_a: float,
    signal: str,
    urns: Dict[str, Dict[str, int]],
) -> float:
    likelihood_a = _signal_probability(urns, "A", signal)
    likelihood_b = _signal_probability(urns, "B", signal)
    numerator = prior_a * likelihood_a
    denominator = numerator + (1.0 - prior_a) * likelihood_b
    return numerator / denominator if denominator else prior_a


def _choice_from_private_posterior(
    prior_a: float,
    private_signal: str,
    urns: Dict[str, Dict[str, int]],
) -> Tuple[str, float]:
    posterior_a = _posterior_after_signal(prior_a, private_signal, urns)
    if math.isclose(posterior_a, 0.5, rel_tol=0.0, abs_tol=1e-12):
        # Floating-point updates can produce 0.4999999999999999 for an exact
        # Bayesian tie. The paper's tie rule follows the private draw.
        return ("A" if private_signal == "L" else "B"), posterior_a
    if posterior_a > 0.5:
        return "A", posterior_a
    if posterior_a < 0.5:
        return "B", posterior_a
    raise AssertionError("unreachable posterior comparison")


def _cascade_direction(
    public_prior_a: float,
    urns: Dict[str, Dict[str, int]],
) -> Optional[str]:
    light_choice, _ = _choice_from_private_posterior(public_prior_a, "L", urns)
    dark_choice, _ = _choice_from_private_posterior(public_prior_a, "D", urns)
    return light_choice if light_choice == dark_choice else None


def _posterior_after_public_decision(
    prior_a: float,
    observed_choice: str,
    urns: Dict[str, Dict[str, int]],
) -> float:
    """Update public belief by integrating over a Bayesian actor's private draw."""

    likelihoods: Dict[str, float] = {}
    for urn in ("A", "B"):
        probability = 0.0
        for signal in ("L", "D"):
            predicted_choice, _ = _choice_from_private_posterior(prior_a, signal, urns)
            if predicted_choice == observed_choice:
                probability += _signal_probability(urns, urn, signal)
        likelihoods[urn] = probability

    numerator = prior_a * likelihoods["A"]
    denominator = numerator + (1.0 - prior_a) * likelihoods["B"]
    if denominator:
        return numerator / denominator

    # Under a deterministic cascade, a contrary action has zero modeled
    # probability. The paper's zero-error analysis treats such an obvious
    # deviation as revealing the contrary private draw.
    inferred_signal = "L" if observed_choice == "A" else "D"
    return _posterior_after_signal(prior_a, inferred_signal, urns)


class InformationCascadePromptBuilder(PromptBuilder):
    """Build a decision prompt from only information visible at that position."""

    def build_system_prompt(self, participant_profile: Dict[str, Any] = None) -> str:
        del participant_profile
        return (
            "Act as one decision maker in an economics experiment. Use only the "
            "experiment rules, your current private draw, public draws explicitly "
            "shown to you, and predictions announced before your turn. Do not infer "
            "access to the selected urn or to another participant's private draw."
        )

    def build_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        return self.build_decision_prompt(
            period_number=int(trial_data["period_number"]),
            decision_position=int(trial_data["decision_position"]),
            private_signal=str(trial_data["private_signal"]),
            prior_decisions=list(trial_data.get("prior_decisions", [])),
            cumulative_earnings=float(trial_data.get("cumulative_earnings", 0.0)),
            previous_period=trial_data.get("previous_period"),
            treatment_name=str(trial_data.get("treatment_name", "symmetric_baseline")),
            urns=trial_data.get("urns"),
            public_draws=list(trial_data.get("public_draws", [])),
            public_draw_rule=trial_data.get("public_draw_rule"),
            initial_instructions=trial_data.get("initial_instructions"),
            practice_periods=trial_data.get("practice_periods"),
        )

    @staticmethod
    def _format_urn(urn: str, composition: Dict[str, int]) -> str:
        return (
            f"Urn {urn} contains {composition['light']} light "
            f"{'ball' if composition['light'] == 1 else 'balls'} and "
            f"{composition['dark']} dark "
            f"{'ball' if composition['dark'] == 1 else 'balls'}."
        )

    @staticmethod
    def _format_practice(practice_periods: Sequence[Dict[str, Any]]) -> str:
        lines = [
            "Completed public practice demonstrations (unpaid; no predictions were made):"
        ]
        for period in practice_periods:
            draws = ", ".join(draw["label"] for draw in period["draws"])
            lines.append(
                f"- Practice {period['practice_number']}: die={period['die_roll']}, "
                f"urn {period['true_urn']} was visibly selected; public draws: {draws}."
            )
        return "\n".join(lines)

    def build_decision_prompt(
        self,
        *,
        period_number: int,
        decision_position: int,
        private_signal: str,
        prior_decisions: Sequence[str],
        cumulative_earnings: float,
        previous_period: Optional[Dict[str, Any]],
        treatment_name: str = "symmetric_baseline",
        urns: Optional[Dict[str, Dict[str, int]]] = None,
        public_draws: Sequence[str] = (),
        public_draw_rule: Optional[Dict[str, Any]] = None,
        initial_instructions: Optional[str] = None,
        practice_periods: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> str:
        urns = urns or SYMMETRIC_URNS
        history = ", ".join(prior_decisions) if prior_decisions else "None"
        sections: List[str] = []

        if initial_instructions:
            sections.append(f"Instructions read before the paid periods:\n{initial_instructions}")
        if practice_periods:
            sections.append(self._format_practice(practice_periods))

        standing_rules = [
            "Standing rules:",
            "- Urn A and urn B are equally likely to be selected.",
            f"- {self._format_urn('A', urns['A'])}",
            f"- {self._format_urn('B', urns['B'])}",
            "- Every draw is made with replacement.",
            "- Decision order is newly randomized in every paid period.",
            "- Earlier predictions are public; other participants' private draws are not.",
            "- A correct prediction earns $2; an incorrect prediction earns $0.",
        ]
        if public_draw_rule and int(public_draw_rule.get("count", 0)) > 0:
            standing_rules.append(
                "- In this treatment, two additional public draws are revealed after "
                "the fourth prediction and before positions 5 and 6 decide."
            )
        sections.append("\n".join(standing_rules))

        if previous_period:
            previous_draws = previous_period.get("public_draws", [])
            public_line = (
                f"- Additional public draws: {', '.join(previous_draws)}\n"
                if previous_draws
                else ""
            )
            sections.append(
                "Previous paid-period feedback:\n"
                f"- Public prediction sequence: {', '.join(previous_period['decisions'])}\n"
                f"{public_line}"
                f"- Urn used: {previous_period['true_urn']}\n"
                f"- Your prediction: {previous_period['own_choice']}\n"
                f"- Your earnings: ${previous_period['earnings']}"
            )

        current_public_draws = ", ".join(
            "Light" if signal == "L" else "Dark" for signal in public_draws
        )
        public_draw_line = (
            f"Additional public draws now visible: {current_public_draws}\n"
            if public_draws
            else ""
        )
        sections.append(
            "Current paid decision:\n"
            f"Treatment: {treatment_name}\n"
            f"Period: {period_number} of {PERIODS_PER_SESSION}\n"
            f"Your position: {decision_position} of {PARTICIPANTS_PER_SESSION}\n"
            f"Your private draw: {'Light' if private_signal == 'L' else 'Dark'}\n"
            f"{public_draw_line}"
            f"Earlier public predictions this period: {history}\n"
            f"Your cumulative earnings before this prediction: ${cumulative_earnings:g}\n\n"
            "Which urn do you predict was selected?\n"
            "Output exactly one line: CHOICE=A or CHOICE=B"
        )
        return "\n\n".join(sections)


class StudyStudy016Config(BaseStudyConfig):
    """Six-person stateful runtime for the evidence-complete treatments."""

    prompt_builder_class = InformationCascadePromptBuilder
    REQUIRES_GROUP_TRIALS = True
    SUPPORTED_SUB_STUDIES = tuple(
        dict.fromkeys(treatment for treatment, _ in IMPLEMENTED_SESSION_SCHEDULE)
    )

    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        schedule = tuple(
            entry
            for entry in IMPLEMENTED_SESSION_SCHEDULE
            if not self.selected_sub_studies
            or entry[0] in self.selected_sub_studies
        )
        if not schedule:
            raise ValueError("study_016 selected scope contains no runnable sessions")
        available_participants = len(schedule) * PARTICIPANTS_PER_SESSION
        participant_count = (
            available_participants if n_trials is None else int(n_trials)
        )
        if participant_count < PARTICIPANTS_PER_SESSION:
            raise ValueError(
                f"study_016 needs at least {PARTICIPANTS_PER_SESSION} participants for one session"
            )
        if participant_count % PARTICIPANTS_PER_SESSION:
            raise ValueError(
                "study_016 participant count must be a multiple of 6 so no partial session is run"
            )
        if participant_count > available_participants:
            raise ValueError(
                f"study_016 selected scope provides {available_participants} "
                "evidence-complete decision-maker slots. Requesting more would "
                "invent sessions outside that scope."
            )

        common = self.load_material("experiment_instructions")
        trials: List[Dict[str, Any]] = []
        session_count = participant_count // PARTICIPANTS_PER_SESSION
        for session_index, (treatment, published_session_number) in enumerate(
            schedule[:session_count]
        ):
            material = self.load_material(treatment)
            combined_instructions = (
                f"{common['instructions']}\n\n"
                f"Treatment-specific rules:\n{material['instructions']}"
            )
            for seat_index in range(PARTICIPANTS_PER_SESSION):
                participant_index = session_index * PARTICIPANTS_PER_SESSION + seat_index
                trials.append(
                    {
                        "trial_number": participant_index + 1,
                        "participant_index": participant_index,
                        "session_index": session_index,
                        "published_session_number": published_session_number,
                        "seat_index": seat_index,
                        "sub_study_id": treatment,
                        "study_type": "sequential_information_cascade",
                        "instructions": combined_instructions,
                        "items": material["items"],
                        "treatment_material": material,
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
    def _draw_signal(
        rng: random.Random,
        true_urn: str,
        urns: Dict[str, Dict[str, int]],
    ) -> str:
        probability_light = _signal_probability(urns, true_urn, "L")
        return "L" if rng.random() < probability_light else "D"

    def _generate_practice_periods(
        self,
        rng: random.Random,
        global_ids: Sequence[int],
        urns: Dict[str, Dict[str, int]],
    ) -> List[Dict[str, Any]]:
        """Run public, unpaid demonstrations until both urns have appeared."""

        periods: List[Dict[str, Any]] = []
        observed_urns = set()
        while observed_urns != {"A", "B"}:
            die_roll = rng.randint(1, 6)
            true_urn = "A" if die_roll <= 3 else "B"
            observed_urns.add(true_urn)
            draw_order = list(global_ids)
            rng.shuffle(draw_order)
            draws = []
            for participant_id in draw_order:
                signal = self._draw_signal(rng, true_urn, urns)
                draws.append(
                    {
                        "participant_id": participant_id,
                        "signal": signal,
                        "label": "Light" if signal == "L" else "Dark",
                    }
                )
            periods.append(
                {
                    "practice_number": len(periods) + 1,
                    "die_roll": die_roll,
                    "true_urn": true_urn,
                    "draw_order": draw_order,
                    "draws": draws,
                    "urn_selection_visible": True,
                    "draws_public": True,
                    "decisions_required": False,
                    "paid": False,
                }
            )
            if len(periods) > 1000:
                raise RuntimeError("practice-period generator failed to display both urns")
        return periods

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
        session_trial = session_trials[0]
        session_index = int(session_trial["session_index"])
        treatment = str(session_trial["sub_study_id"])
        published_session_number = int(session_trial["published_session_number"])
        material = dict(session_trial["treatment_material"])
        urns = material["urns"]
        public_draw_rule = material["public_draws"]
        rng = random.Random(base_seed + 100_003 * session_index)
        use_real_llm = bool(participant_pool_kwargs.get("use_real_llm", False))
        global_ids = [int(trial["participant_index"]) for trial in session_trials]
        practice_periods = self._generate_practice_periods(rng, global_ids, urns)

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
                participant.profile.update(
                    {
                        "participant_id": global_id,
                        "session_index": session_index,
                        "published_session_number": published_session_number,
                        "sub_study_id": treatment,
                        "practice_periods": practice_periods,
                    }
                )
                participant.start_conversation()
                participants[global_id] = participant
                profiles[global_id] = dict(participant.profile)
        else:
            for global_id in global_ids:
                profile = all_profiles[global_id] if all_profiles is not None else {}
                profiles[global_id] = {
                    **profile,
                    "participant_id": global_id,
                    "session_index": session_index,
                    "published_session_number": published_session_number,
                    "sub_study_id": treatment,
                    "practice_periods": practice_periods,
                }

        responses: Dict[int, List[Dict[str, Any]]] = {
            participant_id: [] for participant_id in global_ids
        }
        cumulative_earnings = {participant_id: 0 for participant_id in global_ids}
        previous_period: Dict[int, Optional[Dict[str, Any]]] = {
            participant_id: None for participant_id in global_ids
        }
        initial_context_pending = set(global_ids)
        period_summaries: List[Dict[str, Any]] = []

        for period_number in range(1, PERIODS_PER_SESSION + 1):
            die_roll = rng.randint(1, 6)
            true_urn = "A" if die_roll <= 3 else "B"
            signals = {
                participant_id: self._draw_signal(rng, true_urn, urns)
                for participant_id in global_ids
            }
            decision_order = list(global_ids)
            rng.shuffle(decision_order)
            prior_decisions: List[str] = []
            public_draws: List[str] = []
            public_prior_a = 0.5
            period_records: List[Dict[str, Any]] = []

            for decision_position, participant_id in enumerate(decision_order, start=1):
                reveal_after = public_draw_rule.get("reveal_after_decision_position")
                if (
                    reveal_after is not None
                    and decision_position == int(reveal_after) + 1
                    and not public_draws
                ):
                    public_draws = [
                        self._draw_signal(rng, true_urn, urns)
                        for _ in range(int(public_draw_rule.get("count", 0)))
                    ]
                    for public_signal in public_draws:
                        public_prior_a = _posterior_after_signal(
                            public_prior_a, public_signal, urns
                        )

                private_signal = signals[participant_id]
                public_prior_before_choice = public_prior_a
                expected_choice, private_posterior_a = _choice_from_private_posterior(
                    public_prior_before_choice, private_signal, urns
                )
                cascade_direction = _cascade_direction(public_prior_before_choice, urns)
                private_signal_choice = "A" if private_signal == "L" else "B"
                cascade_opportunity = (
                    cascade_direction is not None
                    and expected_choice == cascade_direction
                    and private_signal_choice != cascade_direction
                )
                is_initial_context = participant_id in initial_context_pending
                decision_context = {
                    "period_number": period_number,
                    "decision_position": decision_position,
                    "private_signal": private_signal,
                    "prior_decisions": list(prior_decisions),
                    "cumulative_earnings": cumulative_earnings[participant_id],
                    "previous_period": previous_period[participant_id],
                    "treatment_name": treatment,
                    "urns": urns,
                    "public_draws": list(public_draws),
                    "public_draw_rule": public_draw_rule,
                    "initial_instructions": (
                        session_trial["instructions"] if is_initial_context else None
                    ),
                    "practice_periods": practice_periods if is_initial_context else None,
                }
                prompt = prompt_builder.build_trial_prompt(decision_context)
                initial_context_pending.discard(participant_id)

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
                public_prior_a = _posterior_after_public_decision(
                    public_prior_before_choice, choice, urns
                )
                prior_decisions.append(choice)
                record = {
                    "participant_id": participant_id,
                    "trial_number": period_number,
                    "response": choice,
                    "response_text": response_text,
                    "raw_response_text": response_text,
                    "prompt": prompt,
                    "usage": usage,
                    "trial_info": {
                        "study_type": "sequential_information_cascade",
                        "sub_study_id": treatment,
                        "session_index": session_index,
                        "published_session_number": published_session_number,
                        "period_number": period_number,
                        "decision_position": decision_position,
                        "private_signal": private_signal,
                        "prior_decisions": prior_snapshot,
                        "public_draws_visible": list(public_draws),
                        "public_prior_a_before_private_signal": public_prior_before_choice,
                        "private_posterior_a": private_posterior_a,
                        "public_posterior_a_after_choice": public_prior_a,
                        "cascade_direction_before_choice": cascade_direction,
                        "cascade_opportunity": cascade_opportunity,
                        "bayesian_choice": expected_choice,
                        "urns": urns,
                        "initial_instructions_shown": is_initial_context,
                    },
                }
                responses[participant_id].append(record)
                period_records.append(record)

            public_sequence = [record["response"] for record in period_records]
            public_draw_labels = [
                "Light" if signal == "L" else "Dark" for signal in public_draws
            ]
            for record in period_records:
                participant_id = int(record["participant_id"])
                is_correct = record["response"] == true_urn
                earnings = CORRECT_PAYOFF_DOLLARS if is_correct else 0
                cumulative_earnings[participant_id] += earnings
                record["correct_answer"] = true_urn
                record["is_correct"] = is_correct
                record["earnings"] = earnings
                record["trial_info"]["true_urn_revealed_after_period"] = true_urn
                record["trial_info"]["die_roll_hidden_during_period"] = die_roll
                previous_period[participant_id] = {
                    "decisions": public_sequence,
                    "public_draws": public_draw_labels,
                    "true_urn": true_urn,
                    "own_choice": record["response"],
                    "earnings": earnings,
                }

            period_summaries.append(
                {
                    "session_index": session_index,
                    "published_session_number": published_session_number,
                    "sub_study_id": treatment,
                    "period_number": period_number,
                    "die_roll": die_roll,
                    "true_urn": true_urn,
                    "decision_order": decision_order,
                    "private_signals": signals,
                    "public_draws_after_position_4": public_draws,
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
                "published_session_number": published_session_number,
                "sub_study_id": treatment,
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
        if any(
            len(session_trials) != PARTICIPANTS_PER_SESSION
            for session_trials in grouped.values()
        ):
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

    @staticmethod
    def _response_summary(responses: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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

        def ratio(numerator: int, denominator: int) -> Optional[float]:
            return numerator / denominator if denominator else None

        return {
            "decisions": len(valid),
            "parse_failures": len(responses) - len(valid),
            "decision_accuracy": ratio(len(correct), len(valid)),
            "bayesian_agreement_rate": ratio(len(bayesian), len(valid)),
            "cascade_opportunities": len(opportunities),
            "cascade_followed": len(followed),
            "cascade_following_rate": ratio(len(followed), len(opportunities)),
        }

    def aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        participants = raw_results.get("individual_data", [])
        responses = [
            response
            for participant in participants
            for response in participant.get("responses", [])
        ]
        by_treatment: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for response in responses:
            treatment = str(response.get("trial_info", {}).get("sub_study_id", "unknown"))
            by_treatment[treatment].append(response)

        overall = self._response_summary(responses)
        overall.update(
            {
                "participants": len(participants),
                "sessions": len({participant.get("session_index") for participant in participants}),
                "periods": len(raw_results.get("session_periods", [])),
                "mean_earnings": (
                    sum(float(row.get("total_earnings", 0)) for row in participants)
                    / len(participants)
                    if participants
                    else None
                ),
                "practice_periods_by_session": {
                    str(participant["session_index"]): len(
                        participant.get("profile", {}).get("practice_periods", [])
                    )
                    for participant in participants[::PARTICIPANTS_PER_SESSION]
                },
                "by_treatment": {
                    treatment: self._response_summary(treatment_responses)
                    for treatment, treatment_responses in sorted(by_treatment.items())
                },
            }
        )

        return {
            "descriptive_statistics": overall,
            "inferential_statistics": {
                "reported_symmetric_cascade_rate": 41.0 / 56.0,
                "reported_asymmetric_cascade_rate": 46.0 / 66.0,
                "scope_note": (
                    "Behavioral rates are diagnostics only. The symmetric 41/56 result pools "
                    "six sessions, including one public-draw session whose procedure is not "
                    "specified and is therefore excluded from this runtime."
                ),
            },
        }
