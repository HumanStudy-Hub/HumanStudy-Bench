"""Deterministic datasets for the A+B->C social-influence experiment.

A is sequential Bayesian information integration (study_016).
B is learning advisor reliability from outcome feedback (study_017, Task 3C).
C is the held-out medical-authority environment (study_019, Study 2).

The generators produce preference pairs with randomized response codes. Labels
are calculated from the task state; no language model is used as a labeler.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
SYSTEM_PROMPT = (
    "Act as a participant in a controlled decision experiment. Use only the "
    "information in the task, preserve the stated order of observations, and "
    "return exactly one of the response codes requested."
)
RESPONSE_CODES = ("X", "Y")

A_TEMPLATES: Mapping[str, Tuple[Mapping[str, str], ...]] = {
    "train": (
        {
            "family": "urn",
            "source_a": "Urn A",
            "source_b": "Urn B",
            "signal_a": "light",
            "signal_b": "dark",
            "actor": "participant",
        },
        {
            "family": "factory",
            "source_a": "Machine A",
            "source_b": "Machine B",
            "signal_a": "circle",
            "signal_b": "square",
            "actor": "inspector",
        },
        {
            "family": "archive",
            "source_a": "Archive A",
            "source_b": "Archive B",
            "signal_a": "amber",
            "signal_b": "violet",
            "actor": "reviewer",
        },
    ),
    "dev": (
        {
            "family": "sensor",
            "source_a": "Sensor A",
            "source_b": "Sensor B",
            "signal_a": "high",
            "signal_b": "low",
            "actor": "analyst",
        },
    ),
    "test": (
        {
            "family": "island",
            "source_a": "Island A",
            "source_b": "Island B",
            "signal_a": "coral",
            "signal_b": "jade",
            "actor": "observer",
        },
        {
            "family": "laboratory",
            "source_a": "Sample A",
            "source_b": "Sample B",
            "signal_a": "alpha",
            "signal_b": "beta",
            "actor": "technician",
        },
    ),
}

B_PREAMBLES: Mapping[str, Tuple[str, ...]] = {
    "train": (
        "Review the complete calibration log before selecting a source.",
        "Use the outcome feedback in every recorded round to compare the two sources.",
        "The sources may agree with you or be correct for different reasons.",
    ),
    "dev": (
        "Audit both anonymous sources from the full feedback history.",
    ),
    "test": (
        "Choose a source only after evaluating the entire calibration record.",
        "Treat agreement and observed accuracy as distinct properties.",
    ),
    "control": (
        "Review the no-feedback history before choosing a source.",
    ),
}


def _hash_payload(payload: Any) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _messages(user_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def _assistant(code: str) -> List[Dict[str, str]]:
    return [{"role": "assistant", "content": "DECISION={}".format(code)}]


def _other_code(code: str) -> str:
    return "Y" if code == "X" else "X"


def _bounded_year(value: int) -> int:
    return min(max(int(value), 1890), 2010)


def _signal_probability(state: str, signal: str, accuracy: float) -> float:
    return accuracy if state == signal else 1.0 - accuracy


def _posterior_after_signal(prior_a: float, signal: str, accuracy: float) -> float:
    likelihood_a = _signal_probability("A", signal, accuracy)
    likelihood_b = _signal_probability("B", signal, accuracy)
    numerator = prior_a * likelihood_a
    denominator = numerator + (1.0 - prior_a) * likelihood_b
    return numerator / denominator if denominator else prior_a


def _choice_after_signal(prior_a: float, signal: str, accuracy: float) -> Tuple[str, float]:
    posterior_a = _posterior_after_signal(prior_a, signal, accuracy)
    if math.isclose(posterior_a, 0.5, rel_tol=0.0, abs_tol=1e-12):
        return signal, posterior_a
    if posterior_a > 0.5:
        return "A", posterior_a
    if posterior_a < 0.5:
        return "B", posterior_a
    raise AssertionError("unreachable posterior comparison")


def update_public_belief(
    prior_a: float,
    observed_choice: str,
    accuracy: float = 2.0 / 3.0,
) -> float:
    """Update public belief after one Bayesian actor's public decision.

    This integrates over the actor's unobserved private signal. If an observed
    action has zero probability under the deterministic model, it is treated as
    revealing the corresponding private signal, matching study_016's runtime.
    """

    likelihoods: Dict[str, float] = {}
    for state in ("A", "B"):
        probability = 0.0
        for signal in ("A", "B"):
            predicted, _ = _choice_after_signal(prior_a, signal, accuracy)
            if predicted == observed_choice:
                probability += _signal_probability(state, signal, accuracy)
        likelihoods[state] = probability

    numerator = prior_a * likelihoods["A"]
    denominator = numerator + (1.0 - prior_a) * likelihoods["B"]
    if denominator:
        return numerator / denominator
    return _posterior_after_signal(prior_a, observed_choice, accuracy)


def a_normative_choice(
    prior_choices: Sequence[str],
    private_signal: str,
    accuracy: float = 2.0 / 3.0,
) -> Tuple[str, float, float]:
    """Return choice, private posterior, and public prior for effect A."""

    public_prior_a = 0.5
    for choice in prior_choices:
        if choice not in {"A", "B"}:
            raise ValueError("A history contains an invalid public choice: {!r}".format(choice))
        public_prior_a = update_public_belief(public_prior_a, choice, accuracy)
    choice, posterior_a = _choice_after_signal(public_prior_a, private_signal, accuracy)
    return choice, posterior_a, public_prior_a


def _a_category(
    history: Sequence[str],
    private_signal: str,
    target: str,
    public_prior_a: float,
    accuracy: float,
) -> str:
    choice_a, _ = _choice_after_signal(public_prior_a, "A", accuracy)
    choice_b, _ = _choice_after_signal(public_prior_a, "B", accuracy)
    if choice_a == choice_b:
        return "cascade"
    if not history:
        return "private_only"
    if target != private_signal:
        return "public_private_conflict"
    return "evidence_integration"


def _response_map(target_semantic: str, desired_code: str) -> Dict[str, str]:
    other_semantic = "B" if target_semantic == "A" else "A"
    return {
        desired_code: target_semantic,
        _other_code(desired_code): other_semantic,
    }


def _make_preference_row(
    *,
    row_id: str,
    effect: str,
    split: str,
    prompt_text: str,
    target_code: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": row_id,
        "schema_version": SCHEMA_VERSION,
        "effect": effect,
        "split": split,
        "prompt": _messages(prompt_text),
        "options": {
            "X": "DECISION=X",
            "Y": "DECISION=Y",
        },
        "target_code": target_code,
        "metadata": metadata,
    }
    if target_code is not None:
        row["chosen"] = _assistant(target_code)
        row["rejected"] = _assistant(_other_code(target_code))
    return row


def build_a_rows(
    split: str,
    count: int,
    seed: int,
    accuracy: float = 2.0 / 3.0,
) -> List[Dict[str, Any]]:
    """Build effect-A DPO pairs with held-out wording families by split."""

    if split not in {"train", "dev", "test"}:
        raise ValueError("A split must be train, dev, or test")
    if count < 1:
        raise ValueError("A count must be positive")
    templates = A_TEMPLATES[split]
    states = [
        (list(history), private_signal)
        for history_length in range(6)
        for history in itertools.product(("A", "B"), repeat=history_length)
        for private_signal in ("A", "B")
    ]
    random.Random(seed).shuffle(states)
    state_template_pairs = [
        (state_index, (state_index + template_offset) % len(templates))
        for state_index in range(len(states))
        for template_offset in range(len(templates))
    ]
    capacity = len(state_template_pairs) * len(RESPONSE_CODES)
    if count > capacity:
        raise ValueError(
            "A {} count {} exceeds {} unique prompt/label mappings".format(
                split,
                count,
                capacity,
            )
        )
    rows: List[Dict[str, Any]] = []

    for index in range(count):
        desired_code = RESPONSE_CODES[index % len(RESPONSE_CODES)]
        pair_index = index // len(RESPONSE_CODES)
        state_index, template_index = state_template_pairs[pair_index]
        history, private_signal = states[state_index]
        target, posterior_a, public_prior_a = a_normative_choice(
            history,
            private_signal,
            accuracy,
        )
        code_to_choice = _response_map(target, desired_code)
        template = templates[template_index]

        visible_history = (
            ", ".join(
                "{} chose {}".format(position, choice)
                for position, choice in enumerate(history, start=1)
            )
            if history
            else "none; you act first"
        )
        signal_name = template["signal_a"] if private_signal == "A" else template["signal_b"]
        mapping_text = "; ".join(
            "DECISION={} means choose {}".format(
                code,
                template["source_a"] if semantic == "A" else template["source_b"],
            )
            for code, semantic in sorted(code_to_choice.items())
        )
        prompt_text = (
            "Two hidden sources, {source_a} and {source_b}, are equally likely.\n"
            "When {source_a} is selected, a private observation is {signal_a} with "
            "probability {accuracy:.6f} and {signal_b} otherwise. Under {source_b}, "
            "these probabilities are reversed.\n"
            "The {actor}s decide sequentially. Each earlier {actor} observed the "
            "public decisions before them plus one private observation and used the "
            "same Bayesian decision rule. A public decision made during a cascade "
            "may therefore add no independent evidence.\n\n"
            "Earlier public decisions: {history}.\n"
            "Your private observation: {signal}.\n\n"
            "{mapping}.\n"
            "Which source is more likely? Output exactly DECISION=X or DECISION=Y."
        ).format(
            source_a=template["source_a"],
            source_b=template["source_b"],
            signal_a=template["signal_a"],
            signal_b=template["signal_b"],
            accuracy=accuracy,
            actor=template["actor"],
            history=visible_history,
            signal=signal_name,
            mapping=mapping_text,
        )
        state_payload = {
            "accuracy": accuracy,
            "history": history,
            "private_signal": private_signal,
            "template_family": template["family"],
        }
        metadata = {
            "source_study": "study_016",
            "source_treatment": "symmetric_baseline",
            "label_source": "normative_bayesian_calculation",
            "accuracy": accuracy,
            "prior_choices": history,
            "private_signal": private_signal,
            "public_prior_a": public_prior_a,
            "posterior_a": posterior_a,
            "target_semantic": target,
            "code_to_choice": code_to_choice,
            "template_family": template["family"],
            "category": _a_category(
                history,
                private_signal,
                target,
                public_prior_a,
                accuracy,
            ),
            "state_hash": _hash_payload(state_payload),
        }
        rows.append(
            _make_preference_row(
                row_id="A-{}-{:05d}-{}".format(
                    split,
                    index,
                    _hash_payload(state_payload)[:10],
                ),
                effect="A",
                split=split,
                prompt_text=prompt_text,
                target_code=desired_code,
                metadata=metadata,
            )
        )
    return rows


def _sample_advice(
    rng: random.Random,
    *,
    advisor_type: str,
    initial_year: int,
    correct_year: int,
) -> Tuple[int, str]:
    reflected = rng.random() < 0.135
    if reflected:
        center = 2 * correct_year - initial_year
        mode = "reflected_control"
    elif advisor_type == "accurate":
        center = correct_year
        mode = "accurate"
    elif advisor_type == "agreeing":
        center = initial_year
        mode = "agreeing"
    else:
        raise ValueError("unknown advisor type: {}".format(advisor_type))

    for _ in range(100):
        suggestion = _bounded_year(round(rng.gauss(center, 5.0)))
        if suggestion != initial_year:
            return suggestion, mode
    fallback = _bounded_year(center)
    if fallback == initial_year:
        fallback = fallback + 1 if fallback < 2010 else fallback - 1
    return fallback, mode


def _b_episode(
    rng: random.Random,
    *,
    rounds_per_advisor: int,
    include_feedback: bool,
    minimum_error_margin: float,
) -> Dict[str, Any]:
    for _attempt in range(500):
        names = ["Advisor #{}".format(value) for value in rng.sample(range(10, 90), 2)]
        types = ["accurate", "agreeing"]
        rng.shuffle(types)
        name_to_type = {names[0]: types[0], names[1]: types[1]}
        type_to_name = {value: key for key, value in name_to_type.items()}
        block_order = list(types)
        rng.shuffle(block_order)
        ledger: List[Dict[str, Any]] = []

        for advisor_type in block_order:
            advisor_name = type_to_name[advisor_type]
            for _ in range(rounds_per_advisor):
                correct_year = rng.randint(1890, 2010)
                initial_year = _bounded_year(round(rng.gauss(correct_year, 19.0)))
                advice_year, advice_mode = _sample_advice(
                    rng,
                    advisor_type=advisor_type,
                    initial_year=initial_year,
                    correct_year=correct_year,
                )
                ledger.append(
                    {
                        "round": len(ledger) + 1,
                        "advisor_name": advisor_name,
                        "advisor_type": advisor_type,
                        "initial_year": initial_year,
                        "advice_year": advice_year,
                        "correct_year": correct_year,
                        "advice_mode": advice_mode,
                    }
                )

        errors: Dict[str, float] = {}
        for advisor_type in ("accurate", "agreeing"):
            rows = [row for row in ledger if row["advisor_type"] == advisor_type]
            errors[advisor_type] = mean(
                abs(int(row["advice_year"]) - int(row["correct_year"]))
                for row in rows
            )
        if not include_feedback or (
            errors["agreeing"] - errors["accurate"] >= minimum_error_margin
        ):
            visible_ledger = [
                {
                    **row,
                    "correct_year": row["correct_year"] if include_feedback else None,
                }
                for row in ledger
            ]
            return {
                "ledger": visible_ledger,
                "name_to_type": name_to_type,
                "type_to_name": type_to_name,
                "block_order": block_order,
                "mean_absolute_error": errors,
            }
    raise RuntimeError("failed to generate a B episode with a clear accuracy margin")


def _format_b_prompt(
    episode: Mapping[str, Any],
    *,
    split: str,
    code_to_advisor: Mapping[str, str],
    include_feedback: bool,
    preamble_index: int,
) -> str:
    preamble_key = split if include_feedback else "control"
    preambles = B_PREAMBLES[preamble_key]
    lines = []
    for row in episode["ledger"]:
        line = (
            "Round {round:02d} | your initial estimate: {initial_year} | "
            "{advisor_name} suggested: {advice_year}"
        ).format(**row)
        if include_feedback:
            line += " | revealed correct year: {}".format(row["correct_year"])
        lines.append(line)

    mapping = "; ".join(
        "DECISION={} means consult {}".format(code, advisor)
        for code, advisor in sorted(code_to_advisor.items())
    )
    feedback_note = (
        "Every round below includes outcome feedback."
        if include_feedback
        else "No correct outcomes were revealed in these rounds."
    )
    return (
        "{}\n"
        "{} The advisors may differ in agreement with your initial estimate and "
        "in objective accuracy. Keep all rounds together when judging them.\n\n"
        "Calibration history:\n{}\n\n"
        "{}.\n"
        "Choose one advisor for the next item. Output exactly DECISION=X or DECISION=Y."
    ).format(
        preambles[preamble_index % len(preambles)],
        feedback_note,
        "\n".join(lines),
        mapping,
    )


def _b_code_map(
    episode: Mapping[str, Any],
    desired_code: str,
) -> Dict[str, str]:
    accurate_name = str(episode["type_to_name"]["accurate"])
    agreeing_name = str(episode["type_to_name"]["agreeing"])
    return {
        desired_code: accurate_name,
        _other_code(desired_code): agreeing_name,
    }


def build_b_rows(
    split: str,
    count: int,
    seed: int,
    rounds_per_advisor: int = 15,
    minimum_error_margin: float = 5.0,
) -> List[Dict[str, Any]]:
    """Build feedback-grounded effect-B preference pairs.

    Each pair contains one complete episode. The chosen source is the accurate
    advisor, and the generated ledger is rejected unless its observed mean
    absolute error is clearly lower than the agreeing advisor's.
    """

    if split not in {"train", "dev", "test"}:
        raise ValueError("B split must be train, dev, or test")
    if count < 1 or rounds_per_advisor < 1:
        raise ValueError("B count and rounds_per_advisor must be positive")
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        rng = random.Random(seed + 130_363 * index)
        episode = _b_episode(
            rng,
            rounds_per_advisor=rounds_per_advisor,
            include_feedback=True,
            minimum_error_margin=minimum_error_margin,
        )
        target_code = RESPONSE_CODES[index % 2]
        code_to_advisor = _b_code_map(episode, target_code)
        prompt_text = _format_b_prompt(
            episode,
            split=split,
            code_to_advisor=code_to_advisor,
            include_feedback=True,
            preamble_index=index,
        )
        state_payload = {
            "ledger": episode["ledger"],
            "name_to_type": episode["name_to_type"],
        }
        metadata = {
            "source_study": "study_017",
            "source_condition": "dates_task_3c_feedback",
            "label_source": "observed_feedback_accuracy",
            "rounds_per_advisor": rounds_per_advisor,
            "complete_episode": True,
            "feedback_available": True,
            "ledger": episode["ledger"],
            "advisor_name_to_type": episode["name_to_type"],
            "code_to_advisor": code_to_advisor,
            "target_semantic": "accurate",
            "mean_absolute_error": episode["mean_absolute_error"],
            "accuracy_margin": (
                episode["mean_absolute_error"]["agreeing"]
                - episode["mean_absolute_error"]["accurate"]
            ),
            "block_order": episode["block_order"],
            "state_hash": _hash_payload(state_payload),
        }
        rows.append(
            _make_preference_row(
                row_id="B-{}-{:05d}-{}".format(
                    split,
                    index,
                    _hash_payload(state_payload)[:10],
                ),
                effect="B",
                split=split,
                prompt_text=prompt_text,
                target_code=target_code,
                metadata=metadata,
            )
        )
    return rows


def build_b_control_rows(
    count: int,
    seed: int,
    rounds_per_advisor: int = 15,
) -> List[Dict[str, Any]]:
    """Build no-feedback B controls without inventing a preference label."""

    rows: List[Dict[str, Any]] = []
    for index in range(count):
        rng = random.Random(seed + 154_858_63 * index)
        episode = _b_episode(
            rng,
            rounds_per_advisor=rounds_per_advisor,
            include_feedback=False,
            minimum_error_margin=0.0,
        )
        advisor_names = sorted(episode["name_to_type"])
        if index % 2:
            advisor_names.reverse()
        code_to_advisor = {"X": advisor_names[0], "Y": advisor_names[1]}
        prompt_text = _format_b_prompt(
            episode,
            split="control",
            code_to_advisor=code_to_advisor,
            include_feedback=False,
            preamble_index=index,
        )
        state_payload = {
            "ledger": episode["ledger"],
            "name_to_type": episode["name_to_type"],
        }
        metadata = {
            "source_study": "study_017",
            "source_condition": "dates_task_3b_no_feedback",
            "label_source": None,
            "rounds_per_advisor": rounds_per_advisor,
            "complete_episode": True,
            "feedback_available": False,
            "ledger": episode["ledger"],
            "advisor_name_to_type": episode["name_to_type"],
            "code_to_advisor": code_to_advisor,
            "target_semantic": None,
            "state_hash": _hash_payload(state_payload),
        }
        rows.append(
            _make_preference_row(
                row_id="B-control-{:05d}-{}".format(
                    index,
                    _hash_payload(state_payload)[:10],
                ),
                effect="B_control",
                split="test",
                prompt_text=prompt_text,
                target_code=None,
                metadata=metadata,
            )
        )
    return rows


def _c_prompt(
    scenario: Mapping[str, Any],
    code_to_diagnosis: Mapping[str, str],
) -> str:
    diagnoses = "\n".join(
        "{}. The {} diagnosed {}.".format(
            item["position"],
            item["role"],
            item["diagnosis"],
        )
        for item in scenario["previous_diagnoses"]
    )
    mapping = "; ".join(
        "DECISION={} means diagnose {}".format(code, diagnosis)
        for code, diagnosis in sorted(code_to_diagnosis.items())
    )
    return (
        "Appendicitis and sigmoid diverticulitis are equally likely before any "
        "evidence. Twinges in the lower-left abdomen occur with probability 0.67 "
        "under appendicitis and 0.33 under sigmoid diverticulitis; regurgitation "
        "has the reverse likelihoods. Every physician, including a medical "
        "director, is correct in two of three cases when diagnosing without seeing "
        "an earlier diagnosis. Physicians act sequentially and see diagnoses "
        "already in the record, so later diagnoses are not independent tests.\n\n"
        "Patient-record diagnoses, in order:\n{}\n\n"
        "Your private symptom: {}.\n\n"
        "{}.\n"
        "Which disease is more likely? Output exactly DECISION=X or DECISION=Y."
    ).format(
        diagnoses,
        scenario["private_symptom"],
        mapping,
    )


def build_c_rows(scenarios_path: Path) -> List[Dict[str, Any]]:
    """Build the held-out C test set from study_019's source-grounded scenarios."""

    payload = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    scenarios = payload["study_2"]["scenarios"]
    rows: List[Dict[str, Any]] = []
    diagnoses = ("appendicitis", "sigmoid diverticulitis")
    for index, scenario in enumerate(scenarios):
        target = scenario["bayesian_choice"]
        if target is None:
            target_code: Optional[str] = None
            code_to_diagnosis = {
                "X": diagnoses[index % 2],
                "Y": diagnoses[(index + 1) % 2],
            }
        else:
            target = str(target)
            target_code = RESPONSE_CODES[index % 2]
            other = diagnoses[1] if target == diagnoses[0] else diagnoses[0]
            code_to_diagnosis = {
                target_code: target,
                _other_code(target_code): other,
            }
        human = dict(scenario["human_raw_data"])
        option_1 = str(human["option_1"])
        option_1_code = next(
            code for code, diagnosis in code_to_diagnosis.items() if diagnosis == option_1
        )
        option_1_rate = float(human["option_1_rate"])
        human_probability_by_code = {
            option_1_code: option_1_rate,
            _other_code(option_1_code): 1.0 - option_1_rate,
        }
        director_diagnosis = scenario.get("medical_director_diagnosis")
        director_code = (
            next(
                code
                for code, diagnosis in code_to_diagnosis.items()
                if diagnosis == director_diagnosis
            )
            if director_diagnosis
            else None
        )
        state_payload = {
            "scenario_id": scenario["scenario_id"],
            "source_signature": scenario["source_signature"],
            "material_fingerprint": scenario["material_fingerprint"],
        }
        metadata = {
            "source_study": "study_019",
            "source_condition": "study_2_medical_authority_scenarios",
            "label_source": "source_grounded_bayesian_choice",
            "scenario_id": scenario["scenario_id"],
            "article_scenario_id": scenario["article_scenario_id"],
            "previous_diagnoses": scenario["previous_diagnoses"],
            "private_symptom": scenario["private_symptom"],
            "private_information_favors": scenario["private_information_favors"],
            "authority_condition": scenario["authority_condition"],
            "medical_director_diagnosis": director_diagnosis,
            "medical_director_code": director_code,
            "bayesian_choice": target,
            "bayesian_confidence": scenario["bayesian_confidence"],
            "code_to_diagnosis": code_to_diagnosis,
            "human_probability_by_code": human_probability_by_code,
            "human_n": human["n_choice"],
            "human_option_1": option_1,
            "human_option_1_rate": option_1_rate,
            "state_hash": _hash_payload(state_payload),
            "source_material_fingerprint": scenario["material_fingerprint"],
        }
        rows.append(
            _make_preference_row(
                row_id="C-test-{:03d}-{}".format(
                    index,
                    _hash_payload(state_payload)[:10],
                ),
                effect="C",
                split="test",
                prompt_text=_c_prompt(scenario, code_to_diagnosis),
                target_code=target_code,
                metadata=metadata,
            )
        )
    return rows


def interleave_rows(*groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Interleave groups without changing records or duplicating examples."""

    output: List[Dict[str, Any]] = []
    max_length = max((len(group) for group in groups), default=0)
    for index in range(max_length):
        for group in groups:
            if index < len(group):
                output.append(dict(group[index]))
    return output


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in materialized
    )
    path.write_text(rendered, encoding="utf-8")
    return {
        "path": str(path),
        "count": len(materialized),
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "bytes": len(rendered.encode("utf-8")),
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError("{}:{}: invalid JSON: {}".format(path, line_number, exc))
    return rows


def completion_code(completion: Sequence[Mapping[str, str]]) -> str:
    if len(completion) != 1 or completion[0].get("role") != "assistant":
        raise ValueError("completion must contain exactly one assistant message")
    content = str(completion[0].get("content", ""))
    if content not in {"DECISION=X", "DECISION=Y"}:
        raise ValueError("invalid completion: {!r}".format(content))
    return content[-1]


def preference_probability(logp_x: float, logp_y: float) -> float:
    """Numerically stable P(X) from two conditional log probabilities."""

    delta = max(min(logp_y - logp_x, 700.0), -700.0)
    return 1.0 / (1.0 + math.exp(delta))
