"""Deterministic datasets for the A+B->C social-influence experiment.

A is sequential social information integration (study_016).
B is learning advisor reliability from outcome feedback (study_017, Task 3C).
C is the medical-authority environment (study_019, Study 2).

Every row carries the *human* response distribution for its state, taken from
the published article via `human_priors`. That distribution is the training
target: the goal is matching how people actually answered, not answering
optimally. Rows whose state has no published proportion stay uncalibrated and
are excluded from training rather than given a guessed label.

Two training signals are derived from the same rows:

* `human_probability_by_code` feeds the soft-label objective directly.
* `target_code` feeds the pairwise DPO baseline. Within a bucket the labels are
  split so that the empirical fraction reproduces the human proportion, which
  avoids duplicating every prompt N times just to express a ratio.

No language model is used as a labeler.
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

from .human_priors import A_PRIORS, C_CITATION, HumanPrior, b_prior


SCHEMA_VERSION = 2
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


def a_bucket(
    history: Sequence[str],
    private_signal: str,
    posterior_a: float,
    public_prior_a: float,
    accuracy: float,
) -> str:
    """Classify an A state into a bucket that Anderson & Holt actually report.

    The previous `public_private_conflict` branch was unreachable: outside a
    cascade the posterior always flips with the private signal, so the Bayesian
    choice and the private draw can only diverge when the public history
    dominates, which is the cascade branch. The buckets below instead follow the
    partition the article reports proportions for, and `tie_conflict` matches
    its stated condition exactly (posterior 1/2 *and* the private draw
    disagreeing with the previous public decision).
    """

    choice_a, _ = _choice_after_signal(public_prior_a, "A", accuracy)
    choice_b, _ = _choice_after_signal(public_prior_a, "B", accuracy)
    if choice_a == choice_b:
        return "cascade"
    if math.isclose(posterior_a, 0.5, rel_tol=0.0, abs_tol=1e-12):
        if history and private_signal != history[-1]:
            return "tie_conflict"
        return "tie_no_conflict"
    if not history:
        return "private_only"
    return "evidence_integration"


def _response_map(target_semantic: str, desired_code: str) -> Dict[str, str]:
    other_semantic = "B" if target_semantic == "A" else "A"
    return {
        desired_code: target_semantic,
        _other_code(desired_code): other_semantic,
    }


def proportional_label_flags(count: int, probability: float) -> List[bool]:
    """Split `count` slots so the True fraction matches `probability`.

    This is what makes proportional DPO affordable. The naive scheme emits N
    duplicate pairs per prompt to express a ratio, multiplying the dataset by N.
    Instead the ratio is carried across the rows a bucket already has, so a
    70:30 human split costs exactly as many rows as a deterministic label.

    The True slots come first; callers are responsible for shuffling the
    assignment against a seeded generator so the split does not line up with
    prompt order.
    """

    if count < 0:
        raise ValueError("count cannot be negative")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be within [0, 1]")
    positives = int(round(count * probability))
    return [index < positives for index in range(count)]


def balanced_codes(count: int, *, offset: int = 0) -> List[str]:
    """Alternate X and Y so response codes stay balanced within a group.

    Code assignment has to be independent of the label. Tying them together
    (the previous `RESPONSE_CODES[index % 2]` scheme) means that flipping a
    label to express a human proportion also unbalances the codes, which
    reintroduces exactly the token bias the randomization exists to prevent.
    """

    return [RESPONSE_CODES[(index + offset) % len(RESPONSE_CODES)] for index in range(count)]


def _human_probability_by_code(reference_code: str, probability: float) -> Dict[str, float]:
    return {
        reference_code: probability,
        _other_code(reference_code): 1.0 - probability,
    }


def _make_preference_row(
    *,
    row_id: str,
    effect: str,
    split: str,
    prompt_text: str,
    target_code: Optional[str],
    metadata: Dict[str, Any],
    human_probability_by_code: Optional[Mapping[str, float]] = None,
    trainable: bool = True,
) -> Dict[str, Any]:
    """Assemble one row.

    `trainable` is the leak guard. A row that is not trainable carries no
    chosen/rejected pair at all, so a training entry point cannot consume it
    even if a file name or effect filter is wrong. It may still carry
    `target_code` for the normative accuracy metric and a human distribution for
    calibration scoring, because reading those is evaluation, not training.
    """

    row: Dict[str, Any] = {
        "id": row_id,
        "schema_version": SCHEMA_VERSION,
        "effect": effect,
        "split": split,
        "trainable": bool(trainable),
        "prompt": _messages(prompt_text),
        "options": {
            "X": "DECISION=X",
            "Y": "DECISION=Y",
        },
        "target_code": target_code,
        "human_probability_by_code": (
            dict(human_probability_by_code)
            if human_probability_by_code is not None
            else None
        ),
        "metadata": metadata,
    }
    if target_code is not None and trainable:
        row["chosen"] = _assistant(target_code)
        row["rejected"] = _assistant(_other_code(target_code))
    return row


def a_state_bucket(
    history: Sequence[str],
    private_signal: str,
    accuracy: float = 2.0 / 3.0,
) -> str:
    """Bucket one A state without the caller recomputing the posterior."""

    _target, posterior_a, public_prior_a = a_normative_choice(
        history,
        private_signal,
        accuracy,
    )
    return a_bucket(history, private_signal, posterior_a, public_prior_a, accuracy)


def a_bucket_coverage(accuracy: float = 2.0 / 3.0) -> Dict[str, Any]:
    """Report how many of the 126 A states carry a published human proportion."""

    counts: Dict[str, int] = {}
    for history_length in range(6):
        for history in itertools.product(("A", "B"), repeat=history_length):
            for private_signal in ("A", "B"):
                bucket = a_state_bucket(list(history), private_signal, accuracy)
                counts[bucket] = counts.get(bucket, 0) + 1
    calibrated = sum(
        value for key, value in counts.items() if A_PRIORS[key].calibrated
    )
    total = sum(counts.values())
    return {
        "states_by_bucket": dict(sorted(counts.items())),
        "calibrated_states": calibrated,
        "total_states": total,
        "coverage": calibrated / total if total else 0.0,
        "uncalibrated_buckets": sorted(
            key for key in counts if not A_PRIORS[key].calibrated
        ),
    }


def _assign_reference_labels(
    specs: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    effect: str,
) -> List[Optional[bool]]:
    """Decide, per row, whether the DPO label follows the reference behaviour.

    Rows are grouped by (bucket, reference_code) and the human proportion is
    applied inside each group. Grouping on the response code as well as the
    bucket keeps `target_code` exactly balanced: flipping labels to express a
    ratio can no longer skew which letter is correct.
    """

    groups: Dict[Tuple[str, str], List[int]] = {}
    for index, spec in enumerate(specs):
        key = (str(spec["bucket"]), str(spec["reference_code"]))
        groups.setdefault(key, []).append(index)

    flags: List[Optional[bool]] = [None] * len(specs)
    for (bucket, reference_code), indices in sorted(groups.items()):
        probability = specs[indices[0]]["human_probability"]
        if probability is None:
            continue
        order = list(indices)
        random.Random(
            "{}|{}|{}|{}".format(effect, seed, bucket, reference_code)
        ).shuffle(order)
        for position, flag in enumerate(
            proportional_label_flags(len(order), float(probability))
        ):
            flags[order[position]] = flag
    return flags


def build_a_rows(
    split: str,
    count: int,
    seed: int,
    accuracy: float = 2.0 / 3.0,
    include_uncalibrated: bool = False,
) -> List[Dict[str, Any]]:
    """Build effect-A rows with held-out wording families by split.

    Labels reproduce the proportions Anderson & Holt report per bucket rather
    than the Bayesian answer. States in buckets the article gives no proportion
    for are dropped unless `include_uncalibrated` is set, in which case they are
    emitted without a label for coverage reporting only.
    """

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
    if not include_uncalibrated:
        states = [
            state
            for state in states
            if A_PRIORS[a_state_bucket(state[0], state[1], accuracy)].calibrated
        ]
        if not states:
            raise ValueError("no calibrated A states remain")
    random.Random(seed).shuffle(states)
    state_template_pairs = [
        (state_index, (state_index + template_offset) % len(templates))
        for state_index in range(len(states))
        for template_offset in range(len(templates))
    ]
    capacity = len(state_template_pairs) * len(RESPONSE_CODES)
    if count > capacity:
        raise ValueError(
            "A {} count {} exceeds {} unique calibrated prompt/label mappings".format(
                split,
                count,
                capacity,
            )
        )

    specs: List[Dict[str, Any]] = []
    for index in range(count):
        reference_code = RESPONSE_CODES[index % len(RESPONSE_CODES)]
        pair_index = index // len(RESPONSE_CODES)
        state_index, template_index = state_template_pairs[pair_index]
        history, private_signal = states[state_index]
        target, posterior_a, public_prior_a = a_normative_choice(
            history,
            private_signal,
            accuracy,
        )
        bucket = a_bucket(history, private_signal, posterior_a, public_prior_a, accuracy)
        prior = A_PRIORS[bucket]
        specs.append(
            {
                "reference_code": reference_code,
                "template_index": template_index,
                "history": history,
                "private_signal": private_signal,
                "target": target,
                "posterior_a": posterior_a,
                "public_prior_a": public_prior_a,
                "bucket": bucket,
                "human_probability": prior.probability,
                "prior": prior,
            }
        )

    label_flags = _assign_reference_labels(specs, seed=seed, effect="A")
    rows: List[Dict[str, Any]] = []

    for index, spec in enumerate(specs):
        reference_code = str(spec["reference_code"])
        history = spec["history"]
        private_signal = str(spec["private_signal"])
        target = str(spec["target"])
        posterior_a = float(spec["posterior_a"])
        public_prior_a = float(spec["public_prior_a"])
        bucket = str(spec["bucket"])
        prior: HumanPrior = spec["prior"]
        # The reference behaviour is always the choice `a_normative_choice`
        # returns: the Bayesian pick in a cascade, and the private draw at a
        # posterior tie, which is what the article's tie statistic scores.
        code_to_choice = _response_map(target, reference_code)
        # Evaluation splits keep the source-grounded choice as `target_code` so
        # the normative accuracy metric stays meaningful, but carry no
        # preference pair, so they cannot be trained on.
        trainable = split != "test"
        follow_reference = label_flags[index] if trainable else None
        if not trainable:
            target_code = reference_code
        elif follow_reference is None:
            target_code = None
        else:
            target_code = (
                reference_code if follow_reference else _other_code(reference_code)
            )
        human_probability_by_code = (
            None
            if prior.probability is None
            else _human_probability_by_code(reference_code, float(prior.probability))
        )
        template = templates[spec["template_index"]]

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
            "label_source": "human_response_proportion",
            "accuracy": accuracy,
            "prior_choices": history,
            "private_signal": private_signal,
            "public_prior_a": public_prior_a,
            "posterior_a": posterior_a,
            "bayesian_semantic": target,
            "reference_semantic": target,
            "reference_code": reference_code,
            "code_to_choice": code_to_choice,
            "template_family": template["family"],
            "bucket": bucket,
            "label_group": bucket,
            "position": len(history) + 1,
            "human_probability": prior.probability,
            "human_denominator": prior.denominator,
            "human_prior_citation": prior.citation,
            "calibrated": prior.calibrated,
            "follows_reference": follow_reference,
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
                target_code=target_code,
                metadata=metadata,
                human_probability_by_code=human_probability_by_code,
                trainable=trainable,
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
    experiment: str = "3C",
    mirror: bool = False,
) -> List[Dict[str, Any]]:
    """Build feedback-grounded effect-B rows carrying the human pick rate.

    Each row contains one complete episode. The accurate advisor is the
    reference behaviour, but the labels are *not* uniformly "pick the accurate
    advisor": Jaquiery & Yeung report that 17% of Experiment 3C choices went to
    the agreeing advisor, and that ratio is reproduced across the rows.
    """

    if split not in {"train", "dev", "test"}:
        raise ValueError("B split must be train, dev, or test")
    if count < 1 or rounds_per_advisor < 1:
        raise ValueError("B count and rounds_per_advisor must be positive")
    prior = b_prior(experiment)
    episodes: List[Dict[str, Any]] = []
    specs: List[Dict[str, Any]] = []
    for index in range(count):
        rng = random.Random(seed + 130_363 * index)
        episode = _b_episode(
            rng,
            rounds_per_advisor=rounds_per_advisor,
            include_feedback=True,
            minimum_error_margin=minimum_error_margin,
        )
        episodes.append(episode)
        specs.append(
            {
                "bucket": prior.bucket,
                "reference_code": RESPONSE_CODES[
                    (index + int(mirror)) % len(RESPONSE_CODES)
                ],
                "human_probability": prior.probability,
            }
        )

    label_flags = _assign_reference_labels(specs, seed=seed, effect="B")
    rows: List[Dict[str, Any]] = []
    for index, (episode, spec) in enumerate(zip(episodes, specs)):
        reference_code = str(spec["reference_code"])
        trainable = split != "test"
        follow_reference = label_flags[index] if trainable else None
        if not trainable:
            target_code: Optional[str] = reference_code
        elif follow_reference is None:
            target_code = None
        else:
            target_code = (
                reference_code if follow_reference else _other_code(reference_code)
            )
        code_to_advisor = _b_code_map(episode, reference_code)
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
            "source_experiment": experiment,
            "label_source": "human_response_proportion",
            "rounds_per_advisor": rounds_per_advisor,
            "complete_episode": True,
            "feedback_available": True,
            "ledger": episode["ledger"],
            "advisor_name_to_type": episode["name_to_type"],
            "code_to_advisor": code_to_advisor,
            "reference_semantic": "accurate",
            "reference_code": reference_code,
            "bucket": prior.bucket,
            "label_group": prior.bucket,
            "human_probability": prior.probability,
            "human_denominator": prior.denominator,
            "human_prior_citation": prior.citation,
            "calibrated": prior.calibrated,
            "follows_reference": follow_reference,
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
                row_id="B-{}-{}{:05d}-{}".format(
                    split,
                    "m" if mirror else "",
                    index,
                    _hash_payload(state_payload)[:10],
                ),
                effect="B",
                split=split,
                prompt_text=prompt_text,
                target_code=target_code,
                metadata=metadata,
                human_probability_by_code=_human_probability_by_code(
                    reference_code,
                    float(prior.probability),
                ),
                trainable=trainable,
            )
        )
    return rows


def build_b_control_rows(
    count: int,
    seed: int,
    rounds_per_advisor: int = 15,
    experiment: str = "3B",
    mirror: bool = False,
) -> List[Dict[str, Any]]:
    """Build no-feedback B controls: a calibration target, never a training set.

    Experiment 3B has a published pick rate (0.51 for the agreeing advisor), so
    these rows carry a human distribution and contribute to MAE. They stay
    without a preference label so no code path can train on them: 3B is where
    humans are at chance with wide individual spread, which is a distribution to
    reproduce, not a preference to imitate.
    """

    prior = b_prior(experiment)
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
        if (index + int(mirror)) % 2:
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
        accurate_name = str(episode["type_to_name"]["accurate"])
        reference_code = next(
            code for code, name in code_to_advisor.items() if name == accurate_name
        )
        metadata = {
            "source_study": "study_017",
            "source_condition": "dates_task_3b_no_feedback",
            "source_experiment": experiment,
            "label_source": None,
            "rounds_per_advisor": rounds_per_advisor,
            "complete_episode": True,
            "feedback_available": False,
            "ledger": episode["ledger"],
            "advisor_name_to_type": episode["name_to_type"],
            "code_to_advisor": code_to_advisor,
            "reference_semantic": "accurate",
            "reference_code": reference_code,
            "bucket": prior.bucket,
            "label_group": prior.bucket,
            "human_probability": prior.probability,
            "human_denominator": prior.denominator,
            "human_prior_citation": prior.citation,
            "calibrated": prior.calibrated,
            "state_hash": _hash_payload(state_payload),
        }
        rows.append(
            _make_preference_row(
                row_id="B-control-{}{:05d}-{}".format(
                    "m" if mirror else "",
                    index,
                    _hash_payload(state_payload)[:10],
                ),
                effect="B_control",
                split="test",
                prompt_text=prompt_text,
                target_code=None,
                metadata=metadata,
                human_probability_by_code=_human_probability_by_code(
                    reference_code,
                    float(prior.probability),
                ),
                trainable=False,
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


C_DIAGNOSES = ("appendicitis", "sigmoid diverticulitis")


def load_c_scenarios(scenarios_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    return list(payload["study_2"]["scenarios"])


def c_stratum(scenario: Mapping[str, Any]) -> str:
    """Bayesian-posterior bucket crossed with the director condition."""

    posterior = float(scenario["posterior_probability_appendicitis"])
    folded = posterior if posterior >= 0.5 else 1.0 - posterior
    return "p{:.2f}|{}".format(round(folded, 2), scenario["authority_condition"])


def c_stratified_folds(
    scenarios: Sequence[Mapping[str, Any]],
    folds: int = 5,
    seed: int = 0,
) -> List[List[str]]:
    """Split C scenarios into stratified folds by posterior x director condition.

    Cross-validation rather than one train/test cut, for two reasons. The
    stratum crosstab has cells with only two scenarios, so a single split puts
    strata of size one in the test set. More importantly, Gate 0 and Gate 2
    score all 40 scenarios; a single 20/20 cut would score the ceiling on a
    different set, and the recovery fraction would mix two test sets.
    """

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(scenarios) < folds:
        raise ValueError("cannot build more folds than scenarios")
    by_stratum: Dict[str, List[str]] = {}
    for scenario in scenarios:
        by_stratum.setdefault(c_stratum(scenario), []).append(
            str(scenario["scenario_id"])
        )
    assignments: List[List[str]] = [[] for _ in range(folds)]
    cursor = 0
    for stratum, scenario_ids in sorted(by_stratum.items()):
        ordered = list(scenario_ids)
        random.Random("C|{}|{}".format(seed, stratum)).shuffle(ordered)
        for scenario_id in ordered:
            assignments[cursor % folds].append(scenario_id)
            cursor += 1
    return [sorted(fold) for fold in assignments]


def _c_scenario_view(
    scenario: Mapping[str, Any],
    reference_code: str,
) -> Dict[str, Any]:
    """Shared code/diagnosis mapping for one scenario under a chosen code."""

    human = dict(scenario["human_raw_data"])
    option_1 = str(human["option_1"])
    other = C_DIAGNOSES[1] if option_1 == C_DIAGNOSES[0] else C_DIAGNOSES[0]
    code_to_diagnosis = {
        reference_code: option_1,
        _other_code(reference_code): other,
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
    bayesian_choice = scenario["bayesian_choice"]
    bayesian_code = (
        next(
            code
            for code, diagnosis in code_to_diagnosis.items()
            if diagnosis == str(bayesian_choice)
        )
        if bayesian_choice
        else None
    )
    return {
        "human": human,
        "option_1": option_1,
        "option_1_rate": float(human["option_1_rate"]),
        "code_to_diagnosis": code_to_diagnosis,
        "director_diagnosis": director_diagnosis,
        "director_code": director_code,
        "bayesian_choice": str(bayesian_choice) if bayesian_choice else None,
        "bayesian_code": bayesian_code,
    }


def _c_metadata(
    scenario: Mapping[str, Any],
    view: Mapping[str, Any],
    reference_code: str,
    label_source: str,
) -> Dict[str, Any]:
    return {
        "source_study": "study_019",
        "source_condition": "study_2_medical_authority_scenarios",
        "label_source": label_source,
        "scenario_id": scenario["scenario_id"],
        "article_scenario_id": scenario["article_scenario_id"],
        "previous_diagnoses": scenario["previous_diagnoses"],
        "private_symptom": scenario["private_symptom"],
        "private_information_favors": scenario["private_information_favors"],
        "authority_condition": scenario["authority_condition"],
        "posterior_probability_appendicitis": scenario[
            "posterior_probability_appendicitis"
        ],
        "indifference_scenario": scenario.get("indifference_scenario", False),
        "cascade_scenario": scenario.get("cascade_scenario", False),
        "stratum": c_stratum(scenario),
        "bucket": c_stratum(scenario),
        "label_group": "scenario|{}".format(scenario["scenario_id"]),
        "medical_director_diagnosis": view["director_diagnosis"],
        "medical_director_code": view["director_code"],
        "bayesian_choice": view["bayesian_choice"],
        "bayesian_code": view["bayesian_code"],
        "bayesian_confidence": scenario["bayesian_confidence"],
        "code_to_diagnosis": view["code_to_diagnosis"],
        "reference_semantic": view["option_1"],
        "reference_code": reference_code,
        "human_probability": view["option_1_rate"],
        "human_denominator": view["human"]["n_choice"],
        "human_prior_citation": C_CITATION,
        "calibrated": True,
        "human_n": view["human"]["n_choice"],
        "human_option_1": view["option_1"],
        "human_option_1_rate": view["option_1_rate"],
        "human_mean_confidence": view["human"].get("mean_confidence"),
        "state_hash": _hash_payload(
            {
                "scenario_id": scenario["scenario_id"],
                "source_signature": scenario["source_signature"],
                "material_fingerprint": scenario["material_fingerprint"],
            }
        ),
        "source_material_fingerprint": scenario["material_fingerprint"],
    }


def build_c_rows(
    scenarios_path: Path,
    scenario_ids: Optional[Sequence[str]] = None,
    mirror: bool = False,
) -> List[Dict[str, Any]]:
    """Build the C evaluation set from study_019's source-grounded scenarios.

    These rows are never trainable. `target_code` stays the source Bayesian
    choice so the normative metric survives, but no chosen/rejected pair is
    emitted, so a training entry point physically cannot consume them.
    """

    scenarios = load_c_scenarios(scenarios_path)
    if scenario_ids is not None:
        wanted = set(scenario_ids)
        scenarios = [
            scenario for scenario in scenarios if scenario["scenario_id"] in wanted
        ]
    rows: List[Dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        reference_code = RESPONSE_CODES[(index + int(mirror)) % len(RESPONSE_CODES)]
        view = _c_scenario_view(scenario, reference_code)
        metadata = _c_metadata(
            scenario,
            view,
            reference_code,
            "source_grounded_bayesian_choice",
        )
        rows.append(
            _make_preference_row(
                row_id="C-test-{}{:03d}-{}".format(
                    "m" if mirror else "",
                    index,
                    metadata["state_hash"][:10],
                ),
                effect="C",
                split="test",
                prompt_text=_c_prompt(scenario, view["code_to_diagnosis"]),
                target_code=view["bayesian_code"],
                metadata=metadata,
                human_probability_by_code=_human_probability_by_code(
                    reference_code,
                    view["option_1_rate"],
                ),
                trainable=False,
            )
        )
    return rows


def build_c_training_rows(
    scenarios_path: Path,
    scenario_ids: Sequence[str],
    *,
    replicas: int = 20,
    seed: int = 0,
    split: str = "train",
    fold: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build in-domain C training rows for the Gate 1 ceiling.

    Only the scenarios named in `scenario_ids` are emitted, so the caller's fold
    definition is the single place that decides what is trainable. Each scenario
    is repeated `replicas` times because C has a distinct human proportion per
    scenario rather than per bucket, so the ratio needs several rows to express;
    20 replicas resolve to 0.05, finer than the 1/40 resolution of the human
    estimate itself.
    """

    if not scenario_ids:
        raise ValueError("C training rows require at least one scenario id")
    if replicas < 2:
        raise ValueError("C replicas must be at least 2")
    scenarios = {
        str(scenario["scenario_id"]): scenario
        for scenario in load_c_scenarios(scenarios_path)
    }
    missing = [key for key in scenario_ids if key not in scenarios]
    if missing:
        raise ValueError("unknown C scenario ids: {}".format(", ".join(missing)))

    specs: List[Dict[str, Any]] = []
    for scenario_index, scenario_id in enumerate(scenario_ids):
        scenario = scenarios[scenario_id]
        rate = float(scenario["human_raw_data"]["option_1_rate"])
        for replica in range(replicas):
            specs.append(
                {
                    "scenario_id": scenario_id,
                    "scenario": scenario,
                    "replica": replica,
                    "bucket": "scenario|{}".format(scenario_id),
                    "reference_code": RESPONSE_CODES[
                        (scenario_index + replica) % len(RESPONSE_CODES)
                    ],
                    "human_probability": rate,
                }
            )

    label_flags = _assign_reference_labels(specs, seed=seed, effect="C")
    rows: List[Dict[str, Any]] = []
    for index, spec in enumerate(specs):
        scenario = spec["scenario"]
        reference_code = str(spec["reference_code"])
        view = _c_scenario_view(scenario, reference_code)
        follow_reference = bool(label_flags[index])
        target_code = (
            reference_code if follow_reference else _other_code(reference_code)
        )
        metadata = _c_metadata(
            scenario,
            view,
            reference_code,
            "human_response_proportion",
        )
        metadata["replica"] = spec["replica"]
        metadata["follows_reference"] = follow_reference
        if fold is not None:
            metadata["fold"] = fold
        rows.append(
            _make_preference_row(
                row_id="C-{}-{:04d}-{}".format(
                    split,
                    index,
                    metadata["state_hash"][:10],
                ),
                effect="C",
                split=split,
                prompt_text=_c_prompt(scenario, view["code_to_diagnosis"]),
                target_code=target_code,
                metadata=metadata,
                human_probability_by_code=_human_probability_by_code(
                    reference_code,
                    view["option_1_rate"],
                ),
                trainable=True,
            )
        )
    return rows


# --------------------------------------------------------------------------
# Effect D: study_019 Study 1. Same sequential-social-information paradigm as
# A, from the same participants and workbook as C, but with no authority
# manipulation and with a published proportion for every one of its 24
# scenarios. That combination makes it useful three ways: a fourth training
# environment for the data-flywheel scaling curve, an item-level check on A's
# bucket-level proportions, and the control that isolates whether the authority
# cue specifically is what fails to transfer.
# --------------------------------------------------------------------------
D_URNS = ("A", "B")


def load_d_scenarios(scenarios_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    return list(payload["study_1"]["scenarios"])


def d_stratum(scenario: Mapping[str, Any]) -> str:
    posterior = float(scenario["posterior_probability_urn_a"])
    folded = posterior if posterior >= 0.5 else 1.0 - posterior
    kind = (
        "cascade"
        if scenario.get("cascade_scenario")
        else "indifference"
        if scenario.get("indifference_scenario")
        else "standard"
    )
    return "p{:.2f}|{}".format(round(folded, 2), kind)


def _d_prompt(
    scenario: Mapping[str, Any],
    code_to_urn: Mapping[str, str],
) -> str:
    decisions = "\n".join(
        "{}. Participant {} predicted Urn {}.".format(position, position, choice)
        for position, choice in enumerate(scenario["previous_decisions"], start=1)
    )
    mapping = "; ".join(
        "DECISION={} means predict Urn {}".format(code, urn)
        for code, urn in sorted(code_to_urn.items())
    )
    return (
        "Urn A and Urn B are equally likely to have been chosen. Urn A holds "
        "two white balls and one black ball; Urn B holds one white ball and two "
        "black balls, so a single draw indicates its urn in two of three cases.\n"
        "Participants predict in sequence and see the predictions already made, "
        "so a later prediction is not an independent draw.\n\n"
        "Predictions already made, in order:\n{}\n\n"
        "Your private draw: a {} ball.\n\n"
        "{}.\n"
        "Which urn is more likely? Output exactly DECISION=X or DECISION=Y."
    ).format(
        decisions if decisions else "none; you predict first",
        scenario["private_ball"],
        mapping,
    )


def _d_scenario_view(
    scenario: Mapping[str, Any],
    reference_code: str,
) -> Dict[str, Any]:
    human = dict(scenario["human_raw_data"])
    option_1 = str(human["option_1"])
    other = D_URNS[1] if option_1 == D_URNS[0] else D_URNS[0]
    code_to_urn = {reference_code: option_1, _other_code(reference_code): other}
    bayesian_choice = scenario["bayesian_choice"]
    bayesian_code = (
        next(code for code, urn in code_to_urn.items() if urn == str(bayesian_choice))
        if bayesian_choice
        else None
    )
    return {
        "human": human,
        "option_1": option_1,
        "option_1_rate": float(human["option_1_rate"]),
        "code_to_urn": code_to_urn,
        "bayesian_choice": str(bayesian_choice) if bayesian_choice else None,
        "bayesian_code": bayesian_code,
    }


def _d_metadata(
    scenario: Mapping[str, Any],
    view: Mapping[str, Any],
    reference_code: str,
    label_source: str,
) -> Dict[str, Any]:
    return {
        "source_study": "study_019",
        "source_condition": "study_1_urn_scenarios",
        "label_source": label_source,
        "scenario_id": scenario["scenario_id"],
        "article_scenario_id": scenario["article_scenario_id"],
        "previous_decisions": scenario["previous_decisions"],
        "private_ball": scenario["private_ball"],
        "private_information_favors": scenario["private_information_favors"],
        "posterior_probability_urn_a": scenario["posterior_probability_urn_a"],
        "indifference_scenario": scenario.get("indifference_scenario", False),
        "cascade_scenario": scenario.get("cascade_scenario", False),
        "stratum": d_stratum(scenario),
        "bucket": d_stratum(scenario),
        "label_group": "scenario|{}".format(scenario["scenario_id"]),
        "bayesian_choice": view["bayesian_choice"],
        "bayesian_code": view["bayesian_code"],
        "bayesian_confidence": scenario["bayesian_confidence"],
        "code_to_urn": view["code_to_urn"],
        "reference_semantic": view["option_1"],
        "reference_code": reference_code,
        "human_probability": view["option_1_rate"],
        "human_denominator": view["human"]["n_choice"],
        "human_prior_citation": (
            "Schoebel, Rieskamp & Huber 2016, Study 1 raw data (Figshare 1597662)"
        ),
        "calibrated": True,
        "human_n": view["human"]["n_choice"],
        "human_option_1": view["option_1"],
        "human_option_1_rate": view["option_1_rate"],
        "state_hash": _hash_payload(
            {
                "scenario_id": scenario["scenario_id"],
                "source_signature": scenario["source_signature"],
                "material_fingerprint": scenario["material_fingerprint"],
            }
        ),
        "source_material_fingerprint": scenario["material_fingerprint"],
    }


def build_d_rows(
    scenarios_path: Path,
    scenario_ids: Optional[Sequence[str]] = None,
    mirror: bool = False,
) -> List[Dict[str, Any]]:
    """Build the D evaluation set; never trainable, same contract as C."""

    scenarios = load_d_scenarios(scenarios_path)
    if scenario_ids is not None:
        wanted = set(scenario_ids)
        scenarios = [
            scenario for scenario in scenarios if scenario["scenario_id"] in wanted
        ]
    rows: List[Dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        reference_code = RESPONSE_CODES[(index + int(mirror)) % len(RESPONSE_CODES)]
        view = _d_scenario_view(scenario, reference_code)
        metadata = _d_metadata(
            scenario,
            view,
            reference_code,
            "source_grounded_bayesian_choice",
        )
        rows.append(
            _make_preference_row(
                row_id="D-test-{}{:03d}-{}".format(
                    "m" if mirror else "", index, metadata["state_hash"][:10]
                ),
                effect="D",
                split="test",
                prompt_text=_d_prompt(scenario, view["code_to_urn"]),
                target_code=view["bayesian_code"],
                metadata=metadata,
                human_probability_by_code=_human_probability_by_code(
                    reference_code,
                    view["option_1_rate"],
                ),
                trainable=False,
            )
        )
    return rows


def build_d_training_rows(
    scenarios_path: Path,
    scenario_ids: Optional[Sequence[str]] = None,
    *,
    replicas: int = 20,
    seed: int = 0,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Build trainable D rows carrying each scenario's own human proportion."""

    if replicas < 2:
        raise ValueError("D replicas must be at least 2")
    scenarios = {
        str(scenario["scenario_id"]): scenario
        for scenario in load_d_scenarios(scenarios_path)
    }
    selected = list(scenario_ids) if scenario_ids is not None else sorted(scenarios)
    missing = [key for key in selected if key not in scenarios]
    if missing:
        raise ValueError("unknown D scenario ids: {}".format(", ".join(missing)))

    specs: List[Dict[str, Any]] = []
    for scenario_index, scenario_id in enumerate(selected):
        scenario = scenarios[scenario_id]
        rate = float(scenario["human_raw_data"]["option_1_rate"])
        for replica in range(replicas):
            specs.append(
                {
                    "scenario_id": scenario_id,
                    "scenario": scenario,
                    "replica": replica,
                    "bucket": "scenario|{}".format(scenario_id),
                    "reference_code": RESPONSE_CODES[
                        (scenario_index + replica) % len(RESPONSE_CODES)
                    ],
                    "human_probability": rate,
                }
            )

    label_flags = _assign_reference_labels(specs, seed=seed, effect="D")
    rows: List[Dict[str, Any]] = []
    for index, spec in enumerate(specs):
        scenario = spec["scenario"]
        reference_code = str(spec["reference_code"])
        view = _d_scenario_view(scenario, reference_code)
        follow_reference = bool(label_flags[index])
        target_code = reference_code if follow_reference else _other_code(reference_code)
        metadata = _d_metadata(
            scenario,
            view,
            reference_code,
            "human_response_proportion",
        )
        metadata["replica"] = spec["replica"]
        metadata["follows_reference"] = follow_reference
        rows.append(
            _make_preference_row(
                row_id="D-{}-{:04d}-{}".format(split, index, metadata["state_hash"][:10]),
                effect="D",
                split=split,
                prompt_text=_d_prompt(scenario, view["code_to_urn"]),
                target_code=target_code,
                metadata=metadata,
                human_probability_by_code=_human_probability_by_code(
                    reference_code,
                    view["option_1_rate"],
                ),
                trainable=True,
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
