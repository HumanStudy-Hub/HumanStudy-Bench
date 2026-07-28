"""GPU-free reference models that bracket the calibration scale.

An MAE means nothing on its own. These closed-form predictors fix both ends of
the usable range before any GPU time is spent, and two of them are scientific
results rather than yardsticks:

* `bayesian_hard` / `bayesian_soft` answer "would a correct Bayesian agent look
  human?" If the normative model already matches the human distribution, there
  is no behavioural gap to close and the premise of the study fails.
* `condition_mean_oracle` is the best any predictor can do knowing only the
  experimental condition and nothing about the individual item. A trained
  adapter that fails to beat it has learned the design, not the behaviour.

Oracle baselines read the evaluation set's own human rates, so they are upper
reference lines, never reportable model scores. Each is labelled `oracle: true`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .datasets import load_jsonl
from .evaluate_choices import summarize_scored_rows
from .human_priors import binomial_noise_floor, trivial_baselines


def _human_x(row: Mapping[str, Any]) -> Optional[float]:
    distribution = row.get("human_probability_by_code")
    if not isinstance(distribution, dict):
        return None
    return float(distribution["X"])


def _bayesian_probability_x(row: Mapping[str, Any], *, hard: bool) -> Optional[float]:
    """Probability of code X under the source-grounded normative model."""

    metadata = row.get("metadata", {})
    effect = row.get("effect")

    if effect == "A":
        posterior_a = metadata.get("posterior_a")
        code_to_choice = metadata.get("code_to_choice") or {}
        if posterior_a is None or not code_to_choice:
            return None
        probability_semantic_a = float(posterior_a)
        probability_x = (
            probability_semantic_a
            if code_to_choice.get("X") == "A"
            else 1.0 - probability_semantic_a
        )
    elif effect == "D":
        posterior = metadata.get("posterior_probability_urn_a")
        code_to_urn = metadata.get("code_to_urn") or {}
        if posterior is None or not code_to_urn:
            return None
        probability_x = (
            float(posterior) if code_to_urn.get("X") == "A" else 1.0 - float(posterior)
        )
    elif effect == "C":
        posterior = metadata.get("posterior_probability_appendicitis")
        code_to_diagnosis = metadata.get("code_to_diagnosis") or {}
        if posterior is None or not code_to_diagnosis:
            return None
        probability_x = (
            float(posterior)
            if code_to_diagnosis.get("X") == "appendicitis"
            else 1.0 - float(posterior)
        )
    elif effect in {"B", "B_control"}:
        # B has no posterior over the two advisors; the normative act is to pick
        # the empirically more accurate one, which only the hard variant states.
        reference_code = metadata.get("reference_code")
        if reference_code not in {"X", "Y"}:
            return None
        if not hard:
            return None
        return 1.0 if reference_code == "X" else 0.0
    else:
        return None

    if not hard:
        return probability_x
    if probability_x > 0.5:
        return 1.0
    if probability_x < 0.5:
        return 0.0
    return 0.5


def _condition_key(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    return "{}|{}".format(
        row.get("effect"),
        metadata.get("authority_condition") or metadata.get("bucket") or "all",
    )


def _predictor_uniform(rows: Sequence[Mapping[str, Any]]) -> Callable[[Mapping[str, Any]], float]:
    return lambda row: 0.5


def _predictor_grand_mean(
    rows: Sequence[Mapping[str, Any]],
) -> Callable[[Mapping[str, Any]], float]:
    values = [value for value in (_human_x(row) for row in rows) if value is not None]
    mean_value = sum(values) / len(values) if values else 0.5
    return lambda row: mean_value


def _predictor_condition_mean(
    rows: Sequence[Mapping[str, Any]],
) -> Callable[[Mapping[str, Any]], float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = _human_x(row)
        if value is not None:
            grouped[_condition_key(row)].append(value)
    means = {
        key: sum(values) / len(values) for key, values in grouped.items() if values
    }
    overall = (
        sum(means.values()) / len(means) if means else 0.5
    )
    return lambda row: means.get(_condition_key(row), overall)


def _predictor_bayesian_hard(
    rows: Sequence[Mapping[str, Any]],
) -> Callable[[Mapping[str, Any]], Optional[float]]:
    return lambda row: _bayesian_probability_x(row, hard=True)


def _predictor_bayesian_soft(
    rows: Sequence[Mapping[str, Any]],
) -> Callable[[Mapping[str, Any]], Optional[float]]:
    return lambda row: _bayesian_probability_x(row, hard=False)


def _predictor_human_oracle(
    rows: Sequence[Mapping[str, Any]],
) -> Callable[[Mapping[str, Any]], Optional[float]]:
    return _human_x


REFERENCE_MODELS: Mapping[str, Mapping[str, Any]] = {
    "uniform_half": {
        "builder": _predictor_uniform,
        "oracle": False,
        "description": "Always 0.5; the no-information floor of the scale.",
    },
    "bayesian_hard": {
        "builder": _predictor_bayesian_hard,
        "oracle": False,
        "description": "Deterministic normative choice; a perfectly rational agent.",
    },
    "bayesian_soft": {
        "builder": _predictor_bayesian_soft,
        "oracle": False,
        "description": "Probability matching to the Bayesian posterior itself.",
    },
    "grand_mean_oracle": {
        "builder": _predictor_grand_mean,
        "oracle": True,
        "description": "One constant fitted to the evaluation set's own mean.",
    },
    "condition_mean_oracle": {
        "builder": _predictor_condition_mean,
        "oracle": True,
        "description": "Best predictor that knows only the experimental condition.",
    },
    "human_oracle": {
        "builder": _predictor_human_oracle,
        "oracle": True,
        "description": "Reproduces the observed rate exactly; equals the noise floor.",
    },
}


def score_reference_model(
    rows: Sequence[Mapping[str, Any]],
    name: str,
) -> Dict[str, Any]:
    if name not in REFERENCE_MODELS:
        raise ValueError("unknown reference model: {!r}".format(name))
    specification = REFERENCE_MODELS[name]
    predictor = specification["builder"](rows)
    scored: List[Dict[str, Any]] = []
    skipped = 0
    for row in rows:
        probability_x = predictor(row)
        if probability_x is None:
            skipped += 1
            continue
        probability_x = min(max(float(probability_x), 0.0), 1.0)
        metadata = row.get("metadata", {})
        agreeing_advisor_code = None
        code_to_advisor = metadata.get("code_to_advisor", {})
        advisor_name_to_type = metadata.get("advisor_name_to_type", {})
        for code, advisor_name in code_to_advisor.items():
            if advisor_name_to_type.get(advisor_name) == "agreeing":
                agreeing_advisor_code = code
                break
        scored.append(
            {
                "id": row["id"],
                "effect": row["effect"],
                "split": row["split"],
                "target_code": row.get("target_code"),
                "predicted_code": "X" if probability_x >= 0.5 else "Y",
                "log_probability_by_code": {"X": 0.0, "Y": 0.0},
                "probability_by_code": {"X": probability_x, "Y": 1.0 - probability_x},
                "pair_id": metadata.get("state_hash"),
                "bucket": metadata.get("bucket"),
                "scenario_id": metadata.get("scenario_id"),
                "authority_condition": metadata.get("authority_condition"),
                "indifference_scenario": metadata.get("indifference_scenario"),
                "medical_director_code": metadata.get("medical_director_code"),
                "reference_code": metadata.get("reference_code"),
                "agreeing_advisor_code": agreeing_advisor_code,
                "human_probability_by_code": row.get("human_probability_by_code"),
                "human_n": metadata.get("human_n") or metadata.get("human_denominator"),
            }
        )
    return {
        "schema_version": 2,
        "model_label": name,
        "base_model": None,
        "adapter": None,
        "reference_model": True,
        "oracle": specification["oracle"],
        "description": specification["description"],
        "skipped_rows": skipped,
        "summary": summarize_scored_rows(scored),
        "rows": scored,
    }


def scale_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Noise floor and trivial baselines for one evaluation set."""

    proportions = [value for value in (_human_x(row) for row in rows) if value is not None]
    counts = [
        int(
            row.get("metadata", {}).get("human_n")
            or row.get("metadata", {}).get("human_denominator")
            or 0
        )
        for row in rows
        if _human_x(row) is not None
    ]
    report: Dict[str, Any] = {
        "rows_with_human_data": len(proportions),
        "trivial_baselines": trivial_baselines(proportions) if proportions else {},
    }
    if proportions and all(count > 0 for count in counts):
        report["noise_floor_mae"] = binomial_noise_floor(proportions, counts)
    return report


def _dataset_argument(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=/path/to/file.jsonl")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_dataset_argument,
        required=True,
        help="Repeatable LABEL=/path/to/file.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(REFERENCE_MODELS),
        help="Restrict to specific reference models (default: all).",
    )
    args = parser.parse_args()

    names = args.model or sorted(REFERENCE_MODELS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview: Dict[str, Any] = {}

    for label, path in args.dataset:
        rows = load_jsonl(path)
        dataset_directory = args.output_dir / label
        dataset_directory.mkdir(parents=True, exist_ok=True)
        overview[label] = {"scale": scale_report(rows), "models": {}}
        for name in names:
            result = score_reference_model(rows, name)
            result["dataset"] = str(path.resolve())
            result["dataset_label"] = label
            (dataset_directory / "{}.json".format(name)).write_text(
                json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            calibration = result["summary"]["calibration"]
            overview[label]["models"][name] = {
                "oracle": result["oracle"],
                "rows": calibration["rows"],
                "mae": calibration["mae"],
                "cross_entropy": calibration["cross_entropy"],
                "accuracy": result["summary"]["normative"]["accuracy"],
            }

    (args.output_dir / "reference_overview.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(overview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
