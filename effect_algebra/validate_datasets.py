"""Fail-closed validation for the A+B->C calibration datasets.

Two properties matter and both are checked here rather than trusted:

* **Labels reproduce the published human proportions.** Every row names the
  bucket it came from and the proportion that bucket carries, and the empirical
  label split inside each bucket is recomputed and compared.
* **Nothing trains on an evaluation set.** The guard is the per-row `trainable`
  flag plus the directory the row lives in, not the effect name. Gate 1 has to
  train on C scenarios, so an effect-level ban would either block the ceiling
  or be quietly disabled; a split-level guard survives that.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .datasets import a_bucket, a_normative_choice, completion_code, load_jsonl
from .human_priors import A_PRIORS, b_prior


# Directory contract: what may be trained on lives in dpo/ or cv/*_train.jsonl,
# and nothing else may carry a preference pair.
TRAINABLE_DIRECTORIES = {"dpo"}
EVAL_DIRECTORIES = {"eval"}

# A bucket's empirical label split is a rounded ratio over a finite number of
# rows, so it cannot match the published proportion exactly. The tolerance below
# covers rounding on groups as small as ~20 rows.
BUCKET_PROPORTION_TOLERANCE = 0.06
MINIMUM_BUCKET_ROWS_FOR_PROPORTION_CHECK = 20


def _error(errors: List[str], path: Path, row_id: str, message: str) -> None:
    errors.append("{} [{}]: {}".format(path, row_id, message))


def _relative_role(path: Path, root: Optional[Path]) -> str:
    """Classify a file as trainable, eval, or cross-validation."""

    parent = path.parent.name
    if parent in TRAINABLE_DIRECTORIES:
        return "trainable"
    if parent in EVAL_DIRECTORIES:
        return "eval"
    if parent == "cv":
        return "cv_train" if path.name.endswith("_train.jsonl") else "cv_eval"
    return "unknown"


def _validate_human_distribution(
    path: Path,
    row: Mapping[str, Any],
    errors: List[str],
) -> None:
    row_id = str(row.get("id", "<missing-id>"))
    distribution = row.get("human_probability_by_code")
    if distribution is None:
        metadata = row.get("metadata", {})
        if metadata.get("calibrated"):
            _error(
                errors,
                path,
                row_id,
                "row is marked calibrated but carries no human distribution",
            )
        return
    if not isinstance(distribution, dict) or set(distribution) != {"X", "Y"}:
        _error(errors, path, row_id, "human distribution must cover exactly X and Y")
        return
    try:
        total = sum(float(value) for value in distribution.values())
    except (TypeError, ValueError):
        _error(errors, path, row_id, "human distribution contains a non-numeric value")
        return
    if abs(total - 1.0) > 1e-9:
        _error(errors, path, row_id, "human distribution does not sum to one")
    for value in distribution.values():
        if not 0.0 <= float(value) <= 1.0:
            _error(errors, path, row_id, "human probability outside [0, 1]")
            break
    reference_code = row.get("metadata", {}).get("reference_code")
    reference_probability = row.get("metadata", {}).get("human_probability")
    if reference_code in {"X", "Y"} and reference_probability is not None:
        if abs(float(distribution[reference_code]) - float(reference_probability)) > 1e-9:
            _error(
                errors,
                path,
                row_id,
                "human distribution disagrees with metadata.human_probability",
            )


def _validate_common(
    path: Path,
    row: Mapping[str, Any],
    errors: List[str],
    *,
    role: str,
) -> None:
    row_id = str(row.get("id", "<missing-id>"))
    if row.get("schema_version") != 2:
        _error(errors, path, row_id, "schema_version must be 2")
    if row.get("effect") not in {"A", "B", "B_control", "C", "D"}:
        _error(errors, path, row_id, "unknown effect")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 2:
        _error(errors, path, row_id, "prompt must contain system and user messages")
    else:
        if [message.get("role") for message in prompt] != ["system", "user"]:
            _error(errors, path, row_id, "prompt roles must be system,user")
        if not all(str(message.get("content", "")).strip() for message in prompt):
            _error(errors, path, row_id, "prompt messages cannot be empty")
    if row.get("options") != {"X": "DECISION=X", "Y": "DECISION=Y"}:
        _error(errors, path, row_id, "options must be randomized X/Y response codes")

    trainable = row.get("trainable")
    if not isinstance(trainable, bool):
        _error(errors, path, row_id, "trainable must be a boolean")
        trainable = False
    if role in {"trainable", "cv_train"} and not trainable:
        _error(errors, path, row_id, "row in a training file is not trainable")
    if role in {"eval", "cv_eval"} and trainable:
        _error(errors, path, row_id, "row in an evaluation file is marked trainable")

    has_pair = "chosen" in row or "rejected" in row
    target_code = row.get("target_code")
    if not trainable:
        if has_pair:
            _error(
                errors,
                path,
                row_id,
                "non-trainable row carries a preference pair",
            )
    elif target_code is None:
        if has_pair:
            _error(errors, path, row_id, "unlabeled row cannot contain a preference pair")
    else:
        if target_code not in {"X", "Y"}:
            _error(errors, path, row_id, "target_code must be X, Y, or null")
        try:
            chosen_code = completion_code(row["chosen"])
            rejected_code = completion_code(row["rejected"])
            if chosen_code != target_code:
                _error(errors, path, row_id, "chosen completion disagrees with target_code")
            if rejected_code == chosen_code:
                _error(errors, path, row_id, "chosen and rejected completions are identical")
        except (KeyError, TypeError, ValueError) as exc:
            _error(errors, path, row_id, "invalid preference completion: {}".format(exc))

    _validate_human_distribution(path, row, errors)


def _validate_a(path: Path, row: Mapping[str, Any], errors: List[str]) -> None:
    row_id = str(row["id"])
    metadata = row.get("metadata", {})
    try:
        choice, posterior_a, public_prior_a = a_normative_choice(
            metadata["prior_choices"],
            metadata["private_signal"],
            float(metadata["accuracy"]),
        )
        if abs(float(metadata["posterior_a"]) - posterior_a) > 1e-9:
            _error(errors, path, row_id, "A posterior_a does not match recomputation")
        if abs(float(metadata["public_prior_a"]) - public_prior_a) > 1e-9:
            _error(errors, path, row_id, "A public prior does not match recomputation")

        expected_bucket = a_bucket(
            metadata["prior_choices"],
            metadata["private_signal"],
            posterior_a,
            public_prior_a,
            float(metadata["accuracy"]),
        )
        if metadata.get("bucket") != expected_bucket:
            _error(errors, path, row_id, "A bucket does not match recomputation")

        # The reference behaviour is the source-grounded choice. The *label* may
        # deviate from it, because a share of humans deviated; what must hold is
        # that the reference itself is computed, not guessed.
        code_to_choice = metadata["code_to_choice"]
        reference_code = str(metadata["reference_code"])
        if code_to_choice[reference_code] != choice:
            _error(errors, path, row_id, "A reference code does not map to the source choice")
        if metadata.get("reference_semantic") != choice:
            _error(errors, path, row_id, "A reference semantic does not match recomputation")

        prior = A_PRIORS[expected_bucket]
        recorded = metadata.get("human_probability")
        if prior.probability is None:
            if recorded is not None:
                _error(errors, path, row_id, "uncalibrated A bucket carries a proportion")
        elif recorded is None or abs(float(recorded) - prior.probability) > 1e-9:
            _error(
                errors,
                path,
                row_id,
                "A human proportion does not match the published bucket value",
            )
    except (KeyError, TypeError, ValueError) as exc:
        _error(errors, path, row_id, "invalid A metadata: {}".format(exc))


def _validate_b(path: Path, row: Mapping[str, Any], errors: List[str]) -> None:
    row_id = str(row["id"])
    metadata = row.get("metadata", {})
    try:
        ledger = metadata["ledger"]
        rounds = int(metadata["rounds_per_advisor"])
        if not metadata.get("complete_episode"):
            _error(errors, path, row_id, "B episode is not marked complete")
        if len(ledger) != 2 * rounds:
            _error(
                errors,
                path,
                row_id,
                "B episode has {} rows; expected {}".format(len(ledger), 2 * rounds),
            )
        counts = Counter(record["advisor_type"] for record in ledger)
        if counts != Counter({"accurate": rounds, "agreeing": rounds}):
            _error(errors, path, row_id, "B advisor round counts are unbalanced")
        if not all(record.get("correct_year") is not None for record in ledger):
            _error(errors, path, row_id, "feedback B row is missing revealed outcomes")
        errors_by_type = {}
        for advisor_type in ("accurate", "agreeing"):
            errors_by_type[advisor_type] = mean(
                abs(int(record["advice_year"]) - int(record["correct_year"]))
                for record in ledger
                if record["advisor_type"] == advisor_type
            )
        if errors_by_type["accurate"] >= errors_by_type["agreeing"]:
            _error(errors, path, row_id, "accurate advisor is not empirically more accurate")
        recorded = metadata["mean_absolute_error"]
        for advisor_type in ("accurate", "agreeing"):
            if abs(float(recorded[advisor_type]) - errors_by_type[advisor_type]) > 1e-9:
                _error(errors, path, row_id, "B error summary does not match ledger")

        code_to_advisor = metadata["code_to_advisor"]
        reference_advisor = code_to_advisor[str(metadata["reference_code"])]
        if metadata["advisor_name_to_type"][reference_advisor] != "accurate":
            _error(errors, path, row_id, "B reference code does not map to the accurate advisor")

        prior = b_prior(str(metadata.get("source_experiment", "3C")))
        if abs(float(metadata["human_probability"]) - float(prior.probability)) > 1e-9:
            _error(
                errors,
                path,
                row_id,
                "B human proportion does not match the published pick rate",
            )

        user_prompt = str(row["prompt"][1]["content"])
        for record in ledger:
            expected = "Round {:02d}".format(int(record["round"]))
            if expected not in user_prompt:
                _error(errors, path, row_id, "B prompt omits {}".format(expected))
                break
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        _error(errors, path, row_id, "invalid B metadata: {}".format(exc))


def _validate_b_control(path: Path, row: Mapping[str, Any], errors: List[str]) -> None:
    row_id = str(row["id"])
    metadata = row.get("metadata", {})
    ledger = metadata.get("ledger", [])
    rounds = int(metadata.get("rounds_per_advisor", 0) or 0)
    if row.get("target_code") is not None:
        _error(errors, path, row_id, "B no-feedback control must remain unlabeled")
    if row.get("trainable"):
        _error(errors, path, row_id, "B no-feedback control must never be trainable")
    if len(ledger) != 2 * rounds:
        _error(errors, path, row_id, "B control does not contain a complete episode")
    if any(record.get("correct_year") is not None for record in ledger):
        _error(errors, path, row_id, "B no-feedback control leaks correct outcomes")


def _validate_c(path: Path, row: Mapping[str, Any], errors: List[str]) -> None:
    row_id = str(row["id"])
    metadata = row.get("metadata", {})
    try:
        code_to_diagnosis = metadata["code_to_diagnosis"]
        bayesian_choice = metadata["bayesian_choice"]
        bayesian_code = metadata["bayesian_code"]
        if bayesian_choice is None:
            if bayesian_code is not None:
                _error(errors, path, row_id, "C indifference case carries a Bayesian code")
        elif code_to_diagnosis[str(bayesian_code)] != bayesian_choice:
            _error(errors, path, row_id, "C Bayesian code does not map to the source choice")

        reference_code = str(metadata["reference_code"])
        if code_to_diagnosis[reference_code] != metadata["human_option_1"]:
            _error(errors, path, row_id, "C reference code does not map to the scored option")
        if abs(
            float(metadata["human_probability"])
            - float(metadata["human_option_1_rate"])
        ) > 1e-9:
            _error(errors, path, row_id, "C human proportion does not match the raw rate")
        if not str(metadata.get("source_material_fingerprint", "")):
            _error(errors, path, row_id, "C source fingerprint is missing")

        if not row.get("trainable") and row.get("target_code") != bayesian_code:
            _error(
                errors,
                path,
                row_id,
                "C evaluation row must expose the source Bayesian choice as target_code",
            )
    except (KeyError, TypeError, ValueError) as exc:
        _error(errors, path, row_id, "invalid C metadata: {}".format(exc))


def _validate_d(path: Path, row: Mapping[str, Any], errors: List[str]) -> None:
    row_id = str(row["id"])
    metadata = row.get("metadata", {})
    try:
        code_to_urn = metadata["code_to_urn"]
        reference_code = str(metadata["reference_code"])
        if code_to_urn[reference_code] != metadata["human_option_1"]:
            _error(errors, path, row_id, "D reference code does not map to the scored urn")
        if abs(
            float(metadata["human_probability"])
            - float(metadata["human_option_1_rate"])
        ) > 1e-9:
            _error(errors, path, row_id, "D human proportion does not match the raw rate")
        bayesian_code = metadata["bayesian_code"]
        if metadata["bayesian_choice"] is None:
            if bayesian_code is not None:
                _error(errors, path, row_id, "D indifference case carries a Bayesian code")
        elif code_to_urn[str(bayesian_code)] != metadata["bayesian_choice"]:
            _error(errors, path, row_id, "D Bayesian code does not map to the source choice")
        if not str(metadata.get("source_material_fingerprint", "")):
            _error(errors, path, row_id, "D source fingerprint is missing")
        if not row.get("trainable") and row.get("target_code") != bayesian_code:
            _error(
                errors,
                path,
                row_id,
                "D evaluation row must expose the source Bayesian choice as target_code",
            )
    except (KeyError, TypeError, ValueError) as exc:
        _error(errors, path, row_id, "invalid D metadata: {}".format(exc))


def _check_bucket_proportions(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    errors: List[str],
) -> Dict[str, Any]:
    """Recompute the label split inside each bucket against its human value."""

    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        metadata = row.get("metadata", {})
        # Evaluation rows expose the source-grounded choice as `target_code` by
        # design, so their split is not a proportional label and must not be
        # compared against the human proportion.
        if not row.get("trainable"):
            continue
        if row.get("target_code") not in {"X", "Y"}:
            continue
        if metadata.get("human_probability") is None:
            continue
        # Group by the key the label assignment actually used. For A and B that
        # is the bucket; for C it is the scenario, because C carries a distinct
        # proportion per scenario rather than per stratum.
        label_group = metadata.get("label_group") or metadata.get("bucket")
        groups.setdefault(
            "{}|{}".format(row.get("effect"), label_group), []
        ).append(row)

    report: Dict[str, Any] = {}
    for key, group in sorted(groups.items()):
        target_probability = float(group[0]["metadata"]["human_probability"])
        observed = sum(
            1.0
            for row in group
            if row["target_code"] == row["metadata"]["reference_code"]
        ) / len(group)
        report[key] = {
            "rows": len(group),
            "human_probability": target_probability,
            "empirical_reference_rate": observed,
        }
        if len(group) < MINIMUM_BUCKET_ROWS_FOR_PROPORTION_CHECK:
            continue
        if abs(observed - target_probability) > BUCKET_PROPORTION_TOLERANCE:
            errors.append(
                "{}: bucket {} label split is {:.3f}, expected {:.3f} "
                "(tolerance {:.2f})".format(
                    path,
                    key,
                    observed,
                    target_probability,
                    BUCKET_PROPORTION_TOLERANCE,
                )
            )
    return report


def validate_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    role: Optional[str] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    ids: Set[str] = set()
    prompt_groups: Dict[str, List[Mapping[str, Any]]] = {}
    targets: Counter = Counter()
    effects: Counter = Counter()
    if role is None:
        role = _relative_role(path, None)

    for row in rows:
        row_id = str(row.get("id", "<missing-id>"))
        if row_id in ids:
            _error(errors, path, row_id, "duplicate row id")
        ids.add(row_id)
        _validate_common(path, row, errors, role=role)
        effect = row.get("effect")
        effects[effect] += 1
        if row.get("target_code") is not None:
            targets[row["target_code"]] += 1
        prompt_hash = json.dumps(row.get("prompt"), sort_keys=True, ensure_ascii=False)
        prompt_groups.setdefault(prompt_hash, []).append(row)

        if effect == "A":
            _validate_a(path, row, errors)
        elif effect == "B":
            _validate_b(path, row, errors)
        elif effect == "B_control":
            _validate_b_control(path, row, errors)
        elif effect == "C":
            _validate_c(path, row, errors)
        elif effect == "D":
            _validate_d(path, row, errors)

    # Repeated prompts are legitimate only where a per-scenario proportion needs
    # several rows to express. Everywhere else a duplicate is padded scale.
    for prompt_hash, group in prompt_groups.items():
        if len(group) == 1:
            continue
        replicated = [row for row in group if "replica" in row.get("metadata", {})]
        if len(replicated) != len(group):
            _error(
                errors,
                path,
                str(group[0].get("id", "<missing-id>")),
                "duplicate prompt that is not a declared proportion replica",
            )
            continue
        distributions = {
            json.dumps(row.get("human_probability_by_code"), sort_keys=True)
            for row in group
        }
        if len(distributions) != 1:
            _error(
                errors,
                path,
                str(group[0]["id"]),
                "replicas of one prompt disagree on the human distribution",
            )

    bucket_report = _check_bucket_proportions(path, rows, errors)

    labeled_count = sum(targets.values())
    if labeled_count >= 20:
        x_rate = targets["X"] / labeled_count
        if not 0.45 <= x_rate <= 0.55:
            errors.append(
                "{}: response-code balance is {:.3f}, outside [0.45, 0.55]".format(
                    path,
                    x_rate,
                )
            )
    return {
        "valid": not errors,
        "errors": errors,
        "count": len(rows),
        "role": role,
        "effects": dict(effects),
        "target_codes": dict(targets),
        "buckets": bucket_report,
        "ids": ids,
        "prompt_hashes": set(prompt_groups),
    }


def _jsonl_files(root: Path) -> Iterable[Path]:
    return (
        sorted(root.glob("dpo/*.jsonl"))
        + sorted(root.glob("eval/*.jsonl"))
        + sorted(root.glob("cv/*.jsonl"))
    )


def _scenario_ids(rows: Sequence[Mapping[str, Any]]) -> Set[str]:
    return {
        str(row["metadata"]["scenario_id"])
        for row in rows
        if "scenario_id" in row.get("metadata", {})
    }


def _validate_cv_folds(root: Path, errors: List[str]) -> Dict[str, Any]:
    """Each fold must train and test on disjoint scenarios."""

    report: Dict[str, Any] = {}
    train_files = sorted(root.glob("cv/*_train.jsonl"))
    for train_path in train_files:
        test_path = train_path.with_name(
            train_path.name.replace("_train.jsonl", "_test.jsonl")
        )
        if not test_path.exists():
            errors.append("{}: fold has no matching test file".format(train_path))
            continue
        train_ids = _scenario_ids(load_jsonl(train_path))
        test_ids = _scenario_ids(load_jsonl(test_path))
        overlap = train_ids & test_ids
        if overlap:
            errors.append(
                "{}: {} scenarios appear in both the fold train and test sets".format(
                    train_path,
                    len(overlap),
                )
            )
        report[train_path.name] = {
            "train_scenarios": len(train_ids),
            "test_scenarios": len(test_ids),
            "overlap": len(overlap),
        }
    return report


def validate_dataset_tree(root: Path) -> Dict[str, Any]:
    root = Path(root)
    errors: List[str] = []
    reports: Dict[str, Dict[str, Any]] = {}
    split_prompts: Dict[str, Set[str]] = {}
    split_states: Dict[str, Set[str]] = {}
    all_ids: Set[str] = set()

    for path in _jsonl_files(root):
        rows = load_jsonl(path)
        role = _relative_role(path, root)
        report = validate_rows(path, rows, role=role)
        errors.extend(report["errors"])
        relative = str(path.relative_to(root))
        reports[relative] = {
            key: value
            for key, value in report.items()
            if key not in {"errors", "ids", "prompt_hashes"}
        }
        is_joint_view = path.name.startswith("AB_")
        # Folds deliberately reuse scenarios across the cross-validation, so
        # they are excluded from the global split comparison and checked
        # per fold instead.
        is_cross_validation = path.parent.name == "cv"
        for row in rows:
            row_id = str(row["id"])
            if not is_joint_view and not is_cross_validation and row_id in all_ids:
                errors.append(
                    "{} [{}]: row id also appears in another non-joint file".format(
                        path, row_id
                    )
                )
            if not is_joint_view and not is_cross_validation:
                all_ids.add(row_id)
            if is_joint_view or is_cross_validation:
                continue
            split = str(row["split"])
            split_prompts.setdefault(split, set()).add(
                json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False)
            )
            split_states.setdefault(split, set()).add(
                str(row.get("metadata", {}).get("state_hash", ""))
            )

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_prompts.get(left, set()) & split_prompts.get(right, set())
        if overlap:
            errors.append(
                "prompt leakage between {} and {}: {} exact prompts".format(
                    left,
                    right,
                    len(overlap),
                )
            )
        state_overlap = (
            split_states.get(left, set()) & split_states.get(right, set())
        ) - {""}
        if state_overlap:
            errors.append(
                "state leakage between {} and {}: {} state hashes".format(
                    left,
                    right,
                    len(state_overlap),
                )
            )

    c_paths = [path for path in _jsonl_files(root) if path.name == "C_test.jsonl"]
    if len(c_paths) != 1:
        errors.append("exactly one C_test.jsonl is required")
    else:
        c_rows = load_jsonl(c_paths[0])
        scenarios = {
            str(row["metadata"]["scenario_id"]) for row in c_rows
        }
        if len(scenarios) != 40:
            errors.append("C_test.jsonl must cover the 40 study_019 Study 2 scenarios")
        # Each scenario appears once per letter assignment so the letter bias
        # can be averaged out at scoring time.
        if len(c_rows) != 2 * len(scenarios):
            errors.append(
                "C_test.jsonl must contain both letter assignments of every scenario"
            )
        if any(row.get("trainable") for row in c_rows):
            errors.append("C_test.jsonl contains a trainable row")

    fold_report = _validate_cv_folds(root, errors)

    trainable_rows = sum(
        1
        for path in _jsonl_files(root)
        for row in load_jsonl(path)
        if row.get("trainable")
    )
    return {
        "valid": not errors,
        "errors": errors,
        "summary": {
            "files": len(reports),
            "rows": sum(report["count"] for report in reports.values()),
            "trainable_rows": trainable_rows,
            "errors": len(errors),
            "c_test_used_for_training": False,
            "cv_folds": fold_report,
        },
        "files": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("effect_algebra/data"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_dataset_tree(args.data_dir)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
