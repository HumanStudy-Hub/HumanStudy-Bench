"""Fail-closed validation for A+B->C preference and evaluation data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from .datasets import a_normative_choice, completion_code, load_jsonl


TRAIN_FILES = {
    "A_train.jsonl",
    "A_dev.jsonl",
    "B_train.jsonl",
    "B_dev.jsonl",
    "AB_train.jsonl",
    "AB_dev.jsonl",
}


def _error(errors: List[str], path: Path, row_id: str, message: str) -> None:
    errors.append("{} [{}]: {}".format(path, row_id, message))


def _validate_common(path: Path, row: Mapping[str, Any], errors: List[str]) -> None:
    row_id = str(row.get("id", "<missing-id>"))
    if row.get("schema_version") != 1:
        _error(errors, path, row_id, "schema_version must be 1")
    if row.get("effect") not in {"A", "B", "B_control", "C"}:
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

    target_code = row.get("target_code")
    if target_code is None:
        if "chosen" in row or "rejected" in row:
            _error(errors, path, row_id, "unlabeled controls cannot contain preference pairs")
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


def _validate_a(path: Path, row: Mapping[str, Any], errors: List[str]) -> None:
    row_id = str(row["id"])
    metadata = row.get("metadata", {})
    try:
        choice, posterior_a, public_prior_a = a_normative_choice(
            metadata["prior_choices"],
            metadata["private_signal"],
            float(metadata["accuracy"]),
        )
        code_to_choice = metadata["code_to_choice"]
        target_code = str(row["target_code"])
        if code_to_choice[target_code] != choice:
            _error(errors, path, row_id, "A label does not match recomputed Bayesian choice")
        if abs(float(metadata["posterior_a"]) - posterior_a) > 1e-9:
            _error(errors, path, row_id, "A posterior_a does not match recomputation")
        if abs(float(metadata["public_prior_a"]) - public_prior_a) > 1e-9:
            _error(errors, path, row_id, "A public prior does not match recomputation")
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
        target_advisor = code_to_advisor[str(row["target_code"])]
        if metadata["advisor_name_to_type"][target_advisor] != "accurate":
            _error(errors, path, row_id, "B chosen code does not map to accurate advisor")
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
    if len(ledger) != 2 * rounds:
        _error(errors, path, row_id, "B control does not contain a complete episode")
    if any(record.get("correct_year") is not None for record in ledger):
        _error(errors, path, row_id, "B no-feedback control leaks correct outcomes")


def _validate_c(path: Path, row: Mapping[str, Any], errors: List[str]) -> None:
    row_id = str(row["id"])
    metadata = row.get("metadata", {})
    try:
        target_code = row["target_code"]
        if target_code is None:
            if metadata["bayesian_choice"] is not None:
                _error(errors, path, row_id, "C null target is not a source indifference case")
        elif metadata["code_to_diagnosis"][str(target_code)] != metadata["bayesian_choice"]:
            _error(errors, path, row_id, "C target is not the source Bayesian choice")
        probabilities = metadata["human_probability_by_code"]
        if set(probabilities) != {"X", "Y"}:
            _error(errors, path, row_id, "C human probabilities must cover X and Y")
        if abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1e-9:
            _error(errors, path, row_id, "C human probabilities do not sum to one")
        if not str(metadata.get("source_material_fingerprint", "")):
            _error(errors, path, row_id, "C source fingerprint is missing")
    except (KeyError, TypeError, ValueError) as exc:
        _error(errors, path, row_id, "invalid C metadata: {}".format(exc))


def validate_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    ids: Set[str] = set()
    prompt_hashes: Set[str] = set()
    targets: Counter = Counter()
    effects: Counter = Counter()

    for row in rows:
        row_id = str(row.get("id", "<missing-id>"))
        if row_id in ids:
            _error(errors, path, row_id, "duplicate row id")
        ids.add(row_id)
        _validate_common(path, row, errors)
        effect = row.get("effect")
        effects[effect] += 1
        if row.get("target_code") is not None:
            targets[row["target_code"]] += 1
        prompt_payload = row.get("prompt")
        prompt_hash = json.dumps(prompt_payload, sort_keys=True, ensure_ascii=False)
        if prompt_hash in prompt_hashes:
            _error(errors, path, row_id, "duplicate prompt within file")
        prompt_hashes.add(prompt_hash)

        if effect == "A":
            _validate_a(path, row, errors)
        elif effect == "B":
            _validate_b(path, row, errors)
        elif effect == "B_control":
            _validate_b_control(path, row, errors)
        elif effect == "C":
            _validate_c(path, row, errors)

        if path.name in TRAIN_FILES and effect not in {"A", "B"}:
            _error(errors, path, row_id, "held-out effect appears in a training file")

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
        "effects": dict(effects),
        "target_codes": dict(targets),
        "ids": ids,
        "prompt_hashes": prompt_hashes,
    }


def _jsonl_files(root: Path) -> Iterable[Path]:
    return sorted(root.glob("dpo/*.jsonl")) + sorted(root.glob("eval/*.jsonl"))


def validate_dataset_tree(root: Path) -> Dict[str, Any]:
    root = Path(root)
    errors: List[str] = []
    reports: Dict[str, Dict[str, Any]] = {}
    split_prompts: Dict[str, Set[str]] = {}
    split_states: Dict[str, Set[str]] = {}
    all_ids: Set[str] = set()

    for path in _jsonl_files(root):
        rows = load_jsonl(path)
        report = validate_rows(path, rows)
        errors.extend(report["errors"])
        relative = str(path.relative_to(root))
        reports[relative] = {
            key: value
            for key, value in report.items()
            if key not in {"errors", "ids", "prompt_hashes"}
        }
        is_joint_view = path.name.startswith("AB_")
        for row in rows:
            row_id = str(row["id"])
            if not is_joint_view and row_id in all_ids:
                errors.append(
                    "{} [{}]: row id also appears in another non-joint file".format(
                        path, row_id
                    )
                )
            if not is_joint_view:
                all_ids.add(row_id)
            split = str(row["split"])
            if is_joint_view:
                continue
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
    elif len(load_jsonl(c_paths[0])) != 40:
        errors.append("C_test.jsonl must contain the 40 study_019 Study 2 scenarios")

    return {
        "valid": not errors,
        "errors": errors,
        "summary": {
            "files": len(reports),
            "rows": sum(report["count"] for report in reports.values()),
            "errors": len(errors),
            "C_used_for_training": False,
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
