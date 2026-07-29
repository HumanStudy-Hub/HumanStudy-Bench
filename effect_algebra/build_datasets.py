"""CLI for building all A+B->C datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .datasets import (
    SCHEMA_VERSION,
    a_bucket_coverage,
    build_a_rows,
    build_b_control_rows,
    build_b_rows,
    build_c_rows,
    build_c_training_rows,
    build_d_rows,
    build_d_training_rows,
    c_stratified_folds,
    d_stratified_holdout,
    load_d_scenarios,
    interleave_rows,
    load_c_scenarios,
    write_jsonl,
)
from .human_priors import binomial_noise_floor, trivial_baselines
from .validate_datasets import validate_dataset_tree


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("effect_algebra/data"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--a-train", type=int, default=512)
    parser.add_argument("--a-dev", type=int, default=128)
    parser.add_argument("--a-test", type=int, default=256)
    parser.add_argument("--b-train", type=int, default=1024)
    parser.add_argument("--b-dev", type=int, default=256)
    parser.add_argument("--b-test", type=int, default=256)
    parser.add_argument("--b-control", type=int, default=256)
    parser.add_argument("--rounds-per-advisor", type=int, default=15)
    parser.add_argument(
        "--b-probe-rounds",
        default="3,5,15",
        help="Ledger lengths to emit as B readability probes; empty to skip.",
    )
    parser.add_argument("--b-probe-count", type=int, default=128)
    parser.add_argument(
        "--d-replicas",
        type=int,
        default=28,
        help="Replicas per D scenario; 28 x 18 = 504 rows, within 2%% of A_train.",
    )
    parser.add_argument("--d-holdout", type=int, default=6)
    parser.add_argument("--c-folds", type=int, default=5)
    parser.add_argument("--c-replicas", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--skip-validation", action="store_true")
    return parser


def build_tree(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir.resolve()
    repo_root = args.repo_root.resolve()
    scenarios_path = (
        repo_root
        / "extended_study"
        / "study_019"
        / "source"
        / "materials"
        / "scenarios.json"
    )
    if not scenarios_path.exists():
        raise FileNotFoundError("study_019 scenarios not found: {}".format(scenarios_path))

    a_train = build_a_rows("train", args.a_train, args.seed + 1000)
    a_dev = build_a_rows("dev", args.a_dev, args.seed + 2000)
    a_test = build_a_rows("test", args.a_test, args.seed + 3000)
    b_train = build_b_rows(
        "train",
        args.b_train,
        args.seed + 4000,
        rounds_per_advisor=args.rounds_per_advisor,
    )
    b_dev = build_b_rows(
        "dev",
        args.b_dev,
        args.seed + 5000,
        rounds_per_advisor=args.rounds_per_advisor,
    )
    # Evaluation sets carry both letter assignments of every item. Scoring an
    # item under only one assignment confounds the content signal with the
    # model's preference for a letter, which on the base model is large enough
    # to dominate the calibration metric outright.
    b_test = build_b_rows(
        "test",
        args.b_test,
        args.seed + 6000,
        rounds_per_advisor=args.rounds_per_advisor,
    ) + build_b_rows(
        "test",
        args.b_test,
        args.seed + 6000,
        rounds_per_advisor=args.rounds_per_advisor,
        mirror=True,
    )
    b_control = build_b_control_rows(
        args.b_control,
        args.seed + 7000,
        rounds_per_advisor=args.rounds_per_advisor,
    ) + build_b_control_rows(
        args.b_control,
        args.seed + 7000,
        rounds_per_advisor=args.rounds_per_advisor,
        mirror=True,
    )
    c_test = build_c_rows(scenarios_path) + build_c_rows(scenarios_path, mirror=True)
    d_test = build_d_rows(scenarios_path) + build_d_rows(scenarios_path, mirror=True)
    # D as a transfer source. Six scenarios are held out so a D-trained adapter
    # can be checked for generalization within D; D has no wording variation, so
    # unlike A there is no held-out-wording split to use instead. 28 replicas on
    # the remaining 18 scenarios put the row count within 2% of A_train, so an
    # A-versus-D comparison is not confounded by dataset size.
    d_scenarios = load_d_scenarios(scenarios_path)
    d_source_ids, d_holdout_ids = d_stratified_holdout(
        d_scenarios, holdout=args.d_holdout, seed=args.seed
    )
    d_train = build_d_training_rows(
        scenarios_path,
        d_source_ids,
        replicas=args.d_replicas,
        seed=args.seed + 8000,
    )
    d_holdout = build_d_rows(scenarios_path, scenario_ids=d_holdout_ids) + build_d_rows(
        scenarios_path, scenario_ids=d_holdout_ids, mirror=True
    )

    file_rows = {
        "A_train": ("dpo/A_train.jsonl", a_train),
        "A_dev": ("dpo/A_dev.jsonl", a_dev),
        "A_test": ("eval/A_test.jsonl", a_test),
        "B_train": ("dpo/B_train.jsonl", b_train),
        "B_dev": ("dpo/B_dev.jsonl", b_dev),
        "B_test": ("eval/B_test.jsonl", b_test),
        "AB_train": ("dpo/AB_train.jsonl", interleave_rows(a_train, b_train)),
        "AB_dev": ("dpo/AB_dev.jsonl", interleave_rows(a_dev, b_dev)),
        "D_train": ("dpo/D_train.jsonl", d_train),
        # Scenarios D_train never saw, so a D-trained adapter can be checked for
        # generalization within D. eval/D_test.jsonl keeps all 24 scenarios and
        # stays the right file for models not trained on D.
        "D_heldout": ("eval/D_heldout.jsonl", d_holdout),
        # Plain union, so it answers "given both sources, is it better?".
        # The matched-budget question of whether diversity beats volume is the
        # flywheel's job and is built separately.
        "AD_train": ("dpo/AD_train.jsonl", interleave_rows(a_train, d_train)),
        "B_no_feedback_control": ("eval/B_no_feedback_control.jsonl", b_control),
        "C_test": ("eval/C_test.jsonl", c_test),
        "D_test": ("eval/D_test.jsonl", d_test),
    }

    # B readability probes. The base model scored at chance on the 15-round
    # ledger, which can mean either that it cannot weigh advisors at all or
    # that it cannot do the arithmetic over that many rounds in one forward
    # pass. Scoring the same mechanism at several ledger lengths separates the
    # two: if accuracy rises as the ledger shortens, the mechanism is intact and
    # the length is the binding constraint.
    for rounds in [
        int(value) for value in str(args.b_probe_rounds).split(",") if value.strip()
    ]:
        file_rows["B_probe_r{}".format(rounds)] = (
            "eval/B_probe_r{}.jsonl".format(rounds),
            build_b_rows(
                "test",
                args.b_probe_count,
                args.seed + 9000 + rounds,
                rounds_per_advisor=rounds,
            )
            + build_b_rows(
                "test",
                args.b_probe_count,
                args.seed + 9000 + rounds,
                rounds_per_advisor=rounds,
                mirror=True,
            ),
        )

    # Gate 1 cross-validation. Every scenario is held out in exactly one fold,
    # so the ceiling is measured on all 40 scenarios, the same set Gate 0 and
    # Gate 2 score.
    scenarios = load_c_scenarios(scenarios_path)
    folds = c_stratified_folds(scenarios, folds=args.c_folds, seed=args.seed)
    all_scenario_ids = [str(scenario["scenario_id"]) for scenario in scenarios]
    for fold_index, held_out in enumerate(folds):
        train_ids = [key for key in all_scenario_ids if key not in set(held_out)]
        file_rows["C_cv_fold{}_train".format(fold_index)] = (
            "cv/C_fold{}_train.jsonl".format(fold_index),
            build_c_training_rows(
                scenarios_path,
                train_ids,
                replicas=args.c_replicas,
                seed=args.seed + fold_index,
                split="train",
                fold=fold_index,
            ),
        )
        file_rows["C_cv_fold{}_test".format(fold_index)] = (
            "cv/C_fold{}_test.jsonl".format(fold_index),
            build_c_rows(scenarios_path, scenario_ids=held_out)
            + build_c_rows(scenarios_path, scenario_ids=held_out, mirror=True),
        )

    files: Dict[str, Dict[str, Any]] = {}
    for name, (relative_path, rows) in file_rows.items():
        files[name] = write_jsonl(output_dir / relative_path, rows)
        files[name]["relative_path"] = relative_path

    human_rates = [
        float(scenario["human_raw_data"]["option_1_rate"]) for scenario in scenarios
    ]
    human_counts = [
        int(scenario["human_raw_data"]["n_choice"]) for scenario in scenarios
    ]

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "rounds_per_advisor": args.rounds_per_advisor,
        "sources": {
            "A": "extended_study/study_016 symmetric baseline; Anderson & Holt bucket proportions",
            "B": "extended_study/study_017 Dates Task 3C; published 0.17 agreeing pick rate",
            "B_control": "extended_study/study_017 Dates Task 3B; scored, never trained",
            "C": str(scenarios_path.relative_to(repo_root)),
            "D": "study_019 Study 1; per-scenario rates, no authority manipulation",
            "D_train": "study_019 Study 1, 18 of 24 scenarios; transfer source for Gate 2b",
            "D_heldout": "study_019 Study 1, the 6 scenarios D_train never sees",
        },
        "label_policy": {
            "objective": "match published human response proportions",
            "a_bucket_coverage": a_bucket_coverage(),
            "d_split": {
                "source_scenarios": len(d_source_ids),
                "holdout_scenarios": d_holdout_ids,
                "replicas_per_scenario": args.d_replicas,
            },
            "c_folds": {
                "count": args.c_folds,
                "replicas_per_scenario": args.c_replicas,
                "held_out_by_fold": {
                    str(index): fold for index, fold in enumerate(folds)
                },
            },
        },
        "calibration_scale": {
            "c_noise_floor_mae": binomial_noise_floor(human_rates, human_counts),
            "c_trivial_baselines": trivial_baselines(human_rates),
        },
        "training_boundary": {
            "trainable_effects_outside_cv": ["A", "B"],
            "never_trainable": ["B_control", "eval/C_test.jsonl"],
            "c_test_used_for_training": False,
        },
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    if not args.skip_validation:
        report = validate_dataset_tree(output_dir)
        if not report["valid"]:
            raise ValueError(
                "generated datasets failed validation:\n{}".format(
                    "\n".join(report["errors"])
                )
            )
        manifest["validation"] = report["summary"]
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return manifest


def main() -> None:
    args = _parser().parse_args()
    manifest = build_tree(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "files": {
                    name: {"count": details["count"], "sha256": details["sha256"]}
                    for name, details in manifest["files"].items()
                },
                "validation": manifest.get("validation"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
