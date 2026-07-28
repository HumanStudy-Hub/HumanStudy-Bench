"""Emit the ordered Colab command list for the whole experiment.

Ordered so a budget that runs out early still leaves a coherent result. Each
stage carries a stop condition; running a later stage after its gate failed
produces a number that cannot be interpreted, which is worse than no number.

Stage 0 needs no GPU at all and should be run before opening Colab: it fixes the
calibration scale (noise floor, chance, and the perfectly-Bayesian reference)
that every later MAE is read against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def _stage(
    name: str,
    gate: str,
    purpose: str,
    commands: Sequence[str],
    *,
    gpu_hours: float,
    stop_condition: str = "",
    needs_gpu: bool = True,
) -> Dict[str, Any]:
    return {
        "stage": name,
        "gate": gate,
        "purpose": purpose,
        "needs_gpu": needs_gpu,
        "estimated_gpu_hours": gpu_hours,
        "stop_condition": stop_condition,
        "commands": list(commands),
    }


def build_plan(
    *,
    data: str,
    drive: str,
    base_model: str,
    folds: int,
    flywheel_conditions: Sequence[str],
) -> List[Dict[str, Any]]:
    evaluation_datasets = " ".join(
        "--dataset {label}={data}/eval/{file}".format(label=label, data=data, file=file)
        for label, file in (
            ("A_test", "A_test.jsonl"),
            ("B_test", "B_test.jsonl"),
            ("B_control", "B_no_feedback_control.jsonl"),
            ("C_test", "C_test.jsonl"),
            ("D_test", "D_test.jsonl"),
        )
    )

    stages: List[Dict[str, Any]] = [
        _stage(
            "build_data",
            "setup",
            "Generate and fail-closed validate every dataset.",
            [
                "python -m effect_algebra.build_datasets "
                "--repo-root . --output-dir {data}".format(data=data),
                "python -m effect_algebra.validate_datasets --data-dir {data}".format(
                    data=data
                ),
            ],
            gpu_hours=0.0,
            needs_gpu=False,
            stop_condition="Stop unless the validator reports errors: 0.",
        ),
        _stage(
            "reference_models",
            "gate_0",
            "Fix the calibration scale before spending GPU time: noise floor, "
            "chance, and how close a perfectly Bayesian agent already is.",
            [
                "python -m effect_algebra.reference_models {datasets} "
                "--output-dir {drive}/results/reference".format(
                    datasets=evaluation_datasets, drive=drive
                ),
            ],
            gpu_hours=0.0,
            needs_gpu=False,
            stop_condition=(
                "If bayesian_hard MAE is already within the noise floor, the "
                "normative model matches humans and there is no gap to close."
            ),
        ),
        _stage(
            "base_profile",
            "gate_0",
            "Base-model calibration on all evaluation sets, plus the overshoot "
            "diagnostic that decides between the soft-label and DPO objectives.",
            [
                "python -m effect_algebra.evaluate_suite "
                "--model-label base --base-model {base} {datasets} "
                "--output-dir {drive}/results/base".format(
                    base=base_model, datasets=evaluation_datasets, drive=drive
                ),
                "python -m effect_algebra.knowledge_probe "
                "--model-label base --base-model {base} "
                "--output {drive}/results/base/knowledge_probe.json".format(
                    base=base_model, drive=drive
                ),
                "python -m effect_algebra.plot_calibration "
                "--result base={drive}/results/base/C_test.json "
                "--noise-floor 0.035 --output {drive}/results/base/calibration_C.svg "
                "--csv {drive}/results/base/calibration_C.csv".format(drive=drive),
            ],
            gpu_hours=0.5,
            stop_condition=(
                "Stop if base MAE is already below 0.10 everywhere (no gap), or "
                "if the knowledge probe shows the model cannot state the "
                "findings (the framing, not the model, is wrong). "
                "If dpo_unreachable_rate is above 0.5, the pairwise objective "
                "is structurally unusable; keep soft-label as the main method."
            ),
        ),
        _stage(
            "gate_1_ceiling",
            "gate_1",
            "In-domain C training, cross-validated so the ceiling is measured on "
            "all 40 scenarios, the same set Gate 0 and Gate 2 score.",
            [
                "python -m effect_algebra.train_soft "
                "--train-file {data}/cv/C_fold{fold}_train.jsonl "
                "--eval-file {data}/cv/C_fold{fold}_test.jsonl "
                "--output-dir {drive}/adapters/C_fold{fold} "
                "--run-name c-fold{fold} --base-model {base}".format(
                    data=data, drive=drive, base=base_model, fold=fold
                )
                for fold in range(folds)
            ]
            + [
                "python -m effect_algebra.evaluate_suite "
                "--model-label C_fold{fold} --base-model {base} "
                "--adapter {drive}/adapters/C_fold{fold} "
                "--dataset C_fold{fold}_test={data}/cv/C_fold{fold}_test.jsonl "
                "--output-dir {drive}/results/C_fold{fold}".format(
                    data=data, drive=drive, base=base_model, fold=fold
                )
                for fold in range(folds)
            ],
            gpu_hours=2.0,
            stop_condition=(
                "Sweep the learning rate on fold 0 only, then report the ceiling "
                "as the mean over folds 1..n so the selection fold is excluded. "
                "If the ceiling does not beat base MAE by more than the noise "
                "floor, calibration training does not work; do not test transfer."
            ),
        ),
        _stage(
            "gate_2_transfer",
            "gate_2",
            "One joint adapter on A+B, evaluated zero-shot on C.",
            [
                "python -m effect_algebra.train_soft "
                "--train-file {data}/dpo/AB_train.jsonl "
                "--eval-file {data}/dpo/AB_dev.jsonl "
                "--output-dir {drive}/adapters/AB_soft "
                "--run-name ab-soft --base-model {base}".format(
                    data=data, drive=drive, base=base_model
                ),
                "python -m effect_algebra.evaluate_suite "
                "--model-label AB_soft --base-model {base} "
                "--adapter {drive}/adapters/AB_soft {datasets} "
                "--output-dir {drive}/results/AB_soft".format(
                    base=base_model, datasets=evaluation_datasets, drive=drive
                ),
            ],
            gpu_hours=1.5,
            stop_condition=(
                "Check A-only on A_test and B-only on B_test improved first. "
                "Without that, A+B has no verified basis and C cannot be read."
            ),
        ),
        _stage(
            "ablation_single_effect",
            "ablation",
            "Which source paradigm carries the transfer.",
            [
                "python -m effect_algebra.train_soft "
                "--train-file {data}/dpo/{effect}_train.jsonl "
                "--eval-file {data}/dpo/{effect}_dev.jsonl "
                "--output-dir {drive}/adapters/{effect}_soft "
                "--run-name {lower}-soft --base-model {base}".format(
                    data=data,
                    drive=drive,
                    base=base_model,
                    effect=effect,
                    lower=effect.lower(),
                )
                for effect in ("A", "B")
            ]
            + [
                "python -m effect_algebra.evaluate_suite "
                "--model-label {effect}_soft --base-model {base} "
                "--adapter {drive}/adapters/{effect}_soft {datasets} "
                "--output-dir {drive}/results/{effect}_soft".format(
                    base=base_model, datasets=evaluation_datasets, drive=drive,
                    effect=effect,
                )
                for effect in ("A", "B")
            ],
            gpu_hours=1.5,
        ),
        _stage(
            "ablation_dpo_objective",
            "ablation",
            "Pairwise DPO under the same data, to show empirically whether the "
            "direction blindness predicted by the Gate 0 diagnostic occurs.",
            [
                "python -m effect_algebra.train_dpo "
                "--train-file {data}/dpo/AB_train.jsonl "
                "--eval-file {data}/dpo/AB_dev.jsonl "
                "--output-dir {drive}/adapters/AB_dpo_beta{tag} "
                "--run-name ab-dpo-beta{tag} --beta {beta} "
                "--base-model {base}".format(
                    data=data,
                    drive=drive,
                    base=base_model,
                    beta=beta,
                    tag=str(beta).replace(".", "p"),
                )
                for beta in (0.1, 0.3, 1.0)
            ]
            + [
                "python -m effect_algebra.evaluate_suite "
                "--model-label AB_dpo_beta{tag} --base-model {base} "
                "--adapter {drive}/adapters/AB_dpo_beta{tag} {datasets} "
                "--output-dir {drive}/results/AB_dpo_beta{tag}".format(
                    base=base_model,
                    datasets=evaluation_datasets,
                    drive=drive,
                    tag=str(beta).replace(".", "p"),
                )
                for beta in (0.1, 0.3, 1.0)
            ],
            gpu_hours=3.0,
        ),
        _stage(
            "flywheel",
            "flywheel",
            "Does zero-shot calibration improve as sourced human data is added, "
            "and is the driver diversity or volume? Diversity conditions all "
            "train on the same number of rows so the two cannot be confused.",
            [
                "python -m effect_algebra.flywheel --repo-root . "
                "--output-dir {data} --tier core --tier extended "
                "--base-model {base} --drive-root {drive}".format(
                    data=data, base=base_model, drive=drive
                ),
            ]
            + [
                "python -m effect_algebra.train_soft "
                "--train-file {data}/flywheel/{name}_train.jsonl "
                "--eval-file {data}/eval/C_test.jsonl "
                "--output-dir {drive}/adapters/flywheel_{name} "
                "--run-name flywheel-{name} --base-model {base}".format(
                    data=data, drive=drive, base=base_model, name=name
                )
                for name in flywheel_conditions
            ]
            + [
                "python -m effect_algebra.evaluate_suite "
                "--model-label flywheel_{name} --base-model {base} "
                "--adapter {drive}/adapters/flywheel_{name} {datasets} "
                "--output-dir {drive}/results/flywheel_{name}".format(
                    base=base_model, datasets=evaluation_datasets, drive=drive,
                    name=name,
                )
                for name in flywheel_conditions
            ],
            gpu_hours=4.0,
            stop_condition=(
                "Read the diversity conditions and the volume series together. "
                "A flat volume curve with a rising diversity curve means the "
                "flywheel should add papers, not rows."
            ),
        ),
        _stage(
            "report",
            "report",
            "Assemble the comparison table and the calibration figure.",
            [
                "python -m effect_algebra.compare_results "
                "{drive}/results/*/C_test.json "
                "--csv {drive}/results/summary.csv "
                "--markdown {drive}/results/summary.md".format(drive=drive),
                "python -m effect_algebra.plot_calibration "
                "--result base={drive}/results/base/C_test.json "
                "--result transfer={drive}/results/AB_soft/C_test.json "
                "--noise-floor 0.035 "
                "--output {drive}/results/calibration_C.svg".format(drive=drive),
            ],
            gpu_hours=0.0,
            needs_gpu=False,
        ),
    ]
    return stages


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="/content/drive/MyDrive/effect_algebra_ab_c/data",
        help="Where build_datasets wrote the dataset tree.",
    )
    parser.add_argument(
        "--drive",
        default="/content/drive/MyDrive/effect_algebra_ab_c",
        help="Persistent root for adapters and results.",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--format",
        default="shell",
        choices=("shell", "json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stage",
        action="append",
        help="Restrict output to specific stages.",
    )
    return parser


def render_shell(stages: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by effect_algebra.run_plan. Run one stage per Colab session,",
        "# restart the runtime between training runs to release GPU memory, and",
        "# read each stop condition before starting the next stage.",
        "set -euo pipefail",
        "",
    ]
    total = 0.0
    for stage in stages:
        total += float(stage["estimated_gpu_hours"])
        lines.append("# " + "=" * 68)
        lines.append("# {}  [{}]".format(stage["stage"], stage["gate"]))
        lines.append("# {}".format(stage["purpose"]))
        lines.append(
            "# GPU: {}   estimated {:.1f} GPU-hours".format(
                "yes" if stage["needs_gpu"] else "no (run before opening Colab)",
                stage["estimated_gpu_hours"],
            )
        )
        if stage["stop_condition"]:
            lines.append("# STOP CONDITION: {}".format(stage["stop_condition"]))
        lines.append("# " + "=" * 68)
        lines.extend(stage["commands"])
        lines.append("")
    lines.append("# Total estimated: {:.1f} GPU-hours".format(total))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parser().parse_args()
    from .flywheel import TIERS

    conditions: List[str] = list(TIERS["core"]) + list(TIERS["extended"])
    conditions += [
        "volume_{:03d}pct".format(int(round(fraction * 100)))
        for fraction in (0.125, 0.25, 0.5)
    ]
    stages = build_plan(
        data=args.data,
        drive=args.drive,
        base_model=args.base_model,
        folds=args.folds,
        flywheel_conditions=conditions,
    )
    if args.stage:
        wanted = set(args.stage)
        stages = [stage for stage in stages if stage["stage"] in wanted]

    rendered = (
        json.dumps(stages, indent=2, ensure_ascii=False)
        if args.format == "json"
        else render_shell(stages)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
