"""Data-flywheel experiment: does more sourced human data improve transfer?

The pipeline turns papers into calibration environments, so the question that
decides whether the flywheel is worth turning is whether zero-shot calibration
on an unseen paradigm improves as environments are added. Two axes, deliberately
separated:

* **Diversity** - how many distinct paradigms are in the pool, at a *fixed*
  total row budget. Without the fixed budget, {A+B} has twice the rows of {A}
  and any improvement is unattributable: more paradigms and more data are
  confounded. Every diversity condition here trains on the same number of rows.
* **Volume** - fractions of the full pool with the paradigm set held fixed.
  Together with the diversity axis this separates "more data" from "more kinds
  of data", and shows where the curve saturates.

Subsampling preserves, exactly rather than in expectation, both the human
proportion inside every label group and the X/Y response-code balance: it
samples within each (label group, response code, label side) cell. A subset that
drifted off the human proportion would change the training target, which is the
one thing that must stay fixed while data volume varies.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .datasets import (
    build_a_rows,
    build_b_rows,
    build_d_training_rows,
    interleave_rows,
    write_jsonl,
)
from .validate_datasets import validate_rows


# Diversity conditions. C never appears: it is the held-out transfer target.
DIVERSITY_CONDITIONS: Mapping[str, Tuple[str, ...]] = {
    "A_only": ("A",),
    "B_only": ("B",),
    "D_only": ("D",),
    "A_plus_B": ("A", "B"),
    "A_plus_D": ("A", "D"),
    "B_plus_D": ("B", "D"),
    "A_plus_B_plus_D": ("A", "B", "D"),
}

VOLUME_FRACTIONS: Tuple[float, ...] = (0.125, 0.25, 0.5, 1.0)

# Ordered so that a truncated budget still answers the most important question
# first: which single paradigm carries the transfer, and whether combining helps.
TIERS: Mapping[str, Tuple[str, ...]] = {
    "core": ("A_only", "B_only", "A_plus_B"),
    "extended": ("D_only", "A_plus_B_plus_D"),
    "full": ("A_plus_D", "B_plus_D"),
}


def _cell_key(row: Mapping[str, Any]) -> Tuple[str, str, str, bool]:
    metadata = row.get("metadata", {})
    return (
        str(row.get("effect")),
        str(metadata.get("label_group") or metadata.get("bucket")),
        str(metadata.get("reference_code")),
        bool(row.get("target_code") == metadata.get("reference_code")),
    )


def subsample_preserving_proportions(
    rows: Sequence[Mapping[str, Any]],
    target_count: int,
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    """Take `target_count` rows without moving the human proportion.

    Sampling uniformly at random would leave the label split correct only in
    expectation, and at small budgets the drift is large enough to change what
    the model is being asked to match. Sampling the same fraction out of every
    (label group, response code, label side) cell keeps the proportion and the
    code balance intact by construction.
    """

    if target_count < 0:
        raise ValueError("target_count cannot be negative")
    if target_count >= len(rows):
        return [dict(row) for row in rows]

    cells: Dict[Tuple[str, str, str, bool], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[_cell_key(row)].append(row)

    fraction = target_count / len(rows)
    selected: List[Dict[str, Any]] = []
    remainders: List[Tuple[float, Tuple[str, str, str, bool], int]] = []
    for key, group in sorted(cells.items(), key=lambda item: str(item[0])):
        ordered = list(group)
        random.Random("flywheel|{}|{}".format(seed, key)).shuffle(ordered)
        exact = len(ordered) * fraction
        take = int(exact)
        selected.extend(dict(row) for row in ordered[:take])
        if take < len(ordered):
            remainders.append((exact - take, key, take))
        cells[key] = ordered

    # Hand out the leftover slots to the cells with the largest fractional part,
    # so the rounding error is spread rather than concentrated in one bucket.
    remainders.sort(key=lambda item: (-item[0], str(item[1])))
    shortfall = target_count - len(selected)
    for _, key, taken in remainders:
        if shortfall <= 0:
            break
        selected.append(dict(cells[key][taken]))
        shortfall -= 1
    return selected


def build_pool(
    repo_root: Path,
    *,
    a_count: int,
    b_count: int,
    d_replicas: int,
    rounds_per_advisor: int,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Trainable rows for every source paradigm, keyed by effect."""

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
    return {
        "A": build_a_rows("train", a_count, seed + 1000),
        "B": build_b_rows(
            "train",
            b_count,
            seed + 4000,
            rounds_per_advisor=rounds_per_advisor,
        ),
        "D": build_d_training_rows(
            scenarios_path,
            replicas=d_replicas,
            seed=seed + 8000,
        ),
    }


def build_conditions(
    pool: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    row_budget: int,
    conditions: Sequence[str],
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """One matched-budget training set per diversity condition."""

    built: Dict[str, Dict[str, Any]] = {}
    for name in conditions:
        effects = DIVERSITY_CONDITIONS[name]
        # Split the fixed budget evenly across the paradigms in the condition,
        # so the comparison is about which paradigms are present, not how many
        # rows each contributes.
        per_effect = row_budget // len(effects)
        groups: List[List[Dict[str, Any]]] = []
        for effect in effects:
            available = pool[effect]
            if len(available) < per_effect:
                raise ValueError(
                    "condition {} needs {} rows of effect {} but the pool has {}; "
                    "lower --row-budget or raise the generator counts".format(
                        name,
                        per_effect,
                        effect,
                        len(available),
                    )
                )
            groups.append(
                subsample_preserving_proportions(available, per_effect, seed=seed)
            )
        built[name] = {
            "effects": list(effects),
            "rows": interleave_rows(*groups),
            "rows_per_effect": per_effect,
        }
    return built


def build_volume_series(
    pool: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    row_budget: int,
    effects: Sequence[str],
    fractions: Sequence[float],
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """The scaling curve: one training set per fraction of a fixed pool."""

    series: Dict[str, Dict[str, Any]] = {}
    for fraction in fractions:
        target = max(int(round(row_budget * fraction)), len(effects))
        per_effect = target // len(effects)
        groups = [
            subsample_preserving_proportions(pool[effect], per_effect, seed=seed)
            for effect in effects
        ]
        series["volume_{:03d}pct".format(int(round(fraction * 100)))] = {
            "effects": list(effects),
            "fraction": fraction,
            "rows": interleave_rows(*groups),
            "rows_per_effect": per_effect,
        }
    return series


def _run_plan(
    output_dir: Path,
    names: Sequence[str],
    *,
    drive_root: str,
    base_model: str,
) -> List[Dict[str, Any]]:
    """Colab commands, one training run and one evaluation per condition."""

    plan: List[Dict[str, Any]] = []
    for name in names:
        adapter = "{}/adapters/flywheel_{}".format(drive_root, name)
        plan.append(
            {
                "condition": name,
                "train": (
                    "python -m effect_algebra.train_soft "
                    "--train-file {data}/flywheel/{name}_train.jsonl "
                    "--eval-file {data}/eval/C_test.jsonl "
                    "--output-dir {adapter} "
                    "--run-name flywheel-{name} "
                    "--base-model {base}"
                ).format(
                    data=str(output_dir),
                    name=name,
                    adapter=adapter,
                    base=base_model,
                ),
                "evaluate": (
                    "python -m effect_algebra.evaluate_suite "
                    "--model-label flywheel_{name} "
                    "--base-model {base} "
                    "--adapter {adapter} "
                    "--dataset C_test={data}/eval/C_test.jsonl "
                    "--dataset D_test={data}/eval/D_test.jsonl "
                    "--dataset A_test={data}/eval/A_test.jsonl "
                    "--dataset B_test={data}/eval/B_test.jsonl "
                    "--output-dir {drive}/results/flywheel_{name}"
                ).format(
                    name=name,
                    base=base_model,
                    adapter=adapter,
                    data=str(output_dir),
                    drive=drive_root,
                ),
            }
        )
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("effect_algebra/data"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--a-count", type=int, default=512)
    parser.add_argument("--b-count", type=int, default=1024)
    parser.add_argument("--d-replicas", type=int, default=20)
    parser.add_argument("--rounds-per-advisor", type=int, default=15)
    parser.add_argument(
        "--row-budget",
        type=int,
        default=480,
        help="Total training rows every diversity condition gets.",
    )
    parser.add_argument(
        "--tier",
        action="append",
        choices=sorted(TIERS),
        help="Which tiers of diversity conditions to build (default: core).",
    )
    parser.add_argument(
        "--volume-effects",
        default="A,B,D",
        help="Comma-separated effects held fixed while volume varies.",
    )
    parser.add_argument("--skip-volume-series", action="store_true")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/effect_algebra_ab_c",
        help="Where the emitted run plan writes adapters and results.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    tiers = args.tier or ["core"]
    conditions: List[str] = []
    for tier in tiers:
        conditions.extend(name for name in TIERS[tier] if name not in conditions)

    pool = build_pool(
        args.repo_root.resolve(),
        a_count=args.a_count,
        b_count=args.b_count,
        d_replicas=args.d_replicas,
        rounds_per_advisor=args.rounds_per_advisor,
        seed=args.seed,
    )
    built = build_conditions(
        pool,
        row_budget=args.row_budget,
        conditions=conditions,
        seed=args.seed,
    )
    if not args.skip_volume_series:
        volume_effects = [
            effect.strip() for effect in args.volume_effects.split(",") if effect.strip()
        ]
        built.update(
            build_volume_series(
                pool,
                row_budget=args.row_budget,
                effects=volume_effects,
                fractions=VOLUME_FRACTIONS,
                seed=args.seed,
            )
        )

    files: Dict[str, Any] = {}
    errors: List[str] = []
    for name, payload in sorted(built.items()):
        relative = "flywheel/{}_train.jsonl".format(name)
        path = output_dir / relative
        record = write_jsonl(path, payload["rows"])
        record["relative_path"] = relative
        record["effects"] = payload["effects"]
        record["rows_per_effect"] = payload["rows_per_effect"]
        if "fraction" in payload:
            record["fraction"] = payload["fraction"]
        report = validate_rows(path, payload["rows"], role="trainable")
        record["valid"] = report["valid"]
        record["buckets"] = report["buckets"]
        errors.extend(report["errors"])
        files[name] = record

    manifest = {
        "experiment": "data flywheel",
        "question": (
            "does zero-shot calibration on an unseen paradigm improve as more "
            "sourced human data is added, and is it diversity or volume?"
        ),
        "held_out_target": "C (study_019 Study 2 medical authority)",
        "row_budget": args.row_budget,
        "matched_budget": True,
        "seed": args.seed,
        "pool_sizes": {effect: len(rows) for effect, rows in sorted(pool.items())},
        "conditions": files,
        "run_plan": _run_plan(
            output_dir,
            sorted(built),
            drive_root=args.drive_root,
            base_model=args.base_model,
        ),
        "valid": not errors,
        "errors": errors,
    }
    manifest_path = output_dir / "flywheel" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(
            "flywheel subsets failed validation:\n{}".format("\n".join(errors[:20]))
        )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir / "flywheel"),
                "conditions": {
                    name: {
                        "count": record["count"],
                        "effects": record["effects"],
                    }
                    for name, record in sorted(files.items())
                },
                "runs": len(manifest["run_plan"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
