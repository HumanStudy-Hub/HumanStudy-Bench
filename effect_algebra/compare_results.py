"""Create a compact CSV and Markdown table from choice-evaluation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else "{:.4f}".format(value)


def _condition_mae(summary: Dict[str, Any], condition: str) -> Optional[float]:
    group = summary["calibration"]["by_authority_condition"].get(condition)
    return group["mae"] if group else None


def result_row(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    calibration = summary["calibration"]
    normative = summary["normative"]
    authority = summary["authority"]
    overshoot = summary.get("overshoot", {})
    advisor_agreement = summary.get("advisor_agreement", {})
    scale = calibration.get("scale", {})
    return {
        "model": payload["model_label"],
        "dataset": Path(payload["dataset"]).name,
        "rows": normative["rows"],
        # Primary metric: distance to the human distribution.
        "mae": calibration["mae"],
        "rmse": calibration["rmse"],
        "cross_entropy": calibration["cross_entropy"],
        "mae_noise_floor": (scale.get("noise_floor_mae") or {}).get("mean"),
        "mae_always_half": (scale.get("trivial_baselines") or {}).get("always_half"),
        "mae_opposes_private": _condition_mae(
            summary, "medical_director_opposes_private"
        ),
        "mae_supports_private": _condition_mae(
            summary, "medical_director_supports_private"
        ),
        "mae_indifference": calibration["indifference_subset"]["mae"],
        # Secondary: normative agreement, reported separately on purpose.
        "accuracy": normative["accuracy"],
        "x_rate": normative["decision_x_rate"],
        "authority_alignment": authority["hard_alignment_rate"],
        "human_authority_alignment": authority.get(
            "human_mean_alignment_probability"
        ),
        "advisor_agreement": advisor_agreement.get(
            "mean_agreeing_choice_probability"
        ),
        "dpo_unreachable_rate": overshoot.get("dpo_unreachable_rate"),
        "source": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    rows: List[Dict[str, Any]] = [result_row(path) for path in args.results]
    fieldnames = list(rows[0])
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "model",
        "dataset",
        "mae",
        "mae_noise_floor",
        "mae_opposes_private",
        "mae_indifference",
        "cross_entropy",
        "accuracy",
        "authority_alignment",
        "human_authority_alignment",
        "dpo_unreachable_rate",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[key]) if key in {"model", "dataset"} else _fmt(row[key])
                for key in headers
            )
            + " |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
