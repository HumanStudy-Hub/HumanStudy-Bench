"""Create a compact CSV and Markdown table from choice-evaluation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else "{:.4f}".format(value)


def result_row(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    overall = summary["overall"]
    human = summary["human_distribution"]
    authority = summary["authority"]
    advisor_agreement = summary.get("advisor_agreement", {})
    return {
        "model": payload["model_label"],
        "dataset": Path(payload["dataset"]).name,
        "rows": overall["rows"],
        "accuracy": overall["accuracy"],
        "target_probability": overall["mean_target_probability"],
        "preference_margin": overall["mean_preference_margin"],
        "x_rate": overall["decision_x_rate"],
        "human_probability_mae": human["weighted_probability_mae"],
        "authority_alignment": authority["hard_alignment_rate"],
        "advisor_agreement": advisor_agreement.get(
            "mean_agreeing_choice_probability"
        ),
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
        "accuracy",
        "target_probability",
        "preference_margin",
        "human_probability_mae",
        "authority_alignment",
        "advisor_agreement",
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
