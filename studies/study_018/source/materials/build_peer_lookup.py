#!/usr/bin/env python3
"""Compile the published LIONESS JavaScript lookup tables into package JSON."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ESTIMATE_MIN = 1
ESTIMATE_MAX = 150
CONDITIONS = {
    2: "LN",
    3: "HN",
    4: "HF",
    5: "HC",
    6: "filler",
}


def extract_array(path: Path, name: str) -> list[Any]:
    """Extract one JavaScript array literal without evaluating JavaScript."""

    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        rf"(?<![A-Za-z0-9_])(?:var\s+)?{re.escape(name)}\s*=\s*\[",
        text,
    )
    if match is None:
        raise ValueError(f"array {name!r} not found in {path}")

    start = text.find("[", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                value = ast.literal_eval(text[start : index + 1])
                if not isinstance(value, list):
                    raise TypeError(f"{name!r} in {path} is not a list")
                return value
    raise ValueError(f"unterminated array {name!r} in {path}")


def normalize_matrix(matrix: list[Any], expected_rows: int, name: str) -> dict[str, Any]:
    if len(matrix) != expected_rows:
        raise ValueError(f"{name} has {len(matrix)} rows; expected {expected_rows}")

    source_lengths: list[int] = []
    normalized: list[list[int]] = []
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) < ESTIMATE_MAX:
            raise ValueError(
                f"{name}[{row_index}] has "
                f"{len(row) if isinstance(row, list) else 'non-list'} values; "
                f"expected at least {ESTIMATE_MAX}"
            )
        source_lengths.append(len(row))
        values = [int(value) for value in row[:ESTIMATE_MAX]]
        if any(not ESTIMATE_MIN <= value <= ESTIMATE_MAX for value in values):
            raise ValueError(f"{name}[{row_index}] contains an out-of-range peer estimate")
        normalized.append(values)
    return {
        "rows": normalized,
        "source_row_lengths": source_lengths,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_lookup(source_dir: Path) -> dict[str, Any]:
    stage_main_setup = source_dir / "stage32741.php"
    stage_main_social = source_dir / "stage32743.php"
    stage_control_setup = source_dir / "stage32725.php"
    stage_control_social = source_dir / "stage32751.php"
    required = (
        stage_main_setup,
        stage_main_social,
        stage_control_setup,
        stage_control_social,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing LIONESS source files: {missing}")

    species = [str(value) for value in extract_array(stage_main_setup, "animal_names")]
    true_counts = [int(value) for value in extract_array(stage_main_setup, "numbers")]
    treatment_codes = [int(value) for value in extract_array(stage_main_setup, "treatments")]
    if not (len(species) == len(true_counts) == len(treatment_codes) == 30):
        raise ValueError("main task setup must define exactly 30 aligned rounds")

    main = {
        name: normalize_matrix(
            extract_array(stage_main_social, name),
            expected_rows=30,
            name=f"main.{name}",
        )
        for name in ("p1", "p2", "p3")
    }
    control = {
        name: normalize_matrix(
            extract_array(stage_control_social, name),
            expected_rows=5,
            name=f"control.{name}",
        )
        for name in ("p1", "p2", "p3")
    }
    anchor_pools = [
        [int(value) for value in extract_array(stage_control_setup, f"e{index}")]
        for index in range(1, 6)
    ]
    if any(len(pool) != 101 for pool in anchor_pools):
        raise ValueError("each four-peer control anchor pool must contain 101 values")

    control_source_rounds = [6, 2, 13, 4, 15]
    for control_index, main_round in enumerate(control_source_rounds):
        for peer_name in ("p1", "p2", "p3"):
            control_row = control[peer_name]["rows"][control_index]
            main_row = main[peer_name]["rows"][main_round - 1]
            if control_row != main_row:
                raise ValueError(
                    f"control row {control_index + 1} no longer matches "
                    f"main round {main_round} for {peer_name}"
                )

    rounds = [
        {
            "round": index + 1,
            "species": species[index],
            "true_count": true_counts[index],
            "treatment_code": treatment_codes[index],
            "condition": CONDITIONS[treatment_codes[index]],
        }
        for index in range(30)
    ]

    source_files = {
        path.name: {
            "sha256": sha256(path),
            "role": {
                "stage32741.php": "main round species, true counts, and treatment sequence",
                "stage32743.php": "main-task three-peer lookup tables",
                "stage32725.php": "four-peer control anchor pools",
                "stage32751.php": "four-peer control lookup tables",
            }[path.name],
        }
        for path in required
    }

    return {
        "schema_version": 1,
        "source": {
            "archive": "OSF LIONESS experiment software",
            "url": "https://osf.io/rmcuy/",
            "files": source_files,
        },
        "runtime_indexing": {
            "estimate_min": ESTIMATE_MIN,
            "estimate_max": ESTIMATE_MAX,
            "index_expression_in_original_code": "peer[round - 1][firstEstimate - 1]",
            "normalization": (
                "Only indices 0 through 149 are reachable for valid slider responses. "
                "Rows with a trailing 151st value are retained in source_row_lengths "
                "but compiled to the reachable 150-value domain."
            ),
        },
        "conditions": {
            str(code): condition for code, condition in CONDITIONS.items()
        },
        "main_task": {
            "rounds": rounds,
            "p1": main["p1"]["rows"],
            "p2": main["p2"]["rows"],
            "p3": main["p3"]["rows"],
            "source_row_lengths": {
                peer: main[peer]["source_row_lengths"] for peer in ("p1", "p2", "p3")
            },
        },
        "four_peer_control": {
            "rounds": [
                {
                    "round": index + 1,
                    "species_label": species[main_round - 1],
                    "emulated_main_round": main_round,
                    "condition": CONDITIONS[treatment_codes[main_round - 1]],
                }
                for index, main_round in enumerate(control_source_rounds)
            ],
            "anchor_pools": anchor_pools,
            "invalid_anchor_policy": (
                "The published pools contain two zero sentinels. Runtime sampling "
                "rejects values outside 1 through 150 because zero cannot index the "
                "published slider lookup and produced a known invalid historical row."
            ),
            "p1": control["p1"]["rows"],
            "p2": control["p2"]["rows"],
            "p3": control["p3"]["rows"],
            "source_row_lengths": {
                peer: control[peer]["source_row_lengths"]
                for peer in ("p1", "p2", "p3")
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path, help="Extracted OSF LIONESS directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("peer_lookup.json"),
    )
    args = parser.parse_args()

    payload = compile_lookup(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
