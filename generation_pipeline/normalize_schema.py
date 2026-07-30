#!/usr/bin/env python3
"""Normalize historical stage4 JSON schema and write a repair report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation_pipeline.verification.schema_validator import normalize_stage4_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan stage4/**/*.json with schema_validator")
    parser.add_argument("--stage4-dir", type=Path, default=Path("stage4"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("stage4/schema_normalization_report.json"),
        help="Where to write the normalization report",
    )
    parser.add_argument("--write", action="store_true", help="Write deterministic repairs back to paper JSONs")
    parser.add_argument("--no-backup", action="store_true", help="Skip .bak backups when --write is used")
    args = parser.parse_args()

    report = normalize_stage4_tree(
        args.stage4_dir,
        write=args.write,
        backup=not args.no_backup,
        report_path=args.report,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Report: {args.report}")
    raise SystemExit(1 if report["summary"]["invalid"] else 0)


if __name__ == "__main__":
    main()
