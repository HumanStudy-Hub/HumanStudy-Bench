#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    job = json.loads((args.job / "job.json").read_text())
    contract = args.contract.read_text()
    osf = job.get("osfUrl")
    external_rule = (
        f"External material was explicitly supplied: `{osf}`. You may access this URL and links directly contained in its materials."
        if osf
        else "No external URL was supplied. Network research is not authorized: do not search, browse, fetch websites, resolve the DOI, or discover OSF materials. Use only the uploaded PDF."
    )
    prompt = f"""{contract}

## Current job

- Job directory: `{args.job.resolve()}`
- Paper: `{(args.job / 'input/paper.pdf').resolve()}`
- Original filename: `{job.get('paperName', 'paper.pdf')}`
- Contributor: `{job.get('contributorName', 'Unknown')}`
- External-source policy: {external_rule}

Complete the full extraction and package build now. Write all deliverables under
`{(args.job / 'package').resolve()}`. Do not modify files outside the job directory.
"""
    args.output.write_text(prompt)


if __name__ == "__main__":
    main()
