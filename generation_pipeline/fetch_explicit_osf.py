"""Fetch one user-supplied OSF project into a pipeline job directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from generation_pipeline.connectors.base_connector import SourceFetchPlan
from generation_pipeline.connectors.osf_connector import OsfConnector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--paper-title", default="")
    parser.add_argument("--token")
    args = parser.parse_args()

    connector = OsfConnector(token=args.token, download_all=True)
    plan = SourceFetchPlan(
        paper_folder=args.dest.parent.name,
        json_path=None,
        paper_title=args.paper_title,
        link=args.url,
        connector="osf",
    )
    fetched = connector.fetch(plan, args.dest)
    print(f"Fetched {len(fetched)} OSF file(s) into {args.dest}")


if __name__ == "__main__":
    main()
