#!/usr/bin/env python3
"""
Download OSF supplementary materials for stage4 papers whose experiments
reference materials not present in the paper (osf_only / not_in_paper / cited_scale).

Usage:
    # Preview which papers need OSF materials
    python generation_pipeline/fetch_osf.py --list

    # Dry-run download plan for one paper
    python generation_pipeline/fetch_osf.py --only TheBenefit --dry-run

    # Download all OSF files for papers with OSF links (includes all-linked by default)
    python generation_pipeline/fetch_osf.py

    # Only papers still missing materials in JSON
    python generation_pipeline/fetch_osf.py --missing-only

Private / view-only projects:
    export OSF_TOKEN=...   # optional bearer token for authenticated projects
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation_pipeline.utils.osf_crawler import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OSF materials for stage4 corpus")
    parser.add_argument(
        "--stage4-dir",
        type=Path,
        default=Path("stage4"),
        help="Root directory containing per-paper JSON folders (default: stage4)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Only process papers whose path/folder matches substring (repeatable)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List papers with missing OSF-referenced materials; do not download",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve OSF and show matched files without writing to disk",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only fetch papers with missing materials in JSON (legacy behavior)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("OSF_TOKEN"),
        help="OSF API bearer token (or set OSF_TOKEN env var)",
    )
    args = parser.parse_args()

    if not args.stage4_dir.exists():
        raise SystemExit(f"stage4 dir not found: {args.stage4_dir}")

    results = run_batch(
        args.stage4_dir,
        only=args.only,
        token=args.token,
        download_all=True,
        dry_run=args.dry_run,
        list_only=args.list,
        all_linked=not args.missing_only,
        missing_only=args.missing_only,
    )

    if args.list:
        if not results:
            print("No papers with OSF link found.")
            return
        label = "missing materials" if args.missing_only else "OSF link"
        print(f"Found {len(results)} paper(s) with {label}:\n")
        for plan in results:
            print(f"- {plan.paper_folder}")
            print(f"  title: {plan.paper_title[:80]}")
            print(f"  osf:   {plan.osf_url or '(no link in JSON)'}")
            files_dir = args.stage4_dir / plan.paper_folder / "osf" / "files"
            n_dl = sum(1 for _ in files_dir.rglob("*") if _.is_file()) if files_dir.exists() else 0
            print(f"  missing slots: {len(plan.effects_needing_osf)} | osf/files on disk: {n_dl}")
            for need in plan.effects_needing_osf[:3]:
                print(f"    • {need.study} [{need.slot}={need.status}] IV={need.iv[:50]}")
            if len(plan.effects_needing_osf) > 3:
                print(f"    • ... +{len(plan.effects_needing_osf) - 3} more")
            print()
        return

    ok, err = 0, 0
    for plan in results:
        if plan.errors and not plan.downloaded_files and not plan.skipped_files:
            err += 1
            print(f"✗ {plan.paper_folder}: {'; '.join(plan.errors)}")
            continue
        ok += 1
        action = "would download" if args.dry_run else "downloaded"
        n = len(plan.skipped_files) if args.dry_run else len(plan.downloaded_files)
        print(f"✓ {plan.paper_folder}: {action} {n} file(s)")
        files = plan.skipped_files if args.dry_run else plan.downloaded_files
        for path in files[:8]:
            print(f"    {path}")
        if len(files) > 8:
            print(f"    ... +{len(files) - 8} more")
        for msg in plan.errors:
            print(f"  ⚠ {msg}")

    print(f"\nDone: {ok} paper(s) processed, {err} error(s).")


if __name__ == "__main__":
    main()
