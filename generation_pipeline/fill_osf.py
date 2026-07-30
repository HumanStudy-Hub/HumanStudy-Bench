#!/usr/bin/env python3
"""Fill JSON slots from downloaded OSF materials."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation_pipeline.utils.osf_crawler import run_batch as fetch_osf
from generation_pipeline.utils.osf_fill import run_fill


def main():
    parser = argparse.ArgumentParser(description="Fill OSF materials into stage4 JSON")
    parser.add_argument("--stage4-dir", type=Path, default=Path("stage4"))
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fetch", action="store_true", help="Download OSF files first")
    parser.add_argument("--token", default=None, help="OSF_TOKEN for private projects")
    args = parser.parse_args()

    if args.fetch:
        print("Downloading OSF files...")
        plans = fetch_osf(
            args.stage4_dir,
            only=args.only,
            token=args.token,
            dry_run=False,
        )
        for p in plans:
            if p.errors:
                print(f"  ⚠ {p.paper_folder}: {p.errors[0]}")

    print("Filling JSON from OSF files...")
    results = run_fill(args.stage4_dir, only=args.only, dry_run=args.dry_run)
    for r in results:
        if r.get("skipped"):
            print(f"- {r['paper']}: skipped ({r['skipped']})")
            continue
        ch = r.get("changes", [])
        if ch:
            print(f"✓ {r['paper']}: {len(ch)} update(s)" + (" (dry-run)" if r.get("dry_run") else ""))
            for c in ch:
                print(f"    • {c}")
        else:
            print(f"- {r['paper']}: no slots filled")


if __name__ == "__main__":
    main()
