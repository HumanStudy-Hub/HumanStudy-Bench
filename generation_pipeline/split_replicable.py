#!/usr/bin/env python3
"""
Split stage4 corpus into runnable / non_runnable based on material completeness for agent replication.

Criterion (per effect, all three slots must have non-empty content):
  - materials.content
  - manipulation.content
  - items.content (cited_scale with <80 chars counts as incomplete — scale items missing)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

RUNNABLE = "runnable"
NOT_RUNNABLE = "non_runnable"
MIN_ITEMS_CHARS = 80


def analyze(json_path: Path) -> tuple[bool, list[dict], int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    issues: list[dict] = []
    n_effects = 0
    for study in data.get("eligible_studies", []):
        sname = study.get("study", "")
        for ei, eff in enumerate(study.get("effects", [])):
            n_effects += 1
            for slot in ("materials", "manipulation", "items"):
                obj = eff.get(slot) or {}
                st = obj.get("status")
                c = obj.get("content")
                has = c is not None and str(c).strip() != ""
                problem = None
                if not has:
                    problem = st or "empty"
                elif slot == "items" and st == "cited_scale" and len(str(c).strip()) < MIN_ITEMS_CHARS:
                    problem = "cited_scale_partial"
                if problem:
                    issues.append(
                        {
                            "study": sname,
                            "effect_index": ei,
                            "slot": slot,
                            "status": problem,
                            "IV": eff.get("IV", ""),
                            "DV": eff.get("DV", ""),
                        }
                    )
    return (len(issues) == 0 and n_effects > 0, issues, n_effects)


def split(stage4_dir: Path, dry_run: bool = False) -> dict:
    runnable_dir = stage4_dir / RUNNABLE
    not_dir = stage4_dir / NOT_RUNNABLE
    skip_names = {RUNNABLE, NOT_RUNNABLE, "replicable", "non_replicable", "能跑", "不能跑"}

    report = {"runnable": [], "not_runnable": [], "moved": []}

    paper_dirs: list[Path] = []
    split_dirs = {RUNNABLE, NOT_RUNNABLE, "能跑", "不能跑"}
    for folder in sorted(stage4_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in split_dirs:
            paper_dirs.extend(sorted(p for p in folder.iterdir() if p.is_dir()))
        elif folder.name not in skip_names:
            paper_dirs.append(folder)

    for folder in paper_dirs:
        jsons = [f for f in folder.glob("*.json") if f.name not in ("sum.json",)]
        jsons = [f for f in jsons if "manifest" not in f.name.lower()]
        if not jsons:
            continue

        can_run, issues, n_eff = analyze(jsons[0])
        entry = {
            "folder": folder.name,
            "effects": n_eff,
            "issues": issues,
        }
        dest_parent = runnable_dir if can_run else not_dir
        if can_run:
            report["runnable"].append(entry)
        else:
            report["not_runnable"].append(entry)

        if not dry_run:
            dest_parent.mkdir(parents=True, exist_ok=True)
            dest = dest_parent / folder.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(folder), str(dest))
            report["moved"].append({"from": folder.name, "to": str(dest.relative_to(stage4_dir))})

    if not dry_run:
        index_path = stage4_dir / "replication_split.json"
        index_path.write_text(
            json.dumps(
                {
                    "criterion": (
                        "Each effect: materials, manipulation, items all have content; "
                        f"items cited_scale must be >= {MIN_ITEMS_CHARS} chars"
                    ),
                    "runnable_count": len(report["runnable"]),
                    "not_runnable_count": len(report["not_runnable"]),
                    "runnable": [x["folder"] for x in report["runnable"]],
                    "not_runnable": [
                        {"folder": x["folder"], "issues": x["issues"]} for x in report["not_runnable"]
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return report


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--stage4-dir", type=Path, default=Path("stage4"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    r = split(args.stage4_dir, dry_run=args.dry_run)
    print(f"runnable: {len(r['runnable'])}")
    for x in r["runnable"]:
        print(f"  ✓ {x['folder']} ({x['effects']} effects)")
    print(f"\nnon_runnable: {len(r['not_runnable'])}")
    for x in r["not_runnable"]:
        print(f"  ✗ {x['folder']} — {len(x['issues'])} gap(s)")
    if not args.dry_run:
        print(f"\nMoved to stage4/{RUNNABLE}/ and stage4/{NOT_RUNNABLE}/")
        print("Index: stage4/replication_split.json")


if __name__ == "__main__":
    main()
