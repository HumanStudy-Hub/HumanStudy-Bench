#!/usr/bin/env python3
"""Validate a HumanStudy package built by the agent pipeline.

The agent writes data files only; the playground supplies the generic runtime
(`scripts/config.py` and `scripts/evaluator.py`). This validator checks the
JSON contract the runtime depends on, plus the cross-file consistency the
generic evaluator relies on (sub-study ids and gt_key references).
"""
import json
import sys
from pathlib import Path


REQUIRED = (
    "index.json",
    "study.json",
    "source/specification.json",
    "source/metadata.json",
    "source/ground_truth.json",
    "source/evidence.json",
    "audit/missing_information.json",
)


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def main() -> None:
    package = Path(sys.argv[1]).resolve()
    roots = [path for path in package.iterdir() if path.is_dir()] if package.exists() else []
    if len(roots) != 1:
        raise SystemExit("package must contain exactly one paper folder")
    root = roots[0]

    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise SystemExit("missing required package files: " + ", ".join(missing))

    for path in root.rglob("*.json"):
        _load(path)  # raises on invalid JSON

    materials_dir = root / "source" / "materials"
    material_files = sorted(materials_dir.glob("*.json")) if materials_dir.is_dir() else []
    if not material_files:
        raise SystemExit("source/materials/ must contain at least one <sub_study_id>.json")

    specification = _load(root / "source/specification.json")
    ground_truth = _load(root / "source/ground_truth.json")

    if not specification.get("study_id"):
        raise SystemExit("specification.json is missing study_id")
    if not specification.get("participants") or not isinstance(specification["participants"], dict):
        raise SystemExit("specification.json is missing participants")

    studies = ground_truth.get("studies")
    if not isinstance(studies, list) or not studies:
        raise SystemExit("ground_truth.json must have a non-empty studies array")

    # Material filenames are the authoritative sub_study_id set.
    sub_ids = {path.stem for path in material_files}

    by_sub_study = specification.get("participants", {}).get("by_sub_study") or {}
    spec_sub_ids = set(by_sub_study.keys()) if isinstance(by_sub_study, dict) else set()
    if spec_sub_ids and sub_ids - spec_sub_ids:
        raise SystemExit(
            f"materials sub_study_ids missing from specification.by_sub_study: {sorted(sub_ids - spec_sub_ids)}"
        )

    # Collect every gt_key referenced by a response_mapping.
    referenced_keys = set()
    for study in studies:
        if not isinstance(study, dict):
            continue
        for finding in study.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            mapping = finding.get("response_mapping")
            if not isinstance(mapping, dict):
                continue
            if mapping.get("sub_study_id") not in sub_ids:
                raise SystemExit(
                    f"response_mapping.sub_study_id {mapping.get('sub_study_id')!r} has no material file"
                )
            for key in (mapping.get("measure_gt_keys") or []):
                referenced_keys.add((mapping["sub_study_id"], key))
            if mapping.get("group_gt_key"):
                referenced_keys.add((mapping["sub_study_id"], mapping["group_gt_key"]))

    # Every referenced gt_key must exist verbatim on a material item.
    gt_by_sub = {}
    for path in material_files:
        material = _load(path)
        items = material.get("items") if isinstance(material, dict) else []
        if not isinstance(items, list):
            items = []
        keys = set()
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("metadata"), dict):
                key = item["metadata"].get("gt_key")
                if key:
                    keys.add(key)
        gt_by_sub[path.stem] = keys

    for sub_id, key in sorted(referenced_keys):
        if key not in gt_by_sub.get(sub_id, set()):
            raise SystemExit(
                f"response_mapping references gt_key {key!r} missing from materials/{sub_id}.json"
            )

    print(f"Validated agent package: {root.name}")


if __name__ == "__main__":
    main()
