"""Deterministic contract audit for Stage 1 study inventories."""

from __future__ import annotations

import re
from typing import Any, Dict, List


_VALID_REPLICABLE = {"YES", "NO", "UNCERTAIN"}


def slugify(value: Any, fallback: str = "study") -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text or fallback


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _source_hints(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _add_missing(missing_by_field: Dict[str, List[str]], field: str, ref: str) -> None:
    missing_by_field.setdefault(field, []).append(ref)


def audit_stage1_study_contract(stage1_json: Dict[str, Any]) -> Dict[str, Any]:
    """Return a field-level audit for a Stage 1 inventory."""
    experiments = [exp for exp in stage1_json.get("experiments", []) or [] if isinstance(exp, dict)]
    studies: Dict[str, Dict[str, Any]] = {}
    missing_by_field: Dict[str, List[str]] = {}
    seen_study_ids: set[str] = set()
    ready = 0
    eligible_or_uncertain = 0

    for idx, exp in enumerate(experiments):
        path = f"$.experiments[{idx}]"
        study_id = str(exp.get("study_id") or exp.get("experiment_id") or f"study_{idx + 1}")
        study_ref = study_id or path
        replicable = str(exp.get("replicable") or "").strip().upper()
        is_downstream_candidate = replicable in {"YES", "UNCERTAIN"}
        if is_downstream_candidate:
            eligible_or_uncertain += 1

        blocking: List[str] = []
        warnings: List[str] = []

        if not _has_text(exp.get("study_id")):
            blocking.append("missing_study_id")
            _add_missing(missing_by_field, "study_id", study_ref)
        if study_id in seen_study_ids:
            blocking.append("duplicate_study_id")
            _add_missing(missing_by_field, "study_id", study_ref)
        seen_study_ids.add(study_id)

        if not _has_text(exp.get("experiment_id")):
            blocking.append("missing_experiment_id")
            _add_missing(missing_by_field, "experiment_id", study_ref)
        if not _has_text(exp.get("study_name")):
            blocking.append("missing_study_name")
            _add_missing(missing_by_field, "study_name", study_ref)
        if replicable not in _VALID_REPLICABLE:
            blocking.append("invalid_replicable_label")
            _add_missing(missing_by_field, "replicable", study_ref)

        if is_downstream_candidate:
            for field in ("design_type", "participant_task", "input", "participants", "output"):
                if not _has_text(exp.get(field)):
                    blocking.append(f"missing_{field}")
                    _add_missing(missing_by_field, field, study_ref)
            if not exp.get("conditions_or_factors"):
                blocking.append("missing_conditions_or_factors")
                _add_missing(missing_by_field, "conditions_or_factors", study_ref)
            hints = _source_hints(exp.get("candidate_source_hints"))
            if not hints:
                blocking.append("missing_candidate_source_hints")
                _add_missing(missing_by_field, "candidate_source_hints", study_ref)
            elif not any(_has_text(hint.get("description")) for hint in hints):
                warnings.append("source_hints_without_description")

        if "has_self_contained_materials" not in exp or not isinstance(exp.get("has_self_contained_materials"), bool):
            warnings.append("self_contained_materials_not_boolean")
            _add_missing(missing_by_field, "has_self_contained_materials", study_ref)

        if not blocking:
            ready += 1
        studies[study_id] = {
            "study_id": exp.get("study_id"),
            "experiment_id": exp.get("experiment_id"),
            "study_name": exp.get("study_name"),
            "replicable": replicable,
            "ready": not blocking,
            "downstream_candidate": is_downstream_candidate,
            "blocking_issues": sorted(set(blocking)),
            "warnings": sorted(set(warnings)),
            "fields": {
                "design_type": exp.get("design_type"),
                "conditions_or_factors": len(exp.get("conditions_or_factors") or []),
                "participant_task": exp.get("participant_task"),
                "input": exp.get("input"),
                "output": exp.get("output"),
                "candidate_source_hints": len(_source_hints(exp.get("candidate_source_hints"))),
            },
        }

    blocking_issue_count = sum(len(study.get("blocking_issues") or []) for study in studies.values())
    return {
        "version": "stage1_study_contract_v1",
        "total_studies": len(experiments),
        "eligible_or_uncertain": eligible_or_uncertain,
        "ready": ready,
        "needs_review": len(experiments) - ready,
        "blocking_issue_count": blocking_issue_count,
        "missing_by_field": {field: sorted(set(refs)) for field, refs in sorted(missing_by_field.items())},
        "studies": studies,
    }


def apply_stage1_study_contract(stage1_json: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the Stage 1 study contract in-place and return it."""
    contract = audit_stage1_study_contract(stage1_json)
    stage1_json["stage1_study_contract"] = contract
    return contract
