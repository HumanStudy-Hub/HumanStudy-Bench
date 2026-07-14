"""Deterministic contract audit for Stage 1 study inventories."""

from __future__ import annotations

import re
from typing import Any, Dict, List


_VALID_REPLICABLE = {"YES", "NO", "UNCERTAIN"}
_VALID_PROVENANCE = {"current_paper", "unclear"}
_VALID_RELATIONSHIP_KINDS = {
    "paired_contrast",
    "multi_unit_comparison",
    "replication_set",
    "sequence",
    "shared_sample",
    "other",
}
_VALID_MATERIAL_VARIANT_ROLES = {
    "condition",
    "stimulus",
    "form",
    "order",
    "item_set",
    "other",
}
_VALID_SIMULATION_BARRIER_KINDS = {
    "physical_action",
    "consequential_commitment",
    "live_interaction",
    "longitudinal_exposure",
    "dynamic_environment",
    "specialized_apparatus",
    "other",
}
_INLINE_BLOCK_REF_GROUP_RE = re.compile(
    r"\s*\[\s*p\d{2,}_[a-z0-9_]+"
    r"(?:\s*,\s*p\d{2,}_[a-z0-9_]+)*\s*\]",
    re.IGNORECASE,
)
_PROVENANCE_KEYS = {
    "evidence_refs",
    "field_evidence",
    "source_mention_ids",
    "source_relation_ids",
}


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


def _remove_inline_evidence_tokens(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        cleaned, count = _INLINE_BLOCK_REF_GROUP_RE.subn("", value)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned).strip()
        return cleaned, count
    if isinstance(value, list):
        output: List[Any] = []
        removed = 0
        for item in value:
            cleaned, count = _remove_inline_evidence_tokens(item)
            output.append(cleaned)
            removed += count
        return output, removed
    if isinstance(value, dict):
        output: Dict[str, Any] = {}
        removed = 0
        for key, item in value.items():
            if str(key) in _PROVENANCE_KEYS:
                output[key] = item
                continue
            cleaned, count = _remove_inline_evidence_tokens(item)
            output[key] = cleaned
            removed += count
        return output, removed
    return value, 0


def normalize_stage1_semantic_fields(stage1_json: Dict[str, Any]) -> int:
    """Remove internal block IDs accidentally embedded in participant-facing text."""
    experiments = stage1_json.get("experiments")
    if not isinstance(experiments, list):
        return 0
    removed = 0
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            continue
        cleaned, count = _remove_inline_evidence_tokens(experiment)
        experiments[index] = cleaned
        removed += count
    return removed


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

        provenance = str(exp.get("unit_provenance") or "").strip().lower()
        if provenance not in _VALID_PROVENANCE:
            blocking.append("invalid_or_external_unit_provenance")
            _add_missing(missing_by_field, "unit_provenance", study_ref)
        if exp.get("is_distinct_empirical_unit") is not True:
            blocking.append("not_a_distinct_empirical_unit")
            _add_missing(missing_by_field, "is_distinct_empirical_unit", study_ref)
        if not _has_text(exp.get("unit_provenance_evidence")):
            blocking.append("missing_unit_provenance_evidence")
            _add_missing(missing_by_field, "unit_provenance_evidence", study_ref)
        if provenance == "unclear" and replicable == "YES":
            blocking.append("unclear_provenance_requires_uncertain_label")
            _add_missing(missing_by_field, "provenance_eligibility_consistency", study_ref)
        empirical_support = exp.get("empirical_support")
        empirical_support = empirical_support if isinstance(empirical_support, dict) else {}
        support_statuses = {
            field: str(empirical_support.get(field) or "").strip().lower()
            for field in (
                "own_sample_or_assignment",
                "participant_facing_task",
                "quantitative_result",
            )
        }
        invalid_support = [
            field
            for field, status in support_statuses.items()
            if status not in {"yes", "no", "unclear"}
        ]
        if invalid_support:
            blocking.append("missing_or_invalid_empirical_support")
            _add_missing(missing_by_field, "empirical_support", study_ref)
        if support_statuses.get("quantitative_result") == "no" and replicable != "NO":
            blocking.append("no_quantitative_result_requires_no_label")
            _add_missing(missing_by_field, "empirical_support_consistency", study_ref)

        simulation_barriers = exp.get("simulation_barriers")
        if not isinstance(simulation_barriers, list):
            blocking.append("simulation_barriers_not_an_array")
            _add_missing(missing_by_field, "simulation_barriers", study_ref)
            simulation_barriers = []
        primary_target_barrier = False
        for barrier_index, barrier in enumerate(simulation_barriers):
            barrier_ref = f"{study_ref}.simulation_barriers[{barrier_index}]"
            if not isinstance(barrier, dict):
                blocking.append("invalid_simulation_barrier")
                _add_missing(missing_by_field, "simulation_barriers", barrier_ref)
                continue
            kind = str(barrier.get("kind") or "").strip().lower()
            if kind not in _VALID_SIMULATION_BARRIER_KINDS:
                blocking.append("invalid_simulation_barrier_kind")
                _add_missing(missing_by_field, "simulation_barrier_kind", barrier_ref)
            if not _has_text(barrier.get("description")):
                blocking.append("missing_simulation_barrier_description")
                _add_missing(missing_by_field, "simulation_barrier_description", barrier_ref)
            if not isinstance(barrier.get("affects_primary_target"), bool):
                blocking.append("invalid_primary_target_barrier_flag")
                _add_missing(missing_by_field, "simulation_barrier_target", barrier_ref)
            elif barrier.get("affects_primary_target") is True:
                primary_target_barrier = True
            barrier_evidence = [
                str(ref).strip()
                for ref in barrier.get("evidence_refs") or []
                if str(ref).strip()
            ]
            if not barrier_evidence:
                blocking.append("missing_simulation_barrier_evidence")
                _add_missing(missing_by_field, "simulation_barrier_evidence", barrier_ref)
        if primary_target_barrier and replicable != "NO":
            blocking.append("primary_target_barrier_requires_no_label")
            _add_missing(missing_by_field, "eligibility_consistency", study_ref)

        exclusion_reasons = exp.get("exclusion_reasons")
        if isinstance(exclusion_reasons, list):
            reasons = [
                str(reason).strip()
                for reason in exclusion_reasons
                if str(reason).strip()
            ]
        else:
            reasons = []
        if is_downstream_candidate and reasons:
            blocking.append("candidate_has_exclusion_reasons")
            _add_missing(missing_by_field, "eligibility_consistency", study_ref)
        if replicable == "NO" and not reasons:
            blocking.append("excluded_without_reason")
            _add_missing(missing_by_field, "exclusion_reasons", study_ref)

        evidence_refs = [
            str(ref).strip()
            for ref in exp.get("evidence_refs") or []
            if str(ref).strip()
        ]
        if not evidence_refs:
            blocking.append("missing_evidence_refs")
            _add_missing(missing_by_field, "evidence_refs", study_ref)

        material_variants = exp.get("material_variants")
        if not isinstance(material_variants, list):
            blocking.append("material_variants_not_an_array")
            _add_missing(missing_by_field, "material_variants", study_ref)
            material_variants = []
        seen_variant_ids: set[str] = set()
        seen_variant_labels: set[str] = set()
        for variant_index, variant in enumerate(material_variants):
            variant_ref = f"{study_ref}.material_variants[{variant_index}]"
            if not isinstance(variant, dict):
                blocking.append("invalid_material_variant")
                _add_missing(missing_by_field, "material_variants", variant_ref)
                continue
            variant_id = str(variant.get("variant_id") or "").strip()
            variant_label = str(variant.get("label") or "").strip()
            if not variant_id or variant_id in seen_variant_ids:
                blocking.append("missing_or_duplicate_material_variant_id")
                _add_missing(missing_by_field, "material_variant_id", variant_ref)
            if not variant_label or variant_label.lower() in seen_variant_labels:
                blocking.append("missing_or_duplicate_material_variant_label")
                _add_missing(missing_by_field, "material_variant_label", variant_ref)
            seen_variant_ids.add(variant_id)
            seen_variant_labels.add(variant_label.lower())
            role = str(variant.get("role") or "").strip().lower()
            if role not in _VALID_MATERIAL_VARIANT_ROLES:
                blocking.append("invalid_material_variant_role")
                _add_missing(missing_by_field, "material_variant_role", variant_ref)
            if variant.get("is_alternative_version") is not True:
                blocking.append("material_variant_is_not_an_alternative_version")
                _add_missing(
                    missing_by_field,
                    "material_variant_alternative_version",
                    variant_ref,
                )
            variant_evidence = [
                str(ref).strip()
                for ref in variant.get("evidence_refs") or []
                if str(ref).strip()
            ]
            if not variant_evidence:
                blocking.append("missing_material_variant_evidence")
                _add_missing(missing_by_field, "material_variant_evidence", variant_ref)

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
            "unit_provenance": provenance,
            "is_distinct_empirical_unit": exp.get("is_distinct_empirical_unit"),
            "empirical_support": support_statuses,
            "simulation_barriers": simulation_barriers,
            "primary_target_barrier": primary_target_barrier,
            "exclusion_reasons": reasons,
            "evidence_refs": evidence_refs,
            "material_variant_count": len(material_variants),
            "ready": not blocking,
            "downstream_candidate": is_downstream_candidate,
            "blocking_issues": sorted(set(blocking)),
            "warnings": sorted(set(warnings)),
            "fields": {
                "design_type": exp.get("design_type"),
                "conditions_or_factors": len(exp.get("conditions_or_factors") or []),
                "material_variants": len(material_variants),
                "participant_task": exp.get("participant_task"),
                "input": exp.get("input"),
                "output": exp.get("output"),
                "candidate_source_hints": len(_source_hints(exp.get("candidate_source_hints"))),
            },
        }

    comparison_groups = [
        group
        for group in stage1_json.get("comparison_groups", []) or []
        if isinstance(group, dict)
    ]
    comparison_group_reports: Dict[str, Dict[str, Any]] = {}
    seen_group_ids: set[str] = set()
    valid_study_ids = set(studies)
    for idx, group in enumerate(comparison_groups):
        path = f"$.comparison_groups[{idx}]"
        group_id = str(group.get("comparison_group_id") or "").strip()
        group_ref = group_id or path
        blocking: List[str] = []
        warnings: List[str] = []
        if not group_id:
            blocking.append("missing_comparison_group_id")
            _add_missing(missing_by_field, "comparison_group_id", group_ref)
        elif group_id in seen_group_ids:
            blocking.append("duplicate_comparison_group_id")
            _add_missing(missing_by_field, "comparison_group_id", group_ref)
        seen_group_ids.add(group_id)
        members = [
            str(value).strip()
            for value in group.get("member_study_ids") or []
            if str(value).strip()
        ]
        if len(set(members)) < 2:
            blocking.append("comparison_group_requires_two_distinct_members")
            _add_missing(missing_by_field, "member_study_ids", group_ref)
        unknown_members = sorted(set(members) - valid_study_ids)
        if unknown_members:
            blocking.append("comparison_group_has_unknown_members")
            _add_missing(missing_by_field, "member_study_ids", group_ref)
        relationship_kind = str(group.get("relationship_kind") or "").strip().lower()
        if relationship_kind not in _VALID_RELATIONSHIP_KINDS:
            blocking.append("invalid_relationship_kind")
            _add_missing(missing_by_field, "relationship_kind", group_ref)
        if not _has_text(group.get("comparison_target")):
            warnings.append("missing_comparison_target")
            _add_missing(missing_by_field, "comparison_target", group_ref)
        evidence_refs = [
            str(ref).strip()
            for ref in group.get("evidence_refs") or []
            if str(ref).strip()
        ]
        if not evidence_refs:
            blocking.append("missing_comparison_evidence_refs")
            _add_missing(missing_by_field, "comparison_evidence_refs", group_ref)
        report_key = group_ref if group_ref not in comparison_group_reports else path
        comparison_group_reports[report_key] = {
            "comparison_group_id": group_id or None,
            "member_study_ids": members,
            "relationship_kind": relationship_kind,
            "comparison_target": group.get("comparison_target"),
            "evidence_refs": evidence_refs,
            "ready": not blocking,
            "blocking_issues": sorted(set(blocking)),
            "warnings": sorted(set(warnings)),
        }

    study_blocking_count = sum(
        len(study.get("blocking_issues") or []) for study in studies.values()
    )
    comparison_blocking_count = sum(
        len(group.get("blocking_issues") or [])
        for group in comparison_group_reports.values()
    )
    blocking_issue_count = study_blocking_count + comparison_blocking_count
    return {
        "version": "stage1_study_contract_v5",
        "total_studies": len(experiments),
        "eligible_or_uncertain": eligible_or_uncertain,
        "ready": ready,
        "needs_review": len(experiments) - ready,
        "blocking_issue_count": blocking_issue_count,
        "study_blocking_issue_count": study_blocking_count,
        "comparison_group_blocking_issue_count": comparison_blocking_count,
        "missing_by_field": {field: sorted(set(refs)) for field, refs in sorted(missing_by_field.items())},
        "studies": studies,
        "comparison_groups": comparison_group_reports,
    }


def apply_stage1_study_contract(stage1_json: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the Stage 1 study contract in-place and return it."""
    previous_contract = stage1_json.get("stage1_study_contract")
    previous_normalization = (
        previous_contract.get("normalization")
        if isinstance(previous_contract, dict)
        and isinstance(previous_contract.get("normalization"), dict)
        else {}
    )
    previous_removed = int(
        previous_normalization.get("inline_evidence_token_groups_removed") or 0
    )
    removed_inline_refs = normalize_stage1_semantic_fields(stage1_json)
    contract = audit_stage1_study_contract(stage1_json)
    contract["normalization"] = {
        "inline_evidence_token_groups_removed": previous_removed + removed_inline_refs,
        "inline_evidence_token_groups_removed_this_pass": removed_inline_refs,
    }
    stage1_json["stage1_study_contract"] = contract
    return contract
