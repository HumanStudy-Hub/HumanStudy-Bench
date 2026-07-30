"""HumanStudy-Bench material contract checks for Stage 3 outputs."""

from __future__ import annotations

from typing import Any, Dict, List


_RESPONSE_OPTION_TYPES = {"multiple_choice", "likert", "scale", "matrix", "slider"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _has_ellipsis(value: Any) -> bool:
    return "..." in _text(value) or "…" in _text(value)


def _item_options(item: Dict[str, Any]) -> List[Any]:
    options = item.get("options")
    if isinstance(options, list) and options:
        return options
    choices = item.get("choices")
    if isinstance(choices, list) and choices:
        return choices
    response_format = item.get("response_format")
    if isinstance(response_format, dict) and isinstance(response_format.get("options"), list):
        return response_format["options"]
    return []


def _item_has_scale(item: Dict[str, Any]) -> bool:
    scale = item.get("scale")
    if isinstance(scale, dict) and (scale.get("min") is not None or scale.get("max") is not None):
        return True
    response_format = item.get("response_format")
    if isinstance(response_format, dict):
        return response_format.get("scale_min") is not None or response_format.get("scale_max") is not None
    return False


def _item_has_matrix(item: Dict[str, Any]) -> bool:
    matrix = item.get("matrix")
    if isinstance(matrix, dict) and matrix.get("rows") and matrix.get("columns"):
        return True
    response_format = item.get("response_format")
    return bool(
        isinstance(response_format, dict)
        and response_format.get("rows")
        and response_format.get("columns")
    )


def _unique(values: List[Any]) -> List[str]:
    return sorted({text for value in values if (text := _text(value))})


def _source_trace(material: Dict[str, Any]) -> Dict[str, Any]:
    return material.get("source_trace") if isinstance(material.get("source_trace"), dict) else {}


def _base_sources(material: Dict[str, Any]) -> Dict[str, List[str]]:
    trace = _source_trace(material)
    stimulus_trace = trace.get("stimulus_source_trace") if isinstance(trace.get("stimulus_source_trace"), dict) else {}
    if _text(trace.get("stimulus_source")):
        return {
            "sources": _unique([trace.get("stimulus_source")]),
            "files": _unique([trace.get("stimulus_source_file"), stimulus_trace.get("source_file")]),
        }
    return {
        "sources": _unique([trace.get("primary_source")]),
        "files": _unique([trace.get("source_file")]),
    }


def _item_source_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing: List[str] = []
    sources: List[Any] = []
    files: List[Any] = []
    for index, item in enumerate(items, start=1):
        item_id = _text(item.get("id") or item.get("data_export_tag") or f"item_{index}")
        source = item.get("source")
        source_file = item.get("source_file")
        if not _text(source) and not _text(source_file):
            missing.append(item_id)
        sources.append(source)
        files.append(source_file)
    return {
        "sources": _unique(sources),
        "files": _unique(files),
        "missing_item_sources": missing,
    }


def _condition_source_summary(conditions: Any, material: Dict[str, Any]) -> Dict[str, Any]:
    normalized = [condition for condition in conditions if isinstance(condition, dict)] if isinstance(conditions, list) else []
    trace = _source_trace(material)
    sources: List[Any] = [trace.get("conditions_source"), trace.get("primary_source")]
    files: List[Any] = [trace.get("conditions_source_file"), trace.get("source_file")]
    missing: List[str] = []
    for index, condition in enumerate(normalized, start=1):
        condition_id = _text(condition.get("name") or condition.get("label") or f"condition_{index}")
        sources.extend([condition.get("source"), condition.get("description_source")])
        files.extend([condition.get("source_file")])
        if not any(_text(condition.get(key)) for key in ("source", "source_file", "description_source")):
            # A source_trace-level primary_source can still provenance extracted
            # condition levels from the same instrument.
            if not _text(trace.get("primary_source")) and not _text(trace.get("source_file")):
                missing.append(condition_id)
    return {
        "sources": _unique(sources),
        "files": _unique(files),
        "missing_condition_sources": missing,
    }


def _field_evidence(
    material: Dict[str, Any],
    *,
    items: List[Dict[str, Any]],
    conditions: Any,
) -> Dict[str, Dict[str, Any]]:
    base = _base_sources(material)
    item_sources = _item_source_summary(items)
    condition_sources = _condition_source_summary(conditions, material)
    has_instructions = bool(_text(material.get("instructions") or material.get("stimulus")))
    has_conditions = bool(conditions)
    return {
        "instructions": {
            "status": (
                "present" if has_instructions and (base["sources"] or base["files"])
                else "missing" if has_instructions
                else "not_applicable"
            ),
            **base,
        },
        "stimulus": {
            "status": (
                "present" if has_instructions and (base["sources"] or base["files"])
                else "missing" if has_instructions
                else "not_applicable"
            ),
            **base,
        },
        "response_items": {
            "status": "present" if item_sources["sources"] or item_sources["files"] else ("missing" if items else "not_applicable"),
            **item_sources,
        },
        "response_options": {
            "status": "present" if item_sources["sources"] or item_sources["files"] else ("missing" if items else "not_applicable"),
            **item_sources,
        },
        "conditions": {
            "status": (
                "present" if has_conditions and (condition_sources["sources"] or condition_sources["files"])
                else "missing" if has_conditions
                else "not_applicable"
            ),
            **condition_sources,
        },
    }


def _condition_summary(conditions: Any) -> Dict[str, Any]:
    if not isinstance(conditions, list):
        conditions = []
    normalized = [condition for condition in conditions if isinstance(condition, dict)]
    levels = 0
    described = 0
    for condition in normalized:
        condition_levels = condition.get("levels")
        if isinstance(condition_levels, list):
            levels += len(condition_levels)
        descriptions = condition.get("level_descriptions")
        if isinstance(descriptions, dict):
            described += sum(1 for value in descriptions.values() if _text(value))
    return {
        "status": "present" if normalized else "missing",
        "count": len(normalized),
        "levels": levels,
        "described_levels": described,
    }


def audit_material_contract(material: Dict[str, Any]) -> Dict[str, Any]:
    """Return a field-level audit for one Stage 3 study material."""
    issues: List[str] = []
    warnings: List[str] = []

    instructions = _text(material.get("instructions") or material.get("stimulus"))
    instruction_field = {
        "status": "present" if instructions else "missing",
        "chars": len(instructions),
        "source": (
            material.get("source_trace", {}).get("stimulus_source")
            or material.get("source_trace", {}).get("primary_source")
        ) if isinstance(material.get("source_trace"), dict) else None,
    }
    if not instructions:
        issues.append("missing_instructions")

    raw_items = material.get("items")
    items = [item for item in raw_items if isinstance(item, dict)] if isinstance(raw_items, list) else []
    item_field = {
        "status": "present" if items else "missing",
        "count": len(items),
        "sources": sorted(
            {
                _text(item.get("source"))
                for item in items
                if _text(item.get("source"))
            }
        ),
        "missing_questions": [],
        "truncated_questions": [],
    }
    if not items:
        issues.append("missing_response_items")
    evidence = _field_evidence(material, items=items, conditions=material.get("conditions"))

    options_present = 0
    options_missing: List[str] = []
    for index, item in enumerate(items, start=1):
        item_id = _text(item.get("id") or item.get("data_export_tag") or f"item_{index}")
        question = _text(item.get("question"))
        if not question:
            item_field["missing_questions"].append(item_id)
            issues.append("missing_response_question")
        if _has_ellipsis(question):
            item_field["truncated_questions"].append(item_id)
            issues.append("truncated_response_question")

        item_type = _text(item.get("type")).lower()
        requires_options = item_type in _RESPONSE_OPTION_TYPES
        has_contract = bool(_item_options(item) or _item_has_scale(item) or _item_has_matrix(item))
        if has_contract:
            options_present += 1
        elif requires_options:
            options_missing.append(item_id)

    option_field = {
        "status": "present" if items and not options_missing else ("missing" if options_missing else "not_required"),
        "items_with_options_or_scale": options_present,
        "items_missing_options_or_scale": options_missing,
    }
    if options_missing:
        issues.append("missing_response_options")

    response_schema = material.get("response_schema") or material.get("response_format")
    response_field = {
        "status": "present" if isinstance(response_schema, dict) and response_schema else "missing",
        "schema_keys": sorted(response_schema.keys()) if isinstance(response_schema, dict) else [],
    }
    if items and response_field["status"] == "missing":
        warnings.append("missing_response_schema")

    conditions = _condition_summary(material.get("conditions"))
    if conditions["status"] == "present" and conditions["levels"] == 0:
        warnings.append("condition_without_levels")

    source_trace = material.get("source_trace") if isinstance(material.get("source_trace"), dict) else {}
    source_field = {
        "primary_source": source_trace.get("primary_source"),
        "source_file": source_trace.get("source_file"),
        "stimulus_source": source_trace.get("stimulus_source"),
        "has_candidates": bool(source_trace.get("candidates")),
    }
    if not source_field["primary_source"] and not source_field["stimulus_source"]:
        warnings.append("missing_source_trace")
    for field, payload in evidence.items():
        status = payload.get("status")
        if status == "missing":
            warnings.append(f"missing_source_evidence:{field}")

    blocking = sorted(set(issues))
    return {
        "version": "human-study-bench-material-v1",
        "sub_study_id": material.get("sub_study_id"),
        "ready": not blocking,
        "blocking_issues": blocking,
        "warnings": sorted(set(warnings)),
        "fields": {
            "instructions": instruction_field,
            "stimulus": instruction_field,
            "response_items": item_field,
            "response_options": option_field,
            "response_schema": response_field,
            "conditions": conditions,
            "source_evidence": source_field,
        },
        "field_evidence": evidence,
    }


def apply_material_contracts(materials: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Annotate materials in-place and return a Stage 3 contract summary."""
    audits: Dict[str, Dict[str, Any]] = {}
    ready_count = 0
    selected_ready_count = 0
    field_missing: Dict[str, List[str]] = {}
    selected_field_missing: Dict[str, List[str]] = {}
    source_missing: Dict[str, List[str]] = {}
    selected_source_missing: Dict[str, List[str]] = {}
    selected_sub_studies: List[str] = []

    for sid, material in materials.items():
        audit = audit_material_contract(material)
        audits[sid] = audit
        material["human_material_contract"] = audit
        selection = material.get("selection") if isinstance(material.get("selection"), dict) else {}
        selected = selection.get("keep") is not False
        if selected:
            selected_sub_studies.append(sid)

        readiness = material.setdefault("readiness", {"ready": False, "blocking_issues": [], "warnings": []})
        blocking = list(readiness.get("blocking_issues") or [])
        blocking.extend(audit["blocking_issues"])
        readiness["blocking_issues"] = sorted(set(blocking))
        warnings = list(readiness.get("warnings") or [])
        warnings.extend(f"contract:{warning}" for warning in audit["warnings"])
        readiness["warnings"] = sorted(set(warnings))
        readiness["ready"] = readiness.get("ready") is not False and not readiness["blocking_issues"]

        if audit["ready"] and readiness["ready"]:
            ready_count += 1
            if selected:
                selected_ready_count += 1
        for field, payload in audit["fields"].items():
            if payload.get("status") == "missing":
                field_missing.setdefault(field, []).append(sid)
                if selected:
                    selected_field_missing.setdefault(field, []).append(sid)
        for field, payload in audit.get("field_evidence", {}).items():
            if payload.get("status") == "missing":
                source_missing.setdefault(field, []).append(sid)
                if selected:
                    selected_source_missing.setdefault(field, []).append(sid)

    return {
        "version": "human-study-bench-material-v1",
        "total_sub_studies": len(materials),
        "ready": ready_count,
        "needs_patch": len(materials) - ready_count,
        "selected_sub_studies": selected_sub_studies,
        "selected_total": len(selected_sub_studies),
        "selected_ready": selected_ready_count,
        "selected_needs_patch": len(selected_sub_studies) - selected_ready_count,
        "missing_by_field": field_missing,
        "selected_missing_by_field": selected_field_missing,
        "missing_source_by_field": source_missing,
        "selected_missing_source_by_field": selected_source_missing,
        "materials": audits,
    }
