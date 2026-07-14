from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from generation_pipeline.identifiers import canonical_sub_study_id
from generation_pipeline.pdf.models import (
    EvidenceContext,
    ParsedPdfDocument,
    RESPONSE_TYPES_REQUIRING_CONTRACT,
    evidence_refs,
    invalid_evidence_refs,
    item_has_response_contract,
)


_VALID_BLOCK_ROLES = {"instruction", "stimulus", "context", "example"}
_VALID_PROVENANCE = {"verbatim", "structured_from_source", "reconstructed", "placeholder"}
_VALID_ITEM_TYPES = {
    "multiple_choice",
    "likert",
    "scale",
    "slider",
    "open_ended",
    "ranking",
    "matrix",
    "text",
}


def normalize_instrument(raw: Dict[str, Any], study: Dict[str, Any]) -> Dict[str, Any]:
    study_name = _first_text(
        study.get("study"),
        study.get("study_id"),
        study.get("experiment_id"),
        study.get("study_name"),
        raw.get("study_name"),
        raw.get("study_id"),
    )
    study_id = canonical_sub_study_id(study_name)
    blocks = [
        _normalize_block(block, index)
        for index, block in enumerate(raw.get("blocks") or [], start=1)
        if isinstance(block, dict)
    ]
    blocks = [block for block in blocks if block.get("text")]
    factors = [
        _normalize_factor(factor, index)
        for index, factor in enumerate(raw.get("factors") or [], start=1)
        if isinstance(factor, dict)
    ]
    factors = [factor for factor in factors if factor.get("name")]
    items = [
        _normalize_item(item, study_id, index)
        for index, item in enumerate(raw.get("items") or [], start=1)
        if isinstance(item, dict)
    ]
    items = [item for item in items if item.get("question")]
    source_structures = [
        _normalize_source_structure(value, index)
        for index, value in enumerate(raw.get("source_structures") or [], start=1)
        if isinstance(value, dict)
    ]
    completeness = raw.get("completeness") if isinstance(raw.get("completeness"), dict) else {}
    return {
        "version": "study-instrument-v2",
        "study_id": study_id,
        "study_name": study_name,
        "participant_flow": _text(raw.get("participant_flow")),
        "assignment": _text(raw.get("assignment")),
        "timepoints": [_text(item) for item in raw.get("timepoints") or [] if _text(item)],
        "factors": factors,
        "blocks": blocks,
        "items": items,
        "source_structures": source_structures,
        "completeness": {
            "status": str(completeness.get("status") or "partial").lower(),
            "execution_fidelity": str(completeness.get("execution_fidelity") or "unknown").lower(),
            "missing_fields": [_text(item) for item in completeness.get("missing_fields") or [] if _text(item)],
            "source_absent_fields": [
                _text(item) for item in completeness.get("source_absent_fields") or [] if _text(item)
            ],
            "pipeline_errors": [
                _text(item) for item in completeness.get("pipeline_errors") or [] if _text(item)
            ],
            "compiler_audit": deepcopy(completeness.get("compiler_audit"))
            if isinstance(completeness.get("compiler_audit"), dict)
            else {},
            "repair_audit": [
                deepcopy(item)
                for item in completeness.get("repair_audit") or []
                if isinstance(item, dict)
            ],
            "runtime_filter_audit": [
                deepcopy(item)
                for item in completeness.get("runtime_filter_audit") or []
                if isinstance(item, dict)
            ],
            "unresolved_visual_material": bool(completeness.get("unresolved_visual_material")),
            "notes": _text(completeness.get("notes")),
        },
    }


def build_coverage_ledger(
    instrument: Dict[str, Any],
    study: Dict[str, Any],
    document: ParsedPdfDocument,
    *,
    verifier: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    valid_refs = set(document.block_map())
    fields: Dict[str, Dict[str, Any]] = {}
    blocking: List[str] = []
    warnings: List[str] = []

    participant_blocks = [
        block
        for block in instrument.get("blocks", [])
        if block.get("role") in {"instruction", "stimulus", "context"} and _text(block.get("text"))
    ]
    fields["instructions"] = _field(bool(participant_blocks), _collect_refs(participant_blocks), valid_refs)
    if not participant_blocks:
        blocking.append("missing_instructions")

    items = [item for item in instrument.get("items", []) if isinstance(item, dict)]
    fields["items"] = _field(bool(items), _collect_refs(items), valid_refs)
    if not items:
        blocking.append("missing_response_items")

    missing_contract = [
        str(item.get("id") or index)
        for index, item in enumerate(items, start=1)
        if str(item.get("type") or "").lower() in RESPONSE_TYPES_REQUIRING_CONTRACT
        and not item_has_response_contract(item)
    ]
    fields["response_options"] = {
        "status": "missing" if missing_contract else ("present" if items else "not_applicable"),
        "missing_item_ids": missing_contract,
    }
    if missing_contract:
        blocking.append("missing_response_options")

    conditions_expected = _conditions_expected(study, instrument)
    factors = [factor for factor in instrument.get("factors", []) if isinstance(factor, dict)]
    complete_factors = [factor for factor in factors if factor.get("levels")]
    fields["conditions"] = _field(bool(complete_factors), _collect_factor_refs(complete_factors), valid_refs)
    fields["conditions"]["required"] = conditions_expected
    if conditions_expected and not complete_factors:
        blocking.append("missing_conditions")

    source_structures = [
        value for value in instrument.get("source_structures", []) if isinstance(value, dict)
    ]
    incomplete_structures = [
        str(value.get("id") or "source_structure")
        for value in source_structures
        if "stimulus" in str(value.get("role") or "").lower()
        and not (
            isinstance(value.get("data"), (dict, list))
            and bool(value.get("data"))
        )
    ]
    structure_refs = _collect_refs(source_structures)
    invalid_structure_refs = invalid_evidence_refs(structure_refs, valid_refs)
    fields["source_structures"] = {
        "status": "missing" if incomplete_structures or invalid_structure_refs else (
            "present" if source_structures else "not_applicable"
        ),
        "incomplete_ids": incomplete_structures,
        "evidence_refs": sorted(set(structure_refs)),
        "invalid_refs": invalid_structure_refs,
    }
    if incomplete_structures:
        blocking.append("incomplete_source_structure")
    if invalid_structure_refs:
        blocking.append("invalid_source_evidence")

    provenance_values = [
        str(value.get("provenance") or "")
        for value in [*participant_blocks, *items]
        if isinstance(value, dict)
    ]
    reconstructed = sum(value in {"reconstructed", "placeholder"} for value in provenance_values)
    fields["participant_facing_provenance"] = {
        "status": "present" if provenance_values else "missing",
        "values": sorted(set(provenance_values)),
        "reconstructed_or_placeholder": reconstructed,
    }
    if reconstructed:
        blocking.append("participant_material_not_source_exact")

    all_refs = _collect_refs([*participant_blocks, *items]) + _collect_factor_refs(factors)
    invalid_refs = invalid_evidence_refs(all_refs, valid_refs)
    fields["source_evidence"] = {
        "status": "present" if all_refs and not invalid_refs else "missing",
        "evidence_refs": sorted(set(all_refs)),
        "invalid_refs": invalid_refs,
    }
    if not all_refs:
        blocking.append("missing_source_evidence")
    if invalid_refs:
        blocking.append("invalid_source_evidence")

    ellipsis_paths = _ellipsis_paths(
        {
            "blocks": participant_blocks,
            "items": items,
        }
    )
    placeholder_paths = _placeholder_paths(
        {
            "blocks": participant_blocks,
            "items": items,
        }
    )
    if ellipsis_paths or placeholder_paths:
        if ellipsis_paths:
            blocking.append("truncated_participant_material")
        if placeholder_paths:
            blocking.append("unresolved_participant_placeholder")
        fields["verbatim_completeness"] = {
            "status": "missing",
            "ellipsis_paths": ellipsis_paths,
            "placeholder_paths": placeholder_paths,
        }
    else:
        fields["verbatim_completeness"] = {
            "status": "present",
            "ellipsis_paths": [],
            "placeholder_paths": [],
        }

    completeness = instrument.get("completeness") if isinstance(instrument.get("completeness"), dict) else {}
    if completeness.get("status") != "complete":
        blocking.append("pdf_material_incomplete")
    fidelity = str(completeness.get("execution_fidelity") or "unknown").lower()
    fields["execution_fidelity"] = {"status": fidelity}
    if fidelity in {"partial", "none", "unknown"}:
        blocking.append("insufficient_execution_fidelity")
    elif fidelity == "semantically_equivalent":
        warnings.append(
            "Participant task is source-complete but display/order details were reconstructed without changing the response contract."
        )
    if completeness.get("unresolved_visual_material"):
        blocking.append("unresolved_visual_material")
    pipeline_errors = [
        _text(item) for item in completeness.get("pipeline_errors") or [] if _text(item)
    ]
    if pipeline_errors:
        blocking.append("pdf_material_pipeline_error")
    if document.degraded:
        warnings.append("PDF used degraded pypdf block parsing; layout and table structure were unavailable.")
    warnings.extend(document.warnings)

    verifier = verifier if isinstance(verifier, dict) else {}
    verifier_status = str(verifier.get("status") or "not_run")
    unsupported = [str(item) for item in verifier.get("unsupported_paths") or [] if str(item).strip()]
    leaks = [str(item) for item in verifier.get("non_participant_paths") or [] if str(item).strip()]
    verifier_missing = [str(item) for item in verifier.get("missing_fields") or [] if str(item).strip()]
    nonblocking_absences = [
        str(item)
        for item in verifier.get("nonblocking_source_absences") or []
        if str(item).strip()
    ]
    fields["semantic_verifier"] = {
        "status": verifier_status,
        "unsupported_paths": unsupported,
        "non_participant_paths": leaks,
        "missing_fields": verifier_missing,
        "nonblocking_source_absences": nonblocking_absences,
        "notes": _text(verifier.get("notes")),
    }
    if verifier and verifier_status != "pass":
        blocking.append("semantic_verification_failed")
    if unsupported:
        blocking.append("unsupported_participant_material")
    if leaks:
        blocking.append("non_participant_text_in_runtime_material")
    if verifier_missing:
        blocking.append("verifier_detected_missing_material")
    if nonblocking_absences:
        warnings.append(
            f"Semantic verifier recorded {len(nonblocking_absences)} non-blocking source absence(s)."
        )

    return {
        "version": "study-instrument-coverage-v1",
        "study_id": instrument.get("study_id"),
        "ready": not blocking,
        "fields": fields,
        "blocking_issues": sorted(set(blocking)),
        "warnings": sorted(set(warnings)),
        "source_absent_fields": list(completeness.get("source_absent_fields") or []),
        "extractor_missing_fields": list(completeness.get("missing_fields") or []),
        "pipeline_errors": pipeline_errors,
    }


def compile_hsb_material(
    instrument: Dict[str, Any],
    coverage: Dict[str, Any],
    document: ParsedPdfDocument,
    context: EvidenceContext,
    *,
    verifier: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    instruction_blocks = [
        {
            "id": block.get("id"),
            "role": block.get("role"),
            "text": block.get("text"),
            "condition": block.get("condition"),
            "provenance": block.get("provenance"),
            "evidence_refs": block.get("evidence_refs", []),
        }
        for block in instrument.get("blocks", [])
        if block.get("role") in {"instruction", "stimulus", "context"} and _text(block.get("text"))
    ]
    instructions = "\n\n".join(str(block["text"]).strip() for block in instruction_blocks)
    primary_source = "pdf_docling_llm" if document.parser.startswith("docling") else "pdf_block_llm"
    conditions = [_compile_condition(factor) for factor in instrument.get("factors", []) if factor.get("name")]
    for condition in conditions:
        condition["source"] = primary_source
        condition["source_file"] = document.source_file
    items = deepcopy(instrument.get("items") or [])
    for item in items:
        item["source"] = primary_source
        item["source_file"] = document.source_file
    response_schema = _response_schema(items)
    return {
        "sub_study_id": instrument.get("study_id"),
        "study_name": instrument.get("study_name"),
        "instructions": instructions,
        "instruction_blocks": instruction_blocks,
        "items": items,
        "conditions": conditions,
        "source_structures": deepcopy(instrument.get("source_structures") or []),
        "response_schema": response_schema,
        "preserve_full_instrument_for_runtime": True,
        "readiness": {
            "ready": bool(coverage.get("ready")),
            "blocking_issues": list(coverage.get("blocking_issues") or []),
            "warnings": list(coverage.get("warnings") or []),
        },
        "source_trace": {
            "primary_source": primary_source,
            "source_file": document.source_file,
            "extractor": "pdf_evidence_provider_v1",
            "document_parser": document.to_dict(include_blocks=False),
            "evidence_context": context.to_dict(),
            "coverage_ledger": coverage,
            "semantic_verifier": verifier or {},
            "preserve_full_instrument_for_runtime": True,
        },
        "study_instrument": deepcopy(instrument),
        "coverage_ledger": deepcopy(coverage),
    }


def material_gaps(coverage: Dict[str, Any]) -> List[str]:
    fields = coverage.get("fields") if isinstance(coverage.get("fields"), dict) else {}
    gaps: List[str] = []
    for field in (
        "instructions",
        "items",
        "response_options",
        "conditions",
        "source_structures",
        "source_evidence",
    ):
        payload = fields.get(field) if isinstance(fields.get(field), dict) else {}
        if payload.get("status") == "missing" and (field != "conditions" or payload.get("required", True)):
            gaps.append(field)
    if "unresolved_visual_material" in set(coverage.get("blocking_issues") or []):
        gaps.append("visual_material")
    if "pdf_material_incomplete" in set(coverage.get("blocking_issues") or []):
        gaps.append("complete_instrument")
    return gaps


def _normalize_block(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    role = str(raw.get("role") or "context").lower()
    if role not in _VALID_BLOCK_ROLES:
        role = "context"
    provenance = str(raw.get("provenance") or "structured_from_source").lower()
    if provenance not in _VALID_PROVENANCE:
        provenance = "structured_from_source"
    return {
        "id": _slug(raw.get("id") or f"block_{index}"),
        "role": role,
        "text": _text(raw.get("text")),
        "condition": _text(raw.get("condition")) or None,
        "provenance": provenance,
        "evidence_refs": evidence_refs(raw.get("evidence_refs")),
    }


def _normalize_factor(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    levels: List[Dict[str, Any]] = []
    for level_index, level in enumerate(raw.get("levels") or [], start=1):
        if isinstance(level, dict):
            label = _text(level.get("label") or level.get("name"))
            description = _text(level.get("participant_facing_text"))
            implementation_notes = _text(
                level.get("implementation_notes")
                or (level.get("description") if not description else "")
            )
            refs = evidence_refs(level.get("evidence_refs"))
        else:
            label = _text(level)
            description = ""
            implementation_notes = ""
            refs = []
        if label:
            levels.append(
                {
                    "label": label,
                    "participant_facing_text": description,
                    "implementation_notes": implementation_notes,
                    "evidence_refs": refs,
                }
            )
    return {
        "name": _text(raw.get("name") or f"factor_{index}"),
        "assignment": _text(raw.get("assignment")),
        "provenance": _normalize_provenance(raw.get("provenance")),
        "levels": levels,
        "evidence_refs": evidence_refs(raw.get("evidence_refs")),
    }


def _normalize_item(raw: Dict[str, Any], study_id: str, index: int) -> Dict[str, Any]:
    item_type = str(raw.get("type") or "open_ended").lower()
    if item_type not in _VALID_ITEM_TYPES:
        item_type = "open_ended"
    options: List[str] = []
    option_records: List[Dict[str, Any]] = []
    raw_option_records = raw.get("option_records")
    option_source = (
        raw_option_records
        if isinstance(raw_option_records, list) and raw_option_records
        else raw.get("options") or []
    )
    for option in option_source:
        if isinstance(option, dict):
            label = _text(
                option.get("label")
                or option.get("text")
                or option.get("value")
                or option.get("id")
            )
            if not label:
                continue
            record = deepcopy(option)
            record["label"] = label
            display_label = _option_display_label(record, label)
            record["display_label"] = display_label
            option_records.append(record)
            options.append(display_label)
        elif _text(option):
            options.append(_text(option))
    scale = raw.get("scale") if isinstance(raw.get("scale"), dict) else {}
    normalized_scale: Dict[str, Any] = {}
    if scale:
        normalized_scale = {
            "min": scale.get("min"),
            "max": scale.get("max"),
            "anchors": _normalize_scale_anchors(scale.get("anchors")),
        }
    matrix = raw.get("matrix") if isinstance(raw.get("matrix"), dict) else {}
    normalized_matrix: Dict[str, Any] = {}
    if matrix:
        normalized_matrix = {
            "rows": [_text(item) for item in matrix.get("rows") or [] if _text(item)],
            "columns": [_text(item) for item in matrix.get("columns") or [] if _text(item)],
            "response_mode": _text(matrix.get("response_mode")) or "single_per_row",
        }
    provenance = str(raw.get("provenance") or "structured_from_source").lower()
    if provenance not in _VALID_PROVENANCE:
        provenance = "structured_from_source"
    item: Dict[str, Any] = {
        "id": _slug(raw.get("id") or f"{study_id}_item_{index}"),
        "question": _text(raw.get("question")),
        "type": item_type,
        "options": options,
        "provenance": provenance,
        "evidence_refs": evidence_refs(raw.get("evidence_refs")),
    }
    for key in ("condition", "block", "timepoint", "trial_group"):
        value = raw.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            item[key] = value
        elif isinstance(value, dict) and value:
            item[key] = deepcopy(value)
    if isinstance(raw.get("attributes"), (dict, list)):
        item["attributes"] = deepcopy(raw["attributes"])
    if isinstance(raw.get("metadata"), dict):
        item["metadata"] = deepcopy(raw["metadata"])
    if option_records:
        item["option_records"] = option_records
    if normalized_scale:
        item["scale"] = normalized_scale
        item["response_format"] = {
            "answer_type": item_type,
            "scale_min": normalized_scale.get("min"),
            "scale_max": normalized_scale.get("max"),
            "anchors": normalized_scale.get("anchors", {}),
            "options": options,
        }
    elif normalized_matrix:
        item["matrix"] = normalized_matrix
        item["response_format"] = {"answer_type": "matrix", **normalized_matrix}
    elif options:
        item["response_format"] = {"answer_type": item_type, "options": options}
    return item


def _normalize_scale_anchors(value: Any) -> Dict[str, str]:
    """Accept the common object and value/label-list anchor encodings."""
    if isinstance(value, dict):
        return {
            str(key): _text(label)
            for key, label in value.items()
            if _text(label)
        }
    if not isinstance(value, list):
        return {}
    anchors: Dict[str, str] = {}
    for entry in value:
        if not isinstance(entry, dict):
            continue
        key = entry.get("value")
        label = _text(entry.get("label") or entry.get("text"))
        if key not in (None, "") and label:
            anchors[str(key)] = label
    return anchors


def _option_display_label(record: Dict[str, Any], label: str) -> str:
    explicit_display = _text(record.get("display"))
    if explicit_display:
        return explicit_display
    # A descriptive label already carries the participant-visible alternative.
    # Appending every structured attribute would leak compiler metadata such as
    # source column names or level mappings back into the runtime option.
    if len(label) > 10 and (":" in label or ";" in label):
        return label
    attributes = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
    displayed: List[str] = []
    paired: Dict[str, Dict[str, str]] = {}
    for key, value in attributes.items():
        match = re.match(r"^(.*?)[ _](label|value|unit)$", str(key), flags=re.IGNORECASE)
        if not match or value in (None, "", [], {}):
            continue
        prefix, field = match.groups()
        paired.setdefault(prefix.lower(), {})[field.lower()] = _text(value)
    consumed_prefixes: set[str] = set()
    for prefix, value in paired.items():
        if value.get("label") and value.get("value"):
            rendered = value["value"]
            unit = value.get("unit")
            if unit and unit.lower() not in rendered.lower():
                rendered = f"{rendered} {unit}"
            displayed.append(f"{value['label']}: {rendered}")
            consumed_prefixes.add(prefix)
    for key, value in attributes.items():
        key_text = _text(key)
        if not key_text or value in (None, "", [], {}):
            continue
        normalized_key = re.sub(r"[^a-z0-9]+", "_", key_text.lower()).strip("_")
        key_prefix = re.sub(r"_(?:label|value|unit)$", "", normalized_key)
        if key_prefix in consumed_prefixes:
            continue
        if normalized_key.endswith(("_label", "_level", "_mapping")) or normalized_key in {
            "label",
            "level",
            "level_mapping",
            "source_mapping",
        }:
            continue
        if isinstance(value, dict) and value.get("value") not in (None, ""):
            key_text = _text(value.get("label") or key_text)
            value_text = _text(value.get("display") or value.get("value"))
            unit = _text(value.get("unit"))
            if unit and unit.lower() not in value_text.lower():
                value_text = f"{value_text} {unit}"
        elif isinstance(value, (dict, list)):
            continue
        else:
            value_text = _text(value)
        if value_text:
            displayed.append(f"{key_text}: {value_text}")
    if displayed and all(part.split(": ", 1)[-1].lower() in label.lower() for part in displayed):
        return label
    return f"{label} - {'; '.join(displayed)}" if displayed else label


def _normalize_source_structure(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    value = {
        "id": _slug(raw.get("id") or f"source_structure_{index}"),
        "role": _text(raw.get("role")) or "stimulus_generation_source",
        "description": _text(raw.get("description") or raw.get("text") or raw.get("content")),
        "provenance": _normalize_provenance(raw.get("provenance")),
        "evidence_refs": evidence_refs(raw.get("evidence_refs")),
        "runtime": False,
    }
    if isinstance(raw.get("data"), (dict, list)):
        value["data"] = deepcopy(raw["data"])
    else:
        canonical = {
            "id",
            "role",
            "description",
            "text",
            "content",
            "provenance",
            "evidence_refs",
            "runtime",
        }
        structured = {
            str(key): deepcopy(item)
            for key, item in raw.items()
            if key not in canonical and item not in (None, "", [], {})
        }
        if structured:
            value["data"] = structured
    return value


def _compile_condition(factor: Dict[str, Any]) -> Dict[str, Any]:
    levels = [level for level in factor.get("levels", []) if isinstance(level, dict)]
    return {
        "name": factor.get("name"),
        "assignment": factor.get("assignment"),
        "levels": [level.get("label") for level in levels if level.get("label")],
        "level_descriptions": {
            str(level.get("label")): str(level.get("participant_facing_text"))
            for level in levels
            if level.get("label") and level.get("participant_facing_text")
        },
        "implementation_notes": {
            str(level.get("label")): str(level.get("implementation_notes"))
            for level in levels
            if level.get("label") and level.get("implementation_notes")
        },
        "evidence_refs": sorted(
            set(evidence_refs(factor.get("evidence_refs")) + _collect_refs(levels))
        ),
    }


def _response_schema(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    types = list(dict.fromkeys(str(item.get("type") or "open_ended") for item in items))
    schema: Dict[str, Any] = {"answer_type": types[0] if len(types) == 1 else "mixed"}
    if len(items) == 1:
        item = items[0]
        response_format = item.get("response_format") if isinstance(item.get("response_format"), dict) else {}
        schema.update(response_format)
    schema["item_count"] = len(items)
    return schema


def _conditions_expected(study: Dict[str, Any], instrument: Dict[str, Any]) -> bool:
    assignment = _text(instrument.get("assignment")).lower()
    if assignment in {"between-subjects", "within-subjects", "mixed"}:
        return True
    if assignment in {"measured", "none"}:
        return False
    design = _text(study.get("design") or study.get("design_type")).lower()
    if any(
        term in design
        for term in ("between", "within", "factorial", "randomized", "manipulated", "mixed")
    ):
        return True
    return False


def _field(present: bool, refs: List[str], valid_refs: set[str]) -> Dict[str, Any]:
    invalid = invalid_evidence_refs(refs, valid_refs)
    return {
        "status": "present" if present and refs and not invalid else "missing",
        "content_present": present,
        "evidence_refs": sorted(set(refs)),
        "invalid_refs": invalid,
    }


def _collect_refs(values: Iterable[Dict[str, Any]]) -> List[str]:
    refs: List[str] = []
    for value in values:
        refs.extend(evidence_refs(value.get("evidence_refs")))
    return refs


def _collect_factor_refs(factors: Iterable[Dict[str, Any]]) -> List[str]:
    refs: List[str] = []
    for factor in factors:
        refs.extend(evidence_refs(factor.get("evidence_refs")))
        refs.extend(_collect_refs([level for level in factor.get("levels", []) if isinstance(level, dict)]))
    return refs


_TRUNCATION_ELLIPSIS_RE = re.compile(r"(?<!\.)\.{3}(?!\.)|…")
_RUNTIME_PLACEHOLDER_RE = re.compile(
    r"\[(?:insert|replace|fill|product|category|attribute|condition|scenario|"
    r"stimulus|question|item|option|response)(?:[^\]]*)\]"
    r"|\{(?:insert|replace|fill|product|category|attribute|condition|scenario|"
    r"stimulus|question|item|option|response)(?:[^}]*)\}"
    r"|<\s*(?:insert|replace|fill|product|category|attribute|condition|scenario|"
    r"stimulus|question|item|option|response)(?:[^>]*)>",
    flags=re.IGNORECASE,
)


def _ellipsis_paths(value: Any, path: str = "$") -> List[str]:
    paths: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(_ellipsis_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_ellipsis_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str) and _TRUNCATION_ELLIPSIS_RE.search(value):
        paths.append(path)
    return paths


def _placeholder_paths(value: Any, path: str = "$") -> List[str]:
    paths: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(_placeholder_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_placeholder_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str) and _RUNTIME_PLACEHOLDER_RE.search(value):
        paths.append(path)
    return paths


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_") or "item"


def _normalize_provenance(value: Any) -> str:
    provenance = str(value or "structured_from_source").lower()
    return provenance if provenance in _VALID_PROVENANCE else "structured_from_source"


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _first_text(*values: Any) -> str:
    return next((_text(value) for value in values if _text(value)), "Study")
