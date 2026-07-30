from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from generation_pipeline.identifiers import canonical_sub_study_id
from generation_pipeline.pdf.contract import (
    build_coverage_ledger,
    compile_hsb_material,
    material_gaps,
    normalize_instrument,
)
from generation_pipeline.pdf.evidence import PdfEvidenceIndex, _stage1_experiment
from generation_pipeline.pdf.models import (
    RESPONSE_TYPES_REQUIRING_CONTRACT,
    item_has_response_contract,
)
from generation_pipeline.pdf.parser import parse_pdf_document
from src.llm.helpers import call_with_timeout


PDF_INSTRUMENT_MAX_TOKENS = 16000
PDF_COMPILER_MAX_TOKENS = 16000
PDF_PLANNER_MAX_TOKENS = 8000
PDF_UNIT_AUDITOR_MAX_TOKENS = 5000
PDF_VERIFIER_MAX_TOKENS = 12000
PDF_COMPILER_WORKERS = 4
PDF_COMPILER_MAX_UNIT_ITEMS = 64
PDF_COMPILER_CACHE_VERSION = "runtime-compiler-v6"
PDF_TABLE_RECOVERY_MAX_TOKENS = 16000
PDF_TABLE_ADJUDICATOR_MAX_TOKENS = 3000
PDF_TABLE_RECOVERY_CACHE_VERSION = "vision-table-recovery-v4"
PDF_TABLE_LINKER_MAX_TOKENS = 6000
PDF_TABLE_LINKER_CACHE_VERSION = "semantic-table-linker-v1"


def extract_pdf_study_materials(
    stage_json: Dict[str, Any],
    pdf_path: Path,
    llm_client: Any,
    *,
    stage1_json: Optional[Dict[str, Any]] = None,
    source_label: str = "pdf_docling_llm",
    timeout: Optional[float] = 180.0,
    max_attempts: int = 2,
    retry_delay: float = 1.0,
    only_sub_study_id: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
    force_parse: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Extract complete, source-grounded study instruments from a paper PDF."""
    del source_label
    if llm_client is None:
        return {}
    studies = _selected_or_all_studies(stage_json)
    if only_sub_study_id:
        wanted = canonical_sub_study_id(only_sub_study_id)
        studies = [study for study in studies if _study_id(study) == wanted]
    if not studies:
        return {}

    document = parse_pdf_document(
        Path(pdf_path),
        artifacts_dir=artifacts_dir,
        force=force_parse,
        prefer_docling=True,
    )
    index = PdfEvidenceIndex(document)
    print(
        f"  PDF evidence parser: {document.parser} pages={document.page_count} "
        f"blocks={len(document.blocks)} chars={document.text_chars}",
        flush=True,
    )

    materials: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for position, study in enumerate(studies, start=1):
        study_name = _study_name(study)
        context = index.context_for_study(study, stage1_json=stage1_json)
        print(
            f"  Stage 3 PDF instrument extraction {position}/{len(studies)}: "
            f"{study_name} ({context.mode}, {context.context_chars} chars)",
            flush=True,
        )
        try:
            initial_images = _context_visual_images(document, context.block_ids)
            if document.parser == "docling_vision" and not initial_images:
                raise RuntimeError(
                    "Image-dominant PDF has no rendered page evidence; provide an artifacts_dir"
                )
            raw = _extract_instrument(
                llm_client,
                study,
                stage1_json,
                context.text,
                valid_block_ids=context.block_ids,
                images=initial_images,
                timeout=timeout,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
            )
            instrument = normalize_instrument(raw, study)
            _link_document_tables(
                instrument,
                document,
                llm_client,
                study,
                stage1_json,
                evidence_block_ids=context.block_ids,
                timeout=timeout,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                cache_dir=(
                    Path(document.artifacts_dir) / "table_linker"
                    if document.artifacts_dir
                    else None
                ),
                use_cache=not force_parse,
            )
            _attach_structured_evidence(instrument, document)
            preliminary = build_coverage_ledger(instrument, study, document)
            gaps = material_gaps(preliminary)
            if context.mode == "full_document":
                gaps = [gap for gap in gaps if gap != "complete_instrument"]
            if _has_runtime_source_structures(instrument):
                # Exact linked tables are compiler inputs, not missing runtime
                # fields. A whole-instrument repair here can overwrite those
                # tables with another LLM transcription before compilation.
                gaps = []
            gaps = list(dict.fromkeys(gaps))
            if gaps:
                repair_context = index.context_for_study(
                    study,
                    stage1_json=stage1_json,
                    gaps=gaps,
                )
                print(
                    f"    targeted recovery: {', '.join(gaps)} "
                    f"({repair_context.context_chars} chars)",
                    flush=True,
                )
                repaired_raw = _repair_instrument(
                    llm_client,
                    study,
                    stage1_json,
                    instrument,
                    gaps,
                    repair_context.text,
                    valid_block_ids=repair_context.block_ids,
                    images=_context_visual_images(document, repair_context.block_ids),
                    timeout=timeout,
                    max_attempts=max_attempts,
                    retry_delay=retry_delay,
                )
                repaired = normalize_instrument(repaired_raw, study)
                _attach_structured_evidence(repaired, document)
                instrument = _merge_instruments(
                    instrument,
                    repaired,
                    replace_runtime=True,
                )
                preliminary = build_coverage_ledger(instrument, study, document)

            if _has_runtime_source_structures(instrument):
                _recover_complex_source_tables(
                    instrument,
                    document,
                    llm_client,
                    timeout=timeout,
                    max_attempts=max_attempts,
                    retry_delay=retry_delay,
                    cache_dir=(
                        Path(document.artifacts_dir) / "table_recovery"
                        if document.artifacts_dir
                        else None
                    ),
                    use_cache=not force_parse,
                )

            if _needs_runtime_compilation(instrument):
                structure_refs = _source_structure_refs(instrument)
                compiler_context = index.context_for_refs(structure_refs)
                print(
                    "    structured runtime compilation "
                    f"({len(compiler_context)} chars)",
                    flush=True,
                )
                try:
                    compiled_raw = _compile_runtime_from_structures(
                        llm_client,
                        study,
                        stage1_json,
                        instrument,
                        compiler_context,
                        valid_block_ids=sorted(index.block_ids),
                        images=_context_visual_images(document, structure_refs),
                        timeout=timeout,
                        max_attempts=max_attempts,
                        retry_delay=retry_delay,
                        cache_dir=(
                            Path(document.artifacts_dir)
                            / "compiler_cache"
                            / str(instrument.get("study_id") or "study")
                            if document.artifacts_dir
                            else None
                        ),
                        use_cache=not force_parse,
                    )
                    compiled = _apply_runtime_compilation(compiled_raw, instrument, study)
                    _attach_structured_evidence(compiled, document)
                    if _runtime_compilation_improves(instrument, compiled):
                        instrument = compiled
                    else:
                        _record_compiler_failure(
                            instrument,
                            "Compiler output did not improve executable response coverage.",
                        )
                except Exception as exc:
                    _record_compiler_failure(
                        instrument,
                        f"{type(exc).__name__}: {exc}",
                    )
                    print(
                        f"    structured runtime compilation failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

            _drop_unsafe_runtime_blocks(instrument)
            refs = _instrument_refs(instrument)
            verifier_context = index.context_for_refs(refs)
            verifier = _verify_instrument(
                llm_client,
                instrument,
                verifier_context,
                valid_block_ids=sorted(index.block_ids),
                images=_context_visual_images(document, refs),
                timeout=timeout,
                retry_delay=retry_delay,
            )
            if _verifier_has_correctable_runtime_errors(verifier):
                correction_context = index.context_for_study(
                    study,
                    stage1_json=stage1_json,
                    gaps=["source_evidence"],
                )
                print("    verifier-guided runtime correction", flush=True)
                try:
                    original_instrument = instrument
                    original_verifier = verifier
                    verifier_patch = _generate_verifier_patch(
                        llm_client,
                        study,
                        stage1_json,
                        instrument,
                        verifier,
                        correction_context.text,
                        valid_block_ids=correction_context.block_ids,
                        images=_context_visual_images(document, correction_context.block_ids),
                        timeout=timeout,
                        max_attempts=max_attempts,
                        retry_delay=retry_delay,
                    )
                    corrected = _apply_verifier_patch(
                        instrument,
                        verifier_patch,
                        verifier,
                        study,
                    )
                    _attach_structured_evidence(corrected, document)
                    corrected_refs = _instrument_refs(corrected)
                    corrected_verifier = _verify_instrument(
                        llm_client,
                        corrected,
                        index.context_for_refs(corrected_refs),
                        valid_block_ids=sorted(index.block_ids),
                        images=_context_visual_images(document, corrected_refs),
                        timeout=timeout,
                        retry_delay=retry_delay,
                    )
                    if str(corrected_verifier.get("status") or "").lower() == "pass":
                        instrument = corrected
                        verifier = corrected_verifier
                    else:
                        instrument = original_instrument
                        verifier = original_verifier
                        print(
                            "    verifier patch discarded: corrected instrument "
                            "did not pass the independent re-audit",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"    verifier patch rejected: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
            _reconcile_completeness_with_verifier(
                instrument,
                study,
                document,
                verifier,
            )
            coverage = build_coverage_ledger(instrument, study, document, verifier=verifier)
            material = compile_hsb_material(
                instrument,
                coverage,
                document,
                context,
                verifier=verifier,
            )
            material["selection"] = deepcopy(study.get("selection")) if isinstance(study.get("selection"), dict) else {"keep": True}
            materials[str(material["sub_study_id"])] = material
            print(
                f"    instrument: blocks={len(instrument.get('blocks', []))} "
                f"items={len(instrument.get('items', []))} "
                f"factors={len(instrument.get('factors', []))} ready={coverage.get('ready')}",
                flush=True,
            )
        except Exception as exc:
            errors.append(f"{study_name}: {type(exc).__name__}: {exc}")
            print(f"    PDF instrument extraction failed: {type(exc).__name__}: {exc}", flush=True)

    if materials:
        return materials
    if errors:
        raise RuntimeError("PDF instrument extraction failed for all studies: " + " | ".join(errors))
    return {}


def _extract_instrument(
    llm_client: Any,
    study: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
    context: str,
    *,
    valid_block_ids: List[str],
    images: List[str],
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
) -> Dict[str, Any]:
    prompt = _instrument_prompt(study, stage1_json, context, valid_block_ids)
    parsed = _generate_json(
        llm_client,
        prompt,
        timeout=timeout,
        max_tokens=PDF_INSTRUMENT_MAX_TOKENS,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        images=images,
    )
    return _instrument_object(parsed)


def _repair_instrument(
    llm_client: Any,
    study: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
    instrument: Dict[str, Any],
    gaps: List[str],
    context: str,
    *,
    valid_block_ids: List[str],
    images: List[str],
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
) -> Dict[str, Any]:
    prompt = f"""Recover missing fields in a source-grounded experimental instrument.

Return a COMPLETE corrected StudyInstrument, not prose and not a loose patch.
Preserve every valid existing item/block/factor. Add or correct only content supported
by the evidence blocks. Never infer exact participant wording from paper results.
If the source does not contain a requested field, list it in
completeness.source_absent_fields. Do not use ellipses.

Keep participant-facing blocks/items separate from non-runtime source_structures.
When source tables specify executable trials, expand those trials into complete
items with options and condition metadata. The returned object must contain:
study_id, study_name, participant_flow, assignment, timepoints, factors, blocks,
items, source_structures, and completeness (including execution_fidelity).

IMPORTANT FOR TABLE-DEFINED TASKS:
- A source table that gives all attribute values, condition assignments, and a
  participant-facing question template is sufficient to build semantically
  equivalent executable trials. Expand it; do not demand photocopies of every
  original booklet page.
- Missing typography, page layout, or randomized order is not a missing field
  unless the paper says it changes the target effect.
- Put strategy definitions, assignment rules, and method prose in
  implementation_notes/source_structures, never participant_facing_text/blocks.
- source_structures.data must transcribe the complete rows/mappings needed to
  audit the expanded items; a prose summary such as "see the table" is invalid.
- When current source_structures.data.raw_evidence_tables is present, it was
  copied deterministically from Docling. Consume every relevant row instead of
  replacing it with a shorter summary.

MISSING OR INVALID FIELDS:
{json.dumps(gaps, ensure_ascii=False)}

CURRENT INSTRUMENT:
{json.dumps(instrument, indent=2, ensure_ascii=False)}

STUDY INVENTORY:
{json.dumps(_study_inventory(study, stage1_json), indent=2, ensure_ascii=False)}

VALID EVIDENCE BLOCK IDS:
{json.dumps(valid_block_ids, ensure_ascii=False)}

TARGETED SOURCE EVIDENCE:
{context}

Return ONLY one JSON object in the StudyInstrument schema from the original task."""
    parsed = _generate_json(
        llm_client,
        prompt,
        timeout=timeout,
        max_tokens=PDF_INSTRUMENT_MAX_TOKENS,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        images=images,
    )
    return _instrument_object(parsed)


def _generate_verifier_patch(
    llm_client: Any,
    study: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
    instrument: Dict[str, Any],
    verifier: Dict[str, Any],
    context: str,
    *,
    valid_block_ids: List[str],
    images: List[str],
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
) -> Dict[str, Any]:
    allowed = _audited_runtime_ids(instrument, verifier)
    prompt = f"""Propose a constrained semantic patch for a StudyInstrument.

The audit is feedback, not source evidence. Return only the smallest patch needed
to address its blocking paths. Do not rewrite the complete instrument. Existing
records not named by the audit must remain byte-for-byte untouched by the patch
applier. Remove method/results text from runtime material; replace a record only
when the source provides the correct participant-facing content. Never invent a
result, statistic, condition, option, scale, or question.

Patch constraints:
- remove_block_ids/replace_blocks may target only block indexes listed by the
  audit's unsupported_paths or non_participant_paths.
- remove_item_ids/replace_items may target only audited item indexes.
- add_items is allowed only when blocking missing_fields is non-empty and each
  added item is directly supported by source evidence.
- replace_factors may target only audited factor indexes.
- Every replace_blocks/replace_items/replace_factors entry is the replacement
  record itself and must carry its existing top-level id/name. Never wrap a record
  as {{"index": 1, "factor": {{...}}}}.
- A method block that is not participant-facing should normally be removed; the
  source document remains available through source_structures/evidence refs.

EXACT AUDITED TARGETS ALLOWED BY THE PATCH APPLIER:
{json.dumps({key: sorted(value) for key, value in allowed.items()}, indent=2, ensure_ascii=False)}

Do not target any runtime record absent from this allowlist. If an audited record
cannot be corrected from the source, leave it unchanged and explain why in notes.

CURRENT INSTRUMENT:
{json.dumps(instrument, indent=2, ensure_ascii=False)}

SEMANTIC AUDIT:
{json.dumps(verifier, indent=2, ensure_ascii=False)}

STUDY INVENTORY:
{json.dumps(_study_inventory(study, stage1_json), indent=2, ensure_ascii=False)}

VALID EVIDENCE BLOCK IDS:
{json.dumps(valid_block_ids, ensure_ascii=False)}

SOURCE EVIDENCE:
{context}

Return ONLY JSON:
{{
  "remove_block_ids": ["block_id"],
  "replace_blocks": [],
  "remove_item_ids": [],
  "replace_items": [],
  "add_items": [],
  "replace_factors": [],
  "notes": "short source-grounded reason"
}}"""
    parsed = _generate_json(
        llm_client,
        prompt,
        timeout=timeout,
        max_tokens=PDF_INSTRUMENT_MAX_TOKENS,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        images=images,
    )
    return parsed.get("patch") if isinstance(parsed.get("patch"), dict) else parsed


def _apply_verifier_patch(
    instrument: Dict[str, Any],
    patch: Dict[str, Any],
    verifier: Dict[str, Any],
    study: Dict[str, Any],
) -> Dict[str, Any]:
    allowed = _audited_runtime_ids(instrument, verifier)
    remove_blocks = _string_list(patch.get("remove_block_ids"))
    remove_items = _string_list(patch.get("remove_item_ids"))
    replace_blocks = [
        value for value in patch.get("replace_blocks", []) or [] if isinstance(value, dict)
    ]
    replace_items = [
        value for value in patch.get("replace_items", []) or [] if isinstance(value, dict)
    ]
    add_items = [value for value in patch.get("add_items", []) or [] if isinstance(value, dict)]
    replace_factors = [
        value for value in patch.get("replace_factors", []) or [] if isinstance(value, dict)
    ]

    requested_block_ids = set(remove_blocks) | {
        str(value.get("id") or "") for value in replace_blocks
    }
    requested_item_ids = set(remove_items) | {
        str(value.get("id") or "") for value in replace_items
    }
    requested_factor_names = {
        str(value.get("name") or "") for value in replace_factors
    }
    illegal = {
        "blocks": sorted(requested_block_ids - allowed["blocks"]),
        "items": sorted(requested_item_ids - allowed["items"]),
        "factors": sorted(requested_factor_names - allowed["factors"]),
    }
    illegal = {key: values for key, values in illegal.items() if values}
    if illegal:
        raise ValueError(f"Verifier patch targeted unaudited runtime records: {illegal}")
    if add_items and not verifier.get("missing_fields"):
        raise ValueError("Verifier patch attempted to add items without a blocking missing field")

    candidate = deepcopy(instrument)
    block_replacements = {str(value.get("id")): value for value in replace_blocks}
    item_replacements = {str(value.get("id")): value for value in replace_items}
    factor_replacements = {str(value.get("name")): value for value in replace_factors}
    candidate["blocks"] = [
        _merge_record_patch(value, block_replacements.get(str(value.get("id"))))
        for value in candidate.get("blocks", []) or []
        if isinstance(value, dict) and str(value.get("id")) not in set(remove_blocks)
    ]
    candidate["items"] = [
        _merge_record_patch(value, item_replacements.get(str(value.get("id"))))
        for value in candidate.get("items", []) or []
        if isinstance(value, dict) and str(value.get("id")) not in set(remove_items)
    ]
    existing_item_ids = {str(value.get("id")) for value in candidate["items"]}
    for value in add_items:
        item_id = str(value.get("id") or "")
        if not item_id or item_id in existing_item_ids:
            raise ValueError(f"Verifier patch added an invalid or duplicate item id: {item_id!r}")
        candidate["items"].append(deepcopy(value))
        existing_item_ids.add(item_id)
    candidate["factors"] = [
        _merge_record_patch(value, factor_replacements.get(str(value.get("name"))))
        for value in candidate.get("factors", []) or []
        if isinstance(value, dict)
    ]

    operations = {
        "removed_blocks": remove_blocks,
        "replaced_blocks": sorted(block_replacements),
        "removed_items": remove_items,
        "replaced_items": sorted(item_replacements),
        "added_items": [str(value.get("id")) for value in add_items],
        "replaced_factors": sorted(factor_replacements),
        "notes": str(patch.get("notes") or "").strip(),
    }
    if not any(value for key, value in operations.items() if key != "notes"):
        raise ValueError("Verifier patch contained no applicable operation")
    completeness = candidate.setdefault("completeness", {})
    repair_audit = [
        deepcopy(value)
        for value in completeness.get("repair_audit", []) or []
        if isinstance(value, dict)
    ]
    repair_audit.append(operations)
    completeness["repair_audit"] = repair_audit
    return normalize_instrument(candidate, study)


def _merge_record_patch(
    current: Dict[str, Any],
    replacement: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(replacement, dict):
        return deepcopy(current)
    merged = {**deepcopy(current), **deepcopy(replacement)}
    if "options" in replacement and "option_records" not in replacement:
        merged.pop("option_records", None)
    return merged


def _audited_runtime_ids(
    instrument: Dict[str, Any],
    verifier: Dict[str, Any],
) -> Dict[str, set[str]]:
    paths = [
        *_string_list(verifier.get("unsupported_paths")),
        *_string_list(verifier.get("non_participant_paths")),
    ]
    blocks = [value for value in instrument.get("blocks", []) or [] if isinstance(value, dict)]
    items = [value for value in instrument.get("items", []) or [] if isinstance(value, dict)]
    factors = [value for value in instrument.get("factors", []) or [] if isinstance(value, dict)]
    allowed = {"blocks": set(), "items": set(), "factors": set()}
    for path in paths:
        selector = re.match(
            r"^\$\.(blocks|items|factors)\[\?\(@\.(id|name)==['\"]([^'\"]+)['\"]\)\](?:\.|$)",
            path,
        )
        if selector:
            kind, key, identifier = selector.groups()
            expected_key = "name" if kind == "factors" else "id"
            if key == expected_key:
                records = {"blocks": blocks, "items": items, "factors": factors}[kind]
                if any(str(value.get(key) or "") == identifier for value in records):
                    allowed[kind].add(identifier)
            continue
        keyed = re.match(
            r"^\$\.(blocks|items|factors)\[['\"]([^'\"]+)['\"]\](?:\.|$)",
            path,
        )
        if keyed:
            kind, identifier = keyed.groups()
            key = "name" if kind == "factors" else "id"
            records = {"blocks": blocks, "items": items, "factors": factors}[kind]
            if any(str(value.get(key) or "") == identifier for value in records):
                allowed[kind].add(identifier)
            continue
        match = re.match(r"^\$\.(blocks|items|factors)\[(\d+)\](?:\.|$)", path)
        if not match:
            continue
        kind, raw_index = match.groups()
        index = int(raw_index)
        records = {"blocks": blocks, "items": items, "factors": factors}[kind]
        if index >= len(records):
            continue
        key = "name" if kind == "factors" else "id"
        identifier = str(records[index].get(key) or "")
        if identifier:
            allowed[kind].add(identifier)
    return allowed


def _compile_runtime_from_structures(
    llm_client: Any,
    study: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
    instrument: Dict[str, Any],
    context: str,
    *,
    valid_block_ids: List[str],
    images: List[str],
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Plan and compile table/template evidence into bounded runtime units."""
    compiler_input = {
        "participant_flow": instrument.get("participant_flow"),
        "assignment": instrument.get("assignment"),
        "timepoints": instrument.get("timepoints") or [],
        "factors": instrument.get("factors") or [],
        "blocks": instrument.get("blocks") or [],
        "current_items": instrument.get("items") or [],
        "source_structures": _deduplicated_source_structures(instrument),
        "completeness": instrument.get("completeness") or {},
    }
    cache_key = _runtime_compiler_cache_key(
        llm_client,
        study,
        compiler_input,
        valid_block_ids,
    )
    cached_plan = (
        _read_compiler_cache(cache_dir / "plan.json", cache_key)
        if cache_dir is not None and use_cache
        else None
    )
    if (
        isinstance(cached_plan, dict)
        and isinstance(cached_plan.get("plan"), dict)
        and "runtime_factors" in cached_plan["plan"]
    ):
        raw_cached_plan = cached_plan["plan"]
        plan = _normalize_runtime_plan(raw_cached_plan, valid_block_ids)
        if isinstance(raw_cached_plan.get("plan_audit"), dict):
            plan["plan_audit"] = deepcopy(raw_cached_plan["plan_audit"])
        print("      compiler plan cache hit", flush=True)
    else:
        plan = _plan_runtime_units(
            llm_client,
            study,
            stage1_json,
            compiler_input,
            context,
            valid_block_ids=valid_block_ids,
            images=images,
            timeout=timeout,
            max_attempts=max(3, max_attempts),
            retry_delay=retry_delay,
        )
        if cache_dir is not None:
            _write_compiler_cache(
                cache_dir / "plan.json",
                cache_key,
                {"plan": plan},
            )
    units = plan["units"]
    print(f"      compiler plan: {len(units)} unit(s)", flush=True)
    results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    pending_units: List[Dict[str, Any]] = []
    for unit in units:
        cached_unit = (
            _read_compiler_cache(cache_dir / "units" / f"{unit['id']}.json", cache_key)
            if cache_dir is not None and use_cache
            else None
        )
        if (
            isinstance(cached_unit, dict)
            and cached_unit.get("unit") == unit
            and isinstance(cached_unit.get("result"), dict)
            and (cached_unit["result"].get("_source_audit") or {}).get("status")
            == "pass"
            and not _compiled_unit_contract_errors(
                cached_unit["result"],
                expected_item_count=unit.get("expected_item_count"),
            )
        ):
            results[unit["id"]] = cached_unit["result"]
        else:
            pending_units.append(unit)
    if len(results):
        print(f"      compiler unit cache hits: {len(results)}/{len(units)}", flush=True)

    def compile_unit(unit: Dict[str, Any]) -> Dict[str, Any]:
        return _compile_runtime_unit(
            llm_client,
            study,
            stage1_json,
            compiler_input,
            unit,
            context,
            valid_block_ids=valid_block_ids,
            images=images,
            timeout=timeout,
            max_attempts=max(3, max_attempts),
            retry_delay=retry_delay,
        )

    workers = min(PDF_COMPILER_WORKERS, len(pending_units))
    if workers == 1:
        unit = pending_units[0]
        try:
            results[unit["id"]] = compile_unit(unit)
            if cache_dir is not None:
                _write_compiler_cache(
                    cache_dir / "units" / f"{unit['id']}.json",
                    cache_key,
                    {"unit": unit, "result": results[unit["id"]]},
                )
        except Exception as exc:
            errors[unit["id"]] = f"{type(exc).__name__}: {exc}"
    elif workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(compile_unit, unit): unit for unit in pending_units}
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    results[unit["id"]] = future.result()
                    if cache_dir is not None:
                        _write_compiler_cache(
                            cache_dir / "units" / f"{unit['id']}.json",
                            cache_key,
                            {"unit": unit, "result": results[unit["id"]]},
                        )
                    print(f"      compiled unit: {unit['label']}", flush=True)
                except Exception as exc:
                    errors[unit["id"]] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"      compiler unit failed: {unit['label']}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
    return _aggregate_runtime_compilations(instrument, plan, results, errors)


def _plan_runtime_units(
    llm_client: Any,
    study: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
    compiler_input: Dict[str, Any],
    context: str,
    *,
    valid_block_ids: List[str],
    images: List[str],
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
) -> Dict[str, Any]:
    planner_source = {
        "participant_facing_blocks": compiler_input.get("blocks") or [],
        "source_structures": compiler_input.get("source_structures") or [],
    }
    analysis_target = {
        "study_label": _study_name(study),
        "effects": [
            {"IV": effect.get("IV"), "DV": effect.get("DV")}
            for effect in study.get("effects", []) or []
            if isinstance(effect, dict)
        ],
    }
    prompt = f"""Plan bounded runtime-compilation units for one psychology study.

The input contains participant-facing material plus non-runtime source tables or
templates. Partition the source-defined task into independent administered units
that can each contain at most {PDF_COMPILER_MAX_UNIT_ITEMS} concrete response
items. A unit must preserve the
assignment coupling of the original experiment: all items jointly administered
to one participant condition/timepoint stay in the same unit. For example, when
table columns are participant groups and rows are repeated stimuli, plan one unit
per group/timepoint containing the relevant rows; do not transpose that table into
one unit per row and invent a full factor crossing. For a non-assigned item battery,
questionnaire block or scenario can be a unit. Use the source structure, not a
fixed list of domain words.

The selected analysis target below is only a scope hint. It may describe marginal
factor labels that were not participant assignment cells. Do not infer assignment
from it, and do not use any previously generated factors/items to plan source axes.
Treat it as an analysis-specific view of the experiment: include only
administrations needed to reproduce that target's IV and DV. Exclude a repeated
session used only by a different analysis target. Do not turn an analytic
comparison or reported "baseline" into a literal participant condition unless the
method or assignment evidence shows that condition was administered.
The source assignment table is the truth. First identify its assignment axis (what one participant/group receives)
and repeated-stimulus axis. If rows have different strategy labels within the same
group column, that group is one coupled unit; never create one all-row unit for
each strategy. Do not call something an initial baseline unless the source shows
that baseline was administered at the initial timepoint.
When `corrected_evidence_tables` is present, it is a bbox-cropped visual
transcription of the same block and takes precedence over noisy
`raw_evidence_tables`; preserve any listed uncertain_cells as uncertainty.

For each unit, report the number of concrete runtime items that the source implies
when it is determinable. Count a control/baseline only at the timepoint and within
the assignment where the source actually administered it. Never synthesize a
baseline or a Cartesian product of theoretical factor levels. A factor level need
not occur for every stimulus. Count repeated timepoint items only when they are
part of the selected study. Do not create a unit for consent, demographics,
analysis tables, or reported outcomes. An appendix sample is a wording/template
source, not an extra practice trial, unless the paper says participants received
it separately. Plan all source-defined task units. Return one unit
covering the complete task only when the source genuinely has no repeated
partition. Evidence refs must be VALID BLOCK IDS.

Define runtime_factors once for the complete plan. Include only axes that control
participant assignment, item routing, or repeated timepoints. A strategy label
that varies across repeated rows inside the same assigned group belongs in item
metadata, not as a crossed participant factor. Every factor and level must cite
the source block that defines it; use empty participant_facing_text for labels the
participant never saw.

Classify every proposed factor with axis_role=assignment, routing, or timepoint.
Repeated stimuli, questionnaire rows, products, scenarios, and template variables
are not runtime factors merely because every participant sees several of them.
For axis_role=routing, set routes_participant=true only when that level sends a
participant to a different task path; repeated items never route participants.
Classify source examples/templates that were not independently administered under
excluded_source_units, never under units. Every runtime unit must explicitly set
runtime_administered=true and cite evidence that the unit was administered; a
source showing only how to instantiate wording is not administration evidence.
Administration may be established compositionally: a method sentence can state
that participants completed a repeated task, while an exhaustive assignment or
stimulus table supplies the per-group/per-item instantiations and a sample item
supplies the display template. In that case, cite all relevant blocks and compile
the mapped trials; the paper need not reprint every instantiated screen. This does
not make the source table itself a participant-facing runtime item. Do not require
subject rosters, per-cell prose, or separately printed copies of deterministic
table expansions. Conversely, a table alone, without evidence that its mapped
task was administered, remains insufficient.

SELECTED ANALYSIS TARGET (SCOPE ONLY):
{json.dumps(analysis_target, indent=2, ensure_ascii=False)}

SOURCE-ONLY PLANNER INPUT:
{json.dumps(planner_source, indent=2, ensure_ascii=False)}

VALID BLOCK IDS:
{json.dumps(valid_block_ids, ensure_ascii=False)}

SOURCE BLOCKS:
{context}

Return ONLY JSON:
{{
  "source_axes": {{
    "assignment_axis": ["source-defined participant groups/conditions"],
    "repeated_stimulus_axis": ["source-defined repeated blocks/items"],
    "timepoint_axis": ["source-defined administration timepoints"],
    "coupling_rule": "which repeated items one assigned participant receives together"
  }},
  "runtime_factors": [
    {{
      "name": "canonical_factor_name",
      "axis_role": "assignment|routing|timepoint",
      "routes_participant": false,
      "assignment": "between-subjects|within-subjects|mixed|measured",
      "provenance": "structured_from_source",
      "levels": [
        {{
          "label": "source-defined level",
          "participant_facing_text": "",
          "implementation_notes": "non-runtime assignment meaning",
          "evidence_refs": ["p001_table_00001"]
        }}
      ],
      "evidence_refs": ["p001_table_00001"]
    }}
  ],
  "units": [
    {{
      "id": "stable_unit_id",
      "label": "human-readable source unit",
      "scope": "exact rows/conditions/timepoint to compile",
      "expected_item_count": 4,
      "assignment_scope": "one source-defined participant group/condition",
      "runtime_administered": true,
      "administration_evidence_refs": ["p001_text_00002"],
      "evidence_refs": ["p001_table_00001"]
    }}
  ],
  "excluded_source_units": [
    {{
      "label": "source example or template not independently administered",
      "reason": "source-only template",
      "evidence_refs": ["p001_table_00001"]
    }}
  ]
}}"""
    parsed = _generate_json(
        llm_client,
        prompt,
        timeout=timeout,
        max_tokens=PDF_PLANNER_MAX_TOKENS,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        images=images,
    )
    plan = _normalize_runtime_plan(parsed, valid_block_ids)
    audit = _runtime_plan_contract_audit(plan) or _audit_runtime_plan(
        llm_client,
        study,
        plan,
        planner_source,
        context,
        valid_block_ids=valid_block_ids,
        images=images,
        timeout=timeout,
        retry_delay=retry_delay,
    )
    if audit["status"] != "pass":
        repair_prompt = f"""{prompt}

The prior plan failed an independent source audit. Return a complete corrected
replacement plan. Remove every unsupported factor/unit, add every genuinely
administered missing unit, and preserve condition/timepoint coupling. Do not
defend or annotate the old plan.

PRIOR PLAN:
{json.dumps(plan, indent=2, ensure_ascii=False)}

INDEPENDENT PLAN AUDIT:
{json.dumps(audit, indent=2, ensure_ascii=False)}
"""
        repaired = _generate_json(
            llm_client,
            repair_prompt,
            timeout=timeout,
            max_tokens=PDF_PLANNER_MAX_TOKENS,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            images=images,
        )
        plan = _normalize_runtime_plan(repaired, valid_block_ids)
        audit = _runtime_plan_contract_audit(plan) or _audit_runtime_plan(
            llm_client,
            study,
            plan,
            planner_source,
            context,
            valid_block_ids=valid_block_ids,
            images=images,
            timeout=timeout,
            retry_delay=retry_delay,
        )
        if audit["status"] != "pass":
            raise ValueError(
                "Runtime compiler plan failed independent audit after repair: "
                + str(audit.get("notes") or audit)
            )
    plan["plan_audit"] = audit
    return plan


def _normalize_runtime_plan(
    parsed: Dict[str, Any],
    valid_block_ids: List[str],
) -> Dict[str, Any]:
    raw_units = parsed.get("units") if isinstance(parsed.get("units"), list) else []
    units: List[Dict[str, Any]] = []
    excluded_source_units = [
        deepcopy(value)
        for value in parsed.get("excluded_source_units", []) or []
        if isinstance(value, dict)
    ]
    used_ids: set[str] = set()
    valid_refs = set(valid_block_ids)
    for index, raw in enumerate(raw_units, start=1):
        if not isinstance(raw, dict):
            continue
        if raw.get("runtime_administered") is not True:
            excluded_source_units.append(
                {
                    "label": str(raw.get("label") or raw.get("scope") or f"Unit {index}"),
                    "reason": "planner did not establish that this source unit was administered",
                    "evidence_refs": _string_list(raw.get("evidence_refs")),
                }
            )
            continue
        label = str(raw.get("label") or raw.get("scope") or f"Unit {index}").strip()
        unit_id = re.sub(r"[^a-z0-9]+", "_", str(raw.get("id") or label).lower()).strip("_")
        expected = raw.get("expected_item_count")
        try:
            expected_count = int(expected) if expected not in (None, "") else None
        except (TypeError, ValueError):
            expected_count = None
        if expected_count is not None and expected_count < 1:
            expected_count = None
        if not unit_id:
            unit_id = f"unit_{index}"
        if unit_id in used_ids:
            unit_id = f"{unit_id}_{index}"
        used_ids.add(unit_id)
        units.append(
            {
                "id": unit_id,
                "label": label,
                "scope": str(raw.get("scope") or label).strip(),
                "expected_item_count": expected_count,
                "assignment_scope": str(raw.get("assignment_scope") or "").strip(),
                "runtime_administered": True,
                "administration_evidence_refs": [
                    ref
                    for ref in _string_list(raw.get("administration_evidence_refs"))
                    if ref in valid_refs
                ],
                "evidence_refs": [
                    ref for ref in _string_list(raw.get("evidence_refs")) if ref in valid_refs
                ],
            }
        )
    if not units:
        raise ValueError("Runtime compiler plan contains no source-validated administered units")
    if len(units) > 32:
        raise ValueError(f"Runtime compiler plan is over-split: {len(units)} units")
    source_axes = parsed.get("source_axes") if isinstance(parsed.get("source_axes"), dict) else {}
    runtime_factors: List[Dict[str, Any]] = []
    excluded_runtime_factors: List[Dict[str, Any]] = []
    for value in parsed.get("runtime_factors", []) or []:
        if not isinstance(value, dict) or not str(value.get("name") or "").strip():
            continue
        axis_role = str(value.get("axis_role") or "").lower()
        if axis_role not in {"assignment", "routing", "timepoint"}:
            continue
        if axis_role == "routing" and value.get("routes_participant") is not True:
            excluded_runtime_factors.append(
                {
                    "name": str(value.get("name") or ""),
                    "reason": "routing factor did not establish participant-level task routing",
                }
            )
            continue
        normalized_factor = {**deepcopy(value), "axis_role": axis_role}
        if axis_role != "routing":
            normalized_factor.pop("routes_participant", None)
        normalized_levels: List[Dict[str, Any]] = []
        for level in normalized_factor.get("levels", []) or []:
            if not isinstance(level, dict):
                continue
            normalized_level = deepcopy(level)
            participant_text = str(
                normalized_level.get("participant_facing_text") or ""
            ).strip()
            if participant_text:
                notes = str(normalized_level.get("implementation_notes") or "").strip()
                normalized_level["implementation_notes"] = " ".join(
                    value for value in (notes, participant_text) if value
                )
            # Planner factors encode researcher assignment/routing. Source-exact
            # participant wording belongs in blocks/items, never factor metadata.
            normalized_level["participant_facing_text"] = ""
            normalized_levels.append(normalized_level)
        normalized_factor["levels"] = normalized_levels
        runtime_factors.append(normalized_factor)
    return {
        "source_axes": deepcopy(source_axes),
        "runtime_factors": runtime_factors,
        "units": units,
        "excluded_source_units": excluded_source_units,
        "excluded_runtime_factors": excluded_runtime_factors,
    }


def _runtime_plan_contract_audit(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    oversized = [
        value
        for value in plan.get("units", []) or []
        if isinstance(value, dict)
        and isinstance(value.get("expected_item_count"), int)
        and value["expected_item_count"] > PDF_COMPILER_MAX_UNIT_ITEMS
    ]
    missing_administration_refs = [
        value
        for value in plan.get("units", []) or []
        if isinstance(value, dict) and not value.get("administration_evidence_refs")
    ]
    if not oversized and not missing_administration_refs:
        return None
    invalid = list(
        dict.fromkeys(
            str(value.get("id") or "")
            for value in [*oversized, *missing_administration_refs]
            if str(value.get("id") or "")
        )
    )
    claims = [
        {
            "path": f"$.units[?(@.id=='{value.get('id')}')].expected_item_count",
            "reason": (
                f"unit exceeds the {PDF_COMPILER_MAX_UNIT_ITEMS}-item compilation "
                "contract and must be split only at a source-supported "
                "administration boundary"
            ),
            "evidence_refs": list(value.get("evidence_refs") or []),
        }
        for value in oversized
    ]
    claims.extend(
        {
            "path": f"$.units[?(@.id=='{value.get('id')}')].administration_evidence_refs",
            "reason": "unit does not cite evidence that it was administered",
            "evidence_refs": list(value.get("evidence_refs") or []),
        }
        for value in missing_administration_refs
    )
    return {
        "status": "fail",
        "invalid_unit_ids": invalid,
        "invalid_factor_names": [],
        "missing_administered_components": [],
        "unsupported_claims": claims,
        "notes": "Deterministic runtime-plan contract failed before semantic audit.",
    }


def _audit_runtime_plan(
    llm_client: Any,
    study: Dict[str, Any],
    plan: Dict[str, Any],
    planner_source: Dict[str, Any],
    context: str,
    *,
    valid_block_ids: List[str],
    images: List[str],
    timeout: Optional[float],
    retry_delay: float,
) -> Dict[str, Any]:
    prompt = f"""Independently audit a proposed runtime-compilation plan against a paper.

This is an adversarial source audit, not a request to rationalize the plan. Mark
the plan fail unless every unit is source-supported as administered at the claimed
condition and timepoint. Administration evidence may be compositional: a method
statement can establish that participants completed the repeated task, an
exhaustive assignment/stimulus table can define each deterministic group/item
instantiation, and a sample can define the participant-facing display template.
Together those blocks support the instantiated runtime trials even when the paper
does not print every screen separately. Do not demand subject rosters, per-cell
administration prose, or duplicate printed screens. The mapping table itself is
still a non-runtime construction source, not a participant-facing item, and a
table without any independent administration statement is insufficient. A source
sentence saying tasks had two or three options does not establish a separate
baseline at every timepoint; read the surrounding administration description. A
sample/example is not an extra trial without evidence that it was administered
separately.

Factors may include participant assignment, genuine participant routing, and
repeated timepoints. Products, scenarios, questionnaire rows, and other repeated
stimuli belong in item metadata, not factors. Reject duplicate or deterministically
collinear factors that encode the same assignment/timepoint twice. Check that
units preserve source coupling: if group-specific mappings differ, do not collapse
them into one generic unit; if repeated rows are jointly administered, do not
transpose them into a factor crossing. Evidence must support administration, not
merely existence or analysis.
Also reject administrations that belong only to a different analysis target, and
reject a literal baseline/control unit inferred only from an analytic comparison
or results-table label.
`routes_participant` applies only to axis_role=routing. Assignment and timepoint
factors route by their own semantics and must not fail because that optional field
is absent. Factor-level participant_facing_text is intentionally empty because
source-exact instructions live in blocks/items; do not treat empty text as missing.

STUDY TARGET:
{json.dumps(_study_inventory(study, None), indent=2, ensure_ascii=False)}

PROPOSED PLAN:
{json.dumps(plan, indent=2, ensure_ascii=False)}

SOURCE-ONLY PLANNER INPUT:
{json.dumps(planner_source, indent=2, ensure_ascii=False)}

VALID BLOCK IDS:
{json.dumps(valid_block_ids, ensure_ascii=False)}

SOURCE BLOCKS:
{context}

Return ONLY JSON:
{{
  "status": "pass|fail",
  "invalid_unit_ids": ["unit_id"],
  "invalid_factor_names": ["factor_name"],
  "missing_administered_components": ["source-grounded missing unit"],
  "unsupported_claims": [
    {{"path": "$.units[0]", "reason": "why unsupported", "evidence_refs": ["p001_text_00001"]}}
  ],
  "notes": "short source-grounded conclusion"
}}"""
    parsed = _generate_json(
        llm_client,
        prompt,
        timeout=timeout,
        max_tokens=PDF_PLANNER_MAX_TOKENS,
        max_attempts=3,
        retry_delay=retry_delay,
        images=images,
    )
    unit_ids = {str(value.get("id") or "") for value in plan.get("units", [])}
    factor_names = {
        str(value.get("name") or "") for value in plan.get("runtime_factors", [])
    }
    invalid_units = [
        value for value in _string_list(parsed.get("invalid_unit_ids")) if value in unit_ids
    ]
    invalid_factors = [
        value
        for value in _string_list(parsed.get("invalid_factor_names"))
        if value in factor_names
    ]
    missing = _string_list(parsed.get("missing_administered_components"))
    unsupported = [
        deepcopy(value)
        for value in parsed.get("unsupported_claims", []) or []
        if isinstance(value, dict)
    ]
    status = str(parsed.get("status") or "fail").lower()
    if invalid_units or invalid_factors or missing or unsupported:
        status = "fail"
    return {
        "status": status if status in {"pass", "fail"} else "fail",
        "invalid_unit_ids": invalid_units,
        "invalid_factor_names": invalid_factors,
        "missing_administered_components": missing,
        "unsupported_claims": unsupported,
        "notes": str(parsed.get("notes") or "").strip(),
    }


def _compile_runtime_unit(
    llm_client: Any,
    study: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
    compiler_input: Dict[str, Any],
    unit: Dict[str, Any],
    context: str,
    *,
    valid_block_ids: List[str],
    images: List[str],
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
) -> Dict[str, Any]:
    unit_source = {
        "participant_facing_blocks": compiler_input.get("blocks") or [],
        "source_structures": compiler_input.get("source_structures") or [],
    }
    analysis_target = {
        "study_label": _study_name(study),
        "effects": [
            {"IV": effect.get("IV"), "DV": effect.get("DV")}
            for effect in study.get("effects", []) or []
            if isinstance(effect, dict)
        ],
    }
    prompt = f"""Compile structured source evidence into an executable study runtime.

This is a constrained compiler step, not a summarization task. The source parser
has already separated participant-facing blocks from non-runtime tables and method
structures. Return all and only the concrete response items needed to execute the
selected study and reproduce its target comparison.

Previously generated items/factors are intentionally absent because they may have
misread the assignment axes. Treat only source blocks, source structures, page
images, and the unit plan as evidence.

COMPILE ONLY THIS UNIT:
{json.dumps(unit, indent=2, ensure_ascii=False)}

COMPILATION RULES:
- Expand every source-defined trial, condition, timepoint, and alternative needed
  by this unit. A repeated table row is data, not an illustrative example.
- `items` contains only concrete response-bearing trials counted by
  expected_item_count. Put shared instructions in participant_flow; never emit an
  instruction-only text item or an unresolved [PLACEHOLDER] template.
- Preserve the unit's participant assignment bundle. Do not reorganize trials by
  a theoretical strategy/factor and do not create factor combinations absent from
  the source table. Never invent an initial baseline from a later follow-up task.
- Every closed-response item must be directly answerable: multiple_choice,
  ranking, likert, scale, slider, and matrix items require complete options,
  anchors, or rows plus columns. Never emit an empty template item.
- Return every choice option as an object. `label` contains only wording or an
  identifier actually shown to participants (for example `I`), `attributes`
  contains every participant-visible attribute/value, and `role` contains
  researcher-only target/competitor/decoy semantics. Never put `role` text in
  the label; the deterministic compiler renders label + attributes.
- Resolve level numbers through the source attribute-value table, then preserve
  concrete labels and values in option_records/attributes. Do not make a
  participant perform the table lookup at runtime.
- Option attributes contain each participant-visible named attribute exactly
  once. Do not append generic compiler fields such as "Dimension 1 value", raw
  level numbers, source columns, or duplicate copies of values to display text.
- Prefer corrected_evidence_tables over raw_evidence_tables for the same block.
  Never fill a cell listed in uncertain_cells by guessing.
- Preserve source-defined control/baseline trials as well as manipulated trials.
- Keep assignment and researcher strategy labels in condition/attributes/metadata,
  not in participant-facing wording unless the participant actually saw them.
- Reuse participant-facing wording from source blocks. A semantically
  equivalent repeated wrapper may use provenance=structured_from_source.
- A construct name, method summary, results label, or note that original wording
  is unavailable is not participant-facing stimulus text. If the paper says a
  questionnaire description/item is not reproduced or is only available
  elsewhere, omit that executable item and report the missing source field.
  Never turn an absence notice into a question or stimulus placeholder.
- Put every participant-visible stimulus, vignette, and question in `question`,
  options, scale/matrix fields, or a cited runtime block. Item-level `attributes`
  and `metadata` are researcher-only and must not hide text needed to distinguish
  one runtime trial from another.
- When a source example names one concrete category, substitute every
  category-specific noun and attribute in other table-defined instances. Never
  present one category's sample wording with another category's options.
- Cite only VALID BLOCK IDS. Never cite a source_structure id as an evidence ref.
- Do not include means, statistics, results, hypotheses, or analysis narration.
- If a damaged source table makes a concrete response option unknowable, omit
  that unusable item and report the exact missing mapping. Do not guess.
- `complete` means every source-defined executable task component for this study
  is represented. Missing font, booklet layout, random seed, or administrative
  wording is non-blocking and belongs in source_absent_fields, not missing_fields.

SELECTED ANALYSIS TARGET (SCOPE ONLY):
{json.dumps(analysis_target, indent=2, ensure_ascii=False)}

SOURCE-ONLY COMPILER INPUT:
{json.dumps(unit_source, indent=2, ensure_ascii=False)}

VALID BLOCK IDS:
{json.dumps(valid_block_ids, ensure_ascii=False)}

SOURCE BLOCKS:
{context}

Attached page images are authoritative when OCR table cells are ambiguous.

Return ONLY JSON:
{{
  "participant_flow": "corrected source-grounded flow",
  "assignment": "between-subjects|within-subjects|mixed|measured|none",
  "timepoints": [],
  "factors": [],
  "items": [
    {{
      "id": "stable_trial_id",
      "question": "participant-facing question only",
      "type": "multiple_choice|likert|scale|slider|open_ended|ranking|matrix|text",
      "options": [
        {{"label": "source-visible identifier only", "role": "target", "attributes": {{}}}}
      ],
      "scale": {{}},
      "matrix": {{}},
      "condition": {{}},
      "block": "task_block",
      "timepoint": "initial",
      "trial_group": "source-defined group",
      "attributes": {{}},
      "metadata": {{}},
      "provenance": "verbatim|structured_from_source",
      "evidence_refs": ["p001_table_00001"]
    }}
  ],
  "completeness": {{
    "status": "complete|partial|none",
    "execution_fidelity": "exact|semantically_equivalent|partial|none",
    "missing_fields": [],
    "source_absent_fields": [],
    "unresolved_visual_material": false,
    "notes": ""
  }}
}}"""
    last_error: Optional[BaseException] = None
    current_prompt = prompt
    for attempt in range(max(1, max_attempts)):
        try:
            parsed = _generate_json(
                llm_client,
                current_prompt,
                timeout=timeout,
                max_tokens=min(PDF_COMPILER_MAX_TOKENS * (2 ** attempt), 32000),
                max_attempts=1,
                retry_delay=retry_delay,
                images=images,
            )
            result = (
                parsed.get("runtime_compilation")
                if isinstance(parsed.get("runtime_compilation"), dict)
                else parsed
            )
            contract_errors = _compiled_unit_contract_errors(
                result,
                expected_item_count=unit.get("expected_item_count"),
            )
            if not contract_errors:
                source_audit = _audit_compiled_runtime_unit(
                    llm_client,
                    unit,
                    result,
                    unit_source,
                    context,
                    valid_block_ids=valid_block_ids,
                    images=images,
                    timeout=timeout,
                    retry_delay=retry_delay,
                )
                if source_audit["status"] == "pass":
                    result["_source_audit"] = source_audit
                    return result
                last_error = ValueError(
                    "Runtime unit failed source audit: "
                    + str(source_audit.get("notes") or source_audit)
                )
                current_prompt = (
                    prompt
                    + "\n\nThe prior compiled unit failed an independent cell-level "
                    + "source audit. Return a complete corrected replacement. "
                    + "Re-resolve every flagged role, level, and concrete value "
                    + "from the corrected source tables; do not merely edit the "
                    + "audit text.\n\nPRIOR COMPILED UNIT:\n"
                    + json.dumps(result, indent=2, ensure_ascii=False)
                    + "\n\nINDEPENDENT UNIT AUDIT:\n"
                    + json.dumps(source_audit, indent=2, ensure_ascii=False)
                )
                if attempt + 1 < max(1, max_attempts) and retry_delay > 0:
                    time.sleep(retry_delay)
                continue
            last_error = ValueError("; ".join(contract_errors))
            current_prompt = (
                prompt
                + "\n\nThe prior response violated the runtime option contract. "
                + "Correct every issue and return the complete unit JSON again:\n- "
                + "\n- ".join(contract_errors)
            )
        except Exception as exc:
            last_error = exc
            current_prompt = prompt + "\n\nThe prior response failed. Return only valid JSON matching the required contract."
        if attempt + 1 < max(1, max_attempts) and retry_delay > 0:
            time.sleep(retry_delay)
    assert last_error is not None
    raise last_error


def _audit_compiled_runtime_unit(
    llm_client: Any,
    unit: Dict[str, Any],
    result: Dict[str, Any],
    unit_source: Dict[str, Any],
    context: str,
    *,
    valid_block_ids: List[str],
    images: List[str],
    timeout: Optional[float],
    retry_delay: float,
) -> Dict[str, Any]:
    prompt = f"""Independently audit one bounded runtime unit against source evidence.

This is a cell-level fidelity check, not a general summary. The compiler handled
one bounded administered unit, so inspect every item and every option. When assignment
tables give researcher-role level codes and a second table maps those codes to
concrete values, independently perform that lookup for every role. Confirm
that each participant-visible option contains the resulting concrete values and
that no role was swapped. Corrected_evidence_tables and attached table images are
authoritative over noisy raw OCR.

Fail for any wrong or missing concrete value, swapped role, wrong group/product/
timepoint, omitted source-assigned trial, extra trial, leaked researcher role, or
participant-visible method/results text. Also fail display options that duplicate
the same value under generic compiler fields (for example "Dimension 1 value"),
show raw source level numbers, or expose source-table metadata. Ignore harmless
punctuation, equivalent numeric notation (for example 0.5 versus 1/2), and
semantically equivalent attribute headings. Researcher-only role metadata is
allowed outside display labels. A source sample may provide the display template
while exhaustive tables provide deterministic instances; do not require every
screen to be printed.

An item is not executable merely because the source names its construct. If its
participant-facing description, vignette, or wording is explicitly absent from
the available paper, report it in missing_items and fail. Never accept an absence
notice, availability note, construct label, or generated summary as a substitute.
All content needed to distinguish trials must be in runtime-rendered fields;
participant stimuli hidden in item attributes/metadata do not count. Distinct
items that render the same question and response contract are duplicates, not a
recovered battery.

UNIT PLAN:
{json.dumps(unit, indent=2, ensure_ascii=False)}

COMPILED UNIT:
{json.dumps(result, indent=2, ensure_ascii=False)}

SOURCE-ONLY UNIT INPUT:
{json.dumps(unit_source, indent=2, ensure_ascii=False)}

VALID BLOCK IDS:
{json.dumps(valid_block_ids, ensure_ascii=False)}

SOURCE BLOCKS:
{context}

Return ONLY JSON:
{{
  "status": "pass|fail",
  "item_errors": [
    {{
      "path": "$.items[?(@.id=='trial_id')].options[2]",
      "reason": "exact source mismatch",
      "expected": "source-grounded role and concrete values",
      "evidence_refs": ["p001_table_00001"]
    }}
  ],
  "missing_items": ["source-defined missing trial"],
  "notes": "short conclusion"
}}"""
    parsed = _generate_json(
        llm_client,
        prompt,
        timeout=timeout,
        max_tokens=PDF_UNIT_AUDITOR_MAX_TOKENS,
        max_attempts=3,
        retry_delay=retry_delay,
        images=images,
    )
    errors = [
        deepcopy(value)
        for value in parsed.get("item_errors", []) or []
        if isinstance(value, dict)
    ]
    missing = _string_list(parsed.get("missing_items"))
    status = str(parsed.get("status") or "fail").lower()
    if errors or missing:
        status = "fail"
    return {
        "status": status if status in {"pass", "fail"} else "fail",
        "item_errors": errors,
        "missing_items": missing,
        "notes": str(parsed.get("notes") or "").strip(),
    }


def _compiled_unit_contract_errors(
    result: Dict[str, Any],
    *,
    expected_item_count: Optional[int] = None,
) -> List[str]:
    errors: List[str] = []
    items = result.get("items") if isinstance(result.get("items"), list) else []
    if not items:
        errors.append("items must contain at least one concrete runtime item")
        return errors
    if expected_item_count is not None and len(items) != expected_item_count:
        errors.append(
            f"items must contain exactly {expected_item_count} source-defined "
            f"response items; received {len(items)}"
        )
    visible_fingerprints: Dict[str, List[int]] = {}
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"items[{item_index}] must be an object")
            continue
        provenance = str(item.get("provenance") or "").lower()
        if provenance not in {"verbatim", "structured_from_source"}:
            errors.append(
                f"items[{item_index}].provenance must establish source-backed "
                "participant content; placeholder/reconstructed items are not executable"
            )
        item_type = str(item.get("type") or "").lower()
        question = str(item.get("question") or "")
        if _contains_unresolved_runtime_placeholder(question):
            errors.append(f"items[{item_index}].question contains an unresolved placeholder")
        options = item.get("options") if isinstance(item.get("options"), list) else []
        if item_type in {"multiple_choice", "ranking"} and not options:
            errors.append(f"items[{item_index}].options must be non-empty")
            continue
        for option_index, option in enumerate(options):
            if not isinstance(option, dict):
                errors.append(
                    f"items[{item_index}].options[{option_index}] must be an object "
                    "with label/role/attributes"
                )
                continue
            label = str(option.get("label") or "").strip()
            if not label:
                errors.append(f"items[{item_index}].options[{option_index}].label is empty")
            role = str(option.get("role") or "").strip().lower()
            if role and re.search(rf"(?<![a-z0-9]){re.escape(role)}(?![a-z0-9])", label.lower()):
                errors.append(
                    f"items[{item_index}].options[{option_index}].label leaks its "
                    f"researcher-only role {role!r}"
                )
            if _contains_unresolved_runtime_placeholder(label):
                errors.append(
                    f"items[{item_index}].options[{option_index}].label contains "
                    "an unresolved placeholder"
                )
        attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        hidden_runtime_fields = sorted(
            key
            for key in attributes
            if re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            in {"prompt", "question", "stimulus", "stimulus_text", "vignette"}
        )
        if hidden_runtime_fields:
            errors.append(
                f"items[{item_index}].attributes hides participant-facing content "
                f"that the runtime does not render: {', '.join(hidden_runtime_fields)}"
            )
        visible = {
            "question": re.sub(r"\s+", " ", question).strip().lower(),
            "options": options,
            "scale": item.get("scale") if isinstance(item.get("scale"), dict) else {},
            "matrix": item.get("matrix") if isinstance(item.get("matrix"), dict) else {},
        }
        fingerprint = json.dumps(visible, sort_keys=True, ensure_ascii=False)
        visible_fingerprints.setdefault(fingerprint, []).append(item_index)
    for duplicate_indexes in visible_fingerprints.values():
        if len(duplicate_indexes) > 1:
            errors.append(
                "participant-visible duplicate items cannot represent distinct source "
                f"trials: {duplicate_indexes}"
            )
    return errors


def _contains_unresolved_runtime_placeholder(text: Any) -> bool:
    value = str(text or "")
    names = (
        "insert|replace|fill|product|category|attribute|condition|scenario|"
        "stimulus|question|item|option|response"
    )
    return bool(
        re.search(rf"\[(?:{names})(?:[^\]]*)\]", value, flags=re.IGNORECASE)
        or re.search(rf"\{{(?:{names})(?:[^}}]*)\}}", value, flags=re.IGNORECASE)
        or re.search(rf"<\s*(?:{names})(?:[^>]*)>", value, flags=re.IGNORECASE)
    )


def _aggregate_runtime_compilations(
    instrument: Dict[str, Any],
    plan: Dict[str, Any],
    results: Dict[str, Dict[str, Any]],
    errors: Dict[str, str],
) -> Dict[str, Any]:
    units = plan["units"]
    # A source-structure plan covers the complete runtime. Starting from prior
    # generated items would preserve exactly the assignment mistakes this stage
    # is intended to correct and could hide a failed unit behind stale content.
    items: List[Dict[str, Any]] = []
    factors: List[Dict[str, Any]] = deepcopy(plan.get("runtime_factors") or [])
    timepoints = list(instrument.get("timepoints") or [])
    participant_flow = instrument.get("participant_flow")
    assignment = _assignment_from_runtime_factors(
        factors,
        instrument.get("assignment"),
    )
    base_complete = instrument.get("completeness") if isinstance(instrument.get("completeness"), dict) else {}
    missing_fields: List[str] = []
    source_absent = list(base_complete.get("source_absent_fields") or [])
    pipeline_errors = list(base_complete.get("pipeline_errors") or [])
    notes: List[str] = []
    fidelities: List[str] = []
    statuses: List[str] = []

    for unit in units:
        unit_id = unit["id"]
        if unit_id in errors:
            pipeline_errors.append(f"Unit {unit['label']}: {errors[unit_id]}")
            continue
        result = results.get(unit_id) if isinstance(results.get(unit_id), dict) else {}
        unit_items = [deepcopy(item) for item in result.get("items", []) or [] if isinstance(item, dict)]
        for item in unit_items:
            condition = (
                deepcopy(item.get("condition"))
                if isinstance(item.get("condition"), dict)
                else {}
            )
            condition["runtime_unit_id"] = unit_id
            condition["assignment_scope"] = str(unit.get("assignment_scope") or "")
            item["condition"] = condition
            item["runtime_unit_id"] = unit_id
        expected = unit.get("expected_item_count")
        if expected is not None and len(unit_items) != expected:
            missing_fields.append(
                f"Runtime unit {unit['label']} expected {expected} source-defined items but compiled {len(unit_items)}."
            )
        items = _merge_records(items, unit_items, keys=("id",))
        timepoints = list(
            dict.fromkeys(
                _timepoint_labels([*timepoints, *(result.get("timepoints") or [])])
            )
        )
        participant_flow = result.get("participant_flow") or participant_flow
        completeness = result.get("completeness") if isinstance(result.get("completeness"), dict) else {}
        statuses.append(str(completeness.get("status") or "partial").lower())
        fidelities.append(str(completeness.get("execution_fidelity") or "partial").lower())
        missing_fields.extend(str(value) for value in completeness.get("missing_fields") or [] if str(value).strip())
        source_absent.extend(str(value) for value in completeness.get("source_absent_fields") or [] if str(value).strip())
        if completeness.get("notes"):
            notes.append(str(completeness["notes"]).strip())

    if errors:
        status = "partial"
    elif statuses and all(value == "complete" for value in statuses) and not missing_fields:
        status = "complete"
    else:
        status = "partial"
    if fidelities and all(value == "exact" for value in fidelities):
        fidelity = "exact"
    elif fidelities and all(value in {"exact", "semantically_equivalent"} for value in fidelities):
        fidelity = "semantically_equivalent"
    else:
        fidelity = "partial"
    if fidelity == "exact" and any(
        str(item.get("provenance") or "").lower() == "structured_from_source"
        for item in items
    ):
        fidelity = "semantically_equivalent"
    return {
        "participant_flow": participant_flow,
        "assignment": assignment,
        "timepoints": timepoints,
        "factors": factors,
        "items": items,
        "completeness": {
            "status": status,
            "execution_fidelity": fidelity,
            "missing_fields": list(dict.fromkeys(missing_fields)),
            "source_absent_fields": list(dict.fromkeys(source_absent)),
            "pipeline_errors": list(dict.fromkeys(pipeline_errors)),
            "compiler_audit": {
                "source_axes": deepcopy(plan.get("source_axes") or {}),
                "units": [
                    {
                        **deepcopy(unit),
                        "compiled_item_count": len(
                            (results.get(unit["id"]) or {}).get("items") or []
                        ),
                        "source_audit": deepcopy(
                            (results.get(unit["id"]) or {}).get("_source_audit") or {}
                        ),
                        "error": errors.get(unit["id"]),
                    }
                    for unit in units
                ],
                "excluded_source_units": deepcopy(plan.get("excluded_source_units") or []),
                "excluded_runtime_factors": deepcopy(
                    plan.get("excluded_runtime_factors") or []
                ),
                "plan_audit": deepcopy(plan.get("plan_audit") or {}),
                "worker_count": min(PDF_COMPILER_WORKERS, len(units)),
            },
            "unresolved_visual_material": bool(base_complete.get("unresolved_visual_material")),
            "notes": " ".join(dict.fromkeys(notes)),
        },
    }


def _assignment_from_runtime_factors(
    factors: List[Dict[str, Any]],
    fallback: Any,
) -> str:
    assignments = {
        str(factor.get("assignment") or "").lower()
        for factor in factors
        if isinstance(factor, dict)
    }
    has_between = any("between" in value for value in assignments)
    has_within = any("within" in value or "repeated" in value for value in assignments)
    if has_between and has_within:
        return "mixed"
    if has_between:
        return "between-subjects"
    if has_within:
        return "within-subjects"
    if assignments and all("measured" in value for value in assignments):
        return "measured"
    return str(fallback or "none")


def _apply_runtime_compilation(
    compiled: Dict[str, Any],
    instrument: Dict[str, Any],
    study: Dict[str, Any],
) -> Dict[str, Any]:
    candidate = deepcopy(instrument)
    for key in (
        "participant_flow",
        "assignment",
        "timepoints",
        "factors",
        "items",
        "completeness",
    ):
        if key in compiled:
            candidate[key] = deepcopy(compiled[key])
    # Blocks and source structures are immutable compiler inputs. They can only
    # be changed by the extraction/repair stages that have the full evidence.
    candidate["blocks"] = deepcopy(instrument.get("blocks") or [])
    candidate["source_structures"] = deepcopy(instrument.get("source_structures") or [])
    return normalize_instrument(candidate, study)


def _verify_instrument(
    llm_client: Any,
    instrument: Dict[str, Any],
    context: str,
    *,
    valid_block_ids: List[str],
    images: List[str],
    timeout: Optional[float],
    retry_delay: float,
) -> Dict[str, Any]:
    prompt = f"""Audit an extracted psychology study instrument against source evidence.

Judge semantics, not keywords. For every participant-facing block, item, option,
scale, matrix, and factor level, verify that its meaning is supported by the cited
evidence. Distinguish participant-facing material from method narration, reported
results, researcher notes, and generated patch instructions. `source_structures`,
`participant_flow`, and factor `implementation_notes` are explicitly non-runtime;
researcher-facing content is allowed there and must not be reported as leakage.

Fail when:
- participant-facing content is unsupported or materially rewritten;
- method/results narration is placed in runtime material;
- a factor level, item battery, option set, anchor, matrix row/column, or visual
  task component needed to execute the study is missing;
- evidence refs do not support the field they are attached to.

Do not require consent forms, demographics, attention checks, payment wording, or
administrative cover pages unless one of them is itself needed to reproduce a
reported target effect. Do not fail merely because original page layout or item
order is absent when all task semantics, conditions, options, and analyzable
responses can be reconstructed exactly from source tables. A trial expanded from
a source table may use provenance=structured_from_source and can be executable.
Audit only condition-by-stimulus combinations that the source actually assigns.
Do not infer a Cartesian product from a list of factor labels, and do not require
every strategy/level to occur for every repeated stimulus. Preserve jointly
assigned item bundles and their source timepoints.

Do not fail merely because punctuation or whitespace was normalized. A concise
task wrapper may be reconstructed only when it is marked provenance=reconstructed;
report that path as non-participant unless the PDF itself contains equivalent
participant-facing wording. This does not apply to deterministic instantiation of
a source-provided participant template: substituting source-defined category,
attribute, condition, or timepoint values without adding new task meaning is
provenance=structured_from_source and remains participant-facing. Do not flag such
template instances merely because only one filled example is printed verbatim.
Neutral option identifiers generated to distinguish complete source-defined
alternatives are semantically equivalent presentation details. Do not require the
paper's original left/right order or label-to-researcher-role mapping when every
alternative's visible attributes and researcher-only role are source-grounded.
Likewise, an empty factor-level participant_facing_text is correct when the level
name was used only for researcher assignment; do not report an empty field as
unsupported or missing.

Classify source absences by execution impact. `blocking_missing_fields` contains
only omissions that prevent a participant from receiving the manipulation/task,
answering it, or producing the response needed for the target comparison.
Randomization seeds, per-subject assignment lists, fonts, screen geometry, exact
booklet order, and administrative wording are `nonblocking_source_absences` unless
the source explicitly says they alter the manipulation or measured response.

INSTRUMENT:
{json.dumps(instrument, indent=2, ensure_ascii=False)}

VALID BLOCK IDS:
{json.dumps(valid_block_ids, ensure_ascii=False)}

SOURCE BLOCKS FOR CITED EVIDENCE:
{context}

Return ONLY JSON:
{{
  "status": "pass|fail",
  "unsupported_paths": ["$.items[0].question"],
  "non_participant_paths": ["$.blocks[0].text"],
  "blocking_missing_fields": ["complete response options"],
  "nonblocking_source_absences": ["original font and page geometry"],
  "notes": "short evidence-grounded explanation"
}}"""
    try:
        parsed = _generate_json(
            llm_client,
            prompt,
            timeout=timeout,
            max_tokens=PDF_VERIFIER_MAX_TOKENS,
            max_attempts=3,
            retry_delay=retry_delay,
            images=images,
        )
    except Exception as exc:
        return {
            "status": "error",
            "unsupported_paths": [],
            "non_participant_paths": [],
            "missing_fields": [],
            "nonblocking_source_absences": [],
            "notes": f"Verifier failed: {type(exc).__name__}: {exc}",
        }
    unsupported = _string_list(parsed.get("unsupported_paths"))
    non_participant = _runtime_non_participant_paths(
        _string_list(parsed.get("non_participant_paths"))
    )
    missing = _string_list(
        parsed.get("blocking_missing_fields")
        if isinstance(parsed.get("blocking_missing_fields"), list)
        else parsed.get("missing_fields")
    )
    nonblocking = _string_list(parsed.get("nonblocking_source_absences"))
    status = str(parsed.get("status") or "fail").lower()
    if unsupported or non_participant or missing:
        status = "fail"
    return {
        "status": status if status in {"pass", "fail"} else "fail",
        "unsupported_paths": unsupported,
        "non_participant_paths": non_participant,
        "missing_fields": missing,
        "nonblocking_source_absences": nonblocking,
        "notes": str(parsed.get("notes") or "").strip(),
    }


def _reconcile_completeness_with_verifier(
    instrument: Dict[str, Any],
    study: Dict[str, Any],
    document: Any,
    verifier: Dict[str, Any],
) -> None:
    """Let an independent passing audit demote non-semantic source absences."""
    if str(verifier.get("status") or "").lower() != "pass":
        return
    completeness = instrument.get("completeness") if isinstance(instrument.get("completeness"), dict) else {}
    if completeness.get("pipeline_errors"):
        return
    if completeness.get("unresolved_visual_material"):
        return
    fidelity = str(completeness.get("execution_fidelity") or "unknown").lower()
    if fidelity not in {"exact", "semantically_equivalent"}:
        return
    probe = build_coverage_ledger(instrument, study, document, verifier=verifier)
    semantic_blockers = set(probe.get("blocking_issues") or []) - {"pdf_material_incomplete"}
    if semantic_blockers:
        return
    missing = [
        str(item).strip()
        for item in completeness.get("missing_fields") or []
        if str(item).strip()
    ]
    source_absent = [
        str(item).strip()
        for item in completeness.get("source_absent_fields") or []
        if str(item).strip()
    ]
    verifier_absent = [
        str(item).strip()
        for item in verifier.get("nonblocking_source_absences") or []
        if str(item).strip()
    ]
    completeness["source_absent_fields"] = list(
        dict.fromkeys([*source_absent, *missing, *verifier_absent])
    )
    completeness["missing_fields"] = []
    completeness["status"] = "complete"
    note = str(completeness.get("notes") or "").strip()
    audit_note = "Independent semantic audit confirmed that remaining source absences do not change the executable response contract."
    completeness["notes"] = f"{note} {audit_note}".strip()


def _instrument_prompt(
    study: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
    context: str,
    valid_block_ids: List[str],
) -> str:
    return f"""Extract the COMPLETE participant-facing instrument for exactly one study.

The output is an evidence object that will be deterministically compiled into a
HumanStudy-Bench simulation package. Preserve the full instrument needed to run
the study; do not select only items whose words overlap a reported DV.

SOURCE RULES:
- Use Stage 1/2 only as a study/design inventory. Never copy its summaries into
  participant-facing material.
- Every block, item, factor, and factor level must cite one or more exact Block IDs.
- Use provenance=verbatim for source wording, structured_from_source when a table
  or repeated structure is transcribed or expanded without changing meaning, reconstructed
  only for an explicitly generated wrapper, and placeholder only when unusable.
- Put ONLY text actually shown to participants in blocks/items. Put method prose,
  assignment tables, codebooks, and stimulus-construction tables in
  source_structures with runtime=false; never show them as participant stimuli.
- Leave factor participant_facing_text empty when participants never saw the
  condition label/definition; put assignment semantics in implementation_notes.
- Keep answer choices and anchors out of the question string and in
  options/scale/matrix.
- For a matrix, preserve all rows and columns. For a choice/conjoint task, preserve
  every alternative and attribute needed for each choice set.
- When a table/template deterministically defines repeated trials, expand every
  trial needed to reproduce the target effect into items. Give each item condition,
  block/timepoint, attributes, complete options, provenance, and evidence refs.
  Do not stop at one illustrative sample when the source specifies a full battery.
- Transcribe complete machine-readable rows/mappings into source_structures.data;
  a summary saying "see table" is not sufficient evidence for deterministic use.
- Do not omit pretests, repeated ratings, or companion items merely because they
  are not the focal DV. Exclude only administration, consent, demographics, and
  attention checks that are not part of the experimental task.
- Never use ellipses. Never include reported means, test statistics, hypotheses,
  or statements about what the paper found as participant-facing material.
- If source information needed to reproduce the target task semantics is absent,
  do not invent it. Record it under completeness.source_absent_fields and set
  status=partial. Missing booklet typography, page layout, administrative text,
  or randomized order alone does not make an otherwise specified task partial.
- `complete` means enough source-grounded task content to reproduce the reported
  target effect; it does not require irrelevant consent/demographic wording.
- Set execution_fidelity=exact when runtime wording/layout is directly reproduced,
  semantically_equivalent when source tables/templates are expanded without
  changing task semantics, and partial when missing information changes execution.

STUDY INVENTORY:
{json.dumps(_study_inventory(study, stage1_json), indent=2, ensure_ascii=False)}

VALID EVIDENCE BLOCK IDS:
{json.dumps(valid_block_ids, ensure_ascii=False)}

SOURCE EVIDENCE:
{context}

Any attached page image is authoritative source evidence. Its filename gives the
page number (for example page_003.png corresponds to p003 blocks). Read the full
page, including tables, forms, captions, and small-print response anchors.

Return ONLY JSON:
{{
  "study_id": "study_1",
  "study_name": "Study 1",
  "participant_flow": "source-grounded participant flow",
  "assignment": "between-subjects|within-subjects|mixed|measured|none",
  "timepoints": ["initial rating", "follow-up rating"],
  "factors": [
    {{
      "name": "condition",
      "assignment": "between-subjects",
      "provenance": "verbatim|structured_from_source",
      "levels": [
        {{
          "label": "control",
          "participant_facing_text": "complete text only if participants saw it",
          "implementation_notes": "non-runtime assignment details if needed",
          "evidence_refs": ["p003_text_00010"]
        }}
      ],
      "evidence_refs": ["p003_text_00010"]
    }}
  ],
  "blocks": [
    {{
      "id": "instructions_1",
      "role": "instruction|stimulus|context|example",
      "text": "complete participant-facing content",
      "condition": null,
      "provenance": "verbatim|structured_from_source|reconstructed|placeholder",
      "evidence_refs": ["p003_text_00010"]
    }}
  ],
  "items": [
    {{
      "id": "item_1",
      "question": "question only, without answer choices appended",
      "type": "multiple_choice|likert|scale|slider|open_ended|ranking|matrix|text",
      "options": ["complete option A", "complete option B"],
      "scale": {{"min": 1, "max": 7, "anchors": {{"1": "not at all", "7": "very much"}}}},
      "matrix": {{"rows": [], "columns": [], "response_mode": "single_per_row"}},
      "condition": {{"condition": "control"}},
      "block": "main_task",
      "timepoint": "initial",
      "attributes": {{}},
      "provenance": "verbatim|structured_from_source|reconstructed|placeholder",
      "evidence_refs": ["p004_table_00020"]
    }}
  ],
  "source_structures": [
    {{
      "id": "assignment_table",
      "role": "stimulus_generation_source|method_context",
      "description": "non-runtime description",
      "data": {{}},
      "provenance": "verbatim|structured_from_source",
      "evidence_refs": ["p004_table_00020"],
      "runtime": false
    }}
  ],
  "completeness": {{
    "status": "complete|partial|none",
    "execution_fidelity": "exact|semantically_equivalent|partial|none",
    "missing_fields": [],
    "source_absent_fields": [],
    "unresolved_visual_material": false,
    "notes": ""
  }}
}}"""


def _study_inventory(study: Dict[str, Any], stage1_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    inventory = {
        "study": _study_name(study),
        "study_id": study.get("study_id"),
        "design": study.get("design") or study.get("design_type"),
        "sample": study.get("sample"),
        "effects": [
            {
                "IV": effect.get("IV"),
                "DV": effect.get("DV"),
                "materials_notes": effect.get("materials_notes"),
                "source_location": effect.get("table_or_page_location"),
            }
            for effect in study.get("effects", []) or []
            if isinstance(effect, dict)
        ],
    }
    stage1 = _stage1_experiment(stage1_json, study)
    if stage1:
        inventory["stage1"] = {
            key: stage1.get(key)
            for key in (
                "experiment_id",
                "study_id",
                "experiment_name",
                "design_type",
                "conditions_or_factors",
                "participant_task",
                "input",
                "output",
                "candidate_source_hints",
            )
        }
    return inventory


def _merge_instruments(
    base: Dict[str, Any],
    repair: Dict[str, Any],
    *,
    replace_runtime: bool = False,
) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key in ("participant_flow", "assignment"):
        if repair.get(key):
            merged[key] = repair[key]
    if replace_runtime:
        merged["timepoints"] = deepcopy(repair.get("timepoints") or [])
        merged["blocks"] = deepcopy(repair.get("blocks") or [])
        merged["items"] = deepcopy(repair.get("items") or [])
        merged["factors"] = deepcopy(repair.get("factors") or [])
    else:
        merged["timepoints"] = list(
            dict.fromkeys([*(base.get("timepoints") or []), *(repair.get("timepoints") or [])])
        )
        merged["blocks"] = _merge_records(
            base.get("blocks") or [], repair.get("blocks") or [], keys=("id", "text")
        )
        merged["items"] = _merge_records(
            base.get("items") or [], repair.get("items") or [], keys=("id", "question")
        )
        merged["factors"] = _merge_records(
            base.get("factors") or [], repair.get("factors") or [], keys=("name",)
        )
    merged["source_structures"] = _merge_records(
        base.get("source_structures") or [],
        repair.get("source_structures") or [],
        keys=("id", "description"),
    )
    base_complete = base.get("completeness") if isinstance(base.get("completeness"), dict) else {}
    repair_complete = repair.get("completeness") if isinstance(repair.get("completeness"), dict) else {}
    repair_is_complete = str(repair_complete.get("status") or "").lower() == "complete"
    merged["completeness"] = {
        **base_complete,
        **repair_complete,
        "missing_fields": list(repair_complete.get("missing_fields") or [])
        if repair_is_complete
        else list(
            dict.fromkeys([*(base_complete.get("missing_fields") or []), *(repair_complete.get("missing_fields") or [])])
        ),
        "source_absent_fields": list(repair_complete.get("source_absent_fields") or [])
        if repair_is_complete
        else list(
            dict.fromkeys(
                [*(base_complete.get("source_absent_fields") or []), *(repair_complete.get("source_absent_fields") or [])]
            )
        ),
    }
    return merged


def _has_runtime_source_structures(instrument: Dict[str, Any]) -> bool:
    return any(
        isinstance(value, dict)
        and "stimulus" in str(value.get("role") or "").lower()
        and isinstance(value.get("data"), (dict, list))
        and bool(value.get("data"))
        for value in instrument.get("source_structures", []) or []
    )


def _needs_runtime_compilation(instrument: Dict[str, Any]) -> bool:
    # A runtime source structure means that a table/template still defines the
    # executable task. Initial extraction may contain one valid example and
    # incorrectly self-report `complete`; the source-driven compiler is the only
    # step that verifies and expands the full repeated structure.
    return _has_runtime_source_structures(instrument)


def _drop_unsafe_runtime_blocks(instrument: Dict[str, Any]) -> None:
    blocks = [value for value in instrument.get("blocks", []) or [] if isinstance(value, dict)]
    removed = [
        str(value.get("id") or "")
        for value in blocks
        if str(value.get("provenance") or "").lower() in {"reconstructed", "placeholder"}
    ]
    if not removed:
        return
    instrument["blocks"] = [
        value
        for value in blocks
        if str(value.get("provenance") or "").lower()
        not in {"reconstructed", "placeholder"}
    ]
    completeness = instrument.setdefault("completeness", {})
    audit = [
        deepcopy(value)
        for value in completeness.get("runtime_filter_audit", []) or []
        if isinstance(value, dict)
    ]
    audit.append(
        {
            "removed_block_ids": removed,
            "reason": "reconstructed or placeholder text is not source-exact participant material",
        }
    )
    completeness["runtime_filter_audit"] = audit


def _missing_response_contracts(instrument: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for index, item in enumerate(instrument.get("items", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").lower() not in RESPONSE_TYPES_REQUIRING_CONTRACT:
            continue
        if not item_has_response_contract(item):
            missing.append(str(item.get("id") or index))
    return missing


def _runtime_compilation_improves(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> bool:
    before_items = [item for item in before.get("items", []) or [] if isinstance(item, dict)]
    after_items = [item for item in after.get("items", []) or [] if isinstance(item, dict)]
    if not after_items:
        return False
    before_missing = len(_missing_response_contracts(before))
    after_missing = len(_missing_response_contracts(after))
    before_executable = len(before_items) - before_missing
    after_executable = len(after_items) - after_missing
    if after_missing < before_missing and after_executable >= before_executable:
        return True
    if after_missing == before_missing == 0 and after_executable > before_executable:
        return True
    before_complete = str((before.get("completeness") or {}).get("status") or "").lower()
    after_complete = str((after.get("completeness") or {}).get("status") or "").lower()
    return (
        after_missing == 0
        and after_executable >= before_executable
        and before_complete != "complete"
        and after_complete == "complete"
    )


def _record_compiler_failure(instrument: Dict[str, Any], message: str) -> None:
    completeness = instrument.setdefault("completeness", {})
    errors = [
        str(item).strip()
        for item in completeness.get("pipeline_errors") or []
        if str(item).strip()
    ]
    errors.append(f"Structured runtime compiler: {message}")
    completeness["pipeline_errors"] = list(dict.fromkeys(errors))
    completeness["status"] = "partial"


def _source_structure_refs(instrument: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for structure in instrument.get("source_structures", []) or []:
        if isinstance(structure, dict):
            refs.extend(_string_list(structure.get("evidence_refs")))
    for block in instrument.get("blocks", []) or []:
        if isinstance(block, dict):
            refs.extend(_string_list(block.get("evidence_refs")))
    return list(dict.fromkeys(refs))


def _deduplicated_source_structures(instrument: Dict[str, Any]) -> List[Dict[str, Any]]:
    structures = deepcopy(instrument.get("source_structures") or [])
    seen_tables: set[str] = set()
    for structure in structures:
        if not isinstance(structure, dict):
            continue
        data = structure.get("data") if isinstance(structure.get("data"), dict) else None
        if data is None:
            continue
        for table_key in ("corrected_evidence_tables", "raw_evidence_tables"):
            if not isinstance(data.get(table_key), list):
                continue
            tables: List[Dict[str, Any]] = []
            for table in data[table_key]:
                if not isinstance(table, dict):
                    continue
                block_id = str(table.get("block_id") or "")
                identity = f"{table_key}:{block_id}" if block_id else ""
                if identity and identity in seen_tables:
                    tables.append({"block_id": block_id, "reuses_table_above": True})
                    continue
                if identity:
                    seen_tables.add(identity)
                tables.append(table)
            data[table_key] = tables
    return structures


def _runtime_compiler_cache_key(
    llm_client: Any,
    study: Dict[str, Any],
    compiler_input: Dict[str, Any],
    valid_block_ids: List[str],
) -> str:
    payload = {
        "version": PDF_COMPILER_CACHE_VERSION,
        "model": str(getattr(llm_client, "model", "")),
        "study": _study_name(study),
        "effects": [
            {"IV": effect.get("IV"), "DV": effect.get("DV")}
            for effect in study.get("effects", []) or []
            if isinstance(effect, dict)
        ],
        "source_tables": _compiler_cache_tables(compiler_input),
        "valid_block_ids": valid_block_ids,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compiler_cache_tables(compiler_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for structure in compiler_input.get("source_structures", []) or []:
        if not isinstance(structure, dict):
            continue
        data = structure.get("data") if isinstance(structure.get("data"), dict) else {}
        for table_key in ("corrected_evidence_tables", "raw_evidence_tables"):
            for table in data.get(table_key, []) or []:
                if not isinstance(table, dict) or table.get("reuses_table_above"):
                    continue
                block_id = str(table.get("block_id") or "")
                identity = f"{table_key}:{block_id}" if block_id else hashlib.sha256(
                    json.dumps(table, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                if identity in seen:
                    continue
                seen.add(identity)
                stable_table = deepcopy(table)
                stable_table.pop("source_image", None)
                tables.append({"table_kind": table_key, **stable_table})
    return sorted(
        tables,
        key=lambda table: (
            str(table.get("table_kind") or ""),
            str(table.get("block_id") or ""),
            json.dumps(table, sort_keys=True, ensure_ascii=False),
        ),
    )


def _read_compiler_cache(path: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != PDF_COMPILER_CACHE_VERSION:
        return None
    if payload.get("cache_key") != cache_key:
        return None
    return payload


def _write_compiler_cache(path: Path, cache_key: str, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PDF_COMPILER_CACHE_VERSION,
        "cache_key": cache_key,
        **deepcopy(value),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _timepoint_labels(values: List[Any]) -> List[str]:
    labels: List[str] = []
    for value in values:
        if isinstance(value, dict):
            label = next(
                (
                    str(value.get(key)).strip()
                    for key in ("label", "name", "id", "timepoint")
                    if value.get(key) not in (None, "")
                ),
                "",
            )
            if not label:
                label = json.dumps(value, sort_keys=True, ensure_ascii=False)
        else:
            label = str(value or "").strip()
        if label:
            labels.append(label)
    return labels


def _verifier_has_correctable_runtime_errors(verifier: Dict[str, Any]) -> bool:
    if str(verifier.get("status") or "").lower() != "fail":
        return False
    return bool(
        verifier.get("unsupported_paths")
        or verifier.get("non_participant_paths")
        or verifier.get("missing_fields")
    )


def _merge_records(base: List[Dict[str, Any]], repair: List[Dict[str, Any]], *, keys: tuple[str, ...]) -> List[Dict[str, Any]]:
    out = [deepcopy(item) for item in base if isinstance(item, dict)]
    for candidate in repair:
        if not isinstance(candidate, dict):
            continue
        index = next(
            (
                position
                for position, current in enumerate(out)
                if any(candidate.get(key) and candidate.get(key) == current.get(key) for key in keys)
            ),
            None,
        )
        if index is None:
            out.append(deepcopy(candidate))
        else:
            out[index] = {**out[index], **deepcopy(candidate)}
    return out


def _instrument_refs(instrument: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for block in instrument.get("blocks", []) or []:
        refs.extend(_string_list(block.get("evidence_refs")))
    for item in instrument.get("items", []) or []:
        refs.extend(_string_list(item.get("evidence_refs")))
    for factor in instrument.get("factors", []) or []:
        refs.extend(_string_list(factor.get("evidence_refs")))
        for level in factor.get("levels", []) or []:
            if isinstance(level, dict):
                refs.extend(_string_list(level.get("evidence_refs")))
    for structure in instrument.get("source_structures", []) or []:
        if isinstance(structure, dict):
            refs.extend(_string_list(structure.get("evidence_refs")))
    return list(dict.fromkeys(refs))


def _link_document_tables(
    instrument: Dict[str, Any],
    document: Any,
    llm_client: Any,
    study: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
    *,
    evidence_block_ids: List[str],
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
    cache_dir: Optional[Path],
    use_cache: bool,
) -> None:
    """Classify Docling tables independently of the initial instrument draft."""
    allowed_ids = set(evidence_block_ids)
    all_table_ids: set[str] = set()
    candidates: List[Dict[str, Any]] = []
    blocks = list(document.blocks)
    by_order = {block.order: block for block in blocks}
    for block in blocks:
        table = block.metadata.get("table") if isinstance(block.metadata, dict) else None
        if not isinstance(table, dict):
            continue
        all_table_ids.add(block.block_id)
        if block.block_id not in allowed_ids:
            continue
        rows = deepcopy(table.get("rows") or [])
        row_preview = rows if len(rows) <= 16 else [*rows[:12], *rows[-4:]]
        neighbors: List[str] = []
        for offset in (-2, -1, 1, 2):
            neighbor = by_order.get(block.order + offset)
            if neighbor is None or abs(neighbor.page_start - block.page_start) > 1:
                continue
            text_value = str(neighbor.text or "").strip()
            if text_value:
                neighbors.append(text_value[:1200])
        candidates.append(
            {
                "block_id": block.block_id,
                "page": block.page_start,
                "section_path": list(block.section_path),
                "columns": deepcopy(table.get("columns") or []),
                "row_count": len(rows),
                "row_preview": row_preview,
                "table_text": str(block.text or "")[:3000],
                "nearby_text": neighbors,
            }
        )
    if not candidates:
        return
    candidates.sort(key=lambda value: str(value["block_id"]))
    cache_key = _table_linker_cache_key(document, llm_client, study, candidates)
    cache_path = (
        cache_dir / f"{_study_id(study)}.json" if cache_dir is not None else None
    )
    cached = (
        _read_table_linker_cache(cache_path, cache_key)
        if cache_path is not None and use_cache
        else None
    )
    if cached is not None:
        classifications = cached["classifications"]
        print(f"      table linker cache hit: {_study_name(study)}", flush=True)
    else:
        prompt = f"""Classify every parsed table in a psychology paper for runtime use.

This table-linking step is independent of the initial extraction draft. For each
candidate, decide whether it contains participant-facing items, concrete stimulus
values, condition/assignment mappings, administration mappings, a source template
used to instantiate runtime material, reported results/analysis, or unrelated
content. Results tables contain observed outcomes, means, tests, coefficients, or
analysis summaries; never use them to construct participant-facing material.

Link a table only when its rows/cells are needed to present the manipulation,
response task, options, scale, repeated trials, condition routing, or source-defined
stimulus values. A method/assignment table may be linked even if participants did
not see it; it remains runtime=false and is used only by the compiler. In the
classification output, runtime_relevant means "needed by the runtime compiler",
not "directly shown to participants", so assignment/stimulus tables may be true.
Classify
every candidate exactly once. Do not infer table content absent from the preview.

STUDY INVENTORY:
{json.dumps(_study_inventory(study, stage1_json), indent=2, ensure_ascii=False)}

INITIAL DRAFT TABLE REFS (UNTRUSTED HINTS):
{json.dumps(sorted(set(_instrument_refs(instrument)) & all_table_ids), ensure_ascii=False)}

CANDIDATE TABLES:
{json.dumps(candidates, indent=2, ensure_ascii=False)}

Return ONLY JSON:
{{
  "classifications": [
    {{
      "block_id": "p001_table_00001",
      "role": "participant_item|stimulus_values|assignment|administration|source_template|results_or_analysis|unrelated",
      "runtime_relevant": true,
      "reason": "short source-grounded reason"
    }}
  ]
}}"""
        current_prompt = prompt
        classifications = []
        expected_ids = {str(value["block_id"]) for value in candidates}
        last_missing: List[str] = []
        for attempt in range(max(2, int(max_attempts))):
            parsed = _generate_json(
                llm_client,
                current_prompt,
                timeout=timeout,
                max_tokens=PDF_TABLE_LINKER_MAX_TOKENS,
                max_attempts=1,
                retry_delay=retry_delay,
            )
            raw = parsed.get("classifications")
            classifications = [
                deepcopy(value)
                for value in raw or []
                if isinstance(value, dict)
                and str(value.get("block_id") or "") in expected_ids
            ] if isinstance(raw, list) else []
            seen = {str(value.get("block_id") or "") for value in classifications}
            last_missing = sorted(expected_ids - seen)
            if not last_missing:
                break
            current_prompt = (
                prompt
                + "\n\nThe prior response omitted candidate table IDs. Return the complete "
                + "classification list including: "
                + json.dumps(last_missing, ensure_ascii=False)
            )
            if retry_delay > 0 and attempt + 1 < max(2, int(max_attempts)):
                time.sleep(retry_delay)
        if last_missing:
            raise ValueError(
                "Table linker omitted candidate tables after retry: "
                + ", ".join(last_missing)
            )
        if cache_path is not None:
            _write_table_linker_cache(cache_path, cache_key, classifications)

    relevant_roles = {
        "participant_item",
        "stimulus_values",
        "assignment",
        "administration",
        "source_template",
    }
    linked = [
        value
        for value in classifications
        if str(value.get("role") or "").lower() in relevant_roles
    ]
    linked_ids = {str(value.get("block_id") or "") for value in linked}

    retained_structures: List[Dict[str, Any]] = []
    for structure in instrument.get("source_structures", []) or []:
        if not isinstance(structure, dict):
            continue
        refs = set(_string_list(structure.get("evidence_refs")))
        table_refs = refs & all_table_ids
        # Once the deterministic linker has classified document tables, discard
        # LLM-transcribed table structures wholesale. Their prose/data may mix
        # linked stimulus cells with unlinked result cells; the exact Docling
        # rows below are the sole table input to the runtime compiler.
        if table_refs:
            continue
        retained_structures.append(structure)
    instrument["source_structures"] = retained_structures
    if not linked_ids:
        return

    by_id = document.block_map()
    raw_tables: List[Dict[str, Any]] = []
    for block_id in sorted(linked_ids):
        block = by_id.get(block_id)
        table = block.metadata.get("table") if block and isinstance(block.metadata, dict) else None
        if not isinstance(table, dict):
            continue
        raw_tables.append(
            {
                "block_id": block_id,
                "columns": deepcopy(table.get("columns") or []),
                "rows": deepcopy(table.get("rows") or []),
            }
        )
    linked_structure = {
        "id": "document_linked_runtime_tables",
        "role": "stimulus_generation_source_candidates",
        "description": "Docling table inventory linked independently to executable study material.",
        "provenance": "structured_from_source",
        "evidence_refs": sorted(linked_ids),
        "runtime": False,
        "data": {
            "table_linker": deepcopy(classifications),
            "raw_evidence_tables": raw_tables,
        },
    }
    instrument["source_structures"] = _merge_records(
        instrument.get("source_structures") or [],
        [linked_structure],
        keys=("id",),
    )


def _table_linker_cache_key(
    document: Any,
    llm_client: Any,
    study: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> str:
    payload = {
        "version": PDF_TABLE_LINKER_CACHE_VERSION,
        "source_sha256": str(document.source_sha256),
        "model": str(getattr(llm_client, "model", "")),
        "study": _study_name(study),
        "effects": [
            {"IV": effect.get("IV"), "DV": effect.get("DV")}
            for effect in study.get("effects", []) or []
            if isinstance(effect, dict)
        ],
        "candidates": candidates,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_table_linker_cache(path: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != PDF_TABLE_LINKER_CACHE_VERSION:
        return None
    if payload.get("cache_key") != cache_key:
        return None
    classifications = payload.get("classifications")
    return payload if isinstance(classifications, list) else None


def _write_table_linker_cache(
    path: Path,
    cache_key: str,
    classifications: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PDF_TABLE_LINKER_CACHE_VERSION,
        "cache_key": cache_key,
        "classifications": deepcopy(classifications),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _attach_structured_evidence(instrument: Dict[str, Any], document: Any) -> None:
    """Embed exact Docling table rows behind every referenced source structure."""
    by_id = document.block_map()
    for structure in instrument.get("source_structures", []) or []:
        if not isinstance(structure, dict):
            continue
        tables: List[Dict[str, Any]] = []
        for ref in _string_list(structure.get("evidence_refs")):
            block = by_id.get(ref)
            if block is None:
                continue
            table = block.metadata.get("table") if isinstance(block.metadata, dict) else None
            if not isinstance(table, dict):
                continue
            tables.append(
                {
                    "block_id": ref,
                    "columns": deepcopy(table.get("columns") or []),
                    "rows": deepcopy(table.get("rows") or []),
                }
            )
        if not tables:
            continue
        data = structure.get("data") if isinstance(structure.get("data"), dict) else {}
        structure["data"] = {
            **deepcopy(data),
            "raw_evidence_tables": tables,
        }


def _recover_complex_source_tables(
    instrument: Dict[str, Any],
    document: Any,
    llm_client: Any,
    *,
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
    cache_dir: Optional[Path],
    use_cache: bool,
) -> None:
    tables: Dict[str, Dict[str, Any]] = {}
    for structure in instrument.get("source_structures", []) or []:
        if not isinstance(structure, dict) or "stimulus" not in str(structure.get("role") or "").lower():
            continue
        data = structure.get("data") if isinstance(structure.get("data"), dict) else {}
        for table in data.get("raw_evidence_tables", []) or []:
            if not isinstance(table, dict):
                continue
            block_id = str(table.get("block_id") or "")
            if block_id and _table_needs_vision_recovery(table):
                tables.setdefault(block_id, table)
    if not tables:
        return

    recovered: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}
    block_map = document.block_map()
    for block_id, raw_table in tables.items():
        block = block_map.get(block_id)
        if block is None:
            failures[block_id] = "Referenced table block is absent from parsed document."
            continue
        cache_key = _table_recovery_cache_key(document, llm_client, block_id, raw_table)
        cached = (
            _read_table_recovery_cache(cache_dir / f"{block_id}.json", cache_key)
            if cache_dir is not None and use_cache
            else None
        )
        if isinstance(cached, dict) and isinstance(cached.get("recovered_table"), dict):
            recovered[block_id] = deepcopy(cached["recovered_table"])
            local_image = _table_crop_image(document, block, cache_dir)
            if local_image is not None:
                recovered[block_id]["source_image"] = str(local_image)
            print(f"      table recovery cache hit: {block_id}", flush=True)
            continue
        image = _table_crop_image(document, block, cache_dir)
        if image is None:
            failures[block_id] = "No page image/bbox was available for visual table recovery."
            continue
        print(
            f"      recovering complex table: {block_id} "
            f"({len(raw_table.get('columns') or [])} columns)",
            flush=True,
        )
        try:
            value = _recover_table_grid(
                llm_client,
                block_id,
                raw_table,
                image,
                timeout=timeout,
                max_attempts=max(3, max_attempts),
                retry_delay=retry_delay,
            )
            recovered[block_id] = value
            if cache_dir is not None:
                _write_table_recovery_cache(
                    cache_dir / f"{block_id}.json",
                    cache_key,
                    value,
                )
        except Exception as exc:
            failures[block_id] = f"{type(exc).__name__}: {exc}"
            print(
                f"      table recovery failed: {block_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    for structure in instrument.get("source_structures", []) or []:
        if not isinstance(structure, dict):
            continue
        data = structure.get("data") if isinstance(structure.get("data"), dict) else {}
        raw_ids = {
            str(table.get("block_id") or "")
            for table in data.get("raw_evidence_tables", []) or []
            if isinstance(table, dict)
        }
        corrected = [deepcopy(recovered[block_id]) for block_id in raw_ids if block_id in recovered]
        if corrected:
            data["corrected_evidence_tables"] = corrected
        table_errors = [
            {"block_id": block_id, "error": failures[block_id]}
            for block_id in raw_ids
            if block_id in failures
        ]
        if table_errors:
            data["table_recovery_errors"] = table_errors
        structure["data"] = data


def _table_needs_vision_recovery(table: Dict[str, Any]) -> bool:
    columns = table.get("columns") if isinstance(table.get("columns"), list) else []
    rows = table.get("rows") if isinstance(table.get("rows"), list) else []
    if len(columns) >= 10:
        return True
    if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
        return True
    empty_headers = sum(not str(value or "").strip() for value in columns)
    return bool(columns and empty_headers / len(columns) >= 0.25)


def _recover_table_grid(
    llm_client: Any,
    block_id: str,
    raw_table: Dict[str, Any],
    image: Path,
    *,
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
) -> Dict[str, Any]:
    prompt = f"""Transcribe one complex experimental-material table from its cropped source image.

The image is authoritative. The Docling OCR below is a noisy hint and may have
merged or shifted columns. Recover the logical grid exactly as printed.

RULES:
- Flatten multi-row headers into unique column labels while preserving the source
  hierarchy (for example `Condition A | Alternative 1`).
- Superscript letters/numbers attached to a header are footnote markers, not part
  of the canonical category/group label. Strip them from the column label and
  record them under footnote_markers; never turn `Condition A` with superscript
  `a` into a new `Condition Aa` level.
- Every output row must contain exactly one cell per output column.
- Preserve row order, labels, symbols, numbers, currency, fractions, and blanks.
- Carry a product/block label into its associated subrows when that makes the
  logical record explicit; do not invent values.
- For a visibly merged cell spanning several subcolumns, write the label once in
  the central/most natural logical cell and leave the other spanned cells blank;
  never duplicate the label across all covered columns.
- Do not summarize and do not calculate new values.
- Record every unreadable or genuinely ambiguous cell in uncertain_cells. Use an
  empty string for that cell instead of guessing.
- `confidence=high` only when all task-defining cells are legible.

BLOCK ID: {block_id}

NOISY DOCLING TABLE:
{json.dumps(raw_table, indent=2, ensure_ascii=False)}

Return ONLY JSON:
{{
  "block_id": "{block_id}",
  "columns": ["logical column 1", "logical column 2"],
  "rows": [["cell", "cell"]],
  "confidence": "high|medium|low",
  "uncertain_cells": [
    {{"row": 0, "column": "logical column 2", "visible_text": "", "reason": "unreadable"}}
  ],
  "footnote_markers": [
    {{"marker": "a", "attached_to": "Condition A", "text": "footnote text if visible"}}
  ],
  "notes": "short transcription note"
}}"""
    parsed = _generate_json(
        llm_client,
        prompt,
        timeout=timeout,
        max_tokens=PDF_TABLE_RECOVERY_MAX_TOKENS,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        images=[str(image)],
    )
    columns = [str(value or "").strip() for value in parsed.get("columns") or []]
    rows = parsed.get("rows") if isinstance(parsed.get("rows"), list) else []
    if len(columns) < 2 or not rows:
        raise ValueError("Recovered table has no usable columns/rows")
    normalized_rows: List[List[str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(columns):
            raise ValueError(
                f"Recovered table row {index} has {len(row) if isinstance(row, list) else 'invalid'} "
                f"cells for {len(columns)} columns"
            )
        normalized_rows.append([str(value or "").strip() for value in row])
    uncertain = [
        deepcopy(value)
        for value in parsed.get("uncertain_cells") or []
        if isinstance(value, dict)
    ]
    uncertain_columns = {
        str(value.get("column") or "")
        for value in uncertain
        if str(value.get("column") or "")
    }
    keep_columns = [
        index
        for index, column in enumerate(columns)
        if any(row[index] for row in normalized_rows) or column in uncertain_columns
    ]
    if len(keep_columns) != len(columns):
        columns = [columns[index] for index in keep_columns]
        normalized_rows = [
            [row[index] for index in keep_columns]
            for row in normalized_rows
        ]
    confidence = str(parsed.get("confidence") or "low").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    adjudication: List[Dict[str, Any]] = []
    if uncertain:
        try:
            adjudication = _adjudicate_uncertain_table_cells(
                llm_client,
                block_id,
                columns,
                normalized_rows,
                uncertain,
                image,
                timeout=timeout,
                retry_delay=retry_delay,
            )
        except Exception as exc:
            adjudication = [
                {
                    "row": value.get("row"),
                    "column": value.get("column"),
                    "status": "unresolved",
                    "value": "",
                    "confidence": "low",
                    "reason": f"adjudication failed: {type(exc).__name__}: {exc}",
                }
                for value in uncertain
            ]
        resolutions = {
            (value.get("row"), str(value.get("column") or "")): value
            for value in adjudication
            if isinstance(value, dict)
        }
        unresolved: List[Dict[str, Any]] = []
        for cell in uncertain:
            row = cell.get("row")
            column = str(cell.get("column") or "")
            resolution = resolutions.get((row, column), {})
            column_index = columns.index(column) if column in columns else None
            valid_position = (
                isinstance(row, int)
                and 0 <= row < len(normalized_rows)
                and column_index is not None
            )
            resolved = (
                resolution.get("status") == "resolved"
                and str(resolution.get("confidence") or "").lower() == "high"
                and str(resolution.get("value") or "").strip()
                and valid_position
            )
            if resolved:
                normalized_rows[row][column_index] = str(resolution["value"]).strip()
                continue
            if valid_position:
                normalized_rows[row][column_index] = ""
            unresolved.append(cell)
        uncertain = unresolved
        if not uncertain:
            confidence = "high"
    footnote_markers = [
        deepcopy(value)
        for value in parsed.get("footnote_markers") or []
        if isinstance(value, dict)
    ]
    return {
        "block_id": block_id,
        "columns": columns,
        "rows": normalized_rows,
        "confidence": confidence,
        "uncertain_cells": uncertain,
        "uncertainty_adjudication": adjudication,
        "footnote_markers": footnote_markers,
        "notes": str(parsed.get("notes") or "").strip(),
        "source": (
            "vision_table_recovery_with_adjudication"
            if adjudication
            else "vision_table_recovery"
        ),
        "source_image": str(image),
    }


def _adjudicate_uncertain_table_cells(
    llm_client: Any,
    block_id: str,
    columns: List[str],
    rows: List[List[str]],
    uncertain_cells: List[Dict[str, Any]],
    image: Path,
    *,
    timeout: Optional[float],
    retry_delay: float,
) -> List[Dict[str, Any]]:
    prompt = f"""Resolve only the flagged uncertain cells in a source table image.

Inspect the cropped image directly. The prior transcription and visible_text are
hints, not authority. Mark a cell resolved only when its printed value is clearly
legible at high confidence. Otherwise mark it unresolved and return an empty
value. Preserve currency, decimals, fractions, symbols, and capitalization. Do
not infer a value from a sequence or experimental design.

BLOCK ID: {block_id}
COLUMNS: {json.dumps(columns, ensure_ascii=False)}
TRANSCRIBED ROWS (zero-based):
{json.dumps(rows, indent=2, ensure_ascii=False)}
FLAGGED CELLS:
{json.dumps(uncertain_cells, indent=2, ensure_ascii=False)}

Return ONLY JSON:
{{
  "resolutions": [
    {{
      "row": 0,
      "column": "exact column label",
      "status": "resolved|unresolved",
      "value": "exact printed value or empty",
      "confidence": "high|medium|low",
      "reason": "brief visual evidence"
    }}
  ]
}}"""
    parsed = _generate_json(
        llm_client,
        prompt,
        timeout=timeout,
        max_tokens=PDF_TABLE_ADJUDICATOR_MAX_TOKENS,
        max_attempts=3,
        retry_delay=retry_delay,
        images=[str(image)],
    )
    resolutions = [
        deepcopy(value)
        for value in parsed.get("resolutions", []) or []
        if isinstance(value, dict)
    ]
    expected = {
        (value.get("row"), str(value.get("column") or ""))
        for value in uncertain_cells
    }
    returned = {
        (value.get("row"), str(value.get("column") or ""))
        for value in resolutions
    }
    if not expected.issubset(returned):
        raise ValueError("Table uncertainty adjudicator omitted one or more flagged cells")
    return resolutions


def _table_crop_image(document: Any, block: Any, cache_dir: Optional[Path]) -> Optional[Path]:
    image_path = Path(str((block.metadata or {}).get("page_image") or ""))
    if not image_path.exists():
        return None
    if not block.bbox or len(block.bbox) != 4:
        return image_path
    target_dir = cache_dir / "crops" if cache_dir is not None else image_path.parent / "table_crops"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{block.block_id}.v2.png"
    if target.exists():
        return target
    try:
        import pypdfium2 as pdfium
        from PIL import Image

        pdf = pdfium.PdfDocument(str(document.source_file))
        page = pdf[int(block.page_start) - 1]
        page_width, page_height = page.get_size()
        page.close()
        pdf.close()
        with Image.open(image_path) as source:
            width, height = source.size
            left, y1, right, y2 = [float(value) for value in block.bbox]
            top_pdf = max(y1, y2)
            bottom_pdf = min(y1, y2)
            x0 = left / page_width * width
            x1 = right / page_width * width
            top = (page_height - top_pdf) / page_height * height
            bottom = (page_height - bottom_pdf) / page_height * height
            pad_x = max(8, int((x1 - x0) * 0.025))
            pad_y = max(8, int((bottom - top) * 0.04))
            box = (
                max(0, int(x0) - pad_x),
                max(0, int(top) - pad_y),
                min(width, int(x1) + pad_x),
                min(height, int(bottom) + pad_y),
            )
            crop = source.crop(box)
            if crop.width < 100 or crop.height < 60:
                return image_path
            if crop.width < 1800:
                scale = min(3.0, 1800 / crop.width)
                crop = crop.resize(
                    (int(crop.width * scale), int(crop.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            crop.save(target)
        return target
    except Exception:
        return image_path


def _table_recovery_cache_key(
    document: Any,
    llm_client: Any,
    block_id: str,
    raw_table: Dict[str, Any],
) -> str:
    payload = {
        "version": PDF_TABLE_RECOVERY_CACHE_VERSION,
        "source_sha256": str(document.source_sha256),
        "model": str(getattr(llm_client, "model", "")),
        "block_id": block_id,
        "raw_table": raw_table,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _read_table_recovery_cache(path: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != PDF_TABLE_RECOVERY_CACHE_VERSION:
        return None
    if payload.get("cache_key") != cache_key:
        return None
    return payload


def _write_table_recovery_cache(
    path: Path,
    cache_key: str,
    recovered_table: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PDF_TABLE_RECOVERY_CACHE_VERSION,
        "cache_key": cache_key,
        "recovered_table": deepcopy(recovered_table),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _generate_json(
    llm_client: Any,
    prompt: str,
    *,
    timeout: Optional[float],
    max_tokens: int,
    max_attempts: int,
    retry_delay: float,
    images: Optional[List[str]] = None,
) -> Dict[str, Any]:
    last_error: Optional[BaseException] = None
    current_prompt = prompt
    for attempt in range(max(1, int(max_attempts))):
        attempt_max_tokens = min(max_tokens * (2 ** attempt), 32000)
        try:
            response = call_with_timeout(
                lambda: _generate_once(
                    llm_client,
                    current_prompt,
                    timeout,
                    attempt_max_tokens,
                    images=images or [],
                ),
                timeout=timeout,
            )
            response_text = str(response or "").strip()
            if not response_text:
                raise ValueError(
                    "LLM returned empty content; the completion budget may have "
                    f"been exhausted (max_tokens={attempt_max_tokens})"
                )
            return _loads_json(response_text)
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max(1, int(max_attempts)):
                break
            current_prompt = prompt + "\n\nThe prior response failed JSON validation. Return only the requested JSON object."
            if retry_delay > 0:
                time.sleep(retry_delay)
    assert last_error is not None
    raise last_error


def _generate_once(
    llm_client: Any,
    prompt: str,
    timeout: Optional[float],
    max_tokens: int,
    *,
    images: List[str],
) -> Any:
    if images and hasattr(llm_client, "generate_text"):
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "text",
                    "text": f"Attached source page image: {Path(image).name}",
                }
            )
            content.append({"type": "image", "image": image})
        return llm_client.generate_text(
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    try:
        return llm_client.generate_content(prompt=prompt, timeout=timeout, max_tokens=max_tokens)
    except TypeError:
        try:
            return llm_client.generate_content(prompt=prompt, timeout=timeout)
        except TypeError:
            return llm_client.generate_content(prompt=prompt)


def _loads_json(text: str) -> Dict[str, Any]:
    value = str(text or "").strip()
    if "```json" in value:
        value = value.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in value:
        value = value.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group())
    if not isinstance(parsed, dict):
        raise ValueError("PDF instrument response must be a JSON object")
    return parsed


def _instrument_object(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(parsed.get("instrument"), dict):
        return parsed["instrument"]
    if isinstance(parsed.get("instruments"), list):
        first = next((item for item in parsed["instruments"] if isinstance(item, dict)), None)
        if first is not None:
            return first
    return parsed


def _selected_or_all_studies(stage_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    studies = [
        study
        for study in stage_json.get("eligible_studies", []) or stage_json.get("studies", []) or []
        if isinstance(study, dict)
    ]
    selected = [
        study
        for study in studies
        if not isinstance(study.get("selection"), dict) or study["selection"].get("keep") is not False
    ]
    return selected or studies


def _study_name(study: Dict[str, Any]) -> str:
    return str(
        study.get("study")
        or study.get("study_id")
        or study.get("experiment_id")
        or study.get("study_name")
        or "Study"
    )


def _study_id(study: Dict[str, Any]) -> str:
    return canonical_sub_study_id(_study_name(study))


def _string_list(value: Any) -> List[str]:
    return [str(item).strip() for item in value or [] if str(item).strip()] if isinstance(value, list) else []


def _runtime_non_participant_paths(paths: List[str]) -> List[str]:
    runtime: List[str] = []
    for path in paths:
        if re.match(r"^\$\.blocks\[(?:\d+|['\"][^'\"]+['\"])\](?:\.|$)", path):
            runtime.append(path)
            continue
        if re.match(
            r"^\$\.items\[(?:\d+|['\"][^'\"]+['\"])\]"
            r"(?:$|\.(?:question|options|scale|matrix|response_format)(?:\.|$))",
            path,
        ):
            runtime.append(path)
            continue
        if re.match(
            r"^\$\.factors\[(?:\d+|['\"][^'\"]+['\"])\]\.levels\[\d+\]"
            r"\.participant_facing_text(?:\.|$)",
            path,
        ):
            runtime.append(path)
    return list(dict.fromkeys(runtime))


def _visual_images(
    document: Any,
    block_ids: List[str],
    *,
    limit: Optional[int] = 12,
) -> List[str]:
    by_id = document.block_map()
    images: List[str] = []
    for block_id in block_ids:
        block = by_id.get(str(block_id))
        if block is None:
            continue
        image = str(block.metadata.get("page_image") or "")
        if image and image not in images:
            images.append(image)
        if limit is not None and len(images) >= limit:
            break
    return images


def _context_visual_images(document: Any, block_ids: List[str]) -> List[str]:
    # A scan contains no alternate text representation, so silently dropping
    # later pages would make completeness impossible to audit. Born-digital
    # papers keep a bounded visual supplement because their text blocks remain
    # the primary evidence.
    parser = str(getattr(document, "parser", ""))
    limit = None if parser in {"docling_vision", "docling_ocr"} else 12
    return _visual_images(document, block_ids, limit=limit)
