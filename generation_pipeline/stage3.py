from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from generation_pipeline.identifiers import canonical_sub_study_id
from generation_pipeline.parsers.effect_consolidator import annotate_study
from generation_pipeline.parsers.material_assembler import assemble_study_materials
from generation_pipeline.parsers.source_linker import LinkResult, link_sources
from generation_pipeline.parsers.stage3_adapter import match_stage3_study
from generation_pipeline.pdf.provider import extract_pdf_study_materials
from generation_pipeline.stage3_material_contract import apply_material_contracts
from generation_pipeline.stage3_pdf_materials import (
    PDF_TEXT_MAX_CHARS,
    extract_pdf_study_materials as extract_linked_source_materials,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _eligible_of(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return data.get("eligible_studies") or data.get("studies") or []


# --------------------------------------------------------------------------- #
# Unified Stage 3 entry: stage2.json -> stage3.json (+ stage3.md)
# --------------------------------------------------------------------------- #

def _find_stage_file(paper_dir: Path, stem: str) -> Optional[Path]:
    """Locate a stage file by canonical name only."""
    canonical = paper_dir / f"{stem}.json"
    if canonical.exists():
        return canonical
    return None


def _find_pdf(paper_dir: Path) -> Optional[Path]:
    pdfs = sorted(p for p in paper_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def _resolve_osf_files_dir(paper_dir: Path, osf_files_dir: Optional[Path]) -> Path:
    """Resolve a Stage 3 OSF/source directory to the actual files directory."""
    candidates: List[Path] = []
    if osf_files_dir is not None:
        src = Path(osf_files_dir)
        candidates.extend(
            [
                src / "osf" / "files",
                src / "sources" / "osf" / "files",
                src / "files",
                src,
            ]
        )
    candidates.extend(
        [
            paper_dir / "sources" / "osf" / "files",
            paper_dir / "osf" / "files",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return candidates[0] if candidates else paper_dir / "sources" / "osf" / "files"


def run_stage3(
    paper_dir: str | Path,
    *,
    stage2_path: Optional[Path] = None,
    pdf_path: Optional[Path] = None,
    osf_files_dir: Optional[Path] = None,
    filler: Any = None,
    llm_client: Any = None,
    backup: bool = True,
    write: bool = True,
    allow_effect_slot_fallback: bool = False,
    selection_votes: int = 3,
    selection_timeout: Optional[float] = 60.0,
    pdf_material_timeout: Optional[float] = 120.0,
) -> Dict[str, Any]:
    """Unified Stage 3 for one paper: produce a canonical `stage3.json` + `stage3.md`.

    Unlike the legacy corpus patch runner (which rewrites the Stage 2 file in
    place), this writes Stage 3 as its OWN artifact, so the pipeline contract is
    `stage2.json -> stage3.json (+ .md)`:

      1. Load stage2.json.
      2. Optionally slot-fill from the PDF/sources when a `filler` is explicitly
         given. Stage 2 is still left untouched.
      3. Write stage3.json.
      4. Run OSF material assembly + effect consolidation + (optional) study
         selection, enriching stage3.json (`study_materials`, consolidation, …).
      5. Write a human-readable stage3.md summary.

    Returns {stage3_json, stage3_md, materials}.
    """
    paper_dir = Path(paper_dir)
    stage2_path = stage2_path or _find_stage_file(paper_dir, "stage2")
    if stage2_path is None or not Path(stage2_path).exists():
        raise FileNotFoundError(f"stage2.json not found under {paper_dir}")
    stage3_path = paper_dir / "stage3.json"

    paper = json.loads(Path(stage2_path).read_text(encoding="utf-8"))

    # --- slot filling (stage2 -> stage3 content), never in place on stage2 ----
    if filler is not None:
        pdf_path = pdf_path or _find_pdf(paper_dir)
        if pdf_path is None:
            raise FileNotFoundError(f"No PDF found under {paper_dir} for slot filling")
        from generation_pipeline.patchers.patch_runner import discover_source_dirs
        from generation_pipeline.verification.schema_validator import validate_paper

        source_dirs = discover_source_dirs(Path(stage2_path), Path(pdf_path))
        patched = filler.patch_paper(paper, Path(pdf_path), source_dirs=source_dirs or None)
        patched, _ = validate_paper(patched, repair=True, path=stage3_path)
        data = patched
    else:
        data = paper

    if write:
        _atomic_write(stage3_path, data, backup=backup)

    # --- OSF assembly + consolidation + selection (enriches stage3.json) ------
    materials = build_study_materials(
        paper_dir,
        stage3_path=stage3_path,
        osf_files_dir=osf_files_dir,
        pdf_path=pdf_path,
        llm_client=llm_client,
        write=write,
        allow_effect_slot_fallback=allow_effect_slot_fallback,
        selection_votes=selection_votes,
        selection_timeout=selection_timeout,
        pdf_material_timeout=pdf_material_timeout,
    )

    stage3_md = paper_dir / "stage3.md"
    if write:
        # reload to pick up the enrichment build_study_materials wrote back
        enriched = json.loads(stage3_path.read_text(encoding="utf-8"))
        stage3_md.write_text(_render_stage3_md(enriched, materials), encoding="utf-8")

    return {"stage3_json": stage3_path, "stage3_md": stage3_md, "materials": materials}


def _atomic_write(path: Path, data: Dict[str, Any], *, backup: bool) -> None:
    if backup and path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _render_stage3_md(data: Dict[str, Any], materials: Dict[str, Dict[str, Any]]) -> str:
    """Human-readable Stage 3 summary (the canonical stage3.md)."""
    title = data.get("paper_title") or data.get("paper_id") or "(untitled)"
    lines = [f"# Stage 3 — {title}", ""]

    cons = data.get("effect_consolidation_summary")
    if cons:
        lines += [
            "## Effects",
            f"- raw effects: {cons.get('raw_effects')}",
            f"- consolidated: {cons.get('consolidated_effects')}",
            f"- primary simulation targets: {cons.get('primary_simulation_targets')}",
            f"- by role: {cons.get('by_role')}",
            "",
        ]
    sel = data.get("study_selection_summary")
    if sel:
        lines += [
            "## Study selection",
            f"- kept: {sel.get('kept')}",
            f"- dropped: {sel.get('dropped')}",
            "",
        ]
    osf = data.get("osf_materials_summary")
    if osf:
        lines += [
            "## Materials",
            f"- ready: {osf.get('ready')}/{osf.get('total_sub_studies')}",
            f"- by source: {osf.get('by_primary_source')}",
            f"- needs review: {osf.get('unready')}",
            "",
        ]
    contract = data.get("human_material_contract")
    if contract:
        missing = contract.get("missing_by_field") or {}
        missing_source = contract.get("missing_source_by_field") or {}
        selected_missing = contract.get("selected_missing_by_field") or {}
        selected_missing_source = contract.get("selected_missing_source_by_field") or {}
        lines += [
            "## HumanStudy material contract",
            f"- ready: {contract.get('ready')}/{contract.get('total_sub_studies')}",
            f"- needs patch: {contract.get('needs_patch')}",
            f"- selected ready: {contract.get('selected_ready', 'N/A')}/{contract.get('selected_total', 'N/A')}",
            f"- selected needs patch: {contract.get('selected_needs_patch', 'N/A')}",
            f"- missing by field: {missing}",
            f"- selected missing by field: {selected_missing}",
            f"- missing source by field: {missing_source}",
            f"- selected missing source by field: {selected_missing_source}",
            "",
        ]
    target_summary = data.get("simulation_target_summary")
    if target_summary:
        lines += [
            "## Simulation targets",
            f"- total targets: {target_summary.get('total_targets')}",
            f"- by sub-study: {target_summary.get('by_sub_study')}",
            "",
        ]
    requirements = data.get("material_requirements_summary")
    if isinstance(requirements, dict):
        lines += [
            "## Material requirements",
            f"- selected ready: {requirements.get('selected_ready')}/{requirements.get('selected_total')}",
            f"- selected needs patch: {requirements.get('selected_needs_patch')}",
            f"- selected missing by field: {requirements.get('selected_missing_by_field')}",
            f"- selected blocking by issue: {requirements.get('selected_blocking_by_issue')}",
            f"- target-bound sub-studies: {requirements.get('target_bound_sub_studies')}",
            "",
        ]

    lines += ["## Per sub-study", "",
              "| sub_study | selected | source | items | ready |",
              "|---|:--:|---|:--:|:--:|"]
    for sid, m in materials.items():
        sel_keep = m.get("selection", {}).get("keep", True)
        src = m.get("source_trace", {}).get("primary_source")
        ready = m.get("readiness", {}).get("ready")
        lines.append(f"| {sid} | {'✅' if sel_keep else '—'} | {src} | {len(m.get('items', []))} | {'✅' if ready else '⚠'} |")
    lines.append("")
    return "\n".join(lines)


def _study_ids(eligible: List[Dict[str, Any]], paper_dir: Path) -> List[str]:
    ids = [
        str(s.get("study") or s.get("study_id") or s.get("experiment_id") or "").strip()
        for s in eligible
    ]
    ids = [i for i in ids if i]
    if ids:
        return ids
    # fall back to Stage 1 experiment ids
    stage1 = paper_dir / "stage1.json"
    if stage1.exists():
        data = json.loads(stage1.read_text(encoding="utf-8"))
        return [e.get("experiment_id") for e in data.get("experiments", []) if e.get("experiment_id")]
    return []


def _load_stage1_json(paper_dir: Path) -> Dict[str, Any]:
    stage1 = paper_dir / "stage1.json"
    if not stage1.exists():
        return {}
    try:
        data = json.loads(stage1.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _stage1_experiment_map(stage1_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for exp in stage1_json.get("experiments", []) or []:
        if not isinstance(exp, dict):
            continue
        for key in (
            exp.get("study_id"),
            exp.get("experiment_id"),
            exp.get("study_name"),
            exp.get("experiment_name"),
        ):
            sid = _material_slug(key)
            if sid and sid not in out:
                out[sid] = exp
    return out


def _targets_by_sub_id(targets: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for target in targets:
        if not isinstance(target, dict):
            continue
        sid = _material_slug(target.get("sub_study_id") or target.get("study_name"))
        if sid:
            out.setdefault(sid, []).append(target)
    return out


def _stage_study_map(eligible: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for study in eligible:
        if not isinstance(study, dict):
            continue
        for key in (study.get("study_id"), study.get("study"), study.get("experiment_id")):
            sid = _material_slug(key)
            if sid and sid not in out:
                out[sid] = study
    return out


def _effect_for_target(study: Optional[Dict[str, Any]], target: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(study, dict):
        return {}
    try:
        index = int(target.get("effect_index"))
    except (TypeError, ValueError):
        return {}
    effects = study.get("effects") if isinstance(study.get("effects"), list) else []
    if 0 <= index < len(effects) and isinstance(effects[index], dict):
        return effects[index]
    return {}


def _contract_status(material: Dict[str, Any], field: str) -> tuple[str, str]:
    contract = material.get("human_material_contract") if isinstance(material.get("human_material_contract"), dict) else {}
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    evidence = contract.get("field_evidence") if isinstance(contract.get("field_evidence"), dict) else {}
    field_payload = fields.get(field) if isinstance(fields.get(field), dict) else {}
    evidence_payload = evidence.get(field) if isinstance(evidence.get(field), dict) else {}
    return str(field_payload.get("status") or "missing"), str(evidence_payload.get("status") or "missing")


def _required_field_record(material: Dict[str, Any], field: str, *, required: bool, purpose: str) -> Dict[str, Any]:
    status, source_status = _contract_status(material, field)
    requires_source = field != "response_schema"
    if not requires_source and source_status == "missing":
        source_status = "not_applicable"
    missing = status == "missing" or (required and requires_source and source_status == "missing")
    return {
        "field": field,
        "required": required,
        "purpose": purpose,
        "status": status,
        "source_status": source_status,
        "missing": missing,
    }


def apply_material_requirements(
    data: Dict[str, Any],
    materials: Dict[str, Dict[str, Any]],
    *,
    eligible: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    stage1_json: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach final HumanStudy-Bench field requirements to Stage 3 materials."""
    stage1_map = _stage1_experiment_map(stage1_json)
    study_map = _stage_study_map(eligible)
    target_map = _targets_by_sub_id(targets)
    summary_missing: Dict[str, List[str]] = {}
    selected_missing: Dict[str, List[str]] = {}
    blocking_by_issue: Dict[str, List[str]] = {}
    selected_blocking_by_issue: Dict[str, List[str]] = {}
    selected_total = 0
    selected_ready = 0

    for sid, material in materials.items():
        if not isinstance(material, dict):
            continue
        study = study_map.get(_material_slug(sid))
        stage1_exp = stage1_map.get(_material_slug(sid))
        study_targets = target_map.get(_material_slug(sid), [])
        selected = material.get("selection", {}).get("keep", True) is not False
        if selected:
            selected_total += 1

        target_effects = [_effect_for_target(study, target) for target in study_targets]
        target_ivs = sorted({str(effect.get("IV")) for effect in target_effects if effect.get("IV")})
        target_dvs = sorted({str(effect.get("DV")) for effect in target_effects if effect.get("DV")})
        conditions_expected = bool((stage1_exp or {}).get("conditions_or_factors") or target_ivs)
        source_hints = (stage1_exp or {}).get("candidate_source_hints")
        source_hints = source_hints if isinstance(source_hints, list) else []

        fields = [
            _required_field_record(
                material,
                "instructions",
                required=True,
                purpose="participant-facing task framing, stimulus, vignette, or manipulation text",
            ),
            _required_field_record(
                material,
                "response_items",
                required=True,
                purpose="participant response questions/items consumed by the simulator",
            ),
            _required_field_record(
                material,
                "response_options",
                required=True,
                purpose="answer options, anchors, or scale bounds for every response item",
            ),
            _required_field_record(
                material,
                "response_schema",
                required=True,
                purpose="dominant simulator answer type plus options/anchors/scale metadata",
            ),
            _required_field_record(
                material,
                "conditions",
                required=conditions_expected,
                purpose="manipulated/measured factors and levels needed to instantiate trials",
            ),
        ]
        source_record = {
            "field": "source_evidence",
            "required": True,
            "purpose": "field-level source provenance for human review and patching",
            "status": "present" if material.get("source_trace") else "missing",
            "source_status": "present" if material.get("source_trace") else "missing",
            "missing": not bool(material.get("source_trace")),
        }
        fields.append(source_record)

        missing_required = [field["field"] for field in fields if field["required"] and field["missing"]]
        expected_missing = [field["field"] for field in fields if field["missing"]]
        readiness = material.get("readiness") if isinstance(material.get("readiness"), dict) else {}
        readiness_blocking = [
            str(issue)
            for issue in readiness.get("blocking_issues", []) or []
            if str(issue).strip()
        ]
        ready = not missing_required and not readiness_blocking and readiness.get("ready") is not False
        if selected and ready:
            selected_ready += 1
        for field in expected_missing:
            summary_missing.setdefault(field, []).append(sid)
            if selected:
                selected_missing.setdefault(field, []).append(sid)
        for issue in readiness_blocking:
            blocking_by_issue.setdefault(issue, []).append(sid)
            if selected:
                selected_blocking_by_issue.setdefault(issue, []).append(sid)

        search_terms = sorted(
            _tokens(
                *(target_ivs + target_dvs),
                (stage1_exp or {}).get("participant_task"),
                (stage1_exp or {}).get("input"),
                *((hint.get("description") if isinstance(hint, dict) else hint) for hint in source_hints),
            )
        )
        material["material_requirements"] = {
            "version": "human-study-bench-material-requirements-v1",
            "sub_study_id": sid,
            "selected": selected,
            "ready": ready,
            "target_ids": [target.get("target_id") for target in study_targets if target.get("target_id")],
            "target_effect_indices": [target.get("effect_index") for target in study_targets],
            "target_ivs": target_ivs,
            "target_dvs": target_dvs,
            "required_fields": fields,
            "missing_required_fields": missing_required,
            "missing_expected_fields": expected_missing,
            "readiness_blocking_issues": sorted(set(readiness_blocking)),
            "stage1_design": {
                "experiment_id": (stage1_exp or {}).get("experiment_id"),
                "study_id": (stage1_exp or {}).get("study_id"),
                "design_type": (stage1_exp or {}).get("design_type"),
                "conditions_or_factors": (stage1_exp or {}).get("conditions_or_factors", []),
                "participant_task": (stage1_exp or {}).get("participant_task"),
            },
            "source_hints": source_hints,
            "search_terms": search_terms[:40],
        }

    summary = {
        "version": "human-study-bench-material-requirements-v1",
        "total_sub_studies": len(materials),
        "selected_total": selected_total,
        "selected_ready": selected_ready,
        "selected_needs_patch": selected_total - selected_ready,
        "missing_by_field": {field: sorted(set(sids)) for field, sids in sorted(summary_missing.items())},
        "selected_missing_by_field": {field: sorted(set(sids)) for field, sids in sorted(selected_missing.items())},
        "blocking_by_issue": {issue: sorted(set(sids)) for issue, sids in sorted(blocking_by_issue.items())},
        "selected_blocking_by_issue": {
            issue: sorted(set(sids))
            for issue, sids in sorted(selected_blocking_by_issue.items())
        },
        "target_bound_sub_studies": sorted(target_map),
    }
    data["material_requirements_summary"] = summary
    return summary


def build_study_materials(
    paper_dir: str | Path,
    *,
    stage3_path: Optional[Path] = None,
    osf_files_dir: Optional[Path] = None,
    pdf_path: Optional[Path] = None,
    llm_client: Any = None,
    write: bool = True,
    allow_effect_slot_fallback: bool = False,
    selection_votes: int = 3,
    selection_timeout: Optional[float] = 60.0,
    pdf_material_timeout: Optional[float] = 120.0,
) -> Dict[str, Dict[str, Any]]:
    """Assemble per-sub-study materials for a paper and (optionally) persist them.

    Returns a mapping {sub_study_id: materials_dict}. When `write=True`, the
    mapping plus a short summary is written back into `stage3.json` under
    `study_materials` / `osf_materials_summary`.

    By default, per-effect `materials/manipulation/items` slots are not treated
    as final materials. They may be recorded in `source_trace.legacy_slot_summary`
    for human review, but `ready` remains false until a structured source or
    later Stage 4 patch provides participant-facing instructions and items.

    When `llm_client` is provided, also runs the study selector (which sub-studies
    are simulation-worthy) and annotates each study with a `selection` block.
    """
    paper_dir = Path(paper_dir)
    stage3_path = stage3_path or (paper_dir / "stage3.json")
    if not stage3_path.exists():
        raise FileNotFoundError(f"stage3.json not found: {stage3_path}")

    osf_files_dir = _resolve_osf_files_dir(paper_dir, osf_files_dir)
    data = json.loads(stage3_path.read_text(encoding="utf-8"))
    stage1_json = _load_stage1_json(paper_dir)
    eligible = _eligible_of(data)
    study_ids = _study_ids(eligible, paper_dir)

    # 1. consolidate + select effects (dedup over-segmentation, role tagging)
    for study in eligible:
        annotate_study(study)

    # 1a. LLM re-judgment of effect roles (manip-checks, mislabelled causal
    #     effects, secondary DVs) — overrides the deterministic effecttype roles.
    if llm_client is not None:
        from generation_pipeline.effect_selector import apply_to_study
        title = data.get("paper_title") or ""
        for study in eligible:
            try:
                apply_to_study(study, title, llm_client, votes=selection_votes, timeout=selection_timeout)
            except Exception:  # fail-open: keep deterministic roles
                pass

    # 1b. study-level selection (which sub-studies are simulation-worthy)
    if llm_client is not None:
        _annotate_selection(
            paper_dir,
            eligible,
            study_ids,
            data,
            llm_client,
            votes=selection_votes,
            timeout=selection_timeout,
        )

    # Keep study-level findings aligned with any LLM role overrides above.
    from generation_pipeline.stage2_findings import annotate_stage2_findings
    annotate_stage2_findings(data, recompute_consolidation=False)

    # 2. link OSF sources + assemble per-study materials
    if osf_files_dir.exists():
        link: LinkResult = link_sources(study_ids, osf_files_dir)
    else:
        # no OSF files at all → empty link result; assembler falls back to Stage 3
        link = LinkResult(by_study={sid: [] for sid in study_ids}, unlinked=[])

    materials: Dict[str, Dict[str, Any]] = {}
    material_study_ids: Dict[str, str] = {}
    for sid in study_ids:
        s3_study = match_stage3_study(sid, eligible)
        built = assemble_study_materials(
            sid,
            link,
            repo_root=REPO_ROOT,
            stage3_study=s3_study,
            allow_pdf_slot_fallback=allow_effect_slot_fallback,
        )
        if isinstance(s3_study, dict) and isinstance(s3_study.get("selection"), dict):
            built["selection"] = s3_study["selection"]
        materials[built["sub_study_id"]] = built
        material_study_ids[built["sub_study_id"]] = sid

    if llm_client is not None:
        _merge_linked_source_materials(
            materials,
            data,
            link,
            material_study_ids,
            llm_client,
            stage1_json=stage1_json,
            timeout=pdf_material_timeout,
        )

    if pdf_path is not None and _selected_material_needs_source_completion(materials):
        try:
            _merge_pdf_materials(
                materials,
                extract_pdf_study_materials(
                    data,
                    Path(pdf_path),
                    llm_client,
                    timeout=pdf_material_timeout,
                    stage1_json=stage1_json,
                    artifacts_dir=paper_dir / "pdf_artifacts",
                ),
            )
        except Exception as exc:
            data["pdf_material_extraction_error"] = f"{type(exc).__name__}: {exc}"
            for material in materials.values():
                readiness = material.setdefault("readiness", {"ready": False, "blocking_issues": [], "warnings": []})
                warnings = readiness.setdefault("warnings", [])
                warnings.append(f"PDF material extraction failed; material left as source-only draft: {type(exc).__name__}: {exc}")

    _apply_stage2_material_scaffolds(materials, eligible)
    _complete_sparse_materials_from_siblings(materials, eligible)
    data["human_material_contract"] = apply_material_contracts(materials)

    from generation_pipeline.simulation_targets import (
        build_simulation_targets,
        summarize_simulation_targets,
    )
    targets = build_simulation_targets(
        eligible,
        materials,
        paper_title=str(data.get("paper_title") or ""),
    )
    data["simulation_targets"] = targets
    data["simulation_target_summary"] = summarize_simulation_targets(targets)
    apply_material_requirements(
        data,
        materials,
        eligible=eligible,
        targets=targets,
        stage1_json=stage1_json,
    )

    if write:
        _write_back(stage3_path, data, materials)

    return materials


def _merge_linked_source_materials(
    materials: Dict[str, Dict[str, Any]],
    data: Dict[str, Any],
    link: LinkResult,
    material_study_ids: Dict[str, str],
    llm_client: Any,
    *,
    stage1_json: Dict[str, Any],
    timeout: Optional[float],
) -> None:
    """Extract missing study materials from linked OSF/source PDF/DOCX/TXT files.

    Structured QSF/SAV remains the first-class source. This pass only fills
    study drafts that are missing a participant-facing source or have codebook
    items without a stimulus. It uses the same HumanStudy-Bench field prompt as
    the paper-PDF fallback, but grounds extraction in linked supplementary
    materials before consulting the main article PDF.
    """
    attempts: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for sub_id, material in materials.items():
        selection = material.get("selection") if isinstance(material.get("selection"), dict) else {}
        if selection.get("keep") is False:
            continue
        if not _material_needs_linked_source(material):
            continue
        study_id = material_study_ids.get(sub_id) or sub_id
        candidates = _rank_linked_source_candidates(link.material_files(study_id))
        if not candidates:
            continue
        for source in candidates[:2]:
            attempt = {
                "sub_study_id": sub_id,
                "study_id": study_id,
                "file": _rel_source_file(source.path),
                "kind": source.kind,
            }
            attempts.append(attempt)
            try:
                extracted = extract_linked_source_materials(
                    data,
                    source.path,
                    llm_client,
                    stage1_json=stage1_json,
                    pdf_text=_source_text_override(source.path, source.kind),
                    source_label="osf_source_llm",
                    timeout=timeout,
                    only_sub_study_id=sub_id,
                )
            except Exception as exc:
                errors.append({
                    **{key: str(value) for key, value in attempt.items()},
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            if not extracted:
                continue
            extracted_ids = sorted(extracted)
            extracted = _filter_pdf_materials_for_target(
                extracted,
                target_sid=sub_id,
                materials=materials,
            )
            attempt["extracted_sub_study_ids"] = extracted_ids
            attempt["merged_sub_study_ids"] = sorted(extracted)
            dropped_ids = sorted(set(extracted_ids) - set(extracted))
            if dropped_ids:
                attempt["dropped_cross_study_ids"] = dropped_ids
            if not extracted:
                attempt["merged"] = False
                continue
            before = materials.get(sub_id, {})
            before_items = len(before.get("items") or []) if isinstance(before, dict) else 0
            before_instructions = str(before.get("instructions") or "") if isinstance(before, dict) else ""
            _merge_pdf_materials(materials, extracted)
            merged = materials.get(sub_id, {})
            if isinstance(merged.get("source_trace"), dict):
                merged["source_trace"]["source_extraction"] = {
                    "mode": "linked_osf_source",
                    "source_file": _rel_source_file(source.path),
                    "source_kind": source.kind,
                }
            after_items = len(merged.get("items") or []) if isinstance(merged, dict) else 0
            after_instructions = str(merged.get("instructions") or "") if isinstance(merged, dict) else ""
            attempt["merged"] = after_items != before_items or after_instructions != before_instructions
            attempt["post_ready"] = merged.get("readiness", {}).get("ready") if isinstance(merged, dict) else None
            if not _material_needs_linked_source(materials[sub_id]):
                break

    if attempts or errors:
        data["linked_source_material_extraction"] = {
            "version": "linked-source-material-extraction-v1",
            "attempts": attempts,
            "errors": errors,
        }


def _rank_linked_source_candidates(files: List[Any]) -> List[Any]:
    source_files = [item for item in files if getattr(item, "kind", None) in {"pdf", "docx", "text"}]

    def score(item: Any) -> tuple[int, str]:
        path_text = str(getattr(item, "path", "")).lower()
        value = 0
        if any(term in path_text for term in ("material", "materials", "survey", "questionnaire", "instrument", "stimuli", "stimulus")):
            value += 6
        if any(term in path_text for term in ("experimental", "experiment")):
            value += 3
        if any(term in path_text for term in ("pre-registration", "prereg", "pre_reg", "analysis", "script", "code")):
            value -= 6
        if getattr(item, "kind", None) == "text":
            value += 2
        elif getattr(item, "kind", None) == "docx":
            value += 1
        return (-value, str(getattr(item, "path", "")))

    return sorted(source_files, key=score)


def _filter_pdf_materials_for_target(
    extracted: Dict[str, Dict[str, Any]],
    *,
    target_sid: str,
    materials: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Keep only extracted materials that map back to the current source target."""
    out: Dict[str, Dict[str, Any]] = {}
    target_only = {target_sid: materials[target_sid]} if target_sid in materials else {}
    for pdf_sid, pdf_material in extracted.items():
        if _match_pdf_material_id(pdf_sid, target_only) == target_sid:
            out[pdf_sid] = pdf_material
            continue
        if _match_pdf_material_id(pdf_sid, materials) == target_sid:
            out[pdf_sid] = pdf_material
    return out


def _source_text_override(path: Path, kind: str) -> Optional[str]:
    if kind == "text":
        return Path(path).read_text(encoding="utf-8", errors="ignore")[:PDF_TEXT_MAX_CHARS]
    if kind == "docx":
        return _extract_docx_text(Path(path))[:PDF_TEXT_MAX_CHARS]
    return None


def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            raw = zf.read("word/document.xml")
    except Exception:
        return ""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return ""
    chunks = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            chunks.append(node.text)
    return "\n".join(chunks)


def _rel_source_file(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _material_needs_linked_source(material: Dict[str, Any]) -> bool:
    return _material_needs_pdf_fallback(material) or _material_needs_stimulus_completion(material)


def _selected_material_needs_source_completion(materials: Dict[str, Dict[str, Any]]) -> bool:
    for material in materials.values():
        if not isinstance(material, dict):
            continue
        selection = material.get("selection") if isinstance(material.get("selection"), dict) else {}
        if selection.get("keep") is False:
            continue
        if _material_needs_linked_source(material):
            return True
    return False


def _material_needs_stimulus_completion(material: Dict[str, Any]) -> bool:
    if not isinstance(material, dict):
        return False
    if str(material.get("instructions") or "").strip():
        return False
    return bool(material.get("items"))


def _material_needs_pdf_fallback(material: Dict[str, Any]) -> bool:
    source = material.get("source_trace", {}).get("primary_source")
    if source:
        return False
    readiness = material.get("readiness") if isinstance(material.get("readiness"), dict) else {}
    blocking = set(readiness.get("blocking_issues") or [])
    if "no_structured_material_source" in blocking:
        return True
    if not material.get("instructions") or not material.get("items"):
        return True
    return readiness.get("ready") is False


def _merge_pdf_materials(
    materials: Dict[str, Dict[str, Any]],
    pdf_materials: Dict[str, Dict[str, Any]],
) -> None:
    """Fill empty/no-source study material drafts from PDF-extracted packages."""
    for sid, pdf_material in pdf_materials.items():
        if not isinstance(pdf_material, dict):
            continue
        target_sid = _match_pdf_material_id(sid, materials)
        if target_sid is None:
            continue
        current = materials.get(target_sid)
        if current is not None and _material_needs_pdf_stimulus(current, pdf_material):
            _merge_pdf_stimulus(current, pdf_material)
            continue
        if current is not None and not _material_needs_pdf_fallback(current):
            continue
        merged = dict(pdf_material)
        if isinstance(current, dict):
            if isinstance(current.get("selection"), dict):
                merged["selection"] = current["selection"]
            trace = merged.setdefault("source_trace", {})
            trace["replaced_empty_source_trace"] = current.get("source_trace", {})
        merged["sub_study_id"] = target_sid
        materials[target_sid] = merged


def _match_pdf_material_id(
    pdf_sid: str,
    materials: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Map PDF-extractor ids back to known Stage 3 study ids.

    The PDF extractor may produce descriptive ids such as
    `study_1a_vignette_paradigm_democrat_sample`; Stage 3 must still keep the
    material unit fixed to the Stage 1/2 study id (`study_1a`).
    """
    if pdf_sid in materials:
        return pdf_sid
    pdf_slug = _material_slug(pdf_sid)
    candidates = []
    for sid in materials:
        sid_slug = _material_slug(sid)
        if pdf_slug == sid_slug:
            return sid
        if pdf_slug.startswith(f"{sid_slug}_"):
            candidates.append((len(sid_slug), sid))
    if candidates:
        return max(candidates)[1]

    pdf_tokens = _tokens(pdf_sid)
    scored: List[tuple[float, str]] = []
    for sid, material in materials.items():
        score = _jaccard(
            pdf_tokens,
            _tokens(
                sid,
                material.get("sub_study_id") if isinstance(material, dict) else "",
            ),
        )
        if score > 0:
            scored.append((score, sid))
    if scored:
        score, sid = max(scored)
        if score >= 0.5:
            return sid
    return None


def _material_needs_pdf_stimulus(
    material: Dict[str, Any],
    pdf_material: Dict[str, Any],
) -> bool:
    """Whether PDF material can complete a structured-source item package."""
    if not isinstance(material, dict) or not isinstance(pdf_material, dict):
        return False
    if str(material.get("instructions") or "").strip():
        return False
    if not material.get("items"):
        return False
    pdf_instructions = str(pdf_material.get("instructions") or "").strip()
    if not pdf_instructions:
        return False
    pdf_readiness = pdf_material.get("readiness") if isinstance(pdf_material.get("readiness"), dict) else {}
    pdf_blocking = set(pdf_readiness.get("blocking_issues") or [])
    return not ({"missing_instructions", "pdf_material_not_found"} & pdf_blocking)


def _merge_pdf_stimulus(
    material: Dict[str, Any],
    pdf_material: Dict[str, Any],
) -> None:
    material["instructions"] = str(pdf_material.get("instructions") or "").strip()
    if not material.get("conditions") and pdf_material.get("conditions"):
        material["conditions"] = pdf_material["conditions"]
    elif pdf_material.get("conditions"):
        _merge_condition_descriptions(material, pdf_material)
    _repair_item_questions_from_pdf_material(material, pdf_material)

    trace = material.setdefault("source_trace", {})
    pdf_trace = pdf_material.get("source_trace") if isinstance(pdf_material.get("source_trace"), dict) else {}
    trace["stimulus_source"] = pdf_trace.get("primary_source") or "pdf_llm"
    trace["stimulus_source_file"] = pdf_trace.get("source_file")
    trace["stimulus_source_trace"] = pdf_trace

    readiness = material.setdefault("readiness", {"ready": False, "blocking_issues": [], "warnings": []})
    issues = readiness.setdefault("blocking_issues", [])
    issues[:] = [
        issue for issue in issues
        if issue not in {"stimulus_not_verbatim", "missing_instructions", "no_structured_material_source"}
    ]
    warnings = readiness.setdefault("warnings", [])
    warnings.append("Stimulus/instructions completed from PDF material extraction; response items retained from structured source.")
    if material.get("items"):
        if not _material_has_truncated_questions(material):
            issues[:] = [
                issue for issue in issues
                if issue not in {"truncated_sav_labels", "truncated_response_question"}
            ]
        readiness["ready"] = not issues


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_'-]{2,}", re.IGNORECASE)
_MATERIAL_STOP_TERMS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about",
    "study", "studies", "effect", "effects", "condition", "conditions",
    "participant", "participants", "message", "messages", "article", "articles",
    "measure", "measures", "item", "items", "scale", "index", "response",
    "responses", "dependent", "variable", "independent", "interaction",
}
_GENERIC_ASSIGNMENT_RE = re.compile(
    r"participants?\s+were\s+randomly\s+presented\s+with\s+either\s+the\s+[^.]+",
    re.IGNORECASE,
)
_CONDITION_DETAIL_RE = re.compile(
    r"\b(?:in|under|for)\s+the\s+[A-Za-z0-9][A-Za-z0-9 /_-]{1,80}?\s+condition\b",
    re.IGNORECASE,
)


def _material_slug(text: Any) -> str:
    if text in (None, ""):
        return ""
    return canonical_sub_study_id(text)


def _stem(token: str) -> str:
    token = token.lower().strip("'_-")
    for suffix in ("iveness", "ation", "ness", "ment", "ity", "ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break
    return token[:7] if len(token) > 7 else token


def _tokens(*values: Any) -> set[str]:
    out: set[str] = set()
    for value in values:
        for tok in _TOKEN_RE.findall(str(value or "")):
            stemmed = _stem(tok)
            if len(stemmed) >= 4 and stemmed not in _MATERIAL_STOP_TERMS and not stemmed.isdigit():
                out.add(stemmed)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _material_has_truncated_questions(material: Dict[str, Any]) -> bool:
    for item in material.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "")
        if "..." in question or "\u2026" in question:
            return True
    return False


def _repair_item_questions_from_pdf_material(
    material: Dict[str, Any],
    pdf_material: Dict[str, Any],
) -> None:
    current_items = [item for item in material.get("items", []) or [] if isinstance(item, dict)]
    pdf_items = [item for item in pdf_material.get("items", []) or [] if isinstance(item, dict)]
    if not current_items or not pdf_items:
        return

    repairs: List[Dict[str, str]] = []
    pdf_trace = pdf_material.get("source_trace") if isinstance(pdf_material.get("source_trace"), dict) else {}
    pdf_source = str(pdf_trace.get("primary_source") or "pdf_llm")
    pdf_source_file = str(pdf_trace.get("source_file") or "")
    for idx, item in enumerate(current_items):
        old_question = str(item.get("question") or "")
        if "..." not in old_question and "\u2026" not in old_question:
            continue
        replacement = _best_pdf_item_question(item, pdf_items, fallback_index=idx)
        if not replacement:
            continue
        item["question"] = replacement
        flags = [flag for flag in item.get("quality_flags", []) if flag != "label_contains_ellipsis"]
        if flags:
            item["quality_flags"] = sorted(set(flags))
        else:
            item.pop("quality_flags", None)
        old_source = str(item.get("source") or "").strip()
        if old_source and pdf_source not in old_source:
            item["source"] = f"{old_source}+{pdf_source}"
        else:
            item["source"] = old_source or pdf_source
        item["question_source"] = pdf_source
        if pdf_source_file:
            item["question_source_file"] = pdf_source_file
        repairs.append(
            {
                "item": str(item.get("data_export_tag") or item.get("id") or idx),
                "old": old_question.replace("\u2026", "..."),
                "new": replacement,
                "source": pdf_source,
            }
        )

    if repairs:
        trace = material.setdefault("source_trace", {})
        trace["item_question_repairs"] = repairs
        warnings = material.setdefault("readiness", {}).setdefault("warnings", [])
        warnings.append(f"Repaired {len(repairs)} truncated item question(s) from PDF/source material extraction.")


def _best_pdf_item_question(
    item: Dict[str, Any],
    pdf_items: List[Dict[str, Any]],
    *,
    fallback_index: int,
) -> str:
    current_terms = _tokens(item.get("id"), item.get("data_export_tag"), item.get("question"))
    scored: List[tuple[float, Dict[str, Any]]] = []
    for pdf_item in pdf_items:
        question = str(pdf_item.get("question") or "").strip()
        if not question or "..." in question or "\u2026" in question:
            continue
        pdf_terms = _tokens(pdf_item.get("id"), pdf_item.get("data_export_tag"), question)
        scored.append((_jaccard(current_terms, pdf_terms), pdf_item))
    if scored:
        score, match = max(scored, key=lambda row: row[0])
        if score >= 0.12:
            return str(match.get("question") or "").strip()
    if 0 <= fallback_index < len(pdf_items):
        question = str(pdf_items[fallback_index].get("question") or "").strip()
        if question and "..." not in question and "\u2026" not in question:
            return question
    return ""


def _effect_terms(study: Optional[Dict[str, Any]]) -> set[str]:
    if not isinstance(study, dict):
        return set()
    terms: set[str] = set()
    effects = [effect for effect in study.get("effects", []) if isinstance(effect, dict)]
    primary = []
    for effect in effects:
        cons = effect.get("consolidation") if isinstance(effect.get("consolidation"), dict) else {}
        if cons.get("is_primary_simulation_target", True) and cons.get("is_representative", True) is not False:
            primary.append(effect)
    for effect in primary or effects:
        terms |= _tokens(effect.get("IV"), effect.get("DV"))
    return terms


def _condition_terms(material: Dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for cond in material.get("conditions", []) or []:
        if not isinstance(cond, dict):
            continue
        terms |= _tokens(cond.get("name"), cond.get("label"))
        for level in cond.get("levels", []) or []:
            terms |= _tokens(level)
    return terms


def _condition_levels(material: Dict[str, Any]) -> set[str]:
    levels: set[str] = set()
    for cond in material.get("conditions", []) or []:
        if not isinstance(cond, dict):
            continue
        for level in cond.get("levels", []) or []:
            norm = re.sub(r"[^a-z0-9]+", "", str(level).lower())
            if norm:
                levels.add(norm)
    return levels


def _item_terms(material: Dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for item in material.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        terms |= _tokens(item.get("id"), item.get("data_export_tag"), item.get("question"))
    return terms


def _instruction_detail_count(text: str) -> int:
    return len(_CONDITION_DETAIL_RE.findall(text or ""))


def _has_condition_descriptions(material: Dict[str, Any]) -> bool:
    for cond in material.get("conditions", []) or []:
        if isinstance(cond, dict) and isinstance(cond.get("level_descriptions"), dict):
            if any(str(v).strip() for v in cond["level_descriptions"].values()):
                return True
    return False


def _instructions_sparse(material: Dict[str, Any]) -> bool:
    text = str(material.get("instructions") or "").strip()
    if len(text) < 180:
        return True
    if _GENERIC_ASSIGNMENT_RE.fullmatch(text.rstrip(".")):
        return True
    return _instruction_detail_count(text) < 2 and not _has_condition_descriptions(material)


def _instructions_rich(material: Dict[str, Any]) -> bool:
    text = str(material.get("instructions") or "").strip()
    return len(text) >= 160 and (_instruction_detail_count(text) >= 2 or _has_condition_descriptions(material))


def _study_by_sub_id(eligible: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for study in eligible:
        sid = _material_slug(study.get("study") or study.get("study_id") or study.get("experiment_id"))
        if sid:
            out[sid] = study
    return out


def _sibling_material_score(
    target: Dict[str, Any],
    candidate: Dict[str, Any],
    target_study: Optional[Dict[str, Any]],
    candidate_study: Optional[Dict[str, Any]],
) -> tuple[float, Dict[str, float]]:
    effect = _jaccard(_effect_terms(target_study), _effect_terms(candidate_study))
    items = _jaccard(_item_terms(target), _item_terms(candidate))
    cond_terms = _jaccard(_condition_terms(target), _condition_terms(candidate))
    target_levels = _condition_levels(target)
    candidate_levels = _condition_levels(candidate)
    level_match = 1.0 if target_levels and target_levels == candidate_levels else 0.0
    score = (0.50 * effect) + (0.30 * items) + (0.10 * cond_terms) + (0.10 * level_match)
    return score, {
        "effect_similarity": round(effect, 3),
        "item_similarity": round(items, 3),
        "condition_similarity": round(cond_terms, 3),
        "level_match": round(level_match, 3),
        "score": round(score, 3),
    }


def _merge_condition_descriptions(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    source_by_name = {
        str(cond.get("name") or cond.get("label") or "").lower(): cond
        for cond in source.get("conditions", []) or []
        if isinstance(cond, dict)
    }
    merged = []
    for cond in target.get("conditions", []) or []:
        if not isinstance(cond, dict):
            merged.append(cond)
            continue
        key = str(cond.get("name") or cond.get("label") or "").lower()
        src = source_by_name.get(key)
        new_cond = dict(cond)
        if isinstance(src, dict) and isinstance(src.get("level_descriptions"), dict):
            existing = new_cond.get("level_descriptions") if isinstance(new_cond.get("level_descriptions"), dict) else {}
            new_cond["level_descriptions"] = {**src["level_descriptions"], **existing}
            new_cond.setdefault("description_source", "sibling_study_material")
        merged.append(new_cond)
    target["conditions"] = merged


def _apply_stage2_material_scaffolds(
    materials: Dict[str, Dict[str, Any]],
    eligible: List[Dict[str, Any]],
) -> None:
    """Preserve Stage 2 study/effect hints when no participant material exists.

    This is deliberately a patch scaffold, not a runnable material source. It
    prevents Stage 3/4 from emitting empty files when Stage 2 already identified
    the study design, target DVs, source locations, and material notes.
    """
    study_map = _study_by_sub_id(eligible)
    for sid, material in materials.items():
        if not isinstance(material, dict):
            continue
        study = study_map.get(sid)
        if not isinstance(study, dict):
            continue
        selection = material.get("selection") if isinstance(material.get("selection"), dict) else {}
        if selection.get("keep") is False:
            continue

        effects = _stage2_scaffold_effects(study)
        if not effects:
            continue

        trace = material.setdefault("source_trace", {})
        had_primary_source = bool(trace.get("primary_source"))
        scaffold = _build_stage2_scaffold(study, effects)
        trace["stage2_scaffold"] = scaffold["source_trace"]["stage2_scaffold"]

        readiness = material.setdefault("readiness", {"ready": False, "blocking_issues": [], "warnings": []})
        blocking = readiness.setdefault("blocking_issues", [])
        warnings = readiness.setdefault("warnings", [])
        added = False

        if not str(material.get("instructions") or "").strip() and not had_primary_source:
            material["instructions"] = scaffold["instructions"]
            trace["primary_source"] = "stage2_scaffold"
            trace["source_file"] = "stage2.json"
            blocking.append("stage2_scaffold_not_participant_facing")
            added = True

        if not material.get("items"):
            material["items"] = scaffold["items"]
            material["response_schema"] = scaffold["response_schema"]
            blocking.append("stage2_scaffold_placeholder_items")
            added = True

        if not material.get("conditions") and scaffold["conditions"] and not had_primary_source:
            material["conditions"] = scaffold["conditions"]
            added = True

        if added:
            blocking.append("stage2_scaffold_requires_source_patch")
            warnings.append(
                "Stage 2 scaffold preserved known IV/DV/source-location hints, "
                "but verbatim participant-facing wording must still be recovered from PDF/OSF."
            )
            readiness["blocking_issues"] = sorted(set(blocking))
            readiness["warnings"] = sorted(set(warnings))
            readiness["ready"] = False


def _stage2_scaffold_effects(study: Dict[str, Any]) -> List[Dict[str, Any]]:
    effects = [effect for effect in study.get("effects", []) or [] if isinstance(effect, dict)]
    if not effects:
        return []
    primary: List[Dict[str, Any]] = []
    for effect in effects:
        cons = effect.get("consolidation") if isinstance(effect.get("consolidation"), dict) else {}
        role = str(cons.get("analysis_role") or effect.get("effecttype") or "").lower()
        if cons and cons.get("is_representative", True) is False:
            continue
        if cons and cons.get("is_primary_simulation_target") is False:
            continue
        if role in {"manipulation_check", "check", "secondary_dv"}:
            continue
        if role in {"primary_finding", "primary", "main", "main_effect", "interaction", "moderation", "int"}:
            primary.append(effect)
    if primary:
        return primary[:8]
    fallback = [
        effect
        for effect in effects
        if str(effect.get("effecttype") or "").lower() in {"main", "int", "interaction", "moderation"}
    ]
    return (fallback or effects)[:8]


def _build_stage2_scaffold(study: Dict[str, Any], effects: List[Dict[str, Any]]) -> Dict[str, Any]:
    study_name = str(study.get("study") or study.get("study_id") or study.get("experiment_id") or "Study")
    source_notes = _stage2_source_notes(effects)
    effect_lines = []
    for idx, effect in enumerate(effects, start=1):
        line = f"{idx}. IV: {effect.get('IV') or 'unknown'} | DV: {effect.get('DV') or 'unknown'}"
        if effect.get("materials_notes"):
            line += f" | materials_notes: {effect.get('materials_notes')}"
        if effect.get("table_or_page_location"):
            line += f" | source_location: {effect.get('table_or_page_location')}"
        effect_lines.append(line)
    instructions = "\n".join(
        [
            "STAGE2 SCAFFOLD ONLY - not participant-facing material.",
            "Recover verbatim stimulus, task instructions, response items, options, and anchors from the PDF appendix or supplementary instrument before simulation.",
            f"Study: {study_name}",
            "Known Stage 2 targets:",
            *effect_lines,
        ]
    )
    items = _stage2_placeholder_items(effects)
    conditions = _stage2_conditions_from_effects(effects)
    return {
        "instructions": instructions,
        "items": items,
        "conditions": conditions,
        "response_schema": {"answer_type": "text", "placeholder": True} if items else {},
        "source_trace": {
            "stage2_scaffold": {
                "mode": "stage2_design_and_source_hints",
                "study": study_name,
                "source_file": "stage2.json",
                "effects": source_notes,
                "limitations": [
                    "Not participant-facing wording.",
                    "Response items are placeholders derived from Stage 2 DVs.",
                    "Conditions are inferred from Stage 2 IV strings and require PDF/OSF confirmation.",
                ],
            }
        },
    }


def _stage2_source_notes(effects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    notes: List[Dict[str, Any]] = []
    for idx, effect in enumerate(effects):
        notes.append(
            {
                "effect_index": idx,
                "effecttype": effect.get("effecttype"),
                "IV": effect.get("IV"),
                "DV": effect.get("DV"),
                "direction": effect.get("direction"),
                "materials_notes": effect.get("materials_notes"),
                "table_or_page_location": effect.get("table_or_page_location"),
                "size": effect.get("size") or effect.get("analysis_n"),
                "stats": effect.get("stats"),
            }
        )
    return notes


def _stage2_placeholder_items(effects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for effect in effects:
        dv = str(effect.get("DV") or "").strip()
        if not dv:
            continue
        key = re.sub(r"\s+", " ", dv.lower())
        if key in seen:
            continue
        seen.add(key)
        item_id = re.sub(r"[^a-z0-9]+", "_", dv.lower()).strip("_") or "dv"
        items.append(
            {
                "id": f"stage2_placeholder_{item_id}",
                "question": f"PATCH REQUIRED: recover verbatim participant-facing item(s) for '{dv}'.",
                "type": "text",
                "options": [],
                "source": "stage2_scaffold",
                "source_file": "stage2.json",
                "metadata": {
                    "placeholder": True,
                    "dv": dv,
                    "related_ivs": sorted({str(e.get("IV")) for e in effects if e.get("DV") == dv and e.get("IV")}),
                    "source_locations": sorted({
                        str(e.get("table_or_page_location"))
                        for e in effects
                        if e.get("DV") == dv and e.get("table_or_page_location")
                    }),
                },
            }
        )
    return items[:12]


def _stage2_conditions_from_effects(effects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for effect in effects:
        iv = str(effect.get("IV") or "").strip()
        if not iv:
            continue
        for part in re.split(r"\s+(?:×|x|X)\s+", iv):
            condition = _condition_from_stage2_iv(part, source_effect=effect)
            if not condition:
                continue
            key = (condition["name"].lower(), tuple(level.lower() for level in condition.get("levels", [])))
            if str(key) in seen:
                continue
            seen.add(str(key))
            conditions.append(condition)
    return conditions[:8]


def _condition_from_stage2_iv(text: str, *, source_effect: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = re.sub(r"\s+", " ", str(text or "")).strip(" ;.")
    if not raw:
        return None
    name = raw
    level_text = ""
    paren = re.search(r"^(?P<name>.*?)\((?P<levels>[^)]*\bvs\.?\b[^)]*)\)", raw, flags=re.IGNORECASE)
    colon = re.search(r"^(?P<name>.*?):\s*(?P<levels>.*\bvs\.?\b.*)$", raw, flags=re.IGNORECASE)
    if paren:
        name = paren.group("name").strip()
        level_text = paren.group("levels")
    elif colon:
        name = colon.group("name").strip()
        level_text = colon.group("levels")
    elif re.search(r"\bvs\.?\b", raw, flags=re.IGNORECASE):
        level_text = raw
    if not level_text:
        return None
    level_text = re.split(r",\s*(?:manipulated|measured|reported|randomized)", level_text, flags=re.IGNORECASE)[0]
    pieces = [
        re.sub(r"\s+", " ", piece).strip(" .;")
        for piece in re.split(r"\bvs\.?\b| versus ", level_text, flags=re.IGNORECASE)
    ]
    levels = [piece for piece in pieces if piece]
    if len(levels) < 2:
        return None
    clean_name = re.sub(r"\s+", " ", name).strip(" :") or "stage2_condition"
    return {
        "name": clean_name,
        "levels": levels[:6],
        "source": "stage2_scaffold",
        "source_file": "stage2.json",
        "review_required": True,
        "metadata": {
            "source_iv": source_effect.get("IV"),
            "source_location": source_effect.get("table_or_page_location"),
            "placeholder": True,
        },
    }


def _complete_sparse_materials_from_siblings(
    materials: Dict[str, Dict[str, Any]],
    eligible: List[Dict[str, Any]],
) -> None:
    """Record sibling stimulus candidates for sparse studies without patching.

    This handles preregistered replications and repeated paradigms where one
    study's PDF slots contain the full manipulation and a sibling only says
    "participants were randomly assigned/presented..." even though the OSF
    codebook shows the same condition and outcome structure.

    Earlier versions copied the sibling's instructions into the sparse material
    and sometimes flipped readiness to true. That is too aggressive for the new
    Stage 3 contract: source-backed fields should remain unchanged, and
    inferred cross-study reuse should be a patch candidate for human review.
    """
    study_map = _study_by_sub_id(eligible)
    for sid, material in materials.items():
        source_trace = material.get("source_trace") if isinstance(material.get("source_trace"), dict) else {}
        if source_trace.get("extractor") == "pdf_evidence_provider_v1":
            continue
        if not _instructions_sparse(material):
            continue
        target_study = study_map.get(sid)
        ranked: List[tuple[float, Dict[str, float], str, Dict[str, Any]]] = []
        for candidate_id, candidate in materials.items():
            if candidate_id == sid or not _instructions_rich(candidate):
                continue
            score, parts = _sibling_material_score(
                material,
                candidate,
                target_study,
                study_map.get(candidate_id),
            )
            if parts["effect_similarity"] >= 0.25 and parts["item_similarity"] >= 0.20 and score >= 0.42:
                ranked.append((score, parts, candidate_id, candidate))
        if not ranked:
            continue
        _, parts, source_id, source = max(ranked, key=lambda row: row[0])
        trace = material.setdefault("source_trace", {})
        trace["material_completion_candidate"] = {
            "mode": "sibling_study_material",
            "source_sub_study_id": source_id,
            "current_instruction_preview": str(material.get("instructions") or "").strip()[:240],
            "suggested_instruction_preview": str(source.get("instructions") or "").strip()[:600],
            "suggested_conditions": source.get("conditions") if isinstance(source.get("conditions"), list) else [],
            **parts,
        }
        warnings = material.setdefault("readiness", {}).setdefault("warnings", [])
        warnings.append(
            f"Candidate sibling stimulus found in {source_id} "
            f"(score={parts['score']}, effect_similarity={parts['effect_similarity']}); material left unchanged."
        )


def _annotate_selection(
    paper_dir: Path,
    eligible: List[Dict[str, Any]],
    study_ids: List[str],
    data: Dict[str, Any],
    client: Any,
    *,
    votes: int = 3,
    timeout: Optional[float] = 60.0,
) -> None:
    """Run the study selector and attach `selection` to each eligible study."""
    from generation_pipeline.study_selector import select_studies, selection_summary

    stage1 = paper_dir / "stage1.json"
    experiments: List[Dict[str, Any]] = []
    if stage1.exists():
        experiments = json.loads(stage1.read_text(encoding="utf-8")).get("experiments", [])
    if not experiments:
        # fall back to building briefs from the eligible studies themselves
        experiments = [{"experiment_id": sid} for sid in study_ids]

    decisions = select_studies(experiments, client, votes=votes, timeout=timeout)
    for study in eligible:
        name = str(study.get("study") or study.get("study_id") or "")
        dec = decisions.get(name)
        if dec is not None:
            study["selection"] = dec
    data["study_selection_summary"] = selection_summary(decisions)


def _summarize(materials: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ready = sum(1 for m in materials.values() if m["readiness"]["ready"])
    by_source: Dict[str, int] = {}
    for m in materials.values():
        src = m["source_trace"].get("primary_source") or "none"
        by_source[src] = by_source.get(src, 0) + 1
    return {
        "total_sub_studies": len(materials),
        "ready": ready,
        "needs_review": len(materials) - ready,
        "by_primary_source": by_source,
        "unready": [
            sid for sid, m in materials.items() if not m["readiness"]["ready"]
        ],
    }


def _consolidation_summary(eligible: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw = cons = prim = 0
    by_role: Dict[str, int] = {}
    for s in eligible:
        cs = s.get("consolidation_summary")
        if not cs:
            continue
        raw += cs["raw_effects"]
        cons += cs["consolidated_effects"]
        prim += cs["primary_simulation_targets"]
        for role, n in cs.get("by_role", {}).items():
            by_role[role] = by_role.get(role, 0) + n
    return {
        "raw_effects": raw,
        "consolidated_effects": cons,
        "primary_simulation_targets": prim,
        "by_role": by_role,
    }


def _write_back(
    stage3_path: Path,
    data: Dict[str, Any],
    materials: Dict[str, Dict[str, Any]],
) -> None:
    # `data` already carries the in-place effect consolidation annotations.
    data["study_materials"] = materials
    data["osf_materials_summary"] = _summarize(materials)
    data["effect_consolidation_summary"] = _consolidation_summary(_eligible_of(data))
    tmp = stage3_path.with_suffix(stage3_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(stage3_path)


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Stage 3 OSF material extraction.")
    ap.add_argument("paper_dir", help="Output paper dir containing stage3.json")
    ap.add_argument("--no-write", action="store_true", help="Do not modify stage3.json")
    args = ap.parse_args(argv)

    mats = build_study_materials(args.paper_dir, write=not args.no_write)
    summary = _summarize(mats)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for sid, m in mats.items():
        rd = m["readiness"]
        src = m["source_trace"].get("primary_source")
        flag = "" if rd["ready"] else f"  !! {rd['blocking_issues']}"
        print(f"  {sid:14s} src={src} items={len(m['items'])} ready={rd['ready']}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
