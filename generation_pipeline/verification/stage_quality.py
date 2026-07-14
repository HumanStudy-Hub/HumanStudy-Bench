from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set


_VALID_REPLICABLE = {"YES", "NO", "UNCERTAIN"}


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _study_key(value: Any) -> str:
    text = _norm(value)
    match = re.search(r"\b(?:study|experiment|exp)\s+(\d+[a-z]?)\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(pilot|validation)\b", text)
    if match:
        return match.group(1)
    text = re.sub(r"\b(study|experiment|exp)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_stats(effect: Dict[str, Any]) -> bool:
    stats = effect.get("stats") if isinstance(effect.get("stats"), dict) else {}
    if effect.get("reported_statistics_text"):
        return True
    for key in ("B", "b", "t", "f", "z", "chi_square", "D", "eta_square", "p_value", "sig"):
        value = stats.get(key)
        if value not in (None, "", [], [None, None]):
            return True
    ci = stats.get("ci")
    return isinstance(ci, list) and any(item is not None for item in ci)


def _slot_status(effect: Dict[str, Any], slot: str) -> Optional[str]:
    value = effect.get(slot)
    if isinstance(value, dict):
        status = value.get("status")
        return str(status).strip() if status is not None else None
    return None


def _regen_item_count(verification: Dict[str, Any]) -> int:
    regen = verification.get("regeneration_instructions")
    if not isinstance(regen, dict):
        return 0
    count = 0
    for value in regen.values():
        if isinstance(value, list):
            count += len(value)
        elif value not in (None, "", {}):
            count += 1
    return count


def _needs_human_review(
    issues: List[Dict[str, Any]],
    *,
    verifier_status: Any,
    verifier_overall: Any,
    verifier_suggestions: int,
) -> bool:
    if any(issue.get("severity") == "error" for issue in issues):
        return True
    if verifier_status == "error":
        return True
    if verifier_overall in {"needs_review", "fail"}:
        return True
    return verifier_suggestions > 0


def _stage1_eligible_keys(stage1_json: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set()
    for exp in stage1_json.get("experiments", []) or []:
        if not isinstance(exp, dict):
            continue
        replicable = str(exp.get("replicable") or "").strip().upper()
        if replicable in {"YES", "UNCERTAIN"}:
            key = _study_key(exp.get("experiment_id") or exp.get("experiment_name"))
            if key:
                keys.add(key)
    return keys


def build_stage1_quality(stage1_json: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact quality report for the Stage 1 study inventory."""
    experiments = [exp for exp in stage1_json.get("experiments", []) or [] if isinstance(exp, dict)]
    comparison_groups = [
        group
        for group in stage1_json.get("comparison_groups", []) or []
        if isinstance(group, dict)
    ]
    issues: List[Dict[str, Any]] = []
    eligible = 0
    material_missing = 0
    material_variants_total = 0

    if not experiments:
        issues.append({"severity": "error", "path": "$.experiments", "message": "no experiments extracted"})

    for idx, exp in enumerate(experiments):
        path = f"$.experiments[{idx}]"
        material_variants_total += len(exp.get("material_variants") or [])
        replicable = str(exp.get("replicable") or "").strip().upper()
        if replicable not in _VALID_REPLICABLE:
            issues.append({"severity": "warning", "path": f"{path}.replicable", "message": "invalid replicable label"})
        if replicable in {"YES", "UNCERTAIN"}:
            eligible += 1
            for field in ("experiment_id", "input", "participants", "output"):
                if not _has_text(exp.get(field)):
                    issues.append({"severity": "warning", "path": f"{path}.{field}", "message": "missing eligible-study anchor"})
            if not _has_text(exp.get("design_type")):
                issues.append({"severity": "warning", "path": f"{path}.design_type", "message": "missing design type"})
            if not exp.get("conditions_or_factors"):
                issues.append({"severity": "warning", "path": f"{path}.conditions_or_factors", "message": "missing factors or conditions"})
            if exp.get("has_self_contained_materials") is False or _has_text(exp.get("missing_materials")):
                material_missing += 1
        simulation_barriers = exp.get("simulation_barriers")
        if not isinstance(simulation_barriers, list):
            issues.append(
                {
                    "severity": "error",
                    "path": f"{path}.simulation_barriers",
                    "message": "simulation barriers are not an array",
                }
            )
            simulation_barriers = []
        if replicable != "NO" and any(
            isinstance(barrier, dict)
            and barrier.get("affects_primary_target") is True
            for barrier in simulation_barriers
        ):
            issues.append(
                {
                    "severity": "error",
                    "path": f"{path}.replicable",
                    "message": "primary-target execution barrier requires NO eligibility",
                }
            )
        provenance = str(exp.get("unit_provenance") or "").strip().lower()
        if provenance not in {"current_paper", "unclear"}:
            issues.append(
                {
                    "severity": "error",
                    "path": f"{path}.unit_provenance",
                    "message": "accepted inventory entry is not established as a unit from the current paper",
                }
            )
        if exp.get("is_distinct_empirical_unit") is not True:
            issues.append(
                {
                    "severity": "error",
                    "path": f"{path}.is_distinct_empirical_unit",
                    "message": "accepted inventory entry is not a distinct empirical unit",
                }
            )

    if eligible == 0 and experiments:
        issues.append({"severity": "error", "path": "$.experiments", "message": "no eligible or uncertain experiments"})

    stage1_evidence = stage1_json.get("stage1_evidence")
    stage1_evidence = stage1_evidence if isinstance(stage1_evidence, dict) else {}
    if not stage1_evidence:
        issues.append(
            {
                "severity": "error",
                "path": "$.stage1_evidence",
                "message": "missing bounded-window Stage 1 evidence audit",
            }
        )
    else:
        if stage1_evidence.get("full_document_llm_calls") != 0:
            issues.append(
                {
                    "severity": "error",
                    "path": "$.stage1_evidence.full_document_llm_calls",
                    "message": "Stage 1 used a full-document LLM request",
                }
            )
        if not stage1_evidence.get("all_mentions_assigned"):
            issues.append(
                {
                    "severity": "error",
                    "path": "$.stage1_evidence.all_mentions_assigned",
                    "message": "one or more discovery mentions were not reconciled",
                }
            )
        if not stage1_evidence.get("extraction_complete"):
            issues.append(
                {
                    "severity": "error",
                    "path": "$.stage1_evidence.extraction_complete",
                    "message": "one or more per-study Stage 1 extractions failed",
                }
            )
        if not stage1_evidence.get("all_comparison_relations_resolved", False):
            issues.append(
                {
                    "severity": "error",
                    "path": "$.stage1_evidence.all_comparison_relations_resolved",
                    "message": "one or more source-explicit comparison relations were not mapped to accepted units",
                }
            )
        if stage1_evidence.get("parser_degraded"):
            issues.append(
                {
                    "severity": "warning",
                    "path": "$.stage1_evidence.parser_degraded",
                    "message": "layout-aware PDF parsing was unavailable",
                }
            )

    study_contract = stage1_json.get("stage1_study_contract") if isinstance(stage1_json.get("stage1_study_contract"), dict) else {}
    contract_blocking = int(study_contract.get("blocking_issue_count") or 0) if study_contract else 0
    if contract_blocking:
        issues.append(
            {
                "severity": "error",
                "path": "$.stage1_study_contract",
                "message": f"Stage 1 study contract has {contract_blocking} blocking issue(s)",
            }
        )

    verification = stage1_json.get("stage1_verification") if isinstance(stage1_json.get("stage1_verification"), dict) else {}
    verifier_status = verification.get("status")
    verifier_overall = verification.get("overall")
    if verifier_status == "error":
        issues.append({"severity": "warning", "path": "$.stage1_verification", "message": "Stage 1 verifier failed"})
    elif verifier_status not in {None, "ok"}:
        issues.append(
            {
                "severity": "error",
                "path": "$.stage1_verification.validation_diagnostics",
                "message": "Stage 1 verifier completed with malformed correction diagnostics",
            }
        )
    elif verifier_overall in {"needs_review", "fail"}:
        severity = "error" if verifier_overall == "fail" else "warning"
        issues.append({"severity": severity, "path": "$.stage1_verification.overall", "message": f"Stage 1 verifier overall={verifier_overall}"})
    verifier_suggestions = _regen_item_count(verification)
    if verifier_suggestions:
        issues.append({"severity": "warning", "path": "$.stage1_verification.regeneration_instructions", "message": f"Stage 1 verifier returned {verifier_suggestions} regeneration suggestion(s)"})
    if verifier_status == "ok":
        study_audit = verification.get("study_audit")
        study_audit = study_audit if isinstance(study_audit, dict) else {}
        if int(study_audit.get("study_count") or 0) != len(experiments):
            issues.append(
                {
                    "severity": "error",
                    "path": "$.stage1_verification.study_audit.study_count",
                    "message": "Stage 1 verifier did not audit every accepted empirical unit",
                }
            )
        if study_audit.get("full_document_llm_calls") != 0:
            issues.append(
                {
                    "severity": "error",
                    "path": "$.stage1_verification.study_audit.full_document_llm_calls",
                    "message": "Stage 1 study-field verifier used a full-document request",
                }
            )
        if not study_audit.get("all_cited_evidence_included") and experiments:
            issues.append(
                {
                    "severity": "error",
                    "path": "$.stage1_verification.study_audit.all_cited_evidence_included",
                    "message": "one or more study audits omitted currently cited evidence",
                }
            )

    return {
        "version": "stage1_quality_v5",
        "experiments_total": len(experiments),
        "comparison_groups_total": len(comparison_groups),
        "material_variants_total": material_variants_total,
        "rejected_candidates_total": int(stage1_evidence.get("rejected_candidate_count") or 0),
        "rejected_comparison_relations_total": int(
            stage1_evidence.get("rejected_comparison_relation_count") or 0
        ),
        "eligible_or_uncertain": eligible,
        "missing_materials_count": material_missing,
        "study_contract_ready": study_contract.get("ready") if study_contract else None,
        "study_contract_needs_review": study_contract.get("needs_review") if study_contract else None,
        "study_contract_blocking_issues": contract_blocking if study_contract else None,
        "verifier_status": verifier_status,
        "verifier_overall": verifier_overall,
        "verifier_suggestions": verifier_suggestions,
        "needs_human_review": _needs_human_review(
            issues,
            verifier_status=verifier_status,
            verifier_overall=verifier_overall,
            verifier_suggestions=verifier_suggestions,
        ),
        "issues": issues,
    }


def build_stage2_quality(stage2_json: Dict[str, Any], stage1_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return quality checks for Stage 2 study/effect extraction."""
    studies = [study for study in stage2_json.get("eligible_studies", []) or [] if isinstance(study, dict)]
    issues: List[Dict[str, Any]] = []
    study_reports: List[Dict[str, Any]] = []
    effects_total = 0
    effects_with_stats = 0
    effects_with_legacy_material_slots = 0
    legacy_material_slot_details: List[Dict[str, Any]] = []
    findings_total = 0
    primary_finding_candidates = 0
    stage2_keys: Set[str] = set()

    if not studies:
        issues.append({"severity": "error", "path": "$.eligible_studies", "message": "no eligible studies extracted"})

    for sidx, study in enumerate(studies):
        path = f"$.eligible_studies[{sidx}]"
        study_name = study.get("study") or study.get("name") or f"Study {sidx + 1}"
        key = _study_key(study_name)
        if key:
            stage2_keys.add(key)
        sample = study.get("sample") if isinstance(study.get("sample"), dict) else {}
        study_issues: List[str] = []
        effects = [effect for effect in study.get("effects", []) or [] if isinstance(effect, dict)]
        findings = [finding for finding in study.get("findings", []) or [] if isinstance(finding, dict)]
        effects_total += len(effects)
        findings_total += len(findings)

        if not effects:
            study_issues.append("no_effects")
            issues.append({"severity": "error", "path": f"{path}.effects", "message": "study has no effects"})
        if not findings:
            study_issues.append("no_findings")
            issues.append({"severity": "warning", "path": f"{path}.findings", "message": "study has no consolidated findings"})
        if not sample.get("total_n") and not sample.get("analyzed_n"):
            study_issues.append("missing_sample_n")
            issues.append({"severity": "warning", "path": f"{path}.sample", "message": "missing study sample N"})

        for fidx, finding in enumerate(findings):
            fpath = f"{path}.findings[{fidx}]"
            if finding.get("simulation_target", {}).get("candidate"):
                primary_finding_candidates += 1
            for field in ("finding_id", "representative_effect_index", "role", "IV", "DV", "reported_statistics"):
                if finding.get(field) in (None, "", []):
                    issues.append({"severity": "warning", "path": f"{fpath}.{field}", "message": "finding missing required mapping field"})

        for eidx, effect in enumerate(effects):
            epath = f"{path}.effects[{eidx}]"
            if not _has_text(effect.get("IV")):
                issues.append({"severity": "warning", "path": f"{epath}.IV", "message": "missing IV"})
            if not _has_text(effect.get("DV")):
                issues.append({"severity": "warning", "path": f"{epath}.DV", "message": "missing DV"})
            if _has_stats(effect):
                effects_with_stats += 1
            else:
                issues.append({"severity": "error", "path": f"{epath}.stats", "message": "effect has no reported statistic"})
            if not _has_text(effect.get("table_or_page_location")):
                issues.append({"severity": "warning", "path": f"{epath}.table_or_page_location", "message": "missing paper location"})
            if not effect.get("evidence_refs"):
                issues.append(
                    {
                        "severity": "error",
                        "path": f"{epath}.evidence_refs",
                        "message": "effect has no grounded PDF evidence block",
                    }
                )
            active_legacy_slots = {
                slot: status
                for slot in ("materials", "manipulation", "items")
                if (status := _slot_status(effect, slot)) in {"verbatim", "paraphrased", "cited_scale", "osf_only"}
            }
            if active_legacy_slots:
                effects_with_legacy_material_slots += 1
                legacy_material_slot_details.append(
                    {
                        "path": epath,
                        "study": study_name,
                        "effect_index": eidx,
                        "statuses": active_legacy_slots,
                    }
                )

        study_reports.append(
            {
                "study": study_name,
                "effects": len(effects),
                "findings": len(findings),
                "sample_n_present": bool(sample.get("total_n") or sample.get("analyzed_n")),
                "issues": sorted(set(study_issues)),
            }
        )

    coverage: Dict[str, Any] = {}
    if isinstance(stage1_json, dict):
        stage1_keys = _stage1_eligible_keys(stage1_json)
        coverage = {
            "stage1_eligible_or_uncertain": len(stage1_keys),
            "stage2_studies": len(stage2_keys),
            "missing_from_stage2": sorted(stage1_keys - stage2_keys),
            "extra_in_stage2": sorted(stage2_keys - stage1_keys),
        }
        for key in coverage["missing_from_stage2"]:
            issues.append({"severity": "warning", "path": "$.eligible_studies", "message": f"Stage1 study not covered in Stage2: {key}"})

    finding_contract = stage2_json.get("stage2_finding_contract") if isinstance(stage2_json.get("stage2_finding_contract"), dict) else {}
    contract_blocking = int(finding_contract.get("blocking_issue_count") or 0) if finding_contract else 0
    if contract_blocking:
        issues.append(
            {
                "severity": "error",
                "path": "$.stage2_finding_contract",
                "message": f"Stage 2 finding contract has {contract_blocking} blocking issue(s)",
            }
        )

    stage2_evidence = stage2_json.get("stage2_evidence")
    stage2_evidence = stage2_evidence if isinstance(stage2_evidence, dict) else {}
    if not stage2_evidence:
        issues.append(
            {
                "severity": "error",
                "path": "$.stage2_evidence",
                "message": "missing per-study Stage 2 evidence audit",
            }
        )
    elif stage2_evidence.get("full_document_llm_calls") != 0:
        issues.append(
            {
                "severity": "error",
                "path": "$.stage2_evidence.full_document_llm_calls",
                "message": "Stage 2 used a full-document LLM request",
            }
        )

    verification = stage2_json.get("stage2_verification") if isinstance(stage2_json.get("stage2_verification"), dict) else {}
    verifier_status = verification.get("status")
    verifier_overall = verification.get("overall")
    if verifier_status == "error":
        issues.append({"severity": "warning", "path": "$.stage2_verification", "message": "Stage 2 verifier failed"})
    elif verifier_overall in {"needs_review", "fail"}:
        severity = "error" if verifier_overall == "fail" else "warning"
        issues.append({"severity": severity, "path": "$.stage2_verification.overall", "message": f"Stage 2 verifier overall={verifier_overall}"})
    verifier_suggestions = _regen_item_count(verification)
    if verifier_suggestions:
        issues.append({"severity": "warning", "path": "$.stage2_verification.regeneration_instructions", "message": f"Stage 2 verifier returned {verifier_suggestions} regeneration suggestion(s)"})

    if effects_with_legacy_material_slots:
        issues.append(
            {
                "severity": "warning",
                "path": "$.eligible_studies[*].effects[*].materials/manipulation/items",
                "message": (
                    f"{effects_with_legacy_material_slots} effect(s) still contain legacy per-effect material "
                    "slot hints; Stage 3 must use study-level PDF/OSF evidence for final materials"
                ),
            }
        )

    return {
        "version": "stage2_quality_v1",
        "studies_total": len(studies),
        "effects_total": effects_total,
        "findings_total": findings_total,
        "primary_finding_candidates": primary_finding_candidates,
        "effects_with_reported_stats": effects_with_stats,
        "legacy_effect_material_slot_hints": effects_with_legacy_material_slots,
        "legacy_effect_material_slot_details": legacy_material_slot_details,
        "stage1_stage2_coverage": coverage,
        "finding_contract_ready": finding_contract.get("ready") if finding_contract else None,
        "finding_contract_needs_review": finding_contract.get("needs_review") if finding_contract else None,
        "finding_contract_blocking_issues": contract_blocking if finding_contract else None,
        "verifier_status": verifier_status,
        "verifier_overall": verifier_overall,
        "verifier_suggestions": verifier_suggestions,
        "study_reports": study_reports,
        "needs_human_review": _needs_human_review(
            issues,
            verifier_status=verifier_status,
            verifier_overall=verifier_overall,
            verifier_suggestions=verifier_suggestions,
        ),
        "issues": issues,
    }
