from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set


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


def _has_stat_value(stats: Dict[str, Any]) -> bool:
    for key in ("B", "b", "t", "f", "z", "chi_square", "D", "eta_square", "p_value", "sig"):
        value = stats.get(key)
        if value not in (None, "", [], [None, None]):
            return True
    ci = stats.get("ci")
    return isinstance(ci, list) and any(item is not None for item in ci)


def _has_reported_statistics(finding: Dict[str, Any]) -> bool:
    if _has_text(finding.get("reported_statistics")):
        return True
    stats = finding.get("stats") if isinstance(finding.get("stats"), dict) else {}
    return _has_stat_value(stats)


def _valid_index(value: Any, effects: List[Dict[str, Any]]) -> Optional[int]:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(effects):
        return idx
    return None


def _stage1_eligible_keys(stage1_json: Optional[Dict[str, Any]]) -> Set[str]:
    if not isinstance(stage1_json, dict):
        return set()
    keys: Set[str] = set()
    for exp in stage1_json.get("experiments", []) or []:
        if not isinstance(exp, dict):
            continue
        replicable = str(exp.get("replicable") or "").strip().upper()
        if replicable not in {"YES", "UNCERTAIN"}:
            continue
        key = _study_key(exp.get("experiment_id") or exp.get("experiment_name"))
        if key:
            keys.add(key)
    return keys


def _add_missing(missing_by_field: Dict[str, List[str]], field: str, ref: str) -> None:
    missing_by_field.setdefault(field, []).append(ref)


def _finding_audit(
    finding: Dict[str, Any],
    *,
    effects: List[Dict[str, Any]],
    path: str,
    missing_by_field: Dict[str, List[str]],
) -> Dict[str, Any]:
    finding_id = str(finding.get("finding_id") or path)
    blocking: List[str] = []
    warnings: List[str] = []

    for field in ("finding_id", "role", "IV", "DV"):
        if finding.get(field) in (None, "", []):
            issue = f"missing_{field}"
            blocking.append(issue)
            _add_missing(missing_by_field, field, finding_id)

    if not _has_reported_statistics(finding):
        blocking.append("missing_reported_statistics")
        _add_missing(missing_by_field, "reported_statistics", finding_id)

    rep_idx = _valid_index(finding.get("representative_effect_index"), effects)
    if rep_idx is None:
        blocking.append("invalid_representative_effect_index")
        _add_missing(missing_by_field, "representative_effect_index", finding_id)

    raw_indices = finding.get("effect_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        blocking.append("missing_effect_indices")
        _add_missing(missing_by_field, "effect_indices", finding_id)
        valid_indices: List[int] = []
    else:
        valid_indices = []
        invalid_indices: List[Any] = []
        for value in raw_indices:
            idx = _valid_index(value, effects)
            if idx is None:
                invalid_indices.append(value)
            else:
                valid_indices.append(idx)
        if invalid_indices:
            blocking.append("invalid_effect_indices")
            _add_missing(missing_by_field, "effect_indices", finding_id)

    if rep_idx is not None and valid_indices and rep_idx not in valid_indices:
        warnings.append("representative_not_in_effect_indices")

    target = finding.get("simulation_target") if isinstance(finding.get("simulation_target"), dict) else {}
    if "candidate" not in target:
        blocking.append("missing_simulation_target_candidate")
        _add_missing(missing_by_field, "simulation_target", finding_id)

    if not _has_text(finding.get("table_or_page_location")):
        warnings.append("missing_table_or_page_location")
        _add_missing(missing_by_field, "table_or_page_location", finding_id)

    return {
        "finding_id": finding.get("finding_id"),
        "ready": not blocking,
        "blocking_issues": sorted(set(blocking)),
        "warnings": sorted(set(warnings)),
        "fields": {
            "representative_effect_index": rep_idx,
            "effect_indices": valid_indices,
            "role": finding.get("role"),
            "simulation_target_candidate": target.get("candidate"),
            "IV": finding.get("IV"),
            "DV": finding.get("DV"),
            "reported_statistics_present": _has_reported_statistics(finding),
            "table_or_page_location": finding.get("table_or_page_location"),
        },
    }


def audit_stage2_finding_contract(
    stage2_json: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a field-level contract audit for a Stage 2 result."""
    studies = [study for study in stage2_json.get("eligible_studies", []) or [] if isinstance(study, dict)]
    missing_by_field: Dict[str, List[str]] = {}
    audits: Dict[str, Dict[str, Any]] = {}
    ready_studies = 0
    total_findings = 0
    primary_targets = 0
    stage2_keys: Set[str] = set()

    for sidx, study in enumerate(studies):
        path = f"$.eligible_studies[{sidx}]"
        study_id = str(study.get("study_id") or study.get("study") or f"study_{sidx + 1}")
        study_name = str(study.get("study_name") or study.get("study") or study_id)
        key = _study_key(study.get("study") or study.get("study_name") or study_id)
        if key:
            stage2_keys.add(key)

        effects = [effect for effect in study.get("effects", []) or [] if isinstance(effect, dict)]
        findings = [finding for finding in study.get("findings", []) or [] if isinstance(finding, dict)]
        sample = study.get("sample") if isinstance(study.get("sample"), dict) else {}
        blocking: List[str] = []
        warnings: List[str] = []

        if not _has_text(study_id) or not _has_text(study_name):
            blocking.append("missing_study_identity")
            _add_missing(missing_by_field, "study_identity", study_id or path)
        if not effects:
            blocking.append("missing_raw_effects")
            _add_missing(missing_by_field, "effects", study_id)
        if not findings:
            blocking.append("missing_consolidated_findings")
            _add_missing(missing_by_field, "findings", study_id)
        if not sample.get("total_n") and not sample.get("analyzed_n"):
            warnings.append("missing_sample_n")
            _add_missing(missing_by_field, "sample_n", study_id)

        finding_audits: Dict[str, Dict[str, Any]] = {}
        study_primary_targets = 0
        seen_reps: Set[int] = set()
        for fidx, finding in enumerate(findings):
            total_findings += 1
            target = finding.get("simulation_target") if isinstance(finding.get("simulation_target"), dict) else {}
            if target.get("candidate") is True:
                study_primary_targets += 1
                primary_targets += 1
            audit = _finding_audit(
                finding,
                effects=effects,
                path=f"{path}.findings[{fidx}]",
                missing_by_field=missing_by_field,
            )
            rep_idx = audit["fields"]["representative_effect_index"]
            if rep_idx is not None:
                if rep_idx in seen_reps:
                    audit["warnings"].append("duplicate_representative_effect_index")
                seen_reps.add(rep_idx)
            if not audit["ready"]:
                blocking.append(f"finding_not_ready:{finding.get('finding_id') or fidx}")
            finding_audits[str(finding.get("finding_id") or f"finding_{fidx + 1}")] = audit

        if findings and study_primary_targets == 0:
            blocking.append("no_primary_simulation_target")
            _add_missing(missing_by_field, "primary_simulation_target", study_id)

        ready = not blocking
        if ready:
            ready_studies += 1
        audits[study_id] = {
            "study_id": study_id,
            "study_name": study_name,
            "ready": ready,
            "blocking_issues": sorted(set(blocking)),
            "warnings": sorted(set(warnings)),
            "raw_effects": len(effects),
            "findings_total": len(findings),
            "primary_simulation_targets": study_primary_targets,
            "fields": {
                "study_identity": "present" if _has_text(study_id) and _has_text(study_name) else "missing",
                "sample_n": "present" if sample.get("total_n") or sample.get("analyzed_n") else "missing",
                "raw_effects": len(effects),
                "consolidated_findings": len(findings),
            },
            "findings": finding_audits,
        }

    stage1_keys = _stage1_eligible_keys(stage1_json)
    coverage = {
        "checked": bool(stage1_keys),
        "stage1_eligible_or_uncertain": len(stage1_keys),
        "stage2_studies": len(stage2_keys),
        "missing_from_stage2": sorted(stage1_keys - stage2_keys),
        "extra_in_stage2": sorted(stage2_keys - stage1_keys) if stage1_keys else [],
    }
    for key in coverage["missing_from_stage2"]:
        _add_missing(missing_by_field, "stage1_coverage", key)

    contract_blockers = sum(
        len(study_audit.get("blocking_issues") or [])
        for study_audit in audits.values()
    ) + len(coverage["missing_from_stage2"])

    return {
        "version": "stage2_finding_contract_v1",
        "total_studies": len(studies),
        "ready": ready_studies,
        "needs_review": len(studies) - ready_studies,
        "total_findings": total_findings,
        "primary_simulation_targets": primary_targets,
        "blocking_issue_count": contract_blockers,
        "missing_by_field": {field: sorted(set(refs)) for field, refs in sorted(missing_by_field.items())},
        "stage1_stage2_coverage": coverage,
        "studies": audits,
    }


def apply_stage2_finding_contract(
    stage2_json: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach the Stage 2 finding contract audit in-place and return it."""
    contract = audit_stage2_finding_contract(stage2_json, stage1_json)
    stage2_json["stage2_finding_contract"] = contract
    return contract
