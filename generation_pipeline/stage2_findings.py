from __future__ import annotations

import re
from typing import Any, Dict, List

from generation_pipeline.identifiers import canonical_sub_study_id
from generation_pipeline.parsers.effect_consolidator import annotate_study
from generation_pipeline.stage2_finding_contract import apply_stage2_finding_contract


def _slug(value: Any, fallback: str = "study") -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text or fallback


def _reported_statistics(effect: Dict[str, Any]) -> str:
    stats = effect.get("stats") if isinstance(effect.get("stats"), dict) else {}
    parts: List[str] = []
    for key in ("B", "b", "t", "f", "z", "chi_square", "D", "eta_square", "p_value", "sig"):
        value = stats.get(key)
        if value not in (None, "", [], [None, None]):
            parts.append(f"{key}={value}")
    ci = stats.get("ci")
    if isinstance(ci, list) and any(item is not None for item in ci):
        parts.append(f"ci={ci}")
    text = effect.get("reported_statistics_text")
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    return ", ".join(parts)


def _effect_at(effects: List[Dict[str, Any]], index: Any) -> Dict[str, Any]:
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return {}
    if idx < 0 or idx >= len(effects):
        return {}
    effect = effects[idx]
    return effect if isinstance(effect, dict) else {}


def _valid_effect_index(effects: List[Dict[str, Any]], index: Any) -> int | None:
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(effects):
        return idx
    return None


def _finding_from_group(
    *,
    study_id: str,
    study_name: str,
    effects: List[Dict[str, Any]],
    group: Dict[str, Any],
    ordinal: int,
) -> Dict[str, Any]:
    rep_index = _valid_effect_index(effects, group.get("representative_effect_index"))
    representative = _effect_at(effects, rep_index)
    effect_indices = [
        idx
        for idx in (_valid_effect_index(effects, value) for value in group.get("effect_indices", []) or [])
        if idx is not None
    ]
    if rep_index is not None and rep_index not in effect_indices:
        effect_indices.insert(0, rep_index)
    role = str(group.get("analysis_role") or "other")
    is_target = bool(group.get("is_primary_simulation_target"))
    finding_id = f"{_slug(study_id)}__finding_{ordinal:02d}"
    return {
        "finding_id": finding_id,
        "study_id": study_id,
        "study_name": study_name,
        "merge_group_id": group.get("merge_group_id"),
        "representative_effect_index": rep_index,
        "effect_indices": effect_indices,
        "role": role,
        "simulation_target": {
            "candidate": is_target,
            "reason": "primary consolidated effect" if is_target else f"demoted role: {role}",
        },
        "IV": representative.get("IV"),
        "DV": representative.get("DV"),
        "effecttype": representative.get("effecttype"),
        "direction": representative.get("direction"),
        "table_or_page_location": representative.get("table_or_page_location"),
        "reported_statistics": _reported_statistics(representative),
        "stats": representative.get("stats") if isinstance(representative.get("stats"), dict) else {},
        "materials_link_hints": {
            "materials_status": (representative.get("materials") or {}).get("status") if isinstance(representative.get("materials"), dict) else None,
            "manipulation_status": (representative.get("manipulation") or {}).get("status") if isinstance(representative.get("manipulation"), dict) else None,
            "items_status": (representative.get("items") or {}).get("status") if isinstance(representative.get("items"), dict) else None,
        },
    }


def _has_reported_statistics(finding: Dict[str, Any]) -> bool:
    if str(finding.get("reported_statistics") or "").strip():
        return True
    stats = finding.get("stats") if isinstance(finding.get("stats"), dict) else {}
    for key in ("B", "b", "t", "f", "z", "chi_square", "D", "eta_square", "p_value", "sig"):
        if stats.get(key) not in (None, "", [], [None, None]):
            return True
    ci = stats.get("ci")
    return isinstance(ci, list) and any(item is not None for item in ci)


def _ensure_fallback_simulation_target(findings: List[Dict[str, Any]]) -> None:
    """Promote one valid finding when a study only has demoted simple effects."""
    if any(finding.get("simulation_target", {}).get("candidate") is True for finding in findings):
        return
    demoted_only_roles = {"secondary_dv", "manipulation_check", "check", "mediation", "correlation"}
    for finding in findings:
        role = str(finding.get("role") or "").strip().lower()
        if role in demoted_only_roles:
            continue
        if not _has_reported_statistics(finding):
            continue
        target = finding.setdefault("simulation_target", {})
        target["candidate"] = True
        target["reason"] = (
            "fallback primary target: no main/primary target was available; "
            f"promoted reported {role or 'finding'}"
        )
        target["fallback"] = True
        return


def annotate_stage2_findings(
    paper: Dict[str, Any],
    *,
    recompute_consolidation: bool = True,
    stage1_json: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Attach `findings[]` and `stage2_finding_summary` to a Stage 2 paper dict."""
    studies = [study for study in paper.get("eligible_studies", []) or [] if isinstance(study, dict)]
    total_findings = 0
    primary_targets = 0
    by_role: Dict[str, int] = {}

    for study_index, study in enumerate(studies, start=1):
        study_name = str(study.get("study") or study.get("name") or f"Study {study_index}")
        study_id = str(study.get("study_id") or "").strip()
        if not study_id:
            study_id = canonical_sub_study_id(
                study_name,
                fallback=f"study_{study_index}",
            )
            study["study_id"] = study_id
        study["study_name"] = study_name
        if recompute_consolidation or not isinstance(study.get("consolidation_summary"), dict):
            annotate_study(study, study_id=study_id)
        effects = [effect for effect in study.get("effects", []) or [] if isinstance(effect, dict)]
        groups = study.get("consolidation_summary", {}).get("groups", [])
        findings = [
            _finding_from_group(
                study_id=study_id,
                study_name=study_name,
                effects=effects,
                group=group,
                ordinal=ordinal,
            )
            for ordinal, group in enumerate(groups, start=1)
            if isinstance(group, dict)
        ]
        _ensure_fallback_simulation_target(findings)
        study["findings"] = findings
        total_findings += len(findings)
        for finding in findings:
            role = str(finding.get("role") or "other")
            by_role[role] = by_role.get(role, 0) + 1
            if finding.get("simulation_target", {}).get("candidate"):
                primary_targets += 1

    paper["stage2_finding_summary"] = {
        "version": "stage2_findings_v1",
        "studies_total": len(studies),
        "raw_effects_total": sum(len(study.get("effects", []) or []) for study in studies),
        "findings_total": total_findings,
        "primary_simulation_target_candidates": primary_targets,
        "by_role": by_role,
    }
    apply_stage2_finding_contract(paper, stage1_json)
    return paper
