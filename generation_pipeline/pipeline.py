"""HumanStudy-Bench generation-pipeline orchestrator."""

from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation_pipeline.filters.replicability_filter import ReplicabilityFilter
from generation_pipeline.extractors.study_data_extractor import StudyDataExtractor
from generation_pipeline.patchers.slot_filler import SlotFiller
from generation_pipeline.patchers.patch_runner import run_patch
from generation_pipeline.stage1_study_contract import apply_stage1_study_contract
from generation_pipeline.stage2_findings import annotate_stage2_findings
from generation_pipeline.stage1_verifier import STAGE1_VERIFIER_VERSION, verify_stage1_inventory
from generation_pipeline.stage2_verifier import (
    DEFAULT_TIMEOUT as STAGE2_VERIFIER_TIMEOUT,
    STAGE2_VERIFIER_VERSION,
    verify_stage2_findings,
)
from generation_pipeline.stage4 import build_human_study_package
from generation_pipeline.stage5 import Stage5Options, run_stage5
from generation_pipeline.utils.output_formatter import OutputFormatter
from generation_pipeline.utils.review_parser import ReviewParser
from generation_pipeline.verification.schema_validator import summarize_report, validate_paper
from generation_pipeline.verification.stage_quality import build_stage1_quality, build_stage2_quality
from generation_pipeline.settings import AppSettings, load_settings, resolve_data_dir, resolve_llm_config, resolve_output_dir, resolve_runs_dir
from src.llm.factory import get_client


def paper_id_from_pdf(pdf_path: Path) -> str:
    return Path(pdf_path).stem.replace(" ", "_").replace("-", "_").lower()


def _count_regeneration_items(regeneration: Dict[str, Any]) -> int:
    count = 0
    for value in regeneration.values():
        if isinstance(value, list):
            count += len(value)
        elif isinstance(value, dict):
            count += len(value)
        elif value not in (None, "", False):
            count += 1
    return count


def _dedupe_feedback_items(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    output: list[Any] = []
    for value in values:
        try:
            key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        except TypeError:
            key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _stage1_regeneration_feedback(verification: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Stage 1 verifier output into filter regeneration feedback."""
    raw = verification.get("regeneration_instructions")
    raw = raw if isinstance(raw, dict) else {}
    feedback: Dict[str, Any] = {
        "missing_studies": list(raw.get("missing_studies") or []),
        "split_merge_corrections": list(raw.get("split_merge_corrections") or []),
        "comparison_group_corrections": list(
            raw.get("comparison_group_corrections") or []
        ),
        "study_field_corrections": list(raw.get("study_field_corrections") or []),
        "eligibility_corrections": list(raw.get("eligibility_corrections") or []),
    }

    if not any(feedback.values()):
        for item in verification.get("inventory_checks", []) or []:
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict") or "").lower()
            if verdict in {"", "ok"}:
                continue
            study = item.get("study") or "unknown study"
            issue = item.get("issue") or item.get("evidence") or verdict
            if verdict == "missing":
                feedback["missing_studies"].append(f"{study}: {issue}")
            elif verdict in {"split_issue", "merge_issue", "extra"}:
                feedback["split_merge_corrections"].append({"study": study, "reason": issue})
            else:
                feedback["split_merge_corrections"].append({"study": study, "reason": f"{verdict}: {issue}"})

        for item in verification.get("eligibility_checks", []) or []:
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict") or "").lower()
            if verdict in {"", "ok"}:
                continue
            study = item.get("study") or "unknown study"
            issue = item.get("issue") or item.get("evidence") or verdict
            feedback["eligibility_corrections"].append(
                {
                    "study": study,
                    "expected_label": item.get("expected_label") or "UNCERTAIN",
                    "reason": f"{verdict}: {issue}",
                }
            )

    return {
        key: _dedupe_feedback_items([item for item in values if item])
        for key, values in feedback.items()
    }


def _stage1_should_auto_refine(verification: Dict[str, Any]) -> bool:
    if verification.get("status") != "ok":
        return False
    if verification.get("overall") not in {"pass", "needs_review", "fail"}:
        return False
    return _count_regeneration_items(_stage1_regeneration_feedback(verification)) > 0


def _stage1_has_targeted_feedback(feedback: Dict[str, Any]) -> bool:
    return bool(
        feedback.get("study_field_corrections")
        or feedback.get("eligibility_corrections")
    ) and not any(
        feedback.get(key)
        for key in (
            "missing_studies",
            "split_merge_corrections",
            "comparison_group_corrections",
        )
    )


def _apply_stage1_field_corrections(
    result: Dict[str, Any],
    corrections: list[Dict[str, Any]],
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Apply already validated verifier replacements without rediscovering the paper."""
    candidate = deepcopy(result)
    experiments = [
        experiment
        for experiment in candidate.get("experiments", []) or []
        if isinstance(experiment, dict)
    ]
    experiment_by_id: Dict[str, Dict[str, Any]] = {}
    for experiment in experiments:
        for value in (
            experiment.get("study_id"),
            experiment.get("experiment_id"),
            experiment.get("study_name"),
        ):
            key = str(value or "").strip()
            if key:
                experiment_by_id[key] = experiment

    applied: list[Dict[str, Any]] = []
    for correction in corrections:
        if not isinstance(correction, dict):
            continue
        study_id = str(correction.get("study") or "").strip()
        field = str(correction.get("field") or "").strip()
        experiment = experiment_by_id.get(study_id)
        if experiment is None or field not in experiment or "expected_value" not in correction:
            continue
        expected_value = deepcopy(correction["expected_value"])
        if experiment.get(field) == expected_value:
            continue
        experiment[field] = expected_value
        refs = [
            str(ref)
            for ref in correction.get("evidence_block_ids") or []
            if str(ref).strip()
        ]
        experiment["evidence_refs"] = list(
            dict.fromkeys([*(experiment.get("evidence_refs") or []), *refs])
        )
        field_evidence = experiment.get("field_evidence")
        if not isinstance(field_evidence, dict):
            field_evidence = {}
            experiment["field_evidence"] = field_evidence
        field_evidence[field] = list(
            dict.fromkeys([*(field_evidence.get(field) or []), *refs])
        )
        applied.append(
            {
                "study": study_id,
                "field": field,
                "evidence_block_ids": refs,
                "correction_basis": correction.get("correction_basis"),
            }
        )
    candidate.pop("stage1_verification", None)
    candidate.pop("stage1_quality", None)
    return candidate, applied


def _apply_stage1_eligibility_corrections(
    result: Dict[str, Any],
    corrections: list[Dict[str, Any]],
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Apply validated eligibility labels and retain their evidence provenance."""
    candidate = deepcopy(result)
    experiments = [
        experiment
        for experiment in candidate.get("experiments", []) or []
        if isinstance(experiment, dict)
    ]
    experiment_by_id: Dict[str, Dict[str, Any]] = {}
    for experiment in experiments:
        for value in (
            experiment.get("study_id"),
            experiment.get("experiment_id"),
            experiment.get("study_name"),
        ):
            key = str(value or "").strip()
            if key:
                experiment_by_id[key] = experiment

    applied: list[Dict[str, Any]] = []
    for correction in corrections:
        if not isinstance(correction, dict):
            continue
        study_id = str(correction.get("study") or "").strip()
        expected_label = str(correction.get("expected_label") or "").strip().upper()
        experiment = experiment_by_id.get(study_id)
        if experiment is None or expected_label not in {"YES", "NO", "UNCERTAIN"}:
            continue
        if str(experiment.get("replicable") or "").strip().upper() == expected_label:
            continue
        experiment["replicable"] = expected_label
        refs = [
            str(ref)
            for ref in correction.get("evidence_block_ids") or []
            if str(ref).strip()
        ]
        experiment["evidence_refs"] = list(
            dict.fromkeys([*(experiment.get("evidence_refs") or []), *refs])
        )
        field_evidence = experiment.get("field_evidence")
        if not isinstance(field_evidence, dict):
            field_evidence = {}
            experiment["field_evidence"] = field_evidence
        field_evidence["replicable"] = list(
            dict.fromkeys([*(field_evidence.get("replicable") or []), *refs])
        )
        applied.append(
            {
                "study": study_id,
                "field": "replicable",
                "expected_label": expected_label,
                "evidence_block_ids": refs,
                "correction_basis": correction.get("correction_basis"),
            }
        )
    candidate.pop("stage1_verification", None)
    candidate.pop("stage1_quality", None)
    return candidate, applied


def _stage1_result_score(result: Dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Lower is better; only deterministic audit signals participate."""
    evidence = result.get("stage1_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    contract = result.get("stage1_study_contract")
    contract = contract if isinstance(contract, dict) else {}
    quality = result.get("stage1_quality")
    quality = quality if isinstance(quality, dict) else {}
    verification = result.get("stage1_verification")
    verification = verification if isinstance(verification, dict) else {}
    extraction_failures = 0 if evidence.get("extraction_complete") else 1
    contract_blocking = int(contract.get("blocking_issue_count") or 0)
    unresolved_relations = int(evidence.get("rejected_comparison_relation_count") or 0)
    quality_errors = sum(
        1
        for issue in quality.get("issues", []) or []
        if isinstance(issue, dict) and issue.get("severity") == "error"
    )
    verifier_items = _count_regeneration_items(
        _stage1_regeneration_feedback(verification)
    )
    if verification.get("status") == "error":
        verifier_items += 1000
    return (
        extraction_failures,
        contract_blocking,
        unresolved_relations,
        quality_errors,
        verifier_items,
    )


def _stage1_refinement_improves(
    before: tuple[int, int, int, int, int],
    after: tuple[int, int, int, int, int],
) -> bool:
    """Accept only non-regressing, material audit improvements."""
    if any(candidate > current for current, candidate in zip(before, after)):
        return False
    if not any(candidate < current for current, candidate in zip(before, after)):
        return False
    if any(after[index] < before[index] for index in range(4)):
        return True
    current_feedback = before[4]
    feedback_reduction = current_feedback - after[4]
    required_reduction = 1 if current_feedback <= 3 else max(2, (current_feedback + 4) // 5)
    return feedback_reduction >= required_reduction


def _stage1_refinement_snapshot(
    *,
    attempt: int,
    trigger: str,
    verification: Dict[str, Any],
    quality: Dict[str, Any],
    regeneration_feedback: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "attempt": attempt,
        "trigger": trigger,
        "verifier_status": verification.get("status"),
        "verifier_overall": verification.get("overall"),
        "verifier_confidence": verification.get("confidence"),
        "verifier_notes": verification.get("notes", ""),
        "quality_needs_human_review": quality.get("needs_human_review"),
        "quality_issues": quality.get("issues", [])[:20],
        "regeneration_instructions": regeneration_feedback,
    }


def _stage2_regeneration_feedback(verification: Dict[str, Any]) -> Dict[str, Any]:
    """Convert verifier output into the extractor feedback shape."""
    raw = verification.get("regeneration_instructions")
    raw = raw if isinstance(raw, dict) else {}
    feedback: Dict[str, Any] = {
        "missing_effects": list(raw.get("missing_effects") or []),
        "exact_stats_needed": list(raw.get("exact_stats_needed") or []),
        "data_corrections": list(raw.get("data_corrections") or []),
    }

    for item in verification.get("study_coverage", []) or []:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "").lower()
        if verdict in {"", "ok"}:
            continue
        study = item.get("study") or "unknown study"
        issue = item.get("issue") or item.get("evidence") or verdict
        reason = f"{study}: verifier marked study coverage as {verdict}; {issue}"
        if verdict == "missing":
            feedback["missing_effects"].append(reason)
        else:
            feedback["data_corrections"].append({"path": "$.eligible_studies", "reason": reason})

    for item in verification.get("finding_checks", []) or []:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "").lower()
        if verdict in {"", "ok"}:
            continue
        finding_id = item.get("finding_id") or "unknown finding"
        issue = item.get("issue") or item.get("evidence") or verdict
        reason = f"{finding_id}: verifier marked finding as {verdict}; {issue}"
        if verdict == "wrong_stats":
            feedback["exact_stats_needed"].append({"finding_id": finding_id, "reason": reason})
        elif verdict == "unsupported":
            feedback["data_corrections"].append({"path": "$.eligible_studies[*].findings", "reason": reason})
        elif verdict == "missing":
            feedback["missing_effects"].append(reason)
        else:
            feedback["data_corrections"].append({"path": "$.eligible_studies[*].findings", "reason": reason})

    return {
        "missing_effects": [item for item in feedback["missing_effects"] if item],
        "exact_stats_needed": [item for item in feedback["exact_stats_needed"] if item],
        "data_corrections": [item for item in feedback["data_corrections"] if item],
    }


def _stage2_should_auto_refine(verification: Dict[str, Any]) -> bool:
    if verification.get("status") != "ok":
        return False
    if verification.get("overall") not in {"pass", "needs_review", "fail"}:
        return False
    return _count_regeneration_items(_stage2_regeneration_feedback(verification)) > 0


def _stage2_refinement_snapshot(
    *,
    attempt: int,
    trigger: str,
    verification: Dict[str, Any],
    quality: Dict[str, Any],
    regeneration_feedback: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "attempt": attempt,
        "trigger": trigger,
        "verifier_status": verification.get("status"),
        "verifier_overall": verification.get("overall"),
        "verifier_confidence": verification.get("confidence"),
        "verifier_notes": verification.get("notes", ""),
        "quality_needs_human_review": quality.get("needs_human_review"),
        "quality_issues": quality.get("issues", [])[:20],
        "regeneration_instructions": regeneration_feedback,
    }


def _write_stage2_schema_failure(
    *,
    paper_dir: Path,
    json_path: Path,
    raw_result: Dict[str, Any],
    schema_report: Any,
    label: str,
) -> None:
    """Persist invalid Stage 2 output so real-API failures are inspectable."""
    invalid_path = paper_dir / "stage2.invalid.json"
    report_path = paper_dir / "stage2.schema_report.json"
    invalid_path.write_text(json.dumps(raw_result, indent=2, ensure_ascii=False), encoding="utf-8")
    report_payload = schema_report.to_dict()
    report_payload["stage"] = 2
    report_payload["attempt_label"] = label
    report_payload["target_json"] = str(json_path)
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _schema_error_preview(schema_report: Any, *, limit: int = 8) -> str:
    errors = [
        issue
        for issue in schema_report.to_dict().get("issues", [])
        if issue.get("severity") == "error" and not issue.get("fixed")
    ]
    if not errors:
        return ""
    lines = []
    for issue in errors[:limit]:
        lines.append(f"  - {issue.get('path')}: {issue.get('message')}")
    if len(errors) > limit:
        lines.append(f"  - ... {len(errors) - limit} more error(s)")
    return "\n".join(lines)


class GenerationPipeline:
    """Source-grounded social-science study generation pipeline."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        output_dir: Optional[Path] = None,
        settings: Optional[AppSettings] = None,
    ):
        settings = settings or load_settings()
        llm_config = resolve_llm_config(
            settings,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
        )
        self.provider = llm_config.provider
        self.model = llm_config.model
        self.client = get_client(
            provider=self.provider,
            model=self.model,
            api_key=llm_config.api_key,
            api_base=llm_config.api_base,
        )
        self.output_dir = Path(output_dir) if output_dir else resolve_output_dir(settings)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.filter = ReplicabilityFilter(self.client)
        self.extractor = StudyDataExtractor(self.client)
        self.slot_filler = SlotFiller(self.client)

    def paper_output_dir(self, paper_id: str) -> Path:
        out = self.output_dir / paper_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    def run_stage1(
        self,
        pdf_path: Path,
        *,
        verify_inventory: bool = True,
        extraction_timeout: float | None = 300.0,
        verifier_timeout: float | None = 300.0,
        workers: int = 4,
        auto_refine_attempts: int = 2,
        force: bool = False,
    ) -> Tuple[Path, Path, Dict[str, Any]]:
        print(f"Running Stage 1: Study Inventory and Simulation Eligibility for {pdf_path.name}")
        paper_id = paper_id_from_pdf(pdf_path)
        paper_dir = self.paper_output_dir(paper_id)

        md_path = paper_dir / "stage1.md"
        json_path = paper_dir / "stage1.json"

        def audit_result(
            result: Dict[str, Any],
            label: str,
            *,
            boundary_baseline: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            apply_stage1_study_contract(result)
            if verify_inventory:
                print(f"  Stage 1 verifier ({label}): checking study inventory coverage", flush=True)
                try:
                    result["stage1_verification"] = verify_stage1_inventory(
                        result,
                        pdf_path,
                        self.client,
                        pdf_artifacts_dir=paper_dir / "pdf_artifacts",
                        artifacts_dir=paper_dir / "stage1_artifacts" / "verifier",
                        timeout=verifier_timeout,
                        workers=workers,
                        force=bool(force and label == "initial"),
                        boundary_baseline=boundary_baseline,
                    )
                except Exception as exc:
                    result["stage1_verification"] = {
                        "version": STAGE1_VERIFIER_VERSION,
                        "status": "error",
                        "overall": "needs_review",
                        "error": f"{type(exc).__name__}: {exc}",
                        "inventory_checks": [],
                        "comparison_group_checks": [],
                        "field_checks": [],
                        "eligibility_checks": [],
                        "regeneration_instructions": {
                            "missing_studies": [],
                            "split_merge_corrections": [],
                            "comparison_group_corrections": [],
                            "study_field_corrections": [],
                            "eligibility_corrections": [],
                        },
                        "window_audit": {
                            "window_count": 0,
                            "full_document_llm_calls": 0,
                            "windows": [],
                        },
                        "study_audit": {
                            "study_count": 0,
                            "full_document_llm_calls": 0,
                            "all_cited_evidence_included": False,
                            "studies": [],
                        },
                    }
            else:
                result["stage1_verification"] = {
                    "version": STAGE1_VERIFIER_VERSION,
                    "status": "skipped",
                    "overall": "needs_review",
                    "notes": "Disabled by --no-stage1-verifier.",
                    "inventory_checks": [],
                    "comparison_group_checks": [],
                    "field_checks": [],
                    "eligibility_checks": [],
                    "regeneration_instructions": {
                        "missing_studies": [],
                        "split_merge_corrections": [],
                        "comparison_group_corrections": [],
                        "study_field_corrections": [],
                        "eligibility_corrections": [],
                    },
                    "window_audit": {
                        "window_count": 0,
                        "full_document_llm_calls": 0,
                        "windows": [],
                    },
                    "study_audit": {
                        "study_count": 0,
                        "full_document_llm_calls": 0,
                        "all_cited_evidence_included": False,
                        "studies": [],
                    },
                }
            result["stage1_quality"] = build_stage1_quality(result)
            verification = result.get("stage1_verification")
            if (
                label != "initial"
                and verify_inventory
                and isinstance(verification, dict)
                and verification.get("status") == "error"
            ):
                raise RuntimeError(
                    "Stage 1 refinement verification failed: "
                    + str(verification.get("error") or "unknown verifier error")
                )
            return result

        def run_attempt(
            regeneration_feedback: Optional[Dict[str, Any]],
            label: str,
        ) -> Dict[str, Any]:
            result = self.filter.process(
                pdf_path,
                regeneration_instructions=regeneration_feedback,
                pdf_artifacts_dir=paper_dir / "pdf_artifacts",
                artifacts_dir=paper_dir / "stage1_artifacts",
                timeout=extraction_timeout,
                workers=workers,
                force=bool(force and label == "initial"),
            )
            result["paper_id"] = paper_id
            return audit_result(result, label)

        def run_targeted_repair_attempt(
            current: Dict[str, Any],
            feedback: Dict[str, Any],
            label: str,
        ) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
            boundary_baseline = current.get("stage1_verification")
            boundary_baseline = (
                boundary_baseline if isinstance(boundary_baseline, dict) else None
            )
            candidate, applied = _apply_stage1_field_corrections(
                current,
                list(feedback.get("study_field_corrections") or []),
            )
            candidate, eligibility_applied = _apply_stage1_eligibility_corrections(
                candidate,
                list(feedback.get("eligibility_corrections") or []),
            )
            applied.extend(eligibility_applied)
            candidate["paper_id"] = paper_id
            candidate["stage1_targeted_repairs"] = {
                "version": "stage1_targeted_repair_v2",
                "source": "independent_stage1_verifier",
                "applied": applied,
            }
            return (
                audit_result(
                    candidate,
                    label,
                    boundary_baseline=boundary_baseline,
                ),
                applied,
            )

        result = run_attempt(None, "initial")
        best_score = _stage1_result_score(result)
        refinement_history = []
        accepted_refinements = 0
        stopped_reason = "no_refinement_needed"
        max_refines = max(0, int(auto_refine_attempts or 0))
        for attempt in range(1, max_refines + 1):
            verification = result.get("stage1_verification") if isinstance(result.get("stage1_verification"), dict) else {}
            if not verify_inventory or not _stage1_should_auto_refine(verification):
                stopped_reason = "verifier_has_no_grounded_feedback"
                break
            feedback = _stage1_regeneration_feedback(verification)
            use_targeted_repairs = _stage1_has_targeted_feedback(feedback)
            snapshot = _stage1_refinement_snapshot(
                attempt=attempt,
                trigger=(
                    "stage1_verifier_targeted_repairs"
                    if use_targeted_repairs
                    else "stage1_verifier_full_recompile"
                ),
                verification=verification,
                quality=result.get("stage1_quality") if isinstance(result.get("stage1_quality"), dict) else {},
                regeneration_feedback=feedback,
            )
            try:
                if use_targeted_repairs:
                    print(
                        f"  Stage 1 auto-refine {attempt}/{max_refines}: applying "
                        f"{len(feedback['study_field_corrections']) + len(feedback['eligibility_corrections'])} "
                        "validated targeted repair(s)",
                        flush=True,
                    )
                    candidate_result, applied = run_targeted_repair_attempt(
                        result,
                        feedback,
                        f"targeted_refine_{attempt}",
                    )
                    snapshot["applied_field_repairs"] = applied
                else:
                    print(
                        f"  Stage 1 auto-refine {attempt}/{max_refines}: recompiling with "
                        f"{_count_regeneration_items(feedback)} boundary/eligibility feedback item(s)",
                        flush=True,
                    )
                    candidate_result = run_attempt(feedback, f"auto_refine_{attempt}")
            except Exception as exc:
                stopped_reason = "candidate_attempt_failed"
                snapshot["score_before"] = list(best_score)
                snapshot["accepted"] = False
                snapshot["error"] = f"{type(exc).__name__}: {exc}"
                refinement_history.append(snapshot)
                print(
                    f"  Stage 1 auto-refine {attempt}: candidate failed; keeping "
                    f"the prior result ({type(exc).__name__}: {exc})",
                    flush=True,
                )
                break
            candidate_score = _stage1_result_score(candidate_result)
            accepted = _stage1_refinement_improves(best_score, candidate_score)
            snapshot["score_before"] = list(best_score)
            snapshot["candidate_score"] = list(candidate_score)
            snapshot["accepted"] = accepted
            refinement_history.append(snapshot)
            if not accepted:
                stopped_reason = "candidate_did_not_improve_audit_score"
                print(
                    f"  Stage 1 auto-refine {attempt}: rejected candidate score "
                    f"{candidate_score}; keeping {best_score}",
                    flush=True,
                )
                break
            result = candidate_result
            best_score = candidate_score
            accepted_refinements += 1
            candidate_verification = result.get("stage1_verification")
            candidate_verification = (
                candidate_verification
                if isinstance(candidate_verification, dict)
                else {}
            )
            if not _stage1_should_auto_refine(candidate_verification):
                stopped_reason = "verifier_has_no_grounded_feedback"
                break
            stopped_reason = "max_attempts_reached"

        if refinement_history:
            result["stage1_refinement"] = {
                "version": "stage1_auto_refine_v3",
                "attempts": len(refinement_history),
                "accepted_attempts": accepted_refinements,
                "max_attempts": max_refines,
                "final_score": list(best_score),
                "stopped_reason": stopped_reason,
                "final_verifier_overall": result.get("stage1_verification", {}).get("overall")
                if isinstance(result.get("stage1_verification"), dict)
                else None,
            }
            result["stage1_refinement_history"] = refinement_history

        md_content = OutputFormatter.format_stage1_review(result)

        md_path.write_text(md_content, encoding="utf-8")
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✓ Stage 1 complete.\n  MD : {md_path}\n  JSON: {json_path}")
        return md_path, json_path, result

    def check_stage1_review(self, review_file: Path) -> Dict[str, Any]:
        parsed = ReviewParser.parse(review_file)
        return {
            "action": ReviewParser.get_action(parsed["review_status"]),
            "review_status": parsed["review_status"],
            "comments": parsed["comments"],
            "checklists": parsed["checklists"],
        }

    def run_stage2(
        self,
        stage1_json_path: Path,
        pdf_path: Path,
        regeneration_instructions_path: Optional[Path] = None,
        *,
        grounded: bool = False,
        ground_threshold: float = 90.0,
        ground_k: int = 8,
        ground_timeout: float | None = 60.0,
        ground_workers: int = 4,
        extraction_timeout: float | None = 300.0,
        extraction_workers: int = 4,
        verify_findings: bool = True,
        verifier_timeout: float | None = STAGE2_VERIFIER_TIMEOUT,
        auto_refine_attempts: int = 1,
        force: bool = False,
    ) -> Tuple[Path, Path, Dict[str, Any]]:
        print(f"Running Stage 2: Study and Finding Extraction for {pdf_path.name}")
        if not stage1_json_path.exists():
            raise FileNotFoundError(f"Stage1 JSON not found: {stage1_json_path}")
        stage1_result = json.loads(stage1_json_path.read_text(encoding="utf-8"))
        if not isinstance(stage1_result, dict):
            raise ValueError(f"Stage1 result not a dict: {type(stage1_result)}")
        stage1_evidence = stage1_result.get("stage1_evidence")
        if not isinstance(stage1_evidence, dict):
            raise ValueError(
                "Stage 2 requires a bounded-window Stage 1 evidence audit; rerun Stage 1 "
                "with the current pipeline."
            )
        if not stage1_evidence.get("all_mentions_assigned") or not stage1_evidence.get(
            "extraction_complete"
        ):
            raise ValueError(
                "Stage 1 evidence compilation is incomplete; resolve discovery/reconciliation "
                "or per-study extraction errors before Stage 2."
            )

        regen = None
        if regeneration_instructions_path:
            if regeneration_instructions_path.exists():
                regen = json.loads(regeneration_instructions_path.read_text(encoding="utf-8"))
                print(f"Using regeneration feedback: {regeneration_instructions_path.name}")
            else:
                print(f"Warning: regeneration file not found: {regeneration_instructions_path}")

        if grounded:
            print(
                "  Stage 2 legacy grounding: checking per-effect material slots "
                "(disabled by default; Stage 3 owns study-level materials)",
                flush=True,
            )

        paper_id = stage1_result.get("paper_id", "unknown")
        paper_dir = self.paper_output_dir(str(paper_id))
        md_path = paper_dir / "stage2.md"
        json_path = paper_dir / "stage2.json"

        def run_attempt(regeneration_feedback: Optional[Dict[str, Any]], label: str) -> Dict[str, Any]:
            raw_result = self.extractor.process(
                stage1_result,
                pdf_path,
                regeneration_instructions=regeneration_feedback,
                grounded=grounded,
                ground_threshold=ground_threshold,
                ground_k=ground_k,
                ground_timeout=ground_timeout,
                ground_workers=ground_workers,
                pdf_artifacts_dir=paper_dir / "pdf_artifacts",
                artifacts_dir=paper_dir / "stage2_artifacts" / "studies",
                extraction_timeout=extraction_timeout,
                extraction_workers=extraction_workers,
                force=bool(force and label == "initial"),
            )
            result, schema_report = validate_paper(raw_result, repair=True, path=json_path)
            if not schema_report.valid:
                _write_stage2_schema_failure(
                    paper_dir=paper_dir,
                    json_path=json_path,
                    raw_result=raw_result,
                    schema_report=schema_report,
                    label=label,
                )
                preview = _schema_error_preview(schema_report)
                if preview:
                    print(f"Stage 2 schema validation errors ({label}):\n{preview}", flush=True)
                print(
                    "Stage 2 invalid artifacts written:\n"
                    f"  JSON  : {paper_dir / 'stage2.invalid.json'}\n"
                    f"  Report: {paper_dir / 'stage2.schema_report.json'}",
                    flush=True,
                )
                raise ValueError(f"Stage 2 schema validation failed: {summarize_report(schema_report)}")
            if schema_report.changed or schema_report.issues:
                print(f"Stage 2 schema validation ({label}): {summarize_report(schema_report)}")

            result = annotate_stage2_findings(result, stage1_json=stage1_result)
            if verify_findings:
                print(f"  Stage 2 verifier ({label}): checking study/finding coverage", flush=True)
                try:
                    result["stage2_verification"] = verify_stage2_findings(
                        result,
                        stage1_result,
                        pdf_path,
                        self.client,
                        pdf_artifacts_dir=paper_dir / "pdf_artifacts",
                        artifacts_dir=paper_dir / "stage2_artifacts" / "verifier",
                        timeout=verifier_timeout,
                        workers=extraction_workers,
                        force=bool(force and label == "initial"),
                    )
                except Exception as exc:
                    result["stage2_verification"] = {
                        "version": STAGE2_VERIFIER_VERSION,
                        "status": "error",
                        "overall": "needs_review",
                        "error": f"{type(exc).__name__}: {exc}",
                        "study_coverage": [],
                        "finding_checks": [],
                        "regeneration_instructions": {
                            "missing_effects": [],
                            "exact_stats_needed": [],
                            "data_corrections": [],
                        },
                    }
            else:
                result["stage2_verification"] = {
                    "version": STAGE2_VERIFIER_VERSION,
                    "status": "skipped",
                    "overall": "needs_review",
                    "notes": "Disabled by --no-stage2-verifier.",
                    "study_coverage": [],
                    "finding_checks": [],
                    "regeneration_instructions": {
                        "missing_effects": [],
                        "exact_stats_needed": [],
                        "data_corrections": [],
                    },
                }
            result["stage2_quality"] = build_stage2_quality(result, stage1_result)
            return result

        result = run_attempt(regen, "initial")
        refinement_history = []
        max_refines = max(0, int(auto_refine_attempts or 0))
        for attempt in range(1, max_refines + 1):
            verification = result.get("stage2_verification") if isinstance(result.get("stage2_verification"), dict) else {}
            if not verify_findings or not _stage2_should_auto_refine(verification):
                break
            feedback = _stage2_regeneration_feedback(verification)
            refinement_history.append(
                _stage2_refinement_snapshot(
                    attempt=attempt,
                    trigger="stage2_verifier",
                    verification=verification,
                    quality=result.get("stage2_quality") if isinstance(result.get("stage2_quality"), dict) else {},
                    regeneration_feedback=feedback,
                )
            )
            print(
                f"  Stage 2 auto-refine {attempt}/{max_refines}: rerunning with "
                f"{_count_regeneration_items(feedback)} verifier feedback item(s)",
                flush=True,
            )
            result = run_attempt(feedback, f"auto_refine_{attempt}")

        if refinement_history:
            result["stage2_refinement"] = {
                "version": "stage2_auto_refine_v1",
                "attempts": len(refinement_history),
                "max_attempts": max_refines,
                "final_verifier_overall": result.get("stage2_verification", {}).get("overall")
                if isinstance(result.get("stage2_verification"), dict)
                else None,
            }
            result["stage2_refinement_history"] = refinement_history

        md_content = OutputFormatter.format_stage2_review(result)

        md_path.write_text(md_content, encoding="utf-8")
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✓ Stage 2 complete.\n  MD : {md_path}\n  JSON: {json_path}")
        return md_path, json_path, result

    def check_stage2_review(self, review_file: Path) -> Dict[str, Any]:
        parsed = ReviewParser.parse(review_file)
        return {
            "action": ReviewParser.get_action(parsed["review_status"]),
            "review_status": parsed["review_status"],
            "comments": parsed["comments"],
            "checklists": parsed["checklists"],
        }

    # ------------------------------------------------------------------
    # Stage 3 — patch existing JSON corpus (fill slots + upgrade sample)
    # ------------------------------------------------------------------
    def run_stage3(
        self,
        json_dir: Path,
        pdf_dir: Path,
        *,
        only: Optional[list] = None,
        overwrite_filled: bool = False,
        backup: bool = True,
        dry_run: bool = False,
        use_fetched_sources: bool = True,
        source_dirs: Optional[list] = None,
    ) -> Dict[str, str]:
        print(f"Running Stage 3: Patch slots in {json_dir} using PDFs from {pdf_dir}")
        results = run_patch(
            Path(json_dir),
            Path(pdf_dir),
            filler=self.slot_filler,
            only=only,
            overwrite_filled=overwrite_filled,
            backup=backup,
            dry_run=dry_run,
            use_fetched_sources=use_fetched_sources,
            source_dirs=[Path(item) for item in source_dirs] if source_dirs else None,
        )
        ok = sum(1 for v in results.values() if v == "ok" or v == "dry-run-ok")
        print(f"✓ Stage 3 complete: {ok}/{len(results)} papers patched.")
        return results

    # ------------------------------------------------------------------
    # Stage 3 (OSF) - assemble per-study materials from OSF instruments
    # ------------------------------------------------------------------
    def run_stage3_paper(
        self,
        paper_dir: Path,
        *,
        stage2_path: Optional[Path] = None,
        pdf_path: Optional[Path] = None,
        osf_files_dir: Optional[Path] = None,
        slot_fill: bool = False,
        select_studies: bool = True,
        backup: bool = True,
        write: bool = True,
        allow_effect_slot_fallback: bool = False,
        selection_votes: int = 3,
        selection_timeout: Optional[float] = 60.0,
        pdf_material_timeout: Optional[float] = 120.0,
    ) -> Dict[str, Any]:
        """Build canonical Stage 3 artifacts for one paper directory.

        This is the OSF-linker path: it writes `stage3.json` and `stage3.md`
        without rewriting `stage2.json`, then adds `study_materials`,
        consolidation annotations, and simulation-target metadata.
        """
        from generation_pipeline.stage3 import run_stage3 as _run_stage3

        print(f"Running Stage 3 (OSF material assembly): {paper_dir}")
        result = _run_stage3(
            Path(paper_dir),
            stage2_path=stage2_path,
            pdf_path=pdf_path,
            osf_files_dir=osf_files_dir,
            filler=self.slot_filler if slot_fill else None,
            llm_client=self.client if select_studies else None,
            backup=backup,
            write=write,
            allow_effect_slot_fallback=allow_effect_slot_fallback,
            selection_votes=selection_votes,
            selection_timeout=selection_timeout,
            pdf_material_timeout=pdf_material_timeout,
        )
        materials = result["materials"]
        ready = sum(1 for m in materials.values() if m["readiness"]["ready"])
        kept = sum(1 for m in materials.values() if m.get("selection", {}).get("keep", True))
        print(
            f"✓ Stage 3 complete: wrote {Path(result['stage3_json']).name} + "
            f"{Path(result['stage3_md']).name}; {ready}/{len(materials)} ready, "
            f"{kept}/{len(materials)} selected."
        )
        return result

    def build_osf_study_materials(
        self,
        paper_dir: Path,
        *,
        select_studies: bool = True,
        write: bool = True,
        allow_effect_slot_fallback: bool = False,
        osf_files_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Rebuild study-level materials in an existing `stage3.json`."""
        from generation_pipeline.stage3 import build_study_materials

        print(f"Running Stage 3 (OSF): assembling materials for {paper_dir}")
        materials = build_study_materials(
            Path(paper_dir),
            osf_files_dir=osf_files_dir,
            llm_client=self.client if select_studies else None,
            write=write,
            allow_effect_slot_fallback=allow_effect_slot_fallback,
        )
        ready = sum(1 for m in materials.values() if m["readiness"]["ready"])
        kept = sum(1 for m in materials.values() if m.get("selection", {}).get("keep", True))
        print(
            f"✓ Stage 3 (OSF) complete: {ready}/{len(materials)} ready; "
            f"{kept}/{len(materials)} selected for simulation."
        )
        return materials

    # ------------------------------------------------------------------
    # Stage 4 - build HumanStudy-Bench study package
    # ------------------------------------------------------------------
    def run_stage4(
        self,
        json_path: Path,
        *,
        data_dir: Path,
        study_dir: Optional[Path] = None,
        study_id: Optional[str] = None,
        pdf_path: Optional[Path] = None,
        max_repair_iters: int = 3,
        use_llm: bool = True,
        generate_config: bool = True,
        update_registry: bool = True,
    ) -> Dict[str, Any]:
        del max_repair_iters
        print(f"Running Stage 4: Build HumanStudy-Bench study package for {json_path}")
        summary = build_human_study_package(
            Path(json_path),
            data_dir=Path(data_dir),
            study_dir=Path(study_dir) if study_dir else None,
            study_id=study_id,
            pdf_path=pdf_path,
            provider=self.provider,
            model=self.model,
            api_key=self.client.api_key,
            api_base=self.client.api_base,
            use_llm=use_llm,
            selection_llm_client=self.client if use_llm else None,
            generate_config=generate_config,
            update_registry=update_registry,
        )
        print(
            "✓ Stage 4 complete: "
            f"study_id={summary['study_id']} json_generation={summary['json_generation']} "
            f"config={summary['config_status']}"
        )
        if summary.get("config_error"):
            print(f"  Stage 4 config error: {summary['config_error']}")
        print(f"  Study package: {summary['study_dir']}")
        return summary

    # Stage 5 - run HumanStudy-Bench participant simulation
    def run_stage5(
        self,
        study: str | Path,
        *,
        data_dir: Path | None = None,
        runs_dir: Path | None = None,
        models: list[str] | None = None,
        options: Stage5Options | None = None,
    ) -> Dict[str, Any]:
        model_names = models or [self.model]

        stage5_options = options or Stage5Options()
        stage5_options.api_key = stage5_options.api_key or self.client.api_key
        stage5_options.api_base = stage5_options.api_base or self.client.api_base

        print(f"Running Stage 5: HumanStudy-Bench simulation for {study}")
        summary = run_stage5(
            study,
            runs_dir=Path(runs_dir) if runs_dir else resolve_runs_dir(),
            models=model_names,
            options=stage5_options,
            data_dir=Path(data_dir) if data_dir else resolve_data_dir(),
        )
        print(
            "✓ Stage 5 complete: "
            f"study_id={summary['study_id']} models={summary['completed']} "
            f"runs={summary['run_count']} use_real_llm={summary['use_real_llm']}"
        )
        return summary
