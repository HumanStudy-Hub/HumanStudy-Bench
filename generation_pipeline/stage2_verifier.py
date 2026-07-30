"""Per-study evidence verifier for Stage 2 findings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from generation_pipeline.pdf.evidence import PdfEvidenceIndex, _stage1_experiment
from generation_pipeline.pdf.models import EvidenceContext
from generation_pipeline.pdf.parser import parse_pdf_document
from generation_pipeline.stage1_compiler import cached_json_call


STAGE2_VERIFIER_VERSION = "stage2-evidence-verifier-v3"
STAGE2_VERIFIER_PROMPT_VERSION = "stage2-evidence-verifier-prompt-v3"
DEFAULT_MAX_TOKENS = 8000
DEFAULT_TIMEOUT = 300.0
DEFAULT_WORKERS = 4
STAGE2_VERIFIER_CONTEXT_MAX_CHARS = 56000


def _compact_stage1(stage1_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for exp in stage1_json.get("experiments", []) or []:
        if not isinstance(exp, dict):
            continue
        out.append(
            {
                "experiment_id": exp.get("experiment_id"),
                "study_id": exp.get("study_id"),
                "study_name": exp.get("study_name"),
                "replicable": exp.get("replicable"),
                "design_type": exp.get("design_type"),
                "conditions_or_factors": exp.get("conditions_or_factors"),
                "material_variants": exp.get("material_variants"),
                "input": exp.get("input"),
                "participant_task": exp.get("participant_task"),
                "output": exp.get("output"),
                "evidence_refs": exp.get("evidence_refs"),
            }
        )
    return out


def _compact_stage2(stage2_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    studies: List[Dict[str, Any]] = []
    for study in stage2_json.get("eligible_studies", []) or []:
        if not isinstance(study, dict):
            continue
        studies.append(
            {
                "study": study.get("study"),
                "study_id": study.get("study_id"),
                "sample": study.get("sample"),
                "raw_effects": len(study.get("effects", []) or []),
                "effects": [
                    {
                        "effect_index": index,
                        "effecttype": effect.get("effecttype"),
                        "IV": effect.get("IV"),
                        "DV": effect.get("DV"),
                        "direction": effect.get("direction"),
                        "stats": effect.get("stats"),
                        "reported_statistics_text": effect.get("reported_statistics_text"),
                        "location": effect.get("table_or_page_location"),
                        "evidence_refs": effect.get("evidence_refs"),
                    }
                    for index, effect in enumerate(study.get("effects", []) or [])
                    if isinstance(effect, dict)
                ],
                "findings": [
                    {
                        "finding_id": finding.get("finding_id"),
                        "role": finding.get("role"),
                        "simulation_target": finding.get("simulation_target"),
                        "representative_effect_index": finding.get("representative_effect_index"),
                        "effect_indices": finding.get("effect_indices"),
                        "IV": finding.get("IV"),
                        "DV": finding.get("DV"),
                        "reported_statistics": finding.get("reported_statistics"),
                        "location": finding.get("table_or_page_location"),
                    }
                    for finding in study.get("findings", []) or []
                    if isinstance(finding, dict)
                ],
            }
        )
    return studies


def _normalize_report(report: Dict[str, Any], *, valid_block_ids: set[str]) -> Dict[str, Any]:
    overall = str(report.get("overall") or "needs_review").strip().lower()
    if overall not in {"pass", "needs_review", "fail"}:
        overall = "needs_review"
    normalized = {
        "version": STAGE2_VERIFIER_VERSION,
        "status": "ok",
        "overall": overall,
        "confidence": _confidence(report.get("confidence")),
        "study_coverage": _list_of_dicts(report.get("study_coverage")),
        "finding_checks": _list_of_dicts(report.get("finding_checks")),
        "regeneration_instructions": (
            dict(report["regeneration_instructions"])
            if isinstance(report.get("regeneration_instructions"), dict)
            else {}
        ),
        "notes": str(report.get("notes") or "").strip(),
    }
    for key in ("study_coverage", "finding_checks"):
        for item in normalized[key]:
            item["evidence_block_ids"] = [
                str(ref)
                for ref in item.get("evidence_block_ids") or []
                if str(ref) in valid_block_ids
            ]
    regen = normalized["regeneration_instructions"]
    regen.setdefault("missing_effects", [])
    regen.setdefault("exact_stats_needed", [])
    regen.setdefault("data_corrections", [])
    return normalized


def build_verifier_prompt(
    stage1_json: Dict[str, Any],
    stage2_json: Dict[str, Any],
    pdf_text: str,
    *,
    valid_block_ids: Optional[Sequence[str]] = None,
) -> str:
    """Build one per-study Stage 2 verifier prompt."""
    return f"""You are verifying one study extraction from a social-science paper.

Check whether Stage 2 covers this Stage 1 study and whether every extracted
effect/finding/statistic is supported by the bounded evidence context. Focus on
study identity, sample, IV/DV, analysis role, effect direction, exact statistics,
and duplicates. Do not evaluate participant-material completeness; Stage 3 owns
materials. Do not borrow a result from another study visible nearby.

Valid evidence block IDs: {json.dumps(list(valid_block_ids or []), ensure_ascii=False)}

BOUNDED STUDY EVIDENCE:
{pdf_text}

STAGE 1 STUDY:
{json.dumps(_compact_stage1(stage1_json), ensure_ascii=False, indent=2)}

STAGE 2 EXTRACTION:
{json.dumps(_compact_stage2(stage2_json), ensure_ascii=False, indent=2)}

Return only this compact JSON object. Include the study in study_coverage. In
finding_checks include only problematic or genuinely borderline findings. Cite
valid evidence block IDs for every reported problem.
{{
  "overall": "pass|needs_review|fail",
  "confidence": 0.0,
  "study_coverage": [
    {{
      "study": "study id",
      "verdict": "ok|missing|extra|split_issue|needs_review",
      "issue": "short issue or empty",
      "evidence": "short direct evidence",
      "evidence_block_ids": ["valid block id"]
    }}
  ],
  "finding_checks": [
    {{
      "finding_id": "finding id or effect index",
      "verdict": "unsupported|wrong_stats|wrong_role|duplicate|needs_review",
      "issue": "short issue",
      "evidence": "short direct evidence",
      "evidence_block_ids": ["valid block id"]
    }}
  ],
  "regeneration_instructions": {{
    "missing_effects": ["short description"],
    "exact_stats_needed": [{{"finding_id": "...", "reason": "..."}}],
    "data_corrections": [{{"path": "$.eligible_studies[0]...", "reason": "..."}}]
  }},
  "notes": "one short summary"
}}"""


def verify_stage2_findings(
    stage2_json: Dict[str, Any],
    stage1_json: Dict[str, Any],
    pdf_path: Path,
    llm_client: Any,
    *,
    pdf_text: Optional[str] = None,
    pdf_artifacts_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    max_attempts: int = 2,
    retry_delay: float = 1.0,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
) -> Dict[str, Any]:
    del retry_delay
    if llm_client is None:
        return _skipped_report("No LLM client provided.")
    studies = [
        study
        for study in stage2_json.get("eligible_studies", []) or []
        if isinstance(study, dict)
    ]
    if not studies:
        return _skipped_report("No Stage 2 studies to verify.")

    index: Optional[PdfEvidenceIndex] = None
    if pdf_text is None:
        document = parse_pdf_document(
            Path(pdf_path),
            artifacts_dir=pdf_artifacts_dir,
            force=False,
            prefer_docling=True,
        )
        index = PdfEvidenceIndex(document)
    cache_dir = Path(artifacts_dir) if artifacts_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, Dict[str, Any]] = {}
    contexts: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    def run(study: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        study_id = str(study.get("study_id") or study.get("study") or "study")
        stage1_study = _stage1_experiment(stage1_json, study)
        single_stage1 = {"experiments": [stage1_study] if stage1_study else []}
        single_stage2 = {"eligible_studies": [study]}
        if pdf_text is not None:
            context_text = pdf_text
            block_ids: List[str] = []
            context_summary = {
                "mode": "provided_text",
                "block_ids": [],
                "pages": [],
                "context_chars": len(pdf_text),
            }
        else:
            assert index is not None
            anchors = _stage2_evidence_refs(study, stage2_json, stage1_study)
            context = index.context_for_study(
                study,
                stage1_json=stage1_json,
                gaps=["findings"],
                allow_full_document=False,
                anchor_refs=anchors,
                anchor_radius=1,
                use_facet_retrieval=not bool(anchors),
                max_chars=STAGE2_VERIFIER_CONTEXT_MAX_CHARS,
            )
            context_text = context.text
            block_ids = list(context.block_ids)
            context_summary = _context_summary(context)
        prompt = build_verifier_prompt(
            single_stage1,
            single_stage2,
            context_text,
            valid_block_ids=block_ids,
        )
        raw = cached_json_call(
            llm_client,
            prompt,
            cache_path=cache_dir / f"{study_id}.json" if cache_dir else None,
            prompt_version=STAGE2_VERIFIER_PROMPT_VERSION,
            timeout=timeout,
            max_tokens=int(max_tokens or DEFAULT_MAX_TOKENS),
            force=force,
            max_attempts=max_attempts,
            validator=lambda value: _validate_stage2_verifier_payload(
                value,
                valid_block_ids=set(block_ids),
            ),
        )
        return _normalize_report(raw, valid_block_ids=set(block_ids)), context_summary

    with ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as pool:
        future_map = {pool.submit(run, study): study for study in studies}
        for future in as_completed(future_map):
            study = future_map[future]
            study_id = str(study.get("study_id") or study.get("study") or "study")
            try:
                report, context = future.result()
                reports[study_id] = report
                contexts[study_id] = context
            except Exception as exc:
                errors[study_id] = f"{type(exc).__name__}: {exc}"
    if errors:
        detail = "; ".join(f"{key}={value}" for key, value in sorted(errors.items()))
        raise RuntimeError(
            "Stage 2 verification was incomplete; refusing to pass a partial audit: " + detail
        )
    return _aggregate_reports(studies, reports, contexts)


def _aggregate_reports(
    studies: Sequence[Dict[str, Any]],
    reports: Dict[str, Dict[str, Any]],
    contexts: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    coverage: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    missing_effects: List[Any] = []
    exact_stats: List[Any] = []
    corrections: List[Any] = []
    notes: List[str] = []
    confidences: List[float] = []
    overall_values: List[str] = []
    for study in studies:
        study_id = str(study.get("study_id") or study.get("study") or "study")
        report = reports[study_id]
        coverage.extend(report["study_coverage"])
        findings.extend(report["finding_checks"])
        regeneration = report["regeneration_instructions"]
        missing_effects.extend(regeneration.get("missing_effects") or [])
        exact_stats.extend(regeneration.get("exact_stats_needed") or [])
        corrections.extend(regeneration.get("data_corrections") or [])
        if report.get("notes"):
            notes.append(f"{study_id}: {report['notes']}")
        confidences.append(_confidence(report.get("confidence")))
        overall_values.append(str(report.get("overall") or "needs_review"))
    overall = "pass"
    if "fail" in overall_values:
        overall = "fail"
    elif "needs_review" in overall_values:
        overall = "needs_review"
    return {
        "version": STAGE2_VERIFIER_VERSION,
        "status": "ok",
        "overall": overall,
        "confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "study_coverage": _dedupe_dicts(coverage, ("study", "verdict", "issue")),
        "finding_checks": _dedupe_dicts(findings, ("finding_id", "verdict", "issue")),
        "regeneration_instructions": {
            "missing_effects": _dedupe_scalars(missing_effects),
            "exact_stats_needed": _dedupe_dicts(_list_of_dicts(exact_stats), ("finding_id", "reason")),
            "data_corrections": _dedupe_dicts(_list_of_dicts(corrections), ("path", "reason")),
        },
        "evidence_audit": {
            "study_count": len(studies),
            "full_document_llm_calls": 0,
            "study_contexts": contexts,
        },
        "notes": " | ".join(notes),
    }


def _validate_stage2_verifier_payload(
    report: Dict[str, Any],
    *,
    valid_block_ids: set[str],
) -> None:
    overall = str(report.get("overall") or "").strip().lower()
    if overall not in {"pass", "needs_review", "fail"}:
        raise ValueError(f"invalid Stage 2 verifier overall: {overall or 'missing'}")
    coverage = report.get("study_coverage")
    findings = report.get("finding_checks")
    if not isinstance(coverage, list) or not coverage:
        raise ValueError("Stage 2 verifier must include one study_coverage entry")
    if not isinstance(findings, list):
        raise ValueError("Stage 2 verifier finding_checks must be an array")
    for key, values in (("study_coverage", coverage), ("finding_checks", findings)):
        for position, item in enumerate(values):
            if not isinstance(item, dict):
                raise ValueError(f"{key}[{position}] must be an object")
            verdict = str(item.get("verdict") or "").strip().lower()
            refs = item.get("evidence_block_ids")
            if verdict != "ok" and (not isinstance(refs, list) or not refs):
                raise ValueError(f"{key}[{position}] problem has no evidence_block_ids")
            invalid = [str(ref) for ref in refs or [] if str(ref) not in valid_block_ids]
            if invalid:
                raise ValueError(f"{key}[{position}] has invalid evidence refs: {invalid}")


def _stage2_evidence_refs(
    study: Dict[str, Any],
    stage2_json: Dict[str, Any],
    stage1_study: Dict[str, Any],
) -> List[str]:
    values: List[Any] = [study.get("evidence_refs"), stage1_study.get("evidence_refs")]
    for effect in study.get("effects", []) or []:
        if isinstance(effect, dict):
            values.append(effect.get("evidence_refs"))
    evidence = stage2_json.get("stage2_evidence")
    contexts = evidence.get("study_contexts") if isinstance(evidence, dict) else {}
    context = contexts.get(study.get("study_id")) if isinstance(contexts, dict) else {}
    if isinstance(context, dict):
        values.append(context.get("block_ids"))
    return list(
        dict.fromkeys(
            str(ref)
            for refs in values
            if isinstance(refs, list)
            for ref in refs
            if str(ref).strip()
        )
    )


def _context_summary(context: EvidenceContext) -> Dict[str, Any]:
    return {
        "mode": context.mode,
        "block_ids": list(context.block_ids),
        "pages": list(context.pages),
        "source_chars": context.source_chars,
        "context_chars": context.context_chars,
    }


def _list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _dedupe_dicts(values: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    out: List[Dict[str, Any]] = []
    for value in values:
        identity = tuple(str(value.get(key) or "").strip().lower() for key in keys)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(value)
    return out


def _dedupe_scalars(values: Sequence[Any]) -> List[Any]:
    seen: set[str] = set()
    out: List[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return round(min(1.0, max(0.0, number)), 3)


def _skipped_report(notes: str) -> Dict[str, Any]:
    return {
        "version": STAGE2_VERIFIER_VERSION,
        "status": "skipped",
        "overall": "needs_review",
        "notes": notes,
        "study_coverage": [],
        "finding_checks": [],
        "regeneration_instructions": {
            "missing_effects": [],
            "exact_stats_needed": [],
            "data_corrections": [],
        },
        "evidence_audit": {
            "study_count": 0,
            "full_document_llm_calls": 0,
            "study_contexts": {},
        },
    }
