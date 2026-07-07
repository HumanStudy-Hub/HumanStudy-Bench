from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from generation_pipeline.utils.pdf_extractor import extract_pdf_text


PDF_TEXT_MAX_CHARS = 180000
DEFAULT_MAX_TOKENS = 8000
RAW_PREVIEW_CHARS = 700


def _compact_stage1(stage1_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for exp in stage1_json.get("experiments", []) or []:
        if not isinstance(exp, dict):
            continue
        out.append(
            {
                "experiment_id": exp.get("experiment_id"),
                "study_id": exp.get("study_id"),
                "experiment_name": exp.get("experiment_name"),
                "study_name": exp.get("study_name"),
                "replicable": exp.get("replicable"),
                "design_type": exp.get("design_type"),
                "conditions_or_factors": exp.get("conditions_or_factors"),
                "input": exp.get("input"),
                "participant_task": exp.get("participant_task"),
                "output": exp.get("output"),
                "candidate_source_hints": exp.get("candidate_source_hints"),
                "missing_materials": exp.get("missing_materials"),
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


def _loads_json(text: str) -> Dict[str, Any]:
    response_text = text.strip()
    if "```json" in response_text:
        response_text = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group())
    if not isinstance(parsed, dict):
        raise ValueError(f"Verifier response is not a JSON object: {type(parsed)}")
    return parsed


def _raw_preview(value: Any) -> str:
    text = str(value or "").replace("\n", "\\n")
    return text[:RAW_PREVIEW_CHARS]


def _retry_prompt(original_prompt: str, bad_response: str) -> str:
    return (
        original_prompt
        + "\n\nYour previous response could not be parsed as JSON. "
        "Return ONLY the compact JSON object requested above. Do not include "
        "markdown, prose, headings, or analysis. Previous response preview:\n"
        + _raw_preview(bad_response)
    )


def _normalize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    overall = str(report.get("overall") or "needs_review").strip().lower()
    if overall not in {"pass", "needs_review", "fail"}:
        overall = "needs_review"
    normalized = {
        "version": "stage2_verifier_v1",
        "status": "ok",
        "overall": overall,
        "confidence": report.get("confidence"),
        "study_coverage": report.get("study_coverage") if isinstance(report.get("study_coverage"), list) else [],
        "finding_checks": report.get("finding_checks") if isinstance(report.get("finding_checks"), list) else [],
        "regeneration_instructions": report.get("regeneration_instructions")
        if isinstance(report.get("regeneration_instructions"), dict)
        else {},
        "notes": report.get("notes", ""),
    }
    regen = normalized["regeneration_instructions"]
    regen.setdefault("missing_effects", [])
    regen.setdefault("exact_stats_needed", [])
    regen.setdefault("data_corrections", [])
    return normalized


def build_verifier_prompt(stage1_json: Dict[str, Any], stage2_json: Dict[str, Any], pdf_text: str) -> str:
    """Build the Stage 2 verifier prompt. Exposed for unit tests."""
    return f"""You are verifying an extraction from a psychology/management paper.

Check whether Stage 2 correctly covers the eligible studies and whether each
consolidated finding is supported by the paper text. Focus on study/effect/
finding/statistics correctness. Do NOT evaluate whether participant materials
are complete; Stage 3 handles materials.

Keep the verifier output compact: include every study in study_coverage, but in
finding_checks include only findings with problems or the most important
borderline cases. Do not list every ok finding.

PAPER TEXT:
{pdf_text}

STAGE 1 STUDY INVENTORY:
{json.dumps(_compact_stage1(stage1_json), ensure_ascii=False, indent=2)}

STAGE 2 CONSOLIDATED FINDINGS:
{json.dumps(_compact_stage2(stage2_json), ensure_ascii=False, indent=2)}

Return ONLY a compact JSON object with this schema. Do not include markdown,
analysis, commentary, or chain-of-thought. Keep evidence strings short.
{{
  "overall": "pass|needs_review|fail",
  "confidence": 0.0,
  "study_coverage": [
    {{
      "study": "Study 1",
      "verdict": "ok|missing|extra|split_issue|needs_review",
      "issue": "short issue or empty",
      "evidence": "short paper evidence"
    }}
  ],
  "finding_checks": [
    {{
      "finding_id": "study_1__finding_01",
      "verdict": "ok|unsupported|wrong_stats|wrong_role|duplicate|needs_review",
      "issue": "short issue or empty",
      "evidence": "short paper evidence"
    }}
  ],
  "regeneration_instructions": {{
    "missing_effects": ["..."],
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
    timeout: Optional[float] = 60.0,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    max_attempts: int = 2,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    if llm_client is None:
        return {
            "version": "stage2_verifier_v1",
            "status": "skipped",
            "overall": "needs_review",
            "notes": "No LLM client provided.",
            "study_coverage": [],
            "finding_checks": [],
            "regeneration_instructions": {
                "missing_effects": [],
                "exact_stats_needed": [],
                "data_corrections": [],
            },
        }
    text = pdf_text if pdf_text is not None else extract_pdf_text(Path(pdf_path), max_chars=PDF_TEXT_MAX_CHARS)
    prompt = build_verifier_prompt(stage1_json, stage2_json, text)
    attempts = max(1, int(max_attempts or 1))
    last_error: Optional[BaseException] = None
    last_response: Optional[str] = None
    current_prompt = prompt
    for attempt in range(1, attempts + 1):
        try:
            try:
                response = llm_client.generate_content(prompt=current_prompt, timeout=timeout, max_tokens=max_tokens)
            except TypeError:
                try:
                    response = llm_client.generate_content(prompt=current_prompt, timeout=timeout)
                except TypeError:
                    response = llm_client.generate_content(prompt=current_prompt)
            if response is None:
                raise ValueError("Stage 2 verifier LLM returned None.")
            last_response = str(response)
            return _normalize_report(_loads_json(last_response))
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            if last_response is not None:
                current_prompt = _retry_prompt(prompt, last_response)
            if retry_delay > 0:
                time.sleep(retry_delay)
    if last_error is not None:
        if last_response is not None:
            raise ValueError(
                f"{type(last_error).__name__}: {last_error}; "
                f"raw_response_preview={_raw_preview(last_response)}"
            ) from last_error
        raise last_error
    raise ValueError("Stage 2 verifier failed without an exception.")
