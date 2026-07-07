from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from generation_pipeline.utils.pdf_extractor import extract_pdf_text


PDF_TEXT_MAX_CHARS = 180000


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
                "design_type": exp.get("design_type"),
                "conditions_or_factors": exp.get("conditions_or_factors"),
                "input": exp.get("input"),
                "participant_task": exp.get("participant_task"),
                "participants": exp.get("participants"),
                "output": exp.get("output"),
                "candidate_source_hints": exp.get("candidate_source_hints"),
                "replicable": exp.get("replicable"),
                "has_self_contained_materials": exp.get("has_self_contained_materials"),
                "exclusion_reasons": exp.get("exclusion_reasons"),
                "missing_materials": exp.get("missing_materials"),
            }
        )
    return out


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


def _normalize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    overall = str(report.get("overall") or "needs_review").strip().lower()
    if overall not in {"pass", "needs_review", "fail"}:
        overall = "needs_review"
    normalized = {
        "version": "stage1_verifier_v1",
        "status": "ok",
        "overall": overall,
        "confidence": report.get("confidence"),
        "inventory_checks": report.get("inventory_checks") if isinstance(report.get("inventory_checks"), list) else [],
        "eligibility_checks": report.get("eligibility_checks") if isinstance(report.get("eligibility_checks"), list) else [],
        "regeneration_instructions": report.get("regeneration_instructions")
        if isinstance(report.get("regeneration_instructions"), dict)
        else {},
        "notes": report.get("notes", ""),
    }
    regen = normalized["regeneration_instructions"]
    regen.setdefault("missing_studies", [])
    regen.setdefault("split_merge_corrections", [])
    regen.setdefault("eligibility_corrections", [])
    return normalized


def build_verifier_prompt(stage1_json: Dict[str, Any], pdf_text: str) -> str:
    """Build the Stage 1 verifier prompt. Exposed for unit tests."""
    return f"""You are verifying a Stage 1 study inventory from a psychology/management paper.

Check whether the extracted inventory covers the paper's empirical studies and
whether each eligibility label is defensible for an automated HumanStudy-Bench
pipeline. Focus on study coverage, split/merge errors, design/sample anchors,
and eligibility labels. Do NOT require complete participant-facing materials;
Stage 3 handles material source recovery.

PAPER TEXT:
{pdf_text}

STAGE 1 STUDY INVENTORY:
{json.dumps(_compact_stage1(stage1_json), ensure_ascii=False, indent=2)}

Return ONLY a compact JSON object with this schema. Do not include markdown,
analysis, commentary, or chain-of-thought. Keep evidence strings short.
{{
  "overall": "pass|needs_review|fail",
  "confidence": 0.0,
  "inventory_checks": [
    {{
      "study": "Study 1",
      "verdict": "ok|missing|extra|split_issue|merge_issue|needs_review",
      "issue": "short issue or empty",
      "evidence": "short paper evidence"
    }}
  ],
  "eligibility_checks": [
    {{
      "study": "Study 1",
      "verdict": "ok|wrong_label|unsupported|needs_review",
      "expected_label": "YES|NO|UNCERTAIN",
      "issue": "short issue or empty",
      "evidence": "short paper evidence"
    }}
  ],
  "regeneration_instructions": {{
    "missing_studies": ["..."],
    "split_merge_corrections": [{{"study": "...", "reason": "..."}}],
    "eligibility_corrections": [{{"study": "...", "expected_label": "...", "reason": "..."}}]
  }},
  "notes": "one short summary"
}}"""


def verify_stage1_inventory(
    stage1_json: Dict[str, Any],
    pdf_path: Path,
    llm_client: Any,
    *,
    pdf_text: Optional[str] = None,
    timeout: Optional[float] = 60.0,
    max_tokens: Optional[int] = 3000,
    max_attempts: int = 2,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    if llm_client is None:
        return {
            "version": "stage1_verifier_v1",
            "status": "skipped",
            "overall": "needs_review",
            "notes": "No LLM client provided.",
            "inventory_checks": [],
            "eligibility_checks": [],
            "regeneration_instructions": {
                "missing_studies": [],
                "split_merge_corrections": [],
                "eligibility_corrections": [],
            },
        }
    text = pdf_text if pdf_text is not None else extract_pdf_text(Path(pdf_path), max_chars=PDF_TEXT_MAX_CHARS)
    prompt = build_verifier_prompt(stage1_json, text)
    attempts = max(1, int(max_attempts or 1))
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            try:
                response = llm_client.generate_content(prompt=prompt, timeout=timeout, max_tokens=max_tokens)
            except TypeError:
                try:
                    response = llm_client.generate_content(prompt=prompt, timeout=timeout)
                except TypeError:
                    response = llm_client.generate_content(prompt=prompt)
            if response is None:
                raise ValueError("Stage 1 verifier LLM returned None.")
            return _normalize_report(_loads_json(str(response)))
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            if retry_delay > 0:
                time.sleep(retry_delay)
    if last_error is not None:
        raise last_error
    raise ValueError("Stage 1 verifier failed without an exception.")
