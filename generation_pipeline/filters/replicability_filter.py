"""
Replicability Filter - Stage 1 (ai-ethics)

Identifies which experiments in a moral / ethical / prosocial-behavior paper
are eligible for our extraction corpus, and pulls paper metadata.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generation_pipeline.filters.base_filter import BaseFilter
from generation_pipeline.stage1_study_contract import slugify
from generation_pipeline.utils.document_loader import DocumentLoader
from generation_pipeline.utils.pdf_extractor import extract_pdf_text


PDF_TEXT_MAX_CHARS = 400000


class ReplicabilityFilter(BaseFilter):
    """Stage 1 filter for ethics papers."""

    def process(
        self,
        pdf_path: Path,
        regeneration_instructions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        loader = DocumentLoader()
        pdf_info = loader.get_pdf_pages(pdf_path)
        pdf_text = extract_pdf_text(pdf_path, max_chars=PDF_TEXT_MAX_CHARS)
        prompt = self._build_prompt(pdf_path.name, len(pdf_info), regeneration_instructions)
        full_prompt = f"PDF content:\n\n{pdf_text}\n\n{prompt}"

        try:
            response = self.client.generate_content(prompt=full_prompt)
        except Exception as e:
            raise RuntimeError(
                f"Error calling LLM API: {e}. Check config/settings.yaml, provider API key env, "
                "and optional BASE_URL."
            )
        if response is None:
            raise ValueError("LLM returned None response.")

        return self._parse_response(response, pdf_path)

    def _build_prompt(
        self,
        pdf_name: str,
        num_pages: int,
        regeneration_instructions: Optional[Dict[str, Any]] = None,
    ) -> str:
        feedback_section = ""
        if regeneration_instructions:
            feedback_section = "\n\n" + "=" * 80 + "\nVALIDATION FEEDBACK FROM PREVIOUS STAGE 1 INVENTORY:\n" + "=" * 80 + "\n"
            if regeneration_instructions.get("missing_studies"):
                feedback_section += "MISSING STUDIES TO ADD OR RECHECK:\n"
                for item in regeneration_instructions["missing_studies"]:
                    feedback_section += f"  - {item}\n"
            if regeneration_instructions.get("split_merge_corrections"):
                feedback_section += "SPLIT/MERGE CORRECTIONS:\n"
                for item in regeneration_instructions["split_merge_corrections"]:
                    if isinstance(item, dict):
                        feedback_section += f"  - {item.get('study', '')}: {item.get('reason', '')}\n"
                    else:
                        feedback_section += f"  - {item}\n"
            if regeneration_instructions.get("eligibility_corrections"):
                feedback_section += "ELIGIBILITY CORRECTIONS:\n"
                for item in regeneration_instructions["eligibility_corrections"]:
                    if isinstance(item, dict):
                        feedback_section += (
                            f"  - {item.get('study', '')}: expected={item.get('expected_label', '')}; "
                            f"{item.get('reason', '')}\n"
                        )
                    else:
                        feedback_section += f"  - {item}\n"
            feedback_section += "=" * 80 + "\n\n"

        return f"""Analyze the research paper in the attached PDF: {pdf_name} ({num_pages} pages).

This corpus studies MORAL, ETHICAL, and PROSOCIAL BEHAVIOR in psychology / management.
Identify each experiment/study and decide whether it is ELIGIBLE for our extraction.
{feedback_section}

ELIGIBILITY CRITERIA (a study is eligible only if ALL hold):
- Empirical / experimental / quasi-experimental — not purely theoretical or qualitative.
- Reports at least one quantitative effect with a usable statistic (t, F, B/b, chi-square, z, d, eta_square, etc.).
- Outcome variable concerns moral / ethical / prosocial / norm-related behavior, judgment, intention, or perception
  (e.g. cheating, lying, helping, fairness, punishment, moral judgment, ethical leadership, CSR perception, etc.).
- Sample size and design are recoverable from the paper.

For EACH study/experiment, report:
- experiment_id (e.g. "Study 1", "Experiment 2a")
- study_id: stable machine id, snake_case if possible (e.g. "study_1",
  "study_2a"); keep unique within the paper
- experiment_name: short description
- study_name: short human-readable study name; usually same as experiment_name
- design_type: concise study design label (e.g. "between-subjects",
  "within-subjects", "mixed", "correlational", "field", "archival"), or null
  if the design cannot be recovered
- conditions_or_factors: array of concise strings naming the manipulated,
  measured, or repeated factors and their levels (e.g.
  ["message sidedness: one-sided vs two-sided", "attitude strength: continuous"]),
  or [] if no clear factors are recoverable
- input: what participants saw/read/did
- participant_task: one sentence describing the participant-facing task,
  including the stimulus category and response action when recoverable
- participants: brief description (N, source)
- output: what was measured
- candidate_source_hints: array of objects describing where Stage 3 should look
  for participant-facing materials, each with:
  {{"kind": "paper|appendix|supplement|osf|cited_scale|unknown",
    "description": "short source hint",
    "expected_fields": ["instructions", "stimulus", "items", "options", "conditions"]}}
- replicable: YES / NO / UNCERTAIN
- has_self_contained_materials: true/false (full stimulus / scale items present in paper or supplement)
- exclusion_reasons: [] if eligible, else list of reasons (e.g. "no quantitative outcome", "outcome not moral/ethical")
- missing_materials: empty string or description of what is missing (e.g. "scale items referenced from Reynolds 2008")

Also extract: paper_title, paper_authors, paper_abstract.

Respond ONLY with valid JSON (no markdown fences):
{{
    "paper_title": "...",
    "paper_authors": ["..."],
    "paper_abstract": "...",
    "experiments": [
        {{
            "experiment_id": "Study 1",
            "study_id": "study_1",
            "experiment_name": "...",
            "study_name": "...",
            "design_type": "between-subjects",
            "conditions_or_factors": ["condition: control vs treatment"],
            "input": "...",
            "participant_task": "...",
            "participants": "...",
            "output": "...",
            "candidate_source_hints": [
                {{
                    "kind": "paper",
                    "description": "stimulus and outcome items appear in Study 1 method section",
                    "expected_fields": ["instructions", "stimulus", "items", "options", "conditions"]
                }}
            ],
            "replicable": "YES",
            "has_self_contained_materials": true,
            "exclusion_reasons": [],
            "missing_materials": ""
        }}
    ],
    "overall_replicable": true,
    "confidence": 0.85,
    "notes": "..."
}}"""

    def _parse_response(self, response: str, pdf_path: Path) -> Dict[str, Any]:
        response_text = response.strip() if isinstance(response, str) else str(response).strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", response_text, re.DOTALL)
            if m:
                result = json.loads(m.group())
            else:
                raise ValueError(f"Could not parse JSON: {response_text[:200]}")

        if not isinstance(result, dict):
            result = {}
        _normalize_experiments(result)
        paper_id = pdf_path.stem.replace(" ", "_").replace("-", "_").lower()
        result["paper_id"] = paper_id
        return result


def _normalize_experiments(result: Dict[str, Any]) -> None:
    experiments = result.get("experiments")
    if not isinstance(experiments, list):
        return
    seen_ids: set[str] = set()
    for index, exp in enumerate(experiments, start=1):
        if not isinstance(exp, dict):
            continue
        label = str(exp.get("experiment_id") or exp.get("study_name") or exp.get("experiment_name") or f"Study {index}").strip()
        exp["experiment_id"] = label
        study_id = slugify(exp.get("study_id") or label, fallback=f"study_{index}")
        if study_id in seen_ids:
            study_id = f"{study_id}_{index}"
        seen_ids.add(study_id)
        exp["study_id"] = study_id
        name = str(exp.get("study_name") or exp.get("experiment_name") or label).strip()
        exp["study_name"] = name
        exp.setdefault("experiment_name", name)
        exp.setdefault("design_type", None)
        conditions = exp.get("conditions_or_factors", [])
        if conditions is None:
            exp["conditions_or_factors"] = []
        elif isinstance(conditions, str):
            text = conditions.strip()
            exp["conditions_or_factors"] = [text] if text else []
        elif isinstance(conditions, list):
            exp["conditions_or_factors"] = [
                str(item).strip() for item in conditions if str(item).strip()
            ]
        else:
            text = str(conditions).strip()
            exp["conditions_or_factors"] = [text] if text else []
        exp["participant_task"] = _normalize_participant_task(exp)
        exp["candidate_source_hints"] = _normalize_source_hints(exp)


def _normalize_participant_task(exp: Dict[str, Any]) -> str:
    task = str(exp.get("participant_task") or "").strip()
    if task:
        return task
    parts: List[str] = []
    input_text = str(exp.get("input") or "").strip()
    output_text = str(exp.get("output") or "").strip()
    if input_text:
        parts.append(f"participants saw/did: {input_text}")
    if output_text:
        parts.append(f"measured: {output_text}")
    return "; ".join(parts)


def _hint_dict(kind: str, description: str, expected_fields: List[str]) -> Dict[str, Any]:
    return {
        "kind": kind,
        "description": description,
        "expected_fields": expected_fields,
    }


def _normalize_source_hints(exp: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = exp.get("candidate_source_hints")
    hints: List[Dict[str, Any]] = []
    if isinstance(raw, str) and raw.strip():
        hints.append(_hint_dict("unknown", raw.strip(), ["instructions", "stimulus", "items", "options", "conditions"]))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                kind = str(item.get("kind") or "unknown").strip().lower() or "unknown"
                description = str(item.get("description") or item.get("source") or item.get("hint") or "").strip()
                expected = item.get("expected_fields")
                if isinstance(expected, str):
                    expected_fields = [expected.strip()] if expected.strip() else []
                elif isinstance(expected, list):
                    expected_fields = [str(field).strip() for field in expected if str(field).strip()]
                else:
                    expected_fields = []
                hints.append(_hint_dict(kind, description, expected_fields))
            elif str(item).strip():
                hints.append(_hint_dict("unknown", str(item).strip(), ["instructions", "stimulus", "items", "options", "conditions"]))
    if hints:
        return hints

    expected_fields = ["instructions", "stimulus", "items", "options", "conditions"]
    missing = str(exp.get("missing_materials") or "").strip()
    if missing:
        return [_hint_dict("unknown", missing, expected_fields)]
    if exp.get("has_self_contained_materials") is True:
        desc = str(exp.get("input") or exp.get("participant_task") or "participant-facing materials should be in the paper").strip()
        return [_hint_dict("paper", desc, expected_fields)]
    return [_hint_dict("unknown", "participant-facing material source not specified by Stage 1", expected_fields)]
