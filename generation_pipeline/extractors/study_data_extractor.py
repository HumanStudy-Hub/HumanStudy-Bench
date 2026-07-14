"""
Study & Data Extractor - Stage 2

Extracts study-level samples plus effect/finding/statistics records for
HumanStudy-Bench simulation candidates. The legacy
`materials / manipulation / items` effect slots remain in the schema for
backward compatibility, but participant-facing material recovery is a Stage 3
study-level responsibility.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generation_pipeline.extractors.base_extractor import BaseExtractor
from generation_pipeline.identifiers import canonical_sub_study_id
from generation_pipeline.utils.document_loader import DocumentLoader
from generation_pipeline.utils.pdf_extractor import extract_pdf_text


PDF_TEXT_MAX_CHARS = 400000


class StudyDataExtractor(BaseExtractor):
    """Extract findings/effect records for downstream material search."""

    def process(
        self,
        stage1_json: Dict[str, Any],
        pdf_path: Path,
        regeneration_instructions: Optional[Dict[str, Any]] = None,
        *,
        grounded: bool = False,
        ground_threshold: float = 90.0,
        ground_k: int = 8,
        ground_timeout: float | None = 60.0,
        ground_workers: int = 4,
    ) -> Dict[str, Any]:
        loader = DocumentLoader()
        pdf_info = loader.get_pdf_pages(pdf_path)
        pdf_text = extract_pdf_text(pdf_path, max_chars=PDF_TEXT_MAX_CHARS)
        prompt = self._build_prompt(stage1_json, pdf_path.name, len(pdf_info), regeneration_instructions)
        full_prompt = f"PDF content:\n\n{pdf_text}\n\n{prompt}"

        try:
            response = self.client.generate_content(prompt=full_prompt)
        except Exception as e:
            raise RuntimeError(f"Error calling LLM API: {e}.")
        if response is None:
            raise ValueError("LLM returned None response.")

        result = self._parse_response(response, stage1_json)
        _retain_stage1_candidates(result, stage1_json)
        if _effect_count(result) == 0:
            raise ValueError(
                "Stage 2 extraction returned zero effects for eligible Stage 1 studies. "
                "Re-run with regeneration feedback or inspect the Stage 1 experiment anchors."
            )
        if grounded:
            result = self._ground_slots(
                result,
                pdf_path,
                threshold=ground_threshold,
                k=ground_k,
                timeout=ground_timeout,
                workers=ground_workers,
            )
        return result

    def _ground_slots(
        self,
        result: Dict[str, Any],
        pdf_path: Path,
        *,
        threshold: float,
        k: int,
        timeout: float | None,
        workers: int,
    ) -> Dict[str, Any]:
        try:
            from generation_pipeline.extractors.grounded_slot_extractor import GroundedSlotExtractor
        except Exception as exc:  # pragma: no cover - import guard
            import warnings

            warnings.warn(f"Grounding pass unavailable ({exc}); returning ungrounded slots.", RuntimeWarning)
            return result
        if not result.get("eligible_studies"):
            return result
        extractor = GroundedSlotExtractor(
            self.client,
            threshold=threshold,
            k=k,
            slot_timeout=timeout,
            max_workers=workers,
        )
        grounded, report = extractor.reground_paper(
            result, pdf_path, only_verbatim_and_empty=True
        )
        grounded["_grounding_report"] = report
        return grounded

    def _build_prompt(
        self,
        stage1_json: Dict[str, Any],
        pdf_name: str,
        num_pages: int,
        regeneration_instructions: Optional[Dict[str, Any]] = None,
    ) -> str:
        experiments_info = json.dumps(
            _eligible_stage1_experiments(stage1_json),
            indent=2,
            ensure_ascii=False,
        )

        feedback_section = ""
        if regeneration_instructions:
            feedback_section = "\n\n" + "=" * 80 + "\nVALIDATION FEEDBACK FROM PREVIOUS EXTRACTION:\n" + "=" * 80 + "\n"
            if regeneration_instructions.get("missing_effects"):
                feedback_section += "MISSING EFFECTS:\n"
                for x in regeneration_instructions["missing_effects"]:
                    feedback_section += f"  - {x}\n"
            if regeneration_instructions.get("exact_stats_needed"):
                feedback_section += "EXACT STATISTICS REQUIRED:\n"
                for item in regeneration_instructions["exact_stats_needed"]:
                    feedback_section += f"  - {item.get('reason', '')}\n"
            if regeneration_instructions.get("data_corrections"):
                feedback_section += "DATA CORRECTIONS NEEDED:\n"
                for item in regeneration_instructions["data_corrections"]:
                    feedback_section += f"  - {item.get('reason', '')}\n"
            feedback_section += "=" * 80 + "\n\n"

        return f"""Extract per-effect records from the paper: {pdf_name} ({num_pages} pages).

STAGE 1 FILTER RESULTS (only extract studies that were marked replicable / eligible):
{experiments_info}
{feedback_section}
Stage 2 is topic-independent and is responsible for study/effect/finding/
statistics extraction, not final participant-facing materials. The output JSON
must match the project schema exactly. Extract ONLY the Stage 1 candidates shown
above. Preserve their `study_id` values exactly. For EACH candidate, list every
reported statistical effect separately under `effects[]`; downstream code will
consolidate those rows into study-level findings and simulation targets.

CRITICAL RULES:
- Numeric fields that are not reported MUST be `null` (not omitted, not empty string).
- `stats.ci` is ALWAYS a two-element array `[low, high]` — use `[null, null]` if not reported.
- `p_value` is stored as a STRING (e.g. ".03", "<.001").
- `sig` is one of: "sig", "ns", "marginal".
- `direction` is one of: "pos", "neg", "null".
- `effecttype` codes: "main" | "int" (interaction) | "simple" | "mediation" | "correlation".
- `size` is not effect size. It is only an integer participant/sample N for this
  effect analysis, or null. Put Cohen's d in `stats.D`, standardized beta/r in
  `stats.b`, eta-square in `stats.eta_square`, and exact result text in
  `reported_statistics_text`.
- `materials_notes` is only a one-line paper/source hint for Stage 3 material search.
- `materials`, `manipulation`, `items` are legacy compatibility slots. Do NOT
  reconstruct participant-facing materials here. Use
  `{{"status": null, "content": null}}` unless the paper prints a short exact
  quote that is directly useful as a source locator. Full instructions,
  stimuli, response options, anchors, and condition levels are recovered at
  study/sub-study level in Stage 3.
- `table_or_page_location`: e.g. "Table 1, p. 489" or "Study 2 Results, Openness".

SAMPLE FIELD — CRITICAL RULES:
- `sample` lives at the STUDY level (not inside each effect).
- It describes the TOTAL participant pool for the whole study — the number of
  people recruited/consented, not any subgroup or analysis cell.
- Most fields are nullable — only fill what the paper actually reports.
- `total_n`: The FULL study recruited / final-sample N (e.g., "We recruited
  4,001 participants via Prolific" → total_n=4001). Do NOT use a subgroup n
  or cell n that appears only in a statistics table or condition row.
- `analyzed_n`: Only fill when the paper reports a DIFFERENT n after exclusions.
  If not stated separately, leave null.
- `female_percent` and `male_percent`: ALWAYS 0-100 scale (e.g. 55.4, not 0.554).
- `platform` (STRICTLY controlled vocab — output EXACTLY one token, no sentences):
    MTurk | Prolific | CloudResearch | Undergraduate | Graduate |
    Lab | Organizational | Online | Field | Archival | Mixed | Other
  WRONG: "we recruited subjects via the platform CloudResearch."
  RIGHT: "CloudResearch"
  Do not infer a platform from the paper topic. Use only the recruitment/sample
  description; use Other when no more specific controlled value is supported.
- `inclusion_criteria`: VERBATIM quote of who was eligible (e.g. "self-identified
  Democrats", "U.S. adults who had been fired from a job"). null if not stated.
- `exclusion_criteria`: VERBATIM quote about who was excluded (attention checks,
  manipulation failure, outliers). null if not stated.
- `notes`: VERBATIM quote for any other unusual sampling detail. null otherwise.
- NEVER write LLM-composed summaries. Every string field is a direct quote or null.

ANALYSIS SAMPLE FIELDS (effect level, both optional):
- `analysis_n` (integer or null): fill ONLY when this specific analysis used a
  SUBSET of the study sample (e.g., Democrats-only, a condition arm, a cell).
  If the full study sample was used, leave null.
- `analysis_scope` (string or null, controlled vocab):
    full_sample | subgroup | condition | cell | simple_effect | other
  Fill only when analysis_n is non-null. Examples:
    Democrats-only subgroup → analysis_n=958, analysis_scope="subgroup"
    One arm of between-subjects → analysis_n=200, analysis_scope="condition"
    2×2 cell → analysis_n=101, analysis_scope="cell"

OUTPUT FORMAT — respond with ONLY this JSON (no markdown fences):

{{
  "paper_title": "...",
  "paper_metadata": {{
    "authors": ["..."],
    "year": 2024,
    "journal": "...",
    "doi": "...",
    "link": "..."
  }},
  "eligible_studies": [
    {{
      "study": "Study 1",
      "study_id": "study_1",
      "eligibility_rationale": "...",
      "sample": {{
        "total_n": 183,
        "analyzed_n": null,
        "mean_age": 36.5,
        "female_percent": 55.2,
        "male_percent": 44.8,
        "platform": "MTurk",
        "country": "United States",
        "inclusion_criteria": "<verbatim quote or null>",
        "exclusion_criteria": "<verbatim quote or null>",
        "notes": null
      }},
      "effects": [
        {{
          "platform": "MTurk",
          "effecttype": "int",
          "IV": "...",
          "DV": "...",
          "size": null,
          "direction": "pos",
          "mean_group1": null,
          "sd_group1": null,
          "mean_group2": null,
          "sd_group2": null,
          "stats": {{
            "B": 0.35, "b": null, "chi_square": null, "D": null,
            "eta_square": 0.027, "f": null, "t": 2.19, "z": null,
            "ci": [null, null],
            "p_value": ".03",
            "sig": "sig"
          }},
          "materials_notes": "One-line source/search hint for Stage 3 material recovery.",
          "table_or_page_location": "Study 1 Results",
          "analysis_n": null,
          "analysis_scope": null,
          "materials":    {{ "status": null, "content": null }},
          "manipulation": {{ "status": null, "content": null }},
          "items":        {{ "status": null, "content": null }}
        }}
      ]
    }}
  ]
}}"""

    def _parse_response(self, response: str, stage1_json: Dict[str, Any]) -> Dict[str, Any]:
        response_text = response.strip() if isinstance(response, str) else str(response).strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        try:
            result = _loads_llm_json(response_text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", response_text, re.DOTALL)
            if m:
                result = _loads_llm_json(m.group())
            else:
                raise ValueError(f"Could not parse JSON: {response_text[:200]}")

        if not isinstance(result, dict):
            result = {}
        result.setdefault("paper_id", stage1_json.get("paper_id", "unknown"))
        return result


def _eligible_stage1_experiments(stage1_json: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [
        experiment
        for experiment in stage1_json.get("experiments", []) or []
        if isinstance(experiment, dict)
        and str(experiment.get("replicable") or "").strip().upper() in {"YES", "UNCERTAIN"}
    ]


def _identity_keys(record: Dict[str, Any], fields: tuple[str, ...]) -> set[str]:
    return {
        canonical_sub_study_id(record.get(field))
        for field in fields
        if record.get(field)
    }


def _retain_stage1_candidates(result: Dict[str, Any], stage1_json: Dict[str, Any]) -> None:
    """Fail closed when Stage 2 invents or reintroduces an excluded study."""
    candidates = _eligible_stage1_experiments(stage1_json)
    indexed = [
        (
            experiment,
            _identity_keys(
                experiment,
                ("study_id", "experiment_id", "study_name", "experiment_name"),
            ),
        )
        for experiment in candidates
    ]
    retained: list[Dict[str, Any]] = []
    for study in result.get("eligible_studies", []) or []:
        if not isinstance(study, dict):
            continue
        keys = _identity_keys(
            study,
            ("study_id", "study", "name", "study_name", "experiment_id"),
        )
        matches = [experiment for experiment, candidate_keys in indexed if keys & candidate_keys]
        if len(matches) != 1:
            continue
        stage1_study_id = str(matches[0].get("study_id") or "").strip()
        if stage1_study_id:
            study["study_id"] = stage1_study_id
        retained.append(study)
    result["eligible_studies"] = retained


def _effect_count(result: Dict[str, Any]) -> int:
    return sum(
        len(study.get("effects", []) or [])
        for study in result.get("eligible_studies", []) or []
        if isinstance(study, dict)
    )


def _loads_llm_json(text: str) -> Any:
    """Parse JSON with a small repair pass for common LLM formatting mistakes."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(repaired)
