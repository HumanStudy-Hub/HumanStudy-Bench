"""
Study & Data Extractor - Stage 2

Extracts study-level samples plus effect/finding/statistics records for
HumanStudy-Bench simulation candidates. The legacy
`materials / manipulation / items` effect slots remain in the schema for
backward compatibility, but participant-facing material recovery is a Stage 3
study-level responsibility.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generation_pipeline.extractors.base_extractor import BaseExtractor
from generation_pipeline.identifiers import canonical_sub_study_id
from generation_pipeline.pdf.evidence import PdfEvidenceIndex
from generation_pipeline.pdf.models import EvidenceContext
from generation_pipeline.pdf.parser import parse_pdf_document
from generation_pipeline.stage1_compiler import cached_json_call


STAGE2_EXTRACTION_VERSION = "stage2-evidence-extraction-v2"
STAGE2_CONTEXT_MAX_CHARS = 56000
STAGE2_MAX_TOKENS = 12000
STAGE2_DEFAULT_TIMEOUT = 300.0
STAGE2_DEFAULT_WORKERS = 4


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
        pdf_artifacts_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
        extraction_timeout: float | None = STAGE2_DEFAULT_TIMEOUT,
        extraction_workers: int = STAGE2_DEFAULT_WORKERS,
        force: bool = False,
    ) -> Dict[str, Any]:
        candidates = _eligible_stage1_experiments(stage1_json)
        if not candidates:
            return {
                "paper_id": stage1_json.get("paper_id", "unknown"),
                "paper_title": stage1_json.get("paper_title", ""),
                "paper_metadata": {},
                "eligible_studies": [],
                "stage2_evidence": {
                    "version": STAGE2_EXTRACTION_VERSION,
                    "full_document_llm_calls": 0,
                    "study_contexts": {},
                },
            }
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
        print(
            f"  Stage 2 evidence parser: {document.parser} pages={document.page_count} "
            f"blocks={len(document.blocks)} candidates={len(candidates)}",
            flush=True,
        )

        studies: Dict[str, Dict[str, Any]] = {}
        contexts: Dict[str, Dict[str, Any]] = {}
        paper_metadata: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        def run(candidate: Dict[str, Any]) -> tuple[Dict[str, Any], EvidenceContext, Dict[str, Any]]:
            study_id = str(candidate.get("study_id") or candidate.get("experiment_id") or "study")
            anchors = _stage1_evidence_refs(candidate, stage1_json)
            context = index.context_for_study(
                candidate,
                stage1_json=stage1_json,
                gaps=["findings"],
                allow_full_document=False,
                anchor_refs=anchors,
                anchor_radius=1,
                use_facet_retrieval=not bool(anchors),
                max_chars=STAGE2_CONTEXT_MAX_CHARS,
            )
            single_stage1 = {
                "paper_id": stage1_json.get("paper_id"),
                "paper_title": stage1_json.get("paper_title"),
                "experiments": [candidate],
                "comparison_groups": _comparison_groups_for_candidate(
                    stage1_json,
                    candidate,
                ),
            }
            prompt = self._build_prompt(
                single_stage1,
                pdf_path.name,
                document.page_count,
                regeneration_instructions,
            )
            prompt = (
                "Use only the bounded, study-targeted evidence context below. "
                "Do not borrow samples, conditions, or findings from another study. "
                "Every effect must include evidence_refs drawn from the valid block IDs.\n\n"
                f"VALID BLOCK IDS:\n{json.dumps(context.block_ids, ensure_ascii=False)}\n\n"
                f"EVIDENCE CONTEXT:\n{context.text}\n\n{prompt}"
            )
            payload = cached_json_call(
                self.client,
                prompt,
                cache_path=cache_dir / f"{study_id}.json" if cache_dir else None,
                prompt_version=STAGE2_EXTRACTION_VERSION,
                timeout=extraction_timeout,
                max_tokens=STAGE2_MAX_TOKENS,
                force=force,
                validator=lambda value: _validate_stage2_payload(value, candidate, context),
            )
            payload.setdefault("paper_id", stage1_json.get("paper_id", "unknown"))
            _retain_stage1_candidates(payload, single_stage1)
            extracted = payload.get("eligible_studies") or []
            if len(extracted) != 1 or not isinstance(extracted[0], dict):
                raise ValueError(
                    f"Stage 2 expected exactly one extracted study for {study_id}; got {len(extracted)}"
                )
            study = extracted[0]
            if not study.get("effects"):
                raise ValueError(f"Stage 2 returned zero effects for {study_id}")
            _ground_effect_refs(study, context)
            study["comparison_group_ids"] = [
                group.get("comparison_group_id")
                for group in single_stage1["comparison_groups"]
                if group.get("comparison_group_id")
            ]
            study["evidence_context"] = _context_summary(context)
            metadata = payload.get("paper_metadata")
            return study, context, metadata if isinstance(metadata, dict) else {}

        with ThreadPoolExecutor(max_workers=max(1, int(extraction_workers or 1))) as pool:
            future_map = {pool.submit(run, candidate): candidate for candidate in candidates}
            for future in as_completed(future_map):
                candidate = future_map[future]
                study_id = str(candidate.get("study_id") or candidate.get("experiment_id") or "study")
                try:
                    study, context, metadata = future.result()
                    studies[study_id] = study
                    contexts[study_id] = _context_summary(context)
                    if len(metadata) > len(paper_metadata):
                        paper_metadata = metadata
                    print(
                        f"  Stage 2 study extraction: {candidate.get('experiment_id') or study_id} "
                        f"effects={len(study.get('effects') or [])} context={context.context_chars} chars",
                        flush=True,
                    )
                except Exception as exc:
                    errors[study_id] = f"{type(exc).__name__}: {exc}"
        if errors:
            detail = "; ".join(f"{key}={value}" for key, value in sorted(errors.items()))
            raise RuntimeError(
                "Stage 2 evidence extraction was incomplete; refusing to emit a partial result: "
                + detail
            )

        result = {
            "paper_id": stage1_json.get("paper_id", "unknown"),
            "paper_title": stage1_json.get("paper_title", ""),
            "paper_metadata": paper_metadata,
            "eligible_studies": [
                studies[str(candidate.get("study_id") or candidate.get("experiment_id") or "study")]
                for candidate in candidates
            ],
            "stage1_comparison_groups": list(stage1_json.get("comparison_groups") or []),
            "stage2_evidence": {
                "version": STAGE2_EXTRACTION_VERSION,
                "source_sha256": document.source_sha256,
                "parser": document.parser,
                "full_document_llm_calls": 0,
                "study_contexts": contexts,
            },
        }
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
        comparison_groups = json.dumps(
            stage1_json.get("comparison_groups") or [],
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

SOURCE-EXPLICIT COMPARISON GROUPS INVOLVING THIS UNIT:
{comparison_groups}
{feedback_section}
Stage 2 is topic-independent and is responsible for study/effect/finding/
statistics extraction, not final participant-facing materials. The output JSON
must match the project schema exactly. Extract ONLY the Stage 1 candidates shown
above. Preserve their `study_id` values exactly. For EACH candidate, list every
reported statistical effect separately under `effects[]`; downstream code will
consolidate those rows into study-level findings and simulation targets.
    Each Stage 1 candidate is one top-level empirical unit. Its
    `material_variants` are conditions/forms inside that unit, not additional
    studies. Preserve variant-specific effects in `effects[]` while returning
    exactly one study record. A comparison group is context for a cross-unit
    result, not permission to copy another unit's sample, task, conditions, or
    materials into this record.

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
- `evidence_refs`: block IDs from the supplied bounded evidence context that
  directly support this effect and its reported statistics. Never invent IDs.

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
          "evidence_refs": ["p010_text_00123"],
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
        study["stage1_material_variants"] = list(
            matches[0].get("material_variants") or []
        )
        retained.append(study)
    result["eligible_studies"] = retained


def _effect_count(result: Dict[str, Any]) -> int:
    return sum(
        len(study.get("effects", []) or [])
        for study in result.get("eligible_studies", []) or []
        if isinstance(study, dict)
    )


def _stage1_evidence_refs(
    candidate: Dict[str, Any],
    stage1_json: Dict[str, Any],
) -> list[str]:
    collected: list[str] = []
    refs = candidate.get("evidence_refs")
    if isinstance(refs, list):
        collected.extend(str(ref) for ref in refs if str(ref).strip())
    evidence = stage1_json.get("stage1_evidence")
    contexts = evidence.get("study_contexts") if isinstance(evidence, dict) else {}
    context = contexts.get(candidate.get("study_id")) if isinstance(contexts, dict) else {}
    values = context.get("block_ids") if isinstance(context, dict) else []
    collected.extend(str(ref) for ref in values or [] if str(ref).strip())
    study_id = str(candidate.get("study_id") or "").strip()
    for group in stage1_json.get("comparison_groups", []) or []:
        if not isinstance(group, dict) or study_id not in {
            str(value) for value in group.get("member_study_ids") or []
        }:
            continue
        collected.extend(
            str(ref) for ref in group.get("evidence_refs") or [] if str(ref).strip()
        )
    return list(dict.fromkeys(collected))


def _comparison_groups_for_candidate(
    stage1_json: Dict[str, Any],
    candidate: Dict[str, Any],
) -> list[Dict[str, Any]]:
    study_id = str(candidate.get("study_id") or "").strip()
    return [
        group
        for group in stage1_json.get("comparison_groups", []) or []
        if isinstance(group, dict)
        and study_id in {str(value) for value in group.get("member_study_ids") or []}
    ]


def _ground_effect_refs(study: Dict[str, Any], context: EvidenceContext) -> None:
    valid = set(context.block_ids)
    study_refs: list[str] = []
    for effect in study.get("effects", []) or []:
        if not isinstance(effect, dict):
            continue
        refs = effect.get("evidence_refs")
        if not isinstance(refs, list):
            refs = []
        grounded = list(dict.fromkeys(str(ref) for ref in refs if str(ref) in valid))
        effect["evidence_refs"] = grounded
        study_refs.extend(grounded)
    study["evidence_refs"] = list(dict.fromkeys(study_refs))


def _validate_stage2_payload(
    payload: Dict[str, Any],
    candidate: Dict[str, Any],
    context: EvidenceContext,
) -> None:
    studies = payload.get("eligible_studies")
    if not isinstance(studies, list) or len(studies) != 1 or not isinstance(studies[0], dict):
        raise ValueError("Stage 2 per-study response must contain exactly one eligible_studies entry")
    study = studies[0]
    expected_id = str(candidate.get("study_id") or "")
    if str(study.get("study_id") or "") != expected_id:
        raise ValueError(
            f"Stage 2 changed study_id: expected={expected_id}, got={study.get('study_id')}"
        )
    effects = study.get("effects")
    if not isinstance(effects, list) or not effects:
        raise ValueError(f"Stage 2 returned no effects for {expected_id}")
    valid_refs = set(context.block_ids)
    for index, effect in enumerate(effects):
        if not isinstance(effect, dict):
            raise ValueError(f"effects[{index}] must be an object")
        refs = effect.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"effects[{index}] has no evidence_refs")
        invalid = [str(ref) for ref in refs if str(ref) not in valid_refs]
        if invalid:
            raise ValueError(f"effects[{index}] has invalid evidence refs: {invalid}")


def _context_summary(context: EvidenceContext) -> Dict[str, Any]:
    return {
        "mode": context.mode,
        "block_ids": list(context.block_ids),
        "pages": list(context.pages),
        "facets": {key: list(value) for key, value in context.facets.items()},
        "source_chars": context.source_chars,
        "context_chars": context.context_chars,
    }
