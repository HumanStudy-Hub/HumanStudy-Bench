"""Independent block-window verifier for Stage 1 study inventories."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

from generation_pipeline.pdf.evidence import PdfEvidenceIndex
from generation_pipeline.pdf.models import DocumentBlock, ParsedPdfDocument
from generation_pipeline.pdf.parser import parse_pdf_document
from generation_pipeline.stage1_compiler import (
    DEFAULT_WORKERS,
    DiscoveryWindow,
    SIMULATION_BARRIER_KINDS,
    build_discovery_windows,
    canonical_unit_number,
    cached_json_call,
    stage1_policy_text,
)


STAGE1_VERIFIER_VERSION = "stage1-boundary-and-study-verifier-v25"
STAGE1_VERIFIER_PROMPT_VERSION = "stage1-boundary-verifier-prompt-v24"
STAGE1_STUDY_VERIFIER_PROMPT_VERSION = "stage1-study-verifier-prompt-v7"
DEFAULT_MAX_TOKENS = 8000
DEFAULT_TIMEOUT = 300.0
STUDY_AUDIT_CONTEXT_MAX_CHARS = 48000
_BOUNDARY_REPORT_KEYS = (
    "missing_studies",
    "split_merge_corrections",
    "comparison_group_corrections",
)
_VALID_FIELD_CORRECTIONS = {
    "design_type",
    "conditions_or_factors",
    "material_variants",
    "input",
    "participant_task",
    "participants",
    "output",
    "unit_provenance",
    "empirical_support",
    "simulation_barriers",
    "exclusion_reasons",
}


def _compact_stage1(stage1_json: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    experiments: List[Dict[str, Any]] = []
    for exp in stage1_json.get("experiments", []) or []:
        if not isinstance(exp, dict):
            continue
        experiments.append(
            {
                "experiment_id": exp.get("experiment_id"),
                "study_id": exp.get("study_id"),
                "experiment_name": exp.get("experiment_name"),
                "study_name": exp.get("study_name"),
                "candidate_aliases": exp.get("candidate_aliases") or [],
                "candidate_components": [
                    {
                        "study_id": component.get("study_id"),
                        "reported_label": component.get("reported_label"),
                        "study_name": component.get("study_name"),
                        "evidence_block_ids": component.get("evidence_block_ids"),
                    }
                    for component in exp.get("candidate_components", []) or []
                    if isinstance(component, dict)
                ],
                "source_task_family_relation_ids": (
                    exp.get("source_task_family_relation_ids") or []
                ),
                "material_variants": [
                    {
                        "variant_id": variant.get("variant_id"),
                        "label": variant.get("label"),
                        "role": variant.get("role"),
                        "evidence_refs": variant.get("evidence_refs"),
                    }
                    for variant in exp.get("material_variants", []) or []
                    if isinstance(variant, dict)
                ],
                "replicable": exp.get("replicable"),
                "unit_provenance": exp.get("unit_provenance"),
                "is_distinct_empirical_unit": exp.get("is_distinct_empirical_unit"),
                "evidence_refs": exp.get("evidence_refs"),
                "directly_evidenced_in_window": exp.get(
                    "directly_evidenced_in_window",
                    True,
                ),
            }
        )
    comparison_groups = [
        {
            "comparison_group_id": group.get("comparison_group_id"),
            "member_study_ids": group.get("member_study_ids"),
            "relationship_kind": group.get("relationship_kind"),
            "comparison_target": group.get("comparison_target"),
            "evidence_refs": group.get("evidence_refs"),
            "directly_evidenced_in_window": group.get(
                "directly_evidenced_in_window",
                True,
            ),
        }
        for group in stage1_json.get("comparison_groups", []) or []
        if isinstance(group, dict)
    ]
    return {
        "empirical_units": experiments,
        "comparison_groups": comparison_groups,
    }


def build_verifier_prompt(
    stage1_json: Dict[str, Any],
    pdf_text: str,
    *,
    window_id: str = "provided_window",
    valid_block_ids: Optional[Sequence[str]] = None,
) -> str:
    """Build one bounded verifier prompt for unit boundaries and relations."""
    return f"""You are independently auditing empirical-unit boundaries in a
social-science paper for HumanStudy-Bench.

Inspect only this bounded paper window. Find top-level empirical units that are
missing, entries that should be split or merged, and source-explicit
relationships between distinct units. Do not audit design, sample, task,
materials, outcomes, execution barriers, or eligibility here. Those fields are
checked separately from a complete per-study evidence bundle, because absence
from one paper window is not evidence that a field is unsupported.

Boundary rules:
- Audit environment-owning EMPIRICAL UNITS, not each participant interaction.
  A source-labeled Study, Experiment, Survey, Pilot, or Validation is one parent
  unit. Its stimuli, choices, percentage estimates, scales, trait judgments,
  dependent measures, and phases remain components of that unit even when they
  use different response formats or target constructs.
- For an unlabeled paper, separate genuinely different task families with
  different participant procedures or target phenomena. A questionnaire-wide
  sample/procedure paragraph is shared context, not a missing simulation unit.
- Repeated/parameterized items using the same task template remain one unit even
  if separate groups receive different versions. Different group Ns alone are
  not independent task units.
- Stories, vignettes, signs, arms, forms, orders, items, response options,
  table rows, subgroups, dependent variables, manipulation checks, analyses,
  and effects inside one parent unit are not additional studies.
- Prior studies merely cited by the paper are not current-paper units.
- Distinct top-level units remain separate when a paper compares them; record
  that link as a comparison-group correction instead of merging them.
- Every source-explicit current-paper unit must remain in the inventory even if
  it is ineligible for simulation.
- An unlabeled narrative example is not a missing unit unless this window shows
  a participant-facing task family and an exact numeric response statistic.
  Stimulus probabilities and amounts are not response statistics.
- An identity with directly_evidenced_in_window=false is global context. Never
  report it missing merely because its evidence is outside this window.
- A comparison group with directly_evidenced_in_window=false is also identity
  context; local absence cannot make it spurious.
- Candidate split/merge and shared-context decisions were already adjudicated
  once from the complete global candidate ledger. This local window must not
  override them. Always return split_merge_corrections=[]. If a distinct task
  family is absent, report it under missing_studies instead.
- candidate_components list the source-labeled tasks already represented by a
  merged inventory entry. A component must never be reported as missing merely
  because it is not a separate top-level entry. source_task_family_relation_ids
  mean that source-explicit relations were already used globally to compile
  those components into one coherent simulation task family.
- Cross-unit relations were assembled from source-explicit discovery evidence
  across all windows. A local window must not rewrite them. Always return
  comparison_group_corrections=[].
- For comparison groups, missing_group means no existing group has that member
  set. wrong_members and spurious_group must identify an existing
  current_group_id. Local absence of relationship evidence is not a direct
  contradiction. A generic phrase such as "these studies" or a discussion
  summary is narrative_synthesis and does not create a comparison group.

Window: {window_id}
Valid block IDs: {json.dumps(list(valid_block_ids or []), ensure_ascii=False)}

PAPER WINDOW:
{pdf_text}

CURRENT GLOBAL INVENTORY:
{json.dumps(_compact_stage1(stage1_json), ensure_ascii=False, indent=2)}

Return only this JSON object. Report issues only. Every evidence_block_id must
come from the valid list.
{{
  "confidence": 0.0,
  "missing_studies": [
    {{
      "study": "paper label or concise description",
      "proposed_study_id": "stable_snake_case_id",
      "source_boundary": "source_labeled|unlabeled_new_collection",
      "current_paper_collection": true,
      "has_exact_response_statistic": true,
      "response_statistic": "exact response percentage/statistic or null",
      "reason": "why this is a distinct empirical unit",
      "evidence": "short direct evidence",
      "evidence_block_ids": ["valid block id"]
    }}
  ],
  "split_merge_corrections": [],
  "comparison_group_corrections": [],
  "notes": "short window-level note"
}}"""


def build_study_verifier_prompt(
    experiment: Dict[str, Any],
    evidence_text: str,
    *,
    valid_block_ids: Sequence[str],
    audited_evidence_refs: Sequence[str],
    numeric_challenges: Sequence[Dict[str, Any]],
) -> str:
    """Build a field audit over all evidence cited by one empirical unit."""
    study_id = str(experiment.get("study_id") or experiment.get("experiment_id") or "study")
    challenger_refs = [
        block_id for block_id in valid_block_ids if block_id not in set(audited_evidence_refs)
    ]
    return f"""You are independently auditing one empirical unit for Stage 1 of
HumanStudy-Bench. Unlike the boundary audit, this context is assembled around
all evidence blocks currently cited by this study, plus nearby and retrieved
method/result evidence. No cited block may be ignored merely because it appeared
in another part of the paper.

{stage1_policy_text()}

Audit the current values for design_type, conditions_or_factors,
material_variants, input, participant_task, participants, output,
unit_provenance, empirical_support, simulation_barriers, exclusion_reasons, and
the replicable eligibility label.

Rules:
- Report a correction only for a direct contradiction, source content omitted
  from the current value, or a policy misclassification. Do not rewrite a field
  just for style or add unsupported detail.
- Audit the empirical unit as one coherent record. Do not require one field to
  repeat information already present in another appropriate field. In
  particular, conditions_or_factors names assigned levels, compared options,
  or measured factors; input carries the full scenario and procedure, and
  output carries the recorded response and result summary. Derived equations
  need not be copied into conditions_or_factors when the actual options are
  already correct.
- Inspect every block in COMPLETE BOUNDED STUDY EVIDENCE, including retrieved
  blocks not already cited by the current field. Existing citations being
  present does not prove the current value is correct.
- Cross-check percentages, probabilities, sample totals, arithmetic identities,
  and stated equivalences across the evidence bundle. When a visually/OCR-
  ambiguous token conflicts with an explicit equation or equivalence elsewhere,
  report the contradiction instead of repeating the malformed token as an
  exact quote.
- NUMERIC CONSISTENCY CHALLENGES are deterministic retrieval hints, not facts.
  Adjudicate every challenge_id. If a challenge is a real extraction error,
  return `correction_required` and add a complete field correction for every
  current_field listed by that challenge. Otherwise explain why it is
  consistent, unrelated, or insufficient. Never silently ignore a challenge.
- `expected_value` must be the complete corrected JSON value and must have the
  same JSON type as the current field. For a nested error, return the complete
  replacement array/object, not prose describing an edit.
- input, participant_task, participants, and output are text fields. Return a
  single string or null for them, even when the source reports several groups.
- design_type must be exactly one of between-subjects, within-subjects, mixed,
  correlational, field, archival, other, or null. Task labels such as "binary
  choice" belong in participant_task, not design_type.
- Material variants are only genuinely different participant-facing versions
  assigned across participants or occasions. Joint response options, result
  groups, table rows, and a synthetic single form are not variants.
  A non-empty replacement must contain at least two complete variant objects;
  one vignette or one choice set alone is not a variant set.
- Missing exact material wording is a Stage 3 readiness gap, not an eligibility
  failure.
- `empirical_support.quantitative_result=yes` requires an exact numeric response
  result reported for this unit. Numeric values that only define the stimulus,
  sample size, response scale, option amounts, or experimental parameters do
  not satisfy this requirement. Qualitative significance or majority language
  without a response count, percentage, estimate, coefficient, or test
  statistic remains `no` or `unclear`.
- `exclusion_reasons` records why this empirical unit cannot be retained as a
  simulatable Stage 1 target. Participant filtering, missing-data handling, and
  observations omitted only from the paper's statistical analysis belong in
  participants, input, or output when relevant; they are not Stage 1 exclusion
  reasons. Never add a non-empty exclusion_reasons value to a YES study unless
  the same evidence also requires an eligibility downgrade.
- A real observed action, enforced consequential commitment, live contingent
  interaction, longitudinal exposure, dynamic environment, or specialized
  apparatus is a simulation barrier when it affects the original target. A
  static or explicitly hypothetical choice/rating is not. A consequential
  choice does not become hypothetical merely because the paper records it
  before physical follow-through.
- Use eligibility corrections only when the evidence and the shared policy
  directly imply a different YES/NO/UNCERTAIN label.

Study ID: {study_id}
All current cited evidence refs included in this audit:
{json.dumps(list(audited_evidence_refs), ensure_ascii=False)}
Retrieved challenger refs not currently cited by the study fields:
{json.dumps(challenger_refs, ensure_ascii=False)}
You must inspect these challenger blocks for contradictions and omitted support;
do not limit the audit to the current citation list.
Valid evidence block IDs:
{json.dumps(list(valid_block_ids), ensure_ascii=False)}
Numeric consistency challenges:
{json.dumps(list(numeric_challenges), ensure_ascii=False, indent=2)}

CURRENT EMPIRICAL UNIT:
{json.dumps(experiment, ensure_ascii=False, indent=2)}

COMPLETE BOUNDED STUDY EVIDENCE:
{evidence_text}

Return only this JSON object. Every correction must cite valid evidence.
{{
  "study_id": {json.dumps(study_id, ensure_ascii=False)},
  "confidence": 0.0,
  "study_field_corrections": [
    {{
      "field": "one field name from the allowed list above, never a pipe-separated list",
      "expected_value": "complete corrected JSON value with the current field's type",
      "correction_basis": "direct_contradiction|source_missing_content|policy_misclassification",
      "reason": "short correction",
      "evidence": "short direct evidence",
      "evidence_block_ids": ["valid block id"]
    }}
  ],
  "eligibility_corrections": [
    {{
      "expected_label": "YES|NO|UNCERTAIN",
      "correction_basis": "direct_contradiction|policy_misclassification",
      "reason": "short correction",
      "evidence": "short direct evidence",
      "evidence_block_ids": ["valid block id"]
    }}
  ],
  "numeric_challenge_results": [
    {{
      "challenge_id": "exact challenge_id from the supplied list",
      "verdict": "correction_required|consistent|unrelated|insufficient",
      "reason": "short adjudication",
      "evidence_block_ids": ["valid block id"]
    }}
  ],
  "notes": "short study-level note"
}}"""


def verify_stage1_inventory(
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
    boundary_baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del retry_delay
    if llm_client is None:
        return _skipped_report("No LLM client provided.")

    if pdf_text is not None:
        provided_block = DocumentBlock(
            block_id="provided_text",
            order=0,
            page_start=1,
            page_end=1,
            block_type="text",
            text=pdf_text,
            section_path=["Provided verifier text"],
        )
        document = ParsedPdfDocument(
            source_file=str(pdf_path),
            source_sha256="provided_text",
            parser="provided_text",
            parser_version="1",
            page_count=1,
            blocks=[provided_block],
            markdown=pdf_text,
            degraded=True,
            warnings=["verifier_received_text_override"],
        )
        windows = [
            DiscoveryWindow(
                window_id="provided_window",
                text=pdf_text,
                block_ids=[provided_block.block_id],
                pages=[1],
                char_count=len(pdf_text),
            )
        ]
    else:
        document = parse_pdf_document(
            Path(pdf_path),
            artifacts_dir=pdf_artifacts_dir,
            force=False,
            prefer_docling=True,
        )
        windows = build_discovery_windows(document)
    if not windows:
        raise ValueError("Stage 1 verifier has no PDF evidence windows")
    evidence_index = PdfEvidenceIndex(document)
    block_map = document.block_map()

    cache_dir = Path(artifacts_dir) if artifacts_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    reports: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    def run_boundary(window: DiscoveryWindow) -> Dict[str, Any]:
        local_stage1 = _inventory_for_window(stage1_json, window)
        prompt = build_verifier_prompt(
            local_stage1,
            window.text,
            window_id=window.window_id,
            valid_block_ids=window.block_ids,
        )
        return cached_json_call(
            llm_client,
            prompt,
            cache_path=(
                cache_dir / "boundaries" / f"{window.window_id}.json"
                if cache_dir
                else None
            ),
            prompt_version=STAGE1_VERIFIER_PROMPT_VERSION,
            timeout=timeout,
            max_tokens=int(max_tokens or DEFAULT_MAX_TOKENS),
            force=force,
            max_attempts=max_attempts,
            validator=lambda value: _validate_window_payload_shape(
                value,
                valid_block_ids=set(window.block_ids),
            ),
        )

    reuse_boundary = (
        isinstance(boundary_baseline, dict)
        and boundary_baseline.get("status") == "ok"
    )
    if reuse_boundary:
        reports = {
            window.window_id: {
                "confidence": _confidence(boundary_baseline.get("confidence")),
                "missing_studies": [],
                "split_merge_corrections": [],
                "comparison_group_corrections": [],
                "notes": "",
            }
            for window in windows
        }
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as pool:
            future_map = {pool.submit(run_boundary, window): window for window in windows}
            for future in as_completed(future_map):
                window = future_map[future]
                try:
                    reports[window.window_id] = _normalize_window_report(
                        future.result(),
                        valid_block_ids=set(window.block_ids),
                        stage1_json=stage1_json,
                    )
                except Exception as exc:
                    errors[f"boundary:{window.window_id}"] = f"{type(exc).__name__}: {exc}"

    experiments = [
        experiment
        for experiment in stage1_json.get("experiments", []) or []
        if isinstance(experiment, dict)
    ]
    study_reports: Dict[str, Dict[str, Any]] = {}
    study_summaries: Dict[str, Dict[str, Any]] = {}

    def run_study(experiment: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        study_id = _study_key(experiment)
        anchor_refs = _study_evidence_refs(experiment, evidence_index.block_ids)
        context = evidence_index.context_for_study(
            experiment,
            stage1_json=stage1_json,
            gaps=["source_evidence"],
            allow_full_document=False,
            anchor_refs=anchor_refs,
            anchor_radius=1,
            use_facet_retrieval=True,
            max_chars=STUDY_AUDIT_CONTEXT_MAX_CHARS,
        )
        missing_refs = [ref for ref in anchor_refs if ref not in set(context.block_ids)]
        if missing_refs:
            raise ValueError(
                f"bounded context omitted {len(missing_refs)} cited evidence block(s): "
                + ", ".join(missing_refs[:8])
            )
        numeric_challenges = _numeric_consistency_challenges(
            experiment,
            [block_map[block_id] for block_id in context.block_ids if block_id in block_map],
        )
        prompt = build_study_verifier_prompt(
            experiment,
            context.text,
            valid_block_ids=context.block_ids,
            audited_evidence_refs=anchor_refs,
            numeric_challenges=numeric_challenges,
        )
        raw = cached_json_call(
            llm_client,
            prompt,
            cache_path=(
                cache_dir / "studies" / f"{_safe_cache_name(study_id)}.json"
                if cache_dir
                else None
            ),
            prompt_version=STAGE1_STUDY_VERIFIER_PROMPT_VERSION,
            timeout=timeout,
            max_tokens=int(max_tokens or DEFAULT_MAX_TOKENS),
            force=force,
            max_attempts=max_attempts,
            validator=lambda value: _validate_study_audit_payload_shape(
                value,
                experiment=experiment,
                valid_block_ids=set(context.block_ids),
            ),
        )
        report = _normalize_study_audit_report(
            raw,
            experiment=experiment,
            valid_block_ids=set(context.block_ids),
            numeric_challenges=numeric_challenges,
        )
        summary = {
            "study_id": study_id,
            "mode": context.mode,
            "pages": list(context.pages),
            "block_ids": list(context.block_ids),
            "audited_evidence_refs": anchor_refs,
            "all_cited_evidence_included": not missing_refs,
            "context_chars": context.context_chars,
            "numeric_challenges": numeric_challenges,
            "numeric_challenge_results": report["numeric_challenge_results"],
        }
        return report, summary

    with ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as pool:
        future_map = {
            pool.submit(run_study, experiment): experiment for experiment in experiments
        }
        for future in as_completed(future_map):
            experiment = future_map[future]
            study_id = _study_key(experiment)
            try:
                report, summary = future.result()
                study_reports[study_id] = report
                study_summaries[study_id] = summary
            except Exception as exc:
                errors[f"study:{study_id}"] = f"{type(exc).__name__}: {exc}"
    if errors:
        detail = "; ".join(f"{key}={value}" for key, value in sorted(errors.items()))
        raise RuntimeError(
            "Stage 1 verification was incomplete; refusing to pass a partial audit: " + detail
        )
    result = _aggregate_window_reports(
        stage1_json,
        windows,
        reports,
        study_reports=study_reports,
        study_summaries=study_summaries,
    )
    if reuse_boundary:
        _reuse_boundary_baseline(result, boundary_baseline)
    return result


def _study_key(experiment: Dict[str, Any]) -> str:
    return str(
        experiment.get("study_id")
        or experiment.get("experiment_id")
        or experiment.get("study_name")
        or "study"
    ).strip()


def _safe_cache_name(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "_", str(value or "study").lower()).strip("_")
    return normalized or "study"


def _study_evidence_refs(
    experiment: Dict[str, Any],
    valid_block_ids: set[str],
) -> List[str]:
    refs: List[str] = []

    def add(values: Any) -> None:
        if not isinstance(values, list):
            return
        for value in values:
            ref = str(value)
            if ref in valid_block_ids and ref not in refs:
                refs.append(ref)

    add(experiment.get("evidence_refs"))
    field_evidence = experiment.get("field_evidence")
    if isinstance(field_evidence, dict):
        for values in field_evidence.values():
            add(values)
    for key in ("material_variants", "simulation_barriers"):
        for item in experiment.get(key, []) or []:
            if isinstance(item, dict):
                add(item.get("evidence_refs"))
    return refs


_PERCENT_TOKEN_RE = re.compile(
    r"(?<![\w.])(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>%|percent(?:age)?(?:\s+points?)?)(?!\w)",
    re.IGNORECASE,
)
_EQUATION_RE = re.compile(
    r"(?P<equation>"
    r"(?:\d*\.\d+|\d+(?:\.\d+)?\s*%)"
    r"(?:\s*(?:x|\u00d7|\*)\s*(?:\d*\.\d+|\d+(?:\.\d+)?\s*%))+"
    r"\s*=\s*(?:\d*\.\d+|\d+(?:\.\d+)?\s*%)"
    r")",
    re.IGNORECASE,
)
_EQUATION_NUMBER_RE = re.compile(r"\d*\.\d+|\d+(?:\.\d+)?\s*%")
_NUMERIC_AUDIT_FIELDS = (
    "conditions_or_factors",
    "material_variants",
    "input",
    "participant_task",
    "output",
)


def _numeric_consistency_challenges(
    experiment: Dict[str, Any],
    blocks: Sequence[DocumentBlock],
) -> List[Dict[str, Any]]:
    """Find scale-like OCR conflicts between extracted percentages and equations.

    These are review obligations, not automatic corrections. Equations often
    survive OCR better than nearby table glyphs, so a 10x/100x conflict is a
    useful generic signal while still requiring evidence-grounded adjudication.
    """
    current_occurrences: List[Dict[str, Any]] = []
    for field in _NUMERIC_AUDIT_FIELDS:
        _collect_percentage_occurrences(
            experiment.get(field),
            field=field,
            path=field,
            output=current_occurrences,
        )

    equation_occurrences: List[Dict[str, Any]] = []
    for block in blocks:
        for match in _EQUATION_RE.finditer(block.text or ""):
            equation = match.group("equation")
            for token in _EQUATION_NUMBER_RE.findall(equation):
                normalized = _probability_percent(token)
                if normalized is None:
                    continue
                equation_occurrences.append(
                    {
                        "block_id": block.block_id,
                        "equation": equation,
                        "token": token.strip(),
                        "percent_value": normalized,
                    }
                )

    grouped: Dict[tuple[float, float, str, str], Dict[str, Any]] = {}
    for current in current_occurrences:
        current_value = float(current["percent_value"])
        if current_value <= 0:
            continue
        for equation_value in equation_occurrences:
            candidate_value = float(equation_value["percent_value"])
            if candidate_value <= 0 or _numbers_equal(current_value, candidate_value):
                continue
            ratio = max(current_value, candidate_value) / min(current_value, candidate_value)
            if not (_numbers_equal(ratio, 10.0) or _numbers_equal(ratio, 100.0)):
                continue
            if _numeric_mantissa(current_value) != _numeric_mantissa(candidate_value):
                continue
            key = (
                current_value,
                candidate_value,
                str(equation_value["block_id"]),
                str(equation_value["equation"]),
            )
            item = grouped.setdefault(
                key,
                {
                    "current_token": current["token"],
                    "current_percent_value": current_value,
                    "current_fields": [],
                    "current_paths": [],
                    "equation_token": equation_value["token"],
                    "equation_percent_value": candidate_value,
                    "equation": equation_value["equation"],
                    "equation_evidence_block_id": equation_value["block_id"],
                    "signal": "power_of_ten_probability_conflict",
                },
            )
            if current["field"] not in item["current_fields"]:
                item["current_fields"].append(current["field"])
            if current["path"] not in item["current_paths"]:
                item["current_paths"].append(current["path"])

    challenges: List[Dict[str, Any]] = []
    for position, item in enumerate(grouped.values(), start=1):
        challenge = dict(item)
        challenge["challenge_id"] = f"numeric_scale_{position:03d}"
        challenges.append(challenge)
    return challenges


def _collect_percentage_occurrences(
    value: Any,
    *,
    field: str,
    path: str,
    output: List[Dict[str, Any]],
) -> None:
    if isinstance(value, str):
        for match in _PERCENT_TOKEN_RE.finditer(value):
            output.append(
                {
                    "field": field,
                    "path": path,
                    "token": match.group(0).strip(),
                    "percent_value": float(match.group("number")),
                }
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_percentage_occurrences(
                item,
                field=field,
                path=f"{path}[{index}]",
                output=output,
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_percentage_occurrences(
                item,
                field=field,
                path=f"{path}.{key}",
                output=output,
            )


def _probability_percent(token: str) -> Optional[float]:
    compact = token.strip().replace(" ", "")
    try:
        if compact.endswith("%"):
            return float(compact[:-1])
        value = float(compact)
    except ValueError:
        return None
    return value * 100.0 if 0 <= value <= 1 else value


def _numeric_mantissa(value: float) -> str:
    rendered = f"{value:.12g}".replace(".", "").lstrip("0").rstrip("0")
    return rendered or "0"


def _numbers_equal(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-8 * max(1.0, abs(left), abs(right))


def _inventory_for_window(
    stage1_json: Dict[str, Any],
    window: DiscoveryWindow,
) -> Dict[str, Any]:
    block_ids = set(window.block_ids)
    contexts = {}
    evidence = stage1_json.get("stage1_evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("study_contexts"), dict):
        contexts = evidence["study_contexts"]
    global_experiments = [
        experiment
        for experiment in stage1_json.get("experiments", []) or []
        if isinstance(experiment, dict)
    ]
    direct_study_ids: set[str] = set()
    refs_by_study_id: Dict[str, set[str]] = {}
    for experiment in global_experiments:
        if not isinstance(experiment, dict):
            continue
        study_id = str(experiment.get("study_id") or "")
        refs = set(str(ref) for ref in experiment.get("evidence_refs") or [])
        for variant in experiment.get("material_variants") or []:
            if isinstance(variant, dict):
                refs.update(str(ref) for ref in variant.get("evidence_refs") or [])
        for barrier in experiment.get("simulation_barriers") or []:
            if isinstance(barrier, dict):
                refs.update(str(ref) for ref in barrier.get("evidence_refs") or [])
        context = contexts.get(experiment.get("study_id")) if isinstance(contexts, dict) else {}
        if not refs and isinstance(context, dict):
            refs.update(str(ref) for ref in context.get("block_ids") or [])
        refs_by_study_id[study_id] = refs
        if refs & block_ids:
            direct_study_ids.add(study_id)
    comparison_groups: List[Dict[str, Any]] = []
    for group in stage1_json.get("comparison_groups", []) or []:
        if not isinstance(group, dict):
            continue
        refs = {str(ref) for ref in group.get("evidence_refs") or []}
        members = {str(value) for value in group.get("member_study_ids") or []}
        local_group = dict(group)
        local_group["evidence_refs"] = sorted(refs & block_ids)
        local_group["directly_evidenced_in_window"] = bool(
            refs & block_ids or members & direct_study_ids
        )
        comparison_groups.append(local_group)
    experiments: List[Dict[str, Any]] = []
    for experiment in global_experiments:
        study_id = str(experiment.get("study_id") or "")
        is_direct = study_id in direct_study_ids
        if is_direct:
            local_experiment = dict(experiment)
        else:
            local_experiment = {
                "experiment_id": experiment.get("experiment_id"),
                "study_id": experiment.get("study_id"),
                "study_name": experiment.get("study_name"),
                "replicable": experiment.get("replicable"),
            }
        local_experiment["evidence_refs"] = sorted(
            refs_by_study_id.get(study_id, set()) & block_ids
        )
        local_variants: List[Dict[str, Any]] = []
        for variant in experiment.get("material_variants") or []:
            if not isinstance(variant, dict):
                continue
            local_refs = sorted(
                {str(ref) for ref in variant.get("evidence_refs") or []} & block_ids
            )
            if not local_refs:
                continue
            local_variant = dict(variant)
            local_variant["evidence_refs"] = local_refs
            local_variants.append(local_variant)
        local_experiment["material_variants"] = local_variants
        local_barriers: List[Dict[str, Any]] = []
        for barrier in experiment.get("simulation_barriers") or []:
            if not isinstance(barrier, dict):
                continue
            local_refs = sorted(
                {str(ref) for ref in barrier.get("evidence_refs") or []} & block_ids
            )
            if not local_refs:
                continue
            local_barrier = dict(barrier)
            local_barrier["evidence_refs"] = local_refs
            local_barriers.append(local_barrier)
        local_experiment["simulation_barriers"] = local_barriers
        local_experiment["directly_evidenced_in_window"] = is_direct
        experiments.append(local_experiment)
    return {
        "experiments": experiments,
        "comparison_groups": comparison_groups,
    }


def _normalize_window_report(
    report: Dict[str, Any],
    *,
    valid_block_ids: set[str],
    stage1_json: Dict[str, Any],
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "confidence": _confidence(report.get("confidence")),
        "missing_studies": [],
        "split_merge_corrections": [],
        "comparison_group_corrections": [],
        "validation_diagnostics": [],
        "dropped_noop_corrections": 0,
        "notes": str(report.get("notes") or "").strip(),
    }
    for key in _BOUNDARY_REPORT_KEYS:
        values = report.get(key) if isinstance(report.get(key), list) else []
        for value in values:
            if not isinstance(value, dict):
                continue
            item = dict(value)
            if key == "split_merge_corrections":
                item = _normalize_boundary_correction(item, stage1_json=stage1_json)
            elif key == "comparison_group_corrections" and str(
                item.get("evidence_kind") or ""
            ).strip() in {"sequence", "shared_sample"}:
                item["evidence_kind"] = "shared_sample_or_sequence"
            item["evidence_block_ids"] = [
                str(ref)
                for ref in item.get("evidence_block_ids") or []
                if str(ref) in valid_block_ids
            ]
            single_report = {
                "missing_studies": [],
                "split_merge_corrections": [],
                "comparison_group_corrections": [],
            }
            single_report[key] = [item]
            try:
                _validate_window_payload(
                    single_report,
                    valid_block_ids=valid_block_ids,
                    stage1_json=stage1_json,
                )
            except ValueError as exc:
                if _is_noop_boundary_validation_error(exc):
                    normalized["dropped_noop_corrections"] += 1
                else:
                    normalized["validation_diagnostics"].append(
                        {
                            "scope": key,
                            "position": len(normalized[key]),
                            "error": str(exc),
                            "raw_correction": item,
                        }
                    )
                continue
            normalized[key].append(item)
    return normalized


def _validate_window_payload_shape(
    report: Dict[str, Any],
    *,
    valid_block_ids: set[str],
) -> None:
    """Validate response/evidence shape without rejecting one semantic no-op."""
    for forbidden_key in ("study_field_corrections", "eligibility_corrections"):
        if report.get(forbidden_key) not in (None, []):
            raise ValueError(
                f"boundary verifier must not return {forbidden_key}; use the per-study audit"
            )
    for key in _BOUNDARY_REPORT_KEYS:
        values = report.get(key)
        if not isinstance(values, list):
            raise ValueError(f"{key} must be an array")
        for position, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError(f"{key}[{position}] must be an object")
            refs = value.get("evidence_block_ids")
            if not isinstance(refs, list) or not refs:
                raise ValueError(f"{key}[{position}] has no evidence_block_ids")
            invalid = [str(ref) for ref in refs if str(ref) not in valid_block_ids]
            if invalid:
                raise ValueError(f"{key}[{position}] has invalid evidence refs: {invalid}")


def _is_noop_boundary_validation_error(exc: ValueError) -> bool:
    message = str(exc)
    return any(
        phrase in message
        for phrase in (
            "duplicates an existing group",
            "requires different expected members",
        )
    )


def _normalize_boundary_correction(
    item: Dict[str, Any],
    *,
    stage1_json: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = dict(item)
    if str(normalized.get("verdict") or "").strip() != "merge_issue":
        return normalized
    current_ids = [
        str(value).strip()
        for value in normalized.get("current_study_ids") or []
        if str(value).strip()
    ]
    proposed_ids = [
        str(value).strip()
        for value in normalized.get("proposed_study_ids") or []
        if str(value).strip()
    ]
    known_ids = {
        str(experiment.get("study_id") or "").strip()
        for experiment in stage1_json.get("experiments", []) or []
        if isinstance(experiment, dict) and str(experiment.get("study_id") or "").strip()
    }
    if (
        len(current_ids) == 1
        and len(proposed_ids) == 1
        and proposed_ids[0] in known_ids
        and proposed_ids[0] != current_ids[0]
    ):
        normalized["current_study_ids"] = [current_ids[0], proposed_ids[0]]
    return normalized


def _validate_window_payload(
    report: Dict[str, Any],
    *,
    valid_block_ids: set[str],
    stage1_json: Dict[str, Any],
) -> None:
    known_study_ids = {
        str(experiment.get("study_id") or "").strip()
        for experiment in stage1_json.get("experiments", []) or []
        if isinstance(experiment, dict) and str(experiment.get("study_id") or "").strip()
    }
    groups_by_id = {
        str(group.get("comparison_group_id") or "").strip(): group
        for group in stage1_json.get("comparison_groups", []) or []
        if isinstance(group, dict) and str(group.get("comparison_group_id") or "").strip()
    }
    current_group_sets = {
        frozenset(str(member).strip() for member in group.get("member_study_ids") or [])
        for group in groups_by_id.values()
    }
    for forbidden_key in ("study_field_corrections", "eligibility_corrections"):
        if report.get(forbidden_key) not in (None, []):
            raise ValueError(
                f"boundary verifier must not return {forbidden_key}; use the per-study audit"
            )
    for key in _BOUNDARY_REPORT_KEYS:
        values = report.get(key)
        if not isinstance(values, list):
            raise ValueError(f"{key} must be an array")
        for position, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError(f"{key}[{position}] must be an object")
            refs = value.get("evidence_block_ids")
            if not isinstance(refs, list) or not refs:
                raise ValueError(f"{key}[{position}] has no evidence_block_ids")
            invalid = [str(ref) for ref in refs if str(ref) not in valid_block_ids]
            if invalid:
                raise ValueError(f"{key}[{position}] has invalid evidence refs: {invalid}")
            if key == "comparison_group_corrections":
                members = value.get("member_study_ids")
                if not isinstance(members, list) or len(
                    {str(member).strip() for member in members if str(member).strip()}
                ) < 2:
                    raise ValueError(
                        f"comparison_group_corrections[{position}] requires two distinct members"
                    )
                member_set = frozenset(str(member).strip() for member in members)
                verdict = str(value.get("verdict") or "").strip()
                current_group_id = str(value.get("current_group_id") or "").strip()
                evidence_kind = str(value.get("evidence_kind") or "").strip()
                if evidence_kind not in {
                    "joint_statistical_analysis",
                    "explicit_cross_unit_contrast",
                    "replication_or_extension",
                    "shared_sample_or_sequence",
                    "narrative_synthesis",
                }:
                    raise ValueError(
                        f"comparison_group_corrections[{position}] has invalid evidence_kind"
                    )
                if verdict == "missing_group":
                    if current_group_id:
                        raise ValueError(
                            f"comparison_group_corrections[{position}] missing_group cannot name current_group_id"
                        )
                    if member_set in current_group_sets:
                        raise ValueError(
                            f"comparison_group_corrections[{position}] duplicates an existing group"
                        )
                elif verdict in {"wrong_members", "spurious_group"}:
                    if current_group_id not in groups_by_id:
                        raise ValueError(
                            f"comparison_group_corrections[{position}] requires a valid current_group_id"
                        )
                    current_members = frozenset(
                        str(member).strip()
                        for member in groups_by_id[current_group_id].get("member_study_ids") or []
                    )
                    if member_set != current_members:
                        raise ValueError(
                            f"comparison_group_corrections[{position}] member_study_ids must identify the current group"
                        )
                    expected_members = {
                        str(member).strip()
                        for member in value.get("expected_member_study_ids") or []
                        if str(member).strip()
                    }
                    if verdict == "wrong_members" and (
                        len(expected_members) < 2 or expected_members == set(current_members)
                    ):
                        raise ValueError(
                            f"comparison_group_corrections[{position}] requires different expected members"
                        )
                else:
                    raise ValueError(
                        f"comparison_group_corrections[{position}] has invalid verdict"
                    )
            if key == "split_merge_corrections":
                verdict = str(value.get("verdict") or "").strip()
                current_ids = [
                    str(study_id).strip()
                    for study_id in value.get("current_study_ids") or []
                    if str(study_id).strip()
                ]
                proposed_ids = [
                    str(study_id).strip()
                    for study_id in value.get("proposed_study_ids") or []
                    if str(study_id).strip()
                ]
                proposed_labels = [
                    str(label).strip()
                    for label in value.get("proposed_source_labels") or []
                    if str(label).strip()
                ]
                boundary_basis = str(value.get("boundary_basis") or "").strip()
                if boundary_basis not in {
                    "distinct_source_labels",
                    "independent_recruitment_or_session",
                    "same_source_unit",
                }:
                    raise ValueError(
                        f"split_merge_corrections[{position}] has invalid boundary_basis"
                    )
                if len(proposed_labels) != len(proposed_ids):
                    raise ValueError(
                        f"split_merge_corrections[{position}] requires one source label per proposed id"
                    )
                if not set(current_ids) <= known_study_ids:
                    raise ValueError(
                        f"split_merge_corrections[{position}] references unknown current studies"
                    )
                if verdict == "split_issue":
                    if (
                        len(set(current_ids)) != 1
                        or len(set(proposed_ids)) < 2
                        or boundary_basis == "same_source_unit"
                    ):
                        raise ValueError(
                            f"split_merge_corrections[{position}] has invalid split operands"
                        )
                elif verdict == "merge_issue":
                    if (
                        len(set(current_ids)) < 2
                        or len(set(proposed_ids)) != 1
                        or boundary_basis != "same_source_unit"
                    ):
                        raise ValueError(
                            f"split_merge_corrections[{position}] has invalid merge operands"
                        )
                else:
                    raise ValueError(f"split_merge_corrections[{position}] has invalid verdict")
            if key == "missing_studies":
                if not str(value.get("proposed_study_id") or "").strip():
                    raise ValueError(
                        f"missing_studies[{position}].proposed_study_id is required"
                    )
                source_boundary = str(value.get("source_boundary") or "").strip()
                if source_boundary not in {
                    "source_labeled",
                    "unlabeled_new_collection",
                }:
                    raise ValueError(
                        f"missing_studies[{position}] has invalid source_boundary"
                    )
                if not isinstance(value.get("current_paper_collection"), bool):
                    raise ValueError(
                        f"missing_studies[{position}].current_paper_collection must be boolean"
                    )
                if not isinstance(value.get("has_exact_response_statistic"), bool):
                    raise ValueError(
                        f"missing_studies[{position}].has_exact_response_statistic must be boolean"
                    )


def _normalize_study_audit_report(
    report: Dict[str, Any],
    *,
    experiment: Dict[str, Any],
    valid_block_ids: set[str],
    numeric_challenges: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    study_id = _study_key(experiment)
    normalized: Dict[str, Any] = {
        "study_id": study_id,
        "confidence": _confidence(report.get("confidence")),
        "study_field_corrections": [],
        "eligibility_corrections": [],
        "numeric_challenge_results": [],
        "validation_diagnostics": [],
        "notes": str(report.get("notes") or "").strip(),
    }
    for key in ("study_field_corrections", "eligibility_corrections"):
        for value in report.get(key, []) or []:
            if not isinstance(value, dict):
                continue
            if key == "study_field_corrections" and str(
                value.get("field") or ""
            ).strip() == "replicable":
                routed = _route_replicable_field_correction(value)
                if routed is None:
                    normalized["validation_diagnostics"].append(
                        {
                            "scope": key,
                            "position": len(normalized[key]),
                            "error": "replicable correction could not be routed to eligibility",
                            "raw_correction": value,
                        }
                    )
                    continue
                routed["study"] = study_id
                try:
                    _validate_single_study_correction(
                        "eligibility_corrections",
                        routed,
                        experiment=experiment,
                        valid_block_ids=valid_block_ids,
                    )
                except ValueError as exc:
                    normalized["validation_diagnostics"].append(
                        {
                            "scope": key,
                            "position": len(normalized[key]),
                            "error": str(exc),
                            "raw_correction": value,
                        }
                    )
                    continue
                normalized["eligibility_corrections"].append(routed)
                continue
            items = (
                _expand_field_correction(value)
                if key == "study_field_corrections"
                else [dict(value)]
            )
            if key == "study_field_corrections" and not items:
                normalized["validation_diagnostics"].append(
                    {
                        "scope": key,
                        "position": len(normalized[key]),
                        "error": "field correction has no usable typed replacement",
                        "raw_correction": value,
                    }
                )
                continue
            for item in items:
                item["study"] = study_id
                item["evidence_block_ids"] = [
                    str(ref)
                    for ref in item.get("evidence_block_ids") or []
                    if str(ref) in valid_block_ids
                ]
                try:
                    _validate_single_study_correction(
                        key,
                        item,
                        experiment=experiment,
                        valid_block_ids=valid_block_ids,
                    )
                except ValueError as exc:
                    normalized["validation_diagnostics"].append(
                        {
                            "scope": key,
                            "position": len(normalized[key]),
                            "error": str(exc),
                            "raw_correction": value,
                        }
                    )
                    continue
                normalized[key].append(item)
    for value in report.get("numeric_challenge_results", []) or []:
        if not isinstance(value, dict):
            continue
        item = dict(value)
        item["evidence_block_ids"] = [
            str(ref)
            for ref in item.get("evidence_block_ids") or []
            if str(ref) in valid_block_ids
        ]
        try:
            _validate_numeric_challenge_result(
                item,
                position=len(normalized["numeric_challenge_results"]),
                valid_block_ids=valid_block_ids,
                numeric_challenges=numeric_challenges,
                existing=normalized["numeric_challenge_results"],
            )
        except ValueError as exc:
            normalized["validation_diagnostics"].append(
                {
                    "scope": "numeric_challenge_results",
                    "position": len(normalized["numeric_challenge_results"]),
                    "error": str(exc),
                    "raw_correction": value,
                }
            )
            continue
        normalized["numeric_challenge_results"].append(item)
    try:
        _validate_study_audit_payload(
            normalized,
            experiment=experiment,
            valid_block_ids=valid_block_ids,
            numeric_challenges=numeric_challenges,
        )
    except ValueError as exc:
        normalized["validation_diagnostics"].append(
            {
                "scope": "study_audit",
                "error": str(exc),
            }
        )
    return normalized


def _route_replicable_field_correction(
    value: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    expected_label = str(value.get("expected_value") or "").strip().upper()
    if expected_label not in {"YES", "NO", "UNCERTAIN"}:
        return None
    routed = dict(value)
    routed.pop("field", None)
    routed.pop("expected_value", None)
    routed["expected_label"] = expected_label
    return routed


def _validate_study_audit_payload_shape(
    report: Dict[str, Any],
    *,
    experiment: Dict[str, Any],
    valid_block_ids: set[str],
) -> None:
    study_id = _study_key(experiment)
    if str(report.get("study_id") or "").strip() != study_id:
        raise ValueError(
            f"study audit returned {report.get('study_id')!r}; expected {study_id!r}"
        )
    for key in ("study_field_corrections", "eligibility_corrections"):
        values = report.get(key)
        if not isinstance(values, list):
            raise ValueError(f"{key} must be an array")
        for position, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError(f"{key}[{position}] must be an object")
            _validate_correction_refs(
                value,
                path=f"{key}[{position}]",
                valid_block_ids=valid_block_ids,
            )
    values = report.get("numeric_challenge_results", [])
    if not isinstance(values, list):
        raise ValueError("numeric_challenge_results must be an array")
    for position, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"numeric_challenge_results[{position}] must be an object")
        _validate_correction_refs(
            value,
            path=f"numeric_challenge_results[{position}]",
            valid_block_ids=valid_block_ids,
        )


def _validate_correction_refs(
    value: Dict[str, Any],
    *,
    path: str,
    valid_block_ids: set[str],
) -> None:
    refs = value.get("evidence_block_ids")
    if not isinstance(refs, list) or not refs:
        raise ValueError(f"{path} has no evidence_block_ids")
    invalid = [str(ref) for ref in refs if str(ref) not in valid_block_ids]
    if invalid:
        raise ValueError(f"{path} has invalid evidence refs: {invalid}")


def _validate_single_study_correction(
    key: str,
    correction: Dict[str, Any],
    *,
    experiment: Dict[str, Any],
    valid_block_ids: set[str],
) -> None:
    report = {
        "study_id": _study_key(experiment),
        "study_field_corrections": [],
        "eligibility_corrections": [],
        "numeric_challenge_results": [],
    }
    report[key] = [correction]
    _validate_study_audit_payload(
        report,
        experiment=experiment,
        valid_block_ids=valid_block_ids,
        numeric_challenges=[],
    )


def _validate_numeric_challenge_result(
    value: Dict[str, Any],
    *,
    position: int,
    valid_block_ids: set[str],
    numeric_challenges: Sequence[Dict[str, Any]],
    existing: Sequence[Dict[str, Any]],
) -> None:
    expected_ids = {
        str(item.get("challenge_id") or "").strip()
        for item in numeric_challenges
        if str(item.get("challenge_id") or "").strip()
    }
    challenge_id = str(value.get("challenge_id") or "").strip()
    if challenge_id not in expected_ids:
        raise ValueError(f"numeric_challenge_results[{position}] has unknown challenge_id")
    if challenge_id in {
        str(item.get("challenge_id") or "").strip() for item in existing
    }:
        raise ValueError(f"numeric challenge {challenge_id} was adjudicated twice")
    verdict = str(value.get("verdict") or "").strip()
    if verdict not in {"correction_required", "consistent", "unrelated", "insufficient"}:
        raise ValueError(f"numeric_challenge_results[{position}] has invalid verdict")
    _validate_correction_refs(
        value,
        path=f"numeric_challenge_results[{position}]",
        valid_block_ids=valid_block_ids,
    )


def _validate_study_audit_payload(
    report: Dict[str, Any],
    *,
    experiment: Dict[str, Any],
    valid_block_ids: set[str],
    numeric_challenges: Sequence[Dict[str, Any]],
) -> None:
    study_id = _study_key(experiment)
    if str(report.get("study_id") or "").strip() != study_id:
        raise ValueError(
            f"study audit returned {report.get('study_id')!r}; expected {study_id!r}"
        )
    for key in ("study_field_corrections", "eligibility_corrections"):
        values = report.get(key)
        if not isinstance(values, list):
            raise ValueError(f"{key} must be an array")
        for position, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError(f"{key}[{position}] must be an object")
            refs = value.get("evidence_block_ids")
            if not isinstance(refs, list) or not refs:
                raise ValueError(f"{key}[{position}] has no evidence_block_ids")
            invalid = [str(ref) for ref in refs if str(ref) not in valid_block_ids]
            if invalid:
                raise ValueError(f"{key}[{position}] has invalid evidence refs: {invalid}")
            basis = str(value.get("correction_basis") or "").strip()
            allowed_basis = (
                {"direct_contradiction", "source_missing_content", "policy_misclassification"}
                if key == "study_field_corrections"
                else {"direct_contradiction", "policy_misclassification"}
            )
            if basis not in allowed_basis:
                raise ValueError(f"{key}[{position}] has invalid correction_basis")
            if key == "study_field_corrections":
                if "expected_value" not in value:
                    raise ValueError(f"{key}[{position}].expected_value is required")
                if not _expand_field_correction(value):
                    raise ValueError(
                        f"{key}[{position}] has no field with a usable typed replacement"
                    )
            else:
                label = str(value.get("expected_label") or "").strip().upper()
                if label not in {"YES", "NO", "UNCERTAIN"}:
                    raise ValueError(f"{key}[{position}] has invalid expected_label")

    challenge_results = report.get("numeric_challenge_results", [])
    if not isinstance(challenge_results, list):
        raise ValueError("numeric_challenge_results must be an array")
    expected_challenges = {
        str(item.get("challenge_id") or "").strip(): item
        for item in numeric_challenges
        if str(item.get("challenge_id") or "").strip()
    }
    returned_results: Dict[str, Dict[str, Any]] = {}
    for position, value in enumerate(challenge_results):
        if not isinstance(value, dict):
            raise ValueError(f"numeric_challenge_results[{position}] must be an object")
        challenge_id = str(value.get("challenge_id") or "").strip()
        if challenge_id not in expected_challenges:
            raise ValueError(
                f"numeric_challenge_results[{position}] has unknown challenge_id"
            )
        if challenge_id in returned_results:
            raise ValueError(f"numeric challenge {challenge_id} was adjudicated twice")
        verdict = str(value.get("verdict") or "").strip()
        if verdict not in {
            "correction_required",
            "consistent",
            "unrelated",
            "insufficient",
        }:
            raise ValueError(
                f"numeric_challenge_results[{position}] has invalid verdict"
            )
        refs = value.get("evidence_block_ids")
        if not isinstance(refs, list) or not refs:
            raise ValueError(
                f"numeric_challenge_results[{position}] has no evidence_block_ids"
            )
        invalid = [str(ref) for ref in refs if str(ref) not in valid_block_ids]
        if invalid:
            raise ValueError(
                f"numeric_challenge_results[{position}] has invalid evidence refs: {invalid}"
            )
        returned_results[challenge_id] = value
    if set(returned_results) != set(expected_challenges):
        missing = sorted(set(expected_challenges) - set(returned_results))
        raise ValueError(f"numeric challenges were not adjudicated: {missing}")

    corrected_fields = {
        item["field"]
        for correction in report.get("study_field_corrections", []) or []
        if isinstance(correction, dict)
        for item in _expand_field_correction(correction)
    }
    for challenge_id, result in returned_results.items():
        if str(result.get("verdict") or "").strip() != "correction_required":
            continue
        required_fields = set(expected_challenges[challenge_id].get("current_fields") or [])
        missing_fields = sorted(required_fields - corrected_fields)
        if missing_fields:
            raise ValueError(
                f"numeric challenge {challenge_id} requires corrections for fields: "
                + ", ".join(missing_fields)
            )


def _field_value_type_is_valid(field: str, value: Any) -> bool:
    if field == "design_type":
        return value is None or value in {
            "between-subjects",
            "within-subjects",
            "mixed",
            "correlational",
            "field",
            "archival",
            "other",
        }
    if field in {"conditions_or_factors", "exclusion_reasons"}:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if field == "material_variants":
        return isinstance(value, list)
    if field == "simulation_barriers":
        return _is_valid_barrier_replacement(value)
    if field == "empirical_support":
        return (
            isinstance(value, dict)
            and set(value) == {
                "own_sample_or_assignment",
                "participant_facing_task",
                "quantitative_result",
            }
            and all(status in {"yes", "no", "unclear"} for status in value.values())
        )
    if field == "unit_provenance":
        return value in {"current_paper", "cited_prior", "unclear"}
    if field in {
        "input",
        "participant_task",
        "participants",
        "output",
    }:
        return value is None or isinstance(value, str)
    return False


def _expand_field_correction(value: Dict[str, Any]) -> List[Dict[str, Any]]:
    fields = [
        field.strip()
        for field in str(value.get("field") or "").split("|")
        if field.strip() in _VALID_FIELD_CORRECTIONS
    ]
    if not fields or len(fields) > 2:
        return []
    expected = value.get("expected_value")
    if len(fields) > 1 and not isinstance(expected, dict):
        return []
    expanded: List[Dict[str, Any]] = []
    for field in fields:
        field_value = (
            expected.get(field)
            if len(fields) > 1 and isinstance(expected, dict) and field in expected
            else expected
        )
        field_value = _normalize_field_replacement(field, field_value)
        if not _field_value_type_is_valid(field, field_value):
            continue
        item = dict(value)
        item["field"] = field
        item["expected_value"] = field_value
        expanded.append(item)
    return expanded


def _normalize_field_replacement(field: str, value: Any) -> Any:
    if field in {"input", "participant_task", "participants", "output"}:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, (dict, list, int, float, bool)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return value
    if field in {"conditions_or_factors", "exclusion_reasons"} and isinstance(value, list):
        return [
            item
            if isinstance(item, str)
            else json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in value
        ]
    if field != "design_type" or value is None or not isinstance(value, str):
        return value
    compact = re.sub(r"[^a-z]+", " ", value.lower()).strip()
    if compact in {
        "between subjects",
        "between subject",
        "between subjects design",
        "between subject design",
    }:
        return "between-subjects"
    if compact in {
        "within subjects",
        "within subject",
        "within subjects design",
        "within subject design",
    }:
        return "within-subjects"
    if compact.startswith("mixed") or ("between" in compact and "within" in compact):
        return "mixed"
    if compact.startswith("correlational"):
        return "correlational"
    if compact in {"field", "field study", "field experiment"}:
        return "field"
    if compact.startswith("archival"):
        return "archival"
    if compact == "other":
        return "other"
    return value


def _aggregate_window_reports(
    stage1_json: Dict[str, Any],
    windows: Sequence[DiscoveryWindow],
    reports: Dict[str, Dict[str, Any]],
    *,
    study_reports: Dict[str, Dict[str, Any]],
    study_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    missing: List[Dict[str, Any]] = []
    split_merge: List[Dict[str, Any]] = []
    comparison_groups: List[Dict[str, Any]] = []
    field_corrections: List[Dict[str, Any]] = []
    eligibility: List[Dict[str, Any]] = []
    validation_diagnostics: List[Dict[str, Any]] = []
    dropped_noop_corrections = 0
    notes: List[str] = []
    confidences: List[float] = []
    window_summaries: List[Dict[str, Any]] = []
    for window in windows:
        report = reports[window.window_id]
        missing.extend(report["missing_studies"])
        split_merge.extend(report["split_merge_corrections"])
        comparison_groups.extend(report["comparison_group_corrections"])
        validation_diagnostics.extend(
            {
                "audit_scope": f"boundary:{window.window_id}",
                **diagnostic,
            }
            for diagnostic in report.get("validation_diagnostics") or []
            if isinstance(diagnostic, dict)
        )
        dropped_noop_corrections += int(report.get("dropped_noop_corrections") or 0)
        if report.get("notes"):
            notes.append(f"boundary {window.window_id}: {report['notes']}")
        confidences.append(_confidence(report.get("confidence")))
        window_summaries.append(
            {
                "window_id": window.window_id,
                "pages": list(window.pages),
                "block_ids": list(window.block_ids),
                "issue_count": (
                    len(report["missing_studies"])
                    + len(report["split_merge_corrections"])
                    + len(report["comparison_group_corrections"])
                ),
                "validation_diagnostic_count": len(
                    report.get("validation_diagnostics") or []
                ),
                "dropped_noop_correction_count": int(
                    report.get("dropped_noop_corrections") or 0
                ),
            }
        )
    study_audit_summaries: List[Dict[str, Any]] = []
    for experiment in stage1_json.get("experiments", []) or []:
        if not isinstance(experiment, dict):
            continue
        study_id = _study_key(experiment)
        report = study_reports[study_id]
        field_corrections.extend(report["study_field_corrections"])
        eligibility.extend(report["eligibility_corrections"])
        validation_diagnostics.extend(
            {
                "audit_scope": f"study:{study_id}",
                **diagnostic,
            }
            for diagnostic in report.get("validation_diagnostics") or []
            if isinstance(diagnostic, dict)
        )
        if report.get("notes"):
            notes.append(f"study {study_id}: {report['notes']}")
        confidences.append(_confidence(report.get("confidence")))
        summary = dict(study_summaries[study_id])
        summary["issue_count"] = len(report["study_field_corrections"]) + len(
            report["eligibility_corrections"]
        )
        summary["validation_diagnostic_count"] = len(
            report.get("validation_diagnostics") or []
        )
        study_audit_summaries.append(summary)
    missing = [item for item in missing if _is_grounded_missing_unit(item)]
    missing = _filter_existing_missing_units(stage1_json, missing)
    missing = _dedupe_dicts(missing, ("study", "reason"))
    # Local windows cannot safely reverse the global candidate-ledger boundary
    # decision. Missing units remain auditable; local split/merge suggestions are
    # deliberately ignored.
    split_merge = []
    comparison_groups = []
    known_group_members = {
        str(experiment.get("study_id") or "").strip()
        for experiment in stage1_json.get("experiments", []) or []
        if isinstance(experiment, dict)
    } | {
        str(item.get("proposed_study_id") or "").strip()
        for item in missing
    }
    comparison_groups = [
        correction
        for correction in comparison_groups
        if set(str(member).strip() for member in correction.get("member_study_ids") or [])
        <= known_group_members
    ]
    comparison_groups = _filter_ungrounded_group_corrections(
        stage1_json,
        comparison_groups,
    )
    field_corrections, eligibility = _filter_policy_inconsistent_feedback(
        stage1_json,
        field_corrections,
        eligibility,
    )
    field_corrections = _dedupe_dicts(
        field_corrections,
        ("study", "field", "expected_value"),
    )
    eligibility = _dedupe_dicts(eligibility, ("study", "expected_label"))

    inventory_checks: List[Dict[str, Any]] = []
    for item in missing:
        inventory_checks.append(
            {
                "study": item.get("study"),
                "verdict": "missing",
                "issue": item.get("reason"),
                "evidence": item.get("evidence"),
                "evidence_block_ids": item.get("evidence_block_ids", []),
            }
        )
    for item in split_merge:
        inventory_checks.append(
            {
                "study": ", ".join(item.get("current_study_ids") or []),
                "verdict": item.get("verdict") or "needs_review",
                "issue": item.get("reason"),
                "evidence": item.get("evidence"),
                "evidence_block_ids": item.get("evidence_block_ids", []),
            }
        )
    if not inventory_checks:
        for experiment in stage1_json.get("experiments", []) or []:
            if isinstance(experiment, dict):
                inventory_checks.append(
                    {
                        "study": experiment.get("study_id") or experiment.get("experiment_id"),
                        "verdict": "ok",
                        "issue": "",
                        "evidence": "No contradictory evidence found across verifier windows.",
                        "evidence_block_ids": experiment.get("evidence_refs") or [],
                    }
                )

    eligibility_checks = [
        {
            "study": item.get("study"),
            "verdict": "wrong_label",
            "expected_label": item.get("expected_label"),
            "correction_basis": item.get("correction_basis"),
            "issue": item.get("reason"),
            "evidence": item.get("evidence"),
            "evidence_block_ids": item.get("evidence_block_ids", []),
        }
        for item in eligibility
    ]
    comparison_group_checks = [
        {
            "member_study_ids": item.get("member_study_ids") or [],
            "verdict": item.get("verdict") or "missing_group",
            "relationship_kind": item.get("relationship_kind") or "other",
            "comparison_target": item.get("comparison_target") or "",
            "issue": item.get("reason"),
            "evidence": item.get("evidence"),
            "evidence_block_ids": item.get("evidence_block_ids", []),
        }
        for item in comparison_groups
    ]
    field_checks = [
        {
            "study": item.get("study"),
            "field": item.get("field") or "other",
            "expected_value": item.get("expected_value"),
            "correction_basis": item.get("correction_basis"),
            "issue": item.get("reason"),
            "evidence": item.get("evidence"),
            "evidence_block_ids": item.get("evidence_block_ids", []),
        }
        for item in field_corrections
    ]
    has_issues = bool(
        missing
        or split_merge
        or comparison_groups
        or field_corrections
        or eligibility
        or validation_diagnostics
    )
    return {
        "version": STAGE1_VERIFIER_VERSION,
        "status": "incomplete" if validation_diagnostics else "ok",
        "overall": "needs_review" if has_issues else "pass",
        "confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "inventory_checks": inventory_checks,
        "comparison_group_checks": comparison_group_checks,
        "field_checks": field_checks,
        "eligibility_checks": eligibility_checks,
        "validation_diagnostics": validation_diagnostics,
        "dropped_noop_correction_count": dropped_noop_corrections,
        "regeneration_instructions": {
            "missing_studies": [
                {
                    "study": item.get("study"),
                    "reason": item.get("reason"),
                    "evidence_block_ids": item.get("evidence_block_ids", []),
                }
                for item in missing
            ],
            "split_merge_corrections": split_merge,
            "comparison_group_corrections": comparison_groups,
            "study_field_corrections": field_corrections,
            "eligibility_corrections": eligibility,
        },
        "window_audit": {
            "window_count": len(windows),
            "full_document_llm_calls": 0,
            "windows": window_summaries,
        },
        "study_audit": {
            "prompt_version": STAGE1_STUDY_VERIFIER_PROMPT_VERSION,
            "study_count": len(study_audit_summaries),
            "full_document_llm_calls": 0,
            "all_cited_evidence_included": all(
                bool(summary.get("all_cited_evidence_included"))
                for summary in study_audit_summaries
            ),
            "studies": study_audit_summaries,
        },
        "notes": " | ".join(notes[:12]),
    }


def _reuse_boundary_baseline(
    result: Dict[str, Any],
    baseline: Dict[str, Any],
) -> None:
    """Keep the independent boundary decision stable during field-only repair."""
    baseline_regeneration = baseline.get("regeneration_instructions")
    baseline_regeneration = (
        baseline_regeneration if isinstance(baseline_regeneration, dict) else {}
    )
    regeneration = result.get("regeneration_instructions")
    regeneration = regeneration if isinstance(regeneration, dict) else {}
    for key in (
        "missing_studies",
        "split_merge_corrections",
        "comparison_group_corrections",
    ):
        regeneration[key] = deepcopy(list(baseline_regeneration.get(key) or []))
    result["regeneration_instructions"] = regeneration
    result["inventory_checks"] = deepcopy(list(baseline.get("inventory_checks") or []))
    result["comparison_group_checks"] = deepcopy(
        list(baseline.get("comparison_group_checks") or [])
    )
    window_audit = deepcopy(baseline.get("window_audit"))
    if not isinstance(window_audit, dict):
        window_audit = {}
    window_audit["reused_for_field_refinement"] = True
    result["window_audit"] = window_audit
    result["overall"] = (
        "needs_review"
        if _regeneration_has_issues(regeneration)
        else "pass"
    )


def _regeneration_has_issues(regeneration: Dict[str, Any]) -> bool:
    return any(
        bool(regeneration.get(key))
        for key in (
            "missing_studies",
            "split_merge_corrections",
            "comparison_group_corrections",
            "study_field_corrections",
            "eligibility_corrections",
        )
    )


def _filter_policy_inconsistent_feedback(
    stage1_json: Dict[str, Any],
    field_corrections: Sequence[Dict[str, Any]],
    eligibility: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    experiments = [
        experiment
        for experiment in stage1_json.get("experiments", []) or []
        if isinstance(experiment, dict)
    ]
    experiment_by_key: Dict[str, Dict[str, Any]] = {}
    for experiment in experiments:
        for value in (
            experiment.get("study_id"),
            experiment.get("experiment_id"),
            experiment.get("study_name"),
        ):
            key = _feedback_key(value)
            if key:
                experiment_by_key[key] = experiment

    valid_barrier_replacements: set[str] = set()
    quantitative_support_repairs: set[str] = set()
    output_replacements = {
        _feedback_key(correction.get("study")): correction.get("expected_value")
        for correction in field_corrections
        if str(correction.get("field") or "").strip() == "output"
    }
    eligibility_downgrades = {
        _feedback_key(correction.get("study"))
        for correction in eligibility
        if str(correction.get("expected_label") or "").strip().upper()
        in {"NO", "UNCERTAIN"}
    }
    for correction in field_corrections:
        field = str(correction.get("field") or "").strip()
        key = _feedback_key(correction.get("study"))
        expected = correction.get("expected_value")
        if field == "simulation_barriers" and expected and _is_valid_barrier_replacement(expected):
            valid_barrier_replacements.add(key)
        if (
            field == "empirical_support"
            and isinstance(expected, dict)
            and str(expected.get("quantitative_result") or "").strip().lower()
            == "yes"
            and _has_exact_numeric_output(
                output_replacements.get(
                    key,
                    (experiment_by_key.get(key) or {}).get("output"),
                )
            )
        ):
            quantitative_support_repairs.add(key)

    filtered_fields: List[Dict[str, Any]] = []
    for correction in field_corrections:
        key = _feedback_key(correction.get("study"))
        experiment = experiment_by_key.get(key)
        field = str(correction.get("field") or "").strip()
        expected = correction.get("expected_value")
        if field not in _VALID_FIELD_CORRECTIONS:
            continue
        if experiment is None:
            filtered_fields.append(correction)
            continue
        if _field_values_equal(experiment.get(field), expected):
            continue
        if field == "simulation_barriers":
            if not _is_valid_barrier_replacement(expected):
                continue
            if _barrier_signature(expected) == _barrier_signature(
                experiment.get("simulation_barriers")
            ):
                continue
        if field == "material_variants":
            if not _is_valid_material_variant_replacement(expected):
                continue
            if _material_variants_cover(
                experiment.get("material_variants"),
                expected,
            ):
                continue
        if (
            field == "other"
            and isinstance(expected, dict)
            and set(expected).issubset(
                {"directly_evidenced_in_window", "evidence_refs", "evidence_pages"}
            )
        ):
            continue
        if (
            field == "exclusion_reasons"
            and _has_primary_target_barrier(experiment)
            and _is_empty_expected_value(expected)
            and key not in valid_barrier_replacements
        ):
            continue
        support = experiment.get("empirical_support")
        support = support if isinstance(support, dict) else {}
        if (
            field == "empirical_support"
            and isinstance(expected, dict)
            and str(expected.get("quantitative_result") or "").strip().lower()
            == "yes"
            and key not in quantitative_support_repairs
        ):
            continue
        if (
            field == "exclusion_reasons"
            and _is_empty_expected_value(expected)
            and str(support.get("quantitative_result") or "").strip().lower()
            == "no"
            and key not in quantitative_support_repairs
        ):
            continue
        if (
            field == "exclusion_reasons"
            and str(experiment.get("replicable") or "").strip().upper() == "YES"
            and not _is_empty_expected_value(expected)
            and key not in eligibility_downgrades
        ):
            continue
        if (
            field == "participant_task"
            and _has_primary_target_barrier(experiment)
            and key not in valid_barrier_replacements
        ):
            continue
        filtered_fields.append(correction)

    filtered_eligibility: List[Dict[str, Any]] = []
    for correction in eligibility:
        key = _feedback_key(correction.get("study"))
        experiment = experiment_by_key.get(key)
        expected_label = str(correction.get("expected_label") or "").strip().upper()
        if (
            experiment is not None
            and expected_label
            == str(experiment.get("replicable") or "").strip().upper()
        ):
            continue
        support = experiment.get("empirical_support") if experiment is not None else {}
        support = support if isinstance(support, dict) else {}
        if (
            experiment is not None
            and expected_label in {"YES", "UNCERTAIN"}
            and str(support.get("quantitative_result") or "").strip().lower()
            == "no"
            and key not in quantitative_support_repairs
        ):
            continue
        if (
            experiment is not None
            and expected_label in {"YES", "UNCERTAIN"}
            and _has_primary_target_barrier(experiment)
            and key not in valid_barrier_replacements
        ):
            continue
        filtered_eligibility.append(correction)
    return filtered_fields, filtered_eligibility


def _has_exact_numeric_output(value: Any) -> bool:
    """Require a numeric result in the result-bearing field before eligibility repair."""
    text = str(value or "").strip()
    return bool(re.search(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z])", text))


def _is_grounded_missing_unit(item: Dict[str, Any]) -> bool:
    if item.get("current_paper_collection") is not True:
        return False
    boundary = str(item.get("source_boundary") or "").strip()
    if boundary == "source_labeled":
        return True
    return (
        boundary == "unlabeled_new_collection"
        and item.get("has_exact_response_statistic") is True
        and bool(str(item.get("response_statistic") or "").strip())
    )


def _filter_existing_missing_units(
    stage1_json: Dict[str, Any],
    items: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    experiments = [
        experiment
        for experiment in stage1_json.get("experiments", []) or []
        if isinstance(experiment, dict)
    ]
    known_ids = {
        str(experiment.get("study_id") or "").strip()
        for experiment in experiments
        if str(experiment.get("study_id") or "").strip()
    }
    known_ids.update(
        str(component.get("study_id") or "").strip()
        for experiment in experiments
        for component in experiment.get("candidate_components") or []
        if isinstance(component, dict)
        and str(component.get("study_id") or "").strip()
    )
    represented_formal_labels = {
        formal_label
        for experiment in experiments
        for identity in _experiment_identity_texts(experiment)
        if (formal_label := _formal_label_from_text(identity))
    }
    output: List[Dict[str, Any]] = []
    for item in items:
        if str(item.get("proposed_study_id") or "").strip() in known_ids:
            continue
        item_formal_label = _formal_label_from_text(item.get("study"))
        if item_formal_label and item_formal_label in represented_formal_labels:
            continue
        item_words = _label_word_set(item.get("study"))
        item_refs = {
            str(ref) for ref in item.get("evidence_block_ids") or [] if str(ref)
        }
        already_represented = False
        for experiment in experiments:
            experiment_refs = _experiment_evidence_refs(experiment)
            if item_refs and not (item_refs & experiment_refs):
                continue
            for identity in _experiment_identity_texts(experiment):
                identity_words = _label_word_set(identity)
                if _semantically_contains(identity_words, item_words):
                    already_represented = True
                    break
            if already_represented:
                break
        if not already_represented:
            output.append(item)
    return output


def _experiment_identity_texts(experiment: Dict[str, Any]) -> List[str]:
    values: List[Any] = [
        experiment.get("study_id"),
        experiment.get("experiment_id"),
        experiment.get("study_name"),
        experiment.get("experiment_name"),
        *(experiment.get("candidate_aliases") or []),
    ]
    for component in experiment.get("candidate_components") or []:
        if isinstance(component, dict):
            values.extend(
                [
                    component.get("study_id"),
                    component.get("reported_label"),
                    component.get("study_name"),
                ]
            )
    for context in experiment.get("shared_contexts") or []:
        if isinstance(context, dict):
            values.extend([context.get("reported_label"), context.get("study_name")])
    return [str(value).strip() for value in values if str(value or "").strip()]


def _experiment_evidence_refs(experiment: Dict[str, Any]) -> set[str]:
    refs = {
        str(ref) for ref in experiment.get("evidence_refs") or [] if str(ref)
    }
    for component in experiment.get("candidate_components") or []:
        if not isinstance(component, dict):
            continue
        refs.update(
            str(ref)
            for ref in component.get("evidence_block_ids") or []
            if str(ref)
        )
    return refs


def _filter_structured_boundary_corrections(
    stage1_json: Dict[str, Any],
    corrections: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    experiments_by_id = {
        str(experiment.get("study_id") or "").strip(): experiment
        for experiment in stage1_json.get("experiments", []) or []
        if isinstance(experiment, dict)
        and str(experiment.get("study_id") or "").strip()
    }
    known_ids = set(experiments_by_id)
    output: List[Dict[str, Any]] = []
    for correction in corrections:
        verdict = str(correction.get("verdict") or "").strip()
        current_ids = {
            str(value).strip()
            for value in correction.get("current_study_ids") or []
            if str(value).strip()
        }
        proposed_ids = {
            str(value).strip()
            for value in correction.get("proposed_study_ids") or []
            if str(value).strip()
        }
        proposed_labels = [
            str(value).strip()
            for value in correction.get("proposed_source_labels") or []
            if str(value).strip()
        ]
        boundary_basis = str(correction.get("boundary_basis") or "").strip()
        if not current_ids or not current_ids <= known_ids:
            continue
        if verdict == "split_issue" and len(current_ids) == 1 and len(proposed_ids) >= 2:
            current_id = next(iter(current_ids))
            current_formal = _formal_top_level_label(experiments_by_id[current_id])
            proposed_formal = {
                label for label in map(_formal_label_from_text, proposed_labels) if label
            }
            all_proposed_formal = bool(proposed_labels) and len(proposed_formal) == len(
                proposed_labels
            )
            if boundary_basis == "distinct_source_labels" and not all_proposed_formal:
                continue
            if current_formal and len(proposed_formal) < 2:
                continue
            if (
                not all_proposed_formal
                and _proposed_partition_matches_material_variants(
                    experiments_by_id[current_id],
                    current_id=current_id,
                    proposed_ids=proposed_ids,
                )
            ):
                continue
            output.append(correction)
        elif (
            verdict == "merge_issue"
            and len(current_ids) >= 2
            and len(proposed_ids) == 1
            and boundary_basis == "same_source_unit"
        ):
            source_labels = [
                _formal_top_level_label(experiments_by_id[study_id])
                for study_id in sorted(current_ids)
            ]
            if all(source_labels) and len(set(source_labels)) > 1:
                continue
            output.append(correction)
    return output


def _proposed_partition_matches_material_variants(
    experiment: Dict[str, Any],
    *,
    current_id: str,
    proposed_ids: set[str],
) -> bool:
    variant_keys: List[set[str]] = []
    for variant in experiment.get("material_variants", []) or []:
        if not isinstance(variant, dict):
            continue
        keys = [
            _label_word_set(variant.get("variant_id")),
            _label_word_set(variant.get("label")),
        ]
        variant_keys.extend(key for key in keys if key)
    if not variant_keys:
        return False
    for proposed_id in proposed_ids:
        suffix = proposed_id
        if suffix.startswith(current_id + "_"):
            suffix = suffix[len(current_id) + 1 :]
        proposed_words = _label_word_set(suffix)
        if not any(_semantically_contains(proposed_words, key) for key in variant_keys):
            return False
    return True


def _formal_top_level_label(experiment: Dict[str, Any]) -> str:
    for key in ("experiment_id", "study_id"):
        label = _formal_label_from_text(experiment.get(key))
        if label:
            return label
    return ""


def _formal_label_from_text(value: Any) -> str:
    match = re.search(
        r"\b(?P<kind>study|experiment|problem|survey|pilot|validation|sample)"
        r"[ _:#-]*(?P<label>(?:\d+|[ivxlcdm]+)[a-z]?)\b",
        str(value or ""),
        re.IGNORECASE,
    )
    if not match:
        return ""
    return (
        f"{match.group('kind').lower()}:"
        f"{canonical_unit_number(match.group('label')).lower()}"
    )


def _feedback_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _filter_ungrounded_group_corrections(
    stage1_json: Dict[str, Any],
    corrections: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups_by_id = {
        str(group.get("comparison_group_id") or "").strip(): group
        for group in stage1_json.get("comparison_groups", []) or []
        if isinstance(group, dict) and str(group.get("comparison_group_id") or "").strip()
    }
    groups_by_members = {
        frozenset(str(member) for member in group.get("member_study_ids") or []): group
        for group in groups_by_id.values()
    }
    output: List[Dict[str, Any]] = []
    for correction in corrections:
        verdict = str(correction.get("verdict") or "").strip()
        evidence_kind = str(correction.get("evidence_kind") or "").strip()
        members = frozenset(
            str(member) for member in correction.get("member_study_ids") or []
        )
        if verdict == "missing_group":
            if evidence_kind in {
                "joint_statistical_analysis",
                "explicit_cross_unit_contrast",
                "replication_or_extension",
                "shared_sample_or_sequence",
            } and members not in groups_by_members:
                output.append(correction)
            continue
        current_group_id = str(correction.get("current_group_id") or "").strip()
        current = groups_by_id.get(current_group_id)
        if current is None:
            continue
        current_members = frozenset(
            str(member) for member in current.get("member_study_ids") or []
        )
        if members != current_members or verdict not in {"wrong_members", "spurious_group"}:
            continue
        if verdict == "wrong_members" and evidence_kind not in {
            "joint_statistical_analysis",
            "explicit_cross_unit_contrast",
            "replication_or_extension",
            "shared_sample_or_sequence",
        }:
            continue
        if verdict == "spurious_group" and evidence_kind != "narrative_synthesis":
            continue
        correction_refs = {
            str(ref) for ref in correction.get("evidence_block_ids") or []
        }
        current_refs = {str(ref) for ref in current.get("evidence_refs") or []}
        if correction_refs & current_refs:
            output.append(correction)
    return output


def _is_empty_expected_value(value: Any) -> bool:
    if value in (None, "") or value == []:
        return True
    return str(value).strip().lower() in {"[]", "null", "none"}


def _is_valid_barrier_replacement(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(
        isinstance(barrier, dict)
        and str(barrier.get("kind") or "").strip().lower() in SIMULATION_BARRIER_KINDS
        and isinstance(barrier.get("affects_primary_target"), bool)
        for barrier in value
    )


def _barrier_signature(value: Any) -> set[tuple[str, bool]]:
    return {
        (
            str(barrier.get("kind") or "").strip().lower(),
            barrier.get("affects_primary_target") is True,
        )
        for barrier in (value if isinstance(value, list) else [])
        if isinstance(barrier, dict)
    }


def _has_primary_target_barrier(experiment: Dict[str, Any]) -> bool:
    return any(
        isinstance(barrier, dict) and barrier.get("affects_primary_target") is True
        for barrier in experiment.get("simulation_barriers") or []
    )


def _material_variants_cover(current: Any, expected: Any) -> bool:
    if not isinstance(current, list) or not isinstance(expected, list) or not expected:
        return False
    current_variants = [value for value in current if isinstance(value, dict)]
    expected_variants = [value for value in expected if isinstance(value, dict)]
    if len(expected_variants) != len(expected):
        return False
    for required in expected_variants:
        required_label = _label_word_set(required.get("label"))
        required_role = str(required.get("role") or "").strip().lower()
        if required_role not in {
            "condition",
            "stimulus",
            "form",
            "order",
            "item_set",
            "other",
        }:
            required_role = ""
        matched = False
        for available in current_variants:
            if required_role and required_role != str(available.get("role") or "").strip().lower():
                continue
            if not _semantically_contains(
                _label_word_set(available.get("label")),
                required_label,
            ):
                continue
            matched = True
            break
        if not matched:
            return False
    return True


def _is_valid_material_variant_replacement(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if not value:
        return True
    if len(value) < 2:
        return False
    valid_roles = {"condition", "stimulus", "form", "order", "item_set", "other"}
    return all(
        isinstance(variant, dict)
        and bool(str(variant.get("variant_id") or "").strip())
        and bool(str(variant.get("label") or "").strip())
        and str(variant.get("role") or "").strip() in valid_roles
        and variant.get("is_alternative_version") is True
        and isinstance(variant.get("evidence_refs"), list)
        and bool(variant.get("evidence_refs"))
        for variant in value
    )


def _field_values_equal(current: Any, expected: Any) -> bool:
    if isinstance(current, str) and isinstance(expected, str):
        return " ".join(current.split()).casefold() == " ".join(expected.split()).casefold()
    return current == expected


def _word_set(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _label_word_set(value: Any) -> set[str]:
    text = re.sub(r"\([^)]*\)", " ", str(value or ""))
    return _word_set(text)


def _semantically_contains(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    smaller = left if len(left) <= len(right) else right
    return len(left & right) / len(smaller) >= 0.8


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


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return round(min(1.0, max(0.0, number)), 3)


def _skipped_report(notes: str) -> Dict[str, Any]:
    return {
        "version": STAGE1_VERIFIER_VERSION,
        "status": "skipped",
        "overall": "needs_review",
        "notes": notes,
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
            "prompt_version": STAGE1_STUDY_VERIFIER_PROMPT_VERSION,
            "study_count": 0,
            "full_document_llm_calls": 0,
            "all_cited_evidence_included": False,
            "studies": [],
        },
    }
