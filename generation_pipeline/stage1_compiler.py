"""Evidence-driven Stage 1 compiler.

The compiler never sends the complete paper in one LLM request. It first maps
every parsed PDF block through bounded discovery windows, reconciles the
resulting study mentions, and then extracts one study at a time from anchored
evidence contexts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from generation_pipeline.identifiers import canonical_sub_study_id
from generation_pipeline.pdf.evidence import PdfEvidenceIndex
from generation_pipeline.pdf.models import DocumentBlock, EvidenceContext, ParsedPdfDocument
from generation_pipeline.pdf.parser import parse_pdf_document


STAGE1_COMPILER_VERSION = "stage1-evidence-compiler-v19"
DISCOVERY_PROMPT_VERSION = "stage1-discovery-v10"
STUDY_EXTRACTION_PROMPT_VERSION = "stage1-study-extraction-v13"
BOUNDARY_ADJUDICATION_PROMPT_VERSION = "stage1-boundary-adjudication-v6"

DISCOVERY_WINDOW_MAX_CHARS = 18000
DISCOVERY_WINDOW_OVERLAP_UNITS = 2
STUDY_CONTEXT_MAX_CHARS = 48000
DISCOVERY_MAX_TOKENS = 8000
STUDY_EXTRACTION_MAX_TOKENS = 12000
BOUNDARY_ADJUDICATION_MAX_TOKENS = 8000
BOUNDARY_ADJUDICATION_MAX_CONTEXT_CHARS = 60000
DEFAULT_TIMEOUT = 300.0
DEFAULT_WORKERS = 4

SIMULATION_BARRIER_KINDS = {
    "physical_action",
    "consequential_commitment",
    "live_interaction",
    "longitudinal_exposure",
    "dynamic_environment",
    "specialized_apparatus",
    "other",
}


@dataclass(frozen=True)
class DiscoveryWindow:
    window_id: str
    text: str
    block_ids: List[str]
    pages: List[int]
    char_count: int

    def summary(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "block_ids": list(self.block_ids),
            "pages": list(self.pages),
            "char_count": self.char_count,
        }


@dataclass(frozen=True)
class _WindowUnit:
    block: DocumentBlock
    part: int
    parts: int
    text: str

    def render(self) -> str:
        pages = (
            str(self.block.page_start)
            if self.block.page_start == self.block.page_end
            else f"{self.block.page_start}-{self.block.page_end}"
        )
        section = " > ".join(self.block.section_path) or "(unsectioned)"
        part = f" | part={self.part}/{self.parts}" if self.parts > 1 else ""
        return (
            f"[Block {self.block.block_id} | page {pages} | "
            f"type={self.block.block_type} | section={section}{part}]\n"
            f"{self.text.strip()}"
        )


def stage1_policy_text() -> str:
    return """HumanStudy-Bench simulation eligibility is topic-independent.

- YES: human participants received an instruction, stimulus, scenario, survey,
  task, or choice set; produced a representable response; and the paper reports
  at least one quantitative target result with enough design/sample structure
  to define the simulation.
- UNCERTAIN: the unit is probably participant-simulatable, but its boundary,
  task, assignment, outcome, or material source is ambiguous.
- NO: theoretical/review/qualitative-only work; non-human or purely archival
  analysis without a participant task; no quantitative target; or a task that
  fundamentally requires an unrepresented physical action, behaviorally
  consequential commitment, longitudinal exposure, or live social interaction.

Scientific topic or discipline is never an exclusion criterion. Missing exact
questionnaire wording, options, or images is a material-readiness gap for Stage
3, not by itself a reason to label an otherwise simulatable study NO. The
legacy field `replicable` means simulation eligibility, not guaranteed
scientific reproduction. Judge the original measured response, not a weakened
substitute: a real physical behavior or live interaction is not made
text-simulatable by replacing it with a hypothetical intention question. A
choice carrying an enforced real-world consequence may itself be a barrier even
when the paper records the choice before physical follow-through; an explicitly
hypothetical choice is not."""


def build_stage1_task_prompt(
    pdf_name: str,
    num_pages: int,
    regeneration_instructions: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the shared Stage 1 task contract without embedding paper text."""
    feedback = _feedback_text(regeneration_instructions)
    return f"""Analyze the social-science research paper {pdf_name} ({num_pages} pages).

First inventory every distinct empirical study, experiment, survey, pilot, and
validation sample. Then classify whether each unit can be represented as a
    HumanStudy-Bench participant task. Topic is unrestricted: psychology,
    behavioral economics, organizational behavior, political science,
    sociology, communication, marketing, education, and HCI are all in scope.

{stage1_policy_text()}
{feedback}

For every study recover stable IDs, design, factors/conditions, nested
participant-facing material variants, participant input and task, sample,
output, material-source hints, missing materials, and source evidence. Preserve
uncertainty instead of guessing."""


def build_discovery_windows(
    document: ParsedPdfDocument,
    *,
    max_chars: int = DISCOVERY_WINDOW_MAX_CHARS,
    overlap_units: int = DISCOVERY_WINDOW_OVERLAP_UNITS,
) -> List[DiscoveryWindow]:
    """Cover every parsed block with bounded, slightly overlapping windows."""
    if max_chars < 2000:
        raise ValueError("Stage 1 discovery windows must allow at least 2000 characters")
    units: List[_WindowUnit] = []
    unit_text_limit = max(1000, max_chars - 600)
    for block in sorted(document.blocks, key=lambda item: item.order):
        fragments = _split_text(block.text, unit_text_limit)
        for part, fragment in enumerate(fragments, start=1):
            units.append(_WindowUnit(block=block, part=part, parts=len(fragments), text=fragment))
    if not units:
        return []

    windows: List[DiscoveryWindow] = []
    start = 0
    while start < len(units):
        chosen: List[_WindowUnit] = []
        used = 0
        end = start
        while end < len(units):
            rendered = units[end].render()
            extra = len(rendered) + (2 if chosen else 0)
            if chosen and used + extra > max_chars:
                break
            chosen.append(units[end])
            used += extra
            end += 1
            if used >= max_chars:
                break
        if not chosen:
            chosen = [units[start]]
            end = start + 1

        text = "\n\n".join(unit.render() for unit in chosen)
        block_ids = list(dict.fromkeys(unit.block.block_id for unit in chosen))
        pages = sorted(
            {
                page
                for unit in chosen
                for page in range(unit.block.page_start, unit.block.page_end + 1)
            }
        )
        windows.append(
            DiscoveryWindow(
                window_id=f"window_{len(windows) + 1:03d}",
                text=text,
                block_ids=block_ids,
                pages=pages,
                char_count=len(text),
            )
        )
        if end >= len(units):
            break
        start = max(start + 1, end - max(0, int(overlap_units)))
    return windows


def compile_stage1_inventory(
    pdf_path: Path,
    llm_client: Any,
    *,
    regeneration_instructions: Optional[Dict[str, Any]] = None,
    pdf_artifacts_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
) -> Dict[str, Any]:
    """Compile a source-grounded Stage 1 inventory from bounded LLM calls."""
    if llm_client is None:
        raise ValueError("Stage 1 evidence compiler requires an LLM client")
    pdf_path = Path(pdf_path)
    compiler_dir = Path(artifacts_dir) if artifacts_dir is not None else None
    if compiler_dir is not None:
        compiler_dir.mkdir(parents=True, exist_ok=True)
    document = parse_pdf_document(
        pdf_path,
        artifacts_dir=pdf_artifacts_dir,
        force=force,
        prefer_docling=True,
    )
    windows = build_discovery_windows(document)
    if not windows:
        raise ValueError("PDF parser returned no text blocks for Stage 1 discovery")
    print(
        f"  Stage 1 evidence parser: {document.parser} pages={document.page_count} "
        f"blocks={len(document.blocks)} windows={len(windows)} chars={document.text_chars}",
        flush=True,
    )

    mentions, relation_mentions, metadata, discovery_summaries = _discover_all_windows(
        windows,
        llm_client,
        pdf_path=pdf_path,
        regeneration_instructions=regeneration_instructions,
        artifacts_dir=compiler_dir / "discovery" if compiler_dir else None,
        timeout=timeout,
        workers=workers,
        force=force,
    )
    candidates, reconciliation = _reconcile_mentions(mentions)

    index = PdfEvidenceIndex(document)
    candidates, boundary_adjudication = _adjudicate_candidate_boundaries(
        candidates,
        relation_mentions,
        index,
        llm_client,
        pdf_path=pdf_path,
        artifacts_dir=compiler_dir / "reconciliation" if compiler_dir else None,
        timeout=timeout,
        force=force,
    )
    reconciliation["actions"] = [
        *(reconciliation.get("actions") or []),
        *(boundary_adjudication.get("actions") or []),
    ]
    shared_context_actions = _attach_discovered_shared_sample_contexts(
        candidates,
        relation_mentions,
        valid_refs=set(index.document.block_map()),
    )
    reconciliation["actions"].extend(shared_context_actions)
    extracted_records, contexts, extraction_errors = _extract_all_studies(
        candidates,
        index,
        llm_client,
        pdf_path=pdf_path,
        regeneration_instructions=regeneration_instructions,
        artifacts_dir=compiler_dir / "studies" if compiler_dir else None,
        timeout=timeout,
        workers=workers,
        force=force,
    )
    experiments, rejected_candidates = _partition_empirical_units(extracted_records)
    rejected_candidates = [
        *(boundary_adjudication.get("rejected_candidates") or []),
        *rejected_candidates,
    ]
    comparison_groups, rejected_relations, ignored_relations = _reconcile_comparison_groups(
        relation_mentions,
        experiments,
        all_records=[*extracted_records, *rejected_candidates],
    )
    result = {
        "paper_title": metadata.get("paper_title") or pdf_path.stem,
        "paper_authors": metadata.get("paper_authors") or [],
        "paper_abstract": metadata.get("paper_abstract") or "",
        "experiments": experiments,
        "comparison_groups": comparison_groups,
        "overall_replicable": any(
            str(experiment.get("replicable") or "").upper() in {"YES", "UNCERTAIN"}
            for experiment in experiments
        ),
        "confidence": _mean_confidence(experiments),
        "notes": (
            "Stage 1 compiled from complete block-window discovery and "
            "per-study evidence contexts; no full-document LLM request was used."
        ),
        "stage1_evidence": {
            "version": STAGE1_COMPILER_VERSION,
            "source_sha256": document.source_sha256,
            "parser": document.parser,
            "parser_version": document.parser_version,
            "parser_degraded": document.degraded,
            "parser_warnings": list(document.warnings),
            "page_count": document.page_count,
            "text_chars": document.text_chars,
            "full_document_llm_calls": 0,
            "discovery_window_count": len(windows),
            "discovery_windows": discovery_summaries,
            "raw_mention_count": len(mentions),
            "raw_comparison_relation_count": len(relation_mentions),
            "comparison_relation_mentions": relation_mentions,
            "reconciled_study_count": len(candidates),
            "accepted_empirical_unit_count": len(experiments),
            "rejected_candidate_count": len(rejected_candidates),
            "rejected_candidates": rejected_candidates,
            "comparison_group_count": len(comparison_groups),
            "rejected_comparison_relation_count": len(rejected_relations),
            "rejected_comparison_relations": rejected_relations,
            "ignored_comparison_relation_count": len(ignored_relations),
            "ignored_comparison_relations": ignored_relations,
            "all_comparison_relations_resolved": not rejected_relations,
            "all_mentions_assigned": reconciliation.get("all_mentions_assigned", False),
            "reconciliation_warnings": reconciliation.get("warnings", []),
            "reconciliation_actions": reconciliation.get("actions", []),
            "boundary_adjudication": {
                key: value
                for key, value in boundary_adjudication.items()
                if key != "rejected_candidates"
            },
            "extraction_complete": not extraction_errors,
            "extraction_errors": extraction_errors,
            "study_contexts": contexts,
        },
    }
    if compiler_dir is not None:
        _write_json(
            compiler_dir / "paper_map.json",
            {
                "version": STAGE1_COMPILER_VERSION,
                "source_sha256": document.source_sha256,
                "metadata": metadata,
                "windows": discovery_summaries,
                "mentions": mentions,
                "comparison_relation_mentions": relation_mentions,
                "candidates": candidates,
                "extracted_records": extracted_records,
                "rejected_candidates": rejected_candidates,
                "comparison_groups": comparison_groups,
                "rejected_comparison_relations": rejected_relations,
                "ignored_comparison_relations": ignored_relations,
                "reconciliation": reconciliation,
                "boundary_adjudication": boundary_adjudication,
                "study_contexts": contexts,
            },
        )
    return result


def _discover_all_windows(
    windows: Sequence[DiscoveryWindow],
    llm_client: Any,
    *,
    pdf_path: Path,
    regeneration_instructions: Optional[Dict[str, Any]],
    artifacts_dir: Optional[Path],
    timeout: Optional[float],
    workers: int,
    force: bool,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    def run(window: DiscoveryWindow) -> Dict[str, Any]:
        prompt = _discovery_prompt(
            window,
            pdf_name=pdf_path.name,
            regeneration_instructions=_feedback_for_window(
                regeneration_instructions,
                window,
            ),
        )
        cache_path = artifacts_dir / f"{window.window_id}.json" if artifacts_dir else None
        return cached_json_call(
            llm_client,
            prompt,
            cache_path=cache_path,
            prompt_version=DISCOVERY_PROMPT_VERSION,
            timeout=timeout,
            max_tokens=DISCOVERY_MAX_TOKENS,
            force=force,
            validator=lambda payload: _validate_discovery_payload(payload, window),
        )

    with ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as pool:
        future_map = {pool.submit(run, window): window for window in windows}
        for future in as_completed(future_map):
            window = future_map[future]
            try:
                results[window.window_id] = future.result()
            except Exception as exc:
                errors[window.window_id] = f"{type(exc).__name__}: {exc}"
    if errors:
        detail = "; ".join(f"{key}={value}" for key, value in sorted(errors.items()))
        raise RuntimeError(
            "Stage 1 discovery was incomplete; refusing to emit a partial study inventory: "
            + detail
        )

    mentions: List[Dict[str, Any]] = []
    relation_mentions: List[Dict[str, Any]] = []
    metadata_values: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for window in windows:
        payload = results.get(window.window_id, {})
        raw_mentions = payload.get("candidate_mentions")
        if not isinstance(raw_mentions, list):
            raw_mentions = []
        normalized: List[Dict[str, Any]] = []
        for position, raw in enumerate(raw_mentions, start=1):
            if not isinstance(raw, dict):
                continue
            mention = _normalize_mention(raw, window, position)
            normalized.append(mention)
            mentions.append(mention)
        raw_relations = payload.get("comparison_relations")
        if not isinstance(raw_relations, list):
            raw_relations = []
        normalized_relations: List[Dict[str, Any]] = []
        for position, raw in enumerate(raw_relations, start=1):
            if not isinstance(raw, dict):
                continue
            relation = _normalize_relation_mention(raw, window, position)
            normalized_relations.append(relation)
            relation_mentions.append(relation)
        raw_metadata = payload.get("paper_metadata")
        if isinstance(raw_metadata, dict):
            metadata_values.append(raw_metadata)
        summaries.append(
            {
                **window.summary(),
                "mention_count": len(normalized),
                "comparison_relation_count": len(normalized_relations),
            }
        )
        print(
            f"  Stage 1 discovery {window.window_id}: pages={_page_label(window.pages)} "
            f"mentions={len(normalized)} relations={len(normalized_relations)}",
            flush=True,
        )
    return mentions, relation_mentions, _merge_metadata(metadata_values), summaries


def _discovery_prompt(
    window: DiscoveryWindow,
    *,
    pdf_name: str,
    regeneration_instructions: Optional[Dict[str, Any]],
) -> str:
    return f"""You are the high-recall discovery pass for Stage 1 of HumanStudy-Bench.

Paper: {pdf_name}
Window: {window.window_id}; pages: {_page_label(window.pages)}
Valid evidence block IDs: {json.dumps(window.block_ids, ensure_ascii=False)}

Inspect this entire bounded window and report every DISTINCT empirical unit that
could own one HumanStudy-Bench simulation environment. Source hierarchy defines
the boundary first: a paper-labeled Study, Experiment, Survey, Pilot, or
Validation is one parent unit. Its participant-facing steps, measures, response
formats, outcomes, and material variants remain components of that parent. For
papers without such parent labels, use one coherent participant-facing task
family with a common procedure and target phenomenon. Do not turn each dependent
variable, statistical test, manipulation check, parameter value, or condition
into a separate unit. Do not report prior studies that are merely cited in the
introduction or references. Preserve genuinely ambiguous candidates for global
reconciliation instead of dismissing them.

Boundary examples that apply across disciplines:
- A paper-labeled Study, Experiment, Problem, Survey, or Pilot is the default
  top-level empirical unit. Copy only that parent label into reported_label.
- Stories, vignettes, signs, randomized arms, questionnaire forms, order
  versions, item sets, response categories, table rows, and condition levels
  inside one parent unit are NOT additional studies. Put participant-facing
  versions in material_variants and factors/arms in the later study extraction.
- Repeated or parameterized items with the same task template are one unit,
  even when separate participant groups receive different item versions.
  Multiple prompts, judgments, scales, or dependent measures administered
  inside one formally labeled Study/Experiment remain task components of that
  parent, even when their response formats or target constructs differ.
- When no formal parent exists, substantively different task families with
  different procedures or target phenomena remain separate even when embedded
  in one questionnaire or answered by the same respondents.
- A questionnaire-wide sample/method description shared by several task
  families is context, not its own simulation unit. Still report it as an
  ambiguous candidate so reconciliation can attach its sample/procedure evidence
  to all affected task units without merging those units.
- A table row, subgroup result, sample-description sentence, narrative
  observation, cited example, or result fragment is never a unit by itself.
- Split a parent label only when the paper explicitly identifies another
  top-level data collection, such as Study 2a vs Study 2b. A separate condition
  sample size, measurement phase, response scale, or outcome inside Study 1
  does not create another Study 1.
- For an unlabeled paper, use a coherent task family as the boundary. A changed
  participant group alone does not split an otherwise identical task template;
  a shared participant collection alone does not merge substantively different
  tasks.
- A Methods/Procedure/Results/Discussion heading, a bare Roman-numeral section,
  or a common procedure described before several labeled experiments is not a
  separate empirical unit without its own participant collection and response
  result. Attach that evidence to the relevant source-labeled parent instead.
- Once a source-labeled parent such as Experiment VI is visible, subordinate
  headings and rows remain inside it until another peer source label begins.

This pass discovers boundaries and source-explicit relationships only. It does
not make the final eligibility decision. Keep distinct top-level empirical
units separate even when the paper compares them in one result. Record that
cross-unit link in comparison_relations. Do not use comparison_relations for
variants inside one unit. Do not infer a comparison from similar wording or
adjacent numbering. Every candidate, variant, and relationship must cite valid
block IDs from this window. Only return metadata fields when directly visible.

For comparison_relations, emit the maximal source-explicit member set for one
finding instead of redundant pairwise subsets. Use shared_sample only when the
text explicitly says the same participants completed all member units; a
shared population, recruitment setting, or statement that per-problem Ns were
reported is not shared-sample evidence. Each member_ref must copy the exact
reported_label from candidate_mentions in this same JSON response. Do not
paraphrase or rename a member in the relation.
{_feedback_text(regeneration_instructions)}

PAPER WINDOW:
{window.text}

Return only this JSON object:
{{
  "paper_metadata": {{
    "paper_title": "string or null",
    "paper_authors": ["directly visible authors"],
    "paper_abstract": "directly visible abstract or null"
  }},
  "candidate_mentions": [
    {{
      "reported_label": "exact top-level label such as Study 1 or Problem 4",
      "study_name": "short name for the complete parent empirical unit",
      "kind": "study|experiment|survey|pilot|validation|field|other",
      "participant_task_hint": "short source-grounded hint or null",
      "quantitative_target_hint": "short source-grounded hint or null",
      "material_variants": [
        {{
          "label": "source label for a participant-facing version",
          "role": "condition|stimulus|form|order|item_set|other",
          "evidence_block_ids": ["valid block id"]
        }}
      ],
      "evidence_block_ids": ["valid block id"],
      "evidence_summary": "brief evidence for this being a distinct empirical unit",
      "boundary_confidence": 0.0
    }}
  ],
  "comparison_relations": [
    {{
      "member_refs": [
        {{"reported_label": "exact top-level candidate label"}},
        {{"reported_label": "another exact top-level candidate label"}}
      ],
      "members_are_distinct_empirical_units": true,
      "relationship_kind": "paired_contrast|multi_unit_comparison|replication_set|sequence|shared_sample|other",
      "comparison_target": "what response, manipulation, or hypothesis is compared",
      "evidence_block_ids": ["valid block id"],
      "evidence_summary": "source-explicit reason these distinct units are analyzed together",
      "confidence": 0.0
    }}
  ]
}}"""


def _normalize_mention(
    raw: Dict[str, Any],
    window: DiscoveryWindow,
    position: int,
) -> Dict[str, Any]:
    valid_refs = set(window.block_ids)
    refs = [
        str(value)
        for value in raw.get("evidence_block_ids") or []
        if str(value) in valid_refs
    ]
    raw_label = str(
        raw.get("reported_label")
        or raw.get("study_name")
        or f"Unlabeled empirical unit in {window.window_id}"
    ).strip()
    label = _canonical_reported_label(raw_label)
    material_variants = _normalize_discovery_variants(
        raw.get("material_variants"),
        valid_refs=valid_refs,
    )
    return {
        "mention_id": f"{window.window_id}_mention_{position:02d}",
        "window_id": window.window_id,
        "reported_label": label,
        "raw_reported_label": raw_label,
        "material_variants": material_variants,
        "study_name": str(raw.get("study_name") or label).strip(),
        "kind": str(raw.get("kind") or "other").strip().lower(),
        "participant_task_hint": _optional_text(raw.get("participant_task_hint")),
        "quantitative_target_hint": _optional_text(raw.get("quantitative_target_hint")),
        "evidence_block_ids": list(dict.fromkeys(refs)),
        "evidence_summary": str(raw.get("evidence_summary") or "").strip(),
        "boundary_confidence": _confidence(raw.get("boundary_confidence")),
    }


def _normalize_relation_mention(
    raw: Dict[str, Any],
    window: DiscoveryWindow,
    position: int,
) -> Dict[str, Any]:
    valid_refs = set(window.block_ids)
    member_refs = [
        {
            "reported_label": _canonical_reported_label(
                str(value.get("reported_label") or "").strip()
            ),
        }
        for value in raw.get("member_refs") or []
        if isinstance(value, dict) and str(value.get("reported_label") or "").strip()
    ]
    if not member_refs:
        member_refs = [
            {
                "reported_label": _canonical_reported_label(str(value).strip()),
            }
            for value in raw.get("member_labels") or []
            if str(value).strip()
        ]
    refs = [
        str(value)
        for value in raw.get("evidence_block_ids") or []
        if str(value) in valid_refs
    ]
    return {
        "relation_mention_id": f"{window.window_id}_relation_{position:02d}",
        "window_id": window.window_id,
        "member_refs": member_refs,
        "member_labels": [_member_ref_display(value) for value in member_refs],
        "members_are_distinct_empirical_units": bool(
            raw.get("members_are_distinct_empirical_units")
        ),
        "relationship_kind": str(raw.get("relationship_kind") or "other").strip().lower(),
        "comparison_target": str(raw.get("comparison_target") or "").strip(),
        "evidence_block_ids": list(dict.fromkeys(refs)),
        "evidence_summary": str(raw.get("evidence_summary") or "").strip(),
        "confidence": _confidence(raw.get("confidence")),
    }


def _member_ref_display(member_ref: Dict[str, Any]) -> str:
    return str(member_ref.get("reported_label") or "").strip()


def _reconcile_mentions(
    mentions: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not mentions:
        return [], {
            "strategy": "deterministic_parent_unit_deduplication",
            "all_mentions_assigned": True,
            "warnings": [],
            "actions": [],
        }
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for mention in mentions:
        key = _mention_identity_key(mention)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(mention)

    groups = [grouped[key] for key in order]
    actions: List[Dict[str, Any]] = []

    # Overlapping windows often describe the same unlabeled task with a short
    # label in one window and a qualified label in another. Merge only when the
    # evidence overlaps and the labels/names semantically contain one another;
    # distinct numbered source labels are never merged by this rule.
    group_index = 0
    while group_index < len(groups):
        other_index = group_index + 1
        while other_index < len(groups):
            if _mention_groups_are_duplicate(groups[group_index], groups[other_index]):
                left_ids = [str(item.get("mention_id") or "") for item in groups[group_index]]
                right_ids = [str(item.get("mention_id") or "") for item in groups[other_index]]
                groups[group_index].extend(groups.pop(other_index))
                actions.append(
                    {
                        "action": "merge_duplicate_mentions",
                        "source_mention_ids": [*left_ids, *right_ids],
                        "reason": "overlapping evidence and semantically equivalent source labels",
                    }
                )
                continue
            other_index += 1
        group_index += 1

    # A weakly named group/table/form that cites evidence already owned by one
    # formal parent is subordinate to that parent. Requiring exactly one parent
    # keeps shared tables that mention several experiments ambiguous rather than
    # assigning them arbitrarily.
    weak_index = 0
    while weak_index < len(groups):
        weak_group = groups[weak_index]
        if _group_has_source_anchor(weak_group):
            weak_index += 1
            continue
        weak_refs = _group_evidence_refs(weak_group)
        parent_indexes = [
            index
            for index, group in enumerate(groups)
            if index != weak_index
            and _group_has_source_anchor(group)
            and bool(weak_refs & _group_evidence_refs(group))
        ]
        if len(parent_indexes) != 1:
            weak_index += 1
            continue
        parent_index = parent_indexes[0]
        child_ids = [str(item.get("mention_id") or "") for item in weak_group]
        parent_ids = [str(item.get("mention_id") or "") for item in groups[parent_index]]
        groups[parent_index].extend(weak_group)
        groups.pop(weak_index)
        if weak_index < parent_index:
            parent_index -= 1
        actions.append(
            {
                "action": "attach_subordinate_mentions",
                "source_mention_ids": child_ids,
                "parent_mention_ids": parent_ids,
                "reason": "weak child boundary overlaps exactly one source-anchored parent",
            }
        )
        if weak_index > parent_index:
            weak_index -= 1

    mention_order = {
        str(mention.get("mention_id") or ""): position
        for position, mention in enumerate(mentions)
    }
    groups.sort(
        key=lambda group: min(
            mention_order.get(str(mention.get("mention_id") or ""), len(mentions))
            for mention in group
        )
    )

    used_study_ids: set[str] = set()
    candidates: List[Dict[str, Any]] = []
    for source_mentions in groups:
        first = _preferred_parent_mention(source_mentions)
        reported_label = first.get("reported_label")
        key = (
            canonical_sub_study_id(_canonical_reported_label(reported_label))
            if _explicit_unit_labels(reported_label)
            else _mention_identity_key(first)
        )
        source_anchor = _mention_has_source_anchor(first)
        candidate = _candidate_from_mentions(
            {
                "study_id": key,
                "reported_label": first.get("reported_label"),
                "study_name": _reconciled_study_name(source_mentions),
                "kind": first.get("kind"),
                "source_anchor": source_anchor,
                "boundary_notes": (
                    "Mentions were reconciled as one source-level empirical unit from "
                    "overlapping labels/evidence and subordinate material mentions."
                    if len(source_mentions) > 1
                    else "Preserved as a distinct reported empirical unit."
                ),
                "boundary_confidence": max(
                    (_confidence(item.get("boundary_confidence")) for item in source_mentions),
                    default=0.0,
                ),
            },
            source_mentions,
            used_study_ids,
        )
        candidates.append(candidate)
    return candidates, {
        "strategy": "evidence_graph_parent_unit_reconciliation",
        "all_mentions_assigned": sum(len(group) for group in groups) == len(mentions),
        "warnings": [],
        "actions": actions,
    }


_EXPLICIT_UNIT_RE = re.compile(
    r"\b(study|experiment|problem|survey|pilot|validation)\s*[-:#]?\s*"
    r"(\d+[a-z]?|[ivxlcdm]+[a-z]?)\b",
    re.IGNORECASE,
)
_VALID_ROMAN_RE = re.compile(
    r"^M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$"
)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

_SOURCE_ANCHOR_RE = re.compile(
    r"\b(stud(?:y|ies)|experiments?|problems?|surveys?|pilots?|pretests?|"
    r"validations?|field\s+stud(?:y|ies)|trials?|waves?)\b",
    re.IGNORECASE,
)
_FALLBACK_ANCHOR_RE = re.compile(
    r"\b(problems?|surveys?|pilots?|pretests?|validations?|field\s+stud(?:y|ies))\b",
    re.IGNORECASE,
)
_LABEL_STOP_WORDS = {
    "a",
    "an",
    "and",
    "check",
    "checklist",
    "experiment",
    "experiments",
    "form",
    "group",
    "groups",
    "item",
    "items",
    "list",
    "new",
    "of",
    "pilot",
    "pretest",
    "problem",
    "problems",
    "procedure",
    "sample",
    "samples",
    "study",
    "studies",
    "survey",
    "table",
    "task",
    "the",
    "trial",
    "validation",
    "wave",
}


def _explicit_unit_labels(value: Any) -> List[str]:
    labels: List[str] = []
    for match in _EXPLICIT_UNIT_RE.finditer(str(value or "")):
        kind = match.group(1).lower().title()
        number = match.group(2)
        number = number.upper() if number.isalpha() else number.lower()
        label = f"{kind} {number}"
        if label not in labels:
            labels.append(label)
    return labels


def _normalized_explicit_unit_labels(value: Any) -> List[str]:
    labels: List[str] = []
    for match in _EXPLICIT_UNIT_RE.finditer(str(value or "")):
        label = (
            f"{match.group(1).lower().title()} "
            f"{canonical_unit_number(match.group(2))}"
        )
        if label not in labels:
            labels.append(label)
    return labels


def canonical_unit_number(value: Any) -> str:
    """Normalize Arabic/Roman source labels while preserving letter suffixes."""
    token = str(value or "").strip()
    digit_match = re.fullmatch(r"0*(\d+)([a-z]?)", token, re.IGNORECASE)
    if digit_match:
        return f"{int(digit_match.group(1))}{digit_match.group(2).upper()}"

    upper = token.upper()
    roman_value = _roman_value(upper)
    if roman_value is not None:
        return str(roman_value)
    if len(upper) > 1:
        roman_value = _roman_value(upper[:-1])
        if roman_value is not None:
            return f"{roman_value}{upper[-1]}"
    return upper


def _roman_value(value: str) -> Optional[int]:
    if not value or not _VALID_ROMAN_RE.fullmatch(value):
        return None
    total = 0
    previous = 0
    for character in reversed(value):
        current = _ROMAN_VALUES[character]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total


def _canonical_reported_label(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    labels = _explicit_unit_labels(text)
    return labels[0] if len(labels) == 1 else text


def _mention_identity_key(mention: Dict[str, Any]) -> str:
    label = str(mention.get("reported_label") or mention.get("study_name") or "empirical_unit")
    labels = _normalized_explicit_unit_labels(label)
    if len(labels) == 1:
        return canonical_sub_study_id(labels[0])
    normalized = re.sub(r"\bn\s*=\s*\d+\b", "", label, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return canonical_sub_study_id(normalized)


def _mention_has_source_anchor(mention: Dict[str, Any]) -> bool:
    label = str(mention.get("reported_label") or mention.get("raw_reported_label") or "")
    if _explicit_unit_labels(label):
        return True
    if _SOURCE_ANCHOR_RE.search(label):
        return True
    # Discovery-generated study names can clarify an unlabeled problem/survey,
    # but a generic word such as "experiment" in a paraphrase is not itself a
    # source label.
    return bool(_FALLBACK_ANCHOR_RE.search(str(mention.get("study_name") or "")))


def _group_has_source_anchor(group: Sequence[Dict[str, Any]]) -> bool:
    return any(_mention_has_source_anchor(mention) for mention in group)


def _group_evidence_refs(group: Sequence[Dict[str, Any]]) -> set[str]:
    return {
        str(ref)
        for mention in group
        for ref in mention.get("evidence_block_ids") or []
        if str(ref)
    }


def _label_words(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if token not in _LABEL_STOP_WORDS and len(token) > 1
    }


def _labels_semantically_contain(left: Any, right: Any) -> bool:
    left_words = _label_words(left)
    right_words = _label_words(right)
    if not left_words or not right_words:
        return False
    shared = left_words & right_words
    return len(shared) >= 2 and len(shared) / min(len(left_words), len(right_words)) >= 0.8


def _mention_groups_are_duplicate(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
) -> bool:
    if not (_group_evidence_refs(left) & _group_evidence_refs(right)):
        return False
    left_explicit = {
        label
        for mention in left
        for label in _normalized_explicit_unit_labels(mention.get("reported_label"))
    }
    right_explicit = {
        label
        for mention in right
        for label in _normalized_explicit_unit_labels(mention.get("reported_label"))
    }
    if left_explicit and right_explicit and left_explicit != right_explicit:
        return False
    return any(
        _labels_semantically_contain(
            left_mention.get(left_field),
            right_mention.get(right_field),
        )
        for left_mention in left
        for right_mention in right
        for left_field, right_field in (
            ("reported_label", "reported_label"),
            ("study_name", "study_name"),
            ("reported_label", "study_name"),
            ("study_name", "reported_label"),
        )
    )


def _preferred_parent_mention(mentions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return min(
        mentions,
        key=lambda mention: (
            0 if _explicit_unit_labels(mention.get("reported_label")) else 1,
            0 if _mention_has_source_anchor(mention) else 1,
            len(str(mention.get("reported_label") or "")),
        ),
    )


def _adjudicate_candidate_boundaries(
    candidates: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
    index: PdfEvidenceIndex,
    llm_client: Any,
    *,
    pdf_path: Path,
    artifacts_dir: Optional[Path],
    timeout: Optional[float],
    force: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Audit the compact candidate ledger without sending the whole paper."""
    if len(candidates) < 2:
        return list(candidates), {
            "status": "not_needed",
            "llm_call_count": 0,
            "full_document_llm_calls": 0,
            "candidate_count_before": len(candidates),
            "candidate_count_after": len(candidates),
            "context_chars": 0,
            "actions": [],
            "rejected_candidates": [],
        }
    ledger, relation_ledger, prompt, context_strategy = _build_bounded_candidate_ledger(
        candidates,
        relations,
        index,
        pdf_name=pdf_path.name,
    )
    if len(prompt) > BOUNDARY_ADJUDICATION_MAX_CONTEXT_CHARS:
        print(
            "  Stage 1 boundary adjudication skipped: compact candidate ledger "
            f"still exceeds {BOUNDARY_ADJUDICATION_MAX_CONTEXT_CHARS} characters",
            flush=True,
        )
        return list(candidates), {
            "status": "skipped_context_limit",
            "prompt_version": BOUNDARY_ADJUDICATION_PROMPT_VERSION,
            "llm_call_count": 0,
            "full_document_llm_calls": 0,
            "candidate_count_before": len(candidates),
            "candidate_count_after": len(candidates),
            "context_chars": len(prompt),
            "context_char_limit": BOUNDARY_ADJUDICATION_MAX_CONTEXT_CHARS,
            "context_strategy": context_strategy,
            "actions": [],
            "rejected_candidates": [],
            "notes": "All candidates were preserved because the complete compact ledger exceeded the request budget.",
        }
    cache_path = artifacts_dir / "candidate_ledger.json" if artifacts_dir else None
    allowed_refs_by_candidate = {
        str(item.get("study_id") or ""): {
            str(ref) for ref in item.get("evidence_block_ids") or [] if str(ref)
        }
        for item in ledger
    }
    allowed_refs_by_relation = {
        str(item.get("relation_mention_id") or ""): {
            str(ref) for ref in item.get("evidence_block_ids") or [] if str(ref)
        }
        for item in relation_ledger
        if str(item.get("relation_mention_id") or "")
    }
    raw = cached_json_call(
        llm_client,
        prompt,
        cache_path=cache_path,
        prompt_version=BOUNDARY_ADJUDICATION_PROMPT_VERSION,
        timeout=timeout,
        max_tokens=BOUNDARY_ADJUDICATION_MAX_TOKENS,
        force=force,
        validator=lambda value: _validate_boundary_adjudication_payload(
            value,
            candidates=candidates,
            relation_ledger=relation_ledger,
            allowed_refs_by_candidate=allowed_refs_by_candidate,
            allowed_refs_by_relation=allowed_refs_by_relation,
        ),
    )
    raw = _augment_relation_problem_family_merges(
        raw,
        candidates=candidates,
        candidate_ledger=ledger,
        relation_ledger=relation_ledger,
    )
    _validate_boundary_adjudication_payload(
        raw,
        candidates=candidates,
        relation_ledger=relation_ledger,
        allowed_refs_by_candidate=allowed_refs_by_candidate,
        allowed_refs_by_relation=allowed_refs_by_relation,
    )
    reconciled, actions, rejected = _apply_boundary_adjudication(candidates, raw)
    print(
        "  Stage 1 boundary adjudication: "
        f"{len(candidates)} candidates -> {len(reconciled)} "
        f"({len(actions)} correction(s), {len(rejected)} rejected)",
        flush=True,
    )
    return reconciled, {
        "status": "ok",
        "prompt_version": BOUNDARY_ADJUDICATION_PROMPT_VERSION,
        "llm_call_count": 1,
        "full_document_llm_calls": 0,
        "candidate_count_before": len(candidates),
        "candidate_count_after": len(reconciled),
        "context_chars": len(prompt),
        "context_char_limit": BOUNDARY_ADJUDICATION_MAX_CONTEXT_CHARS,
        "context_strategy": context_strategy,
        "actions": actions,
        "rejected_candidates": rejected,
        "notes": str(raw.get("notes") or "").strip(),
    }


def _candidate_ledger_entry(
    candidate: Dict[str, Any],
    index: PdfEvidenceIndex,
    *,
    excerpt_limit: int = 4,
    excerpt_chars: int = 420,
    hint_chars: int = 280,
    ref_limit: Optional[int] = None,
) -> Dict[str, Any]:
    block_map = index.document.block_map()
    evidence_refs = [
        str(ref) for ref in candidate.get("evidence_block_ids") or [] if str(ref)
    ]
    if ref_limit is not None:
        evidence_refs = evidence_refs[: max(0, ref_limit)]
    excerpts: List[Dict[str, Any]] = []
    for ref in evidence_refs if excerpt_limit > 0 else []:
        block = block_map.get(str(ref))
        if block is None:
            continue
        excerpts.append(
            {
                "block_id": block.block_id,
                "page": block.page_start,
                "block_type": block.block_type,
                "section_path": [
                    _truncate_text(part, 80) for part in list(block.section_path)[:3]
                ],
                "text": _truncate_text(block.text, excerpt_chars),
            }
        )
        if len(excerpts) >= max(0, excerpt_limit):
            break
    return {
        "study_id": candidate.get("study_id"),
        "reported_label": _truncate_text(
            candidate.get("reported_label"),
            max(120, hint_chars),
        ),
        "study_name": _truncate_text(
            candidate.get("study_name"),
            max(120, hint_chars),
        ),
        "kind": candidate.get("kind"),
        "source_anchor": bool(candidate.get("source_anchor")),
        "participant_task_hint": _truncate_text(
            candidate.get("participant_task_hint"),
            hint_chars,
        ),
        "quantitative_target_hint": _truncate_text(
            candidate.get("quantitative_target_hint"),
            hint_chars,
        ),
        "evidence_block_ids": evidence_refs,
        "total_evidence_block_count": len(candidate.get("evidence_block_ids") or []),
        "evidence_excerpts": excerpts,
    }


def _relation_ledger_entries(
    relations: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    index: PdfEvidenceIndex,
    *,
    hint_chars: int,
    ref_limit: int,
    minimal: bool = False,
) -> List[Dict[str, Any]]:
    exact_aliases: Dict[str, set[str]] = {}
    identity_aliases: Dict[str, set[str]] = {}
    member_ref_aliases: Dict[str, set[str]] = {}
    for candidate in candidates:
        study_id = str(candidate.get("study_id") or "").strip()
        if not study_id:
            continue
        aliases = [
            candidate.get("study_id"),
            candidate.get("reported_label"),
            candidate.get("study_name"),
            *(candidate.get("aliases") or []),
        ]
        for component in candidate.get("component_mentions") or []:
            if isinstance(component, dict):
                aliases.extend(
                    [component.get("study_id"), component.get("reported_label")]
                )
        for alias in aliases:
            if not str(alias or "").strip():
                continue
            identity = _mention_identity_key({"reported_label": str(alias)})
            member_ref_aliases.setdefault(identity, set()).add(study_id)
            exact_aliases.setdefault(_relation_label_key(alias), set()).add(study_id)
            if identity:
                identity_aliases.setdefault(identity, set()).add(study_id)

    valid_refs = set(index.document.block_map())
    output: List[Dict[str, Any]] = []
    for relation in relations:
        member_refs = relation.get("member_refs")
        member_refs = member_refs if isinstance(member_refs, list) else []
        if member_refs:
            resolutions = [
                (
                    _member_ref_display(member_ref),
                    _resolve_relation_member_ref(
                        member_ref,
                        member_ref_aliases=member_ref_aliases,
                    ),
                )
                for member_ref in member_refs
                if isinstance(member_ref, dict)
            ]
        else:
            resolutions = [
                (
                    str(label),
                    _resolve_relation_label(
                        label,
                        exact_aliases=exact_aliases,
                        identity_aliases=identity_aliases,
                    ),
                )
                for label in relation.get("member_labels") or []
            ]
        item = {
                "relation_mention_id": relation.get("relation_mention_id"),
                "member_study_ids": list(
                    dict.fromkeys(study_id for _, study_id in resolutions if study_id)
                ),
                "members_are_distinct_empirical_units": bool(
                    relation.get("members_are_distinct_empirical_units")
                ),
                "relationship_kind": relation.get("relationship_kind"),
                "evidence_block_ids": [
                    str(ref)
                    for ref in relation.get("evidence_block_ids") or []
                    if str(ref) in valid_refs
                ][:ref_limit],
                "confidence": _confidence(relation.get("confidence")),
            }
        if not minimal:
            item["member_labels"] = [
                _truncate_text(label, max(40, hint_chars)) for label, _ in resolutions
            ]
            item["unresolved_member_labels"] = [
                _truncate_text(label, max(40, hint_chars))
                for label, study_id in resolutions
                if not study_id
            ]
            item["comparison_target"] = _truncate_text(
                relation.get("comparison_target"),
                hint_chars,
            )
            item["evidence_summary"] = _truncate_text(
                relation.get("evidence_summary"),
                hint_chars,
            )
        output.append(item)
    return output


def _minimal_candidate_ledger_entry(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "study_id": candidate.get("study_id"),
        "reported_label": _truncate_text(candidate.get("reported_label"), 80),
        "source_anchor": bool(candidate.get("source_anchor")),
        "evidence_block_ids": list(candidate.get("evidence_block_ids") or [])[:1],
    }


def _build_bounded_candidate_ledger(
    candidates: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
    index: PdfEvidenceIndex,
    *,
    pdf_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
    """Keep every candidate identity while bounding the one global LLM request."""
    strategies = (
        {
            "name": "rich",
            "excerpt_limit": 4,
            "excerpt_chars": 420,
            "hint_chars": 280,
            "ref_limit": 12,
        },
        {
            "name": "compact",
            "excerpt_limit": 1,
            "excerpt_chars": 220,
            "hint_chars": 160,
            "ref_limit": 4,
        },
        {
            "name": "identity_only",
            "excerpt_limit": 0,
            "excerpt_chars": 0,
            "hint_chars": 100,
            "ref_limit": 2,
        },
        {
            "name": "minimal",
            "excerpt_limit": 0,
            "excerpt_chars": 0,
            "hint_chars": 0,
            "ref_limit": 1,
        },
    )
    last_ledger: List[Dict[str, Any]] = []
    last_relation_ledger: List[Dict[str, Any]] = []
    last_prompt = ""
    last_strategy = "identity_only"
    for strategy in strategies:
        if strategy["name"] == "minimal":
            ledger = [_minimal_candidate_ledger_entry(candidate) for candidate in candidates]
        else:
            ledger = [
                _candidate_ledger_entry(
                    candidate,
                    index,
                    excerpt_limit=int(strategy["excerpt_limit"]),
                    excerpt_chars=int(strategy["excerpt_chars"]),
                    hint_chars=int(strategy["hint_chars"]),
                    ref_limit=int(strategy["ref_limit"]),
                )
                for candidate in candidates
            ]
        relation_ledger = _relation_ledger_entries(
            relations,
            candidates,
            index,
            hint_chars=int(strategy["hint_chars"]),
            ref_limit=int(strategy["ref_limit"]),
            minimal=strategy["name"] == "minimal",
        )
        prompt = _boundary_adjudication_prompt(
            ledger,
            relation_ledger,
            pdf_name=pdf_name,
        )
        last_ledger = ledger
        last_relation_ledger = relation_ledger
        last_prompt = prompt
        last_strategy = str(strategy["name"])
        if len(prompt) <= BOUNDARY_ADJUDICATION_MAX_CONTEXT_CHARS:
            break
    return last_ledger, last_relation_ledger, last_prompt, last_strategy


def _boundary_adjudication_prompt(
    ledger: Sequence[Dict[str, Any]],
    relation_ledger: Sequence[Dict[str, Any]],
    *,
    pdf_name: str,
) -> str:
    return f"""You are the boundary-adjudication pass for Stage 1 of
HumanStudy-Bench. The paper is a general social-science paper; topic and
discipline are unrestricted.

Paper: {pdf_name}

The discovery layer produced compact high-recall candidate and relationship
ledgers below. They contain short source excerpts and block IDs, not the full
paper. Report only boundary corrections directly supported by these ledgers.

Rules:
- The unit of reconciliation is a SIMULATION TASK FAMILY, not merely a shared
  participant collection. Merge entries when they are the same task family
  described once in Methods/Procedure and again in Results, Discussion, a table,
  or a conclusion. Section numbers such as 2.2.2 and 3.2 are document locations,
  not study IDs.
- A repeated formal parent label is decisive identity evidence. Merge duplicate
  candidates for the same Study/Experiment even when one candidate describes a
  stimulus/choice step and another describes a measure or outcome step; those
  are components needed by one environment, not independent empirical units.
- Merge repeated or parameterized items when participants receive the same task
  template and response format and the paper treats them as one coordinated
  problem set. Separate participant groups for item versions do not by
  themselves create separate units. Conversely, two sequence-frequency items
  answered by the same respondents using the same judgment procedure are one
  task family even if one result is described as a randomness comparison.
- Do NOT merge substantively different unlabeled task families or different
  formal parents merely because they occur in the same questionnaire, use the
  same respondents, or share a sample-description paragraph.
- A questionnaire-wide sample/procedure description can be shared context for
  several task units. Put it in shared_context_links: remove the context-only
  candidate as a unit and attach its evidence to every target, without merging
  the target units. Never use shared_context_links for a candidate that has its
  own participant response task and outcome (for example, a section where
  participants construct sampling distributions); that remains a task unit.
- Merge a condition, arm, form, order, subgroup, or result table into its one
  task-family parent. Do not merge distinct top-level Study 1/Study 2 or
  Experiment I/Experiment II labels.
- Numbered Problems are often questionnaire items rather than independent
  studies. Distinct Problem labels may be merged only when the RELATIONSHIP
  LEDGER explicitly links the same members and shows that their joint contrast,
  coordinated problem set, or logically equivalent versions define one
  simulation target. Cite the supporting relation ID. A topical relationship
  alone is insufficient, and independently reported Studies or Experiments stay
  separate even when compared.
- Keep genuinely distinct pilots, pretests, validation tasks, surveys, waves,
  and experiments separate when their participant-facing task or collection is
  independently reported.
- Reject only entries that are method/discussion/result fragments rather than
  empirical collections. A source-labeled top-level unit must not be rejected
  merely because it lacks a quantitative result or is not simulatable.
- Do not merge merely because two units study the same construct or share a
  population. Do not decide simulation eligibility or rewrite study fields.
- Every correction must cite block IDs already listed on its member candidates.
  When uncertain, emit no correction and leave the verifier to flag it.

CANDIDATE LEDGER:
{json.dumps(list(ledger), ensure_ascii=False, separators=(",", ":"))}

RELATIONSHIP LEDGER:
{json.dumps(list(relation_ledger), ensure_ascii=False, separators=(",", ":"))}

Return only this JSON object:
{{
  "merge_groups": [
    {{
      "member_study_ids": ["existing id", "another existing id"],
      "canonical_study_id": "member id that best names the task family or formal parent",
      "supporting_relation_ids": ["required relation id when merging distinct Problem labels"],
      "reason": "why these entries form one simulation task family",
      "evidence_block_ids": ["listed block id"]
    }}
  ],
  "shared_context_links": [
    {{
      "context_study_id": "method/sample-only candidate id",
      "target_study_ids": ["task candidate id", "another task candidate id"],
      "reason": "why the context applies to these tasks without merging them",
      "evidence_block_ids": ["context candidate block id"]
    }}
  ],
  "reject_candidates": [
    {{
      "study_id": "existing weak candidate id",
      "reason": "why this is a non-unit fragment",
      "evidence_block_ids": ["listed block id"]
    }}
  ],
  "notes": "short adjudication note"
}}"""


def _relation_supports_problem_merge(
    members: Sequence[str],
    *,
    explicit_labels: set[str],
    supporting_relation_ids: Sequence[str],
    relations_by_id: Dict[str, Dict[str, Any]],
) -> bool:
    if len(explicit_labels) < 2 or not all(
        label.lower().startswith("problem ") for label in explicit_labels
    ):
        return False
    member_set = set(members)
    graph: Dict[str, set[str]] = {member: set() for member in member_set}
    for relation_id in supporting_relation_ids:
        relation = relations_by_id.get(relation_id)
        if relation is None:
            return False
        relation_members = {
            str(value).strip()
            for value in relation.get("member_study_ids") or []
            if str(value).strip()
        }
        if (
            len(relation_members) < 2
            or not relation_members <= member_set
            or str(relation.get("relationship_kind") or "").strip().lower()
            not in {"paired_contrast", "multi_unit_comparison"}
            or _confidence(relation.get("confidence")) < 0.7
        ):
            return False
        for member in relation_members:
            graph[member].update(relation_members - {member})
    if not member_set:
        return False
    pending = [next(iter(member_set))]
    reached: set[str] = set()
    while pending:
        member = pending.pop()
        if member in reached:
            continue
        reached.add(member)
        pending.extend(graph.get(member, set()) - reached)
    return reached == member_set


def _augment_relation_problem_family_merges(
    payload: Dict[str, Any],
    *,
    candidates: Sequence[Dict[str, Any]],
    candidate_ledger: Sequence[Dict[str, Any]],
    relation_ledger: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compile explicit high-confidence Problem relations into task families."""
    output = deepcopy(payload)
    candidate_order = {
        str(candidate.get("study_id") or ""): position
        for position, candidate in enumerate(candidates)
    }
    candidates_by_id = {
        str(candidate.get("study_id") or ""): candidate for candidate in candidates
    }
    candidate_refs = {
        str(candidate.get("study_id") or ""): list(
            candidate.get("evidence_block_ids") or []
        )
        for candidate in candidate_ledger
    }
    eligible_relations: List[Dict[str, Any]] = []
    graph: Dict[str, set[str]] = {}
    for relation in relation_ledger:
        members = [
            str(value).strip()
            for value in relation.get("member_study_ids") or []
            if str(value).strip()
        ]
        if (
            len(set(members)) < 2
            or relation.get("unresolved_member_labels")
            or not relation.get("members_are_distinct_empirical_units")
            or str(relation.get("relationship_kind") or "").strip().lower()
            not in {"paired_contrast", "multi_unit_comparison"}
            or _confidence(relation.get("confidence")) < 0.7
            or any(member not in candidates_by_id for member in members)
            or any(
                not (
                    len(
                        labels := _explicit_unit_labels(
                            candidates_by_id[member].get("reported_label")
                        )
                    )
                    == 1
                    and labels[0].lower().startswith("problem ")
                )
                for member in members
            )
        ):
            continue
        eligible_relations.append(relation)
        for member in members:
            graph.setdefault(member, set()).update(set(members) - {member})

    components: List[List[str]] = []
    unseen = set(graph)
    while unseen:
        seed = min(unseen, key=lambda value: candidate_order.get(value, 10**9))
        pending = [seed]
        component: set[str] = set()
        while pending:
            member = pending.pop()
            if member in component:
                continue
            component.add(member)
            pending.extend(graph.get(member, set()) - component)
        unseen -= component
        if len(component) > 1:
            components.append(
                sorted(component, key=lambda value: candidate_order.get(value, 10**9))
            )

    merge_groups = [
        dict(group)
        for group in output.get("merge_groups") or []
        if isinstance(group, dict)
    ]
    for members in components:
        member_set = set(members)
        supporting = [
            relation
            for relation in eligible_relations
            if {
                str(value)
                for value in relation.get("member_study_ids") or []
                if str(value)
            }
            <= member_set
        ]
        supporting_ids = [
            str(relation.get("relation_mention_id") or "")
            for relation in supporting
            if str(relation.get("relation_mention_id") or "")
        ]
        evidence_refs = list(
            dict.fromkeys(
                [
                    *(
                        ref
                        for member in members
                        for ref in candidate_refs.get(member, [])
                    ),
                    *(
                        ref
                        for relation in supporting
                        for ref in relation.get("evidence_block_ids") or []
                    ),
                ]
            )
        )
        retained_groups: List[Dict[str, Any]] = []
        for group in merge_groups:
            existing_members = {
                str(value) for value in group.get("member_study_ids") or [] if str(value)
            }
            if existing_members & member_set and existing_members <= member_set:
                continue
            retained_groups.append(group)
        merge_groups = retained_groups
        merge_groups.append(
            {
                "member_study_ids": members,
                "canonical_study_id": members[0],
                "supporting_relation_ids": supporting_ids,
                "reason": (
                    "Source-explicit Problem relations jointly define one coordinated "
                    "simulation task family."
                ),
                "evidence_block_ids": evidence_refs,
                "merge_source": "relation_graph",
            }
        )
    output["merge_groups"] = merge_groups
    return output


def _validate_boundary_adjudication_payload(
    payload: Dict[str, Any],
    *,
    candidates: Sequence[Dict[str, Any]],
    relation_ledger: Sequence[Dict[str, Any]] = (),
    allowed_refs_by_candidate: Optional[Dict[str, set[str]]] = None,
    allowed_refs_by_relation: Optional[Dict[str, set[str]]] = None,
) -> None:
    candidates_by_id = {
        str(candidate.get("study_id") or ""): candidate for candidate in candidates
    }
    known_ids = set(candidates_by_id)
    candidate_refs = {
        str(candidate.get("study_id") or ""): {
            str(ref)
            for ref in candidate.get("evidence_block_ids") or []
            if str(ref)
        }
        for candidate in candidates
    }
    if allowed_refs_by_candidate is not None:
        candidate_refs = {
            study_id: candidate_refs.get(study_id, set()) & set(refs)
            for study_id, refs in allowed_refs_by_candidate.items()
        }
    valid_refs = {ref for refs in candidate_refs.values() for ref in refs}
    relations_by_id = {
        str(relation.get("relation_mention_id") or "").strip(): relation
        for relation in relation_ledger
        if str(relation.get("relation_mention_id") or "").strip()
    }
    relation_refs = {
        relation_id: {
            str(ref)
            for ref in relation.get("evidence_block_ids") or []
            if str(ref)
        }
        for relation_id, relation in relations_by_id.items()
    }
    if allowed_refs_by_relation is not None:
        relation_refs = {
            relation_id: relation_refs.get(relation_id, set()) & set(refs)
            for relation_id, refs in allowed_refs_by_relation.items()
        }
    valid_refs.update(ref for refs in relation_refs.values() for ref in refs)
    used_merge_ids: set[str] = set()
    merge_groups = payload.get("merge_groups")
    if not isinstance(merge_groups, list):
        raise ValueError("merge_groups must be an array")
    for position, group in enumerate(merge_groups):
        if not isinstance(group, dict):
            raise ValueError(f"merge_groups[{position}] must be an object")
        members = [
            str(value).strip()
            for value in group.get("member_study_ids") or []
            if str(value).strip()
        ]
        if len(set(members)) < 2 or not set(members) <= known_ids:
            raise ValueError(f"merge_groups[{position}] has invalid member_study_ids")
        if used_merge_ids & set(members):
            raise ValueError(f"merge_groups[{position}] overlaps another merge group")
        canonical_study_id = str(group.get("canonical_study_id") or "").strip()
        if canonical_study_id and canonical_study_id not in members:
            raise ValueError(
                f"merge_groups[{position}] canonical_study_id must be a member"
            )
        supporting_relation_ids = [
            str(value).strip()
            for value in group.get("supporting_relation_ids") or []
            if str(value).strip()
        ]
        if not set(supporting_relation_ids) <= set(relations_by_id):
            raise ValueError(
                f"merge_groups[{position}] has unknown supporting_relation_ids"
            )
        explicit_labels = {
            label
            for member in members
            for label in _normalized_explicit_unit_labels(
                candidates_by_id[member].get("reported_label")
            )
        }
        relation_supported_problem_merge = _relation_supports_problem_merge(
            members,
            explicit_labels=explicit_labels,
            supporting_relation_ids=supporting_relation_ids,
            relations_by_id=relations_by_id,
        )
        if len(explicit_labels) > 1 and not relation_supported_problem_merge:
            raise ValueError(
                f"merge_groups[{position}] merges distinct formal source labels"
            )
        formal_members = [
            member
            for member in members
            if _explicit_unit_labels(candidates_by_id[member].get("reported_label"))
        ]
        if canonical_study_id and len(formal_members) == 1 and canonical_study_id != formal_members[0]:
            raise ValueError(
                f"merge_groups[{position}] canonical_study_id must preserve the formal parent"
            )
        if not relation_supported_problem_merge and not _merge_group_is_connected(
            [candidates_by_id[member] for member in members]
        ):
            raise ValueError(
                f"merge_groups[{position}] is not a connected task family; use shared_context_links for common sample/procedure evidence"
            )
        _validate_adjudication_refs(
            group,
            path=f"merge_groups[{position}]",
            valid_refs=valid_refs,
            member_refs={
                ref
                for member in members
                for ref in candidate_refs.get(member, set())
            }
            | {
                ref
                for relation_id in supporting_relation_ids
                for ref in relation_refs.get(relation_id, set())
            },
        )
        used_merge_ids.update(members)

    shared_context_links = payload.get("shared_context_links", [])
    if not isinstance(shared_context_links, list):
        raise ValueError("shared_context_links must be an array")
    shared_context_ids: set[str] = set()
    shared_target_ids: set[str] = set()
    for position, link in enumerate(shared_context_links):
        if not isinstance(link, dict):
            raise ValueError(f"shared_context_links[{position}] must be an object")
        context_study_id = str(link.get("context_study_id") or "").strip()
        target_study_ids = [
            str(value).strip()
            for value in link.get("target_study_ids") or []
            if str(value).strip()
        ]
        if (
            context_study_id not in known_ids
            or context_study_id in shared_context_ids
            or context_study_id in used_merge_ids
            or not target_study_ids
            or not set(target_study_ids) <= known_ids
            or context_study_id in target_study_ids
        ):
            raise ValueError(f"shared_context_links[{position}] has invalid study ids")
        if _explicit_unit_labels(
            candidates_by_id[context_study_id].get("reported_label")
        ):
            raise ValueError(
                f"shared_context_links[{position}] cannot demote a formal source-labeled unit"
            )
        if _candidate_has_response_target(candidates_by_id[context_study_id]):
            raise ValueError(
                f"shared_context_links[{position}] demotes a candidate with its own response target"
            )
        _validate_adjudication_refs(
            link,
            path=f"shared_context_links[{position}]",
            valid_refs=valid_refs,
            member_refs=candidate_refs.get(context_study_id, set()),
        )
        shared_context_ids.add(context_study_id)
        shared_target_ids.update(target_study_ids)

    rejected = payload.get("reject_candidates")
    if not isinstance(rejected, list):
        raise ValueError("reject_candidates must be an array")
    seen_rejected: set[str] = set()
    for position, item in enumerate(rejected):
        if not isinstance(item, dict):
            raise ValueError(f"reject_candidates[{position}] must be an object")
        study_id = str(item.get("study_id") or "").strip()
        if (
            study_id not in known_ids
            or study_id in seen_rejected
            or study_id in used_merge_ids
            or study_id in shared_context_ids
            or study_id in shared_target_ids
        ):
            raise ValueError(f"reject_candidates[{position}] has invalid study_id")
        if _explicit_unit_labels(candidates_by_id[study_id].get("reported_label")):
            raise ValueError(
                f"reject_candidates[{position}] cannot reject a formal source-labeled candidate"
            )
        _validate_adjudication_refs(
            item,
            path=f"reject_candidates[{position}]",
            valid_refs=valid_refs,
            member_refs=candidate_refs.get(study_id, set()),
        )
        seen_rejected.add(study_id)


def _validate_adjudication_refs(
    item: Dict[str, Any],
    *,
    path: str,
    valid_refs: set[str],
    member_refs: set[str],
) -> None:
    refs = [str(ref) for ref in item.get("evidence_block_ids") or [] if str(ref)]
    if not refs:
        raise ValueError(f"{path} has no evidence_block_ids")
    if not set(refs) <= valid_refs or not set(refs) <= member_refs:
        raise ValueError(f"{path} cites evidence outside its candidate members")


def _merge_group_is_connected(candidates: Sequence[Dict[str, Any]]) -> bool:
    if len(candidates) < 2:
        return False
    if sum(
        bool(_explicit_unit_labels(candidate.get("reported_label")))
        for candidate in candidates
    ) == 1:
        return True
    reached = {0}
    frontier = [0]
    while frontier:
        left_index = frontier.pop()
        for right_index, right in enumerate(candidates):
            if right_index in reached:
                continue
            if _candidate_merge_compatible(candidates[left_index], right):
                reached.add(right_index)
                frontier.append(right_index)
    return len(reached) == len(candidates)


def _candidate_merge_compatible(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> bool:
    left_refs = {str(ref) for ref in left.get("evidence_block_ids") or [] if str(ref)}
    right_refs = {str(ref) for ref in right.get("evidence_block_ids") or [] if str(ref)}
    if left_refs & right_refs:
        return True
    left_values = (
        left.get("reported_label"),
        left.get("study_name"),
        left.get("participant_task_hint"),
    )
    right_values = (
        right.get("reported_label"),
        right.get("study_name"),
        right.get("participant_task_hint"),
    )
    if any(
        _labels_semantically_contain(left_value, right_value)
        for left_value in left_values
        for right_value in right_values
    ):
        return True
    left_words = _candidate_task_signature(left)
    right_words = _candidate_task_signature(right)
    shared = left_words & right_words
    return (
        len(shared) >= 3
        and len(shared) / max(1, min(len(left_words), len(right_words))) >= 0.25
    )


_MERGE_SIGNATURE_STOP_WORDS = {
    "analysis",
    "condition",
    "conditions",
    "current",
    "experiment",
    "experiments",
    "measure",
    "measures",
    "paper",
    "participant",
    "participants",
    "present",
    "reported",
    "response",
    "responses",
    "result",
    "results",
    "study",
    "subject",
    "subjects",
    "task",
    "using",
}


def _candidate_task_signature(candidate: Dict[str, Any]) -> set[str]:
    text = " ".join(
        str(candidate.get(field) or "")
        for field in (
            "reported_label",
            "study_name",
            "participant_task_hint",
            "quantitative_target_hint",
        )
    )
    return _label_words(text) - _MERGE_SIGNATURE_STOP_WORDS


_RESPONSE_TARGET_RE = re.compile(
    r"\b(medians?|means?|variances?|proportions?|percent(?:age)?s?|odds|probabilit(?:y|ies)|"
    r"choices?|chose|chosen|judg(?:e|ed|ment|ments)|ratings?|responses?|estimat(?:e|ed|es|ion)|"
    r"distributions?|frequenc(?:y|ies))\b|%",
    re.IGNORECASE,
)


def _candidate_has_response_target(candidate: Dict[str, Any]) -> bool:
    task = str(candidate.get("participant_task_hint") or "").strip()
    target = str(candidate.get("quantitative_target_hint") or "").strip()
    return bool(task and target and _RESPONSE_TARGET_RE.search(target))


def _apply_boundary_adjudication(
    candidates: Sequence[Dict[str, Any]],
    payload: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates_by_id = {
        str(candidate.get("study_id") or ""): deepcopy(candidate)
        for candidate in candidates
    }
    merge_by_member: Dict[str, List[str]] = {}
    canonical_by_member: Dict[str, str] = {}
    group_by_member: Dict[str, Dict[str, Any]] = {}
    actions: List[Dict[str, Any]] = []
    for group in payload.get("merge_groups") or []:
        members = [str(value) for value in group.get("member_study_ids") or []]
        canonical_study_id = str(group.get("canonical_study_id") or "").strip()
        for member in members:
            merge_by_member[member] = members
            group_by_member[member] = group
            if canonical_study_id:
                canonical_by_member[member] = canonical_study_id
        actions.append(
            {
                "action": (
                    "relation_graph_merge_candidates"
                    if group.get("merge_source") == "relation_graph"
                    else "llm_merge_candidates"
                ),
                "member_study_ids": members,
                "canonical_study_id": canonical_study_id or None,
                "supporting_relation_ids": list(
                    group.get("supporting_relation_ids") or []
                ),
                "reason": str(group.get("reason") or "").strip(),
                "evidence_block_ids": list(group.get("evidence_block_ids") or []),
            }
        )
    shared_context_ids: set[str] = set()
    rejected: List[Dict[str, Any]] = []
    for link in payload.get("shared_context_links") or []:
        context_study_id = str(link.get("context_study_id") or "").strip()
        context_candidate = candidates_by_id[context_study_id]
        target_study_ids = [str(value) for value in link.get("target_study_ids") or []]
        for target_study_id in target_study_ids:
            _attach_shared_context(
                candidates_by_id[target_study_id],
                context_candidate,
                reason=str(link.get("reason") or "").strip(),
            )
        shared_context_ids.add(context_study_id)
        actions.append(
            {
                "action": "llm_attach_shared_context",
                "context_study_id": context_study_id,
                "target_study_ids": target_study_ids,
                "reason": str(link.get("reason") or "").strip(),
                "evidence_block_ids": list(link.get("evidence_block_ids") or []),
            }
        )
        rejected.append(
            {
                "study_id": context_study_id,
                "experiment_id": context_candidate.get("reported_label"),
                "study_name": context_candidate.get("study_name"),
                "unit_provenance": "current_paper",
                "is_distinct_empirical_unit": False,
                "reason": "Shared sample/procedure context was attached to task units without becoming a separate simulation unit.",
                "evidence_refs": list(context_candidate.get("evidence_block_ids") or []),
                "rejection_stage": "shared_context_attachment",
            }
        )
    rejected_ids = {
        str(item.get("study_id") or "").strip()
        for item in payload.get("reject_candidates") or []
    }
    for item in payload.get("reject_candidates") or []:
        study_id = str(item.get("study_id") or "").strip()
        candidate = candidates_by_id[study_id]
        rejected.append(
            {
                "study_id": study_id,
                "experiment_id": candidate.get("reported_label"),
                "study_name": candidate.get("study_name"),
                "unit_provenance": "unclear",
                "is_distinct_empirical_unit": False,
                "reason": str(item.get("reason") or "").strip(),
                "evidence_refs": list(item.get("evidence_block_ids") or []),
                "rejection_stage": "boundary_adjudication",
            }
        )
        actions.append(
            {
                "action": "llm_reject_candidate",
                "study_id": study_id,
                "reason": str(item.get("reason") or "").strip(),
                "evidence_block_ids": list(item.get("evidence_block_ids") or []),
            }
        )

    output: List[Dict[str, Any]] = []
    emitted: set[str] = set()
    for candidate in candidates:
        study_id = str(candidate.get("study_id") or "")
        if study_id in rejected_ids or study_id in shared_context_ids or study_id in emitted:
            continue
        members = merge_by_member.get(study_id, [study_id])
        merged = _merge_candidate_records(
            [candidates_by_id[member] for member in members],
            canonical_study_id=canonical_by_member.get(study_id),
        )
        group = group_by_member.get(study_id)
        if group is not None:
            merged["evidence_block_ids"] = list(
                dict.fromkeys(
                    [
                        *(merged.get("evidence_block_ids") or []),
                        *(group.get("evidence_block_ids") or []),
                    ]
                )
            )
            merged["source_task_family_relation_ids"] = list(
                dict.fromkeys(group.get("supporting_relation_ids") or [])
            )
            reason = str(group.get("reason") or "").strip()
            if reason:
                merged["boundary_notes"] = reason
            problem_labels = [
                labels[0]
                for member in members
                for labels in [
                    _explicit_unit_labels(
                        candidates_by_id[member].get("reported_label")
                    )
                ]
                if len(labels) == 1 and labels[0].lower().startswith("problem ")
            ]
            if len(problem_labels) == len(members) and len(problem_labels) > 1:
                merged["reported_label"] = " + ".join(problem_labels)
                merged["study_name"] = "Coordinated " + " / ".join(problem_labels)
        output.append(merged)
        emitted.update(members)
    return output, actions, rejected


def _merge_candidate_records(
    candidates: Sequence[Dict[str, Any]],
    *,
    canonical_study_id: Optional[str] = None,
) -> Dict[str, Any]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            0 if _explicit_unit_labels(candidate.get("reported_label")) else 1,
            0
            if canonical_study_id
            and str(candidate.get("study_id") or "") == canonical_study_id
            else 1,
            0 if _looks_like_collection_or_task_parent(candidate) else 1,
            0 if candidate.get("source_anchor") else 1,
            str(candidate.get("reported_label") or "").lower(),
        ),
    )
    representative = deepcopy(ordered[0])
    if len(ordered) == 1:
        return representative
    representative["aliases"] = list(
        dict.fromkeys(
            [
                *(str(value) for value in representative.get("aliases") or [] if str(value)),
                *(
                    str(value)
                    for candidate in ordered[1:]
                    for value in (
                        candidate.get("study_id"),
                        candidate.get("reported_label"),
                        candidate.get("study_name"),
                        *(candidate.get("aliases") or []),
                    )
                    if str(value or "").strip()
                ),
            ]
        )
    )
    representative["source_mention_ids"] = list(
        dict.fromkeys(
            str(value)
            for candidate in ordered
            for value in candidate.get("source_mention_ids") or []
            if str(value)
        )
    )
    representative["evidence_block_ids"] = list(
        dict.fromkeys(
            str(value)
            for candidate in ordered
            for value in candidate.get("evidence_block_ids") or []
            if str(value)
        )
    )
    representative["material_variants"] = _merge_discovery_variants(
        *(candidate.get("material_variants") or [] for candidate in ordered)
    )
    representative["participant_task_hint"] = _merge_candidate_hints(
        candidate.get("participant_task_hint") for candidate in ordered
    )
    representative["quantitative_target_hint"] = _merge_candidate_hints(
        candidate.get("quantitative_target_hint") for candidate in ordered
    )
    representative["source_anchor"] = any(
        bool(candidate.get("source_anchor")) for candidate in ordered
    )
    representative["boundary_confidence"] = max(
        (_confidence(candidate.get("boundary_confidence")) for candidate in ordered),
        default=0.0,
    )
    representative["boundary_notes"] = (
        "Boundary adjudication merged complementary method/result or subordinate "
        "descriptions of the same empirical collection."
    )
    representative["component_mentions"] = [
        {
            "study_id": candidate.get("study_id"),
            "reported_label": candidate.get("reported_label"),
            "study_name": candidate.get("study_name"),
            "participant_task_hint": candidate.get("participant_task_hint"),
            "quantitative_target_hint": candidate.get("quantitative_target_hint"),
            "evidence_block_ids": list(candidate.get("evidence_block_ids") or []),
        }
        for candidate in ordered
    ]
    return representative


_COLLECTION_OR_PARENT_RE = re.compile(
    r"\b(questionnaire data|data collection|participant collection|main (?:study|survey|experiment)|"
    r"survey of|sample of participants|respondents? (?:completed|answered)|procedure)\b",
    re.IGNORECASE,
)


def _looks_like_collection_or_task_parent(candidate: Dict[str, Any]) -> bool:
    return bool(
        _COLLECTION_OR_PARENT_RE.search(str(candidate.get("reported_label") or ""))
        or _COLLECTION_OR_PARENT_RE.search(str(candidate.get("study_name") or ""))
    )


def _attach_shared_context(
    target: Dict[str, Any],
    context: Dict[str, Any],
    *,
    reason: str,
) -> None:
    target["evidence_block_ids"] = list(
        dict.fromkeys(
            [
                *(str(ref) for ref in target.get("evidence_block_ids") or [] if str(ref)),
                *(str(ref) for ref in context.get("evidence_block_ids") or [] if str(ref)),
            ]
        )
    )
    target["source_mention_ids"] = list(
        dict.fromkeys(
            [
                *(str(value) for value in target.get("source_mention_ids") or [] if str(value)),
                *(str(value) for value in context.get("source_mention_ids") or [] if str(value)),
            ]
        )
    )
    shared_contexts = target.get("shared_contexts")
    shared_contexts = list(shared_contexts) if isinstance(shared_contexts, list) else []
    shared_contexts.append(
        {
            "study_id": context.get("study_id"),
            "reported_label": context.get("reported_label"),
            "study_name": context.get("study_name"),
            "reason": reason,
            "evidence_block_ids": list(context.get("evidence_block_ids") or []),
        }
    )
    target["shared_contexts"] = shared_contexts


def _merge_candidate_hints(values: Iterable[Any]) -> Optional[str]:
    unique = list(
        dict.fromkeys(str(value).strip() for value in values if str(value or "").strip())
    )
    if not unique:
        return None
    return _truncate_text(" | ".join(unique), 900)


def _truncate_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _normalize_discovery_variants(
    values: Any,
    *,
    valid_refs: set[str],
) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for value in values if isinstance(values, list) else []:
        if isinstance(value, str):
            label = value.strip()
            role = "other"
            refs: List[str] = []
        elif isinstance(value, dict):
            label = str(value.get("label") or "").strip()
            role = str(value.get("role") or "other").strip().lower()
            refs = _valid_refs(value.get("evidence_block_ids"), valid_refs)
        else:
            continue
        if not label:
            continue
        if role not in {"condition", "stimulus", "form", "order", "item_set", "other"}:
            role = "other"
        variants.append(
            {
                "label": label,
                "role": role,
                "evidence_block_ids": refs,
            }
        )
    return _merge_discovery_variants(variants)


def _merge_discovery_variants(*groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for group in groups:
        for value in group:
            label = str(value.get("label") or "").strip()
            if not label:
                continue
            key = _relation_label_key(label)
            if key not in merged:
                merged[key] = {
                    "label": label,
                    "role": str(value.get("role") or "other").strip().lower(),
                    "evidence_block_ids": [],
                }
                order.append(key)
            merged[key]["evidence_block_ids"] = list(
                dict.fromkeys(
                    [
                        *merged[key]["evidence_block_ids"],
                        *(str(ref) for ref in value.get("evidence_block_ids") or [] if str(ref)),
                    ]
                )
            )
    return [merged[key] for key in order]


def _candidate_from_mentions(
    raw: Dict[str, Any],
    mentions: Sequence[Dict[str, Any]],
    used_study_ids: set[str],
) -> Dict[str, Any]:
    first = mentions[0]
    label = str(raw.get("reported_label") or first.get("reported_label") or "Study").strip()
    requested_id = raw.get("study_id") or label
    study_id = canonical_sub_study_id(requested_id)
    base = study_id
    suffix = 2
    while study_id in used_study_ids:
        study_id = f"{base}_{suffix}"
        suffix += 1
    used_study_ids.add(study_id)
    refs = list(
        dict.fromkeys(
            str(ref)
            for mention in mentions
            for ref in [
                *(mention.get("evidence_block_ids") or []),
                *(
                    ref
                    for variant in mention.get("material_variants") or []
                    if isinstance(variant, dict)
                    for ref in variant.get("evidence_block_ids") or []
                ),
            ]
            if str(ref)
        )
    )
    aliases = list(
        dict.fromkeys(
            [
                str(value).strip()
                for value in raw.get("aliases") or []
                if str(value).strip()
            ]
            + [
                str(mention.get("reported_label") or "").strip()
                for mention in mentions
                if str(mention.get("reported_label") or "").strip() != label
            ]
            + [
                str(mention.get("raw_reported_label") or "").strip()
                for mention in mentions
                if str(mention.get("raw_reported_label") or "").strip() != label
            ]
        )
    )
    material_variants = _merge_discovery_variants(
        *(mention.get("material_variants") or [] for mention in mentions)
    )
    return {
        "study_id": study_id,
        "reported_label": label,
        "study_name": str(raw.get("study_name") or first.get("study_name") or label).strip(),
        "kind": str(raw.get("kind") or first.get("kind") or "other").strip().lower(),
        "material_variants": material_variants,
        "aliases": aliases,
        "source_mention_ids": [str(mention["mention_id"]) for mention in mentions],
        "evidence_block_ids": refs,
        "participant_task_hint": _first_text(
            mention.get("participant_task_hint") for mention in mentions
        ),
        "quantitative_target_hint": _first_text(
            mention.get("quantitative_target_hint") for mention in mentions
        ),
        "boundary_notes": str(raw.get("boundary_notes") or "").strip(),
        "boundary_confidence": _confidence(
            raw.get("boundary_confidence"),
            fallback=max(
                (_confidence(mention.get("boundary_confidence")) for mention in mentions),
                default=0.0,
            ),
        ),
        "source_anchor": bool(raw.get("source_anchor")),
    }


def _reconciled_study_name(mentions: Sequence[Dict[str, Any]]) -> str:
    names = list(
        dict.fromkeys(
            str(mention.get("study_name") or "").strip()
            for mention in mentions
            if str(mention.get("study_name") or "").strip()
        )
    )
    label = str(mentions[0].get("reported_label") or "Study") if mentions else "Study"
    return names[0] if len(names) == 1 else label


def _extract_all_studies(
    candidates: Sequence[Dict[str, Any]],
    index: PdfEvidenceIndex,
    llm_client: Any,
    *,
    pdf_path: Path,
    regeneration_instructions: Optional[Dict[str, Any]],
    artifacts_dir: Optional[Path],
    timeout: Optional[float],
    workers: int,
    force: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Dict[str, Any]] = {}
    contexts: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []

    def run(candidate: Dict[str, Any]) -> Tuple[Dict[str, Any], EvidenceContext]:
        context = index.context_for_study(
            candidate,
            gaps=["source_evidence"],
            allow_full_document=False,
            anchor_refs=candidate.get("evidence_block_ids") or [],
            anchor_radius=1,
            use_facet_retrieval=not bool(candidate.get("evidence_block_ids")),
            max_chars=STUDY_CONTEXT_MAX_CHARS,
        )
        prompt = _study_extraction_prompt(
            candidate,
            context,
            pdf_name=pdf_path.name,
            regeneration_instructions=_feedback_for_study(
                regeneration_instructions,
                candidate,
            ),
        )
        cache_path = artifacts_dir / f"{candidate['study_id']}.json" if artifacts_dir else None
        raw = cached_json_call(
            llm_client,
            prompt,
            cache_path=cache_path,
            prompt_version=STUDY_EXTRACTION_PROMPT_VERSION,
            timeout=timeout,
            max_tokens=STUDY_EXTRACTION_MAX_TOKENS,
            force=force,
            validator=lambda value: _validate_study_payload(value, candidate, context),
        )
        return _normalize_study_record(raw, candidate, context, index), context

    with ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as pool:
        future_map = {pool.submit(run, candidate): candidate for candidate in candidates}
        for future in as_completed(future_map):
            candidate = future_map[future]
            study_id = str(candidate["study_id"])
            try:
                record, context = future.result()
                records[study_id] = record
                contexts[study_id] = _context_summary(context)
                print(
                    f"  Stage 1 study extraction: {candidate['reported_label']} "
                    f"({context.context_chars} chars, {len(context.block_ids)} blocks)",
                    flush=True,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                records[study_id] = _failed_study_record(candidate, message)
                contexts[study_id] = {
                    "mode": "extraction_error",
                    "block_ids": list(candidate.get("evidence_block_ids") or []),
                    "pages": [],
                    "context_chars": 0,
                }
                errors.append({"study_id": study_id, "error": message})
                print(
                    f"  Stage 1 study extraction failed: {candidate['reported_label']}: {message}",
                    flush=True,
                )
    ordered = [records[str(candidate["study_id"])] for candidate in candidates]
    return ordered, contexts, errors


def _study_extraction_prompt(
    candidate: Dict[str, Any],
    context: EvidenceContext,
    *,
    pdf_name: str,
    regeneration_instructions: Optional[Dict[str, Any]],
) -> str:
    return f"""Extract exactly one empirical unit for Stage 1 of HumanStudy-Bench.

Paper: {pdf_name}
Candidate boundary:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

{stage1_policy_text()}

Use only the bounded evidence context below. Do not import design, sample,
conditions, tasks, or outcomes from neighboring studies. Preserve the candidate
study_id exactly. Every non-null factual field must cite supporting block IDs in
field_evidence. Evidence IDs must come from this valid list:
{json.dumps(context.block_ids, ensure_ascii=False)}

The candidate may contain component_mentions created when duplicate descriptions
or repeated items from one task family were reconciled. Cover every component in
the aggregate input, participant_task, output, and evidence; do not let the
representative label erase sibling items. Conversely, shared_contexts contains
sample/procedure evidence that applies to this task but does not merge it with
other task families.

If exact participant-facing wording or options are absent, describe the task at
the level supported here, set has_self_contained_materials=false, and record the
gap in missing_materials. Do not label the study NO solely for missing materials.
Do not turn multiple effects, dependent variables, table rows, conditions, or
parameterized repetitions of one task template into additional studies. Recover
source-backed alternative versions in material_variants instead. Different task
families sharing one questionnaire remain separate candidates upstream.

Adjudicate candidate provenance explicitly:
- unit_provenance=current_paper only when the authors report this paper's own
  participant/sample procedure or quantitative result;
- unit_provenance=cited_prior for a prior, cited, or external study;
- unit_provenance=unclear when the bounded evidence cannot establish provenance.
Do not treat a parenthetical footnote/citation marker by itself as proof that a
unit belongs to another paper. Require explicit attribution to other authors,
another publication, or an external dataset before using cited_prior.
Set is_distinct_empirical_unit=false for a general sample-description sentence,
narrative robustness statement, method detail, or result fragment that belongs
to another listed unit rather than its own task/sample/result. Do not keep such
fragments as standalone experiments.
Distinctness and simulation eligibility are separate decisions. A separately
labeled top-level data collection remains a distinct empirical unit even when N
or an exact statistic is absent; use replicable=NO for the missing quantitative
target. Conditions or material versions nested under one Study/Experiment/
Problem stay inside that unit even when assigned to different groups. Never set
distinct=false merely to express that a real unit is ineligible.
The candidate field source_anchor records whether discovery found a source-level
Study/Experiment/Problem/Survey/Pilot/etc. label. If source_anchor=false, require
direct evidence for an independent current-paper collection and an exact
quantitative response result before preserving the candidate as a standalone
unit; otherwise it is likely a method/discussion/result fragment.

For material_variants, include only participant-facing content versions that
Stage 3 may need to reconstruct separately, such as different stories,
vignettes, questionnaire forms, stimulus messages, sign texts, item sets, or
presentation orders that were alternatives across participants or occasions.
Every emitted entry must be a genuine alternative version and set
is_alternative_version=true. Do not list response options shown together in one
choice question, response categories, a sole/synthetic "single form", one
unchanged item set, outcome-defined groups, analysis subgroups, or result-table
rows. Put jointly shown options in input/participant_task/conditions instead.
Use a stable variant_id within this study and cite direct evidence for every
variant. If the source supports fewer than two alternative versions, return an
empty array.

Adjudicate execution barriers separately from missing materials. Add a
simulation_barrier only when the ORIGINAL measured response or a necessary
part of the primary quantitative target requires an execution mode that the
benchmark cannot reproduce, such as observed physical action, a consequential
commitment with enforced real-world follow-through, live contingent
interaction, longitudinal real-world exposure, a dynamic environment, or
specialized apparatus. Reading a static vignette and reporting a hypothetical
choice, rating, ranking, estimate, or free-text response is not a barrier.
`affects_primary_target` must describe the original target, not whether a weaker
proxy question could be invented. Missing wording, options, images, or source
files belongs only in missing_materials and is never a simulation barrier.

Also adjudicate empirical_support from direct evidence. quantitative_result=yes
requires an exact numeric result/statistic, not phrases such as "similar
pattern," "most respondents," or "the effect was observed." A standalone unit
normally needs a participant-facing task plus either its own sample/assignment
or an exact quantitative result. Otherwise mark it non-distinct or ineligible
instead of inventing missing evidence.
{_feedback_text(regeneration_instructions)}

EVIDENCE CONTEXT:
{context.text}

Return only one JSON object:
{{
  "experiment_id": "paper label",
  "study_id": "{candidate['study_id']}",
  "experiment_name": "short description",
  "study_name": "short human-readable name",
  "design_type": "between-subjects|within-subjects|mixed|correlational|field|archival|other|null",
  "conditions_or_factors": ["factor: levels or measured variable"],
  "material_variants": [
    {{
      "variant_id": "stable_snake_case_id",
      "label": "source-supported version label",
      "role": "condition|stimulus|form|order|item_set|other",
      "is_alternative_version": true,
      "assignment": "how this version was assigned or presented; null if unsupported",
      "participant_task_difference": "what participant-facing content differs; null if unsupported",
      "sample": "version-specific N/sample; null if unsupported",
      "quantitative_target": "version-specific result; null if unsupported",
      "evidence_refs": ["block id"]
    }}
  ],
  "input": "what participants saw, read, or did; null if unsupported",
  "participant_task": "participant-facing action; null if unsupported",
  "participants": "sample and N; null if unsupported",
  "output": "measured response/target; null if unsupported",
  "candidate_source_hints": [
    {{
      "kind": "paper|appendix|supplement|osf|cited_scale|unknown",
      "description": "where Stage 3 should recover exact materials",
      "expected_fields": ["instructions", "stimulus", "items", "options", "conditions"]
    }}
  ],
  "replicable": "YES|NO|UNCERTAIN",
  "has_self_contained_materials": false,
  "exclusion_reasons": [],
  "missing_materials": "specific missing material or empty string",
  "unit_provenance": "current_paper|cited_prior|unclear",
  "is_distinct_empirical_unit": true,
  "unit_provenance_evidence": "why this is or is not a study conducted in this paper",
  "empirical_support": {{
    "own_sample_or_assignment": "yes|no|unclear",
    "participant_facing_task": "yes|no|unclear",
    "quantitative_result": "yes|no|unclear"
  }},
  "simulation_barriers": [
    {{
      "kind": "physical_action|consequential_commitment|live_interaction|longitudinal_exposure|dynamic_environment|specialized_apparatus|other",
      "description": "source-grounded execution requirement",
      "affects_primary_target": true,
      "evidence_refs": ["block id"]
    }}
  ],
  "evidence_refs": ["block id"],
  "field_evidence": {{
    "design_type": ["block id"],
    "conditions_or_factors": ["block id"],
    "material_variants": ["block id"],
    "input": ["block id"],
    "participant_task": ["block id"],
    "participants": ["block id"],
    "output": ["block id"],
    "simulation_barriers": ["block id"],
    "replicable": ["block id"]
  }},
  "extraction_confidence": 0.0,
  "boundary_notes": "short note about study identity or ambiguity"
}}"""


def _normalize_study_record(
    raw: Dict[str, Any],
    candidate: Dict[str, Any],
    context: EvidenceContext,
    index: PdfEvidenceIndex,
) -> Dict[str, Any]:
    valid_refs = set(context.block_ids)
    label = str(candidate.get("reported_label") or raw.get("experiment_id") or "Study").strip()
    record = dict(raw)
    record["experiment_id"] = label
    record["study_id"] = str(candidate["study_id"])
    record["experiment_name"] = str(
        raw.get("experiment_name") or raw.get("study_name") or candidate.get("study_name") or label
    ).strip()
    record["study_name"] = str(
        raw.get("study_name") or raw.get("experiment_name") or candidate.get("study_name") or label
    ).strip()
    record["material_variants"] = _normalize_material_variants(
        raw.get("material_variants"),
        valid_refs=valid_refs,
    )
    record.setdefault("design_type", None)
    record.setdefault("conditions_or_factors", [])
    record.setdefault("input", None)
    record.setdefault("participant_task", None)
    record.setdefault("participants", None)
    record.setdefault("output", None)
    record.setdefault("candidate_source_hints", [])
    record.setdefault("replicable", "UNCERTAIN")
    record.setdefault("has_self_contained_materials", False)
    record.setdefault("exclusion_reasons", [])
    record.setdefault("missing_materials", "")
    record["unit_provenance"] = str(raw.get("unit_provenance") or "unclear").strip().lower()
    record["is_distinct_empirical_unit"] = bool(raw.get("is_distinct_empirical_unit"))
    record["unit_provenance_evidence"] = str(
        raw.get("unit_provenance_evidence") or ""
    ).strip()
    record["empirical_support"] = _normalize_empirical_support(raw.get("empirical_support"))
    record["simulation_barriers"] = _normalize_simulation_barriers(
        raw.get("simulation_barriers"),
        valid_refs=valid_refs,
    )
    if record["unit_provenance"] == "unclear":
        record["replicable"] = "UNCERTAIN"
    support = record["empirical_support"]
    if support["quantitative_result"] == "no":
        record["replicable"] = "NO"
        reasons = record.get("exclusion_reasons")
        reasons = list(reasons) if isinstance(reasons, list) else []
        reason = "No exact source-grounded quantitative target result is reported."
        if reason not in reasons:
            reasons.append(reason)
        record["exclusion_reasons"] = reasons
    _apply_simulation_barrier_gate(record)
    refs = _valid_refs(raw.get("evidence_refs"), valid_refs)
    if not refs:
        refs = [ref for ref in candidate.get("evidence_block_ids") or [] if ref in valid_refs]
    refs = list(
        dict.fromkeys(
            [
                *refs,
                *(
                    ref
                    for variant in record["material_variants"]
                    for ref in variant.get("evidence_refs") or []
                ),
                *(
                    ref
                    for barrier in record["simulation_barriers"]
                    for ref in barrier.get("evidence_refs") or []
                ),
            ]
        )
    )
    record["evidence_refs"] = refs
    field_evidence: Dict[str, List[str]] = {}
    raw_field_evidence = raw.get("field_evidence")
    if isinstance(raw_field_evidence, dict):
        for field, values in raw_field_evidence.items():
            field_evidence[str(field)] = _valid_refs(values, valid_refs)
    record["field_evidence"] = field_evidence
    record["evidence_pages"] = sorted(
        {
            page
            for ref in refs
            for block in [index.document.block_map().get(ref)]
            if block is not None
            for page in range(block.page_start, block.page_end + 1)
        }
    )
    record["source_mention_ids"] = list(candidate.get("source_mention_ids") or [])
    record["source_task_family_relation_ids"] = list(
        candidate.get("source_task_family_relation_ids") or []
    )
    record["candidate_aliases"] = list(candidate.get("aliases") or [])
    record["candidate_components"] = deepcopy(
        list(candidate.get("component_mentions") or [])
    )
    record["shared_contexts"] = deepcopy(list(candidate.get("shared_contexts") or []))
    record["evidence_ref_repairs"] = deepcopy(
        list(raw.get("_evidence_ref_repairs") or [])
    )
    record["extraction_confidence"] = _confidence(raw.get("extraction_confidence"))
    record["boundary_notes"] = str(
        raw.get("boundary_notes") or candidate.get("boundary_notes") or ""
    ).strip()
    record["source_anchor"] = bool(candidate.get("source_anchor"))
    return record


def _failed_study_record(candidate: Dict[str, Any], error: str) -> Dict[str, Any]:
    label = str(candidate.get("reported_label") or candidate.get("study_name") or "Study")
    return {
        "experiment_id": label,
        "study_id": candidate["study_id"],
        "experiment_name": candidate.get("study_name") or label,
        "study_name": candidate.get("study_name") or label,
        "material_variants": list(candidate.get("material_variants") or []),
        "design_type": None,
        "conditions_or_factors": [],
        "input": candidate.get("participant_task_hint"),
        "participant_task": candidate.get("participant_task_hint"),
        "participants": None,
        "output": candidate.get("quantitative_target_hint"),
        "candidate_source_hints": [
            {
                "kind": "paper",
                "description": "Stage 1 candidate evidence requires extraction retry",
                "expected_fields": ["instructions", "stimulus", "items", "options", "conditions"],
            }
        ],
        "replicable": "UNCERTAIN",
        "has_self_contained_materials": False,
        "exclusion_reasons": [],
        "missing_materials": f"Stage 1 evidence extraction failed: {error}",
        "unit_provenance": "unclear",
        "is_distinct_empirical_unit": True,
        "unit_provenance_evidence": "Extraction failed; preserve for manual review.",
        "empirical_support": {
            "own_sample_or_assignment": "unclear",
            "participant_facing_task": "unclear",
            "quantitative_result": "unclear",
        },
        "simulation_barriers": [],
        "evidence_refs": list(candidate.get("evidence_block_ids") or []),
        "field_evidence": {},
        "source_mention_ids": list(candidate.get("source_mention_ids") or []),
        "source_task_family_relation_ids": list(
            candidate.get("source_task_family_relation_ids") or []
        ),
        "candidate_aliases": list(candidate.get("aliases") or []),
        "candidate_components": deepcopy(list(candidate.get("component_mentions") or [])),
        "shared_contexts": deepcopy(list(candidate.get("shared_contexts") or [])),
        "evidence_ref_repairs": [],
        "extraction_confidence": 0.0,
        "boundary_notes": candidate.get("boundary_notes") or "",
        "source_anchor": bool(candidate.get("source_anchor")),
    }


def _partition_empirical_units(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for record in records:
        provenance = str(record.get("unit_provenance") or "unclear").strip().lower()
        distinct = bool(record.get("is_distinct_empirical_unit"))
        support = record.get("empirical_support")
        support = support if isinstance(support, dict) else {}
        source_anchor = bool(record.get("source_anchor"))
        weak_boundary = not source_anchor and str(
            support.get("quantitative_result") or "unclear"
        ).strip().lower() != "yes"
        if provenance != "cited_prior" and distinct:
            if weak_boundary and str(record.get("replicable") or "").upper() == "YES":
                record["replicable"] = "UNCERTAIN"
            accepted.append(record)
            continue
        rejected.append(
            {
                "study_id": record.get("study_id"),
                "experiment_id": record.get("experiment_id"),
                "study_name": record.get("study_name"),
                "unit_provenance": provenance,
                "is_distinct_empirical_unit": distinct,
                "reason": (
                    record.get("boundary_notes")
                    if not distinct
                    else record.get("unit_provenance_evidence")
                ),
                "evidence_refs": record.get("evidence_refs") or [],
            }
        )
    return accepted, rejected


_RELATIONSHIP_KINDS = {
    "paired_contrast",
    "multi_unit_comparison",
    "replication_set",
    "sequence",
    "shared_sample",
    "other",
}


def _attach_discovered_shared_sample_contexts(
    candidates: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
    *,
    valid_refs: set[str],
) -> List[Dict[str, Any]]:
    """Attach questionnaire-wide sample/procedure evidence before study extraction."""
    candidates_by_id: Dict[str, Dict[str, Any]] = {}
    member_ref_aliases: Dict[str, set[str]] = {}
    exact_aliases: Dict[str, set[str]] = {}
    identity_aliases: Dict[str, set[str]] = {}
    for candidate in candidates:
        study_id = str(candidate.get("study_id") or "").strip()
        if not study_id:
            continue
        candidates_by_id[study_id] = candidate
        aliases = [
            candidate.get("study_id"),
            candidate.get("reported_label"),
            candidate.get("study_name"),
            *(candidate.get("aliases") or []),
        ]
        for component in candidate.get("component_mentions") or []:
            if isinstance(component, dict):
                aliases.extend(
                    [component.get("study_id"), component.get("reported_label")]
                )
        for alias in aliases:
            if not str(alias or "").strip():
                continue
            identity = _mention_identity_key({"reported_label": str(alias)})
            member_ref_aliases.setdefault(identity, set()).add(study_id)
            exact_aliases.setdefault(_relation_label_key(alias), set()).add(study_id)
            if identity:
                identity_aliases.setdefault(identity, set()).add(study_id)

    actions: List[Dict[str, Any]] = []
    for relation in relations:
        kind = str(relation.get("relationship_kind") or "").strip().lower()
        if kind != "shared_sample":
            continue
        member_refs = relation.get("member_refs")
        member_refs = member_refs if isinstance(member_refs, list) else []
        if member_refs:
            resolutions = [
                (
                    _member_ref_display(member_ref),
                    _resolve_relation_member_ref(
                        member_ref,
                        member_ref_aliases=member_ref_aliases,
                    ),
                )
                for member_ref in member_refs
                if isinstance(member_ref, dict)
            ]
        else:
            resolutions = [
                (
                    str(label),
                    _resolve_relation_label(
                        label,
                        exact_aliases=exact_aliases,
                        identity_aliases=identity_aliases,
                    ),
                )
                for label in relation.get("member_labels") or []
            ]
        resolved_ids = list(
            dict.fromkeys(study_id for _, study_id in resolutions if study_id)
        )
        unresolved_labels = [label for label, study_id in resolutions if not study_id]
        evidence_refs = [
            str(ref)
            for ref in relation.get("evidence_block_ids") or []
            if str(ref) in valid_refs
        ]
        if not resolved_ids or not evidence_refs:
            continue
        relation_id = str(relation.get("relation_mention_id") or "shared_sample").strip()
        reason = str(
            relation.get("evidence_summary")
            or relation.get("comparison_target")
            or "Source evidence describes a shared participant sample or procedure."
        ).strip()
        context = {
            "study_id": relation_id,
            "reported_label": "Shared sample/procedure context",
            "study_name": "Shared sample/procedure context",
            "evidence_block_ids": evidence_refs,
            "source_mention_ids": [relation_id] if relation_id else [],
        }
        for study_id in resolved_ids:
            _attach_shared_context(
                candidates_by_id[study_id],
                context,
                reason=reason,
            )
        actions.append(
            {
                "action": "attach_discovered_shared_context",
                "source_relation_id": relation_id,
                "target_study_ids": resolved_ids,
                "unresolved_member_labels": unresolved_labels,
                "reason": reason,
                "evidence_block_ids": evidence_refs,
            }
        )
    return actions


def _reconcile_comparison_groups(
    relations: Sequence[Dict[str, Any]],
    experiments: Sequence[Dict[str, Any]],
    *,
    all_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Map source-explicit relationship labels onto preserved empirical units."""
    exact_aliases: Dict[str, set[str]] = {}
    identity_aliases: Dict[str, set[str]] = {}
    member_ref_aliases: Dict[str, set[str]] = {}
    experiment_by_id: Dict[str, Dict[str, Any]] = {}
    experiment_order: Dict[str, int] = {}
    accepted_ids = {
        str(experiment.get("study_id") or "").strip()
        for experiment in experiments
        if str(experiment.get("study_id") or "").strip()
    }
    source_records = list(all_records) if all_records is not None else list(experiments)
    for position, experiment in enumerate(source_records):
        study_id = str(experiment.get("study_id") or "").strip()
        if not study_id:
            continue
        experiment_by_id[study_id] = experiment
        experiment_order[study_id] = position
        if str(experiment.get("experiment_id") or "").strip():
            member_ref_aliases.setdefault(
                _mention_identity_key(
                    {"reported_label": experiment.get("experiment_id")}
                ),
                set(),
            ).add(study_id)
        aliases = [
            experiment.get("study_id"),
            experiment.get("experiment_id"),
            experiment.get("study_name"),
            experiment.get("experiment_name"),
            *(experiment.get("candidate_aliases") or []),
        ]
        for component in experiment.get("candidate_components") or []:
            if not isinstance(component, dict):
                continue
            aliases.extend(
                [
                    component.get("study_id"),
                    component.get("reported_label"),
                    component.get("study_name"),
                ]
            )
        for alias in aliases:
            if not str(alias or "").strip():
                continue
            identity = _mention_identity_key({"reported_label": str(alias)})
            member_ref_aliases.setdefault(identity, set()).add(study_id)
            exact_aliases.setdefault(_relation_label_key(alias), set()).add(study_id)
            if identity:
                identity_aliases.setdefault(identity, set()).add(study_id)

    grouped: Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]] = {}
    group_order: List[Tuple[Tuple[str, ...], str]] = []
    rejected: List[Dict[str, Any]] = []
    ignored: List[Dict[str, Any]] = []
    for relation in relations:
        relation_kind = str(relation.get("relationship_kind") or "other").strip().lower()
        label_resolutions: List[str] = []
        unresolved: List[str] = []
        member_refs = relation.get("member_refs")
        member_refs = member_refs if isinstance(member_refs, list) else []
        if member_refs:
            members_to_resolve = [
                (
                    _member_ref_display(member_ref),
                    _resolve_relation_member_ref(
                        member_ref,
                        member_ref_aliases=member_ref_aliases,
                    ),
                )
                for member_ref in member_refs
                if isinstance(member_ref, dict)
            ]
        else:
            members_to_resolve = [
                (
                    str(label),
                    _resolve_relation_label(
                        label,
                        exact_aliases=exact_aliases,
                        identity_aliases=identity_aliases,
                    ),
                )
                for label in relation.get("member_labels") or []
            ]
        for label, study_id in members_to_resolve:
            if study_id is None:
                unresolved.append(str(label))
            else:
                label_resolutions.append(study_id)
        resolved = list(dict.fromkeys(label_resolutions))
        resolved.sort(key=lambda value: experiment_order.get(value, 10**9))
        members_are_distinct = bool(relation.get("members_are_distinct_empirical_units"))
        if relation_kind == "shared_sample":
            ignored.append(
                {
                    **relation,
                    "resolved_member_study_ids": resolved,
                    "unresolved_member_labels": unresolved,
                    "ignore_reason": (
                        "shared sample or procedure is represented through shared_contexts, "
                        "not as an effect comparison group"
                    ),
                }
            )
            continue
        excluded_members = [study_id for study_id in resolved if study_id not in accepted_ids]
        if excluded_members:
            ignored.append(
                {
                    **relation,
                    "resolved_member_study_ids": resolved,
                    "unresolved_member_labels": unresolved,
                    "ignored_member_study_ids": excluded_members,
                    "ignore_reason": "relationship includes a candidate excluded by the provenance/distinctness gate",
                }
            )
            continue
        if unresolved:
            rejected.append(
                {
                    **relation,
                    "resolved_member_study_ids": resolved,
                    "unresolved_member_labels": unresolved,
                    "rejection_reason": (
                        "one or more source labels did not map uniquely to discovered empirical units"
                    ),
                }
            )
            continue
        relation_id = str(relation.get("relation_mention_id") or "").strip()
        consumed_by_task_family = (
            len(resolved) == 1
            and relation_id
            and relation_id
            in set(
                experiment_by_id[resolved[0]].get(
                    "source_task_family_relation_ids"
                )
                or []
            )
        )
        if consumed_by_task_family:
            ignored.append(
                {
                    **relation,
                    "resolved_member_study_ids": resolved,
                    "ignore_reason": (
                        "relationship was consumed by the global simulation task-family merge"
                    ),
                }
            )
            continue
        if len(resolved) < 2:
            if not members_are_distinct:
                ignored.append(
                    {
                        **relation,
                        "resolved_member_study_ids": resolved,
                        "ignore_reason": "source describes an intra-unit item or condition relationship",
                    }
                )
            else:
                rejected.append(
                    {
                        **relation,
                        "resolved_member_study_ids": resolved,
                        "unresolved_member_labels": [],
                        "rejection_reason": (
                            "distinct source versions collapsed onto fewer than two inventory units"
                        ),
                    }
                )
            continue
        if not members_are_distinct:
            rejected.append(
                {
                    **relation,
                    "resolved_member_study_ids": resolved,
                    "unresolved_member_labels": [],
                    "rejection_reason": (
                        "inventory split a source relationship that was marked intra-unit"
                    ),
                }
            )
            continue

        kind = relation_kind
        if kind not in _RELATIONSHIP_KINDS:
            kind = "other"
        key = (tuple(resolved), kind)
        if key not in grouped:
            grouped[key] = {
                "comparison_group_id": "",
                "member_study_ids": resolved,
                "member_labels": [
                    str(experiment_by_id[study_id].get("experiment_id") or study_id)
                    for study_id in resolved
                ],
                "relationship_kind": kind,
                "members_are_distinct_empirical_units": True,
                "comparison_target": str(relation.get("comparison_target") or "").strip(),
                "evidence_refs": [],
                "evidence_summary": str(relation.get("evidence_summary") or "").strip(),
                "confidence": _confidence(relation.get("confidence")),
                "source_relation_ids": [],
            }
            group_order.append(key)
        group = grouped[key]
        group["evidence_refs"] = list(
            dict.fromkeys(
                [*group["evidence_refs"], *(relation.get("evidence_block_ids") or [])]
            )
        )
        source_id = str(relation.get("relation_mention_id") or "").strip()
        if source_id and source_id not in group["source_relation_ids"]:
            group["source_relation_ids"].append(source_id)
        candidate_target = str(relation.get("comparison_target") or "").strip()
        if len(candidate_target) > len(str(group.get("comparison_target") or "")):
            group["comparison_target"] = candidate_target
        candidate_summary = str(relation.get("evidence_summary") or "").strip()
        if len(candidate_summary) > len(str(group.get("evidence_summary") or "")):
            group["evidence_summary"] = candidate_summary
        group["confidence"] = max(
            _confidence(group.get("confidence")),
            _confidence(relation.get("confidence")),
        )

    output: List[Dict[str, Any]] = []
    for position, key in enumerate(group_order, start=1):
        group = grouped[key]
        output.append(group)
    output = _merge_overlapping_comparison_groups(
        output,
        experiment_by_id=experiment_by_id,
        experiment_order=experiment_order,
    )
    still_rejected: List[Dict[str, Any]] = []
    for relation in rejected:
        resolved = [str(value) for value in relation.get("resolved_member_study_ids") or []]
        refs = {str(value) for value in relation.get("evidence_block_ids") or []}
        is_redundant_alias = (
            not relation.get("unresolved_member_labels")
            and len(resolved) == 1
            and any(
                resolved[0] in set(group.get("member_study_ids") or [])
                and bool(refs & {str(value) for value in group.get("evidence_refs") or []})
                for group in output
            )
        )
        if is_redundant_alias:
            ignored.append(
                {
                    **relation,
                    "ignore_reason": (
                        "relation is a source-label alias of an already mapped comparison group"
                    ),
                }
            )
        else:
            still_rejected.append(relation)
    for position, group in enumerate(output, start=1):
        group["comparison_group_id"] = f"comparison_group_{position:03d}"
    return output, still_rejected, ignored


def _merge_overlapping_comparison_groups(
    groups: Sequence[Dict[str, Any]],
    *,
    experiment_by_id: Dict[str, Dict[str, Any]],
    experiment_order: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Collapse overlapping same-window contrasts into maximal finding groups."""
    mergeable_kinds = {"paired_contrast", "multi_unit_comparison"}
    pending = [dict(group) for group in groups]
    changed = True
    while changed:
        changed = False
        for left_index in range(len(pending)):
            left = pending[left_index]
            left_members = set(left.get("member_study_ids") or [])
            left_refs = set(left.get("evidence_refs") or [])
            left_windows = {
                str(value).split("_relation_", 1)[0]
                for value in left.get("source_relation_ids") or []
            }
            for right_index in range(left_index + 1, len(pending)):
                right = pending[right_index]
                right_members = set(right.get("member_study_ids") or [])
                same_members = left_members == right_members
                nested_members = left_members < right_members or right_members < left_members
                if nested_members:
                    continue
                if not same_members and (
                    left.get("relationship_kind") not in mergeable_kinds
                    or right.get("relationship_kind") not in mergeable_kinds
                ):
                    continue
                right_windows = {
                    str(value).split("_relation_", 1)[0]
                    for value in right.get("source_relation_ids") or []
                }
                if not same_members and not (left_windows & right_windows):
                    continue
                if not (left_members & right_members):
                    continue
                if not same_members and not (
                    left_refs & set(right.get("evidence_refs") or [])
                ):
                    continue
                members = list(
                    dict.fromkeys(
                        [
                            *(left.get("member_study_ids") or []),
                            *(right.get("member_study_ids") or []),
                        ]
                    )
                )
                members.sort(key=lambda value: experiment_order.get(value, 10**9))
                left["member_study_ids"] = members
                left["member_labels"] = [
                    str(experiment_by_id[study_id].get("experiment_id") or study_id)
                    for study_id in members
                ]
                if left.get("relationship_kind") != right.get("relationship_kind"):
                    left["relationship_kind"] = "multi_unit_comparison"
                for field in ("evidence_refs", "source_relation_ids"):
                    left[field] = list(
                        dict.fromkeys(
                            [*(left.get(field) or []), *(right.get(field) or [])]
                        )
                    )
                for field in ("comparison_target", "evidence_summary"):
                    left_value = str(left.get(field) or "")
                    right_value = str(right.get(field) or "")
                    if right_value and right_value not in left_value:
                        left[field] = " | ".join(value for value in (left_value, right_value) if value)
                left["confidence"] = max(
                    _confidence(left.get("confidence")),
                    _confidence(right.get("confidence")),
                )
                pending.pop(right_index)
                changed = True
                break
            if changed:
                break
    return pending


def _relation_label_key(value: Any) -> str:
    text = re.sub(r"\bn\s*=\s*\d+\b", "", str(value or ""), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _resolve_relation_label(
    label: Any,
    *,
    exact_aliases: Dict[str, set[str]],
    identity_aliases: Dict[str, set[str]],
) -> Optional[str]:
    exact_matches = exact_aliases.get(_relation_label_key(label), set())
    if len(exact_matches) == 1:
        return next(iter(exact_matches))
    identity = _mention_identity_key({"reported_label": str(label or "")})
    identity_matches = identity_aliases.get(identity, set())
    if len(identity_matches) == 1:
        return next(iter(identity_matches))
    return None


def _resolve_relation_member_ref(
    member_ref: Dict[str, Any],
    *,
    member_ref_aliases: Dict[str, set[str]],
) -> Optional[str]:
    key = _mention_identity_key(
        {"reported_label": member_ref.get("reported_label")}
    )
    matches = member_ref_aliases.get(key, set())
    return next(iter(matches)) if len(matches) == 1 else None


def _validate_discovery_payload(
    payload: Dict[str, Any],
    window: DiscoveryWindow,
) -> None:
    mentions = payload.get("candidate_mentions")
    if not isinstance(mentions, list):
        raise ValueError("candidate_mentions must be an array")
    valid_refs = set(window.block_ids)
    candidate_relation_keys: set[str] = set()
    for position, mention in enumerate(mentions):
        if not isinstance(mention, dict):
            raise ValueError(f"candidate_mentions[{position}] must be an object")
        reported_label = str(mention.get("reported_label") or "").strip()
        if not reported_label:
            raise ValueError(f"candidate_mentions[{position}] has no reported_label")
        explicit_labels = _explicit_unit_labels(reported_label)
        if len(explicit_labels) > 1:
            raise ValueError(
                f"candidate_mentions[{position}] combines multiple parent units: "
                f"{explicit_labels}"
            )
        refs = mention.get("evidence_block_ids")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"candidate_mentions[{position}] has no evidence_block_ids")
        invalid = [str(ref) for ref in refs if str(ref) not in valid_refs]
        if invalid:
            raise ValueError(
                f"candidate_mentions[{position}] has invalid evidence refs: {invalid}"
            )
        candidate_key = _mention_identity_key({"reported_label": reported_label})
        if candidate_key in candidate_relation_keys:
            raise ValueError(
                f"candidate_mentions[{position}] duplicates parent empirical unit "
                f"{_canonical_reported_label(reported_label)!r}; merge its variants"
            )
        candidate_relation_keys.add(candidate_key)
        material_variants = mention.get("material_variants")
        if not isinstance(material_variants, list):
            raise ValueError(
                f"candidate_mentions[{position}].material_variants must be an array"
            )
        for variant_position, variant in enumerate(material_variants):
            if not isinstance(variant, dict) or not str(variant.get("label") or "").strip():
                raise ValueError(
                    f"candidate_mentions[{position}].material_variants[{variant_position}] "
                    "must be an object with a label"
                )
            variant_refs = variant.get("evidence_block_ids")
            if not isinstance(variant_refs, list) or not variant_refs:
                raise ValueError(
                    f"candidate_mentions[{position}].material_variants[{variant_position}] "
                    "has no evidence_block_ids"
                )
            invalid_variant_refs = [
                str(ref) for ref in variant_refs if str(ref) not in valid_refs
            ]
            if invalid_variant_refs:
                raise ValueError(
                    f"candidate_mentions[{position}].material_variants[{variant_position}] "
                    f"has invalid evidence refs: {invalid_variant_refs}"
                )
    relations = payload.get("comparison_relations")
    if not isinstance(relations, list):
        raise ValueError("comparison_relations must be an array")
    for position, relation in enumerate(relations):
        if not isinstance(relation, dict):
            raise ValueError(f"comparison_relations[{position}] must be an object")
        member_refs = relation.get("member_refs")
        if not isinstance(member_refs, list) or len(member_refs) < 2:
            raise ValueError(
                f"comparison_relations[{position}] must reference at least two candidates"
            )
        relation_keys: List[str] = []
        for member_position, member_ref in enumerate(member_refs):
            if not isinstance(member_ref, dict):
                raise ValueError(
                    f"comparison_relations[{position}].member_refs[{member_position}] must be an object"
                )
            key = _mention_identity_key(
                {"reported_label": member_ref.get("reported_label")}
            )
            if key not in candidate_relation_keys:
                raise ValueError(
                    f"comparison_relations[{position}].member_refs[{member_position}] does not exactly reference a candidate"
                )
            relation_keys.append(key)
        if len(set(relation_keys)) < 2:
            raise ValueError(
                f"comparison_relations[{position}] must reference two distinct candidates"
            )
        if not isinstance(relation.get("members_are_distinct_empirical_units"), bool):
            raise ValueError(
                f"comparison_relations[{position}].members_are_distinct_empirical_units must be boolean"
            )
        refs = relation.get("evidence_block_ids")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"comparison_relations[{position}] has no evidence_block_ids")
        invalid = [str(ref) for ref in refs if str(ref) not in valid_refs]
        if invalid:
            raise ValueError(
                f"comparison_relations[{position}] has invalid evidence refs: {invalid}"
            )


def _validate_study_payload(
    payload: Dict[str, Any],
    candidate: Dict[str, Any],
    context: EvidenceContext,
) -> None:
    _prune_invalid_study_evidence_refs(payload, valid_refs=set(context.block_ids))
    expected_id = str(candidate["study_id"])
    if str(payload.get("study_id") or "") != expected_id:
        raise ValueError(
            f"study extraction changed study_id: expected={expected_id}, got={payload.get('study_id')}"
        )
    label = str(payload.get("replicable") or "").strip().upper()
    if label not in {"YES", "NO", "UNCERTAIN"}:
        raise ValueError(f"invalid simulation eligibility label: {label or 'missing'}")
    refs = payload.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("study extraction has no evidence_refs")
    invalid = [str(ref) for ref in refs if str(ref) not in set(context.block_ids)]
    if invalid:
        raise ValueError(f"study extraction has invalid evidence refs: {invalid}")
    provenance = str(payload.get("unit_provenance") or "").strip().lower()
    if provenance not in {"current_paper", "cited_prior", "unclear"}:
        raise ValueError(f"invalid unit_provenance: {provenance or 'missing'}")
    if not isinstance(payload.get("is_distinct_empirical_unit"), bool):
        raise ValueError("is_distinct_empirical_unit must be boolean")
    empirical_support = payload.get("empirical_support")
    if not isinstance(empirical_support, dict):
        raise ValueError("empirical_support must be an object")
    for field in (
        "own_sample_or_assignment",
        "participant_facing_task",
        "quantitative_result",
    ):
        status = str(empirical_support.get(field) or "").strip().lower()
        if status not in {"yes", "no", "unclear"}:
            raise ValueError(f"empirical_support.{field} has invalid status: {status or 'missing'}")
    simulation_barriers = payload.get("simulation_barriers")
    if not isinstance(simulation_barriers, list):
        raise ValueError("simulation_barriers must be an array")
    valid_refs = set(context.block_ids)
    for position, barrier in enumerate(simulation_barriers):
        if not isinstance(barrier, dict):
            raise ValueError(f"simulation_barriers[{position}] must be an object")
        kind = str(barrier.get("kind") or "").strip().lower()
        if kind not in SIMULATION_BARRIER_KINDS:
            raise ValueError(f"simulation_barriers[{position}] has invalid kind: {kind}")
        if not str(barrier.get("description") or "").strip():
            raise ValueError(f"simulation_barriers[{position}] has no description")
        if not isinstance(barrier.get("affects_primary_target"), bool):
            raise ValueError(
                f"simulation_barriers[{position}].affects_primary_target must be boolean"
            )
        barrier_refs = barrier.get("evidence_refs")
        if not isinstance(barrier_refs, list) or not barrier_refs:
            raise ValueError(f"simulation_barriers[{position}] has no evidence_refs")
        invalid_barrier_refs = [
            str(ref) for ref in barrier_refs if str(ref) not in valid_refs
        ]
        if invalid_barrier_refs:
            raise ValueError(
                f"simulation_barriers[{position}] has invalid evidence refs: "
                f"{invalid_barrier_refs}"
            )
    material_variants = payload.get("material_variants")
    if not isinstance(material_variants, list):
        raise ValueError("material_variants must be an array")
    seen_variant_ids: set[str] = set()
    seen_material_variant_labels: set[str] = set()
    for position, variant in enumerate(material_variants):
        if not isinstance(variant, dict):
            raise ValueError(f"material_variants[{position}] must be an object")
        variant_id = str(variant.get("variant_id") or "").strip()
        material_variant_label = str(variant.get("label") or "").strip()
        if not variant_id or not material_variant_label:
            raise ValueError(
                f"material_variants[{position}] requires variant_id and label"
            )
        if variant_id in seen_variant_ids:
            raise ValueError(f"duplicate material variant id: {variant_id}")
        label_key = _relation_label_key(material_variant_label)
        if label_key in seen_material_variant_labels:
            raise ValueError(
                f"duplicate material variant label: {material_variant_label}"
            )
        seen_variant_ids.add(variant_id)
        seen_material_variant_labels.add(label_key)
        role = str(variant.get("role") or "").strip().lower()
        if role not in {"condition", "stimulus", "form", "order", "item_set", "other"}:
            raise ValueError(f"material_variants[{position}] has invalid role: {role}")
        if variant.get("is_alternative_version") is not True:
            raise ValueError(
                f"material_variants[{position}] must be a genuine alternative version"
            )
        variant_refs = variant.get("evidence_refs")
        if not isinstance(variant_refs, list) or not variant_refs:
            raise ValueError(f"material_variants[{position}] has no evidence_refs")
        invalid_variant_refs = [
            str(ref) for ref in variant_refs if str(ref) not in valid_refs
        ]
        if invalid_variant_refs:
            raise ValueError(
                f"material_variants[{position}] has invalid evidence refs: "
                f"{invalid_variant_refs}"
            )


def _prune_invalid_study_evidence_refs(
    payload: Dict[str, Any],
    *,
    valid_refs: set[str],
) -> None:
    """Drop isolated citation typos only when the same claim keeps valid evidence."""
    repairs: List[Dict[str, Any]] = []

    def prune(values: Any, *, path: str, require_nonempty: bool) -> Any:
        if not isinstance(values, list):
            return values
        normalized = [str(value) for value in values if str(value)]
        kept = [value for value in normalized if value in valid_refs]
        removed = [value for value in normalized if value not in valid_refs]
        if removed and require_nonempty and not kept:
            raise ValueError(f"{path} has only invalid evidence refs: {removed}")
        if removed:
            repairs.append({"path": path, "removed_evidence_refs": removed})
        return kept

    payload["evidence_refs"] = prune(
        payload.get("evidence_refs"),
        path="evidence_refs",
        require_nonempty=True,
    )
    field_evidence = payload.get("field_evidence")
    if isinstance(field_evidence, dict):
        for field, values in list(field_evidence.items()):
            field_evidence[field] = prune(
                values,
                path=f"field_evidence.{field}",
                require_nonempty=bool(values),
            )
    for position, barrier in enumerate(payload.get("simulation_barriers") or []):
        if isinstance(barrier, dict):
            barrier["evidence_refs"] = prune(
                barrier.get("evidence_refs"),
                path=f"simulation_barriers[{position}].evidence_refs",
                require_nonempty=True,
            )
    for position, variant in enumerate(payload.get("material_variants") or []):
        if isinstance(variant, dict):
            variant["evidence_refs"] = prune(
                variant.get("evidence_refs"),
                path=f"material_variants[{position}].evidence_refs",
                require_nonempty=True,
            )
    payload["_evidence_ref_repairs"] = repairs


def cached_json_call(
    llm_client: Any,
    prompt: str,
    *,
    cache_path: Optional[Path],
    prompt_version: str,
    timeout: Optional[float],
    max_tokens: int,
    force: bool,
    max_attempts: int = 2,
    validator: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    model = str(getattr(llm_client, "model", "unknown"))
    cache_key = hashlib.sha256(
        f"{prompt_version}\n{model}\n{prompt}".encode("utf-8")
    ).hexdigest()
    keyed_cache_path = _keyed_cache_path(cache_path, cache_key)
    if keyed_cache_path is not None and keyed_cache_path.exists() and not force:
        try:
            cached = json.loads(keyed_cache_path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == cache_key and isinstance(cached.get("payload"), dict):
                payload = cached["payload"]
                if validator is not None:
                    validator(payload)
                return payload
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    last_error: Optional[BaseException] = None
    current_prompt = prompt
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        payload: Optional[Dict[str, Any]] = None
        try:
            response = _generate_content(
                llm_client,
                current_prompt,
                timeout=timeout,
                max_tokens=max_tokens if attempt == 1 else min(max_tokens * 2, 32000),
            )
            payload = _loads_json(response)
            if validator is not None:
                validator(payload)
            if keyed_cache_path is not None:
                keyed_cache_path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(
                    keyed_cache_path,
                    {
                        "version": prompt_version,
                        "model": model,
                        "cache_key": cache_key,
                        "payload": payload,
                    },
                )
            return payload
        except Exception as exc:
            last_error = exc
            if keyed_cache_path is not None and payload is not None:
                invalid_path = keyed_cache_path.with_name(
                    f"{keyed_cache_path.stem}.attempt_{attempt}.invalid"
                    f"{keyed_cache_path.suffix or '.json'}"
                )
                _write_json(
                    invalid_path,
                    {
                        "version": prompt_version,
                        "model": model,
                        "cache_key": cache_key,
                        "attempt": attempt,
                        "validation_error": f"{type(exc).__name__}: {exc}",
                        "payload": payload,
                    },
                )
            current_prompt = (
                prompt
                + "\n\nThe prior response failed validation. Return only the requested valid JSON object."
            )
            if attempt < max_attempts:
                time.sleep(1.0)
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM JSON call failed without an exception")


def _keyed_cache_path(cache_path: Optional[Path], cache_key: str) -> Optional[Path]:
    if cache_path is None:
        return None
    suffix = cache_path.suffix or ".json"
    return cache_path.with_name(
        f"{cache_path.stem}.{cache_key[:16]}{suffix}"
    )


def _generate_content(
    llm_client: Any,
    prompt: str,
    *,
    timeout: Optional[float],
    max_tokens: int,
) -> str:
    try:
        response = llm_client.generate_content(
            prompt=prompt,
            temperature=0.0,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    except TypeError:
        try:
            response = llm_client.generate_content(
                prompt=prompt,
                timeout=timeout,
                max_tokens=max_tokens,
            )
        except TypeError:
            try:
                response = llm_client.generate_content(prompt=prompt, timeout=timeout)
            except TypeError:
                response = llm_client.generate_content(prompt=prompt)
    if response is None:
        raise ValueError("LLM returned None")
    text = str(response)
    if not text.strip():
        raise ValueError(
            "LLM returned an empty response; the completion budget may have been consumed by reasoning"
        )
    return text


def _loads_json(text: str) -> Dict[str, Any]:
    value = str(text or "").strip()
    if "```json" in value:
        value = value.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in value:
        value = value.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group())
    if not isinstance(payload, dict):
        raise ValueError(f"LLM response must be a JSON object, got {type(payload)}")
    return payload


def _split_text(text: str, max_chars: int) -> List[str]:
    value = str(text or "").strip()
    if not value:
        return [""]
    if len(value) <= max_chars:
        return [value]
    parts: List[str] = []
    start = 0
    overlap = min(400, max_chars // 8)
    while start < len(value):
        end = min(len(value), start + max_chars)
        if end < len(value):
            boundary = max(
                value.rfind("\n", start + max_chars // 2, end),
                value.rfind(". ", start + max_chars // 2, end),
            )
            if boundary > start:
                end = boundary + 1
        parts.append(value[start:end].strip())
        if end >= len(value):
            break
        start = max(start + 1, end - overlap)
    return parts


def _feedback_text(value: Optional[Dict[str, Any]]) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    compact = {
        key: item
        for key, item in value.items()
        if key
        in {
            "missing_studies",
            "split_merge_corrections",
            "comparison_group_corrections",
            "study_field_corrections",
            "eligibility_corrections",
        }
        and item
    }
    if not compact:
        return ""
    return "\nVERIFIER FEEDBACK TO RECHECK:\n" + json.dumps(
        compact,
        ensure_ascii=False,
        indent=2,
    )


def _feedback_for_window(
    value: Optional[Dict[str, Any]],
    window: DiscoveryWindow,
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    block_ids = set(window.block_ids)
    selected: Dict[str, List[Any]] = {}
    for key in (
        "missing_studies",
        "split_merge_corrections",
        "comparison_group_corrections",
        "eligibility_corrections",
    ):
        items = value.get(key) if isinstance(value.get(key), list) else []
        has_grounded_items = any(
            isinstance(item, dict) and item.get("evidence_block_ids")
            for item in items
        )
        for item in items:
            if isinstance(item, dict) and item.get("evidence_block_ids"):
                refs = {str(ref) for ref in item.get("evidence_block_ids") or []}
                if refs & block_ids:
                    selected.setdefault(key, []).append(item)
            elif not has_grounded_items:
                selected.setdefault(key, []).append(item)
    return selected or None


def _feedback_for_study(
    value: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    candidate_keys = {
        canonical_sub_study_id(candidate.get("study_id")),
        canonical_sub_study_id(candidate.get("reported_label")),
        canonical_sub_study_id(candidate.get("study_name")),
        *(
            canonical_sub_study_id(alias)
            for alias in candidate.get("aliases") or []
            if alias
        ),
    }
    candidate_refs = {
        str(ref) for ref in candidate.get("evidence_block_ids") or [] if str(ref)
    }
    selected: Dict[str, List[Any]] = {}
    for key in (
        "missing_studies",
        "split_merge_corrections",
        "study_field_corrections",
        "eligibility_corrections",
    ):
        for item in value.get(key) or []:
            if not isinstance(item, dict):
                continue
            study_key = canonical_sub_study_id(item.get("study"))
            item_refs = {
                str(ref) for ref in item.get("evidence_block_ids") or [] if str(ref)
            }
            if study_key in candidate_keys or bool(candidate_refs & item_refs):
                selected.setdefault(key, []).append(item)
    return selected or None


def _merge_metadata(values: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    titles = [
        str(value.get("paper_title") or "").strip()
        for value in values
        if str(value.get("paper_title") or "").strip()
    ]
    abstracts = [
        str(value.get("paper_abstract") or "").strip()
        for value in values
        if str(value.get("paper_abstract") or "").strip()
    ]
    author_lists = [
        [str(author).strip() for author in value.get("paper_authors") or [] if str(author).strip()]
        for value in values
        if isinstance(value.get("paper_authors"), list)
    ]
    return {
        "paper_title": max(titles, key=len) if titles else "",
        "paper_authors": max(author_lists, key=len) if author_lists else [],
        "paper_abstract": max(abstracts, key=len) if abstracts else "",
    }


def _valid_refs(values: Any, valid_refs: set[str]) -> List[str]:
    if not isinstance(values, list):
        return []
    return list(
        dict.fromkeys(str(value) for value in values if str(value) in valid_refs)
    )


def _context_summary(context: EvidenceContext) -> Dict[str, Any]:
    return {
        "mode": context.mode,
        "block_ids": list(context.block_ids),
        "pages": list(context.pages),
        "facets": {key: list(value) for key, value in context.facets.items()},
        "source_chars": context.source_chars,
        "context_chars": context.context_chars,
    }


def _normalize_material_variants(
    extracted_values: Any,
    *,
    valid_refs: set[str],
) -> List[Dict[str, Any]]:
    allowed_roles = {"condition", "stimulus", "form", "order", "item_set", "other"}
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for value in extracted_values if isinstance(extracted_values, list) else []:
        if not isinstance(value, dict):
            continue
        if value.get("is_alternative_version") is not True:
            continue
        label = str(value.get("label") or "").strip()
        if not label:
            continue
        key = _relation_label_key(label)
        if key not in merged:
            merged[key] = {
                "variant_id": "",
                "label": label,
                "role": "other",
                "is_alternative_version": True,
                "assignment": None,
                "participant_task_difference": None,
                "sample": None,
                "quantitative_target": None,
                "evidence_refs": [],
            }
            order.append(key)
        target = merged[key]
        requested_id = str(value.get("variant_id") or "").strip()
        if requested_id:
            target["variant_id"] = requested_id
        role = str(value.get("role") or "").strip().lower()
        if role in allowed_roles:
            target["role"] = role
        for field in (
            "assignment",
            "participant_task_difference",
            "sample",
            "quantitative_target",
        ):
            text = _optional_text(value.get(field))
            if text:
                target[field] = text
        target["evidence_refs"] = list(
            dict.fromkeys(
                [
                    *target["evidence_refs"],
                    *_valid_refs(
                        value.get("evidence_refs") or value.get("evidence_block_ids"),
                        valid_refs,
                    ),
                ]
            )
        )

    used_ids: set[str] = set()
    output: List[Dict[str, Any]] = []
    for position, key in enumerate(order, start=1):
        variant = merged[key]
        requested_id = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(variant.get("variant_id") or variant.get("label") or "").lower(),
        ).strip("_") or f"variant_{position}"
        variant_id = requested_id
        suffix = 2
        while variant_id in used_ids:
            variant_id = f"{requested_id}_{suffix}"
            suffix += 1
        used_ids.add(variant_id)
        variant["variant_id"] = variant_id
        output.append(variant)
    return output


def _normalize_empirical_support(value: Any) -> Dict[str, str]:
    raw = value if isinstance(value, dict) else {}
    fields = (
        "own_sample_or_assignment",
        "participant_facing_task",
        "quantitative_result",
    )
    return {
        field: (
            str(raw.get(field) or "unclear").strip().lower()
            if str(raw.get(field) or "").strip().lower() in {"yes", "no", "unclear"}
            else "unclear"
        )
        for field in fields
    }


def _normalize_simulation_barriers(
    value: Any,
    *,
    valid_refs: set[str],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for barrier in value if isinstance(value, list) else []:
        if not isinstance(barrier, dict):
            continue
        kind = str(barrier.get("kind") or "").strip().lower()
        description = str(barrier.get("description") or "").strip()
        if kind not in SIMULATION_BARRIER_KINDS or not description:
            continue
        key = (kind, _relation_label_key(description))
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "kind": kind,
                "description": description,
                "affects_primary_target": barrier.get("affects_primary_target") is True,
                "evidence_refs": _valid_refs(barrier.get("evidence_refs"), valid_refs),
            }
        )
    return output


def _apply_simulation_barrier_gate(record: Dict[str, Any]) -> None:
    barriers = [
        barrier
        for barrier in record.get("simulation_barriers") or []
        if isinstance(barrier, dict) and barrier.get("affects_primary_target") is True
    ]
    if not barriers:
        return
    record["replicable"] = "NO"
    reasons = record.get("exclusion_reasons")
    reasons = [
        str(reason).strip()
        for reason in reasons if str(reason).strip()
    ] if isinstance(reasons, list) else []
    reason = "The original quantitative target requires an unsupported execution mode."
    if reason not in reasons:
        reasons.append(reason)
    record["exclusion_reasons"] = reasons


def _optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _first_text(values: Iterable[Any]) -> Optional[str]:
    for value in values:
        text = _optional_text(value)
        if text:
            return text
    return None


def _confidence(value: Any, *, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(fallback)
    return round(min(1.0, max(0.0, number)), 3)


def _mean_confidence(experiments: Sequence[Dict[str, Any]]) -> float:
    if not experiments:
        return 0.0
    values = [_confidence(item.get("extraction_confidence")) for item in experiments]
    return round(sum(values) / len(values), 3)


def _page_label(pages: Sequence[int]) -> str:
    if not pages:
        return "unknown"
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}-{pages[-1]}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
