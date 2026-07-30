from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from generation_pipeline.identifiers import canonical_sub_study_id
from generation_pipeline.pdf.models import DocumentBlock, EvidenceContext, ParsedPdfDocument
from generation_pipeline.utils.pdf_chunker import Chunk, retrieve


FULL_DOCUMENT_MAX_CHARS = 120000
RETRIEVAL_CONTEXT_MAX_CHARS = 64000
FACET_K = 4

FACETS: Dict[str, Dict[str, Any]] = {
    "sample": {
        "terms": ["participants", "sample", "recruited", "excluded", "demographic"],
        "description": "sample size, recruitment, exclusions, demographics, and analyzed population",
    },
    "design": {
        "terms": ["method", "design", "randomly assigned", "between subjects", "within subjects"],
        "description": "experimental design, assignment, arms, factors, and all condition levels",
    },
    "procedure": {
        "terms": ["procedure", "participants were asked", "instructions", "task", "completed"],
        "description": "participant procedure, task order, timing, and participant-facing instructions",
    },
    "stimuli": {
        "terms": ["materials", "stimulus", "stimuli", "vignette", "scenario", "message"],
        "description": "complete stimuli, vignettes, scenarios, messages, and condition-specific text",
    },
    "items": {
        "terms": ["measure", "measures", "question", "items", "rated", "indicated"],
        "description": "complete response instrument and every participant response item",
    },
    "options": {
        "terms": ["scale", "ranging from", "anchors", "response options", "choose", "percent"],
        "description": "response options, scale bounds, anchors, matrix rows and columns",
    },
    "tables_figures": {
        "terms": ["table", "figure", "choice set", "product", "alternative", "payoff"],
        "description": "tables, figures, forms, choice sets, payoff matrices, and visual task content",
    },
    "appendix": {
        "terms": ["appendix", "supplement", "questionnaire", "instrument", "full wording"],
        "description": "appendix or supplement containing exact materials and item wording",
    },
    "results": {
        "terms": ["results", "analysis", "regression", "significant", "p <", "p ="],
        "description": "quantitative findings, statistical tests, effect directions, and result tables",
    },
}

GAP_TO_FACETS: Dict[str, Sequence[str]] = {
    "instructions": ("procedure", "stimuli"),
    "stimuli": ("stimuli", "appendix"),
    "items": ("items", "appendix"),
    "response_options": ("options", "items", "tables_figures"),
    "conditions": ("design", "stimuli", "procedure"),
    "visual_material": ("tables_figures",),
    "source_evidence": tuple(FACETS),
    "findings": ("design", "sample", "results", "tables_figures"),
}


class PdfEvidenceIndex:
    def __init__(self, document: ParsedPdfDocument):
        self.document = document
        self._blocks = sorted(document.blocks, key=lambda block: block.order)
        self._by_id = document.block_map()
        self._chunks = [self._as_chunk(block) for block in self._blocks]
        self._chunk_to_block = {chunk.id: block for chunk, block in zip(self._chunks, self._blocks)}

    @property
    def block_ids(self) -> set[str]:
        return set(self._by_id)

    def context_for_study(
        self,
        study: Dict[str, Any],
        *,
        stage1_json: Optional[Dict[str, Any]] = None,
        gaps: Optional[Iterable[str]] = None,
        allow_full_document: bool = True,
        anchor_refs: Optional[Iterable[str]] = None,
        anchor_radius: int = 1,
        use_facet_retrieval: bool = True,
        max_chars: Optional[int] = None,
    ) -> EvidenceContext:
        selected_facets = self._facets_for_gaps(gaps)
        if (
            allow_full_document
            and gaps is None
            and self.document.text_chars <= FULL_DOCUMENT_MAX_CHARS
        ):
            blocks = self._blocks
            text, included = self._render_with_budget(
                blocks,
                max_chars or FULL_DOCUMENT_MAX_CHARS + 24000,
            )
            return EvidenceContext(
                text=text,
                mode="full_document",
                block_ids=[block.block_id for block in included],
                pages=sorted({page for block in included for page in range(block.page_start, block.page_end + 1)}),
                facets={facet: [block.block_id for block in included] for facet in selected_facets},
                source_chars=self.document.text_chars,
                context_chars=len(text),
            )

        study_query = self._study_query(study, stage1_json)
        selected: Dict[str, DocumentBlock] = {}
        prioritized: List[DocumentBlock] = []
        anchor_blocks: List[DocumentBlock] = []
        for ref in anchor_refs or []:
            block = self._by_id.get(str(ref))
            if block is None:
                continue
            if block.block_id not in selected:
                selected[block.block_id] = block
                prioritized.append(block)
                anchor_blocks.append(block)
        # Exact citations are authoritative. Add their surrounding blocks only
        # after every citation has been prioritized so a large neighbor cannot
        # consume the budget before a later cited block is rendered.
        for block in anchor_blocks:
            for neighbor in self._neighbors(block, radius=max(0, int(anchor_radius))):
                selected[neighbor.block_id] = neighbor
        facet_blocks: Dict[str, List[str]] = {}
        facets_to_retrieve = selected_facets if use_facet_retrieval or not selected else []
        for facet in facets_to_retrieve:
            config = FACETS[facet]
            query = f"{study_query} {config['description']} {' '.join(config['terms'])}"
            ranked = retrieve(
                self._chunks,
                query,
                k=FACET_K,
                keywords=config["terms"],
                use_bm25=True,
            )
            ids: List[str] = []
            for chunk, _ in ranked:
                block = self._chunk_to_block[chunk.id]
                for neighbor in self._neighbors(block, radius=1):
                    selected[neighbor.block_id] = neighbor
                    if neighbor.block_id not in ids:
                        ids.append(neighbor.block_id)
            facet_blocks[facet] = ids

        remaining = sorted(
            (block for block in selected.values() if block not in prioritized),
            key=lambda block: block.order,
        )
        ordered = [*prioritized, *remaining]
        text, included = self._render_with_budget(
            ordered,
            max_chars or RETRIEVAL_CONTEXT_MAX_CHARS,
        )
        included_ids = {block.block_id for block in included}
        facet_blocks = {
            facet: [block_id for block_id in block_ids if block_id in included_ids]
            for facet, block_ids in facet_blocks.items()
        }
        mode = "facet_retrieval"
        if prioritized:
            mode = "anchored_facet_retrieval" if facets_to_retrieve else "anchored_retrieval"
        return EvidenceContext(
            text=text,
            mode=mode,
            block_ids=[block.block_id for block in included],
            pages=sorted({page for block in included for page in range(block.page_start, block.page_end + 1)}),
            facets=facet_blocks,
            source_chars=self.document.text_chars,
            context_chars=len(text),
        )

    def context_for_refs(self, refs: Iterable[str], *, max_chars: int = 36000) -> str:
        selected: Dict[str, DocumentBlock] = {}
        for ref in refs:
            block = self._by_id.get(str(ref))
            if block is None:
                continue
            for neighbor in self._neighbors(block, radius=1):
                selected[neighbor.block_id] = neighbor
        text, _ = self._render_with_budget(sorted(selected.values(), key=lambda block: block.order), max_chars)
        return text

    def _facets_for_gaps(self, gaps: Optional[Iterable[str]]) -> List[str]:
        if gaps is None:
            return list(FACETS)
        facets: List[str] = []
        for gap in gaps:
            for facet in GAP_TO_FACETS.get(str(gap), tuple(FACETS)):
                if facet not in facets:
                    facets.append(facet)
        return facets or list(FACETS)

    def _study_query(self, study: Dict[str, Any], stage1_json: Optional[Dict[str, Any]]) -> str:
        values: List[Any] = []
        for key in ("study", "study_id", "experiment_id", "study_name", "design", "sample"):
            values.append(study.get(key))
        for effect in study.get("effects", []) or []:
            if isinstance(effect, dict):
                values.extend([effect.get("IV"), effect.get("DV"), effect.get("materials_notes"), effect.get("table_or_page_location")])
        for finding in study.get("findings", []) or []:
            if isinstance(finding, dict):
                values.extend([finding.get("IV"), finding.get("DV")])
        hint = _stage1_experiment(stage1_json, study)
        if hint:
            values.extend(
                [
                    hint.get("experiment_name"),
                    hint.get("design_type"),
                    hint.get("conditions_or_factors"),
                    hint.get("participant_task"),
                    hint.get("input"),
                    hint.get("output"),
                    hint.get("candidate_source_hints"),
                ]
            )
        return " ".join(_compact(value) for value in values if value not in (None, ""))[:12000]

    def _as_chunk(self, block: DocumentBlock) -> Chunk:
        heading = " > ".join(block.section_path)
        text = f"{heading}\n{block.text}" if heading else block.text
        return Chunk(
            id=block.order,
            text=text,
            page_start=block.page_start,
            page_end=block.page_end,
            char_start=block.order,
            source_path=self.document.source_file,
            source_kind=block.block_type,
        )

    def _neighbors(self, block: DocumentBlock, *, radius: int) -> List[DocumentBlock]:
        try:
            index = self._blocks.index(block)
        except ValueError:
            return [block]
        start = max(0, index - radius)
        end = min(len(self._blocks), index + radius + 1)
        return self._blocks[start:end]

    def _render_with_budget(
        self,
        blocks: Iterable[DocumentBlock],
        max_chars: int,
    ) -> tuple[str, List[DocumentBlock]]:
        parts: List[str] = []
        included: List[DocumentBlock] = []
        used = 0
        for block in blocks:
            pages = str(block.page_start) if block.page_start == block.page_end else f"{block.page_start}-{block.page_end}"
            section = " > ".join(block.section_path) or "(unsectioned)"
            rendered = (
                f"[Block {block.block_id} | page {pages} | type={block.block_type} | section={section}]\n"
                f"{block.text.strip()}"
            )
            if used + len(rendered) > max_chars:
                continue
            parts.append(rendered)
            included.append(block)
            used += len(rendered) + 2
        return "\n\n".join(parts), included


def _stage1_experiment(stage1_json: Optional[Dict[str, Any]], study: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(stage1_json, dict):
        return {}
    wanted = {
        canonical_sub_study_id(study.get(key))
        for key in ("study", "study_id", "experiment_id", "study_name")
        if study.get(key)
    }
    for experiment in stage1_json.get("experiments", []) or []:
        if not isinstance(experiment, dict):
            continue
        keys = {
            canonical_sub_study_id(experiment.get(key))
            for key in ("study_id", "experiment_id", "study_name", "experiment_name")
            if experiment.get(key)
        }
        if wanted & keys:
            return experiment
    return {}


def _compact(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value or "")
    return re.sub(r"\s+", " ", text).strip()
