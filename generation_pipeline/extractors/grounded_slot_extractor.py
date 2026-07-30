from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generation_pipeline.utils.pdf_chunker import (
    Chunk,
    build_chunks,
    chunk_pdf,
    format_evidence,
    retrieve,
)
from generation_pipeline.utils.pdf_extractor import extract_pdf_text
from generation_pipeline.verification.verbatim_verifier import (
    best_partial_ratio,
    normalize_for_match,
)
from generation_pipeline.verification.schema_validator import SLOT_NAMES
from src.llm.helpers import generate_json

DEFAULT_THRESHOLD = 90.0
PDF_TEXT_MAX_CHARS = 400000

# Slot-specific retrieval keywords + a short description of what the span is.
SLOT_PROFILE: dict[str, dict[str, Any]] = {
    "materials": {
        "desc": "the exact stimulus text shown to participants (e.g. the vignette, scenario, "
        "passage, or prompt they read)",
        "keywords": ["vignette", "scenario", "stimuli", "stimulus", "material", "passage",
                     "presented", "read", "described", "condition", "instructions"],
    },
    "manipulation": {
        "desc": "the exact description of what was manipulated across conditions "
        "(the independent-variable operationalization)",
        "keywords": ["manipulat", "condition", "randomly assigned", "random assignment",
                     "varied", "between-subjects", "within-subjects", "we manipulated", "we varied"],
    },
    "items": {
        "desc": "the exact dependent-variable measure: the question wording AND the response "
        "scale / anchors / options",
        "keywords": ["scale", "item", "items", "measured", "rated", "response", "responses",
                     "slider", "sliding", "likert", "anchor", "questionnaire", "question",
                     "0 =", "1 =", "7 =", "point", "percentage"],
    },
}


@dataclass
class SlotResult:
    status: str
    content: Optional[str]
    score: Optional[float]
    action: str  # "grounded_verbatim" | "kept_paraphrased" | "downgraded" | "not_present" | "unchanged"
    evidence_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score": round(self.score, 2) if isinstance(self.score, float) else self.score,
            "action": self.action,
            "evidence_pages": self.evidence_pages,
        }


@dataclass(frozen=True)
class GroundingTask:
    order: int
    study_index: int
    effect_index: int
    slot: str
    path: str
    study_label: str
    iv: str
    dv: str
    location_hint: str
    materials_notes: str
    prev_status: Optional[str]
    prev_content: Any
    had_content: bool


class GroundedSlotExtractor:
    """Evidence-grounded extractor/repairer for materials/manipulation/items slots."""

    def __init__(
        self,
        client,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        k: int = 8,
        chunk_size: int = 2500,
        overlap: int = 600,
        retries: int = 1,
        temperature: float = 0.0,
        slot_timeout: float | None = 60.0,
        max_workers: int = 1,
    ):
        self.client = client
        self.threshold = threshold
        self.k = k
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.retries = retries
        self.temperature = temperature
        self.slot_timeout = slot_timeout
        self.max_workers = max(int(max_workers or 1), 1)
        self._chunks: list[Chunk] = []
        self._source_norm: str = ""
        self._source_paths_by_study: dict[str, set[str]] = {}
        self._source_paths_by_study_non_prereg: dict[str, set[str]] = {}
        self._shared_source_paths: set[str] = set()

    # preparation
    def prepare(self, pdf_path: Path) -> None:
        """Parse + chunk the PDF once; cache normalized raw source for verification."""
        pdf_path = Path(pdf_path)
        self._chunks = chunk_pdf(pdf_path, chunk_size=self.chunk_size, overlap=self.overlap)
        raw = extract_pdf_text(pdf_path, max_chars=PDF_TEXT_MAX_CHARS)
        self._source_norm = normalize_for_match(raw)
        self._source_paths_by_study = {}
        self._source_paths_by_study_non_prereg = {}
        self._shared_source_paths = set()

    def prepare_from_text(self, text: str) -> None:
        """Chunk pre-extracted text (e.g. combined_sources.txt from OSF files).

        Use this when the evidence is supplementary source material rather than
        a PDF — the same chunking + lexical retrieval logic applies.
        """
        # Treat the whole block as a single "page" for provenance purposes.
        pages = [(1, text)]
        self._chunks = build_chunks(pages, chunk_size=self.chunk_size, overlap=self.overlap)
        self._source_norm = normalize_for_match(text)
        self._source_paths_by_study = {}
        self._source_paths_by_study_non_prereg = {}
        self._shared_source_paths = set()

    def prepare_from_sources(self, records: list[dict[str, str]]) -> None:
        """Chunk source files separately, preserving source/study provenance.

        Each record must contain ``source_path`` and ``text``.  Optional
        ``text_path`` is ignored here but can be useful in reports.
        """
        chunks: list[Chunk] = []
        source_texts: list[str] = []
        by_study: dict[str, set[str]] = {}
        by_study_non_prereg: dict[str, set[str]] = {}
        shared: set[str] = set()

        for record in records:
            source_path = str(record.get("source_path") or record.get("text_path") or "").strip()
            text = record.get("text") or ""
            if not source_path or not text.strip():
                continue
            study_keys = _study_keys_from_source_path(source_path)
            source_kind = _source_kind_from_path(source_path, study_keys)
            built = build_chunks(
                [(1, text)],
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                source_path=source_path,
                source_kind=source_kind,
                study_keys=study_keys,
            )
            for chunk in built:
                chunk.id = len(chunks)
                chunks.append(chunk)
            source_texts.append(text)
            if study_keys:
                for key in study_keys:
                    by_study.setdefault(key, set()).add(source_path)
                    if source_kind != "prereg":
                        by_study_non_prereg.setdefault(key, set()).add(source_path)
            elif source_kind == "shared":
                shared.add(source_path)

        self._chunks = chunks
        self._source_norm = normalize_for_match("\n\n".join(source_texts))
        self._source_paths_by_study = by_study
        self._source_paths_by_study_non_prereg = by_study_non_prereg
        self._shared_source_paths = shared

    # -- core-------
    def extract_slot(
        self,
        *,
        study: str,
        iv: str,
        dv: str,
        slot: str,
        location_hint: str = "",
        materials_notes: str = "",
        prev_status: Optional[str] = None,
        prev_content: Optional[str] = None,
        paraphrase_threshold: Optional[float] = None,
        assemble: bool = False,
    ) -> SlotResult:
        """Extract a slot from prepared source text.

        Args:
            paraphrase_threshold: If provided, a second tier below ``self.threshold``
                where extracted content is accepted as ``paraphrased`` rather than
                discarded.  Useful for supplementary OSF files (DOCX/QSF) where the
                text format differs from the paper (e.g. Qualtrics piped-text
                variables, pre-registration prose).  Typical value: 75.
            assemble: When True (and ``paraphrase_threshold`` is set), first try a
                MULTI-SPAN assembly: ask the model to gather every verbatim span
                that together constitutes the material (e.g. donation intro + game
                rules + per-condition manipulation + DV items, which in a Qualtrics
                survey live in different blocks/conditions), then verify EACH span
                individually against the source.  This is the PaperQA-style
                "gather evidence then compose" + LangExtract-style per-span
                grounding path.  Falls back to single-span extraction.
        """
        if slot not in SLOT_PROFILE:
            raise ValueError(f"Unknown slot: {slot}")
        if not self._chunks:
            raise RuntimeError("Call prepare(pdf_path) before extract_slot().")

        profile = SLOT_PROFILE[slot]

        if isinstance(prev_content, str) and prev_content.strip() and prev_status == "verbatim":
            pre = best_partial_ratio(normalize_for_match(prev_content), self._source_norm)
            if pre >= self.threshold:
                return SlotResult(
                    status="verbatim", content=prev_content, score=pre, action="kept_verbatim_prechecked"
                )

        draft = prev_content if isinstance(prev_content, str) else ""
        query_parts = (iv, dv, slot, profile["desc"], location_hint, materials_notes, draft[:600])
        if not self._has_source_metadata():
            query_parts = (study, *query_parts)
        query = " ".join(str(x) for x in query_parts if x)
        # Multi-span assembly retrieves a WIDER candidate set so pieces scattered across blocks/conditions all enter the evidence window.
        eff_k = min(self.k * 3, len(self._chunks)) if assemble else self.k
        scored = self._retrieve_scoped(study, query, k=eff_k, keywords=profile["keywords"])
        if not scored:
            if self._has_source_metadata():
                return SlotResult(
                    status="source_missing",
                    content=None,
                    score=None,
                    action="source_missing_no_scoped_evidence",
                )
            return self._fallback(prev_status, prev_content, score=None, action="not_present")

        evidence = format_evidence(scored)
        evidence_pages = sorted({c.page_start for c, _ in scored} | {c.page_end for c, _ in scored})

        # Multi-span assembly
        if assemble and paraphrase_threshold is not None:
            assembled = self._assemble_spans(
                slot, study, iv, dv, profile["desc"], evidence,
                paraphrase_threshold=paraphrase_threshold,
                evidence_pages=evidence_pages,
            )
            if assembled is not None:
                return assembled
            if slot == "materials" and self._only_prereg_sources_for_study(study):
                return SlotResult(
                    status="source_missing",
                    content=None,
                    score=0.0,
                    action="source_missing_only_prereg_no_material_span",
                    evidence_pages=evidence_pages,
                )
        feedback = ""
        best_score = 0.0
        best_quote: str = ""
        for attempt in range(self.retries + 1):
            parsed = self._ask(slot, study, dv, profile["desc"], evidence, feedback, draft=draft)
            if parsed.get("error"):
                return SlotResult(
                    status=prev_status or "not_in_paper",
                    content=prev_content,
                    score=None,
                    action="skipped_api_error",
                )
            if not parsed.get("found"):
                break
            quote = (parsed.get("quote") or "").strip()
            if not quote:
                break
            score = best_partial_ratio(normalize_for_match(quote), self._source_norm)
            if score > best_score:
                best_score = score
                best_quote = quote
            if score >= self.threshold:
                return SlotResult(
                    status="verbatim",
                    content=quote,
                    score=score,
                    action="grounded_verbatim",
                    evidence_pages=evidence_pages,
                )
            feedback = (
                "Your previous answer was NOT found verbatim in the evidence "
                f"(match score {score:.0f} < {self.threshold:.0f}). You MUST copy characters "
                "exactly as they appear in the evidence passages, with no edits. If the text is "
                "not present in the evidence, set found=false."
            )

        # Paraphrase tier A: verbatim extraction scored below strict threshold
        # but above the paraphrase threshold — the text is clearly from the source.
        if (
            paraphrase_threshold is not None
            and best_quote
            and best_score >= paraphrase_threshold
        ):
            return SlotResult(
                status="paraphrased",
                content=best_quote,
                score=best_score,
                action="grounded_paraphrased",
                evidence_pages=evidence_pages,
            )

        if paraphrase_threshold is not None:
            described = self._ask_describe(slot, study, dv, profile["desc"], evidence)
            if described.get("error"):
                return SlotResult(
                    status=prev_status or "not_in_paper",
                    content=prev_content,
                    score=None,
                    action="skipped_api_error",
                )
            desc_text = (described.get("description") or "").strip()
            if desc_text:
                desc_score = best_partial_ratio(normalize_for_match(desc_text), self._source_norm)
                if desc_score >= paraphrase_threshold:
                    return SlotResult(
                        status="paraphrased",
                        content=desc_text,
                        score=desc_score,
                        action="grounded_paraphrased_assembled",
                        evidence_pages=evidence_pages,
                    )

        return self._fallback(prev_status, prev_content, score=best_score, action="downgraded")

    def _has_source_metadata(self) -> bool:
        return any(chunk.source_path for chunk in self._chunks)

    def _retrieve_scoped(
        self,
        study: str,
        query: str,
        *,
        k: int,
        keywords: list[str],
    ) -> list[tuple[Chunk, float]]:
        """Metadata-filtered retrieval: exact study sources, then shared, then global."""
        study_key = _study_key_from_label(study)
        if not self._has_source_metadata() or not study_key:
            return retrieve(self._chunks, query, k=k, keywords=keywords)

        tiers: list[set[str] | None] = []
        exact_non_prereg = self._source_paths_by_study_non_prereg.get(study_key, set())
        exact_all = self._source_paths_by_study.get(study_key, set())
        exact_prereg = exact_all - exact_non_prereg
        if exact_non_prereg:
            tiers.append(exact_non_prereg)
        if exact_prereg:
            tiers.append(exact_prereg)
        if self._shared_source_paths:
            tiers.append(self._shared_source_paths)
        tiers.append(None)

        seen: set[str] = set()
        for allowed in tiers:
            key = "__global__" if allowed is None else "|".join(sorted(allowed))
            if key in seen:
                continue
            seen.add(key)
            scored = retrieve(
                self._chunks,
                query,
                k=k,
                keywords=keywords,
                allowed_sources=allowed,
                use_bm25=True,
            )
            if scored:
                return scored
        return []

    def _only_prereg_sources_for_study(self, study: str) -> bool:
        study_key = _study_key_from_label(study)
        if not study_key:
            return False
        exact_all = self._source_paths_by_study.get(study_key, set())
        if not exact_all:
            return False
        exact_non_prereg = self._source_paths_by_study_non_prereg.get(study_key, set())
        return not exact_non_prereg

    def _fallback(
        self, prev_status: Optional[str], prev_content: Optional[str], *, score, action: str
    ) -> SlotResult:
        """Honest downgrade: never present unverifiable text as verbatim."""
        if isinstance(prev_content, str) and prev_content.strip():
            keep_status = prev_status if prev_status in {"cited_scale", "osf_only"} else "paraphrased"
            return SlotResult(
                status=keep_status,
                content=prev_content,
                score=score,
                action="kept_paraphrased" if keep_status == "paraphrased" else "unchanged",
            )
        return SlotResult(status="not_in_paper", content=None, score=score, action=action)

    def _ask(
        self, slot: str, study: str, dv: str, desc: str, evidence: str, feedback: str, draft: str = ""
    ) -> dict[str, Any]:
        system = (
            "You are a meticulous research assistant extracting VERBATIM text from a scientific "
            "paper. You never paraphrase, summarize, normalize, or invent text. You only copy "
            "characters exactly as they appear in the provided evidence."
        )
        fb = f"\n\nIMPORTANT FEEDBACK:\n{feedback}\n" if feedback else ""
        draft_block = ""
        if draft and draft.strip():
            draft_block = (
                "\n\nAPPROXIMATE DRAFT (may be paraphrased or contain errors) — use it ONLY to "
                "LOCATE the right passage; do NOT copy from it, copy from the EVIDENCE:\n"
                f'"""{draft.strip()[:1500]}"""\n'
            )
        prompt = f"""TASK: From the evidence passages below (extracted from the paper), copy {desc}
for "{study}" (dependent variable: {dv}).

Rules:
- Copy the EXACT contiguous span of text from the evidence. Do not paraphrase, shorten,
  reorder, fix typos, or merge separate passages.
- Prefer the single most complete contiguous passage that fully states the {slot}.
- The same materials/measure may be shared across studies; a passage describing it counts
  even if it names a different study, as long as it states the {slot} for this measure.
- If the {slot} text is NOT contained in the evidence, set "found": false.
- Do NOT use any outside knowledge. Only the evidence below may be quoted.{fb}{draft_block}

EVIDENCE:
{evidence}

Respond with JSON exactly:
{{"found": true|false, "quote": "<exact verbatim span, or empty string>", "reason": "<short>"}}"""
        try:
            data = generate_json(
                self.client,
                [{"role": "user", "content": prompt}],
                system=system,
                temperature=self.temperature,
                retries=0,
                timeout=self.slot_timeout,
            )
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(data, dict):
            return {"found": False, "quote": "", "reason": "bad_response"}
        return data

    def _ask_describe(
        self, slot: str, study: str, dv: str, desc: str, evidence: str
    ) -> dict[str, Any]:
        """Fallback for sources (e.g. QSF) where verbatim extraction fails.

        Asks the LLM to DESCRIBE the slot content based on the evidence rather
        than copy a verbatim span. This handles QSF piped-text and pre-registration
        prose. The result is tagged ``paraphrased``.
        """
        system = (
            "You extract experimental measurement information from research materials. "
            "You summarize what you find in the provided evidence. "
            "Do not invent information not present in the evidence."
        )
        prompt = f"""TASK: Based ONLY on the evidence below (from a study's supplementary materials),
describe {desc} for "{study}" (dependent variable: {dv}).

Rules:
- Summarize what you can find in the evidence about this {slot}. Include question wording,
  scale anchors, and response options if present.
- Do NOT use outside knowledge. Base your answer solely on the evidence.
- If the {slot} is NOT described in the evidence at all, set "found": false.
- Keep the description concise but complete (max 400 words).

EVIDENCE:
{evidence}

Respond with JSON exactly:
{{"found": true|false, "description": "<summary from evidence, or empty string>", "reason": "<short>"}}"""
        try:
            data = generate_json(
                self.client,
                [{"role": "user", "content": prompt}],
                system=system,
                temperature=self.temperature,
                retries=0,
                timeout=self.slot_timeout,
            )
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(data, dict):
            return {"found": False, "description": "", "reason": "bad_response"}
        return data

    # multi-span assembly
    def _assemble_spans(
        self,
        slot: str,
        study: str,
        iv: str,
        dv: str,
        desc: str,
        evidence: str,
        *,
        paraphrase_threshold: float,
        evidence_pages: list[int],
    ) -> Optional[SlotResult]:
        """Gather multiple labeled verbatim spans and verify each individually.

        Returns a SlotResult when at least one span grounds in the source, or
        ``None`` to signal the caller should fall back to single-span extraction.

        Honesty guarantee: every span kept in the assembled content is
        individually present in the source at >= paraphrase_threshold. Spans the
        model returns that do NOT ground (likely edited/paraphrased/invented) are
        dropped, never shown as verbatim.
        """
        parsed = self._ask_assemble(slot, study, iv, dv, desc, evidence)
        if parsed.get("error"):
            return SlotResult(status="not_in_paper", content=None, score=None, action="skipped_api_error")
        spans = parsed.get("spans")
        if not isinstance(spans, list) or not spans:
            return None

        grounded: list[tuple[str, str, float]] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            quote = (span.get("quote") or "").strip()
            label = (span.get("label") or "").strip()
            if not quote:
                continue
            score = best_partial_ratio(normalize_for_match(quote), self._source_norm)
            if score >= paraphrase_threshold:
                grounded.append((label, quote, score))

        if not grounded:
            return None  # nothing grounded — let single-span path try

        all_verbatim = all(score >= self.threshold for _, _, score in grounded)
        content = "\n\n".join(
            (f"[{label}]\n{quote}" if label else quote) for label, quote, _ in grounded
        )
        weakest = min(score for _, _, score in grounded)
        return SlotResult(
            status="verbatim" if all_verbatim else "paraphrased",
            content=content,
            score=weakest,
            action="grounded_assembled" if all_verbatim else "grounded_assembled_paraphrased",
            evidence_pages=evidence_pages,
        )

    def _ask_assemble(
        self, slot: str, study: str, iv: str, dv: str, desc: str, evidence: str
    ) -> dict[str, Any]:
        """Ask the model to gather ALL verbatim spans composing the material.

        Designed for composite stimuli (e.g. a behavioral paradigm whose intro,
        rules, per-condition manipulation, and DV items live in separate Qualtrics
        blocks/conditions). The model returns a list of {label, quote}; the caller
        verifies each quote against the source.
        """
        system = (
            "You are a meticulous research assistant assembling the COMPLETE set of stimulus "
            "materials for one experimental measure from supplementary materials (which may be a "
            "Qualtrics survey export with multiple blocks and conditions). You copy text VERBATIM "
            "and never paraphrase, summarize, or invent. You gather EVERY relevant span, including "
            "one span per experimental condition when the manipulation differs across conditions."
        )
        prompt = f"""TASK: From the evidence passages below (extracted from a study's supplementary
materials), gather the COMPLETE set of verbatim spans that together constitute {desc}
for "{study}" (independent variable: {iv}; dependent variable: {dv}).

These materials are often SPREAD ACROSS MULTIPLE blocks/conditions. For example a behavioral
paradigm may have: an introduction, the rules/instructions, a per-condition manipulation
(e.g. one version shown when condition=public and another when condition=private, or when a
variable like ambiguous=1 vs ambiguous=0), and the response items. The condition labels may
appear in a "CONDITIONS" outline or in "[shown if: ...]" tags rather than in the stimulus text.

Rules:
- Return a LIST of spans. Each span = {{"label": "<short tag e.g. 'intro', 'rules', 'manipulation: public', 'manipulation: private', 'DV items'>", "quote": "<EXACT verbatim text copied from the evidence>"}}.
- Copy each quote EXACTLY as it appears in the evidence — do not paraphrase, shorten, reorder, fix typos, or merge separate passages into one quote. One contiguous passage per span.
- Include a SEPARATE span for each condition's version of the manipulation when they differ.
- Do NOT use outside knowledge. Only quote text present in the evidence below.
- If none of the evidence contains this material, return an empty list.

EVIDENCE:
{evidence}

Respond with JSON exactly:
{{"spans": [{{"label": "...", "quote": "..."}}], "reason": "<short>"}}"""
        try:
            data = generate_json(
                self.client,
                [{"role": "user", "content": prompt}],
                system=system,
                temperature=self.temperature,
                retries=0,
                timeout=self.slot_timeout,
            )
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(data, dict):
            return {"spans": [], "reason": "bad_response"}
        return data

    def reground_paper(
        self,
        paper: dict[str, Any],
        pdf_path: Path,
        *,
        slots: Optional[list[str]] = None,
        only_verbatim_and_empty: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Re-ground slots of an existing paper dict. Returns (new_paper, report)."""
        self.prepare(pdf_path)
        data = deepcopy(paper)
        slots = slots or list(SLOT_NAMES)
        records: list[dict[str, Any]] = []
        counts = {"grounded_verbatim": 0, "kept_paraphrased": 0, "not_in_paper": 0, "skipped": 0}
        tasks, skipped = _collect_grounding_tasks(
            data,
            slots,
            only_verbatim_and_empty=only_verbatim_and_empty,
        )
        counts["skipped"] = skipped
        total_slots = len(tasks)
        worker_count = min(self.max_workers, total_slots) if total_slots else 0
        if total_slots:
            print(
                f"  Stage 2 grounding pass: checking {total_slots} material slots "
                f"with {worker_count} worker(s)",
                flush=True,
            )

        task_results = self._run_grounding_tasks(tasks, worker_count)
        for task, result, record in sorted(task_results, key=lambda item: item[0].order):
            obj = data["eligible_studies"][task.study_index]["effects"][task.effect_index][task.slot]
            obj["status"] = result.status
            obj["content"] = result.content
            counts[result.status] = counts.get(result.status, 0) + 1
            records.append(record)

        report = {
            "threshold": self.threshold,
            "summary": {
                "slots_processed": len(records),
                "slots_skipped": counts.get("skipped", 0),
                "workers": worker_count,
                "grounded_verbatim": counts.get("verbatim", counts.get("grounded_verbatim", 0)),
                "downgraded_or_kept": sum(
                    1 for r in records if r["action"] in {"kept_paraphrased", "downgraded", "not_present"}
                ),
            },
            "records": records,
        }
        return data, report

    def _run_grounding_tasks(
        self,
        tasks: list[GroundingTask],
        worker_count: int,
    ) -> list[tuple[GroundingTask, SlotResult, dict[str, Any]]]:
        if not tasks:
            return []
        if worker_count <= 1:
            results = []
            total = len(tasks)
            for idx, task in enumerate(tasks, start=1):
                print(
                    f"    grounding {idx}/{total}: "
                    f"{task.study_label} effect {task.effect_index + 1} {task.slot}",
                    flush=True,
                )
                results.append(self._run_grounding_task(task))
            return results

        results: list[tuple[GroundingTask, SlotResult, dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {executor.submit(self._run_grounding_task, task): task for task in tasks}
            for idx, future in enumerate(as_completed(future_to_task), start=1):
                task, result, record = future.result()
                print(
                    f"    grounding completed {idx}/{len(tasks)}: "
                    f"{task.study_label} effect {task.effect_index + 1} {task.slot}",
                    flush=True,
                )
                results.append((task, result, record))
        return results

    def _run_grounding_task(self, task: GroundingTask) -> tuple[GroundingTask, SlotResult, dict[str, Any]]:
        result = self.extract_slot(
            study=task.study_label,
            iv=task.iv,
            dv=task.dv,
            slot=task.slot,
            location_hint=task.location_hint,
            materials_notes=task.materials_notes,
            prev_status=task.prev_status,
            prev_content=task.prev_content,
        )
        record = {
            "path": task.path,
            "study": task.study_label,
            "effect_index": task.effect_index,
            "slot": task.slot,
            "before": {"status": task.prev_status, "had_content": task.had_content},
            **result.to_dict(),
        }
        return task, result, record


def _collect_grounding_tasks(
    paper: dict[str, Any],
    slots: list[str],
    *,
    only_verbatim_and_empty: bool,
) -> tuple[list[GroundingTask], int]:
    tasks: list[GroundingTask] = []
    skipped = 0
    order = 0
    for si, study in enumerate(paper.get("eligible_studies", [])):
        if not isinstance(study, dict):
            continue
        study_label = study.get("study", f"study[{si}]")
        for ei, effect in enumerate(study.get("effects", [])):
            if not isinstance(effect, dict):
                continue
            iv, dv = effect.get("IV", ""), effect.get("DV", "")
            loc = effect.get("table_or_page_location", "")
            notes = effect.get("materials_notes", "")
            for slot in slots:
                obj = effect.get(slot)
                if not isinstance(obj, dict):
                    continue
                status = obj.get("status")
                content = obj.get("content")
                has_content = isinstance(content, str) and content.strip() != ""
                if only_verbatim_and_empty and status not in {"verbatim", None} and not (
                    status in {"paraphrased"} and has_content
                ):
                    skipped += 1
                    continue
                tasks.append(
                    GroundingTask(
                        order=order,
                        study_index=si,
                        effect_index=ei,
                        slot=slot,
                        path=f"$.eligible_studies[{si}].effects[{ei}].{slot}",
                        study_label=study_label,
                        iv=iv,
                        dv=dv,
                        location_hint=loc,
                        materials_notes=notes,
                        prev_status=status,
                        prev_content=content,
                        had_content=has_content,
                    )
                )
                order += 1
    return tasks, skipped


def _write_json_atomic(path: Path, data: dict[str, Any], *, backup: bool = True) -> None:
    import shutil

    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _study_key_from_label(study_label: Any) -> str | None:
    text = normalize_for_match(str(study_label or ""))
    match = re.search(r"\b(?:study|experiment)\s*(\d+[a-z]?)\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d+[a-z]?)\b", text)
    return match.group(1) if match else None


def _study_keys_from_source_path(source_path: str) -> tuple[str, ...]:
    """Infer study identifiers from a source filename/path.

    Examples:
      Study 3.qsf -> ("3",)
      Studies 1a-b.qsf -> ("1a", "1b")
      Pre-registrations/Study 5.pdf -> ("5",)
    """
    text = normalize_for_match(Path(source_path).as_posix())
    keys: list[str] = []

    # Ranges like "Studies 1a-b" or "Studies 1-3".
    for match in re.finditer(
        r"\bstudies?\s*(?P<start>\d+[a-z]?)(?:\s*[-–]\s*(?P<end>\d+[a-z]?|[a-z]))?",
        text,
    ):
        start = match.group("start")
        end = match.group("end")
        if end:
            keys.extend(_expand_study_range(start, end))
        else:
            keys.append(start)

    # Directory-style paths such as ".../Study 2a/Code.txt".
    for match in re.finditer(r"\bstudy\s*(?P<key>\d+[a-z]?)\b", text):
        keys.append(match.group("key"))

    seen: set[str] = set()
    return tuple(key for key in keys if not (key in seen or seen.add(key)))


def _expand_study_range(start: str, end: str) -> list[str]:
    start_match = re.fullmatch(r"(\d+)([a-z]?)", start)
    end_match = re.fullmatch(r"(\d+)?([a-z]?)", end)
    if not start_match or not end_match:
        return [start]
    start_num = int(start_match.group(1))
    start_suffix = start_match.group(2)
    end_num = int(end_match.group(1) or start_num)
    end_suffix = end_match.group(2)

    if start_suffix and end_suffix and start_num == end_num:
        return [f"{start_num}{chr(code)}" for code in range(ord(start_suffix), ord(end_suffix) + 1)]
    if not start_suffix and not end_suffix and start_num <= end_num and end_num - start_num <= 20:
        return [str(num) for num in range(start_num, end_num + 1)]
    return [start, f"{end_num}{end_suffix}"]


def _source_kind_from_path(source_path: str, study_keys: tuple[str, ...]) -> str:
    text = normalize_for_match(Path(source_path).as_posix())
    if "pre-registration" in text or "pre registrations" in text or "prereg" in text:
        return "prereg"
    if "experimental materials" in text or Path(source_path).suffix.lower() in {".qsf", ".docx"}:
        return "experimental"
    if not study_keys:
        return "shared"
    return "study"


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-ground verbatim slots of a paper JSON against its PDF")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--provider", default=None, help="Override; else resolved from settings/env")
    parser.add_argument("--model", default=None, help="Override; else resolved from settings/env")
    parser.add_argument("--base-url", default=None, help="Override OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="Override; prefer the provider env var")
    parser.add_argument("--settings", type=Path, default=None, help="Path to settings.yaml")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Do not write JSON back")
    parser.add_argument("--report", type=Path, help="Where to write the reground report")
    args = parser.parse_args()

    from src.llm.factory import get_client
    from generation_pipeline.settings import load_settings, resolve_llm_config

    settings = load_settings(args.settings)
    cfg = resolve_llm_config(
        settings,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_base=args.base_url,
    )
    print(f"LLM: provider={cfg.provider} model={cfg.model} base_url={cfg.api_base}")
    client = get_client(provider=cfg.provider, model=cfg.model, api_key=cfg.api_key, api_base=cfg.api_base)
    extractor = GroundedSlotExtractor(client, threshold=DEFAULT_THRESHOLD, k=args.k)

    paper = json.loads(args.json.read_text(encoding="utf-8"))
    new_paper, report = extractor.reground_paper(paper, args.pdf)

    if not args.dry_run:
        _write_json_atomic(args.json, new_paper, backup=True)
    report_path = args.report or args.json.parent / "reground_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
