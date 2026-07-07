"""
Slot Filler - Stage 3 (ai-ethics)

Patches existing per-paper JSON files (the `输出/*.json` corpus) by:
  1. Filling empty `materials / manipulation / items` slots (status == null)
     with verbatim/paraphrased content from the PDF.
  2. Upgrading the `sample` field from a string to a structured participants
     object. Current canonical schema stores this object at study level; this
     class still accepts old effect-level `sample` strings for migration.

The patcher only writes slots that are currently empty (status == null);
human-reviewed slots are preserved.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generation_pipeline.utils.pdf_extractor import extract_pdf_text


STUDY_SECTION_CHARS = 24000   # window per study fed to LLM
PDF_TEXT_MAX_CHARS = 400000
PatchUpdate = Dict[str, Any]


def _count_non_null_leaves(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_count_non_null_leaves(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_non_null_leaves(item) for item in value)
    return 0 if value is None else 1


def _count_newly_filled_nulls(previous: Any, current: Any) -> int:
    if previous is None:
        return _count_non_null_leaves(current)
    if isinstance(previous, dict) and isinstance(current, dict):
        keys = set(previous) | set(current)
        return sum(_count_newly_filled_nulls(previous.get(key), current.get(key)) for key in keys)
    if isinstance(previous, list) and isinstance(current, list):
        total = 0
        max_len = max(len(previous), len(current))
        for index in range(max_len):
            before = previous[index] if index < len(previous) else None
            after = current[index] if index < len(current) else None
            total += _count_newly_filled_nulls(before, after)
        return total
    return 0


def _content_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if value is None:
        return 0
    return len(json.dumps(value, ensure_ascii=False))


def _make_patch_update(
    study_label: str,
    slot_name: str,
    *,
    source: str,
    previous: Any,
    current: Any,
    effect_index: int | None = None,
    iv: str | None = None,
    dv: str | None = None,
    score: float | None = None,
) -> PatchUpdate:
    content = current.get("content") if isinstance(current, dict) else current
    path = f"{study_label}.{slot_name}"
    if effect_index is not None:
        path = f"{study_label}.effect[{effect_index}].{slot_name}"
    return {
        "path": path,
        "study": study_label,
        "effect_index": effect_index,
        "slot": slot_name,
        "iv": iv,
        "dv": dv,
        "source": source,
        "status": current.get("status") if isinstance(current, dict) else None,
        "score": score,
        "filled_nulls": _count_newly_filled_nulls(previous, current),
        "updated_fields": _count_non_null_leaves(current),
        "content_chars": _content_chars(content),
    }


# ---------------------------------------------------------------------------
# Section locator: pull a window around "Study N" / first Method section
# ---------------------------------------------------------------------------

def find_study_section(pdf_text: str, study_label: str, window: int = STUDY_SECTION_CHARS) -> str:
    """Return a substring of PDF text centered on the study's Method section.

    Tries (in order): "Study N", "Experiment N", "Method"; falls back to the
    first `window` chars of the document.
    """
    # Normalize: "Study 1" → patterns ["Study 1", "Experiment 1", "Study1"]
    candidates = [study_label]
    m = re.search(r"(study|experiment)\s*(\d+[a-z]?)", study_label, re.I)
    if m:
        n = m.group(2)
        candidates.extend([f"Study {n}", f"Experiment {n}", f"Study{n}", f"Experiment{n}"])
    candidates.append("Method")

    for cand in candidates:
        match = re.search(rf"\b{re.escape(cand)}\b", pdf_text, re.I)
        if match:
            start = max(0, match.start() - 200)
            return pdf_text[start : start + window]

    return pdf_text[:window]


class SlotFiller:
    """Patch one paper JSON in place (returns new dict, does not mutate input)."""

    VALID_STATUSES = {
        "verbatim",
        "paraphrased",
        "cited_scale",
        "osf_only",
        "not_in_paper",
        "source_missing",
    }

    def __init__(
        self,
        llm_client,
        *,
        source_workers: int = 4,
        source_slot_timeout: float | None = 60.0,
    ):
        self.client = llm_client
        self.source_workers = max(int(source_workers or 1), 1)
        self.source_slot_timeout = source_slot_timeout

    # Statuses that mean "slot was assessed but content isn't in the PDF" —
    # these become candidates for re-patching when OSF source files are provided.
    OSF_RETRYABLE_STATUSES = {"osf_only", "not_in_paper", "cited_scale"}

    def patch_paper(
        self,
        paper_json: Dict[str, Any],
        pdf_path: Path,
        *,
        overwrite_filled: bool = False,
        source_dirs: Optional[list] = None,
        on_update: Optional[Callable[[Dict[str, Any], Any], None]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            paper_json: parsed JSON from `输出/*.json`
            pdf_path: matched PDF (same filename stem)
            overwrite_filled: if True, also re-run already-filled slots
                              (default: False — preserve human-reviewed work)
            source_dirs: optional list of Path-like source directories (e.g.
                         ``stage4/<paper>/sources``). When provided:
                         1. Pass 1 (PDF): fill status=None slots from the PDF.
                         2. Pass 2 (OSF grounded): for slots still
                            ``osf_only``/``not_in_paper``/``cited_scale`` with
                            no content, run the paper-qa-style GroundedSlotExtractor
                            against the combined source text. Only verified-verbatim
                            quotes are accepted; unverifiable content is honestly kept
                            as the original status with content=null.
        """
        result = deepcopy(paper_json)
        pdf_text = extract_pdf_text(pdf_path, max_chars=PDF_TEXT_MAX_CHARS)

        for study in result.get("eligible_studies", []):
            study_label = study.get("study", "")
            section = find_study_section(pdf_text, study_label)
            for effect_index, effect in enumerate(study.get("effects", []), 1):
                updates = self._patch_effect(
                    effect,
                    section,
                    study_label,
                    effect_index=effect_index,
                    overwrite=overwrite_filled,
                )
                if on_update:
                    for update in updates:
                        on_update(result, update)

        source_records = self._load_supplementary_sources(source_dirs)
        if source_records:
            self._patch_from_sources(result, source_records, on_update=on_update)

        return result

    @staticmethod
    def _load_supplementary(source_dirs: Optional[list]) -> str:
        """Load text from combined_sources.txt in each source directory.

        No hard character cap is applied here: the text is chunked and only
        the top-k retrieved chunks (typically ~20 KB) are ever sent to the
        LLM, so a large corpus does not inflate prompt costs.
        """
        if not source_dirs:
            return ""
        parts: list[str] = []
        for src_dir in source_dirs:
            combined = Path(src_dir) / "combined_sources.txt"
            if combined.exists():
                try:
                    parts.append(combined.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
        return "\n\n".join(parts)

    @staticmethod
    def _load_supplementary_sources(source_dirs: Optional[list]) -> list[dict[str, str]]:
        """Load source text records while preserving source-file provenance.

        Preferred input is ``source_text_index.json`` written by the connector
        registry; it maps each original source file to its extracted text file.
        Fall back to the legacy anonymous ``combined_sources.txt`` only when the
        index is unavailable.
        """
        if not source_dirs:
            return []
        records: list[dict[str, str]] = []
        for src_dir in source_dirs:
            src_dir = Path(src_dir)
            added = False
            index_path = src_dir / "source_text_index.json"
            if index_path.exists():
                try:
                    index = json.loads(index_path.read_text(encoding="utf-8"))
                except Exception:
                    index = []
                if isinstance(index, list):
                    for item in index:
                        if not isinstance(item, dict):
                            continue
                        text_path = Path(str(item.get("text_path") or ""))
                        if not text_path.exists():
                            continue
                        try:
                            text = text_path.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            continue
                        if text.strip():
                            records.append(
                                {
                                    "source_path": str(item.get("source_path") or text_path),
                                    "text_path": str(text_path),
                                    "text": text,
                                }
                            )
                            added = True
            if added:
                continue
            combined = src_dir / "combined_sources.txt"
            if combined.exists():
                try:
                    text = combined.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = ""
                if text.strip():
                    records.append({"source_path": str(combined), "text_path": str(combined), "text": text})
        return records

    # Lower fuzzy-match threshold for OSF supplementary content.
    # QSF piped text (${e://Field/text}) and pre-registration prose won't
    # score ≥90 against the PDF, but ≥75 indicates the content is present.
    OSF_PARAPHRASE_THRESHOLD = 75.0

    def _patch_from_sources(
        self,
        result: Dict[str, Any],
        source_text: str | list[dict[str, str]],
        *,
        on_update: Optional[Callable[[Dict[str, Any], Any], None]] = None,
    ) -> None:
        """Paper-qa-style grounded fill from OSF/supplementary source text.

        Chunks the source text and uses GroundedSlotExtractor (with a two-tier
        threshold: ≥90 → verbatim, ≥75 → paraphrased) to fill slots that
        were previously unfillable from the PDF alone.

        OSF supplementary files come in several formats:
        - DOCX (e.g. manipulation scripts) → clean text, often verbatim-matchable
        - Pre-registration PDFs → abstract prose, often paraphrase-tier
        - QSF Qualtrics surveys → piped variables, needs assembly → paraphrase-tier
        """
        try:
            from generation_pipeline.extractors.grounded_slot_extractor import GroundedSlotExtractor
        except ImportError:
            return  # grounded extractor unavailable — degrade gracefully

        extractor = GroundedSlotExtractor(
            self.client,
            threshold=90.0,
            k=8,
            retries=1,
            slot_timeout=self.source_slot_timeout,
            max_workers=self.source_workers,
        )
        if isinstance(source_text, list):
            extractor.prepare_from_sources(source_text)
        else:
            extractor.prepare_from_text(source_text)

        studies = [s for s in result.get("eligible_studies", []) if isinstance(s, dict)]

        per_study_filled: dict[int, int] = {}
        per_study_empty_osf_only: dict[int, list[tuple[dict, int, str]]] = {}
        source_tasks: list[dict[str, Any]] = []

        for study_index, study in enumerate(studies):
            study_label = study.get("study", "")
            per_study_filled.setdefault(study_index, 0)
            per_study_empty_osf_only.setdefault(study_index, [])
            for effect_index, effect in enumerate(study.get("effects", []), 1):
                if not isinstance(effect, dict):
                    continue
                iv = effect.get("IV", "")
                dv = effect.get("DV", "")
                loc = effect.get("table_or_page_location", "")
                notes = effect.get("materials_notes", "")
                for slot_name in ("materials", "manipulation", "items"):
                    obj = effect.get(slot_name) or {}
                    status = obj.get("status")
                    content = obj.get("content")
                    if status not in self.OSF_RETRYABLE_STATUSES:
                        continue
                    if content:  # already has content (e.g. human-filled)
                        continue
                    source_tasks.append(
                        {
                            "order": len(source_tasks),
                            "study_index": study_index,
                            "study_label": study_label,
                            "effect": effect,
                            "effect_index": effect_index,
                            "slot_name": slot_name,
                            "status": status,
                            "iv": iv,
                            "dv": dv,
                            "location_hint": loc,
                            "materials_notes": notes,
                        }
                    )

        if source_tasks:
            worker_count = min(self.source_workers, len(source_tasks))
            print(
                f"    source patch: checking {len(source_tasks)} OSF/source slots "
                f"with {worker_count} worker(s)",
                flush=True,
            )

        source_results = self._run_source_tasks(extractor, source_tasks)
        for task, res in sorted(source_results, key=lambda item: item[0]["order"]):
            effect = task["effect"]
            slot_name = task["slot_name"]
            obj = effect.get(slot_name) or {}
            if res.content and res.status in {"verbatim", "paraphrased"}:
                previous = deepcopy(obj)
                current = {"status": res.status, "content": res.content}
                effect[slot_name] = current
                per_study_filled[task["study_index"]] += 1
                update = _make_patch_update(
                    task["study_label"],
                    slot_name,
                    source="osf",
                    previous=previous,
                    current=current,
                    effect_index=task["effect_index"],
                    iv=task["iv"],
                    dv=task["dv"],
                    score=res.score,
                )
                if on_update:
                    on_update(result, update)
                else:
                    print(f"    ✓ filled from sources: {task['study_label']}.{slot_name} "
                          f"status={res.status} score={res.score:.0f}")
            elif res.status == "source_missing":
                previous = deepcopy(obj)
                current = {"status": "source_missing", "content": None}
                effect[slot_name] = current
                update = _make_patch_update(
                    task["study_label"],
                    slot_name,
                    source="osf",
                    previous=previous,
                    current=current,
                    effect_index=task["effect_index"],
                    iv=task["iv"],
                    dv=task["dv"],
                    score=res.score,
                )
                if on_update:
                    on_update(result, update)
                else:
                    print(f"    ⚠ source_missing: {task['study_label']}.{slot_name} "
                          f"(no study-scoped source evidence)")
            elif task["status"] == "osf_only":
                per_study_empty_osf_only[task["study_index"]].append(
                    (effect, task["effect_index"], slot_name)
                )

        # Mark study-local OSF gaps only after at least one sibling study was filled.
        any_study_filled = any(count > 0 for count in per_study_filled.values())
        for study_index, study in enumerate(studies):
            if per_study_filled.get(study_index, 0) > 0:
                continue
            if not any_study_filled:
                continue
            study_label = study.get("study", "")
            for effect, effect_index, slot_name in per_study_empty_osf_only.get(study_index, []):
                obj = effect.get(slot_name) or {}
                previous = deepcopy(obj)
                current = {"status": "source_missing", "content": None}
                effect[slot_name] = current
                update = _make_patch_update(
                    study_label,
                    slot_name,
                    source="osf",
                    previous=previous,
                    current=current,
                    effect_index=effect_index,
                    iv=effect.get("IV"),
                    dv=effect.get("DV"),
                )
                if on_update:
                    on_update(result, update)
                else:
                    print(f"    ⚠ source_missing: {study_label}.{slot_name} "
                          f"(no material in fetched sources; sibling studies filled)")

    def _run_source_tasks(self, extractor: Any, tasks: list[dict[str, Any]]) -> list[tuple[dict[str, Any], Any]]:
        if not tasks:
            return []

        def run_one(task: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            res = extractor.extract_slot(
                study=task["study_label"],
                iv=task["iv"],
                dv=task["dv"],
                slot=task["slot_name"],
                location_hint=task["location_hint"],
                materials_notes=task["materials_notes"],
                prev_status=task["status"],
                prev_content=None,
                paraphrase_threshold=self.OSF_PARAPHRASE_THRESHOLD,
                assemble=True,
            )
            return task, res

        worker_count = min(self.source_workers, len(tasks))
        if worker_count <= 1:
            results = []
            for index, task in enumerate(tasks, start=1):
                print(
                    f"      source patch {index}/{len(tasks)}: "
                    f"{task['study_label']} effect {task['effect_index']} {task['slot_name']}",
                    flush=True,
                )
                results.append(run_one(task))
            return results

        results = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {executor.submit(run_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_to_task), start=1):
                task, res = future.result()
                print(
                    f"      source patch completed {index}/{len(tasks)}: "
                    f"{task['study_label']} effect {task['effect_index']} {task['slot_name']}",
                    flush=True,
                )
                results.append((task, res))
        return results

    def _patch_effect(
        self,
        effect: Dict[str, Any],
        section_text: str,
        study_label: str,
        *,
        effect_index: int | None = None,
        overwrite: bool,
    ) -> list[PatchUpdate]:
        """Mutate `effect` in place: fill empty slots + upgrade sample."""
        updates: list[PatchUpdate] = []
        needs_slots = self._empty_slots(effect, overwrite=overwrite)
        needs_sample_upgrade = isinstance(effect.get("sample"), str)
        if not needs_slots and not needs_sample_upgrade:
            return updates

        filled = self._call_llm(
            section_text=section_text,
            study_label=study_label,
            effect=effect,
            need_slots=needs_slots,
            need_sample=needs_sample_upgrade,
        )
        if filled is None:
            return updates

        for slot_name in ("materials", "manipulation", "items"):
            if slot_name in needs_slots and slot_name in filled:
                new = filled[slot_name]
                if isinstance(new, dict) and self._is_valid_slot(new):
                    previous = deepcopy(effect.get(slot_name) or {})
                    effect[slot_name] = new
                    updates.append(
                        _make_patch_update(
                            study_label,
                            slot_name,
                            source="pdf",
                            previous=previous,
                            current=new,
                            effect_index=effect_index,
                            iv=effect.get("IV"),
                            dv=effect.get("DV"),
                        )
                    )

        # Legacy upgrade: old effect.sample string → flat sample object.
        # The schema validator later lifts this to study.sample.
        if needs_sample_upgrade and "sample" in filled and isinstance(filled["sample"], dict):
            original = effect["sample"]
            new_sample = filled["sample"]
            description = new_sample.pop("description", None)
            if not new_sample.get("notes"):
                new_sample["notes"] = description or original
            effect["sample"] = new_sample
            updates.append(
                _make_patch_update(
                    study_label,
                    "sample",
                    source="pdf",
                    previous=original,
                    current=new_sample,
                    effect_index=effect_index,
                    iv=effect.get("IV"),
                    dv=effect.get("DV"),
                )
            )

        return updates

    def _empty_slots(self, effect: Dict[str, Any], *, overwrite: bool) -> set[str]:
        """Slots to fill from the PDF (status=None, or all if overwrite=True)."""
        out = set()
        for slot_name in ("materials", "manipulation", "items"):
            slot = effect.get(slot_name) or {}
            if overwrite or slot.get("status") is None:
                out.add(slot_name)
        return out

    def _is_valid_slot(self, slot: Dict[str, Any]) -> bool:
        return (
            "status" in slot
            and (slot["status"] is None or slot["status"] in self.VALID_STATUSES)
            and "content" in slot
        )

    def _call_llm(
        self,
        *,
        section_text: str,
        study_label: str,
        effect: Dict[str, Any],
        need_slots: set[str],
        need_sample: bool,
    ) -> Optional[Dict[str, Any]]:
        prompt = self._build_prompt(section_text, study_label, effect, need_slots, need_sample)
        try:
            response = self.client.generate_content(prompt=prompt)
        except Exception as e:
            print(f"  LLM error on {study_label}: {e}")
            return None
        if response is None:
            return None
        return self._parse_response(response)

    def _build_prompt(
        self,
        section_text: str,
        study_label: str,
        effect: Dict[str, Any],
        need_slots: set[str],
        need_sample: bool,
    ) -> str:
        wanted = []
        if "materials" in need_slots:
            wanted.append('"materials"   (the stimuli / scenarios / texts participants saw)')
        if "manipulation" in need_slots:
            wanted.append('"manipulation" (exactly what differed between conditions for the IV)')
        if "items" in need_slots:
            wanted.append('"items"       (the DV scale items / response options)')
        if need_sample:
            wanted.append('"sample"      (structured participants profile, see schema below)')

        schema_slot = """{
  "status": "verbatim | paraphrased | cited_scale | osf_only | not_in_paper",
  "content": "<verbatim quote from PDF if status=verbatim; faithful paraphrase if =paraphrased; null otherwise>"
}"""

        sample_schema = """{
  "total_n": <int or null; FULL study recruited/final-sample N, not subgroup/cell/table N>,
  "analyzed_n": <int or null; only if the whole study has a distinct analyzed N>,
  "mean_age": <float|null>,
  "female_percent": <float 0-100|null>,
  "male_percent": <float 0-100|null>,
  "platform": "<MTurk|Prolific|CloudResearch|TurkPrime|Qualtrics|Undergraduate|Graduate|Lab|Organizational|Online|Field|Archival|Mixed|Other|null>",
  "country": "<country string or null>",
  "inclusion_criteria": "<verbatim source text or null>",
  "exclusion_criteria": "<verbatim source text or null>",
  "notes": "<other participant/sample info as verbatim source text or null>"
}"""

        return f"""You are extracting research-study details from a PDF text excerpt for the ai-ethics corpus.

STUDY: {study_label}
KNOWN EFFECT (for context — do not re-extract):
- IV: {effect.get('IV')}
- DV: {effect.get('DV')}
- N: {effect.get('size')}
- materials_notes: {effect.get('materials_notes')}
- table_or_page_location: {effect.get('table_or_page_location')}

PDF EXCERPT FOR THIS STUDY:
\"\"\"
{section_text}
\"\"\"

YOUR TASK — extract the following fields:
{chr(10).join('- ' + w for w in wanted)}

RULES (CRITICAL):
- Each of (materials, manipulation, items) is an object with this schema:
{schema_slot}
- For `content`, use VERBATIM TEXT from the excerpt when status="verbatim".
- If the paper only describes the material in summary form, use status="paraphrased".
- If the paper cites a published scale without reproducing it (e.g. "Reynolds 2008"),
  status="cited_scale" and content=<the citation as quoted>.
- If the paper says materials are on OSF / supplementary only, status="osf_only", content=null.
- If the paper does not include this material at all, status="not_in_paper", content=null.
- DO NOT FABRICATE: never invent stimulus text not present in the excerpt.

- `sample` schema (structured participants profile; canonical location is study.sample):
{sample_schema}
  Fill only fields the paper reports. All others = null. `total_n` must be the
  full final/recruited study N. Do not put subgroup / condition / simple-effect
  table Ns into sample.total_n.

OUTPUT — respond with ONLY this JSON (no markdown fences). Include ONLY the
keys requested above (omit keys you were not asked to fill):
{{
  {", ".join(f'"{k}": ...' for k in (['materials','manipulation','items','sample'] if False else list(need_slots) + (['sample'] if need_sample else [])))}
}}"""

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        text = response.strip() if isinstance(response, str) else str(response).strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    return None
            return None
