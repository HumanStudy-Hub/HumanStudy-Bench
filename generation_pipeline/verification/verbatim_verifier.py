#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generation_pipeline.utils.pdf_extractor import extract_pdf_text
from generation_pipeline.verification.schema_validator import SLOT_NAMES, validate_paper


DEFAULT_THRESHOLD = 90.0
PDF_TEXT_MAX_CHARS = 400000
# Aliases used when looking for platform names in source text.
# Keys are normalized (lowercase, stripped); values are tuples of substrings to search.
SAMPLE_PLATFORM_ALIASES: dict[str, tuple[str, ...]] = {
    # Online crowdsourcing
    "mturk":             ("mturk", "mechanical turk", "amazon turk"),
    "mechanical turk":   ("mturk", "mechanical turk", "amazon turk"),
    "prolific":          ("prolific",),
    "cloudresearch":     ("cloudresearch", "cloud research", "turkprime"),
    "turkprime":         ("turkprime", "cloudresearch"),
    # University pools
    "undergraduate":     ("undergraduate", "undergrad", "sona", "subject pool", "student pool", "college student"),
    "graduate":          ("graduate student", "grad student"),
    # Lab / field
    "lab":               ("lab", "laboratory", "in-person", "in person"),
    "field":             ("field experiment", "field study", "naturalistic"),
    # Organizational
    "organizational":    ("organizational", "employees", "workers", "company", "firm", "organization"),
    # Other online
    "online":            ("online", "internet", "web-based", "web based"),
    # Archival
    "archival":          ("archival", "administrative data", "secondary data", "existing data", "database"),
}


@dataclass
class VerbatimRecord:
    path: str
    study: str | None
    effect_index: int
    slot: str
    score: float | None
    threshold: float
    action: str
    content_preview: str
    span_scores: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "path": self.path,
            "study": self.study,
            "effect_index": self.effect_index,
            "slot": self.slot,
            "score": self.score,
            "threshold": self.threshold,
            "action": self.action,
            "content_preview": self.content_preview,
        }
        if self.span_scores:
            data["span_scores"] = self.span_scores
        return data


@dataclass
class SampleRecord:
    path: str
    study: str | None
    effect_index: int
    field: str
    value_preview: str
    evidence_kind: str
    action: str
    evidence_preview: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "study": self.study,
            "effect_index": self.effect_index,
            "field": self.field,
            "value_preview": self.value_preview,
            "evidence_kind": self.evidence_kind,
            "action": self.action,
            "evidence_preview": self.evidence_preview,
        }


def verify_paper(
    paper: dict[str, Any],
    source_text: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    repair: bool = True,
    path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Verify all verbatim slots in a paper object against source_text.

    Returns a repaired deep copy and a JSON-serializable report.
    """
    data, schema_report = validate_paper(deepcopy(paper), repair=True, path=path)
    records: list[VerbatimRecord] = []
    normalized_source = normalize_for_match(source_text)
    unverified = 0
    downgraded = 0
    passed = 0

    for slot_ref in _iter_slots(data):
        slot = slot_ref["slot_obj"]
        if slot.get("status") != "verbatim":
            continue

        content = slot.get("content")
        if not isinstance(content, str) or not content.strip():
            action = "downgraded_missing_content" if repair else "missing_content"
            score: float | None = 0.0
            if repair:
                slot["status"] = "paraphrased"
                downgraded += 1
            else:
                unverified += 1
            records.append(_record(slot_ref, score, threshold, action, content))
            continue

        if not normalized_source:
            action = "unverified_no_source_text"
            score = None
            unverified += 1
            records.append(_record(slot_ref, score, threshold, action, content))
            continue

        score, span_scores = _best_verbatim_score(content, normalized_source)
        if score >= threshold:
            passed += 1
            records.append(_record(slot_ref, score, threshold, "passed", content, span_scores=span_scores))
        else:
            action = "downgraded_low_match" if repair else "would_downgrade_low_match"
            if repair:
                slot["status"] = "paraphrased"
                downgraded += 1
            else:
                unverified += 1
            records.append(_record(slot_ref, score, threshold, action, content, span_scores=span_scores))

    sample_repairs: list[dict[str, Any]] = []
    repair_schema_report = None
    if repair and source_text:
        sample_repairs = _repair_sample_values_from_source(data, source_text)
        if sample_repairs:
            data, repair_schema_report = validate_paper(data, repair=True, path=path)

    sample_records = verify_sample_fields(data, source_text)
    sample_passed = sum(1 for record in sample_records if record.action == "passed")
    sample_unverified = sum(1 for record in sample_records if record.action != "passed")

    # Hard sample failures: numeric facts (N, age, gender %) and platform not found in source.
    # These indicate real data integrity issues and block the gate.
    # Soft failures: verbatim text fields (inclusion/exclusion_criteria, notes) that are
    # paraphrases. In new extractions these will be null (and not checked); in migrated
    # old data they may be paraphrases, which is a quality warning but not a hard blocker.
    _HARD_SAMPLE_FIELDS = frozenset({"total_n", "analyzed_n", "mean_age", "female_percent",
                                     "male_percent", "platform", "country"})
    sample_hard_failures = sum(
        1 for r in sample_records
        if r.action != "passed" and r.field in _HARD_SAMPLE_FIELDS
    )

    data, final_schema_report = validate_paper(data, repair=True, path=path)
    empty_slot_records = _empty_required_slot_records(data)
    empty_slots_by_status = _count_empty_slots_by_status(empty_slot_records)
    report = {
        "path": str(path) if path is not None else None,
        "threshold": threshold,
        "passed": final_schema_report.valid and unverified == 0 and sample_hard_failures == 0,
        "summary": {
            "verbatim_slots": len(records),
            "passed": passed,
            "downgraded": downgraded,
            "unverified": unverified,
            "sample_fields": len(sample_records),
            "sample_passed": sample_passed,
            "sample_unverified": sample_unverified,
            "sample_hard_failures": sample_hard_failures,
            "sample_repairs": len(sample_repairs),
            "empty_required_slots": len(empty_slot_records),
            "empty_required_slots_by_status": empty_slots_by_status,
            "stage4_ready": len(empty_slot_records) == 0,
            "schema_valid": final_schema_report.valid,
            "schema_fixed": (
                schema_report.fixed_count
                + (repair_schema_report.fixed_count if repair_schema_report is not None else 0)
                + final_schema_report.fixed_count
            ),
        },
        "records": [record.to_dict() for record in records],
        "sample_records": [record.to_dict() for record in sample_records],
        "sample_repairs": sample_repairs,
        "empty_required_slot_records": empty_slot_records,
        "schema_report": final_schema_report.to_dict(),
    }
    return data, report


def verify_json_file(
    json_path: Path,
    source_paths: list[Path],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    write: bool = True,
    backup: bool = True,
    report_path: Path | None = None,
    write_report: bool = True,
    repair: bool = True,
) -> dict[str, Any]:
    """Verify one JSON file against one or more PDF/text source files."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "eligible_studies" not in data:
        report = {
            "json_path": str(json_path),
            "source_paths": [],
            "skipped": True,
            "passed": True,
            "reason": "not a per-paper corpus JSON",
            "summary": {
                "verbatim_slots": 0,
                "passed": 0,
                "downgraded": 0,
                "unverified": 0,
                "sample_fields": 0,
                "sample_passed": 0,
                "sample_unverified": 0,
                "empty_required_slots": 0,
                "empty_required_slots_by_status": {},
                "stage4_ready": True,
                "schema_valid": True,
                "schema_fixed": 0,
            },
            "records": [],
            "sample_records": [],
            "empty_required_slot_records": [],
        }
        if write_report:
            if report_path is None:
                report_path = json_path.parent / f"{json_path.stem}_verification_report.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            report["report_path"] = str(report_path)
        return report

    source_text, loaded_sources = load_source_text(source_paths)
    repaired, report = verify_paper(
        data,
        source_text,
        threshold=threshold,
        repair=repair,
        path=json_path,
    )
    report["json_path"] = str(json_path)
    report["source_paths"] = [str(path) for path in loaded_sources]

    if write and report["passed"] and _json_changed(data, repaired):
        _write_json_atomic(json_path, repaired, backup=backup)
    elif write and not report["passed"]:
        report["write_skipped_reason"] = "verification did not pass"

    if write_report:
        if report_path is None:
            report_path = json_path.parent / "verification_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(report_path)
    return report


def verify_files(
    json_paths: list[Path],
    *,
    pdf_paths: list[Path] | None = None,
    source_paths: list[Path] | None = None,
    source_dirs: list[Path] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    write: bool = True,
    backup: bool = True,
    aggregate_report_path: Path | None = None,
    repair: bool = True,
) -> dict[str, Any]:
    """Verify multiple JSON files and write per-paper plus optional aggregate reports."""
    json_paths = [Path(path) for path in json_paths]
    pdf_paths = [Path(path) for path in (pdf_paths or [])]
    base_sources = [Path(path) for path in (source_paths or [])]
    for source_dir in source_dirs or []:
        base_sources.extend(_files_under(Path(source_dir)))

    reports: list[dict[str, Any]] = []
    for index, json_path in enumerate(json_paths):
        paired_sources = list(base_sources)
        if len(pdf_paths) == len(json_paths):
            paired_sources.append(pdf_paths[index])
        else:
            paired_sources.extend(pdf_paths)
        report = verify_json_file(
            json_path,
            paired_sources,
            threshold=threshold,
            write=write,
            backup=backup,
            write_report=aggregate_report_path is None,
            repair=repair,
        )
        reports.append(report)

    aggregate = {
        "threshold": threshold,
        "write": write,
        "summary": {
            "files": len(reports),
            "skipped": sum(1 for item in reports if item.get("skipped")),
            "passed": sum(1 for item in reports if item.get("passed")),
            "failed": sum(1 for item in reports if not item.get("passed")),
            "verbatim_slots": sum(item["summary"]["verbatim_slots"] for item in reports),
            "downgraded": sum(item["summary"]["downgraded"] for item in reports),
            "unverified": sum(item["summary"]["unverified"] for item in reports),
            "sample_fields": sum(item["summary"].get("sample_fields", 0) for item in reports),
            "sample_passed": sum(item["summary"].get("sample_passed", 0) for item in reports),
            "sample_unverified": sum(item["summary"].get("sample_unverified", 0) for item in reports),
            "empty_required_slots": sum(item["summary"].get("empty_required_slots", 0) for item in reports),
            "stage4_ready_files": sum(
                1 for item in reports
                if not item.get("skipped") and item["summary"].get("stage4_ready", False)
            ),
            "stage4_not_ready_files": sum(
                1 for item in reports
                if not item.get("skipped") and not item["summary"].get("stage4_ready", False)
            ),
        },
        "files": reports,
    }
    if aggregate_report_path is not None:
        aggregate_report_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_report_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate


def load_source_text(source_paths: Iterable[Path]) -> tuple[str, list[Path]]:
    """Load text from PDF and text-like source files."""
    chunks: list[str] = []
    loaded: list[Path] = []
    for source_path in source_paths:
        source_path = Path(source_path)
        if not source_path.exists() or source_path.is_dir():
            continue
        try:
            if source_path.suffix.lower() == ".pdf":
                text = extract_pdf_text(source_path, max_chars=PDF_TEXT_MAX_CHARS)
            else:
                text = _read_text_like_file(source_path)
        except Exception as exc:
            chunks.append(f"\n[Could not read {source_path}: {exc}]\n")
            continue
        if text.strip():
            loaded.append(source_path)
            chunks.append(f"\n--- Source: {source_path} ---\n{text}")
    return "\n\n".join(chunks), loaded


def normalize_for_match(text: str) -> str:
    """Normalize text before fuzzy matching."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"--- Page \d+ ---", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def best_partial_ratio(needle: str, haystack: str) -> float:
    """Return a 0-100 partial fuzzy match score."""
    if not needle or not haystack:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.partial_ratio(needle, haystack))
    except ImportError:
        return _difflib_partial_ratio(needle, haystack)


def verify_sample_fields(data: dict[str, Any], source_text: str) -> list[SampleRecord]:
    """Verify structured sample fields without fuzzy matching or LLM calls.

    Study-level sample string/numeric fields are checked with field-specific rules.
    Effect-level analysis_n is also checked here (number-in-source, no context).

    This is intentionally high-precision: paraphrases are reported as unverified
    rather than treated as passing.
    """
    source_windows = _source_windows(source_text)
    normalized_source = normalize_for_match(source_text)
    records: list[SampleRecord] = []

    # Study-level sample claims (total_n requires context; others do not)
    for claim in _iter_sample_claims(data):
        value = claim["value"]
        if _empty_sample_value(value):
            continue
        if isinstance(value, str):
            record = _verify_sample_string_claim(claim, value, source_windows, normalized_source)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            record = _verify_sample_numeric_claim(claim, value, source_windows)
        else:
            continue
        records.append(record)

    # Effect-level analysis_n claims (no context required — these appear near stats)
    for claim in _iter_effect_analysis_claims(data):
        value = claim["value"]
        if _empty_sample_value(value) or not isinstance(value, (int, float)):
            continue
        records.append(_verify_sample_numeric_claim(claim, value, source_windows))

    return records


def _repair_sample_values_from_source(data: dict[str, Any], source_text: str) -> list[dict[str, Any]]:
    """Repair old study.sample values using source-backed evidence.

    This is intentionally conservative and source-aware.  It only fills or
    replaces study.sample.total_n when the paper reports a study-level final N
    near the study label (e.g. a sample-information table row).  Analysis-table
    Ns that were previously stored in study.sample.total_n are copied to matching
    effect.analysis_n, then the study-level total_n is corrected.
    """
    windows = _source_windows(source_text)
    repairs: list[dict[str, Any]] = []

    for study_index, study in enumerate(data.get("eligible_studies", [])):
        if not isinstance(study, dict):
            continue
        sample = study.get("sample")
        if not isinstance(sample, dict):
            continue

        study_name = study.get("study") or f"study[{study_index}]"
        current_total = _coerce_int(sample.get("total_n"))
        source_total = _find_study_level_total_n(study_name, windows)
        if source_total is not None and source_total != current_total:
            if current_total is not None:
                _copy_matching_effect_size_to_analysis_n(
                    study,
                    current_total,
                    source_total,
                    repairs,
                    reason="old study.sample.total_n matched an effect-level analysis N",
                )
            before = sample.get("total_n")
            sample["total_n"] = source_total
            repairs.append(
                {
                    "path": f"$.eligible_studies[{study_index}].sample.total_n",
                    "study": study_name,
                    "field": "total_n",
                    "before": before,
                    "after": source_total,
                    "action": "repaired_from_study_level_source",
                }
            )

        repaired_total = _coerce_int(sample.get("total_n"))
        if repaired_total is not None:
            _fill_effect_analysis_n_from_size(study, repaired_total, repairs)

    return repairs


def _find_study_level_total_n(study_name: Any, source_windows: list[str]) -> int | None:
    """Find a study-level final sample N from source windows.

    Prefer explicit final-N / final-sample language tied to the study label.
    This intentionally ignores generic analysis table rows such as Table 4
    simple-effect Ns.
    """
    study_id = _study_id(study_name)
    if not study_id:
        return None

    patterns = _study_total_n_patterns(study_id)
    for window in source_windows:
        normalized = normalize_for_match(window)
        if not _has_study_level_sample_language(normalized):
            continue
        prose_total = _find_final_sample_prose_total(study_id, normalized)
        if prose_total is not None:
            return prose_total
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.I)
            if match:
                return _parse_int(match.group("n"))
    return None


def _find_final_sample_prose_total(study_id: str, normalized_window: str) -> int | None:
    """Parse prose like "final samples consisted of n=... in Study 1a and n=... in Study 1b"."""
    escaped = re.escape(study_id)
    for marker in re.finditer(r"\bfinal\s+samples?\b", normalized_window, flags=re.I):
        segment = normalized_window[marker.start() : marker.start() + 1200]
        segment = re.split(
            r"\.\s+(?=(?:in\s+study|preregistration|we\s+|table|for\s+|our\s+|studies?\b))",
            segment,
            maxsplit=1,
            flags=re.I,
        )[0]
        match = re.search(
            rf"\bn\s*=\s*(?P<n>[\d,]+)(?:(?!\bn\s*=).){{0,300}}?\bstudy\s*{escaped}\b",
            segment,
            flags=re.I | re.S,
        )
        if match:
            return _parse_int(match.group("n"))
        match = re.search(
            rf"\bstudy\s*{escaped}\b(?:(?!\bn\s*=).){{0,300}}?"
            rf"\bn\s*=\s*(?P<n>[\d,]+)",
            segment,
            flags=re.I | re.S,
        )
        if match:
            return _parse_int(match.group("n"))
    return None


def _study_total_n_patterns(study_id: str) -> list[str]:
    escaped = re.escape(study_id)
    token = rf"(?<![\w,]){escaped}(?![\w,])"
    month = (
        r"(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)"
    )
    return [
        # Table-style rows: "3 March 2021 ... final n = 4,001"
        rf"(?:^|\s){token}\s+(?:{month}|[a-z]+(?:\s*[–-]\s*[a-z]+)?)[^.{{}}]{{0,500}}?"
        rf"\bfinal\s+n\s*=\s*(?P<n>[\d,]+)",
        # Explicit prose: "Study 3 ... final n = 4,001"
        rf"\bstudy\s*{escaped}\b[^.{{}}]{{0,500}}?\bfinal\s+n\s*=\s*(?P<n>[\d,]+)",
    ]


def _has_study_level_sample_language(normalized_window: str) -> bool:
    return any(
        phrase in normalized_window
        for phrase in (
            "final n",
            "final sample",
            "sample information",
            "sample demographics",
            "target n",
            "we recruited",
            "recruited a target",
            "our final sample",
            "final samples consisted",
        )
    )


def _study_id(study_name: Any) -> str | None:
    text = normalize_for_match(str(study_name or ""))
    match = re.search(r"\b(?:study|experiment)\s*(\d+[a-z]?)\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d+[a-z]?)\b", text)
    return match.group(1) if match else None


def _copy_matching_effect_size_to_analysis_n(
    study: dict[str, Any],
    old_total_n: int,
    study_total_n: int | None,
    repairs: list[dict[str, Any]],
    *,
    reason: str,
) -> None:
    for effect_index, effect in enumerate(study.get("effects", [])):
        if not isinstance(effect, dict):
            continue
        size = _coerce_int(effect.get("size"))
        if size != old_total_n or effect.get("analysis_n") is not None:
            continue
        effect["analysis_n"] = old_total_n
        effect.setdefault("analysis_scope", _infer_analysis_scope(effect, old_total_n, study_total_n))
        repairs.append(
            {
                "path": f"{study.get('study', 'study')}.effects[{effect_index}].analysis_n",
                "study": study.get("study"),
                "effect_index": effect_index,
                "field": "analysis_n",
                "before": None,
                "after": old_total_n,
                "action": "moved_from_study_total_n",
                "reason": reason,
            }
        )


def _fill_effect_analysis_n_from_size(
    study: dict[str, Any],
    study_total_n: int,
    repairs: list[dict[str, Any]],
) -> None:
    for effect_index, effect in enumerate(study.get("effects", [])):
        if not isinstance(effect, dict):
            continue
        size = _coerce_int(effect.get("size"))
        existing_analysis_n = _coerce_int(effect.get("analysis_n"))
        if existing_analysis_n == study_total_n:
            before_n = effect.pop("analysis_n", None)
            before_scope = effect.pop("analysis_scope", None)
            repairs.append(
                {
                    "path": f"{study.get('study', 'study')}.effects[{effect_index}].analysis_n",
                    "study": study.get("study"),
                    "effect_index": effect_index,
                    "field": "analysis_n",
                    "before": before_n,
                    "after": None,
                    "action": "removed_redundant_full_sample_analysis_n",
                    "analysis_scope_before": before_scope,
                }
            )
            continue
        if size is None or size == study_total_n or effect.get("analysis_n") is not None:
            continue
        effect["analysis_n"] = size
        effect.setdefault("analysis_scope", _infer_analysis_scope(effect, size, study_total_n))
        repairs.append(
            {
                "path": f"{study.get('study', 'study')}.effects[{effect_index}].analysis_n",
                "study": study.get("study"),
                "effect_index": effect_index,
                "field": "analysis_n",
                "before": None,
                "after": size,
                "action": "filled_from_effect_size",
            }
        )


def _infer_analysis_scope(effect: dict[str, Any], analysis_n: int, study_total_n: int | None) -> str:
    effecttype = normalize_for_match(effect.get("effecttype") or "")
    if effecttype == "simple":
        return "simple_effect"
    if study_total_n is not None and analysis_n < study_total_n:
        return "subgroup"
    if effecttype == "int":
        return "subgroup"
    return "other"


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.replace(",", "").strip()
        if re.fullmatch(r"\d+", text):
            return int(text)
    return None


def _parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def _iter_effect_analysis_claims(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield analysis_n claims from effect-level fields.

    effect.analysis_n is the N used in a *specific* analysis (e.g. a subgroup,
    a simple-effect cell, or a condition subset).  It is separate from
    study.sample.total_n (the full recruited study N).
    """
    for study_index, study in enumerate(data.get("eligible_studies", [])):
        if not isinstance(study, dict):
            continue
        study_name = study.get("study")
        for effect_index, effect in enumerate(study.get("effects", [])):
            if not isinstance(effect, dict):
                continue
            analysis_n = effect.get("analysis_n")
            if analysis_n is not None and isinstance(analysis_n, (int, float)) and not isinstance(analysis_n, bool):
                yield _sample_claim(
                    f"$.eligible_studies[{study_index}].effects[{effect_index}].analysis_n",
                    study_name,
                    effect_index,
                    "analysis_n",
                    int(analysis_n),
                )


def _difflib_partial_ratio(needle: str, haystack: str) -> float:
    import difflib

    if needle in haystack:
        return 100.0
    window = min(max(len(needle) * 2, 500), max(len(haystack), 1))
    step = max(len(needle) // 2, 200)
    best = 0.0
    for start in range(0, max(len(haystack) - window + 1, 1), step):
        chunk = haystack[start : start + window]
        best = max(best, difflib.SequenceMatcher(None, needle, chunk).ratio() * 100.0)
        if best >= 99.0:
            break
    return best


def _iter_slots(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for study_index, study in enumerate(data.get("eligible_studies", [])):
        if not isinstance(study, dict):
            continue
        study_name = study.get("study")
        for effect_index, effect in enumerate(study.get("effects", [])):
            if not isinstance(effect, dict):
                continue
            for slot_name in SLOT_NAMES:
                slot = effect.get(slot_name)
                if not isinstance(slot, dict):
                    continue
                yield {
                    "path": f"$.eligible_studies[{study_index}].effects[{effect_index}].{slot_name}",
                    "study": study_name,
                    "effect_index": effect_index,
                    "slot": slot_name,
                    "slot_obj": slot,
                }


def _empty_required_slot_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for slot_ref in _iter_slots(data):
        slot = slot_ref["slot_obj"]
        status = slot.get("status")
        content = slot.get("content")
        if isinstance(content, str) and content.strip():
            continue
        if content is not None and not isinstance(content, str):
            continue
        records.append(
            {
                "path": slot_ref["path"],
                "study": slot_ref["study"],
                "effect_index": slot_ref["effect_index"],
                "slot": slot_ref["slot"],
                "status": status,
                "reason": "required slot has no content",
            }
        )
    return records


def _count_empty_slots_by_status(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = record.get("status")
        key = "null" if status is None else str(status)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _iter_sample_claims(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Iterate verifiable claims from study-level sample objects.

    New schema: sample lives at study level, not per-effect.
    Falls back to effect[0].sample for backward-compat with old JSONs.
    """
    for study_index, study in enumerate(data.get("eligible_studies", [])):
        if not isinstance(study, dict):
            continue
        study_name = study.get("study")

        # New format: study.sample
        sample = study.get("sample")
        if isinstance(sample, (dict, str)):
            base_path = f"$.eligible_studies[{study_index}].sample"
            if isinstance(sample, str):
                yield _sample_claim(base_path, study_name, 0, "sample", sample)
            else:
                yield from _iter_study_sample_claims(sample, base_path, study_name)
            continue

        # Backward compat: old effect-level sample (effect[0] only to avoid duplication)
        effects = study.get("effects") or []
        if effects and isinstance(effects[0], dict):
            old_sample = effects[0].get("sample")
            if isinstance(old_sample, dict):
                base_path = f"$.eligible_studies[{study_index}].effects[0].sample"
                yield from _iter_study_sample_claims(
                    _migrate_old_sample_for_verify(old_sample), base_path, study_name
                )


def _migrate_old_sample_for_verify(old: dict[str, Any]) -> dict[str, Any]:
    """Best-effort convert old nested sample → flat for verification pass."""
    from generation_pipeline.verification.schema_validator import _migrate_old_sample
    try:
        return _migrate_old_sample(old)
    except Exception:
        return {}


def _iter_study_sample_claims(
    sample: dict[str, Any],
    base_path: str,
    study_name: str | None,
) -> Iterable[dict[str, Any]]:
    """Yield one claim per non-null sample field for the new flat study.sample schema."""
    # Numeric fields — verified as number-in-source (no context requirement)
    for field in ("total_n", "analyzed_n"):
        val = sample.get(field)
        if val is not None:
            yield _sample_claim(f"{base_path}.{field}", study_name, 0, field, val)

    # Float fields — age in years, gender in 0-100 percent
    for field in ("mean_age", "female_percent", "male_percent"):
        val = sample.get(field)
        if val is not None:
            yield _sample_claim(f"{base_path}.{field}", study_name, 0, field, val)

    # Categorical / short string fields — platform alias match or substring
    for field in ("platform", "country"):
        val = sample.get(field)
        if val is not None:
            yield _sample_claim(f"{base_path}.{field}", study_name, 0, field, val)

    # Verbatim text fields — fuzzy match ≥ 90% required
    for field in ("inclusion_criteria", "exclusion_criteria", "notes"):
        val = sample.get(field)
        if val is not None:
            yield _sample_claim(f"{base_path}.{field}", study_name, 0, field, val)


def _sample_claim(
    path: str,
    study_name: str | None,
    effect_index: int,
    field: str,
    value: Any,
) -> dict[str, Any]:
    return {
        "path": path,
        "study": study_name,
        "effect_index": effect_index,
        "field": field,
        "value": value,
    }


def _empty_sample_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _verify_sample_string_claim(
    claim: dict[str, Any],
    value: str,
    source_windows: list[str],
    normalized_source: str,
) -> SampleRecord:
    field = claim["field"]

    # Platform: check against known aliases (substring match)
    if field == "platform":
        evidence = _find_platform_alias_window(value, source_windows)
        return _sample_record(
            claim, value,
            evidence_kind="platform_alias_substring",
            action="passed" if evidence else "unverified_platform_not_found",
            evidence=evidence,
        )

    # Country: simple case-insensitive substring match
    if field == "country":
        evidence = _find_substring_window(value, source_windows)
        return _sample_record(
            claim, value,
            evidence_kind="country_substring",
            action="passed" if evidence else "unverified_country_not_found",
            evidence=evidence,
        )

    # Verbatim text fields (inclusion_criteria, exclusion_criteria, notes):
    # require fuzzy match ≥ threshold against full source.
    if field in ("inclusion_criteria", "exclusion_criteria", "notes"):
        score = best_partial_ratio(normalize_for_match(value), normalized_source)
        passed = score >= DEFAULT_THRESHOLD
        return _sample_record(
            claim, value,
            evidence_kind="fuzzy_verbatim",
            action="passed" if passed else "unverified_verbatim_below_threshold",
            evidence=f"score={score:.1f}" if passed else None,
        )

    # Legacy fallback: normalized exact text
    evidence = _find_normalized_exact_window(value, source_windows, normalized_source)
    return _sample_record(
        claim, value,
        evidence_kind="normalized_exact_text",
        action="passed" if evidence else "unverified_no_exact_source_span",
        evidence=evidence,
    )


def _verify_sample_numeric_claim(
    claim: dict[str, Any],
    value: int | float,
    source_windows: list[str],
) -> SampleRecord:
    field = claim["field"]
    if field == "total_n":
        source_total = _find_study_level_total_n(claim.get("study"), source_windows)
        value_int = _coerce_int(value)
        if source_total is not None:
            evidence = None
            action = "unverified_not_study_final_n"
            if value_int == source_total:
                evidence = _find_numeric_evidence(
                    _numeric_variants(source_total, field),
                    _numeric_context_terms(field),
                    source_windows,
                )
                action = "passed" if evidence else "unverified_number_not_found"
            return _sample_record(
                claim, value,
                evidence_kind="study_final_n_exact",
                action=action,
                evidence=evidence,
            )

    variants = _numeric_variants(value, field)
    # N fields: just find the number anywhere — no context required.
    # Age/percent fields: same — the number itself is distinctive enough.
    require_context = _numeric_requires_context(field)
    context_terms = _numeric_context_terms(field) if require_context else ()
    evidence = _find_numeric_evidence(variants, context_terms, source_windows)
    return _sample_record(
        claim, value,
        evidence_kind="numeric_in_source" if not require_context else "numeric_context_exact",
        action="passed" if evidence else "unverified_number_not_found",
        evidence=evidence,
    )


def _sample_record(
    claim: dict[str, Any],
    value: Any,
    *,
    evidence_kind: str,
    action: str,
    evidence: str | None,
) -> SampleRecord:
    return SampleRecord(
        path=claim["path"],
        study=claim["study"],
        effect_index=claim["effect_index"],
        field=claim["field"],
        value_preview=_preview(value),
        evidence_kind=evidence_kind,
        action=action,
        evidence_preview=_preview(evidence, limit=320) if evidence else None,
    )


def _source_windows(source_text: str) -> list[str]:
    text = source_text or ""
    candidates: list[str] = []
    candidates.extend(part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip())
    candidates.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip())
    if not candidates and text.strip():
        candidates.append(text.strip())

    windows: list[str] = []
    for candidate in candidates:
        if len(candidate) <= 1200:
            windows.append(candidate)
            continue
        for start in range(0, len(candidate), 900):
            chunk = candidate[start : start + 1200].strip()
            if chunk:
                windows.append(chunk)
    return windows


def _find_normalized_exact_window(
    value: str,
    source_windows: list[str],
    normalized_source: str,
) -> str | None:
    normalized_value = normalize_for_match(value)
    if not normalized_value:
        return None
    for window in source_windows:
        if normalized_value in normalize_for_match(window):
            return window
    if normalized_value in normalized_source:
        return "Exact normalized text is present in the full source corpus, but no compact source window was isolated."
    return None


def _find_substring_window(value: str, source_windows: list[str]) -> str | None:
    """Case-insensitive substring search across source windows."""
    needle = value.lower().strip()
    if not needle:
        return None
    for window in source_windows:
        if needle in window.lower():
            return window
    return None


def _find_platform_alias_window(value: str, source_windows: list[str]) -> str | None:
    aliases = SAMPLE_PLATFORM_ALIASES.get(normalize_for_match(value), (normalize_for_match(value),))
    aliases = tuple(alias for alias in aliases if alias)
    if not aliases:
        return None
    for window in source_windows:
        normalized_window = normalize_for_match(window)
        if any(alias in normalized_window for alias in aliases):
            return window
    return None


def _numeric_variants(value: int | float, field: str) -> list[str]:
    """Return candidate string forms of a numeric value to search for in source text.

    Handles integer N values, age means, and gender percentages (always 0-100 in
    the new schema but we also search for the 0-1 proportion form as a fallback
    since papers sometimes report either).
    """
    variants: set[str] = set()
    if isinstance(value, int) or (isinstance(value, float) and float(value).is_integer()):
        integer = int(value)
        variants.add(str(integer))
        variants.add(f"{integer:,}")           # thousand separator
        variants.add(f"n = {integer}")
        variants.add(f"n={integer}")
        variants.add(f"n = {integer:,}")
        variants.add(f"n={integer:,}")
        variants.add(f"n= {integer}")
    else:
        number = float(value)
        variants.add(_format_float(number))
        variants.add(f"{number:.2f}")
        variants.add(f"{number:.1f}")

        # Gender percentages: new schema stores 0-100; paper may write as % or proportion.
        if field in ("female_percent", "male_percent"):
            # 0-100 forms
            variants.add(f"{number:.2f}%")
            variants.add(f"{number:.1f}%")
            variants.add(f"{number:.0f}%")
            # proportion form (÷100)
            prop = number / 100.0
            variants.add(_format_float(prop))
            variants.add(f"{prop:.4f}")
            variants.add(f"{prop:.3f}")
            variants.add(f"{prop:.2f}")

        # Age mean: common formats
        if field == "mean_age":
            variants.add(f"m = {number:.2f}")
            variants.add(f"m={number:.2f}")
            variants.add(f"mage = {number:.2f}")
            variants.add(f"mage={number:.2f}")
            variants.add(f"mean age = {number:.2f}")

    return sorted(variants, key=len, reverse=True)


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _numeric_requires_context(field: str) -> bool:
    """Return True when the bare number is ambiguous and a context word is needed.

    `total_n` REQUIRES context to avoid confusing the study-level recruited N
    with a subgroup / analysis / cell N that appears elsewhere in the paper
    (e.g. in a Table row reporting a simple-effect with n = 958).

    `analyzed_n` also requires participant-language context for the same reason.

    All other numeric fields (age, gender %) are distinctive enough that the
    bare number is sufficient.

    `analysis_n` (effect-level) intentionally uses NO context — it is supposed
    to match a stats-adjacent subgroup number; see _iter_effect_analysis_claims.
    """
    return field in ("total_n", "analyzed_n")


def _numeric_context_terms(field: str) -> tuple[str, ...]:
    """Context keywords that must appear in the same source window as the number."""
    if field in ("total_n", "analyzed_n"):
        # Study-level N must be near participant/recruitment/final-N language.
        # Do not use bare "sample": analysis tables often have a "Sample" column
        # header next to subgroup Ns, which are effect.analysis_n, not total_n.
        return (
            "participant", "participants", "subject", "subjects",
            "recruited", "enrolled", "respondent", "respondents",
            "final n", "final sample", "our final sample",
            "completed the study",
        )
    if field == "mean_age":
        return ("age", "mage", "mean age", "years")
    if field == "female_percent":
        return ("female", "women", "woman")
    if field == "male_percent":
        return ("male", "men", "man")
    return ()


def _find_numeric_evidence(
    variants: list[str],
    context_terms: tuple[str, ...],
    source_windows: list[str],
) -> str | None:
    for window in source_windows:
        normalized_window = normalize_for_match(window)
        if not any(_contains_numeric_variant(normalized_window, variant) for variant in variants):
            continue
        if context_terms and not any(term in normalized_window for term in context_terms):
            continue
        return window
    return None


def _contains_numeric_variant(normalized_window: str, variant: str) -> bool:
    normalized_variant = normalize_for_match(variant)
    if not normalized_variant:
        return False
    # Use \d only (not [\d.]) so that end-of-sentence periods (e.g., "n = 958.") don't block
    # a match. We still prevent matches inside longer numbers ("1958" won't match "958").
    return re.search(rf"(?<!\d){re.escape(normalized_variant)}(?!\d)", normalized_window) is not None


def _best_verbatim_score(content: str, normalized_source: str) -> tuple[float, list[dict[str, Any]]]:
    """Score a verbatim slot.

    Multi-span Stage 3 content is stored as labeled blocks:

        [intro]
        exact quote A

        [manipulation: public]
        exact quote B

    Those labels and joins are not a contiguous source quote, so verifying the
    whole string creates false negatives. If labeled spans are present, score
    each quote separately and use the weakest span as the slot score.
    """
    spans = _labeled_verbatim_spans(content)
    if not spans:
        return best_partial_ratio(normalize_for_match(content), normalized_source), []

    span_scores: list[dict[str, Any]] = []
    weakest = 100.0
    for label, quote in spans:
        score = best_partial_ratio(normalize_for_match(quote), normalized_source)
        weakest = min(weakest, score)
        span_scores.append(
            {
                "label": label,
                "score": round(score, 2),
                "chars": len(quote),
                "content_preview": _preview(quote),
            }
        )
    return weakest, span_scores


def _labeled_verbatim_spans(content: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"(?:^|\n)\[(?P<label>[^\]\n]{1,80})\]\n(?P<quote>.*?)(?=(?:\n\n\[[^\]\n]{1,80}\]\n)|\Z)",
        flags=re.S,
    )
    spans: list[tuple[str, str]] = []
    for match in pattern.finditer(content.strip()):
        label = match.group("label").strip()
        quote = match.group("quote").strip()
        # Avoid treating ordinary numeric scale labels such as "[1]" as
        # multi-span assembly labels.
        if not re.search(r"[A-Za-z]", label):
            continue
        if len(quote) < 20:
            continue
        spans.append((label, quote))
    return spans


def _record(
    slot_ref: dict[str, Any],
    score: float | None,
    threshold: float,
    action: str,
    content: Any,
    *,
    span_scores: list[dict[str, Any]] | None = None,
) -> VerbatimRecord:
    return VerbatimRecord(
        path=slot_ref["path"],
        study=slot_ref["study"],
        effect_index=slot_ref["effect_index"],
        slot=slot_ref["slot"],
        score=round(score, 2) if isinstance(score, float) else score,
        threshold=threshold,
        action=action,
        content_preview=_preview(content),
        span_scores=span_scores or [],
    )


def _preview(content: Any, limit: int = 220) -> str:
    text = "" if content is None else str(content)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_changed(before: Any, after: Any) -> bool:
    return json.dumps(before, sort_keys=True, ensure_ascii=False) != json.dumps(after, sort_keys=True, ensure_ascii=False)


def _write_json_atomic(path: Path, data: dict[str, Any], *, backup: bool) -> None:
    import shutil

    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _read_text_like_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".qsf", ".csv", ".tsv", ".html", ".htm", ".xml", ".do", ".r", ".py"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.count("\x00") > max(1, len(text) // 20):
        return ""
    return text


def _files_under(source_dir: Path) -> list[Path]:
    if not source_dir.exists() or not source_dir.is_dir():
        return []
    return [path for path in sorted(source_dir.rglob("*")) if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify schema, verbatim slots, and structured sample fields")
    parser.add_argument("--json", nargs="+", required=True, help="One or more per-paper JSON files")
    parser.add_argument("--pdf", nargs="*", default=[], help="PDF source files")
    parser.add_argument("--source", nargs="*", default=[], help="Additional text-like source files")
    parser.add_argument("--source-dir", nargs="*", default=[], help="Additional source directories")
    parser.add_argument("--dry-run", action="store_true", help="Do not write JSON downgrades")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional aggregate report path")
    args = parser.parse_args()

    aggregate = verify_files(
        [Path(item) for item in args.json],
        pdf_paths=[Path(item) for item in args.pdf],
        source_paths=[Path(item) for item in args.source],
        source_dirs=[Path(item) for item in args.source_dir],
        write=not args.dry_run,
        backup=not args.no_backup,
        aggregate_report_path=args.report,
        repair=not args.dry_run,
    )
    print(json.dumps(aggregate["summary"], indent=2, ensure_ascii=False))
    raise SystemExit(1 if aggregate["summary"]["failed"] else 0)


if __name__ == "__main__":
    main()
