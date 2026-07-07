#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SLOT_NAMES = ("materials", "manipulation", "items")
# "source_missing": external sources (OSF, etc.) were fetched and searched, but
# the material for this slot is genuinely not present in them — distinct from
# "osf_only" (believed to be on OSF, not yet fetched) and "not_in_paper".
VALID_SLOT_STATUSES = {"verbatim", "paraphrased", "cited_scale", "osf_only", "not_in_paper", "source_missing"}
VALID_SIG = {"sig", "ns", "marginal"}
VALID_DIRECTION = {"pos", "neg"}
VALID_EFFECT_TYPES = {"main", "int", "simple", "mediation", "correlation"}

STATS_NUMERIC_FIELDS = ("B", "b", "chi_square", "D", "eta_square", "f", "t", "z")
EFFECT_NUMERIC_FIELDS = ("mean_group1", "sd_group1", "mean_group2", "sd_group2")
EFFECT_REQUIRED_FIELDS = (
    # NOTE: "sample" is no longer an effect-level field — it lives at study level.
    "platform",
    "effecttype",
    "IV",
    "DV",
    "size",
    "direction",
    "mean_group1",
    "sd_group1",
    "mean_group2",
    "sd_group2",
    "stats",
    "materials_notes",
    "table_or_page_location",
    "materials",
    "manipulation",
    "items",
)

# Controlled vocabulary for sample.platform.
# Any of these values (plus "Other") are accepted without further normalization.
# The verifier knows platform-specific aliases (e.g. "MTurk" → "mechanical turk").
SAMPLE_PLATFORM_VOCAB = frozenset({
    "MTurk", "Prolific", "CloudResearch", "Undergraduate", "Graduate",
    "Lab", "Organizational", "Online", "Field", "Archival", "Mixed", "Other",
})

# Controlled vocabulary for effect.analysis_scope.
# Describes the scope of the participants used in a specific effect analysis,
# which may differ from the study-level total_n.
VALID_ANALYSIS_SCOPE = frozenset({
    "full_sample",    # all recruited participants were included
    "subgroup",       # a subset based on demographic/screening criterion
    "condition",      # participants in a specific condition only
    "cell",           # a single experimental cell (e.g., 2×2 design cell)
    "simple_effect",  # simple effect analysis within a level of a factor
    "other",          # any other analytical subset
})

STATS_REQUIRED_FIELDS = (*STATS_NUMERIC_FIELDS, "ci", "p_value", "sig")

NULL_STRINGS = {"", "null", "none", "n/a", "na", "nan", "not reported", "not_reported", "-"}
_MISSING = object()


@dataclass
class ValidationIssue:
    path: str
    message: str
    severity: str = "warning"
    fixed: bool = False
    before: Any = _MISSING
    after: Any = _MISSING

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
            "fixed": self.fixed,
        }
        if self.before is not _MISSING:
            out["before"] = self.before
        if self.after is not _MISSING:
            out["after"] = self.after
        return out


class SchemaValidationReport:
    """JSON-serializable schema validation report."""

    def __init__(self, *, path: str | None = None):
        self.path = path
        self.issues: list[ValidationIssue] = []
        self.changed = False

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" and not issue.fixed for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error" and not issue.fixed)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning" and not issue.fixed)

    @property
    def fixed_count(self) -> int:
        return sum(1 for issue in self.issues if issue.fixed)

    def add(
        self,
        path: str,
        message: str,
        *,
        severity: str = "warning",
        fixed: bool = False,
        before: Any = _MISSING,
        after: Any = _MISSING,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                path=path,
                message=message,
                severity=severity,
                fixed=fixed,
                before=_safe_value(before),
                after=_safe_value(after),
            )
        )
        if fixed:
            self.changed = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "valid": self.valid,
            "changed": self.changed,
            "summary": {
                "errors": self.error_count,
                "warnings": self.warning_count,
                "fixed": self.fixed_count,
                "issues": len(self.issues),
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_paper(
    paper: dict[str, Any],
    *,
    repair: bool = True,
    path: str | Path | None = None,
) -> tuple[dict[str, Any], SchemaValidationReport]:
    """
    Validate and optionally normalize a Stage 2/3 paper JSON object.

    Returns a deep-copied paper object plus a report. Deterministic repairs are
    applied only to the returned object, never to the caller's input.
    """
    report = SchemaValidationReport(path=str(path) if path is not None else None)
    data = deepcopy(paper)

    if not isinstance(data, dict):
        report.add("$", f"paper JSON must be an object, got {type(data).__name__}", severity="error")
        return paper, report

    _ensure_key(data, "paper_title", None, "$.paper_title", report, repair)
    _ensure_key(data, "paper_metadata", {}, "$.paper_metadata", report, repair)
    _ensure_key(data, "eligible_studies", [], "$.eligible_studies", report, repair)

    metadata = data.get("paper_metadata")
    if not isinstance(metadata, dict):
        if repair:
            report.add(
                "$.paper_metadata",
                "paper_metadata must be an object; replaced with empty metadata",
                fixed=True,
                before=metadata,
                after={},
            )
            metadata = {}
            data["paper_metadata"] = metadata
        else:
            report.add("$.paper_metadata", "paper_metadata must be an object", severity="error", before=metadata)
            metadata = {}

    _normalize_metadata(metadata, report, repair)

    studies = data.get("eligible_studies")
    if not isinstance(studies, list):
        report.add("$.eligible_studies", "eligible_studies must be a list", severity="error", before=studies)
        return data, report

    for study_index, study in enumerate(studies):
        study_path = f"$.eligible_studies[{study_index}]"
        if not isinstance(study, dict):
            report.add(study_path, "study entry must be an object", severity="error", before=study)
            continue
        _normalize_study(study, study_path, report, repair)

    return data, report


def validate_file(
    path: Path,
    *,
    repair: bool = True,
    write: bool = False,
    backup: bool = True,
) -> tuple[dict[str, Any] | None, SchemaValidationReport]:
    """Validate one JSON file and optionally write deterministic repairs."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        report = SchemaValidationReport(path=str(path))
        report.add("$", f"could not load JSON: {exc}", severity="error")
        return None, report

    if not _looks_like_paper_json(data):
        report = SchemaValidationReport(path=str(path))
        report.add("$", "skipped: JSON does not contain eligible_studies", severity="warning")
        return data, report

    repaired, report = validate_paper(data, repair=repair, path=path)
    if write and report.valid and report.changed:
        _write_json_atomic(Path(path), repaired, backup=backup)
    return repaired, report


def normalize_stage4_tree(
    stage4_dir: Path,
    *,
    write: bool = False,
    backup: bool = True,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Scan stage4/**/*.json, repair paper JSONs in memory, and write a report."""
    stage4_dir = Path(stage4_dir)
    reports: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for json_path in sorted(stage4_dir.glob("**/*.json")):
        rel = str(json_path.relative_to(stage4_dir))
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report = SchemaValidationReport(path=rel)
            report.add("$", f"could not load JSON: {exc}", severity="error")
            reports.append(report.to_dict())
            continue

        if not _looks_like_paper_json(data):
            skipped.append({"path": rel, "reason": "not a per-paper corpus JSON"})
            continue

        repaired, report = validate_paper(data, repair=True, path=rel)
        if write and report.valid and report.changed:
            _write_json_atomic(json_path, repaired, backup=backup)
        reports.append(report.to_dict())

    aggregate = {
        "stage4_dir": str(stage4_dir),
        "write": write,
        "summary": {
            "paper_jsons": len(reports),
            "skipped": len(skipped),
            "changed": sum(1 for item in reports if item["changed"]),
            "invalid": sum(1 for item in reports if not item["valid"]),
            "fixed": sum(item["summary"]["fixed"] for item in reports),
            "errors": sum(item["summary"]["errors"] for item in reports),
            "warnings": sum(item["summary"]["warnings"] for item in reports),
        },
        "files": reports,
        "skipped": skipped,
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    return aggregate


def summarize_report(report: SchemaValidationReport) -> str:
    """Return a compact human-readable summary."""
    return (
        f"valid={report.valid}, changed={report.changed}, "
        f"fixed={report.fixed_count}, errors={report.error_count}, warnings={report.warning_count}"
    )


def _normalize_metadata(metadata: dict[str, Any], report: SchemaValidationReport, repair: bool) -> None:
    for key, default in {"authors": [], "year": None, "journal": None, "doi": None, "link": None}.items():
        _ensure_key(metadata, key, default, f"$.paper_metadata.{key}", report, repair)

    authors = metadata.get("authors")
    if isinstance(authors, str):
        _replace(metadata, "authors", [authors], "$.paper_metadata.authors", report, repair, "authors must be a list")
    elif authors is None:
        _replace(metadata, "authors", [], "$.paper_metadata.authors", report, repair, "authors must be a list")
    elif not isinstance(authors, list):
        report.add("$.paper_metadata.authors", "authors must be a list", severity="error", before=authors)

    _normalize_int(metadata, "year", "$.paper_metadata.year", report, repair, nullable=True)
    for key in ("journal", "doi", "link"):
        _normalize_nullable_string(metadata, key, f"$.paper_metadata.{key}", report, repair)


def _normalize_study(study: dict[str, Any], path: str, report: SchemaValidationReport, repair: bool) -> None:
    _ensure_key(study, "study", None, f"{path}.study", report, repair)
    _ensure_key(study, "eligibility_rationale", None, f"{path}.eligibility_rationale", report, repair)
    _ensure_key(study, "effects", [], f"{path}.effects", report, repair)
    _ensure_key(study, "sample", None, f"{path}.sample", report, repair)
    _normalize_nullable_string(study, "study", f"{path}.study", report, repair)
    _normalize_nullable_string(study, "eligibility_rationale", f"{path}.eligibility_rationale", report, repair)

    # Migration: lift effect-level sample to study level (one-time, repair=True only).
    # Old format had `sample` inside each effect; new format has it at study level.
    if repair and study.get("sample") is None:
        effects_list = study.get("effects") or []
        old_sample = next(
            (e.get("sample") for e in effects_list if isinstance(e, dict) and isinstance(e.get("sample"), dict)),
            None,
        )
        if old_sample is not None:
            study["sample"] = _migrate_old_sample(old_sample)
            report.add(
                f"{path}.sample",
                "migrated sample from effect[0] to study level",
                fixed=True,
                before=None,
                after=study["sample"],
            )

    _normalize_study_sample(study, path, report, repair)

    effects = study.get("effects")
    if not isinstance(effects, list):
        report.add(f"{path}.effects", "effects must be a list", severity="error", before=effects)
        return

    for effect_index, effect in enumerate(effects):
        effect_path = f"{path}.effects[{effect_index}]"
        if not isinstance(effect, dict):
            report.add(effect_path, "effect entry must be an object", severity="error", before=effect)
            continue
        # Remove obsolete effect-level sample during migration.
        if repair and isinstance(effect.get("sample"), dict):
            del effect["sample"]
            report.add(effect_path + ".sample", "removed effect-level sample (now at study level)", fixed=True)
        _normalize_effect(effect, effect_path, report, repair)


def _migrate_old_sample(old: dict[str, Any]) -> dict[str, Any]:
    """Convert old nested effect.sample → new flat study.sample."""
    new: dict[str, Any] = {
        "total_n": None, "analyzed_n": None,
        "mean_age": None, "female_percent": None, "male_percent": None,
        "platform": None, "country": None,
        "inclusion_criteria": None, "exclusion_criteria": None, "notes": None,
    }
    # N fields
    total = old.get("total_n")
    new["total_n"] = int(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else total
    n_excl = old.get("n_excluded")
    if isinstance(n_excl, (int, float)) and not isinstance(n_excl, bool) and isinstance(total, (int, float)):
        new["analyzed_n"] = int(total - n_excl)

    # Age
    age = old.get("age") or {}
    mean = age.get("mean") if isinstance(age, dict) else None
    if isinstance(mean, (int, float)) and not isinstance(mean, bool):
        new["mean_age"] = float(mean)

    # Gender — normalize to 0-100 percent
    gender = old.get("gender") or {}
    if isinstance(gender, dict):
        for old_key, new_key in (("female_pct", "female_percent"), ("male_pct", "male_percent")):
            val = gender.get(old_key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                # If stored as proportion (≤1.0) convert to percent
                pct = float(val) * 100.0 if float(val) <= 1.0 else float(val)
                new[new_key] = round(pct, 4)

    # Platform: prefer recruitment.source over recruitment.platform
    rec = old.get("recruitment") or {}
    if isinstance(rec, dict):
        platform = rec.get("source") or rec.get("platform")
        if isinstance(platform, str) and platform.strip():
            new["platform"] = platform.strip()

    # Exclusion criteria: list → single string (joined)
    excl = old.get("exclusion_criteria")
    if isinstance(excl, list):
        parts = [str(x).strip() for x in excl if x and str(x).strip()]
        new["exclusion_criteria"] = "; ".join(parts) if parts else None
    elif isinstance(excl, str) and excl.strip():
        new["exclusion_criteria"] = excl.strip()

    # notes from old "other" (only if not an LLM summary — no heuristic, just pass through)
    other = old.get("other")
    if isinstance(other, str) and other.strip():
        new["notes"] = other.strip()

    return new


def _extract_platform_canonical(value: str) -> str | None:
    """Try to extract a canonical SAMPLE_PLATFORM_VOCAB token from a freeform string.

    Examples:
      "we recruited subjects via the platform CloudResearch." → "CloudResearch"
      "Amazon MTurk"                                          → "MTurk"
      "Prolific Academic"                                     → "Prolific"
      "something completely unknown"                          → None

    Checks vocab tokens longest-first so e.g. "CloudResearch" beats "Online".
    """
    v_lower = value.lower()
    for token in sorted(SAMPLE_PLATFORM_VOCAB, key=len, reverse=True):
        if token.lower() in v_lower:
            return token
    return None


def _normalize_study_sample(study: dict[str, Any], path: str, report: SchemaValidationReport, repair: bool) -> None:
    """Validate and normalize the study-level sample object.

    The sample object is general enough for all social-science paper types:
    online panels (MTurk/Prolific), undergraduate pools, lab experiments,
    organizational/field studies, archival data, and multi-wave designs.
    Most fields are nullable — only fill what the paper actually reports.

    Forward-compatibility: unknown keys in the sample dict are preserved with
    a warning so future schema additions don't lose data.
    """
    sample = study.get("sample")
    sample_path = f"{path}.sample"

    if _is_null_string(sample):
        _replace(study, "sample", None, sample_path, report, repair, "empty sample normalized to null")
        return
    if sample is None or isinstance(sample, str):
        return
    if not isinstance(sample, dict):
        report.add(sample_path, "sample must be a dict, string, or null", severity="error", before=sample)
        return

    # Ensure all expected keys exist (nullable defaults)
    sample_defaults: dict[str, Any] = {
        "total_n": None,
        "analyzed_n": None,
        "mean_age": None,
        "female_percent": None,
        "male_percent": None,
        "platform": None,
        "country": None,
        "inclusion_criteria": None,
        "exclusion_criteria": None,
        "notes": None,
    }
    for key, default in sample_defaults.items():
        _ensure_key(sample, key, default, f"{sample_path}.{key}", report, repair)

    # Integer fields
    for key in ("total_n", "analyzed_n"):
        _normalize_int(sample, key, f"{sample_path}.{key}", report, repair, nullable=True)

    # Float fields (0-100 percent scale; null if not reported)
    for key in ("mean_age", "female_percent", "male_percent"):
        _normalize_number(sample, key, f"{sample_path}.{key}", report, repair, nullable=True)
        val = sample.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            # Normalize proportion (≤1.0) → percent for gender fields
            if key in ("female_percent", "male_percent") and float(val) <= 1.0:
                corrected = round(float(val) * 100.0, 4)
                _replace(
                    sample, key, corrected, f"{sample_path}.{key}", report, repair,
                    f"{key} normalized from proportion to percent: {val} → {corrected}"
                )

    # String fields (nullable)
    for key in ("platform", "country", "inclusion_criteria", "exclusion_criteria", "notes"):
        _normalize_nullable_string(sample, key, f"{sample_path}.{key}", report, repair)

    # Platform vocab enforcement: if value is set but not a canonical token, try auto-extraction.
    platform = sample.get("platform")
    if isinstance(platform, str) and platform not in SAMPLE_PLATFORM_VOCAB:
        canonical = _extract_platform_canonical(platform)
        if canonical is not None:
            _replace(
                sample, "platform", canonical, f"{sample_path}.platform", report, repair,
                f"platform extracted from sentence to controlled vocab: {platform!r} → {canonical!r}",
            )
        else:
            report.add(
                f"{sample_path}.platform",
                f"platform value {platform!r} is not in SAMPLE_PLATFORM_VOCAB "
                f"{sorted(SAMPLE_PLATFORM_VOCAB)}; cannot auto-canonicalize",
                severity="warning",
            )

    # Forward-compat: warn about unknown keys so they are visible in reports but not deleted.
    known_keys = set(sample_defaults.keys())
    for unknown_key in sorted(set(sample.keys()) - known_keys):
        report.add(
            f"{sample_path}.{unknown_key}",
            f"unknown sample field {unknown_key!r} preserved for forward-compatibility; "
            "consider using 'notes' for free-form text that does not fit the standard fields",
            severity="warning",
        )


def _normalize_effect(effect: dict[str, Any], path: str, report: SchemaValidationReport, repair: bool) -> None:
    defaults = {
        # "sample" intentionally absent — sample now lives at study level.
        "platform": None,
        "effecttype": None,
        "IV": None,
        "DV": None,
        "size": None,
        "direction": None,
        "mean_group1": None,
        "sd_group1": None,
        "mean_group2": None,
        "sd_group2": None,
        "stats": {},
        "materials_notes": None,
        "table_or_page_location": None,
        "materials": {"status": None, "content": None},
        "manipulation": {"status": None, "content": None},
        "items": {"status": None, "content": None},
    }
    for key in EFFECT_REQUIRED_FIELDS:
        _ensure_key(effect, key, deepcopy(defaults[key]), f"{path}.{key}", report, repair)

    for key in ("platform", "IV", "DV", "materials_notes", "table_or_page_location"):
        _normalize_nullable_string(effect, key, f"{path}.{key}", report, repair)
    _normalize_effecttype(effect, f"{path}.effecttype", report, repair)
    _normalize_direction(effect, f"{path}.direction", report, repair)
    _normalize_effect_sample_size(effect, f"{path}.size", report, repair)
    for key in EFFECT_NUMERIC_FIELDS:
        _normalize_number(effect, key, f"{path}.{key}", report, repair, nullable=True)

    stats = effect.get("stats")
    if not isinstance(stats, dict):
        if repair:
            report.add(f"{path}.stats", "stats must be an object; replaced with empty stats", fixed=True, before=stats, after={})
            stats = {}
            effect["stats"] = stats
        else:
            report.add(f"{path}.stats", "stats must be an object", severity="error", before=stats)
            stats = {}
    _normalize_stats(stats, f"{path}.stats", report, repair)

    for slot_name in SLOT_NAMES:
        _normalize_slot(effect, slot_name, f"{path}.{slot_name}", report, repair)

    # Optional effect-level analysis sample fields (not in EFFECT_REQUIRED_FIELDS — they are
    # supplementary). Only normalize when the key is present so existing JSONs are not polluted
    # with extra null fields unless explicitly written by the extractor.
    if "analysis_n" in effect:
        _normalize_int(effect, "analysis_n", f"{path}.analysis_n", report, repair, nullable=True)
    if "analysis_scope" in effect:
        _normalize_nullable_string(effect, "analysis_scope", f"{path}.analysis_scope", report, repair)
        scope_val = effect.get("analysis_scope")
        if isinstance(scope_val, str) and scope_val not in VALID_ANALYSIS_SCOPE:
            scope_norm = re.sub(r"[\s\-]+", "_", scope_val.lower().strip())
            if scope_norm in VALID_ANALYSIS_SCOPE:
                _replace(
                    effect, "analysis_scope", scope_norm, f"{path}.analysis_scope", report, repair,
                    f"analysis_scope normalized: {scope_val!r} → {scope_norm!r}",
                )
            else:
                report.add(
                    f"{path}.analysis_scope",
                    f"analysis_scope {scope_val!r} not in {sorted(VALID_ANALYSIS_SCOPE)}",
                    severity="warning",
                )


# _normalize_sample (old effect-level) removed — replaced by _normalize_study_sample above.


def _normalize_stats(stats: dict[str, Any], path: str, report: SchemaValidationReport, repair: bool) -> None:
    for key in STATS_REQUIRED_FIELDS:
        default = [None, None] if key == "ci" else None
        _ensure_key(stats, key, deepcopy(default), f"{path}.{key}", report, repair)
    for key in STATS_NUMERIC_FIELDS:
        _normalize_number(stats, key, f"{path}.{key}", report, repair, nullable=True)
    _normalize_ci_like(stats, "ci", f"{path}.ci", report, repair, label="ci")
    _normalize_p_value(stats, path, report, repair)
    _normalize_sig(stats, f"{path}.sig", report, repair)


def _normalize_slot(effect: dict[str, Any], key: str, path: str, report: SchemaValidationReport, repair: bool) -> None:
    slot = effect.get(key)
    if isinstance(slot, str):
        replacement = {"status": None, "content": None if _is_null_string(slot) else slot}
        _replace(effect, key, replacement, path, report, repair, "slot string converted to status/content object")
        slot = effect.get(key)
    elif slot is None:
        replacement = {"status": None, "content": None}
        _replace(effect, key, replacement, path, report, repair, "missing slot converted to status/content object")
        slot = effect.get(key)

    if not isinstance(slot, dict):
        report.add(path, "slot must be an object with status/content", severity="error", before=slot)
        return

    _ensure_key(slot, "status", None, f"{path}.status", report, repair)
    _ensure_key(slot, "content", None, f"{path}.content", report, repair)
    _normalize_slot_status(slot, f"{path}.status", report, repair)
    content = slot.get("content")
    if _is_null_string(content):
        _replace(slot, "content", None, f"{path}.content", report, repair, "empty slot content normalized to null")
    elif content is not None and not isinstance(content, str):
        _replace(slot, "content", str(content), f"{path}.content", report, repair, "slot content converted to string")

    if slot.get("status") == "verbatim" and slot.get("content") is None:
        report.add(f"{path}.content", "verbatim slot must have non-empty content", severity="error")


def _normalize_p_value(stats: dict[str, Any], path: str, report: SchemaValidationReport, repair: bool) -> None:
    key = "p_value"
    value = stats.get(key)
    field_path = f"{path}.p_value"
    if _is_null_string(value):
        _replace(stats, key, None, field_path, report, repair, "empty p_value normalized to null")
    elif value is None:
        return
    elif not isinstance(value, str):
        _replace(stats, key, str(value), field_path, report, repair, "p_value must be stored as a string")


def _normalize_sig(stats: dict[str, Any], path: str, report: SchemaValidationReport, repair: bool) -> None:
    mapping = {
        "significant": "sig",
        "statistically significant": "sig",
        "non significant": "ns",
        "non-significant": "ns",
        "nonsig": "ns",
        "non_sig": "ns",
        "not significant": "ns",
        "not_significant": "ns",
        "insignificant": "ns",
        "marginally significant": "marginal",
        "marginal significant": "marginal",
        "unknown": None,
        "mixed": None,
        "not tested": None,
        "not reported": None,
    }
    _normalize_enum(stats, "sig", VALID_SIG, mapping, path, report, repair, nullable=True)


def _normalize_direction(effect: dict[str, Any], path: str, report: SchemaValidationReport, repair: bool) -> None:
    mapping = {
        "positive": "pos",
        "marginal positive": "pos",
        "marginal pos": "pos",
        "positive marginal": "pos",
        "+": "pos",
        "negative": "neg",
        "marginal negative": "neg",
        "marginal neg": "neg",
        "negative marginal": "neg",
        "-": "neg",
        "none": None,
        "no effect": None,
        "no_effect": None,
        "null": None,
        "zero": None,
        "equal": None,
        "same": None,
        "multi group": None,
        "multi-group": None,
        "mixed": None,
        "complex": None,
    }
    _normalize_enum(effect, "direction", VALID_DIRECTION, mapping, path, report, repair, nullable=True)


def _normalize_effecttype(effect: dict[str, Any], path: str, report: SchemaValidationReport, repair: bool) -> None:
    mapping = {
        "interaction": "int",
        "interactions": "int",
        "moderation": "int",
        "main effect": "main",
        "main_effect": "main",
        "simple effect": "simple",
        "simple_effect": "simple",
        "pearson": "correlation",
        "corr": "correlation",
        "r": "correlation",
    }
    _normalize_enum(effect, "effecttype", VALID_EFFECT_TYPES, mapping, path, report, repair, nullable=False)


def _normalize_slot_status(slot: dict[str, Any], path: str, report: SchemaValidationReport, repair: bool) -> None:
    mapping = {
        "quoted": "verbatim",
        "quote": "verbatim",
        "paraphrase": "paraphrased",
        "paraphrase_summary": "paraphrased",
        "scale": "cited_scale",
        "cited scale": "cited_scale",
        "osf": "osf_only",
        "supplement": "osf_only",
        "supplementary": "osf_only",
        "missing": "not_in_paper",
        "not found": "not_in_paper",
    }
    _normalize_enum(slot, "status", VALID_SLOT_STATUSES, mapping, path, report, repair, nullable=True)


def _normalize_enum(
    obj: dict[str, Any],
    key: str,
    valid: set[str],
    mapping: dict[str, str | None],
    path: str,
    report: SchemaValidationReport,
    repair: bool,
    *,
    nullable: bool,
) -> None:
    value = obj.get(key)
    if _is_null_string(value):
        if nullable:
            _replace(obj, key, None, path, report, repair, f"{key} empty value normalized to null")
            return
        report.add(path, f"{key} must be one of {sorted(valid)}", severity="error", before=value)
        return
    if value is None:
        if nullable:
            return
        report.add(path, f"{key} must be one of {sorted(valid)}", severity="error", before=value)
        return
    if not isinstance(value, str):
        value_text = str(value)
    else:
        value_text = value
    normalized_key = _enum_key(value_text)
    if normalized_key in mapping:
        _replace(obj, key, mapping[normalized_key], path, report, repair, f"{key} enum normalized")
        return
    if value_text in valid:
        return
    report.add(path, f"{key} must be one of {sorted(valid)}", severity="error", before=value)


def _normalize_int(
    obj: dict[str, Any],
    key: str,
    path: str,
    report: SchemaValidationReport,
    repair: bool,
    *,
    nullable: bool,
) -> None:
    value = obj.get(key)
    if _is_null_string(value) or value is None:
        if nullable:
            _replace(obj, key, None, path, report, repair, f"{key} empty value normalized to null")
            return
        report.add(path, f"{key} must be an integer", severity="error", before=value)
        return
    if isinstance(value, bool):
        report.add(path, f"{key} must be an integer, not boolean", severity="error", before=value)
        return
    if isinstance(value, int):
        return
    if isinstance(value, float) and value.is_integer():
        _replace(obj, key, int(value), path, report, repair, f"{key} converted to integer")
        return
    parsed = _parse_number(value)
    if parsed is not None and parsed.is_integer():
        _replace(obj, key, int(parsed), path, report, repair, f"{key} converted to integer")
        return
    report.add(path, f"{key} must be an integer or null", severity="error", before=value)


def _normalize_effect_sample_size(
    effect: dict[str, Any],
    path: str,
    report: SchemaValidationReport,
    repair: bool,
) -> None:
    """Normalize effect.size, whose schema meaning is analysis/sample N."""
    value = effect.get("size")
    if _is_null_string(value) or value is None:
        _replace(effect, "size", None, path, report, repair, "size empty value normalized to null")
        return
    if isinstance(value, bool):
        report.add(path, "size must be an integer, not boolean", severity="error", before=value)
        return
    if isinstance(value, int):
        return
    if isinstance(value, float) and value.is_integer():
        _replace(effect, "size", int(value), path, report, repair, "size converted to integer")
        return
    if isinstance(value, float) and repair:
        _preserve_noninteger_size(effect, value)
        _replace(
            effect,
            "size",
            None,
            path,
            report,
            repair,
            "non-integer size looked like an effect-size statistic; preserved in effect_size_notes and normalized to null",
        )
        return
    parsed = _parse_number(value)
    if parsed is not None and parsed.is_integer():
        _replace(effect, "size", int(parsed), path, report, repair, "size converted to integer")
        return
    if parsed is not None and repair:
        _preserve_noninteger_size(effect, value)
        _replace(
            effect,
            "size",
            None,
            path,
            report,
            repair,
            "non-integer size looked like an effect-size statistic; preserved in effect_size_notes and normalized to null",
        )
        return
    report.add(path, "size must be an integer or null", severity="error", before=value)


def _preserve_noninteger_size(effect: dict[str, Any], value: Any) -> None:
    note = (
        f"Non-integer size field value preserved from extraction: {value!r}. "
        "In this schema, size is sample/analysis N; effect-size statistics belong "
        "in stats or reported_statistics_text."
    )
    existing = effect.get("effect_size_notes")
    if isinstance(existing, str) and existing.strip():
        if note not in existing:
            effect["effect_size_notes"] = f"{existing.rstrip()} {note}"
    else:
        effect["effect_size_notes"] = note


def _normalize_number(
    obj: dict[str, Any],
    key: str,
    path: str,
    report: SchemaValidationReport,
    repair: bool,
    *,
    nullable: bool,
) -> None:
    value = obj.get(key)
    if _is_null_string(value) or value is None:
        if nullable:
            _replace(obj, key, None, path, report, repair, f"{key} empty value normalized to null")
            return
        report.add(path, f"{key} must be numeric", severity="error", before=value)
        return
    if isinstance(value, bool):
        report.add(path, f"{key} must be numeric, not boolean", severity="error", before=value)
        return
    if isinstance(value, (int, float)):
        return
    parsed = _parse_number(value)
    if parsed is not None:
        final: int | float = int(parsed) if parsed.is_integer() else parsed
        _replace(obj, key, final, path, report, repair, f"{key} converted to number")
        return
    report.add(path, f"{key} must be numeric or null", severity="error", before=value)


def _normalize_ci_like(
    obj: dict[str, Any],
    key: str,
    path: str,
    report: SchemaValidationReport,
    repair: bool,
    *,
    label: str,
) -> None:
    value = obj.get(key)
    if _is_null_string(value) or value is None:
        _replace(obj, key, [None, None], path, report, repair, f"{label} normalized to [null, null]")
        value = obj.get(key)

    if not isinstance(value, list):
        parsed = _parse_number_list(value)
        if parsed is not None:
            _replace(obj, key, parsed, path, report, repair, f"{label} parsed into two-element array")
            value = obj.get(key)
        else:
            _replace(obj, key, [None, None], path, report, repair, f"{label} must be a two-element array")
            value = obj.get(key)

    if not isinstance(value, list):
        report.add(path, f"{label} must be a two-element array", severity="error", before=value)
        return

    if len(value) != 2:
        replacement = (value + [None, None])[:2]
        _replace(obj, key, replacement, path, report, repair, f"{label} resized to two elements")
        value = obj.get(key)

    if isinstance(value, list) and len(value) == 2:
        for index in range(2):
            wrapper = {"value": value[index]}
            _normalize_number(wrapper, "value", f"{path}[{index}]", report, repair, nullable=True)
            if repair:
                value[index] = wrapper["value"]


def _normalize_nullable_string(
    obj: dict[str, Any],
    key: str,
    path: str,
    report: SchemaValidationReport,
    repair: bool,
) -> None:
    value = obj.get(key)
    if _is_null_string(value):
        _replace(obj, key, None, path, report, repair, f"{key} empty value normalized to null")
    elif value is not None and not isinstance(value, str):
        _replace(obj, key, str(value), path, report, repair, f"{key} converted to string")


def _ensure_key(
    obj: dict[str, Any],
    key: str,
    default: Any,
    path: str,
    report: SchemaValidationReport,
    repair: bool,
) -> None:
    if key in obj:
        return
    if repair:
        obj[key] = deepcopy(default)
        report.add(path, "missing required key added", fixed=True, before="<missing>", after=default)
    else:
        report.add(path, "missing required key", severity="error")


def _replace(
    obj: dict[str, Any],
    key: str,
    value: Any,
    path: str,
    report: SchemaValidationReport,
    repair: bool,
    message: str,
) -> None:
    before = obj.get(key)
    if before == value:
        return
    if repair:
        obj[key] = value
        report.add(path, message, fixed=True, before=before, after=value)
    else:
        report.add(path, message, severity="error", before=before, after=value)


def _parse_number(value: Any) -> float | None:
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if _is_null_string(text):
            return None
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
            try:
                return float(text)
            except ValueError:
                return None
        if re.match(r"^\s*bs?\s*=", text, flags=re.I):
            return None
        statistic_match = re.search(r"\)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", text)
        if statistic_match:
            return float(statistic_match.group(1))
        leading_match = re.match(r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\b", text)
        if leading_match:
            return float(leading_match.group(1))
        equals_matches = re.findall(r"=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", text)
        if len(equals_matches) == 1:
            return float(equals_matches[0])
    return None


def _parse_number_list(value: Any) -> list[Any] | None:
    if not isinstance(value, str):
        return None
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value.replace(",", ""))
    if len(numbers) >= 2:
        parsed = [float(numbers[0]), float(numbers[1])]
        return [int(x) if x.is_integer() else x for x in parsed]
    return None


def _is_null_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in NULL_STRINGS


def _enum_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("_", " "))


def _safe_value(value: Any) -> Any:
    if value is _MISSING:
        return _MISSING
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)


def _looks_like_paper_json(data: Any) -> bool:
    return isinstance(data, dict) and "eligible_studies" in data


def _write_json_atomic(path: Path, data: dict[str, Any], *, backup: bool) -> None:
    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _paths_from_args(paths: Iterable[str]) -> list[Path]:
    return [Path(item) for item in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and normalize ai-ethics paper JSON schema")
    parser.add_argument("json", nargs="*", help="One or more per-paper JSON files")
    parser.add_argument("--stage4-dir", type=Path, help="Scan a stage4 tree instead of explicit JSON files")
    parser.add_argument("--write", action="store_true", help="Write deterministic repairs back to JSON files")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak files when --write is used")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("stage4/schema_normalization_report.json"),
        help="Report path for --stage4-dir scans",
    )
    args = parser.parse_args()

    if args.stage4_dir:
        aggregate = normalize_stage4_tree(
            args.stage4_dir,
            write=args.write,
            backup=not args.no_backup,
            report_path=args.report,
        )
        print(json.dumps(aggregate["summary"], indent=2, ensure_ascii=False))
        print(f"Report: {args.report}")
        raise SystemExit(1 if aggregate["summary"]["invalid"] else 0)

    if not args.json:
        parser.error("pass JSON files or --stage4-dir")

    any_invalid = False
    for json_path in _paths_from_args(args.json):
        _, report = validate_file(json_path, repair=True, write=args.write, backup=not args.no_backup)
        print(f"{json_path}: {summarize_report(report)}")
        any_invalid = any_invalid or not report.valid
    raise SystemExit(1 if any_invalid else 0)


if __name__ == "__main__":
    main()
