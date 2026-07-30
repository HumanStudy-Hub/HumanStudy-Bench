"""Stage 1 inventory and simulation-eligibility filter."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generation_pipeline.filters.base_filter import BaseFilter
from generation_pipeline.stage1_compiler import (
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
    build_stage1_task_prompt,
    compile_stage1_inventory,
)
from generation_pipeline.stage1_study_contract import slugify


class ReplicabilityFilter(BaseFilter):
    """Inventory empirical studies and classify HumanStudy-Bench eligibility."""

    def process(
        self,
        pdf_path: Path,
        regeneration_instructions: Optional[Dict[str, Any]] = None,
        *,
        pdf_artifacts_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        workers: int = DEFAULT_WORKERS,
        force: bool = False,
    ) -> Dict[str, Any]:
        result = compile_stage1_inventory(
            pdf_path,
            self.client,
            regeneration_instructions=regeneration_instructions,
            pdf_artifacts_dir=pdf_artifacts_dir,
            artifacts_dir=artifacts_dir,
            timeout=timeout,
            workers=workers,
            force=force,
        )
        _normalize_experiments(result)
        result["paper_id"] = pdf_path.stem.replace(" ", "_").replace("-", "_").lower()
        return result

    def _build_prompt(
        self,
        pdf_name: str,
        num_pages: int,
        regeneration_instructions: Optional[Dict[str, Any]] = None,
    ) -> str:
        return build_stage1_task_prompt(
            pdf_name,
            num_pages,
            regeneration_instructions,
        )


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
        exp["replicable"] = str(exp.get("replicable") or "UNCERTAIN").strip().upper()
        reasons = exp.get("exclusion_reasons")
        if isinstance(reasons, str):
            exp["exclusion_reasons"] = [reasons.strip()] if reasons.strip() else []
        elif isinstance(reasons, list):
            exp["exclusion_reasons"] = [
                str(reason).strip() for reason in reasons if str(reason).strip()
            ]
        else:
            exp["exclusion_reasons"] = []
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

    result["overall_replicable"] = any(
        isinstance(exp, dict) and exp.get("replicable") in {"YES", "UNCERTAIN"}
        for exp in experiments
    )


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
