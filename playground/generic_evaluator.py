"""Generic evaluator for agent-built buffer studies.

The Build Study agent writes JSON only; this module is the generic scoring
runtime that the playground falls back to when a study ships no
`scripts/evaluator.py`. It is driven entirely by `source/ground_truth.json`:

- each finding's `statistical_tests[0]` carries the paper's reported statistic
  and expected direction;
- each finding's `response_mapping` names the material items whose answers are
  the outcome and, optionally, the item whose A/B answer splits participants
  into groups.

The generic runtime (`GenericGeneratedStudyConfig`) numbers questions Q1..Qn in
the order a material's `items` list is written, so the same item -> question
mapping is rebuilt here from the material files.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def _source_dir(study_path: Path) -> Path:
    source = study_path / "source"
    return source if (source / "specification.json").exists() else study_path


def _material_items(study_path: Path, sub_study_id: str) -> List[Dict[str, Any]]:
    source = _source_dir(study_path)
    material = _load_json(source / "materials" / f"{sub_study_id}.json")
    items = material.get("items") if isinstance(material, dict) else None
    return items if isinstance(items, list) else []


def _parse_q(responses_text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not responses_text:
        return parsed
    for match in re.finditer(r"Q(\d+)\s*[:=]\s*([^,\s\n]+)", responses_text):
        parsed[f"Q{match.group(1)}"] = match.group(2).strip()
    return parsed


def _q_number(item_index: int) -> str:
    # GeneratedStudyPromptBuilder numbers questions by 1-based list position.
    return f"Q{item_index + 1}"


def _numeric(value: str) -> Optional[float]:
    text = str(value or "").strip()
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _choice(value: str) -> Optional[str]:
    text = str(value or "").strip().upper()
    if not text:
        return None
    for letter in ("A", "B", "C", "D"):
        if letter in text:
            return letter
    return None


def _direction_to_int(direction: str) -> int:
    text = str(direction or "").lower()
    if text in ("positive", "greater", ">", "higher"):
        return 1
    if text in ("negative", "less", "<", "lower"):
        return -1
    return 0


def _human_significance(reported_statistics: str) -> Optional[bool]:
    text = str(reported_statistics or "")
    if not text:
        return None
    if re.search(r"p\s*[<=]\s*\.?0*\.?0?0?[0-4]\d", text):
        # p <= .04x is significant at .05 (covers .001, .01, .02, .03, .04, .05 forms)
        pass
    if re.search(r"p\s*[<=]\s*0?\.0?5\b|p\s*[<=]\s*\.05", text):
        return True
    if re.search(r"p\s*<\s*\.0?[1-4]\d*", text):
        return True
    if re.search(r"p\s*[<=]\s*0?\.00\d", text):
        return True
    if re.search(r"n\.?s\.?", text, re.IGNORECASE):
        return False
    return None


def _human_p_value(reported_statistics: str) -> Optional[float]:
    match = re.search(r"p\s*=\s*\.?(\d+)", str(reported_statistics or ""))
    if match:
        digits = match.group(1)
        value = float(f"0.{digits}") if not digits.startswith("0") else float(digits) / (10 ** (len(digits) - 1))
        return value
    return None


def _item_question_map(items: List[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """Map gt_key -> (Q number, question) using 1-based list position."""
    mapping: Dict[str, Tuple[str, str]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        gt_key = metadata.get("gt_key")
        question = item.get("question") or item.get("text") or item.get("label")
        if gt_key and question:
            mapping[str(gt_key)] = (_q_number(index), str(question))
    return mapping


def _condition_level(trial_info: Dict[str, Any], factor: str) -> Optional[str]:
    assignment = trial_info.get("condition_assignment") if isinstance(trial_info.get("condition_assignment"), dict) else {}
    entry = assignment.get(factor)
    if isinstance(entry, dict):
        return entry.get("level")
    return entry


def _t_test(group_a: List[float], group_b: List[float], expected: int) -> Dict[str, Any]:
    if len(group_a) < 2 or len(group_b) < 2:
        return {"n_agent_1": len(group_a), "n_agent_2": len(group_b)}
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    except Exception:
        return {"n_agent_1": len(group_a), "n_agent_2": len(group_b)}
    if t_stat is None or p_value is None or (isinstance(t_stat, float) and math.isnan(t_stat)):
        return {"n_agent_1": len(group_a), "n_agent_2": len(group_b)}
    mean_a = mean(group_a)
    mean_b = mean(group_b)
    direction = 1 if t_stat > 0 else -1 if t_stat < 0 else 0
    return {
        "n_agent_1": len(group_a),
        "n_agent_2": len(group_b),
        "mean_agent_1": mean_a,
        "mean_agent_2": mean_b,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": float(p_value) < 0.05,
        "direction_match": (expected == 0) or (direction == expected),
    }


def _score_finding(
    study_path: Path,
    study_label: Any,
    finding: Dict[str, Any],
    responses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    mapping = finding.get("response_mapping") if isinstance(finding.get("response_mapping"), dict) else {}
    sub_id = mapping.get("sub_study_id")
    tests = finding.get("statistical_tests") if isinstance(finding.get("statistical_tests"), list) else []
    test_gt = tests[0] if tests else {}
    reported = test_gt.get("reported_statistics", "")
    expected = _direction_to_int(test_gt.get("expected_direction", "none"))
    finding_id = finding.get("finding_id")

    row: Dict[str, Any] = {
        "study_id": study_label,
        "sub_study_id": sub_id,
        "finding_id": finding_id,
        "scenario": mapping.get("scenario") or sub_id,
        "human_p_value": _human_p_value(reported),
        "human_significant": _human_significance(reported),
    }
    if not sub_id:
        return row

    items = _material_items(study_path, str(sub_id))
    item_q = _item_question_map(items)

    measure_keys = [str(key) for key in (mapping.get("measure_gt_keys") or [])]
    group_by = mapping.get("group_by") or "none"
    group_key = str(mapping.get("group_gt_key") or "")
    condition_factor = mapping.get("condition_factor") or ""

    group_a: List[float] = []
    group_b: List[float] = []

    for participant in responses:
        for response in participant.get("responses") or []:
            if not isinstance(response, dict):
                continue
            trial_info = response.get("trial_info") if isinstance(response.get("trial_info"), dict) else {}
            if str(trial_info.get("sub_study_id") or "") != str(sub_id) and not (sub_id and sub_id in str(trial_info.get("material_id") or "")):
                continue
            parsed = _parse_q(response.get("response_text") or "")

            values: List[float] = []
            for key in measure_keys:
                q_number = item_q.get(key, (None, None))[0]
                if not q_number:
                    continue
                value = _numeric(parsed.get(q_number, ""))
                if value is not None:
                    values.append(value)
            if not values:
                continue
            outcome = mean(values)

            if group_by == "choice" and group_key:
                q_number = item_q.get(group_key, (None, None))[0]
                group = _choice(parsed.get(q_number, "")) if q_number else None
                if group == "A":
                    group_a.append(outcome)
                elif group == "B":
                    group_b.append(outcome)
                else:
                    continue
            elif group_by == "condition" and condition_factor:
                level = _condition_level(trial_info, condition_factor)
                if level is None:
                    continue
                levels = []
                if isinstance(level, list):
                    levels = [str(item) for item in level]
                else:
                    levels = [str(level)]
                text = " ".join(levels)
                if "A" in text or str(level) in ("0", "level1", "level_1") or str(level).lower() in ("a",):
                    group_a.append(outcome)
                else:
                    group_b.append(outcome)
            else:
                group_a.append(outcome)

    if group_b:
        row.update(_t_test(group_a, group_b, expected))
    else:
        # Single-group: report the observed mean without a significance test.
        row.update({
            "n_agent_1": len(group_a),
            "mean_agent_1": mean(group_a) if group_a else None,
        })

    return row


def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    study_path = Path(results.get("study_path") or ".")
    ground_truth = _load_json(_source_dir(study_path) / "ground_truth.json")
    responses = results.get("individual_data") if isinstance(results.get("individual_data"), list) else []

    test_results: List[Dict[str, Any]] = []
    for study in ground_truth.get("studies") or []:
        if not isinstance(study, dict):
            continue
        for finding in study.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            test_results.append(_score_finding(study_path, study.get("study_id"), finding, responses))

    return {"test_results": test_results}
