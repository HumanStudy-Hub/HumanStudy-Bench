"""Turn a study evaluator's output into one comparison table the web app can plot.

Study evaluators do not agree on a single shape. Most of them enrich their test
results through `stats_lib.add_statistical_replication_fields` and already carry
human and agent effect sizes; a few return raw t-tests. This module normalises
both into the same rows: one row per statistical test, with the human result and
the agent result side by side.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

Number = Optional[float]


def _number(value: Any) -> Number:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) or math.isinf(result) else result


def _flag(value: Any) -> Optional[bool]:
    return bool(value) if isinstance(value, bool) else None


def _first(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def human_effect_from_reported(reported: str) -> Number:
    """Convert a reported test statistic into a Cohen's d equivalent.

    Papers report the statistic, not the effect size, so the published
    `F(1, df)`, `t(df)`, or `r(df)` is converted with the standard identities.
    Anything else (chi-square, multi-numerator F) has no single-d equivalent and
    is left out of the effect-size comparison.
    """
    if not reported:
        return None
    text = str(reported)
    t_match = re.search(r"\bt\s*\(\s*(\d+(?:\.\d+)?)\s*\)\s*=\s*(-?\d+(?:\.\d+)?)", text)
    if t_match:
        df, statistic = float(t_match.group(1)), float(t_match.group(2))
        return (2 * statistic / math.sqrt(df)) if df > 0 else None
    f_match = re.search(r"\bF\s*\(\s*(\d+)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*=\s*(-?\d+(?:\.\d+)?)", text)
    if f_match:
        df1, df2, statistic = int(f_match.group(1)), float(f_match.group(2)), float(f_match.group(3))
        if df1 == 1 and df2 > 0 and statistic >= 0:
            return 2 * math.sqrt(statistic / df2)
        return None
    r_match = re.search(r"\br\s*\(\s*(\d+)\s*\)\s*=\s*(-?\d+(?:\.\d+)?)", text)
    if r_match:
        correlation = float(r_match.group(2))
        denominator = 1 - correlation ** 2
        return (2 * correlation / math.sqrt(denominator)) if denominator > 0 else None
    return None


def agent_effect_from_t(t_statistic: Number, n1: Number, n2: Number) -> Number:
    """Cohen's d for an independent-samples t-test."""
    if t_statistic is None or not n1:
        return None
    if n2:
        return t_statistic * math.sqrt(1 / n1 + 1 / n2)
    return t_statistic / math.sqrt(n1)


def _ground_truth_index(study_path: Path) -> Dict[tuple, Dict[str, Any]]:
    """Map (study label, finding id) to the paper's reported statistics."""
    source = study_path / "source"
    path = (source if source.is_dir() else study_path) / "ground_truth.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            ground_truth = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    index: Dict[tuple, Dict[str, Any]] = {}
    for sub_study in ground_truth.get("studies", []) or []:
        label = sub_study.get("study_id")
        for finding in sub_study.get("findings", []) or []:
            tests = finding.get("statistical_tests") or [{}]
            entry = {
                "hypothesis": finding.get("main_hypothesis"),
                "reported_statistics": tests[0].get("reported_statistics"),
                "test_name": tests[0].get("test_name"),
                "expected_direction": tests[0].get("expected_direction"),
            }
            index[(label, finding.get("finding_id"))] = entry
            index[(None, finding.get("finding_id"))] = entry
    return index


def _label(row: Dict[str, Any], fallback: str) -> str:
    parts = [str(row[key]) for key in ("study_id", "finding_id", "scenario") if row.get(key)]
    if not parts:
        parts = [str(row.get("test_name") or fallback)]
    return " · ".join(parts).replace("_", " ")


def normalise_tests(evaluation: Dict[str, Any], study_path: Path) -> List[Dict[str, Any]]:
    raw_tests = evaluation.get("test_results") or []
    ground_truth = _ground_truth_index(study_path)
    rows: List[Dict[str, Any]] = []

    for index, raw in enumerate(raw_tests):
        if not isinstance(raw, dict):
            continue
        gt = ground_truth.get((raw.get("study_id"), raw.get("finding_id"))) or ground_truth.get((None, raw.get("finding_id"))) or {}

        agent_p = _number(_first(raw, "p_value_agent", "p_value", "agent_p_value"))
        human_p = _number(_first(raw, "p_value_human", "human_p_value"))
        agent_effect = _number(_first(raw, "agent_effect_d", "agent_effect_size"))
        human_effect = _number(_first(raw, "human_effect_d", "human_effect_size"))
        n_agent_1 = _number(_first(raw, "n_agent_1", "n_agent", "n_agent_extracted"))
        n_agent_2 = _number(_first(raw, "n_agent_2", "n2_agent", "n2_agent_extracted"))

        if agent_effect is None:
            agent_effect = agent_effect_from_t(_number(_first(raw, "t_stat", "test_statistic", "statistic")), n_agent_1, n_agent_2)
        if human_effect is None:
            human_effect = human_effect_from_reported(gt.get("reported_statistics") or raw.get("reported_statistics") or "")

        agent_significant = _flag(_first(raw, "is_significant_agent", "significant", "agent_significant"))
        human_significant = _flag(_first(raw, "is_significant_human", "human_significant"))
        direction_match = _flag(raw.get("direction_match"))
        replicated = _flag(raw.get("replication"))
        if replicated is None and None not in (human_significant, agent_significant, direction_match):
            replicated = bool(human_significant and agent_significant and direction_match)

        rows.append({
            "test_id": f"T{index + 1}",
            "study_label": raw.get("study_id"),
            "sub_study_id": raw.get("sub_study_id"),
            "finding_id": raw.get("finding_id"),
            "scenario": raw.get("scenario"),
            "label": _label(raw, f"Test {index + 1}"),
            "test_type": _first(raw, "statistical_test_type", "test_type", "test"),
            "hypothesis": gt.get("hypothesis"),
            "reported_statistics": gt.get("reported_statistics"),
            "human_effect": human_effect,
            "agent_effect": agent_effect,
            "human_p": human_p,
            "agent_p": agent_p,
            "human_significant": human_significant,
            "agent_significant": agent_significant,
            "direction_match": direction_match,
            "replicated": replicated,
            "agent_mean_1": _number(_first(raw, "mean_agent_1", "agent_mean_1")),
            "agent_mean_2": _number(_first(raw, "mean_agent_2", "agent_mean_2")),
            "n_agent_1": int(n_agent_1) if n_agent_1 else None,
            "n_agent_2": int(n_agent_2) if n_agent_2 else None,
            "pas": _number(_first(raw, "pas", "score")),
        })
    return rows


def summarise(rows: List[Dict[str, Any]], evaluation: Dict[str, Any]) -> Dict[str, Any]:
    scored = [row for row in rows if row["replicated"] is not None]
    replicated = [row for row in scored if row["replicated"]]
    comparable = [row for row in rows if row["human_effect"] is not None and row["agent_effect"] is not None]
    directional = [row for row in rows if row["direction_match"] is not None]

    gaps = [abs(row["agent_effect"] - row["human_effect"]) for row in comparable]
    human_effects = [row["human_effect"] for row in comparable]
    agent_effects = [row["agent_effect"] for row in comparable]

    return {
        "totalTests": len(rows),
        "scoredTests": len(scored),
        "replicatedTests": len(replicated),
        "replicationRate": (len(replicated) / len(scored)) if scored else None,
        "directionMatchRate": (sum(1 for row in directional if row["direction_match"]) / len(directional)) if directional else None,
        "meanAbsoluteEffectGap": (sum(gaps) / len(gaps)) if gaps else None,
        "meanHumanEffect": (sum(human_effects) / len(human_effects)) if human_effects else None,
        "meanAgentEffect": (sum(agent_effects) / len(agent_effects)) if agent_effects else None,
        "effectCorrelation": _correlation(human_effects, agent_effects),
        "studyScore": _number(evaluation.get("score")),
    }


def _correlation(x: List[float], y: List[float]) -> Number:
    if len(x) < 3:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    variance_x = sum((a - mean_x) ** 2 for a in x)
    variance_y = sum((b - mean_y) ** 2 for b in y)
    if variance_x <= 0 or variance_y <= 0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def build_analysis(evaluation: Dict[str, Any], study_path: Path, run: Dict[str, Any], responses: Dict[str, Any]) -> Dict[str, Any]:
    rows = normalise_tests(evaluation, study_path)
    return {
        "runId": run.get("id"),
        "studyId": run.get("studyId"),
        "studyTitle": run.get("studyTitle"),
        "model": run.get("model"),
        "promptPreset": run.get("preset"),
        "participants": responses.get("participants"),
        "trials": responses.get("trials"),
        "completedTrials": responses.get("completedTrials"),
        "summary": summarise(rows, evaluation),
        "tests": rows,
        "findings": evaluation.get("finding_results") or [],
    }
