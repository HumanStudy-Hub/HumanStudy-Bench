import json
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from stats_lib import parse_p_value_from_reported
from study_utils import compute_construct_scores, iter_response_records

# Module-level cache for ground truth and metadata
_ground_truth_cache = None
_metadata_cache = None


def _expected_direction_to_int(expected_dir_str: str) -> int:
    """Convert expected_direction string to int: 1 = positive, -1 = negative, 0 = unknown."""
    if not expected_dir_str:
        return 0
    s = str(expected_dir_str).lower()
    if s in ("positive", "greater", ">"):
        return 1
    if s in ("negative", "less", "<"):
        return -1
    return 0


def _select_complete_cases(
    participant_scores: Sequence[Dict[str, Any]],
    outcome: str,
    predictors: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Build complete-case outcome and predictor matrices for OLS."""
    rows: List[Dict[str, Any]] = []
    for participant in participant_scores:
        required_values = [participant.get(outcome)]
        required_values.extend(participant.get(name) for name in predictors)
        if any(value is None for value in required_values):
            continue
        rows.append(participant)

    if not rows:
        return None

    y = np.array([row[outcome] for row in rows], dtype=float)
    x = np.array([[row[name] for name in predictors] for row in rows], dtype=float)
    return {"rows": rows, "y": y, "x": x}


def _fit_ols(y: np.ndarray, x: np.ndarray, predictor_names: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Fit an ordinary least squares model and return coefficients and t-tests."""
    if y.ndim != 1 or x.ndim != 2:
        return None

    n_obs, n_predictors = x.shape
    if n_obs <= n_predictors + 1:
        return None

    design = np.column_stack([np.ones(n_obs), x])
    rank = np.linalg.matrix_rank(design)
    if rank < design.shape[1]:
        return None

    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    dof = n_obs - design.shape[1]
    if dof <= 0:
        return None

    mse = float(np.sum(residuals ** 2) / dof)
    covariance = mse * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = coefficients / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), dof)

    names = ["intercept", *predictor_names]
    coefficient_map = {}
    for index, name in enumerate(names):
        coefficient_map[name] = {
            "coefficient": float(coefficients[index]),
            "standard_error": float(standard_errors[index]),
            "t_value": float(t_values[index]),
            "p_value": float(p_values[index]),
        }

    return {
        "n_obs": n_obs,
        "degrees_of_freedom": dof,
        "coefficients": coefficient_map,
    }


def evaluate_study(results):
    """
    Evaluates the agent's performance on Study 013 (Keh, Foo & Lim 2002).
    Computes construct scores from agent responses and tests hypothesized relationships.
    Returns test_results with raw stats; no BF/PAS aggregation.
    """
    global _ground_truth_cache, _metadata_cache

    # 1. Load Ground Truth and Metadata (with caching)
    study_dir = Path(__file__).resolve().parent.parent / "source"

    if _ground_truth_cache is None:
        with open(study_dir / "ground_truth.json", "r") as f:
            _ground_truth_cache = json.load(f)

    if _metadata_cache is None:
        metadata_path = study_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                _metadata_cache = json.load(f)
        else:
            _metadata_cache = {}

    ground_truth = _ground_truth_cache
    # 2. Parse all agent responses into participant score vectors
    participant_scores = []

    for response_record in iter_response_records(results):
        scores = compute_construct_scores(
            response_record.get("response_text", ""),
            response_record.get("trial_info", {}),
        )
        if scores is not None:
            participant_scores.append(scores)

    # 3. Build test results for each finding
    test_results = []

    if not participant_scores:
        # Return empty results if no valid participants
        for study_gt in ground_truth.get("studies", []):
            for finding in study_gt.get("findings", []):
                test_results.append({
                    "study_id": "study_013",
                    "sub_study_id": "keh_foo_lim_opportunity_evaluation",
                    "finding_id": finding["finding_id"],
                    "n_agent": 0,
                    "error": "No valid participant data",
                })
        return {"test_results": test_results}

    overconfidence_scores = np.array([p["overconfidence"] for p in participant_scores], dtype=float)
    n_agent = len(participant_scores)

    for study_gt in ground_truth.get("studies", []):
        for finding in study_gt.get("findings", []):
            finding_id = finding["finding_id"]
            stat_tests = finding.get("statistical_tests", [])
            test_gt = stat_tests[0] if stat_tests else {}
            expected_dir_str = test_gt.get("expected_direction", "")
            h_expected = _expected_direction_to_int(expected_dir_str)
            reported_stats = test_gt.get("reported_statistics", "")
            sig_level = test_gt.get("significance_level") or 0.05

            # Parse human p-value
            human_p_value = None
            human_significant = None
            parsed_p, parsed_sig, _ = parse_p_value_from_reported(reported_stats, sig_level)
            if parsed_p is not None:
                human_p_value = parsed_p
                human_significant = parsed_sig

            t_stat = None
            r_stat = None
            p_value = None
            agent_significant = None
            direction_match = None
            mean_agent = None
            sd_agent = None
            coefficient_agent = None
            standard_error_agent = None
            model_n = n_agent
            model_predictors = None
            human_coefficient = test_gt.get("reported_coefficient")
            human_t_value = test_gt.get("reported_t_value")

            if finding_id == "F1":
                # Overconfidence: one-sample t-test against baseline of 1
                baseline = 1.0
                mean_agent = float(np.mean(overconfidence_scores))
                sd_agent = float(np.std(overconfidence_scores, ddof=1)) if n_agent > 1 else 0.0

                # For F1, human significance must be computed from reported stats
                # since the reported_statistics string has no p-value
                human_mean = 5.17
                human_sd = 2.64
                human_n = 77
                human_t = (human_mean - baseline) / (human_sd / np.sqrt(human_n))
                human_p_value = float(stats.t.sf(human_t, human_n - 1))
                human_significant = human_p_value < sig_level

                if sd_agent > 0 and n_agent >= 2:
                    t_stat_val, p_value_val = stats.ttest_1samp(overconfidence_scores, baseline)
                    t_stat = float(t_stat_val)
                    # One-sided test: mean > baseline
                    p_value = float(p_value_val / 2) if t_stat > 0 else float(1 - p_value_val / 2)
                    agent_significant = p_value < sig_level
                    direction_match = (mean_agent > baseline)

            elif finding_id == "F2":
                model_predictors = ["risk_perception"]
                model_data = _select_complete_cases(
                    participant_scores,
                    outcome="opportunity_evaluation",
                    predictors=model_predictors,
                )
                if model_data is not None:
                    model = _fit_ols(model_data["y"], model_data["x"], model_predictors)
                    if model is not None:
                        model_n = model["n_obs"]
                        coefficient = model["coefficients"]["risk_perception"]
                        coefficient_agent = coefficient["coefficient"]
                        standard_error_agent = coefficient["standard_error"]
                        t_stat = coefficient["t_value"]
                        p_value = coefficient["p_value"]
                        agent_significant = p_value < sig_level
                        direction_match = (coefficient_agent < 0) if h_expected == -1 else (coefficient_agent > 0)

            elif finding_id in ("F3", "F4", "F5"):
                model_predictors = [
                    "overconfidence",
                    "small_numbers",
                    "planning_fallacy",
                    "illusion_of_control",
                    "risk_propensity",
                    "age",
                ]
                target_variable = {
                    "F3": ("risk_perception", "illusion_of_control"),
                    "F4": ("opportunity_evaluation", "illusion_of_control"),
                    "F5": ("opportunity_evaluation", "small_numbers"),
                }
                outcome_name, predictor_of_interest = target_variable[finding_id]
                model_data = _select_complete_cases(
                    participant_scores,
                    outcome=outcome_name,
                    predictors=model_predictors,
                )
                if model_data is not None:
                    model = _fit_ols(model_data["y"], model_data["x"], model_predictors)
                    if model is not None:
                        model_n = model["n_obs"]
                        coefficient = model["coefficients"][predictor_of_interest]
                        coefficient_agent = coefficient["coefficient"]
                        standard_error_agent = coefficient["standard_error"]
                        t_stat = coefficient["t_value"]
                        p_value = coefficient["p_value"]
                        agent_significant = p_value < sig_level
                        if h_expected == -1:
                            direction_match = coefficient_agent < 0
                        elif h_expected == 1:
                            direction_match = coefficient_agent > 0
                        else:
                            direction_match = True

            # Compute replication metric
            replication = None
            if human_significant is not None and agent_significant is not None and direction_match is not None:
                replication = human_significant and agent_significant and direction_match

            test_result = {
                "study_id": "study_013",
                "sub_study_id": "keh_foo_lim_opportunity_evaluation",
                "finding_id": finding_id,
                "n_agent": n_agent,
                "model_n": model_n,
                "model_predictors": model_predictors,
                "mean_agent": mean_agent,
                "sd_agent": sd_agent,
                "coefficient_agent": coefficient_agent,
                "standard_error_agent": standard_error_agent,
                "human_coefficient": human_coefficient,
                "human_t_value": human_t_value,
                "t_stat": t_stat,
                "r_stat": r_stat,
                "p_value": float(p_value) if p_value is not None else None,
                "significant": agent_significant,
                "direction_match": direction_match,
                "human_p_value": human_p_value,
                "human_significant": human_significant,
                "replication": replication,
            }
            test_results.append(test_result)

    return {"test_results": test_results}
