import json
import re
from math import erfc, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

_ground_truth_cache = None

SOURCE_ID = "davis2011"
SIG_LEVEL = 0.05


def _expected_direction_to_int(expected_dir_str: str) -> int:
    if not expected_dir_str:
        return 0
    s = str(expected_dir_str).lower()
    if s in ("positive", "greater", ">"):
        return 1
    if s in ("negative", "less", "<"):
        return -1
    return 0


def random_effects_regression(y, X, groups) -> Optional[Dict[str, Any]]:
    """Random-intercept GLS regression (Swamy-Arora variance components).

    y_it = b0 + X_it b + u_i + e_it, with a random effect u_i per subject. This
    is the estimator Stata's ``xtreg, re`` uses by default: sigma_e^2 from the
    within (fixed-effects) regression, sigma_u^2 from the between regression,
    then OLS on the quasi-demeaned data y_it - theta_i * mean_i(y). Standard
    errors are conventional, sigma_e^2 (X*'X*)^-1, with normal (z) p-values.

    Returns None when the model is not estimable (fewer than K + 2 subjects,
    no within-subject variation in a regressor, or no residual variance).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    groups = np.asarray(groups)
    n_obs, k = X.shape
    labels, inverse = np.unique(groups, return_inverse=True)
    n_groups = len(labels)
    if n_groups < k + 2 or n_obs - n_groups - k <= 0:
        return None

    t_i = np.bincount(inverse).astype(float)
    y_bar = np.bincount(inverse, weights=y) / t_i
    X_bar = np.column_stack([np.bincount(inverse, weights=X[:, j]) / t_i for j in range(k)])

    # Within regression -> idiosyncratic variance.
    y_w = y - y_bar[inverse]
    X_w = X - X_bar[inverse]
    if np.linalg.matrix_rank(X_w) < k:
        return None
    b_w, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    sigma_e2 = float(np.sum((y_w - X_w @ b_w) ** 2) / (n_obs - n_groups - k))
    if sigma_e2 <= 0:
        return None

    # Between regression (one row per subject) -> subject-level variance.
    Z_b = np.column_stack([np.ones(n_groups), X_bar])
    b_b, *_ = np.linalg.lstsq(Z_b, y_bar, rcond=None)
    sigma_b2 = float(np.sum((y_bar - Z_b @ b_b) ** 2) / (n_groups - k - 1))
    t_harmonic = n_groups / float(np.sum(1.0 / t_i))
    sigma_u2 = max(0.0, sigma_b2 - sigma_e2 / t_harmonic)

    theta = 1.0 - np.sqrt(sigma_e2 / (t_i * sigma_u2 + sigma_e2))
    th = theta[inverse]
    y_star = y - th * y_bar[inverse]
    X_star = np.column_stack([1.0 - th, X - th[:, None] * X_bar[inverse]])
    xtx = X_star.T @ X_star
    if np.linalg.matrix_rank(xtx) < k + 1:
        return None
    beta = np.linalg.solve(xtx, X_star.T @ y_star)
    se = np.sqrt(np.diag(sigma_e2 * np.linalg.inv(xtx)))
    z = beta / se
    p = np.array([erfc(abs(v) / sqrt(2)) for v in z])
    return {
        "coef": beta.tolist(),
        "se": se.tolist(),
        "z": z.tolist(),
        "p": p.tolist(),
        "sigma_u": sqrt(sigma_u2),
        "sigma_e": sqrt(sigma_e2),
        "n_obs": int(n_obs),
        "n_groups": int(n_groups),
    }


def _agent_panel(results: Dict[str, Any]) -> Dict[str, List]:
    """One row per valid reserve-price decision: price, bidders, period, agent."""
    prices: List[int] = []
    bidders: List[float] = []
    periods: List[float] = []
    agents: List[int] = []
    for agent_index, participant in enumerate(results.get("individual_data", [])):
        for position, response in enumerate(participant.get("responses", []), start=1):
            trial_info = response.get("trial_info", {}) or {}
            num_bidders = trial_info.get("num_bidders")
            if num_bidders is None:
                continue
            match = re.search(r"(\d+)", str(response.get("response_text", "")))
            if not match:
                continue
            rp = int(match.group(1))
            if rp < 0 or rp > 100:
                continue
            prices.append(rp)
            bidders.append(float(num_bidders))
            # Davis et al. (2011) use the decision period; the round number when
            # the response carries one, else the position in the agent's sequence.
            periods.append(float(trial_info.get("round") or position))
            agents.append(agent_index)
    return {"price": prices, "bidders": bidders, "period": periods, "agent": agents}


def evaluate_study(results):
    """
    Evaluates the agent's performance on the Davis et al. (2011) auction study (F1).

    The study follows one paper. The human result is Davis et al. (2011) Table 2,
    Model (1) (printed p. 182): a regression of the observed reserve price in a
    period on the number of bidders minus its average, with random effects for
    individual subjects. The coefficient on n - mean(n) is 2.98 [SE 0.08], p < 0.01
    (only the bound is printed). The agent decisions are analysed the same way:
    reserve price on n - mean(n) and period - mean(period), random intercept per
    agent. Replication = the agent coefficient is significant (0.05) and positive.
    Effect comparison = agent coefficient vs 2.98, both in reserve-price points per
    additional bidder.
    """
    global _ground_truth_cache

    if _ground_truth_cache is None:
        study_dir = Path(__file__).resolve().parent.parent / "source"
        with open(study_dir / "ground_truth.json", "r") as f:
            _ground_truth_cache = json.load(f)

    finding = _ground_truth_cache["studies"][0]["findings"][0]
    scoring = finding["scoring"]
    tests = finding["statistical_tests"]
    test = tests[scoring["statistical_test_index"]]
    if test["source_id"] != SOURCE_ID or scoring["source_id"] != SOURCE_ID:
        raise ValueError("study_014 is scored against Davis et al. (2011) only")
    if any(t.get("source_id") != SOURCE_ID for t in tests):
        raise ValueError("study_014 statistical_tests must all come from Davis et al. (2011)")

    expected_dir = _expected_direction_to_int(test.get("expected_direction", "positive"))
    human_coefficient = test["coefficient"]
    human_se = test["standard_error"]
    human_direction = 1 if human_coefficient > 0 else -1 if human_coefficient < 0 else 0
    if human_direction != expected_dir:
        raise ValueError("Davis 2011 coefficient sign disagrees with expected_direction")

    # Only the bound p < 0.01 is printed; no exact human p-value is derived.
    human_p_value_bound = test["p_value_bound"]
    human_significant = None
    if test.get("p_value_status") == "bound_printed" and human_p_value_bound is not None:
        human_significant = float(human_p_value_bound) <= SIG_LEVEL

    panel = _agent_panel(results)
    mean_by_bidders = {}
    for price, n in zip(panel["price"], panel["bidders"]):
        mean_by_bidders.setdefault(str(int(n)), []).append(price)
    mean_by_bidders = {k: float(np.mean(v)) for k, v in mean_by_bidders.items()}

    fit = None
    if len(panel["price"]) >= 10:
        n_arr = np.array(panel["bidders"])
        per_arr = np.array(panel["period"])
        X = np.column_stack([n_arr - n_arr.mean(), per_arr - per_arr.mean()])
        fit = random_effects_regression(panel["price"], X, panel["agent"])

    agent_coefficient = agent_se = agent_z = agent_p = None
    agent_ci = None
    agent_significant = direction_match = None
    if fit is not None:
        agent_coefficient = float(fit["coef"][1])
        agent_se = float(fit["se"][1])
        agent_z = float(fit["z"][1])
        agent_p = float(fit["p"][1])
        agent_ci = [agent_coefficient - 1.96 * agent_se, agent_coefficient + 1.96 * agent_se]
        agent_significant = agent_p < SIG_LEVEL
        agent_dir = 1 if agent_coefficient > 0 else -1 if agent_coefficient < 0 else 0
        direction_match = (expected_dir == 0) or (agent_dir == expected_dir)

    replication = None
    if human_significant is not None and agent_significant is not None:
        replication = bool(human_significant and agent_significant and direction_match)

    row = {
        "study_id": "second_price_auction",
        "sub_study_id": "second_price_auction",
        "finding_id": finding["finding_id"],
        "scenario": "reserve_price_on_bidder_count",
        "scoring_note": scoring["summary"],
        "reported_statistics": f"{test['reported_statistics']} (Davis et al. 2011, {test['location']})",
        # Human result: printed in Davis et al. (2011).
        "human_source_id": test["source_id"],
        "human_source_location": test["location"],
        "human_test": test["test_name"],
        "human_coefficient": human_coefficient,
        "human_coefficient_label": "printed",
        "human_standard_error": human_se,
        "human_direction": human_direction,
        "human_p_value_status": test["p_value_status"],
        "human_p_value_bound": human_p_value_bound,
        "human_p_value": None,
        "human_significant": human_significant,
        # Agent result: computed here with the same dependent variable and predictor.
        "agent_test": scoring["agent_model"]["description"],
        "agent_coefficient": agent_coefficient,
        "agent_coefficient_label": "computed",
        "agent_standard_error": agent_se,
        "agent_coefficient_ci95": agent_ci,
        "z_stat": agent_z,
        "p_value": agent_p,
        "significant": agent_significant,
        "direction_match": direction_match,
        "agent_sigma_u": fit["sigma_u"] if fit else None,
        "agent_sigma_e": fit["sigma_e"] if fit else None,
        "n_agent_observations": len(panel["price"]),
        "n_agents": fit["n_groups"] if fit else len(set(panel["agent"])),
        "mean_agent_by_bidders": mean_by_bidders,
        # Effect comparison in the paper's own units.
        "effect_size_metric": "re_coefficient_reserve_price_points_per_bidder",
        "human_effect_size": human_coefficient,
        "agent_effect_size": agent_coefficient,
        "effect_gap": (agent_coefficient - human_coefficient) if agent_coefficient is not None else None,
        "replication": replication,
    }
    return {"test_results": [row]}
