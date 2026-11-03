import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

_ground_truth_cache = None


def _expected_direction_to_int(expected_dir_str: str) -> int:
    if not expected_dir_str:
        return 0
    s = str(expected_dir_str).lower()
    if s in ("positive", "greater", ">"):
        return 1
    if s in ("negative", "less", "<"):
        return -1
    return 0


def _r_to_d(r):
    if r is None or np.isnan(r) or abs(r) >= 1:
        return None
    return float(2 * r / np.sqrt(1 - r ** 2))


def evaluate_study(results):
    """
    Evaluates the agent's performance on the Davis et al. (2011) auction study.
    Tests whether the agent exhibits a positive correlation between reserve price
    and number of bidders (F1).

    Human significance and direction are scored against the original Davis et al.
    (2011) Table 2 regression (coefficient 2.98, p < .01). The human effect size is
    the later Feng et al. (2025) Pearson r = .42, which has no printed p-value and is
    used for the effect-size comparison only. Each output field names its source.
    """
    global _ground_truth_cache

    if _ground_truth_cache is None:
        study_dir = Path(__file__).resolve().parent.parent / "source"
        with open(study_dir / "ground_truth.json", "r") as f:
            _ground_truth_cache = json.load(f)

    ground_truth = _ground_truth_cache
    sig_level = 0.05

    # Extract agent reserve prices and bidder counts
    reserve_prices = []
    bidder_counts = []
    by_bidder_count = {}

    for participant in results.get("individual_data", []):
        for response in participant.get("responses", []):
            response_text = response.get("response_text", "")
            trial_info = response.get("trial_info", {})
            num_bidders = trial_info.get("num_bidders")

            if num_bidders is None:
                continue

            match = re.search(r"(\d+)", str(response_text))
            if not match:
                continue
            rp = int(match.group(1))
            if rp < 0 or rp > 100:
                continue

            reserve_prices.append(rp)
            bidder_counts.append(num_bidders)

            key = str(num_bidders)
            if key not in by_bidder_count:
                by_bidder_count[key] = []
            by_bidder_count[key].append(rp)

    # Ground truth for F1. Two human statistics feed this row, from two papers
    # (see source/PROVENANCE.md and the finding's "scoring" block):
    #   significance and direction: Davis et al. (2011) Table 2 regression
    #   effect size:                Feng et al. (2025) Table 2 Pearson r
    test_results = []
    study_gt = ground_truth["studies"][0]
    finding = study_gt["findings"][0]
    finding_id = finding["finding_id"]
    scoring = finding["scoring"]
    tests = finding["statistical_tests"]
    sig_test = tests[scoring["significance_and_direction"]["statistical_test_index"]]
    effect_test = tests[scoring["effect_size"]["statistical_test_index"]]
    assert sig_test["source_id"] == scoring["significance_and_direction"]["source_id"]
    assert effect_test["source_id"] == scoring["effect_size"]["source_id"]

    expected_dir = _expected_direction_to_int(sig_test.get("expected_direction", "positive"))

    # Davis 2011 prints only a bound (p < .01), so the human p-value stays None;
    # significance is decided by the printed bound against the 0.05 level.
    human_p_value = None
    human_p_value_bound = sig_test.get("p_value_bound")
    human_significant = None
    if sig_test.get("p_value_status") == "bound_printed" and human_p_value_bound is not None:
        human_significant = float(human_p_value_bound) <= sig_level
    human_coefficient = sig_test.get("coefficient")
    human_direction = None
    if human_coefficient is not None:
        human_direction = 1 if human_coefficient > 0 else -1 if human_coefficient < 0 else 0
        # The printed coefficient's sign is the human direction the agent must match.
        assert human_direction == expected_dir, "Davis 2011 coefficient sign disagrees with expected_direction"

    human_r = finding["verified_printed_benchmark"]["pearson_r"]
    human_effect_d = _r_to_d(human_r)

    # Compute agent statistics
    agent_r = None
    p_value = None
    agent_significant = None
    direction_match = None
    agent_effect_d = None
    mean_by_bidders = {}

    for bc, prices in by_bidder_count.items():
        mean_by_bidders[bc] = float(np.mean(prices))

    if len(reserve_prices) >= 10:
        rp_arr = np.array(reserve_prices, dtype=float)
        bc_arr = np.array(bidder_counts, dtype=float)
        agent_r = float(np.corrcoef(rp_arr, bc_arr)[0, 1])

        if not np.isnan(agent_r):
            # Approximate p-value from r and n using t-distribution
            n = len(reserve_prices)
            t_stat = agent_r * np.sqrt((n - 2) / (1 - agent_r ** 2))
            # Two-tailed p-value approximation (normal for large n)
            from math import erfc, sqrt
            p_value = erfc(abs(t_stat) / sqrt(2))

            agent_significant = p_value < sig_level
            agent_dir = 1 if agent_r > 0 else -1
            direction_match = (expected_dir == 0) or (agent_dir == expected_dir)
            agent_effect_d = _r_to_d(agent_r)

    replication = None
    if human_significant is not None and agent_significant is not None and direction_match is not None:
        replication = human_significant and agent_significant and direction_match

    test_results.append({
        "study_id": "second_price_auction",
        "sub_study_id": "second_price_auction",
        "finding_id": finding_id,
        "scenario": "reserve_price_bidder_correlation",
        "scoring_note": scoring["summary"],
        "reported_statistics": (
            f"Significance and direction: {scoring['significance_and_direction']['statistic']} "
            f"(Davis et al. 2011, {sig_test['location']}). "
            f"Effect size: {scoring['effect_size']['statistic']} "
            f"(Feng et al. 2025, {effect_test['location']})."
        ),
        "n_agent_observations": len(reserve_prices),
        "mean_agent_by_bidders": mean_by_bidders,
        "agent_test": "Pearson correlation of reserve price with bidder count",
        "agent_pearson_r": agent_r,
        "t_stat": float(t_stat) if agent_r is not None and not np.isnan(agent_r) else None,
        "p_value": float(p_value) if p_value is not None else None,
        "significant": agent_significant,
        "direction_match": direction_match,
        # Human significance and direction: Davis et al. (2011) regression.
        "human_significance_source_id": sig_test["source_id"],
        "human_significance_source_location": sig_test["location"],
        "human_significance_test": sig_test["test_name"],
        "human_coefficient": human_coefficient,
        "human_direction": human_direction,
        "human_p_value_status": sig_test["p_value_status"],
        "human_p_value_bound": human_p_value_bound,
        "human_p_value": human_p_value,
        "human_significant": human_significant,
        # Human effect size: Feng et al. (2025) Pearson r (no p-value printed).
        "human_effect_source_id": effect_test["source_id"],
        "human_effect_source_location": effect_test["location"],
        "human_effect_test": effect_test["test_name"],
        "human_effect_p_value_status": effect_test["p_value_status"],
        "human_pearson_r": human_r,
        # Cohen's d equivalents of the two Pearson r values, d = 2r / sqrt(1 - r^2),
        # the same identity playground/analysis.py applies to a reported r.
        "effect_size_metric": "cohens_d_from_pearson_r",
        "human_effect_d": human_effect_d,
        "agent_effect_d": agent_effect_d,
        "replication": replication,
    })

    return {"test_results": test_results}
