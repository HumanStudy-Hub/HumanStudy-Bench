import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from stats_lib import parse_p_value_from_reported

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


def evaluate_study(results):
    """
    Evaluates the agent's performance on the Davis et al. (2011) auction study.
    Tests whether the agent exhibits a positive correlation between reserve price
    and number of bidders (F1).
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

    # Get ground truth for F1
    test_results = []
    study_gt = ground_truth["studies"][0]
    finding = study_gt["findings"][0]
    finding_id = finding["finding_id"]
    stat_test = finding["statistical_tests"][0]
    expected_dir = _expected_direction_to_int(stat_test.get("expected_direction", "positive"))
    reported_stats = stat_test.get("reported_statistics", "")

    human_p_value = None
    human_significant = None
    parsed_p, parsed_sig, _ = parse_p_value_from_reported(reported_stats, sig_level)
    if parsed_p is not None:
        human_p_value = parsed_p
        human_significant = parsed_sig

    human_r = ground_truth["studies"][0]["findings"][0]["original_data_points"]["data"]["overall"]["pearson_r"]

    # Compute agent statistics
    agent_r = None
    p_value = None
    agent_significant = None
    direction_match = None
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

    replication = None
    if human_significant is not None and agent_significant is not None and direction_match is not None:
        replication = human_significant and agent_significant and direction_match

    test_results.append({
        "study_id": "second_price_auction",
        "sub_study_id": "second_price_auction",
        "finding_id": finding_id,
        "scenario": "reserve_price_bidder_correlation",
        "n_agent_observations": len(reserve_prices),
        "mean_agent_by_bidders": mean_by_bidders,
        "agent_pearson_r": agent_r,
        "t_stat": float(t_stat) if agent_r is not None and not np.isnan(agent_r) else None,
        "p_value": float(p_value) if p_value is not None else None,
        "significant": agent_significant,
        "direction_match": direction_match,
        "human_pearson_r": human_r,
        "human_p_value": human_p_value,
        "human_significant": human_significant,
        "replication": replication,
    })

    return {"test_results": test_results}
