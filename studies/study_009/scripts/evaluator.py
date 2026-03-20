import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from stats_lib import (
    calc_bf_t, 
    calc_bf_chisq, 
    calc_posteriors_3way,
    calc_pas,
    calc_bf_from_p,
    calc_bf_mannwhitneyu,
    calc_bf_binomial,
    parse_p_value_from_reported,
    get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, float]:
    """
    Parses the LLM response to extract the numbers chosen for each round.
    Format expected: Q1=<number>, Q2=<number>, Q3=<number>, Q4=<number>
    """
    results = {}
    # Use regex to find Q1=val, Q2=val, etc.
    pattern = re.compile(r"Q(\d+)\s*=\s*(\d+(?:\.\d+)?)")
    matches = pattern.findall(response_text)
    for q_idx, val in matches:
        try:
            results[f"Q{q_idx}"] = float(val)
        except ValueError:
            continue
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_009: 固定Q1-Q4或从round_feedbacks动态确定
    """
    round_feedbacks = trial_info.get("round_feedbacks", [])
    if round_feedbacks:
        num_rounds = len(round_feedbacks)
        return {f"Q{i+1}" for i in range(num_rounds)}
    else:
        # 默认4个rounds
        return {"Q1", "Q2", "Q3", "Q4"}

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates agent performance for Study 009 (p-Beauty Contest).
    Calculates Bayesian Alignment Score (PAS).
    """
    # 1. Load ground truth and metadata
    study_id = "study_009"
    study_dir = Path(__file__).resolve().parent.parent / "source"
    gt_path = study_dir / "ground_truth.json"
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    # Load metadata for finding and test weights
    metadata = {}
    metadata_path = study_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # Build weight maps: finding_id -> weight, (finding_id, test_name) -> weight
    finding_weights = {}
    test_weights = {}
    for finding in metadata.get("findings", []):
        finding_id = finding.get("finding_id")
        finding_weight = finding.get("weight", 1.0)
        if finding_id:
            finding_weights[finding_id] = finding_weight
        
        for test in finding.get("tests", []):
            test_name = test.get("test_name")
            test_weight = test.get("weight", 1.0)
            if finding_id and test_name:
                test_weights[(finding_id, test_name)] = test_weight

    # 2. Extract agent data
    # Structure: agent_data[sub_study_id] = [ {Q1: val, Q2: val, ...}, ... ]
    agent_data = {
        "p_0.5_condition": [],
        "p_0.66_condition": [],
        "p_1.33_condition": []
    }
    
    # Also extract session data from round_feedbacks for F5 and F6
    # Structure: sessions[sub_study_id] = [session1, session2, ...]
    # Each session is a dict with round data: {round_1: {choices, mean, p_times_mean, winning_number}, ...}
    sessions = {
        "p_0.5_condition": [],
        "p_0.66_condition": [],
        "p_1.33_condition": []
    }
    
    # Track unique sessions by session_id (instead of round_feedbacks signature)
    seen_sessions = {
        "p_0.5_condition": set(),
        "p_0.66_condition": set(),
        "p_1.33_condition": set()
    }

    individual_data = results.get("individual_data", [])
    if not individual_data:
        return {
            "score": 0.5,
            "error": "No individual_data found in results",
            "test_results": [],
            "finding_results": [],
            "substudy_results": []
        }
    
    # Check if data is flat or nested structure
    is_flat = len(individual_data) > 0 and 'responses' not in individual_data[0]
    
    na_count = 0
    valid_count = 0
    
    for participant in individual_data:
        if is_flat:
            # Flat structure: participant is actually a response
            responses = [participant]
        else:
            # Nested structure: participant has responses list
            responses = participant.get("responses", [])
        
        for resp in responses:
            # Handle both nested and flat structures
            if isinstance(resp, dict):
                trial_info = resp.get("trial_info", {})
                response_text = resp.get("response_text", "")
            else:
                continue
            
            sub_id = trial_info.get("sub_study_id")
            if sub_id not in agent_data:
                continue
            
            parsed = parse_agent_responses(response_text)
            # Only append if we have at least one valid round (not all N/A)
            if parsed and len(parsed) > 0:
                agent_data[sub_id].append(parsed)
                valid_count += 1
            else:
                na_count += 1
            
            # Extract session data using session_id (more reliable than round_feedbacks signature)
            session_id = trial_info.get("session_id")
            round_feedbacks = trial_info.get("round_feedbacks", [])
            
            if session_id and round_feedbacks:
                # Use session_id to identify unique sessions
                if session_id not in seen_sessions[sub_id]:
                    seen_sessions[sub_id].add(session_id)
                    # Build session data structure
                    session_data = {}
                    for rf in round_feedbacks:
                        round_num = rf.get("round")
                        feedback = rf.get("feedback", {})
                        session_data[f"round_{round_num}"] = {
                            "choices": feedback.get("choices", {}),
                            "mean": feedback.get("mean", 0),
                            "p_times_mean": feedback.get("p_times_mean", 0),
                            "winning_number": feedback.get("winning_number", 0)
                        }
                    sessions[sub_id].append(session_data)
    
    # Check data availability and log warnings
    total_agent_data = sum(len(data) for data in agent_data.values())
    if total_agent_data == 0:
        return {
            "score": 0.5,
            "error": f"No valid agent data extracted. Found {na_count} N/A responses and {valid_count} valid responses.",
            "test_results": [],
            "finding_results": [],
            "substudy_results": []
        }
    
    print(f"Extracted agent data: {total_agent_data} valid responses ({valid_count} valid, {na_count} N/A)")
    for sub_id, data in agent_data.items():
        print(f"  {sub_id}: {len(data)} responses")

    test_results = []

    # 3. Process Findings
    # F1: Mann-Whitney tests (distribution differences)
    # F2: Binomial test (concentration in neighborhoods)
    # F3: Round 1 vs Round 4 comparison (convergence toward equilibrium)
    findings_gt = ground_truth["studies"][0]["findings"]
    
    # --- Finding 1: Distributions influenced by p ---
    f1 = next(f for f in findings_gt if f["finding_id"] == "F1")
    
    # Test 1: p=1/2 vs p=2/3
    t1_gt = f1["statistical_tests"][0]
    bf_h1 = calc_bf_t(3.29, 48, 67, independent=True)
    pi_h1 = calc_posteriors_3way(bf_h1, 1, prior_odds=10.0) # Expected 0.5 < 0.66
    
    group_05 = [d["Q1"] for d in agent_data["p_0.5_condition"] if "Q1" in d]
    group_066 = [d["Q1"] for d in agent_data["p_0.66_condition"] if "Q1" in d]
    
    p_val_agent_f1 = None
    u_stat_agent_f1 = None
    if len(group_05) > 5 and len(group_066) > 5:
        u_stat_agent_f1, p_val_agent_f1 = stats.mannwhitneyu(group_05, group_066)
        z_stat = abs(stats.norm.ppf(p_val_agent_f1 / 2)) if p_val_agent_f1 > 0 else 5.0
        bf_a1 = calc_bf_t(z_stat, len(group_05), len(group_066), independent=True)
        # Direction: expect mean(0.5) < mean(0.66)
        a_dir = 1 if np.mean(group_066) > np.mean(group_05) else -1
        pi_a1 = calc_posteriors_3way(bf_a1, a_dir)
        reason1 = f"Mann-Whitney U={u_stat_agent_f1}, p={p_val_agent_f1:.4f}, n1={len(group_05)}, n2={len(group_066)}"
    else:
        pi_a1 = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        reason1 = "Insufficient data"
    
    test_result_f1 = {
        "study_id": "Experiment 1",
        "sub_study_id": "p_0.5_vs_p_0.66",
        "finding_id": "F1",
        "test_name": t1_gt["test_name"],
        "pi_human": float(pi_h1['pi_plus'] + pi_h1['pi_minus']),
        "pi_agent": float(pi_a1['pi_plus'] + pi_a1['pi_minus']),
        "pi_human_3way": pi_h1,
        "pi_agent_3way": pi_a1,
        "pas": float(calc_pas(pi_h1, pi_a1)),
        "pi_human_source": t1_gt["reported_statistics"],
        "agent_reason": reason1,
        "statistical_test_type": "mannwhitneyu",
        "human_test_statistic": "3.29", 
        "agent_test_statistic": f"{u_stat_agent_f1:.2f}" if u_stat_agent_f1 is not None else ""
    }
    add_statistical_replication_fields(
        test_result_f1, t1_gt, p_val_agent_f1, u_stat_agent_f1, "mannwhitneyu",
        n_agent=len(group_05), n2_agent=len(group_066),
        n_human=48, n2_human=67
    )
    test_results.append(test_result_f1)

    # Test 2: p=2/3 vs p=4/3
    t2_gt = f1["statistical_tests"][1]
    bf_h2 = calc_bf_t(3.89, 67, 51, independent=True)
    pi_h2 = calc_posteriors_3way(bf_h2, 1, prior_odds=10.0)
    
    group_133 = [d["Q1"] for d in agent_data["p_1.33_condition"] if "Q1" in d]
    
    p_val_agent_f1_2 = None
    u_stat_agent_f1_2 = None
    if len(group_066) > 5 and len(group_133) > 5:
        u_stat_agent_f1_2, p_val_agent_f1_2 = stats.mannwhitneyu(group_066, group_133)
        z_stat = abs(stats.norm.ppf(p_val_agent_f1_2 / 2)) if p_val_agent_f1_2 > 0 else 5.0
        bf_a2 = calc_bf_t(z_stat, len(group_066), len(group_133), independent=True)
        # Direction: expect mean(0.66) < mean(1.33)
        a_dir = 1 if np.mean(group_133) > np.mean(group_066) else -1
        pi_a2 = calc_posteriors_3way(bf_a2, a_dir)
        reason2 = f"Mann-Whitney U={u_stat_agent_f1_2}, p={p_val_agent_f1_2:.4f}, n1={len(group_066)}, n2={len(group_133)}"
    else:
        pi_a2 = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        reason2 = "Insufficient data"

    test_result_f1_2 = {
        "study_id": "Experiment 1",
        "sub_study_id": "p_0.66_vs_p_1.33",
        "finding_id": "F1",
        "test_name": t2_gt["test_name"],
        "pi_human": float(pi_h2['pi_plus'] + pi_h2['pi_minus']),
        "pi_agent": float(pi_a2['pi_plus'] + pi_a2['pi_minus']),
        "pi_human_3way": pi_h2,
        "pi_agent_3way": pi_a2,
        "pas": float(calc_pas(pi_h2, pi_a2)),
        "pi_human_source": t2_gt["reported_statistics"],
        "agent_reason": reason2,
        "statistical_test_type": "mannwhitneyu",
        "human_test_statistic": "3.89",
        "agent_test_statistic": f"{u_stat_agent_f1_2:.2f}" if u_stat_agent_f1_2 is not None else ""
    }
    add_statistical_replication_fields(
        test_result_f1_2, t2_gt, p_val_agent_f1_2, u_stat_agent_f1_2, "mannwhitneyu",
        n_agent=len(group_066), n2_agent=len(group_133),
        n_human=67, n2_human=51
    )
    test_results.append(test_result_f1_2)

    # --- Finding 2: Concentration in neighborhood intervals ---
    # Paper: Nagel (1995), p. 1318
    # Statistical test: One-sided Binomial Test (NOT Chi-square approximation)
    # Neighborhood definition: 50p^(n±1/4) for step n (NOT a fixed ±5 interval)
    # Sample sizes from paper: p=1/2: N=48, p=2/3: N=67, p=4/3: N=51
    f2 = next(f for f in findings_gt if f["finding_id"] == "F2")
    t3_gt = f2["statistical_tests"][0]
    
    # Get human data from ground truth (estimated_k and n from paper)
    f2_data = f2.get("original_data_points", {}).get("data", {})

    p_map = {"p_0.5_condition": 0.5, "p_0.66_condition": 0.66, "p_1.33_condition": 1.33}
    
    # Paper sample sizes and estimated successes from ground truth
    # Use actual data (estimated_k, n) instead of p-values to calculate Pi_Human
    paper_f2_data = {
        "p_0.5_condition": {
            "n": f2_data.get("p_0.5_neighborhoods", {}).get("n", 48),
            "estimated_k": f2_data.get("p_0.5_neighborhoods", {}).get("estimated_k", 24)
        },
        "p_0.66_condition": {
            "n": f2_data.get("p_0.66_neighborhoods", {}).get("n", 67),
            "estimated_k": f2_data.get("p_0.66_neighborhoods", {}).get("estimated_k", 33)
        },
        "p_1.33_condition": {
            "n": f2_data.get("p_1.33_neighborhoods", {}).get("n", 51),
            "estimated_k": f2_data.get("p_1.33_neighborhoods", {}).get("estimated_k")
        }
    }
    
    # Test concentration for each p-value condition and each round separately
    for sub_id in ["p_0.5_condition", "p_0.66_condition", "p_1.33_condition"]:
        p_val = p_map[sub_id]
        n_condition = len(agent_data[sub_id])
        if n_condition == 0:
            continue
        
        # Calculate human Pi from actual data (estimated_k and n) instead of p-value
        human_f2_info = paper_f2_data[sub_id]
        n_paper = human_f2_info["n"]
        
        # For p=1.33, we have round-specific estimated_k from Figure 2C and text
        # For p=0.5 and p=0.66, we have overall estimated_k (applies to all rounds)
        p_133_round_data = f2_data.get("p_1.33_neighborhoods", {})
        
        # Test each round (Q1, Q2, Q3, Q4) separately
        for round_num in range(1, 5):
            # Get estimated_k for this round
            round_data_key = f"round_{round_num}"
            if sub_id == "p_1.33_condition" and round_data_key in p_133_round_data:
                # Use round-specific data from Figure 2C and text descriptions
                estimated_k = p_133_round_data[round_data_key].get("estimated_k")
                estimated_k_note = p_133_round_data[round_data_key].get("note", "")
            else:
                # For p=0.5 and p=0.66, use overall estimated_k (applies to all rounds)
                estimated_k = human_f2_info.get("estimated_k")
                estimated_k_note = ""
            
            if estimated_k is not None:
                # Use actual data: estimated_k successes out of n_paper
                # Paper reports "Almost 50%" in neighborhoods with p < 0.01 (for p=0.5 and p=0.66)
                # For p=1.33, data from Figure 2C (Round 1) and text (Rounds 2-4)
                # Calculate exact expected probability based on neighborhood coverage for this p value
                # We need to calculate neighborhoods for human data too (same as agent calculation)
                human_neighborhoods = []
                for n in [0, 1, 2]:
                    step_value = 50 * (p_val**n)
                    step_value = min(step_value, 100)
                    lower_exp = n - 0.25
                    upper_exp = n + 0.25
                    lower_bound = 50 * (p_val**lower_exp)
                    upper_bound = 50 * (p_val**upper_exp)
                    lower_bound = max(0, min(lower_bound, 100))
                    upper_bound = max(0, min(upper_bound, 100))
                    if p_val < 1:
                        lower_bound, upper_bound = min(lower_bound, upper_bound), max(lower_bound, upper_bound)
                    if n == 0:
                        if p_val < 1:
                            upper_bound = 50
                        elif p_val > 1:
                            lower_bound = 50
                    human_neighborhoods.append({
                        "lower": lower_bound,
                        "upper": upper_bound
                    })
                
                # Calculate exact expected probability from neighborhoods
                human_covered_intervals = []
                for nb in human_neighborhoods:
                    if nb["upper"] > nb["lower"]:
                        human_covered_intervals.append((nb["lower"], nb["upper"]))
                
                if human_covered_intervals:
                    sorted_intervals = sorted(human_covered_intervals)
                    merged = [sorted_intervals[0]]
                    for current in sorted_intervals[1:]:
                        if current[0] <= merged[-1][1]:
                            merged[-1] = (merged[-1][0], max(merged[-1][1], current[1]))
                        else:
                            merged.append(current)
                    exact_coverage_human = sum(upper - lower for lower, upper in merged)
                    expected_p_human = exact_coverage_human / 100.0
                else:
                    expected_p_human = 0.3  # Fallback
                
                # Use calc_bf_binomial with actual k and n (not chi-square or p-value)
                bf_h_f2 = calc_bf_binomial(estimated_k, n_paper, expected_p_human)
                pi_h_f2 = calc_posteriors_3way(bf_h_f2, 1, prior_odds=10.0)
                if estimated_k_note:
                    human_source_f2 = f"k={estimated_k}, n={n_paper} (from {estimated_k_note})"
                else:
                    human_source_f2 = f"k={estimated_k}, n={n_paper} (from paper data)"
            else:
                # Fallback: use p-value method (should not happen with updated data)
                z_human = 1.96
                bf_h_f2 = calc_bf_chisq(z_human**2, n_paper, df=1)
                pi_h_f2 = calc_posteriors_3way(bf_h_f2, 1, prior_odds=10.0)
                human_source_f2 = f"p < 0.05, n={n_paper} (fallback: no estimated_k)"
            round_key_agent = f"Q{round_num}"
            
            # Calculate theoretical steps and neighborhoods for this p value
            # Paper definition (Nagel 1995): neighborhood for step n is [50p^(n-1/4), 50p^(n+1/4)]
            # This is a DYNAMIC interval that depends on p and n, NOT a fixed ±5 interval
            # IMPORTANT: For p < 1, Step 0 (50) neighborhood is bounded from the right by 50
            # Paper: "the neighborhood of 50p^0 is bounded from the right side by 50 for p < 1"
            # Source: Paper footnote/Table 1 description of neighborhood boundaries
            neighborhoods = []
            for n in [0, 1, 2]:
                step_value = 50 * (p_val**n)
                # Cap at 100 for p=1.33
                step_value = min(step_value, 100)
                
                # Calculate neighborhood bounds: 50p^(n±1/4)
                lower_exp = n - 0.25
                upper_exp = n + 0.25
                lower_bound = 50 * (p_val**lower_exp)
                upper_bound = 50 * (p_val**upper_exp)
                
                # Cap bounds at [0, 100]
                lower_bound = max(0, min(lower_bound, 100))
                upper_bound = max(0, min(upper_bound, 100))
                
                # For p < 1, lower_bound > upper_bound, so we need to swap
                if p_val < 1:
                    lower_bound, upper_bound = min(lower_bound, upper_bound), max(lower_bound, upper_bound)
                
                # CRITICAL FIX: Step 0 (n=0) neighborhood boundary truncation (Paper Footnote 12)
                # For p < 1: Step 0 is bounded from the right by 50
                # Paper: "the neighborhood of 50p^0 is bounded from the right side by 50 for p < 1"
                # This is because choosing >50 is a dominated strategy for p < 1
                # For p > 1: Step 0 is bounded from the left by 50
                # Paper: "the neighborhood of 50p^0 is bounded from the left side by 50 for p > 1"
                if n == 0:
                    if p_val < 1:
                        upper_bound = 50
                    elif p_val > 1:
                        lower_bound = 50
                
                neighborhoods.append({
                    "step": step_value,
                    "n": n,
                    "lower": lower_bound,
                    "upper": upper_bound
                })
            
            # Count successes (choices within neighborhood of any step)
            successes = 0
            total_n = 0
            
            for d in agent_data[sub_id]:
                if round_key_agent in d and d[round_key_agent] is not None:
                    total_n += 1
                    choice = d[round_key_agent]
                    # Check if choice is within any neighborhood
                    in_neighborhood = False
                    for nb in neighborhoods:
                        if nb["lower"] <= choice <= nb["upper"]:
                            in_neighborhood = True
                            break
                    if in_neighborhood:
                        successes += 1
    
            if total_n >= 10:  # Minimum sample size for meaningful test
                # Expected probability calculation:
                # Calculate exact expected probability based on actual neighborhood coverage
                # Paper uses dynamic neighborhoods 50p^(n±1/4) for n=0,1,2
                # Total coverage depends on p value and varies by condition
                
                # Calculate total coverage of all neighborhoods (handling overlaps)
                covered_intervals = []
                for nb in neighborhoods:
                    if nb["upper"] > nb["lower"]:  # Valid interval
                        covered_intervals.append((nb["lower"], nb["upper"]))
                
                # Merge overlapping intervals
                if covered_intervals:
                    sorted_intervals = sorted(covered_intervals)
                    merged = [sorted_intervals[0]]
                    for current in sorted_intervals[1:]:
                        if current[0] <= merged[-1][1]:
                            # Overlapping: merge
                            merged[-1] = (merged[-1][0], max(merged[-1][1], current[1]))
                        else:
                            # Non-overlapping: add new interval
                            merged.append(current)
                    
                    # Calculate total coverage
                    exact_coverage = sum(upper - lower for lower, upper in merged)
                    p0 = exact_coverage / 100.0  # Expected probability = total coverage / 100
                else:
                    # Fallback if no valid intervals (should not happen)
                    p0 = 0.3
                
                # One-sided Binomial Test (Paper p. 1318)
                # H0: p = p0 (expected probability based on neighborhood coverage)
                # H1: p > p0 (more concentrated than expected by random chance)
                # This is the exact test used in the original paper, not Chi-square approximation
                # Use calc_bf_binomial with actual k and n (not p-value)
                bf_f2 = calc_bf_binomial(successes, total_n, p0)
                pi_a_f2 = calc_posteriors_3way(bf_f2, 1) # Direction: concentration is "positive" finding
                
                # Also calculate p-value for reporting
                from scipy.stats import binomtest
                binom_result = binomtest(successes, total_n, p0, alternative='greater')
                p_val_binom = binom_result.pvalue
                
                step_values = [f"{nb['step']:.1f}" for nb in neighborhoods]
                reason_f2 = f"k={successes}, n={total_n}, expected_p={p0:.4f} (exact coverage), binomial_p={p_val_binom:.4f}, steps={step_values}, neighborhoods=50p^(n±1/4)"
                agent_stat = f"{p_val_binom:.4f}"
            else:
                pi_a_f2 = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                reason_f2 = f"Insufficient data: n={total_n}"
                agent_stat = ""
            
            test_result_f2 = {
                "study_id": "Experiment 1",
                "sub_study_id": f"{sub_id}_Round{round_num}",
                "finding_id": "F2",
                "test_name": f"Concentration test (Round {round_num})",
                "scenario": f"Round {round_num}",
                "pi_human": float(pi_h_f2['pi_plus'] + pi_h_f2['pi_minus']),
                "pi_agent": float(pi_a_f2['pi_plus'] + pi_a_f2['pi_minus']),
                "pi_human_3way": pi_h_f2,
                "pi_agent_3way": pi_a_f2,
                "pas": float(calc_pas(pi_h_f2, pi_a_f2)),
                "pi_human_source": human_source_f2,
                "agent_reason": reason_f2,
                "statistical_test_type": "binomial",
                "human_test_statistic": "",  # Binomial test doesn't have a single test statistic
                "agent_test_statistic": agent_stat,
                "human_k": estimated_k if 'estimated_k' in locals() and estimated_k is not None else None,
                "human_p0": expected_p_human if 'expected_p_human' in locals() else None
            }
            # Get test_gt for F2
            test_gt_f2 = {}
            for study in ground_truth.get("studies", []):
                for finding in study.get("findings", []):
                    if finding.get("finding_id") == "F2":
                        statistical_tests = finding.get("statistical_tests", [])
                        if statistical_tests:
                            test_gt_f2 = statistical_tests[0]
                        break
            add_statistical_replication_fields(
                test_result_f2, test_gt_f2, p_val_binom if 'p_val_binom' in locals() else None, successes if 'successes' in locals() else None, "binomial",
                n_agent=total_n,
                n_human=n_paper
            )
            test_results.append(test_result_f2)

    # --- Finding 3: Directional Movement (Unraveling) ---
    # Paper: Nagel (1995), p. 7
    # Statistical test: Binomial Test on transition directions
    # For p < 1: choices should decrease over time
    # For p > 1: choices should increase over time
    f3 = next(f for f in findings_gt if f["finding_id"] == "F3")
    f3_data = f3.get("original_data_points", {}).get("data", {})
    
    # Test directional movement for each condition
    for sub_id in ["p_0.5_condition", "p_0.66_condition", "p_1.33_condition"]:
        # Calculate human Pi from transition data
        p_val_map = {"p_0.5_condition": 0.5, "p_0.66_condition": 0.66, "p_1.33_condition": 1.33}
        p_val = p_val_map[sub_id]
        
        if sub_id in ["p_0.5_condition", "p_0.66_condition"]:
            # For p < 1: expect decrease (choices go down)
            human_data = f3_data.get(sub_id, {})
            human_k = human_data.get("decrease_count", 135 if sub_id == "p_0.5_condition" else 163)
            human_n = human_data.get("total_transitions", 144 if sub_id == "p_0.5_condition" else 201)
            # Use calc_bf_binomial with actual k and n
            bf_f3 = calc_bf_binomial(human_k, human_n, 0.5)
            pi_h_f3 = calc_posteriors_3way(bf_f3, 1, prior_odds=10.0)
            human_source = f"Binomial Test (Directional Movement): k={human_k}, n={human_n}, decrease rate={human_k/human_n:.2%} (Paper p. 7)"
        else:  # p_1.33_condition
            # For p > 1: expect increase (choices go up)
            human_data = f3_data.get(sub_id, {})
            human_k = human_data.get("increase_count", 133)
            human_n = human_data.get("total_transitions", 153)
            # Use calc_bf_binomial with actual k and n
            bf_f3 = calc_bf_binomial(human_k, human_n, 0.5)
            pi_h_f3 = calc_posteriors_3way(bf_f3, 1, prior_odds=10.0)
            human_source = f"Binomial Test (Directional Movement): k={human_k}, n={human_n}, increase rate={human_k/human_n:.2%} (Paper p. 7)"
        
        # Agent test: Count transitions in correct direction
        correct_transitions = 0
        total_transitions = 0
        
        # Check all transitions: R1->R2, R2->R3, R3->R4
        for d in agent_data[sub_id]:
            for t in range(1, 4):  # t = 1, 2, 3
                round_key_t = f"Q{t}"
                round_key_t1 = f"Q{t+1}"
                if round_key_t in d and round_key_t1 in d:
                    choice_t = d[round_key_t]
                    choice_t1 = d[round_key_t1]
                    if choice_t is not None and choice_t1 is not None:
                        total_transitions += 1
                        if p_val < 1:
                            # Expect decrease
                            if choice_t1 < choice_t:
                                correct_transitions += 1
                        else:  # p_val > 1
                            # Expect increase
                            if choice_t1 > choice_t:
                                correct_transitions += 1
        
        if total_transitions >= 10:
            # Use calc_bf_binomial with actual k and n
            bf_agent = calc_bf_binomial(correct_transitions, total_transitions, 0.5)
            pi_a_f3 = calc_posteriors_3way(bf_agent, 1)
            success_rate = correct_transitions / total_transitions if total_transitions > 0 else 0
            from scipy.stats import binomtest
            binom_result = binomtest(correct_transitions, total_transitions, 0.5, alternative='greater')
            p_val_binom = binom_result.pvalue
            reason_f3 = f"Binomial Test (Directional Movement): k={correct_transitions}, n={total_transitions}, success_rate={success_rate:.2%}, p={p_val_binom:.4f}"
            agent_stat_str = f"{correct_transitions}/{total_transitions}"
        else:
            pi_a_f3 = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
            reason_f3 = f"Insufficient transitions: n={total_transitions}"
            agent_stat_str = ""
        
        test_results.append({
            "study_id": "Experiment 1",
            "sub_study_id": sub_id,
            "finding_id": "F3",
            "test_name": "Directional Movement (Unraveling)",
            "scenario": "Transitions R1->R2, R2->R3, R3->R4",
            "pi_human": float(pi_h_f3['pi_plus'] + pi_h_f3['pi_minus']),
            "pi_agent": float(pi_a_f3['pi_plus'] + pi_a_f3['pi_minus']),
            "pi_human_3way": pi_h_f3,
            "pi_agent_3way": pi_a_f3,
            "pas": float(calc_pas(pi_h_f3, pi_a_f3)),
            "pi_human_source": human_source,
            "agent_reason": reason_f3,
            "statistical_test_type": "Binomial Test",
            "human_test_statistic": "",
            "agent_test_statistic": agent_stat_str
        })

    # --- Finding 4: Rate Comparison (Convergence Speed) ---
    # Paper: Nagel (1995), p. 9-10, Table 1
    # Statistical test: Mann-Whitney U Test comparing median rates of decrease
    # Compare p=1/2 median rates vs p=2/3 median rates
    f4 = next(f for f in findings_gt if f["finding_id"] == "F4")
    f4_data = f4.get("original_data_points", {}).get("data", {})
    
    # Human data: median rates from Table 1
    median_rates_p_05_human = f4_data.get("p_0.5_condition", {}).get("median_rates", [0.88, 0.98, 0.97])
    median_rates_p_066_human = f4_data.get("p_0.66_condition", {}).get("median_rates", [0.7, 0.91, 0.71, 0.76])
    
    # Calculate human Pi from actual U statistic
    u_stat_human, _ = stats.mannwhitneyu(median_rates_p_05_human, median_rates_p_066_human, alternative='two-sided')
    bf_f4 = calc_bf_mannwhitneyu(u_stat_human, len(median_rates_p_05_human), len(median_rates_p_066_human))
    pi_h_f4 = calc_posteriors_3way(bf_f4, 1, prior_odds=10.0)
    human_source_f4 = f"Mann-Whitney U (median rates): U={u_stat_human:.2f}, rates_p_05={median_rates_p_05_human}, rates_p_066={median_rates_p_066_human}, n1=3, n2=4, p=0.05 (one-tailed, Paper Table 1)"
    
    # Agent test: Calculate median rates per session
    def calculate_session_median_rates(agent_data_list, participants_per_session=16):
        """Calculate median rate of decrease for each simulated session"""
        rates = []
        for i in range(0, len(agent_data_list), participants_per_session):
            session_data = agent_data_list[i:i+participants_per_session]
            round1_values = [d["Q1"] for d in session_data if "Q1" in d and d["Q1"] is not None]
            round4_values = [d["Q4"] for d in session_data if "Q4" in d and d["Q4"] is not None]
            
            if len(round1_values) >= 3 and len(round4_values) >= 3:
                median_r1 = np.median(round1_values)
                median_r4 = np.median(round4_values)
                if median_r1 > 0:
                    rate = (median_r1 - median_r4) / median_r1
                    rates.append(rate)
        return rates
    
    median_rates_p_05 = calculate_session_median_rates(agent_data["p_0.5_condition"])
    median_rates_p_066 = calculate_session_median_rates(agent_data["p_0.66_condition"])
    
    p_val_agent_f4 = None
    u_stat_agent_f4 = None
    if len(median_rates_p_05) >= 2 and len(median_rates_p_066) >= 2:
        u_stat_agent_f4, p_val_agent_f4 = stats.mannwhitneyu(median_rates_p_05, median_rates_p_066, alternative='two-sided')
        bf_agent = calc_bf_mannwhitneyu(u_stat_agent_f4, len(median_rates_p_05), len(median_rates_p_066))
        # Direction: expect median rate(0.5) > median rate(0.66)
        a_dir = 1 if np.median(median_rates_p_05) > np.median(median_rates_p_066) else -1
        pi_a_f4 = calc_posteriors_3way(bf_agent, a_dir)
        reason_f4 = f"Mann-Whitney U (median rates): U={u_stat_agent_f4:.2f}, p={p_val_agent_f4:.4f}, rates_p_05={[f'{r:.2f}' for r in median_rates_p_05]}, rates_p_066={[f'{r:.2f}' for r in median_rates_p_066]}"
        agent_stat_str = f"{u_stat_agent_f4:.2f}"
    else:
        pi_a_f4 = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        reason_f4 = f"Insufficient session data: n1={len(median_rates_p_05)}, n2={len(median_rates_p_066)}"
        agent_stat_str = ""

    test_result_f4 = {
        "study_id": "Experiment 1",
        "sub_study_id": "p_0.5_vs_p_0.66",
        "finding_id": "F4",
        "test_name": "Rate Comparison (Convergence Speed)",
        "scenario": "Median rate of decrease comparison",
        "pi_human": float(pi_h_f4['pi_plus'] + pi_h_f4['pi_minus']),
        "pi_agent": float(pi_a_f4['pi_plus'] + pi_a_f4['pi_minus']),
        "pi_human_3way": pi_h_f4,
        "pi_agent_3way": pi_a_f4,
        "pas": float(calc_pas(pi_h_f4, pi_a_f4)),
        "pi_human_source": human_source_f4,
        "agent_reason": reason_f4,
        "statistical_test_type": "mannwhitneyu",
        "human_test_statistic": f"{u_stat_human:.2f}",
        "agent_test_statistic": agent_stat_str
    }
    # Get test_gt for F4
    test_gt_f4 = {}
    for study in ground_truth.get("studies", []):
        for finding in study.get("findings", []):
            if finding.get("finding_id") == "F4":
                statistical_tests = finding.get("statistical_tests", [])
                if statistical_tests:
                    test_gt_f4 = statistical_tests[0]
                break
    add_statistical_replication_fields(
        test_result_f4, test_gt_f4, p_val_agent_f4, u_stat_agent_f4, "mannwhitneyu",
        n_agent=len(median_rates_p_05), n2_agent=len(median_rates_p_066),
        n_human=3, n2_human=4
    )
    test_results.append(test_result_f4)

    # --- Finding 5: Reference Point (p × mean_{t-1}) ---
    # Paper: Nagel (1995), p. 9, Footnote 15
    # Two hypotheses: Weak (choices < mean_{t-1}) and Strong (choices < p × mean_{t-1} in Period 4)
    f5 = next(f for f in findings_gt if f["finding_id"] == "F5")
    f5_data = f5.get("original_data_points", {}).get("data", {})
    
    # Test F5 Weak Hypothesis: choices < mean_{t-1} (Periods 2-4)
    weak_hyp_data = f5_data.get("weak_hypothesis", {})
    p_value_weak = weak_hyp_data.get("p_value", 0.01)
    # Use p-value to estimate Pi_Human (as requested)
    bf_f5_weak = calc_bf_from_p(p_value_weak, n=100, effect_direction_match=True)  # n is not used in calc_bf_from_p
    pi_h_f5_weak = calc_posteriors_3way(bf_f5_weak, 1, prior_odds=10.0)
    human_source_f5_weak = f"Binomial Test (Weak Hypothesis): p={p_value_weak}, choices < mean_{t-1} significant at 1% level in all sessions and periods (Paper p. 9, Footnote 15)"
    
    # Test F5 Strong Hypothesis: choices < p × mean_{t-1} (Period 4 only, session-level)
    strong_hyp_data = f5_data.get("strong_hypothesis", {})
    p_value_strong = strong_hyp_data.get("p_value", 0.01)
    sessions_rejecting = strong_hyp_data.get("sessions_rejecting", 6)
    total_sessions = strong_hyp_data.get("total_sessions", 7)
    # Use p-value to estimate Pi_Human (as requested)
    bf_f5_strong = calc_bf_from_p(p_value_strong, n=total_sessions, effect_direction_match=True)
    pi_h_f5_strong = calc_posteriors_3way(bf_f5_strong, 1, prior_odds=10.0)
    human_source_f5_strong = f"Binomial Test (Strong Hypothesis - Period 4): p={p_value_strong}, {sessions_rejecting}/{total_sessions} sessions reject at 1% level (Paper p. 9, Footnote 15)"
    
    # Agent test: Use real session data from round_feedbacks
    for sub_id in ["p_0.5_condition", "p_0.66_condition"]:
        p_val = 0.5 if sub_id == "p_0.5_condition" else 0.66
        
        # Weak Hypothesis: choices < mean_{t-1} in Periods 2-4
        below_mean_count = 0
        total_choices_weak = 0
        
        # Strong Hypothesis: choices < p × mean_{t-1} in Period 4 (session-level)
        sessions_below_p_mean = 0
        total_sessions_f5 = 0
        
        for session in sessions[sub_id]:
            # Extract round data
            round_data = {}
            for r in range(1, 5):
                round_key = f"round_{r}"
                if round_key in session:
                    round_data[r] = session[round_key]
            
            # Weak Hypothesis: Periods 2-4
            for t in range(2, 5):  # Periods 2, 3, 4
                if t in round_data and (t-1) in round_data:
                    mean_prev = round_data[t-1]["mean"]
                    choices = round_data[t]["choices"]
                    
                    for pid, choice_val in choices.items():
                        if choice_val is not None:
                            total_choices_weak += 1
                            if choice_val < mean_prev:
                                below_mean_count += 1
            
            # Strong Hypothesis: Period 4 only, session-level
            if 4 in round_data and 3 in round_data:
                mean_3 = round_data[3]["mean"]
                target = p_val * mean_3
                choices_4 = round_data[4]["choices"]
                
                # Count choices < p × mean_{t-1} in this session
                below_p_mean_count = 0
                total_choices_session = 0
                for pid, choice_val in choices_4.items():
                    if choice_val is not None:
                        total_choices_session += 1
                        if choice_val < target:
                            below_p_mean_count += 1
                
                if total_choices_session > 0:
                    total_sessions_f5 += 1
                    # Test if this session rejects H0: choices < p × mean (i.e., most choices are > p × mean)
                    # If below_p_mean_count / total < 0.5, then most choices are > p × mean, so we reject H0
                    from scipy.stats import binomtest
                    binom_result = binomtest(below_p_mean_count, total_choices_session, 0.5, alternative='less')
                    if binom_result.pvalue < 0.01:  # Reject at 1% level
                        sessions_below_p_mean += 1
        
        # Weak Hypothesis result
        if total_choices_weak >= 10:
            bf_agent_weak = calc_bf_binomial(below_mean_count, total_choices_weak, 0.5)
            pi_a_f5_weak = calc_posteriors_3way(bf_agent_weak, 1)
            success_rate_weak = below_mean_count / total_choices_weak if total_choices_weak > 0 else 0
            from scipy.stats import binomtest
            binom_result_weak = binomtest(below_mean_count, total_choices_weak, 0.5, alternative='greater')
            p_val_binom_weak = binom_result_weak.pvalue
            reason_f5_weak = f"Binomial Test (Weak): k={below_mean_count}, n={total_choices_weak}, success_rate={success_rate_weak:.2%}, p={p_val_binom_weak:.4f}"
            agent_stat_str_weak = f"{below_mean_count}/{total_choices_weak}"
        else:
            pi_a_f5_weak = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
            reason_f5_weak = f"Insufficient data: n={total_choices_weak}"
            agent_stat_str_weak = ""
        
        test_result_f5_weak = {
            "study_id": "Experiment 1",
            "sub_study_id": sub_id,
            "finding_id": "F5",
            "test_name": "Reference Point (Weak: choices < mean_{t-1})",
            "scenario": "Periods 2-4: choices < mean_{t-1}",
            "pi_human": float(pi_h_f5_weak['pi_plus'] + pi_h_f5_weak['pi_minus']),
            "pi_agent": float(pi_a_f5_weak['pi_plus'] + pi_a_f5_weak['pi_minus']),
            "pi_human_3way": pi_h_f5_weak,
            "pi_agent_3way": pi_a_f5_weak,
            "pas": float(calc_pas(pi_h_f5_weak, pi_a_f5_weak)),
            "pi_human_source": human_source_f5_weak,
            "agent_reason": reason_f5_weak,
            "statistical_test_type": "binomial",
            "human_test_statistic": "",
            "agent_test_statistic": agent_stat_str_weak
        }
        # Get test_gt for F5
        test_gt_f5 = {}
        for study in ground_truth.get("studies", []):
            for finding in study.get("findings", []):
                if finding.get("finding_id") == "F5":
                    statistical_tests = finding.get("statistical_tests", [])
                    if statistical_tests:
                        test_gt_f5 = statistical_tests[0]
                    break
        add_statistical_replication_fields(test_result_f5_weak, test_gt_f5, p_val_binom_weak if 'p_val_binom_weak' in locals() else None, below_mean_count if 'below_mean_count' in locals() else None, "binomial")
        test_results.append(test_result_f5_weak)
        
        # Strong Hypothesis result (Period 4, session-level)
        if total_sessions_f5 >= 3:
            # Session-level binomial: k sessions reject, n total sessions
            bf_agent_strong = calc_bf_binomial(sessions_below_p_mean, total_sessions_f5, 0.5)
            pi_a_f5_strong = calc_posteriors_3way(bf_agent_strong, 1)
            from scipy.stats import binomtest
            binom_result_strong = binomtest(sessions_below_p_mean, total_sessions_f5, 0.5, alternative='greater')
            p_val_binom_strong = binom_result_strong.pvalue
            reason_f5_strong = f"Binomial Test (Strong - Period 4, session-level): k={sessions_below_p_mean}, n={total_sessions_f5} sessions, p={p_val_binom_strong:.4f}"
            agent_stat_str_strong = f"{sessions_below_p_mean}/{total_sessions_f5}"
        else:
            pi_a_f5_strong = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
            reason_f5_strong = f"Insufficient sessions: n={total_sessions_f5}"
            agent_stat_str_strong = ""
        
        test_result_f5_strong = {
            "study_id": "Experiment 1",
            "sub_study_id": sub_id,
            "finding_id": "F5",
            "test_name": "Reference Point (Strong: choices < p × mean_{t-1}, Period 4)",
            "scenario": "Period 4: session-level rejection of choices < p × mean_{t-1}",
            "pi_human": float(pi_h_f5_strong['pi_plus'] + pi_h_f5_strong['pi_minus']),
            "pi_agent": float(pi_a_f5_strong['pi_plus'] + pi_a_f5_strong['pi_minus']),
            "pi_human_3way": pi_h_f5_strong,
            "pi_agent_3way": pi_a_f5_strong,
            "pas": float(calc_pas(pi_h_f5_strong, pi_a_f5_strong)),
            "pi_human_source": human_source_f5_strong,
            "agent_reason": reason_f5_strong,
            "statistical_test_type": "binomial",
            "human_test_statistic": "",
            "agent_test_statistic": agent_stat_str_strong
        }
        add_statistical_replication_fields(test_result_f5_strong, test_gt_f5, p_val_binom_strong if 'p_val_binom_strong' in locals() else None, sessions_below_p_mean if 'sessions_below_p_mean' in locals() else None, "binomial")
        test_results.append(test_result_f5_strong)

    # --- Finding 6: Learning Direction Theory (LDT) ---
    # Paper: Nagel (1995), p. 12
    # Statistical test: Session-level Binomial Test (k=7, n=7)
    # Only for p<1 conditions (Sessions 1-7), exclude winners
    f6 = next(f for f in findings_gt if f["finding_id"] == "F6")
    f6_data = f6.get("original_data_points", {}).get("data", {})
    
    # Human data: Session-level test (k=7, n=7, p0=0.5)
    session_level_data = f6_data.get("session_level_results", {})
    test_a_data = session_level_data.get("test_a", {})
    human_k_f6 = test_a_data.get("successes", 7)
    human_n_f6 = test_a_data.get("total_sessions", 7)
    # Use actual k and n for binomial test
    bf_f6 = calc_bf_binomial(human_k_f6, human_n_f6, 0.5)
    pi_h_f6 = calc_posteriors_3way(bf_f6, 1, prior_odds=10.0)
    human_source_f6 = f"Binomial Test (LDT, Session-level): k={human_k_f6}, n={human_n_f6}, all sessions > 50% (Paper p. 12)"
    
    # Agent test: Only p<1 conditions (Sessions 1-7)
    # For each session, calculate LDT alignment rate (excluding winners)
    # Then count how many sessions have alignment > 50%
    for sub_id in ["p_0.5_condition", "p_0.66_condition"]:
        p_val = 0.5 if sub_id == "p_0.5_condition" else 0.66
        
        sessions_with_ldt_above_50 = 0
        total_sessions_f6 = 0
        
        for session in sessions[sub_id]:
            # Extract round data
            round_data = {}
            for r in range(1, 5):
                round_key = f"round_{r}"
                if round_key in session:
                    round_data[r] = session[round_key]
            
            # Calculate LDT alignment for this session (excluding winners)
            correct_ldt = 0
            total_transitions_session = 0
            
            # Check transitions: R1->R2, R2->R3, R3->R4
            for t in range(1, 4):  # t = 1, 2, 3
                if t in round_data and (t+1) in round_data:
                    round_t = round_data[t]
                    round_t1 = round_data[t+1]
                    mean_t = round_t["mean"]
                    choices_t = round_t["choices"]
                    choices_t1 = round_t1["choices"]
                    winning_number_t = round_t["winning_number"]
                    
                    if mean_t > 0:
                        target_t = p_val * mean_t  # Optimal target
                        
                        # Check each participant's transition (excluding winners)
                        for pid in choices_t.keys():
                            if pid in choices_t1:
                                choice_t = choices_t[pid]
                                choice_t1 = choices_t1[pid]
                                
                                # EXCLUDE WINNERS: if this participant won in round t, skip
                                if choice_t is not None and abs(choice_t - winning_number_t) < 1e-6:
                                    continue  # Skip winners
                                
                                if choice_t is not None and choice_t1 is not None:
                                    total_transitions_session += 1
                                    
                                    # LDT: Check if adjustment direction is correct
                                    if choice_t > target_t:
                                        # Should decrease
                                        if choice_t1 < choice_t:
                                            correct_ldt += 1
                                    elif choice_t < target_t:
                                        # Should increase
                                        if choice_t1 > choice_t:
                                            correct_ldt += 1
            
            # Check if this session has LDT alignment > 50%
            if total_transitions_session >= 3:
                total_sessions_f6 += 1
                alignment_rate = correct_ldt / total_transitions_session if total_transitions_session > 0 else 0
                if alignment_rate > 0.5:
                    sessions_with_ldt_above_50 += 1
        
        # Session-level binomial test: k sessions with alignment > 50%, n total sessions
        if total_sessions_f6 >= 3:
            bf_agent = calc_bf_binomial(sessions_with_ldt_above_50, total_sessions_f6, 0.5)
            pi_a_f6 = calc_posteriors_3way(bf_agent, 1)
            from scipy.stats import binomtest
            binom_result = binomtest(sessions_with_ldt_above_50, total_sessions_f6, 0.5, alternative='greater')
            p_val_binom = binom_result.pvalue
            reason_f6 = f"Binomial Test (LDT, Session-level): k={sessions_with_ldt_above_50}, n={total_sessions_f6} sessions, p={p_val_binom:.4f}"
            agent_stat_str = f"{sessions_with_ldt_above_50}/{total_sessions_f6}"
        else:
            pi_a_f6 = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
            reason_f6 = f"Insufficient sessions: n={total_sessions_f6}"
            agent_stat_str = ""
        
        test_result_f6 = {
            "study_id": "Experiment 1",
            "sub_study_id": sub_id,
            "finding_id": "F6",
            "test_name": "Learning Direction Theory (LDT, Session-level)",
            "scenario": "Session-level: sessions with LDT alignment > 50% (excluding winners)",
            "pi_human": float(pi_h_f6['pi_plus'] + pi_h_f6['pi_minus']),
            "pi_agent": float(pi_a_f6['pi_plus'] + pi_a_f6['pi_minus']),
            "pi_human_3way": pi_h_f6,
            "pi_agent_3way": pi_a_f6,
            "pas": float(calc_pas(pi_h_f6, pi_a_f6)),
            "pi_human_source": human_source_f6,
            "agent_reason": reason_f6,
            "statistical_test_type": "binomial",
            "human_test_statistic": "",  # Binomial test doesn't have a single test statistic
            "agent_test_statistic": agent_stat_str,
            "human_k": human_k_f6,
            "human_p0": 0.5  # p0 = 0.5 for LDT test
        }
        # Get test_gt for F6
        test_gt_f6 = {}
        for study in ground_truth.get("studies", []):
            for finding in study.get("findings", []):
                if finding.get("finding_id") == "F6":
                    statistical_tests = finding.get("statistical_tests", [])
                    if statistical_tests:
                        test_gt_f6 = statistical_tests[0]
                    break
        add_statistical_replication_fields(test_result_f6, test_gt_f6, p_val_binom if 'p_val_binom' in locals() else None, sessions_with_ldt_above_50 if 'sessions_with_ldt_above_50' in locals() else None, "binomial")
        test_results.append(test_result_f6)

    # 4. Two-level Weighted Aggregation
    # Add test_weight to each test result
    for tr in test_results:
        finding_id = tr.get("finding_id")
        test_name = tr.get("test_name", "")
        tr["test_weight"] = float(test_weights.get((finding_id, test_name), 1.0))
    
    # Level 1: Aggregate tests into findings (weighted by test weights)
    finding_results = []
    for fid in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        relevant_tests = [t for t in test_results if t["finding_id"] == fid]
        if relevant_tests:
            # Weighted average: Σ (PAS * weight) / Σ weights
            total_weighted_pas = sum(t["pas"] * t.get("test_weight", 1.0) for t in relevant_tests)
            total_weight = sum(t.get("test_weight", 1.0) for t in relevant_tests)
            finding_score = total_weighted_pas / total_weight if total_weight > 0 else 0.5
            
            finding_weight = finding_weights.get(fid, 1.0)
            finding_results.append({
                "sub_study_id": "Multiple",
                "finding_id": fid,
                "finding_score": float(finding_score),
                "finding_weight": float(finding_weight),
                "n_tests": len(relevant_tests)
            })

    # Level 2: Aggregate findings into study score (weighted by finding weights)
    total_weighted_finding_score = sum(fr["finding_score"] * fr["finding_weight"] for fr in finding_results)
    total_finding_weight = sum(fr["finding_weight"] for fr in finding_results)
    final_score = total_weighted_finding_score / total_finding_weight if total_finding_weight > 0 else 0.5

    # substudy_results removed - using two-level aggregation (Tests -> Findings -> Study)
    substudy_results = []

    return {
        "score": float(final_score),
        "substudy_results": substudy_results,
        "finding_results": finding_results,
        "test_results": test_results
    }