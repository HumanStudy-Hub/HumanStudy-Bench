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
    calc_bf_anova, 
    calc_bf_variance_f,
    calc_posteriors_3way,
    calc_pas,
    parse_p_value_from_reported,
    get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """Parse Qk=<value> or Qk: <value> or Qk.n=<value> format from response text."""
    results = {}
    # Supports both = and : separators
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*[:=]\s*([^,\n\s]+)")
    for k, v in pattern.findall(response_text):
        results[k.strip()] = v.strip()
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_008: 从items中提取q_idx_choice
    """
    required = set()
    items = trial_info.get("items", [])
    
    for item in items:
        q_idx = item.get("q_idx_choice")
        if q_idx:
            # 如果q_idx已经包含"Q"前缀，直接使用；否则添加"Q"前缀
            if isinstance(q_idx, str) and q_idx.startswith("Q"):
                required.add(q_idx)
            else:
                required.add(f"Q{q_idx}")
        # 如果没有q_idx_choice，使用索引推断
        elif not required:
            for idx, _ in enumerate(items):
                required.add(f"Q{idx + 1}")
            break
    
    return required

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates agent's Bayesian Alignment Score (PAS) for Study 008.
    Study 008: Pluralistic Ignorance (Prentice & Miller, 1993).
    """
    # 1. Load ground truth and metadata
    study_id = "study_008"
    study_dir = Path(__file__).resolve().parent.parent / "source"
    gt_path = study_dir / "ground_truth.json"
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Load metadata for finding and test weights
    metadata = {}
    metadata_path = study_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
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

    # Cache for materials files to avoid reloading
    materials_cache = {}
    
    def load_materials(sub_study_id: str) -> List[Dict[str, Any]]:
        """Load items from materials file if not in cache."""
        if sub_study_id not in materials_cache:
            materials_path = study_dir / "materials" / f"{sub_study_id}.json"
            if materials_path.exists():
                try:
                    with open(materials_path, 'r') as f:
                        material_data = json.load(f)
                        materials_cache[sub_study_id] = material_data.get("items", [])
                except Exception:
                    materials_cache[sub_study_id] = []
            else:
                materials_cache[sub_study_id] = []
        return materials_cache[sub_study_id]

    # 2. Extract agent data
    # We organize data by sub-study for specific analysis
    agent_data = {
        "study_1_comfort_estimation": [],
        "study_2_order_and_friend_comparison": [],
        "study_4_keg_ban_alienation": []
    }

    for participant in results.get("individual_data", []):
        # Get gender from participant profile, with fallback to trial_info profile
        gender_raw = participant.get("profile", {}).get("gender", "Unknown")
        # Normalize gender to capitalized format
        if gender_raw and isinstance(gender_raw, str):
            gender = gender_raw.capitalize() if gender_raw.lower() in ["male", "female"] else gender_raw
        else:
            gender = "Unknown"
        
        for response in participant.get("responses", []):
            response_text = response.get("response_text", "")
            trial_info = response.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            items = trial_info.get("items", [])
            
            # If items are missing from trial_info, load from materials file
            if not items and sub_id:
                items = load_materials(sub_id)
            
            # Use the extraction logic from the config or parse manually
            # The config maps response_text to item IDs (q1, q2...)
            parsed = parse_agent_responses(response_text)
            
            entry = {"gender": gender}
            # Map items based on the metadata provided in trial_info
            items_extracted = 0
            for item in items:
                item_id = item.get("id")
                # Find which Qx corresponds to this item_id in the prompt
                # Based on CustomPromptBuilder: Q1 is the first item, Q2 is second...
                q_idx_in_prompt = None
                for i, itm in enumerate(items):
                    if itm.get("id") == item_id:
                        q_idx_in_prompt = f"Q{i+1}"
                        break
                
                if q_idx_in_prompt and q_idx_in_prompt in parsed:
                    try:
                        entry[item_id] = float(parsed[q_idx_in_prompt])
                        items_extracted += 1
                    except (ValueError, TypeError):
                        continue
            
            # Only append entries that have at least one valid item extracted
            if sub_id in agent_data and items_extracted > 0:
                # Add order condition for Study 2
                if sub_id == "study_2_order_and_friend_comparison":
                    entry["order"] = trial_info.get("order_condition")
                agent_data[sub_id].append(entry)

    test_results = []

    # 3. Process each Finding in Ground Truth
    for study_gt in ground_truth["studies"]:
        sub_study_id_map = {
            "Study 1": "study_1_comfort_estimation",
            "Study 2": "study_2_order_and_friend_comparison",
            "Study 4": "study_4_keg_ban_alienation"
        }
        sub_id = sub_study_id_map.get(study_gt["study_id"])
        data_subset = agent_data.get(sub_id, [])

        for finding in study_gt["findings"]:
            for test in finding["statistical_tests"]:
                reported = test["reported_statistics"]
                pi_h = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                reason = "Insufficient data"

                # Initialize statistics variables
                p_val_agent = None
                t_stat_agent = None
                f_stat_agent = None
                f_val = None
                
                # --- Calculate pi_human ---
                try:
                    if "F(" in reported:
                        # Extract F(df1, df2) = value
                        match = re.search(r"F\((\d+),\s*(\d+)\)\s*=\s*([\d\.]+)", reported)
                        if match:
                            df1, df2, f_val = int(match.group(1)), int(match.group(2)), float(match.group(3))
                            # Check if this is a variance F-test
                            if "variance" in test["test_name"].lower() or "var" in test["test_name"].lower():
                                # Use variance F-test calculator
                                bf_h = calc_bf_variance_f(f_val, df1, df2)
                                pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                            else:
                                # Use ANOVA F-test calculator
                                bf_h = calc_bf_anova(f_val, df1, df2, df1 + df2 + 1)
                                pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                except Exception:
                    pi_h = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                # --- Calculate pi_agent ---
                if not data_subset:
                    reason = "No agent data for this sub-study"
                else:
                    try:
                        # Study 1: Alcohol Comfort
                        if sub_id == "study_1_comfort_estimation":
                            if finding["finding_id"] == "F1":
                                test_name_lower = test["test_name"].lower()
                                if "target" in test_name_lower and "interaction" not in test_name_lower:
                                    # Main effect of Target (Self vs Average)
                                    s_vals = [d["q1"] for d in data_subset if "q1" in d and "q2" in d]
                                    a_vals = [d["q2"] for d in data_subset if "q1" in d and "q2" in d]
                                    if len(s_vals) > 5:
                                        t_stat_agent, p_val_agent = stats.ttest_rel(a_vals, s_vals)
                                        bf_a = calc_bf_t(abs(t_stat_agent), len(s_vals), independent=False)
                                        a_dir = 1 if t_stat_agent > 0 else -1
                                        pi_a = calc_posteriors_3way(bf_a, a_dir)
                                        reason = f"t_rel={t_stat_agent:.2f}, n={len(s_vals)}"
                                    else:
                                        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                                        reason = f"Insufficient data: only {len(s_vals)} valid pairs"
                                elif "interaction" in test["test_name"].lower():
                                    # Sex x Target Interaction
                                    diff_m = [d["q2"] - d["q1"] for d in data_subset if d.get("gender", "").lower() in ["male", "m"] and "q1" in d and "q2" in d]
                                    diff_f = [d["q2"] - d["q1"] for d in data_subset if d.get("gender", "").lower() in ["female", "f"] and "q1" in d and "q2" in d]
                                    if len(diff_m) > 2 and len(diff_f) > 2:
                                        t_stat_agent, p_val_agent = stats.ttest_ind(diff_m, diff_f)
                                        bf_a = calc_bf_t(t_stat_agent, len(diff_m), len(diff_f))
                                        pi_a = calc_posteriors_3way(bf_a, 0)
                                        reason = f"t_ind={t_stat_agent:.2f}, n_m={len(diff_m)}, n_f={len(diff_f)}"
                                    else:
                                        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                                        reason = f"Insufficient data: n_m={len(diff_m)}, n_f={len(diff_f)}"
                            
                            elif finding["finding_id"] == "F2":
                                # Variance comparison
                                s_vals = [d["q1"] for d in data_subset if "q1" in d]
                                a_vals = [d["q2"] for d in data_subset if "q2" in d]
                                if len(s_vals) > 5 and len(a_vals) > 5:
                                    var_s = np.var(s_vals, ddof=1)
                                    var_a = np.var(a_vals, ddof=1)
                                    f_stat = var_s / var_a if var_a != 0 else 1.0
                                    df1_var = len(s_vals) - 1
                                    df2_var = len(a_vals) - 1
                                    bf_a = calc_bf_variance_f(f_stat, df1_var, df2_var)
                                    a_dir = 1 if f_stat > 1 else -1
                                    pi_a = calc_posteriors_3way(bf_a, a_dir)
                                    reason = f"F_var={f_stat:.2f}, n1={len(s_vals)}, n2={len(a_vals)}"
                                else:
                                    pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                                    reason = f"Insufficient data: n1={len(s_vals)}, n2={len(a_vals)}"
                        
                        # Study 2: Friends and Order
                        elif sub_id == "study_2_order_and_friend_comparison":
                            if "interaction" not in test["test_name"].lower():
                                # Main effect of target (Self, Friend, Average)
                                s = [d["q1"] for d in data_subset if all(k in d for k in ["q1", "q2", "q3"])]
                                a = [d["q2"] for d in data_subset if all(k in d for k in ["q1", "q2", "q3"])]
                                f = [d["q3"] for d in data_subset if all(k in d for k in ["q1", "q2", "q3"])]
                                if len(s) > 5:
                                    f_stat_agent, p_val_agent = stats.f_oneway(s, a, f)
                                    m_s, m_f, m_a = np.mean(s), np.mean(f), np.mean(a)
                                    direction_match = (m_s < m_f) and (m_f < m_a)
                                    bf_a = calc_bf_anova(abs(f_stat_agent), 2, 3*len(s)-3, 3*len(s))
                                    a_dir = 1 if direction_match else -1
                                    pi_a = calc_posteriors_3way(bf_a, a_dir)
                                    reason = f"F={abs(f_stat_agent):.2f}, n_total={3*len(s)} [Means: Self={m_s:.2f}, Friend={m_f:.2f}, Avg={m_a:.2f}]"
                                else:
                                    pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                                    reason = f"Insufficient data: n={len(s)}"
                            else:
                                # Target x Order Interaction
                                diff_self_first = [d["q2"] - d["q1"] for d in data_subset if d.get("order") == "self_first" and "q1" in d and "q2" in d]
                                diff_avg_first = [d["q2"] - d["q1"] for d in data_subset if d.get("order") == "average_first" and "q1" in d and "q2" in d]
                                if len(diff_self_first) > 2 and len(diff_avg_first) > 2:
                                    t_stat_agent, p_val_agent = stats.ttest_ind(diff_self_first, diff_avg_first)
                                    bf_a = calc_bf_t(t_stat_agent, len(diff_self_first), len(diff_avg_first))
                                    pi_a = calc_posteriors_3way(bf_a, 0)
                                    reason = f"t_interaction={t_stat_agent:.2f}, n1={len(diff_self_first)}, n2={len(diff_avg_first)}"
                                else:
                                    pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                                    reason = f"Insufficient data: n1={len(diff_self_first)}, n2={len(diff_avg_first)}"

                        # Study 4: Keg Ban Alienation
                        elif sub_id == "study_4_keg_ban_alienation":
                            test_name_lower = test["test_name"].lower()
                            dv_map = {"signatures": "q3", "hours": "q4", "reunions": "q5"}
                            dv_key = next((key for keyword, key in dv_map.items() if keyword in test_name_lower), None)
                            if dv_key:
                                data_valid = [d for d in data_subset if "q1" in d and "q2" in d and dv_key in d]
                                if len(data_valid) > 10:
                                    deviance = [abs(d["q1"] - d["q2"]) for d in data_valid]
                                    med = np.median(deviance)
                                    group_low = [d[dv_key] for i, d in enumerate(data_valid) if deviance[i] <= med]
                                    group_high = [d[dv_key] for i, d in enumerate(data_valid) if deviance[i] > med]
                                    if len(group_low) > 2 and len(group_high) > 2:
                                        t_stat_agent, p_val_agent = stats.ttest_ind(group_low, group_high)
                                        bf_a = calc_bf_t(abs(t_stat_agent), len(group_low), len(group_high))
                                        a_dir = 1 if t_stat_agent > 0 else -1
                                        pi_a = calc_posteriors_3way(bf_a, a_dir)
                                        reason = f"t_deviance={t_stat_agent:.2f}, n_low={len(group_low)}, n_high={len(group_high)}"
                                    else:
                                        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                                        reason = f"Insufficient data: n_low={len(group_low)}, n_high={len(group_high)}"
                                else:
                                    pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                                    reason = f"Insufficient data: n_valid={len(data_valid)}"
                            else:
                                reason = f"Could not identify DV from test name: {test['test_name']}"

                    except Exception as e:
                        reason = f"Error in calculation: {str(e)}"
                        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                # Calculate PAS
                pas = float(calc_pas(pi_h, pi_a))
                
                # Determine test type and statistic
                test_type = "t-test"
                test_stat = t_stat_agent
                if f_stat_agent is not None:
                    test_type = "f-test"
                    test_stat = f_stat_agent
                
                # Create test result dict
                test_result = {
                    "study_id": study_gt["study_id"],
                    "sub_study_id": sub_id,
                    "finding_id": finding["finding_id"],
                    "test_name": test["test_name"],
                    "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
                    "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
                    "pi_human_3way": pi_h,
                    "pi_agent_3way": pi_a,
                    "pas": pas,
                    "pi_human_source": reported,
                    "agent_reason": reason,
                    "statistical_test_type": test_type,
                    "human_test_statistic": f"{f_val:.2f}" if f_val is not None else "",
                    "agent_test_statistic": f"{test_stat:.2f}" if test_stat is not None else ""
                }

                # Extract sample sizes for t-tests
                n_agent = None
                n2_agent = None
                n_human = None
                n2_human = None
                independent = True
                
                if test_type == "t-test" and t_stat_agent is not None:
                    # Extract sample sizes based on which branch was executed
                    if 'diff_m' in locals() and 'diff_f' in locals() and len(diff_m) > 2 and len(diff_f) > 2:
                        n_agent = len(diff_m)
                        n2_agent = len(diff_f)
                        # Human: approximate from reported statistics (typically balanced groups)
                        n_human = 63  # Men n=63, Women n=69 in GT
                        n2_human = 69
                        independent = True
                    elif 's_vals' in locals() and 'a_vals' in locals() and len(s_vals) > 5:
                        n_agent = len(s_vals)
                        n_human = 132  # Total n in GT
                        independent = False
                    elif 'diff_self_first' in locals() and 'diff_avg_first' in locals() and len(diff_self_first) > 2 and len(diff_avg_first) > 2:
                        n_agent = len(diff_self_first)
                        n2_agent = len(diff_avg_first)
                        n_human = 30  # Approximate
                        n2_human = 30
                        independent = True
                    elif 'group_low' in locals() and 'group_high' in locals() and len(group_low) > 2 and len(group_high) > 2:
                        n_agent = len(group_low)
                        n2_agent = len(group_high)
                        n_human = 45  # Total n=90/91 in GT
                        n2_human = 45
                        independent = True
                
                # Add statistical replication fields
                if test_type == "t-test":
                    add_statistical_replication_fields(
                        test_result, test, p_val_agent, test_stat, test_type,
                        n_agent=n_agent,
                        n2_agent=n2_agent,
                        n_human=n_human,
                        n2_human=n2_human,
                        independent=independent
                    )
                else:
                    # F-test - not supported yet for effect sizes
                    add_statistical_replication_fields(test_result, test, p_val_agent, test_stat, test_type)
                
                test_results.append(test_result)

    # 4. Two-level Weighted Aggregation
    # Add test_weight to each test result
    for tr in test_results:
        finding_id = tr.get("finding_id")
        test_name = tr.get("test_name", "")
        tr["test_weight"] = float(test_weights.get((finding_id, test_name), 1.0))
    
    # Level 1: Aggregate tests into findings (weighted by test weights)
    finding_results = []
    # Group by (sub_study_id, finding_id) pairs
    finding_groups = {}
    for t in test_results:
        key = (t["sub_study_id"], t["finding_id"])
        if key not in finding_groups:
            finding_groups[key] = []
        finding_groups[key].append(t)
    
    for (sub_id_str, finding_id_str), tests in finding_groups.items():
        if tests:  # Only compute mean if there are tests
            # Weighted average: Σ (PAS * weight) / Σ weights
            total_weighted_pas = sum(t["pas"] * t.get("test_weight", 1.0) for t in tests)
            total_weight = sum(t.get("test_weight", 1.0) for t in tests)
            finding_score = total_weighted_pas / total_weight if total_weight > 0 else 0.5
            
            finding_weight = finding_weights.get(finding_id_str, 1.0)
            finding_results.append({
                "sub_study_id": sub_id_str,
                "finding_id": finding_id_str,
                "finding_score": float(finding_score),
                "finding_weight": float(finding_weight),
                "n_tests": len(tests)
            })

    # Level 2: Aggregate findings into study score (weighted by finding weights)
    total_weighted_finding_score = sum(fr["finding_score"] * fr["finding_weight"] for fr in finding_results)
    total_finding_weight = sum(fr["finding_weight"] for fr in finding_results)
    overall_score = total_weighted_finding_score / total_finding_weight if total_finding_weight > 0 else 0.5

    # substudy_results removed - using two-level aggregation (Tests -> Findings -> Study)
    substudy_results = []

    return {
        "score": overall_score,
        "substudy_results": substudy_results,
        "finding_results": finding_results,
        "test_results": test_results
    }