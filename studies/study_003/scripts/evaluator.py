import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from stats_lib import (
    calc_bf_chisq, calc_posteriors_3way, calc_pas, chi2_contingency_safe,
    parse_p_value_from_reported, get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """Parse Qk=<value> or Qk: <value> or Qk.n=<value> format from agent response."""
    results = {}
    # Supports both = and : separators
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*[:=]\s*([^,\n\r]+)")
    for k, v in pattern.findall(response_text):
        results[k.strip()] = v.strip()
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_003: 从items中提取q_idx_choice
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
        # 如果没有q_idx_choice，使用索引推断（Q1, Q2, ...）
        elif not required:
            for idx, _ in enumerate(items):
                required.add(f"Q{idx + 1}")
            break
    
    return required

def get_clean_choice(text: str) -> str:
    """Standardize choice labels for comparison."""
    if not text:
        return ""
    text = text.upper()
    if "A & D" in text: return "AD"
    if "B & C" in text: return "BC"
    if "YES" in text: return "YES"
    if "NO" in text: return "NO"
    if "OPTION A" in text or "PROGRAM A" in text: return "A"
    if "OPTION B" in text or "PROGRAM B" in text: return "B"
    if "OPTION C" in text or "PROGRAM C" in text: return "C"
    if "OPTION D" in text or "PROGRAM D" in text: return "D"
    if "OPTION E" in text or "PROGRAM E" in text: return "E"
    if "OPTION F" in text or "PROGRAM F" in text: return "F"
    
    # Fallback to single characters
    for char in "ABCDEF":
        if text.startswith(char) or text.endswith(char):
            return char
    return text

def safe_chi2_contingency(table):
    """
    Safely compute chi-square test with proper error handling for zero expected frequencies.
    
    Args:
        table: 2D array-like contingency table
        
    Returns:
        chi2 statistic (float), or 0.0 if test cannot be computed
    """
    obs = np.array(table)
    # Check for zero total
    if np.sum(obs) == 0:
        return 0.0
    
    # Check for zero row or column sums (which lead to zero expected frequencies)
    row_sums = np.sum(obs, axis=1)
    col_sums = np.sum(obs, axis=0)
    if any(row_sums == 0) or any(col_sums == 0):
        # Can't compute chi-square with zero row/column sums
        return 0.0
    
    try:
        chi2, _, _, expected = stats.chi2_contingency(obs)
        # Check if any expected frequency is zero (which causes the error)
        if expected is not None and np.any(expected == 0):
            return 0.0
        return chi2
    except (ValueError, RuntimeWarning) as e:
        # Handle the specific error about zero expected frequencies
        if "zero element" in str(e).lower() or "expected frequencies" in str(e).lower():
            return 0.0
        # Re-raise other errors
        raise

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Load ground truth and metadata
    study_id = "study_003"
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
    # We need to group responses by sub_study_id
    agent_data = {
        "problem_1": [], "problem_2": [], "problem_3": [], "problem_4": [],
        "problem_5": [], "problem_6": [], "problem_7": [], "problem_8": [],
        "problem_9": [], "problem_10_version_1": [], "problem_10_version_2": []
    }

    for participant in results.get("individual_data", []):
        for resp in participant.get("responses", []):
            trial_info = resp.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            if sub_id in agent_data:
                items = trial_info.get("items", [])
                
                # If items are missing from trial_info, load from materials file
                if not items and sub_id:
                    items = load_materials(sub_id)
                
                parsed = parse_agent_responses(resp.get("response_text", ""))
                agent_data[sub_id].append(parsed)

    # 3. Process Findings
    test_results = []
    
    # Pre-calculate Human Pi values for each finding based on Ground Truth JSON
    # F1: Framing (Prob 1 vs Prob 2)
    # Human counts: P1 (N=152, 72% A -> 109 A, 43 B); P2 (N=155, 22% C -> 34 C, 121 D)
    chi2_h1, _, _, _ = stats.chi2_contingency([[109, 43], [34, 121]])
    bf_h1 = calc_bf_chisq(chi2_h1, 152 + 155, df=1)
    pi_h1 = calc_posteriors_3way(bf_h1, 1, prior_odds=10.0)

    # F2: Dominance (Prob 3 vs Prob 4)
    # Human counts: P3 (N=150, 73% AD -> 110 AD, 40 Other); P4 (N=86, 0% AD -> 0 AD, 86 BC)
    chi2_h2, _, _, _ = stats.chi2_contingency([[110, 40], [0, 86]])
    bf_h2 = calc_bf_chisq(chi2_h2, 150 + 86, df=1)
    pi_h2 = calc_posteriors_3way(bf_h2, 1, prior_odds=10.0)

    # F3: Pseudocertainty (Prob 6 vs Prob 7)
    # Human counts: P6 (N=85, 74% C -> 63 C, 22 D); P7 (N=81, 42% E -> 34 E, 47 F)
    chi2_h3, _, _, _ = stats.chi2_contingency([[63, 22], [34, 47]])
    bf_h3 = calc_bf_chisq(chi2_h3, 85 + 81, df=1)
    pi_h3 = calc_posteriors_3way(bf_h3, 1, prior_odds=10.0)

    # F4: Psychological Accounting (Prob 8 vs Prob 9)
    # Human counts: P8 (N=183, 88% Yes -> 161 Yes, 22 No); P9 (N=200, 46% Yes -> 92 Yes, 108 No)
    chi2_h4, _, _, _ = stats.chi2_contingency([[161, 22], [92, 108]])
    bf_h4 = calc_bf_chisq(chi2_h4, 183 + 200, df=1)
    pi_h4 = calc_posteriors_3way(bf_h4, 1, prior_odds=10.0)

    # F5: Mental Accounting (Prob 10 V1 vs V2)
    # Human counts: V1 (N=93, 68% Yes -> 63 Yes, 30 No); V2 (N=88, 29% Yes -> 26 Yes, 62 No)
    chi2_h5, _, _, _ = stats.chi2_contingency([[63, 30], [26, 62]])
    bf_h5 = calc_bf_chisq(chi2_h5, 93 + 88, df=1)
    pi_h5 = calc_posteriors_3way(bf_h5, 1, prior_odds=10.0)

    pi_human_map = {"F1": pi_h1, "F2": pi_h2, "F3": pi_h3, "F4": pi_h4, "F5": pi_h5}
    pi_human_source_map = {
        "F1": "P1 (72% A, N=152) vs P2 (22% C, N=155)",
        "F2": "P3 (73% AD, N=150) vs P4 (0% AD, N=86)",
        "F3": "P6 (74% C, N=85) vs P7 (42% E, N=81)",
        "F4": "P8 (88% Yes, N=183) vs P9 (46% Yes, N=200)",
        "F5": "V1 (68% Yes, N=93) vs V2 (29% Yes, N=88)"
    }
    human_chi2_map = {"F1": chi2_h1, "F2": chi2_h2, "F3": chi2_h3, "F4": chi2_h4, "F5": chi2_h5}

    # Map findings to their required sub_studies
    finding_configs = [
        {"id": "F1", "sub_studies": ["problem_1", "problem_2"], "study_id": "Problem 1 & 2", "scenario": "Gains vs Losses Framing"},
        {"id": "F2", "sub_studies": ["problem_3", "problem_4"], "study_id": "Problem 3 & 4", "scenario": "Concurrent vs Integrated Decisions"},
        {"id": "F3", "sub_studies": ["problem_6", "problem_7"], "study_id": "Problem 5, 6, & 7", "scenario": "Pseudocertainty Effect"},
        {"id": "F4", "sub_studies": ["problem_8", "problem_9"], "study_id": "Problem 8 & 9", "scenario": "Theater Ticket Loss"},
        {"id": "F5", "sub_studies": ["problem_10_version_1", "problem_10_version_2"], "study_id": "Problem 10", "scenario": "Calculator Price Saving"}
    ]

    for config in finding_configs:
        fid = config["id"]
        pi_h = pi_human_map[fid]
        
        # Get test_gt from ground truth for SRS fields
        test_gt = {}
        for study in ground_truth.get("studies", []):
            for finding in study.get("findings", []):
                if finding.get("finding_id") == fid:
                    statistical_tests = finding.get("statistical_tests", [])
                    if statistical_tests:
                        test_gt = statistical_tests[0]
                    break
        
        # Calculate Agent Pi (3-way)
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        reason = "Insufficient data"
        p_val_agent = None
        chi2_agent = None
        contingency_agent = None
        
        try:
            if fid == "F1":
                p1_a = sum(1 for d in agent_data["problem_1"] if get_clean_choice(d.get("Q1")) == "A")
                p1_b = len(agent_data["problem_1"]) - p1_a
                p2_c = sum(1 for d in agent_data["problem_2"] if get_clean_choice(d.get("Q1")) == "C")
                p2_d = len(agent_data["problem_2"]) - p2_c
                contingency_agent = [[p1_a, p1_b], [p2_c, p2_d]]
                if len(agent_data["problem_1"]) > 5 and len(agent_data["problem_2"]) > 5:
                    chi2_agent, p_val_agent, _, _ = chi2_contingency_safe(contingency_agent)
                    bf_a = calc_bf_chisq(chi2_agent, p1_a+p1_b+p2_c+p2_d)
                    # Direction: Expect proportion in first group (A) > proportion in second (C)
                    # Since we use 2x2 chi-square, we check if rate1 > rate2
                    rate1 = p1_a / (p1_a + p1_b)
                    rate2 = p2_c / (p2_c + p2_d)
                    a_dir = 1 if rate1 > rate2 else -1
                    pi_a = calc_posteriors_3way(bf_a, a_dir)
                    reason = f"Chi2={chi2_agent:.2f}, N1={p1_a+p1_b}, N2={p2_c+p2_d}"
                else:
                    contingency_agent = None
            
            elif fid == "F2":
                p3_ad = sum(1 for d in agent_data["problem_3"] if get_clean_choice(d.get("Q1")) == "A" and get_clean_choice(d.get("Q2")) == "D")
                p3_other = len(agent_data["problem_3"]) - p3_ad
                p4_ad = sum(1 for d in agent_data["problem_4"] if get_clean_choice(d.get("Q1")) == "AD")
                p4_other = len(agent_data["problem_4"]) - p4_ad
                contingency_agent = [[p3_ad, p3_other], [p4_ad, p4_other]]
                if len(agent_data["problem_3"]) > 5 and len(agent_data["problem_4"]) > 5:
                    chi2_agent, p_val_agent, _, _ = chi2_contingency_safe(contingency_agent)
                    bf_a = calc_bf_chisq(chi2_agent, p3_ad+p3_other+p4_ad+p4_other)
                    rate1 = p3_ad / (p3_ad + p3_other)
                    rate2 = p4_ad / (p4_ad + p4_other)
                    a_dir = 1 if rate1 > rate2 else -1
                    pi_a = calc_posteriors_3way(bf_a, a_dir)
                    reason = f"Chi2={chi2_agent:.2f}, N1={p3_ad+p3_other}, N2={p4_ad+p4_other}"
                else:
                    contingency_agent = None

            elif fid == "F3":
                p6_c = sum(1 for d in agent_data["problem_6"] if get_clean_choice(d.get("Q1")) == "C")
                p6_d = len(agent_data["problem_6"]) - p6_c
                p7_e = sum(1 for d in agent_data["problem_7"] if get_clean_choice(d.get("Q1")) == "E")
                p7_f = len(agent_data["problem_7"]) - p7_e
                contingency_agent = [[p6_c, p6_d], [p7_e, p7_f]]
                if len(agent_data["problem_6"]) > 5 and len(agent_data["problem_7"]) > 5:
                    chi2_agent, p_val_agent, _, _ = chi2_contingency_safe(contingency_agent)
                    bf_a = calc_bf_chisq(chi2_agent, p6_c+p6_d+p7_e+p7_f)
                    rate1 = p6_c / (p6_c + p6_d)
                    rate2 = p7_e / (p7_e + p7_f)
                    a_dir = 1 if rate1 > rate2 else -1
                    pi_a = calc_posteriors_3way(bf_a, a_dir)
                    reason = f"Chi2={chi2_agent:.2f}, N1={p6_c+p6_d}, N2={p7_e+p7_f}"
                else:
                    contingency_agent = None
            
            elif fid == "F4":
                p8_y = sum(1 for d in agent_data["problem_8"] if get_clean_choice(d.get("Q1")) == "YES")
                p8_n = len(agent_data["problem_8"]) - p8_y
                p9_y = sum(1 for d in agent_data["problem_9"] if get_clean_choice(d.get("Q1")) == "YES")
                p9_n = len(agent_data["problem_9"]) - p9_y
                contingency_agent = [[p8_y, p8_n], [p9_y, p9_n]]
                if len(agent_data["problem_8"]) > 5 and len(agent_data["problem_9"]) > 5:
                    chi2_agent, p_val_agent, _, _ = chi2_contingency_safe(contingency_agent)
                    bf_a = calc_bf_chisq(chi2_agent, p8_y+p8_n+p9_y+p9_n)
                    rate1 = p8_y / (p8_y + p8_n)
                    rate2 = p9_y / (p9_y + p9_n)
                    a_dir = 1 if rate1 > rate2 else -1
                    pi_a = calc_posteriors_3way(bf_a, a_dir)
                    reason = f"Chi2={chi2_agent:.2f}, N1={p8_y+p8_n}, N2={p9_y+p9_n}"
                else:
                    contingency_agent = None

            elif fid == "F5":
                v1_y = sum(1 for d in agent_data["problem_10_version_1"] if get_clean_choice(d.get("Q1")) == "YES")
                v1_n = len(agent_data["problem_10_version_1"]) - v1_y
                v2_y = sum(1 for d in agent_data["problem_10_version_2"] if get_clean_choice(d.get("Q1")) == "YES")
                v2_n = len(agent_data["problem_10_version_2"]) - v2_y
                contingency_agent = [[v1_y, v1_n], [v2_y, v2_n]]
                if len(agent_data["problem_10_version_1"]) > 5 and len(agent_data["problem_10_version_2"]) > 5:
                    chi2_agent, p_val_agent, _, _ = chi2_contingency_safe(contingency_agent)
                    bf_a = calc_bf_chisq(chi2_agent, v1_y+v1_n+v2_y+v2_n)
                    rate1 = v1_y / (v1_y + v1_n)
                    rate2 = v2_y / (v2_y + v2_n)
                    a_dir = 1 if rate1 > rate2 else -1
                    pi_a = calc_posteriors_3way(bf_a, a_dir)
                    reason = f"Chi2={chi2_agent:.2f}, N1={v1_y+v1_n}, N2={v2_y+v2_n}"
                else:
                    contingency_agent = None

        except Exception as e:
            reason = f"Error in stats: {str(e)}"

        pas = float(calc_pas(pi_h, pi_a))
        
        # Extract chi-square statistic from agent_reason for agent_test_statistic
        agent_test_stat = ""
        if "Chi2=" in reason:
            try:
                chi2_match = reason.split("Chi2=")[1].split(",")[0]
                agent_test_stat = chi2_match.strip()
            except:
                pass
        
        # Get human chi-square statistic
        human_chi2 = human_chi2_map.get(fid, "")
        human_test_stat = f"{human_chi2:.2f}" if human_chi2 else ""
        
        # Create combined sub_study_id to show both problems being compared
        sub_study_ids = config["sub_studies"]
        if len(sub_study_ids) == 2:
            combined_sub_id = f"{sub_study_ids[0]}_vs_{sub_study_ids[1]}"
        else:
            combined_sub_id = "_vs_".join(sub_study_ids)
        
        # Create test result dict
        test_result = {
            "study_id": config["study_id"],
            "sub_study_id": combined_sub_id,
            "finding_id": fid,
            "test_name": "Percentage Comparison / Chi-Square",
            "scenario": config.get("scenario", "Percentage Comparison / Chi-Square"),
            "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
            "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
            "pi_human_3way": pi_h,
            "pi_agent_3way": pi_a,
            "pas": pas,
            "test_weight": float(test_weights.get((fid, config.get("test_name", "")), 1.0)),
            "pi_human_source": pi_human_source_map[fid],
            "agent_reason": reason,
            "statistical_test_type": "Chi-square",
            "human_test_statistic": human_test_stat,
            "agent_test_statistic": agent_test_stat
        }
        
        # Extract human contingency table from ground truth
        # For Study 003, we need to reconstruct from the pre-calculated chi-square values
        # The human data is: F1: [[109, 43], [34, 121]], F2: [[110, 40], [0, 86]], etc.
        contingency_human = None
        if fid == "F1":
            contingency_human = [[109, 43], [34, 121]]
        elif fid == "F2":
            contingency_human = [[110, 40], [0, 86]]
        elif fid == "F3":
            contingency_human = [[63, 22], [34, 47]]
        elif fid == "F4":
            contingency_human = [[161, 22], [92, 108]]
        elif fid == "F5":
            contingency_human = [[63, 30], [26, 62]]
        
        # Add statistical replication fields with contingency tables for frequentist consistency
        add_statistical_replication_fields(
            test_result, test_gt, p_val_agent, chi2_agent, "chi-square",
            contingency_agent=contingency_agent,
            contingency_human=contingency_human
        )
        
        test_results.append(test_result)

    # 4. Two-level Weighted Aggregation
    # Level 1: Aggregate tests into findings (weighted by test weights)
    finding_results = []
    for fid in ["F1", "F2", "F3", "F4", "F5"]:
        f_tests = [t for t in test_results if t["finding_id"] == fid]
        if f_tests:
            # Weighted average: Σ (PAS * weight) / Σ weights
            total_weighted_pas = sum(t["pas"] * t.get("test_weight", 1.0) for t in f_tests)
            total_weight = sum(t.get("test_weight", 1.0) for t in f_tests)
            finding_score = total_weighted_pas / total_weight if total_weight > 0 else 0.5
            
            finding_weight = finding_weights.get(fid, 1.0)
            finding_results.append({
                "finding_id": fid,
                "finding_score": float(finding_score),
                "finding_weight": float(finding_weight),
                "n_tests": len(f_tests)
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