import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from stats_lib import (
    calc_bf_chisq, 
    calc_posteriors_3way,
    calc_pas,
    parse_p_value_from_reported,
    get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """Parse Qk=<value> or Qk: <value> or Qk.n=<value> format from raw response text."""
    results = {}
    # Matches Q1=choice, Q1: choice, Q2=choice, etc.
    # Supports both = and : separators
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*[:=]\s*([^,\n\s]+)")
    for k, v in pattern.findall(response_text):
        results[k.strip()] = v.strip().lower()
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_010: 从items中提取，使用索引+1 (Q1, Q2, Q3对应items[0], items[1], items[2])
    """
    required = set()
    items = trial_info.get("items", [])
    
    # Study_010: Q编号对应items的索引+1
    for idx, item in enumerate(items):
        required.add(f"Q{idx + 1}")
    
    return required

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates Shafir & Tversky (1992) - Study 010.
    Focuses on:
    1. Prisoner's Dilemma Triads (Disjunction Effect)
    2. Newcomb's Problem (Computer version)
    3. Information Seeking in PD
    """
    # 1. Load ground truth and metadata
    study_id = "study_010"
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
    # Structure: sub_id -> list of participant data
    agent_raw = {
        "pd_triad_tasks": [],
        "newcombs_computer_task": [],
        "pd_info_seeking_variation": []
    }

    for participant in results.get("individual_data", []):
        for resp in participant.get("responses", []):
            trial_info = resp.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            
            # Use parsed_response from config or fallback to manual parse
            parsed = resp.get("parsed_response", {})
            if not parsed:
                # Parse Q1=value, Q2=value format
                q_parsed = parse_agent_responses(resp.get("response_text", ""))
                # Map Q1, Q2, Q3 to item IDs based on trial_info items
                parsed = {}
                items = trial_info.get("items", [])
                
                # If items are missing from trial_info, load from materials file
                if not items and sub_id:
                    items = load_materials(sub_id)
                
                for idx, item in enumerate(items):
                    q_idx = f"Q{idx + 1}"
                    item_id = item.get("id")
                    if q_idx in q_parsed and item_id:
                        choice_value = q_parsed[q_idx]
                        # Convert A/B/C to option text if needed
                        options = item.get("options", [])
                        if choice_value.upper() in ['A', 'B', 'C', 'D', 'E']:
                            # It's a letter, convert to option index
                            letter_idx = ord(choice_value.upper()) - ord('A')
                            if letter_idx < len(options):
                                parsed[item_id] = options[letter_idx]
                            else:
                                parsed[item_id] = choice_value  # Fallback
                        else:
                            # Already option text or other format
                            parsed[item_id] = choice_value
            
            if sub_id in agent_raw:
                agent_raw[sub_id].append(parsed)

    # 3. Process each finding/test
    test_results = []
    
    for study in ground_truth["studies"]:
        for finding in study["findings"]:
            fid = finding["finding_id"]
            
            for test in finding["statistical_tests"]:
                pi_h_plus, pi_h_minus, pi_h_zero = 1/3, 1/3, 1/3
                pi_a_plus, pi_a_minus, pi_a_zero = 1/3, 1/3, 1/3
                reason = "Insufficient data"
                sub_id = ""
                p_val_agent = None
                chi2_agent = None
                test_stat_human = ""
                contingency_agent = None
                contingency_human = None

                # --- Finding F1: PD Triads ---
                if fid == "F1":
                    sub_id = "pd_triad_tasks"
                    data = agent_raw[sub_id]
                    
                    if test["test_name"] == "Chi-square or proportion test (implied)":
                        # Human stats from GT
                        # Known Comp: 14/444, Known Coop: 73/444, Unknown: 164/444
                        n_h_triads = 444
                        h_counts = [[14, 430], [73, 371], [164, 280]]
                        chi2_h, p_val_h, _, _ = stats.chi2_contingency(h_counts)
                        test_stat_human = f"{chi2_h:.2f}"
                        # Human direction: 1 if Cooperate rate is higher in Unknown than in Known
                        coop_rates_h = [row[0]/sum(row) for row in h_counts]
                        h_dir = 1 if coop_rates_h[2] > max(coop_rates_h[0], coop_rates_h[1]) else -1
                        pi_h = calc_posteriors_3way(calc_bf_chisq(chi2_h, n_h_triads * 3, df=2), h_dir, prior_odds=10.0)
                        pi_h_plus, pi_h_minus, pi_h_zero = pi_h['pi_plus'], pi_h['pi_minus'], pi_h['pi_zero']
                        
                        # Agent stats
                        a_counts = [[0, 0], [0, 0], [0, 0]] # [Coop, Comp] for Comp, Coop, Unk
                        for d in data:
                            if "pd_known_compete" in d:
                                a_counts[0][0 if "cooperate" in d["pd_known_compete"] else 1] += 1
                            if "pd_known_cooperate" in d:
                                a_counts[1][0 if "cooperate" in d["pd_known_cooperate"] else 1] += 1
                            if "pd_unknown" in d:
                                a_counts[2][0 if "cooperate" in d["pd_unknown"] else 1] += 1
                        
                        n_a = sum([sum(row) for row in a_counts])
                        if n_a > 10: # Minimum threshold
                            # Robust chi-square test: filter out zero rows and check columns
                            valid_rows = [row for row in a_counts if sum(row) > 0]
                            if len(valid_rows) >= 2:
                                # Check columns: ensure each column has at least one non-zero
                                col_sums_valid = [sum(row[i] for row in valid_rows) for i in range(2)]
                                if all(col_sum > 0 for col_sum in col_sums_valid):
                                    try:
                                        chi2_a, p_val_a, _, _ = stats.chi2_contingency(valid_rows)
                                        if not np.isnan(chi2_a) and chi2_a >= 0:
                                            chi2_agent = chi2_a
                                            p_val_agent = p_val_a
                                            # Agent direction: 1 if Cooperate rate is higher in Unknown than in Known
                                            # (Using same logic as human, even if some rows are 0 in agent data, we map back to original 3 rows if possible)
                                            rates_a = []
                                            for row in a_counts:
                                                row_sum = sum(row)
                                                rates_a.append(row[0]/row_sum if row_sum > 0 else 0)
                                            
                                            a_dir = 1 if rates_a[2] > max(rates_a[0], rates_a[1]) else -1
                                            pi_a = calc_posteriors_3way(calc_bf_chisq(chi2_a, n_a, df=len(valid_rows)-1), a_dir)
                                            pi_a_plus, pi_a_minus, pi_a_zero = pi_a['pi_plus'], pi_a['pi_minus'], pi_a['pi_zero']
                                            reason = f"Chi2={chi2_a:.2f}, n={n_a}, Dir={a_dir}"
                                        else:
                                            pi_a_plus, pi_a_minus, pi_a_zero = 0.0, 0.0, 1.0
                                            reason = f"Chi2=NaN (invalid), n={n_a}"
                                    except (ValueError, RuntimeWarning) as e:
                                        pi_a_plus, pi_a_minus, pi_a_zero = 0.0, 0.0, 1.0
                                        reason = f"Chi2 error (zero expected freq), n={n_a}"
                                else:
                                    pi_a_plus, pi_a_minus, pi_a_zero = 0.0, 0.0, 1.0
                                    reason = f"LLM behavior lacks variation (zero columns), n={n_a}"
                            else:
                                pi_a_plus, pi_a_minus, pi_a_zero = 0.0, 0.0, 1.0
                                reason = f"LLM behavior lacks variation (only {len(valid_rows)} valid rows), n={n_a}"
                        else:
                            # Handle case when n_a <= 10
                            pi_a_plus, pi_a_minus, pi_a_zero = 0.0, 0.0, 1.0
                            reason = f"Insufficient data (n={n_a}, need >10)"
                        
                        # Set contingency tables for this test
                        contingency_agent = a_counts
                        contingency_human = h_counts
                            
                    elif test["test_name"] == "Descriptive frequency analysis":
                        # Pattern: Compete in Known, Cooperate in Unknown
                        n_h = 444
                        h_pattern_count = 113
                        # Test against a null of 5% (to check if effect exists)
                        chi2_h = ((h_pattern_count - (n_h * 0.05))**2 / (n_h * 0.05)) + \
                                 (( (n_h - h_pattern_count) - (n_h * 0.95))**2 / (n_h * 0.95))
                        test_stat_human = f"{chi2_h:.2f}"
                        # Direction 1 if pattern count > 5%
                        h_dir = 1 if h_pattern_count > n_h * 0.05 else -1
                        pi_h = calc_posteriors_3way(calc_bf_chisq(chi2_h, n_h, df=1), h_dir, prior_odds=10.0)
                        pi_h_plus, pi_h_minus, pi_h_zero = pi_h['pi_plus'], pi_h['pi_minus'], pi_h['pi_zero']
                        
                        a_pattern_count = 0
                        valid_n_a = 0
                        for d in data:
                            if all(k in d for k in ["pd_known_compete", "pd_known_cooperate", "pd_unknown"]):
                                valid_n_a += 1
                                if "compete" in d["pd_known_compete"] and \
                                   "compete" in d["pd_known_cooperate"] and \
                                   "cooperate" in d["pd_unknown"]:
                                    a_pattern_count += 1
                        
                        if valid_n_a > 0:
                            exp_a = valid_n_a * 0.05
                            chi2_a = ((a_pattern_count - exp_a)**2 / max(0.1, exp_a)) + \
                                     (((valid_n_a - a_pattern_count) - (valid_n_a * 0.95))**2 / (valid_n_a * 0.95))
                            chi2_agent = chi2_a
                            p_val_agent = 1 - stats.chi2.cdf(chi2_a, 1) if chi2_a > 0 else 1.0
                            a_dir = 1 if a_pattern_count > valid_n_a * 0.05 else -1
                            pi_a = calc_posteriors_3way(calc_bf_chisq(chi2_a, valid_n_a, df=1), a_dir)
                            pi_a_plus, pi_a_minus, pi_a_zero = pi_a['pi_plus'], pi_a['pi_minus'], pi_a['pi_zero']
                            reason = f"Pattern count={a_pattern_count}/{valid_n_a}, Dir={a_dir}"
                            
                            # Set contingency tables for this test
                            contingency_agent = [[a_pattern_count, valid_n_a - a_pattern_count], [round(valid_n_a * 0.05), round(valid_n_a * 0.95)]]
                            contingency_human = [[h_pattern_count, n_h - h_pattern_count], [round(n_h * 0.05), round(n_h * 0.95)]]

                # --- Finding F2: Newcomb ---
                elif fid == "F2":
                    sub_id = "newcombs_computer_task"
                    data = agent_raw[sub_id]
                    n_h = 40
                    h_count = 26 # 65% of 40
                    # Test against 50%
                    chi2_h = ((h_count - 20)**2 / 20) + ((14 - 20)**2 / 20)
                    test_stat_human = f"{chi2_h:.2f}"
                    # Direction 1 if Box B count > 50%
                    h_dir = 1 if h_count > n_h * 0.5 else -1
                    pi_h = calc_posteriors_3way(calc_bf_chisq(chi2_h, n_h, df=1), h_dir, prior_odds=10.0)
                    pi_h_plus, pi_h_minus, pi_h_zero = pi_h['pi_plus'], pi_h['pi_minus'], pi_h['pi_zero']
                    
                    # Check for Box B only: value could be "2", "box b", "only", etc.
                    a_count = 0
                    for d in data:
                        choice_val = str(d.get("newcomb_choice", "")).lower()
                        # Match: "2", "box b", "only", or contains "box b only"
                        if choice_val == "2" or "box b" in choice_val or ("only" in choice_val and "both" not in choice_val):
                            a_count += 1
                    n_a = len(data)
                    if n_a > 0:
                        exp = n_a * 0.5
                        chi2_a = ((a_count - exp)**2 / max(0.1, exp)) + (((n_a - a_count) - exp)**2 / max(0.1, exp))
                        chi2_agent = chi2_a
                        p_val_agent = 1 - stats.chi2.cdf(chi2_a, 1) if chi2_a > 0 else 1.0
                        a_dir = 1 if a_count > n_a * 0.5 else -1
                        pi_a = calc_posteriors_3way(calc_bf_chisq(chi2_a, n_a, df=1), a_dir)
                        pi_a_plus, pi_a_minus, pi_a_zero = pi_a['pi_plus'], pi_a['pi_minus'], pi_a['pi_zero']
                        reason = f"Box B count={a_count}/{n_a}, Dir={a_dir}"
                        
                        # Set contingency tables for this test
                        contingency_agent = [[a_count, n_a - a_count], [round(n_a * 0.5), round(n_a * 0.5)]]
                        contingency_human = [[h_count, n_h - h_count], [round(n_h * 0.5), round(n_h * 0.5)]]

                # --- Finding F3: Info Seeking ---
                elif fid == "F3":
                    sub_id = "pd_info_seeking_variation"
                    data = agent_raw[sub_id]
                    # GT: 81%. Assume N=80 (from Exp 1 sample size pool)
                    n_h = 80
                    h_count = 65 # ~81%
                    chi2_h = ((h_count - 40)**2 / 40) + ((15 - 40)**2 / 40)
                    test_stat_human = f"{chi2_h:.2f}"
                    # Direction 1 if Pay count > 50%
                    h_dir = 1 if h_count > n_h * 0.5 else -1
                    pi_h = calc_posteriors_3way(calc_bf_chisq(chi2_h, n_h, df=1), h_dir, prior_odds=10.0)
                    pi_h_plus, pi_h_minus, pi_h_zero = pi_h['pi_plus'], pi_h['pi_minus'], pi_h['pi_zero']
                    
                    # Check for paying for information: value could be "yes", "pay fee", etc.
                    a_count = 0
                    for d in data:
                        choice_val = str(d.get("info_search_choice", "")).lower()
                        # Match: "yes", "pay", or contains "pay fee"
                        if "yes" in choice_val or "pay" in choice_val:
                            a_count += 1
                    n_a = len(data)
                    if n_a > 0:
                        exp = n_a * 0.5
                        chi2_a = ((a_count - exp)**2 / max(0.1, exp)) + (((n_a - a_count) - exp)**2 / max(0.1, exp))
                        chi2_agent = chi2_a
                        p_val_agent = 1 - stats.chi2.cdf(chi2_a, 1) if chi2_a > 0 else 1.0
                        a_dir = 1 if a_count > n_a * 0.5 else -1
                        pi_a = calc_posteriors_3way(calc_bf_chisq(chi2_a, n_a, df=1), a_dir)
                        pi_a_plus, pi_a_minus, pi_a_zero = pi_a['pi_plus'], pi_a['pi_minus'], pi_a['pi_zero']
                        reason = f"Pay count={a_count}/{n_a}, Dir={a_dir}"
                        
                        # Set contingency tables for this test
                        contingency_agent = [[a_count, n_a - a_count], [round(n_a * 0.5), round(n_a * 0.5)]]
                        contingency_human = [[h_count, n_h - h_count], [round(n_h * 0.5), round(n_h * 0.5)]]

                # Calculate PAS
                pi_h_dict = {"pi_plus": pi_h_plus, "pi_minus": pi_h_minus, "pi_zero": pi_h_zero}
                pi_a_dict = {"pi_plus": pi_a_plus, "pi_minus": pi_a_minus, "pi_zero": pi_a_zero}
                pas = calc_pas(pi_h_dict, pi_a_dict)
                
                # Create test result dict
                test_result = {
                    "study_id": study["study_id"],
                    "sub_study_id": sub_id,
                    "finding_id": fid,
                    "test_name": test["test_name"],
                    "pi_human": float(pi_h_plus + pi_h_minus),
                    "pi_agent": float(pi_a_plus + pi_a_minus),
                    "pi_human_3way": pi_h_dict,
                    "pi_agent_3way": pi_a_dict,
                    "pas": float(pas),
                    "pi_human_source": test.get("reported_statistics", "GT"),
                    "agent_reason": reason,
                    "statistical_test_type": "chi-square",
                    "human_test_statistic": test_stat_human,
                    "agent_test_statistic": f"{chi2_agent:.2f}" if chi2_agent is not None else ""
                }
                
                # Contingency tables are now set directly in each test processing block above
                # Add statistical replication fields
                add_statistical_replication_fields(
                    test_result, test, p_val_agent, chi2_agent, "chi-square",
                    contingency_agent=contingency_agent,
                    contingency_human=contingency_human
                )
                
                test_results.append(test_result)

    # 4. Two-level Weighted Aggregation
    # Add test_weight to each test result
    for tr in test_results:
        finding_id = tr.get("finding_id")
        test_name = tr.get("test_name", "")
        tr["test_weight"] = float(test_weights.get((finding_id, test_name), 1.0))
    
    if not test_results:
        return {"score": 0.5, "test_results": [], "substudy_results": [], "finding_results": []}

    # Level 1: Aggregate tests into findings (weighted by test weights)
    finding_map = {}
    for t in test_results:
        key = (t["sub_study_id"], t["finding_id"])
        if key not in finding_map:
            finding_map[key] = []
        finding_map[key].append(t)
    
    finding_results = []
    for k, tests in finding_map.items():
        # Weighted average: Σ (PAS * weight) / Σ weights
        total_weighted_pas = sum(t["pas"] * t.get("test_weight", 1.0) for t in tests)
        total_weight = sum(t.get("test_weight", 1.0) for t in tests)
        finding_score = total_weighted_pas / total_weight if total_weight > 0 else 0.5
        
        finding_weight = finding_weights.get(k[1], 1.0)
        finding_results.append({
            "sub_study_id": k[0],
            "finding_id": k[1],
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

    # Total score
    score = np.mean([t["pas"] for t in test_results])

    return {
        "score": score,
        "substudy_results": substudy_results,
        "finding_results": finding_results,
        "test_results": test_results
    }