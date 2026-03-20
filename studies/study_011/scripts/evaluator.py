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
    calc_posteriors_3way,
    calc_pas,
    parse_p_value_from_reported,
    get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, float]:
    """
    Parses Qk=value or Qk: value format from response text.
    Matches integers and decimals.
    
    Handles multiple formats:
    - Q1=2.50, Q1: 2.50, Q1 = 2.50 (standard formats)
    - **Q1=2.50** (markdown bold formatting - common in Claude refusals)
    - Q1=2.50 (with markdown asterisks around it)
    - Standalone dollar amounts if Q format not found (fallback)
    """
    results = {}
    
    # First, try to match explicit Q formats (most reliable)
    # Pattern handles: Q1=2.50, Q1: 2.50, Q1 = 2.50, **Q1=2.50**, *Q1=2.50*, etc.
    # The pattern is flexible with whitespace and markdown formatting
    patterns = [
        # Standard format: Q1=2.50 or Q1: 2.50
        re.compile(r"Q(\d+(?:\.\d+)?)\s*[:=]\s*(\d*\.?\d+)"),
        # Markdown bold: **Q1=2.50** or *Q1=2.50*
        re.compile(r"[*]{1,2}Q(\d+(?:\.\d+)?)\s*[:=]\s*(\d*\.?\d+)[*]{1,2}"),
        # With parentheses or brackets: (Q1=2.50) or [Q1=2.50]
        re.compile(r"[\[\(]Q(\d+(?:\.\d+)?)\s*[:=]\s*(\d*\.?\d+)[\]\)]"),
    ]
    
    for pattern in patterns:
        for q_idx, val in pattern.findall(response_text):
            try:
                q_key = f"Q{q_idx}"
                # Only add if not already found (first match wins)
                if q_key not in results:
                    results[q_key] = float(val)
            except ValueError:
                continue
    
    # If no Q format found, try to extract standalone dollar amounts as fallback
    # This handles cases where Claude mentions amounts but doesn't use Q format
    if not results:
        # Look for dollar amounts that might be the answer
        # Pattern: $2.50, $2, 2.50, etc. (reasonable range for $5 or $10 splits)
        dollar_patterns = [
            r"\$(\d+\.?\d*)",  # $2.50 format
            r"\b(\d+\.\d{2})\b",  # 2.50 format (two decimal places)
            r"\b(\d+)\b",  # Integer amounts
        ]
        
        amounts_found = []
        for pattern_str in dollar_patterns:
            for match in re.finditer(pattern_str, response_text):
                try:
                    amount = float(match.group(1))
                    # Validate: should be between 0 and 10 (reasonable for $5 or $10 games)
                    if 0 <= amount <= 10 and amount not in amounts_found:
                        amounts_found.append(amount)
                except (ValueError, IndexError):
                    continue
        
        # If we found a reasonable amount, map it to Q1
        if amounts_found:
            # Prefer amounts in the middle range (more likely to be offers)
            # Filter out very small amounts (< 0.10) unless it's the only option
            reasonable_amounts = [a for a in amounts_found if a >= 0.10] or amounts_found
            if reasonable_amounts:
                results["Q1"] = reasonable_amounts[0]  # Take first reasonable amount
    
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_011: 从items中提取q_idx
    """
    required = set()
    items = trial_info.get("items", [])
    
    for item in items:
        q_idx = item.get("q_idx")
        if q_idx:
            # 如果q_idx已经包含"Q"前缀，直接使用；否则添加"Q"前缀
            if isinstance(q_idx, str) and q_idx.startswith("Q"):
                required.add(q_idx)
            else:
                required.add(f"Q{q_idx}")
    
    return required

def calculate_t_from_summary(m1, sd1, n1, m2, sd2, n2):
    """Calculates independent samples t-statistic from summary statistics."""
    if n1 < 2 or n2 < 2:
        return 0.0
    # Pooled standard deviation
    var1 = sd1**2
    var2 = sd2**2
    sp = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    t_stat = (m1 - m2) / (sp * np.sqrt(1/n1 + 1/n2))
    return t_stat

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates Study 011: Fairness in Simple Bargaining Experiments.
    Calculates PAS for findings comparing DG/UG, Pay/No-Pay, and Pie Sizes.
    """
    # 1. Load ground truth and metadata
    study_id = "study_011"
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
    # Organize by sub_study_id
    agent_raw = {
        "DG-P_April_Sept": [],
        "UG-P_April_Sept": [],
        "DG-NP_April_Sept": [],
        "UG-NP_April_Sept": [],
        "DG-P_10_dollars": [],
        "UG-P_10_dollars": []
    }

    for participant in results.get("individual_data", []):
        for resp in participant.get("responses", []):
            trial_info = resp.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            if sub_id not in agent_raw:
                continue
            
            parsed = parse_agent_responses(resp.get("response_text", ""))
            
            # Usually one offer per trial in this study design
            items = trial_info.get("items", [])
            
            # If items are missing from trial_info, load from materials file
            if not items and sub_id:
                items = load_materials(sub_id)
            
            for item in items:
                q_idx = f"Q{item.get('q_idx')}"
                if q_idx in parsed:
                    agent_raw[sub_id].append(parsed[q_idx])
                elif "proposal_form" in item.get("id", ""): # Fallback for different item IDs
                    # If specific Q index not found, take the first parsed value
                    if parsed:
                        agent_raw[sub_id].append(list(parsed.values())[0])

    # 3. Define Finding Logic
    # We use t-tests as the standard comparison tool for PAS calculation
    test_results = []
    
    # Map finding IDs to data processing logic
    findings_to_process = []
    for exp in ground_truth["studies"]:
        for finding in exp["findings"]:
            findings_to_process.append(finding)

    for finding in findings_to_process:
        fid = finding["finding_id"]
        
        # Skip replicability tests (F1, F3, F4, F6) - not needed for simulation
        # These test if April vs September sessions are the same, which is irrelevant for LLM simulation
        if fid in ["F1", "F3", "F4", "F6"]:
            continue
        
        original_data = finding.get("original_data_points", {}).get("data", {})
        keys = list(original_data.keys())
        
        if len(keys) < 2:
            continue
            
        # Human Stats
        h1, h2 = original_data[keys[0]], original_data[keys[1]]
        t_human = calculate_t_from_summary(h1["mean"], h1["sd"], h1["n"], h2["mean"], h2["sd"], h2["n"])
        # Determine human direction
        h_dir = get_direction_from_statistic(t_human, "t-test")
        pi_h_dict = calc_posteriors_3way(calc_bf_t(t_human, h1["n"], h2["n"], independent=True), h_dir, prior_odds=10.0)
        pi_h_plus, pi_h_minus, pi_h_zero = pi_h_dict['pi_plus'], pi_h_dict['pi_minus'], pi_h_dict['pi_zero']
        
        # Agent Stats
        a_group1, a_group2 = [], []
        sub_id_label = ""

        if fid == "F1": # DG-P April vs Sept
            data = agent_raw["DG-P_April_Sept"]
            a_group1, a_group2 = data[:21], data[21:45]
            sub_id_label = "DG-P_April_Sept"
        elif fid == "F3": # UG-P April vs Sept
            data = agent_raw["UG-P_April_Sept"]
            a_group1, a_group2 = data[:20], data[20:43]
            sub_id_label = "UG-P_April_Sept"
        elif fid == "F4": # DG-NP April vs Sept
            data = agent_raw["DG-NP_April_Sept"]
            a_group1, a_group2 = data[:22], data[22:46]
            sub_id_label = "DG-NP_April_Sept"
        elif fid == "F6": # UG-NP April vs Sept
            data = agent_raw["UG-NP_April_Sept"]
            a_group1, a_group2 = data[:24], data[24:48]
            sub_id_label = "UG-NP_April_Sept"
        elif fid == "F2": # DG-P vs UG-P
            a_group1, a_group2 = agent_raw["DG-P_April_Sept"], agent_raw["UG-P_April_Sept"]
            sub_id_label = "DG-P_vs_UG-P"
        elif fid == "F5": # DG-P vs DG-NP
            a_group1, a_group2 = agent_raw["DG-P_April_Sept"], agent_raw["DG-NP_April_Sept"]
            sub_id_label = "DG-P_vs_DG-NP"
        elif fid == "F7": # UG-P vs UG-NP
            a_group1, a_group2 = agent_raw["UG-P_April_Sept"], agent_raw["UG-NP_April_Sept"]
            sub_id_label = "UG-P_vs_UG-NP"
        elif fid == "F8": # DG-P $5 vs $10 (Proportions)
            a_group1 = [x/5.0 for x in agent_raw["DG-P_April_Sept"]]
            a_group2 = [x/10.0 for x in agent_raw["DG-10_dollars"]] if "DG-10_dollars" in agent_raw else [x/10.0 for x in agent_raw.get("DG-P_10_dollars", [])]
            sub_id_label = "DG-P_5_vs_10"
        elif fid == "F9": # UG-P $5 vs $10 (Proportions)
            a_group1 = [x/5.0 for x in agent_raw["UG-P_April_Sept"]]
            a_group2 = [x/10.0 for x in agent_raw["UG-10_dollars"]] if "UG-10_dollars" in agent_raw else [x/10.0 for x in agent_raw.get("UG-P_10_dollars", [])]
            sub_id_label = "UG-P_5_vs_10"
        elif fid == "F10": # DG-P $10 vs UG-P $10
            a_group1, a_group2 = agent_raw.get("DG-P_10_dollars", []), agent_raw.get("UG-P_10_dollars", [])
            sub_id_label = "DG-P_vs_UG-P_10"

        # Calculate Agent PI
        p_val_agent = None
        t_agent = None
        pi_a_plus, pi_a_minus, pi_a_zero = 1/3, 1/3, 1/3
        if len(a_group1) >= 2 and len(a_group2) >= 2:
            t_agent, p_val_agent = stats.ttest_ind(a_group1, a_group2, equal_var=False)
            if np.isnan(t_agent): t_agent = 0.0
            a_dir = get_direction_from_statistic(t_agent, "t-test")
            pi_a_dict = calc_posteriors_3way(calc_bf_t(t_agent, len(a_group1), len(a_group2), independent=True), a_dir)
            pi_a_plus, pi_a_minus, pi_a_zero = pi_a_dict['pi_plus'], pi_a_dict['pi_minus'], pi_a_dict['pi_zero']
            reason = f"t={t_agent:.3f}, n1={len(a_group1)}, n2={len(a_group2)}"
        else:
            reason = f"Insufficient data (n1={len(a_group1)}, n2={len(a_group2)})"

        # Calculate PAS for each specific test mentioned in GT
        pi_h_dict = {"pi_plus": pi_h_plus, "pi_minus": pi_h_minus, "pi_zero": pi_h_zero}
        pi_a_dict = {"pi_plus": pi_a_plus, "pi_minus": pi_a_minus, "pi_zero": pi_a_zero}
        for test in finding["statistical_tests"]:
            pas = calc_pas(pi_h_dict, pi_a_dict)
            test_result = {
                "study_id": study_id,
                "sub_study_id": sub_id_label,
                "finding_id": fid,
                "test_name": test["test_name"],
                "pi_human": float(pi_h_plus + pi_h_minus),
                "pi_agent": float(pi_a_plus + pi_a_minus),
                "pi_human_3way": pi_h_dict,
                "pi_agent_3way": pi_a_dict,
                "pas": float(pas),
                "pi_human_source": test["reported_statistics"],
                "agent_reason": reason,
                "statistical_test_type": "t-test",
                "human_test_statistic": f"{t_human:.3f}" if t_human is not None else "",
                "agent_test_statistic": f"{t_agent:.3f}" if t_agent is not None else ""
            }
            
            # Add statistical replication fields with sample sizes
            if len(a_group1) >= 2 and len(a_group2) >= 2:
                add_statistical_replication_fields(
                    test_result, test, p_val_agent, t_agent, "t-test",
                    n_agent=len(a_group1),
                    n2_agent=len(a_group2),
                    n_human=h1["n"],
                    n2_human=h2["n"],
                    independent=True
                )
            else:
                add_statistical_replication_fields(test_result, test, p_val_agent, t_agent, "t-test")
            
            test_results.append(test_result)

    # 4. Two-level Weighted Aggregation
    # Add test_weight to each test result
    for tr in test_results:
        finding_id = tr.get("finding_id")
        test_name = tr.get("test_name", "")
        tr["test_weight"] = float(test_weights.get((finding_id, test_name), 1.0))
    
    if not test_results:
        return {"score": 0.0, "test_results": [], "finding_results": [], "substudy_results": []}

    # Level 1: Aggregate tests into findings (weighted by test weights)
    finding_results = []
    for fid in sorted(list(set(t["finding_id"] for t in test_results))):
        f_tests = [t for t in test_results if t["finding_id"] == fid]
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
    overall_score = total_weighted_finding_score / total_finding_weight if total_finding_weight > 0 else 0.0

    # substudy_results removed - using two-level aggregation (Tests -> Findings -> Study)
    substudy_results = []

    return {
        "score": overall_score,
        "substudy_results": substudy_results,
        "finding_results": finding_results,
        "test_results": test_results
    }