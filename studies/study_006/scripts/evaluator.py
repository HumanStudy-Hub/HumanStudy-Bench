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
    chi2_contingency_safe,
    parse_p_value_from_reported,
    get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """Parse Qk=<value>, Qk: <value>, or Qk.n=<value> format"""
    results = {}
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*[:=]\s*([^,\n\r]+)")
    for k, v in pattern.findall(response_text):
        results[k.strip()] = v.strip()
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_006: 从items中提取q_idx，如果没有则使用索引+1 (Q1, Q2, Q3对应items[0], items[1], items[2])
    """
    required = set()
    items = trial_info.get("items", [])
    
    # Study_006: Q编号对应items的索引+1，或者从item中提取q_idx
    for idx, item in enumerate(items):
        q_idx = item.get("q_idx")
        if q_idx:
            if isinstance(q_idx, str) and q_idx.startswith("Q"):
                required.add(q_idx)
            else:
                required.add(f"Q{q_idx}")
        else:
            # 如果没有q_idx，使用索引+1
                required.add(f"Q{idx + 1}")
    
    return required

def get_chi2_2sample(c1, n1, c2, n2):
    """Calculate Chi-square for 2x2 contingency table safely. Returns (chi2, p_value)."""
    # chi2_contingency_safe returns (chi2, p, dof, expected)
    chi2, p_val_safe, _, _ = chi2_contingency_safe([[c1, max(0, n1 - c1)], [c2, max(0, n2 - c2)]])
    # Calculate p-value from chi-square
    p_val = None
    if chi2 > 0:
        try:
            chi2_val, p_val, _, _ = stats.chi2_contingency([[c1, max(0, n1 - c1)], [c2, max(0, n2 - c2)]])
            if not np.isnan(p_val):
                return chi2_val, p_val
        except:
            pass
        # Fallback: calculate p-value from chi2 distribution
        from scipy.stats import chi2 as chi2_dist
        p_val = 1 - chi2_dist.cdf(chi2, 1) if chi2 > 0 else 1.0
    return chi2, p_val

def get_chi2_1sample(c, n, p_null=0.5):
    """Calculate Chi-square for 1-sample goodness of fit. Returns (chi2, p_value)."""
    if n == 0: return 0.0, None
    expected = n * p_null
    if expected == 0 or expected == n: return 0.0, None
    chi2 = ((c - expected)**2 / expected) + (((n - c) - (n - expected))**2 / (n - expected))
    # Calculate p-value from chi-square distribution
    from scipy.stats import chi2 as chi2_dist
    p_val = 1 - chi2_dist.cdf(chi2, 1) if chi2 > 0 else 1.0
    return chi2, p_val

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Load ground truth and metadata
    study_id = "study_006"
    study_dir = Path(__file__).resolve().parent.parent / "source"
    
    with open(study_dir / "ground_truth.json", 'r') as f:
        ground_truth = json.load(f)
    
    metadata = {}
    metadata_path = study_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Build weight maps
    finding_weights = {}
    test_weights = {}
    for finding in metadata.get("findings", []):
        fid = finding.get("finding_id")
        finding_weights[fid] = finding.get("weight", 1.0)
        for test in finding.get("tests", []):
            test_weights[(fid, test.get("test_name"))] = test.get("weight", 1.0)

    # Finding ID mapping (GT study_id to Metadata finding_id)
    study_to_fid = {
        "Experiment I": "F1",
        "Experiment II": "F2",
        "Experiment III": "F3",
        "Experiment IV": "F4",
        "Experiment V": "F5",
        "Experiment VI": "F6",
        "Experiment VII": "F7",
        "Experiment VIII": "F8",
        "Experiment IX": "F9",
        "Experiment IXa": "F10",
        "Experiment X": "F11"
    }

    # 2. Extract agent data organized by sub-study and condition
    agent_raw = {} # sub_study_id -> condition -> item_id -> [values]
    
    def normalize_condition(cond: str) -> str:
        """Normalize condition name to match ground truth keys (title case for multi-word)"""
        if not cond:
            return None
        # Convert to title case (e.g., "warm" -> "Warm", "positive first" -> "Positive First")
        return cond.title()

    # Sub-studies that should default to "Default" condition if None
    default_condition_substudies = {"Experiment II Task", "Experiment IX Task", "Experiment X Task"}

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

    for participant in results.get("individual_data", []):
        for resp in participant.get("responses", []):
            trial_info = resp.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            # Try condition first, then assigned_condition, normalize to match ground truth
            cond = trial_info.get("condition") or trial_info.get("assigned_condition")
            # If condition is None and this sub-study should have a default, use "Default"
            if not cond and sub_id in default_condition_substudies:
                cond = "Default"
            cond = normalize_condition(cond)
            items = trial_info.get("items", [])
            
            # If items are missing from trial_info, load from materials file
            if not items and sub_id:
                items = load_materials(sub_id)
            
            parsed = parse_agent_responses(resp.get("response_text", ""))
            
            if sub_id not in agent_raw: agent_raw[sub_id] = {}
            if cond not in agent_raw[sub_id]: agent_raw[sub_id][cond] = {}
            
            for i, item in enumerate(items):
                # Standard item indexing Q1, Q2...
                q_key = f"Q{i+1}"
                val = parsed.get(q_key)
                if val:
                    item_id = item.get("id") or str(i+1)
                    if item_id not in agent_raw[sub_id][cond]:
                        agent_raw[sub_id][cond][item_id] = []
                    agent_raw[sub_id][cond][item_id].append(val)

    # 3. Process Tests
    test_results = []
    
    for study in ground_truth["studies"]:
        gt_study_id = study["study_id"]
        fid = study_to_fid.get(gt_study_id)
        sub_study_id = f"{gt_study_id} Task"
        
        for finding in study["findings"]:
            data_points = finding.get("original_data_points", {}).get("data", {})
            
            for test_idx, test in enumerate(finding["statistical_tests"]):
                test_name = test["test_name"]
                reported = test["reported_statistics"]
                
                pi_h, pi_a = 0.5, 0.5
                reason = "Insufficient data"
                p_val_agent = None
                chi2_agent = None
                chi_h = None
                contingency_agent = None
                contingency_human = None
                
                # Logic per experiment
                try:
                    if gt_study_id == "Experiment I":
                        # 18 trait pairs. Item IDs 1-18.
                        trait_keys = list(data_points.keys())
                        trait_key = trait_keys[test_idx]
                        dp = data_points[trait_key]
                        
                        # Human
                        n_w, n_c = 90, 76 
                        c_w = round((dp["warm_percentage"]/100) * n_w)
                        c_c = round((dp["cold_percentage"]/100) * n_c)
                        chi_h, _ = get_chi2_2sample(c_w, n_w, c_c, n_c)
                        bf_h = calc_bf_chisq(chi_h, n_w + n_c)
                        # Direction: some central traits are Warm > Cold (1), some are null (0)
                        h_dir = 1 if dp["warm_percentage"] > dp["cold_percentage"] + 10 else 0
                        pi_h = calc_posteriors_3way(bf_h, h_dir, prior_odds=10.0 if h_dir != 0 else 0.1)
                        
                        # Agent
                        item_id = str(test_idx + 1)
                        vals_w = agent_raw.get(sub_study_id, {}).get("Warm", {}).get(item_id, [])
                        vals_c = agent_raw.get(sub_study_id, {}).get("Cold", {}).get(item_id, [])
                        
                        if vals_w and vals_c:
                            count_w = sum(1 for v in vals_w if v.upper().startswith('A'))
                            count_c = sum(1 for v in vals_c if v.upper().startswith('A'))
                            chi2_agent, p_val_agent = get_chi2_2sample(count_w, len(vals_w), count_c, len(vals_c))
                            bf_a = calc_bf_chisq(chi2_agent, len(vals_w) + len(vals_c))
                            a_dir = 1 if count_w > count_c else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"Warm: {count_w}/{len(vals_w)}, Cold: {count_c}/{len(vals_c)}"
                            contingency_agent = [[count_w, len(vals_w) - count_w], [count_c, len(vals_c) - count_c]]
                            contingency_human = [[c_w, n_w - c_w], [c_c, n_c - c_c]]
                        else:
                            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                    elif gt_study_id == "Experiment II":
                        dp = data_points["warm_vs_cold_choice"]
                        n_h = dp["n"]
                        c_h = round((dp["warm_percentage"]/100) * n_h)
                        chi_h, _ = get_chi2_1sample(c_h, n_h)
                        bf_h = calc_bf_chisq(chi_h, n_h)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        vals = agent_raw.get(sub_study_id, {}).get("Default", {}).get("1", [])
                        if vals:
                            count_a = sum(1 for v in vals if "warm" in v.lower())
                            chi2_agent, p_val_agent = get_chi2_1sample(count_a, len(vals))
                            bf_a = calc_bf_chisq(chi2_agent, len(vals))
                            a_dir = 1 if count_a > len(vals) * 0.5 else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"Warm choices: {count_a}/{len(vals)}"
                            contingency_agent = [[count_a, len(vals) - count_a], [round(len(vals) * 0.5), round(len(vals) * 0.5)]]
                            contingency_human = [[c_h, n_h - c_h], [round(n_h * 0.5), round(n_h * 0.5)]]
                        else:
                            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                    elif gt_study_id == "Experiment III":
                        trait_key = "generous" if test_idx == 0 else "humorous"
                        dp = data_points[trait_key]
                        n_p, n_b = 20, 26
                        c_p = round((dp["polite_percentage"]/100) * n_p)
                        c_b = round((dp["blunt_percentage"]/100) * n_b)
                        chi_h, _ = get_chi2_2sample(c_p, n_p, c_b, n_b)
                        bf_h = calc_bf_chisq(chi_h, n_p + n_b)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        vals_p = agent_raw.get(sub_study_id, {}).get("Polite", {}).get(trait_key, [])
                        vals_b = agent_raw.get(sub_study_id, {}).get("Blunt", {}).get(trait_key, [])
                        if vals_p and vals_b:
                            count_p = sum(1 for v in vals_p if v.upper().startswith('A'))
                            count_b = sum(1 for v in vals_b if v.upper().startswith('A'))
                            chi2_agent, p_val_agent = get_chi2_2sample(count_p, len(vals_p), count_b, len(vals_b))
                            bf_a = calc_bf_chisq(chi2_agent, len(vals_p) + len(vals_b))
                            a_dir = 1 if count_p > count_b else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"Polite: {count_p}/{len(vals_p)}, Blunt: {count_b}/{len(vals_b)}"
                            contingency_agent = [[count_p, len(vals_p) - count_p], [count_b, len(vals_b) - count_b]]
                            contingency_human = [[c_p, n_p - c_p], [c_b, n_b - c_b]]
                        else:
                            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                    elif gt_study_id == "Experiment IV":
                        dp = data_points["lowest_importance_rank"]
                        n_h = dp["n"]
                        c_h = round((dp["series_b_warm_percentage"]/100) * n_h)
                        chi_h, _ = get_chi2_1sample(c_h, n_h)
                        bf_h = calc_bf_chisq(chi_h, n_h)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        vals = agent_raw.get(sub_study_id, {}).get("Series B", {}).get("1", [])
                        if vals:
                            def is_low(v):
                                try: return int(re.search(r'\d+', v).group()) >= 6
                                except: return False
                            count_low = sum(1 for v in vals if is_low(v))
                            chi2_agent, p_val_agent = get_chi2_1sample(count_low, len(vals))
                            bf_a = calc_bf_chisq(chi2_agent, len(vals))
                            a_dir = 1 if count_low > len(vals) * 0.5 else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"Rank >= 6: {count_low}/{len(vals)}"
                            contingency_agent = [[count_low, len(vals) - count_low], [round(len(vals) * 0.5), round(len(vals) * 0.5)]]
                            contingency_human = [[c_h, n_h - c_h], [round(n_h * 0.5), round(n_h * 0.5)]]
                        else:
                            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                    elif gt_study_id == "Experiment V":
                        dp = data_points["serene" if test_idx == 0 else "cold_frigid_calculating"]
                        n_k, n_c = 38, 41
                        c_k = dp["kind_series_count"]
                        c_c = dp["cruel_series_count"]
                        chi_h, _ = get_chi2_2sample(c_k, n_k, c_c, n_c)
                        bf_h = calc_bf_chisq(chi_h, n_k + n_c)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        item_id = str(test_idx + 1)
                        vals_k = agent_raw.get(sub_study_id, {}).get("Kind", {}).get(item_id, [])
                        vals_c = agent_raw.get(sub_study_id, {}).get("Cruel", {}).get(item_id, [])
                        keywords = ["seren"] if test_idx == 0 else ["cold", "frigid", "calculat"]
                        if vals_k and vals_c:
                            count_k = sum(1 for v in vals_k if any(k in v.lower() for k in keywords))
                            count_c = sum(1 for v in vals_c if any(k in v.lower() for k in keywords))
                            chi2_agent, p_val_agent = get_chi2_2sample(count_k, len(vals_k), count_c, len(vals_c))
                            bf_a = calc_bf_chisq(chi2_agent, len(vals_k) + len(vals_c))
                            a_dir = 1 if count_k > count_c else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"Kind: {count_k}/{len(vals_k)}, Cruel: {count_c}/{len(vals_c)}"
                            contingency_agent = [[count_k, len(vals_k) - count_k], [count_c, len(vals_c) - count_c]]
                            contingency_human = [[c_k, n_k - c_k], [c_c, n_c - c_c]]
                        else:
                            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                    elif gt_study_id in ["Experiment VI", "Experiment VII"]:
                        trait_keys = list(data_points.keys())
                        trait_key = trait_keys[test_idx]
                        dp = data_points[trait_key]
                        n_a = 34 if gt_study_id == "Experiment VI" else 46
                        n_b = 24 if gt_study_id == "Experiment VI" else 53
                        c_a = round((dp["a_percentage"]/100) * n_a)
                        c_b = round((dp["b_percentage"]/100) * n_b)
                        chi_h, _ = get_chi2_2sample(c_a, n_a, c_b, n_b)
                        bf_h = calc_bf_chisq(chi_h, n_a + n_b)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        vals_a = agent_raw.get(sub_study_id, {}).get("Positive First", {}).get(trait_key if gt_study_id == "Experiment VI" else "1", [])
                        vals_b = agent_raw.get(sub_study_id, {}).get("Negative First", {}).get(trait_key if gt_study_id == "Experiment VI" else "1", [])
                        if vals_a and vals_b:
                            count_a = sum(1 for v in vals_a if v.upper().startswith('A'))
                            count_b = sum(1 for v in vals_b if v.upper().startswith('A'))
                            chi2_agent, p_val_agent = get_chi2_2sample(count_a, len(vals_a), count_b, len(vals_b))
                            bf_a = calc_bf_chisq(chi2_agent, len(vals_a) + len(vals_b))
                            a_dir = 1 if count_a > count_b else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"PosFirst: {count_a}/{len(vals_a)}, NegFirst: {count_b}/{len(vals_b)}"
                            contingency_agent = [[count_a, len(vals_a) - count_a], [count_b, len(vals_b) - count_b]]
                            contingency_human = [[c_a, n_a - c_a], [c_b, n_b - c_b]]
                        else:
                            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                    elif gt_study_id == "Experiment VIII":
                        dp = data_points["difficulty_forming_impression"]
                        n1, n2 = 52, 24
                        c1, c2 = dp["group_1_count"], dp["group_2_count"]
                        chi_h, _ = get_chi2_2sample(c1, n1, c2, n2)
                        bf_h = calc_bf_chisq(chi_h, n1 + n2)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        vals1 = agent_raw.get(sub_study_id, {}).get("Broken", {}).get("1", [])
                        vals2 = agent_raw.get(sub_study_id, {}).get("Continuous", {}).get("1", [])
                        if vals1 and vals2:
                            count1 = sum(1 for v in vals1 if v.upper().startswith('A')) # A=Yes/Difficult
                            count2 = sum(1 for v in vals2 if v.upper().startswith('A'))
                            chi2_agent, p_val_agent = get_chi2_2sample(count1, len(vals1), count2, len(vals2))
                            bf_a = calc_bf_chisq(chi2_agent, len(vals1) + len(vals2))
                            a_dir = 1 if count1 > count2 else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"Broken: {count1}/{len(vals1)}, Cont: {count2}/{len(vals2)}"
                            contingency_agent = [[count1, len(vals1) - count1], [count2, len(vals2) - count2]]
                            contingency_human = [[c1, n1 - c1], [c2, n2 - c2]]
                        else:
                            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                    elif gt_study_id == "Experiment IX":
                        trait_keys = list(data_points.keys())
                        trait_key = trait_keys[test_idx]
                        dp = data_points[trait_key]
                        n_h = dp["n"]
                        c_h = round((dp["percentage"]/100) * n_h)
                        chi_h, _ = get_chi2_1sample(c_h, n_h)
                        bf_h = calc_bf_chisq(chi_h, n_h)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        vals = agent_raw.get(sub_study_id, {}).get("Default", {}).get(trait_key, [])
                        if vals:
                            count_a = sum(1 for v in vals if v.upper().startswith('A'))
                            chi2_agent, p_val_agent = get_chi2_1sample(count_a, len(vals))
                            bf_a = calc_bf_chisq(chi2_agent, len(vals))
                            a_dir = 1 if count_a > len(vals) * 0.5 else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"Positive choices: {count_a}/{len(vals)}"
                            contingency_agent = [[count_a, len(vals) - count_a], [round(len(vals) * 0.5), round(len(vals) * 0.5)]]
                            contingency_human = [[c_h, n_h - c_h], [round(n_h * 0.5), round(n_h * 0.5)]]
                        else:
                            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                    elif gt_study_id == "Experiment IXa":
                        trait_keys = list(data_points.keys())
                        trait_key = trait_keys[test_idx]
                        dp = data_points[trait_key]
                        n_w, n_c = 22, 33
                        c_w = round((dp["warm_percentage"]/100) * n_w)
                        c_c = round((dp["cold_percentage"]/100) * n_c)
                        chi_h, _ = get_chi2_2sample(c_w, n_w, c_c, n_c)
                        bf_h = calc_bf_chisq(chi_h, n_w + n_c)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        item_id = str(test_idx + 1)
                        vals_w = agent_raw.get(sub_study_id, {}).get("Warm", {}).get(item_id, [])
                        vals_c = agent_raw.get(sub_study_id, {}).get("Cold", {}).get(item_id, [])
                        if vals_w and vals_c:
                            count_w = sum(1 for v in vals_w if v.upper().startswith('A'))
                            count_c = sum(1 for v in vals_c if v.upper().startswith('A'))
                            chi2_agent, p_val_agent = get_chi2_2sample(count_w, len(vals_w), count_c, len(vals_c))
                            bf_a = calc_bf_chisq(chi2_agent, len(vals_w) + len(vals_c))
                            a_dir = 1 if count_w > count_c else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"Warm: {count_w}/{len(vals_w)}, Cold: {count_c}/{len(vals_c)}"
                            contingency_agent = [[count_w, len(vals_w) - count_w], [count_c, len(vals_c) - count_c]]
                            contingency_human = [[c_w, n_w - c_w], [c_c, n_c - c_c]]
                        else:
                            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                    elif gt_study_id == "Experiment X":
                        trait_key = "set_1_resembles_set_3" if test_idx == 0 else "set_2_resembles_set_4"
                        dp = data_points[trait_key]
                        n_h = dp["n"]
                        c_h = round((dp["percentage"]/100) * n_h)
                        chi_h, _ = get_chi2_1sample(c_h, n_h)
                        bf_h = calc_bf_chisq(chi_h, n_h)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        item_id = str(test_idx + 1)
                        vals = agent_raw.get(sub_study_id, {}).get("Default", {}).get(item_id, [])
                        if vals:
                            count_a = sum(1 for v in vals if v.upper().startswith('A'))
                            chi2_agent, p_val_agent = get_chi2_1sample(count_a, len(vals))
                            bf_a = calc_bf_chisq(chi2_agent, len(vals))
                            a_dir = 1 if count_a > len(vals) * 0.5 else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"Aligned choices: {count_a}/{len(vals)}"
                            contingency_agent = [[count_a, len(vals) - count_a], [round(len(vals) * 0.5), round(len(vals) * 0.5)]]
                            contingency_human = [[c_h, n_h - c_h], [round(n_h * 0.5), round(n_h * 0.5)]]
                        else:
                            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}

                except Exception as e:
                    reason = f"Error: {str(e)}"
                    pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                
                pas = float(calc_pas(pi_h, pi_a))
                t_weight = test_weights.get((fid, test_name), 1.0)
                
                # Create test result dict
                test_result = {
                    "study_id": gt_study_id,
                    "sub_study_id": sub_study_id,
                    "finding_id": fid,
                    "test_name": test_name,
                    "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
                    "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
                    "pi_human_3way": pi_h,
                    "pi_agent_3way": pi_a,
                    "pas": pas,
                    "test_weight": float(t_weight),
                    "pi_human_source": reported,
                    "agent_reason": reason,
                    "statistical_test_type": "Chi-square",
                    "human_test_statistic": f"{chi_h:.2f}" if chi_h is not None else "",
                    "agent_test_statistic": f"{chi2_agent:.2f}" if chi2_agent is not None else ""
                }
                
                # Add statistical replication fields with contingency tables
                add_statistical_replication_fields(
                    test_result, test, p_val_agent, chi2_agent, "chi-square",
                    contingency_agent=contingency_agent,
                    contingency_human=contingency_human
                )
                
                test_results.append(test_result)

    # 4. Aggregation
    finding_results = []
    finding_ids = sorted(list(set(tr["finding_id"] for tr in test_results)))
    for fid in finding_ids:
        fid_tests = [tr for tr in test_results if tr["finding_id"] == fid]
        sub_id = fid_tests[0]["sub_study_id"]
        
        total_weighted_pas = sum(tr["pas"] * tr["test_weight"] for tr in fid_tests)
        total_weight = sum(tr["test_weight"] for tr in fid_tests)
        finding_score = total_weighted_pas / total_weight if total_weight > 0 else 0.5
        
        finding_results.append({
            "sub_study_id": sub_id,
            "finding_id": fid,
            "finding_score": float(finding_score),
            "finding_weight": float(finding_weights.get(fid, 1.0)),
            "n_tests": len(fid_tests)
        })

    total_weighted_finding_score = sum(fr["finding_score"] * fr["finding_weight"] for fr in finding_results)
    total_finding_weight = sum(fr["finding_weight"] for fr in finding_results)
    overall_score = total_weighted_finding_score / total_finding_weight if total_finding_weight > 0 else 0.5

    substudy_results = []
    substudy_ids = sorted(list(set(fr["sub_study_id"] for fr in finding_results)))
    for sid in substudy_ids:
        sid_findings = [fr for fr in finding_results if fr["sub_study_id"] == sid]
        score = np.mean([fr["finding_score"] for fr in sid_findings]) if sid_findings else 0.5
        substudy_results.append({
            "sub_study_id": sid,
            "substudy_score": float(score),
            "n_findings": len(sid_findings)
        })

    return {
        "score": float(overall_score),
        "substudy_results": substudy_results,
        "finding_results": finding_results,
        "test_results": test_results
    }