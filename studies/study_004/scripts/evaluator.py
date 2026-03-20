import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from stats_lib import (
    calc_bf_t, calc_bf_chisq, calc_bf_anova, 
    calc_posteriors_3way, calc_pas,
    parse_p_value_from_reported,
    get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """Parse Qk=<value> or Qk: <value> or Qk.n=<value> format from agent response text."""
    results = {}
    # Matches Q1=A, Q1: 36, Q1.1=50, Q10=0.85 etc.
    # Supports both = and : separators
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*[:=]\s*([^,\n\s]+)")
    for k, v in pattern.findall(response_text):
        # Clean the value (remove trailing punctuation often added by LLMs)
        # Handle percentage signs (common in Study 6)
        clean_v = v.strip().rstrip('.,;)')
        if clean_v.endswith('%'):
            clean_v = clean_v[:-1]  # Remove % symbol
        results[k.strip()] = clean_v
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_004: 从items中提取q_idx_choice或q_indices（用于子问题，如Q1.1, Q1.2, ...）
    """
    required = set()
    items = trial_info.get("items", [])
    
    for item in items:
        # 检查是否有 q_indices（用于子问题，如 Q1.1, Q1.2, ..., Q1.11）
        q_indices = item.get("q_indices", [])
        if q_indices:
            # 如果存在 q_indices，使用它们
            for q_idx in q_indices:
                if isinstance(q_idx, str) and q_idx.startswith("Q"):
                    required.add(q_idx)
                else:
                    required.add(f"Q{q_idx}")
        else:
            # 否则检查 q_idx_choice 或 q_idx
            q_idx = item.get("q_idx_choice") or item.get("q_idx")
            if q_idx:
                if isinstance(q_idx, str) and q_idx.startswith("Q"):
                    required.add(q_idx)
                else:
                    required.add(f"Q{q_idx}")
    
    # 如果没有找到任何Q编号，使用索引推断（向后兼容）
    if not required:
        for idx, _ in enumerate(items):
            required.add(f"Q{idx + 1}")
    
    return required

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the agent's performance on Study 004 (Subjective Probability / Representativeness).
    Calculates Bayesian Alignment Score (PAS) for multiple findings.
    """
    # 1. Load ground truth
    study_id = "study_004"
    study_dir = Path(__file__).resolve().parent.parent / "source"
    gt_path = study_dir / "ground_truth.json"
    
    if not gt_path.exists():
        # Fallback for different environment structures
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

    # 2. Extract and group agent data by sub-study
    agent_data = {
        "study_1_proportion": [],
        "study_1_randomness": [],
        "study_2_programs": [],
        "study_5_marbles": [],
        "study_6_sampling_distributions": [],
        "study_7_ordinal": [],
        "study_9_heights": []
    }

    for participant in results.get("individual_data", []):
        for response in participant.get("responses", []):
            response_text = response.get("response_text", "")
            trial_info = response.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            items = trial_info.get("items", [])
            
            # If items are missing from trial_info, load from materials file
            if not items and sub_id:
                items = load_materials(sub_id)
            
            parsed = parse_agent_responses(response_text)
            if not parsed:
                continue

            if sub_id in agent_data:
                agent_data[sub_id].append({
                    "parsed": parsed,
                    "items": items,
                    "condition": trial_info.get("condition", {})
                })

    test_results = []

    def add_sign_test(finding_id, sub_id, n_h, k_h, agent_successes, agent_total, test_name, reported, scenario=None):
        """Helper to calculate PAS for a sign test comparison."""
        # Get test_gt from ground truth
        test_gt = {}
        for study in ground_truth.get("studies", []):
            for finding in study.get("findings", []):
                if finding.get("finding_id") == finding_id:
                    statistical_tests = finding.get("statistical_tests", [])
                    if statistical_tests:
                        # Find matching test by name or use first one
                        for test in statistical_tests:
                            if test.get("test_name") == test_name or test_name in test.get("test_name", ""):
                                test_gt = test
                                break
                        if not test_gt and statistical_tests:
                            test_gt = statistical_tests[0]
                    break
        
        # Human Pi (3-way)
        chi2_h = 4 * (k_h - n_h/2)**2 / n_h if n_h > 0 else 0
        bf_h = calc_bf_chisq(chi2_h, n_h)
        h_dir = 1 if k_h > n_h/2 else -1
        pi_h = calc_posteriors_3way(bf_h, h_dir, prior_odds=10.0)
        
        # Agent Pi (3-way)
        p_val_agent = None
        chi2_agent = None
        if agent_total >= 3:
            chi2_agent = 4 * (agent_successes - agent_total/2)**2 / agent_total
            from scipy.stats import chi2 as chi2_dist
            p_val_agent = 1 - chi2_dist.cdf(chi2_agent, 1) if chi2_agent > 0 else 1.0
            
            bf_a = calc_bf_chisq(chi2_agent, agent_total)
            a_dir = 1 if agent_successes > agent_total/2 else -1
            pi_a = calc_posteriors_3way(bf_a, a_dir)
            reason = f"k={agent_successes}, n={agent_total}, chi2={chi2_agent:.2f}"
        else:
            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
            reason = f"Insufficient data (n={agent_total})"
            
        pas = float(calc_pas(pi_h, pi_a))
        # Get test weight
        test_weight = float(test_weights.get((finding_id, test_name), 1.0))
        
        # Create test result dict
        test_result = {
            "study_id": "Study 004",
            "sub_study_id": sub_id,
            "finding_id": finding_id,
            "test_weight": test_weight,
            "test_name": test_name,
            "scenario": scenario or test_name,
            "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
            "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
            "pi_human_3way": pi_h,
            "pi_agent_3way": pi_a,
            "pas": pas,
            "pi_human_source": reported,
            "agent_reason": reason,
            "statistical_test_type": "Sign test",
            "human_test_statistic": f"k={k_h}, n={n_h}",
            "agent_test_statistic": f"k={agent_successes}, n={agent_total}" if agent_total >= 3 else ""
        }
        
        # Build contingency tables for chi-square (sign test)
        # For sign test: [successes, failures] for each group
        contingency_agent = None
        contingency_human = None
        if agent_total >= 3:
            agent_failures = agent_total - agent_successes
            contingency_agent = [[agent_successes, agent_failures], [agent_total/2, agent_total/2]]  # Expected values
        if n_h > 0:
            human_failures = n_h - k_h
            contingency_human = [[k_h, human_failures], [n_h/2, n_h/2]]  # Expected values
        
        # Add statistical replication fields with contingency tables
        add_statistical_replication_fields(
            test_result, test_gt, p_val_agent, chi2_agent, "chi-square",
            contingency_agent=contingency_agent,
            contingency_human=contingency_human
        )
        
        test_results.append(test_result)

    # --- F1: Study 1 Proportion (Birth Sequences) ---
    # Finding: BGBBBB is judged less likely than the population ratio (0.50).
    # Human: 75/92 judged it less likely than a representative sequence (72% frequency given).
    successes = 0
    total = 0
    for entry in agent_data["study_1_proportion"]:
        for item in entry["items"]:
            if item["id"] == "estimate_bgbbbb":
                q_idx = item.get("q_idx")
                val = entry["parsed"].get(q_idx)
                try:
                    # Human reference was 72. If agent estimate < 72, it follows the bias.
                    if float(val) < 72:
                        successes += 1
                    total += 1
                except (ValueError, TypeError): continue
    add_sign_test("F1", "study_1_proportion", 92, 75, successes, total, "Sign test", "p < .01, n=92", "Birth Sequence Proportion")

    # --- F2: Study 1 Randomness (Birth Sequences) ---
    # Finding: Disordered sequences (GBBGBG) judged more likely than ordered (BBBGGG).
    # Material: Multiple choice - A) BBBGGG, B) GBBGBG. Finding: B is more likely.
    successes = 0
    total = 0
    for entry in agent_data["study_1_randomness"]:
        for item in entry["items"]:
            if item["id"] == "compare_randomness":
                q_idx = item.get("q_idx")
                val = entry["parsed"].get(q_idx)
                if val:
                    val_upper = str(val).upper().strip()
                    # Material mapping: A=BBBGGG (ordered), B=GBBGBG (disordered). Finding: B is more likely.
                    if val_upper == 'B':
                        successes += 1
                    if val_upper in ['A', 'B']:
                        total += 1
    add_sign_test("F2", "study_1_randomness", 92, 75, successes, total, "Sign test", "p < .01, n=92", "Birth Sequence Randomness")

    # --- F3: Study 2 Programs (Majority-Minority) ---
    # Finding: Participants choose the more representative population (majority status).
    successes = 0
    total = 0
    for entry in agent_data["study_2_programs"]:
        for item in entry["items"]:
            if item["id"] == "program_guess":
                q_idx = item.get("q_idx")
                val = entry["parsed"].get(q_idx)
                # Material mapping: A=Program A (65% boys), B=Program B (45% boys). Finding: A.
                if val == 'A':
                    successes += 1
                total += 1
    add_sign_test("F3", "study_2_programs", 89, 67, successes, total, "Sign test", "p < .01, n=89", "Majority-Minority Programs")

    # --- F5: Study 5 Marbles (Randomness) ---
    # Finding: Disordered distributions judged more likely than perfectly symmetric ones.
    successes = 0
    total = 0
    for entry in agent_data["study_5_marbles"]:
        for item in entry["items"]:
            if item["id"] == "marble_distribution":
                q_idx = item.get("q_idx")
                val = entry["parsed"].get(q_idx)
                # Material mapping: A=Distribution I (4,4,5,4,3), B=Distribution II (4,4,4,4,4). Finding: A.
                if val == 'A':
                    successes += 1
                total += 1
    add_sign_test("F5", "study_5_marbles", 52, 36, successes, total, "Sign test", "p < .01, n=52", "Marble Distribution Randomness")

    # --- F7: Study 9 Heights (Posterior Odds) ---
    # Finding: Confidence determined by sample mean representativeness, not sample size.
    # Case (i) Single 5'10" person > Case (ii) Six 5'8" persons.
    successes = 0
    total = 0
    for entry in agent_data["study_9_heights"]:
        odds_i, odds_ii = None, None
        for item in entry["items"]:
            q_idx = item.get("q_idx")
            val = entry["parsed"].get(q_idx)
            try:
                if item["id"] == "odds_case_i": odds_i = float(val)
                if item["id"] == "odds_case_ii": odds_ii = float(val)
            except (ValueError, TypeError): continue
        if odds_i is not None and odds_ii is not None:
            if odds_i > odds_ii:
                successes += 1
            total += 1
    add_sign_test("F7", "study_9_heights", 115, 86, successes, total, "Median test", "p < .01, n=115", "Height Posterior Odds")

    # --- F6: Study 6 Subjective Sampling Distributions (N-Invariance) ---
    # Finding: Distributions for N=10, 100, 1000 are identical.
    # Human study: N=1500 total, distributed across 9 conditions (3 populations × 3 sample sizes)
    # Expected: ~500 per sample size group (166 per condition × 3 populations)
    # If agent n=1500, we should get ~500 per group, not just 39!
    means_by_n = {10: [], 100: [], 1000: []}
    midpoints = np.array([2.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97.5])
    total_entries = len(agent_data["study_6_sampling_distributions"])
    valid_entries = 0
    for entry in agent_data["study_6_sampling_distributions"]:
        n_val = entry["condition"].get("sample_size")
        q_indices = []
        for item in entry["items"]:
            if item["id"] == "sampling_distribution_estimate":
                # Try to get q_indices from item, or infer from structure
                q_indices = item.get("q_indices", [])
                if not q_indices:
                    # Infer q_indices: Q1.1, Q1.2, ..., Q1.11 (11 options)
                    # Find which Q number this item should be (usually Q1 for first item)
                    item_index = entry["items"].index(item) if item in entry["items"] else 0
                    base_q = item_index + 1
                    num_options = len(item.get("options", []))
                    if num_options == 0:
                        num_options = 11  # Default to 11 for sampling distribution
                    q_indices = [f"Q{base_q}.{i+1}" for i in range(num_options)]
                break  # Only process the first matching item
        
        probs = []
        for q in q_indices:
            try:
                val = entry["parsed"].get(q, '0')
                # Handle percentage signs if parser didn't catch them
                if isinstance(val, str) and val.endswith('%'):
                    val = val[:-1]
                probs.append(float(val))
            except (ValueError, TypeError):
                probs.append(0)
        
        if len(probs) == 11 and sum(probs) > 0:
            probs_norm = np.array(probs) / sum(probs)
            mean_dist = np.sum(probs_norm * midpoints)
            if n_val in means_by_n:
                means_by_n[n_val].append(mean_dist)
                valid_entries += 1

    all_groups = [means_by_n[10], means_by_n[100], means_by_n[1000]]
    # Debug: Log if we're losing too many responses
    # If total_entries is 1500 but valid_entries << 1500, there's a parsing/filtering issue
    if total_entries > 0 and valid_entries / total_entries < 0.5:
        # Most responses are being filtered - likely parsing issue
        pass  # Could add logging here if needed
    
    # Get test_gt for F6
    test_gt_f6 = {}
    for study in ground_truth.get("studies", []):
        for finding in study.get("findings", []):
            if finding.get("finding_id") == "F6":
                statistical_tests = finding.get("statistical_tests", [])
                if statistical_tests:
                    test_gt_f6 = statistical_tests[0]
                break
    
    p_val_agent_f6 = None
    f_stat_agent = None
    if all(len(g) >= 3 for g in all_groups):
        f_stat_agent, p_val_agent_f6 = stats.f_oneway(*all_groups)
        n_total = sum(len(g) for g in all_groups)
        bf_a = calc_bf_anova(f_stat_agent, 2, n_total - 3, n_total)
        # For agent finding null (no difference), any significant difference is a failure.
        # But we treat it as an omnibus F-test. a_dir=0 for ANOVA null tests.
        pi_a = calc_posteriors_3way(bf_a, 0)
        
        # Human Finding: Indistinguishable (Very low F, high N=1500)
        bf_h = calc_bf_anova(0.1, 2, 1497, 1500)
        pi_h = calc_posteriors_3way(bf_h, 0, prior_odds=0.1) # Prior odds for null finding
        pas = float(calc_pas(pi_h, pi_a))
        
        # Create test result dict
        test_result_f6 = {
            "study_id": "Study 004",
            "sub_study_id": "study_6_sampling_distributions",
            "finding_id": "F6",
            "test_name": "ANOVA (N-Invariance)",
            "scenario": "N-Invariance",
            "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
            "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
            "pi_human_3way": pi_h,
            "pi_agent_3way": pi_a,
            "pas": pas,
            "pi_human_source": "Indistinguishable (N=1500)",
            "agent_reason": f"F={f_stat_agent:.2f}, n={n_total}",
            "statistical_test_type": "ANOVA",
            "human_test_statistic": "0.1",
            "agent_test_statistic": f"{f_stat_agent:.2f}"
        }
        
        # Add statistical replication fields
        add_statistical_replication_fields(test_result_f6, test_gt_f6, p_val_agent_f6, f_stat_agent, "f-test")
        
        test_results.append(test_result_f6)

    # --- Study 7: Hospital (Ordinal Judgments) ---
    # Finding: "Same Likelihood" is the modal choice across sample sizes.
    successes = 0
    total = 0
    for entry in agent_data["study_7_ordinal"]:
        for item in entry["items"]:
            if item["id"] == "hospital_problem":
                q_idx = item.get("q_idx")
                val = entry["parsed"].get(q_idx)
                # Material mapping: A=Larger, B=Smaller, C=Same. Finding: C.
                if val == 'C':
                    successes += 1
                total += 1
    if total >= 3:
        # Ground truth shows n=97 for Study 7, with 28 choosing "Same" for Hospital problem
        # But the data table shows 28/50 for Hospital, so using n=50, k=28 from the table
        add_sign_test("F8", "study_7_ordinal", 50, 28, successes, total, "Sign test (Hospital)", "Modal=Same (28/50)", "Hospital Ordinal Judgment")

    # 4. Two-level Weighted Aggregation
    # Add test_weight to each test result if not already present
    for tr in test_results:
        if "test_weight" not in tr:
            finding_id = tr.get("finding_id")
            test_name = tr.get("test_name", "")
            tr["test_weight"] = float(test_weights.get((finding_id, test_name), 1.0))
    
    if not test_results:
        return {"score": 0.5, "substudy_results": [], "finding_results": [], "test_results": []}

    # Level 1: Aggregate tests into findings (weighted by test weights)
    finding_scores = {}
    for t in test_results:
        fid = t["finding_id"]
        if fid not in finding_scores:
            finding_scores[fid] = []
        finding_scores[fid].append(t)
    
    finding_results = []
    for fid, tests in finding_scores.items():
        # Weighted average: Σ (PAS * weight) / Σ weights
        total_weighted_pas = sum(t["pas"] * t.get("test_weight", 1.0) for t in tests)
        total_weight = sum(t.get("test_weight", 1.0) for t in tests)
        finding_score = total_weighted_pas / total_weight if total_weight > 0 else 0.5
        
        finding_weight = finding_weights.get(fid, 1.0)
        finding_results.append({
            "sub_study_id": next(t["sub_study_id"] for t in tests),
            "finding_id": fid,
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