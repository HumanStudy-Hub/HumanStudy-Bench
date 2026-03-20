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
    calc_bf_r, 
    calc_bf_chisq, 
    calc_posteriors_3way,
    calc_pas,
    parse_p_value_from_reported,
    get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """Parse Qk=<value> or Qk.n=<value> format from agent response.
    
    Handles multiple formats:
    - Q1=Yes, Q2=4 (equals sign)
    - Q1: Yes, Q2: 4 (colon)
    - Just "Yes" or "No" (maps to Q1 if no Q prefix found)
    - Just numbers (maps to Q1, Q2, etc. based on position)
    """
    results = {}
    
    # First, try to match explicit Q formats (Q1=Yes, Q1: Yes, etc.)
    # Matches Q1=Yes, Q2=4, Q1.1=No etc. (with equals sign)
    pattern_equals = re.compile(r"(Q\d+(?:\.\d+)?)\s*=\s*([^,\n\s]+)")
    for k, v in pattern_equals.findall(response_text):
        results[k.strip()] = v.strip()
    
    # Also match Q1: Yes, Q2: 4 (with colon) - but only if not already found
    # This handles models like mistral nemo that use colon format
    pattern_colon = re.compile(r"(Q\d+(?:\.\d+)?)\s*:\s*([^,\n\s]+)")
    for k, v in pattern_colon.findall(response_text):
        k = k.strip()
        # Only add if not already found (equals takes precedence)
        if k not in results:
            results[k] = v.strip()
    
    # If no Q prefixes found, try to extract values directly
    # This handles cases where model just responds "Yes" or "No" or numbers
    if not results:
        # Clean the text - remove extra whitespace but preserve structure
        text = response_text.strip()
        
        # Split by newlines to handle multi-line responses
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Try to find Yes/No (case insensitive) - usually first
        yes_no_match = re.search(r'\b(Yes|No)\b', text, re.IGNORECASE)
        if yes_no_match:
            results["Q1"] = yes_no_match.group(1).capitalize()  # Normalize to Yes/No
        
        # Try to find numbers (0-6 for blame/praise scale is most common)
        # Look for standalone numbers, prioritizing 0-6 range
        numbers_found = []
        
        # First, look for single digits 0-6 (blame/praise scale)
        for match in re.finditer(r'\b([0-6])\b', text):
            num = match.group(1)
            if num not in numbers_found:
                numbers_found.append(num)
        
        # If no 0-6 found, look for any number
        if not numbers_found:
            for match in re.finditer(r'\b(\d+)\b', text):
                num = match.group(1)
                if num not in numbers_found:
                    numbers_found.append(num)
        
        # Map numbers to Q keys
        # If we found Yes/No, numbers go to Q2, Q3, etc.
        # Otherwise, first number goes to Q1
        start_idx = 2 if "Q1" in results else 1
        for i, num in enumerate(numbers_found):
            q_key = f"Q{start_idx + i}"
            if q_key not in results:
                results[q_key] = num
    
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    Extract all required Q numbers from trial_info.
    Study_005: Q numbers are based on item index (Q1, Q2, Q3...)
    """
    required = set()
    items = trial_info.get("items", [])
    
    # Study_005 uses simple Q{idx+1} numbering based on item index
    for idx, item in enumerate(items):
        required.add(f"Q{idx + 1}")
    
    return required

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates Study 005 (Knobe 2003 - The Knobe Effect).
    Calculates PAS for intentionality asymmetries, praise/blame asymmetries, and correlations.
    """
    # 1. Load ground truth and metadata
    study_id = "study_005"
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
    # Structure: sub_id -> list of participant dicts
    processed_data = {
        "experiment_1_harm": [],
        "experiment_1_help": [],
        "experiment_2_harm": [],
        "experiment_2_help": []
    }

    for participant in results.get("individual_data", []):
        # Handle both nested structure (responses list) and flat structure (single response)
        responses = participant.get("responses", [])
        if not responses and "trial_info" in participant:
            # Flat structure: participant itself is the response
            responses = [participant]
        
        for response in responses:
            trial_info = response.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            if sub_id not in processed_data:
                continue
                
            parsed = parse_agent_responses(response.get("response_text", ""))
            items = trial_info.get("items", [])
            
            # If items are missing from trial_info, load from materials file
            if not items and sub_id:
                items = load_materials(sub_id)
            
            p_data = {}
            for i, item in enumerate(items):
                q_key = f"Q{i+1}"
                val = parsed.get(q_key)
                if val:
                    # Map item IDs to data - materials use "id" field, not "item_id"
                    item_id = item.get("id") or item.get("item_id", "")
                    if "intentionality" in item_id.lower():
                        # Convert Yes/No to 1/0
                        p_data["intentional"] = 1 if val.lower().startswith("yes") else 0
                    elif "blame" in item_id.lower() or "praise" in item_id.lower():
                        try:
                            p_data["moral_score"] = float(val)
                        except ValueError:
                            pass
            
            if p_data:
                processed_data[sub_id].append(p_data)

    test_results = []

    # Helper to calculate stats for specific findings
    # Finding 1 & 3: Chi-square (Intentionality asymmetry)
    def run_chi2_test(harm_sub, help_sub, finding_id, human_chi, human_n, study_label):
        harm_ints = [d["intentional"] for d in processed_data[harm_sub] if "intentional" in d]
        help_ints = [d["intentional"] for d in processed_data[help_sub] if "intentional" in d]
        
        bf_h = calc_bf_chisq(human_chi, human_n, df=1)
        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0) # Human expected direction is match
        
        if len(harm_ints) > 0 and len(help_ints) > 0:
            table = [
                [sum(harm_ints), len(harm_ints) - sum(harm_ints)],
                [sum(help_ints), len(help_ints) - sum(help_ints)]
            ]
            
            harm_all_same = len(set(harm_ints)) <= 1
            help_all_same = len(set(help_ints)) <= 1
            both_groups_same = harm_all_same and help_all_same and (set(harm_ints) == set(help_ints))
            
            if both_groups_same:
                pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                reason = f"Zero variance (all={harm_ints[0] if harm_ints else 'N/A'}), supports H0"
                return pi_h, pi_a, reason, None, None
            else:
                try:
                    chi2, p_val, _, _ = stats.chi2_contingency(table)
                    n_agent = len(harm_ints) + len(help_ints)
                    bf_a = calc_bf_chisq(chi2, n_agent, df=1)
                    
                    # Direction: Harm_Yes_rate > Help_Yes_rate
                    rate_harm = sum(harm_ints) / len(harm_ints)
                    rate_help = sum(help_ints) / len(help_ints)
                    a_dir = 1 if rate_harm > rate_help else -1
                    pi_a = calc_posteriors_3way(bf_a, a_dir)
                    
                    reason = f"chi2={chi2:.2f}, n={n_agent}, table={table}"
                    return pi_h, pi_a, reason, p_val, chi2
                except Exception as e:
                    pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                    reason = f"Chi-square calculation failed: {str(e)}"
                    return pi_h, pi_a, reason, None, None
        else:
            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
            reason = "Insufficient data"
            return pi_h, pi_a, reason, None, None

    # Finding 2: t-test (Blame vs Praise)
    def run_t_test_moral():
        # Combine Exp 1 and Exp 2 moral scores if available
        # Based on materials, only Exp 1 has blame/praise items
        harm_moral = [d["moral_score"] for sub in ["experiment_1_harm", "experiment_2_harm"] 
                      for d in processed_data[sub] if "moral_score" in d]
        help_moral = [d["moral_score"] for sub in ["experiment_1_help", "experiment_2_help"] 
                      for d in processed_data[sub] if "moral_score" in d]
        
        # GT: t(120)=8.4, df=120 means n1+n2-2=120, so n1+n2=122
        # Assuming equal groups: n1=61, n2=61
        bf_h = calc_bf_t(8.4, 61, n2=61, independent=True)
        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
        
        p_val_agent = None
        t_stat_agent = None
        if len(harm_moral) > 2 and len(help_moral) > 2:
            t_stat_agent, p_val_agent = stats.ttest_ind(harm_moral, help_moral)
            bf_a = calc_bf_t(t_stat_agent, len(harm_moral), len(help_moral), independent=True)
            a_dir = 1 if t_stat_agent > 0 else -1
            pi_a = calc_posteriors_3way(bf_a, a_dir)
            reason = f"t={t_stat_agent:.2f}, n1={len(harm_moral)}, n2={len(help_moral)}"
        else:
            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
            reason = "Insufficient data"
        return pi_h, pi_a, reason, p_val_agent, t_stat_agent

    # Finding 4: Correlation
    def run_correlation():
        # Correlation between intentionality and moral score
        ints = []
        moral = []
        for sub in processed_data:
            for d in processed_data[sub]:
                if "intentional" in d and "moral_score" in d:
                    ints.append(d["intentional"])
                    moral.append(d["moral_score"])
        
        # GT: r(120)=0.53
        bf_h = calc_bf_r(0.53, 122)
        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
        
        if len(ints) > 5:
            # Check for zero variance before computing correlation
            ints_variance = len(set(ints)) > 1
            moral_variance = len(set(moral)) > 1
            
            if not ints_variance or not moral_variance:
                pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                reason = f"Zero variance (ints={len(set(ints))} unique, moral={len(set(moral))} unique)"
                return pi_h, pi_a, reason, None, None
            
            try:
                r_val, p_val = stats.pearsonr(ints, moral)
                bf_a = calc_bf_r(r_val, len(ints))
                a_dir = 1 if r_val > 0 else -1
                pi_a = calc_posteriors_3way(bf_a, a_dir)
                reason = f"r={r_val:.2f}, n={len(ints)}"
                return pi_h, pi_a, reason, p_val, r_val
            except Exception as e:
                pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                reason = f"Correlation calculation failed: {str(e)}"
                return pi_h, pi_a, reason, None, None
        else:
            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
            reason = "Insufficient data"
            return pi_h, pi_a, reason, None, None

    # 3. Process each finding
    # F1: Exp 1 Chi-Square
    pi_h1, pi_a1, r1, p_val_f1, chi2_f1 = run_chi2_test("experiment_1_harm", "experiment_1_help", "F1", 27.2, 78, "Exp 1")
    # Get test_gt for F1
    test_gt_f1 = {}
    for study in ground_truth.get("studies", []):
        for finding in study.get("findings", []):
            if finding.get("finding_id") == "F1":
                statistical_tests = finding.get("statistical_tests", [])
                if statistical_tests:
                    test_gt_f1 = statistical_tests[0]
                break
    
    # Extract contingency tables for F1
    harm_ints_f1 = [d["intentional"] for d in processed_data["experiment_1_harm"] if "intentional" in d]
    help_ints_f1 = [d["intentional"] for d in processed_data["experiment_1_help"] if "intentional" in d]
    contingency_agent_f1 = None
    if len(harm_ints_f1) > 0 and len(help_ints_f1) > 0:
        contingency_agent_f1 = [
            [sum(harm_ints_f1), len(harm_ints_f1) - sum(harm_ints_f1)],
            [sum(help_ints_f1), len(help_ints_f1) - sum(help_ints_f1)]
        ]
    # Human contingency: From chi2(1, N=78)=27.2
    # Harm: 82% of 39 = 32 Yes, 7 No; Help: 23% of 39 = 9 Yes, 30 No
    contingency_human_f1 = [[32, 7], [9, 30]]
    
    test_result_f1 = {
        "study_id": "Experiment 1", "sub_study_id": "experiment_1_harm", "finding_id": "F1",
        "test_name": "Chi-square test", 
        "pi_human": float(pi_h1['pi_plus'] + pi_h1['pi_minus']),
        "pi_agent": float(pi_a1['pi_plus'] + pi_a1['pi_minus']),
        "pi_human_3way": pi_h1,
        "pi_agent_3way": pi_a1,
        "pas": float(calc_pas(pi_h1, pi_a1)),
        "test_weight": test_weights.get(("F1", "Chi-square test"), 1.0), "pi_human_source": "chi2(1, N=78)=27.2", 
        "agent_reason": r1, "statistical_test_type": "Chi-square", "human_test_statistic": "27.2",
        "agent_test_statistic": f"{chi2_f1:.2f}" if chi2_f1 is not None else ""
    }
    add_statistical_replication_fields(
        test_result_f1, test_gt_f1, p_val_f1, chi2_f1, "chi-square",
        contingency_agent=contingency_agent_f1,
        contingency_human=contingency_human_f1
    )
    test_results.append(test_result_f1)

    # F2: Combined t-test
    pi_h2, pi_a2, r2, p_val_f2, t_stat_f2 = run_t_test_moral()
    # Get test_gt for F2
    test_gt_f2 = {}
    for study in ground_truth.get("studies", []):
        for finding in study.get("findings", []):
            if finding.get("finding_id") == "F2":
                statistical_tests = finding.get("statistical_tests", [])
                if statistical_tests:
                    test_gt_f2 = statistical_tests[0]
                break
    
    # Extract sample sizes for F2
    harm_moral_f2 = [d["moral_score"] for sub in ["experiment_1_harm", "experiment_2_harm"] 
                      for d in processed_data[sub] if "moral_score" in d]
    help_moral_f2 = [d["moral_score"] for sub in ["experiment_1_help", "experiment_2_help"] 
                      for d in processed_data[sub] if "moral_score" in d]
    # Human: t(120) means df=120, so n1+n2=122. Assume balanced: n1=61, n2=61
    
    test_result_f2 = {
        "study_id": "Experiment 1", "sub_study_id": "experiment_1_harm", "finding_id": "F2",
        "test_name": "t-test", 
        "pi_human": float(pi_h2['pi_plus'] + pi_h2['pi_minus']),
        "pi_agent": float(pi_a2['pi_plus'] + pi_a2['pi_minus']),
        "pi_human_3way": pi_h2,
        "pi_agent_3way": pi_a2,
        "pas": float(calc_pas(pi_h2, pi_a2)),
        "test_weight": test_weights.get(("F2", "t-test"), 1.0), "pi_human_source": "t(120)=8.4", 
        "agent_reason": r2, "statistical_test_type": "t-test", "human_test_statistic": "8.4",
        "agent_test_statistic": f"{t_stat_f2:.2f}" if t_stat_f2 is not None else ""
    }
    add_statistical_replication_fields(
        test_result_f2, test_gt_f2, p_val_f2, t_stat_f2, "t-test",
        n_agent=len(harm_moral_f2) if len(harm_moral_f2) > 2 else None,
        n2_agent=len(help_moral_f2) if len(help_moral_f2) > 2 else None,
        n_human=61,  # From t(120), assuming balanced
        n2_human=61,
        independent=True
    )
    test_results.append(test_result_f2)

    # F3: Exp 2 Chi-Square
    pi_h3, pi_a3, r3, p_val_f3, chi2_f3 = run_chi2_test("experiment_2_harm", "experiment_2_help", "F3", 9.5, 42, "Exp 2")
    # Get test_gt for F3
    test_gt_f3 = {}
    for study in ground_truth.get("studies", []):
        for finding in study.get("findings", []):
            if finding.get("finding_id") == "F3":
                statistical_tests = finding.get("statistical_tests", [])
                if statistical_tests:
                    test_gt_f3 = statistical_tests[0]
                break
    
    # Extract contingency tables for F3
    harm_ints_f3 = [d["intentional"] for d in processed_data["experiment_2_harm"] if "intentional" in d]
    help_ints_f3 = [d["intentional"] for d in processed_data["experiment_2_help"] if "intentional" in d]
    contingency_agent_f3 = None
    if len(harm_ints_f3) > 0 and len(help_ints_f3) > 0:
        contingency_agent_f3 = [
            [sum(harm_ints_f3), len(harm_ints_f3) - sum(harm_ints_f3)],
            [sum(help_ints_f3), len(help_ints_f3) - sum(help_ints_f3)]
        ]
    # Human contingency: From chi2(1, N=42)=9.5
    # Harm: 77% of 21 = 16 Yes, 5 No; Help: 30% of 21 = 6 Yes, 15 No
    contingency_human_f3 = [[16, 5], [6, 15]]
    
    test_result_f3 = {
        "study_id": "Experiment 2", "sub_study_id": "experiment_2_harm", "finding_id": "F3",
        "test_name": "Chi-square test", 
        "pi_human": float(pi_h3['pi_plus'] + pi_h3['pi_minus']),
        "pi_agent": float(pi_a3['pi_plus'] + pi_a3['pi_minus']),
        "pi_human_3way": pi_h3,
        "pi_agent_3way": pi_a3,
        "pas": float(calc_pas(pi_h3, pi_a3)),
        "test_weight": test_weights.get(("F3", "Chi-square test"), 1.0), "pi_human_source": "chi2(1, N=42)=9.5", 
        "agent_reason": r3, "statistical_test_type": "Chi-square", "human_test_statistic": "9.5",
        "agent_test_statistic": f"{chi2_f3:.2f}" if chi2_f3 is not None else ""
    }
    add_statistical_replication_fields(
        test_result_f3, test_gt_f3, p_val_f3, chi2_f3, "chi-square",
        contingency_agent=contingency_agent_f3,
        contingency_human=contingency_human_f3
    )
    test_results.append(test_result_f3)

    # F4: Correlation
    pi_h4, pi_a4, r4, p_val_f4, r_val_f4 = run_correlation()
    # Get test_gt for F4
    test_gt_f4 = {}
    for study in ground_truth.get("studies", []):
        for finding in study.get("findings", []):
            if finding.get("finding_id") == "F4":
                statistical_tests = finding.get("statistical_tests", [])
                if statistical_tests:
                    test_gt_f4 = statistical_tests[0]
                break
    
    # Extract sample size for F4
    ints_f4 = []
    moral_f4 = []
    for sub in processed_data:
        for d in processed_data[sub]:
            if "intentional" in d and "moral_score" in d:
                ints_f4.append(d["intentional"])
                moral_f4.append(d["moral_score"])
    # Human: r(120) means n=122 (df=120 for correlation test)
    
    test_result_f4 = {
        "study_id": "Experiment 2", "sub_study_id": "experiment_2_harm", "finding_id": "F4",
        "test_name": "Correlation", 
        "pi_human": float(pi_h4['pi_plus'] + pi_h4['pi_minus']),
        "pi_agent": float(pi_a4['pi_plus'] + pi_a4['pi_minus']),
        "pi_human_3way": pi_h4,
        "pi_agent_3way": pi_a4,
        "pas": float(calc_pas(pi_h4, pi_a4)),
        "test_weight": test_weights.get(("F4", "Correlation"), 1.0), "pi_human_source": "r(120)=0.53", 
        "agent_reason": r4, "statistical_test_type": "correlation", "human_test_statistic": "0.53",
        "agent_test_statistic": f"{r_val_f4:.2f}" if r_val_f4 is not None else ""
    }
    add_statistical_replication_fields(
        test_result_f4, test_gt_f4, p_val_f4, r_val_f4, "correlation",
        n_agent=len(ints_f4) if len(ints_f4) > 5 else None,
        n_human=122  # From r(120), n=122
    )
    test_results.append(test_result_f4)

    # 4. Two-level weighted aggregation
    finding_results = []
    finding_ids = sorted(list(set(tr["finding_id"] for tr in test_results)))
    for fid in finding_ids:
        fid_tests = [tr for tr in test_results if tr["finding_id"] == fid]
        total_weighted_pas = sum(tr["pas"] * tr["test_weight"] for tr in fid_tests)
        total_test_weight = sum(tr["test_weight"] for tr in fid_tests)
        finding_score = total_weighted_pas / total_test_weight if total_test_weight > 0 else 0.5
        
        finding_results.append({
            "sub_study_id": fid_tests[0]["sub_study_id"],
            "finding_id": fid,
            "finding_score": float(finding_score),
            "finding_weight": float(finding_weights.get(fid, 1.0)),
            "n_tests": len(fid_tests)
        })

    total_weighted_finding_score = sum(fr["finding_score"] * fr["finding_weight"] for fr in finding_results)
    total_finding_weight = sum(fr["finding_weight"] for fr in finding_results)
    overall_score = total_weighted_finding_score / total_finding_weight if total_finding_weight > 0 else 0.5

    # substudy_results for backward compatibility
    substudy_results = []
    for sid in sorted(processed_data.keys()):
        sid_findings = [fr for fr in finding_results if fr["sub_study_id"] == sid]
        if sid_findings:
            score = np.mean([fr["finding_score"] for fr in sid_findings])
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