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
    calc_posteriors_3way,
    calc_pas,
    parse_p_value_from_reported,
    get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """
    Parses Qk=X format from agent responses (where X is a number).
    Also supports legacy Qk=Box X format for backward compatibility.
    """
    results = {}
    
    # Pattern 1: Match Q1=7, Q1: 7 (direct number format - preferred)
    # Also handles <7> format
    pattern1 = re.compile(r"(Q\d+)\s*[:=]\s*(?:<)?(\d+)(?:>)?")
    for k, v in pattern1.findall(response_text):
        k_stripped = k.strip()
        if k_stripped not in results:
            # Store as direct number (string format for consistency)
            results[k_stripped] = v.strip()
    
    # Pattern 2: Match Q1=Box 7, Q1: Box 7 (legacy format with "Box" prefix, case-insensitive)
    # Also handles <Box 7> format
    # Only add if not already found by pattern1 (to avoid duplicates)
    pattern2 = re.compile(r"(Q\d+)\s*[:=]\s*(?:<)?(?:Box|box)\s*(\d+)(?:>)?", re.IGNORECASE)
    for k, v in pattern2.findall(response_text):
        k_stripped = k.strip()
        if k_stripped not in results:
            # Extract just the number from "Box 7" format
            results[k_stripped] = v.strip()
    
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_007: 从items中提取q_idx
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
        # 如果没有q_idx，使用索引推断
        elif not required:
            for idx, _ in enumerate(items):
                required.add(f"Q{idx + 1}")
            break
    
    return required

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the agent's performance for study_007 (Billig & Tajfel, 1973).
    """
    # 1. Load ground truth and metadata
    study_id = "study_007"
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
            # Study 007 materials might be under specific names, but here we expect the sub_study_id
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

    # 2. Matrix Definitions from GT
    matrix_defs = ground_truth["overall_original_results"]["all_raw_data"]["matrix_definitions"]
    
    # 3. Extract agent data
    # Categorized by condition: cat_sim, cat_non_sim, non_cat_sim, non_cat_non_sim
    agent_data = {
        "cat_sim": [],
        "cat_non_sim": [],
        "non_cat_sim": [],
        "non_cat_non_sim": []
    }

    for participant in results.get("individual_data", []):
        for response in participant.get("responses", []):
            response_text = response.get("response_text", "")
            trial_info = response.get("trial_info", {})
            condition = trial_info.get("condition")
            items = trial_info.get("items", [])
            sub_id = trial_info.get("sub_study_id") or "2x2_factorial_design"
            
            # If items are missing from trial_info, load from materials file
            if not items and sub_id:
                items = load_materials(sub_id)
            
            parsed = parse_agent_responses(response_text)
            
            participant_scores = {}
            for item in items:
                q_idx = item.get("q_idx")
                item_id = item.get("id") # Matrix_1, Matrix_2, Matrix_3
                
                if q_idx in parsed:
                    choice_text = parsed[q_idx]
                    try:
                        # Extract box number: support both "Box 7" format and direct number "7"
                        # parse_agent_responses already converts "Box 7" to "7", but handle legacy format just in case
                        if isinstance(choice_text, str) and re.search(r"Box", choice_text, re.IGNORECASE):
                            # Legacy format: "Box 7" or "box 7"
                            match = re.search(r"(\d+)", choice_text)
                            if match:
                                box_num = int(match.group(1))
                            else:
                                continue
                        else:
                            # Direct number format: "7" or just the number
                            box_num = int(choice_text.strip())
                        idx = box_num - 1
                        
                        # Calculate specific pulls based on Tajfel matrix logic
                        if item_id == "Matrix_1":
                            # Score = Top Row value. GT says < 6.5 is Favoritism.
                            val = matrix_defs["Matrix_1"]["top_row"][idx]
                            participant_scores["m1_val"] = val
                        elif item_id == "Matrix_2":
                            # FAV vs MJP. Index 0 is Max FAV (Diff 18), Index 12 is Max MJP (Sum 32).
                            # Neutral point is index 6. 
                            # Pull = 7 - box_num (Box 1 -> 6, Box 7 -> 0, Box 13 -> -6)
                            participant_scores["m2_pull"] = 7 - box_num
                        elif item_id == "Matrix_3":
                            # FAV vs Fairness. Box 1 (Index 0) is Fairness (14/14), Box 13 (Index 12) is FAV (26/2).
                            # Pull = box_num - 7 (Box 1 -> -6, Box 7 -> 0, Box 13 -> 6)
                            participant_scores["m3_pull"] = box_num - 7
                    except (AttributeError, ValueError, IndexError):
                        continue
            
            if len(participant_scores) == 3:
                # Calculate Overall FAV score (proxy for Rank Mean)
                # Lower M1 = More Fav, Higher M2/M3 Pull = More Fav
                overall = (6.5 - participant_scores["m1_val"]) + participant_scores["m2_pull"] + participant_scores["m3_pull"]
                participant_scores["overall"] = overall
                agent_data[condition].append(participant_scores)

    # 4. Define Test Processing
    test_results = []
    
    # Human statistics parsing and Bayesian probability calculation
    # F1: ANOVA (2x2)
    # F2: Wilcoxon (Matrix 2)
    # F3: Wilcoxon (Matrix 3)
    # F4: One-sample t-test (Matrix 1)

    for study in ground_truth["studies"]:
        for finding in study["findings"]:
            fid = finding["finding_id"]
            for test in finding["statistical_tests"]:
                pi_h = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
                reason = "Insufficient data"
                reported = test["reported_statistics"]
                test_stat_human = ""
                
                try:
                    if fid == "F1":
                        # Main effects of Categorization and Similarity
                        # Human: F(1, 72) = 14.96 (Cat), F(1, 72) = 4.13 (Sim)
                        n_h = 75
                        if "Categorization" in test["test_name"] or "Categorization" in reported:
                            f_h = 14.96
                            test_stat_human = "14.96"
                            bf_h = calc_bf_anova(f_h, 1, 72, n_h)
                            pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                            
                            # Agent: T-test between Cat and Non-Cat groups (F = t^2)
                            cat_group = [d["overall"] for d in agent_data["cat_sim"] + agent_data["cat_non_sim"]]
                            non_cat_group = [d["overall"] for d in agent_data["non_cat_sim"] + agent_data["non_cat_non_sim"]]
                            
                            p_val_agent_f1_cat = None
                            t_stat_agent_f1_cat = None
                            if len(cat_group) > 2 and len(non_cat_group) > 2:
                                t_stat_agent_f1_cat, p_val_agent_f1_cat = stats.ttest_ind(cat_group, non_cat_group)
                                t_val = t_stat_agent_f1_cat if not np.isnan(t_stat_agent_f1_cat) else 0
                                f_a = t_val**2
                                n_a = len(cat_group) + len(non_cat_group)
                                bf_a = calc_bf_anova(f_a, 1, n_a - 4, n_a)
                                a_dir = 1 if t_val > 0 else -1
                                pi_a = calc_posteriors_3way(bf_a, a_dir)
                                reason = f"F={f_a:.2f}, n={n_a}"
                        
                        elif "Similarity" in test["test_name"] or "Similarity" in reported:
                            f_h = 4.13
                            test_stat_human = "4.13"
                            bf_h = calc_bf_anova(f_h, 1, 72, n_h)
                            pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                            
                            # Agent: T-test between Sim and Non-Sim groups
                            sim_group = [d["overall"] for d in agent_data["cat_sim"] + agent_data["non_cat_sim"]]
                            non_sim_group = [d["overall"] for d in agent_data["cat_non_sim"] + agent_data["non_cat_non_sim"]]
                            
                            p_val_agent_f1_sim = None
                            t_stat_agent_f1_sim = None
                            if len(sim_group) > 2 and len(non_sim_group) > 2:
                                t_stat_agent_f1_sim, p_val_agent_f1_sim = stats.ttest_ind(sim_group, non_sim_group)
                                t_val = t_stat_agent_f1_sim if not np.isnan(t_stat_agent_f1_sim) else 0
                                f_a = t_val**2
                                n_a = len(sim_group) + len(non_sim_group)
                                bf_a = calc_bf_anova(f_a, 1, n_a - 4, n_a)
                                a_dir = 1 if t_val > 0 else -1
                                pi_a = calc_posteriors_3way(bf_a, a_dir)
                                reason = f"F={f_a:.2f}, n={n_a}"

                    elif fid == "F2":
                        # Wilcoxon/T-test on Matrix 2 Pull for Cat groups
                        # Human: p < .01 (Cat:Sim), p < .05 (Cat:Non-sim)
                        # We'll use the Cat:Sim group (strongest finding)
                        # N per condition ~ 19. p=0.01 for df=18 -> t ~ 2.88
                        t_h = 2.88
                        test_stat_human = "2.88"
                        bf_h = calc_bf_t(t_h, 19, independent=False)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        # Agent: One-sample t-test vs 0 pull
                        p_val_agent_f2 = None
                        t_stat_agent_f2 = None
                        group = [d["m2_pull"] for d in agent_data["cat_sim"]]
                        if len(group) > 2:
                            t_stat_agent_f2, p_val_agent_f2 = stats.ttest_1samp(group, 0)
                            bf_a = calc_bf_t(t_stat_agent_f2, len(group), independent=False)
                            a_dir = 1 if t_stat_agent_f2 > 0 else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"t={t_stat_agent_f2:.2f}, n={len(group)}"

                    elif fid == "F3":
                        # Wilcoxon/T-test on Matrix 3 Pull for Cat groups
                        # Human: p < .01. t ~ 2.88
                        t_h = 2.88
                        test_stat_human = "2.88"
                        bf_h = calc_bf_t(t_h, 19, independent=False)
                        pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
                        
                        p_val_agent_f3 = None
                        t_stat_agent_f3 = None
                        group = [d["m3_pull"] for d in agent_data["cat_sim"]]
                        if len(group) > 2:
                            t_stat_agent_f3, p_val_agent_f3 = stats.ttest_1samp(group, 0)
                            bf_a = calc_bf_t(t_stat_agent_f3, len(group), independent=False)
                            a_dir = 1 if t_stat_agent_f3 > 0 else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"t={t_stat_agent_f3:.2f}, n={len(group)}"

                    elif fid == "F4":
                        # One-sample t-test Matrix 1 vs 6.5
                        # Human: Mean 4.38, p < .01. t ~ 2.88
                        t_h = -2.88 # Negative t for < 6.5
                        test_stat_human = "-2.88"
                        bf_h = calc_bf_t(abs(t_h), 19, independent=False)
                        pi_h = calc_posteriors_3way(bf_h, -1, prior_odds=10.0)
                        
                        p_val_agent_f4 = None
                        t_stat_agent_f4 = None
                        group = [d["m1_val"] for d in agent_data["cat_sim"]]
                        if len(group) > 2:
                            # Note: Direction matters (Ingroup fav is < 6.5)
                            t_stat_agent_f4, p_val_agent_f4 = stats.ttest_1samp(group, 6.5)
                            bf_a = calc_bf_t(t_stat_agent_f4, len(group), independent=False)
                            a_dir = 1 if t_stat_agent_f4 > 0 else -1
                            pi_a = calc_posteriors_3way(bf_a, a_dir)
                            reason = f"mean={np.mean(group):.2f}, t={t_stat_agent_f4:.2f}, n={len(group)}"

                except Exception as e:
                    reason = f"Error: {str(e)}"
                    p_val_agent = None
                    t_stat_agent = None

                # Determine p_val_agent and t_stat_agent based on finding_id
                p_val_agent = None
                t_stat_agent = None
                if fid == "F1":
                    if "Categorization" in test.get("test_name", "") or "Categorization" in reported:
                        p_val_agent = p_val_agent_f1_cat if 'p_val_agent_f1_cat' in locals() else None
                        t_stat_agent = t_stat_agent_f1_cat if 't_stat_agent_f1_cat' in locals() else None
                    elif "Similarity" in test.get("test_name", "") or "Similarity" in reported:
                        p_val_agent = p_val_agent_f1_sim if 'p_val_agent_f1_sim' in locals() else None
                        t_stat_agent = t_stat_agent_f1_sim if 't_stat_agent_f1_sim' in locals() else None
                elif fid == "F2":
                    p_val_agent = p_val_agent_f2 if 'p_val_agent_f2' in locals() else None
                    t_stat_agent = t_stat_agent_f2 if 't_stat_agent_f2' in locals() else None
                elif fid == "F3":
                    p_val_agent = p_val_agent_f3 if 'p_val_agent_f3' in locals() else None
                    t_stat_agent = t_stat_agent_f3 if 't_stat_agent_f3' in locals() else None
                elif fid == "F4":
                    p_val_agent = p_val_agent_f4 if 'p_val_agent_f4' in locals() else None
                    t_stat_agent = t_stat_agent_f4 if 't_stat_agent_f4' in locals() else None

                # Create test result dict
                test_result = {
                    "study_id": study["study_id"],
                    "sub_study_id": "2x2_factorial_design",
                    "finding_id": fid,
                    "test_name": test["test_name"],
                    "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
                    "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
                    "pi_human_3way": pi_h,
                    "pi_agent_3way": pi_a,
                    "pas": float(calc_pas(pi_h, pi_a)),
                    "pi_human_source": reported,
                    "agent_reason": reason,
                    "statistical_test_type": "t-test" if fid in ["F2", "F3", "F4"] else "f-test",
                    "human_test_statistic": test_stat_human,
                    "agent_test_statistic": f"{t_stat_agent:.2f}" if t_stat_agent is not None else ""
                }
                
                # Extract sample sizes for t-tests
                n_agent = None
                n_human = None
                if fid == "F2" and 'group' in locals() and len(group) > 2:
                    n_agent = len(group)
                    n_human = 19  # From t(18), n=19
                elif fid == "F3" and 'group' in locals() and len(group) > 2:
                    n_agent = len(group)
                    n_human = 19  # From t(18), n=19
                elif fid == "F4" and 'group' in locals() and len(group) > 2:
                    n_agent = len(group)
                    n_human = 19  # From t(18), n=19
                
                # Add statistical replication fields
                if fid in ["F2", "F3", "F4"]:
                    add_statistical_replication_fields(
                        test_result, test, p_val_agent, t_stat_agent, "t-test",
                        n_agent=n_agent,
                        n_human=n_human,
                        independent=False  # One-sample t-tests
                    )
                else:
                    # F-test (F1) - not supported yet for effect sizes
                    add_statistical_replication_fields(test_result, test, p_val_agent, t_stat_agent, "f-test")
                
                test_results.append(test_result)

    # 5. Two-level Weighted Aggregation
    # Add test_weight to each test result
    for tr in test_results:
        finding_id = tr.get("finding_id")
        test_name = tr.get("test_name", "")
        tr["test_weight"] = float(test_weights.get((finding_id, test_name), 1.0))
    
    if not test_results:
        return {"score": 0.5, "substudy_results": [], "finding_results": [], "test_results": []}

    # Level 1: Aggregate tests into findings (weighted by test weights)
    finding_results = []
    finding_ids = sorted(list(set(t["finding_id"] for t in test_results)))
    for f_id in finding_ids:
        f_tests = [t for t in test_results if t["finding_id"] == f_id]
        # Weighted average: Σ (PAS * weight) / Σ weights
        total_weighted_pas = sum(t["pas"] * t.get("test_weight", 1.0) for t in f_tests)
        total_weight = sum(t.get("test_weight", 1.0) for t in f_tests)
        finding_score = total_weighted_pas / total_weight if total_weight > 0 else 0.5
        
        finding_weight = finding_weights.get(f_id, 1.0)
        finding_results.append({
            "sub_study_id": f_tests[0]["sub_study_id"],
            "finding_id": f_id,
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