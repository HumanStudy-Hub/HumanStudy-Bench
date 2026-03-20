import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional

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

def extract_numeric(val: Any) -> Optional[float]:
    """Extracts the first numeric value from a string."""
    if val is None:
        return None
    try:
        # Handle cases like "1,000" or "1.5"
        clean_val = str(val).replace(",", "")
        match = re.search(r"[-+]?\d*\.?\d+", clean_val)
        return float(match.group()) if match else None
    except (ValueError, AttributeError):
        return None

def extract_choice(val: Any) -> Optional[str]:
    """Extracts A or B from the response."""
    if not val:
        return None
    match = re.search(r"\b([A-B])\b", str(val).upper())
    return match.group(1) if match else None

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """Parses response_text into a dictionary of Qk=value or Qk: value."""
    results = {}
    # Matches Q1=A, Q1: 36, Q1.1=100, Q2 = 75, etc.
    # Supports both = and : separators
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*[:=]\s*([^,\n\s]+)")
    matches = pattern.findall(response_text)
    for k, v in matches:
        results[k.strip()] = v.strip()
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_002: 从items中提取q_idx_estimate, q_idx_choice, q_idx_confidence等
    """
    required = set()
    items = trial_info.get("items", [])
    
    for item in items:
        for key in ['q_idx', 'q_idx_choice', 'q_idx_estimate', 'q_idx_confidence']:
            q_idx = item.get(key)
            if q_idx:
                # 如果q_idx已经包含"Q"前缀，直接使用；否则添加"Q"前缀
                if isinstance(q_idx, str) and q_idx.startswith("Q"):
                    required.add(q_idx)
                else:
                    required.add(f"Q{q_idx}")
    
    return required

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the agent's performance for Study 002 (Anchoring).
    """
    study_id = "study_002"
    study_dir = Path(__file__).resolve().parent.parent / "source"
    
    with open(study_dir / "ground_truth.json", "r") as f:
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

    # 1. Organize Data
    # sub_study_id -> [ {item_label, estimate, anchor_type, choice, confidence, anchor_val} ]
    agent_data = {
        "exp_1_calibration": [],
        "exp_1_anchored_estimation": [],
        "exp_2_discredited_anchor": [],
        "exp_3_wtp_estimation": []
    }

    # Helper to normalize labels to match ground truth table
    def normalize_label(label: str) -> str:
        """Normalize label to match ground truth table format."""
        if not label:
            return label
        label_lower = label.lower()
        # Map full labels to GT table format
        mappings = {
            "mississippi": "Mississippi River",
            "everest": "Mount Everest",
            "meat": "Meat eaten",
            "sf to nyc": "SF to NYC",
            "distance sf": "SF to NYC",
            "redwood": "Redwood height",
            "tallest redwood": "Redwood height",
            "height of tallest redwood": "Redwood height",
            "un members": "UN members",
            "number of un": "UN members",
            "female professors": "Female professors",
            "chicago": "Chicago population",
            "population of chicago": "Chicago population",
            "telephone": "Telephone year",
            "year telephone": "Telephone year",
            "babies": "Babies born",
            "babies born per day": "Babies born",
            "cat": "Cat speed",
            "speed of house cat": "Cat speed",
            "gas": "Gas used",
            "gas used per month": "Gas used",
            "bars": "Berkeley bars",
            "number of bars": "Berkeley bars",
            "bars in berkeley": "Berkeley bars",
            "state colleges": "State colleges",
            "lincoln": "Lincoln presidency",
            "lincoln presidency": "Lincoln presidency"
        }
        for key, value in mappings.items():
            if key in label_lower:
                return value
        return label  # Return original if no match
    
    # GT table labels (from ground_truth.json data_tables)
    gt_table_labels = [
        "Mississippi River", "Mount Everest", "Meat eaten",
        "SF to NYC", "Redwood height", "UN members",
        "Female professors", "Chicago population", "Telephone year",
        "Babies born", "Cat speed", "Gas used",
        "Berkeley bars", "State colleges", "Lincoln presidency"
    ]

    for participant in results.get("individual_data", []):
        for response in participant.get("responses", []):
            response_text = response.get("response_text", "")
            trial_info = response.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            items = trial_info.get("items", [])
            
            # If items are missing from trial_info, load from materials file
            if not items and sub_id:
                items = load_materials(sub_id)

            if sub_id not in agent_data:
                continue

            parsed = parse_agent_responses(response_text)
            
            for item in items:
                # Determine item label (GT name)
                # Try gt_key first as per instructions
                label = item.get("metadata", {}).get("gt_key")
                if not label:
                    # Fallback to metadata label
                    label = item.get("metadata", {}).get("label")
                if not label:
                    # Fallback to index mapping if gt_key missing
                    item_id = item.get("id", "")
                    try:
                        idx = int(re.search(r"(\d+)", item_id).group(1)) - 1
                        if 0 <= idx < len(gt_table_labels):
                            label = gt_table_labels[idx]
                    except:
                        label = item_id
                
                # Normalize label to match ground truth table format
                label = normalize_label(label)

                entry = {"label": label, "participant_id": participant.get("participant_id")}
                
                if sub_id == "exp_1_calibration":
                    est_key = item.get("q_idx_estimate", "Q1.1")
                    conf_key = item.get("q_idx_confidence", "Q1.2")
                    entry["estimate"] = extract_numeric(parsed.get(est_key))
                    entry["confidence"] = extract_numeric(parsed.get(conf_key))
                    agent_data[sub_id].append(entry)

                elif sub_id in ["exp_1_anchored_estimation", "exp_2_discredited_anchor"]:
                    choice_key = item.get("q_idx_choice", "Q1.1")
                    est_key = item.get("q_idx_estimate", "Q1.2")
                    conf_key = item.get("q_idx_confidence") # May not exist in Exp 2
                    
                    entry["choice"] = extract_choice(parsed.get(choice_key))
                    entry["estimate"] = extract_numeric(parsed.get(est_key))
                    if conf_key:
                        entry["confidence"] = extract_numeric(parsed.get(conf_key))
                    
                    entry["anchor_type"] = item.get("assigned_anchor_type")
                    # Extract actual anchor value from metadata
                    meta = item.get("metadata", {})
                    anchor_type = entry.get("anchor_type")
                    if anchor_type:
                        entry["anchor_val"] = meta.get(f"{anchor_type}_anchor")
                    else:
                        entry["anchor_val"] = None
                    agent_data[sub_id].append(entry)

                elif sub_id == "exp_3_wtp_estimation":
                    # WTP items are paired: referendum then amount
                    q_idx = item.get("q_idx", "")
                    if "referendum" in item.get("id", ""):
                        entry["choice"] = extract_choice(parsed.get(q_idx))
                    else:
                        entry["estimate"] = extract_numeric(parsed.get(q_idx))
                    # WTP also needs anchor_type and anchor_val for analysis
                    entry["anchor_type"] = item.get("assigned_anchor_type")
                    meta = item.get("metadata", {})
                    anchor_type = entry.get("anchor_type")
                    if anchor_type:
                        entry["anchor_val"] = meta.get(f"{anchor_type}_anchor")
                    else:
                        entry["anchor_val"] = None
                    agent_data[sub_id].append(entry)

    # 2. Extract Calibration Baselines for Transformed Scores
    # Table 1: [Median, Low (15th), High (85th)]
    baselines = {}
    cal_table = ground_truth.get("overall_original_results", {}).get("data_tables", [{}])[0]
    for row in cal_table.get("rows", []):
        name = row[0]
        # Normalize the name to ensure consistent matching
        normalized_name = normalize_label(name)
        baselines[normalized_name] = {
            "median": extract_numeric(row[1]),
            "low": extract_numeric(row[2]),
            "high": extract_numeric(row[3])
        }
        # Also store with original name for backward compatibility
        baselines[name] = baselines[normalized_name]

    test_results = []

    # --- Experiment 1 Analysis ---
    exp1_finding_map = {f["finding_id"]: f for f in ground_truth["studies"][0]["findings"]}
    e1_data = agent_data["exp_1_anchored_estimation"]
    e1_valid = [d for d in e1_data if d.get("estimate") is not None and d.get("anchor_val") is not None]
    cal_data = agent_data["exp_1_calibration"]
    
    # F1 Test 1: Mean Anchoring Index (AI) (Human AI=0.49, n=103)
    f1_gt_1 = exp1_finding_map["F1"]["statistical_tests"][0]
    # Calculate AI for each item: AI = (Est - CalMedian) / (Anchor - CalMedian)
    all_ais = []
    for d in e1_valid:
        base = baselines.get(d["label"])
        if base:
            cal_median = base["median"]
            anchor = d["anchor_val"]
            denom = anchor - cal_median
            if denom != 0:
                ai = (d["estimate"] - cal_median) / denom
                all_ais.append(ai)

    # Human: AI=0.49, n=103, reported as highly significant
    # Since we don't have exact t-stat, but paper reports it as significant (p<0.05),
    # we estimate t-stat from AI value. For AI=0.49 with n=103, a one-sample t-test
    # against 0 would give approximately t ≈ 5.0 (conservative estimate for strong effect)
    # This gives pi_h ≈ 0.99 which matches the "highly significant" description
    estimated_t_human = 5.0  # Conservative estimate for AI=0.49, n=103
    bf_h = calc_bf_t(estimated_t_human, 103, independent=False)
    pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
    
    if len(all_ais) > 2:
        t_stat, p_val_agent = stats.ttest_1samp(all_ais, 0)
        bf_a = calc_bf_t(t_stat, len(all_ais), independent=False)
        a_dir = 1 if t_stat > 0 else -1
        pi_a = calc_posteriors_3way(bf_a, a_dir)
        mean_ai = np.mean(all_ais)
        reason = f"t={t_stat:.2f}, mean_ai={mean_ai:.2f}, n={len(all_ais)}"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        p_val_agent = None
        t_stat = None
        reason = "Insufficient data"
    
    test_result = {
        "study_id": "Experiment 1", "sub_study_id": "exp_1_anchored_estimation", "finding_id": "F1",
        "test_name": "Mean Anchoring Index (AI)", "scenario": "Global",
        "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']), 
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h, pi_a)),
        "pi_human_source": "AI=0.49, n=103", "agent_reason": reason,
        "statistical_test_type": "t-test", "human_test_statistic": str(estimated_t_human), 
        "agent_test_statistic": f"{t_stat:.2f}" if t_stat is not None else ""
    }
    # One-sample t-test: n_human=103, n_agent=len(all_ais)
    add_statistical_replication_fields(
        test_result, f1_gt_1, p_val_agent, t_stat, "t-test",
        n_agent=len(all_ais) if len(all_ais) > 2 else None,
        n_human=103,
        independent=False
    )
    test_results.append(test_result)

    # F1 Test 2: Correlation (Human r=0.42, n=103)
    f1_gt_2 = exp1_finding_map["F1"]["statistical_tests"][1]
    bf_h = calc_bf_r(0.42, 103)
    pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
    
    if len(e1_valid) > 5:
        r_val, p_val_agent_2 = stats.pearsonr([d["estimate"] for d in e1_valid], [d["anchor_val"] for d in e1_valid])
        bf_a = calc_bf_r(r_val, len(e1_valid))
        a_dir = 1 if r_val > 0 else -1
        pi_a = calc_posteriors_3way(bf_a, a_dir)
        reason = f"r={r_val:.2f}, n={len(e1_valid)}"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        p_val_agent_2 = None
        r_val = None
        reason = "Insufficient data"
    
    test_result_2 = {
        "study_id": "Experiment 1", "sub_study_id": "exp_1_anchored_estimation", "finding_id": "F1",
        "test_name": "Correlation Estimate vs Anchor", "scenario": "Global",
        "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h, pi_a)),
        "pi_human_source": "r=0.42, n=103", "agent_reason": reason,
        "statistical_test_type": "correlation",
        "human_test_statistic": "0.42",
        "agent_test_statistic": f"{r_val:.2f}" if r_val is not None else ""
    }
    # Correlation: n_human=103, n_agent=len(e1_valid)
    add_statistical_replication_fields(
        test_result_2, f1_gt_2, p_val_agent_2, r_val, "correlation",
        n_agent=len(e1_valid) if len(e1_valid) > 5 else None,
        n_human=103
    )
    test_results.append(test_result_2)

    # F2: Asymmetry (Human t(102)=7.99)
    # Transformed Score = 100 * (Est - P15) / (P85 - P15)
    f2_gt = exp1_finding_map["F2"]["statistical_tests"][0]
    scores_high = []
    scores_low = []
    for d in e1_valid:
        base = baselines.get(d["label"])
        if base and base["high"] != base["low"]:
            score = 100 * (d["estimate"] - base["low"]) / (base["high"] - base["low"])
            if d["anchor_type"] == "high": scores_high.append(score)
            else: scores_low.append(score)
    
    bf_h = calc_bf_t(7.99, len(scores_high), len(scores_low), independent=True)
    pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
    
    if len(scores_high) > 2 and len(scores_low) > 2:
        t_stat, p_val_agent_f2 = stats.ttest_ind(scores_high, scores_low)
        bf_a = calc_bf_t(t_stat, len(scores_high), len(scores_low), independent=True)
        a_dir = 1 if t_stat > 0 else -1
        pi_a = calc_posteriors_3way(bf_a, a_dir)
        reason = f"t={t_stat:.2f}, n_h={len(scores_high)}, n_l={len(scores_low)}"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        p_val_agent_f2 = None
        t_stat = None
        reason = "Insufficient data"
        
    test_result_f2 = {
        "study_id": "Experiment 1", "sub_study_id": "exp_1_anchored_estimation", "finding_id": "F2",
        "test_name": "Asymmetry of Anchoring", "scenario": "Global",
        "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h, pi_a)),
        "pi_human_source": "t(102)=7.99", "agent_reason": reason,
        "statistical_test_type": "t-test", "human_test_statistic": "7.99", 
        "agent_test_statistic": f"{t_stat:.2f}" if t_stat is not None else ""
    }
    # Independent t-test: t(102) means df=102, so n1+n2≈104. Assume roughly balanced: n1≈52, n2≈52
    # For human, we'll use approximate values; for agent we have exact counts
    add_statistical_replication_fields(
        test_result_f2, f2_gt, p_val_agent_f2, t_stat, "t-test",
        n_agent=len(scores_high) if len(scores_high) > 2 else None,
        n2_agent=len(scores_low) if len(scores_low) > 2 else None,
        n_human=52,  # Approximate from df=102 (balanced groups)
        n2_human=52,
        independent=True
    )
    test_results.append(test_result_f2)
    
    # F3: Extreme Estimates (Human t(102)=6.12)
    f3_gt = exp1_finding_map["F3"]["statistical_tests"][0]
    # Count estimates that exceed anchor (high anchor) or below anchor (low anchor)
    high_extreme_rates = []
    low_extreme_rates = []
    
    # Group by item label
    by_label = {}
    for d in e1_valid:
        label = d["label"]
        if label not in by_label:
            by_label[label] = {"high": [], "low": []}
        by_label[label][d["anchor_type"]].append(d)
    
    for label, data in by_label.items():
        base = baselines.get(label)
        if not base:
            continue
        high_anchor = base["high"]
        low_anchor = base["low"]
        
        high_ests = [d["estimate"] for d in data["high"]]
        low_ests = [d["estimate"] for d in data["low"]]
        
        if high_ests:
            extreme_count = sum(1 for e in high_ests if e > high_anchor)
            high_extreme_rates.append(extreme_count / len(high_ests))
        if low_ests:
            extreme_count = sum(1 for e in low_ests if e < low_anchor)
            low_extreme_rates.append(extreme_count / len(low_ests))

    # Human: t(102)=6.12, n=103 (use HUMAN sample size, not agent)
    bf_h = calc_bf_t(6.12, 103, independent=False)
    pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
    if len(high_extreme_rates) > 2 and len(low_extreme_rates) > 2:
        t_stat, p_val_agent_f3 = stats.ttest_ind(high_extreme_rates, low_extreme_rates)
        bf_a = calc_bf_t(t_stat, len(high_extreme_rates), len(low_extreme_rates), independent=True)
        a_dir = 1 if t_stat > 0 else -1
        pi_a = calc_posteriors_3way(bf_a, a_dir)
        reason = f"t={t_stat:.2f}, n_h={len(high_extreme_rates)}, n_l={len(low_extreme_rates)}"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        p_val_agent_f3 = None
        t_stat = None
        reason = "Insufficient data"

    test_result_f3 = {
        "study_id": "Experiment 1", "sub_study_id": "exp_1_anchored_estimation", "finding_id": "F3",
        "test_name": "Extreme Estimates (High vs Low)", "scenario": "Global",
        "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h, pi_a)),
        "pi_human_source": "t(102)=6.12", "agent_reason": reason,
        "statistical_test_type": "t-test", "human_test_statistic": "6.12", 
        "agent_test_statistic": f"{t_stat:.2f}" if t_stat is not None else ""
    }
    # Independent t-test: t(102) means df=102, so n1+n2≈104. Assume roughly balanced
    add_statistical_replication_fields(
        test_result_f3, f3_gt, p_val_agent_f3, t_stat, "t-test",
        n_agent=len(high_extreme_rates) if len(high_extreme_rates) > 2 else None,
        n2_agent=len(low_extreme_rates) if len(low_extreme_rates) > 2 else None,
        n_human=52,  # Approximate from df=102
        n2_human=52,
        independent=True
    )
    test_results.append(test_result_f3)

    # F4 Test 1: Pearson Correlation (AI vs Confidence) (Human r=-0.68, n=15)
    f4_gt_1 = exp1_finding_map["F4"]["statistical_tests"][0]
    # Calculate mean AI and mean confidence per item
    item_ais = []
    item_confs = []
    by_label_ai = {}
    by_label_conf = {}
    
    for d in e1_valid:
        label = d["label"]
        base = baselines.get(label)
        if base:
            cal_median = base["median"]
            anchor = d["anchor_val"]
            denom = anchor - cal_median
            if denom != 0:
                ai = (d["estimate"] - cal_median) / denom
                if label not in by_label_ai:
                    by_label_ai[label] = []
                    by_label_conf[label] = []
                by_label_ai[label].append(ai)
                if d.get("confidence") is not None:
                    by_label_conf[label].append(d["confidence"])
    
    for label in by_label_ai:
        if label in by_label_conf and by_label_conf[label]:
            item_ais.append(np.mean(by_label_ai[label]))
            item_confs.append(np.mean(by_label_conf[label]))
    
    # F4 Test 1: Pearson Correlation (AI vs. Confidence) (Human r=-0.68, n=15)
    f4_gt_1 = exp1_finding_map["F4"]["statistical_tests"][0]
    bf_h = calc_bf_r(-0.68, 15)
    pi_h = calc_posteriors_3way(bf_h, -1, prior_odds=10.0)
    if len(item_ais) > 3 and len(item_confs) > 3:
        r_val, p_val_agent_f4_1 = stats.pearsonr(item_ais, item_confs)
        bf_a = calc_bf_r(r_val, len(item_ais))
        a_dir = 1 if r_val > 0 else -1
        pi_a = calc_posteriors_3way(bf_a, a_dir)
        reason = f"r={r_val:.2f}, n={len(item_ais)}"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        p_val_agent_f4_1 = None
        r_val = None
        reason = "Insufficient data"
    
    test_result_f4_1 = {
        "study_id": "Experiment 1", "sub_study_id": "exp_1_anchored_estimation", "finding_id": "F4",
        "test_name": "Pearson Correlation (AI vs. Confidence)", "scenario": "Global",
        "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h, pi_a)),
        "pi_human_source": "r=-0.68, n=15", "agent_reason": reason,
        "statistical_test_type": "correlation",
        "human_test_statistic": "-0.68",
        "agent_test_statistic": f"{r_val:.2f}" if r_val is not None else ""
    }
    # Correlation: n_human=15 (items), n_agent=len(item_ais)
    add_statistical_replication_fields(
        test_result_f4_1, f4_gt_1, p_val_agent_f4_1, r_val, "correlation",
        n_agent=len(item_ais) if len(item_ais) > 3 else None,
        n_human=15
    )
    test_results.append(test_result_f4_1)
    
    # F4 Test 2: High Anchor vs Confidence (Human t(14)=2.37)
    # Pool data across all items to increase sample size
    f4_gt_2 = exp1_finding_map["F4"]["statistical_tests"][1]
    high_ais_pooled = []
    high_confs_pooled = []
    for label in by_label_ai:
        high_data = [d for d in e1_valid if d["label"] == label and d["anchor_type"] == "high" and d.get("confidence") is not None]
        base = baselines.get(label)
        if base and len(high_data) > 0:
            cal_median = base["median"]
            anchor = base["high"]
            denom = anchor - cal_median
            if denom != 0:
                for d in high_data:
                    ai = (d["estimate"] - cal_median) / denom
                    high_ais_pooled.append(ai)
                    high_confs_pooled.append(d["confidence"])

    bf_h = calc_bf_t(2.37, 15, independent=False)
    pi_h = calc_posteriors_3way(bf_h, -1, prior_odds=10.0) # Expected negative
    t_stat_agent = None
    p_val_agent_f4_2 = None
    if len(high_ais_pooled) > 10 and len(high_confs_pooled) == len(high_ais_pooled):
        r, p_val_agent_f4_2 = stats.pearsonr(high_ais_pooled, high_confs_pooled)
        if not np.isnan(r):
            # Test if correlation differs from 0 (expected negative)
            t_stat_agent = r * np.sqrt((len(high_ais_pooled) - 2) / (1 - r**2)) if abs(r) < 0.999 else 0
            bf_a = calc_bf_t(t_stat_agent, len(high_ais_pooled), independent=False)
            a_dir = 1 if t_stat_agent > 0 else -1
            pi_a = calc_posteriors_3way(bf_a, a_dir)
            reason = f"r={r:.2f}, t={t_stat_agent:.2f}, n={len(high_ais_pooled)}"
        else:
            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
            reason = "Insufficient data (NaN correlation)"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        reason = "Insufficient data"

    test_result_f4_2 = {
        "study_id": "Experiment 1", "sub_study_id": "exp_1_anchored_estimation", "finding_id": "F4",
        "test_name": "High Anchor vs Confidence (Individual Level)", "scenario": "Global",
        "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h, pi_a)),
        "pi_human_source": "t(14)=2.37", "agent_reason": reason,
        "statistical_test_type": "t-test", "human_test_statistic": "-2.37", 
        "agent_test_statistic": f"{t_stat_agent:.2f}" if t_stat_agent is not None else ""
    }
    # One-sample t-test: t(14) means df=14, so n=15
    add_statistical_replication_fields(
        test_result_f4_2, f4_gt_2, p_val_agent_f4_2, t_stat_agent, "t-test",
        n_agent=len(high_ais_pooled) if len(high_ais_pooled) > 10 else None,
        n_human=15,
        independent=False
    )
    test_results.append(test_result_f4_2)
    
    # F4 Test 3: Low Anchor vs Confidence (Human t(14)=4.80)
    # Pool data across all items to increase sample size
    f4_gt_3 = exp1_finding_map["F4"]["statistical_tests"][2]
    low_ais_pooled = []
    low_confs_pooled = []
    for label in by_label_ai:
        low_data = [d for d in e1_valid if d["label"] == label and d["anchor_type"] == "low" and d.get("confidence") is not None]
        base = baselines.get(label)
        if base and len(low_data) > 0:
            cal_median = base["median"]
            anchor = base["low"]
            denom = anchor - cal_median
            if denom != 0:
                for d in low_data:
                    ai = (d["estimate"] - cal_median) / denom
                    low_ais_pooled.append(ai)
                    low_confs_pooled.append(d["confidence"])
    
    bf_h = calc_bf_t(4.80, 15, independent=False)
    pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
    t_stat_agent = None
    p_val_agent_f4_3 = None
    if len(low_ais_pooled) > 10 and len(low_confs_pooled) == len(low_ais_pooled):
        r, p_val_agent_f4_3 = stats.pearsonr(low_ais_pooled, low_confs_pooled)
        if not np.isnan(r):
            # Test if correlation differs from 0
            t_stat_agent = r * np.sqrt((len(low_ais_pooled) - 2) / (1 - r**2)) if abs(r) < 0.999 else 0
            bf_a = calc_bf_t(t_stat_agent, len(low_ais_pooled), independent=False)
            a_dir = 1 if t_stat_agent > 0 else -1
            pi_a = calc_posteriors_3way(bf_a, a_dir)
            reason = f"r={r:.2f}, t={t_stat_agent:.2f}, n={len(low_ais_pooled)}"
        else:
            pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
            reason = "Insufficient data (NaN correlation)"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        reason = "Insufficient data"
    
    test_result_f4_3 = {
        "study_id": "Experiment 1", "sub_study_id": "exp_1_anchored_estimation", "finding_id": "F4",
        "test_name": "Low Anchor vs Confidence (Individual Level)", "scenario": "Global",
        "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h, pi_a)),
        "pi_human_source": "t(14)=4.80", "agent_reason": reason,
        "statistical_test_type": "t-test", "human_test_statistic": "4.80", 
        "agent_test_statistic": f"{t_stat_agent:.2f}" if t_stat_agent is not None else ""
    }
    # One-sample t-test: t(14) means df=14, so n=15
    add_statistical_replication_fields(
        test_result_f4_3, f4_gt_3, p_val_agent_f4_3, t_stat_agent, "t-test",
        n_agent=len(low_ais_pooled) if len(low_ais_pooled) > 10 else None,
        n_human=15,
        independent=False
    )
    test_results.append(test_result_f4_3)
    
    # F4 Test 4: Confidence Ratings Comparison (Human t(154)=3.53)
    f4_gt_4 = exp1_finding_map["F4"]["statistical_tests"][3]
    conf_anchored = [d["confidence"] for d in e1_data if d.get("confidence") is not None]
    conf_calib = [d["confidence"] for d in agent_data["exp_1_calibration"] if d.get("confidence") is not None]
    
    # Human: t(154)=3.53, n=155 (use HUMAN sample size, not agent)
    bf_h = calc_bf_t(3.53, 155, independent=False)
    pi_h = calc_posteriors_3way(bf_h, 1, prior_odds=10.0)
    if len(conf_anchored) > 2 and len(conf_calib) > 2:
        t_stat, p_val_agent_f4_4 = stats.ttest_ind(conf_anchored, conf_calib)
        bf_a = calc_bf_t(t_stat, len(conf_anchored), len(conf_calib), independent=True)
        a_dir = 1 if t_stat > 0 else -1
        pi_a = calc_posteriors_3way(bf_a, a_dir)
        reason = f"t={t_stat:.2f}, n1={len(conf_anchored)}, n2={len(conf_calib)}"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        p_val_agent_f4_4 = None
        t_stat = None
        reason = "Insufficient data"

    test_result_f4_4 = {
        "study_id": "Experiment 1", "sub_study_id": "exp_1_anchored_estimation", "finding_id": "F4",
        "test_name": "Confidence Ratings Comparison (Anchored vs Calibration)", "scenario": "Global",
        "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h, pi_a)),
        "pi_human_source": "t(154)=3.53", "agent_reason": reason,
        "statistical_test_type": "t-test", "human_test_statistic": "3.53", 
        "agent_test_statistic": f"{t_stat:.2f}" if t_stat is not None else ""
    }
    # Independent t-test: t(154) means df=154, so n1+n2≈156. Assume roughly balanced
    add_statistical_replication_fields(
        test_result_f4_4, f4_gt_4, p_val_agent_f4_4, t_stat, "t-test",
        n_agent=len(conf_anchored) if len(conf_anchored) > 2 else None,
        n2_agent=len(conf_calib) if len(conf_calib) > 2 else None,
        n_human=78,  # Approximate from df=154
        n2_human=78,
        independent=True
    )
    test_results.append(test_result_f4_4)

    # --- Experiment 2 Analysis ---
    exp2_finding_map = {f["finding_id"]: f for f in ground_truth["studies"][1]["findings"]}
    e2_data = agent_data["exp_2_discredited_anchor"]
    high_judgments = [d["choice"] for d in e2_data if d["anchor_type"] == "high" and d.get("choice")]
    low_judgments = [d["choice"] for d in e2_data if d["anchor_type"] == "low" and d.get("choice")]
    
    # F1 Test 1: High Anchor Judged Low (Human: 28% wrong-way)
    f2_gt_1 = exp2_finding_map["F1"]["statistical_tests"][0]
    # Hypothesis: Anchoring persists even when discredited
    # Human: 28% judged "Higher" (wrong direction) for high anchor
    # This means 72% judged correctly (toward anchor), significantly > 50% chance
    # Calculate chi-square: test if 72% differs from 50% (chance level)
    # We need human sample size - using n=103 from Experiment 1 (only sample size reported in paper)
    human_n_exp2 = 103  # Using n=103 from Experiment 1 as reported sample size
    human_wrong_rate_high = 0.28
    human_correct_count_high = int((1 - human_wrong_rate_high) * human_n_exp2)  # 72% correct
    human_wrong_count_high = int(human_wrong_rate_high * human_n_exp2)  # 28% wrong
    expected = human_n_exp2 * 0.5
    chi2_human_high = ((human_correct_count_high - expected)**2 / expected) + ((human_wrong_count_high - expected)**2 / expected)
    bf_h_high = calc_bf_chisq(chi2_human_high, human_n_exp2)
    pi_h_high = calc_posteriors_3way(bf_h_high, 1, prior_odds=10.0)
    
    contingency_agent_exp2_1 = None
    if len(high_judgments) > 5:
        count_wrong = high_judgments.count('A')  # A = Higher, but anchor is high, so wrong
        count_correct = len(high_judgments) - count_wrong
        rate = count_wrong / len(high_judgments)
        # Build 2x2 contingency: [correct, wrong] vs [expected_correct, expected_wrong]
        expected_correct = 0.5 * len(high_judgments)
        expected_wrong = 0.5 * len(high_judgments)
        contingency_agent_exp2_1 = [[count_correct, count_wrong], [expected_correct, expected_wrong]]
        chi2 = ((count_wrong - 0.5*len(high_judgments))**2 / (0.5*len(high_judgments))) * 2
        bf_a = calc_bf_chisq(chi2, len(high_judgments))
        # Direction for chi-square: we treat count_correct > count_wrong as match (1)
        a_dir = 1 if count_correct > count_wrong else -1
        pi_a = calc_posteriors_3way(bf_a, a_dir)
        # For chi-square, p-value from chi2 distribution
        from scipy.stats import chi2 as chi2_dist
        p_val_agent_exp2_1 = 1 - chi2_dist.cdf(chi2, 1) if chi2 > 0 else 1.0
        reason = f"rate={rate:.2f}, n={len(high_judgments)}"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        p_val_agent_exp2_1 = None
        chi2 = None
        reason = "Insufficient data"

    # Build human contingency table
    contingency_human_exp2_1 = [[human_correct_count_high, human_wrong_count_high], [expected, expected]]

    test_result_exp2_1 = {
        "study_id": "Experiment 2", "sub_study_id": "exp_2_discredited_anchor", "finding_id": "F1",
        "test_name": "Discredited Anchor Judgment", "scenario": "High Anchor",
        "pi_human": float(pi_h_high['pi_plus'] + pi_h_high['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h_high,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h_high, pi_a)),
        "pi_human_source": f"28% wrong-way (72% correct, n={human_n_exp2})", "agent_reason": reason,
        "statistical_test_type": "chi-square",
        "human_test_statistic": f"{chi2_human_high:.2f}",
        "agent_test_statistic": f"{chi2:.2f}" if chi2 is not None else ""
    }
    add_statistical_replication_fields(
        test_result_exp2_1, f2_gt_1, p_val_agent_exp2_1, chi2, "chi-square",
        contingency_agent=contingency_agent_exp2_1,
        contingency_human=contingency_human_exp2_1
    )
    test_results.append(test_result_exp2_1)
    
    # F1 Test 2: Low Anchor Judged High (Human: 15% wrong-way)
    f2_gt_2 = exp2_finding_map["F1"]["statistical_tests"][1]
    # Human: 15% judged "Lower" (wrong direction) for low anchor
    # This means 85% judged correctly (toward anchor), significantly > 50% chance
    human_wrong_rate_low = 0.15
    human_correct_count_low = int((1 - human_wrong_rate_low) * human_n_exp2)  # 85% correct
    human_wrong_count_low = int(human_wrong_rate_low * human_n_exp2)  # 15% wrong
    chi2_human_low = ((human_correct_count_low - expected)**2 / expected) + ((human_wrong_count_low - expected)**2 / expected)
    bf_h_low = calc_bf_chisq(chi2_human_low, human_n_exp2)
    pi_h_low = calc_posteriors_3way(bf_h_low, 1, prior_odds=10.0)
    
    contingency_agent_exp2_2 = None
    if len(low_judgments) > 5:
        count_wrong = low_judgments.count('B')  # B = Lower, but anchor is low, so wrong
        count_correct = len(low_judgments) - count_wrong
        rate = count_wrong / len(low_judgments)
        expected_correct = 0.5 * len(low_judgments)
        expected_wrong = 0.5 * len(low_judgments)
        contingency_agent_exp2_2 = [[count_correct, count_wrong], [expected_correct, expected_wrong]]
        chi2 = ((count_wrong - 0.5*len(low_judgments))**2 / (0.5*len(low_judgments))) * 2
        bf_a = calc_bf_chisq(chi2, len(low_judgments))
        a_dir = 1 if count_correct > count_wrong else -1
        pi_a = calc_posteriors_3way(bf_a, a_dir)
        from scipy.stats import chi2 as chi2_dist
        p_val_agent_exp2_2 = 1 - chi2_dist.cdf(chi2, 1) if chi2 > 0 else 1.0
        reason = f"rate={rate:.2f}, n={len(low_judgments)}"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        p_val_agent_exp2_2 = None
        chi2 = None
        reason = "Insufficient data"

    # Build human contingency table
    contingency_human_exp2_2 = [[human_correct_count_low, human_wrong_count_low], [expected, expected]]

    test_result_exp2_2 = {
        "study_id": "Experiment 2", "sub_study_id": "exp_2_discredited_anchor", "finding_id": "F1",
        "test_name": "Discredited Anchor Judgment", "scenario": "Low Anchor",
        "pi_human": float(pi_h_low['pi_plus'] + pi_h_low['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h_low,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h_low, pi_a)),
        "pi_human_source": f"15% wrong-way (85% correct, n={human_n_exp2})", "agent_reason": reason,
        "statistical_test_type": "chi-square",
        "human_test_statistic": f"{chi2_human_low:.2f}",
        "agent_test_statistic": f"{chi2:.2f}" if chi2 is not None else ""
    }
    add_statistical_replication_fields(
        test_result_exp2_2, f2_gt_2, p_val_agent_exp2_2, chi2, "chi-square",
        contingency_agent=contingency_agent_exp2_2,
        contingency_human=contingency_human_exp2_2
    )
    test_results.append(test_result_exp2_2)
    
    # --- Experiment 3 Analysis ---
    exp3_finding_map = {f["finding_id"]: f for f in ground_truth["studies"][2]["findings"]}
    e3_data = agent_data["exp_3_wtp_estimation"]
    
    # F1 Test 1: High Anchor WTP (Human: 24% anchored, baseline 4.2%)
    f3_gt_1 = exp3_finding_map["F1"]["statistical_tests"][0]
    # Hypothesis: Anchoring occurs in WTP tasks
    # Human: 24% exceeded high anchor vs 4.2% baseline - test if 24% > 4.2%
    # We need human sample size - using n=103 from Experiment 1 (only sample size reported in paper)
    human_n_exp3 = 103  # Using n=103 from Experiment 1 as reported sample size
    human_anchored_rate_high = 0.24
    human_baseline_high = 0.042
    human_anchored_count_high = int(human_anchored_rate_high * human_n_exp3)
    human_baseline_count_high = int(human_baseline_high * human_n_exp3)
    # Test if anchored rate (24%) differs from baseline (4.2%)
    # Use chi-square test for proportions
    total_expected = (human_anchored_rate_high + human_baseline_high) * human_n_exp3 / 2
    chi2_human_high_wtp = ((human_anchored_count_high - total_expected)**2 / total_expected) + ((human_baseline_count_high - total_expected)**2 / total_expected)
    bf_h_high_wtp = calc_bf_chisq(chi2_human_high_wtp, human_n_exp3)
    pi_h_high_wtp = calc_posteriors_3way(bf_h_high_wtp, 1, prior_odds=10.0)
    
    # Count WTP estimates that exceed high anchor (anchored = estimate > anchor)
    high_wtp_data = [d for d in e3_data if d.get("anchor_type") == "high" and d.get("estimate") is not None and d.get("anchor_val") is not None]
    
    contingency_agent_exp3_1 = None
    if len(high_wtp_data) > 5:
        # Count estimates that exceed the anchor (anchored behavior)
        anchored_count = sum(1 for d in high_wtp_data if d.get("estimate", 0) > d.get("anchor_val", 0))
        not_anchored_count = len(high_wtp_data) - anchored_count
        rate = anchored_count / len(high_wtp_data) if high_wtp_data else 0
        expected_anchored = 0.5 * len(high_wtp_data)
        expected_not = 0.5 * len(high_wtp_data)
        contingency_agent_exp3_1 = [[anchored_count, not_anchored_count], [expected_anchored, expected_not]]
        chi2 = ((anchored_count - 0.5*len(high_wtp_data))**2 / (0.5*len(high_wtp_data))) * 2
        bf_a = calc_bf_chisq(chi2, len(high_wtp_data))
        a_dir = 1 if anchored_count > not_anchored_count else -1
        pi_a = calc_posteriors_3way(bf_a, a_dir)
        from scipy.stats import chi2 as chi2_dist
        p_val_agent_exp3_1 = 1 - chi2_dist.cdf(chi2, 1) if chi2 > 0 else 1.0
        reason = f"rate={rate:.2f}, n={len(high_wtp_data)}"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        p_val_agent_exp3_1 = None
        chi2 = None
        reason = "Insufficient data"

    # Build human contingency table: anchored vs baseline
    human_not_anchored_high = human_n_exp3 - human_anchored_count_high
    human_baseline_not = human_n_exp3 - human_baseline_count_high
    contingency_human_exp3_1 = [[human_anchored_count_high, human_not_anchored_high], [human_baseline_count_high, human_baseline_not]]

    test_result_exp3_1 = {
        "study_id": "Experiment 3", "sub_study_id": "exp_3_wtp_estimation", "finding_id": "F1",
        "test_name": "Percentage of anchored estimates (High)", "scenario": "High Anchor",
        "pi_human": float(pi_h_high_wtp['pi_plus'] + pi_h_high_wtp['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h_high_wtp,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h_high_wtp, pi_a)),
        "pi_human_source": f"24% anchored (vs 4.2% baseline, n={human_n_exp3})", "agent_reason": reason,
        "statistical_test_type": "chi-square",
        "human_test_statistic": f"{chi2_human_high_wtp:.2f}",
        "agent_test_statistic": f"{chi2:.2f}" if chi2 is not None else ""
    }
    add_statistical_replication_fields(
        test_result_exp3_1, f3_gt_1, p_val_agent_exp3_1, chi2, "chi-square",
        contingency_agent=contingency_agent_exp3_1,
        contingency_human=contingency_human_exp3_1
    )
    test_results.append(test_result_exp3_1)
    
    # F1 Test 2: Low Anchor WTP (Human: 15% anchored, baseline 4.6%)
    f3_gt_2 = exp3_finding_map["F1"]["statistical_tests"][1]
    # Human: 15% anchored vs 4.6% baseline - test if 15% > 4.6%
    human_anchored_rate_low = 0.15
    human_baseline_low = 0.046
    human_anchored_count_low = int(human_anchored_rate_low * human_n_exp3)
    human_baseline_count_low = int(human_baseline_low * human_n_exp3)
    total_expected_low = (human_anchored_rate_low + human_baseline_low) * human_n_exp3 / 2
    chi2_human_low_wtp = ((human_anchored_count_low - total_expected_low)**2 / total_expected_low) + ((human_baseline_count_low - total_expected_low)**2 / total_expected_low)
    bf_h_low_wtp = calc_bf_chisq(chi2_human_low_wtp, human_n_exp3)
    pi_h_low_wtp = calc_posteriors_3way(bf_h_low_wtp, 1, prior_odds=10.0)
    
    # Count WTP estimates that are below low anchor (anchored = estimate < anchor)
    low_wtp_data = [d for d in e3_data if d.get("anchor_type") == "low" and d.get("estimate") is not None and d.get("anchor_val") is not None]
    
    contingency_agent_exp3_2 = None
    if len(low_wtp_data) > 5:
        # Count estimates that are below the anchor (anchored behavior)
        anchored_count = sum(1 for d in low_wtp_data if d.get("estimate", float('inf')) < d.get("anchor_val", 0))
        not_anchored_count = len(low_wtp_data) - anchored_count
        rate = anchored_count / len(low_wtp_data) if low_wtp_data else 0
        expected_anchored = 0.5 * len(low_wtp_data)
        expected_not = 0.5 * len(low_wtp_data)
        contingency_agent_exp3_2 = [[anchored_count, not_anchored_count], [expected_anchored, expected_not]]
        chi2 = ((anchored_count - 0.5*len(low_wtp_data))**2 / (0.5*len(low_wtp_data))) * 2
        bf_a = calc_bf_chisq(chi2, len(low_wtp_data))
        a_dir = 1 if anchored_count > not_anchored_count else -1
        pi_a = calc_posteriors_3way(bf_a, a_dir)
        from scipy.stats import chi2 as chi2_dist
        p_val_agent_exp3_2 = 1 - chi2_dist.cdf(chi2, 1) if chi2 > 0 else 1.0
        reason = f"rate={rate:.2f}, n={len(low_wtp_data)}"
    else:
        pi_a = {'pi_plus': 0.0, 'pi_minus': 0.0, 'pi_zero': 1.0}
        p_val_agent_exp3_2 = None
        chi2 = None
        reason = "Insufficient data"
    
    # Build human contingency table: anchored vs baseline
    human_not_anchored_low = human_n_exp3 - human_anchored_count_low
    human_baseline_not_low = human_n_exp3 - human_baseline_count_low
    contingency_human_exp3_2 = [[human_anchored_count_low, human_not_anchored_low], [human_baseline_count_low, human_baseline_not_low]]

    test_result_exp3_2 = {
        "study_id": "Experiment 3", "sub_study_id": "exp_3_wtp_estimation", "finding_id": "F1",
        "test_name": "Percentage of anchored estimates (Low)", "scenario": "Low Anchor",
        "pi_human": float(pi_h_low_wtp['pi_plus'] + pi_h_low_wtp['pi_minus']),
        "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
        "pi_human_3way": pi_h_low_wtp,
        "pi_agent_3way": pi_a,
        "pas": float(calc_pas(pi_h_low_wtp, pi_a)),
        "pi_human_source": f"15% anchored (vs 4.6% baseline, n={human_n_exp3})", "agent_reason": reason,
        "statistical_test_type": "chi-square",
        "human_test_statistic": f"{chi2_human_low_wtp:.2f}",
        "agent_test_statistic": f"{chi2:.2f}" if chi2 is not None else ""
    }
    add_statistical_replication_fields(
        test_result_exp3_2, f3_gt_2, p_val_agent_exp3_2, chi2, "chi-square",
        contingency_agent=contingency_agent_exp3_2,
        contingency_human=contingency_human_exp3_2
    )
    test_results.append(test_result_exp3_2)

    # 3. Two-level Weighted Aggregation
    # Add test_weight to each test result
    for tr in test_results:
        finding_id = tr.get("finding_id")
        test_name = tr.get("test_name")
        if finding_id and test_name:
            tr["test_weight"] = float(test_weights.get((finding_id, test_name), 1.0))
        else:
            tr["test_weight"] = 1.0
    
    # Level 1: Aggregate tests into findings (weighted by test weights)
    finding_results = []
    from collections import defaultdict
    grouped = defaultdict(list)
    for tr in test_results:
        key = (tr["sub_study_id"], tr["finding_id"])
        grouped[key].append(tr)
    
    for (ss_id, f_id), tests in grouped.items():
        # Weighted average: Σ (PAS * weight) / Σ weights
        total_weighted_pas = sum(tr["pas"] * tr.get("test_weight", 1.0) for tr in tests)
        total_weight = sum(tr.get("test_weight", 1.0) for tr in tests)
        finding_score = total_weighted_pas / total_weight if total_weight > 0 else 0.5
        
        finding_weight = finding_weights.get(f_id, 1.0)
        finding_results.append({
            "sub_study_id": ss_id,
            "finding_id": f_id,
            "finding_score": float(finding_score),
            "finding_weight": float(finding_weight),
            "n_tests": len(tests)
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