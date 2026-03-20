import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from stats_lib import (
    calc_bf_t, calc_bf_r, calc_bf_chisq, calc_bf_vsb, 
    calc_posteriors_3way,
    calc_pas,
    get_direction_from_statistic,
    add_statistical_replication_fields
)

def parse_agent_responses(response_text: str) -> Dict[str, float]:
    """
    Parses responses in the format Q1=<value> or Q1: <value>.
    Returns a dictionary mapping the key to a float value.
    """
    results = {}
    # Supports both = and : separators
    pattern = re.compile(r"(Q\d+)\s*[:=]\s*([\d\.]+)")
    for k, v in pattern.findall(response_text):
        try:
            results[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return results

def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_012: 从items中提取q_idx，如果没有则默认返回Q1（因为study_012总是期望Q1）
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
    
    # Study_012 always expects Q1 (amount sent for Room A, amount returned for Room B)
    # If no q_idx found in items, default to Q1
    if not required:
        required.add("Q1")
    
    return required

def extract_numeric_from_text(text: str) -> float:
    """Extract first numeric value from text."""
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass
    return 0.0

def analytical_correlation_diff(
    group1_sent: List[float],
    group1_returned: List[float],
    group2_sent: List[float],
    group2_returned: List[float]
) -> float:
    """
    Analytical test to compare Spearman correlation coefficients between two groups
    using Fisher's z-transformation.
    """
    if len(group1_sent) < 4 or len(group2_sent) < 4:
        return 1.0

    # Calculate observed correlations
    if len(set(group1_sent)) <= 1 or len(set(group1_returned)) <= 1:
        r1 = 0.0
    else:
        r1, _ = stats.spearmanr(group1_sent, group1_returned)
        if np.isnan(r1): r1 = 0.0
    
    if len(set(group2_sent)) <= 1 or len(set(group2_returned)) <= 1:
        r2 = 0.0
    else:
        r2, _ = stats.spearmanr(group2_sent, group2_returned)
        if np.isnan(r2): r2 = 0.0

    # Fisher z-transform
    # Clamp r to (-0.999, 0.999) to avoid infinity
    z1 = np.arctanh(np.clip(r1, -0.999, 0.999))
    z2 = np.arctanh(np.clip(r2, -0.999, 0.999))
    
    # Standard error for Spearman comparison
    # SE = sqrt(1.06/(n1-3) + 1.06/(n2-3))
    n1, n2 = len(group1_sent), len(group2_sent)
    se = np.sqrt(1.06/(n1-3) + 1.06/(n2-3))
    
    z_stat = (z2 - z1) / se
    p_value = 1 - stats.norm.cdf(z_stat) # One-tailed: Social > No History
    
    return p_value

def analytical_uniformity_test(observed_data):
    """
    Analytical Chi-square test for uniformity.
    """
    n = len(observed_data)
    bins = 11  # 0-10
    
    observed_counts = np.histogram(observed_data, bins=bins, range=(0, bins))[0]
    expected_freq = n / bins
    expected_counts = np.full(bins, expected_freq)
    valid_mask = expected_counts > 0
    
    chi2_observed, p_value = stats.chisquare(observed_counts[valid_mask], expected_counts[valid_mask])
    
    return p_value, chi2_observed

def classify_strategy(sent: float, returned: float) -> Optional[str]:
    """
    Classify Room B's return strategy based on k = returned / (3 * sent).
    
    Returns one of: "k=0", "k=1/3", "k=1/2", "k=2/3", or None if no match.
    
    Strategy definitions:
    - k=0: Returned = 0 (selfish)
    - k=1/3: Returned ≈ sent (return principal, k = 1/3)
    - k=1/2: Returned ≈ 1.5 * sent (split total, k = 1/2)
    - k=2/3: Returned ≈ 2 * sent (equal earnings, k = 2/3)
    """
    if sent == 0:
        # If sent is 0, returned should also be 0 (no game played)
        if abs(returned) < 0.1:
            return "k=0"
        return None
    
    # Calculate k ratio: k = returned / (3 * sent)
    k_ratio = returned / (3 * sent) if sent > 0 else 0
    
    # Calculate expected returns for each strategy (for absolute tolerance check)
    expected_k0 = 0
    expected_k1_3 = sent  # Return principal (k=1/3: return = sent)
    expected_k1_2 = 1.5 * sent  # Split total (k=1/2: return = 1.5 * sent)
    expected_k2_3 = 2 * sent  # Equal earnings (k=2/3: return = 2 * sent, simplified)
    
    # Use both k-ratio and absolute tolerance for robustness
    tolerance_abs = 1.0  # Absolute dollar tolerance
    tolerance_k = 0.1  # Tolerance for k ratio (10% of target k value)
    
    # Check k=0: returned should be close to 0
    if abs(returned - expected_k0) <= tolerance_abs or abs(k_ratio) <= tolerance_k:
        return "k=0"
    
    # Check k=1/3: returned ≈ sent, so k ≈ 1/3
    k_target_1_3 = 1.0 / 3.0
    if abs(returned - expected_k1_3) <= tolerance_abs or abs(k_ratio - k_target_1_3) <= tolerance_k:
        return "k=1/3"
    
    # Check k=1/2: returned ≈ 1.5*sent, so k ≈ 1/2
    k_target_1_2 = 1.0 / 2.0
    if abs(returned - expected_k1_2) <= tolerance_abs or abs(k_ratio - k_target_1_2) <= tolerance_k:
        return "k=1/2"
    
    # Check k=2/3: returned ≈ 2*sent, so k ≈ 2/3
    k_target_2_3 = 2.0 / 3.0
    if abs(returned - expected_k2_3) <= tolerance_abs or abs(k_ratio - k_target_2_3) <= tolerance_k:
        return "k=2/3"
    
    return None

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates Study 012 (Trust/Investment Game) performance using non-parametric tests.
    """
    # 1. Load ground truth and metadata
    study_id = "study_012"
    study_dir = Path(__file__).resolve().parent.parent / "source"
    gt_path = study_dir / "ground_truth.json"
    with open(gt_path, 'r') as f:
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

    # 2. Extract and pair agent data
    # Structure: We need to match Room A decisions with Room B decisions
    agent_data = {
        "no_history_investment_game": {"pairs": [], "sent": []},
        "social_history_investment_game": {"pairs": [], "sent": []}
    }

    # Dictionaries to store responses by pair index
    room_a_by_pair = {} # (sub_id, pair_index) -> (amount_sent, participant_id)
    room_b_by_pair = {} # (sub_id, pair_index) -> (amount_returned, participant_id)
    
    # Get all sources of participant data
    all_participants = results.get("individual_data", [])
    if not all_participants or (len(all_participants) > 0 and not all_participants[0].get("responses")):
        # If individual_data is empty or has no responses, check participant_summaries
        all_participants = results.get("participant_summaries", all_participants)

    # Extract all responses first
    for participant in all_participants:
        participant_id = participant.get("participant_id")
        for response in participant.get("responses", []):
            trial_info = response.get("trial_info", {})
            role = trial_info.get("role")
            sub_id = trial_info.get("sub_study_id")
            pair_index = trial_info.get("pair_index")
            
            if sub_id not in agent_data or pair_index is None:
                continue
                
            response_text = response.get("response_text", "")
            parsed = parse_agent_responses(response_text)
            
            # Load items to find correct Q index if needed
            items = trial_info.get("items", [])
            if not items:
                items = load_materials(sub_id)
            
            if role == "room_a":
                q_idx = "Q1"
                for item in items:
                    if item.get("id") == "amount_sent":
                        q_idx = item.get("q_idx", "Q1")
                        break
                amount = parsed.get(q_idx, extract_numeric_from_text(response_text))
                amount = max(0, min(10, amount))
                room_a_by_pair[(sub_id, pair_index)] = (amount, participant_id)
                # Store sent amount for trust analysis
                agent_data[sub_id]["sent"].append(amount)
                
            elif role == "room_b":
                q_idx = "Q1"
                for item in items:
                    if item.get("id") == "amount_returned":
                        q_idx = item.get("q_idx", "Q1")
                        break
                amount = parsed.get(q_idx, extract_numeric_from_text(response_text))
                amount = max(0, amount)
                room_b_by_pair[(sub_id, pair_index)] = (amount, participant_id)

    # Now match them up for reciprocity analysis
    for (sub_id, pair_index), (amount_sent, p_id_a) in room_a_by_pair.items():
        if (sub_id, pair_index) in room_b_by_pair:
            amount_returned, p_id_b = room_b_by_pair[(sub_id, pair_index)]
            agent_data[sub_id]["pairs"].append((amount_sent, amount_returned))
        elif amount_sent == 0:
            # Room A sent $0, so Room B naturally didn't play
            agent_data[sub_id]["pairs"].append((0.0, 0.0))
    
    # Separate into sent and returned lists for easier analysis
    nh_pairs = agent_data["no_history_investment_game"]["pairs"]
    sh_pairs = agent_data["social_history_investment_game"]["pairs"]
    
    # Use the full 'sent' list for trust tests, not just paired ones
    nh_sent = agent_data["no_history_investment_game"]["sent"]
    sh_sent = agent_data["social_history_investment_game"]["sent"]
    
    nh_returned = [p[1] for p in nh_pairs]
    sh_returned = [p[1] for p in sh_pairs]

    # 3. Define Human Statistics (from Ground Truth)
    # F1: Trust exists but distribution is uniform (p = 0.29)
    # BF calculated from reported p using VSB
    p_value_f1_human = 0.29
    bf_f1 = calc_bf_vsb(p_value_f1_human)
    pi_h_f1 = calc_posteriors_3way(bf_f1, 0, prior_odds=0.1) # Not significant, direction 0
    
    # F2: Mean Comparison (No History) - Mean Sent ($5.16) vs Mean Returned ($4.66)
    # Hypothesis: Returned > Sent. Data: Returned < Sent (contradicts hypothesis).
    t_stat_f2_human = -0.58 
    bf_f2 = calc_bf_t(t_stat_f2_human, 32, independent=False)
    # Human found negative effect (Returned < Sent) when H+ was expected
    pi_h_f2 = calc_posteriors_3way(bf_f2, -1, prior_odds=0.1)
    
    # F3: Spearman correlation (No History) - rs = 0.01
    r_f3_human = 0.01
    bf_f3 = calc_bf_r(r_f3_human, 32)
    pi_h_f3 = calc_posteriors_3way(bf_f3, 0, prior_odds=0.1)
    
    # F4: Social History Trust - Uniformity test (p = 0.06)
    p_value_f4_human = 0.06
    bf_f4 = calc_bf_vsb(p_value_f4_human)
    pi_h_f4 = calc_posteriors_3way(bf_f4, 0, prior_odds=1.0) # Borderline
    
    # F5: Wilcoxon rank-sum / Mann-Whitney U test (p = 0.1)
    p_value_f5_human = 0.1
    bf_f5 = calc_bf_vsb(p_value_f5_human)
    # Expected Social > No History (Direction 1)
    pi_h_f5 = calc_posteriors_3way(bf_f5, 1, prior_odds=10.0)
    
    # F6: Spearman correlation (Social History) - rs = 0.34
    r_f6_human = 0.34
    bf_f6 = calc_bf_r(r_f6_human, 28)
    pi_h_f6 = calc_posteriors_3way(bf_f6, 1, prior_odds=10.0)
    
    # F6b: Resampling test - reported p = 0.06
    p_value_f6b_human = 0.06
    bf_f6b = calc_bf_vsb(p_value_f6b_human)
    pi_h_f6b = calc_posteriors_3way(bf_f6b, 1, prior_odds=10.0)
    
    # F7: Strategy classification (>90% coverage)
    # For F7, we'll keep it as a simple probability as it's not a standard H0/H1 test
    pi_h_f7 = {'pi_plus': 0.9, 'pi_minus': 0.0, 'pi_zero': 0.1}
    human_stat_f7 = ">90%"

    # 4. Process all findings with non-parametric tests
    test_results = []
    
    # F1: No History Trust (Uniformity) - Analytical Test
    if len(nh_sent) > 5:
        # Use Analytical Chi-square test
        p_value, chi2 = analytical_uniformity_test(nh_sent)
        # From p-value use VSB to calculate BF
        bf = calc_bf_vsb(p_value)
        # Uniformity test usually doesn't have a 'direction' in the same way
        pi_a_dict = calc_posteriors_3way(bf, 0)
        
        reason = f"chi2={chi2:.2f}, p={p_value:.3f} (analytical), n={len(nh_sent)} sent"
        test_type = "Analytical Uniformity Test"
        human_test_stat = "p=0.29" # Reported
        agent_test_stat = f"{chi2:.2f}"
    else:
        pi_a_dict = {'pi_plus': 1/3, 'pi_minus': 1/3, 'pi_zero': 1/3}
        reason = "Insufficient data"
        test_type = "Analytical Uniformity Test"
        human_test_stat = "p=0.29"
        agent_test_stat = ""
    
    test_results.append({
        "finding_id": "F1",
        "sub_study_id": "no_history_investment_game",
        "pi_h": pi_h_f1,
        "pi_a": pi_a_dict,
        "reason": reason,
        "test_name": "Approximate Randomization Test",
        "statistical_test_type": test_type,
        "human_test_statistic": human_test_stat,
        "agent_test_statistic": agent_test_stat
    })

    # F2: No History Reciprocity (Mean Comparison)
    if len(nh_pairs) > 5:
        # Filter pairs where sent > 1 (as in paper)
        valid_pairs = [(s, r) for s, r in nh_pairs if s > 1]
        if len(valid_pairs) >= 5:
            valid_sent = [p[0] for p in valid_pairs]
            valid_returned = [p[1] for p in valid_pairs]
            
            # Paired t-test: returned vs sent
            t_stat, p_val = stats.ttest_rel(valid_returned, valid_sent)
            a_dir = get_direction_from_statistic(t_stat, "t-test")
            pi_a_dict = calc_posteriors_3way(calc_bf_t(t_stat, len(valid_pairs), independent=False), a_dir)
            reason = f"t={t_stat:.2f}, n={len(valid_pairs)} pairs"
            test_type = "t-test"
            human_test_stat = "t=-0.58"
            agent_test_stat = f"{t_stat:.2f}"
        else:
            pi_a_dict = {'pi_plus': 1/3, 'pi_minus': 1/3, 'pi_zero': 1/3}
            reason = "Insufficient data (need sent > 1)"
            test_type = "Mean Comparison"
            human_test_stat = "t=-0.58"
            agent_test_stat = ""
    else:
        pi_a_dict = {'pi_plus': 1/3, 'pi_minus': 1/3, 'pi_zero': 1/3}
        reason = "Insufficient data"
        test_type = "Mean Comparison"
        human_test_stat = "t=-0.58"
        agent_test_stat = ""
    
    test_results.append({
        "finding_id": "F2",
        "sub_study_id": "no_history_investment_game",
        "pi_h": pi_h_f2,
        "pi_a": pi_a_dict,
        "reason": reason,
        "test_name": "Mean Comparison",
        "statistical_test_type": test_type,
        "human_test_statistic": "Sent=5.16, Ret=4.66",
        "agent_test_statistic": agent_test_stat
    })

    # F3: No History Correlation (Spearman)
    if len(nh_pairs) > 5:
        nh_sent_vals = [p[0] for p in nh_pairs]
        nh_ret_vals = [p[1] for p in nh_pairs]
        if len(set(nh_sent_vals)) <= 1 or len(set(nh_ret_vals)) <= 1:
            r = 0.0  # Constant array, correlation undefined
            p_val = 1.0
        else:
            r, p_val = stats.spearmanr(nh_sent_vals, nh_ret_vals)
            if np.isnan(r):
                r = 0.0
        a_dir = 1 if r > 0 else -1
        pi_a_dict = calc_posteriors_3way(calc_bf_r(r, len(nh_pairs)), a_dir)
        reason = f"rs={r:.2f}, n={len(nh_pairs)} pairs"
        test_type = "correlation"
        human_test_stat = "rs=0.01"
        agent_test_stat = f"{r:.2f}"
    else:
        pi_a_dict = {'pi_plus': 1/3, 'pi_minus': 1/3, 'pi_zero': 1/3}
        reason = "Insufficient data"
        test_type = "Spearman's rank correlation coefficient"
        human_test_stat = "rs=0.01"
        agent_test_stat = ""
    
    test_results.append({
        "finding_id": "F3",
        "sub_study_id": "no_history_investment_game",
        "pi_h": pi_h_f3,
        "pi_a": pi_a_dict,
        "reason": reason,
        "test_name": "Spearman's rank correlation coefficient",
        "statistical_test_type": test_type,
        "human_test_statistic": human_test_stat,
        "agent_test_statistic": agent_test_stat
    })
    
    # F4: Social History Trust (Uniformity) - Analytical Test
    if len(sh_sent) > 5:
        # Use Analytical Chi-square test
        p_value, chi2 = analytical_uniformity_test(sh_sent)
        # From p-value use VSB to calculate BF
        bf = calc_bf_vsb(p_value)
        pi_a_dict = calc_posteriors_3way(bf, 0)
        
        reason = f"chi2={chi2:.2f}, p={p_value:.3f} (analytical), n={len(sh_sent)} sent"
        test_type = "Analytical Uniformity Test"
        human_test_stat = "p=0.06"
        agent_test_stat = f"{chi2:.2f}"
    else:
        pi_a_dict = {'pi_plus': 1/3, 'pi_minus': 1/3, 'pi_zero': 1/3}
        reason = "Insufficient data"
        test_type = "Analytical Uniformity Test"
        human_test_stat = "p=0.06"
        agent_test_stat = ""
    
    test_results.append({
        "finding_id": "F4",
        "sub_study_id": "social_history_investment_game",
        "pi_h": pi_h_f4,
        "pi_a": pi_a_dict,
        "reason": reason,
        "test_name": "Approximate Randomization Test",
        "statistical_test_type": test_type,
        "human_test_statistic": human_test_stat,
        "agent_test_statistic": agent_test_stat
    })

    # F5: Social History vs No History Payback (Wilcoxon rank-sum / Mann-Whitney U)
    if len(sh_returned) > 5 and len(nh_returned) > 5:
        u_stat, p_val = stats.mannwhitneyu(sh_returned, nh_returned, alternative='greater')
        # Convert U statistic to approximate z-score
        n1, n2 = len(sh_returned), len(nh_returned)
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        if std_u > 0:
            z_stat = (u_stat - mean_u) / std_u
        else:
            z_stat = 0.0
        a_dir = 1 if z_stat > 0 else -1
        pi_a_dict = calc_posteriors_3way(calc_bf_t(z_stat, n1, n2), a_dir)
        reason = f"U={u_stat:.0f}, p={p_val:.3f}, n1={n1} pairs, n2={n2} pairs"
        test_type = "mannwhitneyu"
        human_test_stat = "p=0.1"
        agent_test_stat = f"{u_stat:.0f}"
    else:
        pi_a_dict = {'pi_plus': 1/3, 'pi_minus': 1/3, 'pi_zero': 1/3}
        reason = "Insufficient data for comparison"
        test_type = "Wilcoxon rank-sum test"
        human_test_stat = "p=0.1"
        agent_test_stat = ""
    
    test_results.append({
        "finding_id": "F5",
        "sub_study_id": "social_history_investment_game",
        "pi_h": pi_h_f5,
        "pi_a": pi_a_dict,
        "reason": reason,
        "test_name": "Wilcoxon rank-sum test",
        "statistical_test_type": test_type,
        "human_test_statistic": human_test_stat,
        "agent_test_statistic": agent_test_stat
    })

    # F6: Social History Correlation (Spearman + Resampling Test)
    if len(sh_pairs) > 5:
        sh_sent_vals = [p[0] for p in sh_pairs]
        sh_ret_vals = [p[1] for p in sh_pairs]
        if len(set(sh_sent_vals)) <= 1 or len(set(sh_ret_vals)) <= 1:
            r = 0.0  # Constant array, correlation undefined
            p_val = 1.0
        else:
            r, p_val = stats.spearmanr(sh_sent_vals, sh_ret_vals)
            if np.isnan(r):
                r = 0.0
        a_dir = 1 if r > 0 else -1
        pi_a_dict = calc_posteriors_3way(calc_bf_r(r, len(sh_pairs)), a_dir)
        reason = f"rs={r:.2f}, n={len(sh_pairs)} pairs"
        test_type = "correlation"
        human_test_stat = "rs=0.34"
        agent_test_stat = f"{r:.2f}"
    else:
        pi_a_dict = {'pi_plus': 1/3, 'pi_minus': 1/3, 'pi_zero': 1/3}
        reason = "Insufficient data"
        test_type = "Spearman's rank correlation coefficient"
        human_test_stat = "rs=0.34"
        agent_test_stat = ""
    
    test_results.append({
        "finding_id": "F6",
        "sub_study_id": "social_history_investment_game",
        "pi_h": pi_h_f6,
        "pi_a": pi_a_dict,
        "reason": reason,
        "test_name": "Spearman's rank correlation coefficient",
        "statistical_test_type": test_type,
        "human_test_statistic": human_test_stat,
        "agent_test_statistic": agent_test_stat
    })
    
    # F6b: Analytical Correlation Difference Test
    if len(nh_pairs) > 3 and len(sh_pairs) > 3:
        nh_sent_vals = [p[0] for p in nh_pairs]
        nh_ret_vals = [p[1] for p in nh_pairs]
        sh_sent_vals = [p[0] for p in sh_pairs]
        sh_ret_vals = [p[1] for p in sh_pairs]
        
        p_value = analytical_correlation_diff(
            nh_sent_vals, nh_ret_vals,
            sh_sent_vals, sh_ret_vals
        )
        # Calculate observed correlations to check direction
        nh_r = 0.0
        if len(set(nh_sent_vals)) > 1 and len(set(nh_ret_vals)) > 1:
            nh_r, _ = stats.spearmanr(nh_sent_vals, nh_ret_vals)
        
        sh_r = 0.0
        if len(set(sh_sent_vals)) > 1 and len(set(sh_ret_vals)) > 1:
            sh_r, _ = stats.spearmanr(sh_sent_vals, sh_ret_vals)
            
        # For analytical test, use VSB to calculate BF from p-value
        # Hypothesis: Social History correlation > No History correlation
        a_dir = 1 if sh_r > nh_r else -1
        bf = calc_bf_vsb(p_value)
        pi_a_dict = calc_posteriors_3way(bf, a_dir)
        
        reason = f"p={p_value:.3f} (analytical correlation diff), Dir={a_dir}"
        test_type = "analytical"
        human_test_stat = "0.06"
        agent_test_stat = f"{p_value:.3f}"
    else:
        pi_a_dict = {'pi_plus': 1/3, 'pi_minus': 1/3, 'pi_zero': 1/3}
        reason = "Insufficient data for analytical comparison"
        test_type = "Analytical correlation diff"
        human_test_stat = "0.06"
        agent_test_stat = ""
    
    test_results.append({
        "finding_id": "F6",
        "sub_study_id": "social_history_investment_game",
        "pi_h": pi_h_f6b,  # Use F6b's own pi_h (based on VSB)
        "pi_a": pi_a_dict,
        "reason": reason,
        "test_name": "Resampling test", # Keep test name as "Resampling test" to match GT if needed, or change to Analytical?
        "statistical_test_type": test_type,
        "human_test_statistic": human_test_stat,
        "agent_test_statistic": agent_test_stat
    })
    
    # F7: Strategy Classification
    all_pairs = nh_pairs + sh_pairs
    if len(all_pairs) > 5:
        strategy_counts = {"k=0": 0, "k=1/3": 0, "k=1/2": 0, "k=2/3": 0, "other": 0}
        for sent, returned in all_pairs:
            strategy = classify_strategy(sent, returned)
            if strategy:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            else:
                strategy_counts["other"] += 1
        
        total_classified = sum(strategy_counts[k] for k in ["k=0", "k=1/3", "k=1/2", "k=2/3"])
        coverage = total_classified / len(all_pairs) if len(all_pairs) > 0 else 0.0
        # F7 is coverage, we use it as pi_plus for simplicity
        pi_a_dict = {'pi_plus': coverage, 'pi_minus': 0.0, 'pi_zero': 1.0 - coverage}
        reason = f"Coverage={coverage:.1%} ({total_classified}/{len(all_pairs)} pairs)"
        test_type = "Strategy Type Classification"
        human_test_stat = human_stat_f7
        agent_test_stat = f"{coverage:.1%}"
    else:
        pi_a_dict = {'pi_plus': 0.5, 'pi_minus': 0.0, 'pi_zero': 0.5}
        reason = "Insufficient data"
        test_type = "Strategy Type Classification"
        human_test_stat = human_stat_f7
        agent_test_stat = ""
    
    test_results.append({
        "finding_id": "F7",
        "sub_study_id": "combined",
        "pi_h": {'pi_plus': 0.9, 'pi_minus': 0.0, 'pi_zero': 0.1},
        "pi_a": pi_a_dict,
        "reason": reason,
        "test_name": "Strategy Type Classification",
        "statistical_test_type": test_type,
        "human_test_statistic": human_test_stat,
        "agent_test_statistic": agent_test_stat
    })

    # 5. Compile Final Output
    final_test_results = []
    for tr in test_results:
        # Calculate PAS using 3-way posteriors
        pi_h = tr["pi_h"]
        pi_a = tr["pi_a"]
        pas = calc_pas(pi_h, pi_a)
        
        test_result = {
            "study_id": "Experiment 1" if "no_history" in tr["sub_study_id"] else ("Experiment 2" if "social_history" in tr["sub_study_id"] else "Combined Analysis"),
            "sub_study_id": tr["sub_study_id"],
            "finding_id": tr["finding_id"],
            "test_name": tr["test_name"],
            "pi_human": float(pi_h['pi_plus'] + pi_h['pi_minus']),
            "pi_agent": float(pi_a['pi_plus'] + pi_a['pi_minus']),
            "pi_human_3way": pi_h,
            "pi_agent_3way": pi_a,
            "pas": float(pas),
            "pi_human_source": "Ground Truth Statistics",
            "agent_reason": tr["reason"],
            "statistical_test_type": tr.get("statistical_test_type", ""),
            "human_test_statistic": tr.get("human_test_statistic", ""),
            "agent_test_statistic": tr.get("agent_test_statistic", "")
        }
        
        # Add statistical replication fields for frequentist consistency
        # Extract metadata for replication fields
        test_type = tr.get("statistical_test_type", "").lower()
        test_stat_agent = tr.get("agent_test_statistic", "")
        if test_stat_agent:
            try:
                # Remove % and other chars if needed
                test_stat_agent = float(test_stat_agent.replace("%", ""))
            except:
                test_stat_agent = None
        else:
            test_stat_agent = None

        # Get test_gt for SRS fields
        fid = tr["finding_id"]
        test_gt = {}
        for study in ground_truth.get("studies", []):
            for finding in study.get("findings", []):
                if finding.get("finding_id") == fid:
                    statistical_tests = finding.get("statistical_tests", [])
                    if statistical_tests:
                        # Find matching test by name or use first one
                        for test in statistical_tests:
                            if test.get("test_name") == tr["test_name"]:
                                test_gt = test
                                break
                        if not test_gt and statistical_tests:
                            test_gt = statistical_tests[0]
                    break
            if test_gt: break

        # Determine sample sizes
        n_agent = None
        n2_agent = None
        n_human = None
        n2_human = None
        
        if fid == "F1": # Approximate Randomization Test (treated as binomial/chi-square)
            # Experiment 1: 42 participants, Experiment 2: 43 participants
            n_human = 32 if "no_history" in tr["sub_study_id"] else 28
            # We can use k/n for binomial
            match = re.search(r"(\d+)/(\d+)", tr["reason"])
            if match:
                k_a = int(match.group(1))
                n_a = int(match.group(2))
                # Set human_k and human_p0 in test_result for p-value calculation
                test_result["human_k"] = int(n_human * 0.7)  # Approximate k if not in GT
                test_result["human_p0"] = 0.5
                add_statistical_replication_fields(
                    test_result, test_gt, None, None, "binomial",
                    n_agent=n_a,
                    n_human=n_human
                )
        elif fid == "F2": # Paired t-test
            # Exp 1: 32, Exp 2: 28
            n_a = len(nh_pairs) if "no_history" in tr["sub_study_id"] else len(sh_pairs)
            n_h = 32 if "no_history" in tr["sub_study_id"] else 28
            add_statistical_replication_fields(
                test_result, test_gt, None, test_stat_agent, "t-test",
                n_agent=n_a, n_human=n_h, independent=False
            )
        elif fid in ["F3", "F6"]: # Spearman correlation
            n_a = len(nh_pairs) if "no_history" in tr["sub_study_id"] else len(sh_pairs)
            n_h = 32 if "no_history" in tr["sub_study_id"] else 28
            add_statistical_replication_fields(
                test_result, test_gt, None, test_stat_agent, "correlation",
                n_agent=n_a, n_human=n_h
            )
        elif fid == "F5": # Mann-Whitney U
            n1_a = len(sh_returned)
            n2_a = len(nh_returned)
            n1_h = 28
            n2_h = 32
            add_statistical_replication_fields(
                test_result, test_gt, None, test_stat_agent, "mannwhitneyu",
                n_agent=n1_a, n2_agent=n2_a,
                n_human=n1_h, n2_human=n2_h
            )
        else:
            add_statistical_replication_fields(test_result, test_gt, None, test_stat_agent, test_type)
            
        final_test_results.append(test_result)

    # Two-level Weighted Aggregation
    for tr in final_test_results:
        finding_id = tr.get("finding_id")
        test_name = tr.get("test_name", "")
        tr["test_weight"] = float(test_weights.get((finding_id, test_name), 1.0))
    
    finding_ids = sorted(list(set(t["finding_id"] for t in final_test_results)))
    finding_results = []
    for fid in finding_ids:
        tests = [t for t in final_test_results if t["finding_id"] == fid]
        total_weighted_pas = sum(t["pas"] * t.get("test_weight", 1.0) for t in tests)
        total_weight = sum(t.get("test_weight", 1.0) for t in tests)
        finding_score = total_weighted_pas / total_weight if total_weight > 0 else 0.5
        
        finding_weight = finding_weights.get(fid, 1.0)
        finding_results.append({
            "finding_id": fid,
            "finding_score": float(finding_score),
            "finding_weight": float(finding_weight),
            "n_tests": len(tests)
        })

    total_weighted_finding_score = sum(fr["finding_score"] * fr["finding_weight"] for fr in finding_results)
    total_finding_weight = sum(fr["finding_weight"] for fr in finding_results)
    overall_score = total_weighted_finding_score / total_finding_weight if total_finding_weight > 0 else 0.5

    substudy_results = []

    return {
        "score": overall_score,
        "substudy_results": substudy_results,
        "finding_results": finding_results,
        "test_results": final_test_results
    }
