import json
import math
import re
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Minimal BAS infrastructure (inlined because study_015 stats_lib is minimal)
# ---------------------------------------------------------------------------

def calc_bf_chisq(chi2: float, n: int, df: int = 1) -> float:
    """BF10 for chi-square test via BIC approximation."""
    try:
        log_bf = (chi2 - df * math.log(n)) / 2.0
        return math.exp(log_bf)
    except Exception:
        return 1.0


def chi2_contingency_safe(observed):
    """Perform chi-square contingency test safely."""
    obs = np.array(observed)
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1) if obs.ndim == 2 else 1
    if np.sum(obs) == 0:
        return 0.0, 1.0, dof, None
    if np.any(np.sum(obs, axis=0) == 0) or np.any(np.sum(obs, axis=1) == 0):
        return 0.0, 1.0, dof, None
    try:
        chi2, p, res_dof, expected = stats.chi2_contingency(obs)
        if expected is not None and np.any(expected == 0):
            return 0.0, 1.0, res_dof, expected
        return chi2, p, res_dof, expected
    except (ValueError, RuntimeWarning):
        return 0.0, 1.0, dof, None


def calc_posteriors_3way(bf10: float, direction: int, prior_odds: float = 1.0) -> dict:
    """Calculate 3-way posterior probabilities (H+, H-, H0)."""
    if bf10 is None or math.isnan(bf10):
        pi_zero = 0.5
        pi_one = 0.5
    elif math.isinf(bf10):
        pi_zero = 0.0
        pi_one = 1.0
    else:
        odds = bf10 * prior_odds
        pi_one = odds / (1.0 + odds)
        pi_zero = 1.0 / (1.0 + odds)

    if direction > 0:
        pi_plus = pi_one * 0.9999
        pi_minus = pi_one * 0.0001
    elif direction < 0:
        pi_plus = pi_one * 0.0001
        pi_minus = pi_one * 0.9999
    else:
        pi_plus = pi_one * 0.5
        pi_minus = pi_one * 0.5

    return {
        "pi_plus": float(pi_plus),
        "pi_minus": float(pi_minus),
        "pi_zero": float(pi_zero),
    }


POSTERIOR_NULL = {"pi_plus": 0.0, "pi_minus": 0.0, "pi_zero": 1.0}


def calc_pas(pi_h, pi_a) -> float:
    """Probability Alignment Score (BAS). Supports 3-way dict and scalar inputs."""
    if isinstance(pi_h, dict) and isinstance(pi_a, dict):
        return (
            pi_h.get("pi_plus", 0.0) * pi_a.get("pi_plus", 0.0)
            + pi_h.get("pi_minus", 0.0) * pi_a.get("pi_minus", 0.0)
            + pi_h.get("pi_zero", 0.0) * pi_a.get("pi_zero", 0.0)
        )
    try:
        ph = max(1e-6, min(1.0 - 1e-6, float(pi_h)))
        pa = max(1e-6, min(1.0 - 1e-6, float(pi_a)))
        return ph * pa + (1 - ph) * (1 - pa)
    except (TypeError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# Target-option mapping per condition
# ---------------------------------------------------------------------------
# In decoy_target_first: column[0]=target → Option A=target, B=decoy
# In decoy_decoy_first:  column[0]=decoy  → Option A=decoy,  B=target

TARGET_OPTION = {
    "decoy_target_first": "A",
    "decoy_decoy_first": "B",
}


# ---------------------------------------------------------------------------
# Required interface functions
# ---------------------------------------------------------------------------

def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """Parse agent response text into {key: value} dict.

    Handles:
      CHOICE=COMPLETE, CHOICE=A, choice = b, etc.
    """
    parsed = {}
    if not response_text:
        return parsed

    pattern = re.compile(r"(CHOICE)\s*[:=]\s*(\S+)", re.IGNORECASE)
    for m in pattern.finditer(response_text):
        parsed[m.group(1).upper()] = m.group(2).upper().strip(".")

    if not parsed:
        text_up = response_text.upper()
        for token in ["COMPLETE", "DECLINE"]:
            if token in text_up:
                parsed["CHOICE"] = token
                break
        if not parsed:
            for token in ["A", "B", "C"]:
                if re.search(rf"\b{token}\b", text_up):
                    parsed["CHOICE"] = token
                    break

    return parsed


def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """Return required question identifiers for each trial."""
    return {"CHOICE"}


# ---------------------------------------------------------------------------
# Helper: determine whether participant chose the target survey
# ---------------------------------------------------------------------------

def _chose_target(condition: str, choice: str) -> bool:
    """Return True if the agent chose the target survey."""
    choice = choice.upper()
    if condition == "control":
        return choice == "COMPLETE"
    target_letter = TARGET_OPTION.get(condition)
    if target_letter is None:
        return False
    return choice == target_letter


# ---------------------------------------------------------------------------
# Helper: compute human posteriors from ground-truth p-value strings
# ---------------------------------------------------------------------------

def _parse_p(p_str: str) -> Optional[float]:
    """Parse a p-value string like 'p < 0.001' or 'p = 0.165'."""
    if not p_str or p_str == "NOT PROVIDED":
        return None
    m = re.search(r"p\s*[=:]\s*([0-9.]+)", p_str, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"p\s*<\s*([0-9.]+)", p_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 2.0
    return None


def _human_posterior_from_chi2(test_data: dict, prior_odds: float = 10.0) -> dict:
    """Compute human posterior from a ground-truth chi-square/contingency test.

    Attempts to reconstruct chi2 from raw_data, otherwise approximates from p-value.
    """
    rd = test_data.get("raw_data", {})

    # --- Try to reconstruct chi2 from 2x2 raw counts ---
    chi2_val = None
    n_total = None

    # F2 primary outcome format
    if "control_completed" in rd and "control_n" in rd and "decoy_completed" in rd and "decoy_n" in rd:
        c_yes = rd["control_completed"]
        c_n = rd["control_n"]
        d_yes = rd["decoy_completed"]
        d_n = rd["decoy_n"]
        table = np.array([[c_yes, c_n - c_yes], [d_yes, d_n - d_yes]])
        chi2_val, _, _, _ = chi2_contingency_safe(table)
        n_total = c_n + d_n

    # F3 order-effect formats
    if chi2_val is None and "decoy_target_first_completed" in rd and "control_completed" in rd:
        c_yes = rd["control_completed"]
        c_n = rd["control_n"]
        d_yes = rd["decoy_target_first_completed"]
        d_n = rd["decoy_target_first_n"]
        table = np.array([[c_yes, c_n - c_yes], [d_yes, d_n - d_yes]])
        chi2_val, _, _, _ = chi2_contingency_safe(table)
        n_total = c_n + d_n

    if chi2_val is None and "decoy_decoy_first_completed" in rd and "control_completed" in rd:
        c_yes = rd["control_completed"]
        c_n = rd["control_n"]
        d_yes = rd["decoy_decoy_first_completed"]
        d_n = rd["decoy_decoy_first_n"]
        table = np.array([[c_yes, c_n - c_yes], [d_yes, d_n - d_yes]])
        chi2_val, _, _, _ = chi2_contingency_safe(table)
        n_total = c_n + d_n

    if chi2_val is not None and n_total is not None and n_total > 0:
        bf_h = calc_bf_chisq(chi2_val, n_total, df=1)
        direction = 1
        return calc_posteriors_3way(bf_h, direction, prior_odds=prior_odds)

    # Fallback: use reported p-value to approximate BF
    p_val = _parse_p(test_data.get("p_value", ""))
    if p_val is not None and p_val < 1.0:
        # Use -2 ln(p) as a rough chi2 surrogate with n=200
        approx_chi = -2.0 * math.log(max(p_val, 1e-300))
        bf_h = calc_bf_chisq(approx_chi, 200, df=1)
        direction = 1
        return calc_posteriors_3way(bf_h, direction, prior_odds=prior_odds)

    return calc_posteriors_3way(1.0, 0, prior_odds=prior_odds)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate agent performance on the decoy-effect study."""

    study_path = Path(results.get("study_path", results.get("study_dir", ".")))
    source_path = study_path / "source"

    with open(source_path / "ground_truth.json", "r") as f:
        gt = json.load(f)
    with open(source_path / "metadata.json", "r") as f:
        meta = json.load(f)

    finding_weights = {fw["finding_id"]: fw["weight"] for fw in meta.get("findings", [])}

    # ------------------------------------------------------------------
    # 1. Parse agent responses and group by condition
    # ------------------------------------------------------------------
    agent_by_condition: Dict[str, List[bool]] = {
        "control": [],
        "decoy_target_first": [],
        "decoy_decoy_first": [],
    }

    individual_data = results.get("individual_data", results.get("participants", []))
    for participant in individual_data:
        responses = participant.get("responses", [participant])
        for resp in responses:
            trial_info = resp.get("trial_info", resp)
            condition = trial_info.get("condition", "")
            response_text = resp.get("response_text", resp.get("response", ""))

            parsed = parse_agent_responses(response_text)
            choice = parsed.get("CHOICE", "")

            if condition in agent_by_condition:
                agent_by_condition[condition].append(_chose_target(condition, choice))

    # Convenience counts
    ctrl = agent_by_condition["control"]
    dtf = agent_by_condition["decoy_target_first"]
    ddf = agent_by_condition["decoy_decoy_first"]
    decoy_all = dtf + ddf

    n_ctrl = len(ctrl)
    n_dtf = len(dtf)
    n_ddf = len(ddf)
    n_decoy = len(decoy_all)

    ctrl_completed = sum(ctrl)
    dtf_completed = sum(dtf)
    ddf_completed = sum(ddf)
    decoy_completed = dtf_completed + ddf_completed

    # ------------------------------------------------------------------
    # 2. Evaluate each finding
    # ------------------------------------------------------------------
    test_results: List[Dict[str, Any]] = []
    finding_results: List[Dict[str, Any]] = []

    # ---- F1: Randomization balance checks (weight 0.01) ----
    # Agent data has no demographics to compare, so we assign a neutral BAS.
    f1_tests = gt.get("main_study", {}).get("findings", [])
    f1_finding = next((f for f in f1_tests if f["finding_id"] == "F1"), None)
    if f1_finding:
        n_f1_tests = len(f1_finding.get("statistical_tests", []))
        for st in f1_finding.get("statistical_tests", []):
            # Cannot reconstruct balance checks from agent data (no demographics).
            test_results.append({
                "finding_id": "F1",
                "test_id": st["test_id"],
                "test_name": st["test_name"],
                "pi_human": {"pi_plus": 0.0, "pi_minus": 0.0, "pi_zero": 1.0},
                "pi_agent": {"pi_plus": 0.0, "pi_minus": 0.0, "pi_zero": 1.0},
                "pas": 1.0,  # Both are null → perfect agreement
                "test_weight": 1.0 / max(n_f1_tests, 1),
                "note": "Balance check not reconstructible from agent data; scored as null agreement.",
            })
        f1_score = 1.0
        finding_results.append({
            "finding_id": "F1",
            "finding_weight": finding_weights.get("F1", 0.01),
            "finding_score": f1_score,
        })

    # ---- F2: Primary outcome — decoy increases target completion ----
    f2_finding = next((f for f in f1_tests if f["finding_id"] == "F2"), None)
    f2_test_scores = []
    if f2_finding:
        for st in f2_finding.get("statistical_tests", []):
            test_id = st["test_id"]
            pi_h = _human_posterior_from_chi2(st)

            # Agent posterior
            if n_ctrl >= 2 and n_decoy >= 2:
                table = np.array([
                    [ctrl_completed, n_ctrl - ctrl_completed],
                    [decoy_completed, n_decoy - decoy_completed],
                ])
                chi2_a, p_a, _, _ = chi2_contingency_safe(table)
                bf_a = calc_bf_chisq(chi2_a, n_ctrl + n_decoy, df=1)
                a_rate_ctrl = ctrl_completed / n_ctrl if n_ctrl else 0
                a_rate_decoy = decoy_completed / n_decoy if n_decoy else 0
                a_dir = 1 if a_rate_decoy > a_rate_ctrl else (-1 if a_rate_decoy < a_rate_ctrl else 0)
                pi_a = calc_posteriors_3way(bf_a, a_dir)
            else:
                pi_a = dict(POSTERIOR_NULL)

            pas = calc_pas(pi_h, pi_a)
            test_results.append({
                "finding_id": "F2",
                "test_id": test_id,
                "test_name": st["test_name"],
                "pi_human": pi_h,
                "pi_agent": pi_a,
                "pas": float(pas),
                "test_weight": 1.0 / max(len(f2_finding["statistical_tests"]), 1),
                "agent_stats": {
                    "control_n": n_ctrl,
                    "control_completed": ctrl_completed,
                    "decoy_n": n_decoy,
                    "decoy_completed": decoy_completed,
                },
            })
            f2_test_scores.append(pas)

    f2_score = float(np.mean(f2_test_scores)) if f2_test_scores else 0.5
    finding_results.append({
        "finding_id": "F2",
        "finding_weight": finding_weights.get("F2", 0.35),
        "finding_score": f2_score,
    })

    # ---- F3: Order effect (target-first vs decoy-first vs control) ----
    f3_finding = next((f for f in f1_tests if f["finding_id"] == "F3"), None)
    f3_test_scores = []
    if f3_finding:
        for st in f3_finding.get("statistical_tests", []):
            test_id = st["test_id"]
            pi_h = _human_posterior_from_chi2(st)

            # Determine which sub-comparison
            if "target_first" in test_id:
                sub_n = n_dtf
                sub_completed = dtf_completed
            elif "decoy_first" in test_id:
                sub_n = n_ddf
                sub_completed = ddf_completed
            else:
                sub_n = n_decoy
                sub_completed = decoy_completed

            if n_ctrl >= 2 and sub_n >= 2:
                table = np.array([
                    [ctrl_completed, n_ctrl - ctrl_completed],
                    [sub_completed, sub_n - sub_completed],
                ])
                chi2_a, p_a, _, _ = chi2_contingency_safe(table)
                bf_a = calc_bf_chisq(chi2_a, n_ctrl + sub_n, df=1)
                a_rate_ctrl = ctrl_completed / n_ctrl if n_ctrl else 0
                a_rate_sub = sub_completed / sub_n if sub_n else 0
                a_dir = 1 if a_rate_sub > a_rate_ctrl else (-1 if a_rate_sub < a_rate_ctrl else 0)
                pi_a = calc_posteriors_3way(bf_a, a_dir)
            else:
                pi_a = dict(POSTERIOR_NULL)

            # Handle NOT PROVIDED p-values: if human posterior is fully null,
            # direction from ground truth claim should still be respected.
            p_str = st.get("p_value", "")
            if p_str == "NOT PROVIDED":
                # Non-significant in human data → direction 0
                pi_h = calc_posteriors_3way(1.0, 0, prior_odds=10.0)

            pas = calc_pas(pi_h, pi_a)
            test_results.append({
                "finding_id": "F3",
                "test_id": test_id,
                "test_name": st["test_name"],
                "pi_human": pi_h,
                "pi_agent": pi_a,
                "pas": float(pas),
                "test_weight": 1.0 / max(len(f3_finding["statistical_tests"]), 1),
                "agent_stats": {
                    "control_n": n_ctrl,
                    "control_completed": ctrl_completed,
                    "sub_condition_n": sub_n,
                    "sub_condition_completed": sub_completed,
                },
            })
            f3_test_scores.append(pas)

    f3_score = float(np.mean(f3_test_scores)) if f3_test_scores else 0.5
    finding_results.append({
        "finding_id": "F3",
        "finding_weight": finding_weights.get("F3", 0.35),
        "finding_score": f3_score,
    })

    # ---- F1_prelim and F2_prelim ----
    # Preliminary findings about question-type preference and payment-delay aversion.
    # Not directly testable from the main experiment agent data (no preliminary survey).
    # Score as neutral (0.5) to avoid penalizing or rewarding by default.
    for fid in ["F1_prelim", "F2_prelim"]:
        finding_results.append({
            "finding_id": fid,
            "finding_weight": finding_weights.get(fid, 0.05),
            "finding_score": 0.5,
            "note": "Preliminary finding not reconstructible from main experiment agent data.",
        })

    # ---- F4: Perceived influence of decoy (secondary) ----
    # This finding reports that 57.9% of decoy-condition target completers agreed
    # the decoy influenced them. We cannot directly measure self-report from agent
    # choice data. Score as neutral.
    finding_results.append({
        "finding_id": "F4",
        "finding_weight": finding_weights.get("F4", 0.10),
        "finding_score": 0.5,
        "note": "Self-reported decoy influence not measurable from agent choice data.",
    })

    # ---- F5: Non-response bias checks (secondary) ----
    # Demographic balance among completers vs non-completers and FCQ scores.
    # Agent data lacks demographics and FCQ responses at the evaluation level.
    # Score as neutral.
    finding_results.append({
        "finding_id": "F5",
        "finding_weight": finding_weights.get("F5", 0.14),
        "finding_score": 0.5,
        "note": "Non-response bias checks not reconstructible from agent choice data.",
    })

    # ------------------------------------------------------------------
    # 3. Two-level weighted aggregation
    # ------------------------------------------------------------------
    total_w = 0.0
    total_ws = 0.0
    for fr in finding_results:
        w = fr["finding_weight"]
        total_w += w
        total_ws += w * fr["finding_score"]

    overall_score = total_ws / total_w if total_w > 0 else 0.5

    # ------------------------------------------------------------------
    # 4. Build return dict
    # ------------------------------------------------------------------
    return {
        "score": float(overall_score),
        "pi_human": None,
        "pi_agent": None,
        "finding_results": finding_results,
        "test_results": test_results,
        "details": {
            "agent_condition_counts": {
                "control": n_ctrl,
                "decoy_target_first": n_dtf,
                "decoy_decoy_first": n_ddf,
            },
            "agent_target_completions": {
                "control": ctrl_completed,
                "decoy_target_first": dtf_completed,
                "decoy_decoy_first": ddf_completed,
            },
            "agent_completion_rates": {
                "control": ctrl_completed / n_ctrl if n_ctrl else 0.0,
                "decoy_target_first": dtf_completed / n_dtf if n_dtf else 0.0,
                "decoy_decoy_first": ddf_completed / n_ddf if n_ddf else 0.0,
            },
        },
    }
