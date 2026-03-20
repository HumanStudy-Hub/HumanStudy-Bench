"""
Statistical Library for Probability Alignment Score (PAS).

This library provides functions to calculate Bayes Factors (BF10) for common
statistical tests and convert them to posterior probabilities.

Core Philosophy:
All statistical evidence is converted to a Likelihood Ratio (Bayes Factor):
    BF10 = P(Data | H1) / P(Data | H0)

This unified metric allows us to compare evidence strength across diverse
study types (t-tests, correlations, chi-square, etc.).
"""

import math
import re
import warnings
import numpy as np
from scipy import stats, special, integrate
from scipy.integrate import IntegrationWarning
from typing import Optional, Tuple, Dict, List, Any

# Suppress scipy integration tolerance warnings globally for this module
# These warnings occur when numerical integration can't achieve exact tolerance
# but the results are still accurate enough for our purposes
warnings.filterwarnings('ignore', category=IntegrationWarning)
warnings.filterwarnings('ignore', message='.*The algorithm.*')
warnings.filterwarnings('ignore', message='.*Integration.*')
warnings.filterwarnings('ignore', message='.*tolerance.*')

def calc_posteriors_3way(bf10: float, direction: int, prior_odds: float = 1.0) -> dict:
    """
    Calculate 3-way posterior probabilities (H+, H-, H0).
    
    Args:
        bf10: Bayes Factor (H1 vs H0)
        direction: Direction of effect (1 for H+, -1 for H-, 0 for unknown)
        prior_odds: Prior odds P(H1)/P(H0)
        
    Returns:
        dict: {'pi_plus': float, 'pi_minus': float, 'pi_zero': float}
    """
    # 1. Calculate P(H0 | Data) and P(H1 | Data)
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
    
    # 2. Split P(H1 | Data) into P(H+ | Data) and P(H- | Data)
    # If direction is clear, we assign almost all H1 probability to that direction.
    # If direction is unknown (0), we split it 50/50.
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
        'pi_plus': float(pi_plus),
        'pi_minus': float(pi_minus),
        'pi_zero': float(pi_zero)
    }

def prob_from_bf(bf, prior_odds=1.0, direction_match=True):
    """
    Convert Bayes Factor (BF10) to Posterior Probability of H1.
    
    DEPRECATED: Use calc_posteriors_3way instead for better direction handling.
    
    Args:
        bf: Bayes Factor (BF10)
        prior_odds: Prior odds P(H1)/P(H0), default 1.0 (neutral)
        direction_match: Whether the agent effect direction matches the human effect direction.
                         If False, the posterior probability is set to 0.0.
        
    Returns:
        float: Posterior probability P(H1 | Data)
    """
    if not direction_match:
        return 0.0
        
    if bf is None or math.isnan(bf):
        return 0.5
    # Handle infinity
    if math.isinf(bf):
        return 1.0
        
    odds = bf * prior_odds
    return odds / (1.0 + odds)


def prob_from_bf_human(bf, prior_odds=None, direction_match=True, is_significant=True):
    """
    Convert Bayes Factor (BF10) to Posterior Probability of H1.
    
    DEPRECATED: Use calc_posteriors_3way instead for better direction handling.
    """
    if not direction_match:
        return 0.0
        
    if bf is None or math.isnan(bf):
        return 0.5
    # Handle infinity
    if math.isinf(bf):
        return 1.0

    # Determine prior odds if not explicitly provided
    if prior_odds is None:
        prior_odds = 10.0 if is_significant else 0.1

    odds = bf * prior_odds
    return odds / (1.0 + odds)

def recalculate_pi_human(pi_h_old, is_significant):
    """
    Recalculate pi_human with correct prior odds if it was originally
    calculated with the default 10:1 prior but the finding is actually
    not significant (or vice versa).
    """
    if pi_h_old is None or math.isnan(pi_h_old):
        return 0.5
        
    # Check if pi_h_old is already consistent with significance
    # (High prob for significant, low prob for non-significant)
    if is_significant and pi_h_old >= 0.5:
        return pi_h_old
    if not is_significant and pi_h_old <= 0.5:
        return pi_h_old
        
    # Otherwise, the probability is on the "wrong side" of 0.5 relative to significance.
    # This usually means the prior used was incorrect (e.g., 10:1 used for non-sig finding).
    
    # Back-calculate BF assuming it used the WRONG prior
    # If is_sig is True but pi < 0.5, it likely used 0.1 prior
    if is_significant and pi_h_old < 0.5:
        # odds = bf * 0.1 => bf = odds / 0.1 = (pi / (1-pi)) / 0.1
        bf_h = pi_h_old / (0.1 * (1.0 - pi_h_old))
        new_prior = 10.0
    # If is_sig is False but pi > 0.5, it likely used 10.0 prior
    else: # not is_significant and pi_h_old > 0.5
        # odds = bf * 10.0 => bf = odds / 10.0 = (pi / (1-pi)) / 10.0
        bf_h = pi_h_old / (10.0 * (1.0 - pi_h_old))
        new_prior = 0.1
        
    # Recalculate pi with correct prior
    new_odds = bf_h * new_prior
    return new_odds / (1.0 + new_odds)


def parse_p_value_from_reported(reported_statistics: str, significance_level: float = 0.05) -> tuple:
    """
    Parse p-value from reported_statistics string.
    
    Handles formats like:
    - "p < .001" or "p < 0.001"
    - "p < .05" or "p < 0.05"
    - "p = 0.023"
    - "F(1, 312) = 49.1, p < .001"
    
    Args:
        reported_statistics: String containing reported statistics
        significance_level: Default significance level (usually 0.05)
        
    Returns:
        tuple: (p_value, is_significant, confidence)
            - p_value: Parsed p-value (float) or None if not found
            - is_significant: bool indicating if p < significance_level
            - confidence: "high" if exact value, "medium" if inequality, "low" if inferred
    """
    # Ensure significance_level is not None
    if significance_level is None:
        significance_level = 0.05
    
    if not reported_statistics:
        return None, False, "low"
    
    text = reported_statistics.lower()
    
    # Try to extract exact p-value: "p = 0.023" or "p=0.023"
    exact_match = re.search(r'p\s*[=:]\s*([0-9.]+)', text)
    if exact_match:
        p_val = float(exact_match.group(1))
        is_sig = p_val <= significance_level if significance_level is not None else False
        return p_val, is_sig, "high"
    
    # Try to extract inequality: "p < .001" or "p < 0.05"
    inequality_match = re.search(r'p\s*[<>]\s*([0-9.]+)', text)
    if inequality_match:
        threshold = float(inequality_match.group(1))
        is_less_than = '<' in text[inequality_match.start():inequality_match.end()]
        
        if is_less_than:
            # p < threshold: use threshold/2 as conservative estimate
            p_val = threshold / 2.0
            is_sig = True
            return p_val, is_sig, "medium"
        else:
            # p > threshold: use threshold as conservative estimate
            p_val = threshold
            is_sig = threshold < significance_level if significance_level is not None else False
            return p_val, is_sig, "medium"
    
    # If no p-value found, check if test statistic suggests significance
    # For t-tests: large |t| usually means significant
    t_match = re.search(r't\s*\([^)]+\)\s*=\s*([0-9.]+)', text)
    if t_match:
        t_val = abs(float(t_match.group(1)))
        # Rough heuristic: |t| > 2 usually means p < 0.05 for reasonable sample sizes
        if t_val > 2.0:
            # Conservative estimate: p < 0.05
            return 0.025, True, "low"
        else:
            # Conservative estimate: p > 0.05
            return 0.10, False, "low"
    
    # For F-tests: large F usually means significant
    f_match = re.search(r'f\s*\([^)]+\)\s*=\s*([0-9.]+)', text)
    if f_match:
        f_val = float(f_match.group(1))
        # Rough heuristic: F > 4 usually means p < 0.05
        if f_val > 4.0:
            return 0.025, True, "low"
        else:
            return 0.10, False, "low"
    
    # For correlations: large |r| usually means significant
    r_match = re.search(r'r\s*=\s*([-0-9.]+)', text)
    if r_match:
        r_val = abs(float(r_match.group(1)))
        # Rough heuristic: |r| > 0.3 usually means p < 0.05 for n > 30
        if r_val > 0.3:
            return 0.025, True, "low"
        else:
            return 0.10, False, "low"
    
    # Default: assume not significant if we can't determine
    return None, False, "low"


def calculate_human_p_value(
    test_type: str,
    test_statistic: Optional[float] = None,
    n_human: Optional[int] = None,
    n2_human: Optional[int] = None,
    k_human: Optional[int] = None,
    p0: Optional[float] = None,
    alternative: str = "two-sided"
) -> Optional[float]:
    """
    Calculate human p-value from test statistic and sample sizes.
    
    This function computes p-values directly from statistical test results,
    similar to how agent p-values are calculated, rather than parsing from text.
    
    Args:
        test_type: Type of statistical test ("t-test", "mannwhitneyu", "binomial", "f-test", "correlation", etc.)
        test_statistic: Test statistic value (t, U, F, r, etc.)
        n_human: Sample size for first group (or total n for one-sample tests)
        n2_human: Sample size for second group (for two-sample tests)
        k_human: Number of successes (for binomial tests)
        p0: Expected probability under H0 (for binomial tests)
        alternative: Alternative hypothesis ("two-sided", "greater", "less")
        
    Returns:
        float: Calculated p-value, or None if calculation is not possible
    """
    if test_statistic is None and k_human is None:
        return None
    
    try:
        if test_type in ["t-test", "ttest", "ttest_ind", "ttest_rel", "ttest_1samp"]:
            if test_statistic is None or n_human is None:
                return None
            
            # Calculate degrees of freedom
            if n2_human is not None:
                # Two-sample t-test
                df = n_human + n2_human - 2
            else:
                # One-sample t-test
                df = n_human - 1
            
            if df <= 0:
                return None
            
            # Calculate p-value from t-distribution
            if alternative == "two-sided":
                p_val = 2 * (1 - stats.t.cdf(abs(test_statistic), df))
            elif alternative == "greater":
                p_val = 1 - stats.t.cdf(test_statistic, df)
            elif alternative == "less":
                p_val = stats.t.cdf(test_statistic, df)
            else:
                p_val = 2 * (1 - stats.t.cdf(abs(test_statistic), df))
            
            return float(np.clip(p_val, 0.0, 1.0))
        
        elif test_type in ["mannwhitneyu", "mann-whitney", "wilcoxon"]:
            if test_statistic is None:
                return None
            
            # Check if test_statistic is already a z-statistic (common for Mann-Whitney U)
            # Z-statistics are typically > 1 in absolute value, while U-statistics are typically < n1*n2
            # If we have n_human and n2_human, we can check if it's likely a U-statistic
            if n_human is not None and n2_human is not None:
                max_U = n_human * n2_human
                # If test_statistic is much larger than max_U, it's likely a z-statistic
                if abs(test_statistic) > max_U * 0.1:  # Heuristic: z-statistics are usually > 10% of max_U
                    # Treat as z-statistic
                    z = test_statistic
                else:
                    # Treat as U-statistic, convert to z
                    U = test_statistic
                    n1, n2 = n_human, n2_human
                    mean_U = n1 * n2 / 2.0
                    var_U = n1 * n2 * (n1 + n2 + 1) / 12.0
                    
                    if var_U <= 0:
                        return None
                    
                    z = (U - mean_U) / np.sqrt(var_U)
            else:
                # No sample sizes provided, assume it's a z-statistic
                z = test_statistic
            
            if alternative == "two-sided":
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            elif alternative == "greater":
                p_val = 1 - stats.norm.cdf(z)
            elif alternative == "less":
                p_val = stats.norm.cdf(z)
            else:
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
            
            return float(np.clip(p_val, 0.0, 1.0))
        
        elif test_type in ["binomial", "binomtest"]:
            if k_human is None or n_human is None or p0 is None:
                return None
            
            # Use scipy.stats.binomtest
            if alternative == "two-sided":
                alt = "two-sided"
            elif alternative == "greater":
                alt = "greater"
            elif alternative == "less":
                alt = "less"
            else:
                alt = "two-sided"
            
            binom_result = stats.binomtest(k_human, n_human, p0, alternative=alt)
            return float(binom_result.pvalue)
        
        elif test_type in ["f-test", "f_test", "anova"]:
            if test_statistic is None or n_human is None:
                return None
            
            # For F-test, need df1 and df2
            # If n2_human is provided, assume it's df_error
            if n2_human is not None:
                df1 = n_human  # df_effect
                df2 = n2_human  # df_error
            else:
                # Default: assume balanced design
                df1 = 1
                df2 = n_human - 2
            
            if df1 <= 0 or df2 <= 0:
                return None
            
            p_val = 1 - stats.f.cdf(test_statistic, df1, df2)
            return float(np.clip(p_val, 0.0, 1.0))
        
        elif test_type in ["correlation", "pearson", "spearman"]:
            if test_statistic is None or n_human is None:
                return None
            
            r = test_statistic
            # Convert correlation to t-statistic: t = r * sqrt((n-2)/(1-r^2))
            if abs(r) >= 1.0:
                return 0.0 if abs(r) == 1.0 else None
            
            t_stat = r * np.sqrt((n_human - 2) / (1 - r**2))
            df = n_human - 2
            
            if df <= 0:
                return None
            
            if alternative == "two-sided":
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            elif alternative == "greater":
                p_val = 1 - stats.t.cdf(t_stat, df)
            elif alternative == "less":
                p_val = stats.t.cdf(t_stat, df)
            else:
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            return float(np.clip(p_val, 0.0, 1.0))
        
        else:
            # Unknown test type
            return None
    
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def add_statistical_replication_fields(
    test_result: dict,
    test_gt: dict,
    p_val_agent: float = None,
    test_stat_agent: float = None,
    test_type: str = "t-test",
    n_agent: Optional[int] = None,
    n2_agent: Optional[int] = None,
    n_human: Optional[int] = None,
    n2_human: Optional[int] = None,
    contingency_agent: Optional[list] = None,
    contingency_human: Optional[list] = None,
    independent: bool = True,
    agent_direction_override: Optional[int] = None
) -> dict:
    """
    Add statistical replication fields to a test result dictionary.
    
    This function extracts human p-value and direction from ground truth,
    determines agent significance and direction, and adds all necessary
    fields for statistical replication analysis, including frequentist
    consistency metrics (Z-difference and Consistency Score).
    
    If the agent direction does not match the human direction, Pi_Agent is 
    set to 0.0 and PAS is recalculated.
    
    CRITICAL: This function now automatically extracts sample sizes from multiple sources
    if not provided, ensuring n_human and n_agent are never None when possible.
    """
    sig_level = test_gt.get("significance_level", 0.05)
    if sig_level is None:
        sig_level = 0.05
    
    reported_stats = test_gt.get("reported_statistics", "")
    expected_dir_str = test_gt.get("expected_direction", "positive")
    
    # ============================================================================
    # AUTOMATIC SAMPLE SIZE EXTRACTION (CRITICAL: Ensure n_human/n_agent are never None)
    # ============================================================================
    
    # Extract human sample sizes from multiple sources if missing
    if n_human is None:
        # Method 1: Try to extract from reported_statistics (e.g., "t(79) = 2.66" or "F(1, 78) = 17.7")
        if reported_stats:
            # Pattern: t(df) or F(df1, df2) or r(df) or chi2(df)
            # For t-test: df = n - 1 (one-sample) or df = n1 + n2 - 2 (two-sample)
            # For F-test: df2 = total_n - df1 - 1
            t_match = re.search(r't\((\d+)\)', reported_stats)
            f_match = re.search(r'F\((\d+),\s*(\d+)\)', reported_stats)
            r_match = re.search(r'r\((\d+)\)', reported_stats)
            chi2_match = re.search(r'chi2\((\d+)\)', reported_stats)
            
            if f_match:
                # F-test: df2 gives us total_n - df1 - 1, so total_n ≈ df2 + df1 + 1
                df1, df2 = int(f_match.group(1)), int(f_match.group(2))
                if test_type in ["f-test", "f_test", "anova"]:
                    # For balanced design: n1 ≈ n2 ≈ (df2 + df1 + 1) / 2
                    total_n = df2 + df1 + 1
                    n_human = total_n // 2
                    n2_human = total_n - n_human
            elif t_match:
                df = int(t_match.group(1))
                if independent:
                    # Two-sample: df = n1 + n2 - 2, assume balanced
                    total_n = df + 2
                    n_human = total_n // 2
                    n2_human = total_n - n_human
                else:
                    # Paired/one-sample: df = n - 1
                    n_human = df + 1
            elif r_match:
                df = int(r_match.group(1))
                # Correlation: df = n - 2
                n_human = df + 2
            elif chi2_match:
                df = int(chi2_match.group(1))
                # Chi-square: rough estimate, df often = (rows-1)*(cols-1)
                # For 2x2: df=1, n ≈ 100-200 typically
                # Use conservative estimate
                n_human = max(50, df * 50)  # Rough estimate
        
        # Method 2: Try to extract from test_result's pi_human_source (e.g., "18 vs 3, n=79")
        if n_human is None:
            pi_human_source = test_result.get("pi_human_source", "")
            if pi_human_source:
                # Pattern: "n=79" or "n1=40, n2=40" or "18 vs 3, n=79"
                n_match = re.search(r'n\s*=\s*(\d+)', pi_human_source)
                n1_match = re.search(r'n1\s*=\s*(\d+)', pi_human_source)
                n2_match = re.search(r'n2\s*=\s*(\d+)', pi_human_source)
                
                if n1_match and n2_match:
                    n_human = int(n1_match.group(1))
                    n2_human = int(n2_match.group(1))
                elif n_match:
                    total_n = int(n_match.group(1))
                    if independent:
                        n_human = total_n // 2
                        n2_human = total_n - n_human
                    else:
                        n_human = total_n
        
        # Method 3: Try to extract from test_result's agent_reason (for agent sample size)
        if n_agent is None:
            agent_reason = test_result.get("agent_reason", "")
            if agent_reason:
                # Pattern: "n=64 pairs" or "n1=50, n2=67" or "t(62), n=64"
                n_match = re.search(r'n\s*=\s*(\d+)', agent_reason)
                n1_match = re.search(r'n1\s*=\s*(\d+)', agent_reason)
                n2_match = re.search(r'n2\s*=\s*(\d+)', agent_reason)
                nA_match = re.search(r'nA\s*=\s*(\d+)', agent_reason)
                nB_match = re.search(r'nB\s*=\s*(\d+)', agent_reason)
                
                if n1_match and n2_match:
                    n_agent = int(n1_match.group(1))
                    n2_agent = int(n2_match.group(1))
                elif nA_match and nB_match:
                    n_agent = int(nA_match.group(1))
                    n2_agent = int(nB_match.group(1))
                elif n_match:
                    total_n = int(n_match.group(1))
                    if independent:
                        n_agent = total_n // 2
                        n2_agent = total_n - n_agent
                    else:
                        n_agent = total_n
        
        # Method 4: For binomial tests, extract from test_result fields
        if test_type in ["binomial", "binomtest", "sign_test"]:
            if n_human is None:
                # Try to extract from human_k and human_p0 if available
                human_k = test_result.get("human_k")
                if human_k is not None:
                    # Try to infer n from context or use a default
                    # Often binomial tests have n in the reported stats
                    n_match = re.search(r'n\s*=\s*(\d+)', reported_stats)
                    if n_match:
                        n_human = int(n_match.group(1))
            
            if n_agent is None:
                # Extract from agent_reason (e.g., "k=13, n=50")
                agent_reason = test_result.get("agent_reason", "")
                if agent_reason:
                    n_match = re.search(r'n\s*=\s*(\d+)', agent_reason)
                    if n_match:
                        n_agent = int(n_match.group(1))
    
    # Store extracted sample sizes in test_result for future reference
    if n_human is not None:
        test_result["n_human_extracted"] = n_human
    if n2_human is not None:
        test_result["n2_human_extracted"] = n2_human
    if n_agent is not None:
        test_result["n_agent_extracted"] = n_agent
    if n2_agent is not None:
        test_result["n2_agent_extracted"] = n2_agent
    
    # Try to calculate human p-value from test statistic first (preferred method)
    p_val_human = None
    is_sig_human = False
    confidence_human = "low"
    
    # Determine alternative hypothesis from expected_direction
    alternative = "two-sided"
    if expected_dir_str in ["positive", "greater", ">"]:
        alternative = "greater"
    elif expected_dir_str in ["negative", "less", "<"]:
        alternative = "less"
    
    # For binomial tests, check if we have human_k and human_p0
    if test_type in ["binomial", "binomtest"]:
        human_k = test_result.get("human_k")
        human_p0 = test_result.get("human_p0")
        if human_k is not None and human_p0 is not None and n_human is not None:
            try:
                calculated_p = calculate_human_p_value(
                    test_type=test_type,
                    k_human=human_k,
                    n_human=n_human,
                    p0=human_p0,
                    alternative=alternative
                )
                if calculated_p is not None:
                    p_val_human = calculated_p
                    is_sig_human = p_val_human < sig_level
                    confidence_human = "high"  # Calculated from data is high confidence
            except (ValueError, TypeError):
                pass
    
    # For other test types, try to extract from human_test_statistic
    if p_val_human is None:
        human_test_stat_str = test_result.get("human_test_statistic", "")
        if human_test_stat_str:
            try:
                # Extract numeric value from test statistic string
                clean_stat = re.sub(r'[^0-9.-]', '', str(human_test_stat_str))
                if clean_stat:
                    test_stat_val = float(clean_stat)
                    
                    # Try to calculate p-value from test statistic
                    calculated_p = calculate_human_p_value(
                        test_type=test_type,
                        test_statistic=test_stat_val,
                        n_human=n_human,
                        n2_human=n2_human,
                        alternative=alternative
                    )
                    
                    if calculated_p is not None:
                        p_val_human = calculated_p
                        is_sig_human = p_val_human < sig_level
                        confidence_human = "high"  # Calculated from statistic is high confidence
            except (ValueError, TypeError):
                pass
    
    # If calculation failed, fall back to parsing from reported statistics
    if p_val_human is None:
        parsed_p, parsed_sig, parsed_conf = parse_p_value_from_reported(reported_stats, sig_level)
        if parsed_p is not None:
            p_val_human = parsed_p
            is_sig_human = parsed_sig
            confidence_human = parsed_conf
        elif test_result.get("is_significant_human"):
            # NEW: If still None but marked significant in GT, use a small p-value
            p_val_human = 0.01
            is_sig_human = True
            confidence_human = "low"

    # NEW: Store r-equivalent effect sizes for plotting
    test_result["human_effect_r"] = FrequentistConsistency.effect_to_r_equiv(test_type, test_result.get("human_effect_size") or test_result.get("human_effect_d") or 0.0)
    test_result["agent_effect_r"] = FrequentistConsistency.effect_to_r_equiv(test_type, test_result.get("agent_effect_size") or test_result.get("agent_effect_d") or 0.0)

    
    # NEW: Automatically correct pi_human if it was calculated with the wrong prior odds
    # But ONLY if we have high or medium confidence in the p-value parsing.
    # For low confidence (e.g. non-p-value strings), we trust the evaluator's pi_h.
    pi_h_old = test_result.get("pi_human") or test_result.get("pi_h")
    if pi_h_old is not None and confidence_human != "low":
        pi_h_new = recalculate_pi_human(pi_h_old, is_sig_human)
        if abs(pi_h_new - pi_h_old) > 1e-6:
            if "pi_human" in test_result:
                test_result["pi_human"] = float(pi_h_new)
            if "pi_h" in test_result:
                test_result["pi_h"] = float(pi_h_new)
            
            # Recalculate PAS with corrected pi_human
            pi_a = test_result.get("pi_agent") or test_result.get("pi_a") or 0.5
            new_pas = float(calc_pas(pi_h_new, pi_a))
            if "pas" in test_result:
                test_result["pas"] = new_pas
            if "score" in test_result:
                test_result["score"] = new_pas

    # Human direction
    human_test_stat_str = test_result.get("human_test_statistic", "")
    val_human = None
    if human_test_stat_str:
        try:
            # Clean string: remove non-numeric
            clean_h = re.sub(r'[^0-9.-]', '', human_test_stat_str)
            val_h_raw = float(clean_h) if clean_h else None
            
            # Convert F to t-equivalent for df1=1
            if test_type in ["f-test", "f_test", "anova"] and val_h_raw is not None:
                val_human = math.sqrt(abs(val_h_raw))
            else:
                val_human = val_h_raw
        except (ValueError, TypeError):
            pass
            
    human_direction = get_direction_from_statistic(val_human, test_type, expected_dir_str, contingency_table=contingency_human)
    
    # Agent significance and direction
    is_sig_agent = p_val_agent < sig_level if p_val_agent is not None else False
    
    if agent_direction_override is not None:
        agent_direction = agent_direction_override
    else:
        agent_direction = get_direction_from_statistic(test_stat_agent, test_type, expected_dir_str, contingency_table=contingency_agent)
    
    # Direction match
    direction_match = (human_direction == agent_direction) if agent_direction != 0 else False
    
    # NOTE: Old logic of setting pi_agent to 0.0 on direction mismatch is REMOVED.
    # The 3-way posterior model (H+, H-, H0) handles directionality naturally
    # in the PAS calculation: pas = sum(pi_h_i * pi_a_i).
    
    # Add fields to test_result
    test_result["p_value_human"] = float(p_val_human) if p_val_human is not None else None
    test_result["p_value_agent"] = float(p_val_agent) if p_val_agent is not None else None
    test_result["is_significant_human"] = bool(is_sig_human)
    test_result["is_significant_agent"] = bool(is_sig_agent)
    test_result["human_direction"] = int(human_direction)
    test_result["agent_direction"] = int(agent_direction)
    test_result["direction_match"] = bool(direction_match)
    test_result["significance_level"] = float(sig_level)
    test_result["statistical_test_type"] = test_type
    
    # Calculate frequentist consistency metrics
    z_diff = None
    replication_consistency = None
    agent_effect_size = None
    human_effect_size = None
    
    try:
        if test_type in ["t-test", "t_test", "f-test", "f_test", "anova"]:
            # Use extracted sample sizes if original ones were None
            n_human_use = n_human if n_human is not None else test_result.get("n_human_extracted")
            n2_human_use = n2_human if n2_human is not None else test_result.get("n2_human_extracted")
            n_agent_use = n_agent if n_agent is not None else test_result.get("n_agent_extracted")
            n2_agent_use = n2_agent if n2_agent is not None else test_result.get("n2_agent_extracted")
            
            # Human Effect Size
            if val_human is not None and n_human_use is not None:
                human_effect_size = FrequentistConsistency.t_to_cohens_d(val_human, n_human_use, n2_human_use, independent=independent)
            
            # Agent Effect Size
            if test_stat_agent is not None and not math.isnan(float(test_stat_agent)) and n_agent_use is not None:
                val_agent = float(test_stat_agent)
                if test_type in ["f-test", "f_test", "anova"]:
                    val_agent = math.sqrt(abs(val_agent))
                agent_effect_size = FrequentistConsistency.t_to_cohens_d(val_agent, n_agent_use, n2_agent_use, independent=independent)
            
            # Ensure consistency: if agent has effect size, try to compute human effect size
            if agent_effect_size is not None and human_effect_size is None:
                # Try to compute human effect size from available data
                if val_human is not None and n_human_use is not None:
                    human_effect_size = FrequentistConsistency.t_to_cohens_d(val_human, n_human_use, n2_human_use, independent=independent)
            
            # Ensure consistency: if human has effect size, try to compute agent effect size
            if human_effect_size is not None and agent_effect_size is None:
                # Try to compute agent effect size from available data
                if test_stat_agent is not None and not math.isnan(float(test_stat_agent)) and n_agent_use is not None:
                    val_agent = float(test_stat_agent)
                    if test_type in ["f-test", "f_test", "anova"]:
                        val_agent = math.sqrt(abs(val_agent))
                    agent_effect_size = FrequentistConsistency.t_to_cohens_d(val_agent, n_agent_use, n2_agent_use, independent=independent)
                
                # Consistency
                if agent_effect_size is not None and n_agent_use is not None and n_human_use is not None:
                    se_a = FrequentistConsistency.cohens_d_se(agent_effect_size, n_agent_use, n2_agent_use if independent else 0)
                    se_h = FrequentistConsistency.cohens_d_se(human_effect_size, n_human_use, n2_human_use if independent else 0)
                    z_diff, replication_consistency = FrequentistConsistency.calculate_z_diff(agent_effect_size, se_a, human_effect_size, se_h)
            elif human_effect_size is not None and agent_effect_size is not None:
                # Both exist, compute consistency
                if n_agent_use is not None and n_human_use is not None:
                    se_a = FrequentistConsistency.cohens_d_se(agent_effect_size, n_agent_use, n2_agent_use if independent else 0)
                    se_h = FrequentistConsistency.cohens_d_se(human_effect_size, n_human_use, n2_human_use if independent else 0)
                    z_diff, replication_consistency = FrequentistConsistency.calculate_z_diff(agent_effect_size, se_a, human_effect_size, se_h)

        elif test_type in ["correlation", "pearson", "spearman"]:
            # Use extracted sample sizes if original ones were None
            n_human_use = n_human if n_human is not None else test_result.get("n_human_extracted")
            n_agent_use = n_agent if n_agent is not None else test_result.get("n_agent_extracted")
            
            if val_human is not None:
                human_effect_size = FrequentistConsistency.correlation_to_fisher_z(val_human)
            if test_stat_agent is not None and not math.isnan(float(test_stat_agent)):
                agent_effect_size = FrequentistConsistency.correlation_to_fisher_z(float(test_stat_agent))
            
            # Ensure consistency: if agent has effect size, try to compute human effect size
            if agent_effect_size is not None and human_effect_size is None:
                if val_human is not None:
                    human_effect_size = FrequentistConsistency.correlation_to_fisher_z(val_human)
            
            # Ensure consistency: if human has effect size, try to compute agent effect size
            if human_effect_size is not None and agent_effect_size is None:
                if test_stat_agent is not None and not math.isnan(float(test_stat_agent)):
                    agent_effect_size = FrequentistConsistency.correlation_to_fisher_z(float(test_stat_agent))
            
            # Compute consistency if both exist
            if human_effect_size is not None and agent_effect_size is not None and n_agent_use is not None and n_human_use is not None:
                se_a = FrequentistConsistency.correlation_se(float(test_stat_agent), n_agent_use)
                se_h = FrequentistConsistency.correlation_se(val_human, n_human_use)
                z_diff, replication_consistency = FrequentistConsistency.calculate_z_diff(agent_effect_size, se_a, human_effect_size, se_h)

        elif test_type in ["chi-square", "chi2", "chi_square"]:
            if contingency_human is not None:
                a_h, b_h = contingency_human[0]
                c_h, d_h = contingency_human[1]
                human_effect_size = FrequentistConsistency.log_odds_ratio(a_h, b_h, c_h, d_h)
            if contingency_agent is not None:
                a_a, b_a = contingency_agent[0]
                c_a, d_a = contingency_agent[1]
                agent_effect_size = FrequentistConsistency.log_odds_ratio(a_a, b_a, c_a, d_a)
            
            # Ensure consistency: if agent has effect size, try to compute human effect size
            if agent_effect_size is not None and human_effect_size is None:
                if contingency_human is not None:
                    a_h, b_h = contingency_human[0]
                    c_h, d_h = contingency_human[1]
                    human_effect_size = FrequentistConsistency.log_odds_ratio(a_h, b_h, c_h, d_h)
            
            # Ensure consistency: if human has effect size, try to compute agent effect size
            if human_effect_size is not None and agent_effect_size is None:
                if contingency_agent is not None:
                    a_a, b_a = contingency_agent[0]
                    c_a, d_a = contingency_agent[1]
                    agent_effect_size = FrequentistConsistency.log_odds_ratio(a_a, b_a, c_a, d_a)
            
            # Compute consistency if both exist
            if human_effect_size is not None and agent_effect_size is not None:
                se_a = FrequentistConsistency.log_odds_ratio_se(a_a, b_a, c_a, d_a)
                se_h = FrequentistConsistency.log_odds_ratio_se(a_h, b_h, c_h, d_h)
                z_diff, replication_consistency = FrequentistConsistency.calculate_z_diff(agent_effect_size, se_a, human_effect_size, se_h)

        elif test_type in ["mannwhitneyu", "wilcoxon"]:
            # Mann-Whitney U: Convert U statistic to rank-biserial correlation
            # U statistic is stored in agent_test_statistic
            # n1, n2 can be parsed from agent_reason or provided as parameters
            
            # Use extracted sample sizes if original ones were None
            n_agent_use = n_agent if n_agent is not None else test_result.get("n_agent_extracted")
            n2_agent_use = n2_agent if n2_agent is not None else test_result.get("n2_agent_extracted")
            n_human_use = n_human if n_human is not None else test_result.get("n_human_extracted")
            n2_human_use = n2_human if n2_human is not None else test_result.get("n2_human_extracted")
            
            # Agent effect size
            if test_stat_agent is not None and not math.isnan(float(test_stat_agent)):
                try:
                    u_agent = float(test_stat_agent)
                    # Try to get n1, n2 from parameters first
                    if n_agent_use is not None and n2_agent_use is not None:
                        n1_a, n2_a = n_agent_use, n2_agent_use
                    else:
                        # Parse from agent_reason string like "Mann-Whitney U=496.0, p=0.0000, n1=50, n2=67"
                        agent_reason = test_result.get("agent_reason", "")
                        n1_match = re.search(r'n1=(\d+)', agent_reason)
                        n2_match = re.search(r'n2=(\d+)', agent_reason)
                        if n1_match and n2_match:
                            n1_a, n2_a = int(n1_match.group(1)), int(n2_match.group(1))
                        else:
                            n1_a, n2_a = None, None
                    
                    if n1_a is not None and n2_a is not None and n1_a > 0 and n2_a > 0:
                        # Convert U to AUC: A = U / (n1 * n2)
                        auc = u_agent / (n1_a * n2_a)
                        # Convert AUC to rank-biserial correlation: r_rb = 2A - 1
                        r_rb = 2.0 * auc - 1.0
                        agent_effect_size = r_rb
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            # Human effect size
            # For human side, we might have Z-score or need to parse from human_test_statistic
            human_test_stat_str = test_result.get("human_test_statistic", "")
            if human_test_stat_str:
                try:
                    # Try to parse as Z-score
                    z_human = float(re.sub(r'[^0-9.-]', '', str(human_test_stat_str)))
                    # Convert Z to rank-biserial correlation: r_rb ≈ z / sqrt(N)
                    # But we need N, try to get from n_human or parse
                    if n_human_use is not None and n2_human_use is not None:
                        n_total = n_human_use + n2_human_use
                    elif n_human_use is not None:
                        n_total = n_human_use
                    else:
                        # Try to parse from pi_human_source or other fields
                        n_total = None
                    
                    if n_total is not None and n_total > 0:
                        # Approximate: r_rb ≈ z / sqrt(N)
                        r_rb_human = z_human / math.sqrt(n_total)
                        # Clamp to valid range
                        r_rb_human = max(-1.0, min(1.0, r_rb_human))
                        human_effect_size = r_rb_human
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            # Compute consistency if both exist
            if human_effect_size is not None and agent_effect_size is not None:
                # For rank-biserial correlation, use correlation SE formula
                # SE_r ≈ 1 / sqrt(N - 3) for large samples
                if n_agent_use is not None and n2_agent_use is not None:
                    n_total_a = n_agent_use + n2_agent_use
                    se_a = 1.0 / math.sqrt(max(1, n_total_a - 3)) if n_total_a > 3 else 1.0
                else:
                    se_a = 0.1  # Default fallback
                
                if n_human_use is not None and n2_human_use is not None:
                    n_total_h = n_human_use + n2_human_use
                    se_h = 1.0 / math.sqrt(max(1, n_total_h - 3)) if n_total_h > 3 else 1.0
                elif n_human_use is not None:
                    se_h = 1.0 / math.sqrt(max(1, n_human_use - 3)) if n_human_use > 3 else 1.0
                else:
                    se_h = 0.1  # Default fallback
                
                z_diff, replication_consistency = FrequentistConsistency.calculate_z_diff(agent_effect_size, se_a, human_effect_size, se_h)

        elif test_type in ["binomial", "sign_test", "binomtest"]:
            # Binomial: Convert proportion to effect size
            # For agent: parse k, n from agent_reason or agent_test_statistic
            # agent_test_statistic might be the p-value or proportion
            
            # Use extracted sample sizes if original ones were None
            n_human_use = n_human if n_human is not None else test_result.get("n_human_extracted")
            n_agent_use = n_agent if n_agent is not None else test_result.get("n_agent_extracted")
            
            # Agent effect size
            agent_reason = test_result.get("agent_reason", "")
            # Parse k, n from agent_reason like "k=13, n=50, expected_p=0.2102"
            k_match = re.search(r'k=(\d+)', agent_reason)
            n_match = re.search(r'n=(\d+)', agent_reason)
            expected_p_match = re.search(r'expected_p=([\d.]+)', agent_reason)
            
            if k_match and n_match:
                try:
                    k_a = int(k_match.group(1))
                    n_a = int(n_match.group(1))
                    p0_a = float(expected_p_match.group(1)) if expected_p_match else 0.5
                    
                    if n_a > 0:
                        p_a = k_a / n_a
                        # Store proportion as effect_size (will be converted to d via effect_to_d_equiv)
                        agent_effect_size = p_a
                        # Update n_agent_use if we extracted it from agent_reason
                        if n_agent_use is None:
                            n_agent_use = n_a
                            test_result["n_agent_extracted"] = n_a
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            # Human effect size
            # Parse from pi_human_source like "k=24, n=48 (from paper data)"
            pi_human_source = test_result.get("pi_human_source", "")
            k_h_match = re.search(r'k=(\d+)', pi_human_source)
            n_h_match = re.search(r'n=(\d+)', pi_human_source)
            
            if k_h_match and n_h_match:
                try:
                    k_h = int(k_h_match.group(1))
                    n_h = int(n_h_match.group(1))
                    p0_h = 0.5  # Default null for binomial
                    
                    if n_h > 0:
                        p_h = k_h / n_h
                        human_effect_size = p_h
                        # Update n_human_use if we extracted it from pi_human_source
                        if n_human_use is None:
                            n_human_use = n_h
                            test_result["n_human_extracted"] = n_h
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            # Compute consistency if both exist
            if human_effect_size is not None and agent_effect_size is not None:
                # For proportions, use SE_p = sqrt(p*(1-p)/n)
                # Use extracted sample sizes
                n_a_use = n_a if 'n_a' in locals() and n_a is not None else n_agent_use
                n_h_use = n_h if 'n_h' in locals() and n_h is not None else n_human_use
                
                if n_a_use is not None and n_a_use > 0:
                    se_a = math.sqrt(agent_effect_size * (1 - agent_effect_size) / n_a_use)
                else:
                    se_a = 0.1  # Default fallback
                
                if n_h_use is not None and n_h_use > 0:
                    se_h = math.sqrt(human_effect_size * (1 - human_effect_size) / n_h_use)
                else:
                    se_h = 0.1  # Default fallback
                
                z_diff, replication_consistency = FrequentistConsistency.calculate_z_diff(agent_effect_size, se_a, human_effect_size, se_h)

    except Exception:
        pass
    
    # Save results
    if z_diff is not None and not (math.isnan(z_diff) or math.isinf(z_diff)):
        test_result["z_diff"] = float(z_diff)
    
    if replication_consistency is not None and not (math.isnan(replication_consistency) or math.isinf(replication_consistency)):
        test_result["replication_consistency"] = float(replication_consistency)
    else:
        test_result["replication_consistency"] = 0.0
        
    if agent_effect_size is not None and not (math.isnan(agent_effect_size) or math.isinf(agent_effect_size)):
        test_result["agent_effect_size"] = float(agent_effect_size)
    if human_effect_size is not None and not (math.isnan(human_effect_size) or math.isinf(human_effect_size)):
        test_result["human_effect_size"] = float(human_effect_size)
    
    # Compute and store Cohen's d-equivalent for correlation-based ECS
    agent_effect_d = None
    human_effect_d = None
    if agent_effect_size is not None and not (math.isnan(agent_effect_size) or math.isinf(agent_effect_size)):
        agent_effect_d = FrequentistConsistency.effect_to_d_equiv(test_type, agent_effect_size)
    if human_effect_size is not None and not (math.isnan(human_effect_size) or math.isinf(human_effect_size)):
        human_effect_d = FrequentistConsistency.effect_to_d_equiv(test_type, human_effect_size)
    
    if agent_effect_d is not None and not (math.isnan(agent_effect_d) or math.isinf(agent_effect_d)):
        test_result["agent_effect_d"] = float(agent_effect_d)
    if human_effect_d is not None and not (math.isnan(human_effect_d) or math.isinf(human_effect_d)):
        test_result["human_effect_d"] = float(human_effect_d)
    
    # Compute and store n_eff for Fisher-z pooling weights
    n_eff = None
    if test_type in ["t-test", "t_test", "f-test", "f_test", "anova"]:
        if independent:
            if n_agent is not None and n2_agent is not None:
                n_eff = int(n_agent + n2_agent)
            elif n_agent is not None:
                n_eff = int(n_agent)
        else:
            if n_agent is not None:
                n_eff = int(n_agent)
    elif test_type in ["correlation", "pearson", "spearman"]:
        if n_agent is not None:
            n_eff = int(n_agent)
    elif test_type in ["chi-square", "chi2", "chi_square"]:
        if contingency_agent is not None:
            total = sum(sum(row) if isinstance(row, list) else row for row in contingency_agent)
            if total > 0:
                n_eff = int(total)
    elif test_type in ["mannwhitneyu", "wilcoxon"]:
        if n_agent is not None and n2_agent is not None:
            n_eff = int(n_agent + n2_agent)
    elif test_type in ["binomial", "sign_test"]:
        if n_agent is not None:
            n_eff = int(n_agent)
    
    if n_eff is not None and n_eff > 0:
        test_result["n_eff"] = n_eff
    else:
        test_result["n_eff"] = 1  # Default fallback
    
    return test_result


def get_direction_from_statistic(test_statistic: float, test_type: str, expected_direction: str = None, contingency_table: list = None) -> int:
    """
    Determine direction of effect from test statistic.
    
    Args:
        test_statistic: Test statistic value (t, F, r, etc.) or tuple (will extract first element)
        test_type: Type of test ("t-test", "correlation", "f-test", etc.)
        expected_direction: Expected direction from ground truth ("positive", "negative", or None)
        contingency_table: Optional 2x2 contingency table for chi-square tests
        
    Returns:
        int: 1 for positive direction, -1 for negative, 0 if unclear
    """
    # Handle contingency table for chi-square (more reliable than unsigned statistic)
    if test_type in ["chi-square", "chi2", "chi_square"] and contingency_table is not None:
        try:
            if len(contingency_table) == 2 and len(contingency_table[0]) == 2:
                a, b = contingency_table[0]
                c, d = contingency_table[1]
                p1 = a / (a + b) if (a + b) > 0 else 0
                p2 = c / (c + d) if (c + d) > 0 else 0
                if p1 > p2: return 1
                if p1 < p2: return -1
                return 0
        except (ValueError, TypeError, ZeroDivisionError, IndexError):
            pass

    # Handle tuple input (extract first element)
    if isinstance(test_statistic, tuple):
        if len(test_statistic) > 0:
            test_statistic = test_statistic[0]
        else:
            test_statistic = None
    
    # If statistic is missing, use expected_direction
    if test_statistic is None:
        if expected_direction:
            return 1 if expected_direction.lower() in ["positive", "pos", "+"] else -1
        return 0
    
    # Check if it's a valid number
    try:
        test_statistic = float(test_statistic)
        if math.isnan(test_statistic):
            if expected_direction:
                return 1 if expected_direction.lower() in ["positive", "pos", "+"] else -1
            return 0
    except (ValueError, TypeError):
        if expected_direction:
            return 1 if expected_direction.lower() in ["positive", "pos", "+"] else -1
        return 0
    
    if test_type in ["t-test", "t_test"]:
        # For t-test: sign of t-statistic indicates direction
        return 1 if test_statistic > 0 else -1
    
    elif test_type in ["correlation", "pearson", "spearman"]:
        # For correlation: sign of r indicates direction
        return 1 if test_statistic > 0 else -1
    
    elif test_type in ["f-test", "anova", "f_test"]:
        # For F-test: normally always positive. 
        # But if the evaluator passes a negative value, we treat it as reverse direction.
        if test_statistic < 0:
            return -1
        if expected_direction:
            return 1 if expected_direction.lower() in ["positive", "pos", "+"] else -1
        return 1  # Default to positive for F-tests
    
    elif test_type in ["chi-square", "chi2", "chi_square"]:
        # For chi-square: normally always positive.
        if test_statistic < 0:
            return -1
        if expected_direction:
            return 1 if expected_direction.lower() in ["positive", "pos", "+"] else -1
        return 1  # Default to positive for chi-square
    
    # Default: use expected_direction if provided
    if expected_direction:
        return 1 if expected_direction.lower() in ["positive", "pos", "+"] else -1
    
    return 0

def calc_pas(pi_h_input, pi_a_input):
    """
    Calculate Probability Alignment Score (PAS) per test.
    
    Supports both 2-way (scalar) and 3-way (dict) inputs.
    
    If dict: PAS = π_h+ * π_a+ + π_h- * π_a- + π_h0 * π_a0
    If scalar: PAS = π_h * π_a + (1 - π_h) * (1 - π_a)
    
    This measures the probability that Human and Agent are in the same state.
    """
    if isinstance(pi_h_input, dict) and isinstance(pi_a_input, dict):
        # 3-way calculation: dot product of posterior probability vectors
        return (pi_h_input.get('pi_plus', 0.0) * pi_a_input.get('pi_plus', 0.0) +
                pi_h_input.get('pi_minus', 0.0) * pi_a_input.get('pi_minus', 0.0) +
                pi_h_input.get('pi_zero', 0.0) * pi_a_input.get('pi_zero', 0.0))
    
    # Legacy 2-way scalar calculation
    try:
        pi_h = max(1e-6, min(1.0 - 1e-6, float(pi_h_input)))
        pi_a = max(1e-6, min(1.0 - 1e-6, float(pi_a_input)))
        return pi_h * pi_a + (1 - pi_h) * (1 - pi_a)
    except (TypeError, ValueError):
        # Fallback if inputs are mixed or invalid
        return 0.5

def rescale_score(score):
    """
    Rescale score from [0, 1] to [-1, 1] using 2x - 1.
    """
    return 2.0 * score - 1.0

def normalize_test_score(f_i: float, pi_h_input) -> float:
    """
    Per-test normalization to [0,1].
    
    Supports both 2-way (scalar) and 3-way (dict) pi_h.
    """
    if isinstance(pi_h_input, dict):
        # 3-way normalization
        p0 = pi_h_input.get('pi_zero', 0.5)
        pp = pi_h_input.get('pi_plus', 0.25)
        pm = pi_h_input.get('pi_minus', 0.25)
        
        # hi: perfect replication (agent posteriors match human posteriors exactly)
        hi = p0**2 + pp**2 + pm**2
        
        # lo: random agreement (agent posteriors are neutral: 0.5, 0.25, 0.25)
        lo = 0.5 * p0 + 0.25 * pp + 0.25 * pm
    else:
        # Legacy 2-way normalization
        pi_h = float(pi_h_input)
        lo = min(pi_h, 1.0 - pi_h)
        hi = max(pi_h, 1.0 - pi_h)
        
    denom = hi - lo
    if denom < 1e-8:
        return 0.0
    g_i = (f_i - lo) / denom
    return max(0.0, min(1.0, g_i))

def normalize_study(test_results):
    """
    Study-wide normalization.
    
    F_norm = mean_i(g_i)
    NormScore = 2*F_norm - 1
    
    where g_i is per-test normalized score from normalize_test_score.
    """
    if not test_results:
        return 0.5, 0.0  # F_norm=0.5 -> NormScore=0
    g_vals = []
    for t in test_results:
        # Support both "score" and "pas" field names
        f_i = t.get("score") or t.get("pas", 0.5)
        pi_h = t.get("pi_human", 0.5)
        g_vals.append(normalize_test_score(f_i, pi_h))
        F_norm = float(np.mean(g_vals))
    NormScore = rescale_score(F_norm)
    return F_norm, NormScore

def normalize_pas(pas_score, pi_human):
    """
    Normalize PAS score to [-1, 1] range considering the human ceiling.
    
    Formula: (PAS - 0.5) / (max(pi_h, 1-pi_h) - 0.5)
    
    This allows the score to reach 1.0 if the agent matches the human evidence ceiling,
    or -1.0 if it perfectly contradicts it.
    """
    ceiling = max(pi_human, 1.0 - pi_human)
    denom = ceiling - 0.5
    if denom < 1e-6:
        return rescale_score(pas_score)
    
    norm = (pas_score - 0.5) / denom
    return max(-1.0, min(1.0, norm))

def calc_field_comparable_agent_score(pas: float, pi_human, pi_agent=None):
    """
    Calculate a field-comparable agent performance score that is not confounded by pi_human.
    
    This metric allows fair comparison of agent performance across fields/studies regardless
    of the strength of human evidence (pi_human). It addresses the issue where:
    - Field 1 with pi_h=0.6 and pi_a=0.8 gives PAS=0.56
    - Field 2 with pi_h=0.9 and pi_a=0.8 gives PAS=0.74
    Even though agent performance (pi_a=0.8) is the same in both fields.
    
    Two approaches are provided:
    
    1. Normalized PAS (default): Uses normalize_test_score to normalize PAS by human ceiling
       - Range: [0, 1] where 1 = perfect alignment, 0 = worst possible alignment
       - This accounts for the maximum possible PAS given pi_human
    
    2. Direct pi_agent (if provided): Returns pi_agent directly when direction matches
       - Range: [0, 1] where 1 = perfect agent evidence, 0 = no agent evidence
       - Simpler but doesn't account for direction matching
    
    Args:
        pas: Probability Alignment Score (PAS) between human and agent
        pi_human: Human posterior probability (scalar or dict)
        pi_agent: Optional agent posterior probability (scalar or dict). If provided,
                  returns pi_agent directly for simpler comparison.
    
    Returns:
        float: Field-comparable agent performance score in [0, 1]
               Higher values indicate better agent performance relative to human evidence.
    
    Example:
        # Field 1: weak human evidence (pi_h=0.6), good agent (pi_a=0.8)
        pas1 = calc_pas(0.6, 0.8)  # = 0.56
        score1 = calc_field_comparable_agent_score(pas1, 0.6, 0.8)  # ≈ 0.8 (normalized)
        
        # Field 2: strong human evidence (pi_h=0.9), good agent (pi_a=0.8)  
        pas2 = calc_pas(0.9, 0.8)  # = 0.74
        score2 = calc_field_comparable_agent_score(pas2, 0.9, 0.8)  # ≈ 0.8 (normalized)
        
        # Now score1 ≈ score2, allowing fair comparison across fields
    """
    # Option 1: Use direct pi_agent if provided (simpler, but requires direction matching check)
    if pi_agent is not None:
        # For scalar pi_agent, return it directly
        if isinstance(pi_agent, (int, float)):
            return float(pi_agent)
        # For dict (3-way), return max(pi_plus, pi_minus) as the agent evidence strength
        elif isinstance(pi_agent, dict):
            return max(pi_agent.get('pi_plus', 0.0), pi_agent.get('pi_minus', 0.0))
    
    # Option 2: Normalize PAS by human ceiling (accounts for direction matching)
    # This uses the same normalization as normalize_test_score
    normalized_pas = normalize_test_score(pas, pi_human)
    return normalized_pas

def aggregate_field_comparable_scores(test_results: list, use_pi_agent_direct: bool = False) -> float:
    """
    Aggregate field-comparable agent scores across tests for fair field-wise comparison.
    
    This function aggregates agent performance scores that are normalized to account
    for differences in human evidence strength (pi_human) across fields/studies.
    
    Args:
        test_results: List of test result dicts, each containing:
            - "pas": PAS score
            - "pi_human": Human posterior probability
            - "pi_agent": (optional) Agent posterior probability
        use_pi_agent_direct: If True, use pi_agent directly. If False, normalize PAS.
    
    Returns:
        float: Aggregated field-comparable score in [0, 1]
    
    Example:
        # Compare agent performance across Field 1 vs Field 2
        field1_tests = [{"pas": 0.56, "pi_human": 0.6, "pi_agent": 0.8}, ...]
        field2_tests = [{"pas": 0.74, "pi_human": 0.9, "pi_agent": 0.8}, ...]
        
        score1 = aggregate_field_comparable_scores(field1_tests)
        score2 = aggregate_field_comparable_scores(field2_tests)
        
        # Now score1 and score2 are directly comparable
    """
    if not test_results:
        return 0.5
    
    scores = []
    for test in test_results:
        pas_val = test.get("pas") or test.get("score", 0.5)
        pi_h = test.get("pi_human", 0.5)
        pi_a = test.get("pi_agent") if use_pi_agent_direct else None
        
        score = calc_field_comparable_agent_score(pas_val, pi_h, pi_a)
        scores.append(score)
    
    return float(np.mean(scores))

# -----------------------------------------------------------------------------
# Bayes Factor Calculators
# -----------------------------------------------------------------------------

# Legacy BIC approximation versions (deprecated, use exact versions below)
def calc_bf_t_bic(t_stat, n1, n2=None, independent=True):
    """
    [LEGACY] Calculate Bayes Factor (BF10) for t-test using BIC approximation.
    
    DEPRECATED: This function uses BIC approximation. Use calc_bf_t() instead for exact JZS calculation.
    
    Approximation based on Rouder et al. (2009).
    
    Args:
        t_stat: t-statistic
        n1: Sample size of group 1
        n2: Sample size of group 2 (if independent)
        independent: True for independent samples, False for paired/one-sample
        
    Returns:
        float: BF10
    """
    try:
        if independent:
            if n2 is None:
                raise ValueError("n2 must be provided for independent t-test")
            # Effective sample size for independent t-test
            n_eff = (n1 * n2) / (n1 + n2)
            nu = n1 + n2 - 2  # Degrees of freedom
        else:
            # Paired or one-sample
            n_eff = n1
            nu = n1 - 1
            
        t2 = t_stat**2
        
        # JZS Bayes Factor approximation
        # Formula from Rouder et al. (2009) "Bayesian t tests for accepting and rejecting the null hypothesis"
        # This is a numerical integration. For simplicity and robustness in this library,
        # we use the BIC approximation for large N, or a simplified analytical bound.
        
        # Using BIC approximation which is robust and standard for this scale
        # BF ≈ exp((BIC_H0 - BIC_H1) / 2)
        # However, precise JZS is better for small N.
        # Let's use a standard approximation for JZS:
        
        r_scale = 0.707  # Standard Cauchy width
        
        def integrand(g):
            return ((1 + n_eff * g * r_scale**2)**(-0.5)) * ((1 + t2 / ((1 + n_eff * g * r_scale**2) * nu))**(-(nu + 1) / 2)) * ((2 * math.pi)**(-0.5)) * (g**(-1.5)) * math.exp(-1 / (2 * g))

        # Since we can't easily do integration without more dependencies, 
        # we use the BIC approximation which is asymptotically equivalent and easier to implement reliably.
        # BF10 ≈ sqrt(n_eff) * exp(0.5 * t^2) is too simple.
        
        # Better: k-factor approximation
        # For our purposes, we will use the rigorous conversion from p-value if t is large,
        # or a BIC approximation: 
        # BIC(H1) = n * ln(1 - R^2) + k*ln(n)
        # Convert t to R^2: t^2 = R^2 * df / (1 - R^2)
        
        # Let's use the exact JZS formula if possible, or a high-quality approximation.
        # Here is the Rouder approximation implementation:
        
        if abs(t_stat) > 10:
             # For very large t, BF is huge. Return a large number.
             return 1e6
             
        # JZS analytic approximation
        term1 = (1 + t2/nu)**(-(nu+1)/2)
        # Integral part is complex.
        
        # FALLBACK: Use BIC approximation which is standard in many ML benchmarks
        # BF10 = exp( (BIC0 - BIC1) / 2 )
        # ∆BIC ≈ chi2 - df * ln(N)
        # Here chi2 ~ t^2 (for 1 df difference)
        # ln(BF10) ≈ (t^2 - ln(n_eff))/2
        log_bf = (t2 - math.log(n_eff)) / 2
        return math.exp(log_bf)
        
    except Exception as e:
        print(f"Error calculating BF for t-test: {e}")
        return 1.0

def calc_bf_t(t_stat, n1, n2=None, independent=True, r_scale=0.707):
    """
    Calculate JZS Bayes Factor (BF10) for t-test using exact numerical integration.
    
    This is the default implementation using exact JZS calculation.
    Ref: Rouder et al. (2009). Bayesian t tests for accepting and rejecting the null hypothesis.
    
    For the legacy BIC approximation version, use calc_bf_t_bic().
    
    Args:
        t_stat: t-statistic
        n1: Sample size of group 1
        n2: Sample size of group 2 (if independent)
        independent: True for independent samples, False for paired/one-sample
        r_scale: Scale parameter for Cauchy prior (default: 0.707, standard JZS)
        
    Returns:
        float: BF10 (exact Bayes Factor)
    """
    try:
        # 1. Determine Effective Sample Size & DoF
        if independent:
            if n2 is None:
                raise ValueError("n2 must be provided for independent t-test")
            # Effective sample size for independent t-test
            n_eff = (n1 * n2) / (n1 + n2)
            nu = n1 + n2 - 2  # Degrees of freedom
        else:
            # Paired or one-sample
            n_eff = n1
            nu = n1 - 1
            
        t2 = t_stat**2
        
        # 2. Optimization: For very large t, return bound to avoid integration overflow
        if t2 > 100:
            return 1e6

        # 3. Define the numerator integrand (Marginal Likelihood of H1)
        # Equation 1 from Rouder et al. (2009)
        def numerator_integrand(g):
            term1 = (1 + n_eff * g * r_scale**2)**(-0.5)
            term2 = (1 + t2 / ((1 + n_eff * g * r_scale**2) * nu))**(-(nu + 1) / 2)
            term3 = (2 * math.pi)**(-0.5) * (g**(-1.5)) * math.exp(-1 / (2 * g))
            return term1 * term2 * term3

        # Integrate g from 0 to infinity
        # Using scipy.integrate.quad is standard and accurate
        # Suppress tolerance warnings and use relaxed tolerance for numerical stability
        with warnings.catch_warnings():
            # Suppress all integration-related warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=IntegrationWarning)
            warnings.filterwarnings('ignore', message='.*tolerance.*')
            warnings.filterwarnings('ignore', message='.*Integration.*')
            warnings.filterwarnings('ignore', message='.*The algorithm.*')
            # Use relaxed tolerance (epsabs=1e-6, epsrel=1e-6) to avoid warnings
            integral_val, _ = integrate.quad(
                numerator_integrand, 
                0, 
                np.inf,
                epsabs=1e-6,
                epsrel=1e-6,
                limit=100
            )
        
        # 4. Denominator (Marginal Likelihood of H0)
        # This is simply the t-distribution density under H0
        denominator = (1 + t2 / nu)**(-(nu + 1) / 2)
        
        bf10 = integral_val / denominator
        
        # Handle edge cases
        if math.isnan(bf10) or math.isinf(bf10):
            # Fallback to BIC if integration fails
            log_bf = (t2 - math.log(n_eff)) / 2
            return math.exp(log_bf)
            
        return bf10
        
    except Exception as e:
        print(f"Error in exact JZS calc: {e}, falling back to BIC")
        # Fallback to BIC approximation if integration fails
        try:
            n_eff = (n1 * n2) / (n1 + n2) if independent and n2 is not None else n1
            return calc_bf_t_bic(t_stat, n1, n2, independent)
        except:
            return 1.0

# Legacy BIC approximation version
def calc_bf_r_bic(r, n):
    """
    [LEGACY] Calculate Bayes Factor (BF10) for Pearson correlation using BIC approximation.
    
    DEPRECATED: This function uses BIC approximation. Use calc_bf_r() instead for exact calculation.
    
    H0: r = 0, H1: r != 0
    
    Using BIC approximation for linear regression model comparison.
    """
    try:
        r2 = r**2
        if r2 >= 1.0:
            return 1e6
            
        # Compare Model with correlation (1 param) vs Null (0 param)
        # Bayes Factor for correlation can be approximated via linear regression
        # BF10 ≈ sqrt(n) * (1 - r^2)^(-(n-3)/2) / 2  (rough Jeffreys approximation)
        
        # Better: Wetzels & Wagenmakers (2012) default prior
        # Or simpler BIC approximation for regression:
        # BF10 ≈ exp((n * R2 - log(n))/2) ? No.
        
        # BIC Approximation:
        # BIC0 - BIC1 ≈ n * ln(1/(1-r^2)) - k * ln(n) where k=1
        log_bf = (n * math.log(1/(1-r2)) - math.log(n)) / 2
        return math.exp(log_bf)
        
    except Exception:
        return 1.0

def calc_bf_r(r, n):
    """
    Calculate Bayes Factor for Pearson Correlation using exact JZS calculation.
    
    Strategy: Convert r to t-statistic and use the exact JZS t-test formula.
    H0: rho = 0
    
    This is the default implementation. For the legacy BIC approximation version, use calc_bf_r_bic().
    
    Ref: Wetzels & Wagenmakers (2012). A default Bayesian hypothesis test for correlations.
    
    Args:
        r: Pearson correlation coefficient
        n: Sample size
        
    Returns:
        float: BF10 (exact Bayes Factor)
    """
    try:
        r = float(r)
        if abs(r) >= 1.0:
            return 1e6
        if n <= 2:
            return 1.0
            
        # Convert r to t-statistic
        # t = r * sqrt(df / (1 - r^2))
        # where df = n - 2 for correlation test
        df = n - 2
        if df <= 0:
            return 1.0
            
        r2 = r**2
        if r2 >= 1.0:
            return 1e6
            
        t_val = r * math.sqrt(df / (1 - r2))
        
        # Use the exact JZS t-test function
        # Treat as one-sample t-test with sample size n
        return calc_bf_t(t_val, n, independent=False)
        
    except Exception as e:
        print(f"Error in exact correlation BF: {e}")
        return 1.0

def calc_bf_chisq(chi2, n, df=1):
    """
    Calculate BF10 for Chi-Square test.
    """
    try:
        # BIC approximation
        # ∆BIC = chi2 - df * ln(n)
        log_bf = (chi2 - df * math.log(n)) / 2
        return math.exp(log_bf)
    except Exception:
        return 1.0

def chi2_contingency_safe(observed):
    """
    Perform chi-square contingency test safely. 
    Returns (chi2, p, dof, expected).
    
    If the test cannot be computed (e.g., zero expected frequencies due to 
    zero row/column sums), it returns (0.0, 1.0, dof, None) which effectively 
    represents no evidence for a difference (H0).
    """
    obs = np.array(observed)
    # Default degrees of freedom for 2x2
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1) if obs.ndim == 2 else 1
    
    if np.sum(obs) == 0:
        return 0.0, 1.0, dof, None
    
    # Check for zero row or column sums
    if np.any(np.sum(obs, axis=0) == 0) or np.any(np.sum(obs, axis=1) == 0):
        return 0.0, 1.0, dof, None
        
    try:
        chi2, p, res_dof, expected = stats.chi2_contingency(obs)
        if expected is not None and np.any(expected == 0):
            return 0.0, 1.0, res_dof, expected
        return chi2, p, res_dof, expected
    except (ValueError, RuntimeWarning):
        return 0.0, 1.0, dof, None

def calc_bf_anova(f_stat, df1, df2, n_total, r_scale=0.5):
    """
    Calculate JZS Bayes Factor (BF10) for ANOVA F-test using exact numerical integration.
    Ref: Rouder et al. (2012). Default Bayes Factors for ANOVA Designs.
    
    Args:
        f_stat: F-statistic
        df1: Numerator degrees of freedom (number of groups - 1)
        df2: Denominator degrees of freedom (N - number of groups)
        n_total: Total sample size
        r_scale: Scale parameter for Cauchy prior (default 0.5 for ANOVA)
        
    Returns:
        float: BF10 (exact Bayes Factor)
    """
    try:
        # Calculate R-squared from F-statistic
        r2 = (df1 * f_stat) / (df1 * f_stat + df2)
        
        # Optimization: for very large F, return bound
        if f_stat > 100:
            return 1e6

        # Rouder et al. (2012) JZS ANOVA integration
        # p is the number of parameters (df1)
        p = df1
        N = n_total
        
        def integrand(g):
            # Prior on g: Inverse-Gamma(1/2, r_scale^2 * N / 2)? 
            # No, standard JZS uses r_scale^2 / 2.
            # Term for marginal likelihood under H1 vs H0
            # BF = (1+Ng)^((N-p-1)/2) * (1+Ng(1-R2))^(-(N-1)/2)
            
            log_term = ((N - p - 1) / 2.0) * math.log(1 + N * g) - ((N - 1) / 2.0) * math.log(1 + N * g * (1 - r2))
            
            # Prior part (Inverse-Gamma density)
            # shape = 0.5, scale = r_scale^2 / 2
            log_prior = 0.5 * math.log(r_scale**2 / 2.0) - special.gammaln(0.5) - 1.5 * math.log(g) - (r_scale**2 / (2.0 * g))
            
            return math.exp(log_term + log_prior)

        # Integrate g from 0 to infinity
        # Suppress tolerance warnings and use relaxed tolerance for numerical stability
        with warnings.catch_warnings():
            # Suppress all integration-related warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=IntegrationWarning)
            warnings.filterwarnings('ignore', message='.*tolerance.*')
            warnings.filterwarnings('ignore', message='.*Integration.*')
            warnings.filterwarnings('ignore', message='.*The algorithm.*')
            # Use relaxed tolerance (epsabs=1e-6, epsrel=1e-6) to avoid warnings
            bf10, _ = integrate.quad(
                integrand, 
                0, 
                np.inf,
                epsabs=1e-6,
                epsrel=1e-6,
                limit=100
            )
        
        if math.isnan(bf10) or math.isinf(bf10):
            raise ValueError("Integration resulted in non-finite value")
            
        return bf10
        
    except Exception as e:
        # Fallback to BIC approximation
        try:
            r2 = (df1 * f_stat) / (df1 * f_stat + df2)
            log_bf = (n_total * math.log(1/(1-r2)) - df1 * math.log(n_total)) / 2
            return math.exp(log_bf)
        except:
            return 1.0

# Legacy BIC approximation version
def calc_bf_variance_f_bic(f_stat, df1, df2):
    """
    [LEGACY] Calculate BF10 for variance F-test using BIC approximation.
    
    DEPRECATED: This function uses BIC approximation. Use calc_bf_variance_f() instead for robust VSB calculation.
    
    Calculate BF10 for variance F-test (testing equality of variances).
    
    H0: σ1² = σ2² (variances equal)
    H1: σ1² ≠ σ2² (variances differ)
    
    The F-statistic is F = s1²/s2², which follows F(df1, df2) under H0.
    
    Args:
        f_stat: F-statistic (ratio of variances)
        df1: Degrees of freedom for numerator (n1 - 1)
        df2: Degrees of freedom for denominator (n2 - 1)
        
    Returns:
        float: BF10 (Bayes Factor favoring H1 over H0)
    """
    try:
        # For variance F-tests, we use the F-distribution directly
        # Under H0: F ~ F(df1, df2)
        # Under H1: F follows a scaled F-distribution
        
        # BIC approximation for variance comparison
        # The F-statistic provides evidence against H0 when it deviates from 1
        # We can use the likelihood ratio approach
        
        # Log-likelihood ratio approximation
        # For large samples, the F-test for variances can be approximated
        # Using the fact that log(F) is approximately normal under H0
        
        # More direct approach: Use the p-value from F-distribution
        # But we want BF, not p-value
        
        # Alternative: Use BIC approximation based on F-statistic
        # The evidence against H0 increases as F deviates from 1
        # We can use: BF ≈ exp(0.5 * (F - 1)^2 / variance_of_F_under_H0)
        
        # For F-distribution, under H0, E[F] = df2/(df2-2) for df2 > 2
        # Var[F] = 2*df2^2*(df1+df2-2) / (df1*(df2-2)^2*(df2-4)) for df2 > 4
        
        # Simpler BIC approximation:
        # BF10 ≈ exp(0.5 * chi2) where chi2 is the test statistic
        # For F-test: we can convert F to a test statistic
        
        # Use the fact that for variance F-test with large df:
        # The evidence is approximately proportional to |log(F)|
        # BF ≈ exp(0.5 * df1 * (log(F))^2) for F > 1
        # For F < 1, use 1/F and swap df1/df2
        
        if f_stat >= 1.0:
            # F >= 1: use direct calculation
            log_f = math.log(f_stat)
            # BIC approximation: evidence scales with df and log(F)^2
            log_bf = 0.5 * df1 * log_f**2
        else:
            # F < 1: use reciprocal (swap df1 and df2)
            log_f = math.log(1.0 / f_stat)
            log_bf = 0.5 * df2 * log_f**2
        
        # Cap the BF to avoid extreme values
        bf = math.exp(log_bf)
        return min(bf, 1e6)
        
    except Exception as e:
        print(f"Error calculating BF for variance F-test: {e}")
        return 1.0

def calc_bf_variance_f(f_stat, df1, df2):
    """
    Calculate BF10 for variance F-test using Vovk-Sellke Bound (VSB).
    
    This is the default implementation. Exact Bayesian variance comparison is sensitive 
    to priors and assumptions. A robust upper bound based on the p-value from the 
    F-distribution is safer for this specific test, especially when dealing with non-normal data.
    
    This function uses scipy.stats.f.cdf to compute the p-value, then applies
    the Vovk-Sellke Bound to get a conservative estimate of the Bayes Factor.
    
    For the legacy BIC approximation version, use calc_bf_variance_f_bic().
    
    Args:
        f_stat: F-statistic (ratio of variances)
        df1: Degrees of freedom for numerator (n1 - 1)
        df2: Degrees of freedom for denominator (n2 - 1)
        
    Returns:
        float: BF10 (upper bound using VSB)
    """
    try:
        # Calculate p-value from F-distribution (two-tailed)
        # Using scipy.stats.f.cdf (standard library)
        p_right = 1 - stats.f.cdf(f_stat, df1, df2)
        p_left = stats.f.cdf(f_stat, df1, df2)
        p_value = 2 * min(p_right, p_left)
        
        # Use Vovk-Sellke Bound (already implemented in calc_bf_vsb)
        return calc_bf_vsb(p_value)
    except Exception as e:
        print(f"Error in robust variance F-test BF: {e}")
        return 1.0

# Legacy BIC approximation version
def calc_bf_mannwhitneyu_bic(u_stat, n1, n2):
    """
    [LEGACY] Calculate Bayes Factor (BF10) for Mann-Whitney U test using BIC approximation.
    
    DEPRECATED: This function uses BIC approximation. Use calc_bf_mannwhitneyu() instead for exact calculation.
    
    Mann-Whitney U test compares two independent groups using ranks.
    We convert U statistic to z-statistic, then use t-test approximation.
    
    This uses the BIC approximation version of t-test calculation.
    
    Args:
        u_stat: Mann-Whitney U statistic
        n1: Sample size of group 1
        n2: Sample size of group 2
        
    Returns:
        float: BF10
    """
    try:
        # Convert U to z-statistic
        # Under H0, U has mean = n1*n2/2 and variance = n1*n2*(n1+n2+1)/12
        u_mean = n1 * n2 / 2.0
        u_var = n1 * n2 * (n1 + n2 + 1) / 12.0
        u_std = math.sqrt(u_var)
        
        # z = (U - mean) / std
        z_stat = (u_stat - u_mean) / u_std if u_std > 0 else 0.0
        
        # Use t-test BF calculation with z as t-statistic
        # For Mann-Whitney U, we treat it as independent samples
        return calc_bf_t_bic(z_stat, n1, n2, independent=True)
        
    except Exception as e:
        print(f"Error calculating BF for Mann-Whitney U: {e}")
        return 1.0

def calc_bf_mannwhitneyu(u_stat, n1, n2):
    """
    Calculate Bayes Factor (BF10) for Mann-Whitney U test using exact JZS calculation.
    
    Mann-Whitney U test compares two independent groups using ranks.
    We convert U statistic to z-statistic, then use the exact JZS t-test calculation.
    
    This is the default implementation. Note: The U-to-z conversion itself is an 
    asymptotic approximation, but using the exact JZS Bayes Factor for the resulting 
    z-statistic provides better precision than the BIC approximation, especially for small sample sizes.
    
    For the legacy BIC approximation version, use calc_bf_mannwhitneyu_bic().
    
    Args:
        u_stat: Mann-Whitney U statistic
        n1: Sample size of group 1
        n2: Sample size of group 2
        
    Returns:
        float: BF10 (exact calculation using JZS)
    """
    try:
        # Convert U to z-statistic
        # Under H0, U has mean = n1*n2/2 and variance = n1*n2*(n1+n2+1)/12
        u_mean = n1 * n2 / 2.0
        u_var = n1 * n2 * (n1 + n2 + 1) / 12.0
        u_std = math.sqrt(u_var)
        
        # z = (U - mean) / std
        z_stat = (u_stat - u_mean) / u_std if u_std > 0 else 0.0
        
        # Use exact JZS t-test calculation with z as t-statistic
        # For Mann-Whitney U, we treat it as independent samples
        return calc_bf_t(z_stat, n1, n2, independent=True)
        
    except Exception as e:
        print(f"Error calculating BF for Mann-Whitney U: {e}")
        # Fallback to BIC version
        return calc_bf_mannwhitneyu_bic(u_stat, n1, n2)

# Legacy BIC approximation version
def calc_bf_binomial_bic(k, n, p0):
    """
    [LEGACY] Calculate Bayes Factor (BF10) for Binomial Test using BIC approximation.
    
    DEPRECATED: This function uses BIC approximation. Use calc_bf_binomial() instead for exact calculation.
    
    H0: p = p0 (null probability)
    H1: p != p0 (alternative)
    
    Uses BIC approximation for binomial model comparison.
    
    Args:
        k: Number of successes
        n: Total number of trials
        p0: Null hypothesis probability
        
    Returns:
        float: BF10
    """
    try:
        if n == 0 or k < 0 or k > n:
            return 1.0
        
        # Observed proportion
        p_obs = k / n if n > 0 else 0.0
        
        # For binomial test, we compare:
        # H0: p = p0 (0 parameters, fixed)
        # H1: p = p_obs (1 parameter, estimated from data)
        
        # Log-likelihood under H0
        # L0 = log(n choose k) + k*log(p0) + (n-k)*log(1-p0)
        # But we can use the fact that for large n, binomial approximates normal
        # Or use BIC approximation directly
        
        # BIC approximation for binomial:
        # BIC = -2*log(L) + k*log(n)
        # where k is number of parameters
        
        # For H0: BIC0 = -2*log(L0) + 0*log(n) = -2*log(L0)
        # For H1: BIC1 = -2*log(L1) + 1*log(n)
        # BF10 = exp((BIC0 - BIC1) / 2)
        
        # Log-likelihood under H0
        if p0 <= 0 or p0 >= 1:
            return 1.0
        log_l0 = k * math.log(p0) + (n - k) * math.log(1 - p0)
        
        # Log-likelihood under H1 (using observed p)
        if p_obs <= 0 or p_obs >= 1:
            # Edge case: all successes or all failures
            if p_obs == 0:
                log_l1 = (n - k) * math.log(1.0 - 1e-10) if n > k else 0
            else:  # p_obs == 1
                log_l1 = k * math.log(1.0 - 1e-10) if k > 0 else 0
        else:
            log_l1 = k * math.log(p_obs) + (n - k) * math.log(1 - p_obs)
        
        # BIC
        bic0 = -2 * log_l0  # 0 parameters
        bic1 = -2 * log_l1 + 1 * math.log(n)  # 1 parameter
        
        # BF10 = exp((BIC0 - BIC1) / 2)
        log_bf = (bic0 - bic1) / 2.0
        return math.exp(log_bf)
        
    except Exception as e:
        print(f"Error calculating BF for Binomial Test: {e}")
        return 1.0

def calc_bf_binomial(k, n, p0=0.5):
    """
    Calculate Bayes Factor for Binomial Test using exact conjugate prior.
    
    H0: p = p0 (null probability)
    H1: p ~ Beta(1, 1) (Uniform Prior)
    
    This is the default implementation using exact conjugate prior solution.
    For Beta(1,1) uniform prior, the marginal likelihood has a closed form: 1/(n+1).
    
    For the legacy BIC approximation version, use calc_bf_binomial_bic().
    
    Ref: For Beta-Binomial with uniform prior, P(D|H1) = 1/(n+1) (exact result).
    
    Args:
        k: Number of successes
        n: Total number of trials
        p0: Null hypothesis probability (default: 0.5)
        
    Returns:
        float: BF10 (exact Bayes Factor)
    """
    try:
        if n == 0:
            return 1.0
        k = int(k)
        n = int(n)
        
        if k < 0 or k > n:
            return 1.0
        
        # Marginal Likelihood H0: Simple binomial probability
        # P(D|H0) = C(n,k) * p0^k * (1-p0)^(n-k)
        # Using scipy.special.comb for binomial coefficient (standard library)
        log_l0 = math.log(special.comb(n, k, exact=False))  # Use scipy's comb function
        log_l0 += k * math.log(p0) + (n - k) * math.log(1 - p0)
        
        # Marginal Likelihood H1: Beta-Binomial with alpha=1, beta=1 (Uniform prior)
        # For Beta(1,1) uniform prior, P(D|H1) = 1/(n+1) (exact closed form)
        # This is a known mathematical result for uniform prior on binomial parameter
        log_l1 = -math.log(n + 1)
        
        # BF10 = exp(LogL1 - LogL0)
        log_bf = log_l1 - log_l0
        return math.exp(log_bf)
        
    except Exception as e:
        print(f"Error in exact Binomial BF: {e}")
        return 1.0

def calc_bf_from_p(p_value, n, effect_direction_match=True):
    """
    Universal fallback: Estimate BF from p-value using Vovk-Sellke Upper Bound.
    
    BF10 <= 1 / (-e * p * ln(p))
    
    Args:
        effect_direction_match: If False (effect in wrong direction), BF < 1.
    """
    if not effect_direction_match:
        # Evidence AGAINST H1 if direction is wrong
        # e.g. we expected positive correlation but got negative
        return 0.1
        
    if p_value >= 1/math.e: # p >= 0.37
        return 0.5 # Inconclusive / Weak H0
        
    if p_value < 1e-300:
        return 1e6
        
    # Vovk-Sellke Bound (Maximum possible BF for a given p)
    # We use a slightly more conservative version (divide by 2) to be safe
    max_bf = 1 / (-math.e * p_value * math.log(p_value))
    return max_bf / 2

def calc_bf_vsb(p_value):
    """
    Calculate BF10 using Vovk-Sellke Upper Bound (VSB) for resampling tests.
    
    Formula: BF10 ≈ 1 / (-e·p·ln(p)) when p < 1/e
    
    This is the theoretical upper bound for BF10 given only the p-value,
    representing the maximum possible evidence for H1.
    
    This function uses the exact VSB formula (not divided by 2, unlike calc_bf_from_p),
    which is appropriate for resampling/permutation tests where we only have the p-value.
    
    Args:
        p_value: p-value from resampling/permutation test
    
    Returns:
        float: BF10 (upper bound)
    """
    if p_value >= 1/math.e:  # p >= 0.3679
        return 1.0  # Weak evidence, BF ≈ 1
    
    if p_value < 1e-300:
        return 1e6  # Very strong evidence
    
    # Vovk-Sellke Upper Bound (exact formula, not divided by 2)
    bf = 1 / (-math.e * p_value * math.log(p_value))
    return bf

# -----------------------------------------------------------------------------
# Helpers for Human Ground Truth
# -----------------------------------------------------------------------------

def get_human_pi_from_findings(findings, n=100):
    """
    Heuristic to estimate human posterior from finding description if stats missing.
    """
    # This is a fallback if exact p-value is not parsed
    # Ideally we parse p-value from ground_truth
    return 0.95 # Assume published significant effect has high posterior

# -----------------------------------------------------------------------------
# Bootstrap Utilities
# -----------------------------------------------------------------------------

def bootstrap_metric(data_pool, metric_func, n_bootstrap=1000, random_seed=None):
    """
    Calculate bootstrap standard error for a metric computed on participant data.
    
    This function resamples participants with replacement from the data pool,
    computes the metric on each bootstrap sample, and returns the mean and
    standard error of the bootstrap distribution.
    
    Args:
        data_pool: List of participant data dictionaries (individual_data format)
        metric_func: Function that takes a list of participant data and returns a metric value
                     Should accept: metric_func(participant_data_list) -> float
        n_bootstrap: Number of bootstrap iterations (default: 1000)
        random_seed: Optional random seed for reproducibility
        
    Returns:
        tuple: (mean, standard_error) of the bootstrap distribution
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if not data_pool:
        return 0.0, 0.0
    
    n_participants = len(data_pool)
    if n_participants == 0:
        return 0.0, 0.0
    
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data_pool, size=n_participants, replace=True).tolist()
        
        try:
            # Compute metric on bootstrap sample
            metric_value = metric_func(bootstrap_sample)
            if not (np.isnan(metric_value) or np.isinf(metric_value)):
                bootstrap_values.append(metric_value)
        except Exception as e:
            # Skip this bootstrap iteration if metric computation fails
            continue
    
    if not bootstrap_values:
        # Fallback: compute metric on original data
        try:
            original_metric = metric_func(data_pool)
            return float(original_metric), 0.0
        except Exception:
            return 0.0, 0.0
    
    bootstrap_mean = float(np.mean(bootstrap_values))
    bootstrap_std = float(np.std(bootstrap_values, ddof=1))
    bootstrap_se = bootstrap_std  # Standard error is the standard deviation of bootstrap distribution
    
    return bootstrap_mean, bootstrap_se


# -----------------------------------------------------------------------------
# Frequentist Replication Consistency Metric
# -----------------------------------------------------------------------------

class FrequentistConsistency:
    """
    Calculate Frequentist Replication Consistency using Z-difference metric.
    
    This class implements the standard meta-analysis approach for comparing
    effect sizes between two studies (human vs agent), normalized by their
    combined standard errors.
    
    Formula: Z_diff = (delta_a - delta_h) / sqrt(SE_a^2 + SE_h^2)
    Consistency Score = 2 * (1 - Phi(|Z_diff|))
    
    Where Phi is the standard normal cumulative distribution function.
    """
    
    @staticmethod
    def cohens_d_se(d: float, n1: int, n2: int = 0) -> float:
        """
        Calculate standard error for Cohen's d.
        
        For independent samples (n2 > 0): SE_d ≈ sqrt((n1+n2)/(n1*n2) + d^2/(2*(n1+n2)))
        For one-sample/paired (n2 = 0): SE_d ≈ sqrt(1/n1 + d^2/(2*n1))
        """
        if n1 <= 0:
            return float('inf')
            
        if n2 > 0:
            # Independent samples
            n_total = n1 + n2
            se_sq = (n_total / (n1 * n2)) + (d**2 / (2 * n_total))
        else:
            # One-sample / Paired
            se_sq = (1.0 / n1) + (d**2 / (2 * n1))
            
        return math.sqrt(se_sq)
    
    @staticmethod
    def correlation_se(r: float, n: int) -> float:
        """
        Calculate standard error for correlation coefficient using Fisher's z-transform.
        
        First converts r to Fisher's z: z = 0.5 * ln((1+r)/(1-r))
        Then: SE_z = 1 / sqrt(N-3)
        
        Args:
            r: Pearson correlation coefficient
            n: Sample size
            
        Returns:
            float: Standard error of Fisher's z-transformed correlation
        """
        if n <= 3:
            return float('inf')
        if abs(r) >= 1.0:
            return float('inf')
        return 1.0 / math.sqrt(n - 3)
    
    @staticmethod
    def correlation_to_fisher_z(r: float) -> float:
        """
        Convert Pearson correlation to Fisher's z-scale.
        
        Formula: z = 0.5 * ln((1+r)/(1-r))
        
        Args:
            r: Pearson correlation coefficient
            
        Returns:
            float: Fisher's z-transformed correlation
        """
        if abs(r) >= 1.0:
            return float('inf') if r > 0 else float('-inf')
        if r == 0:
            return 0.0
        return 0.5 * math.log((1 + r) / (1 - r))
    
    @staticmethod
    def log_odds_ratio(a: float, b: float, c: float, d: float) -> float:
        """
        Calculate log odds ratio with Haldane-Anscombe correction for zero counts.
        
        Formula: logOR = ln((A * D) / (B * C))
        With correction: add 0.5 to all cells if any cell is zero.
        """
        if a == 0 or b == 0 or c == 0 or d == 0:
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
        
        # Avoid log(0)
        return math.log((a * d) / (b * c))

    @staticmethod
    def log_odds_ratio_se(a: float, b: float, c: float, d: float) -> float:
        """
        Calculate standard error for log odds ratio with Haldane-Anscombe correction.
        
        Formula: SE_logOR = sqrt(1/A + 1/B + 1/C + 1/D)
        With correction: add 0.5 to all cells if any cell is zero.
        """
        if a == 0 or b == 0 or c == 0 or d == 0:
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
            
        se_sq = (1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d)
        return math.sqrt(se_sq)
    
    @staticmethod
    def t_to_cohens_d(t_stat: float, n1: int, n2: Optional[int] = None, independent: bool = True) -> float:
        """
        Convert t-statistic to Cohen's d effect size.
        
        For independent samples: d = t * sqrt((n1+n2)/(n1*n2))
        For paired/one-sample: d = t / sqrt(n)
        
        Args:
            t_stat: t-statistic
            n1: Sample size of group 1 (or total n for paired)
            n2: Sample size of group 2 (if independent)
            independent: True for independent samples, False for paired/one-sample
            
        Returns:
            float: Cohen's d
        """
        if independent:
            if n2 is None:
                raise ValueError("n2 must be provided for independent t-test")
            if n1 <= 0 or n2 <= 0:
                return 0.0
            return t_stat * math.sqrt((n1 + n2) / (n1 * n2))
        else:
            if n1 <= 0:
                return 0.0
            return t_stat / math.sqrt(n1)
    
    @staticmethod
    def calculate_z_diff(
        effect_agent: float,
        se_agent: float,
        effect_human: float,
        se_human: float
    ) -> Tuple[float, float]:
        """
        Calculate Z-difference and Consistency Score.
        
        Z_diff = (delta_a - delta_h) / sqrt(SE_a^2 + SE_h^2)
        Consistency Score = 2 * (1 - Phi(|Z_diff|))
        
        Args:
            effect_agent: Agent effect size (Cohen's d, Fisher's z, or log OR)
            se_agent: Standard error of agent effect size
            effect_human: Human effect size (same scale as agent)
            se_human: Standard error of human effect size
            
        Returns:
            tuple: (z_diff, consistency_score)
                - z_diff: Z-score of the difference
                - consistency_score: P-value of the difference (0-1 scale, where 1 = perfect match)
        """
        # Calculate combined standard error
        se_combined_sq = se_agent**2 + se_human**2
        if se_combined_sq <= 0:
            # If both SEs are 0 or invalid, return extreme values
            if effect_agent == effect_human:
                return 0.0, 1.0  # Perfect match
            else:
                return float('inf'), 0.0  # Completely different
        
        se_combined = math.sqrt(se_combined_sq)
        
        # Calculate Z-difference
        z_diff = (effect_agent - effect_human) / se_combined
        
        # Calculate Consistency Score (p-value of the difference)
        # Using two-tailed test: P(|Z| > |z_diff|)
        abs_z = abs(z_diff)
        if math.isinf(abs_z) or math.isnan(abs_z):
            consistency_score = 0.0
        else:
            # Phi is the standard normal CDF
            # P(|Z| > |z_diff|) = 2 * (1 - Phi(|z_diff|))
            consistency_score = 2 * (1 - stats.norm.cdf(abs_z))
        
        return z_diff, consistency_score
    
    @staticmethod
    def calculate_consistency_for_t_test(
        t_agent: float,
        n1_agent: int,
        n2_agent: Optional[int],
        t_human: float,
        n1_human: int,
        n2_human: Optional[int],
        independent: bool = True
    ) -> Tuple[float, float, float, float]:
        """
        Calculate Z-difference and Consistency Score for t-test.
        
        Args:
            t_agent: Agent t-statistic
            n1_agent, n2_agent: Agent sample sizes
            t_human: Human t-statistic
            n1_human, n2_human: Human sample sizes
            independent: True for independent samples, False for paired
            
        Returns:
            tuple: (d_agent, d_human, z_diff, consistency_score)
        """
        # Convert t-statistics to Cohen's d
        d_agent = FrequentistConsistency.t_to_cohens_d(t_agent, n1_agent, n2_agent, independent)
        d_human = FrequentistConsistency.t_to_cohens_d(t_human, n1_human, n2_human, independent)
        
        # Calculate standard errors
        if independent:
            se_agent = FrequentistConsistency.cohens_d_se(d_agent, n1_agent, n2_agent or n1_agent)
            se_human = FrequentistConsistency.cohens_d_se(d_human, n1_human, n2_human or n1_human)
        else:
            se_agent = FrequentistConsistency.cohens_d_se(d_agent, n1_agent, 0)
            se_human = FrequentistConsistency.cohens_d_se(d_human, n1_human, 0)
        
        # Calculate Z-difference
        z_diff, consistency_score = FrequentistConsistency.calculate_z_diff(
            d_agent, se_agent, d_human, se_human
        )
        
        return d_agent, d_human, z_diff, consistency_score
    
    @staticmethod
    def calculate_consistency_for_correlation(
        r_agent: float,
        n_agent: int,
        r_human: float,
        n_human: int
    ) -> Tuple[float, float, float, float]:
        """
        Calculate Z-difference and Consistency Score for correlation.
        
        Uses Fisher's z-transform for proper comparison.
        
        Args:
            r_agent: Agent correlation coefficient
            n_agent: Agent sample size
            r_human: Human correlation coefficient
            n_human: Human sample size
            
        Returns:
            tuple: (z_agent, z_human, z_diff, consistency_score)
                where z_agent and z_human are Fisher's z-transformed correlations
        """
        # Convert to Fisher's z
        z_agent = FrequentistConsistency.correlation_to_fisher_z(r_agent)
        z_human = FrequentistConsistency.correlation_to_fisher_z(r_human)
        
        # Calculate standard errors
        se_agent = FrequentistConsistency.correlation_se(r_agent, n_agent)
        se_human = FrequentistConsistency.correlation_se(r_human, n_human)
        
        # Calculate Z-difference
        z_diff, consistency_score = FrequentistConsistency.calculate_z_diff(
            z_agent, se_agent, z_human, se_human
        )
        
        return z_agent, z_human, z_diff, consistency_score
    
    @staticmethod
    def calculate_consistency_for_chi_square(
        chi2_agent: float,
        contingency_agent: list,
        chi2_human: float,
        contingency_human: list
    ) -> Tuple[float, float, float, float]:
        """
        Calculate Z-difference and Consistency Score for chi-square test.
        
        Uses log odds ratio as the effect size metric.
        
        Args:
            chi2_agent: Agent chi-square statistic
            contingency_agent: Agent 2x2 contingency table [[a, b], [c, d]]
            chi2_human: Human chi-square statistic
            contingency_human: Human 2x2 contingency table [[a, b], [c, d]]
            
        Returns:
            tuple: (log_or_agent, log_or_human, z_diff, consistency_score)
        """
        # Extract contingency table counts
        if len(contingency_agent) != 2 or len(contingency_agent[0]) != 2:
            raise ValueError("contingency_agent must be a 2x2 table")
        if len(contingency_human) != 2 or len(contingency_human[0]) != 2:
            raise ValueError("contingency_human must be a 2x2 table")
        
        a_a, b_a = contingency_agent[0]
        c_a, d_a = contingency_agent[1]
        a_h, b_h = contingency_human[0]
        c_h, d_h = contingency_human[1]
        
        # Calculate log odds ratios using corrected helper
        log_or_agent = FrequentistConsistency.log_odds_ratio(a_a, b_a, c_a, d_a)
        log_or_human = FrequentistConsistency.log_odds_ratio(a_h, b_h, c_h, d_h)
        
        # Calculate standard errors using corrected helper
        se_agent = FrequentistConsistency.log_odds_ratio_se(a_a, b_a, c_a, d_a)
        se_human = FrequentistConsistency.log_odds_ratio_se(a_h, b_h, c_h, d_h)
        
        # Calculate Z-difference
        z_diff, consistency_score = FrequentistConsistency.calculate_z_diff(
            log_or_agent, se_agent, log_or_human, se_human
        )
        
        return log_or_agent, log_or_human, z_diff, consistency_score

    @staticmethod
    def mann_whitney_u_to_r_rb(u_stat: float, n1: int, n2: int) -> float:
        """
        Convert Mann-Whitney U statistic to Rank-Biserial Correlation (r_rb).
        Formula: r_rb = 1 - (2 * U) / (n1 * n2)
        """
        if n1 <= 0 or n2 <= 0:
            return 0.0
        return 1 - (2 * u_stat) / (n1 * n2)

    @staticmethod
    def r_rb_se(r_rb: float, n1: int, n2: int) -> float:
        """
        Approximate Standard Error for Rank-Biserial Correlation (r_rb).
        Using simpler approximation for large N: SE_r_rb = sqrt((1 - r_rb^2) / (n1 + n2 - 1))
        """
        n_total = n1 + n2
        if n_total <= 1:
            return float('inf')
        return math.sqrt((1 - r_rb**2) / (n_total - 1))

    @staticmethod
    def proportion_difference_se(p1: float, n1: int, p2: float, n2: int) -> float:
        """
        Calculate Standard Error for the difference between two proportions.
        Formula: SE_diff = sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        """
        if n1 <= 0 or n2 <= 0:
            return float('inf')
        se1_sq = (p1 * (1 - p1)) / n1
        se2_sq = (p2 * (1 - p2)) / n2
        return math.sqrt(se1_sq + se2_sq)

    @staticmethod
    def calculate_consistency_for_mann_whitney(
        u_agent: float, n1_agent: int, n2_agent: int,
        u_human: float, n1_human: int, n2_human: int
    ) -> Tuple[float, float, float, float]:
        """
        Calculate Z-difference and Consistency Score for Mann-Whitney U test.
        """
        r_rb_agent = FrequentistConsistency.mann_whitney_u_to_r_rb(u_agent, n1_agent, n2_agent)
        r_rb_human = FrequentistConsistency.mann_whitney_u_to_r_rb(u_human, n1_human, n2_human)

        se_agent = FrequentistConsistency.r_rb_se(r_rb_agent, n1_agent, n2_agent)
        se_human = FrequentistConsistency.r_rb_se(r_rb_human, n1_human, n2_human)

        z_diff, consistency_score = FrequentistConsistency.calculate_z_diff(
            r_rb_agent, se_agent, r_rb_human, se_human
        )
        return r_rb_agent, r_rb_human, z_diff, consistency_score

    @staticmethod
    def calculate_consistency_for_binomial(
        k_agent: int, n_agent: int,
        k_human: int, n_human: int
    ) -> Tuple[float, float, float, float]:
        """
        Calculate Z-difference and Consistency Score for Binomial/Sign test.
        Uses proportion difference as effect size.
        """
        p_agent = k_agent / n_agent if n_agent > 0 else 0.0
        p_human = k_human / n_human if n_human > 0 else 0.0

        # We treat this as a comparison between two proportions (p_agent vs p_human)
        # Standard error for p_agent and p_human
        se_agent = math.sqrt(p_agent * (1 - p_agent) / n_agent) if n_agent > 0 else 0
        se_human = math.sqrt(p_human * (1 - p_human) / n_human) if n_human > 0 else 0

        z_diff, consistency_score = FrequentistConsistency.calculate_z_diff(
            p_agent, se_agent, p_human, se_human
        )
        return p_agent, p_human, z_diff, consistency_score
    
    @staticmethod
    def effect_to_r_equiv(test_type: str, effect_value: float) -> Optional[float]:
        """
        Convert effect size to Pearson correlation r-equivalent.
        
        This matches the metric used in OSC (2015).
        Conversion: r = d / sqrt(d^2 + 4)
        """
        if effect_value is None or math.isnan(effect_value):
            return 0.0
        
        test_type_lower = test_type.lower()
        
        # If it's already a correlation, it's stored as Fisher z, convert back to r
        if test_type_lower in ["correlation", "pearson", "spearman"]:
            return math.tanh(effect_value)
            
        # For others, convert Cohen's d to r
        # Formula: r = d / sqrt(d^2 + 4)
        d = float(effect_value)
        r = d / math.sqrt(d**2 + 4.0)
        return float(r)

    @staticmethod
    def effect_to_d_equiv(test_type: str, effect_value: float) -> Optional[float]:
        """
        Convert effect size to Cohen's d-equivalent for correlation-based ECS.
        
        Conversion rules:
        - t-test/f-test/anova: already d -> return as-is
        - correlation (stored as Fisher z): r = tanh(z), d = 2r/sqrt(1-r^2)
        - chi-square (stored as log OR): d ≈ log(OR) * sqrt(3)/π
        - Mann-Whitney (stored as rank-biserial r_rb): d = 2r_rb/sqrt(1-r_rb^2)
        - binomial (stored as proportion p, signed): clamp p, z = Φ^-1(p), d ≈ sqrt(2)*z
        
        Args:
            test_type: Type of statistical test
            effect_value: Effect size value in its native unit
            
        Returns:
            float: Cohen's d-equivalent, or None if conversion fails
        """
        if effect_value is None or math.isnan(effect_value) or math.isinf(effect_value):
            return None
        
        test_type_lower = test_type.lower()
        
        # t-test, f-test, anova: already Cohen's d
        if test_type_lower in ["t-test", "t_test", "f-test", "f_test", "anova"]:
            return float(effect_value)
        
        # correlation: stored as Fisher z, convert to r then to d
        elif test_type_lower in ["correlation", "pearson", "spearman"]:
            # Convert Fisher z to r
            r = math.tanh(effect_value)
            # Clamp r to avoid division by zero
            eps = 1e-6
            r = max(-1.0 + eps, min(1.0 - eps, r))
            # Convert r to d: d = 2r / sqrt(1-r^2)
            if abs(r) >= 1.0 - eps:
                return None  # Invalid
            d = (2.0 * r) / math.sqrt(1.0 - r**2)
            return float(d)
        
        # chi-square: stored as log OR, convert to d
        elif test_type_lower in ["chi-square", "chi2", "chi_square"]:
            # d ≈ log(OR) * sqrt(3) / π
            d = effect_value * math.sqrt(3.0) / math.pi
            return float(d)
        
        # Mann-Whitney: stored as rank-biserial correlation r_rb
        elif test_type_lower in ["mannwhitneyu", "wilcoxon"]:
            # Treat r_rb as r, then d = 2r / sqrt(1-r^2)
            r_rb = effect_value
            eps = 1e-6
            r_rb = max(-1.0 + eps, min(1.0 - eps, r_rb))
            if abs(r_rb) >= 1.0 - eps:
                return None
            d = (2.0 * r_rb) / math.sqrt(1.0 - r_rb**2)
            return float(d)
        
        # binomial: stored as proportion p, convert via signed difference from null
        elif test_type_lower in ["binomial", "sign_test"]:
            # For binomial tests, effect_value is the proportion p (0-1)
            # We need to compute the signed difference from null (typically 0.5)
            # Then convert to d using the z-score method
            p0 = 0.5  # Typical null for binomial tests
            p = float(effect_value)
            # Clamp p to [eps, 1-eps] to avoid edge cases
            eps = 1e-6
            p = max(eps, min(1.0 - eps, p))
            
            # Compute signed difference: delta_p = p - p0
            delta_p = p - p0
            
            # For small differences, use linear approximation
            # For larger differences, use z-score conversion
            # Standard error of proportion: SE_p = sqrt(p*(1-p)/n)
            # But we don't have n here, so we use a normalized approach
            # Convert proportion difference to z-score: z = (p - p0) / SE_p
            # Approximate SE_p using p0 for normalization: SE_p ≈ sqrt(p0*(1-p0)/n)
            # For large n, we can use: d ≈ (p - p0) / sqrt(p0*(1-p0)) * sqrt(n)
            # But without n, we use a simpler approximation: d ≈ 2 * (p - p0) / sqrt(p0*(1-p0))
            # This gives d in units of standard deviations
            d = 2.0 * delta_p / math.sqrt(p0 * (1.0 - p0))
            return float(d)
        
        # Unknown test type
        return None


# -----------------------------------------------------------------------------
# PAS (Probability Alignment Score) Aggregation via Fisher-z Pooling
# -----------------------------------------------------------------------------

def fisher_z_transform(r: float, clamp: bool = True) -> float:
    """
    Fisher z-transform: z = atanh(r) = 0.5 * ln((1+r)/(1-r))
    
    Args:
        r: Correlation-like value in [-1, 1]
        clamp: If True, clamp r to (-1+eps, 1-eps) before transform to avoid infinities
        
    Returns:
        float: Fisher z value
    """
    if clamp:
        # Clamp to avoid infinities
        eps = 1e-6
        r = max(-1.0 + eps, min(1.0 - eps, r))
    
    if abs(r) >= 1.0:
        return float('inf') if r > 0 else float('-inf')
    if abs(r) < 1e-10:
        return 0.0
    
    return 0.5 * math.log((1 + r) / (1 - r))


def fisher_z_inverse(z: float) -> float:
    """
    Inverse Fisher z-transform: r = tanh(z)
    
    Args:
        z: Fisher z value
        
    Returns:
        float: Correlation-like value in [-1, 1]
    """
    if math.isinf(z) or math.isnan(z):
        return 1.0 if z > 0 else -1.0
    return math.tanh(z)


def compute_n_eff_for_test(test_result: dict) -> int:
    """
    Compute effective sample size (n_eff) for a test result.
    Used as weights in Fisher-z pooling.
    
    Args:
        test_result: Test result dictionary with test type and sample size info
        
    Returns:
        int: Effective sample size (fallback to 1 if unavailable)
    """
    test_type = test_result.get("statistical_test_type", "").lower()
    
    # Try to get n_eff if already stored
    if "n_eff" in test_result:
        n_eff = test_result.get("n_eff")
        if n_eff is not None and n_eff > 0:
            return int(n_eff)
    
    # Extract from test-specific fields
    if test_type in ["t-test", "t_test", "f-test", "f_test", "anova"]:
        # For t/f tests, try to get n1, n2
        # Check if we have n_agent, n2_agent or similar fields
        n1 = test_result.get("n_agent") or test_result.get("n1_agent")
        n2 = test_result.get("n2_agent")
        if n1 is not None:
            if n2 is not None and n2 > 0:
                return int(n1 + n2)  # Independent samples
            else:
                return int(n1)  # Paired/one-sample
    elif test_type in ["correlation", "pearson", "spearman"]:
        n = test_result.get("n_agent")
        if n is not None and n > 0:
            return int(n)
    elif test_type in ["chi-square", "chi2", "chi_square"]:
        # For chi-square, sum all contingency table cells
        contingency = test_result.get("contingency_table_agent")
        if contingency and isinstance(contingency, list):
            total = sum(sum(row) if isinstance(row, list) else row for row in contingency)
            if total > 0:
                return int(total)
    elif test_type in ["mannwhitneyu", "wilcoxon"]:
        n1 = test_result.get("n1_agent")
        n2 = test_result.get("n2_agent")
        if n1 is not None and n2 is not None:
            return int(n1 + n2)
    elif test_type in ["binomial", "sign_test"]:
        n = test_result.get("n_agent")
        if n is not None and n > 0:
            return int(n)
    
    # Fallback: return 1 (unweighted)
    return 1


def aggregate_finding_pas_raw(test_results_for_finding: list) -> float:
    """
    Aggregate PAS (raw) for a finding using Fisher-z pooling.
    
    Formula:
    1. Convert each PAS_i to r_i = 2*PAS_i - 1
    2. If K=1: return PAS_1
    3. If K>1: Fisher transform, unweighted average (1/N * sum(z)), inverse transform
    
    Args:
        test_results_for_finding: List of test result dicts for a single finding
        
    Returns:
        float: Aggregated PAS (raw) for the finding
    """
    if not test_results_for_finding:
        return 0.5
    
    if len(test_results_for_finding) == 1:
        # Single test: no pooling needed
        pas = test_results_for_finding[0].get("pas") or test_results_for_finding[0].get("score", 0.5)
        return float(pas)
    
    # Multiple tests: Fisher-z pool with unweighted average
    z_values = []
    
    for test in test_results_for_finding:
        pas = test.get("pas") or test.get("score", 0.5)
        r = 2.0 * float(pas) - 1.0  # Convert to [-1, 1]
        
        # Fisher transform
        z = fisher_z_transform(r, clamp=True)
        if math.isinf(z) or math.isnan(z):
            continue
        
        z_values.append(z)
    
    if not z_values:
        # Fallback: simple mean of PAS
        pas_list = [t.get("pas") or t.get("score", 0.5) for t in test_results_for_finding]
        return float(np.mean(pas_list))
    
    # Unweighted average of Fisher z: 1/N * sum(z)
    z_bar = float(np.mean(z_values))
    
    # Inverse transform
    r_finding = fisher_z_inverse(z_bar)
    
    # Convert back to PAS
    pas_finding = (r_finding + 1.0) / 2.0
    return float(pas_finding)


def aggregate_finding_pas_norm(test_results_for_finding: list) -> float:
    """
    Aggregate PAS (normalized) for a finding using Fisher-z pooling.
    
    Formula:
    1. For each test: H_i = π_h,i^2 + (1-π_h,i)^2
    2. r_i = 2*PAS_i - 1, r_h,i = 2*H_i - 1
    3. r_i' = r_i / r_h,i (normalized ratio)
    4. If K=1: return r_1'
    5. If K>1: Fisher transform r_i', weighted average, inverse transform
    
    Args:
        test_results_for_finding: List of test result dicts for a single finding
        
    Returns:
        float: Aggregated PAS (normalized) for the finding
    """
    if not test_results_for_finding:
        return 0.0
    
    if len(test_results_for_finding) == 1:
        # Single test: compute normalized ratio directly
        test = test_results_for_finding[0]
        pas = test.get("pas") or test.get("score", 0.5)
        pi_h = test.get("pi_human", 0.5)
        
        # Handle dict pi_h (3-way) vs scalar (2-way)
        if isinstance(pi_h, dict):
            p0 = pi_h.get('pi_zero', 0.5)
            pp = pi_h.get('pi_plus', 0.25)
            pm = pi_h.get('pi_minus', 0.25)
            H = p0**2 + pp**2 + pm**2
        else:
            H = float(pi_h)**2 + (1.0 - float(pi_h))**2
        
        r = 2.0 * float(pas) - 1.0
        r_h = 2.0 * H - 1.0
        
        if abs(r_h) < 1e-8:
            return 0.0  # No meaningful normalization
        
        r_norm = r / r_h
        # Clamp to valid range
        r_norm = max(-1.0 + 1e-6, min(1.0 - 1e-6, r_norm))
        return float(r_norm)
    
    # Multiple tests: Fisher-z pool normalized ratios
    z_values = []
    weights = []
    
    for test in test_results_for_finding:
        pas = test.get("pas") or test.get("score", 0.5)
        pi_h = test.get("pi_human", 0.5)
        
        # Handle dict pi_h (3-way) vs scalar (2-way)
        if isinstance(pi_h, dict):
            p0 = pi_h.get('pi_zero', 0.5)
            pp = pi_h.get('pi_plus', 0.25)
            pm = pi_h.get('pi_minus', 0.25)
            H = p0**2 + pp**2 + pm**2
        else:
            H = float(pi_h)**2 + (1.0 - float(pi_h))**2
        
        r = 2.0 * float(pas) - 1.0
        r_h = 2.0 * H - 1.0
        
        if abs(r_h) < 1e-8:
            continue  # Skip tests with no meaningful normalization
        
        r_norm = r / r_h
        # Clamp to valid range before Fisher transform
        r_norm = max(-1.0 + 1e-6, min(1.0 - 1e-6, r_norm))
        
        z = fisher_z_transform(r_norm, clamp=False)  # Already clamped
        if math.isinf(z) or math.isnan(z):
            continue
        
        n_eff = compute_n_eff_for_test(test)
        z_values.append(z)
        weights.append(float(n_eff))
    
    if not z_values:
        # Fallback: simple mean of normalized ratios
        r_norm_list = []
        for test in test_results_for_finding:
            pas = test.get("pas") or test.get("score", 0.5)
            pi_h = test.get("pi_human", 0.5)
            if isinstance(pi_h, dict):
                p0 = pi_h.get('pi_zero', 0.5)
                pp = pi_h.get('pi_plus', 0.25)
                pm = pi_h.get('pi_minus', 0.25)
                H = p0**2 + pp**2 + pm**2
            else:
                H = float(pi_h)**2 + (1.0 - float(pi_h))**2
            r = 2.0 * float(pas) - 1.0
            r_h = 2.0 * H - 1.0
            if abs(r_h) >= 1e-8:
                r_norm_list.append(r / r_h)
        if r_norm_list:
            return float(np.mean(r_norm_list))
        return 0.0
    
    # Weighted average of Fisher z
    total_weight = sum(weights)
    if total_weight <= 0:
        z_bar = float(np.mean(z_values))
    else:
        z_bar = sum(z * w for z, w in zip(z_values, weights)) / total_weight
    
    # Inverse transform
    r_finding_norm = fisher_z_inverse(z_bar)
    return float(r_finding_norm)


def aggregate_finding_ecs(test_results_for_finding: list) -> float:
    """
    Aggregate ECS (Effect Consistency Score) for a finding using RMS-Stouffer method.
    
    Mathematical Framework (RMS-Stouffer):
    
    Level 1: Test-level Z-scores
        Z_{j,k} = (δ_agent - δ_human) / sqrt(SE_agent² + SE_human²)
    
    Level 2: Finding-level RMS pooling
        RMS_j = sqrt(mean(Z_{j,k}²))
        χ²_j = K_j * RMS_j² = sum(Z_{j,k}²)
    
    Level 3: Finding-level p-value (Chi-squared test)
        p_j = P(χ²_{K_j} ≥ χ²_j | H_0)
        where H_0: δ_agent = δ_human for all tests in finding j
    
    Args:
        test_results_for_finding: List of test result dicts for a single finding
        
    Returns:
        float: Aggregated ECS (p-value) for the finding. Higher values indicate
               better consistency (smaller effect size differences).
    """
    if not test_results_for_finding:
        return 0.0
    
    z_diff_values = []
    for test in test_results_for_finding:
        z_diff = test.get("z_diff")
        if z_diff is not None and not (math.isnan(z_diff) or math.isinf(z_diff)):
            z_diff_values.append(float(z_diff))
    
    if not z_diff_values:
        return 0.0
    
    K = len(z_diff_values)
    
    if K == 1:
        # Single test: use two-tailed p-value from standard normal
        z_abs = abs(z_diff_values[0])
        if math.isinf(z_abs) or math.isnan(z_abs):
            return 0.0
        p_value = 2.0 * (1.0 - stats.norm.cdf(z_abs))
    else:
        # Multiple tests: RMS pooling → Chi-squared test
        # Step 1: Calculate RMS
        rms = math.sqrt(np.mean([z**2 for z in z_diff_values]))
        
        # Step 2: Calculate chi-squared statistic
        # χ² = K * RMS² = sum(Z²)
        chi_squared = K * (rms ** 2)
        
        if math.isinf(chi_squared) or math.isnan(chi_squared):
            return 0.0
        
        # Step 3: Convert to p-value using chi-squared distribution
        # p = P(χ²_K ≥ χ²_observed | H_0)
        # This is a one-sided test (we only care about large discrepancies)
        p_value = 1.0 - stats.chi2.cdf(chi_squared, df=K)
    
    return float(p_value)


def aggregate_study_ecs_strict(finding_p_values: list) -> float:
    """
    Aggregate finding-level ECS p-values to study-level using Stouffer's method.
    
    Mathematical Framework (RMS-Stouffer):
    
    Level 4: Convert finding-level p-values to standard normal Z-scores
        Z_j* = Φ⁻¹(1 - p_j)
        where p_j is the finding-level p-value from chi-squared test
    
    Level 5: Study-level Stouffer aggregation (equal weights)
        Z_study = (1/√m) * sum(Z_j*)
        where m = number of findings
    
    Level 6: Study-level p-value
        p_study = 1 - Φ(Z_study)  (one-sided: testing for inconsistency)
        OR
        p_study = 2 * (1 - Φ(|Z_study|))  (two-sided)
    
    Global null hypothesis:
        H_0: δ_agent = δ_human for all findings in the study
    
    Args:
        finding_p_values: List of finding-level p-values (from aggregate_finding_ecs)
        
    Returns:
        float: Study-level ECS_Strict (p-value). Higher values indicate better
               consistency. Interpretation:
               - p ≥ 0.10: No evidence of inconsistency
               - 0.05 ≤ p < 0.10: Weak evidence of inconsistency
               - 0.01 ≤ p < 0.05: Moderate evidence of inconsistency
               - p < 0.01: Strong evidence of inconsistency
    """
    if not finding_p_values:
        return 0.0
    
    # Filter out invalid p-values
    valid_p_values = []
    for p in finding_p_values:
        if p is not None and not (math.isnan(p) or math.isinf(p)):
            # Clamp p-values to valid range [0, 1]
            p_clamped = max(0.0, min(1.0, float(p)))
            valid_p_values.append(p_clamped)
    
    if not valid_p_values:
        return 0.0
    
    m = len(valid_p_values)
    
    if m == 1:
        # Single finding: return its p-value directly
        return valid_p_values[0]
    
    # Step 1: Convert p-values to standard normal Z-scores
    # Z_j* = Φ⁻¹(1 - p_j)
    # Note: We use (1 - p_j) because p_j represents "probability of this much or more inconsistency"
    # For p_j close to 1 (high consistency), we want Z_j* close to -∞ (negative)
    # For p_j close to 0 (low consistency), we want Z_j* close to +∞ (positive)
    z_scores = []
    for p in valid_p_values:
        # Handle edge cases
        if p >= 1.0:
            # Perfect consistency: Z = -∞, but we'll use a large negative value
            z_scores.append(-10.0)  # Approximate -∞
        elif p <= 0.0:
            # Perfect inconsistency: Z = +∞, but we'll use a large positive value
            z_scores.append(10.0)  # Approximate +∞
        else:
            # Standard case: Z = Φ⁻¹(1 - p)
            z = stats.norm.ppf(1.0 - p)
            if math.isinf(z) or math.isnan(z):
                # Fallback for numerical issues
                if p > 0.5:
                    z_scores.append(-10.0)
                else:
                    z_scores.append(10.0)
            else:
                z_scores.append(float(z))
    
    # Step 2: Stouffer aggregation (equal weights)
    # Z_study = (1/√m) * sum(Z_j*)
    z_study = (1.0 / math.sqrt(m)) * sum(z_scores)
    
    if math.isinf(z_study) or math.isnan(z_study):
        return 0.0
    
    # Step 3: Convert back to p-value
    # One-sided: p = 1 - Φ(Z_study)  (testing for inconsistency)
    # This tests whether Z_study is significantly positive (indicating inconsistency)
    p_study = 1.0 - stats.norm.cdf(z_study)
    
    # Clamp to valid range
    p_study = max(0.0, min(1.0, p_study))
    
    return float(p_study)


def aggregate_pas_inverse_variance(
    pas_values: list,
    se_values: list,
    min_se: float = 1e-10
) -> tuple:
    """
    Aggregate per-study PAS using inverse-variance (Z-score) weighting.

    When studies have valid bootstrap SEs, this is the meta-analytic (inverse-variance
    weighted) mean and its SE. When SEs are missing or zero, falls back to the
    simple mean and the propagated SE of the mean.

    Formulas:
    - With valid SE_k: w_k = 1/SE_k^2, PAS_agg = sum(w_k * PAS_k)/sum(w_k),
      SE(PAS_agg) = 1/sqrt(sum(w_k)).
    - Fallback: PAS_agg = mean(PAS_k), SE = (1/K)*sqrt(sum(SE_k^2)).

    Args:
        pas_values: Per-study PAS (raw, in [0,1]).
        se_values: Per-study standard errors (same order as pas_values).
        min_se: Minimum SE to treat as "has SE" (avoids div-by-zero).

    Returns:
        (pas_agg, pas_agg_se): Aggregated PAS and its standard error.
    """
    if not pas_values or len(pas_values) != len(se_values):
        return (0.0, 0.0)
    n = len(pas_values)
    pas = []
    se = []
    for i in range(n):
        p = pas_values[i]
        if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
            continue
        s = se_values[i] if i < len(se_values) else 0.0
        s = float(s) if s is not None else 0.0
        if math.isnan(s) or math.isinf(s):
            s = 0.0
        pas.append(float(p))
        se.append(s)
    if not pas:
        return (0.0, 0.0)
    k = len(pas)
    # Studies with valid SE
    weights = []
    for i, s in enumerate(se):
        if s is not None and not (math.isnan(s) or math.isinf(s) or s < min_se):
            weights.append(1.0 / (s * s))
        else:
            weights.append(0.0)
    total_w = sum(weights)
    if total_w > 0:
        pas_agg = sum(weights[i] * pas[i] for i in range(k)) / total_w
        pas_agg_se = 1.0 / math.sqrt(total_w)
        return (float(pas_agg), float(pas_agg_se))
    # Fallback: equal-weight mean and SE(mean) = (1/K)*sqrt(sum(SE_k^2))
    pas_agg = sum(pas) / k
    pas_agg_se = math.sqrt(sum(s * s for s in se)) / k if k else 0.0
    return (float(pas_agg), float(pas_agg_se))


def weighted_corr(x: list, y: list, w: list) -> Optional[float]:
    """
    Compute weighted Pearson correlation between x and y.
    
    Formula:
    r_w = sum(w_i * (x_i - mu_x) * (y_i - mu_y)) / sqrt(sum(w_i * (x_i - mu_x)^2) * sum(w_i * (y_i - mu_y)^2))
    where mu_x = sum(w_i * x_i) / sum(w_i), mu_y = sum(w_i * y_i) / sum(w_i)
    
    Args:
        x: List of x values
        y: List of y values
        w: List of weights (must be same length as x and y)
        
    Returns:
        float: Weighted correlation coefficient, or None if undefined
    """
    if len(x) != len(y) or len(x) != len(w):
        return None
    
    if len(x) < 2:
        return None  # Need at least 2 points for correlation
    
    # Filter out invalid values
    valid_indices = [i for i in range(len(x)) 
                     if x[i] is not None and y[i] is not None and w[i] is not None
                     and not (math.isnan(x[i]) or math.isnan(y[i]) or math.isnan(w[i]))
                     and not (math.isinf(x[i]) or math.isinf(y[i]) or math.isinf(w[i]))
                     and w[i] > 0]
    
    if len(valid_indices) < 2:
        return None
    
    x_vals = [x[i] for i in valid_indices]
    y_vals = [y[i] for i in valid_indices]
    w_vals = [w[i] for i in valid_indices]
    
    # Compute weighted means
    total_weight = sum(w_vals)
    if total_weight <= 0:
        return None
    
    mu_x = sum(x_vals[i] * w_vals[i] for i in range(len(x_vals))) / total_weight
    mu_y = sum(y_vals[i] * w_vals[i] for i in range(len(y_vals))) / total_weight
    
    # Compute weighted covariance and variances
    cov_xy = sum(w_vals[i] * (x_vals[i] - mu_x) * (y_vals[i] - mu_y) for i in range(len(x_vals)))
    var_x = sum(w_vals[i] * (x_vals[i] - mu_x)**2 for i in range(len(x_vals)))
    var_y = sum(w_vals[i] * (y_vals[i] - mu_y)**2 for i in range(len(y_vals)))
    
    # Check for zero variance (all x or all y are identical)
    if var_x <= 0 or var_y <= 0:
        return 0.0  # No correlation if no variance
    
    # Compute correlation
    r = cov_xy / math.sqrt(var_x * var_y)
    
    # Clamp to valid range
    r = max(-1.0, min(1.0, r))
    return float(r)


def weighted_linreg(x: list, y: list, w: list) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute weighted least squares regression: y = a*x + b.
    
    Args:
        x: List of x values (independent variable)
        y: List of y values (dependent variable)
        w: List of weights (must be same length as x and y)
        
    Returns:
        tuple: (slope a, intercept b), or (None, None) if regression fails
    """
    if len(x) != len(y) or len(x) != len(w):
        return None, None
    
    if len(x) < 2:
        return None, None  # Need at least 2 points for regression
    
    # Filter out invalid values
    valid_indices = [i for i in range(len(x)) 
                     if x[i] is not None and y[i] is not None and w[i] is not None
                     and not (math.isnan(x[i]) or math.isnan(y[i]) or math.isnan(w[i]))
                     and not (math.isinf(x[i]) or math.isinf(y[i]) or math.isinf(w[i]))
                     and w[i] > 0]
    
    if len(valid_indices) < 2:
        return None, None
    
    x_vals = [x[i] for i in valid_indices]
    y_vals = [y[i] for i in valid_indices]
    w_vals = [w[i] for i in valid_indices]
    
    # Compute weighted means
    total_weight = sum(w_vals)
    if total_weight <= 0:
        return None, None
    
    mu_x = sum(x_vals[i] * w_vals[i] for i in range(len(x_vals))) / total_weight
    mu_y = sum(y_vals[i] * w_vals[i] for i in range(len(y_vals))) / total_weight
    
    # Compute weighted covariance and variance
    cov_xy = sum(w_vals[i] * (x_vals[i] - mu_x) * (y_vals[i] - mu_y) for i in range(len(x_vals)))
    var_x = sum(w_vals[i] * (x_vals[i] - mu_x)**2 for i in range(len(x_vals)))
    
    # Check for zero variance in x
    if var_x <= 0:
        return None, None  # Cannot compute slope if x has no variance
    
    # Compute slope: a = cov_xy / var_x
    a = cov_xy / var_x
    
    # Compute intercept: b = mu_y - a * mu_x
    b = mu_y - a * mu_x
    
    return float(a), float(b)


def weighted_ccc(x: list, y: list, w: list) -> Optional[float]:
    """
    Compute weighted Lin's Concordance Correlation Coefficient (CCC) between x and y.
    
    Lin's CCC measures agreement between two measurements, accounting for both
    correlation and accuracy (bias). It ranges from -1 to 1, where 1 indicates
    perfect agreement.
    
    Formula:
    ρ_c = (2 * cov(x,y)) / (var(x) + var(y) + (μ_x - μ_y)²)
    
    where μ, var, cov are weighted using the same filtering rules as weighted_corr.
    
    Args:
        x: List of x values (e.g., human effect sizes)
        y: List of y values (e.g., agent effect sizes)
        w: List of weights (must be same length as x and y)
        
    Returns:
        float: Weighted CCC, or None if undefined
    """
    if len(x) != len(y) or len(x) != len(w):
        return None
    
    if len(x) < 2:
        return None  # Need at least 2 points for CCC
    
    # Filter out invalid values (same logic as weighted_corr)
    valid_indices = [i for i in range(len(x)) 
                     if x[i] is not None and y[i] is not None and w[i] is not None
                     and not (math.isnan(x[i]) or math.isnan(y[i]) or math.isnan(w[i]))
                     and not (math.isinf(x[i]) or math.isinf(y[i]) or math.isinf(w[i]))
                     and w[i] > 0]
    
    if len(valid_indices) < 2:
        return None
    
    x_vals = [x[i] for i in valid_indices]
    y_vals = [y[i] for i in valid_indices]
    w_vals = [w[i] for i in valid_indices]
    
    # Compute weighted means
    total_weight = sum(w_vals)
    if total_weight <= 0:
        return None
    
    mu_x = sum(x_vals[i] * w_vals[i] for i in range(len(x_vals))) / total_weight
    mu_y = sum(y_vals[i] * w_vals[i] for i in range(len(y_vals))) / total_weight
    
    # Compute weighted covariance and variances
    cov_xy = sum(w_vals[i] * (x_vals[i] - mu_x) * (y_vals[i] - mu_y) for i in range(len(x_vals)))
    var_x = sum(w_vals[i] * (x_vals[i] - mu_x)**2 for i in range(len(x_vals)))
    var_y = sum(w_vals[i] * (y_vals[i] - mu_y)**2 for i in range(len(y_vals)))
    
    # Check for zero variance (all x or all y are identical)
    if var_x <= 0 and var_y <= 0:
        return 1.0  # Perfect agreement if both have no variance
    if var_x <= 0 or var_y <= 0:
        return 0.0  # No agreement if one has no variance
    
    # Compute CCC: ρ_c = (2 * cov) / (var_x + var_y + (μ_x - μ_y)²)
    mean_diff_sq = (mu_x - mu_y) ** 2
    denominator = var_x + var_y + mean_diff_sq
    
    if denominator <= 0:
        return 0.0
    
    ccc = (2.0 * cov_xy) / denominator
    
    # Clamp to valid range
    ccc = max(-1.0, min(1.0, ccc))
    return float(ccc)


def compute_ecs_corr(
    test_results: list,
    study_groups: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Compute ECS (Effect Consistency Score) at multiple levels using both CCC and Pearson correlation.
    
    Main ECS metric: Lin's Concordance Correlation Coefficient (CCC) - measures agreement accounting
    for both correlation and accuracy (bias).
    
    Retained metric: Pearson correlation (r) - measures linear association only.
    Caricature regression (slope a, intercept b) is computed using Pearson-based weighted regression.
    
    Args:
        test_results: List of test result dicts (must have 'agent_effect_d' and 'human_effect_d')
        study_groups: Optional dict mapping domain names to study IDs (e.g., {"Cognition": ["study_001", ...]})
        
    Returns:
        dict: {
            # Main ECS (CCC-based)
            'ecs_overall': float,  # CCC overall
            'ecs_domain': Dict[str, float],  # CCC per domain
            'ecs_per_study': Dict[str, float],  # CCC per study
            
            # Retained Pearson correlation (for figures/appendix)
            'ecs_corr_overall': float,  # Pearson r overall
            'ecs_corr_domain': Dict[str, float],  # Pearson r per domain
            'ecs_corr_per_study': Dict[str, float],  # Pearson r per study
            
            # Caricature regression (Pearson-based)
            'caricature_overall': {'a': float, 'b': float},
            'caricature_domain': Dict[str, {'a': float, 'b': float}],
            'caricature_per_study': Dict[str, {'a': float, 'b': float}],
            
            'n_tests_overall': int,
            'n_tests_per_study': Dict[str, int]
        }
    """
    result = {
        # Main ECS (CCC)
        'ecs_overall': None,
        'ecs_domain': {},
        'ecs_per_study': {},
        
        # Retained Pearson correlation
        'ecs_corr_overall': None,
        'ecs_corr_domain': {},
        'ecs_corr_per_study': {},
        
        # Caricature regression (Pearson-based)
        'caricature_overall': {'a': None, 'b': None},
        'caricature_domain': {},
        'caricature_per_study': {},
        
        'n_tests_overall': 0,
        'n_tests_per_study': {}
    }
    
    if not test_results:
        return result
    
    # Group tests by study_id
    from collections import defaultdict
    study_tests = defaultdict(list)
    for test in test_results:
        study_id = test.get('study_id')
        if study_id:
            study_tests[study_id].append(test)
    
    # Collect all valid test points for overall correlation
    all_x = []  # human_effect_d
    all_y = []  # agent_effect_d
    all_w = []  # weights (1/N_tests per study)
    
    # Also collect per-study and per-domain points
    study_x = defaultdict(list)
    study_y = defaultdict(list)
    study_w = defaultdict(list)
    
    domain_x = defaultdict(list)
    domain_y = defaultdict(list)
    domain_w = defaultdict(list)
    
    # First pass: count findings and tests per study, collect VALID points
    # Two-level weighting: Study → Finding → Test
    # Each finding within a study gets equal weight, then each test within a finding gets equal weight
    study_finding_tests = defaultdict(lambda: defaultdict(list))  # {study_id: {finding_id: [tests...]}}
    study_valid_points_with_finding = defaultdict(list)  # {study_id: [(h_d, a_d, finding_id), ...]}
    
    for study_id, tests in study_tests.items():
        for test in tests:
            finding_id = test.get('finding_id', 'unknown')
            
            # Try to get d-equivalent directly first
            h_d = test.get('human_effect_d')
            a_d = test.get('agent_effect_d')
            
            # If not available, convert from effect_size using test type (to reduce missing data)
            if h_d is None:
                es_h_raw = test.get('human_effect_size')
                if es_h_raw is not None:
                    test_type = test.get('statistical_test_type', 't-test')
                    h_d = FrequentistConsistency.effect_to_d_equiv(test_type, es_h_raw)
            
            if a_d is None:
                es_a_raw = test.get('agent_effect_size')
                if es_a_raw is not None:
                    test_type = test.get('statistical_test_type', 't-test')
                    a_d = FrequentistConsistency.effect_to_d_equiv(test_type, es_a_raw)
            
            # Track all tests by finding (for weight calculation)
            study_finding_tests[study_id][finding_id].append(test)
            
            # Include if both are available and valid
            if h_d is not None and a_d is not None:
                if not (math.isnan(h_d) or math.isnan(a_d) or math.isinf(h_d) or math.isinf(a_d)):
                    study_valid_points_with_finding[study_id].append((h_d, a_d, finding_id))
    
    # Second pass: compute correct two-level weights
    # weight = (1 / n_findings_in_study) × (1 / n_tests_in_finding)
    for study_id, points in study_valid_points_with_finding.items():
        n_valid = len(points)
        result['n_tests_per_study'][study_id] = n_valid  # Store valid count for reporting
        
        if n_valid == 0:
            continue
        
        # Count findings in this study
        n_findings = len(study_finding_tests[study_id])
        
        # Count tests per finding (total, not just valid)
        finding_total_tests = {}
        for finding_id, tests in study_finding_tests[study_id].items():
            finding_total_tests[finding_id] = len(tests)
        
        for h_d, a_d, finding_id in points:
            # Correct two-level weight:
            # weight = (1 / n_findings) × (1 / n_tests_in_finding)
            n_tests_in_finding = finding_total_tests.get(finding_id, 1)
            weight_per_test = (1.0 / n_findings) * (1.0 / n_tests_in_finding)
            
            # Overall
            all_x.append(h_d)
            all_y.append(a_d)
            all_w.append(weight_per_test)
            
            # Per-study
            study_x[study_id].append(h_d)
            study_y[study_id].append(a_d)
            study_w[study_id].append(1.0)  # Equal weight within study
            
            # Per-domain (if study_groups provided)
            if study_groups:
                for domain, study_list in study_groups.items():
                    if study_id in study_list:
                        domain_x[domain].append(h_d)
                        domain_y[domain].append(a_d)
                        domain_w[domain].append(weight_per_test)
                        break
    
    result['n_tests_overall'] = len(all_x)
    
    # Compute overall correlation and regression
    # Require at least 3 data points for meaningful correlation (with 2 points, correlation is always perfect)
    if len(all_x) >= 3:
        # Compute CCC (main ECS metric)
        ccc_overall = weighted_ccc(all_x, all_y, all_w)
        result['ecs_overall'] = ccc_overall
        
        # Compute Pearson correlation (retained for figures/appendix)
        r_overall = weighted_corr(all_x, all_y, all_w)
        result['ecs_corr_overall'] = r_overall
        
        # Compute caricature regression (Pearson-based)
        a_overall, b_overall = weighted_linreg(all_x, all_y, all_w)
        result['caricature_overall'] = {'a': a_overall, 'b': b_overall}
    
    # Compute per-study correlation and regression
    for study_id in study_tests.keys():
        if len(study_x[study_id]) >= 3:
            # Compute CCC (main ECS metric)
            ccc_study = weighted_ccc(study_x[study_id], study_y[study_id], study_w[study_id])
            result['ecs_per_study'][study_id] = ccc_study
            
            # Compute Pearson correlation (retained for figures/appendix)
            r_study = weighted_corr(study_x[study_id], study_y[study_id], study_w[study_id])
            result['ecs_corr_per_study'][study_id] = r_study
            
            # Compute caricature regression (Pearson-based)
            a_study, b_study = weighted_linreg(study_x[study_id], study_y[study_id], study_w[study_id])
            result['caricature_per_study'][study_id] = {'a': a_study, 'b': b_study}
        else:
            # Study has < 3 tests: correlation undefined (insufficient data)
            result['ecs_per_study'][study_id] = None
            result['ecs_corr_per_study'][study_id] = None
            result['caricature_per_study'][study_id] = {'a': None, 'b': None}
    
    # Compute per-domain correlation and regression
    if study_groups:
        for domain in study_groups.keys():
            if len(domain_x[domain]) >= 3:
                # Compute CCC (main ECS metric)
                ccc_domain = weighted_ccc(domain_x[domain], domain_y[domain], domain_w[domain])
                result['ecs_domain'][domain] = ccc_domain
                
                # Compute Pearson correlation (retained for figures/appendix)
                r_domain = weighted_corr(domain_x[domain], domain_y[domain], domain_w[domain])
                result['ecs_corr_domain'][domain] = r_domain
                
                # Compute caricature regression (Pearson-based)
                a_domain, b_domain = weighted_linreg(domain_x[domain], domain_y[domain], domain_w[domain])
                result['caricature_domain'][domain] = {'a': a_domain, 'b': b_domain}
            else:
                # Domain has < 3 tests: correlation undefined (insufficient data)
                result['ecs_domain'][domain] = None
                result['ecs_corr_domain'][domain] = None
                result['caricature_domain'][domain] = {'a': None, 'b': None}
    
    return result


def aggregate_study_pas(test_results: list) -> Tuple[float, float, dict]:
    """
    Aggregate PAS (raw and normalized) at study level.
    
    Process:
    1. Group tests by finding_id
    2. For each finding: compute z_avg = (1/N_test) * sum(z_test) (unweighted average of z)
    3. Average z_avg across findings, then convert to PAS
    
    Args:
        test_results: List of all test result dicts for a study
        
    Returns:
        tuple: (study_pas_raw, study_pas_norm, per_finding_breakdown)
            - study_pas_raw: Study-level raw PAS (average z per finding, then convert to PAS)
            - study_pas_norm: Study-level normalized PAS (simple mean of finding-level PAS)
            - per_finding_breakdown: Dict mapping finding_id -> {"pas_raw": float, "pas_norm": float}
    """
    if not test_results:
        return 0.5, 0.0, {}
    
    from collections import defaultdict
    finding_tests = defaultdict(list)
    
    for test in test_results:
        fid = test.get("finding_id", "default")
        finding_tests[fid].append(test)
    
    finding_scores = {}
    finding_z_avgs = []  # Store z_avg for each finding
    
    for fid, tests in finding_tests.items():
        # Compute z_avg = (1/N_test) * sum(z_test) for this finding
        z_values = []
        for test in tests:
            pas = test.get("pas") or test.get("score", 0.5)
            r = 2.0 * float(pas) - 1.0  # Convert to [-1, 1]
            z = fisher_z_transform(r, clamp=True)
            if not (math.isinf(z) or math.isnan(z)):
                z_values.append(z)
        
        if z_values:
            z_avg_finding = float(np.mean(z_values))  # 1/N * sum(z)
            finding_z_avgs.append(z_avg_finding)
            # Convert back to PAS for breakdown
            r_finding = fisher_z_inverse(z_avg_finding)
            pas_raw = (r_finding + 1.0) / 2.0
        else:
            # Fallback: simple mean of PAS
            pas_list = [t.get("pas") or t.get("score", 0.5) for t in tests]
            pas_raw = float(np.mean(pas_list))
            z_avg_finding = fisher_z_transform(2.0 * pas_raw - 1.0, clamp=True)
            finding_z_avgs.append(z_avg_finding)
        
        pas_norm = aggregate_finding_pas_norm(tests)
        finding_scores[fid] = {
            "pas_raw": pas_raw,
            "pas_norm": pas_norm
        }
    
    # Average z_avg across findings, then convert to PAS
    if finding_z_avgs:
        z_study = float(np.mean(finding_z_avgs))
        r_study = fisher_z_inverse(z_study)
        study_pas_raw = (r_study + 1.0) / 2.0
    else:
        study_pas_raw = 0.5
    
    # For normalized PAS, still use simple mean of finding-level PAS
    if finding_scores:
        study_pas_norm = float(np.mean([f["pas_norm"] for f in finding_scores.values()]))
    else:
        study_pas_norm = 0.0
    
    return study_pas_raw, study_pas_norm, finding_scores


def aggregate_study_pas_mean_only(test_results: list) -> float:
    """
    PAS with mean at finding and study level; only test level keeps its special (PAS per test) definition.

    - Test level: unchanged (PAS or score per test).
    - Finding level: mean of test-level PAS in that finding.
    - Study level: mean of finding-level means.

    Returns a single study-level PAS in [0, 1]. No Fisher-z; avoids sensitivity to 0/1 extremes.
    """
    if not test_results:
        return 0.5
    from collections import defaultdict
    finding_tests = defaultdict(list)
    for t in test_results:
        fid = t.get("finding_id", "default")
        finding_tests[fid].append(t)
    finding_means = []
    for tests in finding_tests.values():
        vals = []
        for t in tests:
            b = t.get("pas") or t.get("score")
            if b is not None and not (math.isnan(b) or math.isinf(b)):
                v = float(b)
                if 0 <= v <= 1:
                    vals.append(v)
        if vals:
            finding_means.append(float(np.mean(vals)))
    if not finding_means:
        return 0.5
    return float(np.mean(finding_means))
