"""
Minimal stats utilities for evaluator.
Parse p-value from reported_statistics strings; no BF/PAS.
"""

import re
from typing import Tuple, Optional


def parse_p_value_from_reported(
    reported_statistics: str, significance_level: float = 0.05
) -> Tuple[Optional[float], bool, str]:
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
    if significance_level is None:
        significance_level = 0.05

    if not reported_statistics:
        return None, False, "low"

    text = reported_statistics.lower()

    # Try to extract exact p-value: "p = 0.023" or "p=0.023"
    exact_match = re.search(r"p\s*[=:]\s*([0-9.]+)", text)
    if exact_match:
        p_val = float(exact_match.group(1))
        is_sig = p_val <= significance_level if significance_level is not None else False
        return p_val, is_sig, "high"

    # Try to extract inequality: "p < .001" or "p < 0.05"
    inequality_match = re.search(r"p\s*[<>]\s*([0-9.]+)", text)
    if inequality_match:
        threshold = float(inequality_match.group(1))
        is_less_than = "<" in text[
            inequality_match.start() : inequality_match.end()
        ]

        if is_less_than:
            p_val = threshold / 2.0
            is_sig = True
            return p_val, is_sig, "medium"
        else:
            p_val = threshold
            is_sig = threshold < significance_level if significance_level is not None else False
            return p_val, is_sig, "medium"

    # If no p-value found, infer from test statistic
    t_match = re.search(r"t\s*\([^)]+\)\s*=\s*([0-9.]+)", text)
    if t_match:
        t_val = abs(float(t_match.group(1)))
        if t_val > 2.0:
            return 0.025, True, "low"
        return 0.10, False, "low"

    f_match = re.search(r"f\s*\([^)]+\)\s*=\s*([0-9.]+)", text)
    if f_match:
        f_val = float(f_match.group(1))
        if f_val > 4.0:
            return 0.025, True, "low"
        return 0.10, False, "low"

    r_match = re.search(r"r\s*=\s*([-0-9.]+)", text)
    if r_match:
        r_val = abs(float(r_match.group(1)))
        if r_val > 0.3:
            return 0.025, True, "low"
        return 0.10, False, "low"

    return None, False, "low"
