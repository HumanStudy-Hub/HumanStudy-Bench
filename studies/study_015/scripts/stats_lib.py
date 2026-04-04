"""
Parse p-value from reported_statistics strings.
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
      - "r = 0.42, p < .001"

    Returns:
        tuple: (p_value, is_significant, confidence)
    """
    if significance_level is None:
        significance_level = 0.05

    if not reported_statistics:
        return None, False, "low"

    text = reported_statistics.lower()

    exact_match = re.search(r"p\s*[=:]\s*([0-9.]+)", text)
    if exact_match:
        p_val = float(exact_match.group(1))
        is_sig = p_val <= significance_level if significance_level is not None else False
        return p_val, is_sig, "high"

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

    r_match = re.search(r"r\s*=\s*([-0-9.]+)", text)
    if r_match:
        r_val = abs(float(r_match.group(1)))
        if r_val > 0.3:
            return 0.025, True, "low"
        return 0.10, False, "low"

    return None, False, "low"
