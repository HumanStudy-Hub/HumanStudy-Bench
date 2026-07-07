"""
Metrics calculator for statistical computations.
"""

import numpy as np
from typing import Tuple
from scipy import stats


class MetricsCalculator:
    """Calculate statistical metrics and effect sizes."""

    @staticmethod
    def cohens_d(mean1: float, mean2: float, sd1: float, sd2: float,
                 n1: int, n2: int) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            mean1, mean2: Means of the two groups
            sd1, sd2: Standard deviations
            n1, n2: Sample sizes

        Returns:
            Cohen's d
        """
        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
        return (mean1 - mean2) / pooled_sd if pooled_sd > 0 else 0.0

    @staticmethod
    def eta_squared(f_stat: float, df_effect: int, df_error: int) -> float:
        """
        Calculate eta-squared effect size from F-statistic.

        Args:
            f_stat: F-statistic
            df_effect: Degrees of freedom for effect
            df_error: Degrees of freedom for error

        Returns:
            Eta-squared
        """
        ss_effect = f_stat * df_effect
        ss_total = ss_effect + df_error
        return ss_effect / ss_total if ss_total > 0 else 0.0

    @staticmethod
    def confidence_interval(mean: float, se: float, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval.

        Args:
            mean: Mean value
            se: Standard error
            confidence: Confidence level (default: 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * se
        return (mean - margin, mean + margin)

    @staticmethod
    def relative_error(observed: float, expected: float) -> float:
        """
        Calculate relative error.

        Args:
            observed: Observed value
            expected: Expected value

        Returns:
            Relative error (as proportion)
        """
        if expected == 0:
            return abs(observed) if observed != 0 else 0.0
        return abs(observed - expected) / abs(expected)

    @staticmethod
    def standard_error(sd: float, n: int) -> float:
        """Calculate standard error."""
        return sd / np.sqrt(n) if n > 0 else 0.0
