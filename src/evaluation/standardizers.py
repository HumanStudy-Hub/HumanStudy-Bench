"""
Standardizers for converting different data types to standardized effect sizes.

All standardizers convert to "standardized distance in standard deviation units",
enabling comparison across different measurement types.
"""

import numpy as np
from typing import Dict, Any, Tuple


class BaseStandardizer:
    """Base class for all standardizers."""

    def compute(self, agent_data: Dict, human_data: Dict) -> Tuple[float, Dict]:
        """
        Compute standardized effect size.

        Args:
            agent_data: Agent statistics
            human_data: Human baseline statistics

        Returns:
            d: Standardized effect size (distance in SD units)
            details: Computation details for reporting
        """
        raise NotImplementedError


class ProportionStandardizer(BaseStandardizer):
    """
    Standardizer for proportion data (binary choices).

    Uses Freeman-Tukey arcsine transformation for variance stabilization,
    then computes standardized distance.

    Formula:
        θ = arcsin(√p)
        SE = 1/(2√n)
        d = |θ_agent - θ_human| / SE_pooled
    """

    def compute(self, agent_data: Dict, human_data: Dict) -> Tuple[float, Dict]:
        """
        Args:
            agent_data: {"proportion": float, "n": int}
            human_data: {"proportion": float, "n": int}
        """
        # Extract data
        p_agent = agent_data["proportion"]
        n_agent = agent_data["n"]
        p_human = human_data["proportion"]
        n_human = human_data["n"]

        # Freeman-Tukey transformation
        theta_agent = np.arcsin(np.sqrt(p_agent))
        theta_human = np.arcsin(np.sqrt(p_human))

        # Standard errors
        se_agent = 1.0 / (2.0 * np.sqrt(n_agent))
        se_human = 1.0 / (2.0 * np.sqrt(n_human))
        se_pooled = np.sqrt(se_agent**2 + se_human**2)

        # Standardized effect size
        d = abs(theta_agent - theta_human) / se_pooled

        details = {
            "method": "Freeman-Tukey",
            "p_agent": p_agent,
            "p_human": p_human,
            "theta_agent": theta_agent,
            "theta_human": theta_human,
            "se_pooled": se_pooled,
            "n_agent": n_agent,
            "n_human": n_human
        }

        return d, details


class RatingStandardizer(BaseStandardizer):
    """
    Standardizer for rating/scale data (continuous).

    Uses Cohen's d: standardized mean difference.

    Formula:
        SD_pooled = √[((n₁-1)·SD₁² + (n₂-1)·SD₂²) / (n₁+n₂-2)]
        d = |M_agent - M_human| / SD_pooled
    """

    def compute(self, agent_data: Dict, human_data: Dict) -> Tuple[float, Dict]:
        """
        Args:
            agent_data: {"mean": float, "sd": float, "n": int}
            human_data: {"mean": float, "sd": float, "n": int}
        """
        # Extract data
        m_agent = agent_data["mean"]
        sd_agent = agent_data["sd"]
        n_agent = agent_data["n"]
        m_human = human_data["mean"]
        sd_human = human_data["sd"]
        n_human = human_data["n"]

        # Pooled standard deviation
        sd_pooled = np.sqrt(
            ((n_agent - 1) * sd_agent**2 + (n_human - 1) * sd_human**2) /
            (n_agent + n_human - 2)
        )

        # Cohen's d
        d = abs(m_agent - m_human) / sd_pooled

        details = {
            "method": "Cohen's d",
            "m_agent": m_agent,
            "m_human": m_human,
            "sd_agent": sd_agent,
            "sd_human": sd_human,
            "sd_pooled": sd_pooled,
            "n_agent": n_agent,
            "n_human": n_human
        }

        return d, details


class EffectSizeStandardizer(BaseStandardizer):
    """
    Standardizer for comparing effect sizes directly.

    Computes standardized distance between two effect size estimates.

    Formula:
        d = |ES_agent - ES_human| / SE_combined
    """

    def compute(self, agent_data: Dict, human_data: Dict) -> Tuple[float, Dict]:
        """
        Args:
            agent_data: {"effect_size": float, "se": float}
            human_data: {"effect_size": float, "se": float}
        """
        # Extract data
        es_agent = agent_data["effect_size"]
        se_agent = agent_data.get("se", 0)
        es_human = human_data["effect_size"]
        se_human = human_data.get("se", 0)

        # Combined SE
        se_combined = np.sqrt(se_agent**2 + se_human**2)

        # Avoid division by zero
        if se_combined == 0:
            # If no SE provided, use absolute difference scaled by typical SE
            se_combined = 0.1  # Assume 10% SE as reasonable default

        # Standardized difference
        d = abs(es_agent - es_human) / se_combined

        details = {
            "method": "Direct Effect Size",
            "es_agent": es_agent,
            "es_human": es_human,
            "se_agent": se_agent,
            "se_human": se_human,
            "se_combined": se_combined
        }

        return d, details


class StandardizerRegistry:
    """Registry of standardizers for different data types."""

    _registry = {
        "proportion": ProportionStandardizer,
        "rating": RatingStandardizer,
        "effect_size": EffectSizeStandardizer
    }

    @classmethod
    def get(cls, data_type: str) -> BaseStandardizer:
        """Get standardizer for given data type."""
        if data_type not in cls._registry:
            raise ValueError(f"Unknown data type: {data_type}. "
                           f"Available: {list(cls._registry.keys())}")
        return cls._registry[data_type]()

    @classmethod
    def register(cls, data_type: str, standardizer_class: type):
        """Register a new standardizer."""
        cls._registry[data_type] = standardizer_class
