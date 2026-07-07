"""
Base agent class for HumanStudyBench.

All agents must inherit from this class and implement the run_study method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """Abstract base class for all agents in HumanStudyBench."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def run_study(self, specification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a study based on the specification.

        Args:
            specification: Study specification dictionary containing:
                - design: Experimental design (IVs, DVs, etc.)
                - participants: Participant information
                - procedure: Experimental procedure
                - materials: Study materials

        Returns:
            Dictionary containing results in standardized format:
            {
                "descriptive_statistics": {
                    "variable_name": {
                        "condition_1": {"mean": float, "sd": float, "n": int},
                        "condition_2": {"mean": float, "sd": float, "n": int}
                    }
                },
                "inferential_statistics": {
                    "test_name": {
                        "test": str,  # e.g., "t_test", "ANOVA"
                        "statistic": float,  # t, F, etc.
                        "p_value": float,
                        "df": int | List[int],
                        "effect_size": str,  # e.g., "cohens_d"
                        "effect_size_value": float
                    }
                },
                "raw_data": List[Dict]  # Optional: trial-by-trial data
            }

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Agents must implement run_study method")

    def reset(self) -> None:
        """
        Reset agent state between studies.

        Override this method if your agent maintains state that should be
        reset between different studies.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
