"""
Base Filter Class
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional


class BaseFilter(ABC):
    """Base class for all filters"""

    def __init__(self, llm_client):
        """
        Initialize filter.

        Args:
            llm_client: LLM client instance (GeminiClient)
        """
        self.client = llm_client

    @abstractmethod
    def process(
        self,
        pdf_path: Path,
        regeneration_instructions: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process PDF and return filter results.

        Args:
            pdf_path: Path to PDF file
            regeneration_instructions: Optional verifier feedback from a prior
                extraction attempt.
            kwargs: Stage-specific parser, cache, timeout, and worker controls.

        Returns:
            Dictionary with filter results
        """
        raise NotImplementedError
