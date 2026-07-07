"""
Base Extractor Class
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class BaseExtractor(ABC):
    """Base class for all extractors"""

    def __init__(self, llm_client):
        """
        Initialize extractor.

        Args:
            llm_client: LLM client instance (GeminiClient)
        """
        self.client = llm_client

    @abstractmethod
    def process(self, stage1_json: Dict[str, Any], pdf_path: Path) -> Dict[str, Any]:
        """
        Process stage1 results and extract study data.

        Args:
            stage1_json: Results from stage1 filter
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted data
        """
        raise NotImplementedError
