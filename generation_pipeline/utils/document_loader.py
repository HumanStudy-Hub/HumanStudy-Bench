"""
Document loader shared by the generation pipeline.
Only the methods used by the filter / extractor.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


class DocumentLoader:
    """Loads PDF / JSON / markdown for the extraction pipeline."""

    @staticmethod
    def get_pdf_pages(pdf_path: Path) -> List[int]:
        """Return list of 1-indexed page numbers in the PDF."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except ImportError:
                raise ImportError("Install pypdf: pip install pypdf")
        reader = PdfReader(str(pdf_path))
        return list(range(1, len(reader.pages) + 1))

    @staticmethod
    def load_json(json_path: Path) -> Dict[str, Any]:
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        return json.loads(json_path.read_text(encoding="utf-8"))

    @staticmethod
    def load_markdown(md_path: Path) -> str:
        md_path = Path(md_path)
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
        return md_path.read_text(encoding="utf-8")
