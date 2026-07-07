"""
PDF text extraction for multi-provider pipeline (no upload_file dependency).
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional


DEFAULT_OCR_MIN_CHARS = 800


class OCRUnavailableError(RuntimeError):
    """Raised internally when optional OCR dependencies are unavailable."""


def extract_pdf_text(
    pdf_path: Path,
    max_chars: Optional[int] = None,
    *,
    ocr_fallback: bool = True,
    ocr_min_chars: int = DEFAULT_OCR_MIN_CHARS,
) -> str:
    """
    Extract text from PDF with page markers.

    Uses pypdf first. If the extracted text is too short or appears garbled,
    tries optional OCR fallbacks (`ocrmypdf`, then `pytesseract` with PyMuPDF or
    pdf2image). If OCR dependencies are absent, returns the pypdf text and emits
    a warning rather than failing the pipeline.

    Args:
        pdf_path: Path to PDF file
        max_chars: If set, truncate to this many characters (for context limits)

    Returns:
        Text with "--- Page N ---" markers between pages
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pypdf_text = ""
    pypdf_error: Exception | None = None
    try:
        pypdf_text = _extract_with_pypdf(pdf_path, max_chars=max_chars)
    except ImportError as exc:
        pypdf_error = exc

    if pypdf_text and not _needs_ocr(pypdf_text, max_chars=max_chars, min_chars=ocr_min_chars):
        return pypdf_text

    if ocr_fallback:
        try:
            ocr_text = _extract_with_ocr(pdf_path, max_chars=max_chars)
            if ocr_text.strip():
                return ocr_text
        except OCRUnavailableError as exc:
            if pypdf_text:
                warnings.warn(
                    f"PDF text looked unreliable for {pdf_path}; OCR fallback unavailable: {exc}",
                    RuntimeWarning,
                )
                return pypdf_text
            if pypdf_error is not None:
                raise ImportError(
                    "pypdf is required for PDF extraction unless OCR dependencies are installed. "
                    "Install pypdf or install ocrmypdf/pytesseract OCR support."
                ) from pypdf_error
            raise
        except Exception as exc:
            if pypdf_text:
                warnings.warn(
                    f"PDF text looked unreliable for {pdf_path}; OCR fallback failed: {exc}",
                    RuntimeWarning,
                )
                return pypdf_text
            raise

    if pypdf_text:
        return pypdf_text
    if pypdf_error is not None:
        raise ImportError("pypdf required for PDF text extraction. Install with: pip install pypdf") from pypdf_error
    return ""


def _extract_with_pypdf(pdf_path: Path, max_chars: Optional[int] = None) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise ImportError("pypdf required for PDF text extraction. Install with: pip install pypdf") from exc

    reader = PdfReader(str(pdf_path))
    parts = []
    total = 0
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        block = f"--- Page {i} ---\n{text}"
        if max_chars and total + len(block) > max_chars:
            remaining = max_chars - total - 50
            if remaining > 0:
                block = block[:remaining] + "\n[... truncated]"
            parts.append(block)
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _needs_ocr(text: str, *, max_chars: Optional[int], min_chars: int) -> bool:
    content = text.replace("\n", " ")
    content = " ".join(part for part in content.split() if not part.startswith("---"))
    stripped = content.strip()
    if not stripped:
        return True
    threshold = min_chars if max_chars is None else min(min_chars, max_chars)
    if len(stripped) < threshold:
        return True

    replacement_ratio = stripped.count("\ufffd") / max(len(stripped), 1)
    control_count = sum(1 for ch in stripped if ord(ch) < 32 and ch not in "\n\t\r")
    control_ratio = control_count / max(len(stripped), 1)
    alpha_num_ratio = sum(1 for ch in stripped if ch.isalnum()) / max(len(stripped), 1)
    return replacement_ratio > 0.01 or control_ratio > 0.02 or alpha_num_ratio < 0.20


def _extract_with_ocr(pdf_path: Path, max_chars: Optional[int]) -> str:
    errors: list[str] = []
    try:
        return _extract_with_ocrmypdf(pdf_path, max_chars=max_chars)
    except OCRUnavailableError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"ocrmypdf failed: {exc}")

    try:
        return _extract_with_pytesseract(pdf_path, max_chars=max_chars)
    except OCRUnavailableError as exc:
        errors.append(str(exc))

    raise OCRUnavailableError("; ".join(errors) if errors else "no OCR backend available")


def _extract_with_ocrmypdf(pdf_path: Path, max_chars: Optional[int]) -> str:
    if shutil.which("ocrmypdf") is None:
        raise OCRUnavailableError("ocrmypdf command not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_pdf = Path(tmpdir) / "ocr.pdf"
        cmd = ["ocrmypdf", "--force-ocr", "--quiet", str(pdf_path), str(out_pdf)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return _extract_with_pypdf(out_pdf, max_chars=max_chars)


def _extract_with_pytesseract(pdf_path: Path, max_chars: Optional[int]) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise OCRUnavailableError("pytesseract package not installed") from exc

    try:
        return _extract_with_pymupdf(pdf_path, pytesseract, max_chars=max_chars)
    except OCRUnavailableError:
        return _extract_with_pdf2image(pdf_path, pytesseract, max_chars=max_chars)


def _extract_with_pymupdf(pdf_path: Path, pytesseract, max_chars: Optional[int]) -> str:
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError as exc:
        raise OCRUnavailableError("PyMuPDF/Pillow not installed for pytesseract PDF rendering") from exc

    doc = fitz.open(str(pdf_path))
    parts: list[str] = []
    total = 0
    for index, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image)
        block = f"--- Page {index} ---\n{text}"
        if max_chars and total + len(block) > max_chars:
            remaining = max_chars - total - 50
            if remaining > 0:
                block = block[:remaining] + "\n[... truncated]"
            parts.append(block)
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _extract_with_pdf2image(pdf_path: Path, pytesseract, max_chars: Optional[int]) -> str:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise OCRUnavailableError("pdf2image not installed for pytesseract PDF rendering") from exc

    parts: list[str] = []
    total = 0
    for index, image in enumerate(convert_from_path(str(pdf_path), dpi=200), 1):
        text = pytesseract.image_to_string(image)
        block = f"--- Page {index} ---\n{text}"
        if max_chars and total + len(block) > max_chars:
            remaining = max_chars - total - 50
            if remaining > 0:
                block = block[:remaining] + "\n[... truncated]"
            parts.append(block)
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
