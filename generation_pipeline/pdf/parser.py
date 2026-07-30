from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from generation_pipeline.pdf.models import DocumentBlock, ParsedPdfDocument
from generation_pipeline.utils.pdf_chunker import clean_text, split_pages
from generation_pipeline.utils.pdf_extractor import extract_pdf_text


PARSER_CACHE_VERSION = "pdf-parser-v8"
IMAGE_DOMINANT_MIN_CHARS = 1500
IMAGE_DOMINANT_CHARS_PER_PAGE = 250
_HEADING_RE = re.compile(
    r"^(?:study|experiment|appendix|supplement|method|methods|materials|procedure|measures|results|participants)\b",
    re.IGNORECASE,
)


def parse_pdf_document(
    pdf_path: Path,
    *,
    artifacts_dir: Optional[Path] = None,
    force: bool = False,
    prefer_docling: bool = True,
) -> ParsedPdfDocument:
    """Parse a PDF into a cached, page-grounded block representation.

    Docling is the primary parser. A pypdf block fallback is deliberately marked
    degraded so downstream readiness never confuses flat text with layout-aware
    extraction.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    source_hash = _sha256(pdf_path)
    cache_dir = Path(artifacts_dir) if artifacts_dir is not None else None
    if cache_dir is not None and not force:
        cached = _load_cache(cache_dir, source_hash)
        if cached is not None and not (prefer_docling and cached.degraded):
            cached.source_file = str(pdf_path)
            return cached

    baseline = _parse_with_pypdf(pdf_path, source_hash, warnings=[])
    errors: List[str] = []
    if prefer_docling:
        try:
            force_ocr = _needs_full_page_ocr(baseline)
            try:
                document, raw_document = _parse_with_docling(
                    pdf_path,
                    source_hash,
                    force_ocr=force_ocr,
                )
            except Exception as ocr_exc:
                if not force_ocr:
                    raise
                document, raw_document = _parse_with_docling(
                    pdf_path,
                    source_hash,
                    force_ocr=False,
                )
                document.warnings.append(
                    f"full_page_ocr_failed:{type(ocr_exc).__name__}:{ocr_exc};using_page_vision"
                )
            document = _merge_text_baseline(document, baseline)
            document = _prepare_image_dominant_document(document, baseline)
            document = _repair_systematic_text_layer_artifacts(document)
            document = _repair_ambiguous_numeric_ocr(
                document,
                pdf_path,
                artifacts_dir=cache_dir,
            )
            if cache_dir is not None:
                _write_artifacts(cache_dir, document, raw_document=raw_document)
            return document
        except Exception as exc:
            errors.append(f"docling_failed:{type(exc).__name__}:{exc}")

    document = _repair_systematic_text_layer_artifacts(baseline)
    document = _repair_ambiguous_numeric_ocr(
        document,
        pdf_path,
        artifacts_dir=cache_dir,
    )
    document.warnings.extend(errors)
    if cache_dir is not None:
        _write_artifacts(cache_dir, document, raw_document=None)
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_with_docling(
    pdf_path: Path,
    source_hash: str,
    *,
    force_ocr: bool = False,
) -> Tuple[ParsedPdfDocument, Dict[str, Any]]:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Docling is required for layout-aware PDF extraction. Install with `pip install docling`."
        ) from exc

    options = PdfPipelineOptions()
    options.do_ocr = force_ocr
    if force_ocr:
        options.ocr_options = RapidOcrOptions(
            lang=["english"],
            force_full_page_ocr=True,
            backend="torch",
        )
    options.do_table_structure = True
    local_models = Path.home() / ".cache" / "docling" / "models"
    if local_models.exists():
        options.artifacts_path = local_models
    options.generate_page_images = False
    options.generate_picture_images = False
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )
    result = converter.convert(pdf_path)
    doc = result.document
    raw_document = doc.export_to_dict()
    markdown = doc.export_to_markdown()
    blocks = _docling_blocks(doc)
    if not blocks:
        raise ValueError("Docling returned no document blocks")
    page_count = max((block.page_end for block in blocks), default=len(getattr(doc, "pages", {}) or {}))
    try:
        version = importlib.metadata.version("docling")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    warnings: List[str] = []
    status = str(getattr(result, "status", ""))
    if status and "success" not in status.lower():
        warnings.append(f"docling_conversion_status:{status}")
    return (
        ParsedPdfDocument(
            source_file=str(pdf_path),
            source_sha256=source_hash,
            parser="docling_ocr" if force_ocr else "docling",
            parser_version=version,
            page_count=page_count,
            blocks=blocks,
            markdown=markdown,
            degraded=False,
            warnings=[*warnings, "full_page_ocr_applied"] if force_ocr else warnings,
        ),
        raw_document,
    )


def _docling_blocks(doc: Any) -> List[DocumentBlock]:
    blocks: List[DocumentBlock] = []
    section_path: List[str] = []
    for order, pair in enumerate(doc.iterate_items()):
        item, level = pair
        label_obj = getattr(item, "label", "text")
        label = str(getattr(label_obj, "value", label_obj) or "text").lower()
        text = clean_text(str(getattr(item, "text", "") or ""))
        metadata: Dict[str, Any] = {}

        if label == "table" or item.__class__.__name__.lower().startswith("table"):
            try:
                frame = item.export_to_dataframe(doc=doc)
                text = frame.to_markdown(index=False)
                metadata["table"] = {
                    "columns": [str(column) for column in frame.columns],
                    "rows": frame.fillna("").astype(str).values.tolist(),
                    "html": frame.to_html(index=False),
                }
            except Exception as exc:
                metadata["table_error"] = f"{type(exc).__name__}: {exc}"

        if label in {"picture", "figure"} or item.__class__.__name__.lower().startswith("picture"):
            caption = ""
            try:
                caption = clean_text(str(doc.caption_text(item) or ""))
            except Exception:
                caption = ""
            if caption:
                text = caption
            metadata["visual_required"] = True

        if label in {"title", "section_header"} and text:
            heading_level = max(0, int(level or 0))
            section_path = section_path[:heading_level]
            section_path.append(text)

        provenance = list(getattr(item, "prov", []) or [])
        pages = [int(getattr(prov, "page_no", 1) or 1) for prov in provenance] or [1]
        bbox = _bbox_of(provenance[0]) if provenance else None
        if not text and label not in {"picture", "figure", "table"}:
            continue
        if not text:
            text = f"[{label.upper()} on page {min(pages)}; inspect page image]"
        block_id = f"p{min(pages):03d}_{label}_{order:05d}"
        blocks.append(
            DocumentBlock(
                block_id=block_id,
                order=order,
                page_start=min(pages),
                page_end=max(pages),
                block_type=label,
                text=text,
                section_path=list(section_path),
                bbox=bbox,
                metadata=metadata,
            )
        )
    return blocks


def _bbox_of(provenance: Any) -> Optional[List[float]]:
    bbox = getattr(provenance, "bbox", None)
    if bbox is None:
        return None
    values = []
    for key in ("l", "t", "r", "b"):
        value = getattr(bbox, key, None)
        if value is None:
            return None
        values.append(round(float(value), 3))
    return values


def _parse_with_pypdf(pdf_path: Path, source_hash: str, *, warnings: List[str]) -> ParsedPdfDocument:
    raw = extract_pdf_text(pdf_path, max_chars=None, ocr_fallback=False)
    pages = split_pages(raw)
    blocks: List[DocumentBlock] = []
    section_path: List[str] = []
    order = 0
    for page_no, page_text in pages:
        paragraphs = [clean_text(value) for value in re.split(r"\n\s*\n", page_text)]
        paragraphs = [value for value in paragraphs if value]
        if not paragraphs and clean_text(page_text):
            paragraphs = [clean_text(page_text)]
        for paragraph in paragraphs:
            block_type = "section_header" if len(paragraph) <= 120 and _HEADING_RE.match(paragraph) else "text"
            if block_type == "section_header":
                section_path = [paragraph]
            blocks.append(
                DocumentBlock(
                    block_id=f"p{page_no:03d}_{block_type}_{order:05d}",
                    order=order,
                    page_start=page_no,
                    page_end=page_no,
                    block_type=block_type,
                    text=paragraph,
                    section_path=list(section_path),
                )
            )
            order += 1
    return ParsedPdfDocument(
        source_file=str(pdf_path),
        source_sha256=source_hash,
        parser="pypdf_block_fallback",
        parser_version="1",
        page_count=max((page for page, _ in pages), default=0),
        blocks=blocks,
        markdown="\n\n".join(block.text for block in blocks),
        degraded=True,
        warnings=[*warnings, "layout_and_table_structure_unavailable"],
    )


def _merge_text_baseline(
    document: ParsedPdfDocument,
    baseline: ParsedPdfDocument,
) -> ParsedPdfDocument:
    """Keep Docling layout objects while recovering text its backend missed."""
    document.page_count = max(document.page_count, baseline.page_count)
    docling_text_chars = sum(
        len(block.text)
        for block in document.blocks
        if block.block_type not in {"picture", "figure"}
        and "inspect page image" not in block.text
    )
    baseline_chars = baseline.text_chars
    if baseline_chars <= 0 or docling_text_chars >= max(1000, int(baseline_chars * 0.65)):
        return document

    visual_blocks = [
        block
        for block in document.blocks
        if block.block_type in {"table", "picture", "figure"}
    ]
    merged = [*baseline.blocks, *visual_blocks]
    merged.sort(key=lambda block: (block.page_start, block.bbox[1] if block.bbox else 10**9, block.order))
    for order, block in enumerate(merged):
        block.order = order
    document.blocks = merged
    document.page_count = max(document.page_count, baseline.page_count)
    document.parser = "docling+pypdf"
    document.markdown = baseline.markdown + "\n\n## Docling visual/table blocks\n\n" + "\n\n".join(
        block.text for block in visual_blocks
    )
    document.warnings.append(
        f"docling_text_coverage_low:{docling_text_chars}/{baseline_chars};pypdf_text_blocks_merged"
    )
    return document


def _prepare_image_dominant_document(
    document: ParsedPdfDocument,
    baseline: ParsedPdfDocument,
) -> ParsedPdfDocument:
    """Expose every page as evidence when neither parser recovered usable text."""
    page_count = max(document.page_count, baseline.page_count)
    document.page_count = page_count
    substantive_chars = sum(
        len(block.text)
        for block in document.blocks
        if block.block_type not in {"picture", "figure", "page_image"}
        and "inspect page image" not in block.text.lower()
    )
    threshold = max(IMAGE_DOMINANT_MIN_CHARS, page_count * IMAGE_DOMINANT_CHARS_PER_PAGE)
    if document.parser == "docling_ocr":
        _add_full_page_evidence(document, page_count)
    if page_count <= 0 or substantive_chars >= threshold:
        return document

    _add_full_page_evidence(document, page_count)
    document.parser = "docling_vision"
    document.warnings.append(
        f"image_dominant_document:{substantive_chars}/{threshold};full_page_vision_required"
    )
    document.markdown = (
        document.markdown.rstrip()
        + "\n\n## Parser note\n\n"
        + "The PDF is image-dominant. Full rendered pages are authoritative evidence.\n"
    )
    return document


def _add_full_page_evidence(document: ParsedPdfDocument, page_count: int) -> None:
    existing = {block.block_id for block in document.blocks}
    next_order = max((block.order for block in document.blocks), default=-1) + 1
    for page_no in range(1, page_count + 1):
        block_id = f"p{page_no:03d}_page_image"
        if block_id in existing:
            continue
        document.blocks.append(
            DocumentBlock(
                block_id=block_id,
                order=next_order,
                page_start=page_no,
                page_end=page_no,
                block_type="page_image",
                text=f"[Full rendered page {page_no}; inspect the attached page image as primary evidence]",
                metadata={
                    "visual_required": True,
                    "synthetic_page_evidence": True,
                },
            )
        )
        next_order += 1


def _needs_full_page_ocr(document: ParsedPdfDocument) -> bool:
    threshold = max(
        IMAGE_DOMINANT_MIN_CHARS,
        document.page_count * IMAGE_DOMINANT_CHARS_PER_PAGE,
    )
    return document.page_count > 0 and document.text_chars < threshold


_BROKEN_N_BRACKET_RE = re.compile(
    r"(\[\s*N\s*=\s*)(\d{1,6})1(\s*[:.])",
    re.IGNORECASE,
)
_BROKEN_THIRDS_RE = re.compile(
    r"\b([12])13(\s+probabil(?:ity|ities|i-))",
    re.IGNORECASE,
)
_BROKEN_BRACKET_PERCENT_RE = re.compile(
    r"(?<!\[)\b1(\d{2})(\s+percent\])",
    re.IGNORECASE,
)
_BROKEN_BRACKET_CURRENCY_RE = re.compile(r"\[\$?I(\d{2,3})1(?=\s)")


def _repair_systematic_text_layer_artifacts(
    document: ParsedPdfDocument,
) -> ParsedPdfDocument:
    """Repair repeated impossible glyph substitutions in a PDF text layer.

    The trailing-N repair is enabled only when the same malformed bracket
    pattern occurs at least three times in one document. This avoids treating a
    single genuine sample size ending in 1 as a corrupted closing bracket.
    """
    broken_n_count = sum(len(_BROKEN_N_BRACKET_RE.findall(block.text)) for block in document.blocks)
    thirds_count = sum(len(_BROKEN_THIRDS_RE.findall(block.text)) for block in document.blocks)
    bracket_percent_count = sum(
        len(_BROKEN_BRACKET_PERCENT_RE.findall(block.text))
        for block in document.blocks
    )
    bracket_currency_count = sum(
        len(_BROKEN_BRACKET_CURRENCY_RE.findall(block.text))
        for block in document.blocks
    )
    if broken_n_count < 3 and thirds_count == 0 and bracket_currency_count == 0:
        return document

    def repair(text: str) -> str:
        if broken_n_count >= 3:
            text = _BROKEN_N_BRACKET_RE.sub(r"\1\2]\3", text)
            text = _BROKEN_BRACKET_PERCENT_RE.sub(r"[\1\2", text)
        text = _BROKEN_THIRDS_RE.sub(r"\1/3\2", text)
        return _BROKEN_BRACKET_CURRENCY_RE.sub(
            lambda match: f"[$1{match.group(1)}]",
            text,
        )

    for block in document.blocks:
        block.text = repair(block.text)
    document.markdown = repair(document.markdown)
    if broken_n_count >= 3:
        document.warnings.append(
            f"systematic_n_bracket_glyph_repaired:{broken_n_count}"
        )
    if thirds_count:
        document.warnings.append(f"systematic_fraction_glyph_repaired:{thirds_count}")
    if broken_n_count >= 3 and bracket_percent_count:
        document.warnings.append(
            f"systematic_bracket_percent_glyph_repaired:{bracket_percent_count}"
        )
    if bracket_currency_count:
        document.warnings.append(
            f"systematic_currency_bracket_glyph_repaired:{bracket_currency_count}"
        )
    return document


_AMBIGUOUS_DECIMAL_RE = re.compile(
    r"(?<![\w.])\.(?P<body>[OoIiLlSsBbGg][0-9OoIiLlSsBbGg]{0,2})(?!\w)"
)
_OCR_DECIMAL_RE = re.compile(r"(?<![\w.])(?:\.\d{1,3}|0\d{1,2})(?!\d)")
_AMBIGUOUS_STATISTIC_RE = re.compile(
    r"\b(?P<label>[tzrx])\s*=\s*(?P<number>\d{3,})"
    r"(?=\s*,?\s*p\s*[<=>])",
    re.IGNORECASE,
)
_OCR_STATISTIC_RE = re.compile(
    r"\b(?P<label>[tzrx])\s*=\s*(?P<number>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_CONTEXT_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _repair_ambiguous_numeric_ocr(
    document: ParsedPdfDocument,
    pdf_path: Path,
    *,
    artifacts_dir: Optional[Path],
) -> ParsedPdfDocument:
    suspicious = [block for block in document.blocks if _has_suspicious_numeric_ocr(block.text)]
    if not suspicious:
        return document
    tesseract = shutil.which("tesseract")
    if not tesseract:
        document.warnings.append(
            f"ambiguous_numeric_ocr_unresolved:{sum(_suspicious_numeric_count(block.text) for block in suspicious)}:tesseract_unavailable"
        )
        return document
    try:
        import pypdfium2 as pdfium
    except ImportError:
        document.warnings.append(
            f"ambiguous_numeric_ocr_unresolved:{sum(_suspicious_numeric_count(block.text) for block in suspicious)}:pypdfium2_unavailable"
        )
        return document

    pages = sorted(
        {
            page
            for block in suspicious
            for page in range(block.page_start, block.page_end + 1)
            if 1 <= page <= document.page_count
        }
    )
    if not pages:
        return document

    temporary: Optional[tempfile.TemporaryDirectory[str]] = None
    if artifacts_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="humanstudy_numeric_ocr_")
        image_dir = Path(temporary.name)
    else:
        image_dir = Path(artifacts_dir) / "page_images"
        image_dir.mkdir(parents=True, exist_ok=True)

    page_text: Dict[int, str] = {}
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            for page_no in pages:
                page = pdf[page_no - 1]
                bitmap = page.render(scale=3.0)
                image = bitmap.to_pil()
                image_path = image_dir / f"page_{page_no:03d}_numeric_ocr.png"
                try:
                    image.save(image_path)
                finally:
                    image.close()
                    bitmap.close()
                    page.close()
                completed = subprocess.run(
                    [tesseract, str(image_path), "stdout", "--psm", "3", "-l", "eng"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if completed.returncode == 0 and completed.stdout.strip():
                    page_text[page_no] = completed.stdout
        finally:
            pdf.close()
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        document.warnings.append(
            f"ambiguous_numeric_ocr_failed:{type(exc).__name__}:{exc}"
        )
    finally:
        if temporary is not None:
            temporary.cleanup()

    repair_count = 0
    unresolved_count = 0
    markdown_replacements: List[Tuple[str, str]] = []
    for block in suspicious:
        original_text = block.text
        repairs: List[Dict[str, Any]] = []
        for page_no in range(block.page_start, block.page_end + 1):
            if not _has_suspicious_numeric_ocr(block.text):
                break
            ocr_text = page_text.get(page_no)
            if not ocr_text:
                continue
            block.text, page_repairs = _repair_ambiguous_decimal_tokens(
                block.text,
                ocr_text,
            )
            block.text, statistic_repairs = _repair_ambiguous_statistic_tokens(
                block.text,
                ocr_text,
            )
            page_repairs.extend(statistic_repairs)
            for item in page_repairs:
                item["page"] = page_no
                item["engine"] = "tesseract_context_alignment"
            repairs.extend(page_repairs)
        remaining = [
            *[match.group(0) for match in _AMBIGUOUS_DECIMAL_RE.finditer(block.text)],
            *[match.group(0) for match in _AMBIGUOUS_STATISTIC_RE.finditer(block.text)],
        ]
        if repairs:
            block.metadata["numeric_ocr_repairs"] = repairs
            repair_count += len(repairs)
            markdown_replacements.extend(
                (str(item["original"]), str(item["replacement"])) for item in repairs
            )
        if remaining:
            block.metadata["numeric_ocr_ambiguities"] = remaining
            unresolved_count += len(remaining)
        if block.text != original_text:
            block.metadata["text_before_numeric_ocr_repair"] = original_text

    for original, replacement in markdown_replacements:
        document.markdown = document.markdown.replace(original, replacement, 1)
    if repair_count:
        document.warnings.append(f"ambiguous_numeric_ocr_repaired:{repair_count}")
    if unresolved_count:
        document.warnings.append(f"ambiguous_numeric_ocr_unresolved:{unresolved_count}")
    return document


def _repair_ambiguous_decimal_tokens(
    text: str,
    ocr_text: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    repairs: List[Dict[str, Any]] = []
    replacements: List[Tuple[int, int, str, float, str]] = []
    ocr_candidates = list(_OCR_DECIMAL_RE.finditer(ocr_text))
    for match in _AMBIGUOUS_DECIMAL_RE.finditer(text):
        left_words = _context_words(text[max(0, match.start() - 140) : match.start()])[-8:]
        right_words = _context_words(text[match.end() : match.end() + 140])[:8]
        scored_by_replacement: Dict[str, float] = {}
        for candidate in ocr_candidates:
            replacement = candidate.group(0)
            if replacement.startswith("0") and not replacement.startswith("0."):
                replacement = "." + replacement
            try:
                if not 0 <= float(replacement) <= 1:
                    continue
            except ValueError:
                continue
            candidate_left = _context_words(
                ocr_text[max(0, candidate.start() - 180) : candidate.start()]
            )[-8:]
            candidate_right = _context_words(
                ocr_text[candidate.end() : candidate.end() + 180]
            )[:8]
            score = _context_alignment_score(
                left_words,
                right_words,
                candidate_left,
                candidate_right,
            )
            scored_by_replacement[replacement] = max(
                score,
                scored_by_replacement.get(replacement, 0.0),
            )
        scored = sorted(
            ((score, replacement) for replacement, score in scored_by_replacement.items()),
            reverse=True,
        )
        if not scored:
            continue
        best_score, replacement = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < 0.68 or best_score - second_score < 0.08:
            continue
        replacements.append(
            (match.start(), match.end(), replacement, best_score, match.group(0))
        )

    for start, end, replacement, score, original in reversed(replacements):
        text = text[:start] + replacement + text[end:]
        repairs.append(
            {
                "original": original,
                "replacement": replacement,
                "context_alignment_score": round(score, 3),
            }
        )
    repairs.reverse()
    return text, repairs


def _repair_ambiguous_statistic_tokens(
    text: str,
    ocr_text: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    repairs: List[Dict[str, Any]] = []
    replacements: List[Tuple[int, int, str, float, str]] = []
    ocr_candidates = list(_OCR_STATISTIC_RE.finditer(ocr_text))
    for match in _AMBIGUOUS_STATISTIC_RE.finditer(text):
        left_words = _context_words(text[max(0, match.start() - 140) : match.start()])[-8:]
        right_words = _context_words(text[match.end() : match.end() + 140])[:8]
        scored_by_replacement: Dict[str, float] = {}
        for candidate in ocr_candidates:
            number = candidate.group("number")
            if "." not in number:
                continue
            replacement = f"{candidate.group('label')} = {number}"
            candidate_left = _context_words(
                ocr_text[max(0, candidate.start() - 180) : candidate.start()]
            )[-8:]
            candidate_right = _context_words(
                ocr_text[candidate.end() : candidate.end() + 180]
            )[:8]
            score = _context_alignment_score(
                left_words,
                right_words,
                candidate_left,
                candidate_right,
            )
            scored_by_replacement[replacement] = max(
                score,
                scored_by_replacement.get(replacement, 0.0),
            )
        scored = sorted(
            ((score, replacement) for replacement, score in scored_by_replacement.items()),
            reverse=True,
        )
        if not scored:
            continue
        best_score, replacement = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < 0.68 or best_score - second_score < 0.08:
            continue
        replacements.append(
            (match.start(), match.end(), replacement, best_score, match.group(0))
        )

    for start, end, replacement, score, original in reversed(replacements):
        text = text[:start] + replacement + text[end:]
        repairs.append(
            {
                "original": original,
                "replacement": replacement,
                "context_alignment_score": round(score, 3),
            }
        )
    repairs.reverse()
    return text, repairs


def _has_suspicious_numeric_ocr(text: str) -> bool:
    return bool(
        _AMBIGUOUS_DECIMAL_RE.search(text)
        or _AMBIGUOUS_STATISTIC_RE.search(text)
    )


def _suspicious_numeric_count(text: str) -> int:
    return len(_AMBIGUOUS_DECIMAL_RE.findall(text)) + len(
        _AMBIGUOUS_STATISTIC_RE.findall(text)
    )


def _context_words(value: str) -> List[str]:
    return [token.lower() for token in _CONTEXT_WORD_RE.findall(value)]


def _context_alignment_score(
    expected_left: List[str],
    expected_right: List[str],
    actual_left: List[str],
    actual_right: List[str],
) -> float:
    left = SequenceMatcher(None, expected_left, actual_left).ratio()
    right = SequenceMatcher(None, expected_right, actual_right).ratio()
    expected = set(expected_left + expected_right)
    actual = set(actual_left + actual_right)
    overlap = len(expected & actual) / max(1, len(expected))
    return (left + right + overlap) / 3.0


def _write_artifacts(
    artifacts_dir: Path,
    document: ParsedPdfDocument,
    *,
    raw_document: Optional[Dict[str, Any]],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    document.artifacts_dir = str(artifacts_dir)
    _render_visual_pages(document, artifacts_dir)
    _write_json(artifacts_dir / "blocks.json", document.to_dict(include_blocks=True))
    _write_json(
        artifacts_dir / "parser_report.json",
        {
            **document.to_dict(include_blocks=False),
            "cache_version": PARSER_CACHE_VERSION,
            "block_types": _counts(block.block_type for block in document.blocks),
        },
    )
    (artifacts_dir / "document.md").write_text(document.markdown, encoding="utf-8")
    if raw_document is not None:
        _write_json(artifacts_dir / "document.json", raw_document)


def _render_visual_pages(document: ParsedPdfDocument, artifacts_dir: Path) -> None:
    pages = sorted(
        {
            block.page_start
            for block in document.blocks
            if block.block_type in {"table", "picture", "figure", "page_image"}
        }
    )
    if not pages:
        return
    try:
        import pypdfium2 as pdfium
    except ImportError:
        document.warnings.append("visual_pages_not_rendered:pypdfium2_unavailable")
        return
    image_dir = artifacts_dir / "page_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rendered: Dict[int, str] = {}
    pdf = pdfium.PdfDocument(document.source_file)
    try:
        for page_no in pages:
            if page_no < 1 or page_no > len(pdf):
                continue
            image_path = image_dir / f"page_{page_no:03d}.png"
            if not image_path.exists():
                page = pdf[page_no - 1]
                bitmap = page.render(scale=1.8)
                pil_image = bitmap.to_pil()
                try:
                    pil_image.save(image_path)
                finally:
                    pil_image.close()
                    bitmap.close()
                    page.close()
            rendered[page_no] = str(image_path)
    finally:
        pdf.close()
    for block in document.blocks:
        if block.page_start in rendered and block.block_type in {"table", "picture", "figure", "page_image"}:
            block.metadata["page_image"] = rendered[block.page_start]


def _load_cache(artifacts_dir: Path, source_hash: str) -> Optional[ParsedPdfDocument]:
    report_path = artifacts_dir / "parser_report.json"
    blocks_path = artifacts_dir / "blocks.json"
    if not report_path.exists() or not blocks_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        payload = json.loads(blocks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("cache_version") != PARSER_CACHE_VERSION or report.get("source_sha256") != source_hash:
        return None
    markdown_path = artifacts_dir / "document.md"
    markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
    document = ParsedPdfDocument.from_dict(payload, markdown=markdown)
    document.artifacts_dir = str(artifacts_dir)
    image_dir = artifacts_dir / "page_images"
    for block in document.blocks:
        old_image = str((block.metadata or {}).get("page_image") or "")
        if not old_image:
            continue
        candidate = image_dir / Path(old_image).name
        if candidate.exists():
            block.metadata["page_image"] = str(candidate)
    return document


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _counts(values: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts
