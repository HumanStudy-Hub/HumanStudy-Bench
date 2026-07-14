from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from generation_pipeline.pdf.models import DocumentBlock, ParsedPdfDocument
from generation_pipeline.utils.pdf_chunker import clean_text, split_pages
from generation_pipeline.utils.pdf_extractor import extract_pdf_text


PARSER_CACHE_VERSION = "pdf-parser-v3"
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
            if cache_dir is not None:
                _write_artifacts(cache_dir, document, raw_document=raw_document)
            return document
        except Exception as exc:
            errors.append(f"docling_failed:{type(exc).__name__}:{exc}")

    document = baseline
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
