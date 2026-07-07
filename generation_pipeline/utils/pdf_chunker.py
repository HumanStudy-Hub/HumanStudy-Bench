from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generation_pipeline.utils.pdf_extractor import extract_pdf_text

PDF_TEXT_MAX_CHARS = 400000
DEFAULT_CHUNK_SIZE = 2500
DEFAULT_OVERLAP = 600
_PAGE_MARKER = re.compile(r"---\s*Page\s+(\d+)\s*---", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """
    the a an and or of to in on for with without that this these those is are was were be been being
    by as at from into over under between within across about not no yes it its their our your his her
    they we you he she them us i me my study studies effect effects paper participants subject subjects
    """.split()
)


@dataclass
class Chunk:
    """A contiguous slice of text with page and optional source provenance."""

    id: int
    text: str
    page_start: int
    page_end: int
    char_start: int
    source_path: str | None = None
    source_kind: str | None = None
    study_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_start": self.char_start,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "study_keys": list(self.study_keys),
            "text": self.text,
        }


def parse_pages(
    pdf_path: Path,
    *,
    max_chars: Optional[int] = PDF_TEXT_MAX_CHARS,
) -> list[tuple[int, str]]:
    """Extract text and split into (page_no, page_text) using the page markers."""
    raw = extract_pdf_text(Path(pdf_path), max_chars=max_chars)
    return split_pages(raw)


def split_pages(raw_text: str) -> list[tuple[int, str]]:
    """Split text produced by extract_pdf_text into (page_no, text) pairs."""
    if not raw_text:
        return []
    parts = _PAGE_MARKER.split(raw_text)
    pages: list[tuple[int, str]] = []
    # parts = [pre, page_no, body, page_no, body, ...]
    if parts and parts[0].strip() and not _PAGE_MARKER.match("--- Page 0 ---"):
        # leading text before first marker (rare); attach as page 0
        lead = parts[0].strip()
        if lead:
            pages.append((0, lead))
    it = iter(parts[1:])
    for page_no, body in zip(it, it):
        try:
            num = int(page_no)
        except (TypeError, ValueError):
            num = len(pages) + 1
        pages.append((num, body))
    if not pages and raw_text.strip():
        pages.append((1, raw_text))
    return pages


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
    )
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # de-hyphenate across line breaks
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_chunks(
    pages: list[tuple[int, str]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    source_path: str | None = None,
    source_kind: str | None = None,
    study_keys: Iterable[str] = (),
) -> list[Chunk]:
    """Concatenate cleaned page text and slice into overlapping windows.

    Each chunk records the page range it spans so evidence can be cited.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    # Build one cleaned string + a per-character page map.
    segments: list[str] = []
    page_marks: list[tuple[int, int]] = []  # (char_offset_in_joined, page_no)
    offset = 0
    for page_no, body in pages:
        cleaned = clean_text(body)
        if not cleaned:
            continue
        if segments:
            segments.append(" ")
            offset += 1
        page_marks.append((offset, page_no))
        segments.append(cleaned)
        offset += len(cleaned)
    joined = "".join(segments)
    if not joined:
        return []

    def page_at(char_index: int) -> int:
        page = page_marks[0][1] if page_marks else 1
        for mark_offset, page_no in page_marks:
            if mark_offset <= char_index:
                page = page_no
            else:
                break
        return page

    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    cid = 0
    for start in range(0, len(joined), step):
        end = min(start + chunk_size, len(joined))
        body = joined[start:end].strip()
        if body:
            chunks.append(
                Chunk(
                    id=cid,
                    text=body,
                    page_start=page_at(start),
                    page_end=page_at(end - 1),
                    char_start=start,
                    source_path=source_path,
                    source_kind=source_kind,
                    study_keys=tuple(study_keys),
                )
            )
            cid += 1
        if end >= len(joined):
            break
    return chunks


def chunk_pdf(
    pdf_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_chars: Optional[int] = PDF_TEXT_MAX_CHARS,
) -> list[Chunk]:
    """Convenience: parse + chunk a PDF in one call."""
    return build_chunks(parse_pages(pdf_path, max_chars=max_chars), chunk_size=chunk_size, overlap=overlap)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text or "").lower())


def _content_tokens(text: str, *, min_len: int = 3) -> set[str]:
    return {t for t in _tokens(text) if len(t) >= min_len and t not in _STOPWORDS}


def score_chunk(query: str, chunk_text: str, *, keywords: Iterable[str] = ()) -> float:
    """Lexical relevance score for a chunk against a query.

    coverage of distinct query content-tokens (0..1) + bonus for slot keywords,
    with a small fuzzy tie-breaker. Deterministic; no embeddings.
    """
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return 0.0
    chunk_tokens = set(_tokens(chunk_text))
    coverage = len(q_tokens & chunk_tokens) / len(q_tokens)

    low = chunk_text.lower()
    kw_hits = sum(1 for kw in keywords if kw.lower() in low)
    kw_bonus = min(kw_hits, 5) * 0.15

    fuzzy = _fuzzy_ratio(query.lower(), low) / 100.0 * 0.1
    return coverage + kw_bonus + fuzzy


def retrieve(
    chunks: list[Chunk],
    query: str,
    *,
    k: int = 8,
    keywords: Iterable[str] = (),
    allowed_sources: set[str] | None = None,
    use_bm25: bool = True,
) -> list[tuple[Chunk, float]]:
    """Return the top-k chunks for a query, highest score first."""
    keywords = list(keywords)
    if allowed_sources is not None:
        chunks = [chunk for chunk in chunks if chunk.source_path in allowed_sources]
    if use_bm25:
        scored = _bm25_score_chunks(query, chunks, keywords=keywords)
    else:
        scored = [(chunk, score_chunk(query, chunk.text, keywords=keywords)) for chunk in chunks]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [item for item in scored[:k] if item[1] > 0.0]


def format_evidence(scored_chunks: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks as a numbered evidence block for an LLM prompt."""
    blocks = []
    for chunk, score in scored_chunks:
        pages = (
            f"p.{chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pp.{chunk.page_start}-{chunk.page_end}"
        )
        source = f" | source={chunk.source_path}" if chunk.source_path else ""
        blocks.append(f"[Evidence {chunk.id} | {pages}{source} | relevance={score:.2f}]\n{chunk.text}")
    return "\n\n".join(blocks)


def _bm25_score_chunks(
    query: str,
    chunks: list[Chunk],
    *,
    keywords: Iterable[str] = (),
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[Chunk, float]]:
    """Pure-Python BM25 over the candidate chunks, with existing keyword/fuzzy tiebreaks."""
    if not chunks:
        return []
    query_tokens = [t for t in _tokens(query) if len(t) >= 3 and t not in _STOPWORDS]
    if not query_tokens:
        return []

    docs = [[t for t in _tokens(chunk.text) if len(t) >= 3 and t not in _STOPWORDS] for chunk in chunks]
    avg_len = sum(len(doc) for doc in docs) / max(len(docs), 1)
    df: dict[str, int] = {}
    for doc in docs:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1

    import math

    n_docs = len(docs)
    scores: list[tuple[Chunk, float]] = []
    for chunk, doc in zip(chunks, docs):
        if not doc:
            continue
        tf: dict[str, int] = {}
        for token in doc:
            tf[token] = tf.get(token, 0) + 1
        bm25 = 0.0
        doc_len = len(doc)
        for token in query_tokens:
            freq = tf.get(token, 0)
            if not freq:
                continue
            idf = math.log(1 + (n_docs - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / max(avg_len, 1))
            bm25 += idf * (freq * (k1 + 1)) / max(denom, 1e-9)

        low = chunk.text.lower()
        kw_hits = sum(1 for kw in keywords if kw.lower() in low)
        kw_bonus = min(kw_hits, 5) * 0.15
        fuzzy = _fuzzy_ratio(query.lower(), low) / 100.0 * 0.1
        scores.append((chunk, bm25 + kw_bonus + fuzzy))
    return scores


def _fuzzy_ratio(needle: str, haystack: str) -> float:
    if not needle or not haystack:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.partial_ratio(needle, haystack))
    except ImportError:
        import difflib

        return difflib.SequenceMatcher(None, needle, haystack[: max(len(needle) * 4, 1000)]).ratio() * 100.0
