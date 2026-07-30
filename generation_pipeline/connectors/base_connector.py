"""Base classes and shared utilities for source connectors.

Connectors are deterministic fetchers for external paper materials. Phase 1 is
OSF-only: stages fetch OSF sources once, store them under
``stage4/<paper>/sources/``, and then feed that source directory into
verification or patching logic.
"""

from __future__ import annotations

import json
import mimetypes
import re
import urllib.parse
import zipfile
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from generation_pipeline.utils.pdf_extractor import extract_pdf_text


TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".json",
    ".qsf",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".xml",
    ".do",
    ".r",
    ".py",
    ".js",
    ".css",
}
DOCX_SUFFIXES = {".docx"}
PDF_SUFFIXES = {".pdf"}


@dataclass(frozen=True)
class SourceFetchPlan:
    """Connector-independent plan for fetching a paper's external source."""

    paper_folder: str
    json_path: str | None
    paper_title: str
    link: str
    connector: str | None = None
    doi: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FetchedSource:
    """One fetched file, with provenance."""

    path: str
    connector: str
    source_url: str
    kind: str = "file"
    size: int | None = None
    content_type: str | None = None
    text_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConnectorFetchResult:
    """Fetch result for one connector invocation."""

    connector: str
    plan: SourceFetchPlan
    dest: str
    fetched: list[FetchedSource] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector": self.connector,
            "plan": self.plan.to_dict(),
            "dest": self.dest,
            "fetched": [item.to_dict() for item in self.fetched],
            "skipped": list(self.skipped),
            "errors": list(self.errors),
        }


class BaseSourceConnector(ABC):
    """Base class for external source connectors."""

    name = "base"

    @abstractmethod
    def detect(self, link: str) -> bool:
        """Return True when this connector can handle ``link``."""

    @abstractmethod
    def fetch(self, plan: SourceFetchPlan, dest: Path) -> list[FetchedSource]:
        """Fetch source files for ``plan`` into ``dest`` and return files."""

    def extract_text(self, file: Path) -> str:
        """Extract text from a fetched file.

        Unsupported binary formats return an empty string rather than raising.
        Parse failures raise so the registry can record precise provenance.
        """
        return extract_text_from_file(file)


def extract_text_from_file(path: Path, *, max_chars: int = 200000) -> str:
    """Extract text from PDF, DOCX, Qualtrics/text, and simple text-like files."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return extract_pdf_text(path, max_chars=max_chars)
    if suffix in DOCX_SUFFIXES:
        return _docx_text(path)[:max_chars]
    if suffix == ".qsf":
        # Qualtrics Survey Format: JSON containing HTML-wrapped question text.
        # Raw JSON is 90 %+ metadata noise — parse to human-readable Q&A.
        return _qsf_text(path)[:max_chars]
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]

    # Try lightweight text detection for unknown files, but avoid binary dumps.
    raw = path.read_bytes()[:max_chars]
    if not raw:
        return ""
    if raw.count(b"\x00") > max(1, len(raw) // 20):
        return ""
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    printable = sum(1 for ch in text if ch.isprintable() or ch.isspace())
    if printable / max(len(text), 1) < 0.8:
        return ""
    return text


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_filename(name: str, *, fallback: str = "source") -> str:
    """Return a filesystem-safe filename while preserving useful suffixes."""
    name = urllib.parse.unquote(name or "").strip()
    if not name:
        name = fallback
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name or fallback


def safe_relative_path(path: str, *, fallback: str = "source") -> Path:
    """Sanitize a remote relative path and prevent directory traversal."""
    parts: list[str] = []
    for part in PurePosixPath(path or fallback).parts:
        if part in {"", ".", "..", "/"}:
            continue
        parts.append(safe_filename(part, fallback=fallback))
    return Path(*parts) if parts else Path(safe_filename(fallback))


def filename_from_url(url: str, *, fallback: str = "source") -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    if not name:
        guessed = mimetypes.guess_extension(parsed.path) or ""
        name = f"{fallback}{guessed}"
    return safe_filename(name, fallback=fallback)


def iter_text_files(files: Iterable[Path]) -> Iterable[Path]:
    """Yield files that are likely to produce text."""
    for path in files:
        suffix = Path(path).suffix.lower()
        if suffix in PDF_SUFFIXES | DOCX_SUFFIXES | TEXT_SUFFIXES:
            yield Path(path)


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    return "\n".join(node.text for node in root.iter() if node.tag.endswith("}t") and node.text)


_HTML_TAG = re.compile(r"<[^>]+>")
_LEFTOP = re.compile(r'class="LeftOpDesc">([^<]+)<')
_OPDESC = re.compile(r'class="OpDesc">([^<]+)<')


def _qsf_clean(s: str) -> str:
    import html as _html

    return re.sub(r"\s+", " ", _html.unescape(_HTML_TAG.sub(" ", s or ""))).strip()


def _qsf_readable_logic(logic: Any) -> str:
    """Render Qualtrics BranchLogic / DisplayLogic as a short human-readable condition."""
    descriptions: list[str] = []

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                if key == "Description" and isinstance(val, str):
                    descriptions.append(val)
                else:
                    collect(val)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(logic)

    parts: list[str] = []
    for desc in descriptions:
        left_m = _LEFTOP.search(desc)
        if not left_m:
            continue
        left = _qsf_clean(left_m.group(1))
        op_m = _OPDESC.search(desc)
        op = _qsf_clean(op_m.group(1)) if op_m else ""
        if left:
            parts.append(f"{left} {op}".strip())
    # Dedupe while preserving order
    seen: set[str] = set()
    uniq = [p for p in parts if not (p in seen or seen.add(p))]
    return "; ".join(uniq)


def _render_sq(payload: dict[str, Any]) -> str:
    """Render one Survey Question (SQ) payload to clean text with choices + display logic."""
    qtext = _qsf_clean(payload.get("QuestionText", ""))
    if not qtext:
        return ""
    sub = _qsf_clean(payload.get("QuestionDescription", ""))
    header = f"Q: {qtext}"
    if sub and sub != qtext and len(sub) < 120:
        header += f" ({sub})"
    parts = [header]

    cond = _qsf_readable_logic(payload.get("DisplayLogic"))
    if cond:
        parts.append(f"  [shown if: {cond}]")

    choices = payload.get("Choices") or payload.get("Answers") or {}
    if isinstance(choices, dict):
        for key in sorted(choices, key=lambda k: (int(k) if str(k).isdigit() else 0, str(k))):
            val = choices[key]
            lbl = _qsf_clean(val.get("Display") or val.get("Text") or "") if isinstance(val, dict) else ""
            if lbl:
                parts.append(f"  {key}. {lbl}")
    labels = payload.get("Labels") or {}
    if isinstance(labels, dict):
        for key, val in labels.items():
            lbl = _qsf_clean(val.get("Display") or "") if isinstance(val, dict) else ""
            if lbl:
                parts.append(f"  [{key}] {lbl}")
    return "\n".join(parts)


def _qsf_text(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        data = json.loads(raw)
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")

    elements = data.get("SurveyElements") if isinstance(data, dict) else []
    if not isinstance(elements, list):
        return ""

    # --- index questions by QID and parse block definitions ------------------
    qid_map: dict[str, dict[str, Any]] = {}
    block_map: dict[str, dict[str, Any]] = {}
    flow: list[Any] = []
    survey_name = ""
    for el in elements:
        if not isinstance(el, dict):
            continue
        kind = el.get("Element")
        if kind == "SQ":
            payload = el.get("Payload")
            qid = el.get("PrimaryAttribute") or (payload.get("QuestionID") if isinstance(payload, dict) else None)
            if qid and isinstance(payload, dict):
                qid_map[qid] = payload
        elif kind == "BL":
            payload = el.get("Payload")
            iterable = payload.values() if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
            for block in iterable:
                if not isinstance(block, dict):
                    continue
                bid = block.get("ID")
                if not bid:
                    continue
                qids = [
                    item.get("QuestionID")
                    for item in (block.get("BlockElements") or [])
                    if isinstance(item, dict) and item.get("Type") == "Question" and item.get("QuestionID")
                ]
                block_map[bid] = {
                    "description": _qsf_clean(block.get("Description") or ""),
                    "type": block.get("Type") or "Standard",
                    "qids": qids,
                }
        elif kind == "FL":
            payload = el.get("Payload")
            if isinstance(payload, dict) and isinstance(payload.get("Flow"), list):
                flow = payload["Flow"]
        elif kind == "SO":
            payload = el.get("Payload")
            if isinstance(payload, dict):
                survey_name = _qsf_clean(payload.get("SurveyName") or "")

    def render_block(bid: str, context: list[str]) -> str:
        block = block_map.get(bid)
        if not block or block.get("type") == "Trash":
            return ""
        rendered = [_render_sq(qid_map[q]) for q in block["qids"] if q in qid_map]
        rendered = [r for r in rendered if r]
        if not rendered:
            return ""
        desc = block["description"] or bid
        ctx = f" | condition: {', '.join(context)}" if context else ""
        return f"\n--- BLOCK: {desc}{ctx} ---\n" + "\n".join(rendered)

    # --- walk the flow, accumulating branch/embedded-data context ------------
    body_parts: list[str] = []
    condition_vars: dict[str, set] = {}
    branch_labels: list[str] = []

    def walk(items: list[Any], context: list[str]) -> None:
        for it in items:
            if not isinstance(it, dict):
                continue
            t = it.get("Type")
            if t == "Block":
                bid = it.get("ID")
                if bid:
                    rendered = render_block(bid, context)
                    if rendered:
                        body_parts.append(rendered)
            elif t == "EmbeddedData":
                local = list(context)
                for ed in it.get("EmbeddedData", []) or []:
                    if not isinstance(ed, dict):
                        continue
                    field_name, value = ed.get("Field"), ed.get("Value")
                    if field_name and value not in (None, ""):
                        condition_vars.setdefault(field_name, set()).add(str(value))
                        local.append(f"{field_name}={value}")
                # ED assignments apply to siblings that follow; mutate context in place
                context[:] = local
            elif t in ("Branch",):
                label = _qsf_readable_logic(it.get("BranchLogic"))
                if label:
                    branch_labels.append(label)
                child_ctx = context + ([label] if label else [])
                walk(it.get("Flow", []) or [], child_ctx)
            elif t in ("BlockRandomizer", "Randomizer", "Group", "Root"):
                walk(it.get("Flow", []) or [], list(context))
            else:
                # Unknown node may still carry a nested Flow
                if isinstance(it.get("Flow"), list):
                    walk(it["Flow"], list(context))

    if flow:
        walk(flow, [])

    # --- assemble final text -------------------------------------------------
    out: list[str] = []
    if survey_name:
        out.append(f"=== QSF SURVEY: {survey_name} ===")
    if condition_vars or branch_labels:
        lines = ["CONDITIONS defined in the survey flow (manipulation variables):"]
        for field_name in sorted(condition_vars):
            values = sorted(condition_vars[field_name])
            if len(values) == 1:
                lines.append(f"  - {field_name} = {values[0]}")
            else:
                lines.append(f"  - {field_name} ∈ {{{', '.join(values)}}} (randomized between conditions)")
        seen_b: set[str] = set()
        for label in branch_labels:
            if label not in seen_b:
                seen_b.add(label)
                lines.append(f"  - branch: {label}")
        out.append("\n".join(lines))

    if body_parts:
        out.extend(body_parts)
    else:
        # Fallback: flat SQ dump in element order (old behavior) when flow parsing
        # produced nothing (e.g. malformed flow or single implicit block).
        flat = [_render_sq(qid_map[q]) for q in qid_map]
        out.extend(r for r in flat if r)

    return "\n\n".join(out)
