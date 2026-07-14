from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class DocumentBlock:
    block_id: str
    order: int
    page_start: int
    page_end: int
    block_type: str
    text: str
    section_path: List[str] = field(default_factory=list)
    bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "DocumentBlock":
        return cls(
            block_id=str(value.get("block_id") or ""),
            order=int(value.get("order") or 0),
            page_start=int(value.get("page_start") or 1),
            page_end=int(value.get("page_end") or value.get("page_start") or 1),
            block_type=str(value.get("block_type") or "text"),
            text=str(value.get("text") or ""),
            section_path=[str(item) for item in value.get("section_path") or [] if str(item).strip()],
            bbox=[float(item) for item in value.get("bbox") or []] or None,
            metadata=value.get("metadata") if isinstance(value.get("metadata"), dict) else {},
        )


@dataclass
class ParsedPdfDocument:
    source_file: str
    source_sha256: str
    parser: str
    parser_version: str
    page_count: int
    blocks: List[DocumentBlock]
    markdown: str = ""
    degraded: bool = False
    warnings: List[str] = field(default_factory=list)
    artifacts_dir: Optional[str] = None

    @property
    def text_chars(self) -> int:
        return sum(len(block.text) for block in self.blocks)

    def block_map(self) -> Dict[str, DocumentBlock]:
        return {block.block_id: block for block in self.blocks}

    def to_dict(self, *, include_blocks: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "version": "pdf-document-v1",
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "page_count": self.page_count,
            "text_chars": self.text_chars,
            "degraded": self.degraded,
            "warnings": list(self.warnings),
            "artifacts_dir": self.artifacts_dir,
        }
        if include_blocks:
            payload["blocks"] = [block.to_dict() for block in self.blocks]
        return payload

    @classmethod
    def from_dict(cls, value: Dict[str, Any], *, markdown: str = "") -> "ParsedPdfDocument":
        return cls(
            source_file=str(value.get("source_file") or ""),
            source_sha256=str(value.get("source_sha256") or ""),
            parser=str(value.get("parser") or "unknown"),
            parser_version=str(value.get("parser_version") or "unknown"),
            page_count=int(value.get("page_count") or 0),
            blocks=[
                DocumentBlock.from_dict(item)
                for item in value.get("blocks") or []
                if isinstance(item, dict)
            ],
            markdown=markdown,
            degraded=bool(value.get("degraded")),
            warnings=[str(item) for item in value.get("warnings") or []],
            artifacts_dir=str(value.get("artifacts_dir") or "") or None,
        )


@dataclass
class EvidenceContext:
    text: str
    mode: str
    block_ids: List[str]
    pages: List[int]
    facets: Dict[str, List[str]]
    source_chars: int
    context_chars: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


RESPONSE_TYPES_REQUIRING_CONTRACT = {
    "multiple_choice",
    "likert",
    "scale",
    "slider",
    "ranking",
    "matrix",
}


def evidence_refs(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def item_has_response_contract(item: Dict[str, Any]) -> bool:
    if isinstance(item.get("options"), list) and item["options"]:
        return True
    scale = item.get("scale") if isinstance(item.get("scale"), dict) else {}
    if scale.get("min") is not None and scale.get("max") is not None:
        return True
    matrix = item.get("matrix") if isinstance(item.get("matrix"), dict) else {}
    if isinstance(matrix.get("rows"), list) and matrix["rows"] and isinstance(matrix.get("columns"), list) and matrix["columns"]:
        return True
    response_format = item.get("response_format") if isinstance(item.get("response_format"), dict) else {}
    return bool(
        response_format.get("options")
        or response_format.get("scale_min") is not None
        or response_format.get("scale_max") is not None
        or (response_format.get("rows") and response_format.get("columns"))
    )


def invalid_evidence_refs(values: Iterable[str], valid_block_ids: set[str]) -> List[str]:
    return sorted({value for value in values if value not in valid_block_ids})
