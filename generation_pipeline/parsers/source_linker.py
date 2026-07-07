from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

# Extensions we can parse structurally vs. treat as free text.
STRUCTURED_EXTS = {".qsf": "qsf", ".sav": "sav"}
TEXT_EXTS = {".pdf": "pdf", ".docx": "docx", ".txt": "text", ".md": "text"}

DATA_EXTS = {".csv", ".xlsx", ".xls", ".do", ".r", ".sps", ".dta", ".json"}

# Named (non-numeric) study tokens that recur across papers.
_NAMED_TOKENS = ("pilot", "validation")

_TOKEN_RE = re.compile(r"\b(?:study|studies|exp(?:eriment)?s?)[\s_-]*"
                       r"(\d+[a-z]?(?:-[a-z0-9]+)?)(?![a-z0-9])", re.IGNORECASE)

_SHORT_RE = re.compile(r"(?<![A-Za-z0-9])S(\d+[a-z]?)(?![a-z0-9])")


def _norm_token(tok: str) -> str:
    return re.sub(r"\s+", "", tok).lower()


def _expand_range(raw: str) -> Set[str]:
    """Expand compact study refs to a token set.

    "1a-b"  -> {1a, 1b}      "2-3" -> {2, 3}      "1a/1b" -> {1a, 1b}
    "2a"    -> {2a}          "1"   -> {1}
    """
    raw = _norm_token(raw)
    m = re.match(r"^(\d+)([a-z]?)[-–](\d*)([a-z]?)$", raw)
    if not m:
        return {raw}
    n1, l1, n2, l2 = m.groups()
    base = n1
    out: Set[str] = set()
    # letter range on same base number: 1a-b -> 1a, 1b
    if l1 and l2 and not n2:
        for code in range(ord(l1), ord(l2) + 1):
            out.add(f"{base}{chr(code)}")
        return out
    # numeric range: 2-3 -> 2, 3
    if n2 and not l1 and not l2:
        for num in range(int(n1), int(n2) + 1):
            out.add(str(num))
        return out
    # explicit two endpoints: 1a/1b style already split by separator
    out.add(f"{n1}{l1}")
    if n2:
        out.add(f"{n2}{l2}")
    elif l2:
        out.add(f"{n1}{l2}")
    return out


def study_tokens(study_id: str) -> Set[str]:
    """Token set identifying a sub-study, e.g. 'Study 2a' -> {'2a'}."""
    text = (study_id or "").lower()
    tokens: Set[str] = set()
    for m in _TOKEN_RE.finditer(text):
        tokens |= _expand_range(m.group(1))
    for m in _SHORT_RE.finditer(study_id or ""):
        tokens.add(_norm_token(m.group(1)))
    for named in _NAMED_TOKENS:
        if named in text:
            tokens.add(named)
    return tokens


def file_tokens(path: str) -> Set[str]:
    """Token set a file/folder name refers to, ranges expanded."""
    name = str(path)
    tokens: Set[str] = set()
    for m in _TOKEN_RE.finditer(name):
        tokens |= _expand_range(m.group(1))
    for m in _SHORT_RE.finditer(name):
        tokens.add(_norm_token(m.group(1)))
    low = name.lower()
    for named in _NAMED_TOKENS:
        if named in low:
            tokens.add(named)
    return tokens


def route_ext(path: str | Path) -> Optional[str]:
    """Return parser kind for a path: 'qsf'|'sav'|'pdf'|'docx'|'text'|'data'|None."""
    ext = Path(path).suffix.lower()
    if ext in STRUCTURED_EXTS:
        return STRUCTURED_EXTS[ext]
    if ext in TEXT_EXTS:
        return TEXT_EXTS[ext]
    if ext in DATA_EXTS:
        return "data"
    return None


@dataclass
class LinkedFile:
    path: Path
    kind: str                 # qsf | sav | pdf | docx | text | data
    tokens: Set[str] = field(default_factory=set)

    @property
    def is_material(self) -> bool:
        """Whether this file can yield participant-facing materials."""
        return self.kind in ("qsf", "sav", "pdf", "docx", "text")


@dataclass
class LinkResult:
    by_study: Dict[str, List[LinkedFile]]
    unlinked: List[LinkedFile]

    def material_files(self, study_id: str) -> List[LinkedFile]:
        files = self.by_study.get(study_id, [])
        # Prefer structured instruments first, then text fallback.
        order = {"qsf": 0, "sav": 1, "docx": 2, "pdf": 3, "text": 4, "data": 9}
        return sorted([f for f in files if f.is_material],
                      key=lambda f: order.get(f.kind, 8))


def _iter_files(root: str | Path) -> List[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def link_sources(
    study_ids: Sequence[str],
    files_root: str | Path,
    *,
    include_data: bool = False,
) -> LinkResult:
    studies = {sid: study_tokens(sid) for sid in study_ids}
    by_study: Dict[str, List[LinkedFile]] = {sid: [] for sid in study_ids}
    unlinked: List[LinkedFile] = []

    for path in _iter_files(files_root):
        kind = route_ext(path)
        if kind is None:
            continue
        if kind == "data" and not include_data:
            continue
        # match on the path relative to root so folder names count
        try:
            rel = path.relative_to(files_root)
        except ValueError:
            rel = path
        ftoks = file_tokens(str(rel))
        lf = LinkedFile(path=path, kind=kind, tokens=ftoks)

        matched = [sid for sid, stoks in studies.items() if stoks & ftoks]
        if matched:
            for sid in matched:
                by_study[sid].append(lf)
        else:
            unlinked.append(lf)

    return LinkResult(by_study=by_study, unlinked=unlinked)
