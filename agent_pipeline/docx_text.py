#!/usr/bin/env python3
"""Text of a Word file (``.docx``), page by page, with its paragraph breaks kept.

    python agent_pipeline/docx_text.py <job-dir>/input/external/<file>.docx [--json <out.json>]

Use this to read a survey's Word copy (a Qualtrics "Export Survey to Word" file, a
materials document) before transcribing from it. Joining Word's text runs by hand
loses what separates them:

- a line break inside a paragraph (``<w:br/>``) becomes a new line, so
  "Scenario 1" and "In this study ..." stay apart;
- every paragraph and table cell is its own line;
- a page break (``<w:br w:type="page"/>``, a page-break-before paragraph) and the
  grey "Page Break" line a Qualtrics export prints between pages end the page.
  The marker itself is never text: it is not in any page's paragraphs.

Lines a Qualtrics export adds around the survey ("Start of Block: ...", "End of
Block: ...", "Display This Question:", "Skip To: ...", "Standard: ... (n
Questions)") are kept apart under ``export_lines`` so they can guide the flow; they
are not participant text either.

Printed as plain text: one paragraph per line, pages separated by a line
``--- page break ---``. ``--json`` writes ``{"pages": [[paragraph, ...], ...],
"export_lines": [...]}``. Reading a file runs nothing in it.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

PAGE_SEPARATOR = "--- page break ---"
# The marker line a Qualtrics Word export prints at each page break.
PAGE_MARKER_RE = re.compile(r"^\s*Page Break\s*$")
# Qualtrics starts every block on a new page.
BLOCK_LINE_RE = re.compile(r"^\s*(Start of Block:|End of Block:)")
# Lines a Qualtrics Word export adds that no participant saw.
EXPORT_LINE_RE = re.compile(
    r"(?i)^\s*(start of block:|end of block:|display this question:|skip to:|carry forward|"
    r"standard:.*\(\d+ questions?\)\s*$|block:.*\(\d+ questions?\)\s*$|branch:|endsurvey:|"
    r"(new )?question separator\s*$)")
# Text a participant never saw that only a Word or Qualtrics export puts in a file. The package
# check refuses it in any participant-visible string.
EXPORT_ARTIFACT_RE = re.compile(
    r"(?<!\w)(Page Break|Start of Block:|End of Block:|Display This Question:|Skip To:|"
    r"This question was not displayed|End of Survey:|QuestionSeparator)(?!\w)")


TOKEN_RE = re.compile(
    r"<w:pPr\b[^>]*?(?:/>|>(?P<ppr>.*?)</w:pPr>)"        # paragraph properties: only a page break before counts
    r"|<w:p(?=[\s>/])[^>]*?(?P<empty>/)?>|(?P<pend></w:p>)"
    r"|<w:t(?=[\s>])[^>]*>(?P<text>[^<]*)</w:t>"
    r"|<w:(?P<brk>br|cr)\b(?P<brattrs>[^>]*?)/?>|(?P<tab><w:tab\b[^>]*?/?>)|(?P<hyphen><w:noBreakHyphen\b[^>]*?/?>)",
    re.S)
PAGE_BEFORE_RE = re.compile(r"<w:pageBreakBefore\b(?![^>]*w:val=\"(?:0|false|off)\")")


def read(path: Path) -> Dict[str, List]:
    """``{"pages": [[paragraph, ...], ...], "export_lines": [...]}`` of a .docx file."""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", "replace")
    # A text box is stored twice (mc:Choice and an mc:Fallback copy for old Word): read it once.
    xml = re.sub(r"<mc:Fallback\b.*?</mc:Fallback>", "", xml, flags=re.S)
    pages: List[List[str]] = [[]]
    export_lines: List[str] = []
    current: List[str] = []

    def new_page() -> None:
        if pages[-1]:
            pages.append([])

    def end_paragraph() -> None:
        text = "".join(current)
        current.clear()
        for piece_number, piece in enumerate(text.split("\f")):
            if piece_number:
                new_page()
            for line in piece.split("\n"):
                line = line.replace("\u00a0", " ").rstrip()
                if PAGE_MARKER_RE.match(line):
                    new_page()
                elif EXPORT_LINE_RE.match(line):
                    export_lines.append(line.strip())
                    if BLOCK_LINE_RE.match(line):
                        new_page()
                elif line.strip():
                    pages[-1].append(line)

    for match in TOKEN_RE.finditer(xml):
        if match.group(0).startswith("<w:pPr"):
            if PAGE_BEFORE_RE.search(match.group("ppr") or ""):
                end_paragraph()
                new_page()
        elif match.group("pend") is not None or match.group("empty"):
            end_paragraph()
        elif match.group(0).startswith("<w:p"):
            if current:
                end_paragraph()          # a paragraph inside another (a text box): its own line
        elif match.group("text") is not None:
            current.append(html.unescape(match.group("text")))
        elif match.group("brk"):
            current.append("\f" if re.search(r'w:type="page"', match.group("brattrs") or "") else "\n")
        elif match.group("tab"):
            current.append("\t")
        elif match.group("hyphen"):
            current.append("-")
    end_paragraph()
    if not pages[-1] and len(pages) > 1:
        pages.pop()
    return {"pages": pages, "export_lines": export_lines}


def plain_text(path: Path) -> str:
    """The file as text: one paragraph per line, pages separated by ``--- page break ---``."""
    return f"\n{PAGE_SEPARATOR}\n".join("\n".join(page) for page in read(path)["pages"])


# --------------------------------------------------------------------------- package check

TEXT_KEYS_RE = re.compile(r"(?i)(note|reason|comment|description|derivation|evidence|source|search|why|"
                          r"implementation|departure|impact|action|what$)")


def _visible_strings(value: object, key: str = "") -> List[str]:
    """Participant text in materials.json: every string except notes, reasons and source records."""
    if isinstance(value, dict):
        return [s for k, v in value.items() if not TEXT_KEYS_RE.search(str(k)) for s in _visible_strings(v, str(k))]
    if isinstance(value, list):
        return [s for v in value for s in _visible_strings(v, key)]
    return [value] if isinstance(value, str) else []


def _lost_breaks(lines: List[str]) -> List[str]:
    """The two neighbouring lines of a page run together with nothing between them ("Scenario 1In this")."""
    joins = []
    for first, second in zip(lines, lines[1:]):
        tail, head = first.rstrip()[-12:], second.lstrip()[:12]
        if len(tail.strip()) >= 4 and len(head.strip()) >= 4 and not tail[-1].isspace() and not head[0].isspace():
            joins.append(tail + head)
    return joins


def check(root: Path, job: Path, result: dict | None = None) -> List[str]:
    """Participant text must not carry Word or Qualtrics export artifacts or lose a Word file's line breaks."""
    texts: List[str] = []
    for run in (result or {}).get("runs") or []:
        texts += [str(call.get("prompt") or "") for call in run.get("calls") or []]
    try:
        materials = json.loads((root / "materials" / "materials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        materials = None
    texts += _visible_strings(materials)
    problems: List[str] = []
    artifacts = sorted({m.group(1) for text in texts for m in EXPORT_ARTIFACT_RE.finditer(text)})
    if artifacts:
        example = next(text for text in texts if EXPORT_ARTIFACT_RE.search(text))
        where = EXPORT_ARTIFACT_RE.search(example)
        problems.append(
            "participant text carries survey-export markup no participant saw ("
            + ", ".join(f"'{a}'" for a in artifacts[:4]) + f"; e.g. '...{example[max(0, where.start() - 60):where.end()]}'"
            "). Read Word copies with agent_pipeline/docx_text.py: a 'Page Break' line ends a page and is never text")
    try:
        metadata = json.loads((root / "source" / "paper_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metadata = {}
    sources = metadata.get("external_sources") if isinstance(metadata, dict) else None
    joined = re.sub(r"[ \t]+", " ", "\n".join(texts))
    lost: List[str] = []
    for source in sources if isinstance(sources, list) else []:
        path = job / str(source.get("path") or "") if isinstance(source, dict) else None
        if path is None or path.suffix.lower() != ".docx" or not path.is_file():
            continue
        try:
            pages = read(path)["pages"]
        except (OSError, KeyError, zipfile.BadZipFile):
            continue
        for page in pages:
            for join in _lost_breaks(page):
                if re.sub(r"[ \t]+", " ", join) in joined and join not in lost:
                    lost.append(join)
    if lost:
        problems.append(
            "participant text runs two lines of a Word source together with nothing between them ("
            + "; ".join(f"'{j}'" for j in lost[:3]) + "). Keep the file's paragraph and line breaks (read it with "
            "agent_pipeline/docx_text.py)")
    return problems


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("docx", type=Path)
    parser.add_argument("--json", type=Path, help="write pages and export lines as JSON here")
    args = parser.parse_args(argv)
    try:
        record = read(args.docx)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f"cannot read {args.docx}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{PAGE_SEPARATOR}\n".join("\n".join(page) for page in record["pages"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
