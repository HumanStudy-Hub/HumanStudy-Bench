from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional

_TYPE_MAP: Dict[str, str] = {
    "MC": "multiple_choice",        # multiple choice (single/multi answer)
    "Matrix": "matrix",             # grid of rows x scale
    "Slider": "slider",
    "TE": "text_entry",
    "DB": "descriptive_text",       # display-only block of text (no response)
    "Timing": "meta",               # page timers
    "Captcha": "meta",
    "Meta": "meta",
    "RO": "rank_order",
    "CS": "constant_sum",
    "DD": "drill_down",
    "PGR": "pick_group_rank",
}

NON_RESPONSE_TYPES = {"descriptive_text", "meta"}


@dataclass
class QsfItem:
    qid: str
    data_export_tag: Optional[str]
    block: Optional[str]
    qualtrics_type: str
    type: str
    text: str
    choices: List[str] = field(default_factory=list)
    rows: List[str] = field(default_factory=list)
    scale: Optional[Dict[str, Any]] = None

    def is_response_item(self) -> bool:
        return self.type not in NON_RESPONSE_TYPES


@dataclass
class QsfBlock:
    id: str
    description: str
    is_trash: bool
    items: List[QsfItem] = field(default_factory=list)


@dataclass
class ParsedSurvey:
    survey_name: str
    blocks: List[QsfBlock] = field(default_factory=list)
    flow_conditions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def items(self) -> List[QsfItem]:
        out: List[QsfItem] = []
        for block in self.blocks:
            out.extend(block.items)
        return out

    def response_items(self, *, include_trash: bool = False) -> List[QsfItem]:
        """Participant-answerable items, trash blocks excluded by default."""
        out: List[QsfItem] = []
        for block in self.blocks:
            if block.is_trash and not include_trash:
                continue
            out.extend(it for it in block.items if it.is_response_item())
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "survey_name": self.survey_name,
            "flow_conditions": self.flow_conditions,
            "blocks": [
                {
                    "id": b.id,
                    "description": b.description,
                    "is_trash": b.is_trash,
                    "items": [asdict(it) for it in b.items],
                }
                for b in self.blocks
            ],
            "items": [asdict(it) for it in self.items],
        }

# Text normalization

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t ]+")


def clean_html(raw: Any) -> str:
    """Strip Qualtrics rich-text HTML to readable plain text.

    Block tags become line breaks so multi-paragraph prompts stay legible;
    inline tags are removed; entities are unescaped; whitespace collapsed.
    """
    if raw is None:
        return ""
    text = str(raw)
    # turn block-level boundaries into newlines before stripping tags
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*(p|div|li|tr)\s*>", "\n", text)
    text = _TAG_RE.sub("", text)
    text = unescape(text)
    # collapse intra-line whitespace, trim each line, drop empties
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _choice_displays(choices: Any) -> List[str]:
    """Qualtrics Choices/Answers dicts -> ordered list of display strings.

    Keys are numeric strings ("1","2","10"); preserve numeric order, not the
    dict's insertion/lexical order.
    """
    if not isinstance(choices, dict):
        return []
    def _key(k: str) -> Any:
        try:
            return (0, int(k))
        except (TypeError, ValueError):
            return (1, str(k))
    out: List[str] = []
    for k in sorted(choices.keys(), key=_key):
        v = choices[k]
        disp = v.get("Display") if isinstance(v, dict) else v
        out.append(clean_html(disp))
    return out


def _scale_from_answers(answers: Any) -> Optional[Dict[str, Any]]:
    """Likert/matrix scale points -> {min, max, anchors}.

    Qualtrics blanks intermediate anchors with "&nbsp;"; keep only the points
    that carry a real label as the anchors map.
    """
    if not isinstance(answers, dict):
        return None
    numeric = []
    anchors: Dict[str, str] = {}
    for k, v in answers.items():
        disp = clean_html(v.get("Display") if isinstance(v, dict) else v)
        try:
            n = int(k)
        except (TypeError, ValueError):
            continue
        numeric.append(n)
        if disp:
            anchors[str(n)] = disp
    if not numeric:
        return None
    return {"min": min(numeric), "max": max(numeric), "anchors": anchors}


def _slider_scale(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cfg = payload.get("Configuration") if isinstance(payload.get("Configuration"), dict) else {}
    lo, hi = cfg.get("CSSliderMin"), cfg.get("CSSliderMax")
    try:
        lo = int(lo)
        hi = int(hi)
    except (TypeError, ValueError):
        # fall back to label-based scale if the config range is absent
        return _scale_from_answers(payload.get("Labels")) or _scale_from_answers(
            payload.get("Answers")
        )
    anchors: Dict[str, str] = {}
    labels = payload.get("Labels")
    if isinstance(labels, dict):
        displays = []
        def label_key(item: Any) -> tuple[int, Any]:
            return (0, int(item)) if str(item).isdigit() else (1, str(item))

        for k in sorted(labels.keys(), key=label_key):
            v = labels[k]
            disp = clean_html(v.get("Display") if isinstance(v, dict) else v)
            if disp:
                displays.append(disp)
        anchors = _slider_anchors(displays, lo, hi)
    return {"min": lo, "max": hi, "anchors": anchors}


def _slider_anchors(labels: List[str], lo: int, hi: int) -> Dict[str, str]:
    if not labels:
        return {}
    explicit: Dict[str, str] = {}
    for label in labels:
        match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:[-:–—]\s*)?(.*)$", label, flags=re.S)
        if not match:
            explicit = {}
            break
        value = float(match.group(1))
        if value < lo or value > hi:
            explicit = {}
            break
        key = str(int(value)) if value.is_integer() else str(value)
        explicit[key] = label
    if explicit and len(explicit) == len(labels):
        return explicit

    if len(labels) == 1:
        return {str(lo): labels[0]}
    span = hi - lo
    anchors: Dict[str, str] = {}
    for idx, label in enumerate(labels):
        value = lo + (span * idx / (len(labels) - 1))
        key = str(int(round(value))) if abs(value - round(value)) < 1e-9 else str(value)
        anchors[key] = label
    return anchors


def _extract_flow_conditions(flow_payload: Any) -> List[Dict[str, Any]]:
    """Recover simple embedded-data condition factors from Qualtrics Survey Flow."""
    values: Dict[str, set[str]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("Type") == "EmbeddedData":
            for entry in node.get("EmbeddedData") or []:
                if not isinstance(entry, dict):
                    continue
                field = entry.get("Field") or entry.get("Description")
                if not field:
                    continue
                values.setdefault(str(field), set()).add(str(entry.get("Value", "")))
        for key in ("Flow", "Payload"):
            if key in node:
                visit(node[key])

    visit(flow_payload)
    out = []
    for field, levels in sorted(values.items()):
        clean_levels = sorted(level for level in levels if level != "")
        if len(clean_levels) > 1:
            out.append({
                "name": field,
                "levels": clean_levels,
                "source": "qsf_flow_embedded_data",
            })
    return out


# Core parse

def _normalized_type(qtype: str, selector: str) -> str:
    base = _TYPE_MAP.get(qtype, "other")
    # Bipolar matrices are really a forced A/B choice, not a rating grid.
    if qtype == "Matrix" and selector == "Bipolar":
        return "multiple_choice"
    return base


def _bipolar_choices(payload: Dict[str, Any]) -> List[str]:
    """Bipolar matrix choices encode 'A...:B...' poles; reduce to ['A','B']."""
    rows = _choice_displays(payload.get("Choices"))
    for row in rows:
        if ":" in row:
            left, right = row.split(":", 1)
            la = left.strip()[:1].upper()
            rb = right.strip()[:1].upper()
            if la and rb:
                return [la, rb]
    return ["A", "B"] if rows else []


def _build_item(payload: Dict[str, Any], qid: str, block_name: Optional[str]) -> QsfItem:
    qtype = str(payload.get("QuestionType") or "")
    selector = str(payload.get("Selector") or "")
    norm = _normalized_type(qtype, selector)
    qualtrics_type = f"{qtype}/{selector}" if selector else qtype

    text = clean_html(payload.get("QuestionText"))
    choices: List[str] = []
    rows: List[str] = []
    scale: Optional[Dict[str, Any]] = None

    if qtype == "Matrix" and selector == "Bipolar":
        choices = _bipolar_choices(payload)
        rows = _choice_displays(payload.get("Choices"))  # the per-trial rows
    elif qtype == "Matrix":
        rows = _choice_displays(payload.get("Choices"))   # statement rows
        scale = _scale_from_answers(payload.get("Answers"))
        choices = [a for a in (scale or {}).get("anchors", {}).values()] if scale else []
    elif qtype == "MC":
        choices = _choice_displays(payload.get("Choices"))
    elif qtype == "Slider":
        scale = _slider_scale(payload)
        rows = _choice_displays(payload.get("Choices"))  # statement(s) being rated

    return QsfItem(
        qid=qid,
        data_export_tag=payload.get("DataExportTag") or None,
        block=block_name,
        qualtrics_type=qualtrics_type,
        type=norm,
        text=text,
        choices=[c for c in choices if c],
        rows=[r for r in rows if r],
        scale=scale,
    )


def _iter_blocks(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [b for b in payload if isinstance(b, dict)]
    if isinstance(payload, dict):
        return [b for b in payload.values() if isinstance(b, dict)]
    return []


def parse_qsf(data: Dict[str, Any]) -> ParsedSurvey:
    """Parse a loaded .qsf dict into a `ParsedSurvey`."""
    elements = data.get("SurveyElements", [])
    survey_name = ""
    entry = data.get("SurveyEntry")
    if isinstance(entry, dict):
        survey_name = clean_html(entry.get("SurveyName") or "")

    # 1) index every question payload by QID
    questions: Dict[str, Dict[str, Any]] = {}
    for el in elements:
        if el.get("Element") == "SQ":
            qid = el.get("PrimaryAttribute") or (el.get("Payload") or {}).get("QuestionID")
            payload = el.get("Payload")
            if qid and isinstance(payload, dict):
                questions[qid] = payload

    # 2) find the block element (ordering + grouping)
    block_defs: List[Dict[str, Any]] = []
    for el in elements:
        if el.get("Element") == "BL":
            block_defs = _iter_blocks(el.get("Payload"))
            break

    blocks: List[QsfBlock] = []
    seen: set[str] = set()

    for bdef in block_defs:
        desc = clean_html(bdef.get("Description") or "") or "Block"
        is_trash = str(bdef.get("Type") or "").lower() == "trash" or "trash" in desc.lower()
        block_id = str(bdef.get("ID") or desc)
        block = QsfBlock(id=block_id, description=desc, is_trash=is_trash)
        for be in bdef.get("BlockElements", []) or []:
            if be.get("Type") != "Question":
                continue
            qid = be.get("QuestionID")
            payload = questions.get(qid)
            if not payload:
                continue
            block.items.append(_build_item(payload, qid, desc))
            seen.add(qid)
        blocks.append(block)

    # 3) any question not referenced by a block (rare) -> "Unblocked"
    orphans = [qid for qid in questions if qid not in seen]
    if orphans:
        block = QsfBlock(id="__unblocked__", description="Unblocked", is_trash=False)
        for qid in orphans:
            block.items.append(_build_item(questions[qid], qid, None))
        blocks.append(block)

    flow_conditions: List[Dict[str, Any]] = []
    for el in elements:
        if el.get("Element") == "FL":
            flow_conditions = _extract_flow_conditions(el.get("Payload"))
            break

    return ParsedSurvey(survey_name=survey_name, blocks=blocks, flow_conditions=flow_conditions)


def parse_qsf_file(path: str | Path) -> ParsedSurvey:
    """Load and parse a .qsf file from disk."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return parse_qsf(data)

# CLI

def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Parse a Qualtrics .qsf survey file.")
    ap.add_argument("qsf", help="Path to the .qsf file")
    ap.add_argument("--json", action="store_true", help="Emit full parsed JSON")
    ap.add_argument("--all", action="store_true", help="Include trash/meta items")
    args = ap.parse_args(argv)

    survey = parse_qsf_file(args.qsf)

    if args.json:
        out = survey.to_dict()
        if not args.all:
            out["blocks"] = [b for b in out["blocks"] if not b["is_trash"]]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"Survey: {survey.survey_name or '(unnamed)'}")
    for block in survey.blocks:
        if block.is_trash and not args.all:
            continue
        marker = " [trash]" if block.is_trash else ""
        print(f"\n## Block: {block.description}{marker}  ({len(block.items)} items)")
        for it in block.items:
            if not args.all and not it.is_response_item():
                continue
            tag = it.data_export_tag or it.qid
            preview = it.text.replace("\n", " ")
            if len(preview) > 90:
                preview = preview[:90] + "…"
            extra = ""
            if it.choices:
                extra = f"  choices={it.choices[:4]}"
            elif it.scale:
                extra = f"  scale={it.scale['min']}-{it.scale['max']}"
            print(f"  - [{tag}] ({it.type}) {preview}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
