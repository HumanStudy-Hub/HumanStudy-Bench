from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm.helpers import call_with_timeout

from generation_pipeline.utils.pdf_extractor import extract_pdf_text


PDF_TEXT_MAX_CHARS = 500000
PDF_CONTEXT_MAX_CHARS = 220000
PDF_STUDY_CONTEXT_MAX_CHARS = 24000
PDF_APPENDIX_CONTEXT_MAX_CHARS = 16000
PDF_MATERIAL_MAX_TOKENS = 5000
_VALID_TYPES = {"multiple_choice", "likert", "scale", "slider", "open_ended", "ranking", "matrix", "text"}
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_'-]{2,}", re.IGNORECASE)
_TARGET_STOP_TERMS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about",
    "study", "studies", "effect", "effects", "condition", "conditions",
    "measure", "measures", "item", "items", "index", "scale", "article",
    "message", "messages", "dependent", "variable", "variables", "participant",
    "participants", "toward", "after", "before", "interaction", "direct",
    "indirect", "main", "simple", "primary", "reported", "report", "anova",
    "correlation", "regression", "decomposition", "analysis", "finding",
    "one-way", "two-way", "three-way",
}
_CHECK_ITEM_RE = re.compile(
    r"\b(attention|attn|careless|comprehension|manipulation|bot|captcha|"
    r"screen(?:er|ing)|instructional|quality)\s*[_ -]?\s*check\b|"
    r"(?:^|[_ -])(attention|attn|careless|comprehension|manipulation|captcha)(?:[_ -]|$)|"
    r"\bplease\s+(?:select|choose|pick)\b",
    re.IGNORECASE,
)
_ADMIN_ITEM_RE = re.compile(
    r"(?:^|[_ -])(consent|debrief|demograph|age|gender|sex|race|ethnic|"
    r"education|income|prolific|mturk|duration|finished|progress|timer|timing)(?:[_ -]|$)",
    re.IGNORECASE,
)
_META_INSTRUCTION_RE = re.compile(
    r"\b("
    r"from\s+(?:the\s+)?(?:abstract|overview|paper|method|methods)|"
    r"general\s+description|available\s+in\s+the\s+paper|paper\s+text|"
    r"conceptual\s+framing|method\s+description|pages?\s+\d+"
    r")\b",
    re.IGNORECASE,
)


def _slug(text: Any, fallback: str = "study") -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return value or fallback


def _sub_study_id(value: Any) -> str:
    slug = _slug(value)
    if not slug.startswith(("study", "pilot")):
        slug = f"study_{slug}"
    return slug


def _loads_json(text: str) -> Dict[str, Any]:
    response_text = str(text or "").strip()
    if "```json" in response_text:
        response_text = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group())
    if not isinstance(parsed, dict):
        raise ValueError(f"PDF material response is not a JSON object: {type(parsed)}")
    return parsed


def _text(value: Any) -> str:
    text = str(value or "").replace("\u2026", "...").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_ellipsis(value: Any) -> bool:
    text = str(value or "")
    return "..." in text or "\u2026" in text


def _stage1_hint_map(stage1_json: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(stage1_json, dict):
        return out
    for exp in stage1_json.get("experiments", []) or []:
        if not isinstance(exp, dict):
            continue
        compact = {
            "experiment_id": exp.get("experiment_id"),
            "study_id": exp.get("study_id"),
            "design_type": exp.get("design_type"),
            "conditions_or_factors": exp.get("conditions_or_factors"),
            "participant_task": exp.get("participant_task"),
            "input": exp.get("input"),
            "output": exp.get("output"),
            "candidate_source_hints": exp.get("candidate_source_hints"),
        }
        for key in (exp.get("study_id"), exp.get("experiment_id"), exp.get("study_name"), exp.get("experiment_name")):
            sid = _sub_study_id(key)
            if sid and sid not in out:
                out[sid] = compact
    return out


def _selected_or_all_studies(stage_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    studies = [
        study
        for study in stage_json.get("eligible_studies", []) or stage_json.get("studies", []) or []
        if isinstance(study, dict)
    ]
    selected = [
        study
        for study in studies
        if isinstance(study.get("selection"), dict) and study["selection"].get("keep") is True
    ]
    if selected:
        studies = selected
    return studies


def _study_matches_sub_id(study: Dict[str, Any], sub_id: str) -> bool:
    wanted = _sub_study_id(sub_id)
    keys = {
        _sub_study_id(study.get("study")),
        _sub_study_id(study.get("study_id")),
        _sub_study_id(study.get("experiment_id")),
        _sub_study_id(study.get("study_name")),
        _sub_study_id(study.get("experiment_name")),
    }
    return wanted in keys


def _stage_json_with_studies(stage_json: Dict[str, Any], studies: List[Dict[str, Any]]) -> Dict[str, Any]:
    subset = dict(stage_json)
    subset["eligible_studies"] = studies
    subset.pop("studies", None)
    return subset


def _compact_studies(stage_json: Dict[str, Any], stage1_json: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    stage1_hints = _stage1_hint_map(stage1_json)
    studies = _selected_or_all_studies(stage_json)
    for study in studies:
        if not isinstance(study, dict):
            continue
        findings = []
        for finding in study.get("findings", []) or []:
            if isinstance(finding, dict):
                findings.append(
                    {
                        "finding_id": finding.get("finding_id"),
                        "role": finding.get("role"),
                        "IV": finding.get("IV"),
                        "DV": finding.get("DV"),
                        "simulation_target": finding.get("simulation_target"),
                    }
                )
        effects = []
        for effect in study.get("effects", []) or []:
            if isinstance(effect, dict):
                effects.append(
                    {
                        "IV": effect.get("IV"),
                        "DV": effect.get("DV"),
                        "effecttype": effect.get("effecttype"),
                        "direction": effect.get("direction"),
                    }
                )
        study_name = study.get("study") or study.get("study_id") or study.get("experiment_id")
        item = {
            "study": study_name,
            "study_id": study.get("study_id"),
            "design": study.get("design") or study.get("design_type"),
            "sample": study.get("sample"),
            "effects": effects[:12],
            "findings": findings[:12],
            "selection": study.get("selection"),
            "required_hsb_fields": [
                "instructions/stimulus/vignette/manipulation",
                "participant response items",
                "options/anchors/scales",
                "conditions/factors/levels",
            ],
        }
        hint = stage1_hints.get(_sub_study_id(study_name)) or stage1_hints.get(_sub_study_id(study.get("study_id")))
        if hint:
            item["stage1_source_hints"] = hint
        out.append(item)
    return out


def _material_context(pdf_text: str) -> str:
    """Keep material-heavy parts of a long PDF instead of only the front."""
    if len(pdf_text) <= PDF_CONTEXT_MAX_CHARS:
        return pdf_text

    markers = [
        "appendix",
        "supplement",
        "stimuli",
        "stimulus",
        "materials",
        "measures",
        "manipulation",
        "question stem",
        "items were",
        "participants were asked",
    ]
    chunks: List[str] = [pdf_text[:50000]]
    appendix = _appendix_context(pdf_text, max_chars=90000)
    if appendix:
        chunks.append(appendix)
    chunks.append(pdf_text[-90000:])
    lowered = pdf_text.lower()
    for marker in markers:
        start = 0
        while True:
            idx = lowered.find(marker, start)
            if idx < 0:
                break
            chunks.append(pdf_text[max(0, idx - 12000): idx + 28000])
            start = idx + len(marker)
            if sum(len(c) for c in chunks) >= PDF_CONTEXT_MAX_CHARS * 2:
                break

    seen = set()
    deduped: List[str] = []
    for chunk in chunks:
        key = chunk[:500]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    text = "\n\n--- MATERIAL CONTEXT BREAK ---\n\n".join(deduped)
    if len(text) > PDF_CONTEXT_MAX_CHARS:
        text = text[:PDF_CONTEXT_MAX_CHARS]
    return text


def _appendix_blocks(pdf_text: str) -> List[Dict[str, str]]:
    """Return appendix sections as first-class material contexts.

    Many papers place exact stimuli and scale items only in appendices. Treating
    appendix text as ordinary tail text makes it easy to truncate away the
    participant-facing details after methods/results chunks are added.
    """
    text = str(pdf_text or "")
    if not text:
        return []
    matches = list(
        re.finditer(
            r"(?im)(?:^|\n)\s*(Appendix\s+[A-Z][^\n]{0,140}|Supplement(?:al|ary)?\s+(?:Materials?|Appendix)[^\n]{0,120})\s*(?=\n)",
            text,
        )
    )
    blocks: List[Dict[str, str]] = []
    if not matches:
        idx = text.lower().find("appendix")
        if idx >= 0:
            return [{"heading": "Appendix", "text": text[idx:]}]
        return []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = re.sub(r"\s+", " ", match.group(1)).strip()
        body = text[start:end].strip()
        if body:
            blocks.append({"heading": heading, "text": body})
    return blocks


def _appendix_context(
    pdf_text: str,
    *,
    study: Optional[Dict[str, Any]] = None,
    stage1_json: Optional[Dict[str, Any]] = None,
    max_chars: int = PDF_APPENDIX_CONTEXT_MAX_CHARS,
) -> str:
    blocks = _appendix_blocks(pdf_text)
    if not blocks:
        return ""
    terms = set(_study_context_terms(study, stage1_json)) if isinstance(study, dict) else set()
    priority_terms = {
        "measure",
        "measures",
        "item",
        "items",
        "stimuli",
        "stimulus",
        "manipulation",
        "question stem",
        "appendix",
    }

    def score(block: Dict[str, str]) -> tuple[int, int]:
        haystack = f"{block.get('heading', '')}\n{block.get('text', '')}".lower()
        value = 0
        for term in terms:
            if len(term) >= 4 and term in haystack:
                value += 3
        for term in priority_terms:
            if term in haystack:
                value += 2
        # Preserve original appendix order as a secondary key.
        return value, -len(haystack)

    ranked = sorted(enumerate(blocks), key=lambda item: (score(item[1]), -item[0]), reverse=True)
    chunks: List[str] = []
    total = 0
    per_block_limit = max(3000, min(9000, max_chars // max(1, min(len(ranked), 4))))
    for _, block in ranked:
        chunk = (
            f"--- PRIORITY APPENDIX MATERIAL BLOCK: {block.get('heading', 'Appendix')} ---\n"
            f"{block.get('text', '').strip()}"
        )
        if not chunk.strip():
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        chunks.append(chunk[:min(remaining, per_block_limit)])
        total += len(chunks[-1])
    return "\n\n".join(chunks)


def _appendix_retry_context(
    pdf_text: str,
    *,
    study: Optional[Dict[str, Any]],
    stage1_json: Optional[Dict[str, Any]],
) -> str:
    appendix = _appendix_context(
        pdf_text,
        study=study,
        stage1_json=stage1_json,
        max_chars=PDF_STUDY_CONTEXT_MAX_CHARS,
    )
    if not appendix:
        return ""
    return (
        "--- APPENDIX-ONLY MATERIAL RETRY CONTEXT ---\n"
        "The prior full-PDF material extraction failed. Use this appendix-only "
        "context to recover exact participant-facing stimuli, measures, items, "
        "conditions, options, and anchors.\n\n"
        f"{appendix}"
    )


def _should_appendix_retry(exc: BaseException) -> bool:
    if not isinstance(exc, ValueError):
        return False
    text = str(exc).lower()
    return "valid json" in text or "no response" in text or "response_preview='empty'" in text


def _clean_pdf_phrase(value: Any) -> str:
    text = str(value or "")
    text = (
        text.replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\u0081", "-")
        .replace("", "-")
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = _fix_pdf_ligature_splits(text)
    text = re.sub(r"\bCOVID-\s*19\b", "COVID-19", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"\s+([’'])", r"\1", text)
    text = re.sub(r"([’'])\s+", r"\1", text)
    text = re.sub(r"\s*/\s*", "/", text)
    return text.strip()


def _fix_pdf_ligature_splits(text: str) -> str:
    """Repair common PDF words split around fi/fl ligatures."""
    replacements = {
        "of fice": "office",
        "of fices": "offices",
        "of ficial": "official",
        "of ficials": "officials",
        "con firm": "confirm",
        "con firmed": "confirmed",
        "dif ficult": "difficult",
        "ef fect": "effect",
        "ef fects": "effects",
        "af fect": "affect",
        "af fects": "affects",
        "speci fic": "specific",
        "scienti fic": "scientific",
        "signi ficant": "significant",
        "coef ficient": "coefficient",
    }
    out = str(text or "")
    for broken, fixed in replacements.items():
        out = re.sub(rf"\b{re.escape(broken)}\b", fixed, out, flags=re.IGNORECASE)
    return out


def _clean_pdf_multiline(value: Any) -> str:
    text = str(value or "")
    text = (
        text.replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\u0081", "-")
        .replace("", "-")
    )
    lines = [_clean_pdf_phrase(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _strip_pdf_artifacts(value: str) -> str:
    text = str(value or "")
    markers = [
        "This document is copyrighted",
        "All rights, including",
        "This article is intended solely",
        "--- Page",
        "LEADERS' ANTI-ASIAN COMMUNICATION",
        "LEADERS’ ANTI-ASIAN COMMUNICATION",
        "JUN AND WU",
    ]
    cut = len(text)
    lowered = text.lower()
    for marker in markers:
        idx = lowered.find(marker.lower())
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut].strip()


def _study_number(value: Any) -> Optional[str]:
    match = re.search(r"\bstudy\s+([0-9]+[a-z]?)\b", str(value or ""), flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _stage_study_number(study: Dict[str, Any]) -> Optional[str]:
    for key in ("study", "study_id", "experiment_id", "study_name", "experiment_name"):
        number = _study_number(study.get(key))
        if number:
            return number
    return None


def _appendix_text_by_heading(pdf_text: str, heading_prefix: str) -> str:
    chunks = [
        block.get("text", "")
        for block in _appendix_blocks(pdf_text)
        if str(block.get("heading") or "").lower().startswith(heading_prefix.lower())
    ]
    return "\n\n".join(chunk for chunk in chunks if chunk).strip()


def _study_section_from_appendix(appendix_text: str, study_number: Optional[str]) -> str:
    text = str(appendix_text or "")
    if not text:
        return ""
    if not study_number:
        return text
    pattern = re.compile(rf"(?im)^\s*Study\s+{re.escape(study_number)}\b[^\n]*")
    match = pattern.search(text)
    if not match:
        return ""
    next_match = re.search(r"(?im)^\s*Study\s+[0-9]+[a-z]?\b[^\n]*", text[match.end():])
    end = match.end() + next_match.start() if next_match else len(text)
    return text[match.start():end].strip()


def _is_measure_heading(line: str) -> bool:
    clean = _clean_pdf_phrase(line)
    if not clean:
        return False
    lowered = clean.lower()
    if lowered in {"measures", "control variables"} or lowered.startswith(("study ", "appendix ")):
        return False
    if lowered.startswith(
        (
            "all measures",
            "we used",
            "we adapted",
            "to measure",
            "before measuring",
            "participants",
            "question items",
            "items were",
            "the items",
            "the question",
            "carelessness",
        )
    ):
        return False
    if re.search(r"[.;:?]\s*$", clean):
        return False
    words = clean.split()
    if not 1 <= len(words) <= 8:
        return False
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    titleish = sum(1 for word in alpha_words if word[:1].isupper())
    return bool(alpha_words and titleish >= max(1, len(alpha_words) // 2))


def _measure_sections(study_section: str) -> List[tuple[str, str]]:
    lines = _clean_pdf_multiline(_strip_pdf_artifacts(study_section)).splitlines()
    heading_indices = [
        (idx, line)
        for idx, line in enumerate(lines)
        if _is_measure_heading(line)
    ]
    sections: List[tuple[str, str]] = []
    for pos, (start, heading) in enumerate(heading_indices):
        end = heading_indices[pos + 1][0] if pos + 1 < len(heading_indices) else len(lines)
        body = "\n".join(lines[start + 1:end]).strip()
        if re.search(r"\b(?:question\s+)?items?\s+were\b", body, flags=re.IGNORECASE):
            sections.append((heading, body))
    return sections


def _scale_options(scale: Dict[str, Any]) -> List[str]:
    try:
        lo = int(float(scale["min"]))
        hi = int(float(scale["max"]))
    except (KeyError, TypeError, ValueError):
        return []
    anchors = scale.get("anchors") if isinstance(scale.get("anchors"), dict) else {}
    out: List[str] = []
    for value in range(lo, hi + 1):
        label = _clean_pdf_phrase(anchors.get(str(value), ""))
        out.append(f"{value} - {label}" if label and label != str(value) else str(value))
    return out


def _scale_from_text(text: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = dict(fallback or {})
    clean = _clean_pdf_phrase(text)
    match = re.search(
        r"\b([0-9]+)\s*=\s*(.+?)\s+to\s+([0-9]+)\s*=\s*(.+?)(?:[).,;]|$)",
        clean,
        flags=re.IGNORECASE,
    )
    if not match:
        return base
    lo, left, hi, right = match.groups()
    left = _clean_pdf_phrase(left).strip(" .")
    right = _clean_pdf_phrase(right).strip(" .")
    return {
        "min": int(lo),
        "max": int(hi),
        "anchors": {str(int(lo)): left, str(int(hi)): right},
    }


def _extract_stem(section_text: str) -> str:
    clean = _clean_pdf_phrase(section_text)
    patterns = [
        r"question stem read,?\s*[\"“](.+?)[\"”]",
        r"with the following question stem\s*[\"“](.+?)[\"”]",
        r"with the question stem,?\s*[\"“](.+?)[\"”]",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            stem = _clean_pdf_phrase(match.group(1))
            stem = re.sub(r"\?\.$", "?", stem)
            return stem
    return ""


def _split_lettered_items(items_text: str) -> List[str]:
    clean = _clean_pdf_phrase(items_text)
    matches = list(re.finditer(r"(?:^|[\s;,])(?:and\s+)?\(([a-z])\)\s*", clean, flags=re.IGNORECASE))
    items: List[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean)
        item = _clean_pdf_phrase(clean[start:end])
        item = re.sub(r"^(?:and|or)\s+", "", item, flags=re.IGNORECASE)
        item = item.strip(" ;,")
        item = re.sub(r"\.\s*[\"”]?$", "", item).strip()
        if item:
            items.append(item)
    return items


def _extract_items_text(section_text: str) -> str:
    clean = _clean_pdf_phrase(section_text)
    match = re.search(
        r"\b(?:Question\s+items|The\s+(?:three\s+)?items|Items)\s+were,?\s*(.+)$",
        clean,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    tail = match.group(1)
    tail = re.split(
        r"\b(?:To note|Carelessness Checks|Control Variables)\b|This document is copyrighted",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return tail.strip()


def _extract_measure_items(
    study_section: str,
    *,
    material_id: str,
    pdf_path: Path,
    source_label: str,
) -> List[Dict[str, Any]]:
    default_scale = _scale_from_text(study_section, {"min": 1, "max": 7, "anchors": {}})
    items: List[Dict[str, Any]] = []
    for heading, body in _measure_sections(study_section):
        stem = _extract_stem(body)
        scale = _scale_from_text(body, default_scale)
        options = _scale_options(scale)
        raw_items = _split_lettered_items(_extract_items_text(body))
        for item_index, item_text in enumerate(raw_items, start=1):
            reverse_coded = bool(re.search(r"reverse[- ]coded", item_text, flags=re.IGNORECASE))
            item_text = re.sub(r"\s*\(reverse[- ]coded\)\s*", "", item_text, flags=re.IGNORECASE).strip(" ;,.")
            question = _clean_pdf_phrase(f"{stem} {item_text}".strip()) if stem else _clean_pdf_phrase(item_text)
            item = {
                "id": f"{_slug(heading, material_id)}_{item_index}",
                "question": question,
                "options": options,
                "type": "likert",
                "scale": scale,
                "response_format": {
                    "answer_type": "likert",
                    "scale_min": scale.get("min"),
                    "scale_max": scale.get("max"),
                    "anchors": scale.get("anchors", {}),
                    "options": options,
                },
                "source": source_label,
                "source_file": str(pdf_path),
                "block": heading,
                "metadata": {
                    "construct": heading,
                    "reverse_coded": reverse_coded,
                    "evidence": {"section": "Appendix A", "heading": heading},
                },
            }
            items.append(item)
    return items


def _condition_label(value: str) -> str:
    text = _clean_pdf_phrase(value).replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:] if text else text


def _condition_block_study_number(block_text: str) -> Optional[str]:
    match = re.search(r"\bStudy\s+([0-9]+[a-z]?)\s+Manipulation\b", block_text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else _study_number(block_text)


def _extract_bracketed_condition(block_text: str) -> tuple[List[Dict[str, Any]], str]:
    text = _clean_pdf_multiline(_strip_pdf_artifacts(block_text))
    if not text or "[" not in text or "/" not in text:
        return [], ""
    label_match = re.search(r"\[([^/\[\]]+)/([^\[\]]+)\]\s+condition", text, flags=re.IGNORECASE)
    if not label_match:
        return [], ""
    levels = [_condition_label(label_match.group(1)), _condition_label(label_match.group(2))]
    body_match = re.search(r"\bTo:\s*.+$", text, flags=re.IGNORECASE | re.DOTALL)
    if not body_match:
        return [], ""
    template = body_match.group(0).strip()
    alt_re = re.compile(r"\[([^\[\]/]+)/([^\[\]]+)\]")
    descriptions: Dict[str, str] = {}
    for idx, level in enumerate(levels):
        def repl(match: re.Match[str]) -> str:
            return _clean_pdf_phrase(match.group(idx + 1))

        descriptions[level] = _clean_pdf_phrase(alt_re.sub(repl, template))
    condition = {
        "name": "label condition",
        "levels": levels,
        "level_descriptions": descriptions,
        "source": "pdf_appendix_parser",
        "source_file": "",
        "metadata": {
            "evidence": {"section": "Appendix B"},
            "stimulus_template": _clean_pdf_phrase(template),
        },
    }
    return [condition], _clean_pdf_phrase(template)


def _appendix_material_instructions(conditions: List[Dict[str, Any]], study_number: Optional[str]) -> str:
    if conditions:
        level_text = ", ".join(
            str(level)
            for condition in conditions
            for level in condition.get("levels", []) or []
        )
        return (
            "Read the team manager email assigned by the study condition "
            f"({level_text}). Then answer the following questions about your reactions "
            "and perceptions after reading the email."
        )
    label = f"Study {study_number}" if study_number else "the study"
    return f"Complete the participant-facing survey questions for {label}."


def _deterministic_appendix_materials(
    stage_json: Dict[str, Any],
    pdf_path: Path,
    pdf_text: str,
    *,
    source_label: str = "pdf_appendix_parser",
) -> Dict[str, Dict[str, Any]]:
    appendix_a = _appendix_text_by_heading(pdf_text, "Appendix A")
    if not appendix_a:
        return {}
    appendix_b = _appendix_text_by_heading(pdf_text, "Appendix B")
    condition_study = _condition_block_study_number(appendix_b) if appendix_b else None
    parsed_conditions, stimulus_template = _extract_bracketed_condition(appendix_b)
    for condition in parsed_conditions:
        condition["source_file"] = str(pdf_path)

    materials: Dict[str, Dict[str, Any]] = {}
    for study in _selected_or_all_studies(stage_json):
        if not isinstance(study, dict):
            continue
        study_name = study.get("study") or study.get("study_id") or study.get("experiment_id")
        study_number = _stage_study_number(study)
        study_section = _study_section_from_appendix(appendix_a, study_number)
        if not study_section:
            continue
        sub_id = _sub_study_id(study_name)
        items = _extract_measure_items(
            study_section,
            material_id=sub_id,
            pdf_path=pdf_path,
            source_label=source_label,
        )
        if not items:
            continue
        conditions = parsed_conditions if parsed_conditions and study_number == condition_study else []
        instructions = _appendix_material_instructions(conditions, study_number)
        warnings: List[str] = []
        blocking: List[str] = []
        if not conditions and any("condition" in str(effect.get("IV") or "").lower() for effect in study.get("effects", []) or [] if isinstance(effect, dict)):
            warnings.append("No appendix condition block matched this study; verify manipulated conditions manually.")
        if any(_has_ellipsis(item.get("question")) for item in items):
            blocking.append("truncated_pdf_item_text")
        ready = bool(instructions and items) and not blocking
        material = {
            "sub_study_id": sub_id,
            "instructions": instructions,
            "items": items,
            "conditions": conditions,
            "response_schema": _infer_response_schema(items),
            "readiness": {
                "ready": ready,
                "blocking_issues": blocking,
                "warnings": warnings,
            },
            "source_trace": {
                "primary_source": source_label,
                "source_file": str(pdf_path),
                "extractor": "stage3_pdf_appendix_parser_v1",
                "preserve_full_instrument_for_runtime": True,
                "conditions_source": "Appendix B" if conditions else None,
                "items_source": "Appendix A",
                "stimulus_template": stimulus_template if conditions else "",
                "evidence": {
                    "sections": ["Appendix A", *(["Appendix B"] if conditions else [])],
                    "study_section": f"Study {study_number}" if study_number else "",
                    "item_constructs": sorted({item.get("block") for item in items if item.get("block")}),
                },
            },
        }
        materials[sub_id] = material
    return materials


def _study_context_terms(study: Dict[str, Any], stage1_json: Optional[Dict[str, Any]]) -> List[str]:
    values: List[Any] = [
        study.get("study"),
        study.get("study_id"),
        study.get("experiment_id"),
        study.get("study_name"),
        study.get("experiment_name"),
        study.get("design") or study.get("design_type"),
    ]
    for key in ("conditions_or_factors", "participant_task", "input", "output"):
        values.append(study.get(key))
    for effect in study.get("effects", []) or []:
        if isinstance(effect, dict):
            values.extend([effect.get("IV"), effect.get("DV"), effect.get("claim")])
    for finding in study.get("findings", []) or []:
        if isinstance(finding, dict):
            values.extend([finding.get("IV"), finding.get("DV"), finding.get("claim")])

    stage1_hints = _stage1_hint_map(stage1_json)
    for key in (
        _sub_study_id(study.get("study")),
        _sub_study_id(study.get("study_id")),
        _sub_study_id(study.get("experiment_id")),
    ):
        hint = stage1_hints.get(key)
        if isinstance(hint, dict):
            values.extend(hint.values())

    terms: List[str] = []
    seen = set()
    for value in values[:5]:
        phrase = str(value or "").lower().strip()
        if len(phrase) >= 4 and phrase not in seen:
            seen.add(phrase)
            terms.append(phrase)
    for value in values:
        for token in _TOKEN_RE.findall(str(value or "")):
            token = token.lower().strip("'_-")
            if len(token) < 4 or token in _TARGET_STOP_TERMS or token.isdigit():
                continue
            if token not in seen:
                seen.add(token)
                terms.append(token)
    return terms[:40]


def _study_material_context(
    pdf_text: str,
    study: Dict[str, Any],
    stage1_json: Optional[Dict[str, Any]],
) -> str:
    """Keep a smaller, study-specific source context for one material call."""
    if len(pdf_text) <= PDF_STUDY_CONTEXT_MAX_CHARS:
        return pdf_text

    lowered = pdf_text.lower()
    chunks: List[str] = [pdf_text[:6000]]
    appendix = _appendix_context(
        pdf_text,
        study=study,
        stage1_json=stage1_json,
        max_chars=PDF_APPENDIX_CONTEXT_MAX_CHARS,
    )
    if appendix:
        chunks.append(appendix)
    study_terms = _study_context_terms(study, stage1_json)
    terms = [
        *study_terms[:12],
        "procedure",
        "method",
        "appendix",
        "supplement",
        "supplemental",
        "stimuli",
        "stimulus",
        "measures",
        "manipulation",
        "participants were asked",
        *study_terms[12:],
    ]
    total_chars = sum(len(chunk) for chunk in chunks)
    for term in terms:
        start = 0
        hits = 0
        needle = str(term or "").lower().strip()
        if len(needle) < 4:
            continue
        while hits < 3:
            idx = lowered.find(needle, start)
            if idx < 0:
                break
            chunks.append(pdf_text[max(0, idx - 3500): idx + 9000])
            total_chars += len(chunks[-1])
            hits += 1
            start = idx + len(needle)
            if total_chars >= PDF_STUDY_CONTEXT_MAX_CHARS * 2:
                break
        if total_chars >= PDF_STUDY_CONTEXT_MAX_CHARS * 2:
            break

    seen = set()
    deduped: List[str] = []
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk[:600]).strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    context = "\n\n--- STUDY MATERIAL CONTEXT BREAK ---\n\n".join(deduped)
    if len(context) > PDF_STUDY_CONTEXT_MAX_CHARS:
        context = context[:PDF_STUDY_CONTEXT_MAX_CHARS]
    return context


def _infer_response_schema(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    dominant = str(items[0].get("type") or "open_ended")
    schema: Dict[str, Any] = {"answer_type": dominant}
    options = items[0].get("options")
    if isinstance(options, list) and options:
        schema["options"] = options
    scale = items[0].get("scale") if isinstance(items[0].get("scale"), dict) else {}
    if scale:
        if scale.get("min") is not None:
            schema["scale_min"] = scale.get("min")
        if scale.get("max") is not None:
            schema["scale_max"] = scale.get("max")
        anchors = scale.get("anchors")
        if isinstance(anchors, dict) and anchors:
            schema["anchors"] = anchors
    return schema


def _normalize_item(raw: Dict[str, Any], material_id: str, index: int) -> Dict[str, Any]:
    item_type = str(raw.get("type") or raw.get("answer_type") or "open_ended").strip().lower()
    if item_type not in _VALID_TYPES:
        item_type = "open_ended"
    options = raw.get("options")
    if not isinstance(options, list):
        options = []
    options = [_text(option) for option in options if _text(option)]
    scale = raw.get("scale") if isinstance(raw.get("scale"), dict) else {}
    anchors = raw.get("anchors") if isinstance(raw.get("anchors"), dict) else {}
    if anchors and not scale:
        scale = {"anchors": anchors}
    item = {
        "id": _slug(raw.get("id") or raw.get("data_export_tag") or f"{material_id}_item_{index}", f"{material_id}_item_{index}"),
        "question": _text(raw.get("question") or raw.get("text") or raw.get("label")),
        "type": item_type,
        "options": options,
    }
    if scale:
        item["scale"] = scale
    response_format = raw.get("response_format") if isinstance(raw.get("response_format"), dict) else {}
    if response_format:
        item["response_format"] = response_format
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
    if evidence:
        metadata = {**metadata, "evidence": evidence}
    if metadata:
        item["metadata"] = metadata
    return item


def _token_stem(token: str) -> str:
    token = token.lower().strip("'_-")
    for suffix in ("iveness", "ation", "ness", "ment", "ity", "ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break
    return token[:6] if len(token) > 6 else token


def _target_stems(*values: Any) -> set[str]:
    stems: set[str] = set()
    for value in values:
        for token in _TOKEN_RE.findall(str(value or "").lower()):
            token = token.strip("'_-")
            if len(token) < 4 or token in _TARGET_STOP_TERMS or token.isdigit():
                continue
            stems.add(_token_stem(token))
    return stems


def _study_lookup_key(value: Any) -> str:
    return _sub_study_id(value)


def _matching_stage_study(stage_json: Dict[str, Any], sub_id: str, study_name: Any) -> Optional[Dict[str, Any]]:
    studies = [
        study
        for study in stage_json.get("eligible_studies", []) or stage_json.get("studies", []) or []
        if isinstance(study, dict)
    ]
    wanted = {_study_lookup_key(sub_id), _study_lookup_key(study_name)}
    for study in studies:
        keys = {
            _study_lookup_key(study.get("study")),
            _study_lookup_key(study.get("study_id")),
            _study_lookup_key(study.get("experiment_id")),
        }
        if wanted & keys:
            return study
    return studies[0] if len(studies) == 1 else None


def _target_dv_values(study: Optional[Dict[str, Any]]) -> List[Any]:
    if not isinstance(study, dict):
        return []

    findings = [finding for finding in study.get("findings", []) or [] if isinstance(finding, dict)]
    candidate_findings = [
        finding
        for finding in findings
        if isinstance(finding.get("simulation_target"), dict)
        and finding["simulation_target"].get("candidate") is True
    ]
    role_findings = [
        finding
        for finding in findings
        if str(finding.get("role") or "").lower() in {"primary_finding", "interaction", "main_effect"}
    ]
    selected_findings = candidate_findings or role_findings or findings
    values = [finding.get("DV") for finding in selected_findings if finding.get("DV")]

    effects = [effect for effect in study.get("effects", []) or [] if isinstance(effect, dict)]
    for effect in effects:
        consolidation = effect.get("consolidation") if isinstance(effect.get("consolidation"), dict) else {}
        if consolidation.get("is_primary_simulation_target") is True and consolidation.get("is_representative", True) is not False:
            if effect.get("DV"):
                values.append(effect.get("DV"))

    return values


def _item_match_score(item: Dict[str, Any], stems: set[str]) -> int:
    if not stems:
        return 0
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("id", "data_export_tag", "question")
    )
    item_stems = _target_stems(haystack)
    score = len(item_stems & stems)
    tag = str(item.get("data_export_tag") or item.get("id") or "").lower()
    tag_parts = [_token_stem(part) for part in re.split(r"[^a-z0-9]+", tag) if part]
    for stem in stems:
        if any(part and (stem.startswith(part) or part.startswith(stem[:4])) for part in tag_parts):
            score += 2
    return score


def _non_response_item_reason(item: Dict[str, Any]) -> Optional[str]:
    item_id = str(item.get("id") or item.get("data_export_tag") or "")
    question = _text(item.get("question"))
    haystack = f"{item_id} {question}"
    if _ADMIN_ITEM_RE.search(haystack):
        return "admin_or_demographic_item"
    if _CHECK_ITEM_RE.search(haystack):
        return "attention_or_manipulation_check"
    item_type = str(item.get("type") or "").lower()
    if item_type == "text" and not item.get("options") and not item.get("scale"):
        return "display_text_not_response_item"
    return None


def _filter_pdf_response_items(
    items: List[Dict[str, Any]],
    *,
    stage_json: Dict[str, Any],
    sub_id: str,
    study_name: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    report: Dict[str, Any] = {
        "mode": "none",
        "dropped_non_response": [],
        "dropped_non_target": [],
        "target_stems": [],
        "kept": [],
    }

    substantive: List[Dict[str, Any]] = []
    for item in items:
        reason = _non_response_item_reason(item)
        item_id = str(item.get("data_export_tag") or item.get("id") or item.get("question") or "")
        if reason:
            report["dropped_non_response"].append({"id": item_id, "reason": reason})
            continue
        substantive.append(item)

    study = _matching_stage_study(stage_json, sub_id, study_name)
    stems = _target_stems(*_target_dv_values(study))
    report["target_stems"] = sorted(stems)
    scored = [(item, _item_match_score(item, stems)) for item in substantive]
    target_items = [item for item, score in scored if score > 0]

    if target_items and len(target_items) < len(substantive):
        target_ids = {id(item) for item in target_items}
        report["mode"] = "target_dv_terms"
        report["dropped_non_target"] = [
            str(item.get("data_export_tag") or item.get("id") or item.get("question") or "")
            for item in substantive
            if id(item) not in target_ids
        ]
        report["kept"] = [
            str(item.get("data_export_tag") or item.get("id") or item.get("question") or "")
            for item in target_items
        ]
        return target_items, report

    report["mode"] = "non_response_removed" if report["dropped_non_response"] else "all_items"
    report["kept"] = [
        str(item.get("data_export_tag") or item.get("id") or item.get("question") or "")
        for item in substantive
    ]
    return substantive, report


def _tag_pdf_item_sources(items: List[Dict[str, Any]], pdf_path: Path, source_label: str) -> None:
    source_file = str(pdf_path)
    for item in items:
        item.setdefault("source", source_label)
        item.setdefault("source_file", source_file)


def _normalize_conditions(raw_conditions: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw_conditions, list):
        return out
    for idx, cond in enumerate(raw_conditions, start=1):
        if not isinstance(cond, dict):
            continue
        levels = cond.get("levels")
        if not isinstance(levels, list):
            levels = []
        item = {
            "name": _text(cond.get("name") or cond.get("label") or f"condition_{idx}"),
            "levels": [_text(level) for level in levels if _text(level)],
        }
        descriptions = cond.get("level_descriptions")
        if isinstance(descriptions, dict):
            item["level_descriptions"] = {
                _text(k): _text(v) for k, v in descriptions.items() if _text(k) and _text(v)
            }
        out.append(item)
    return out


def _study_mentions(value: Any) -> set[str]:
    text = str(value or "")
    return {
        match.group(1).lower()
        for match in re.finditer(r"\bstudy\s+([0-9]+[a-z]?)\b", text, flags=re.IGNORECASE)
    }


def _expected_study_mentions(stage_json: Dict[str, Any], sub_id: str, study_name: Any) -> set[str]:
    study = _matching_stage_study(stage_json, sub_id, study_name)
    if not isinstance(study, dict):
        return _study_mentions(study_name)
    mentions: set[str] = set()
    for key in ("study", "study_id", "experiment_id", "study_name", "experiment_name"):
        mentions |= _study_mentions(study.get(key))
    return mentions or _study_mentions(study_name)


def _has_cross_study_evidence(
    *,
    stage_json: Dict[str, Any],
    sub_id: str,
    study_name: Any,
    instructions: str,
    evidence: Dict[str, Any],
) -> bool:
    expected = _expected_study_mentions(stage_json, sub_id, study_name)
    if not expected:
        return False
    evidence_text = instructions + "\n" + json.dumps(evidence, ensure_ascii=False)
    mentioned = _study_mentions(evidence_text)
    return bool(mentioned and not (mentioned & expected))


def _normalize_material(
    raw: Dict[str, Any],
    pdf_path: Path,
    stage_json: Dict[str, Any],
    *,
    source_label: str = "pdf_llm",
) -> Optional[Dict[str, Any]]:
    study_name = raw.get("study") or raw.get("study_id") or raw.get("sub_study_id")
    sub_id = _sub_study_id(study_name)
    instructions = _text(raw.get("instructions") or raw.get("stimulus") or raw.get("vignette"))
    material_id = sub_id
    items = [
        _normalize_item(item, material_id, idx)
        for idx, item in enumerate(raw.get("items") or [], start=1)
        if isinstance(item, dict)
    ]
    items = [item for item in items if item.get("question")]
    items, item_filter = _filter_pdf_response_items(
        items,
        stage_json=stage_json,
        sub_id=sub_id,
        study_name=study_name,
    )
    _tag_pdf_item_sources(items, pdf_path, source_label)
    conditions = _normalize_conditions(raw.get("conditions"))

    completeness = str(raw.get("completeness") or "partial").strip().lower()
    evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
    blocking: List[str] = []
    warnings: List[str] = []
    if not instructions:
        blocking.append("missing_instructions")
    if not items:
        blocking.append("no_participant_items")
    if completeness in {"none", "missing"}:
        blocking.append("pdf_material_not_found")
    elif completeness not in {"complete", "ready"}:
        blocking.append("pdf_material_incomplete")
    if _has_cross_study_evidence(
        stage_json=stage_json,
        sub_id=sub_id,
        study_name=study_name,
        instructions=instructions,
        evidence=evidence,
    ):
        blocking.append("pdf_material_cross_study_evidence")
        warnings.append("PDF material evidence mentions a different study than the target study.")
    if _META_INSTRUCTION_RE.search(instructions):
        blocking.append("pdf_material_not_participant_facing")
        warnings.append("PDF material instructions contain source commentary rather than participant-facing wording.")
    if any(_has_ellipsis(item.get("question")) or any(_has_ellipsis(opt) for opt in item.get("options", [])) for item in items):
        blocking.append("truncated_pdf_item_text")
        warnings.append("PDF material extraction contains ellipsis markers; recover full wording before simulation.")
    if raw.get("verbatim") is False:
        warnings.append("PDF material extractor marked the material as non-verbatim or reconstructed.")
    if raw.get("missing"):
        warnings.append("Missing PDF material details: " + _text(raw.get("missing")))

    ready = bool(instructions and items) and not blocking
    return {
        "sub_study_id": sub_id,
        "instructions": instructions,
        "items": items,
        "conditions": conditions,
        "response_schema": _infer_response_schema(items),
        "readiness": {
            "ready": ready,
            "blocking_issues": sorted(dict.fromkeys(blocking)),
            "warnings": warnings,
        },
        "source_trace": {
            "primary_source": source_label,
            "source_file": str(pdf_path),
            "extractor": "stage3_pdf_materials_v1",
            "evidence": evidence,
            "item_filter": item_filter,
        },
    }


def build_pdf_material_prompt(
    stage_json: Dict[str, Any],
    pdf_text: str,
    stage1_json: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the PDF material extractor prompt. Exposed for unit tests."""
    return f"""Extract participant-facing study materials from this psychology/management paper PDF or supplementary source document.

Use the Stage 2 study/finding inventory only to decide which studies and
outcomes matter. Extract materials from the source text itself. Do not invent
question wording, options, anchors, condition labels, or vignettes. If the PDF
only names a cited scale without item wording, mark that study as partial or
missing. Do not use ellipses; if wording is truncated, mark it incomplete.
Put display text, vignettes, and stimuli in instructions/conditions, not in
items. Exclude attention checks, careless-response checks, comprehension
checks, manipulation checks, consent, demographics, timers, and debriefing from
items unless Stage 2 explicitly makes that construct the simulation target.

For each eligible study, return a study-level material package matching the
HumanStudy-Bench material shape:
- instructions: participant-facing task framing, scenario, vignette, or stimulus.
- items: participant response questions/items with type, options or scale/anchors.
- conditions: manipulated/measured factors and condition levels.
- completeness: complete, partial, or none.
- evidence: short source sections/pages/quotes proving the extraction.

SOURCE TEXT:
{pdf_text}

STAGE 1/2 STUDY, TARGET, AND SOURCE-HINT INVENTORY:
{json.dumps(_compact_studies(stage_json, stage1_json), ensure_ascii=False, indent=2)}

Return ONLY JSON with this schema:
{{
  "materials": [
    {{
      "study": "Study 1",
      "instructions": "<participant-facing instructions or stimulus text>",
      "items": [
        {{
          "id": "interpersonal_justice_1",
          "question": "<full participant-facing question text>",
          "type": "likert|scale|slider|multiple_choice|open_ended|ranking|matrix|text",
          "options": [],
          "scale": {{"min": 1, "max": 7, "anchors": {{"1": "Never", "7": "Always"}}}},
          "evidence": {{"section": "Appendix A"}}
        }}
      ],
      "conditions": [
        {{
          "name": "label condition",
          "levels": ["neutral labels", "stigmatizing labels"],
          "level_descriptions": {{"neutral labels": "<neutral stimulus text>", "stigmatizing labels": "<stigmatizing stimulus text>"}}
        }}
      ],
      "completeness": "complete|partial|none",
      "verbatim": true,
      "missing": "",
      "evidence": {{"sections": ["Appendix A"], "quotes": ["short quote"]}}
    }}
  ],
  "notes": "short extraction note"
}}"""


def extract_pdf_study_materials(
    stage_json: Dict[str, Any],
    pdf_path: Path,
    llm_client: Any,
    *,
    stage1_json: Optional[Dict[str, Any]] = None,
    pdf_text: Optional[str] = None,
    source_label: str = "pdf_llm",
    timeout: Optional[float] = 60.0,
    max_attempts: int = 2,
    retry_delay: float = 1.0,
    only_sub_study_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    text = pdf_text if pdf_text is not None else extract_pdf_text(Path(pdf_path), max_chars=PDF_TEXT_MAX_CHARS)
    studies = _selected_or_all_studies(stage_json)
    if only_sub_study_id:
        studies = [study for study in studies if _study_matches_sub_id(study, only_sub_study_id)]
        stage_json = _stage_json_with_studies(stage_json, studies)
        if not studies:
            return {}

    deterministic = _deterministic_appendix_materials(
        stage_json,
        Path(pdf_path),
        text,
        source_label="pdf_appendix_parser",
    )
    if deterministic:
        target_ids = {
            _sub_study_id(study.get("study") or study.get("study_id") or study.get("experiment_id"))
            for study in studies
            if isinstance(study, dict)
        }
        if llm_client is None or target_ids <= set(deterministic):
            return deterministic
    if llm_client is None:
        return deterministic or {}
    if len(studies) > 1:
        try:
            materials = _extract_pdf_study_materials_by_study(
                stage_json,
                Path(pdf_path),
                llm_client,
                text,
                studies=studies,
                stage1_json=stage1_json,
                source_label=source_label,
                timeout=timeout,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
            )
        except Exception:
            if deterministic:
                return deterministic
            raise
        materials.update(deterministic)
        return materials

    context = _material_context(text)
    prompt = build_pdf_material_prompt(stage_json, context, stage1_json=stage1_json)
    try:
        parsed = _generate_json_with_retries(
            llm_client,
            prompt=prompt,
            timeout=timeout,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
        )
    except Exception as exc:
        if not _should_appendix_retry(exc):
            raise
        retry_context = _appendix_retry_context(
            text,
            study=studies[0] if studies else None,
            stage1_json=stage1_json,
        )
        if not retry_context:
            raise
        parsed = _generate_json_with_retries(
            llm_client,
            prompt=build_pdf_material_prompt(stage_json, retry_context, stage1_json=stage1_json),
            timeout=timeout,
            max_attempts=1,
            retry_delay=retry_delay,
        )
    return _normalize_pdf_material_response(parsed, Path(pdf_path), stage_json, source_label=source_label)


def _normalize_pdf_material_response(
    parsed: Dict[str, Any],
    pdf_path: Path,
    stage_json: Dict[str, Any],
    *,
    source_label: str,
) -> Dict[str, Dict[str, Any]]:
    materials: Dict[str, Dict[str, Any]] = {}
    for raw in parsed.get("materials") or []:
        if not isinstance(raw, dict):
            continue
        material = _normalize_material(raw, Path(pdf_path), stage_json, source_label=source_label)
        if material is None:
            continue
        materials[material["sub_study_id"]] = material
    return materials


def _extract_pdf_study_materials_by_study(
    stage_json: Dict[str, Any],
    pdf_path: Path,
    llm_client: Any,
    pdf_text: str,
    *,
    studies: List[Dict[str, Any]],
    stage1_json: Optional[Dict[str, Any]],
    source_label: str,
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
) -> Dict[str, Dict[str, Any]]:
    materials: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    total = len(studies)
    for index, study in enumerate(studies, start=1):
        single_stage = _stage_json_with_studies(stage_json, [study])
        context = _study_material_context(pdf_text, study, stage1_json)
        study_name = study.get("study") or study.get("study_id") or study.get("experiment_id") or "unknown"
        print(
            f"  Stage 3 PDF material extraction {index}/{total}: {study_name} "
            f"({len(context)} chars)",
            flush=True,
        )
        prompt = build_pdf_material_prompt(
            single_stage,
            context,
            stage1_json=stage1_json,
        )
        try:
            parsed = _generate_json_with_retries(
                llm_client,
                prompt=prompt,
                timeout=timeout,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
            )
            materials.update(
                _normalize_pdf_material_response(
                    parsed,
                    pdf_path,
                    single_stage,
                    source_label=source_label,
                )
            )
        except Exception as exc:
            if not _should_appendix_retry(exc):
                errors.append(f"{study_name}: {type(exc).__name__}: {exc}")
                continue
            retry_context = _appendix_retry_context(
                pdf_text,
                study=study,
                stage1_json=stage1_json,
            )
            if retry_context:
                print(
                    f"  Stage 3 PDF appendix-only retry {index}/{total}: {study_name} "
                    f"({len(retry_context)} chars)",
                    flush=True,
                )
                try:
                    parsed = _generate_json_with_retries(
                        llm_client,
                        prompt=build_pdf_material_prompt(
                            single_stage,
                            retry_context,
                            stage1_json=stage1_json,
                        ),
                        timeout=timeout,
                        max_attempts=1,
                        retry_delay=retry_delay,
                    )
                    materials.update(
                        _normalize_pdf_material_response(
                            parsed,
                            pdf_path,
                            single_stage,
                            source_label=source_label,
                        )
                    )
                    continue
                except Exception as retry_exc:
                    errors.append(
                        f"{study_name}: {type(exc).__name__}: {exc}; "
                        f"appendix_retry={type(retry_exc).__name__}: {retry_exc}"
                    )
                    continue
            errors.append(f"{study_name}: {type(exc).__name__}: {exc}")
            continue
    if materials:
        return materials
    if errors:
        raise RuntimeError("Stage 3 PDF material extraction failed for all study-level calls: " + " | ".join(errors))
    return {}


def _generate_json_with_retries(
    llm_client: Any,
    *,
    prompt: str,
    timeout: Optional[float],
    max_attempts: int,
    retry_delay: float,
) -> Dict[str, Any]:
    attempts = max(1, int(max_attempts or 1))
    current_prompt = prompt
    last_error: Optional[BaseException] = None
    last_parse_error: Optional[BaseException] = None
    last_preview = "empty"
    for attempt in range(1, attempts + 1):
        try:
            response = call_with_timeout(
                lambda: _generate_once(llm_client, prompt=current_prompt, timeout=timeout),
                timeout=timeout,
            )
        except Exception as exc:
            last_error = exc
            if type(exc).__name__ == "LLMTimeoutError":
                break
            if attempt >= attempts:
                break
            if retry_delay > 0:
                time.sleep(retry_delay)
            continue

        text = str(response or "").strip()
        last_preview = text[:700] if text else "empty"
        try:
            return _loads_json(text)
        except Exception as exc:
            last_parse_error = exc
            if attempt >= attempts:
                break
            current_prompt = (
                prompt
                + "\n\nYour previous answer was not parseable JSON. "
                + "Return ONLY the JSON object matching the requested schema; no prose, no markdown, no commentary."
            )
            if retry_delay > 0:
                time.sleep(retry_delay)
    if last_parse_error is not None:
        raise ValueError(
            "Stage 3 PDF material extractor did not return valid JSON: "
            f"{type(last_parse_error).__name__}: {last_parse_error}; response_preview={last_preview!r}"
        ) from last_parse_error
    if last_error is not None:
        raise last_error
    raise ValueError("Stage 3 PDF material extractor returned no response.")


def _generate_once(llm_client: Any, *, prompt: str, timeout: Optional[float]) -> Any:
    try:
        return llm_client.generate_content(prompt=prompt, timeout=timeout, max_tokens=PDF_MATERIAL_MAX_TOKENS)
    except TypeError:
        try:
            return llm_client.generate_content(prompt=prompt, timeout=timeout)
        except TypeError:
            # Some local test/fallback clients do not accept a timeout kwarg.
            return llm_client.generate_content(prompt=prompt)
