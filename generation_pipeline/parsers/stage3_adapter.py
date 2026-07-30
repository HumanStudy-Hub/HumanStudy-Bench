from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from generation_pipeline.parsers.source_linker import study_tokens

# Status ranked by trust; used to pick the best when several effects disagree.
_STATUS_RANK = {
    "verbatim": 0,
    "osf_only": 1,
    "paraphrased": 2,
    "not_in_paper": 8,
    "source_missing": 8,
    None: 9,
}
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_'-]{2,}", re.IGNORECASE)
_STOP_TERMS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about",
    "study", "studies", "participants", "participant", "condition",
    "conditions", "message", "messages", "article", "articles", "measure",
    "measures", "item", "items", "scale", "were", "was", "are", "they",
    "their", "then", "after", "before", "effect", "effects",
    "one", "two", "one-sided", "two-sided", "sidedness", "side", "sides",
    "author", "authors", "argued", "arguments", "acknowledged", "briefly",
    "reasonable", "points", "same", "first", "fact", "few", "some", "such",
    "version", "versions", "written", "read", "presented",
}
_PARTICIPANT_TEXT_RE = re.compile(
    r"\b(participants?|you|your|please|read|rate|respond|answer|choose|"
    r"write|presented|assigned|told|condition|scenario|message|article|task)\b",
    re.IGNORECASE,
)
_ANALYSIS_TEXT_RE = re.compile(
    r"\b(regression|model|anova|coefficient|hypothes(?:is|ized|es)|"
    r"independent variables?|dependent variable|p\s*[<=>]|B\s*=|t\(|"
    r"partial\s+η|figure\s+\d+|results?)\b",
    re.IGNORECASE,
)


def load_stage3_studies(stage3_path: str | Path) -> List[Dict[str, Any]]:
    """Return the list of study dicts from a stage3.json file."""
    data = json.loads(Path(stage3_path).read_text(encoding="utf-8"))
    return data.get("eligible_studies") or data.get("studies") or []


def match_stage3_study(study_id: str, studies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the Stage 3 study whose name shares a study-token with `study_id`."""
    want = study_tokens(study_id)
    if not want:
        return None
    for s in studies:
        name = s.get("study") or s.get("study_id") or s.get("experiment_id") or ""
        if study_tokens(str(name)) & want:
            return s
    return None


def _slot(effect: Dict[str, Any], name: str) -> tuple[Optional[str], str]:
    """Return (status, content) for a slot, tolerating string or dict shape."""
    v = effect.get(name)
    if isinstance(v, dict):
        return v.get("status"), _clean_slot_text(v.get("content"))
    if isinstance(v, str):
        return ("verbatim" if v.strip() else None), _clean_slot_text(v)
    return None, ""


def _clean_slot_text(value: Any) -> str:
    text = str(value or "").replace("…", " ").replace("...", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if re.match(r"^[a-z]{2,}\b", text):
        colon = text.find(":")
        if 0 < colon < 240:
            after = text[colon + 1:].strip()
            if re.match(r'^[“"A-Z]|\b(please|imagine|read|rate|you|your)\b', after, flags=re.IGNORECASE):
                text = after
    return text


def _status_rank(status: Optional[str]) -> int:
    return _STATUS_RANK.get(status, 5)


def _text_tokens(text: str) -> set[str]:
    return {
        tok.strip("'_-")
        for tok in _TOKEN_RE.findall(str(text or "").lower())
        if tok.strip("'_-") not in _STOP_TERMS and not tok.isdigit()
    }


def _similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _is_analysis_text(text: str) -> bool:
    return bool(_ANALYSIS_TEXT_RE.search(text))


def _is_participant_text(text: str) -> bool:
    return bool(_PARTICIPANT_TEXT_RE.search(text))


def _effect_is_primary(effect: Dict[str, Any]) -> bool:
    cons = effect.get("consolidation")
    if isinstance(cons, dict):
        return bool(cons.get("is_primary_simulation_target", True)) and cons.get("is_representative", True) is not False
    return True


def _slot_records(effects: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    source_effects = [effect for effect in effects if _effect_is_primary(effect)] or effects
    for idx, effect in enumerate(source_effects):
        status, content = _slot(effect, name)
        if not content:
            continue
        if name in {"materials", "manipulation"} and (_is_analysis_text(content) or not _is_participant_text(content)):
            continue
        records.append({
            "status": status or "unknown",
            "content": content,
            "tokens": _text_tokens(content),
            "index": idx,
        })
    return records


def _dedup_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        key = re.sub(r"\s+", " ", rec["content"].lower()).strip()
        if key not in best or _status_rank(rec["status"]) < _status_rank(best[key]["status"]):
            best[key] = rec
    return list(best.values())


def _coherent_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = _dedup_records(records)
    if len(records) <= 1:
        return records

    clusters: List[List[Dict[str, Any]]] = []
    for rec in records:
        placed = False
        for cluster in clusters:
            if any(_similarity(rec["tokens"], other["tokens"]) >= 0.16 for other in cluster):
                cluster.append(rec)
                placed = True
                break
        if not placed:
            clusters.append([rec])

    def score(cluster: List[Dict[str, Any]]) -> tuple[int, int, int]:
        status_bonus = sum(max(0, 8 - _status_rank(rec["status"])) for rec in cluster)
        length_bonus = min(3000, sum(len(rec["content"]) for rec in cluster)) // 250
        return (len(cluster), status_bonus, length_bonus)

    chosen = max(clusters, key=score)
    return sorted(chosen, key=lambda rec: (_status_rank(rec["status"]), rec["index"]))


def _best_unique(slot_values: List[tuple[Optional[str], str]]) -> List[tuple[str, str]]:
    """Dedup (status, content) by normalized content, keeping best status."""
    best: Dict[str, tuple[str, str]] = {}
    for status, content in slot_values:
        if not content:
            continue
        key = re.sub(r"\s+", " ", content.lower()).strip()
        rank = _status_rank(status)
        if key not in best or rank < _status_rank(best[key][0]):
            best[key] = (status or "unknown", content)
    return list(best.values())


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return s or fallback


def _condition_levels_from_texts(texts: List[str]) -> List[Dict[str, Any]]:
    """Extract generic condition-level descriptions from participant text."""
    level_map: Dict[str, List[str]] = {}
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(texts))
    current_level: Optional[str] = None
    for sentence in sentences:
        sent = sentence.strip()
        if not sent:
            continue
        match = re.search(
            r"\b(?:in|under|for)\s+the\s+([A-Za-z0-9][A-Za-z0-9 /_-]{1,70}?\s+condition)\b",
            sent,
            flags=re.IGNORECASE,
        )
        if match:
            current_level = re.sub(r"\s+", " ", match.group(1)).strip()
            level_map.setdefault(current_level, []).append(sent)
            continue
        if current_level and re.match(r"^(?:In addition|Then|Next|The author|Participants|They)\b", sent):
            level_map[current_level].append(sent)
    if len(level_map) < 2:
        return []
    return [
        {
            "level": level,
            "description": " ".join(parts),
            "source": "pdf_stage3_slot",
        }
        for level, parts in level_map.items()
    ]


def materials_from_stage3(stage3_study: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidate a Stage 3 study's effect slots into a materials block.

    Returns {instructions, items, conditions, slot_summary} where each item and
    the instructions carry their Stage 3 `status` so trust is visible downstream.
    """
    effects = [e for e in stage3_study.get("effects", []) if isinstance(e, dict)]

    materials = [(rec["status"], rec["content"]) for rec in _coherent_records(_slot_records(effects, "materials"))]
    manipulations = [(rec["status"], rec["content"]) for rec in _coherent_records(_slot_records(effects, "manipulation"))]

    # instructions = stimulus (materials) followed by manipulation framing
    instr_parts: List[str] = [c for _, c in materials] + [c for _, c in manipulations]
    instructions = "\n\n".join(instr_parts)

    # items: one per distinct DV measure wording, labelled by the effect's DV
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for i, e in enumerate(effects, start=1):
        status, content = _slot(e, "items")
        if not content:
            continue
        key = re.sub(r"\s+", " ", content.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        dv = e.get("DV") or f"measure {i}"
        items.append({
            "id": _slug(dv, f"item_{i}"),
            "question": content,
            "options": [],
            "type": "open_ended",
            "source": "pdf_stage3",
            "source_file": "stage3.json",
            "data_export_tag": None,
            "slot_status": status or "unknown",
            "dv": dv,
        })

    # conditions from manipulation slots (named, when distinguishable)
    condition_levels = _condition_levels_from_texts([c for _, c in manipulations])
    conditions: List[Dict[str, Any]] = []
    if condition_levels:
        conditions.append({
            "name": "pdf_stage3_conditions",
            "levels": [level["level"] for level in condition_levels],
            "level_descriptions": {
                level["level"]: level["description"] for level in condition_levels
            },
            "source": "pdf_stage3_slot",
            "review_required": True,
        })

    instruction_statuses = [status for status, _ in materials + manipulations]
    best_instruction_status = (
        min(instruction_statuses, key=_status_rank) if instruction_statuses else "source_missing"
    )

    return {
        "instructions": instructions,
        "items": items,
        "conditions": conditions,
        "slot_summary": {
            "instructions_status": best_instruction_status,
            "materials_status": materials[0][0] if materials else "source_missing",
            "manipulation_status": manipulations[0][0] if manipulations else "source_missing",
            "items_status": items[0]["slot_status"] if items else "source_missing",
            "n_effects": len(effects),
        },
    }
