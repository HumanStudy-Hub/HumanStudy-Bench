from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from generation_pipeline.identifiers import canonical_sub_study_id

from generation_pipeline.parsers.qsf_parser import ParsedSurvey, parse_qsf_file
from generation_pipeline.parsers.sav_parser import ParsedDataset, parse_sav_file
from generation_pipeline.parsers.source_linker import LinkedFile, LinkResult
from generation_pipeline.parsers.stage3_adapter import materials_from_stage3

# Blocks / items that are survey plumbing, not the experimental task.
_ADMIN_BLOCK_RE = re.compile(
    r"consent|debrief|demograph|prolific|mturk|\battention\b|attn|comprehension|"
    r"manipulation check|^instructions?$|practice|captcha|short demos?|timing",
    re.IGNORECASE,
)
_ADMIN_ITEM_RE = re.compile(
    r"^(?:english|gender|sex|age|race|ethnic|education|income|prolific|mturk|"
    r"attention|attn|duration|finished|progress|obj_?side)$",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_'-]{2,}", re.IGNORECASE)
_TARGET_STOP_TERMS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about",
    "study", "studies", "effect", "effects", "condition", "conditions",
    "measure", "measures", "item", "items", "index", "scale", "article",
    "message", "messages", "counter", "attitudinal", "dependent", "variable",
    "variables", "participant", "participants", "toward", "after", "before",
    "interaction", "direct", "indirect",
}


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return s or fallback


def _is_admin_block(name: Optional[str]) -> bool:
    return bool(name and _ADMIN_BLOCK_RE.search(name))


def _has_truncation_marker(text: str) -> bool:
    return bool(text and ("..." in text or "…" in text))


def _scale_key_text(value: Any) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return str(int(n)) if n == int(n) else str(n)


def _clean_scale_label(key: Any, label: Any) -> str:
    key_text = _scale_key_text(key)
    text = re.sub(r"\s+", " ", str(label or "")).strip()
    if not text:
        return key_text

    leading = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*(.*)$", text)
    if leading:
        leading_key = _scale_key_text(leading.group(1))
        rest = leading.group(2).strip()
        if leading_key == key_text:
            rest = re.sub(
                r"^(?:[?=:.\-]|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|\u2212)+\s*",
                "",
                rest,
            ).strip()
            return f"{key_text} - {rest}" if rest else key_text

    return f"{key_text} - {text}" if text != key_text else key_text


def _normalized_scale_anchors(scale: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(scale, dict) or not isinstance(scale.get("anchors"), dict):
        return {}
    return {
        _scale_key_text(key): _clean_scale_label(key, value)
        for key, value in scale.get("anchors", {}).items()
    }


def _scale_sort_key(value: Any) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _scale_options(scale: Optional[Dict[str, Any]], *, enumerate_discrete: bool) -> List[str]:
    if not isinstance(scale, dict):
        return []
    anchors = _normalized_scale_anchors(scale)
    try:
        lo = int(float(scale["min"]))
        hi = int(float(scale["max"]))
    except (KeyError, TypeError, ValueError):
        return [str(v) for _, v in sorted(anchors.items(), key=lambda item: _scale_sort_key(item[0]))]

    if enumerate_discrete and 0 <= hi - lo <= 20:
        return [_clean_scale_label(n, anchors.get(str(n), str(n))) for n in range(lo, hi + 1)]

    return [str(v) for _, v in sorted(anchors.items(), key=lambda item: _scale_sort_key(item[0]))]


def _response_format(ntype: str, scale: Optional[Dict[str, Any]], options: List[str]) -> Dict[str, Any]:
    fmt: Dict[str, Any] = {"answer_type": ntype}
    if scale:
        fmt.update({
            "scale_min": scale.get("min"),
            "scale_max": scale.get("max"),
            "anchors": _normalized_scale_anchors(scale),
        })
        if ntype == "slider":
            fmt["response_mode"] = "numeric_slider"
        elif ntype == "scale":
            fmt["response_mode"] = "discrete_scale"
    if options:
        fmt["options"] = options
    return fmt


def _clean_matrix_text(text: str) -> str:
    text = re.sub(r"\s*(?:\.\.\.|…)\s*", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


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


def _looks_instruction_like(name: str, label: str) -> bool:
    if re.search(r"(?:^|_)(instruction|intro)(?:_|$)", name or "", flags=re.IGNORECASE):
        return True
    return bool(re.match(r"^\s*(now|thank you|to minimize)\b", label or "", flags=re.IGNORECASE))

# QSF -> materials

def _expand_question(it: Any) -> str:
    stem = it.text
    if it.rows:
        clean_stem = _clean_matrix_text(stem)
        trials = " ".join(
            f"{i}. {_clean_matrix_text(row)}"
            for i, row in enumerate(it.rows, start=1)
        )
        return f"{clean_stem} {trials}".strip() if clean_stem else trials
    return stem


def _qsf_condition_candidates(survey: ParsedSurvey) -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = [dict(c) for c in getattr(survey, "flow_conditions", [])]
    names = {c.get("name") for c in conditions}

    vignette_groups: set[str] = set()
    vignette_numbers: set[str] = set()
    for block in survey.blocks:
        if block.is_trash or _is_admin_block(block.description):
            continue
        match = re.match(r"\s*([A-Za-z][\w-]*)\s+Vignette\s+(\d+)", block.description or "")
        if match:
            vignette_groups.add(match.group(1))
            vignette_numbers.add(match.group(2))

    if len(vignette_groups) > 1 and "qsf_vignette_group" not in names:
        conditions.append({
            "name": "qsf_vignette_group",
            "levels": sorted(vignette_groups),
            "source": "qsf_block_description",
            "review_required": True,
        })
    if len(vignette_numbers) > 1 and "qsf_vignette" not in names:
        conditions.append({
            "name": "qsf_vignette",
            "levels": sorted(vignette_numbers, key=lambda item: int(item)),
            "source": "qsf_block_description",
            "review_required": True,
        })
    return conditions


def _from_qsf(survey: ParsedSurvey, source_file: str) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    instructions: List[str] = []
    instruction_blocks: List[Dict[str, Any]] = []

    for block in survey.blocks:
        if block.is_trash or _is_admin_block(block.description):
            continue
        for it in block.items:
            # stimulus / framing text -> instructions
            if not it.is_response_item():
                if len(it.text) >= 60:
                    instructions.append(it.text)
                    instruction_blocks.append(
                        {
                            "block": it.block or block.description,
                            "qid": it.qid,
                            "data_export_tag": it.data_export_tag,
                            "text": it.text,
                            "source": "osf_qsf",
                            "source_file": source_file,
                        }
                    )
                continue
            tag = it.data_export_tag or it.qid
            question = _expand_question(it)
            options = list(it.choices)
            if it.scale and it.type in ("scale", "matrix"):
                options = options or _scale_options(it.scale, enumerate_discrete=True)
            elif it.scale and it.type == "slider":
                options = options or _scale_options(it.scale, enumerate_discrete=False)
            item: Dict[str, Any] = {
                "id": _slug(tag, it.qid.lower()),
                "question": question,
                "options": options,
                "type": it.type,
                "source": "osf_qsf",
                "source_file": source_file,
                "data_export_tag": it.data_export_tag,
                "block": it.block,
            }
            if it.scale:
                item["scale"] = it.scale
            item["response_format"] = _response_format(it.type, it.scale, options)
            items.append(item)

    return {
        "instructions": "\n\n".join(instructions[:8]),
        "instruction_blocks": instruction_blocks,
        "items": items,
        "conditions": _qsf_condition_candidates(survey),
        "response_schema": _response_schema(items),
    }


# SAV -> materials

def _from_sav(dataset: ParsedDataset, source_file: str) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for v in dataset.response_items():
        if _looks_instruction_like(v.name, v.label):
            skipped.append(v.name)
            continue
        opts = v.choices
        if not opts and v.scale:
            opts = _scale_options(v.scale, enumerate_discrete=True)
        item = {
            "id": _slug(v.name, v.name),
            "question": v.label,
            "options": opts,
            "type": v.type,
            "source": "osf_sav",
            "source_file": source_file,
            "data_export_tag": v.name,
            **({"scale": v.scale} if v.scale else {}),
        }
        item["response_format"] = _response_format(v.type, v.scale, opts)
        if _has_truncation_marker(v.label):
            item.setdefault("quality_flags", []).append("label_contains_ellipsis")
        items.append(item)

    conditions = [
        {"name": v.name, "label": v.label or v.name, "levels": v.choices}
        for v in dataset.condition_variables()
    ]
    return {
        "instructions": "",  # .sav carries no stimulus text (provides_stimulus=False)
        "instruction_blocks": [],
        "items": items,
        "conditions": conditions,
        "response_schema": _response_schema(items),
        "skipped_items": skipped,
    }


# Shared

def _response_schema(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Infer a dominant response schema from the assembled items."""
    if not items:
        return {}
    from collections import Counter
    dominant = Counter(it["type"] for it in items).most_common(1)[0][0]
    rep = next((it for it in items if it["type"] == dominant), items[0])
    schema: Dict[str, Any] = {"answer_type": dominant}
    if rep.get("scale"):
        schema["scale_min"] = rep["scale"]["min"]
        schema["scale_max"] = rep["scale"]["max"]
        schema["anchors"] = _normalized_scale_anchors(rep["scale"])
    if rep.get("options"):
        schema["options"] = rep["options"]
    return schema


def _primary_effects(stage3_study: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(stage3_study, dict):
        return []
    effects = [effect for effect in stage3_study.get("effects", []) if isinstance(effect, dict)]
    primary = []
    for effect in effects:
        cons = effect.get("consolidation")
        if isinstance(cons, dict):
            if cons.get("is_primary_simulation_target", True) and cons.get("is_representative", True) is not False:
                primary.append(effect)
    return primary or effects


def _focal_item_stems(stage3_study: Optional[Dict[str, Any]]) -> set[str]:
    effects = _primary_effects(stage3_study)
    stems: set[str] = set()
    for effect in effects:
        stems |= _target_stems(effect.get("DV"))
    return stems


def _item_match_score(item: Dict[str, Any], stems: set[str]) -> int:
    if not stems:
        return 0
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("id", "data_export_tag", "question")
    )
    item_stems = _target_stems(haystack)
    score = len(item_stems & stems)
    # Short analysis tags often use an abbreviated construct prefix
    # (e.g. open_1 for openness, recep1 for receptiveness).
    tag = str(item.get("data_export_tag") or item.get("id") or "").lower()
    tag_parts = [_token_stem(part) for part in re.split(r"[^a-z0-9]+", tag) if part]
    for stem in stems:
        if any(part and (stem.startswith(part) or part.startswith(stem[:4])) for part in tag_parts):
            score += 2
    return score


def _annotate_focal_item_candidates(
    items: List[Dict[str, Any]],
    stage3_study: Optional[Dict[str, Any]],
    warnings: List[str],
    source_trace: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if len(items) < 8:
        return items
    stems = _focal_item_stems(stage3_study)
    if not stems:
        return items
    scored = [(item, _item_match_score(item, stems)) for item in items]
    selected = [
        item for item, score in scored
        if score > 0 and not _ADMIN_ITEM_RE.match(str(item.get("id") or item.get("data_export_tag") or ""))
    ]
    if len(selected) < 2 or len(selected) == len(items):
        return items
    warnings.append(
        f"Identified {len(selected)}/{len(items)} likely focal response items; full source instrument retained."
    )
    source_trace["focal_item_candidates"] = {
        "mode": "focal_outcome_terms_non_destructive",
        "target_stems": sorted(stems),
        "candidate_items": [item.get("data_export_tag") or item.get("id") for item in selected],
        "instrument_items_retained": len(items),
    }
    return items


def _stage3_item_texts(stage3_study: Optional[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for effect in _primary_effects(stage3_study):
        slot = effect.get("items")
        content = slot.get("content") if isinstance(slot, dict) else slot
        if content:
            texts.append(str(content))
    return texts


def _extract_question_phrases(text: str) -> List[str]:
    """Pull participant-facing item phrases out of prose item summaries."""
    text = _clean_matrix_text(text)
    if not text:
        return []
    candidates: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if "used to measure" in sentence.lower() and "all participants rated" not in sentence.lower():
            continue
        if not re.search(r"\b(how|what extent|rate|rated|likely|appreciat|recogniz|share|want)\b", sentence, re.I):
            continue
        if re.search(r"\ball participants rated\b", sentence, flags=re.I):
            sentence = re.sub(r"^.*?\ball participants rated\b\s+", "", sentence, flags=re.I)
        elif re.search(r"\b(?:please\s+)?rate\b", sentence, flags=re.I):
            sentence = re.sub(r"^.*?\b(?:please\s+)?rate\b\s+", "", sentence, flags=re.I)
        elif not re.match(r"\s*(?:how|to what extent|what)\b", sentence, flags=re.I):
            continue
        parts = re.split(r",\s+|\s+and\s+", sentence)
        for part in parts:
            part = part.strip(" ;.")
            part = re.sub(r"^(?:and|or)\s+", "", part, flags=re.IGNORECASE)
            if not re.search(r"\b(how|what extent|likely|appreciat|recogniz|share|want)\b", part, re.I):
                continue
            if len(part) < 18:
                continue
            part = _participantize_question(part)
            if not part.endswith("?"):
                part += "?"
            candidates.append(part[:1].upper() + part[1:])
    deduped: List[str] = []
    seen = set()
    for item in candidates:
        key = re.sub(r"\s+", " ", item.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _participantize_question(text: str) -> str:
    """Convert prose summaries of participant items back into direct questions."""
    text = re.sub(r"\b[hH]ow likely they were to\b", "how likely are you to", text)
    text = re.sub(r"\b[hH]ow much they appreciated\b", "how much do you appreciate", text)
    text = re.sub(r"\b[hH]ow much they recognized\b", "how much do you recognize", text)
    text = re.sub(r"\bthey just read\b", "you just read", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthey read\b", "you read", text, flags=re.IGNORECASE)
    text = re.sub(r"\btheir\b", "your", text, flags=re.IGNORECASE)
    return text


def _item_numeric_group_key(item: Dict[str, Any]) -> tuple[str, Optional[int]]:
    tag = str(item.get("data_export_tag") or item.get("id") or "")
    match = re.match(r"^([A-Za-z_]+?)[_-]?(\d+)$", tag)
    if not match:
        return tag.lower(), None
    return match.group(1).lower().rstrip("_"), int(match.group(2))


def _record_truncated_item_patch_hints(
    items: List[Dict[str, Any]],
    stage3_study: Optional[Dict[str, Any]],
    warnings: List[str],
    source_trace: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not any(_has_truncation_marker(item.get("question", "")) for item in items):
        return items
    phrases: List[str] = []
    for text in _stage3_item_texts(stage3_study):
        phrases.extend(_extract_question_phrases(text))
    if not phrases:
        return items

    groups: Dict[str, List[tuple[int, Optional[int], Dict[str, Any]]]] = {}
    for idx, item in enumerate(items):
        key, number = _item_numeric_group_key(item)
        groups.setdefault(key, []).append((idx, number, item))

    hints: List[Dict[str, str]] = []
    for _, group in groups.items():
        numbered = [(idx, number, item) for idx, number, item in group if number is not None]
        if len(numbered) < 2 or len(phrases) < len(numbered):
            continue
        numbered.sort(key=lambda entry: entry[1] or 0)
        group_questions = [entry[2].get("question", "") for entry in numbered]
        if not any(_has_truncation_marker(question) for question in group_questions):
            continue
        for idx, number, item in numbered:
            old = str(item.get("question") or "")
            if not _has_truncation_marker(old):
                continue
            phrase_index = (number or 1) - 1
            if phrase_index < 0 or phrase_index >= len(phrases):
                continue
            new = phrases[phrase_index]
            hints.append({
                "item": str(item.get("data_export_tag") or item.get("id")),
                "old": old.replace("...", "[truncated]").replace("…", "[truncated]"),
                "suggested_question": new,
                "suggested_source": "stage3_effect_item_slot",
            })
        break

    if hints:
        warnings.append(
            f"Recorded {len(hints)} Stage 3 item patch hint(s); SAV codebook labels were left unchanged."
        )
        source_trace["item_patch_hints"] = hints
    return items


def _norm_condition_label(value: str) -> str:
    value = value.lower()
    value = value.replace("two-sided", "twosided").replace("one-sided", "onesided")
    value = value.replace("two-side", "twosided").replace("one-side", "onesided")
    value = re.sub(r"\b(condition|conditions|strong|weak|high|low)\b", " ", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def _record_stage3_condition_patch_hints(
    conditions: List[Dict[str, Any]],
    s3: Optional[Dict[str, Any]],
    warnings: List[str],
    source_trace: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not s3 or not s3.get("conditions"):
        return conditions
    stage3_conditions = s3.get("conditions") or []
    stage3_levels: Dict[str, str] = {}
    for cond in stage3_conditions:
        descriptions = cond.get("level_descriptions") if isinstance(cond, dict) else None
        if isinstance(descriptions, dict):
            for level, desc in descriptions.items():
                stage3_levels[_norm_condition_label(str(level))] = str(desc)
    if not stage3_levels:
        return conditions
    hints: List[Dict[str, Any]] = []
    for cond in conditions:
        level_desc: Dict[str, str] = {}
        for level in cond.get("levels", []):
            norm = _norm_condition_label(str(level))
            match = next(
                (
                    desc for key, desc in stage3_levels.items()
                    if norm in key or key in norm or bool(set(re.findall(r"[a-z]+", norm)) & set(re.findall(r"[a-z]+", key)))
                ),
                None,
            )
            if match:
                level_desc[str(level)] = match
        if level_desc:
            hints.append({
                "condition": cond.get("name") or cond.get("label"),
                "suggested_level_descriptions": level_desc,
                "suggested_source": "stage3_condition_slot",
            })
    if hints:
        warnings.append(
            f"Recorded {len(hints)} Stage 3 condition patch hint(s); source conditions were left unchanged."
        )
        source_trace["condition_patch_hints"] = hints
    return conditions


def assemble_study_materials(
    study_id: str,
    link_result: LinkResult,
    *,
    repo_root: Optional[Path] = None,
    stage3_study: Optional[Dict[str, Any]] = None,
    allow_pdf_slot_fallback: bool = False,
) -> Dict[str, Any]:
    """Assemble one materials dict for `study_id` from its linked sources.

    Source priority:
      1. OSF .qsf instrument          - verbatim participant-facing survey.
      2. OSF .sav codebook            - items + conditions; stimulus can be
                                        filled from legacy Stage 3 slots only
                                        when `allow_pdf_slot_fallback=True`.
      3. Stage 3 PDF-extracted slots  - optional legacy fallback. The new
                                        pipeline keeps incomplete packages as
                                        `ready=false` instead of treating
                                        per-effect slots as final materials.
    """
    sub_id = canonical_sub_study_id(study_id)

    mats = link_result.material_files(study_id)
    base = {
        "sub_study_id": sub_id,
        "instructions": "",
        "instruction_blocks": [],
        "items": [],
        "conditions": [],
        "response_schema": {},
        "readiness": {"ready": False, "blocking_issues": [], "warnings": []},
        "source_trace": {"primary_source": None, "source_file": None, "candidates": [
            {"kind": f.kind, "file": _relpath(f.path, repo_root)} for f in mats
        ]},
    }
    s3 = materials_from_stage3(stage3_study) if stage3_study else None

    primary = _pick_primary(mats)

    # --- 3. no OSF instrument: keep a draft unless legacy fallback is explicit
    if primary is None:
        if allow_pdf_slot_fallback and s3 and s3["items"]:
            base.update({k: s3[k] for k in ("instructions", "items", "conditions")})
            base["response_schema"] = _response_schema(s3["items"])
            base["source_trace"]["primary_source"] = "pdf_stage3"
            base["source_trace"]["source_file"] = "stage3.json"
            base["source_trace"]["slot_summary"] = s3["slot_summary"]
            base["readiness"]["ready"] = True
            base["readiness"]["warnings"].append(
                "No OSF instrument; materials taken from Stage 3 PDF slots "
                f"(status: {s3['slot_summary']}). Review verbatim-ness."
            )
            _apply_quality_gate(base, primary_kind="pdf_stage3")
            base["readiness"]["ready"] = bool(base["items"]) and not base["readiness"]["blocking_issues"]
        else:
            base["readiness"]["blocking_issues"].append("no_structured_material_source")
            base["readiness"]["warnings"].append(
                "No structured OSF/supplement instrument was linked. Legacy "
                "per-effect slots were not used as final materials; recover the "
                "participant-facing stimulus and response items from the PDF or "
                "supplement in the Stage 4 draft."
            )
            if s3:
                base["source_trace"]["legacy_slot_summary"] = s3["slot_summary"]
        return base

    # --- 1/2. OSF instrument present -----------------------------------------
    src_rel = _relpath(primary.path, repo_root)
    try:
        if primary.kind == "qsf":
            built = _from_qsf(parse_qsf_file(primary.path), src_rel)
        else:  # sav
            built = _from_sav(parse_sav_file(primary.path), src_rel)
    except Exception as exc:  # pragma: no cover - parser robustness guard
        base["readiness"]["blocking_issues"].append(f"parse_error:{type(exc).__name__}")
        return base

    base.update({k: built[k] for k in ("instructions", "items", "conditions", "response_schema")})
    if "instruction_blocks" in built:
        base["instruction_blocks"] = built["instruction_blocks"]
    base["source_trace"]["primary_source"] = f"osf_{primary.kind}"
    base["source_trace"]["source_file"] = src_rel
    if built.get("skipped_items"):
        base["readiness"]["warnings"].append(
            "Skipped non-response SAV fields: " + ", ".join(built["skipped_items"])
        )

    # --- 2. enrich .sav (no stimulus) with Stage 3 PDF material --------------
    if primary.kind == "sav" and not base["instructions"]:
        if allow_pdf_slot_fallback and s3 and s3["instructions"]:
            base["instructions"] = s3["instructions"]
            base["source_trace"]["stimulus_source"] = "pdf_stage3"
            base["source_trace"]["slot_summary"] = s3["slot_summary"]
            base["readiness"]["warnings"].append(
                "Items from .sav codebook; stimulus filled from Stage 3 PDF "
                f"slots (status: {s3['slot_summary'].get('materials_status')})."
            )
        else:
            if s3:
                base["source_trace"]["legacy_slot_summary"] = s3["slot_summary"]
            base["readiness"]["warnings"].append(
                "Items recovered from .sav codebook; stimulus text not available "
                "from data file. Legacy per-effect slots were not used as final "
                "stimulus; complete from PDF/supplement."
            )

    base["items"] = _annotate_focal_item_candidates(
        base["items"],
        stage3_study,
        base["readiness"]["warnings"],
        base["source_trace"],
    )
    if primary.kind == "sav":
        base["items"] = _record_truncated_item_patch_hints(
            base["items"],
            stage3_study,
            base["readiness"]["warnings"],
            base["source_trace"],
        )
    base["conditions"] = _record_stage3_condition_patch_hints(
        base["conditions"],
        s3,
        base["readiness"]["warnings"],
        base["source_trace"],
    )
    base["response_schema"] = _response_schema(base["items"])

    n = len(base["items"])
    if n == 0:
        base["readiness"]["blocking_issues"].append("no_participant_items")
    _apply_quality_gate(base, primary_kind=primary.kind)
    base["readiness"]["ready"] = n > 0 and not base["readiness"]["blocking_issues"]
    return base


def _apply_quality_gate(material: Dict[str, Any], *, primary_kind: str) -> None:
    issues = material["readiness"]["blocking_issues"]
    warnings = material["readiness"]["warnings"]

    if primary_kind == "sav":
        truncated = [
            item.get("data_export_tag") or item.get("id")
            for item in material.get("items", [])
            if _has_truncation_marker(item.get("question", ""))
        ]
        if truncated:
            issues.append("truncated_sav_labels")
            warnings.append(
                "SAV variable labels contain ellipses/truncation markers: "
                + ", ".join(str(tag) for tag in truncated[:12])
                + (" ..." if len(truncated) > 12 else "")
            )

        trace = material.get("source_trace", {}) if isinstance(material.get("source_trace"), dict) else {}
        stimulus_source = trace.get("stimulus_source")
        has_instructions = bool(str(material.get("instructions") or "").strip())
        if not has_instructions or not stimulus_source:
            issues.append("stimulus_not_verbatim")
            warnings.append(
                "SAV files do not contain participant-facing stimulus text; "
                "complete stimulus from a participant-facing PDF/supplement source."
            )

    if primary_kind == "qsf":
        ellipsis = [
            item.get("data_export_tag") or item.get("id")
            for item in material.get("items", [])
            if _has_truncation_marker(item.get("question", ""))
        ]
        if ellipsis:
            warnings.append(
                "QSF item text still contains ellipsis markers; verify matrix expansion: "
                + ", ".join(str(tag) for tag in ellipsis[:12])
                + (" ..." if len(ellipsis) > 12 else "")
            )

    if primary_kind == "pdf_stage3":
        cited = [
            item.get("data_export_tag") or item.get("id")
            for item in material.get("items", [])
            if str(item.get("slot_status") or "").lower() == "cited_scale"
        ]
        if cited:
            warnings.append(
                "PDF Stage 3 items include cited/non-verbatim scales: "
                + ", ".join(str(tag) for tag in cited[:12])
                + (" ..." if len(cited) > 12 else "")
            )
        if cited and len(cited) == len(material.get("items", [])):
            issues.append("pdf_items_not_verbatim")


def _pick_primary(mats: List[LinkedFile]) -> Optional[LinkedFile]:
    for kind in ("qsf", "sav"):
        for f in mats:
            if f.kind == kind:
                return f
    return None


def _relpath(path: Path, repo_root: Optional[Path]) -> str:
    if repo_root:
        try:
            return str(Path(path).resolve().relative_to(Path(repo_root).resolve()))
        except ValueError:
            pass
    return str(path)
