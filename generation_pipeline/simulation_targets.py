from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from generation_pipeline.identifiers import canonical_sub_study_id

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]+", re.IGNORECASE)
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "using",
    "used", "study", "studies", "effect", "effects", "condition",
    "conditions", "measure", "measures", "item", "items", "scale",
    "participant", "participants", "response", "responses", "reported",
    "results", "result", "table", "index", "direct", "focal", "primary",
    "continuous", "between", "within", "point", "rating",
    "more", "less", "same", "than", "today", "across",
    "future", "generat", "people", "person", "persons", "human", "humans",
}
_DROP_ROLES = {
    "simple", "simple_effect", "mediation", "correlation",
    "manipulation_check", "check", "secondary_dv",
}
_ALLOW_ROLES = {"primary", "primary_finding", "interaction", "moderation", "main"}


def _slug(value: Any, fallback: str = "study") -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text or fallback


def _stem(token: str) -> str:
    token = token.lower().strip("'_-")
    if token in {"sided", "sidedness"}:
        return "side"
    for suffix in ("iveness", "ation", "ness", "ment", "ility", "ity", "ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break
    return token[:7] if len(token) > 7 else token


def _tokens(*values: Any) -> set[str]:
    out: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("content")
        text = str(value or "")
        text = re.sub(r"[-/]+", " ", text)
        for token in _TOKEN_RE.findall(text):
            stem = _stem(token)
            if len(stem) >= 4 and stem not in _STOP and not stem.isdigit():
                out.add(stem)
    return out


def _soft_overlap(a: set[str], b: set[str]) -> int:
    hits = 0
    used: set[str] = set()
    for x in a:
        for y in b:
            if y in used:
                continue
            if x == y or (min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x))):
                hits += 1
                used.add(y)
                break
    return hits


def _stats_p_value(stats: Dict[str, Any]) -> Optional[float]:
    raw = stats.get("p_value") if isinstance(stats, dict) else None
    if raw in (None, "", [None, None]):
        return None
    text = str(raw).strip().lower()
    if text in {"ns", "n.s.", "not significant", "non-significant"}:
        return 1.0
    match = re.search(r"([<>]=?)?\s*\.?(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    number = match.group(2)
    if "." in match.group(0) and "." not in number:
        return float("0." + number)
    return float(number)


def _effect_is_significant(effect: Dict[str, Any]) -> bool:
    stats = effect.get("stats") if isinstance(effect.get("stats"), dict) else {}
    sig = str(stats.get("sig") or effect.get("sig") or "").strip().lower()
    if any(token in sig for token in ("marginal", "ns", "n.s", "non", "not sig")):
        return False
    p_value = _stats_p_value(stats)
    return not (p_value is not None and p_value > 0.05)


def _material_selected(material: Any) -> bool:
    if not isinstance(material, dict):
        return True
    selection = material.get("selection")
    if not isinstance(selection, dict):
        return True
    return bool(selection.get("keep", True))


def _study_selected(study: Dict[str, Any]) -> bool:
    selection = study.get("selection")
    if not isinstance(selection, dict):
        return True
    return bool(selection.get("keep", True))


def _condition_terms(material: Dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for cond in material.get("conditions", []) or []:
        if not isinstance(cond, dict):
            continue
        terms |= _tokens(cond.get("name"), cond.get("label"))
        for level in cond.get("levels", []) or []:
            terms |= _tokens(level)
        descriptions = cond.get("level_descriptions")
        if isinstance(descriptions, dict):
            for value in descriptions.values():
                terms |= _tokens(value)
    return terms


def _item_terms(material: Dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for item in material.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        terms |= _tokens(item.get("id"), item.get("data_export_tag"), item.get("question"))
    return terms


def _item_match_count(material: Dict[str, Any], dv_terms: set[str]) -> int:
    count = 0
    for item in material.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        item_terms = _tokens(item.get("id"), item.get("data_export_tag"), item.get("question"))
        if _soft_overlap(dv_terms, item_terms) > 0:
            count += 1
    return count


def _role(effect: Dict[str, Any]) -> str:
    cons = effect.get("consolidation") if isinstance(effect.get("consolidation"), dict) else {}
    return str(cons.get("analysis_role") or effect.get("effecttype") or "").strip().lower()


def _is_representative(effect: Dict[str, Any], index: int) -> bool:
    cons = effect.get("consolidation") if isinstance(effect.get("consolidation"), dict) else {}
    if not cons:
        return True
    rep = cons.get("representative_effect_index", index)
    return bool(cons.get("is_representative", index == rep))


def _candidate(
    *,
    effect: Dict[str, Any],
    effect_index: int,
    material: Dict[str, Any],
    title_terms: set[str],
) -> Optional[Dict[str, Any]]:
    role = _role(effect)
    etype = str(effect.get("effecttype") or "").strip().lower()
    if role in _DROP_ROLES or etype in _DROP_ROLES:
        return None
    if role and role not in _ALLOW_ROLES and etype not in {"main", "int", "interaction", "moderation"}:
        return None
    if not _effect_is_significant(effect):
        return None
    iv = str(effect.get("IV") or "")
    dv = str(effect.get("DV") or "")
    materials_notes = str(effect.get("materials_notes") or "")
    if "manipulation check" in f"{iv} {dv} {materials_notes}".lower():
        return None

    iv_terms = _tokens(iv)
    dv_terms = _tokens(dv)
    condition_terms = _condition_terms(material)
    item_terms = _item_terms(material)
    condition_hits = _soft_overlap(iv_terms, condition_terms)
    item_hits = _soft_overlap(dv_terms, item_terms)
    item_match_count = _item_match_count(material, dv_terms)
    title_hits = _soft_overlap(dv_terms, title_terms)
    cons = effect.get("consolidation") if isinstance(effect.get("consolidation"), dict) else {}
    llm_keep = cons.get("llm_keep")
    primary_flag = bool(cons.get("is_primary_simulation_target", True))

    score = 0
    score += 35 if role in {"interaction", "moderation"} or etype in {"int", "interaction"} else 20
    score += 20 if primary_flag else 0
    score += 12 if llm_keep is True else 0
    score += 12 * min(condition_hits, 2)
    score += 8 * min(item_hits, 2)
    score += 5 * min(item_match_count, 5)
    score += 14 * min(title_hits, 2)
    score += 6 if _effect_is_significant(effect) else 0

    return {
        "effect_index": effect_index,
        "effect": effect,
        "score": score,
        "role": role or etype or "primary",
        "condition_hits": condition_hits,
        "item_hits": item_hits,
        "item_match_count": item_match_count,
        "title_hits": title_hits,
        "has_conditions": bool(condition_terms),
        "dv_key": " ".join(sorted(dv_terms)) or _slug(dv, "dv"),
        "merge_group_id": cons.get("merge_group_id"),
    }


def _pick_study_targets(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    # If the material exposes condition variables, prefer effects whose IV maps
    # to those participant-facing conditions. This removes measured-only direct
    # effects while preserving manipulation x moderator interactions.
    if any(c["condition_hits"] > 0 for c in candidates):
        candidates = [c for c in candidates if c["condition_hits"] > 0]

    if any(c["item_hits"] > 0 for c in candidates):
        candidates = [c for c in candidates if c["item_hits"] > 0]

    # When the paper title clearly names an item-backed focal outcome, use it to
    # drop secondary DVs such as mediators or appreciation checks. If the only
    # title hits are derived metrics with no direct item match, keep them as
    # possible additions below instead of letting them suppress item-backed DVs.
    title_backed = [c for c in candidates if c["title_hits"] > 0 and c["item_match_count"] > 0]
    if title_backed:
        candidates = title_backed
    elif any(c["has_conditions"] and c["condition_hits"] > 0 for c in candidates):
        # Explicit condition structure supports multiple co-primary outcomes in
        # the same study (for example manipulation -> beliefs and support).
        return sorted(
            [c for c in candidates if c["condition_hits"] > 0],
            key=lambda item: item["effect_index"],
        )

    by_dv: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        by_dv.setdefault(candidate["dv_key"], []).append(candidate)

    def group_score(items: List[Dict[str, Any]]) -> tuple[int, int, int]:
        return (
            max(int(item["score"]) for item in items),
            len(items),
            sum(int(item["condition_hits"]) for item in items),
        )

    ranked_groups = sorted(by_dv.values(), key=group_score, reverse=True)
    selected = list(ranked_groups[0])
    selected_ids = {id(item) for item in selected}
    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        if candidate["title_hits"] > 0 and candidate["item_match_count"] == 0 and candidate["score"] >= 55:
            selected.append(candidate)
    return sorted(selected, key=lambda item: item["effect_index"])


def build_simulation_targets(
    eligible_studies: List[Dict[str, Any]],
    materials: Dict[str, Dict[str, Any]],
    *,
    paper_title: str = "",
) -> List[Dict[str, Any]]:
    """Build and stamp target records for Stage 4.

    Returns one or more target findings per selected sub-study. Each target is
    bound to a material id and to a representative effect index, so Stage 4 no
    longer has to infer which paper analysis the prompt should represent.
    """
    title_terms = _tokens(paper_title)
    targets: List[Dict[str, Any]] = []

    for study in eligible_studies:
        if not isinstance(study, dict) or not _study_selected(study):
            continue
        study_name = str(study.get("study") or study.get("study_id") or study.get("experiment_id") or "")
        sub_id = canonical_sub_study_id(study_name)
        material = materials.get(sub_id)
        if not isinstance(material, dict) or not _material_selected(material):
            continue

        candidates: List[Dict[str, Any]] = []
        effects = [effect for effect in study.get("effects", []) or [] if isinstance(effect, dict)]
        for index, effect in enumerate(effects):
            if not _is_representative(effect, index):
                continue
            candidate = _candidate(
                effect=effect,
                effect_index=index,
                material=material,
                title_terms=title_terms,
            )
            if candidate is not None:
                candidates.append(candidate)

        picked = _pick_study_targets(candidates)
        for rank, candidate in enumerate(picked, start=1):
            effect = candidate["effect"]
            target_id = f"{sub_id}__effect_{rank:02d}"
            target = {
                "target_id": target_id,
                "sub_study_id": sub_id,
                "study_name": study_name or sub_id,
                "effect_index": candidate["effect_index"],
                "merge_group_id": candidate.get("merge_group_id"),
                "analysis_role": candidate["role"],
                "selection_basis": {
                    "score": candidate["score"],
                    "condition_hits": candidate["condition_hits"],
                    "item_hits": candidate["item_hits"],
                    "item_match_count": candidate["item_match_count"],
                    "title_hits": candidate["title_hits"],
                    "dv_key": candidate["dv_key"],
                },
            }
            effect["simulation_target"] = {
                "selected": True,
                "target_id": target_id,
                "rank": rank,
                **target["selection_basis"],
            }
            targets.append(target)
    return targets


def summarize_simulation_targets(targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_study: Dict[str, int] = {}
    for target in targets:
        sid = str(target.get("sub_study_id") or "study")
        by_study[sid] = by_study.get(sid, 0) + 1
    return {
        "total_targets": len(targets),
        "by_sub_study": by_study,
    }
