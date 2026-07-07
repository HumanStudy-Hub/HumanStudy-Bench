from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# effecttype -> (analysis_role, is_primary_simulation_target)
_ROLE_MAP: Dict[str, tuple[str, bool]] = {
    "main": ("primary_finding", True),
    "int": ("interaction", True),
    "interaction": ("interaction", True),
    "simple": ("simple_effect", False),
    "simple_effect": ("simple_effect", False),
    "correlation": ("correlation", False),
    "mediation": ("mediation", False),
    "moderation": ("moderation", True),
    "manipulation_check": ("manipulation_check", False),
    "check": ("manipulation_check", False),
}

# tokens that mark a factor LEVEL, not the construct itself -> stripped before
_LEVEL_NOISE = re.compile(
    r"\b(\d[\d,\.]*\s*(years?|yrs?|%|percent|days?|months?)?|"
    r"standardi[sz]ed|raw|within[- ]subjects?|between[- ]subjects?|stated|"
    r"per[- ]trial|omnibus|overall|baseline|post|pre)\b",
    re.IGNORECASE,
)
_PARENS = re.compile(r"\([^)]*\)")
_NONWORD = re.compile(r"[^a-z0-9]+")
_AS_ABOVE = re.compile(r"\b(as above|as previous|same as (above|previous)|ditto)\b", re.IGNORECASE)

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "by", "with",
    "that", "this", "is", "are", "be", "vs", "versus", "choose", "choice",
    "rating", "scale", "measure", "score", "level", "average", "mean",
}


def _norm(text: Any) -> str:
    return str(text or "").strip()


def _construct_tokens(text: Any, *, drop_parens: bool = False) -> Set[str]:
    s = _norm(text).lower()
    if drop_parens:
        s = _PARENS.sub(" ", s)
    s = _LEVEL_NOISE.sub(" ", s)
    s = _NONWORD.sub(" ", s)
    toks = {t for t in s.split() if t and t not in _STOP and len(t) > 1}
    return toks


def _table_base(loc: Any) -> str:
    s = _norm(loc).lower()
    s = _PARENS.sub("", s)
    s = re.sub(r"\bp\.?\s*\d+\b", "", s)        # drop page refs
    s = re.sub(r"[,;].*$", "", s)               # drop trailing clauses
    return _NONWORD.sub(" ", s).strip()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _role_for(effecttype: Any) -> tuple[str, bool]:
    return _ROLE_MAP.get(str(effecttype or "").strip().lower(), ("other", False))


def _level_label(effect: Dict[str, Any]) -> str:
    """A short label distinguishing this row from its siblings (the moderator)."""
    for key in ("IV", "DV", "table_or_page_location"):
        m = _LEVEL_NOISE.search(_norm(effect.get(key)))
        if m:
            return m.group(0).strip()
    return _norm(effect.get("table_or_page_location"))[:40]


@dataclass
class EffectGroup:
    merge_group_id: str
    representative_index: int
    effect_indices: List[int] = field(default_factory=list)
    effecttype: str = ""
    analysis_role: str = "other"
    is_primary_simulation_target: bool = False
    iv: str = ""
    dv: str = ""
    table: str = ""
    moderator_levels: List[str] = field(default_factory=list)


def consolidate_effects(
    effects: List[Dict[str, Any]],
    *,
    iv_threshold: float = 0.6,
    dv_threshold: float = 0.6,
    study_id: str = "study",
) -> List[EffectGroup]:
    """Group a study's effect rows into consolidated effects."""
    groups: List[EffectGroup] = []
    prev_idx: Optional[int] = None

    for idx, eff in enumerate(effects):
        if not isinstance(eff, dict):
            continue
        etype = str(eff.get("effecttype") or "").strip().lower()
        role, primary = _role_for(etype)
        iv = _norm(eff.get("IV"))
        dv = _norm(eff.get("DV"))
        table = _table_base(eff.get("table_or_page_location"))

        # explicit "as above" duplicate -> attach to the previous group
        merged = False
        if (_AS_ABOVE.search(dv) or _AS_ABOVE.search(iv)) and prev_idx is not None and groups:
            g = next((g for g in groups if prev_idx in g.effect_indices), None)
            if g is not None:
                g.effect_indices.append(idx)
                g.moderator_levels.append(_level_label(eff))
                merged = True

        if not merged:
            # Keep parentheticals
            iv_tok = _construct_tokens(iv)
            dv_tok = _construct_tokens(dv)
            for g in groups:
                if g.effecttype != etype or g.table != table or not table:
                    continue
                if (_jaccard(_construct_tokens(g.iv), iv_tok) >= iv_threshold
                        and _jaccard(_construct_tokens(g.dv), dv_tok) >= dv_threshold):
                    g.effect_indices.append(idx)
                    g.moderator_levels.append(_level_label(eff))
                    merged = True
                    break

        if not merged:
            groups.append(EffectGroup(
                merge_group_id=f"{study_id}::g{len(groups) + 1}",
                representative_index=idx,
                effect_indices=[idx],
                effecttype=etype,
                analysis_role=role,
                is_primary_simulation_target=primary,
                iv=iv, dv=dv, table=table,
            ))
        prev_idx = idx

    # choose representative = member with the most complete reported statistics
    for g in groups:
        g.representative_index = _pick_representative(effects, g.effect_indices)
        g.moderator_levels = [m for m in dict.fromkeys(g.moderator_levels) if m]
    return groups


def _completeness(effect: Dict[str, Any]) -> int:
    stats = effect.get("stats") if isinstance(effect.get("stats"), dict) else {}
    return sum(1 for v in stats.values() if v not in (None, "", [None, None]))


def _pick_representative(effects: List[Dict[str, Any]], indices: List[int]) -> int:
    return max(indices, key=lambda i: _completeness(effects[i])) if indices else (indices or [0])[0]


def annotate_study(study: Dict[str, Any], *, study_id: Optional[str] = None) -> Dict[str, Any]:
    """Attach `consolidation` to each effect and a `consolidation_summary`.

    Returns the same study dict (mutated in place) for convenience.
    """
    effects = [e for e in study.get("effects", []) if isinstance(e, dict)]
    sid = study_id or str(study.get("study") or study.get("study_id") or "study")
    groups = consolidate_effects(effects, study_id=sid)

    for g in groups:
        for i in g.effect_indices:
            effects[i]["consolidation"] = {
                "merge_group_id": g.merge_group_id,
                "representative_effect_index": g.representative_index,
                "effect_indices": g.effect_indices,
                "is_representative": i == g.representative_index,
                "analysis_role": g.analysis_role,
                "is_primary_simulation_target": g.is_primary_simulation_target,
                "moderator_levels": g.moderator_levels,
            }

    primary = [g for g in groups if g.is_primary_simulation_target]
    study["consolidation_summary"] = {
        "raw_effects": len(effects),
        "consolidated_effects": len(groups),
        "primary_simulation_targets": len(primary),
        "by_role": _role_histogram(groups),
        "groups": [
            {
                "merge_group_id": g.merge_group_id,
                "analysis_role": g.analysis_role,
                "is_primary_simulation_target": g.is_primary_simulation_target,
                "representative_effect_index": g.representative_index,
                "effect_indices": g.effect_indices,
                "n_merged": len(g.effect_indices),
            }
            for g in groups
        ],
    }
    return study


def _role_histogram(groups: List[EffectGroup]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for g in groups:
        out[g.analysis_role] = out.get(g.analysis_role, 0) + 1
    return out
