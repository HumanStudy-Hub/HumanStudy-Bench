from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.llm.helpers import call_with_timeout

EFFECT_SELECTION_RUBRIC = """Select the PRIMARY findings of ONE study for a simulation benchmark. The
"effecttype" label MAY BE WRONG; judge from IV/DV semantics.

STEP 1: From the paper title + study, name the study's FOCAL OUTCOME construct(s) - the
dependent measure(s) the study is fundamentally about (often named in the title).

STEP 2: KEEP an effect iff ALL hold:
 - its IV is the study's experimental MANIPULATION / assigned condition (a manipulated
   between- or within-subjects factor), not a measured individual-difference scale;
 - its DV is a FOCAL outcome (or a distinct focal outcome), NOT a manipulation check
   (a DV that merely re-measures the manipulated construct itself), NOT a secondary /
   auxiliary measure outside the focal outcome(s);
 - keep main effects AND interactions of the manipulation; a condition->focal-outcome
   effect is primary even if mislabelled 'correlation' / 'mediation'.
DROP everything else (manipulation checks, non-focal secondary DVs, pure correlations of
measured scales, mediation / process tests).

Return ONLY JSON:
{"focal_outcomes":["..."],"effects":[{"effect_index":int,"keep":true,"role":"primary|manip_check|secondary_dv|correlation|mediation","reason":"<=12 words"}]}"""


def build_prompt(title: str, study_name: str, effects: List[Dict[str, Any]]) -> str:
    lines = [
        f"  [{i}] effecttype={e.get('effecttype')} | IV={e.get('IV')} | DV={e.get('DV')}"
        for i, e in enumerate(effects)
    ]
    return (
        f"{EFFECT_SELECTION_RUBRIC}\n\nPaper: {title}\nStudy: {study_name}\n"
        f"Effects:\n" + "\n".join(lines)
    )


def _parse(response: str) -> Optional[List[Dict[str, Any]]]:
    text = response if isinstance(response, str) else str(response)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    eff = obj.get("effects") if isinstance(obj, dict) else None
    return eff if isinstance(eff, list) else None


_NON_PRIMARY_EFFECT_TYPES = {
    "simple",
    "simple_effect",
    "mediation",
    "correlation",
    "manipulation_check",
    "check",
}


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


def _deterministic_primary_veto(effect: Dict[str, Any]) -> Optional[str]:
    etype = str(effect.get("effecttype") or "").strip().lower()
    if etype in _NON_PRIMARY_EFFECT_TYPES:
        return f"drop deterministic {etype}"
    stats = effect.get("stats") if isinstance(effect.get("stats"), dict) else {}
    sig = str(stats.get("sig") or effect.get("sig") or "").strip().lower()
    if any(token in sig for token in ("marginal", "ns", "n.s", "non", "not sig")):
        return "drop deterministic non-significant"
    p_value = _stats_p_value(stats)
    if p_value is not None and p_value > 0.05:
        return "drop deterministic p>.05"
    dv = str(effect.get("DV") or "").lower()
    iv = str(effect.get("IV") or "").lower()
    if "manipulation check" in dv or "manipulation check" in iv:
        return "drop deterministic manipulation check"
    return None


def select_effects(
    title: str,
    study_name: str,
    effects: List[Dict[str, Any]],
    client: Any,
    *,
    votes: int = 1,
    timeout: Optional[float] = 60.0,
) -> Dict[int, Dict[str, Any]]:
    """Return {effect_index: {"keep": bool, "role": str, "reason": str}}.

    Fail-open: indices the model could not score keep their default (keep=True).
    With votes>1 each effect's keep decision is a majority vote.
    """
    n = len(effects)
    keep_votes = [0] * n
    drop_votes = [0] * n
    roles: List[str] = ["primary"] * n
    reasons: List[str] = [""] * n
    rounds = 0

    for _ in range(max(1, votes)):
        try:
            response = _generate_selection_response(
                client,
                prompt=build_prompt(title, study_name, effects),
                timeout=timeout,
            )
            parsed = _parse(response)
        except Exception:  # pragma: no cover - network/LLM dependent
            parsed = None
        if parsed is None:
            continue
        rounds += 1
        for d in parsed:
            idx = d.get("effect_index")
            if not isinstance(idx, int) or not (0 <= idx < n):
                continue
            if bool(d.get("keep", True)):
                keep_votes[idx] += 1
            else:
                drop_votes[idx] += 1
            if d.get("role"):
                roles[idx] = str(d.get("role"))
            if d.get("reason"):
                reasons[idx] = str(d.get("reason"))[:200]

    out: Dict[int, Dict[str, Any]] = {}
    for i in range(n):
        if rounds == 0:
            out[i] = {"keep": True, "role": "primary", "reason": "selection skipped: LLM unavailable"}
            continue
        keep = keep_votes[i] >= drop_votes[i]  # ties -> keep (fail-open)
        veto = _deterministic_primary_veto(effects[i])
        if veto:
            out[i] = {"keep": False, "role": roles[i], "reason": veto}
        else:
            out[i] = {"keep": keep, "role": roles[i], "reason": reasons[i]}
    return out


def _generate_selection_response(client: Any, *, prompt: str, timeout: Optional[float]) -> str:
    def _call() -> str:
        try:
            return client.generate_content(prompt=prompt, timeout=timeout, max_tokens=1200)
        except TypeError:
            try:
                return client.generate_content(prompt=prompt, timeout=timeout)
            except TypeError:
                return client.generate_content(prompt=prompt)

    return call_with_timeout(_call, timeout=timeout)


def apply_to_study(
    study: Dict[str, Any],
    title: str,
    client: Any,
    *,
    votes: int = 1,
    timeout: Optional[float] = 60.0,
) -> Dict[int, Dict[str, Any]]:
    """Re-judge a consolidated study's effects and override the primary flags."""
    effects = [e for e in study.get("effects", []) if isinstance(e, dict)]
    study_name = str(study.get("study") or study.get("study_id") or "")
    decisions = select_effects(title, study_name, effects, client, votes=votes, timeout=timeout)

    for i, e in enumerate(effects):
        dec = decisions.get(i)
        cons = e.get("consolidation")
        if dec and isinstance(cons, dict):
            cons["llm_role"] = dec["role"]
            cons["llm_keep"] = dec["keep"]
            cons["llm_reason"] = dec["reason"]

    summary = study.get("consolidation_summary")
    if isinstance(summary, dict):
        primary = 0
        for g in summary.get("groups", []):
            rep = g.get("representative_effect_index")
            dec = decisions.get(rep) if isinstance(rep, int) else None
            if dec is not None:
                g["is_primary_simulation_target"] = dec["keep"]
                g["analysis_role"] = dec["role"]
                # mirror onto the effect-level consolidation blocks
                for idx in g.get("effect_indices", []):
                    if 0 <= idx < len(effects) and isinstance(effects[idx].get("consolidation"), dict):
                        effects[idx]["consolidation"]["is_primary_simulation_target"] = dec["keep"]
                        effects[idx]["consolidation"]["analysis_role"] = dec["role"]
            if g.get("is_primary_simulation_target"):
                primary += 1
        summary["primary_simulation_targets"] = primary
        summary["llm_reselected"] = True
    return decisions
