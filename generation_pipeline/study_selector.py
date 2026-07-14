from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.llm.helpers import call_with_timeout

STUDY_SELECTION_RUBRIC = """You are curating the sub-studies of ONE research paper for an LLM-participant
simulation benchmark. A simulated participant can only READ self-contained written
materials and return a written response. Decide for EACH study: keep or drop.

KEEP a study when an LLM participant can faithfully complete it:
- Vignette / scenario studies (read a scenario, give judgments or ratings).
- Message / article reading studies (read text, then rate attitudes/openness).
- Clean experimental manipulations delivered as written instructions / assignment
  to a between-subjects condition, measured by ratings / Likert / a single choice.
- Written predictions, ratings, or choices remain simulatable even when the
  scenario concerns money, donation, work, or another real-world behavior.
- When a study mixes a written prediction task with a later physical action,
  keep the written task and explain which component is simulatable.
- Keep BOTH an original and its replication when the replication adds a new
  measure/construct/population and is NOT described as fixing the earlier one.

DROP a study only when it cannot be faithfully simulated or is clearly secondary:
- Pilot / norming / validation / manipulation-check-only studies.
- Interactive behavioral paradigms: economic games, public-vs-private decisions
  with real partners, real money/behavior - anything needing live interaction.
- Repetitive multi-trial psychophysical / discounting tasks whose result is a
  derived curve / indifference point rather than a self-contained response.
- An EARLIER study when a later study is described as a "higher-powered" (or
  better-powered / corrected) replication of essentially the same manipulation,
  because that label means the earlier study was under-powered; keep the
  higher-powered version and drop the earlier one.

Return ONLY a JSON array, one object per study:
[{"experiment_id": "...", "keep": true, "reason": "<=15 words"}]"""


def build_brief(exp: Dict[str, Any]) -> str:
    """Compact, decision-relevant brief for one Stage 1 experiment record."""
    return (
        "- experiment_id: %s\n"
        "  name: %s\n"
        "  design: %s\n"
        "  conditions/factors: %s\n"
        "  input (what participant does): %s\n"
        "  participant_task: %s\n"
        "  output (DV): %s\n"
        "  source_hints: %s"
        % (
            exp.get("experiment_id"),
            exp.get("experiment_name"),
            _brief_value(exp.get("design_type")),
            _brief_value(exp.get("conditions_or_factors")),
            str(exp.get("input"))[:240],
            str(exp.get("participant_task") or "")[:240],
            str(exp.get("output"))[:160],
            _brief_value(exp.get("candidate_source_hints"))[:240],
        )
    )


def _brief_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("description") or item.get("kind") or item
            else:
                text = item
            if str(text).strip():
                parts.append(str(text).strip())
        return "; ".join(parts)
    return str(value).strip()


def build_prompt(experiments: List[Dict[str, Any]]) -> str:
    briefs = "\n".join(build_brief(e) for e in experiments)
    return f"{STUDY_SELECTION_RUBRIC}\n\nPaper studies:\n{briefs}"


def _parse_decisions(response: str) -> List[Dict[str, Any]]:
    text = response if isinstance(response, str) else str(response)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and d.get("experiment_id")]


def _select_once(
    experiments: List[Dict[str, Any]],
    client: Any,
    *,
    timeout: Optional[float] = 60.0,
) -> Optional[List[Dict[str, Any]]]:
    try:
        response = _generate_selection_response(client, prompt=build_prompt(experiments), timeout=timeout)
    except Exception:  # pragma: no cover - network/LLM dependent
        return None
    return _parse_decisions(response)


def _generate_selection_response(client: Any, *, prompt: str, timeout: Optional[float]) -> str:
    def _call() -> str:
        try:
            return client.generate_content(prompt=prompt, timeout=timeout, max_tokens=1000)
        except TypeError:
            try:
                return client.generate_content(prompt=prompt, timeout=timeout)
            except TypeError:
                return client.generate_content(prompt=prompt)

    return call_with_timeout(_call, timeout=timeout)


_DROP_REASON_RE = re.compile(
    r"\b("
    r"requires\s+real|no\s+self[- ]contained\s+scenario|field\s+survey|"
    r"longitudinal|multi[- ]wave|supervisor[- ]rated|real[- ]world\s+interaction|"
    r"real\s+employees|not\s+faithfully\s+simulat"
    r")\b",
    re.IGNORECASE,
)

_FIELD_STUDY_RE = re.compile(
    r"\b(field\s+survey|multi[- ]wave|longitudinal|time[- ]lagged|"
    r"supervisor[- ]rated|real\s+employees|workplace\s+survey)\b",
    re.IGNORECASE,
)

_SELF_CONTAINED_EXPERIMENT_RE = re.compile(
    r"\b(vignette|scenario|random\s+assignment|randomly\s+assigned|"
    r"between[- ]subjects|within[- ]subjects|manipulat|written\s+instruction)\b",
    re.IGNORECASE,
)

def _deterministic_drop(exp: Dict[str, Any], reason: str = "") -> Optional[str]:
    """Hard vetoes for cases the LLM sometimes labels keep despite its reason.

    These are not replacements for the LLM selector. They only catch obvious
    non-simulatable field/longitudinal studies and contradictory reasons such as
    "requires real employees; no self-contained scenario".
    """
    reason_s = str(reason or "")
    if _DROP_REASON_RE.search(reason_s):
        return "deterministic-drop: " + reason_s[:180]

    text = " ".join(
        str(exp.get(key) or "")
        for key in ("experiment_name", "design_type", "conditions_or_factors", "input", "participant_task", "output")
    )
    if _FIELD_STUDY_RE.search(text) and not _SELF_CONTAINED_EXPERIMENT_RE.search(text):
        return "deterministic-drop: field/longitudinal study without self-contained scenario"
    return None


def select_studies(
    experiments: List[Dict[str, Any]],
    client: Any,
    *,
    votes: int = 1,
    timeout: Optional[float] = 60.0,
) -> Dict[str, Dict[str, Any]]:
    """Return {experiment_id: {"keep": bool, "reason": str}} for a paper.

    On any LLM/parse failure a study defaults to keep=True (fail-open: never
    silently drop a study the model could not score). With `votes` > 1 the judge
    is polled several times and each study's decision is a majority vote, which
    stabilizes the borderline calls (ties resolve to keep).
    """
    ids = [str(e.get("experiment_id")) for e in experiments if e.get("experiment_id")]
    decisions: Dict[str, Dict[str, Any]] = {
        eid: {"keep": True, "reason": "default-keep (not scored)"} for eid in ids
    }

    keep_votes: Dict[str, int] = {eid: 0 for eid in ids}
    drop_votes: Dict[str, int] = {eid: 0 for eid in ids}
    reasons: Dict[str, str] = {eid: "" for eid in ids}
    rounds = 0
    for _ in range(max(1, votes)):
        parsed = _select_once(experiments, client, timeout=timeout)
        if parsed is None:
            continue
        rounds += 1
        for d in parsed:
            eid = str(d.get("experiment_id"))
            if eid not in keep_votes:
                continue
            if bool(d.get("keep", True)):
                keep_votes[eid] += 1
            else:
                drop_votes[eid] += 1
            if d.get("reason"):
                reasons[eid] = str(d.get("reason"))[:200]

    if rounds == 0:
        for eid in ids:
            exp = next((item for item in experiments if str(item.get("experiment_id")) == eid), {})
            veto = _deterministic_drop(exp, "")
            if veto:
                decisions[eid] = {"keep": False, "reason": veto}
            else:
                decisions[eid]["reason"] = "selection skipped: LLM unavailable"
        return decisions

    for eid in ids:
        # tie or no votes -> keep (fail-open)
        keep = keep_votes[eid] >= drop_votes[eid]
        suffix = ""
        if votes > 1:
            suffix = f" (votes {keep_votes[eid]}K/{drop_votes[eid]}d)"
        reason = (reasons[eid] + suffix).strip()
        exp = next((item for item in experiments if str(item.get("experiment_id")) == eid), {})
        veto = _deterministic_drop(exp, reason)
        if veto:
            decisions[eid] = {"keep": False, "reason": veto}
        else:
            decisions[eid] = {"keep": keep, "reason": reason}
    return decisions


def selection_summary(decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    kept = [eid for eid, d in decisions.items() if d["keep"]]
    dropped = [eid for eid, d in decisions.items() if not d["keep"]]
    return {
        "total": len(decisions),
        "kept": kept,
        "dropped": dropped,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
    }
