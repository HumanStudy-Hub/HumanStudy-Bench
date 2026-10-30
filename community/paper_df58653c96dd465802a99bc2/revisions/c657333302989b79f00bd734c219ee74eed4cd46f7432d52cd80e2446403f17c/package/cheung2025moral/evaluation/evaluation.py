#!/usr/bin/env python3
"""The paper's own comparisons, computed from AI-participant sessions alone.

Study 1: CBR option chosen when it coincides with omission vs action (within participant).
Study 2: Yes-No bias (yes = CBR vs no = CBR; original + Yes-No arms) and omission bias
(CBR on omission vs action; original + Action-Omission arms).
The paper's human-versus-LLM ANOVAs, Study 1 collective action problems, Study 3 and Study 4 are
reported as not_ready because the data they need are not part of these sessions.
"""
import math


def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-30, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-12:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def _p_two_sided(t, df):
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def _mean(xs):
    return sum(xs) / len(xs)


def _var(xs):
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _welch(a, b):
    """a, b: lists of 0/1 outcomes. Returns difference of means (a - b) with Welch t."""
    if len(a) < 2 or len(b) < 2:
        return {"n_a": len(a), "n_b": len(b), "status": "not_ready", "reason": "fewer than two valid responses in a group"}
    va, vb = _var(a) / len(a), _var(b) / len(b)
    out = {"n_a": len(a), "n_b": len(b), "mean_a": _mean(a), "mean_b": _mean(b), "difference": _mean(a) - _mean(b)}
    se = math.sqrt(va + vb)
    if se == 0:
        out.update({"t": None, "df": None, "p": None, "note": "no variance in either group"})
        return out
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = out["difference"] / se
    out.update({"t": t, "df": df, "p": _p_two_sided(t, df)})
    return out


def _valid(trials):
    return [t for t in trials if t.get("cbr") is not None]


def _study1(sessions):
    diffs, n_valid, n_all = [], 0, 0
    for s in sessions:
        if s.get("study") != "s1":
            continue
        n_all += 1
        tr = _valid(s["trials"])
        act = [t["cbr"] for t in tr if t["framing"] == "action"]
        omi = [t["cbr"] for t in tr if t["framing"] == "omission"]
        if act and omi:
            n_valid += 1
            diffs.append((_mean(omi), _mean(act)))
    if not n_all:
        return None
    if n_valid < 2:
        return {"status": "not_ready", "reason": "fewer than two participants with valid answers in both framings", "participants": n_all}
    d = [o - a for o, a in diffs]
    out = {"status": "ok", "participants": n_valid,
           "proportion_cbr_when_action": _mean([a for _, a in diffs]), "proportion_cbr_when_omission": _mean([o for o, _ in diffs]),
           "omission_minus_action": _mean(d)}
    if _var(d) > 0:
        t = _mean(d) / math.sqrt(_var(d) / len(d))
        out.update({"paired_t": t, "df": len(d) - 1, "p": _p_two_sided(t, len(d) - 1)})
    else:
        out.update({"paired_t": None, "note": "no variance in the paired differences"})
    return out


def _arm_trials(sessions, arm):
    return [t for s in sessions if s.get("study") == "s2" and s.get("arm") == arm for t in s["trials"]]


def _study2(sessions):
    if not any(s.get("study") == "s2" for s in sessions):
        return None
    arms = {a: _valid(_arm_trials(sessions, a)) for a in ("s2_original", "s2_yes_no", "s2_omission")}
    out = {"per_arm": {a: {"n": len(t), "proportion_cbr": _mean([x["cbr"] for x in t]) if t else None} for a, t in arms.items()}}
    yn = arms["s2_original"] + arms["s2_yes_no"]
    if arms["s2_original"] and arms["s2_yes_no"]:
        yes = [t["cbr"] for t in yn if t["yes_is_cbr"]]
        no = [t["cbr"] for t in yn if not t["yes_is_cbr"]]
        r = _welch(yes, no)
        out["yes_no_bias"] = dict(r, definition="proportion CBR when yes = CBR minus proportion CBR when no = CBR; negative = bias toward answering no")
    else:
        out["yes_no_bias"] = {"status": "not_ready", "reason": "needs both the s2_original and s2_yes_no arms"}
    om = arms["s2_original"] + arms["s2_omission"]
    if arms["s2_original"] and arms["s2_omission"]:
        on_omission = [t["cbr"] for t in om if not t["cbr_is_action"]]
        on_action = [t["cbr"] for t in om if t["cbr_is_action"]]
        r = _welch(on_omission, on_action)
        out["omission_bias"] = dict(r, definition="proportion CBR when it coincides with omission minus when it coincides with action; positive = omission bias")
    else:
        out["omission_bias"] = {"status": "not_ready", "reason": "needs both the s2_original and s2_omission arms"}
    return out


def evaluate(sessions):
    result = {"note": "AI participants: a simulation, not observations of people. Normal-theory p values.",
              "not_ready": [
                  {"analysis": "Study 1 collective action problems (altruism score, t-tests)", "missing": "vignettes and slider ranges not available"},
                  {"analysis": "Study 3 framing effects", "missing": "vignettes not available"},
                  {"analysis": "Study 4 model comparison", "missing": "model checkpoints; no human participants to replace"},
                  {"analysis": "Human-versus-LLM ANOVA interactions (F tests) in Studies 1-3", "missing": "requires the human and LLM response data, which are not part of these sessions"}]}
    s1, s2 = _study1(sessions), _study2(sessions)
    if s1 is not None:
        result["study_1_omission_bias"] = s1
    if s2 is not None:
        result["study_2"] = s2
    if s1 is None and s2 is None:
        result["status"] = "not_ready"
    return result


if __name__ == "__main__":
    print(__doc__)


# ---- effect_estimates: blocks for effects.json (compute.op = "package"), built from evaluate() ----------

_NOT_READY = {
    "paper_llm_respondents": "the paper's LLM respondents (each model x prompt) are a separate sample whose "
                             "responses are not part of these sessions",
    "llm_vs_human_anova": "the framing x model ANOVA needs the paper's LLM response data alongside the human data; "
                          "not part of these sessions",
    "s1_collective_action": "Study 1 collective action problems: vignettes, slider ranges and responses are not "
                            "available",
    "s3_humans": "Study 3: human data and five of six vignettes are not available",
    "s4_models": "Study 4 compares model checkpoints (pre-trained Llama 3.1, Llama 3.1-Instruct, Centaur); their "
                 "responses are not part of these sessions",
}


def _diff_block(r, t_name, r_name, method):
    if not r or r.get("status") == "not_ready" or r.get("difference") is None:
        return {"status": "not_ready", "reason": (r or {}).get("reason") or "not computed"}
    t, df, p = r.get("t"), r.get("df"), r.get("p")
    se = abs(r["difference"] / t) if t else None
    cells = {t_name: {"n": r["n_a"], "k": round(r["mean_a"] * r["n_a"]), "proportion": r["mean_a"]},
             r_name: {"n": r["n_b"], "k": round(r["mean_b"] * r["n_b"]), "proportion": r["mean_b"]}}
    return {"status": "computed", "metric": "risk_difference", "value": r["difference"], "se": se,
            "n": r["n_a"] + r["n_b"], "cells": cells,
            "test": {"statistic": "t", "value": t, "df": df, "p": p}, "method": method}


def effect_estimates(evaluation):
    out = {k: {"status": "not_ready", "reason": v} for k, v in _NOT_READY.items()}
    s1 = evaluation.get("study_1_omission_bias")
    if not s1:
        out["s1_omission_bias"] = {"status": "not_ready", "reason": "no Study 1 sessions (the Study 1 human data are "
                                                                   "not in the human-data folder)"}
    elif s1.get("status") != "ok":
        out["s1_omission_bias"] = {"status": "not_ready", "reason": s1.get("reason") or "not computed"}
    else:
        t = s1.get("paired_t")
        d = s1["omission_minus_action"]
        out["s1_omission_bias"] = {
            "status": "computed", "metric": "risk_difference", "value": d,
            "se": abs(d / t) if t else None, "n": s1["participants"],
            "cells": {"omission": {"n": s1["participants"], "proportion": s1["proportion_cbr_when_omission"]},
                      "action": {"n": s1["participants"], "proportion": s1["proportion_cbr_when_action"]}},
            "test": {"statistic": "t", "value": t, "df": s1.get("df"), "p": s1.get("p")},
            "method": "per-participant proportion CBR on omission minus on action vignettes; paired t-test"}
    s2 = evaluation.get("study_2") or {}
    out["s2_yes_no_bias"] = _diff_block(
        s2.get("yes_no_bias"), "yes = CBR", "no = CBR",
        "proportion CBR when yes = CBR minus when no = CBR, original + Yes-No arms pooled; Welch t on 0/1 choices")
    out["s2_omission_bias"] = _diff_block(
        s2.get("omission_bias"), "omission", "action",
        "proportion CBR when CBR = omission minus CBR = action, original + Action-Omission arms pooled; Welch t")
    return out
