"""The paper's own analysis on replay sessions: OLS of each outcome on condition indicators.

Control is the reference for Favorable and Bribes; innuendo is the reference for Investigation
(control was not asked). Estimates are unweighted: YouGov survey weights and block inverse-probability
weights cannot be reproduced (see program issue i-weights). Reported Table 2 / appendix values are
returned alongside for comparison; they are not used to compute anything.
"""
import math

CONDITIONS = ["control", "innuendo", "denial", "causal"]
DVS = {"favorable": ("control", ["innuendo", "denial", "causal"]),
       "bribes": ("control", ["innuendo", "denial", "causal"]),
       "investigation": ("innuendo", ["denial", "causal"])}
REPORTED = {  # Table 2 (IPW models 1, 3, 5) and appendix pairwise means, paper.pdf pages 15 and 24
    "favorable": {"means": {"control": 2.75, "innuendo": 2.17, "denial": 2.36, "causal": 2.83},
                  "coef_vs_reference": {"innuendo": -0.58, "denial": -0.39, "causal": 0.08}},
    "bribes": {"means": {"control": 2.86, "innuendo": 3.65, "denial": 3.41, "causal": 2.93},
               "coef_vs_reference": {"innuendo": 0.78, "denial": 0.55, "causal": 0.07}},
    "investigation": {"means": {"innuendo": 3.46, "denial": 3.21, "causal": 2.87},
                      "coef_vs_reference": {"denial": -0.25, "causal": -0.58}},
}


def _p(z):
    return math.erfc(abs(z) / math.sqrt(2))  # normal approximation


def _values(sessions, dv, cond):
    return [s["responses"][dv] for s in sessions
            if s.get("condition") == cond and isinstance(s.get("responses", {}).get(dv), (int, float))]


def evaluate(sessions):
    sessions = list(sessions or [])
    out = {"status": "ok", "weights": "unweighted", "n_sessions": len(sessions), "models": {}}
    missing = []
    for dv, (ref, others) in DVS.items():
        groups = {c: _values(sessions, dv, c) for c in [ref] + others}
        short = [c for c, v in groups.items() if len(v) < 2]
        if short:
            missing.append({"outcome": dv, "needs_at_least_2_valid_responses_in": short})
            continue
        means = {c: sum(v) / len(v) for c, v in groups.items()}
        dof = sum(len(v) for v in groups.values()) - len(groups)
        sse = sum((x - means[c]) ** 2 for c, v in groups.items() for x in v)
        s2 = sse / dof if dof > 0 else float("nan")
        def contrast(a, b):
            d = means[a] - means[b]
            se = math.sqrt(s2 * (1 / len(groups[a]) + 1 / len(groups[b])))
            z = d / se if se > 0 else float("nan")
            return {"difference": round(d, 3), "se": round(se, 3), "p_normal_approx": None if z != z else round(_p(z), 4)}
        pair = {}
        names = [ref] + others
        for i, a in enumerate(names):
            for b in names[:i]:
                pair["%s_vs_%s" % (a, b)] = contrast(a, b)
        out["models"][dv] = {
            "reference": ref, "n": {c: len(v) for c, v in groups.items()},
            "means": {c: round(m, 3) for c, m in means.items()},
            "coef_vs_reference": {c: contrast(c, ref) for c in others},
            "pairwise": pair,
            "reported_in_paper": REPORTED[dv],
        }
    if missing and not out["models"]:
        return {"status": "not_ready", "reason": "no outcome has enough valid responses in every needed condition",
                "missing": missing}
    if missing:
        out["status"] = "partial"
        out["not_ready"] = missing
    return out
