"""evaluate(sessions) -> dict. Derives everything from the sessions alone.

Fits the 2x2 OLS model of the paper's within-site analysis: likelihood ~ tempt * load, with the
effect of tempting fate in the no-load condition as the main effect (treatment coding) and the
Tempting Fate x Cognitive Load interaction. The paper's mixed model across sites does not apply to a
single simulated population, so only the within-site OLS is computed.
"""
import math

REPORTED = {
    "original_study_6": {"main_effect": 1.03, "main_ci": [0.09, 1.97], "interaction": 1.54, "interaction_ci": [0.05, 3.03]},
    "similar_sites": {"main_effect": 0.11, "main_ci": [-0.34, 0.56], "interaction": -0.02, "interaction_ci": [-0.68, 0.63]},
    "mturk_rpp_protocol": {"main_effect": 0.21, "main_ci": [-0.01, 0.43], "interaction": -0.20, "interaction_ci": [-0.53, 0.13]},
}


def evaluate(sessions):
    cells = {(0, 0): [], (1, 0): [], (0, 1): [], (1, 1): []}
    for s in sessions or []:
        v = (s.get("responses") or {}).get("likelihood")
        if v is not None and (s.get("tempting_fate"), s.get("cognitive_load")) in {(a, b) for a in (0, 1) for b in (0, 1)}:
            cells[(int(s["tempting_fate"]), int(s["cognitive_load"]))].append(float(v))
    short = [k for k, v in cells.items() if len(v) < 2]
    if short:
        return {"status": "not_ready", "missing": "at least 2 valid likelihood ratings in each cell (tempting fate, load); short cells: %s" % short,
                "reported": REPORTED}
    mean = {k: sum(v) / len(v) for k, v in cells.items()}
    n = {k: len(v) for k, v in cells.items()}
    df = sum(n.values()) - 4
    s2 = sum(sum((x - mean[k]) ** 2 for x in v) for k, v in cells.items()) / df if df > 0 else float("nan")
    main = mean[(1, 0)] - mean[(0, 0)]
    inter = (mean[(1, 1)] - mean[(0, 1)]) - main
    se_main = math.sqrt(s2 * (1 / n[(1, 0)] + 1 / n[(0, 0)]))
    se_int = math.sqrt(s2 * sum(1 / x for x in n.values()))
    try:
        from scipy import stats
        crit = stats.t.ppf(0.975, df)
    except Exception:
        crit = 1.96
    ci = lambda b, se: [round(b - crit * se, 3), round(b + crit * se, 3)]
    return {"status": "ok", "type": "simulation", "n": n, "cell_means": {"%d%d" % k: round(v, 3) for k, v in mean.items()},
            "main_effect": {"estimate": round(main, 3), "se": round(se_main, 3), "ci95": ci(main, se_main)},
            "interaction": {"estimate": round(inter, 3), "se": round(se_int, 3), "ci95": ci(inter, se_int)},
            "reported": REPORTED,
            "note": "Simulation result next to the paper's reported estimates; agreement would not show that people behave the same way."}


# ---- effects.json link (appended): effect blocks for compute.op = "package" ----------------------------------

def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta (Numerical Recipes, Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
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
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-13:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_p(t, df):
    """Two-sided p of Student's t; standard library only."""
    if t is None or df is None or df <= 0:
        return None
    return min(1.0, _ibeta(df / 2.0, 0.5, df / (df + t * t)))


_NOT_LOADED = ("the package's human loader keeps only the similar university sites ('b.similar'), and its "
               "evaluation fits one pooled 2x2 OLS; this estimate needs %s")
_NOT_READY = {
    "tf_main_mturk": _NOT_LOADED % "the MTurk subjects",
    "tf_main_dissimilar": _NOT_LOADED % "the dissimilar-site subjects and the paper's mixed model",
    "tf_main_universities": _NOT_LOADED % "all university sites and the paper's mixed model",
    "tf_main_ml2_ml5_pooled": "needs Many Labs 2 subject-level data, which this package does not hold",
    "tf_main_ml5_minus_ml2": "needs Many Labs 2 subject-level data, which this package does not hold",
    "tf_x_load_mturk": _NOT_LOADED % "the MTurk subjects",
    "tf_x_load_dissimilar": _NOT_LOADED % "the dissimilar-site subjects and the paper's mixed model",
    "tf_x_load_universities": _NOT_LOADED % "all university sites and the paper's mixed model",
    "site_main_similar_vs_mturk": _NOT_LOADED % "MTurk subjects and a mixed model across site types",
    "site_int_similar_vs_mturk": _NOT_LOADED % "MTurk subjects and a mixed model across site types",
    "site_main_similar_vs_dissimilar": _NOT_LOADED % "dissimilar-site subjects and a mixed model across site types",
    "site_int_similar_vs_dissimilar": _NOT_LOADED % "dissimilar-site subjects and a mixed model across site types",
    "site_main_university_vs_mturk": _NOT_LOADED % "all site types and a mixed model",
    "site_int_university_vs_mturk": _NOT_LOADED % "all site types and a mixed model",
    "sat_moderates_main": _NOT_LOADED % "all US universities with their median SAT and a mixed model",
    "sat_moderates_int": _NOT_LOADED % "all US universities with their median SAT and a mixed model",
    "load_effort_mturk_vs_university": _NOT_LOADED % "MTurk subjects' effort ratings and a mixed model",
    "load_diff_mturk_vs_university": _NOT_LOADED % "MTurk subjects' difficulty ratings and a mixed model",
    "importance_mturk_vs_similar": _NOT_LOADED % "MTurk subjects' importance ratings and a mixed model",
    "importance_mturk_vs_dissimilar": _NOT_LOADED % "MTurk and dissimilar-site importance ratings",
    "negativity_mturk_vs_similar": _NOT_LOADED % "MTurk subjects' negativity ratings and a mixed model",
    "negativity_mturk_vs_dissimilar": _NOT_LOADED % "MTurk and dissimilar-site negativity ratings",
}


def effect_estimates(evaluation):
    """{key: block} for effects.json entries with compute.op = "package", built from evaluate()'s output."""
    out = {k: {"status": "not_ready", "reason": r} for k, r in _NOT_READY.items()}
    if not evaluation or evaluation.get("status") != "ok":
        out["tf_x_load_similar"] = {"status": "not_ready",
                                    "reason": (evaluation or {}).get("missing") or "evaluation not computed"}
        return out
    n, means, inter = evaluation["n"], evaluation["cell_means"], evaluation["interaction"]
    cells = {"c%d%d" % k: {"n": v, "mean": means["%d%d" % k]} for k, v in n.items()}
    df = sum(n.values()) - 4
    est, se = inter["estimate"], inter["se"]
    t = est / se if se else None
    out["tf_x_load_similar"] = {
        "status": "computed", "metric": "regression_coefficient", "value": est, "se": se, "n": sum(n.values()),
        "cells": cells, "test": {"statistic": "t", "value": t, "df": df, "p": _t_p(t, df)},
        "method": "OLS likelihood ~ tempt * load (treatment coding), tempt:load coefficient = (c11 - c01) - (c10 - c00); "
                  "pooled residual variance, df = N - 4 (as evaluate() computes it)"}
    return out
