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
