#!/usr/bin/env python3
"""The paper's main comparison from AI-participant sessions: update value on neutral-rated trials,
upwards vs downwards direction of error, per study (paper pp. 8-10).

evaluate(sessions) -> dict. Everything is derived from the sessions. Returns per study the trial counts,
cell means and SDs, the signed-update coding of the paper, and a random-intercept mixed model of
update ~ direction (statsmodels, Wald test; the paper used Satterthwaite F, so the statistic is not the
same). When data are too thin, or statsmodels is missing, the affected test is {"status": "not_ready", ...}.
Reported values are returned beside the simulated ones for comparison; agreement is not evidence about people.
"""
import math
from collections import defaultdict

REPORTED = {
    "study1": {"n_up": 801, "mean_up": 2.11, "n_down": 720, "mean_down": 11.33, "F": "F(1,1515) = 244.47", "estimate": 9.13},
    "study2": {"n_up": 831, "mean_up": 4.16, "n_down": 868, "mean_down": 10.08, "F": "F(1,1694) = 77.09", "estimate": 6.24},
    "study3": {"n_up": 828, "mean_up": 4.13, "n_down": 839, "mean_down": 10.56, "F": "F(1,1662) = 68.48", "estimate": 6.51},
}


def _valence_class(v):
    return None if v is None else "negative" if v <= 2 else "neutral" if v == 3 else "positive"


def trial_rows(session):
    rows = []
    for t in session.get("trials", []):
        e1, e2, br = t.get("E1"), t.get("E2"), t.get("BR")
        if None in (e1, e2, br) or e1 == br:
            continue
        direction = "upwards" if br > e1 else "downwards"
        move = e2 - e1
        update = abs(move) if (move > 0) == (br > e1) and move != 0 else (-abs(move) if move != 0 else 0.0)
        rows.append({"participant": session["participant"], "direction": direction, "update": update,
                     "valence": _valence_class(t.get("valence"))})
    return rows


def _quartiles(xs):
    xs = sorted(xs)
    n = len(xs)

    def q(p):
        k = (n - 1) * p
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    return q(0.25), q(0.75)


def _drop_outliers(rows):
    """Paper p. 7: mean updates in each of the six valence x direction cells, outliers beyond 3 x IQR removed.
    Applied here to participant-level cell means (the unit is ambiguous in the paper: see iss-exclusion-detail)."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["valence"], r["direction"])][r["participant"]].append(r["update"])
    bad = set()
    for key, per in cells.items():
        means = {p: sum(v) / len(v) for p, v in per.items()}
        if len(means) < 4:
            continue
        q1, q3 = _quartiles(list(means.values()))
        for p, m in means.items():
            if m < q1 - 3 * (q3 - q1) or m > q3 + 3 * (q3 - q1):
                bad.add((key, p))
    return [r for r in rows if ((r["valence"], r["direction"]), r["participant"]) not in bad]


def _mean_sd(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return m, sd


def _lmm(rows):
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return {"status": "not_ready", "reason": "statsmodels/pandas not installed"}
    df = pd.DataFrame(rows)
    if df["direction"].nunique() < 2 or df["participant"].nunique() < 3 or len(df) < 10:
        return {"status": "not_ready", "reason": "need both directions, at least 3 participants and 10 neutral trials"}
    try:
        fit = smf.mixedlm("update ~ C(direction, Treatment('downwards'))", df, groups=df["participant"]).fit(reml=True)
    except Exception as exc:  # singular fits etc.
        return {"status": "not_ready", "reason": "model did not fit: %s" % str(exc)[:120]}
    name = [k for k in fit.params.index if "upwards" in k][0]
    return {"status": "ok", "estimate_upwards_vs_downwards": float(fit.params[name]), "se": float(fit.bse[name]),
            "z": float(fit.tvalues[name]), "p": float(fit.pvalues[name]),
            "note": "Wald z from a random-intercept model; the paper reports a Satterthwaite F"}


def evaluate(sessions):
    if not sessions:
        return {"status": "not_ready", "reason": "no sessions"}
    out = {"status": "ok", "studies": {}, "comparison_note": "Simulated AI participants; reported values are the paper's, shown for reference."}
    for study in sorted({s["study"] for s in sessions}):
        subset = [s for s in sessions if s["study"] == study]
        rows = [r for s in subset for r in trial_rows(s)]
        kept = _drop_outliers(rows)
        neutral = [r for r in kept if r["valence"] == "neutral"]
        up = [r["update"] for r in neutral if r["direction"] == "upwards"]
        down = [r["update"] for r in neutral if r["direction"] == "downwards"]
        mu, su = _mean_sd(up)
        md, sd = _mean_sd(down)
        entry = {"participants": len(subset), "trials_scored": len(rows), "trials_after_outlier_rule": len(kept),
                 "neutral": {"upwards": {"n": len(up), "mean": mu, "sd": su}, "downwards": {"n": len(down), "mean": md, "sd": sd}},
                 "reported": REPORTED.get(study)}
        entry["lmm"] = _lmm(neutral) if neutral else {"status": "not_ready", "reason": "no trial was rated neutral"}
        # secondary analysis cells (descriptives only; the valence x direction LMM is not recomputed)
        entry["by_valence"] = {v: {d: len([r for r in kept if r["valence"] == v and r["direction"] == d]) for d in ("upwards", "downwards")}
                               for v in ("negative", "neutral", "positive")}
        entry["secondary_lmm"] = {"status": "not_ready", "reason": "valence x direction model with random slopes is not implemented offline; compare the cell counts and means"}
        out["studies"][study] = entry
    out["aggregate_table1"] = {"status": "not_ready", "reason": "Bayesian difference/ratio, learning-rate and regression analyses need SI Supplementary Analyses 1.1-1.4, which are not available"}
    return out


if __name__ == "__main__":
    print(__doc__)


# ---- effects.json link (appended) ---------------------------------------------------------------------

def effect_estimates(evaluation):
    """Result blocks for effects.json (compute.op = "package"), read from what evaluate() returned.

    neutral_asymmetry_<study>: downwards minus upwards update on neutral-rated trials. The random-intercept
    model's fixed effect (sign flipped from evaluate's upwards-vs-downwards coding) when it fit; otherwise the
    difference of the neutral-trial cell means. Every other key is not_ready: evaluate() does not compute it."""
    out = {}
    studies = (evaluation or {}).get("studies") or {}
    for study in ("study1", "study2", "study3"):
        e = studies.get(study)
        key = "neutral_asymmetry_%s" % study
        if not e:
            out[key] = {"status": "not_ready", "reason": "no sessions for %s" % study}
        else:
            nu = e.get("neutral") or {}
            up, down = nu.get("upwards") or {}, nu.get("downwards") or {}
            cells = {"downwards": dict(down), "upwards": dict(up)}
            n = (up.get("n") or 0) + (down.get("n") or 0)
            lmm = e.get("lmm") or {}
            if lmm.get("status") == "ok":
                out[key] = {"status": "computed", "metric": "regression_coefficient",
                            "value": -lmm["estimate_upwards_vs_downwards"], "se": lmm["se"], "n": n, "cells": cells,
                            "test": {"statistic": "z", "value": -lmm["z"], "df": None, "p": lmm["p"]},
                            "method": "statsmodels MixedLM (REML), update ~ direction, random intercepts by participant, "
                                      "neutral trials; Wald z (the paper used a Satterthwaite F)"}
            elif up.get("mean") is not None and down.get("mean") is not None:
                out[key] = {"status": "computed", "metric": "mean_difference",
                            "value": down["mean"] - up["mean"], "se": None, "n": n, "cells": cells,
                            "test": {"statistic": "other", "value": None, "df": None, "p": None},
                            "method": "difference of neutral-trial cell means (the mixed model did not run: %s)"
                                      % lmm.get("reason", "no model")}
            else:
                out[key] = {"status": "not_ready", "reason": "no neutral trials in one direction of error"}
        for val in ("negative", "positive"):
            out["%s_asymmetry_%s" % (val, study)] = {
                "status": "not_ready",
                "reason": "evaluate() returns only trial counts for %s-rated trials, not cell means" % val}
        out["direction_all_valences_%s" % study] = {
            "status": "not_ready", "reason": "the valence x direction LMM is not implemented (secondary_lmm)"}
        out["valence_x_direction_%s" % study] = {
            "status": "not_ready", "reason": "the valence x direction LMM is not implemented (secondary_lmm)"}
    reason = (evaluation or {}).get("aggregate_table1", {}).get("reason") or "Table 1 techniques are not implemented"
    for tech in ("bayes_diff", "bayes_ratio", "learning_rates", "regression"):
        for study in ("study1", "study2", "study3", "agg123"):
            out["table1_%s_%s" % (tech, study)] = {"status": "not_ready", "reason": reason}
    return out
