"""The paper's own comparisons computed from sessions alone (Franke & Degen 2016, Experiments 1 and 2).

Computed: exclusion of the 15% of participants with the highest distractor-error rate on non-ambiguous
trials, choice proportions per trial type, Helmert-contrast logistic regression of target vs competitor
choices (fixed effects only, so p-values are not those of the paper's mixed-effects model), and each
participant's target rate on simple and complex trials (Fig 2/3 right).
Not computed (returned as not_ready): the JAGS hierarchical model and Bayes factors.
"""
import math

REPORTED = {
    "exp1": {"n_analysed": 51, "unamb": [0.99, 0.0004], "amb": [0.46, 0.51], "simple": [0.77, 0.23], "complex": [0.57, 0.42],
             "beta_simple_vs_complex": 1.28, "beta_complex_vs_amb": 0.44},
    "exp2": {"n_analysed": 50, "unamb": [0.96, 0.01], "amb": [0.48, 0.48], "simple": [0.82, 0.16], "complex": [0.53, 0.45],
             "beta_simple_vs_complex": 1.63, "beta_complex_vs_amb": 0.20},
}


def _solve(a, b):
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("singular")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def _inv(a):
    n = len(a)
    cols = []
    for j in range(n):
        cols.append(_solve(a, [1.0 if i == j else 0.0 for i in range(n)]))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _logit(X, y):
    p = len(X[0])
    beta = [0.0] * p
    for _ in range(60):
        mu = [1 / (1 + math.exp(-sum(b * x for b, x in zip(beta, row)))) for row in X]
        grad = [sum((y[i] - mu[i]) * X[i][j] for i in range(len(X))) for j in range(p)]
        H = [[sum(mu[i] * (1 - mu[i]) * X[i][j] * X[i][k] for i in range(len(X))) + (1e-6 if j == k else 0) for k in range(p)] for j in range(p)]
        step = _solve(H, grad)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    mu = [1 / (1 + math.exp(-sum(b * x for b, x in zip(beta, row)))) for row in X]
    H = [[sum(mu[i] * (1 - mu[i]) * X[i][j] * X[i][k] for i in range(len(X))) + (1e-6 if j == k else 0) for k in range(p)] for j in range(p)]
    cov = _inv(H)
    return beta, [math.sqrt(max(cov[i][i], 0)) for i in range(p)]


def _study(sessions, study):
    ss = [s for s in sessions if s.get("study") == study]
    if not ss:
        return {"status": "not_ready", "missing": ["no sessions for " + study]}
    err = []
    for s in ss:
        t = [x for x in s["trials"] if x["phase"] == "main" and x["analysis_cond"] != "amb" and x["role"] is not None]
        err.append(sum(x["role"] == "distractor" for x in t) / max(len(t), 1))
    k = int(0.15 * len(ss) + 1e-9)
    drop = set(sorted(range(len(ss)), key=lambda i: -err[i])[:k])
    kept = [s for i, s in enumerate(ss) if i not in drop]
    props = {}
    for cond in ("unamb", "amb", "simple", "complex"):
        rows = [x for s in kept for x in s["trials"] if x["phase"] == "main" and x["analysis_cond"] == cond and x["role"]]
        props[cond] = {r: round(sum(x["role"] == r for x in rows) / len(rows), 3) if rows else None for r in ("target", "competitor", "distractor")}
        props[cond]["n_trials"] = len(rows)
    X, y = [], []
    for s in kept:
        for x in s["trials"]:
            if x["phase"] != "main" or x["analysis_cond"] not in ("simple", "complex", "amb") or x["role"] not in ("target", "competitor"):
                continue
            c = x["analysis_cond"]
            h1 = {"simple": 2 / 3, "complex": -1 / 3, "amb": -1 / 3}[c]
            h2 = {"simple": 0.0, "complex": 0.5, "amb": -0.5}[c]
            tr = x["trial"] - 33.5
            X.append([1.0, h1, h2, tr, h1 * tr, h2 * tr])
            y.append(1.0 if x["role"] == "target" else 0.0)
    reg = {"status": "not_ready", "missing": ["too few target/competitor choices or no variation"]}
    if len(set(y)) == 2 and len(y) > 12:
        try:
            b, se = _logit(X, y)
            names = ["Intercept", "CONDITION (HARDER VS. SIMPLE)", "CONDITION (AMBIGUOUS VS. COMPLEX)", "TRIAL", "HARDER.VS.SIMPLE:TRIAL", "AMBIGUOUS.VS.COMPLEX:TRIAL"]
            reg = {"status": "ok", "note": "fixed-effects logistic regression with standard Helmert weights; the paper used a mixed-effects model with control predictors, so values are only roughly comparable",
                   "coefficients": {n: {"beta": round(v, 3), "se": round(e, 3)} for n, v, e in zip(names, b, se)}}
        except (ValueError, OverflowError) as exc:
            reg = {"status": "not_ready", "missing": ["regression did not converge: %s" % exc]}
    indiv = []
    for s in kept:
        r = {}
        for cond in ("simple", "complex"):
            rows = [x for x in s["trials"] if x["phase"] == "main" and x["analysis_cond"] == cond and x["role"]]
            r[cond] = round(sum(x["role"] == "target" for x in rows) / len(rows), 3) if rows else None
        indiv.append(r)
    return {"status": "ok", "n_sessions": len(ss), "n_excluded_for_error_rate": len(drop), "n_analysed": len(kept),
            "proportions": props, "regression": reg, "individual_target_rates": indiv, "paper_reported": REPORTED[study]}


def evaluate(sessions):
    out = {"studies": {s: _study(sessions, s) for s in ("exp1", "exp2")},
           "model_comparison": {"status": "not_ready",
                                "missing": ["JAGS hierarchical type model and Bayes factors (S5 Text)", "salience priors from Experiment 3 (S4 Text) needed by the R2 listener"],
                                "paper_reported": "Bayes factors hom/het (tolerance 0.05): production about 3.955, comprehension about 2.8e-11"}}
    return out


# ---- effects.json: the package-computed effects (built only from what evaluate() returns) ----------

def _z_test(beta, se):
    if beta is None or not se:
        return {"statistic": "z", "value": None, "df": None, "p": None}
    z = beta / se
    return {"statistic": "z", "value": z, "df": None, "p": math.erfc(abs(z) / math.sqrt(2))}


def _t_p(t, df):
    """Two-sided p of a t statistic (regularised incomplete beta, continued fraction)."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def cf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((a + m2) * (qam + m2))
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1 + aa / c if abs(1 + aa / c) > 1e-30 else 1e-30
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1 + aa / c if abs(1 + aa / c) > 1e-30 else 1e-30
            de = d * c
            h *= de
            if abs(de - 1) < 1e-12:
                break
        return h

    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    ib = front * cf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - front * cf(b, a, 1 - x) / b
    return min(1.0, max(0.0, ib))


_REG = {"simple_vs_harder": "CONDITION (HARDER VS. SIMPLE)", "complex_vs_amb": "CONDITION (AMBIGUOUS VS. COMPLEX)",
        "trial": "TRIAL"}
_REG_METHOD = ("fixed-effects logistic regression of target vs competitor choices on Helmert-coded CONDITION (simple vs "
               "complex+ambiguous: +2/3, -1/3, -1/3; complex vs ambiguous: +1/2, -1/2), centred trial number and its "
               "interactions with both contrasts, after excluding the 15%% of participants with the highest distractor "
               "rate on non-ambiguous trials and all distractor choices; coefficient %s with Wald z. No random effects and "
               "no target-position or message-type controls, so this is not the paper's mixed-effects b (Table 2) and "
               "its p-value is not the paper's")


def effect_estimates(evaluation):
    out = {}
    for study in ("exp1", "exp2"):
        st = (evaluation.get("studies") or {}).get(study) or {}
        nr = lambda why: {"status": "not_ready", "reason": why}
        if st.get("status") != "ok":
            why = "; ".join(st.get("missing") or ["no analysable sessions for " + study])
            for k in list(_REG) + ["rd_simple_complex", "rd_complex_amb"]:
                out["%s_%s" % (study, k)] = nr(why)
            continue
        n = st.get("n_analysed")
        reg = st.get("regression") or {}
        coefs = reg.get("coefficients") or {}
        for k, name in _REG.items():
            c = coefs.get(name)
            if reg.get("status") != "ok" or not c:
                out["%s_%s" % (study, k)] = nr("; ".join(reg.get("missing") or ["regression not computed"]))
                continue
            out["%s_%s" % (study, k)] = {
                "status": "computed", "metric": "other", "value": c["beta"], "se": c["se"], "n": n, "cells": {},
                "test": _z_test(c["beta"], c["se"]), "method": _REG_METHOD % name}
        props = st.get("proportions") or {}

        def cell(cond):
            p = props.get(cond) or {}
            return {"n": p.get("n_trials"), "proportion": p.get("target"),
                    "competitor": p.get("competitor"), "distractor": p.get("distractor")}

        for k, t, r in (("rd_simple_complex", "simple", "complex"), ("rd_complex_amb", "complex", "amb")):
            ct, cr = cell(t), cell(r)
            if ct["proportion"] is None or cr["proportion"] is None:
                out["%s_%s" % (study, k)] = nr("no %s or %s trials after exclusions" % (t, r))
                continue
            block = {"status": "computed", "metric": "risk_difference", "value": ct["proportion"] - cr["proportion"],
                     "se": None, "n": n, "cells": {t: ct, r: cr},
                     "test": {"statistic": "other", "value": None, "df": None, "p": None},
                     "method": ("share of target choices among all %s trials minus share among all %s trials (target, "
                                "competitor and distractor choices all counted, as in the paper's percentages), pooled "
                                "over the participants kept after the 15%% error-rate exclusion" % (t, r))}
            if k == "rd_simple_complex":
                d = [x["simple"] - x["complex"] for x in st.get("individual_target_rates") or []
                     if x.get("simple") is not None and x.get("complex") is not None]
                if len(d) > 1:
                    m = sum(d) / len(d)
                    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1))
                    se = sd / math.sqrt(len(d))
                    block["se"] = se
                    block["cells"]["per_participant_difference"] = {"n": len(d), "mean": m, "sd": sd}
                    if se > 0:
                        tv = m / se
                        block["test"] = {"statistic": "t", "value": tv, "df": len(d) - 1, "p": _t_p(tv, len(d) - 1)}
                    block["method"] += ("; se and paired t from each participant's simple minus complex target rate "
                                        "(12 trials per condition, so their mean equals the pooled difference)")
            out["%s_%s" % (study, k)] = block
    mc = evaluation.get("model_comparison") or {}
    why = "; ".join(mc.get("missing") or ["the hierarchical model is not run by the package"])
    out["bf_production"] = {"status": "not_ready", "reason": why}
    out["bf_comprehension"] = {"status": "not_ready", "reason": why}
    return out
