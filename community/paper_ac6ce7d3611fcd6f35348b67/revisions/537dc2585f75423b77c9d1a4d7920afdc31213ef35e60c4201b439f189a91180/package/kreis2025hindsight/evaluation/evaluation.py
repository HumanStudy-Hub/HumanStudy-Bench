"""Evaluation for Kreis et al. (2025), study s1. evaluate(sessions) derives everything from the sessions.

The paper's tests are Bayesian linear mixed-effects models (brms, bridge-sampling Bayes factors). These are not
reproduced offline: this module applies the paper's exclusion rules and reports participant-level paired
comparisons of |Δ| (approximation: mean difference, SD, t, df, dz, no p-values) and marks Bayes factors not_ready.
"""
import math
import statistics as st

REPORTED = {
    "transfer_learning_b": {"feedback": -6.59, "domain": -3.71, "irrelevant": 0.52, "control": -0.92},
    "hindsight_b": {"feedback": -2.88, "domain": -2.25, "irrelevant": 0.87, "control": 1.10},
    "source": "Kreis et al. (2025), Results, pages 17-18 (b on |Δ| from Bayesian mixed models; negative = more accurate than OJ)",
}
ARMS = ("control", "feedback", "domain", "irrelevant")


def _quartiles(xs):
    xs = sorted(xs)
    q = st.quantiles(xs, n=4, method="inclusive") if len(xs) >= 2 else [xs[0]] * 3
    return q[0], q[2]


def _valid(est, weight):
    return est is not None and 0 <= est <= weight


def _paired(diffs):
    n = len(diffs)
    if n < 2:
        return {"n": n, "status": "not_ready", "reason": "fewer than 2 participants"}
    m, sd = st.mean(diffs), st.stdev(diffs)
    t = m / (sd / math.sqrt(n)) if sd > 0 else None
    return {"n": n, "mean_diff": round(m, 4), "sd": round(sd, 4), "t": None if t is None else round(t, 3), "df": n - 1, "dz": None if sd == 0 else round(m / sd, 3)}


def _welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return {"status": "not_ready", "reason": "fewer than 2 participants in a group"}
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    return {"mean_diff": round(st.mean(a) - st.mean(b), 4), "t": None if se == 0 else round((st.mean(a) - st.mean(b)) / se, 3)}


def _person(s):
    """Per-participant item-level |Δ| after the paper's item rules."""
    oj, roj, nj = {}, {}, {}
    for t in s["oj"]:
        oj[t["item"]] = t
    for t in s["roj"]:
        roj[t["item"]] = t
    pairs = []
    for i, o in oj.items():
        r = roj.get(i)
        if r is None or not _valid(o["estimate"], o["weight"]) or not _valid(r["estimate"], r["weight"]):
            continue
        if o["estimate"] == o["actual"] or r["estimate"] == o["estimate"]:  # perfect OJ, perfect reproduction
            continue
        pairs.append((abs(o["estimate"] - o["actual"]), abs(r["estimate"] - r["actual"])))
    nj_d = [abs(t["estimate"] - t["actual"]) for t in s["nj"] if _valid(t["estimate"], t["weight"])]
    return pairs, nj_d, [abs(t["estimate"] - t["actual"]) for t in s["oj"] if _valid(t["estimate"], t["weight"])], \
        [abs(t["estimate"] - t["actual"]) for t in s["roj"] if _valid(t["estimate"], t["weight"])]


def evaluate(sessions):
    if not sessions:
        return {"status": "not_ready", "reason": "no sessions"}
    people = []
    for s in sessions:
        pairs, nj_d, oj_all, roj_all = _person(s)
        people.append({"arm": s.get("arm"), "pairs": pairs, "nj": nj_d, "med": [st.median(x) if x else None for x in (oj_all, roj_all, nj_d)]})
    excluded = []
    if len(people) >= 8:  # Q3 + 3 x IQR of person-level median |Δ| per task (interpretation, see issue iss_outlier)
        limits = []
        for k in range(3):
            vals = [p["med"][k] for p in people if p["med"][k] is not None]
            q1, q3 = _quartiles(vals) if vals else (0, 0)
            limits.append(q3 + 3 * (q3 - q1))
        for p in people:
            if any(p["med"][k] is not None and p["med"][k] > limits[k] for k in range(3)):
                p["out"] = True
                excluded.append(p["arm"])
    groups = {}
    for arm in ARMS:
        ps = [p for p in people if p["arm"] == arm and not p.get("out") and p["pairs"] and p["nj"]]
        if not any(p["arm"] == arm for p in people):
            groups[arm] = {"status": "not_run"}
            continue
        oj = [st.mean(a for a, _ in p["pairs"]) for p in ps]
        roj = [st.mean(b for _, b in p["pairs"]) for p in ps]
        njm = [st.mean(p["nj"]) for p in ps]
        hs = [r - o for o, r in zip(oj, roj)]
        tl = [n - o for o, n in zip(oj, njm)]
        groups[arm] = {"n": len(ps), "mean_abs_dev": {"OJ": round(st.mean(oj), 3) if oj else None, "ROJ": round(st.mean(roj), 3) if roj else None, "NJ": round(st.mean(njm), 3) if njm else None},
                       "hindsight_ROJ_minus_OJ": _paired(hs), "transfer_NJ_minus_OJ": _paired(tl), "_hs": hs, "_tl": tl}
    inter = {}
    for name, (a, b) in {"feedback_vs_domain": ("feedback", "domain"), "irrelevant_vs_control": ("irrelevant", "control")}.items():
        ga, gb = groups.get(a, {}), groups.get(b, {})
        if "_hs" in ga and "_hs" in gb:
            inter[name] = {"hindsight": _welch(ga["_hs"], gb["_hs"]), "transfer": _welch(ga["_tl"], gb["_tl"])}
        else:
            inter[name] = {"status": "not_ready", "reason": "arm not run"}
    for g in groups.values():
        g.pop("_hs", None), g.pop("_tl", None)
    return {"status": "approximate", "note": "Participant-level paired comparisons of |Δ| after the paper's exclusions; not the paper's Bayesian mixed models.",
            "groups": groups, "interactions": inter,
            "bayes_factors": {"status": "not_ready", "reason": "brms/Stan bridge-sampling Bayes factors, priors and MCMC settings are not available offline (issue iss_mcmc)"},
            "participants_excluded_outlier": excluded, "reported_in_paper": REPORTED}
