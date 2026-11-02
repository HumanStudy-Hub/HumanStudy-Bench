"""Scores sessions from task/adapter.py run_sessions with the paper's comparisons (offline approximations).

The paper used Bayesian models (see audit/missing_information.json: i-priors). This offline evaluation reports
the same quantities with frequentist summaries and says so. Everything derives from `sessions` alone plus the
replication outcomes recorded in task/task.json.
"""
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, stdev

HERE = Path(__file__).resolve().parent
TASK = json.loads((HERE.parent / "task" / "task.json").read_text(encoding="utf-8"))
OUTCOME = {k: v == "replicated" for k, v in TASK["replicationOutcomes"].items()}
PAPER = {"accuracy": {"description_only": 0.59, "description_plus_evidence": 0.67},
         "auc": {"description_only": 0.62, "description_plus_evidence": 0.74},
         "log_brier_delta_median": -0.96,
         "confidence_effect_size_rho": {"description_only": 0.61, "description_plus_evidence": 0.77}}
ND = NormalDist()


def _p_replicate(a):
    c = a["confidence"] / 100.0
    return c if a["replicate"] == "yes" else 1.0 - c


def _participant(s):
    ans = [a for a in s["answers"] if not a["not_understood"]]
    if not ans:
        return None
    acc = mean(1.0 if (a["replicate"] == "yes") == OUTCOME[a["key"]] else 0.0 for a in ans)
    brier = mean((_p_replicate(a) - (1.0 if OUTCOME[a["key"]] else 0.0)) ** 2 for a in ans)
    return {"accuracy": acc, "brier": brier, "log_brier": math.log(max(brier, 1e-6)), "answers": ans}


def _t(xs, mu=0.0):
    if len(xs) < 2:
        return None
    sd = stdev(xs)
    return None if sd == 0 else (mean(xs) - mu) / (sd / math.sqrt(len(xs)))


def _sdt(rows):
    h = f = nh = nf = 0
    for r in rows:
        for a in r["answers"]:
            yes = a["replicate"] == "yes"
            if OUTCOME[a["key"]]:
                nh += 1; h += yes
            else:
                nf += 1; f += yes
    if not nh or not nf:
        return None
    hr = (h + 0.5) / (nh + 1); fr = (f + 0.5) / (nf + 1)
    d = ND.inv_cdf(hr) - ND.inv_cdf(fr)
    return {"hit_rate": hr, "false_alarm_rate": fr, "d_prime": d, "bias_c": -(ND.inv_cdf(hr) + ND.inv_cdf(fr)) / 2,
            "auc": ND.cdf(d / math.sqrt(2))}


def evaluate(sessions):
    valid = [s for s in sessions if s.get("valid")]
    if not valid:
        return {"status": "not_ready", "missing": "no valid sessions to score",
                "invalid": [s.get("error") for s in sessions]}
    arms = {}
    for s in valid:
        p = _participant(s)
        if p:
            arms.setdefault(s["arm"], []).append(p)
    out = {"status": "ok", "note": "Offline frequentist summaries; the paper used Bayesian hierarchical models.",
           "n_sessions": len(sessions), "n_invalid": len(sessions) - len(valid), "arms": {}, "paper_reported": PAPER}
    for arm, rows in arms.items():
        accs = [r["accuracy"] for r in rows]
        out["arms"][arm] = {"n": len(rows), "accuracy_mean": mean(accs), "accuracy_t_vs_.5": _t(accs, 0.5),
                            "mean_log_brier": mean(r["log_brier"] for r in rows), "sdt": _sdt(rows)}
    a, b = arms.get("description_only"), arms.get("description_plus_evidence")
    if a and b and len(a) > 1 and len(b) > 1:
        x, y = [r["log_brier"] for r in a], [r["log_brier"] for r in b]
        se = math.sqrt(stdev(x) ** 2 / len(x) + stdev(y) ** 2 / len(y))
        sp = math.sqrt(((len(x) - 1) * stdev(x) ** 2 + (len(y) - 1) * stdev(y) ** 2) / (len(x) + len(y) - 2))
        out["condition_comparison"] = {"measure": "log Brier score, evidence minus description-only (negative = better with evidence)",
                                       "difference": mean(y) - mean(x), "welch_t": (mean(y) - mean(x)) / se if se else None,
                                       "cohens_d": (mean(y) - mean(x)) / sp if sp else None}
    else:
        out["condition_comparison"] = {"status": "not_ready", "missing": "needs at least 2 valid participants in each arm"}
    out["confidence_vs_replication_effect_size"] = {"status": "not_ready",
        "missing": "replication effect sizes (Camerer et al., 2018; Klein et al., 2018) are not printed in the paper (issue i-reff)"}
    return out


if __name__ == "__main__":
    print("evaluate(sessions) -> dict; run via the study runner")


# ---- effects.json link (appended) ---------------------------------------------------------------------------
# effect_estimates(evaluation) -> {key: result block} for the effects with compute.op = "package".
# Every block is built from what evaluate() returned above; se and sd are recovered algebraically from the
# t statistics evaluate() reports. Frequentist summaries; the paper's Bayesian estimates are not recomputed.

def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < 1e-12:
            break
    return h


def _p_two_sided_t(t, df):
    """Two-sided p of Student's t (regularized incomplete beta)."""
    if t is None or df is None or df <= 0:
        return None
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    if x <= 0:
        return 0.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b + lbeta)
    ib = front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1.0 - front * _betacf(b, a, 1 - x) / b
    return min(1.0, max(0.0, ib))


def _acc_cell(arm):
    """n, mean and sd of per-participant accuracy, sd recovered from evaluate()'s t against .50."""
    n, m, t = arm["n"], arm["accuracy_mean"], arm.get("accuracy_t_vs_.5")
    se = (m - 0.5) / t if t else None
    return {"n": n, "mean": m, "sd": se * math.sqrt(n) if se is not None else None}, se, t


EFFECT_KEYS = ("accuracy_desc", "accuracy_evid", "accuracy_evid_vs_desc", "log_brier_evid_vs_desc",
               "confidence_vs_replication_es_desc", "confidence_vs_replication_es_evid", "auc_desc", "auc_evid",
               "quality_check_accuracy", "original_bf_vs_replication_es", "expert_accuracy_ssrp",
               "expert_accuracy_ml2")


def effect_estimates(evaluation):
    def nr(reason):
        return {"status": "not_ready", "reason": reason}
    if not evaluation or evaluation.get("status") != "ok":
        return {k: nr("no valid sessions were scored") for k in EFFECT_KEYS}
    arms = evaluation.get("arms", {})
    d, e = arms.get("description_only"), arms.get("description_plus_evidence")
    out = {}
    for key, cond, arm in (("accuracy_desc", "c-desc", d), ("accuracy_evid", "c-evid", e)):
        if not arm or arm["n"] < 2 or not arm.get("accuracy_t_vs_.5"):
            out[key] = nr("needs at least 2 participants with varying accuracy in this condition")
            continue
        cell, se, t = _acc_cell(arm)
        out[key] = {"status": "computed", "metric": "proportion", "value": arm["accuracy_mean"], "se": se,
                    "n": arm["n"], "cells": {cond: cell},
                    "test": {"statistic": "t", "value": t, "df": arm["n"] - 1, "p": _p_two_sided_t(t, arm["n"] - 1),
                             "against": 0.5},
                    "method": "mean of per-participant accuracy rates ('did not understand' answers dropped); "
                              "one-sample t test against .50, two-sided (the paper: Bayesian hierarchical model)"}
    if d and e and d["n"] > 1 and e["n"] > 1 and d.get("accuracy_t_vs_.5") and e.get("accuracy_t_vs_.5"):
        cd, sd_, _ = _acc_cell(d)
        ce, se_, _ = _acc_cell(e)
        v1, v2 = se_ ** 2, sd_ ** 2
        se = math.sqrt(v1 + v2)
        t = (ce["mean"] - cd["mean"]) / se if se else None
        df = (v1 + v2) ** 2 / (v1 ** 2 / (e["n"] - 1) + v2 ** 2 / (d["n"] - 1)) if se else None
        out["accuracy_evid_vs_desc"] = {
            "status": "computed", "metric": "risk_difference", "value": ce["mean"] - cd["mean"], "se": se,
            "n": d["n"] + e["n"], "cells": {"c-evid": ce, "c-desc": cd},
            "test": {"statistic": "t", "value": t, "df": df, "p": _p_two_sided_t(t, df)},
            "method": "difference of mean per-participant accuracy rates (evidence minus description only), Welch t"}
    else:
        out["accuracy_evid_vs_desc"] = nr("needs at least 2 participants with varying accuracy in each condition")
    cmp_ = evaluation.get("condition_comparison") or {}
    if d and e and cmp_.get("difference") is not None and cmp_.get("welch_t"):
        diff, t = cmp_["difference"], cmp_["welch_t"]
        p = 2 * (1 - ND.cdf(abs(t)))
        out["log_brier_evid_vs_desc"] = {
            "status": "computed", "metric": "mean_difference", "value": diff, "se": diff / t, "n": d["n"] + e["n"],
            "cells": {"c-evid": {"n": e["n"], "mean": e["mean_log_brier"]},
                      "c-desc": {"n": d["n"], "mean": d["mean_log_brier"]}},
            "test": {"statistic": "t", "value": t, "df": None, "p": p},
            "method": "mean log Brier score per participant, evidence minus description only; Welch t, p by normal "
                      "approximation (Welch df not returned by evaluate); Cohen's d = %s (pooled sd). The paper "
                      "reports the posterior median of delta from a Bayesian t test, not this mean difference."
                      % (("%.3f" % cmp_["cohens_d"]) if cmp_.get("cohens_d") is not None else "n/a")}
    else:
        out["log_brier_evid_vs_desc"] = nr("needs at least 2 valid participants in each arm")
    reff = (evaluation.get("confidence_vs_replication_effect_size") or {}).get(
        "missing", "replication effect sizes are not in the package")
    out["confidence_vs_replication_es_desc"] = nr(reff)
    out["confidence_vs_replication_es_evid"] = nr(reff)
    for key, cond, arm in (("auc_desc", "c-desc", d), ("auc_evid", "c-evid", e)):
        sdt = (arm or {}).get("sdt")
        if not sdt:
            out[key] = nr("needs predictions for both replicated and non-replicated studies")
            continue
        out[key] = {"status": "computed", "metric": "auc", "value": sdt["auc"], "se": None, "n": arm["n"],
                    "cells": {cond: {"n": arm["n"], "hit_rate": sdt["hit_rate"],
                                     "false_alarm_rate": sdt["false_alarm_rate"], "d_prime": sdt["d_prime"],
                                     "bias_c": sdt["bias_c"]}},
                    "test": {"statistic": None, "value": None, "df": None, "p": None},
                    "method": "AUC = Phi(d'/sqrt 2), equal-variance SDT on pooled hit and false-alarm rates "
                              "(log-linear correction); no test (the paper ran none)"}
    rows = [a for a in (d, e) if a and a.get("accuracy_t_vs_.5")]
    if rows and sum(a["n"] for a in rows) > 1:
        cells = [_acc_cell(a)[0] for a in rows]
        n = sum(c["n"] for c in cells)
        m = sum(c["n"] * c["mean"] for c in cells) / n
        ss = sum((c["n"] - 1) * c["sd"] ** 2 + c["n"] * (c["mean"] - m) ** 2 for c in cells)
        sd = math.sqrt(ss / (n - 1))
        se = sd / math.sqrt(n)
        t = (m - 0.5) / se if se else None
        out["quality_check_accuracy"] = {
            "status": "computed", "metric": "proportion", "value": m, "se": se, "n": n,
            "cells": {"all": {"n": n, "mean": m, "sd": sd}},
            "test": {"statistic": "t", "value": t, "df": n - 1, "p": _p_two_sided_t(t, n - 1), "against": 0.5},
            "method": "mean per-participant accuracy over both conditions, one-sample t against .50"}
    else:
        out["quality_check_accuracy"] = nr("needs at least 2 scored participants")
    out["original_bf_vs_replication_es"] = nr("study-level analysis of the original Bayes factors and replication "
                                              "effect sizes; replication effect sizes are not in the package and "
                                              "no participant session holds them")
    for proj in ("ssrp", "ml2"):
        out["expert_accuracy_%s" % proj] = nr("expert predictions from the %s project are published data outside "
                                              "this study's sessions" % proj.upper())
    return out
