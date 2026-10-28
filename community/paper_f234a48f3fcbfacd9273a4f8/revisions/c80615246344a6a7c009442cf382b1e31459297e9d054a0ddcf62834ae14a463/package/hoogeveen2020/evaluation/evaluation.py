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
