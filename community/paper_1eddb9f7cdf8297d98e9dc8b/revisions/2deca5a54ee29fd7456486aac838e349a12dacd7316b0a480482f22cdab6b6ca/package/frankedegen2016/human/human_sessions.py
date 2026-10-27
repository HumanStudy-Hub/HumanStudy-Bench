"""Human data for Franke & Degen (2016) -> sessions for the package's evaluation (standard library only)."""
import csv
import json
import math
from pathlib import Path

SOURCE = ("human-data/data/exp_s007.csv and exp_s008.csv (tab-separated; columns workerid, language, trial, rt, "
          "condition, response), supplied with the project. The two files are byte-identical (52 workers x 66 trials); "
          "their choice proportions match the paper's Experiment 2 (production), so one copy is used for exp2.")

EXCLUSIONS = ("Load keeps every worker whose language field is English (case-insensitive; all 52 are). The paper's "
              "exclusion of the 15% of participants with the highest distractor-error rate on non-ambiguous trials is "
              "applied inside evaluation.evaluate, not here. Distractor trials are kept as role 'distractor' (evaluate "
              "drops them from the regression). Experiment 1's four speaker warm-up trials are not in the data.")

NOT_MAPPED = [
    {"arm": "exp1", "reason": "No separate Experiment 1 (comprehension) file: exp_s007.csv and exp_s008.csv are identical "
                              "and match Experiment 2's production proportions; the data has no objects/messages/choice index either."},
]

# Codebook: none supplied; column meanings taken from the headers and values.
# workerid -> "participant"; trial (0-based position, 0..65) -> trial = trial + 1;
# condition (unambiguous|ambiguous|simple|complex) -> analysis_cond (unamb|amb|simple|complex);
# response (target|competitor|distractor) -> role; language -> English filter. rt is unused.
_COND = {"unambiguous": "unamb", "ambiguous": "amb", "simple": "simple", "complex": "complex"}


def load(data_dir):
    rows = list(csv.DictReader(open(Path(data_dir) / "data" / "exp_s007.csv", encoding="utf-8"), delimiter="\t"))
    by = {}
    for r in rows:
        if r["language"].strip().lower() != "english":
            continue
        by.setdefault(int(r["workerid"]), []).append(r)
    sessions = []
    for wid in sorted(by):
        trials = []
        for r in sorted(by[wid], key=lambda r: int(r["trial"])):
            c = _COND[r["condition"]]
            trials.append({"trial": int(r["trial"]) + 1, "phase": "main", "cond": c, "analysis_cond": c,
                           "choice": None, "role": r["response"], "reply": r["response"]})
        roles = [t["role"] for t in trials]
        sessions.append({"study": "exp2", "arm": "exp2", "condition": "within-participant: all trial types",
                         "participant": wid, "trials": trials,
                         "parsed": {"choice": None, "raw": ",".join(roles)},
                         "valid": all(r in ("target", "competitor", "distractor") for r in roles),
                         "source": "human"})
    return sessions


def _p(beta, se):
    return math.erfc(abs(beta / se) / math.sqrt(2)) if se else None


def key_effects(evaluation):
    out = []
    for study, label in (("exp1", "Exp 1"), ("exp2", "Exp 2")):
        st = evaluation.get("studies", {}).get(study, {})
        rep = st.get("paper_reported") or {}
        co = (st.get("regression") or {}).get("coefficients") or {}
        for key, name, rk in (("CONDITION (HARDER VS. SIMPLE)", "simple vs harder (complex+ambiguous): target-choice log-odds, higher for simple", "beta_simple_vs_complex"),
                              ("CONDITION (AMBIGUOUS VS. COMPLEX)", "complex vs ambiguous: target-choice log-odds, higher for complex", "beta_complex_vs_amb")):
            c = co.get(key)
            est = c["beta"] if c else float("nan")
            out.append({"id": "%s_%s" % (study, "simple_vs_harder" if "HARDER" in key else "complex_vs_amb"),
                        "label": "%s %s" % (label, name), "estimate": est,
                        "p": _p(c["beta"], c["se"]) if c else None, "reported": rep.get(rk)})
    return out
