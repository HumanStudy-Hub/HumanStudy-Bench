"""Human sessions for Nyhan & Reifler (2015), built from the authors' replication data (one row = one respondent)."""
import csv
import math
from pathlib import Path

SOURCE = ("human-data/causal-replication.csv: the authors' replication data for the CCAP 2012 YouGov panel "
          "(Nyhan_JournExpPoliSci_2015_DEqr; for-cos.do, from Brendan Nyhan, received 7/13/19, also on his "
          "website as nyhan-reifler-jeps.zip). One row per respondent, 1000 rows.")
EXCLUSIONS = ("None applied. The paper states no exclusions beyond the survey sample itself; all 1000 respondents "
              "are kept, each in the one arm they were assigned to. Unweighted, like the package evaluation "
              "(YouGov weight 'weight' and block IPW 'aw' are not used). A respondent with a missing outcome keeps "
              "None for it and valid=False.")
NOT_MAPPED = []  # all four arms are in the data

# Codebook entries (no codebook file is in human-data; README.pdf only lists the files; meanings come from
# for-cos.do variable names and the paper's measures, and were checked against the paper's reported means):
#   ccap_id      respondent id                                     -> participant
#   swensenrand  assigned condition 1=control 2=innuendo 3=denial 4=causal (matches swensencond1-4 dummies)
#   swensenfav   opinion of Swensen, 1 very unfavorable .. 6 very favorable -> favorable (same codes as materials)
#   resigninvest resigned due to investigation, 1 not likely at all .. 4 very likely; blank for control
#                (not asked)                                       -> investigation
#   acceptedbribes accepted bribes / acted illegally, 1 not at all .. 5 extremely likely -> bribes
#   crooked      pre-treatment 'crooked officials' item (not used in the analysis)
ARMS = {1: "control", 2: "innuendo", 3: "denial", 4: "causal"}
OUTCOMES = {"favorable": "swensenfav", "investigation": "resigninvest", "bribes": "acceptedbribes"}
ASKED = {"control": ["favorable", "bribes"], "innuendo": ["favorable", "investigation", "bribes"],
         "denial": ["favorable", "investigation", "bribes"], "causal": ["favorable", "investigation", "bribes"]}


def _num(x):
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return None


def load(data_dir):
    path = Path(data_dir) / "causal-replication.csv"
    sessions = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            arm = ARMS.get(_num(row["swensenrand"]))
            if arm is None:
                continue
            parsed = {v: _num(row[OUTCOMES[v]]) for v in ASKED[arm]}
            sessions.append({"study": "S1", "condition": arm, "arm": arm, "participant": row["ccap_id"],
                             "parsed": parsed, "responses": parsed,  # evaluate() reads "responses"; same dict
                             "valid": all(v is not None for v in parsed.values()),
                             "source": "human"})
    return sessions


# (id, label, outcome, contrast, sign): sign +1 when the predicted effect raises the outcome, -1 when it lowers it.
_TESTS = [
    ("innuendo_favorable", "Innuendo vs control: lower favorability of Swensen", "favorable", "innuendo", "reference", -1),
    ("innuendo_bribes", "Innuendo vs control: higher belief he took bribes", "bribes", "innuendo", "reference", 1),
    ("denial_favorable", "Denial vs innuendo: repairs favorability", "favorable", "denial", "innuendo", 1),
    ("causal_favorable", "Causal correction vs innuendo: repairs favorability", "favorable", "causal", "innuendo", 1),
    ("denial_bribes", "Denial vs innuendo: lowers belief in bribes", "bribes", "denial", "innuendo", -1),
    ("causal_bribes", "Causal correction vs innuendo: lowers belief in bribes", "bribes", "causal", "innuendo", -1),
    ("denial_investigation", "Denial vs innuendo: lowers belief he resigned due to investigation", "investigation", "denial", "innuendo", -1),
    ("causal_investigation", "Causal correction vs innuendo: lowers belief he resigned due to investigation", "investigation", "causal", "innuendo", -1),
]


def key_effects(evaluation):
    models = (evaluation or {}).get("models", {})
    out = []
    for tid, label, dv, arm, ref, sign in _TESTS:
        m = models.get(dv)
        est = p = rep = None
        if m:
            if ref == "reference":
                ref = m["reference"]
            c = m["coef_vs_reference"].get(arm) if ref == m["reference"] else m["pairwise"].get("%s_vs_%s" % (arm, ref))
            if c:
                est, p = sign * c["difference"], c["p_normal_approx"]
            r = m["reported_in_paper"]["coef_vs_reference"].get(arm) if ref == m["reference"] else None
            if r is not None:
                rep = sign * r
        out.append({"id": tid, "label": label, "estimate": float("nan") if est is None else float(est),
                    "p": p, "reported": rep})
    return out
