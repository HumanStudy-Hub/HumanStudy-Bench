#!/usr/bin/env python3
"""Human sessions for the Thomas et al. (2014) replay package, built from the Exeter/Prolific replication data."""
import csv
import os

SOURCE = ("human-data/xvrb_z591_data.csv (Qualtrics export, Prolific sample, 11-13 Aug 2020) with its codebook "
          "human-data/xvrb_z591_dictionary.pdf; the replication of the Experiment 1 two-player butcher-baker game "
          "(survey wording: sources/xvrb_z591_survey.pdf). One payoff level only (GBP 1.60 alone / 1.80 together), "
          "so it is mapped to the package's exp1 $2 arms as the nearest payoff; this is an approximation.")

EXCLUSIONS = ("Dropped: rows with no Cooperate decision or no Knowledge/Role (incomplete, 22 rows) and Qualtrics preview "
              "rows (DistributionChannel='preview'). Comprehension exclusion as in the paper (and the codebook's INCLUDE "
              "variable, which is not in the csv, so recomputed): sessions are kept in the data but flagged "
              "payoff_correct=False and valid=False unless all three payoff checks are right "
              "(CC1=1.80, CC2=0, CC3=1.60). evaluate() applies the exclusion via payoff_correct.")

NOT_MAPPED = [
    "exp1 arms with payoff $1, $5, $10: the data contain one payoff level only (GBP 1.60 alone)",
    "exp2 (four-player game): not in the data",
    "exp3 (similarity manipulation, Big Five, reported level): not in the data (no similarity or Big Five columns)",
]

# Codebook entries used (xvrb_z591_dictionary.pdf):
#   Cooperate   0=No, 1=Yes (work together)            -> parsed["decision"]
#   Knowledge   1=Level 1 .. 3=Level 3, 4=Common        -> condition knowledge (primary..common)
#   Role        Butcher / Baker                          -> session["role"]
#   CC1/CC2/CC3 (_bt for butchers, _bk for bakers)       -> payoff_answers; 1=0, 2=1.40, 3=1.60, 4=1.80
#   MC1..MC4    (_bt / _bk)  1=Yes 2=No 3=Not possible   -> knowledge_answers
#   DistributionChannel (Qualtrics, preview rows)        -> exclusion
#   PROLIFIC_PID                                         -> participant id (falls back to ResponseId)
_LEVEL = {"1": "primary", "2": "secondary", "3": "tertiary", "4": "common"}
_CC = {"1": 0.0, "2": 1.40, "3": 1.60, "4": 1.80}
_MC = {"1": "yes", "2": "no", "3": "cant_tell"}
_PAYOFF = 2  # nearest package payoff to GBP 1.60


def load(data_dir):
    with open(os.path.join(data_dir, "xvrb_z591_data.csv"), encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    sessions = []
    for r in rows[2:]:  # row 1 is the question text
        d = dict(zip(header, r))
        if d.get("Cooperate", "") not in ("0", "1") or d.get("Knowledge", "") not in _LEVEL:
            continue
        if d.get("DistributionChannel", "").lower() == "preview":
            continue
        role = d["Role"].strip().lower()
        suf = "_bt" if role == "butcher" else "_bk"
        cc = [_CC.get(d.get(f"CC{i}{suf}", "")) for i in (1, 2, 3)]
        mc = [_MC.get(d.get(f"MC{i}{suf}", "")) for i in (1, 2, 3, 4)]
        k = _LEVEL[d["Knowledge"]]
        arm = f"exp1/{k}/{_PAYOFF}"
        parsed = {"decision": "together" if d["Cooperate"] == "1" else "alone", "payoff_answers": cc,
                  "payoff_correct": cc == [1.80, 0.0, 1.60], "knowledge_answers": mc}
        s = {"study": "exp1", "condition": {"arm": arm, "study": "exp1", "knowledge": k, "payoff": _PAYOFF, "similarity": None},
             "arm": arm, "participant": d.get("PROLIFIC_PID") or d["ResponseId"], "role": role,
             "parsed": parsed, "valid": parsed["payoff_correct"], "source": "human"}
        s.update(parsed)  # evaluate() reads these at top level, as run_sessions emits them
        sessions.append(s)
    return sessions


def _signed(test, sign):
    chi2 = test.get("chi2")
    return None if chi2 is None else sign * chi2


def key_effects(evaluation):
    block = (evaluation.get("with_comprehension_exclusion") or {}).get("exp1") or {}
    cell = (block.get("by_payoff") or {}).get(f"${_PAYOFF}") or {}
    prop = cell.get("proportions") or {}
    paper = evaluation.get("paper_reported", {}).get("exp1_table1_chi2", {})

    def rep(key):
        try:
            return float(paper[key][1].rstrip("*"))  # the $2 column
        except (KeyError, IndexError, ValueError):
            return None

    def diff(a, b):
        return 1 if prop.get(a) is not None and prop.get(b) is not None and prop[a] >= prop[b] else -1

    specs = [("exp1_omnibus", "Knowledge level omnibus, $2 (signed chi2, df 3)", "all_levels", "all", 1),
             ("exp1_secondary_vs_primary", "Secondary > primary knowledge, $2 (chi2)", "1o_vs_2o", "1o_vs_2o", diff("secondary", "primary")),
             ("exp1_tertiary_vs_secondary", "Tertiary vs secondary (predicted no difference), $2 (chi2)", "2o_vs_3o", "2o_vs_3o", diff("tertiary", "secondary")),
             ("exp1_common_vs_tertiary", "Common > tertiary knowledge, $2 (chi2)", "3o_vs_ck", "3o_vs_ck", diff("common", "tertiary"))]
    out = []
    for id_, label, ek, pk, sign in specs:
        t = cell.get(ek) or {}
        out.append({"id": id_, "label": label, "estimate": _signed(t, sign), "p": t.get("p"), "reported": rep(pk)})
    return out
