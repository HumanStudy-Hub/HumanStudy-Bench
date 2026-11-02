"""Human sessions for Pennycook et al. (2020), Studies 1 and 2, built from the authors' released data.

Standard library only. The data files have no separate codebook; field meanings come from the Qualtrics
export in human-data/study2_materials.docx (Study 2 survey flow) and from the column structure of the files.
"""
import csv
import os

SOURCE = ("human-data/study1.csv (853 participants) and human-data/study2.csv (856 participants), the authors' "
          "public data files for Pennycook, McPhetres, Zhang, Lu & Rand (2020), 'Fighting COVID-19 "
          "misinformation on social media', Psychological Science; human-data/study2_materials.docx is the "
          "Study 2 Qualtrics survey export used as codebook.")

EXCLUSIONS = ("None applied here. The files already hold the paper's analysed samples (853 in Study 1, 856 in "
              "Study 2, i.e. after the paper's removal of non-completers/non-passers of the screeners); every row "
              "with a complete set of 30 headline answers is kept, including the few rows with Finished=0 or "
              "English!=1, because the paper's N matches the row count.")

NOT_MAPPED = []

# Codebook mapping (column -> meaning)
#  study1.csv  Acc / Sharing: 1 when the participant was in the accuracy / sharing condition, 'NA' otherwise
#              (Qualtrics randomised-condition flags; also the first column 'Question' = 1 accuracy, 2 sharing).
#  study1.csv  Fake1_1..Fake1_15, Real1_1..Real1_15 (+ suffixes .0, .1, .2): yes(1)/no(0) answer to
#              "is the claim accurate?" or "would you consider sharing?" for the 15 false / 15 true headlines.
#              The four suffix blocks are the four Qualtrics blocks (condition x yes/no option order); exactly one
#              block is non-NA per participant.
#  study2.csv  Condition (first column): 1 = control, 2 = treatment (treatment rows have a pretest answer).
#  study2.csv  NR_1_2..NR_4_2: 0/1 (No/Yes) answer to the pretest accuracy question for neutral headline 1..4
#              (study2_materials.docx, blocks "neu r1".."neu r4"); exactly one is non-NA per treatment participant.
#  study2.csv  Fake1_1..15, Real1_1..15: 1-6 sharing likelihood for the false / true headlines.
# Headline k of Fake1_k / Real1_k is taken to be False-k / True-k in materials.json.

_SUFFIXES = ["", ".0", ".1", ".2"]


def _read(path):
    with open(path, newline="", encoding="latin-1") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    header[0] = "Condition"  # first column name is mangled by a BOM
    return [dict(zip(header, r)) for r in rows[1:]]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _items(row, suffix):
    out = {}
    for k in range(1, 16):
        for pref, hid in (("Fake1_", "False-%d" % k), ("Real1_", "True-%d" % k)):
            v = _num(row.get("%s%d%s" % (pref, k, suffix)))
            if v is None:
                return None
            out[hid] = v
    return out


def _study1(data_dir):
    sessions = []
    for i, row in enumerate(_read(os.path.join(data_dir, "study1.csv"))):
        arm_cond = "accuracy" if row.get("Acc") == "1" else "sharing" if row.get("Sharing") == "1" else None
        resp = None
        for suf in _SUFFIXES:
            resp = _items(row, suf)
            if resp is not None:
                break
        ok = arm_cond is not None and resp is not None and all(v in (0.0, 1.0) for v in resp.values())
        if arm_cond is None:
            continue
        parsed = {"valid": ok, "responses": {h: int(v) for h, v in resp.items()} if ok else {},
                  "pretest": None, "errors": [] if ok else ["incomplete answers"]}
        sessions.append({"study": "study1", "condition": arm_cond, "arm": "study1/" + arm_cond,
                         "participant": "study1/%s#%d" % (arm_cond, i), "parsed": parsed, "valid": ok, "source": "human"})
    return sessions


def _study2(data_dir):
    sessions = []
    for i, row in enumerate(_read(os.path.join(data_dir, "study2.csv"))):
        cond = {"1": "control", "2": "treatment"}.get(row["Condition"])
        if cond is None:
            continue
        resp = _items(row, "")
        ok = resp is not None and all(1 <= v <= 6 and v == int(v) for v in resp.values())
        pretest = None
        if cond == "treatment":
            vals = [_num(row.get("NR_%d_2" % k)) for k in range(1, 5)]
            vals = [v for v in vals if v is not None]
            if len(vals) == 1 and vals[0] in (0.0, 1.0):
                pretest = int(vals[0])
            else:
                ok = False
        parsed = {"valid": ok, "responses": {h: int(v) for h, v in resp.items()} if ok else {},
                  "pretest": pretest, "errors": [] if ok else ["incomplete answers"]}
        sessions.append({"study": "study2", "condition": cond, "arm": "study2/" + cond,
                         "participant": "study2/%s#%d" % (cond, i), "parsed": parsed, "valid": ok, "source": "human"})
    return sessions


def load(data_dir):
    sessions = _study1(data_dir) + _study2(data_dir)
    # evaluation.py reads the parse_reply keys at top level of a session, as run_sessions returns them
    for s in sessions:
        s.update(s["parsed"])
        s["valid"] = s["parsed"]["valid"]
    return sessions


def _get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def key_effects(evaluation):
    ev = evaluation or {}
    paper = ev.get("paper_reported", {})
    out = []

    def add(id_, label, est, p, rep):
        out.append({"id": id_, "label": label, "estimate": float(est) if est is not None else float("nan"),
                    "p": p, "reported": rep})

    t = _get(ev, "study1", "veracity_x_condition") or {}
    b = t.get("beta")
    add("study1_veracity_x_condition",
        "Study 1: sharing reduces sensitivity to veracity (-beta of veracity x condition, z-scored yes/no)",
        None if b is None else -b, t.get("p"), None)
    # reported is None: the paper's -0.126 is on a different coding scale than evaluation.py's +-0.5 codes
    # (the human difference in discernment, .32 vs .06 raw, gives 0.52 here and 0.52/4 = 0.13 under +-1 codes).
    t = _get(ev, "study2", "veracity_x_treatment") or {}
    add("study2_veracity_x_treatment",
        "Study 2: accuracy prompt increases sharing discernment (beta of veracity x treatment, 0-1 scale)",
        t.get("beta"), t.get("p"), None if not paper else paper["study2"]["veracity_x_treatment"]["beta"])
    r = _get(ev, "study2", "discernment_ratio_treatment_over_control")
    add("study2_discernment_ratio", "Study 2: treatment/control sharing discernment ratio", r, None,
        None if not paper else paper["study2"]["discernment_ratio"])
    t = _get(ev, "study3_items") or {}
    add("items_accuracy_vs_treatment_effect",
        "Item level: Study 1 accuracy rating vs Study 2 treatment effect (r)", t.get("r"), t.get("p"),
        None if not paper else paper["study3"]["r"])
    return out
