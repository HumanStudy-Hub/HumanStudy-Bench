#!/usr/bin/env python3
"""Human sessions for Burton et al. (2022), built from the authors' trial-level data (standard library only)."""
import csv
from pathlib import Path

SOURCE = ("human-data/data/study1.csv, study2.csv, study3.csv: trial-level data of Burton, Harris, Shah & Hahn (2022), "
          "100 UK Prolific participants x 51 events per study (5,100 trials each); supplied with the project, no codebook file.")

EXCLUSIONS = ("Paper p. 7: trials with E1 equal to BR are excluded (also dropped again by evaluation.evaluate), and trials whose "
              "signed update lies beyond 3 x IQR of the quartiles within their valence (negative/neutral/positive) x direction "
              "(upwards/downwards) cell are removed as outliers. load() applies this at trial level, which reproduces the "
              "paper's neutral-trial counts and means; evaluation.evaluate's own participant-level outlier rule then runs on top.")

NOT_MAPPED = [
    {"arm": "study1:e1first / study1:ebrfirst / study2:... / study3:...",
     "reason": "The data files do not record whether E1 or eBR was asked first, so the two order arms cannot be separated. "
               "Each participant is loaded once per study with arm = condition = the study id ('study1', 'study2', 'study3')."},
]

# Codebook: none supplied. Columns are used by header name; meanings follow the paper and the adapter's trial keys.
#   Participant -> session "participant" (prefixed with the study id)
#   Event       -> trial "event" (event number 1-51)
#   Valence     -> trial "valence" (1-5 valence rating, adapter parse_reply screen "valence")
#   E1          -> trial "E1"  (initial estimate, 0-100)
#   eBR         -> trial "eBR" (estimate of base rate expected by others, 0-100)
#   BR          -> trial "BR"  (presented base rate)
#   E2          -> trial "E2"  (second estimate, 0-100)
# BR_recall is not in the data and is left out of the trials.
STUDIES = ("study1", "study2", "study3")


def _num(x):
    x = (x or "").strip()
    if x == "":
        return None
    v = float(x)
    return int(v) if v.is_integer() else v


def _update(t):
    e1, e2, br = t["E1"], t["E2"], t["BR"]
    move = e2 - e1
    return abs(move) if (move > 0) == (br > e1) and move != 0 else (-abs(move) if move != 0 else 0.0)


def _q(xs, p):
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _outliers(sessions):
    """ids of trials beyond 3 x IQR in their valence x direction cell."""
    cells = {}
    for s in sessions:
        for t in s["trials"]:
            if None in (t["E1"], t["E2"], t["BR"], t["valence"]) or t["E1"] == t["BR"]:
                continue
            v = t["valence"]
            key = ("negative" if v <= 2 else "neutral" if v == 3 else "positive", t["BR"] > t["E1"])
            cells.setdefault(key, []).append((t, _update(t)))
    bad = set()
    for items in cells.values():
        xs = sorted(u for _, u in items)
        q1, q3 = _q(xs, 0.25), _q(xs, 0.75)
        for t, u in items:
            if u < q1 - 3 * (q3 - q1) or u > q3 + 3 * (q3 - q1):
                bad.add(id(t))
    return bad


def load(data_dir):
    sessions = []
    for study in STUDIES:
        path = Path(data_dir) / "data" / f"{study}.csv"
        if not path.exists():
            path = Path(data_dir) / f"{study}.csv"
        by_p, study_sessions = {}, []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                t = {"event": _num(r["Event"]), "BR": _num(r["BR"]), "E1": _num(r["E1"]), "eBR": _num(r["eBR"]),
                     "E2": _num(r["E2"]), "valence": _num(r["Valence"])}
                by_p.setdefault(r["Participant"], []).append(t)
        for pid, trials in sorted(by_p.items(), key=lambda kv: int(kv[0])):
            trials.sort(key=lambda t: t["event"])
            valid = all(t[k] is not None for t in trials for k in ("E1", "E2", "BR", "valence"))
            study_sessions.append({"study": study, "condition": study, "arm": study, "participant": f"{study}-{pid}",
                             "parsed": {"trials": trials}, "trials": trials, "valid": valid, "source": "human"})
        bad = _outliers(study_sessions)
        for ss in study_sessions:
            ss["trials"] = [t for t in ss["trials"] if id(t) not in bad]
            ss["parsed"] = {"trials": ss["trials"]}
        sessions.extend(study_sessions)
    return sessions


def key_effects(evaluation):
    """Neutral-event asymmetry per study: mean update downwards minus upwards (positive = larger update when BR is
    below the estimate, the direction the paper finds). Model estimate (sign flipped) when the LMM fit, else mean difference."""
    out = []
    for study in STUDIES:
        e = (evaluation.get("studies") or {}).get(study) or {}
        rep = (e.get("reported") or {}).get("estimate")
        lmm = e.get("lmm") or {}
        nu = (e.get("neutral") or {})
        est, p = None, None
        if lmm.get("status") == "ok":
            est, p = -lmm["estimate_upwards_vs_downwards"], lmm["p"]
        elif nu.get("upwards", {}).get("mean") is not None and nu.get("downwards", {}).get("mean") is not None:
            est = nu["downwards"]["mean"] - nu["upwards"]["mean"]
        out.append({"id": f"asymmetry_{study}", "label": f"{study}: neutral-event update, downwards minus upwards",
                    "estimate": est if est is not None else float("nan"), "p": p, "reported": rep})
    return out
