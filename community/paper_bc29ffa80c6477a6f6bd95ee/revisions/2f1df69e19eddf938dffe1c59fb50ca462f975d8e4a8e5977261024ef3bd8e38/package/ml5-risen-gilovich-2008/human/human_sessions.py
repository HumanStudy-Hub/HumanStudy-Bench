"""Human sessions for ML5 Risen & Gilovich (2008) "tempting fate" replication, built from the ML5 data."""
import csv
import math
from pathlib import Path

SOURCE = ("human-data/full_prepped_dataset.csv: the Many Labs 5 (Ebersole et al., 2020) prepared dataset for the "
          "Risen & Gilovich tempting-fate replication; one row per participant, all sites.")

EXCLUSIONS = ("The file is already the prepared dataset: its 'excluded' column is 0 for every row, i.e. the a priori "
              "exclusions were applied upstream (n.excl gives the per-site count removed). Rows with a missing "
              "likelihood rating are marked invalid. Only the 'b.similar' group (U Penn, UCB, UVA, Stanford) is "
              "loaded, the 'similar U.S. university sites' population the package materials target; the MTurk "
              "('a.mturk') and 'c.dissimilar' groups are left out so the pooled estimate matches one reported analysis.")

NOT_MAPPED = []  # all four arms (tempting fate x load) are built; no codebook file was supplied, columns are mapped by name below
GROUPS = ("b.similar",)

# Column -> meaning (no codebook shipped; inferred from names and value ranges):
#   id          participant id                              -> participant
#   group       site cluster (a.mturk / b.similar / c.dissimilar) -> filter
#   tempt       1 = tempted-fate scenario, 0 = not          -> tempting_fate
#   load        1 = counting-backward cognitive load        -> cognitive_load
#   lkl         likelihood of being called on, 0-10         -> parsed["likelihood"]
#   count.eff   effort of counting task, 0-10 (load only)   -> parsed["effort"]
#   count.hard  difficulty of counting task, 0-10           -> parsed["difficulty"]
#   importance  importance of answering correctly, 0-10     -> parsed["importance"]
#   badness     how bad if called on and unable to answer   -> parsed["negativity"]
# Unused: eff.split, had.read (= 1 - tempt), end.num, n.excl, is.mturk, SAT, excluded, site.


def _num(v):
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return x if 0 <= x <= 10 else None


def load(data_dir):
    sessions = []
    with open(Path(data_dir) / "full_prepped_dataset.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["group"] not in GROUPS or r.get("excluded") == "1":
                continue
            tempt, load_ = int(r["tempt"]), int(r["load"])
            arm = "c%d%d" % (tempt, load_)
            parsed = {"likelihood": _num(r["lkl"])}
            if load_:
                parsed["effort"] = _num(r["count.eff"])
                parsed["difficulty"] = _num(r["count.hard"])
            parsed["importance"] = _num(r["importance"])
            parsed["negativity"] = _num(r["badness"])
            sessions.append({"study": "s1", "condition": arm, "arm": arm, "participant": "%s-%s" % (r["site"], r["id"]),
                             "parsed": parsed, "valid": parsed["likelihood"] is not None, "source": "human",
                             # fields the package evaluation reads
                             "tempting_fate": tempt, "cognitive_load": load_, "responses": parsed})
    return sessions


def _p(est, se, df):
    if not se or se != se:
        return None
    t = abs(est / se)
    try:
        from scipy import stats
        return float(2 * stats.t.sf(t, df))
    except Exception:
        return math.erfc(t / math.sqrt(2))


def key_effects(evaluation):
    if evaluation.get("status") != "ok":
        return []
    df = sum(evaluation["n"].values()) - 4
    rep = evaluation.get("reported", {}).get("similar_sites", {})
    out = []
    for id_, label, key, rk in [
            ("tempting_fate_main", "Tempting fate raises likelihood (no-load condition, 0-10 points)", "main_effect", "main_effect"),
            ("tempting_fate_x_load", "Tempting fate x cognitive load interaction (0-10 points)", "interaction", "interaction")]:
        e = evaluation[key]
        out.append({"id": id_, "label": label, "estimate": float(e["estimate"]), "p": _p(e["estimate"], e["se"], df),
                    "reported": rep.get(rk)})
    return out
