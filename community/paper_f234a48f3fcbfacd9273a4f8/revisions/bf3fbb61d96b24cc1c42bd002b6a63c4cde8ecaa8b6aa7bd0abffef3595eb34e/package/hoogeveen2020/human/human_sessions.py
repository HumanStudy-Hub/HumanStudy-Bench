"""Human sessions for Hoogeveen, Sarafoglou & Wagenmakers (2020), built from the authors' public data files."""
import csv
from statistics import NormalDist
from pathlib import Path

SOURCE = ("Data_MTurk_AN.csv, Data_Social_Media_AN.csv, Data_UvA_Lab_AN.csv (Qualtrics exports, one row per respondent) and "
          "DescriptionAndStatisticsPerStudy.csv (study numbers/IDs), the authors' OSF data release for the paper, in human-data/.")

EXCLUSIONS = ("Paper's preregistered exclusions: (a) PhD in psychology (phd == 1), (b) not understanding >50% of descriptions, "
              "(c) failing the bogus attention-check item 28 (must answer 'no' with confidence 70-80), (d) familiar with the "
              "replication projects (knowledge != 5). Only respondents who answered all 28 pages are kept (the paper's N=257 "
              "before exclusions). 'I don't understand' ticks are kept as not_understood flags (evaluate drops those answers).")

NOT_MAPPED = []

# Study number in the data (1..27) -> adapter item key; same order as DescriptionAndStatisticsPerStudy.csv / materials.json.
KEYS = ["aviezer", "balafoutas", "derex", "duncan", "gervais", "gneezy", "hauser", "bauer", "critcher", "karpicke",
        "anderson", "giessner", "risen", "shafir", "zaval", "tversky", "kidd", "shah", "zhong", "kovacs", "lee",
        "morewedge", "nishi", "pyc", "sparrow", "wilson", "alter"]
ARMS = {"1": ("c-desc", "description_only", "des"), "2": ("c-evid", "description_plus_evidence", "bf")}
FILES = ["Data_MTurk_AN.csv", "Data_Social_Media_AN.csv", "Data_UvA_Lab_AN.csv"]


def _cols(i, kind, nl):
    # Codebook (Qualtrics export column names):
    #   Conditie            -> condition: 1 = description only, 2 = description + Bayes factor
    #   {i}_des/bf.belief   -> "Will this study replicate?" 1 = yes, 2 = no (attention check requires 2 = "no")
    #   {i}_des/bf.conf_1   -> confidence slider 0-100
    #   {i}_des/bf.0        -> tick box "I don't understand the study description" (1 = ticked)
    #   ...NL variants      -> same items in the Dutch survey version (Social_Media / UvA_Lab files)
    s = "NL" if nl else ""
    return "%d_%s.belief%s" % (i, kind, s), "%d_%s.conf%s_1" % (i, kind, s), "%d_%s.0%s" % (i, kind, s)


def _order(r):
    # preorder: presentation order of study numbers (Qualtrics export, leading '|')
    try:
        o = [int(x) for x in r.get("preorder", "").split("|") if x]
        if sorted(o) == list(range(1, 29)):
            return [x for x in o if x != 28]
    except ValueError:
        pass
    return list(range(1, 28))


def load(data_dir):
    sessions = []
    for fn in FILES:
        with open(Path(data_dir) / fn, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                arm = ARMS.get(r.get("Conditie"))
                if not arm:
                    continue
                cond, name, kind = arm
                rows = None
                for nl in (False, True):
                    if nl and "1_%s.belief%s" % (kind, "NL") not in r:
                        continue
                    got = []
                    for i in range(1, 29):
                        b, c, u = _cols(i, kind, nl)
                        got.append((r[b], r[c], r[u]))
                    if all(g[0] in ("1", "2") and g[1] != "NA" for g in got):
                        rows = got
                if rows is None:
                    continue
                att = rows[27]
                if not (att[0] == "2" and 70 <= float(att[1]) <= 80):
                    continue
                if r.get("phd") == "1" or r.get("knowledge") != "5":
                    continue
                if sum(g[2] == "1" for g in rows[:27]) > 13.5:
                    continue
                order = _order(r)
                answers = []
                for page, n in enumerate(order, 1):
                    b, c, u = rows[n - 1]
                    answers.append({"key": KEYS[n - 1], "page": page, "replicate": "yes" if b == "1" else "no",
                                    "confidence": float(c), "not_understood": u == "1"})
                sessions.append({"study": "s1", "condition": cond, "arm": name, "participant": "h-" + r["ResponseId"],
                                 "parsed": {"valid": True, "answers": answers, "error": None}, "valid": True,
                                 "source": "human",
                                 # this package's evaluate() reads answers at the top level, as adapter.run_sessions emits
                                 "answers": answers, "error": None})
    return sessions


def key_effects(evaluation):
    arms = evaluation.get("arms", {})
    d, e = arms.get("description_only", {}), arms.get("description_plus_evidence", {})
    cmp_ = evaluation.get("condition_comparison", {})
    out = []

    def add(id_, label, est, p, rep):
        out.append({"id": id_, "label": label, "estimate": est, "p": p, "reported": rep})
    add("accuracy_desc", "Accuracy above chance, description only (mean accuracy - .50)",
        (d["accuracy_mean"] - 0.5) if d else None, None, 0.59 - 0.5)
    add("accuracy_evid", "Accuracy above chance, description + evidence (mean accuracy - .50)",
        (e["accuracy_mean"] - 0.5) if e else None, None, 0.67 - 0.5)
    sd, se = (d.get("sdt") or {}), (e.get("sdt") or {})
    add("auc_desc", "AUC (SDT, from d'), description only", sd.get("auc"), None, 0.62)
    add("auc_evid", "AUC (SDT, from d'), description + evidence", se.get("auc"), None, 0.74)
    # sign flipped: negative log-Brier difference = better with evidence = predicted direction
    diff = cmp_.get("difference")
    # paper's -0.96 is a Bayesian hierarchical-model median on another scale, so it is not comparable: reported=None
    wt = cmp_.get("welch_t")
    pv = 2 * (1 - NormalDist().cdf(abs(wt))) if wt is not None else None  # normal approximation, large n
    add("brier_condition", "Evidence benefit: mean log Brier per participant (description only - evidence)",
        -diff if diff is not None else None, pv, None)
    return out
