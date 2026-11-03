"""Human data for Tessler & Franke (2018): builds sessions in the replay package's shape. Standard library only."""
import csv
import os

SOURCE = ("Tessler & Franke (2018) public data, human-data/data/: expt1-trials.csv, expt1-subject_information.csv, "
          "expt2-trials.csv, expt2-subject_information.csv (MTurk participants; ratings on a 0-1 slider, rescaled to 0-100).")

EXCLUSIONS = ("Participants whose self-reported native language (subject_information 'language') is not an English "
              "variant are dropped (paper: Exp1 6 of 120, Exp2 35 of 750). Blank / 'n/a' answers and mixed answers "
              "('spanish & english') count as not English; typos of 'english' are kept. This gives 7 (Exp1) and 37 (Exp2) "
              "exclusions, not exactly the paper's counts, because the paper does not list its coding.")

NOT_MAPPED = []  # every adapter arm has human data
# Caveat: explicit-condition participants rated 12 four-slider displays; the AI adapter shows 3. All 12 are kept.

# Codebook-free dataset: column names below are those of the csv headers.
#  workerid -> participant id; language (subject_information) -> exclusion
#  Exp1 trials: response -> rating (0-1); negation (morphological/lexical) -> antonym_type;
#               sentence_type (positive/antonym/neg_positive/neg_antonym) -> adjective_type; positive -> positive adjective; adjective -> statement word
#  Exp2 trials: response -> rating; antonym_type (morphant/lexant) -> antonym_type; adjective_type -> adjective_type;
#               condition (implicit = one slider per trial; explicit = four sliders on one screen) -> arm context; trial_type; trial_num -> display
ATYPE = {"positive": "positive", "antonym": "antonym", "morphant": "antonym", "lexant": "antonym",
         "neg_positive": "negated_positive", "neg_antonym": "negated_antonym",
         "neg_morphant": "negated_antonym", "neg_lexant": "negated_antonym"}
NTYPE = {"morphological": "morphological", "morphant": "morphological", "lexical": "lexical", "lexant": "lexical"}
ENGLISH = {"english", "eng", "en", "engish", "englise", "emglish", "englih", "engilish", "engl iush", "englisg",
           "eeglish", "egnlish", "american english", "(american) english", "english (us)"}


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(x):
    try:
        return min(100.0, max(0.0, float(x) * 100.0))
    except (TypeError, ValueError):
        return None


def _keep(subjects):
    return {s["workerid"] for s in subjects if s["language"].strip().lower() in ENGLISH}


def _session(study, arm, condition, context, pid, rows):
    trials = [{"adjective_type": ATYPE[r["_atype"]], "antonym_type": NTYPE[r["_ntype"]], "positive": r["positive"],
               "statement": "%s is %s." % (r["name"], r["adjective"]), "rating": _num(r["response"])} for r in rows]
    ratings = [t["rating"] for t in trials]
    parsed = {"ratings": ratings} if context == "multiple" else {"rating": ratings[0] if ratings else None}
    return {"study": study, "condition": condition, "arm": arm, "context": context, "participant": pid,
            "parsed": parsed, "valid": bool(ratings) and all(v is not None for v in ratings),
            "trials": trials, "source": "human"}


def load(data_dir):
    d = os.path.join(data_dir, "data") if os.path.isdir(os.path.join(data_dir, "data")) else data_dir
    out = []
    keep = _keep(_read(os.path.join(d, "expt1-subject_information.csv")))
    by = {}
    for r in _read(os.path.join(d, "expt1-trials.csv")):
        if r["workerid"] in keep:
            r["_atype"], r["_ntype"] = r["sentence_type"], r["negation"]
            by.setdefault(r["workerid"], []).append(r)
    for w, rows in by.items():
        rows.sort(key=lambda r: int(r["trial_num"]))
        out.append(_session("exp1", "exp1", None, "single", "exp1-" + w, rows))
    subj = _read(os.path.join(d, "expt2-subject_information.csv"))
    keep = _keep(subj)
    by = {}
    for r in _read(os.path.join(d, "expt2-trials.csv")):
        if r["workerid"] in keep:
            r["_atype"], r["_ntype"] = r["adjective_type"], r["antonym_type"]
            by.setdefault((r["workerid"], r["condition"]), []).append(r)
    arms = {("implicit", "morphant"): ("exp2-morphological-single", "c2-morph-single", "single"),
            ("implicit", "lexant"): ("exp2-lexical-single", "c2-lex-single", "single"),
            ("explicit", "morphant"): ("exp2-morphological-multiple", "c2-morph-multi", "multiple"),
            ("explicit", "lexant"): ("exp2-lexical-multiple", "c2-lex-multi", "multiple")}
    for (w, cond), rows in by.items():
        rows.sort(key=lambda r: (int(r["trial_num"]), int(r["slider_position"]) if r["slider_position"].isdigit() else 0))
        arm, c, ctx = arms[(cond, rows[0]["antonym_type"])]
        out.append(_session("exp2", arm, c, ctx, "exp2-%s-%s" % (cond, w), rows))
    return out


# (id, label, evaluation test, field, sign, reported). Estimates are raw differences in proportion units; the paper reports
# lme4 b on a contrast-coded predictor (about half the raw difference), so "reported" is None, not a mismatched number.
_SPEC = [
    ("exp1_interaction", "Exp1: antonym-vs-negated-positive contrast, morphological minus lexical (lexical antonyms lower)", "exp1_interaction", "mean", 1, None),
    ("exp2_interaction_single", "Exp2 single: morphological minus lexical contrast", "exp2_interaction_single", "diff", 1, None),
    ("exp2_morphological_vs_negated_positive", "Exp2 single, morphological: negated positive minus antonym (paper: ~0)", "exp2_morphological_vs_negated_positive", "mean", -1, None),
    ("exp2_lexical_vs_negated_positive", "Exp2 single, lexical: negated positive minus antonym (lexical more negative)", "exp2_lexical_vs_negated_positive", "mean", -1, None),
    ("exp2_context_morphological", "Exp2 morphological: multiple-context makes antonym lower than single (single minus multiple contrast)", "exp2_context_morphological", "diff", -1, None),
    ("exp2_three_way", "Exp2 three-way: context effect larger for morphological than lexical", "exp2_three_way", "diff", -1, None),
]


def key_effects(evaluation):
    tests = (evaluation or {}).get("tests", {})
    out = []
    for id_, label, test, field, sign, rep in _SPEC:
        t = tests.get(test)
        if t is None:
            out.append({"id": id_, "label": label, "estimate": None, "p": None, "reported": rep})
            continue
        out.append({"id": id_, "label": label, "estimate": sign * t[field] / 100.0, "p": t.get("p"), "reported": rep})
    return out
