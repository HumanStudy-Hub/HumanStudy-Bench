#!/usr/bin/env python3
"""
Evaluation for the Ross, Greene, & House (1977) "False Consensus Effect" package.

Consumes a JSON-lines file of participant response records produced by
task/adapter.py (one record per line, one simulated participant each) and
tests the two hypotheses common to Study 1, Study 3, and Study 4:

  H1 (perceived consensus): participants who personally choose option X
     estimate a higher percentage of peers choosing X than participants who
     chose the alternative.
  H2 (trait-rating extremity): participants give less extreme trait ratings
     to a "typical" actor who shares their own choice than to one who does not.

Study 2 is evaluated separately (per-item direction/count check, matching
the paper's "32 of 34 items in predicted direction" summary statistic).

No network access is used or required. Where too few records are supplied to
run a given check, that check reports status "not_ready" with the specific
missing requirement rather than fabricating a result.
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normal_two_tailed_p(z):
    """Two-tailed p-value from the standard normal CDF (erf-based)."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))


def welch_t_test(sample1, sample2):
    """Welch's t-test for unequal variances. Returns dict with t, df, p (normal
    approximation to the t-distribution -- adequate for the n>~20 per-group
    sizes this package targets; not an exact Student-t CDF, to avoid a SciPy
    dependency in an offline-only adapter)."""
    n1, n2 = len(sample1), len(sample2)
    if n1 < 2 or n2 < 2:
        return None
    m1, m2 = statistics.mean(sample1), statistics.mean(sample2)
    v1, v2 = statistics.variance(sample1), statistics.variance(sample2)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return {"t": 0.0, "df": n1 + n2 - 2, "p_two_tailed_normal_approx": 1.0, "mean1": m1, "mean2": m2}
    t = (m1 - m2) / se
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = df_num / df_den if df_den > 0 else n1 + n2 - 2
    p = normal_two_tailed_p(t)
    return {"t": t, "df": df, "p_two_tailed_normal_approx": p, "mean1": m1, "mean2": m2}


def one_sample_t_test(diffs):
    n = len(diffs)
    if n < 2:
        return None
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return {"t": float("inf") if m != 0 else 0.0, "df": n - 1, "p_two_tailed_normal_approx": 0.0 if m != 0 else 1.0, "mean_diff": m, "n": n}
    se = sd / math.sqrt(n)
    t = m / se
    p = normal_two_tailed_p(t)
    return {"t": t, "df": n - 1, "p_two_tailed_normal_approx": p, "mean_diff": m, "n": n}


def evaluate_consensus_hypothesis(records, study_id, condition_id):
    """H1 for study1/study3/study4: own-choice group's estimate of their own
    option should exceed the other group's estimate of that same option."""
    subset = [r for r in records if r.get("study_id") == study_id and r.get("condition_id") == condition_id]
    if not subset:
        return {"status": "not_ready", "missing_requirement": f"no records for {study_id}/{condition_id}"}

    options = sorted({opt for r in subset for opt in r.get("consensus_estimate", {})})
    if len(options) != 2:
        return {"status": "not_ready", "missing_requirement": "records do not agree on a two-option consensus_estimate schema"}
    optA, optB = options

    group_est_for_A = defaultdict(list)  # own_choice -> list of estimate(%A)
    for r in subset:
        est = r.get("consensus_estimate", {})
        if optA not in est:
            continue
        choice = r.get("own_choice")
        if choice not in (optA, optB):
            continue
        group_est_for_A[choice].append(float(est[optA]))

    if len(group_est_for_A.get(optA, [])) < 2 or len(group_est_for_A.get(optB, [])) < 2:
        return {
            "status": "not_ready",
            "missing_requirement": (
                f"need >=2 participants choosing '{optA}' and >=2 choosing '{optB}' in {study_id}/{condition_id}; "
                f"have {len(group_est_for_A.get(optA, []))} and {len(group_est_for_A.get(optB, []))}"
            ),
        }

    test = welch_t_test(group_est_for_A[optA], group_est_for_A[optB])
    predicted_direction = test["mean1"] > test["mean2"]
    return {
        "status": "ok",
        "study_id": study_id,
        "condition_id": condition_id,
        "option_compared": optA,
        "n_chose_optA": len(group_est_for_A[optA]),
        "n_chose_optB": len(group_est_for_A[optB]),
        "mean_estimate_pctA_by_optA_choosers": test["mean1"],
        "mean_estimate_pctA_by_optB_choosers": test["mean2"],
        "predicted_direction_confirmed": predicted_direction,
        "t": test["t"],
        "df": test["df"],
        "p_two_tailed_normal_approx": test["p_two_tailed_normal_approx"],
        "note": "p-value uses a normal approximation to the t-distribution; see welch_t_test docstring.",
    }


def evaluate_trait_extremity_hypothesis(records, study_id, condition_id):
    """H2 for study1/study3/study4: within each participant, the trait-rating
    extremity (|rating|, averaged over traits) for the actor who differs from
    the rater's own choice should exceed the extremity for the actor who
    matches the rater's own choice."""
    subset = [r for r in records if r.get("study_id") == study_id and r.get("condition_id") == condition_id and r.get("trait_ratings")]
    if len(subset) < 2:
        return {"status": "not_ready", "missing_requirement": f"need >=2 records with trait_ratings for {study_id}/{condition_id}; have {len(subset)}"}

    diffs = []
    for r in subset:
        choice = r.get("own_choice")
        traits = r.get("trait_ratings", {})
        matched_vals, mismatched_vals = [], []
        for trait, ratings in traits.items():
            for target, val in ratings.items():
                if target == choice:
                    matched_vals.append(abs(float(val)))
                else:
                    mismatched_vals.append(abs(float(val)))
        if not matched_vals or not mismatched_vals:
            continue
        diffs.append(statistics.mean(mismatched_vals) - statistics.mean(matched_vals))

    if len(diffs) < 2:
        return {"status": "not_ready", "missing_requirement": "could not compute matched/mismatched extremity pairs from trait_ratings"}

    test = one_sample_t_test(diffs)
    return {
        "status": "ok",
        "study_id": study_id,
        "condition_id": condition_id,
        "n": test["n"],
        "mean_extremity_difference_mismatched_minus_matched": test["mean_diff"],
        "predicted_direction_confirmed": test["mean_diff"] > 0,
        "t": test["t"],
        "df": test["df"],
        "p_two_tailed_normal_approx": test["p_two_tailed_normal_approx"],
        "note": "p-value uses a normal approximation to the t-distribution; see welch_t_test docstring.",
    }


def evaluate_study2(records):
    subset = [r for r in records if r.get("study_id") == "study2"]
    if not subset:
        return {"status": "not_ready", "missing_requirement": "no study2 records supplied"}

    per_item = defaultdict(lambda: {"cat1": [], "cat2": []})
    for r in subset:
        self_cat = {row["item_index"]: row["category_chosen"] for row in r.get("self_category", [])}
        peer_est = {row["item_index"]: row["pct_category_1"] for row in r.get("peer_estimate_pct", [])}
        for idx, cat in self_cat.items():
            if idx not in peer_est:
                continue
            bucket = "cat1" if cat == "category_1" else "cat2"
            per_item[idx][bucket].append(float(peer_est[idx]))

    item_results = {}
    predicted_direction_count = 0
    tested_count = 0
    for idx, groups in per_item.items():
        if len(groups["cat1"]) < 2 or len(groups["cat2"]) < 2:
            item_results[idx] = {"status": "not_ready", "missing_requirement": "need >=2 self-raters in each category for this item"}
            continue
        test = welch_t_test(groups["cat1"], groups["cat2"])
        predicted = test["mean1"] > test["mean2"]
        tested_count += 1
        if predicted:
            predicted_direction_count += 1
        item_results[idx] = {
            "status": "ok",
            "n_cat1": len(groups["cat1"]),
            "n_cat2": len(groups["cat2"]),
            "mean_estimate_by_cat1_raters": test["mean1"],
            "mean_estimate_by_cat2_raters": test["mean2"],
            "predicted_direction_confirmed": predicted,
            "t": test["t"],
            "p_two_tailed_normal_approx": test["p_two_tailed_normal_approx"],
        }

    if tested_count == 0:
        return {"status": "not_ready", "missing_requirement": "no item had >=2 respondents in both self-categories yet"}

    return {
        "status": "ok",
        "items_tested": tested_count,
        "items_total_available": len(per_item),
        "items_in_predicted_direction": predicted_direction_count,
        "paper_reported_baseline": "32 of 34 items in predicted direction (Ross, Greene & House, 1977, p.286)",
        "per_item": item_results,
    }


STUDY_CONDITIONS = {
    "study1": ["supermarket", "term_paper", "traffic_ticket", "space_program"],
    "study3": ["eat_at_joes", "repent"],
    "study4": ["generic"],
}


def run_evaluation(records, only_study=None):
    result = {"consensus_hypothesis": {}, "trait_extremity_hypothesis": {}, "study2": None}

    for study_id, conditions in STUDY_CONDITIONS.items():
        if only_study and study_id != only_study:
            continue
        for condition_id in conditions:
            key = f"{study_id}/{condition_id}"
            result["consensus_hypothesis"][key] = evaluate_consensus_hypothesis(records, study_id, condition_id)
            result["trait_extremity_hypothesis"][key] = evaluate_trait_extremity_hypothesis(records, study_id, condition_id)

    if only_study is None or only_study == "study2":
        result["study2"] = evaluate_study2(records)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="path to a JSON-lines file of adapter.py response records")
    parser.add_argument("--study", choices=["study1", "study2", "study3", "study4"], help="restrict evaluation to one study")
    args = parser.parse_args()

    try:
        records = load_records(args.records)
    except FileNotFoundError:
        print(json.dumps({"status": "not_ready", "missing_requirement": f"records file not found: {args.records}"}, indent=2))
        sys.exit(0)

    if not records:
        print(json.dumps({"status": "not_ready", "missing_requirement": "records file is empty"}, indent=2))
        sys.exit(0)

    result = run_evaluation(records, only_study=args.study)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
