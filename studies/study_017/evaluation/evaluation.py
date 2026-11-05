"""
Evaluation for Kannan, Pamuru & Rosokha (2023) task runs.

Consumes one or more session-record JSON files produced by
task/adapter.py (run_study1_session / run_study2_session output) and checks
the paper's reported findings (Results 6-10, Section 4.6). Each Result
requires specific conditions to be present across the supplied session files;
when a required condition/comparison is absent, that check returns a
structured `not_ready` result naming exactly what is missing -- it never
fabricates a comparison from data that was not run.

No network access is used or required.
"""

import argparse
import json
import random
from statistics import mean


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _rank_label(rank):
    return {1: "highest_valued", 2: "medium_valued", 3: "lowest_valued"}.get(rank)


def load_sessions(paths):
    sessions = []
    for p in paths:
        with open(p) as f:
            sessions.append(json.load(f))
    return sessions


def _matches_in_window(session, window):
    return [m for m in session["matches"] if window is None or m["match_index"] in window]


def extract_rows(session, window=None):
    """One row per (match, participant): rank label, num_adjustments,
    bid_to_value_ratio, allocative_efficiency (match-level, repeated per row),
    alpha, cost_c."""
    rows = []
    for m in _matches_in_window(session, window):
        for pid, payoff in m["payoffs"].items():
            rows.append({
                "match_index": m["match_index"],
                "rank": payoff["rank"],
                "rank_label": _rank_label(payoff["rank"]),
                "num_adjustments": payoff["num_adjustments"],
                "bid_to_value_ratio": payoff["bid_to_value_ratio"],
                "allocative_efficiency": m["allocative_efficiency"],
                "alpha": m["alpha"],
                "cost_c": m["cost_c"],
            })
    return rows


def permutation_test(sample_a, sample_b, n_permutations=10000, seed=0):
    """Two-tailed permutation test on the difference in means (Good 2013),
    matching the statistical test cited throughout Section 4.6 of the paper."""
    a = [x for x in sample_a if x is not None]
    b = [x for x in sample_b if x is not None]
    if len(a) == 0 or len(b) == 0:
        return None
    rng = random.Random(seed)
    observed = mean(a) - mean(b) if a and b else None
    if observed is None:
        return None
    pooled = a + b
    n_a = len(a)
    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_a = pooled[:n_a]
        perm_b = pooled[n_a:]
        diff = mean(perm_a) - mean(perm_b)
        if abs(diff) >= abs(observed):
            count_extreme += 1
    p_value = count_extreme / n_permutations
    return {"observed_difference": observed, "p_value": p_value, "n_permutations": n_permutations}


def bootstrap_se(sample, n_boot=2000, seed=0):
    xs = [x for x in sample if x is not None]
    if not xs:
        return None
    rng = random.Random(seed)
    boot_means = []
    n = len(xs)
    for _ in range(n_boot):
        resample = [xs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(mean(resample))
    m = mean(boot_means)
    var = sum((x - m) ** 2 for x in boot_means) / (len(boot_means) - 1) if len(boot_means) > 1 else 0.0
    return var ** 0.5


def _find_condition_sessions(sessions, key, value_a, value_b):
    a = [s for s in sessions if s.get(key) == value_a]
    b = [s for s in sessions if s.get(key) == value_b]
    return a, b


def check_result_6_lower_cost_more_exploration(sessions, window=None):
    """Result 6: 'Lower costs lead to more exploration.' Requires at least one
    session with cost_c == 0.0 and one with cost_c == 0.1 (same study_id)."""
    c0, c1 = _find_condition_sessions(sessions, "condition_c", 0.0, 0.1)
    if not c0 or not c1:
        return {"result": "Result 6", "status": "not_ready",
                "missing_requirement": "Need at least one session run with condition_c=0.0 "
                                        "and at least one with condition_c=0.1 to compare."}
    rows0 = [r for s in c0 for r in extract_rows(s, window)]
    rows1 = [r for s in c1 for r in extract_rows(s, window)]
    test = permutation_test([r["num_adjustments"] for r in rows0],
                             [r["num_adjustments"] for r in rows1])
    supported = test is not None and mean([r["num_adjustments"] for r in rows0]) > \
        mean([r["num_adjustments"] for r in rows1])
    return {"result": "Result 6", "status": "evaluated",
            "mean_adjustments_c0.0": _mean(r["num_adjustments"] for r in rows0),
            "mean_adjustments_c0.1": _mean(r["num_adjustments"] for r in rows1),
            "permutation_test": test, "directionally_consistent_with_paper": supported}


def check_result_7_lowest_valued_overbid_most(sessions, window=None):
    """Result 7: 'The bid-to-value ratio is higher for lower valued agents.'
    Evaluable within any single session that has at least one match."""
    rows = [r for s in sessions for r in extract_rows(s, window)]
    if not rows:
        return {"result": "Result 7", "status": "not_ready",
                "missing_requirement": "No match data supplied."}
    by_rank = {label: [r["bid_to_value_ratio"] for r in rows if r["rank_label"] == label]
               for label in ("highest_valued", "medium_valued", "lowest_valued")}
    means = {k: _mean(v) for k, v in by_rank.items()}
    test = permutation_test(by_rank["lowest_valued"], by_rank["highest_valued"])
    supported = (means["lowest_valued"] is not None and means["highest_valued"] is not None
                 and means["lowest_valued"] > means["highest_valued"])
    return {"result": "Result 7", "status": "evaluated", "mean_bid_to_value_by_rank": means,
            "permutation_test_lowest_vs_highest": test, "directionally_consistent_with_paper": supported}


def check_result_8_lowest_valued_friction_effect(sessions, window=None):
    """Result 8: 'For the lowest valued agents, the bid-to-value ratio
    increases as friction decreases.' Requires both cost_c conditions."""
    c0, c1 = _find_condition_sessions(sessions, "condition_c", 0.0, 0.1)
    if not c0 or not c1:
        return {"result": "Result 8", "status": "not_ready",
                "missing_requirement": "Need sessions at both condition_c=0.0 and condition_c=0.1."}
    r0 = [r for s in c0 for r in extract_rows(s, window) if r["rank_label"] == "lowest_valued"]
    r1 = [r for s in c1 for r in extract_rows(s, window) if r["rank_label"] == "lowest_valued"]
    test = permutation_test([r["bid_to_value_ratio"] for r in r0],
                             [r["bid_to_value_ratio"] for r in r1])
    m0, m1 = _mean(r["bid_to_value_ratio"] for r in r0), _mean(r["bid_to_value_ratio"] for r in r1)
    supported = m0 is not None and m1 is not None and m0 > m1
    return {"result": "Result 8", "status": "evaluated",
            "mean_bid_to_value_lowest_valued_c0.0": m0, "mean_bid_to_value_lowest_valued_c0.1": m1,
            "permutation_test": test, "directionally_consistent_with_paper": supported}


def check_result_9_alpha_effect(sessions, window=None):
    """Result 9: 'The bid-to-value ratio decreases as alpha increases for
    medium and highest valued agents.' Requires within-subjects alpha
    variation (Study 1 only; Study 2 fixes alpha=0.5)."""
    rows = [r for s in sessions for r in extract_rows(s, window)]
    alphas_present = sorted(set(r["alpha"] for r in rows))
    if len(alphas_present) < 2:
        return {"result": "Result 9", "status": "not_ready",
                "missing_requirement": "Need matches spanning at least two distinct alpha values "
                                        "(Study 1 varies alpha within-subjects; Study 2 fixes alpha=0.5 "
                                        "and cannot support this check)."}
    low_alpha, high_alpha = alphas_present[0], alphas_present[-1]
    out = {}
    for label in ("medium_valued", "highest_valued"):
        low = [r["bid_to_value_ratio"] for r in rows if r["rank_label"] == label and r["alpha"] == low_alpha]
        high = [r["bid_to_value_ratio"] for r in rows if r["rank_label"] == label and r["alpha"] == high_alpha]
        test = permutation_test(low, high)
        m_low, m_high = _mean(low), _mean(high)
        out[label] = {"mean_at_low_alpha": m_low, "mean_at_high_alpha": m_high,
                       "permutation_test": test,
                       "directionally_consistent_with_paper": (m_low is not None and m_high is not None
                                                                and m_low > m_high)}
    return {"result": "Result 9", "status": "evaluated", "low_alpha": low_alpha, "high_alpha": high_alpha,
            "by_rank": out}


def check_result_10_efficiency_increases_with_friction(sessions, window=None):
    """Result 10: 'Allocative efficiency of the market increases as friction
    costs increase.' Requires both cost_c conditions."""
    c0, c1 = _find_condition_sessions(sessions, "condition_c", 0.0, 0.1)
    if not c0 or not c1:
        return {"result": "Result 10", "status": "not_ready",
                "missing_requirement": "Need sessions at both condition_c=0.0 and condition_c=0.1."}
    eff0 = [m["allocative_efficiency"] for s in c0 for m in _matches_in_window(s, window)]
    eff1 = [m["allocative_efficiency"] for s in c1 for m in _matches_in_window(s, window)]
    test = permutation_test(eff0, eff1)
    m0, m1 = _mean(eff0), _mean(eff1)
    supported = m0 is not None and m1 is not None and m1 > m0
    return {"result": "Result 10", "status": "evaluated",
            "mean_efficiency_c0.0": m0, "mean_efficiency_c0.1": m1,
            "permutation_test": test, "directionally_consistent_with_paper": supported}


ALL_CHECKS = [
    check_result_6_lower_cost_more_exploration,
    check_result_7_lowest_valued_overbid_most,
    check_result_8_lowest_valued_friction_effect,
    check_result_9_alpha_effect,
    check_result_10_efficiency_increases_with_friction,
]


def evaluate(sessions, matches_window=range(5, 11)):
    """matches_window mirrors the paper's Section 4.6 analysis window
    (matches 5-10, evidence E21). Pass None to use all matches."""
    return {"analysis_window_match_indices": list(matches_window) if matches_window else "all",
            "num_sessions_supplied": len(sessions),
            "checks": [check(sessions, matches_window) for check in ALL_CHECKS]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", required=True, dest="sessions",
                         help="Path to a session-record JSON file from task/adapter.py. "
                              "May be repeated to supply multiple conditions/runs.")
    parser.add_argument("--all-matches", action="store_true",
                         help="Use all matches instead of the paper's matches-5-10 analysis window.")
    args = parser.parse_args()

    sessions = load_sessions(args.sessions)
    window = None if args.all_matches else range(5, 11)
    print(json.dumps(evaluate(sessions, window), indent=2, default=str))


if __name__ == "__main__":
    main()
