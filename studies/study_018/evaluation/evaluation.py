"""Evaluation for the decentralized two-sided matching-market experiment.

evaluate(sessions) derives every metric directly from the session logs returned by
task/adapter.py's run_sessions() -- no separate data file, no pre-aggregation. It
reproduces (on whatever data the agents actually generate) the paper's headline
outcome measures: market-level and pair-level stability rates, and, for market
structures small enough to enumerate exactly (n_per_side <= 9, i.e. all of this
package's 'main' and 'unilateral' arms), the share of stable matchings/matches that
are the median, food-optimal, or color-optimal stable outcome (paper Tables 2-4).
"""

import itertools
from collections import defaultdict

MAX_ENUMERABLE_N = 9  # 9! = 362,880 permutations; brute-force enumeration is fast up to here


def _payoff(matrix, f, c):
    ci = int(c.split("-")[1]) - 1
    return matrix[f][ci][0], matrix[f][ci][1]


def blocking_pairs(matrix, matching, foods, colors):
    """matching: food -> color or None. Returns list of (f, c) blocking pairs."""
    rev = {c: f for f, c in matching.items() if c}
    blockers = []
    for f in foods:
        cur_c = matching.get(f)
        f_cur_pay = _payoff(matrix, f, cur_c)[0] if cur_c else -1
        for c in colors:
            if c == cur_c:
                continue
            f_pay, c_pay = _payoff(matrix, f, c)
            if f_pay > f_cur_pay:
                other_f = rev.get(c)
                c_cur_pay = _payoff(matrix, other_f, c)[1] if other_f else -1
                if c_pay > c_cur_pay:
                    blockers.append((f, c))
    return blockers


def deferred_acceptance(matrix, foods, colors, proposing_side):
    """Standard Gale-Shapley DA. proposing_side: 'food' or 'color'. Returns a matching
    dict food->color (proposer-optimal for the given proposing side)."""
    n = len(foods)
    if proposing_side == "food":
        prefs = {f: sorted(colors, key=lambda c: -_payoff(matrix, f, c)[0]) for f in foods}
        next_idx = {f: 0 for f in foods}
        held = {}  # color -> food currently held
        free = list(foods)
        while free:
            f = free.pop()
            if next_idx[f] >= n:
                continue
            c = prefs[f][next_idx[f]]
            next_idx[f] += 1
            if c not in held:
                held[c] = f
            else:
                incumbent = held[c]
                f_pay = _payoff(matrix, f, c)[1]
                inc_pay = _payoff(matrix, incumbent, c)[1]
                if f_pay > inc_pay:
                    held[c] = f
                    free.append(incumbent)
                else:
                    free.append(f)
        return {f: c for c, f in held.items()}
    else:
        prefs = {c: sorted(foods, key=lambda f: -_payoff(matrix, f, c)[1]) for c in colors}
        next_idx = {c: 0 for c in colors}
        held = {}  # food -> color currently held
        free = list(colors)
        while free:
            c = free.pop()
            if next_idx[c] >= n:
                continue
            f = prefs[c][next_idx[c]]
            next_idx[c] += 1
            if f not in held:
                held[f] = c
            else:
                incumbent = held[f]
                c_pay = _payoff(matrix, f, c)[0]
                inc_pay = _payoff(matrix, f, incumbent)[0]
                if c_pay > inc_pay:
                    held[f] = c
                    free.append(incumbent)
                else:
                    free.append(c)
        return dict(held)


def _is_stable(matrix, matching, foods, colors):
    return len(blocking_pairs(matrix, matching, foods, colors)) == 0


def enumerate_stable_matchings(matrix, foods, colors):
    """Brute-force enumeration by permutation search. Only called for n <= MAX_ENUMERABLE_N."""
    n = len(foods)
    stable = []
    for perm in itertools.permutations(range(n)):
        matching = {foods[i]: colors[perm[i]] for i in range(n)}
        if _is_stable(matrix, matching, foods, colors):
            stable.append(matching)
    return stable


def classify_matching(matching, stable_matchings, foods, colors, matrix):
    """Classify a (stable) matching as median / food_optimal / color_optimal / other,
    given the full enumerated set of stable matchings. Returns one of those 4 labels,
    or None if `matching` is not itself stable / not in the enumerated set."""
    if not stable_matchings or matching not in stable_matchings:
        return None
    # food-optimal: matching each food to its best stable partner (by food payoff)
    food_optimal = {}
    color_optimal = {}
    for f in foods:
        best_c, best_pay = None, -1
        worst_c, worst_pay = None, 10 ** 9
        for sm in stable_matchings:
            c = sm[f]
            pay = _payoff(matrix, f, c)[0]
            if pay > best_pay:
                best_pay, best_c = pay, c
            if pay < worst_pay:
                worst_pay, worst_c = pay, c
        food_optimal[f] = best_c
        color_optimal[f] = worst_c  # food's worst partner = color-optimal outcome for that food
    if matching == food_optimal:
        return "food_optimal"
    if matching == color_optimal:
        return "color_optimal"
    # median: each agent matched to the median-ranked partner among its stable partners
    is_median = True
    for f in foods:
        partners = sorted({sm[f] for sm in stable_matchings}, key=lambda c: -_payoff(matrix, f, c)[0])
        median_partner = partners[(len(partners) - 1) // 2] if len(partners) % 2 == 1 else None
        if median_partner is None or matching[f] != median_partner:
            is_median = False
            break
    if is_median:
        return "median"
    return "non_extremal_non_median"


def _agent_stable_partner_rank(matrix, foods, colors, stable_matchings, matched_food, matched_color, side, agent):
    """Return 'best'/'median'/'worst'/None classification of `matched_color` (or food)
    among `agent`'s own set of stable partners, from `agent`'s own preference order."""
    if side == "food":
        partners = sorted({sm[agent] for sm in stable_matchings}, key=lambda c: -_payoff(matrix, agent, c)[0])
        target = matched_color
    else:
        partners = sorted({f for f in foods if any(sm[f] == agent for sm in stable_matchings)},
                           key=lambda f: -_payoff(matrix, f, agent)[1])
        target = matched_food
    if target not in partners:
        return None
    idx = partners.index(target)
    if idx == 0:
        return "best"
    if idx == len(partners) - 1 and len(partners) > 1:
        return "worst"
    if len(partners) % 2 == 1 and idx == (len(partners) - 1) // 2:
        return "median"
    return "other_stable"


def evaluate(sessions):
    if not sessions:
        return {"not_ready": {"reason": "no sessions provided"}}

    by_arm = defaultdict(list)
    for s in sessions:
        by_arm[s["arm"]].append(s)

    arm_results = {}
    for arm, arm_sessions in by_arm.items():
        n = arm_sessions[0]["n_per_side"]
        foods = ["food-%d" % (i + 1) for i in range(n)]
        colors = ["color-%d" % (j + 1) for j in range(n)]

        n_markets = len(arm_sessions)
        n_stable_markets = 0
        pair_stability_ratios = []
        unmatched_fracs = []
        offers_per_market = []
        ticks_per_market = []
        parse_fallback_count = 0
        total_events = 0
        median_share_market = []
        food_optimal_share_market = []
        color_optimal_share_market = []
        median_share_match = 0
        food_optimal_share_match = 0
        color_optimal_share_match = 0
        classified_matches = 0
        enumeration_skipped = n > MAX_ENUMERABLE_N

        for s in arm_sessions:
            matrix = s["payoff_matrix"]
            matching = s["final_matching"]
            offers = [e for e in s["events"] if e["action"] == "OFFER"]
            offers_per_market.append(len(offers))
            ticks_per_market.append(s["n_ticks_used"])
            total_events += len(s["events"])
            parse_fallback_count += sum(1 for e in s["events"] if e.get("parse_fallback") or e.get("response_parse_fallback"))

            bp = blocking_pairs(matrix, matching, foods, colors)
            is_stable = len(bp) == 0
            if is_stable:
                n_stable_markets += 1
            n_pairs = sum(1 for v in matching.values() if v)
            n_agents_in_bp = len({a for pair in bp for a in pair})
            pair_stability_ratio = 1.0 if n_pairs == 0 else 1 - (n_agents_in_bp / (2 * n))
            pair_stability_ratios.append(pair_stability_ratio)
            unmatched_fracs.append(1 - n_pairs / n)

            if not enumeration_skipped:
                stable_matchings = enumerate_stable_matchings(matrix, foods, colors)
                if is_stable and stable_matchings:
                    label = classify_matching(matching, stable_matchings, foods, colors, matrix)
                    median_share_market.append(1 if label == "median" else 0)
                    food_optimal_share_market.append(1 if label == "food_optimal" else 0)
                    color_optimal_share_market.append(1 if label == "color_optimal" else 0)
                    for f, c in matching.items():
                        if not c:
                            continue
                        classified_matches += 1
                        rank = _agent_stable_partner_rank(matrix, foods, colors, stable_matchings, f, c, "food", f)
                        # a stable *match* is one where both f and c are mutually stable partners in some sm
                        is_stable_pair = any(sm[f] == c for sm in stable_matchings)
                        if is_stable_pair:
                            if rank == "median":
                                median_share_match += 1
                            elif rank == "best":
                                food_optimal_share_match += 1
                            elif rank == "worst":
                                color_optimal_share_match += 1

        arm_results[arm] = {
            "n_markets": n_markets,
            "n_per_side": n,
            "pct_markets_stable": 100.0 * n_stable_markets / n_markets,
            "avg_pct_pairs_without_blocking_partner": 100.0 * sum(pair_stability_ratios) / n_markets,
            "avg_pct_unmatched_agents": 100.0 * sum(unmatched_fracs) / n_markets,
            "avg_n_offers_per_market": sum(offers_per_market) / n_markets,
            "avg_n_ticks_per_market": sum(ticks_per_market) / n_markets,
            "parse_fallback_rate": (parse_fallback_count / total_events) if total_events else 0.0,
            "median_and_extremal_classification": (
                {
                    "skipped_reason": "n_per_side=%d exceeds brute-force enumeration cap (%d); exact stable-matching"
                                       " set, and hence median/extremal classification, is not computed for this arm."
                                       " See audit/missing_information.json:'median_classification_large_markets'." % (n, MAX_ENUMERABLE_N)
                } if enumeration_skipped else {
                    "pct_of_stable_markets_that_are_median": (
                        100.0 * sum(median_share_market) / len(median_share_market) if median_share_market else None
                    ),
                    "pct_of_stable_markets_that_are_food_optimal": (
                        100.0 * sum(food_optimal_share_market) / len(food_optimal_share_market) if food_optimal_share_market else None
                    ),
                    "pct_of_stable_markets_that_are_color_optimal": (
                        100.0 * sum(color_optimal_share_market) / len(color_optimal_share_market) if color_optimal_share_market else None
                    ),
                    "pct_of_stable_matches_that_are_median": (
                        100.0 * median_share_match / classified_matches if classified_matches else None
                    ),
                    "pct_of_stable_matches_that_are_food_optimal": (
                        100.0 * food_optimal_share_match / classified_matches if classified_matches else None
                    ),
                    "pct_of_stable_matches_that_are_color_optimal": (
                        100.0 * color_optimal_share_match / classified_matches if classified_matches else None
                    ),
                }
            ),
        }

    comparisons = {}
    pairs = [
        ("main_unique_sm", "unilateral_unique_sm", "bilateral_vs_unilateral__unique_sm"),
        ("main_four_by_four", "unilateral_four_by_four", "bilateral_vs_unilateral__four_by_four"),
        ("main_five_stable", "unilateral_five_stable", "bilateral_vs_unilateral__five_stable"),
    ]
    for a, b, key in pairs:
        if a in arm_results and b in arm_results:
            comparisons[key] = {
                "bilateral_pct_markets_stable": arm_results[a]["pct_markets_stable"],
                "unilateral_pct_markets_stable": arm_results[b]["pct_markets_stable"],
                "paper_reports": "bilateral 88.24% / unilateral 60.47% overall (paper Table 4); "
                                  "see source/evidence.json:finding_unilateral_comparison",
            }
    if "main_unique_sm" in arm_results and "large_unique_sm" in arm_results:
        comparisons["main_vs_large__unique_sm"] = {
            "main_pct_markets_stable": arm_results["main_unique_sm"]["pct_markets_stable"],
            "large_pct_markets_stable": arm_results["large_unique_sm"]["pct_markets_stable"],
            "paper_reports": "main 88.24% / large 66.67% overall (paper Table 4); "
                              "see source/evidence.json:finding_large_market_comparison",
        }

    return {
        "by_arm": arm_results,
        "cross_arm_comparisons": comparisons,
        "notes": [
            "All figures above are computed from the sessions actually produced by run_sessions() for "
            "whatever model was injected as llm() -- they are not copied from the paper. Compare them "
            "against the paper's own figures recorded in source/evidence.json (entries prefixed 'finding_') "
            "and paper.pdf Tables 2-4, which describe human-subject behavior, not a ground truth the agents "
            "are expected to reproduce.",
            "Sample sizes here are typically much smaller than the paper's (85/43/12 real experimental "
            "markets); treat any single evaluate() call as illustrative unless the researcher requests a "
            "much larger n in run_sessions().",
        ],
    }
