#!/usr/bin/env python3
"""
Evaluation checks for the GSP bid-adjustment-friction task
(task/adapter.py, task/task.json).

Implements the descriptive checks that a single adapter run (or a small set
of runs across conditions) actually supports, modeled on the paper's own
qualitative Results:

  Result 1/6:  lower bid-adjustment cost -> more bid adjustments (exploration)
  Result 2/7:  bid-to-value ratio is higher for lower-valued participants
  Result 3/8:  for the lowest-valued participants, bid-to-value ratio
               increases as friction (cost) decreases
  Result 5/10: allocative efficiency increases as friction cost increases

Each check needs one or more run logs produced by
`task/adapter.py ... --output run.json` (or the `run_session()` /
`run(...)` return value in-process). Checks that need data this package
does not have (e.g., the paper's bootstrapped permutation-test p-values
computed over real independent human sessions) return a structured
`not_ready` result naming the missing requirement, rather than fabricating
a p-value. See audit/missing_information.json (AUD-08, AUD-09).

Runs fully offline (stdlib only).
"""

from __future__ import annotations

import json
import sys
from typing import Optional


RANK_LABELS = ("highest", "medium", "lowest")


def _rank_metrics(run_output: dict) -> dict:
    """Per-rank (highest/medium/lowest private value) descriptive means,
    pooling across ALL THREE participants (agent-under-test + the two
    simulated co-players) in every match of one run. This mirrors how the
    paper pools across all subjects who happened to hold a given rank
    (Tables 2/3/5/6/8/9), except the paper pools across real human subjects
    and this pools across one agent-under-test plus simulated co-players.
    """
    adj = {r: [] for r in RANK_LABELS}
    ratio = {r: [] for r in RANK_LABELS}
    efficiency = []

    for m in run_output["matches"]:
        participant_by_rank = m.get("participant_by_rank")
        if participant_by_rank is None:
            raise ValueError(
                "match record missing 'participant_by_rank'; run log was produced "
                "by an incompatible/older adapter.py version"
            )
        for label, participant in zip(RANK_LABELS, participant_by_rank):
            adj[label].append(m["num_bid_adjustments_by_participant"][participant])
            bv = m["bid_to_value_by_participant"][participant]
            if bv is not None:
                ratio[label].append(bv)
        if m["allocative_efficiency"] is not None:
            efficiency.append(m["allocative_efficiency"])

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "mean_bid_adjustments_by_rank": {r: mean(adj[r]) for r in RANK_LABELS},
        "mean_bid_to_value_ratio_by_rank": {r: mean(ratio[r]) for r in RANK_LABELS},
        "mean_allocative_efficiency": mean(efficiency),
        "n_matches": len(run_output["matches"]),
    }


def check_exploration_direction(run_low_cost: dict, run_high_cost: dict) -> dict:
    """Result 1/6: lower cost -> more bid adjustments, checked within a
    matched pair of runs that differ ONLY in `cost` (same study/alpha/
    info_cost/seed-generating conditions otherwise)."""
    c_low = run_low_cost["condition"]["cost"]
    c_high = run_high_cost["condition"]["cost"]
    if c_low >= c_high:
        return {
            "check": "exploration_direction",
            "status": "not_ready",
            "reason": "run_low_cost.condition.cost must be strictly less than run_high_cost.condition.cost",
        }
    low = _rank_metrics(run_low_cost)
    high = _rank_metrics(run_high_cost)
    per_rank = {}
    all_consistent = True
    for r in RANK_LABELS:
        lo, hi = low["mean_bid_adjustments_by_rank"][r], high["mean_bid_adjustments_by_rank"][r]
        consistent = (lo is not None and hi is not None and lo >= hi)
        all_consistent = all_consistent and consistent
        per_rank[r] = {"low_cost_mean_adjustments": lo, "high_cost_mean_adjustments": hi,
                        "direction_matches_paper": consistent}
    return {
        "check": "exploration_direction",
        "paper_result_ids": ["Result 1", "Result 6"],
        "evidence_ids": ["ev.comp.result1", "ev.human_main.result6"],
        "status": "computed",
        "direction_matches_paper_for_all_ranks": all_consistent,
        "per_rank": per_rank,
        "caveat": "Descriptive comparison of two simulated-co-player runs; not a significance test over independent human sessions (see check_significance_not_ready below).",
    }


def check_overbidding_by_lowest_valued(run: dict) -> dict:
    """Result 2/7: bid-to-value ratio is higher for lower-valued participants
    within a single run."""
    metrics = _rank_metrics(run)
    r = metrics["mean_bid_to_value_ratio_by_rank"]
    if any(r[k] is None for k in RANK_LABELS):
        return {"check": "overbidding_by_lowest_valued", "status": "not_ready",
                "reason": "one or more ranks had no valid bid-to-value observations in this run"}
    ordered_correctly = r["lowest"] >= r["medium"] >= r["highest"]
    return {
        "check": "overbidding_by_lowest_valued",
        "paper_result_ids": ["Result 2", "Result 7"],
        "evidence_ids": ["ev.comp.result2to4", "ev.human_main.result7to8"],
        "status": "computed",
        "mean_bid_to_value_ratio_by_rank": r,
        "direction_matches_paper": ordered_correctly,
    }


def check_friction_reduces_lowest_valued_overbidding(run_low_cost: dict, run_high_cost: dict) -> dict:
    """Result 3/8: for the lowest-valued participants, bid-to-value ratio
    increases as friction (cost) decreases, i.e. is higher in the low-cost
    run than the high-cost run."""
    c_low = run_low_cost["condition"]["cost"]
    c_high = run_high_cost["condition"]["cost"]
    if c_low >= c_high:
        return {
            "check": "friction_reduces_lowest_valued_overbidding",
            "status": "not_ready",
            "reason": "run_low_cost.condition.cost must be strictly less than run_high_cost.condition.cost",
        }
    low = _rank_metrics(run_low_cost)["mean_bid_to_value_ratio_by_rank"]["lowest"]
    high = _rank_metrics(run_high_cost)["mean_bid_to_value_ratio_by_rank"]["lowest"]
    if low is None or high is None:
        return {"check": "friction_reduces_lowest_valued_overbidding", "status": "not_ready",
                "reason": "missing lowest-rank bid-to-value observations in one of the two runs"}
    return {
        "check": "friction_reduces_lowest_valued_overbidding",
        "paper_result_ids": ["Result 3", "Result 8"],
        "evidence_ids": ["ev.comp.result2to4", "ev.human_main.result7to8"],
        "status": "computed",
        "lowest_rank_bid_to_value_ratio": {"low_cost_run": low, "high_cost_run": high},
        "direction_matches_paper": low >= high,
    }


def check_efficiency_increases_with_friction(run_low_cost: dict, run_high_cost: dict) -> dict:
    """Result 5/10: allocative efficiency increases as friction cost
    increases."""
    c_low = run_low_cost["condition"]["cost"]
    c_high = run_high_cost["condition"]["cost"]
    if c_low >= c_high:
        return {
            "check": "efficiency_increases_with_friction",
            "status": "not_ready",
            "reason": "run_low_cost.condition.cost must be strictly less than run_high_cost.condition.cost",
        }
    low = _rank_metrics(run_low_cost)["mean_allocative_efficiency"]
    high = _rank_metrics(run_high_cost)["mean_allocative_efficiency"]
    if low is None or high is None:
        return {"check": "efficiency_increases_with_friction", "status": "not_ready",
                "reason": "missing allocative-efficiency observations in one of the two runs"}
    return {
        "check": "efficiency_increases_with_friction",
        "paper_result_ids": ["Result 5", "Result 10"],
        "evidence_ids": ["ev.comp.result5", "ev.human_main.result10"],
        "status": "computed",
        "mean_allocative_efficiency": {"low_cost_run": low, "high_cost_run": high},
        "direction_matches_paper": high >= low,
    }


def check_significance_not_ready(reason: Optional[str] = None) -> dict:
    """The paper's inferential statistics (bootstrapped standard errors and
    two-tailed permutation tests across MULTIPLE INDEPENDENT SESSIONS of
    real human subjects; Tables 5-10) cannot be honestly reproduced from
    adapter.py runs, because:
      (a) co-players are simulated policies, not independent human sessions
          (audit AUD-04), so there is no valid cross-session sampling
          variation to bootstrap or permute; and
      (b) the paper's raw session-level data / Online Appendix M robustness
          tables are not included in the uploaded PDF (audit AUD-08).
    This function documents that gap structurally instead of fabricating a
    p-value.
    """
    return {
        "check": "significance_tests_matching_tables_5_through_10",
        "status": "not_ready",
        "missing_requirement": (
            reason or
            "Independent per-session human (or validated behavioral-clone) data "
            "across the C in {0.0,0.1} x alpha in {0.2,0.5,0.8} (human_main) or "
            "C x info_cost (human_mechanism) cells, to bootstrap standard errors "
            "and run two-tailed permutation tests as in the paper (Good 2013)."
        ),
        "audit_ref": "AUD-08",
    }


def evaluate_from_files(paths: list[str]) -> dict:
    runs = []
    for path in paths:
        with open(path) as f:
            payload = json.load(f)
        runs.append(payload["run"] if "run" in payload else payload)
    return evaluate(runs)


def evaluate(runs: list[dict]) -> dict:
    """Top-level entry point. `runs` is a list of run-log dicts as produced
    by adapter.run_session(...) (or the "run" field of adapter.py's JSON
    stdout/--output). Automatically pairs up runs that share study/alpha/
    info_cost but differ in cost, to run the cost-contrast checks; always
    runs the single-run checks on every run provided."""
    results = {"single_run_checks": [], "paired_run_checks": [], "not_ready": []}

    for run in runs:
        results["single_run_checks"].append({
            "condition": run["condition"],
            "result": check_overbidding_by_lowest_valued(run),
        })

    by_key = {}
    for run in runs:
        c = run["condition"]
        key = (c["study"], c.get("alpha"), c.get("info_cost"))
        by_key.setdefault(key, {})[c["cost"]] = run

    for key, by_cost in by_key.items():
        if 0.0 in by_cost and (0.1 in by_cost):
            low, high = by_cost[0.0], by_cost[0.1]
            results["paired_run_checks"].append({
                "matched_on": {"study": key[0], "alpha": key[1], "info_cost": key[2]},
                "exploration_direction": check_exploration_direction(low, high),
                "friction_reduces_lowest_valued_overbidding": check_friction_reduces_lowest_valued_overbidding(low, high),
                "efficiency_increases_with_friction": check_efficiency_increases_with_friction(low, high),
            })

    if not results["paired_run_checks"]:
        results["not_ready"].append({
            "check": "cost_contrast_checks",
            "status": "not_ready",
            "missing_requirement": (
                "Need at least one matched pair of runs with cost=0.0 and cost=0.1 "
                "(same study/alpha/info_cost) to evaluate Results 1/3/5/6/8/10."
            ),
        })

    results["not_ready"].append(check_significance_not_ready())
    return results


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(json.dumps({
            "usage": "evaluation.py RUN_LOG.json [RUN_LOG2.json ...]",
            "note": "Each RUN_LOG.json is produced by: python3 ../task/adapter.py ... --output RUN_LOG.json",
        }, indent=2))
        return 0
    print(json.dumps(evaluate_from_files(argv), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
