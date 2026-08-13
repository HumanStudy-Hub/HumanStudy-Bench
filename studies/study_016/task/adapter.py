#!/usr/bin/env python3
"""
Runnable adapter for the GSP bid-adjustment-friction task.

Models the two human-subject studies from Kannan, Pamuru, and Rosokha (2023),
"Analyzing Frictions in Generalized Second-Price Auction Markets",
Information Systems Research 34(4):1437-1454.

- study "human_main": Section 4 main experiment (between-subjects cost C in
  {0.0, 0.1}; within-subjects alpha in {0.2, 0.5, 0.8}).
- study "human_mechanism": Section 5 discrete-time / information-cost
  follow-up (2x2: cost C in {0.0, 0.1} x info_cost in {0.0, 0.5}; alpha fixed
  at 0.5).

See task/task.json for the full I/O contract, condition tables, and the
documented derived design decisions (continuous time is discretized into
decision checkpoints; the other two group members each match are simulated
policies, not replayed human data -- see audit/missing_information.json).

Runs fully offline (stdlib only, no network access) and exposes --smoke-test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Environment model (payoff formulas verbatim from the paper, Section 3.1,
# Equations 1-3; see source/evidence.json#ev.comp.payoff.formula)
# ---------------------------------------------------------------------------

ALPHA1 = 1.0  # click-through rate of slot 1, fixed by the paper (Section 3.1)


@dataclass
class Condition:
    study: str  # "human_main" or "human_mechanism"
    cost: float  # bid-adjustment cost C
    alpha: float  # value of Good 2 relative to Good 1
    info_cost: Optional[float] = None  # human_mechanism only

    def validate(self) -> None:
        if self.study not in ("human_main", "human_mechanism"):
            raise ValueError(f"unknown study '{self.study}'")
        if self.study == "human_main" and self.cost not in (0.0, 0.1):
            raise ValueError("human_main defines cost in {0.0, 0.1} (ev.human_main.treatments)")
        if self.study == "human_main" and self.alpha not in (0.2, 0.5, 0.8):
            raise ValueError("human_main defines alpha in {0.2, 0.5, 0.8} (ev.human_main.treatments)")
        if self.study == "human_mechanism":
            if self.cost not in (0.0, 0.1):
                raise ValueError("human_mechanism defines cost in {0.0, 0.1} (ev.human_mechanism.design)")
            if self.info_cost not in (0.0, 0.5):
                raise ValueError("human_mechanism defines info_cost in {0.0, 0.5} (ev.human_mechanism.design)")
            if self.alpha != 0.5:
                raise ValueError("human_mechanism fixes alpha = 0.5 (ev.human_mechanism.design)")


PolicyFn = Callable[[dict, random.Random], float]


def policy_truthful(state: dict, rng: random.Random) -> float:
    """Bid = own value. The weakly-dominant theoretical benchmark the paper
    itself invokes for the lowest-valued bidder (ev.comp.env.setup). Used as
    the default co-player and as the smoke-test agent-under-test stand-in."""
    return float(state["own_value_good1"])


def policy_softmax_explorer(state: dict, rng: random.Random) -> float:
    """Stylized Boltzmann explorer parameterized like the paper's OWN
    Q-learning agents (delta/gamma/lambda from Table 1,
    ev.comp.qlearning.params). This is a stress-test opponent provided for
    runnability -- it is NOT a claim about real human bidding, which the
    paper does not release at the individual-trajectory level.

    Candidates near the current bid get a cheap-proxy payoff of
    -|candidate - value|; any candidate that differs from the current bid
    additionally pays the session's bid_adjustment_cost, so -- consistent
    with the paper's own qualitative logic (Result 1/6) -- higher cost makes
    this policy adjust its bid less often."""
    value = state["own_value_good1"]
    current = state["current_own_bid"]
    cost = state["bid_adjustment_cost"]
    base = current if current is not None else value
    candidates = [max(0.0, min(10.0, base + step)) for step in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    lam = 1.0
    payoffs = []
    for c in candidates:
        p = -(abs(c - value))
        if current is not None and c != current:
            p -= cost
        payoffs.append(p)
    weights = [pow(2.718281828, lam * p) for p in payoffs]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for c, w in zip(candidates, weights):
        acc += w
        if r <= acc:
            return round(c, 2)
    return candidates[-1]


BUILTIN_POLICIES = {
    "truthful": policy_truthful,
    "softmax_explorer": policy_softmax_explorer,
}


@dataclass
class ParticipantLog:
    values_seen: list = field(default_factory=list)
    bids_history: list = field(default_factory=list)
    num_adjustments: int = 0
    num_info_purchases: int = 0
    payoff: float = 0.0


def _decision_schedule(study: str, rng: random.Random) -> int:
    """Draw the number of decision checkpoints for one match.

    human_main: derived discretization of the continuous-time / random-
    termination protocol (min 20s, 1%/s thereafter, expected 2 minutes) --
    see task.json derived_design_decisions D1 and ev.human_main.random_termination.
    Capped at 100 checkpoints (matches T=100 in the paper's own computational
    model, ev.comp.qlearning.params) purely to bound runtime.

    human_mechanism: reported discrete-time protocol -- termination
    probability 0.1 per period, no stated minimum, expected 10 periods
    (ev.human_mechanism.discrete_time_protocol). Exact duration-sequence
    realizations (Online Table N.1) are unavailable; periods are drawn fresh
    (task.json derived_design_decisions D3).
    """
    if study == "human_main":
        periods = 20
        while periods < 100 and rng.random() > 0.01:
            periods += 1
        return periods
    else:
        periods = 1
        while periods < 200 and rng.random() > 0.1:
            periods += 1
        return periods


def run_match(condition: Condition, match_index: int, agent_fn: PolicyFn,
              co_player_fn: PolicyFn, rng: random.Random) -> dict:
    """Simulate one match: agent-under-test plus two simulated co-players."""
    values = [rng.randint(1, 10) for _ in range(3)]
    values.sort(reverse=True)
    v_good1 = values  # [v_highest, v_medium, v_lowest] for good 1
    alpha = condition.alpha

    participants = ["agent", "co_player_1", "co_player_2"]
    fns = {"agent": agent_fn, "co_player_1": co_player_fn, "co_player_2": co_player_fn}
    logs = {p: ParticipantLog() for p in participants}
    current_bids = {p: None for p in participants}

    n_periods = _decision_schedule(condition.study, rng)

    for period in range(1, n_periods + 1):
        for idx, p in enumerate(participants):
            state = {
                "study": condition.study,
                "match_index": match_index,
                "period_index": period,
                "is_first_decision_in_match": current_bids[p] is None,
                "own_value_good1": v_good1[idx],
                "alpha": alpha,
                "bid_adjustment_cost": condition.cost,
                "current_own_bid": current_bids[p],
                "info_cost": condition.info_cost if condition.study == "human_mechanism" else None,
                "counterfactual_payoff_table": None,
            }
            new_bid = float(fns[p](state, rng))
            new_bid = max(0.0, min(10.0, new_bid))
            if current_bids[p] is not None and new_bid != current_bids[p]:
                logs[p].num_adjustments += 1
                logs[p].payoff -= condition.cost
            current_bids[p] = new_bid
            logs[p].bids_history.append(new_bid)
            logs[p].values_seen.append(v_good1[idx])

    ranked = sorted(zip(participants, current_bids.values(), v_good1),
                     key=lambda t: t[1], reverse=True)
    (p1, b1, val1), (p2, b2, val2), (p3, b3, val3) = ranked

    payoff1 = ALPHA1 * (val1 - b2)
    payoff2 = alpha * (val2 - b3)
    payoff3 = 0.0
    logs[p1].payoff += payoff1
    logs[p2].payoff += payoff2
    logs[p3].payoff += payoff3

    max_welfare = ALPHA1 * v_good1[0] + alpha * v_good1[1]
    realized_welfare = ALPHA1 * val1 + alpha * val2
    allocative_efficiency = (realized_welfare / max_welfare) if max_welfare > 0 else None

    bid_to_value = {p: (current_bids[p] / v) if v > 0 else None
                    for p, v in zip(participants, v_good1)}

    return {
        "match_index": match_index,
        "n_periods": n_periods,
        "values_by_rank": [val1, val2, val3],
        "final_bids_by_rank": [b1, b2, b3],
        "participant_by_rank": [p1, p2, p3],
        "winner_by_rank": [p1, p2, None],
        "payoff_by_participant": {p: round(logs[p].payoff, 4) for p in participants},
        "num_bid_adjustments_by_participant": {p: logs[p].num_adjustments for p in participants},
        "bid_to_value_by_participant": bid_to_value,
        "allocative_efficiency": allocative_efficiency,
        "rank_of_agent": [p1, p2, p3].index("agent") + 1,
    }


def run_session(condition: Condition, num_matches: int, agent_fn: PolicyFn,
                 co_player_fn: PolicyFn, seed: int) -> dict:
    condition.validate()
    rng = random.Random(seed)
    match_logs = [run_match(condition, m + 1, agent_fn, co_player_fn, rng)
                  for m in range(num_matches)]
    return {
        "condition": {
            "study": condition.study,
            "cost": condition.cost,
            "alpha": condition.alpha,
            "info_cost": condition.info_cost,
        },
        "num_matches": num_matches,
        "seed": seed,
        "matches": match_logs,
        "fidelity_note": (
            "Co-players are simulated policies, not replayed human data "
            "(see task/task.json 'co_players' and audit/missing_information.json AUD-04). "
            "Metrics below describe the agent-under-test's behavior in this "
            "simulated environment; they are not a claim of statistical "
            "equivalence to the paper's human sessions."
        ),
    }


def summarize(run_output: dict) -> dict:
    """Descriptive metrics analogous to the paper's Tables 2/5/8 (adjustments),
    3/6/9 (bid-to-value ratio), and Figure 1 / Tables 7/10 (allocative
    efficiency), computed for the agent-under-test only."""
    matches = run_output["matches"]
    if not matches:
        return {"error": "no matches to summarize"}
    agent_adjustments = [m["num_bid_adjustments_by_participant"]["agent"] for m in matches]
    agent_ratios = [m["bid_to_value_by_participant"]["agent"] for m in matches
                    if m["bid_to_value_by_participant"]["agent"] is not None]
    efficiencies = [m["allocative_efficiency"] for m in matches if m["allocative_efficiency"] is not None]
    rank_counts = {1: 0, 2: 0, 3: 0}
    for m in matches:
        rank_counts[m["rank_of_agent"]] += 1

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "mean_agent_bid_adjustments": mean(agent_adjustments),
        "mean_agent_bid_to_value_ratio": mean(agent_ratios),
        "mean_allocative_efficiency": mean(efficiencies),
        "agent_rank_distribution": rank_counts,
        "num_matches_summarized": len(matches),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke-test", action="store_true",
                    help="Run a short offline sanity check across both studies and exit.")
    p.add_argument("--study", choices=["human_main", "human_mechanism"], default="human_main")
    p.add_argument("--cost", type=float, default=0.0, help="Bid-adjustment cost C.")
    p.add_argument("--alpha", type=float, default=0.5, help="alpha (ignored/must be 0.5 for human_mechanism).")
    p.add_argument("--info-cost", type=float, default=None,
                   help="Information cost for human_mechanism (0.0 or 0.5). Required if --study human_mechanism.")
    p.add_argument("--num-matches", type=int, default=10)
    p.add_argument("--agent-policy", choices=list(BUILTIN_POLICIES.keys()), default="truthful",
                    help="Built-in stand-in for the agent-under-test (a real harness should inject its own agent_fn).")
    p.add_argument("--co-player-policy", choices=list(BUILTIN_POLICIES.keys()), default="truthful")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", type=str, default=None, help="Optional path to write full JSON run log.")
    return p


def run_smoke_test() -> dict:
    """Offline, dependency-free sanity check: runs a few matches for each
    study's default condition using built-in policies and returns True/False
    fitness plus a readiness note. No network access is used or required."""
    results = {}
    ok = True
    try:
        out_main = run_session(
            Condition(study="human_main", cost=0.1, alpha=0.5),
            num_matches=3, agent_fn=policy_truthful, co_player_fn=policy_softmax_explorer, seed=42,
        )
        results["human_main"] = summarize(out_main)
    except Exception as exc:  # pragma: no cover - smoke test must not crash the CLI
        ok = False
        results["human_main"] = {"error": str(exc)}

    try:
        out_mech = run_session(
            Condition(study="human_mechanism", cost=0.0, alpha=0.5, info_cost=0.5),
            num_matches=3, agent_fn=policy_truthful, co_player_fn=policy_softmax_explorer, seed=42,
        )
        results["human_mechanism"] = summarize(out_mech)
    except Exception as exc:  # pragma: no cover
        ok = False
        results["human_mechanism"] = {"error": str(exc)}

    return {
        "smoke_test_passed": ok,
        "results": results,
        "readiness_status": "runnable_with_documented_limitations",
        "limitations": (
            "Co-players are simulated (not replayed human data); continuous-time "
            "bidding in human_main is discretized into decision checkpoints. "
            "See task/task.json#derived_design_decisions and "
            "audit/missing_information.json for the full list."
        ),
    }


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.smoke_test:
        print(json.dumps(run_smoke_test(), indent=2))
        return 0

    if args.study == "human_mechanism" and args.info_cost is None:
        print(json.dumps({
            "blocked": True,
            "reason": "study=human_mechanism requires --info-cost {0.0 or 0.5} (ev.human_mechanism.design).",
        }, indent=2))
        return 1

    condition = Condition(
        study=args.study,
        cost=args.cost,
        alpha=(0.5 if args.study == "human_mechanism" else args.alpha),
        info_cost=args.info_cost,
    )
    try:
        condition.validate()
    except ValueError as exc:
        print(json.dumps({"blocked": True, "reason": str(exc)}, indent=2))
        return 1

    agent_fn = BUILTIN_POLICIES[args.agent_policy]
    co_player_fn = BUILTIN_POLICIES[args.co_player_policy]

    run_output = run_session(condition, args.num_matches, agent_fn, co_player_fn, args.seed)
    result = {"run": run_output, "summary": summarize(run_output)}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps({"written_to": args.output, "summary": result["summary"]}, indent=2))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
