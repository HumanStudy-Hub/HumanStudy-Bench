"""
Runnable-task adapter for Kannan, Pamuru & Rosokha (2023), "Analyzing Frictions
in Generalized Second-Price Auction Markets", Information Systems Research.

Implements the two human-subject studies (Section 4 "main experiment" and
Section 5 "mechanism experiment") as a multiplayer bidding simulation in which
every original human subject is one agent participant. No network access is
used or required. See task/task.json for the full design contract and
audit/missing_information.json for disclosed departures/assumptions.

This module is a library first: a harness supplies real agent decision logic
by implementing ParticipantAgent and passing instances into run_study1_session
/ run_study2_session. `python adapter.py --smoke-test` runs a tiny self-test
with a trivial scripted stand-in agent (SmokeTestAgent) purely to verify the
mechanics execute end-to-end -- that stand-in is NOT a valid substitute for a
real participant in a research run.
"""

import argparse
import json
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Agent interface -- a real run must supply subclasses backed by an LLM or
# other genuine decision-making process. One instance per participant.
# ---------------------------------------------------------------------------

class ParticipantAgent(ABC):
    """One instance represents one human-subject stand-in. Never share state
    between agents that were not allowed to communicate in the paper."""

    @abstractmethod
    def initial_bid(self, observation: dict) -> float:
        """observation contains: value_good1, value_good2, alpha, cost_c,
        match_index, study. Must return a bid in [0, 10]."""
        raise NotImplementedError

    @abstractmethod
    def decide_bid_update(self, observation: dict) -> Optional[float]:
        """Called every period after the initial bid. observation additionally
        contains: period, current_own_bid, provisional_rank (1/2/3),
        price_if_winning (revealed only if provisional_rank is 1 or 2, else
        None), periods_elapsed. Return None to keep the current bid (no cost),
        or a new bid in [0, 10] (incurs cost_c)."""
        raise NotImplementedError

    def decide_info_access(self, observation: dict) -> bool:
        """Study 2 only. observation additionally contains info_cost (0 or
        0.5) and, if available, the counterfactual payoff plot inputs.
        Default: never access. Override for real Study-2 participants."""
        return False


class SmokeTestAgent(ParticipantAgent):
    """Deterministic, minimal stand-in used ONLY by --smoke-test to check that
    the simulation mechanics run without error. Bids true value once, never
    revises. This is a code self-test fixture, not a research participant."""

    def initial_bid(self, observation: dict) -> float:
        return float(observation["value_good1"])

    def decide_bid_update(self, observation: dict) -> Optional[float]:
        return None

    def decide_info_access(self, observation: dict) -> bool:
        return False


# ---------------------------------------------------------------------------
# Shared GSP payoff mechanics (Section 3.1 / 4.3, evidence E07/E18)
# ---------------------------------------------------------------------------

def compute_match_payoffs(values_good1, alpha, final_bids, adjustment_counts, cost_c,
                           info_costs=None):
    """values_good1, final_bids, adjustment_counts: dict participant_id -> value.
    Returns dict participant_id -> record with rank, good_won, price_paid, payoff,
    allocative efficiency (shared across the match) and bid-to-value ratio.
    Ties are broken uniformly at random, matching Section 3.2.1 ('ties broken
    randomly')."""
    ids = list(values_good1.keys())
    order = sorted(ids, key=lambda i: (-final_bids[i], random.random()))
    rank_of = {pid: r + 1 for r, pid in enumerate(order)}
    b = {rank_of[pid]: final_bids[pid] for pid in ids}
    v = {rank_of[pid]: values_good1[pid] for pid in ids}

    price_paid = {1: b.get(2, 0.0), 2: alpha * b.get(3, 0.0), 3: 0.0}
    value_realized = {1: v[1], 2: alpha * v[2], 3: 0.0}

    total_value = sum(value_realized.values())
    max_possible = max(v[1], alpha * v[1])  # best single allocation of good1's top value
    # Allocative efficiency per Section 3.1 eq. (1): realized welfare / max possible welfare
    # Max possible welfare allocates the two goods to the two highest-value agents.
    sorted_vals = sorted(v.values(), reverse=True)
    max_possible = sorted_vals[0] * 1.0 + sorted_vals[1] * alpha
    efficiency = (total_value / max_possible) if max_possible > 0 else None

    out = {}
    for pid in ids:
        r = rank_of[pid]
        friction_cost = cost_c * adjustment_counts.get(pid, 0)
        info_cost = (info_costs or {}).get(pid, 0.0)
        payoff = value_realized[r] - price_paid[r] - friction_cost - info_cost
        bid_to_value_ratio = (final_bids[pid] / values_good1[pid]) if values_good1[pid] else None
        out[pid] = {
            "rank": r,
            "good_won": 1 if r == 1 else (2 if r == 2 else None),
            "final_bid": final_bids[pid],
            "value_good1": values_good1[pid],
            "price_paid": price_paid[r],
            "friction_cost": friction_cost,
            "info_cost": info_cost,
            "payoff": payoff,
            "bid_to_value_ratio": bid_to_value_ratio,
            "num_adjustments": adjustment_counts.get(pid, 0),
        }
    return out, efficiency


def draw_termination_period(rng: random.Random, min_periods: int, hazard: float,
                             max_periods: int) -> int:
    """Section 4.2 Remark 1 / Section 5: minimum periods, then i.i.d. per-period
    hazard of ending. max_periods is a disclosed practical safety cap (see
    task/task.json known_departures_from_original_design)."""
    t = min_periods
    while t < max_periods:
        if rng.random() < hazard:
            break
        t += 1
    return t


def make_triads(participant_ids, rng: random.Random):
    """Random partition into groups of three (Section 4.2: 'randomly split into
    groups of three ... remained so until the end of the match')."""
    if len(participant_ids) % 3 != 0:
        raise ValueError("num_agents must be a multiple of 3 (see task/task.json)")
    shuffled = list(participant_ids)
    rng.shuffle(shuffled)
    return [shuffled[i:i + 3] for i in range(0, len(shuffled), 3)]


# ---------------------------------------------------------------------------
# Study 1: main experiment (Section 4)
# ---------------------------------------------------------------------------

@dataclass
class Study1Config:
    num_agents: int = 3
    num_matches: int = 10
    condition_c: float = 0.0                # between-subjects: 0.0 or 0.1
    alpha_values: tuple = (0.2, 0.5, 0.8)    # within-subjects
    min_periods: int = 20
    hazard: float = 0.01
    max_periods: int = 300                  # practical cap, see task.json
    value_low: int = 1
    value_high: int = 10
    seed: int = 0


def run_study1_match(triad_ids, agents, alpha, cost_c, stop_period, rng, study="study_1_main_experiment"):
    values = {pid: rng.randint(1, 10) for pid in triad_ids}
    bids = {}
    adjustments = {pid: 0 for pid in triad_ids}
    period_log = []

    obs_base = {"study": study, "alpha": alpha, "cost_c": cost_c}
    for pid in triad_ids:
        obs = dict(obs_base, value_good1=values[pid], value_good2=alpha * values[pid])
        b = agents[pid].initial_bid(obs)
        bids[pid] = max(0.0, min(10.0, float(b)))

    for period in range(1, stop_period + 1):
        order = sorted(triad_ids, key=lambda i: -bids[i])
        rank_of = {p: r + 1 for r, p in enumerate(order)}
        for pid in triad_ids:
            r = rank_of[pid]
            price_if_winning = None
            if r == 1:
                price_if_winning = sorted([bids[o] for o in triad_ids if o != pid])[-1]
            elif r == 2:
                price_if_winning = alpha * min(bids[o] for o in triad_ids if o != pid)
            obs = dict(obs_base, value_good1=values[pid], value_good2=alpha * values[pid],
                       period=period, current_own_bid=bids[pid], provisional_rank=r,
                       price_if_winning=price_if_winning, periods_elapsed=period)
            new_bid = agents[pid].decide_bid_update(obs)
            if new_bid is not None:
                new_bid = max(0.0, min(10.0, float(new_bid)))
                if new_bid != bids[pid]:
                    adjustments[pid] += 1
                    bids[pid] = new_bid
        period_log.append({"period": period, "bids": dict(bids)})

    payoffs, efficiency = compute_match_payoffs(values, alpha, bids, adjustments, cost_c)
    return {
        "triad": triad_ids, "alpha": alpha, "cost_c": cost_c,
        "stop_period": stop_period, "values": values, "final_bids": dict(bids),
        "adjustments": adjustments, "payoffs": payoffs, "allocative_efficiency": efficiency,
        "period_log": period_log,
    }


def run_study1_session(agents: dict, config: Study1Config = Study1Config()):
    """agents: dict participant_id -> ParticipantAgent, len multiple of 3."""
    rng = random.Random(config.seed)
    participant_ids = list(agents.keys())
    if len(participant_ids) != config.num_agents:
        raise ValueError("agents dict size must equal config.num_agents")

    matches = []
    for m in range(1, config.num_matches + 1):
        triads = make_triads(participant_ids, rng)
        alpha = rng.choice(list(config.alpha_values))
        stop_period = draw_termination_period(rng, config.min_periods, config.hazard, config.max_periods)
        for triad in triads:
            record = run_study1_match(triad, agents, alpha, config.condition_c, stop_period, rng)
            record["match_index"] = m
            matches.append(record)

    return {
        "study_id": "study_1_main_experiment",
        "condition_c": config.condition_c,
        "num_agents": config.num_agents,
        "num_matches": config.num_matches,
        "seed": config.seed,
        "matches": matches,
        "assumptions": [
            "value_good1 drawn Uniform{1,...,10} (evidence_label: derived, see task/task.json)",
            "bid domain clamped to [0,10] (evidence_label: derived)",
            "no rounding applied to value_good2 = alpha * value_good1 (evidence_label: missing, see E23)",
            f"max_periods safety cap = {config.max_periods} applied to the theoretically unbounded random-termination process",
        ],
    }


# ---------------------------------------------------------------------------
# Study 2: mechanism experiment (Section 5)
# ---------------------------------------------------------------------------

@dataclass
class Study2Config:
    num_agents: int = 3
    num_matches: int = 10
    condition_c: float = 0.0                 # between-subjects: 0.0 or 0.1
    info_cost: float = 0.0                    # between-subjects: 0.0 (free) or 0.5 (costly)
    alpha: float = 0.5                        # fixed in Study 2
    hazard: float = 0.1
    max_periods: int = 60                    # practical cap, see task.json
    seed: int = 0


def run_study2_match(triad_ids, agents, alpha, cost_c, info_cost, stop_period, rng):
    values = {pid: rng.randint(1, 10) for pid in triad_ids}
    bids = {}
    adjustments = {pid: 0 for pid in triad_ids}
    info_access_counts = {pid: 0 for pid in triad_ids}
    period_log = []

    obs_base = {"study": "study_2_mechanism_experiment", "alpha": alpha, "cost_c": cost_c,
                "info_cost": info_cost}
    for pid in triad_ids:
        obs = dict(obs_base, value_good1=values[pid], value_good2=alpha * values[pid])
        b = agents[pid].initial_bid(obs)
        bids[pid] = max(0.0, min(10.0, float(b)))

    prev_bids = dict(bids)
    for period in range(1, stop_period + 1):
        # Simultaneous decisions within the period; each agent may consult the
        # counterfactual-payoff plot (built from the OTHER two agents' previous
        # period bids, per evidence E15) before deciding whether to adjust.
        order = sorted(triad_ids, key=lambda i: -prev_bids[i])
        rank_of = {p: r + 1 for r, p in enumerate(order)}
        new_bids = dict(bids)
        for pid in triad_ids:
            others_prev = [prev_bids[o] for o in triad_ids if o != pid]
            r = rank_of[pid]
            price_if_winning = None
            if r == 1:
                price_if_winning = sorted(others_prev)[-1]
            elif r == 2:
                price_if_winning = alpha * min(others_prev)
            info_obs = dict(obs_base, value_good1=values[pid], value_good2=alpha * values[pid],
                             period=period, current_own_bid=bids[pid],
                             counterfactual_grid=[
                                 {"candidate_bid": cb,
                                  "payoff_if_others_hold_previous_bids":
                                      _counterfactual_payoff(cb, others_prev, values[pid], alpha)}
                                 for cb in [x / 2 for x in range(0, 21)]
                             ])
            accessed = agents[pid].decide_info_access(info_obs)
            if accessed:
                info_access_counts[pid] += 1

            obs = dict(obs_base, value_good1=values[pid], value_good2=alpha * values[pid],
                       period=period, current_own_bid=bids[pid], provisional_rank=r,
                       price_if_winning=price_if_winning, periods_elapsed=period,
                       info_plot=info_obs["counterfactual_grid"] if accessed else None)
            new_bid = agents[pid].decide_bid_update(obs)
            if new_bid is not None:
                new_bid = max(0.0, min(10.0, float(new_bid)))
                if new_bid != bids[pid]:
                    adjustments[pid] += 1
                new_bids[pid] = new_bid
        prev_bids = dict(bids)
        bids = new_bids
        period_log.append({"period": period, "bids": dict(bids),
                            "info_accessed": {pid: info_access_counts[pid] for pid in triad_ids}})

    info_costs = {pid: info_cost * info_access_counts[pid] for pid in triad_ids}
    payoffs, efficiency = compute_match_payoffs(values, alpha, bids, adjustments, cost_c,
                                                 info_costs=info_costs)
    return {
        "triad": triad_ids, "alpha": alpha, "cost_c": cost_c, "info_cost": info_cost,
        "stop_period": stop_period, "values": values, "final_bids": dict(bids),
        "adjustments": adjustments, "info_access_counts": info_access_counts,
        "payoffs": payoffs, "allocative_efficiency": efficiency, "period_log": period_log,
    }


def _counterfactual_payoff(candidate_bid, others_prev_bids, own_value, alpha):
    all_bids = sorted(others_prev_bids + [candidate_bid], reverse=True)
    r = all_bids.index(candidate_bid) + 1
    if r == 1:
        price = sorted(others_prev_bids)[-1]
        return own_value - price
    elif r == 2:
        price = alpha * min(others_prev_bids)
        return alpha * own_value - price
    return 0.0


def run_study2_session(agents: dict, config: Study2Config = Study2Config()):
    rng = random.Random(config.seed)
    participant_ids = list(agents.keys())
    if len(participant_ids) != config.num_agents:
        raise ValueError("agents dict size must equal config.num_agents")

    matches = []
    for m in range(1, config.num_matches + 1):
        triads = make_triads(participant_ids, rng)
        stop_period = draw_termination_period(rng, min_periods=1, hazard=config.hazard,
                                                max_periods=config.max_periods)
        for triad in triads:
            record = run_study2_match(triad, agents, config.alpha, config.condition_c,
                                       config.info_cost, stop_period, rng)
            record["match_index"] = m
            matches.append(record)

    return {
        "study_id": "study_2_mechanism_experiment",
        "condition_c": config.condition_c,
        "info_cost": config.info_cost,
        "num_agents": config.num_agents,
        "num_matches": config.num_matches,
        "seed": config.seed,
        "matches": matches,
        "assumptions": [
            "value_good1 drawn Uniform{1,...,10} independently every match (evidence_label: derived)",
            "termination hazard p=0.1 with no minimum-period floor substitutes for the four unavailable "
            "Online Table N.1 duration sequences (evidence_label: missing, see audit/missing_information.json)",
            f"max_periods safety cap = {config.max_periods} applied to the theoretically unbounded process",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_smoke_test():
    ok = True
    messages = []

    agents1 = {i: SmokeTestAgent() for i in range(3)}
    r1 = run_study1_session(agents1, Study1Config(num_agents=3, num_matches=2,
                                                    min_periods=2, max_periods=5, seed=1))
    if len(r1["matches"]) != 2:
        ok = False
        messages.append("study1: expected 2 matches")
    for match in r1["matches"]:
        payoffs = match["payoffs"]
        if len(payoffs) != 3:
            ok = False
            messages.append("study1: expected 3 payoff records per match")
        if match["allocative_efficiency"] is None or not (0 <= match["allocative_efficiency"] <= 1.0001):
            ok = False
            messages.append("study1: allocative efficiency out of expected [0,1] range")

    agents2 = {i: SmokeTestAgent() for i in range(3)}
    r2 = run_study2_session(agents2, Study2Config(num_agents=3, num_matches=2,
                                                    max_periods=3, seed=2))
    if len(r2["matches"]) != 2:
        ok = False
        messages.append("study2: expected 2 matches")

    result = {
        "smoke_test": "PASSED" if ok else "FAILED",
        "messages": messages,
        "study1_sample_match": r1["matches"][0],
        "study2_sample_match": r2["matches"][0],
    }
    print(json.dumps(result, indent=2, default=str))
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run a minimal end-to-end self-test with a scripted "
                              "stand-in agent (not a valid research participant) "
                              "and print the result as JSON.")
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(_run_smoke_test())

    print(json.dumps({
        "status": "blocked",
        "reason": "No agent implementation supplied on the CLI. This adapter is a "
                   "library: import task.adapter and call run_study1_session / "
                   "run_study2_session with real ParticipantAgent subclasses backed "
                   "by genuine per-participant decision logic (see task/task.json "
                   "for the required observation/action contract). Use --smoke-test "
                   "to verify the simulation mechanics only.",
    }, indent=2))
    sys.exit(2)


if __name__ == "__main__":
    main()
