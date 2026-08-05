#!/usr/bin/env python3
"""
Adapter for Anderson & Holt (1997), "Information Cascades in the Laboratory".

Reconstructs the sequential urn-prediction task described in
materials/materials.json: each period, a die roll (simulated) picks urn A or
urn B; six decision-maker "seats" are approached in a random order; each seat
sees one private ball draw (with replacement) and the public decisions
announced so far this period, then announces its own decision (A or B).

This module never makes network calls. It exposes:
  - run_session(agent_fn, study, condition, n_periods=15, seed=None)
  - run_smoke_test() -> dict   (uses a built-in baseline policy, no agent_fn needed)
  - CLI: `python adapter.py --smoke-test`

`agent_fn` is supplied by the harness running the model under test. It must be
a callable: agent_fn(round_input: dict) -> {"decision": "A"|"B"}. See
task/task.json ("per_round_interaction") for the exact schema of round_input.

If a study/condition combination requires a researcher decision that has not
been made (see audit/missing_information.json), run_session raises
AdapterBlocked with a clear message rather than guessing.
"""
import argparse
import json
import os
import random
import sys

CONDITIONS = {
    ("study_1_symmetric", "baseline_private_draws_only"): {
        "urn_A": {"a": 2, "b": 1, "die_rolls": (1, 2, 3)},
        "urn_B": {"a": 1, "b": 2, "die_rolls": (4, 5, 6)},
        "public_draws_after_round": None,
    },
    ("study_1_symmetric", "public_draws_after_round4"): {
        "urn_A": {"a": 2, "b": 1, "die_rolls": (1, 2, 3)},
        "urn_B": {"a": 1, "b": 2, "die_rolls": (4, 5, 6)},
        "public_draws_after_round": 4,
        "n_public_draws": 2,
    },
    ("study_2_asymmetric", "asymmetric_urns"): {
        "urn_A": {"a": 6, "b": 1, "die_rolls": (1, 2, 3)},
        "urn_B": {"a": 5, "b": 2, "die_rolls": (4, 5, 6)},
        "public_draws_after_round": None,
    },
}

N_SEATS_PER_SESSION = 6
N_PAID_PERIODS_DEFAULT = 15
PAYOFF_CORRECT_USD = 2
PAYOFF_INCORRECT_USD = 0


class AdapterBlocked(Exception):
    """Raised when faithful execution requires a missing researcher decision."""


def _draw_ball(urn, rng):
    total = urn["a"] + urn["b"]
    return "light" if rng.random() < urn["a"] / total else "dark"


def _roll_urn(config, rng):
    roll = rng.randint(1, 6)
    if roll in config["urn_A"]["die_rolls"]:
        return "A", config["urn_A"]
    return "B", config["urn_B"]


def _load_materials():
    materials_path = os.path.join(os.path.dirname(__file__), "..", "materials", "materials.json")
    with open(materials_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _instructions_text(study, materials):
    if study == "study_1_symmetric":
        return materials["study_1_symmetric"]["instructions_full_text"]["text"]
    if study == "study_2_asymmetric":
        return materials["study_2_asymmetric"]["instructions_full_text"]["text"]
    raise AdapterBlocked(f"Unknown study '{study}'")


def run_period(agent_fn, study, condition, config, period_index, rng, instructions_text):
    true_urn_label, urn = _roll_urn(config, rng)
    seat_order = list(range(1, N_SEATS_PER_SESSION + 1))
    rng.shuffle(seat_order)

    rounds_log = []
    decisions_so_far = []
    public_draws_after = config.get("public_draws_after_round")
    n_public_draws = config.get("n_public_draws", 0)
    inserted_public_draws = None

    for round_number in range(1, N_SEATS_PER_SESSION + 1):
        private_draw = _draw_ball(urn, rng)

        if public_draws_after and round_number == public_draws_after + 1 and inserted_public_draws is None:
            inserted_public_draws = [_draw_ball(urn, rng) for _ in range(n_public_draws)]

        extra_public_draws = inserted_public_draws if (public_draws_after and round_number > public_draws_after) else None

        round_input = {
            "study": study,
            "condition": condition,
            "instructions_text": instructions_text,
            "period": period_index,
            "round": round_number,
            "seat": seat_order[round_number - 1],
            "private_draw": private_draw,
            "decisions_so_far_this_period": list(decisions_so_far),
            "extra_public_draws": extra_public_draws,
        }

        result = agent_fn(round_input)
        decision = result.get("decision") if isinstance(result, dict) else result
        if decision not in ("A", "B"):
            raise ValueError(f"agent_fn returned invalid decision {decision!r}; expected 'A' or 'B'")

        correct = decision == true_urn_label
        decisions_so_far.append({"round": round_number, "decision": decision})
        rounds_log.append({
            "round": round_number,
            "private_draw": private_draw,
            "decision": decision,
            "correct": correct,
            "payoff_usd": PAYOFF_CORRECT_USD if correct else PAYOFF_INCORRECT_USD,
        })

    return {
        "period": period_index,
        "true_urn": true_urn_label,
        "public_draws_inserted": inserted_public_draws,
        "rounds": rounds_log,
    }


def run_session(agent_fn, study, condition, n_periods=N_PAID_PERIODS_DEFAULT, seed=None, session_id=None):
    key = (study, condition)
    if key not in CONDITIONS:
        raise AdapterBlocked(
            f"No configuration for study={study!r} condition={condition!r}. "
            f"Valid combinations: {sorted(CONDITIONS.keys())}"
        )
    config = CONDITIONS[key]
    materials = _load_materials()
    instructions_text = _instructions_text(study, materials)

    rng = random.Random(seed)
    periods = [
        run_period(agent_fn, study, condition, config, p, rng, instructions_text)
        for p in range(1, n_periods + 1)
    ]
    return {
        "study": study,
        "condition": condition,
        "session_id": session_id or f"{study}:{condition}:seed={seed}",
        "periods": periods,
    }


# ---------------------------------------------------------------------------
# Built-in baseline policies, used only for --smoke-test (no external agent
# required). These are NOT claims about human behavior; they only exist to
# exercise the simulation mechanics end-to-end.
# ---------------------------------------------------------------------------

def _baseline_follow_private_signal(round_input):
    """Always predicts the urn matching the subject's own private draw, ignoring public history."""
    return {"decision": "A" if round_input["private_draw"] == "light" else "B"}


def run_smoke_test():
    results = {}
    for (study, condition) in CONDITIONS:
        session = run_session(
            _baseline_follow_private_signal,
            study,
            condition,
            n_periods=3,
            seed=12345,
            session_id=f"smoke:{study}:{condition}",
        )
        n_rounds = sum(len(p["rounds"]) for p in session["periods"])
        n_correct = sum(r["correct"] for p in session["periods"] for r in p["rounds"])
        results[f"{study}::{condition}"] = {
            "n_periods": len(session["periods"]),
            "n_rounds": n_rounds,
            "n_correct": n_correct,
            "sample_period_0": session["periods"][0],
        }
    return {"status": "ok", "policy": "baseline_follow_private_signal", "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Run a network-free smoke test of the simulation mechanics and exit.")
    parser.add_argument("--study", default=None, help="Study key, e.g. study_1_symmetric")
    parser.add_argument("--condition", default=None, help="Condition key, e.g. baseline_private_draws_only")
    parser.add_argument("--periods", type=int, default=N_PAID_PERIODS_DEFAULT)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        print(json.dumps(run_smoke_test(), indent=2))
        return 0

    if args.study and args.condition:
        session = run_session(_baseline_follow_private_signal, args.study, args.condition, n_periods=args.periods, seed=args.seed)
        print(json.dumps(session, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
