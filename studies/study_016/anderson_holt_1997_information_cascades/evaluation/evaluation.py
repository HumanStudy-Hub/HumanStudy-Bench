#!/usr/bin/env python3
"""
Evaluation for Anderson & Holt (1997), "Information Cascades in the Laboratory".

Consumes session logs produced by task/adapter.py's run_session() (schema
documented in task/task.json under "outputs_consumed_by_evaluation") and
computes the paper's core behavioral measures:

  1. Cascade formation / reverse-cascade classification per period (Section
     III, Table 7), using the paper's own "Bayes distribution" method: later
     rounds' relevant signals are inferred from EARLIER PUBLIC DECISIONS only
     (never from another round's hidden private draw), exactly as a real
     subsequent decision-maker would have to do.
  2. Actual / private-information / optimal / random expected-payoff
     efficiency (Section III, equations for "actual efficiency" and "private
     information efficiency"), and counting efficiency for the asymmetric
     design (Section V.B).
  3. Status-quo bias check (Section V.A) and representativeness bias check
     (Section V.A, public-draw condition only).
  4. Counting-heuristic-vs-Bayes' rule check (Section V.B, asymmetric design
     only).

Each check reports both the agent's own numbers and, for descriptive
comparison only, the corresponding human benchmark transcribed verbatim from
the paper (source/evidence.json). This module does NOT attempt to reproduce
the paper's Table 3 round-by-round logit error-rate estimates (see
`logit_error_analysis()` below), because that requires many repeated
independent replications per round and an MLE fit the paper performed with a
proprietary GAUSS routine on human subjects' raw per-decision data, which is
not published in the paper (see audit/missing_information.json) and would not
be statistically meaningful on a handful of agent-generated sessions. That
check returns a structured "not_ready" result instead of a fabricated number.

No network access is used or required.
"""
import json
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Design parameters (must match task/adapter.py's CONDITIONS)
# ---------------------------------------------------------------------------

DESIGNS = {
    "study_1_symmetric": {
        "p_a_given_A": 2 / 3, "p_b_given_A": 1 / 3,
        "p_a_given_B": 1 / 3, "p_b_given_B": 2 / 3,
    },
    "study_2_asymmetric": {
        "p_a_given_A": 6 / 7, "p_b_given_A": 1 / 7,
        "p_a_given_B": 5 / 7, "p_b_given_B": 2 / 7,
    },
}

PAYOFF_CORRECT_USD = 2
PAYOFF_RANDOM_USD = 1.0  # 0.5 * PAYOFF_CORRECT_USD, expected payoff of a coin flip

_SIGNAL_OF = {"light": "a", "dark": "b"}


def posterior_A(design, n_a, n_b):
    """Bayesian posterior probability of urn A given n_a relevant a-signals and n_b relevant b-signals."""
    lr_a = design["p_a_given_A"] / design["p_a_given_B"]
    lr_b = design["p_b_given_A"] / design["p_b_given_B"]
    odds = (lr_a ** n_a) * (lr_b ** n_b)  # prior odds are 1:1 (P(A)=P(B)=0.5), so posterior odds = likelihood ratio
    return odds / (odds + 1.0)


def is_cascade_state(design, n_a, n_b):
    """True iff the Bayes-optimal decision for a hypothetical next round is the same
    whether that round's own signal turns out to be 'a' or 'b' -- i.e. a cascade is
    already in force given tally (n_a, n_b)."""
    p_if_a = posterior_A(design, n_a + 1, n_b)
    p_if_b = posterior_A(design, n_a, n_b + 1)
    dec_if_a = "A" if p_if_a >= 0.5 else "B"
    dec_if_b = "A" if p_if_b >= 0.5 else "B"
    if dec_if_a == dec_if_b:
        return True, dec_if_a
    return False, None


def _bayes_optimal_decision(design, n_a, n_b, own_signal):
    n_a2, n_b2 = (n_a + 1, n_b) if own_signal == "a" else (n_a, n_b + 1)
    p = posterior_A(design, n_a2, n_b2)
    if p > 0.5:
        return "A", p
    if p < 0.5:
        return "B", p
    return ("A" if own_signal == "a" else "B"), p  # tie-break: footnote 10, match own signal


def analyze_period(study, period):
    """Replays one period's rounds, inferring the 'Bayes distribution' tally from
    public decisions only (Section III methodology), and returns a per-round and
    per-period analysis."""
    design = DESIGNS[study]
    n_a, n_b = 0, 0
    cascade_active, cascade_dir = False, None
    round_analyses = []

    for r in period["rounds"]:
        own_signal = _SIGNAL_OF[r["private_draw"]]
        pre_cascade_active, pre_cascade_dir = is_cascade_state(design, n_a, n_b)
        optimal_decision, posterior_for_A = _bayes_optimal_decision(design, n_a, n_b, own_signal)
        actual_decision = r["decision"]
        actual_posterior = posterior_for_A if actual_decision == "A" else (1 - posterior_for_A)
        signal_only_decision = "A" if own_signal == "a" else "B"

        round_analyses.append({
            "round": r["round"],
            "pre_round_relevant_tally": {"a": n_a, "b": n_b},
            "cascade_in_force_pre_round": pre_cascade_active,
            "own_signal": own_signal,
            "optimal_decision": optimal_decision,
            "actual_decision": actual_decision,
            "matches_optimal": actual_decision == optimal_decision,
            "matches_private_signal_only": actual_decision == signal_only_decision,
            "bayes_inconsistent_with_private_info": pre_cascade_active and optimal_decision != signal_only_decision,
            "expected_payoff_actual_usd": actual_posterior * PAYOFF_CORRECT_USD,
            "expected_payoff_optimal_usd": max(posterior_for_A, 1 - posterior_for_A) * PAYOFF_CORRECT_USD,
            "expected_payoff_private_info_only_usd": (
                max(posterior_A(design, 1, 0), 1 - posterior_A(design, 1, 0)) * PAYOFF_CORRECT_USD
                if own_signal == "a" else
                max(posterior_A(design, 0, 1), 1 - posterior_A(design, 0, 1)) * PAYOFF_CORRECT_USD
            ),
        })

        # Update the inferred relevant tally exactly as a subsequent rational
        # decision-maker would, using ONLY the announced decision.
        if cascade_active:
            if actual_decision != cascade_dir:
                # deviation: reveals a signal favoring the non-cascade urn
                if cascade_dir == "A":
                    n_b += 1
                else:
                    n_a += 1
                cascade_active, cascade_dir = is_cascade_state(design, n_a, n_b)
            # else: decision matches the cascade, adds no new information
        else:
            inferred_signal = "a" if actual_decision == "A" else "b"
            if inferred_signal == "a":
                n_a += 1
            else:
                n_b += 1
            cascade_active, cascade_dir = is_cascade_state(design, n_a, n_b)

    cascade_possible = any(ra["cascade_in_force_pre_round"] for ra in round_analyses)
    cascade_formed = cascade_possible and all(
        ra["matches_optimal"] for ra in round_analyses if ra["cascade_in_force_pre_round"]
    )
    reverse_cascade = False
    if cascade_formed:
        cascade_rounds = [ra for ra in round_analyses if ra["cascade_in_force_pre_round"]]
        reverse_cascade = cascade_rounds[0]["optimal_decision"] != period["true_urn"]

    return {
        "period": period["period"],
        "true_urn": period["true_urn"],
        "cascade_possible": cascade_possible,
        "cascade_formed": cascade_formed,
        "reverse_cascade": reverse_cascade,
        "rounds": round_analyses,
    }


# ---------------------------------------------------------------------------
# Aggregate checks
# ---------------------------------------------------------------------------

def cascade_summary(study, period_analyses):
    normal_formed = sum(1 for p in period_analyses if p["cascade_formed"] and not p["reverse_cascade"])
    reverse_formed = sum(1 for p in period_analyses if p["cascade_formed"] and p["reverse_cascade"])
    possible_not_formed = sum(1 for p in period_analyses if p["cascade_possible"] and not p["cascade_formed"])
    n_possible = sum(1 for p in period_analyses if p["cascade_possible"])

    benchmark = {
        "study_1_symmetric": {"cascades_formed": 41, "periods_possible": 56, "text": "41 of 56 possible periods (Section III); Table 7: 28 normal + 13 reverse formed, 10 normal + 5 reverse possible-but-not-formed."},
        "study_2_asymmetric": {"cascades_formed": 46, "periods_possible": 66, "text": "46 of 66 possible periods (Section V.B); Table 7: 28 normal + 18 reverse formed, 12 normal + 8 reverse possible-but-not-formed."},
    }[study]

    return {
        "n_periods_analyzed": len(period_analyses),
        "n_periods_cascade_possible": n_possible,
        "n_cascades_formed_normal": normal_formed,
        "n_cascades_formed_reverse": reverse_formed,
        "n_cascade_possible_but_not_formed": possible_not_formed,
        "cascade_formation_rate": (normal_formed + reverse_formed) / n_possible if n_possible else None,
        "human_benchmark": benchmark,
    }


def efficiency_summary(study, period_analyses):
    pi_actual = pi_optimal = pi_random_total = pi_private = 0.0
    n_rounds = 0
    for p in period_analyses:
        for ra in p["rounds"]:
            pi_actual += ra["expected_payoff_actual_usd"]
            pi_optimal += ra["expected_payoff_optimal_usd"]
            pi_private += ra["expected_payoff_private_info_only_usd"]
            pi_random_total += PAYOFF_RANDOM_USD
            n_rounds += 1

    denom = pi_optimal - pi_random_total
    actual_efficiency = 100 * (pi_actual - pi_random_total) / denom if denom else None
    private_info_efficiency = 100 * (pi_private - pi_random_total) / denom if denom else None

    benchmark = {
        "study_1_symmetric": {"actual_efficiency_pct": 91.4, "private_information_efficiency_pct": 72.1,
                               "text": "Averaged over all 36 symmetric-design subjects (Section III)."},
        "study_2_asymmetric": {"actual_efficiency_pct": 67.6, "private_information_efficiency_pct": 45.2,
                                "text": "Averaged over all 36 asymmetric-design subjects (Section V.B)."},
    }[study]

    return {
        "n_rounds": n_rounds,
        "sum_expected_payoff_actual_usd": round(pi_actual, 4),
        "sum_expected_payoff_optimal_usd": round(pi_optimal, 4),
        "sum_expected_payoff_private_info_only_usd": round(pi_private, 4),
        "sum_expected_payoff_random_usd": round(pi_random_total, 4),
        "actual_efficiency_pct": round(actual_efficiency, 2) if actual_efficiency is not None else None,
        "private_information_efficiency_pct": round(private_info_efficiency, 2) if private_info_efficiency is not None else None,
        "human_benchmark": benchmark,
    }


def status_quo_bias_check(period_analyses):
    """Section V.A: instances where the pre-round Bayes-distribution posterior for A
    was exactly 1/2 and the subject's own signal did NOT match the immediately
    preceding announced decision. Checks whether the agent followed its own signal
    (rational under the paper's null finding) or the preceding decision (status-quo bias)."""
    n_instances = 0
    n_followed_own_signal = 0
    for p in period_analyses:
        prev_decision = None
        for ra in p["rounds"]:
            n_a, n_b = ra["pre_round_relevant_tally"]["a"], ra["pre_round_relevant_tally"]["b"]
            is_tie = n_a == n_b  # posterior 1/2 for A pre-round tally, symmetric design property
            own_label = "A" if ra["own_signal"] == "a" else "B"
            if prev_decision is not None and is_tie and own_label != prev_decision:
                n_instances += 1
                if ra["actual_decision"] == own_label:
                    n_followed_own_signal += 1
            prev_decision = ra["actual_decision"]
    return {
        "n_qualifying_instances": n_instances,
        "n_followed_own_private_signal_rather_than_status_quo": n_followed_own_signal,
        "share_followed_own_signal": (n_followed_own_signal / n_instances) if n_instances else None,
        "human_benchmark": {
            "n_qualifying_instances": 68, "n_followed_own_signal": 57,
            "share": round(57 / 68, 4),
            "text": "57 of 68 qualifying instances, symmetric design, all 6 sessions (Section V.A). Paper concludes there is no detectable status-quo bias beyond Bayesian updating.",
        },
    }


def counting_heuristic_check(study, period_analyses):
    """Section V.B (asymmetric design only): among rounds where the plain
    signal-count rule and Bayes' rule prescribe different decisions, does the agent
    follow Bayes' rule (correct) or the counting heuristic?"""
    if study != "study_2_asymmetric":
        return {"applicable": False, "note": "Counting and Bayes' rule always coincide in the symmetric design (paper, Section V.B)."}

    n_disagree = 0
    n_bayes_correct_when_disagree = 0
    for p in period_analyses:
        for ra in p["rounds"]:
            n_a, n_b = ra["pre_round_relevant_tally"]["a"], ra["pre_round_relevant_tally"]["b"]
            own = ra["own_signal"]
            total_a = n_a + (1 if own == "a" else 0)
            total_b = n_b + (1 if own == "b" else 0)
            counting_decision = "A" if total_a > total_b else ("B" if total_b > total_a else ("A" if own == "a" else "B"))
            if counting_decision != ra["optimal_decision"]:
                n_disagree += 1
                if ra["actual_decision"] == ra["optimal_decision"]:
                    n_bayes_correct_when_disagree += 1
    return {
        "applicable": True,
        "n_rounds_bayes_and_counting_disagree": n_disagree,
        "n_bayes_correct_decisions_when_disagree": n_bayes_correct_when_disagree,
        "share_bayes_correct_when_disagree": (n_bayes_correct_when_disagree / n_disagree) if n_disagree else None,
        "human_benchmark": {
            "n_disagree": 82, "n_bayes_correct": 41, "share": 0.5,
            "text": "41 of 82 cases (50%) where Bayes' rule and counting disagree, subjects made the Bayes-correct decision (Section V.B).",
        },
    }


def logit_error_analysis():
    """Table 3's round-by-round logit beta estimates were fit on the ORIGINAL HUMAN
    subjects' raw per-decision data using a recursive Newton-Raphson MLE in GAUSS.
    That raw per-subject dataset is not published in the paper (it states a 'complete
    data appendix is available from the authors on request', footnote 18) and is not
    included in this package (see audit/missing_information.json). Refitting an
    equivalent model on a handful of agent-generated sessions would not be a faithful
    replication of Table 3 and could produce a misleadingly precise-looking number
    from a tiny, non-independent sample. This check is therefore reported as not_ready."""
    return {
        "status": "not_ready",
        "check": "round_by_round_logit_error_rate_estimation (paper Table 3, Section IV)",
        "missing_requirement": (
            "The original per-subject, per-round raw decision/draw dataset for the six "
            "symmetric-design sessions (540 decisions), used by the authors to fit a "
            "recursive logit model in GAUSS. The paper reports only the fitted "
            "coefficients (Table 3), not the underlying per-decision data, and states "
            "the full data appendix is available from the authors on request -- which "
            "this job was not authorized to seek out (no external URL was supplied)."
        ),
        "would_require": "A researcher-supplied raw dataset (or a much larger bank of independent agent-run sessions, e.g. dozens per round) plus a maximum-likelihood logit fitting routine (e.g. via statsmodels/scipy), and a decision about whether to replicate the paper's exact recursive round-by-round procedure or a simpler pooled logit.",
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def evaluate(session_logs):
    """session_logs: list of dicts matching task/task.json's session_log_schema."""
    by_study = defaultdict(list)
    for session in session_logs:
        by_study[session["study"]].append(session)

    report = {}
    for study, sessions in by_study.items():
        all_period_analyses = []
        for session in sessions:
            all_period_analyses.extend(analyze_period(study, p) for p in session["periods"])

        study_report = {
            "n_sessions": len(sessions),
            "cascade_summary": cascade_summary(study, all_period_analyses),
            "efficiency_summary": efficiency_summary(study, all_period_analyses),
            "counting_heuristic_check": counting_heuristic_check(study, all_period_analyses),
        }
        if study == "study_1_symmetric":
            study_report["status_quo_bias_check"] = status_quo_bias_check(all_period_analyses)
        report[study] = study_report

    report["logit_error_analysis"] = logit_error_analysis()
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-test":
        # Exercise the evaluator end-to-end on adapter.py's own smoke-test sessions,
        # with no network access and no external agent required.
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "task"))
        import adapter  # noqa: E402

        sessions = []
        for (study, condition) in adapter.CONDITIONS:
            sessions.append(adapter.run_session(
                adapter._baseline_follow_private_signal,
                study, condition, n_periods=5, seed=42,
            ))
        print(json.dumps(evaluate(sessions), indent=2))
        return 0

    data = json.load(sys.stdin)
    print(json.dumps(evaluate(data), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
