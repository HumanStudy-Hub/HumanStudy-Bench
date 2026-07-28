"""Source-grounded human response proportions for effects A, B, and C.

Every number here is transcribed from the published article that defines the
effect. Nothing is inferred, smoothed, or borrowed across studies. A state that
has no published proportion is marked uncalibrated and is excluded from
training rather than given a guessed label.

The calibration framing needs one number per bucket: the probability that a
human participant produced the *reference* behaviour named by that bucket.
`reference_semantic` names which behaviour the probability refers to, so the
consumer never has to guess the direction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple


class HumanPrior(NamedTuple):
    """One published human response proportion attached to a state bucket."""

    bucket: str
    probability: Optional[float]
    reference_semantic: str
    numerator: Optional[int]
    denominator: Optional[int]
    citation: str

    @property
    def calibrated(self) -> bool:
        return self.probability is not None


# --------------------------------------------------------------------------
# Effect A: Anderson, L. R., & Holt, C. A. (1997), American Economic Review
# 87(5), 847-862. Symmetric design only (six sessions, 540 decisions).
# --------------------------------------------------------------------------
#
# Bucket definitions, and why each proportion is the right one:
#
# cascade            The public history dominates a conflicting private draw,
#                    so the Bayesian choice is independent of the signal.
#                    "Cascade behavior was observed in 41 of the 56 periods in
#                    which such an imbalance occurred." (Results, p. 9)
#
# tie_conflict       Bayesian posterior is exactly 1/2 and the private draw
#                    disagrees with the previous public decision.
#                    "there were 68 instances in which the Bayes distribution
#                    was 1/2 and the private information did not match the
#                    label of the previous decision. In 57 of these 68 cases,
#                    the subject did not follow the previous decision."
#                    (Biases, pp. 16-17)
#
# tie_no_conflict    Posterior is 1/2 but the private draw agrees with the
#                    previous decision. The article reports no proportion for
#                    this configuration, so it stays uncalibrated.
#
# evidence_integration
#                    Neither a cascade nor a tie, so the Bayesian choice and
#                    the private draw coincide and a deviation contradicts
#                    both. "about 4 percent of the decisions were inconsistent
#                    with both Bayes' rule and private information." (p. 10)
#
# private_only       First position, no public history. Table 3 records 4
#                    errors in round 1 against a denominator of 90
#                    (6 sessions x 15 periods).
#
A_PRIORS: Mapping[str, HumanPrior] = {
    "cascade": HumanPrior(
        bucket="cascade",
        probability=41.0 / 56.0,
        reference_semantic="bayesian",
        numerator=41,
        denominator=56,
        citation="Anderson & Holt 1997, Results p. 9",
    ),
    "tie_conflict": HumanPrior(
        bucket="tie_conflict",
        probability=57.0 / 68.0,
        reference_semantic="private_signal",
        numerator=57,
        denominator=68,
        citation="Anderson & Holt 1997, Biases pp. 16-17",
    ),
    "tie_no_conflict": HumanPrior(
        bucket="tie_no_conflict",
        probability=None,
        reference_semantic="private_signal",
        numerator=None,
        denominator=None,
        citation="Anderson & Holt 1997 reports no proportion for this bucket",
    ),
    "evidence_integration": HumanPrior(
        bucket="evidence_integration",
        probability=0.96,
        reference_semantic="bayesian",
        numerator=None,
        denominator=540,
        citation="Anderson & Holt 1997, p. 10 (4% inconsistent with both)",
    ),
    "private_only": HumanPrior(
        bucket="private_only",
        probability=86.0 / 90.0,
        reference_semantic="bayesian",
        numerator=86,
        denominator=90,
        citation="Anderson & Holt 1997, Table 3 round 1 (4 errors of 90)",
    ),
}

# Reported in the article but deliberately not used as training targets,
# because each overlaps a bucket above and would double-count the same
# decisions. Kept for the cross-checks in `A_CROSS_CHECKS`.
A_CROSS_CHECKS: Mapping[str, Mapping[str, float]] = {
    "second_round_follows_private_when_conflicting": {
        "value": 0.95,
        "citation": "Anderson & Holt 1997, p. 14",
    },
    "cascade_formation_rate_all_twelve_sessions": {
        "value": 87.0 / 122.0,
        "citation": "Anderson & Holt 1997, Table 7",
    },
    "deviations_from_cascade_with_opposing_private_draw": {
        "value": 15.0 / 16.0,
        "citation": "Anderson & Holt 1997, p. 15",
    },
}

A_ROUND_DEVIATIONS: Mapping[int, int] = {1: 4, 2: 3, 3: 6, 4: 14, 5: 13, 6: 7}
A_ROUND_DENOMINATOR = 90


# --------------------------------------------------------------------------
# Effect B: Jaquiery, M., & Yeung, N. (2024), PLOS ONE 19(9), e0311211.
# Pick rates are the proportion of test-block choices that went to the
# advisor named by `reference_semantic`, averaged over participants.
# --------------------------------------------------------------------------
class BExperiment(NamedTuple):
    experiment: str
    task: str
    contrast: str
    feedback: bool
    pick_rate: float
    ci95: Sequence[float]
    n: int
    reference_semantic: str


B_EXPERIMENTS: Mapping[str, BExperiment] = {
    "1A": BExperiment("1A", "dots", "accuracy", False, 0.57, (0.52, 0.61), 50, "accurate"),
    "1B": BExperiment("1B", "dates", "accuracy", False, 0.45, (0.33, 0.57), 28, "accurate"),
    "1C": BExperiment("1C", "dates", "accuracy", True, 0.67, (0.57, 0.78), 34, "accurate"),
    "2A": BExperiment("2A", "dots", "agreement", False, 0.61, (0.57, 0.65), 50, "agreeing"),
    "2B": BExperiment("2B", "dates", "agreement", False, 0.63, (0.53, 0.73), 35, "agreeing"),
    "2C": BExperiment("2C", "dates", "agreement", True, 0.52, (0.42, 0.62), 39, "agreeing"),
    "3A": BExperiment("3A", "dots", "accuracy_vs_agreement", False, 0.46, (0.38, 0.54), 64, "agreeing"),
    "3B": BExperiment("3B", "dates", "accuracy_vs_agreement", False, 0.51, (0.39, 0.64), 29, "agreeing"),
    "3C": BExperiment("3C", "dates", "accuracy_vs_agreement", True, 0.17, (0.08, 0.25), 31, "agreeing"),
}

B_DEFAULT_EXPERIMENT = "3C"


def b_prior(experiment: str = B_DEFAULT_EXPERIMENT) -> HumanPrior:
    """Return the accurate-advisor pick probability for one B experiment."""

    if experiment not in B_EXPERIMENTS:
        raise ValueError(
            "unknown B experiment {!r}; expected one of {}".format(
                experiment,
                ", ".join(sorted(B_EXPERIMENTS)),
            )
        )
    record = B_EXPERIMENTS[experiment]
    # Normalize every experiment onto the accurate advisor so that one code
    # path serves all nine, regardless of which advisor the paper scored.
    probability = (
        record.pick_rate
        if record.reference_semantic == "accurate"
        else 1.0 - record.pick_rate
    )
    return HumanPrior(
        bucket="jaquiery_yeung_{}".format(experiment.lower()),
        probability=probability,
        reference_semantic="accurate",
        numerator=None,
        denominator=record.n,
        citation="Jaquiery & Yeung 2024, Experiment {} advisor choice".format(experiment),
    )


# --------------------------------------------------------------------------
# Effect C: Schoebel, Rieskamp, & Huber (2016), PLOS ONE 11(1), e0146536.
# Proportions are per scenario and come from the public Figshare workbook via
# `study_019/source/materials/scenarios.json`, so no table is duplicated here.
# --------------------------------------------------------------------------
C_HUMAN_N = 40
C_CITATION = "Schoebel, Rieskamp & Huber 2016, Study 2 raw data (Figshare 1597662)"


def _binomial_absolute_deviation(probability: float, count: int) -> Tuple[float, float]:
    """Exact mean and variance of |X/n - p| for X ~ Binomial(n, p)."""

    log_choose = 0.0
    first = 0.0
    second = 0.0
    for successes in range(count + 1):
        if successes:
            log_choose += math.log((count - successes + 1) / successes)
        log_pmf = log_choose
        if probability <= 0.0:
            log_pmf = 0.0 if successes == 0 else float("-inf")
        elif probability >= 1.0:
            log_pmf = 0.0 if successes == count else float("-inf")
        else:
            log_pmf += successes * math.log(probability)
            log_pmf += (count - successes) * math.log1p(-probability)
        if log_pmf == float("-inf"):
            continue
        mass = math.exp(log_pmf)
        deviation = abs(successes / count - probability)
        first += mass * deviation
        second += mass * deviation * deviation
    return first, max(second - first * first, 0.0)


def binomial_noise_floor(
    proportions: Sequence[float],
    counts: Sequence[int],
    **_ignored: Any,
) -> Dict[str, float]:
    """Mean absolute error a *perfect* model still incurs from sampling noise.

    Human proportions are estimated from finitely many participants, so the
    observed rate differs from the underlying rate. A model reproducing the
    underlying rate exactly would still show non-zero MAE against the observed
    rates, and every reported MAE has to be read against that floor.

    Computed exactly by summing over the binomial support rather than by
    simulation: it is both faster and free of Monte Carlo error, which matters
    because the floor is quoted as a fixed reference line in every table.
    """

    if len(proportions) != len(counts):
        raise ValueError("proportions and counts must have equal length")
    if not proportions:
        raise ValueError("cannot compute a noise floor for an empty set")

    means: List[float] = []
    variances: List[float] = []
    for probability, count in zip(proportions, counts):
        if count <= 0:
            raise ValueError("participant counts must be positive")
        mean_value, variance = _binomial_absolute_deviation(float(probability), int(count))
        means.append(mean_value)
        variances.append(variance)

    scenarios = len(means)
    expected = sum(means) / scenarios
    # The reported floor is a mean over independent scenarios, so its own
    # sampling spread shrinks with the number of scenarios.
    standard_error = math.sqrt(sum(variances)) / scenarios
    return {
        "mean": expected,
        "standard_error": standard_error,
        "ci95_low": max(expected - 1.96 * standard_error, 0.0),
        "ci95_high": expected + 1.96 * standard_error,
        "scenarios": float(scenarios),
    }


def trivial_baselines(proportions: Sequence[float]) -> Dict[str, float]:
    """MAE of two models that ignore the prompt, for scale."""

    if not proportions:
        raise ValueError("cannot compute baselines for an empty set")
    grand_mean = sum(proportions) / len(proportions)
    return {
        "always_half": sum(abs(value - 0.5) for value in proportions) / len(proportions),
        "always_grand_mean": sum(
            abs(value - grand_mean) for value in proportions
        ) / len(proportions),
        "grand_mean": grand_mean,
    }


def logit(probability: float) -> float:
    """Log odds, clamped so that certainty does not produce an infinity."""

    clamped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clamped / (1.0 - clamped))


def dpo_reachable(human_probability: float, base_probability: float) -> bool:
    """Whether proportional DPO can reach a target from a given base model.

    At the population optimum of the DPO loss,

        logit(p_model) = logit(p_base) + logit(p_human) / beta

    so hitting `p_human` needs beta = logit(p_h) / (logit(p_h) - logit(p_base)),
    and beta must be positive. That holds only when the base model sits closer
    to 0.5 than the human proportion does. When the base model already
    overshoots in the same direction, every positive beta pushes it further
    away: proportional DPO can sharpen a distribution but never soften one.
    """

    target = logit(human_probability)
    base = logit(base_probability)
    denominator = target - base
    if denominator == 0.0:
        return True
    return (target / denominator) > 0.0


def dpo_optimal_beta(
    human_probability: float,
    base_probability: float,
) -> Optional[float]:
    """The beta that lands exactly on the human proportion, if one exists."""

    target = logit(human_probability)
    base = logit(base_probability)
    denominator = target - base
    if denominator == 0.0:
        return None
    beta = target / denominator
    return beta if beta > 0.0 else None
