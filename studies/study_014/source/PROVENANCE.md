# Ground-truth source audit

The study follows one paper: Davis et al. (2011), DOI 10.1287/mnsc.1100.1258.
Its printed regression is the only human result used in scoring. The original
copyrighted PDF remains a source key only; its exact hash is in SOURCE.json.

A second document is declared, Feng et al. (2025), DOI
10.18653/v1/2025.emnlp-main.391 (included, CC BY 4.0). It is a later paper that
compares LLM agents with a human benchmark on this task. It is kept for
reference ("see also") and because some retained legacy aggregates are printed
there. It is not used in scoring.

The retained legacy aggregates (`original_data_points`) are not a set of
numbers extracted from the 2011 paper. Every one has a JSON-pointer entry in
ground_truth.json's value_provenance. Exact printed values, differences from
printed values, lower-precision figure labels, and values not located in either
document are explicitly distinguished. No expert aggregate has been deleted or
silently corrected. University-of-Michigan/2023 attribution is retained as
legacy attribution, not independently established from inspected replication
data. None of these aggregates is scored.

Davis Table 2 (page 182) reports the original bidder-count coefficient 2.98
with a significance marker defined as p < .01. It does not establish the legacy
p < .001 claim. Feng Table 2 (page 7698) prints human r = .42, mean 27.31, SD
23.22, entropy 5.08, sell-through .76 and profit 32.83, without a Pearson
p-value. The legacy SD 23.23 and sell-through .74 remain preserved and flagged.
Feng Figure 2 prints bidder-group means at one decimal place; it does not
establish all retained two-decimal values. Human sample size 40 is printed in
Feng Figure 1A (page 7694, "40 Humans"), so it is labelled reported; it
describes the later paper's human benchmark, not the 2011 sample, which Davis
Table 1 (page 181) prints as 120 seller subjects, 20 per treatment cell. The
total of 2400 observations is derived as 40 x 60 printed rounds.

The 2011 paper's own per-n means are stored separately, unscored, under
`original_paper_descriptives`: Davis Figure 2(a) (page 182) prints NoInfo
Cuberoot bars of 19.2, 23.4, 27.0, 29.5, 38.7 and 48.2 for n = 1, 2, 3, 4, 7
and 10. The figure does not say whether n = 1 and n = 4 pool the two bidder
sets.

Evidence labels are kept true: `missing` is used only for a gap with no value
(the Pearson p-value). Retained expert values that could not be located in
either document carry `reported_from: "hsb_legacy_expert_annotation"` and an
empty printed_in list. For these records, `reported` means reported by the
legacy benchmark annotation, not by either paper. Their status is
`unverified_legacy`, or `differs_from_printed` for the two legacy values
(SD 23.23, sell-through .74) that conflict with the values Feng Table 2 prints.

## Replication scoring: Davis et al. (2011) only

Scoring rule: a study follows its paper and is scored the way that paper tests
its finding. The rule is recorded in the finding's
`scoring` block in ground_truth.json.

**Human result.** Davis et al. (2011), Table 2, Model (1), printed page 182.
The paper ran "all regressions with random effects for individual subjects and
used the observed reserve price in a period as the dependent variable" (page
181). Model (1), which the paper calls the baseline model, regresses the
reserve price on n - mean(n), Per - mean(Per) and a Cube indicator. The
coefficient on n - mean(n) is 2.98 [SE 0.08], marked with an asterisk that the
table note defines as p < 0.01; the paper reads it as "sellers in the Cuberoot
treatment increase the reservation price by 2.98 for each additional bidder"
(page 181). Only the bound is printed, so `human_p_value` stays empty,
`human_p_value_bound` is 0.01, and the human result counts as significant
because the bound is below 0.05. The other Model (1) rows are stored as
printed: constant 29.45 [1.83], period 0.04 [0.011], Cube 31.78 [2.59],
R-squared 0.499. Model (2) keeps 2.98; Model (3) adds an n x Cube interaction
and prints 3.24 [0.11] on n - mean(n). Those are recorded and not scored.

**Agent analysis.** The agent decisions are fitted the same way, as far as the
agent data allow: reserve price on n - mean(n) and period - mean(period), with
a random intercept per agent (one agent = one `individual_data` entry). The
estimator is random-intercept GLS with Swamy-Arora variance components and
conventional standard errors (z-test), written in numpy in
`scripts/evaluator.py`, because the study's `stats_lib.py` has no panel
regression and statsmodels is not a dependency. What differs from Davis:

- No Cube indicator. The agent task replays only the Cuberoot distribution, so
  Cube is constant and is dropped.
- Bidder set. The agent sees n in {1, 4, 7, 10} only. Davis Model (1) pools the
  NoInfo n in {1, 2, 3, 4} and n in {1, 4, 7, 10} sets and both distributions
  (Table 1: 80 seller subjects, 20 per cell, 60 decisions each).
- Davis does not name the random-effects estimator. Swamy-Arora is the default
  random-effects estimator in Stata's `xtreg`. The Hausman test of footnote 5 is
  not run.
- n and period are centered on the agent sample's own means. Centering does not
  change the slope.
- With fewer than 4 agents the between-subject variance cannot be estimated, so
  the model is not fitted and the row is not scored.

**Score.** `replication` is true when the human result is significant (Davis
bound p < 0.01), the agent coefficient on n - mean(n) is significant at 0.05,
and it is positive, the sign of the printed 2.98. The effect comparison is the
agent coefficient (labelled `computed`) against 2.98 (labelled `printed`),
both in reserve-price points per additional bidder: `human_effect_size`,
`agent_effect_size`, `effect_gap` and `agent_coefficient_ci95`. No Cohen's d is
derived; the paper prints a regression coefficient, and the comparison stays
in its units.

**Running it.** The evaluator expects one response per round, with
`trial_info.num_bidders` (and `round` when available), one `individual_data`
entry per agent. The shipped 60-round baselines in `benchmark/llm_data` are
scored this way offline. The playground cannot yet run this multi-round game:
it sends each agent one trial holding all 60 rounds in `trial_info.rounds`, so
a playground run yields no decisions and stays unscored until a multi-round
runner exists.

**Units.** `human_effect_size` and `agent_effect_size` are reserve-price points
per additional bidder (`effect_size_metric`), not Cohen's d. Do not average
them with d-based effects of other studies.

**History.** From HumanStudy-Bench PR #50 to PR #54 the evaluator left
`human_significant` and `replication` empty and study_014 was excluded from the
playground `replicationRate`. PR #54 scored significance against Davis 2011 and
effect size against the Feng 2025 Pearson r = .42 (as Cohen's d), with the agent
side a Pearson correlation. That split is replaced by the rule above. A
replication rate or effect gap computed before this change used a different
agent test and a different effect metric for study_014.

For paper-extraction scoring, use only items whose printed_in includes a
provided source identifier, plus derivations whose inputs and reasoning are
fully stated. Unverified legacy aggregates and mismatched values are not exact
paper-extraction targets. The provenance annotations define that boundary;
they do not add an external scoring system.
