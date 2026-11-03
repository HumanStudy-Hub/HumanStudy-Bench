# Ground-truth source audit

This package uses two declared documents: Davis et al. (2011), DOI
10.1287/mnsc.1100.1258, and Feng et al. (2025), DOI
10.18653/v1/2025.emnlp-main.391. The original copyrighted PDF remains a source
key only; its exact hash is in SOURCE.json. The later paper is already included.

The human benchmark is not a set of numbers extracted entirely from the 2011
paper. Every retained aggregate has a JSON-pointer entry in ground_truth.json's
value_provenance. Exact printed values, differences from printed values,
lower-precision figure labels, and values not located in either document are
explicitly distinguished. No expert aggregate has been deleted or silently
corrected. University-of-Michigan/2023 attribution is retained as legacy
attribution, not independently established from inspected replication data.

Davis Table 2 (page 182) reports the original bidder-count coefficient 2.98 with
a significance marker defined as p < .01. It does not establish the existing
p < .001 claim. Feng Table 2 (page 7698) prints human r = .42, mean 27.31,
SD 23.22, entropy 5.08, sell-through .76 and profit 32.83, without a Pearson
p-value. The legacy SD 23.23 and sell-through .74 remain preserved and flagged.
Figure 2 prints bidder-group means at one decimal place; it does not establish
all retained two-decimal values. Human sample size 40 is printed in Feng
Figure 1A (page 7694, "40 Humans"), so it is labelled reported; it describes
the later paper's human benchmark, not the 2011 sample, which Davis Table 1
(page 181) prints as 120 seller subjects, 20 per treatment cell. The total of
2400 observations is derived as 40 x 60 printed rounds.

Evidence labels are kept true: `missing` is used only for a gap with no value
(the Pearson p-value). Retained expert values that could not be located in
either document carry `reported_from: "hsb_legacy_expert_annotation"` and an
empty printed_in list. For these records, `reported` means reported by the
legacy benchmark annotation, not by either paper. Their status is
`unverified_legacy`, or `differs_from_printed` for the two legacy values
(SD 23.23, sell-through .74) that conflict with the values Feng Table 2 prints.

## Replication scoring: two tests, two sources

The evaluator scores one finding (F1) with statistics from two different tests
in two different papers. Each statistic keeps its own source, and the evaluator
output names both (`scoring_note`, `reported_statistics`, and the
`human_significance_*` and `human_effect_*` fields). The rule is recorded in
the finding's `scoring` block in ground_truth.json.

- **Human significance and direction** come from the original Davis et al.
  (2011) Table 2 random-effects regression (printed page 182; the coefficient is
  discussed on page 181): the coefficient on the centered bidder count is 2.98
  [SE 0.08] in Models (1) and (2), marked with an asterisk that the table note
  defines as p < 0.01. Only the bound is printed. The evaluator stores the bound
  (`human_p_value_bound = 0.01`), leaves `human_p_value` empty, and treats the
  result as significant because the bound is below the 0.05 level. The sign of
  the coefficient (positive) is the direction the agent must match. This
  regression is estimated on the NoInfo treatments of the original experiment
  (Table 1, page 181): Cuberoot and Cube pooled with a Cube indicator, and both
  bidder sets n in {1, 2, 3, 4} and n in {1, 4, 7, 10}. It is not limited to the
  Cuberoot n in {1, 4, 7, 10} cell that the agent task replays. Model (3), which
  adds a bidder-count x Cube interaction, prints a coefficient of 3.24; the
  scored value is the 2.98 of Models (1) and (2).
- **Human effect size** comes from the later Feng et al. (2025) Table 2 Human
  row (printed page 7698): Pearson r = .42, with no p-value printed. It is used
  only for the effect-size comparison. For the playground's effect-size table,
  both the human r and the agent r are converted to a Cohen's d equivalent with
  d = 2r / sqrt(1 - r^2), the identity `playground/analysis.py` applies to any
  reported r.

No p-value is inferred from r = .42, and the original regression's p-value
bound is not attached to the later Pearson coefficient: it is attached to the
regression it was printed for, and the row reports the two tests side by side.
The agent's own significance test is a Pearson correlation of reserve price
with bidder count on the agent decisions. It treats every decision as
independent, whereas Davis et al. (2011) used random effects for subjects, so
with thousands of decisions a small positive correlation is already
significant. The replication verdict therefore checks only that reserve prices
rise with bidder count; how closely the pattern matches the human one is shown
by the effect-size comparison, not by the verdict.

`replication` is true when the human result is significant (Davis 2011), the
agent correlation is significant at 0.05, and its sign matches the Davis 2011
coefficient. With this rule study_014 counts toward a replication rate again.
From HumanStudy-Bench PR #50 until this decision, the evaluator left `human_significant` and `replication` empty and study_014
was excluded from the playground `replicationRate`; replication rates computed
in that period exclude it.

For paper-extraction scoring, use only items whose printed_in includes a
provided source identifier, plus derivations whose inputs and reasoning are
fully stated. Unverified legacy aggregates and mismatched values are not exact
paper-extraction targets. The provenance annotations define that boundary;
they do not add an external scoring system.
