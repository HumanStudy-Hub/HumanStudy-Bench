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

The evaluator uses the later paper's Pearson benchmark and leaves human
significance and the replication verdict unavailable because that paper does
not print its p-value. The original regression's significance cannot be attached
to the later Pearson coefficient. The original regression remains separately
recorded, and the former combined statistic is preserved for audit.

For paper-extraction scoring, use only items whose printed_in includes a
provided source identifier, plus derivations whose inputs and reasoning are
fully stated. Unverified legacy aggregates and mismatched values are not exact
paper-extraction targets. The provenance annotations define that boundary;
they do not add an external scoring system.
