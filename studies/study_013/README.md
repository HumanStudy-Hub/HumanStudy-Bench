# Study 013: Opportunity Evaluation under Risky Conditions

**Authors:** Hean Tat Keh, Maw Der Foo, Boon Chong Lim

**Year:** 2002

**Journal:** *Entrepreneurship Theory and Practice*, 27(2), 125-148

## Description

This study examines how cognitive biases affect entrepreneurs' opportunity evaluation under risky conditions. Using a survey of 77 founders of top SMEs in Singapore, the study measures four cognitive biases (overconfidence, illusion of control, belief in the law of small numbers, and planning fallacy) and tests how they influence risk perception and opportunity evaluation of a standardized business case vignette. The benchmark implementation focuses on the paper's calibration test and its reported regression findings.

## Participants

- **N = 77** founders and owners of the top 500 SMEs in Singapore
- 97% male, mean age 46.6 years
- 92.4% Chinese, 79% founded their business
- Business revenue: 48.6% between S$1M-S$25M, 44.4% between S$25M-S$50M

## Key Findings Tested

| Finding | Hypothesis | Human Result |
|---------|-----------|--------------|
| F1 | Entrepreneurs are overconfident (mean items outside 90% CI > 1) | Mean = 5.17, SD = 2.64 |
| F2 | Risk perception negatively predicts opportunity evaluation (H1) | beta = -0.50, t = -5.98, p < .001 |
| F3 | Illusion of control negatively predicts risk perception in Model 1 (H5) | beta = -0.76, t = -3.34, p < .01 |
| F4 | Illusion of control positively predicts opportunity evaluation in Model 2 | beta = 0.40, t = 2.23, p < .05 |
| F5 | Belief in the law of small numbers positively predicts opportunity evaluation in Model 2 | beta = 1.17, t = 1.91, p < .06 |

## Questionnaire Structure

- **Section A:** 5 forced-choice gamble items (risk propensity)
- **Section B:** 7 Likert items (2 filler, 2 planning fallacy, 3 illusion of control)
- **Section C:** 10 confidence-interval estimation items (overconfidence)
- **Section D:** Business case vignette + 4 risk perception items + 3 opportunity evaluation items + 1 optional open-ended item coded for belief in the law of small numbers

## File Structure

```
study_013/
├── index.json
├── README.md
├── source/
│   ├── Keh-Foo-Lim-2002-Opportunity-Evaluation.pdf
│   ├── metadata.json
│   ├── specification.json
│   ├── ground_truth.json
│   └── materials/
│       ├── section_a_risk_propensity.json
│       ├── section_b_cognitive_biases.json
│       ├── section_c_overconfidence.json
│       └── section_d_case_vignette.json
└── scripts/
    ├── config.py
    ├── evaluator.py
    ├── study_utils.py
    └── stats_lib.py
```

## Overconfidence Answer Key

The 10 confidence-interval items reference Singapore statistics circa 1999-2000. Correct answers have been verified against:
- Yearbook of Statistics Singapore 2000 (Department of Statistics)
- Changi Airport Group corporate history
- LTA Vehicle Quota Tender Results 2000-2004
- SingStat residential dwelling datasets

## Contributor

Guankai Zhai ([@zgk2003](https://github.com/zgk2003))
