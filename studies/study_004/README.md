# Subjective Probability: A Judgment of Representativeness

**Authors:** Daniel Kahneman, Amos Tversky  
**Year:** 1972

---

## Description

This paper explores a heuristic—representativeness—according to which the subjective probability of an event, or a sample, is determined by the degree to which it: (i) is similar in essential characteristics to its parent population; and (ii) reflects the salient features of the process by which it is generated. This heuristic is explicated in a series of empirical examples demonstrating predictable and systematic errors in the evaluation of uncertain events. In particular, since sample size does not represent any property of the population, it is expected to have little or no effect on judgment of likelihood. This prediction is confirmed in studies showing that subjective sampling distributions and posterior probability judgments are determined by the most salient characteristic of the sample (e.g., proportion, mean) without regard to the size of the sample. The present heuristic approach is contrasted with the normative (Bayesian) approach to the analysis of the judgment of uncertainty....

## Participants

- **N:** 537
- **Population:** High school students
- **age_range:** 15-18
- **grades:** 10, 11, and 12
- **background:** No background in probability or statistics
- **education:** At least one course in statistics
- **By sub-study:**
  - `study_1_proportion`: n = 92
  - `study_1_randomness`: n = 92
  - `study_2_programs`: n = 89
  - `study_5_marbles`: n = 52
  - `study_6_sampling_distributions`: n = 1500
  - `study_7_ordinal`: n = 97
  - `study_9_heights`: n = 115

## Replicated tests (human data)

- **Sign test**  
  Reported: p < .01  
  Significance level: 0.01
- **Sign test**  
  Reported: p < .01  
  Significance level: 0.01
- **Sign test**  
  Reported: p < .01  
  Significance level: 0.01
- **Sign test**  
  Reported: p < .01  
  Significance level: 0.01
- **Descriptive comparison**  
  Reported: Indistinguishable (Figs 1a, 1b, 1c)  
  Significance level: —
- **Median test**  
  Reported: p < .01  
  Significance level: 0.01

## Effect sizes & auxiliary statistics

- F1: Comparison of birth sequences GBGBBG (given frequency 72) and BGBBBB.
  Raw/summary data present in `ground_truth.json`.
- F2: Comparison of perceived randomness in birth sequences.
  Raw/summary data present in `ground_truth.json`.
- F3: Guessing program origin for a class with 55% boys (Program A: 65% boys; Program B: 45% boys).
  Raw/summary data present in `ground_truth.json`.
- F4: Subjective probability of replication (n=10) after initial significant result (n=20, p < .05).
  Raw/summary data present in `ground_truth.json`.
- F5: Comparison of two marble distributions among five children.
  Raw/summary data present in `ground_truth.json`.
- F6: Subjective sampling distributions for three populations across three sample sizes.
  Raw/summary data present in `ground_truth.json`.
- F7: Odds that a sample came from a male vs female population based on height.
  Raw/summary data present in `ground_truth.json`.

## Files

**source/**
- Subjective_probability_A_judgment_of_rep.pdf

- ground_truth.json

- materials/study_1_proportion.json

- materials/study_1_randomness.json

- materials/study_2_programs.json

- materials/study_3_binomial.json

- materials/study_4_psychologists.json

- materials/study_5_marbles.json

- materials/study_6_sampling_distributions.json

- materials/study_7_ordinal.json

- materials/study_8_posterior.json

- materials/study_9_heights.json

- metadata.json

- specification.json

**scripts/**
- config.py

- evaluator.py

- stats_lib.py

- study_utils.py
