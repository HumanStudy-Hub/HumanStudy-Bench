# Measures of Anchoring in Estimation Tasks

**Authors:** Karen E. Jacowitz, Daniel Kahneman  
**Year:** 1995

---

## Description

The authors describe a method for the quantitative study of anchoring effects in estimation tasks. A calibration group provides estimates of a set of uncertain quantities. Subjects in the anchored condition first judge whether a specified number (the anchor) is higher or lower than the true value before estimating each quantity. The anchors are set at predetermined percentiles of the distribution of estimates in the calibration group (15th and 85th percentiles in this study). This procedure permits the transformation of anchored estimates into percentiles in the calibration group, allows pooling of results across problems, and provides a natural measure of the size of the effect. The authors illustrate the method by a demonstration that the initial judgment of the anchor is susceptible to an anchoring-like bias and by an analysis of the relation between anchoring and subjective confidence....

## Participants

- **N:** 156
- **Population:** Students
- **By sub-study:**
  - `exp_1_calibration`: n = 53
  - `exp_1_anchored_estimation`: n = 103
  - `exp_2_discredited_anchor`: n = 0
  - `exp_3_wtp_estimation`: n = 0

## Replicated tests (human data)

- **Mean Anchoring Index (AI)**  
  Reported: AI=0.49, n=103  
  Significance level: 0.05
- **Point-Biserial Correlation**  
  Reported: r=0.42, n=103  
  Significance level: 0.05
- **Student's t-test (Asymmetry of Anchoring)**  
  Reported: t(102)=7.99, p=0.01  
  Significance level: 0.01
- **Student's t-test (Extreme Estimates)**  
  Reported: t(102)=6.12, p=0.001  
  Significance level: 0.001
- **Pearson Correlation (AI vs. Confidence)**  
  Reported: r=-0.68, p=0.05  
  Significance level: 0.05
- **Student's t-test (High Anchor vs. Confidence)**  
  Reported: t(14)=2.37, p=0.05  
  Significance level: 0.05
- **Student's t-test (Low Anchor vs. Confidence)**  
  Reported: t(14)=4.80, p=0.001  
  Significance level: 0.001
- **Student's t-test (Confidence Ratings Comparison)**  
  Reported: t(154)=3.53, p=0.001  
  Significance level: 0.001
- **Percentage of judgments (High judged low)**  
  Reported: p=null, n=null, percentage=28  
  Significance level: 0.05
- **Percentage of judgments (Low judged high)**  
  Reported: p=null, n=null, percentage=15  
  Significance level: 0.05
- **Percentage of anchored estimates (High)**  
  Reported: p=null, n=null, percentage=24  
  Significance level: 0.05
- **Percentage of anchored estimates (Low)**  
  Reported: p=null, n=null, percentage=15  
  Significance level: 0.05

## Effect sizes & auxiliary statistics

- F1: Item-level anchoring indices and median estimates for 15 quantities.
  Raw/summary data present in `ground_truth.json`.
- F2: Transformed scores for high and low anchors.
  Raw/summary data present in `ground_truth.json`.
- F3: Percentage of estimates falling beyond the anchor.
  Raw/summary data present in `ground_truth.json`.
- F4: Mean confidence ratings and correlations.
  Raw/summary data present in `ground_truth.json`.
- F1: Percentage of judgments indicating the anchor was in the wrong direction despite discrediting.
  Raw/summary data present in `ground_truth.json`.
- F1: Comparison of WTP anchoring percentages against calibration group baselines.
  Raw/summary data present in `ground_truth.json`.

## Files

**source/**
- ground_truth.json

- ground_truth_raw_response.txt

- jacowitz-kahneman-1995-measures-of-anchoring-in-estimation-tasks.pdf

- materials/birth_order_likelihood.json

- materials/exp_1_anchored_estimation.json

- materials/exp_1_calibration.json

- materials/exp_2_discredited_anchor.json

- materials/exp_3_wtp_estimation.json

- materials/height_posterior_odds.json

- materials/high_school_programs.json

- materials/marble_distribution.json

- materials/ordinal_sample_size_judgments.json

- materials/replicability_of_significance.json

- materials/sampling_distribution_tasks.json

- materials/symmetric_binomial_posterior.json

- metadata.json

- specification.json

**scripts/**
- config.py

- evaluator.py

- stats_lib.py

- study_utils.py
