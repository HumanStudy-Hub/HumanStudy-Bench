# Intentional Action and Side-Effects in Ordinary Language

**Authors:** Joshua Knobe  
**Year:** —

---

## Description

The paper addresses a controversy in the concept of intentional action: whether side-effects of an action are brought about intentionally. The author hypothesizes that people's intuitions are influenced by their attitude toward the specific side-effect. Through two experiments, the author demonstrates an asymmetry: people are considerably more willing to say a side-effect was brought about intentionally when they regard that side-effect as bad (harmful) than when they regard it as good (helpful). This finding suggests that the assignment of praise and blame is at the root of how people apply the concept of intentionality....

## Participants

- **N:** 120
- **Population:** People spending time in a Manhattan public park.
- **By sub-study:**
  - `Experiment 1`: n = 78
  - `Experiment 2`: n = 42

## Replicated tests (human data)

- **Chi-square test**  
  Reported: chi2(1, N=78)=27.2, p=0.001  
  Significance level: 0.05
- **t-test (Combined with Exp 2)**  
  Reported: t(120)=8.4, p=0.001  
  Significance level: 0.05
- **Chi-square test**  
  Reported: chi2(1, N=42)=9.5, p=0.01  
  Significance level: 0.05
- **Correlation (Combined with Exp 1)**  
  Reported: r(120)=0.53, p=0.001  
  Significance level: 0.05

## Effect sizes & auxiliary statistics

- Finding 1: Percentage of participants judging the side-effect as intentional.
  Raw/summary data present in `ground_truth.json`.
- Finding 2: Mean praise/blame scores on a 0-6 scale.
  Raw/summary data present in `ground_truth.json`.
- Finding 1: Percentage of participants judging the side-effect as intentional.
  Raw/summary data present in `ground_truth.json`.
- Finding 3: Correlation between moral judgment and intentionality judgment across both experiments.
  Raw/summary data present in `ground_truth.json`.

## Files

**source/**
- KNOBE.pdf

- ground_truth.json

- ground_truth_raw_response.txt

- materials/experiment_1_harm.json

- materials/experiment_1_help.json

- materials/experiment_2_harm.json

- materials/experiment_2_help.json

- metadata.json

- specification.json

**scripts/**
- config.py

- evaluator.py

- stats_lib.py

- study_utils.py
