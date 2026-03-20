# Unraveling in Guessing Games: An Experimental Study

**Authors:** Rosemarie Nagel  
**Year:** 1995

---

## Description

Consider the following game: a large number of players have to state simultaneously a number in the closed interval [0, 100]. The winner is the person whose chosen number is closest to the mean of all chosen numbers multiplied by a parameter $p$. For $0 \leq p < 1$, the unique Nash equilibrium is zero. This experimental study investigates how players incorporate the behavior of others, focusing on finite depth of reasoning. Experiments were conducted using $p = 1/2, 2/3$, and $4/3$. First-period choices suggest boundedly rational behavior starting from an initial reference point of 50, with clustering around iteration steps 1 and 2 ($50p^n$), accounting for the difference in distributions across parameter values. Over four periods, choices converge toward the equilibrium (0 for $p < 1$, 100 for $p = 4/3$). The modal depth of reasoning does not increase over time. A qualitative learning-direction theory, based on individual experience (adjusting choice direction based on whether the previous adjustment factor was above or below the optimal factor), is proposed as a better explanation for the adjustment process over time than increasing depth of reasoning....

## Participants

- **N:** 168
- **Population:** —
- **By sub-study:**
  - `p_0.5_condition`: n = 50
  - `p_0.66_condition`: n = 67
  - `p_1.33_condition`: n = 51

## Replicated tests (human data)

- **Mann-Whitney U test**  
  Reported: p < 0.001  
  Significance level: 0.05
- **Mann-Whitney U test**  
  Reported: p < 0.0001  
  Significance level: 0.05
- **One-sided Binomial Test**  
  Reported: p < 0.01 (for p=1/2 and p=2/3), p < 0.05 (for p=4/3)  
  Significance level: 0.05
- **Binomial Test**  
  Reported: Significant at 5% level  
  Significance level: 0.05
- **Mann-Whitney U test**  
  Reported: p = 0.05 (one-tailed)  
  Significance level: 0.05
- **Binomial Test (Weak Hypothesis)**  
  Reported: p < 0.01  
  Significance level: 0.01
- **Binomial Test (Strong Hypothesis - Period 4)**  
  Reported: p < 0.01 (6 out of 7 sessions reject at 1% level)  
  Significance level: 0.01
- **Binomial Test (Session-level)**  
  Reported: p = 0.0078 (0.5^7) < 0.01  
  Significance level: 0.01

## Effect sizes & auxiliary statistics

- F1: First period mean and median choices across different parameter conditions.
  Raw/summary data present in `ground_truth.json`.
- F2: Concentration of choices around theoretical iteration steps.
  Raw/summary data present in `ground_truth.json`.
- F3: Transition counts showing directional movement over time.
  Raw/summary data present in `ground_truth.json`.
- F4: Median rate of decrease (w_med) calculated per session from Round 1 to Round 4.
  Raw/summary data present in `ground_truth.json`.
- F5: Reference point test: comparing choices to p × mean_{t-1} in Periods 2-4. Session-level analysis.
  Raw/summary data present in `ground_truth.json`.
- F6: Learning Direction Theory alignment: session-level analysis for p<1 conditions only (Sessions 1-7).
  Raw/summary data present in `ground_truth.json`.

## Files

**source/**
- Unraveling in Guessing Games- An Experimental Study.pdf

- ground_truth.json

- materials/p_0.5_condition.json

- materials/p_0.66_condition.json

- materials/p_1.33_condition.json

- metadata.json

- specification.json

**scripts/**
- config.py

- evaluator.py

- participant_pool.py

- stats_lib.py

- study_utils.py
