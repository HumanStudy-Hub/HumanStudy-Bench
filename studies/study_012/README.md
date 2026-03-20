# Trust, Reciprocity, and Social History

**Authors:** Joyce Berg, John Dickhaut, Kevin McCabe  
**Year:** 1995

---

## Description

We designed an experiment to study trust and reciprocity in an investment setting. This design controls for alternative explanations of behavior including repeat game reputation effects, contractual precommitments, and punishment threats. Observed decisions suggest that reciprocity exists as a basic element of human behavior and that this is accounted for in the trust extended to an anonymous counterpart. A second treatment, social history, identifies conditions which strengthen the relationship between trust and reciprocity....

## Participants

- **N:** 120
- **Population:** undergraduate students
- **By sub-study:**
  - `no_history_investment_game`: n = 64
  - `social_history_investment_game`: n = 56

## Replicated tests (human data)

- **Approximate Randomization Test**  
  Reported: p = 0.29  
  Significance level: 0.05
- **Mean Comparison**  
  Reported: Mean Sent = $5.16, Mean Returned = $4.66  
  Significance level: —
- **Spearman's rank correlation coefficient**  
  Reported: rs = 0.01  
  Significance level: —
- **Approximate Randomization Test**  
  Reported: p = 0.06  
  Significance level: 0.05
- **Wilcoxon rank-sum test**  
  Reported: r = 776, p = 0.1  
  Significance level: 0.1
- **Spearman's rank correlation coefficient**  
  Reported: rs = 0.34  
  Significance level: —
- **Resampling test**  
  Reported: p = 0.06  
  Significance level: 0.1
- **Strategy Type Classification**  
  Reported: >90% of pairs follow one of four strategies  
  Significance level: —

## Effect sizes & auxiliary statistics

- F1: Investment decisions (amount sent) by Room A subjects.
  Raw/summary data present in `ground_truth.json`.
- F2: Comparison of average investment vs average payback.
  Raw/summary data present in `ground_truth.json`.
- F3: Correlation between the amount sent by Room A and the amount returned by Room B.
  Raw/summary data present in `ground_truth.json`.
- F4: Investment decisions by Room A subjects provided with social history.
  Raw/summary data present in `ground_truth.json`.
- F5: Comparison of payback amounts between Experiment 1 and Experiment 2.
  Raw/summary data present in `ground_truth.json`.
- F6: Correlation between amount sent and amount returned in the Social History condition.
  Raw/summary data present in `ground_truth.json`.
- F7: Classification of Room B return strategies based on k = Return / (3 * Sent).
  Raw/summary data present in `ground_truth.json`.

## Files

**source/**
- berg_geb95.pdf

- ground_truth.json

- materials/no_history_investment_game.json

- materials/social_history_investment_game.json

- metadata.json

- specification.json

**scripts/**
- config.py

- evaluator.py

- participant_pool.py

- stats_lib.py

- study_utils.py
