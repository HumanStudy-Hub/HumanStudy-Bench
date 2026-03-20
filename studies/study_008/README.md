# Pluralistic Ignorance and Alcohol Use on Campus: Some Consequences of Misperceiving the Social Norm

**Authors:** Deborah A. Prentice, Dale T. Miller  
**Year:** 1993

---

## Description

Four studies examined the relation between college students' own attitudes toward alcohol use and their estimates of the attitudes of their peers. All studies found widespread evidence of pluralistic ignorance: Students believed that they were more uncomfortable with campus alcohol practices than was the average student. Study 2 demonstrated this perceived self-other difference also with respect to one's friends. Study 3 tracked attitudes toward drinking over the course of a semester and found gender differences in response to perceived deviance: Male students shifted their attitudes over time in the direction of what they mistakenly believed to be the norm, whereas female students showed no such attitude change. Study 4 found that students' perceived deviance correlated with various measures of campus alienation, even though that deviance was illusory. The implications of these results for general issues of norm estimation and responses to perceived deviance are discussed....

## Participants

- **N:** 468
- **Population:** Princeton University undergraduates
- **gender:** {'women': 266, 'men': 202}
- **education_level:** 1st- through 4th-year classes
- **By sub-study:**
  - `study_1_comfort_estimation`: n = 132
  - `study_2_order_and_friend_comparison`: n = 242
  - `study_4_keg_ban_alienation`: n = 94

## Replicated tests (human data)

- **2 (sex) X 2 (target) analysis of variance (ANOVA)**  
  Reported: F(1, 130) = 55.52, p < .0001  
  Significance level: 0.05
- **Sex X Target interaction ANOVA**  
  Reported: F(1, 130) = 9.96, p < .005  
  Significance level: 0.05
- **Statistical comparison of variances (F-test)**  
  Reported: F(131, 131) = 2.99, p < .0001  
  Significance level: 0.05
- **2 (sex) X 3 (target) X 2 (question order) ANOVA**  
  Reported: F(2, 476) = 54.52, p < .0001  
  Significance level: 0.05
- **Target X Order interaction ANOVA**  
  Reported: F(2, 476) = 3.45, p < .05  
  Significance level: 0.05
- **ANCOVA on willingness to collect signatures (controlling for attitudes)**  
  Reported: F(1, 90) = 18.94, p < .0001  
  Significance level: 0.05
- **ANCOVA on willingness to work hours (controlling for attitudes)**  
  Reported: F(1, 90) = 10.99, p < .005  
  Significance level: 0.05
- **ANCOVA on percentage of reunions expected (controlling for attitudes)**  
  Reported: F(1, 89) = 8.10, p < .01  
  Significance level: 0.05

## Effect sizes & auxiliary statistics

- F1: Self-comfort vs. estimated average student comfort ratings on an 11-point scale.
  Raw/summary data present in `ground_truth.json`.
- F2: Comparison of the variance in actual self-ratings versus the variance in estimates of the average student.
  Raw/summary data present in `ground_truth.json`.
- F1: Comfort ratings for self, friends, and average student across two question-order conditions.
  Raw/summary data present in `ground_truth.json`.
- F1: Willingness to engage in social action (signatures and hours) based on perceived deviance from the norm.
  Raw/summary data present in `ground_truth.json`.
- F2: Expected reunion attendance as a measure of institutional connection/alienation.
  Raw/summary data present in `ground_truth.json`.

## Files

**source/**
- Social Norm.pdf

- ground_truth.json

- materials/study_1_comfort_estimation.json

- materials/study_2_order_and_friend_comparison.json

- materials/study_4_keg_ban_alienation.json

- metadata.json

- specification.json

**scripts/**
- config.py

- evaluator.py

- stats_lib.py

- study_utils.py
