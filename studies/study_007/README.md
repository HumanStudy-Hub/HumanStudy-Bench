# Social categorization and similarity in intergroup behaviour

**Authors:** Michael Billig, Henri Tajfel  
**Year:** 1973

---

## Description

The present study is one of a series exploring the role of social categorization in intergroup behaviour. It has been found in our previous studies that in ‘minimal’ situations, in which the subjects were categorized into groups on the basis of visual judgments they had made or of their esthetic preferences, they clearly discriminated against members of an outgroup although this gave them no personal advantage. However, in these previous studies division into groups was still made on the basis of certain criteria of ‘real‘ similarity between subjects who were assigned to the same category. Therefore, the present study established social categories on an explicitly random basis without any reference to any such real similarity. It was found that, as soon as the notion of ‘group’ was introduced into the situation, the subjects still discriminated against those assigned to another random category. This discrimination was considerably more marked than the one based on a division of subjects in terms of interindividual similarities in which the notion of ‘group’ was never explicitly introduced. In addition, it was found that fairness was also a determinant of the subjects’ decisions.…

## Participants

- **N:** 75
- **Population:** Schoolboys
- **age_range:** 14-16
- **gender:** male
- **By sub-study:**
  - `2x2_factorial_design`: n = 75

## Replicated tests (human data)

- **ANOVA (2x2) on Overall FAV scores**  
  Reported: Categorization F(1, 72) = 14.96; Similarity F(1, 72) = 4.13  
  Significance level: 0.05
- **Wilcoxon Matched Pairs Test (Matrix 2 FAV vs MJP)**  
  Reported: Difference = 3.87 (Cat: Sim), p < .01; Difference = 1.98 (Cat: Non-sim), p < .05  
  Significance level: 0.05
- **Wilcoxon Matched Pairs Test (Matrix 3 FAV vs Fairness)**  
  Reported: Difference = 4.00 (Cat: Sim), p < .01; Difference = 2.67 (Cat: Non-sim), p < .01  
  Significance level: 0.01
- **One-sample t-test (Matrix 1 vs no-bias point 6.5)**  
  Reported: Mean = 4.38 (Cat: Sim), p < .01  
  Significance level: 0.01

## Effect sizes & auxiliary statistics

- F1: Overall FAV rank means across the 2x2 factorial design conditions.
  Raw/summary data present in `ground_truth.json`.
- F2: Mean difference scores between Ingroup Favouritism (FAV) and Maximum Joint Profit (MJP) for Matrix 2.
  Raw/summary data present in `ground_truth.json`.
- F3: Mean difference scores between Ingroup Favouritism (FAV) and Fairness (F) for Matrix 3.
  Raw/summary data present in `ground_truth.json`.
- F4: Mean scores for Matrix 1 (Straightforward FAV). Lower scores indicate higher ingroup favouritism.
  Raw/summary data present in `ground_truth.json`.

## Files

**source/**
- ground_truth.json

- materials/2x2_factorial_design.json

- metadata.json

- socialcateg.pdf

- specification.json

**scripts/**
- config.py

- evaluator.py

- stats_lib.py

- study_utils.py
