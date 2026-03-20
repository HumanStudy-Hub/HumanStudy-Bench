# The “False Consensus Effect”: An Egocentric Bias in Social Perception and Attribution Processes

**Authors:** Lee Ross, David Greene, Pamela House  
**Year:** 1977

---

## Description

Evidence from four studies demonstrates that social observers tend to perceive a “false consensus” with respect to the relative commonness of their own responses. A related bias was shown to exist in the observers’ social inferences. Thus, raters estimated particular responses to be relatively common and relatively unrevealing concerning the actors’ distinguishing personal dispositions when the responses in question were similar to the raters’ own responses; responses differing from those of the rater, by contrast, were perceived to be relatively uncommon and revealing of the actor. These results were obtained both in questionnaire studies presenting subjects with hypothetical situations and choices and in authentic conflict situations. The implications of these findings for our understanding of social perception phenomena and for our analysis of the divergent perceptions of actors and observers are discussed. Finally, cognitive and perceptual mechanisms are proposed which might account for distortions in perceived consensus and for corresponding biases in social inference and attributional processes....

## Participants

- **N:** 504
- **Population:** Stanford undergraduates
- **By sub-study:**
  - `study_1_hypothetical_stories`: n = 320
  - `study_2_personal_description_items`: n = 80
  - `study_3_sandwich_board_hypothetical`: n = 104

## Replicated tests (human data)

- **Analysis of Variance (Fixed variable: Story)**  
  Reported: F(1, 312) = 49.1, p < .001  
  Significance level: 0.05
- **Analysis of Variance (Trait Ratings)**  
  Reported: F(1, 312) = 37.40, p < .001  
  Significance level: 0.05
- **Student's t-test**  
  Reported: Multiple t-values ranging from < 1 to 6.19, p < .10 to p < .001  
  Significance level: 0.05
- **Analysis of Variance (Combined Signs)**  
  Reported: F = 56.2  
  Significance level: 0.05
- **Analysis of Variance (Combined Trait Ratings)**  
  Reported: F = 17.79  
  Significance level: 0.05

## Effect sizes & auxiliary statistics

- F1: Estimates of commonness for Choice 1 by subjects who chose Choice 1 vs Choice 2 across four stories.
  Raw/summary data present in `ground_truth.json`.
- F2: Trait rating differences (Choice 1 - Choice 2) by subjects who chose Choice 1 vs Choice 2.
  Raw/summary data present in `ground_truth.json`.
- F3: Percentage estimates of Category 1 by members of Category 1 vs Category 2 for 17 significant items.
  Raw/summary data present in `ground_truth.json`.
- F4: Estimates of percentage who would wear the sign by those who agreed vs refused.
  Raw/summary data present in `ground_truth.json`.
- F5: Trait rating differences (Wear - Not Wear) by subjects who agreed vs refused.
  Raw/summary data present in `ground_truth.json`.

## Files

**source/**
- 4705-Ross-et-al-False-Consensus-Effect.pdf

- ground_truth.json

- materials/study_1_hypothetical_stories.json

- materials/study_2_personal_description_items.json

- materials/study_3_sandwich_board_hypothetical.json

- metadata.json

- specification.json

**scripts/**
- config.py

- evaluator.py

- stats_lib.py

- study_utils.py
