# Preregistered online experiment: hindsight bias and transfer learning for the sugar content of food items

`s1`

Single empirical study (N = 300 analysed) with a 3 (task: OJ, ROJ, NJ) by 4 (group: control, feedback, domain information, irrelevant information) mixed design. Appendices B (priors), C (mapping knowledge) and D (anchoring) are further analyses of the same sample, not separate experiments.

## Hypotheses

### Hindsight and transfer learning in the feedback group

`n_h1`

Conditions: Feedback group

- **Statement**: Presenting the actual sugar content values for previously estimated food items triggers hindsight effects for those items and improves estimation accuracy for new items. (p. 7)
- **Scope [derived]**: Tested separately in each group with the model comparison task (OJ versus ROJ for hindsight, OJ versus NJ for transfer learning). (p. 17)
  - how it was derived: Combines the hypothesis sentence with the described tests of hindsight and transfer learning effects.

### Hindsight and transfer learning in the domain-information group

`n_h2`

Conditions: Domain-information group

- **Statement**: Presenting values for other items from the same domain also triggers hindsight effects for the previously estimated items and improves estimation accuracy for new items. (p. 7)
- **Scope [derived]**: Tested separately in each group with the model comparison task (OJ versus ROJ for hindsight, OJ versus NJ for transfer learning). (p. 17)
  - how it was derived: Combines the hypothesis sentence with the described tests of hindsight and transfer learning effects.

### No hindsight or transfer learning in the irrelevant-information group

`n_h3`

Conditions: Irrelevant-information group

- **Statement**: Presenting the actual values relabeled as longitudes of European cities leads to neither hindsight effects nor improved estimation accuracy for new food items. (p. 7)
- **Scope [derived]**: Tested separately in each group with the model comparison task (OJ versus ROJ for hindsight, OJ versus NJ for transfer learning). (p. 17)
  - how it was derived: Combines the hypothesis sentence with the described tests of hindsight and transfer learning effects.

### No hindsight or transfer learning in the control group

`n_h4`

Conditions: Control group

- **Statement**: Without actual values (unrelated text), neither hindsight nor transfer learning effects occur. (p. 7)
- **Scope [derived]**: Tested separately in each group with the model comparison task (OJ versus ROJ for hindsight, OJ versus NJ for transfer learning). (p. 17)
  - how it was derived: Combines the hypothesis sentence with the described tests of hindsight and transfer learning effects.

## Participants

### Participants

`n_participants`

- **Setting**: Online, via Prolific; hosted with lab.js and Jatos; device had to be laptop, desktop or tablet (no smartphone). (p. 7, p. 11)
- **Roles [derived]**: All participants are respondents (estimators); no confederates or scripted roles are described. (p. 11)
  - how it was derived: The paper describes only participants who respond to the tasks; no other role appears.
- **Recruitment**: Prolific, with the same general eligibility criteria as Groß et al. (2023, Exp. 2). (p. 7)
- **Eligibility**: Native German speakers aged 18 to 45 years; study completed on laptop, desktop or tablet; individuals reporting type 1 or type 2 diabetes or an eating disorder were excluded. (p. 7)
- **N completed**: 340 participants completed the experiment. (p. 8)
- **N analysed**: 300 after data diagnostics: control 76, feedback 73, domain information 74, irrelevant information 77. (p. 13, p. 2)
- **N randomized per group**: Not stated in the paper.
- **Characteristics**: Completers: mean age 27.9 years; 142 women, 193 men, 5 diverse. Highest education: university degree 157, higher secondary certificate 122, vocational training 24, lower secondary certificate 24, PhD 13. (p. 8)
- **Sampling unit [derived]**: Individual participant; each contributes up to 32 OJ/ROJ pairs and 32 NJs. (p. 11, p. 10)
  - how it was derived: One participant per session; 32 items per task per the procedure.
- **Compensation**: Fixed £4.80; median completion time 18.09 minutes. (p. 8)
- **Exclusions**: Eight preregistered diagnostic steps; participants excluded for smartphone use (6), non-compliance (2), technical problems (6), impossible estimates (1) and outlying accuracy (21), plus item-level exclusions (31 responses). (p. 12, p. 13)
- **Sample size planning**: A priori simulation-based power analysis (mixedpower, R) on the hindsight effect in the domain-information group of Groß et al. (2023, Exp. 2); 80 participants per group reach 80% power, so 320 for four groups, plus 20 extra participants. (p. 8)
- **Ethics**: Approved by the Ethics Committee of the University of Mannheim. (p. 24)

## Design

### Design

`n_design`

- **Design type**: 3 (task: OJ, ROJ, NJ; within) by 4 (group: control, feedback, domain information, irrelevant information; between) mixed design. (p. 11)
- **Control**: Control group reads a text unrelated to the study topic instead of values (filler task, 256 seconds). (p. 11, p. 6)
- **Assignment**: Participants were randomly assigned to one of the four groups. (p. 11)
- **Allocation procedure**: Not stated in the paper.
- **Order**: Phases in fixed order OJ, information, ROJ, NJ; items presented sequentially in an individually randomized order in the OJ task and the same order in the ROJ task; assignment of item sets to roles counterbalanced with comparable frequency across participants and groups. (p. 11, p. 12)
- **Counterbalancing scheme**: Not stated in the paper.

## Conditions

### Control group

`c_control`

- **Description**: Read a text unrelated to the topic of the study (filler task) for 256 seconds instead of seeing values. (p. 11)

### Feedback group

`c_feedback`

- **Description**: Shown the actual sugar content values of the 32 food items from the OJ task. (p. 11)

### Domain-information group

`c_domain`

- **Description**: Shown the actual values of a different set of 32 food items from the same domain. (p. 11)

### Irrelevant-information group

`c_irrelevant`

- **Description**: Shown the same values as the feedback group but relabeled as longitudes of 32 well-known European cities, with the flag of the respective country. (p. 11)

## Materials

### Food items, item sets and images

`m_items`

- **Item list (verbatim table)** (verbatim, p. 30, p. 31, p. 32, p. 33, p. 9):
  > 96 food items in eight subcategories (12 each) with serving or purchase size (g or ml), sugar content (g) and set (A, B, C) as printed in Table A1; Table A2 lists the sets by sugar content and the city set. The table is carried in materials/materials.json.
- **Three sets**: From the 96 items, three sets of 32 items each with four items from each subcategory, comparable in degree of processing and in mean, median and range of sugar content. One set is used in the OJ and ROJ task; the same set (feedback, irrelevant) or a second set (domain) is shown in the information phase; a third set is used in the NJ task. (p. 10)
- **Sugar content statistics**: M = 7.6 grams, SD = 7.7, min = 0, max = 33. (p. 9)
- **Serving/purchase size**: Typical serving size for most items (for example a banana); typical purchase size for others (for example a can of coconut milk). (p. 9)
- **Image treatment**: Every item was photographed; pictures were edited to blur labels, nutritional information and numbers. Images are said to be at https://osf.io/hr5bv/ (not consulted). (p. 10, p. 9)
- **Food images**: Not stated in the paper.

### OJ question wording

`m_oj_prompt`

- **Printed example (translated from German)** (verbatim, p. 11):
  > How many grams of sugar does a kiwi that weighs 85 grams contain?
- **Wording in the text next to Figure 1** (verbatim, p. 9):
  > How many grams of sugar are in a kiwi that weighs 85 grams?
- **Task instructions and response-field label**: Not stated in the paper.

### ROJ question wording

`m_roj_prompt`

- **Printed example** (verbatim, p. 12):
  > How many grams of sugar does a kiwi that weighs 85 grams contain? What was your ORIGINAL answer?

### NJ question wording

`m_nj_prompt`

- **Printed example** (verbatim, p. 12):
  > How many grams of sugar does a tangerine that weighs 90 grams contain?

### Information screens: actual values (feedback group)

`m_info_feedback`

Conditions: Feedback group

- **Printed example** (verbatim, p. 11):
  > A kiwi that weighs 85 grams contains 8 grams of sugar.
- **Presentation time**: Each item was presented for 8 seconds, 256 seconds overall. (p. 11)

### Information screens: values of other items (domain-information group)

`m_info_domain`

Conditions: Domain-information group

- **Printed example** (verbatim, p. 11):
  > A plum that weighs 75 grams contains 7 grams of sugar.
- **Presentation time**: Each item was presented for 8 seconds, 256 seconds overall. (p. 11)

### Information screens: values relabeled as longitudes (irrelevant-information group)

`m_info_irrelevant`

Conditions: Irrelevant-information group

- **Printed example** (verbatim, p. 11):
  > Basel, a major city in Switzerland, is located at longitude 8.
- **City set** (verbatim, p. 33):
  > Table A2 lists 32 cities with longitudes approximating the sugar values; the values shown were the sugar contents, so they only approximated the actual longitudes.
- **Pairing of cities with values and flags**: Not stated in the paper.

### Filler text (control group)

`m_filler`

Conditions: Control group

- **Description**: A text unrelated to the topic of the study, shown for an overall duration of 256 seconds. (p. 11, p. 6)
- **Filler text content**: Not stated in the paper.

### Prior engagement question

`m_engagement`

- **Question (translated from German)** (verbatim, p. 10):
  > At the beginning of this study, you answered several questions about the sugar content of food items. How frequently, prior to this study, have you engaged with this topic (e.g., at school, at work, or in your leisure time)?
- **Response scale**: 7-point scale from very rarely to very frequently. (p. 10)
- **Wording of the cheating, click-through and display questions**: Not stated in the paper.

## Procedure

Order of steps:

- Informed consent (p. 11)
- Repeated: Phase 1: original judgments (OJ) of 32 food items (participant sees: Food items, item sets and images; OJ question wording) (p. 11, p. 10)
  - **Item loop**: 32 items presented sequentially in a randomized order for each participant; one numeric estimate in grams per item. (p. 11)
- Depending on condition: Phase 2: information phase (depends on group) (p. 11, p. 6)
  - **Order of values within the information phase**: Not stated in the paper.
  - Read the actual values of the OJ items (8 s each) (participant sees: Information screens: actual values (feedback group)) (p. 11)
    - conditions: Feedback group
  - Read the values of a different set of 32 items (8 s each) (participant sees: Information screens: values of other items (domain-information group)) (p. 11)
    - conditions: Domain-information group
  - Read the same values relabeled as longitudes of European cities (8 s each) (participant sees: Information screens: values relabeled as longitudes (irrelevant-information group)) (p. 11)
    - conditions: Irrelevant-information group
  - Read a filler text unrelated to the study (256 s) (participant sees: Filler text (control group)) (p. 11)
    - conditions: Control group
- Repeated: Phase 3: recall of original judgments (ROJ) for the same 32 items (participant sees: Food items, item sets and images; ROJ question wording) (p. 11, p. 12)
  - **Item loop**: Same 32 items in the same randomized order as in the OJ task; for each item, recall the original answer. (p. 12)
- Repeated: Phase 4: new judgments (NJ) of 32 new food items (participant sees: Food items, item sets and images; NJ question wording) (p. 12, p. 10)
  - **Item loop**: A new set of 32 food items; one estimate per item. (p. 12)
- Post-experiment questions and comments (participant sees: Prior engagement question) (p. 12, p. 10)

### Procedure

`n_procedure`

- **Task**: Estimate the sugar content in grams of common food items shown with their weight (and an image); later recall one's own original estimates; estimate a new set of items. (p. 2, p. 9, p. 11, p. 12)
- **Goal**: Measure estimation accuracy |Δ| in three tasks to detect hindsight effects (OJ versus ROJ) and transfer learning (OJ versus NJ). (p. 6, p. 7, p. 13)
- **Phases**: (1) OJ task, 32 items sequentially in random order; (2) information phase by group, 8 s per item, 256 s overall (control: filler text, 256 s); (3) ROJ task, the same 32 items in the same random order; (4) NJ task, a new set of 32 items; then post-experiment questions. (p. 11, p. 12)
- **What participants see [derived]**: OJ, ROJ and NJ questions are shown one item at a time; in the information phase the values (or text) of their group. Participants are not shown their own OJs in the ROJ task (they are asked to recall them). (p. 11, p. 12)
  - how it was derived: The ROJ example asks 'What was your ORIGINAL answer?' for each item; no screen showing the OJ is described.
- **Feedback [derived]**: No feedback is given during OJ, ROJ or NJ; values are shown only in the information phase. (p. 6, p. 11)
  - how it was derived: Derived from the four-phase description, in which actual values appear only in phase 2.
- **Stopping**: Participants complete all four phases; Prolific excluded participants not finishing within 87 minutes. (p. 11, p. 12)
- **Software**: Programmed with lab.js and hosted via Jatos. (p. 11)

## Variables

### Information group (between-participants factor)

`v_group`

- **Role [derived]**: iv (p. 11)
  - how it was derived: Group is the manipulated between-participants factor.
- **Type**: categorical: control, feedback, domain information, irrelevant information (p. 11)
- **Observation unit [derived]**: participant (p. 11)
  - how it was derived: Each participant belongs to one group.

### Task (within-participants factor)

`v_task`

- **Role [derived]**: iv (p. 11, p. 14)
  - how it was derived: Task is the within-participants fixed effect.
- **Type**: categorical: OJ, ROJ, NJ (p. 11)
- **Observation unit [derived]**: participant by item response (p. 13)
  - how it was derived: Random effects for participants and items imply item-level rows.

### Estimate of sugar content

`v_estimate`

- **Role [derived]**: raw response (p. 11)
  - how it was derived: Response from which |Δ| is computed.
- **Type [derived]**: numeric (p. 11)
  - how it was derived: Estimates are in grams and compared with numeric actual values.
- **Unit**: grams of sugar (p. 11)
- **Observation unit [derived]**: one estimate per participant, item and task (p. 13)
  - how it was derived: Eq. 1 defines |Δ| for each item i and participant j.

### Absolute deviation |Δ| (estimation accuracy)

`v_absdev`

- **Role**: dv (p. 13)
- **Type [derived]**: numeric, non-negative (p. 13)
  - how it was derived: Absolute deviation is non-negative.
- **Definition**: |Δij| = |estimateij − actuali|, the absolute deviation of the estimate from the actual value for item i and participant j; smaller values indicate higher accuracy. (p. 13)
- **Unit [derived]**: grams (p. 13)
  - how it was derived: Estimates and actual values are in grams.
- **Transformation**: None (no logarithmic order-of-magnitude error); criterion mean-centered in the models. (p. 13, p. 34)
- **Observation unit [derived]**: one row per participant, item and task (p. 13)
  - how it was derived: Defined for each item and participant in each task.

### Prior engagement with the domain

`v_engage`

- **Role [derived]**: covariate (p. 16)
  - how it was derived: Examined as a predictor of initial accuracy and effect sizes.
- **Type**: ordinal, 7-point (p. 10)
- **Observation unit [derived]**: participant (p. 10)
  - how it was derived: One answer per participant.

### Rank-order correlation between estimates and actual values (mapping knowledge)

`v_rank`

- **Role**: dv (p. 35)
- **Type**: numeric correlation, Fisher r-to-z transformed and mean-centered (p. 35)
- **Observation unit [derived]**: participant by task (p. 35)
  - how it was derived: A rank-order correlation is computed for each participant.

## Analysis

### Data diagnostics and exclusions

`a_exclusions`

- **Aim**: Exclude extreme outliers and non-compliance (preregistered steps, comparable to Groß et al., 2023, Exp. 2). (p. 12)
- **Method**: Eight steps: Prolific time limit (87 min); smartphone use; non-compliance; technical problems; impossible estimates (above item weight for 10+ items); estimates under 1000 ms for 10+ items (none); 21 participants with median absolute deviation above the threefold interquartile range in at least one task; item-level exclusion of estimates above item weight (30) or under 1000 ms (1). (p. 12, p. 13)
- **Exact outlier criterion**: Not stated in the paper.

### Treatment of perfectly accurate judgments and perfect OJ reproductions

`a_perfect`

- **Aim**: Avoid distorting hindsight and transfer effects with cases where no effect can occur. (p. 14, p. 15)
- **Method**: Exclude OJs equal to the actual value together with their ROJs (7.4%); exclude OJ/ROJ pairs with ROJ equal to OJ (all analyses, not preregistered for transfer learning); NJ items have no corresponding exclusions. (p. 14, p. 15)
- **Coding**: OJ = actual value excluded; ROJ = OJ excluded. (p. 14, p. 15)

### Transfer learning test (OJ versus NJ)

`a_tl`

- **Aim**: Test whether |Δ| is smaller for NJs than for OJs in each group. (p. 17)
- **Inputs [derived]**: |Δ| per participant, item and task after exclusions. (p. 13, p. 14)
  - how it was derived: Combines the definition of |Δ| with the stated exclusions.
- **Method**: Bayesian linear mixed-effects models (brms, Stan); fixed effect task; random intercepts and slopes for participants and items; full model M1 versus baseline M0 without the predictor; Bayes factors by bridge sampling. (p. 13, p. 14)
- **Criterion**: BF10 scale: above 10 strong, 3 to 10 moderate, 1 to 3 weak evidence for M1; reciprocals for M0. (p. 14)
- **MCMC settings and contrast coding**: Not stated in the paper.

### Hindsight test (OJ versus ROJ)

`a_hs`

- **Aim**: Test whether |Δ| is smaller for ROJs than for OJs in each group. (p. 17)
- **Method**: Same mixed-effects model comparison as the transfer learning test with task OJ versus ROJ. (p. 17, p. 13, p. 14)

### Interaction analyses between groups

`a_int`

- **Aim**: Compare effect sizes between feedback and domain-information groups and between control and irrelevant-information groups. (p. 17)
- **Method**: Models including the task by group interaction; fixed effects task, group and their interaction; footnote notes that preregistered analyses were the task effect per group and interaction analyses were additional. (p. 13, p. 14)

### Priors and sensitivity analysis (Appendix B)

`a_priors`

- **Method**: Criterion |Δ| mean-centered; intercept normal(0,15); slopes either skeptical normal(0,10) or weakly informative normal(-5,10) (main text); random-effect SDs and residual SD half-normal(0,15); LKJ(2) prior on random-effect correlations; prior predictive checks. (p. 34)
- **Robustness**: General conclusions were robust across the two prior specifications. (p. 14)

### Mapping knowledge (Appendix C)

`a_map`

- **Aim**: Test improvement in the rank-order correlation: no transfer learning expected; hindsight expected in the feedback group only. (p. 35)
- **Method**: Person-level rank-order correlations, Fisher r-to-z and mean-centered; Bayesian linear mixed-effects models with task as fixed effect and random intercepts and slopes for participants; priors normal(0,0.5); perfect judgments and reproductions included. (p. 35)

### Anchoring analyses in the irrelevant-information group (Appendix D)

`a_anchor`

- **Aim**: Check whether relabeled values influenced ROJs via their mean, median or the last values presented. (p. 37)
- **Method**: Same models as the main analysis with |Δ| from the mean or median of presented values, and recency mean and median of the last five values. (p. 37)

### Prior engagement analyses

`a_eng`

- **Method**: Association of self-reported engagement with initial accuracy and with hindsight and transfer learning effects. (p. 16)

### Comparison across food subcategories

`a_subcat`

- **Method**: Transfer learning and hindsight effects compared across the eight food subcategories against initial accuracy of the OJs (conditional predictions from the mixed-effects models). (p. 20)

## Reported results

### Reported sample flow

`res_sample`

- **Result kind**: reported (p. 13)
- **Descriptives**: 340 completed; 300 analysed: control 76, feedback 73, domain 74, irrelevant 77. (p. 8, p. 13)

### Reported perfect reproductions and accurate judgments

`res_perfect`

- **Result kind**: reported (p. 15)
- **Descriptives**: OJ equal to the actual value: 7.4%. Perfect OJ reproductions: feedback 35.23%, domain information 33.08%, irrelevant information 39.67%, control 39.67%. (p. 14, p. 15)
- **Test**: Percentages differed between groups, X2(3) = 34.04, p < .001. (p. 15)

### Reported self-reported engagement

`res_engage`

- **Result kind**: reported (p. 16)
- **Descriptives**: M = 3.44, SD = 1.45 on the 7-point scale. (p. 16)

### Reported transfer learning effects

`res_tl`

- **Result kind**: reported (p. 17)
- **Effect (b, CI 95%, BF10)**: Feedback: b = −6.59, [−8.63, −4.55], BF10 > 10,000. Domain information: b = −3.71, [−5.09, −2.32], BF10 > 10,000. Irrelevant information: b = 0.52, [−0.45, 1.49], BF10 = 0.08. Control: b = −0.92, [−2.00, 0.15], BF10 = 0.19. (p. 17)
- **Interactions**: Feedback versus domain: b = −3.11, [−5.06, −1.16], BF10 = 11 (larger in feedback; no difference in Groß et al., 2023). Irrelevant versus control: b = 1.35, [−0.06, 2.77], BF10 = 0.37. (p. 17)

### Reported hindsight effects

`res_hs`

- **Result kind**: reported (p. 17)
- **Effect (b, CI 95%, BF10)**: Feedback: b = −2.88, [−4.19, −1.54], BF10 = 268. Domain information: b = −2.25, [−3.32, −1.16], BF10 = 80. Irrelevant information: b = 0.87, [−0.06, 1.81], BF10 = 0.23. Control: b = 1.10, [−0.15, 2.35], BF10 = 0.25. (p. 17)
- **Interactions**: Feedback versus domain: b = −0.70, [−2.11, 0.71], BF10 = 0.07. Control versus irrelevant: b = −0.26, [−1.15, 0.64], BF10 = 0.05. (p. 17, p. 18)

### Reported size of hindsight versus transfer learning effects

`res_hs_tl`

- **Result kind**: reported (p. 22)
- **Descriptives**: In the feedback and domain-information groups hindsight effects were smaller than transfer learning effects; ROJs were further from the actual values than NJs. (p. 22)

### Reported Bayes factors with skeptical prior (Table B1)

`res_sens`

- **Result kind**: reported (p. 34)
- **Table B1 (skeptical prior; weakly informative in main text)**: Transfer learning, BF10 for control, feedback, domain, irrelevant, interaction control×irrelevant, interaction feedback×domain: 0.20, >10,000, >10,000, 0.08, 0.42, 13. Hindsight: 0.06, 280, 76, 0.25, 0.29, 0.11. (p. 34)

### Reported mapping knowledge (Tables C1, C2)

`res_map`

- **Result kind**: reported (p. 36)
- **Rank-order correlations M (SD): control, feedback, domain, irrelevant**: OJ 0.61 (0.13), 0.63 (0.10), 0.60 (0.14), 0.60 (0.14); ROJ 0.62 (0.15), 0.70 (0.11), 0.63 (0.14), 0.60 (0.13); NJ 0.64 (0.13), 0.69 (0.12), 0.65 (0.14), 0.61 (0.14). (p. 36)
- **Bayes factors BF10 (control, feedback, domain, irrelevant)**: Transfer learning (OJ vs NJ) 0.70, 3, 8, 0.05; hindsight (OJ vs ROJ) 0.08, >10,000, 5, 0.03. (p. 36)
- **Summary**: Improved mapping knowledge for transfer learning in feedback and domain groups and for hindsight in both groups; none in control or irrelevant groups. (p. 35, p. 36)

### Reported anchoring analyses (Appendix D)

`res_anchor`

- **Result kind**: reported (p. 37)
- **BF10**: ROJs not influenced by the mean (BF10 = 0.20) or median (BF10 = 0.32) of presented values; recency mean 0.15, recency median 0.16. (p. 37)

### Reported subcategory comparison

`res_subcat`

- **Result kind**: reported (p. 20)
- **Descriptives**: Subcategories with higher initial accuracy showed smaller transfer learning and hindsight effects; the paper contrasts Oils and fats (high initial accuracy) with Drinks (low initial accuracy) in the feedback group. (p. 20)

### Figure 2: estimation accuracy by group and task

`res_fig2`

- **Result kind**: reported (p. 15)
- **Mean |Δ| per group and task**: Not stated in the paper.

## Simulation scope

### Simulation scope

`sim_s1`

- **Steps [implementation]**: Phases OJ, information, ROJ and NJ and the post-experiment engagement question can be replayed as text. Consent, device checks, Prolific time limits, display problems and response-time exclusions cannot be replayed. (p. 11, p. 12)
- **Roles [implementation]**: Every participant is an AI participant, run sequentially with its own conversation history. No scripted roles exist in the paper. (p. 11)
- **Conditions [implementation]**: Feedback, domain-information and irrelevant-information arms can run from printed materials (city–value pairing and countries are assumptions to confirm). The control arm cannot run until the filler text is supplied. (p. 11)
- **Outcomes [implementation]**: |Δ| per task, hindsight and transfer effects per group; the paper's Bayesian mixed models are not reproduced offline (approximate paired comparisons only). Prior engagement is a self-report of real-world experience and is not simulated. (p. 13, p. 14)
- **Limits [implementation]**: AI participants have no real food experience, time pressure, reaction times or visual perception of the images (text only unless images are supplied); an AI that keeps its transcript can reproduce its OJ exactly in the ROJ task, which differs from human memory. (p. 9, p. 15)

## Open issues

- **Instruction and consent texts are not printed** [decision, `iss_instructions`, m_oj_prompt]: Only example question wordings are printed; instruction screens, response-field label and consent text are not. Suggested: Supply the original (German) instruction screens or approve a minimal framing.
- **Food images are not available** [decision, `iss_images`, m_items]: The images are on OSF (not consulted) and are not in the paper. Suggested: Supply the images or accept a text-only replay. (p. 10, p. 9)
- **Filler text for the control group is missing** [blocking, `iss_filler`, m_filler]: The text is described as unrelated to the study but not printed. Suggested: Supply the filler text in materials/materials.json (control_filler_text). (p. 11)
- **City–value pairing, countries and flags are not printed** [decision, `iss_city`, m_info_irrelevant]: Only the Basel example and the city list with approximate longitudes are given. Suggested: Confirm the pairing and supply country names or flags. (p. 11, p. 33)
- **Counterbalancing scheme for item sets is not described** [decision, `iss_counter`, n_design]: Only 'comparable frequency' is stated. Suggested: Provide the assignment scheme. (p. 12)
- **Group allocation procedure and randomized n are not reported** [check, `iss_alloc`, n_participants]: Random assignment is stated; allocation ratio and the numbers randomized per group are not. Suggested: Provide allocation and randomized n. (p. 11)
- **Order of values in the information phase is not stated** [check, `iss_order`, s1]: The step description gives timing (8 s per item) but not the order. Suggested: State the order. (p. 11)
- **Two printed OJ wordings differ** [check, `iss_wording`, m_oj_prompt]: Figure 1 text reads 'How many grams of sugar are in a kiwi that weighs 85 grams?' and the procedure example reads 'How many grams of sugar does a kiwi that weighs 85 grams contain?'; both are translations. Suggested: Confirm which wording was shown. (p. 11, p. 9)
- **MCMC settings, coding and outlier reference are not stated** [check, `iss_mcmc`, a_tl]: Chains, iterations, task coding and the exact threefold-IQR rule are not given. Suggested: Provide the analysis code (OSF). (p. 14)
- **Outlier criterion not fully specified** [check, `iss_outlier`, a_exclusions]: Seventh exclusion step lacks the reference for the interquartile range. Suggested: Confirm in the analysis code. (p. 12, p. 13)
- **Cell means of |Δ| are not reported numerically** [check, `iss_cells`, res_fig2]: They appear only in Figure 2. Suggested: Obtain means from the OSF data. (p. 15)
- **Wording of compliance and display questions is not printed** [check, `iss_compliance`, m_engagement]: Described but not worded. Suggested: Supply wording if needed. (p. 12)
- **Units and phrasing for liquid and counted items are not printed** [check, `iss_units`, m_items]: Table A1 gives size in g or ml without marking which; examples only cover solid items with 'weighs … grams'. Suggested: Confirm the original item phrasing and units. (p. 30, p. 9)
- **Allocation procedure for random assignment is not stated** [check, `iss_alloc_rule`, n_design]: The paper says participants were randomly assigned but not how. Suggested: State the allocation procedure. (p. 11)
