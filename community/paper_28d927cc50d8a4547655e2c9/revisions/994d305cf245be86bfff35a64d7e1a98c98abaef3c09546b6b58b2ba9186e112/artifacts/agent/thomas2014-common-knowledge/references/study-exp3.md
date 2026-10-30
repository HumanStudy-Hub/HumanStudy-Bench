# Experiment 3: replication of the $2 condition with similarity manipulation, Big Five and knowledge-level comprehension test

`exp3`

MTurk participants, 800 recruited, 550 analysed; knowledge x similarity design; Big Five questionnaire and a harder knowledge-level comprehension test whose errors are analysed as a confusion matrix.

## Hypotheses

### Common knowledge recognition hypothesis

`hyp_ck`

- **Statement**: In coordination environments, people strategically differentiate between shared knowledge and common knowledge; participants will try to work together more frequently with common knowledge of the payoffs than with shared or primary knowledge. (p. 13)
- **Scope**: Experiment 1, Experiment 2, Experiment 3 (p. 13)

### Shared knowledge hypothesis

`hyp_sk`

- **Statement**: People do not represent common knowledge as a distinct state; the only major distinction affecting coordination decisions is between primary and shared knowledge. (p. 13)
- **Scope**: Experiment 1, Experiment 2, Experiment 3 (p. 13)

### Curse of knowledge hypothesis

`hyp_curse`

- **Statement**: Participants will try to work together with the same frequency across all knowledge levels. (p. 14)
- **Scope**: Experiment 1, Experiment 2, Experiment 3 (p. 14)

### Single versus distinct representation hypotheses

`hyp_rep`

- **Statement**: Single-representation: most comprehension errors in the common knowledge condition, decreasing as levels of shared knowledge decrease. Distinct-representation: errors increase only with the number of levels of shared knowledge, errors on common knowledge are few, and errors are systematic: levels of shared knowledge are mistaken for each other, not for common knowledge. (p. 14, p. 15)
- **Scope**: Experiment 3 (p. 14, p. 15)

### Perceived similarity hypothesis

`hyp_sim`

- **Statement**: The more similar people perceive themselves to be to their partners, the more likely they will be to predict that their partner will choose as they do, and thus to attempt risky coordination. (p. 16)
- **Scope**: Experiment 3 (p. 16)

### Altruistic motives hypothesis

`hyp_alt`

- **Statement**: People who are higher on Agreeableness will be more likely to try to work together. (p. 16)
- **Scope**: Experiment 3 (p. 16)

### Risk-seeking hypothesis

`hyp_risk`

- **Statement**: People who are higher on Openness will be more likely to take the bet by trying to work together. (p. 16, p. 17)
- **Scope**: Experiment 3 (p. 16, p. 17)

## Participants

### Experimenter / interface (scripted role)

`scripted_researchers`

- **Role**: Experimenter and survey interface: presents the printed instructions, the private (messenger boy) and public (loudspeaker) boxes, and the comprehension questions; performs the offline random pairing and payment after data collection. (p. 19, p. 20)
- **Scripted [implementation]**: Yes: this is the paper's own scripted role (an interface and offline payment); it is not an original participant.

### Experiment 3 participants

`part_e3`

- **Setting**: Online study on Amazon Mechanical Turk. (p. 27)
- **Roles [derived, to check]**: Butcher or baker as in the $2 condition of Experiment 1 (role split not stated). (p. 27)
  - how it was derived: Design 'added three components to the $2 condition from Experiment 1'.
- **Recruitment**: Amazon Mechanical Turk. (p. 27)
- **Recruited**: 800, evenly distributed across similarity and knowledge conditions (8 cells of 100). (p. 27)
- **Analysed [derived, to check]**: 550 after eliminating people who failed the comprehension questions. Derived knowledge-condition sizes from the pairwise Ns: primary 131, secondary 149, tertiary 135, common 135. (p. 27, p. 28, p. 29)
  - how it was derived: From N=280 (prim+sec), 284 (sec+ter), 266 (prim+ter), 270 (common+ter) with total 550.
- **Characteristics**: About 46% male; Mage = 31.6, SDage = 11.3. (p. 27)
- **Eligibility**: Not stated in the paper. (p. 27)
- **Sampling unit [derived]**: The individual participant. (p. 27)
  - how it was derived: One decision per participant as in Experiment 1.

## Design

### Experiment 3 design

`design_e3`

- **Design type**: Replication of the $2 payoff condition of Experiment 1 with a 4 (knowledge) x 2 (similar vs. dissimilar partner) between-subjects design, adding a Big Five questionnaire and a harder knowledge-level comprehension test. (p. 26, p. 27)
- **Assignment**: Not stated in the paper. (p. 27)
- **Order [to check]**: Four similarity questions at the start; participants are told whether the matched partner is similar or dissimilar and rate perceived similarity; scenario and decision; Big Five questionnaire; finally the knowledge-level comprehension questions on a separate page without reference to the instructions. (p. 27, p. 28)
- **Manipulation check**: Perceived similarity rating (0-100%) checks the similarity manipulation; knowledge-level comprehension questions are the test of representation. (p. 28)

## Conditions

### Experiment 3: Primary knowledge

`kc_prim_e3`

- **Definition**: Primary knowledge as in Experiment 1 ($2 payoff condition). (p. 26, p. 27)

### Experiment 3: Secondary knowledge

`kc_sec_e3`

- **Definition**: Secondary knowledge as in Experiment 1 ($2 payoff condition). (p. 26, p. 27)

### Experiment 3: Tertiary knowledge

`kc_ter_e3`

- **Definition**: Tertiary knowledge as in Experiment 1 ($2 payoff condition). (p. 26, p. 27)

### Experiment 3: Common knowledge

`kc_ck_e3`

- **Definition**: Common knowledge as in Experiment 1 ($2 payoff condition). (p. 26, p. 27)

### Experiment 3: Similar partner

`sc_sim_e3`

- **Definition**: Told they would be matched with a partner who gave the same answers to three or more of the four questions. (p. 28)

### Experiment 3: Dissimilar partner

`sc_dis_e3`

- **Definition**: Told they would be matched with a partner who gave the same answers to two or fewer of the four questions. (p. 28)

### Experiment 3: $2 payoff structure

`pc_2_e3`

- **Definition [derived, to check]**: $2.00 alone / $2.10 together as in Experiment 1. (p. 27, p. 18)
  - how it was derived: Replicates the $2 condition of Experiment 1.

## Materials

### Primary knowledge message (baker version)

`mat_prim`

Conditions: Experiment 1: Primary knowledge; Experiment 3: Primary knowledge

- **Private box (verbatim)** (verbatim, p. 19):
  > “The Messenger Boy has not seen the Butcher today, so he cannot tell you anything about what the Butcher knows.”
- **Public box [to check]**: The public box stated that the loudspeaker was silent. The exact on-screen wording of the silent loudspeaker is not printed. (p. 19)

### Secondary knowledge message (baker version)

`mat_sec`

Conditions: Experiment 1: Secondary knowledge; Experiment 3: Secondary knowledge

- **Private box (verbatim)** (verbatim, p. 19):
  > “The Messenger Boy says he stopped by the butcher shop before coming to your bakery. He tells you that the Butcher knows what today's hot dog price is. However, he says that he forgot to mention to the Butcher that he was coming to see you, so the Butcher is not aware that you know today's hot dog price.”
- **Public box [to check]**: The public box stated that the loudspeaker was silent. The exact on-screen wording is not printed. (p. 19)

### Tertiary knowledge message (baker version)

`mat_ter`

Conditions: Experiment 1: Tertiary knowledge; Experiment 3: Tertiary knowledge

- **Private box (verbatim)** (verbatim, p. 19, p. 20):
  > “The Messenger Boy mentions that he is heading over to the butcher shop, and will let the Butcher know today's price as well. The Messenger Boy will also tell the Butcher that he just came from your bakery and told you the price. However, the Messenger Boy will not inform the Butcher that he told you he would be heading over there. So, while the Butcher is aware that you know today's price, he is not aware that you know that he knows that.”
- **Public box [to check]**: The public box stated that the loudspeaker was silent. The exact on-screen wording is not printed. (p. 20)

### Common knowledge message (baker version)

`mat_ck`

Conditions: Experiment 1: Common knowledge; Experiment 3: Common knowledge

- **Public box (verbatim, with the printed placeholders)** (verbatim, p. 20):
  > “The loudspeaker broadcast the market price of [today’s price] (of which you could earn [earnings for working together]).”
- **Private box (verbatim, with the printed placeholder)** (verbatim, p. 20):
  > “The messenger boy did not come by. Because the market price was broadcast on the loudspeaker, the Butcher knows [today’s price], and he knows that you know this information as well.”
- **Values for [today's price] and [earnings for working together]**: Not stated in the paper. (p. 20, p. 19)

### Butcher-perspective versions

`mat_persp`

- **Butcher version**: Not stated in the paper. (p. 19)

### Comprehension questions

`mat_comp`

- **Printed example payoff question (verbatim)** (verbatim, p. 20):
  > “If you chose to make hot dogs but the butcher did not, then how much would you have made?”
- **Printed example knowledge question (verbatim)** (verbatim, p. 20):
  > “Does the butcher know the price of hot dogs today? (yes/no/can’t tell)”
- **Payoff question set (three questions)** (verbatim, to check, p. 16, p. 17):
  > Baker wording from the supplied replication survey (Exeter, Qualtrics): (1) If you and the Butcher both chose to make hot dogs, then how much would you earn for the interaction? (2) If you chose to make hot dogs but the Butcher did not, then how much would you earn for the interaction? (3) If you chose not to make hot dogs, regardless of what the Butcher decided, then how much money would you earn for the interaction?
- **Knowledge question set (four questions)** (verbatim, to check, p. 17, p. 18):
  > Baker wording from the supplied replication survey: (1) Do you know the price of hot dogs today? (Yes/No) (2) Does the Butcher know the price of hot dogs today? (3) Does the Butcher know that you know the price of hot dogs today? (4) Does the Butcher know that you know that he knows the price of hot dogs today? Questions 2-4: Yes / No / It was not possible to tell.
- **Original question set and answer options**: Not stated in the paper. (p. 20)

### Similarity questions (verbatim)

`mat_sim_q`

- **Question 1** (verbatim, p. 27):
  > “Do you prefer more intense kinds of music (e.g., rock or rap) or more mellow kinds of music (e.g., classical or jazz)?”
- **Question 2** (verbatim, p. 27):
  > “If you had to pick, would you say you are more liberal or more conservative?”
- **Question 3** (verbatim, p. 27):
  > “How old are you? [available answers: “I'm 35 years old or older,” and “I'm younger than 35 years old”]
- **Question 4** (verbatim, p. 27):
  > “When making decisions do you tend to rely more on intuition or more on reason?”

### Similarity feedback and rating

`mat_sim_told`

- **Similar text [to check]**: Participants were told that they would be matched with a partner who gave the same answers to three or more of these questions (wording not printed). (p. 28)
- **Dissimilar text [to check]**: Told they would be matched with a partner who gave the same answers to two or fewer of these questions (wording not printed). (p. 28)
- **Rating [to check]**: Report how similar they perceived their partner to be, on a scale from 0% to 100%; wording not printed. (p. 28)
- **Actual matching**: Not stated in the paper. (p. 28)

### Big Five questionnaire

`mat_big5`

- **Measure**: A standard 50-question survey measuring the Big Five personality traits (Goldberg, 1999). (p. 28)
- **Items and response scale**: Not stated in the paper. (p. 28, p. 57)

### Experiment 3 scenario (as in the $2 condition of Experiment 1)

`mat_e3_instr`

- **Content [derived, to check]**: Same role-play scenario and instructions as the $2 payoff condition of Experiment 1 (see mat_e1_instr). (p. 27, p. 18)
  - how it was derived: Experiment 3 'added three components to the $2 condition from Experiment 1.'

### Knowledge-level comprehension test

`mat_e3_comp`

- **How administered**: On a separate page at the end of the survey, after the Big Five task, without access to the initial instructions or the knowledge information. (p. 26, p. 28)
- **Question wording**: Not stated in the paper. (p. 28)
- **Classification rule**: Responses classified into a reported level of knowledge (primary, secondary, tertiary, common) or 'unclassifiable' (logically inconsistent, incomplete, or 'can't tell' for a level the participant had). (p. 51)

## Procedure

Order of steps:

- Answer four similarity questions (participant sees: Similarity questions (verbatim)) (p. 27)
- Told partner is similar or dissimilar, then rate perceived similarity (0-100%) (participant sees: Similarity feedback and rating) (p. 28)
- Read the $2 butcher-baker scenario (participant sees: Experiment 3 scenario (as in the $2 condition of Experiment 1)) (p. 27)
- Depending on condition: Reveal the day's information for the assigned knowledge condition (p. 27)
  - **Which message [derived]**: Follows the participant's knowledge condition as in Experiment 1. (p. 27)
    - how it was derived: Replication of the Experiment 1 $2 condition.
  - Primary knowledge message shown (participant sees: Primary knowledge message (baker version)) (p. 27)
    - conditions: Experiment 3: Primary knowledge
  - Secondary knowledge message shown (participant sees: Secondary knowledge message (baker version)) (p. 27)
    - conditions: Experiment 3: Secondary knowledge
  - Tertiary knowledge message shown (participant sees: Tertiary knowledge message (baker version)) (p. 27)
    - conditions: Experiment 3: Tertiary knowledge
  - Common knowledge message shown (participant sees: Common knowledge message (baker version)) (p. 27)
    - conditions: Experiment 3: Common knowledge
- Decide to work alone or together (participant sees: Similarity feedback and rating) (p. 28)
- Complete the 50-item Big Five questionnaire (p. 28)
- Answer knowledge-level comprehension questions (information hidden) (p. 28)

### Experiment 3 procedure

`proc_e3`

- **Task**: As in the $2 condition of Experiment 1, with similarity manipulation, Big Five and a harder comprehension test. (p. 27)
- **Visible information**: Knowledge information is concealed while answering the comprehension questions at the end (visible in Experiments 1 and 2). (p. 26)
- **Actions**: Answer four similarity questions, rate similarity, decide, complete the Big Five, answer knowledge comprehension questions. (p. 27, p. 28)
- **Payoff [derived, to check]**: $2.00 alone; $2.10 together if partner also works together; $0 otherwise. (p. 27, p. 52)
  - how it was derived: Inherited from the $2 payoff condition of Experiment 1.

## Variables

### Knowledge state (IV)

`iv_kn_e3`

- **Role**: iv (p. 26)
- **Type**: categorical, 4 levels (p. 28)
- **Observation unit**: participant (p. 27)

### Partner similarity (IV)

`iv_sim_e3`

- **Role**: iv (p. 28)
- **Type**: binary: similar vs. dissimilar (p. 28)
- **Observation unit**: participant (p. 27)

### Tried to work together (DV)

`dv_dec_e3`

- **Role**: dv (p. 28)
- **Type**: binary (p. 28)
- **Observation unit**: participant (p. 27)

### Perceived similarity

`dv_percsim_e3`

- **Role**: mediator (p. 28)
- **Type**: rating 0%-100% (p. 28)
- **Observation unit**: participant (p. 27)

### Agreeableness

`cov_agree_e3`

- **Role**: moderator (p. 29)
- **Type**: continuous Big Five scale score (range 1-5) (p. 57)
- **Observation unit**: participant (p. 27)
- **Scoring**: Not stated in the paper. (p. 28)

### Openness to Experience

`cov_open_e3`

- **Role**: moderator (p. 30)
- **Type**: continuous Big Five scale score (range 1-5) (p. 58)
- **Observation unit**: participant (p. 27)

### Extraversion, Conscientiousness, Neuroticism

`cov_other_e3`

- **Role**: covariate (p. 30)
- **Type**: continuous Big Five scale score (p. 30)
- **Observation unit**: participant (p. 27)

### Reported level of knowledge

`dv_level_e3`

- **Role**: dv (p. 29)
- **Type**: categorical: primary, secondary, tertiary, common, unclassifiable (p. 51)
- **Observation unit**: participant (p. 27)

## Analysis

### Experiment 3: similarity effects

`an_sim_e3`

- **Aim**: Check the similarity manipulation and test the perceived similarity hypothesis. (p. 28)
- **Method**: t-test of perceived similarity by condition; Wald chi-square test of perceived similarity on decisions (logistic); similarity condition collapsed thereafter. (p. 28)

### Experiment 3: knowledge-level comparisons

`an_know_e3`

- **Aim**: Replicate the knowledge effect. (p. 28)
- **Method**: Chi-square tests: common vs tertiary, secondary vs primary, secondary vs tertiary. (p. 28)

### Experiment 3: confusion matrix

`an_conf_e3`

- **Aim**: Test single- versus distinct-representation hypotheses. (p. 29, p. 51)
- **Method**: Planned chi-square comparisons of error rates between conditions; sign test for tertiary mistaken for secondary vs. common. (p. 29, p. 51)

### Experiment 3: Agreeableness

`an_agr_e3`

- **Aim**: Test the altruistic motives hypothesis. (p. 29, p. 30)
- **Method**: Logistic regression of the decision on knowledge condition, Agreeableness and their interaction (Wald chi-square), then post-hoc t-tests of Agreeableness by decision within each knowledge condition. (p. 29, p. 30)

### Experiment 3: Openness

`an_open_e3`

- **Aim**: Test the risk-seeking hypothesis. (p. 30)
- **Method**: Same interaction model and post-hoc t-tests for Openness. (p. 30)

## Reported results

### Experiment 3: similarity

`res_sim_e3`

- **Result kind**: reported (p. 28)
- **Statistics**: Perceived similarity higher in the Similar condition, t(548) = 13.87, p < .001; perceived similarity did not affect decisions, Wald chi2(1, N = 550) = 0.01, p = .944; similarity made no systematic difference. (p. 28)
- **Descriptives [to check]**: Percentages only in Figure 5. (p. 56)

### Experiment 3: knowledge effect

`res_know_e3`

- **Result kind**: reported (p. 28)
- **Statistics**: Common vs tertiary: chi2(1, N = 270) = 22.28, p < .001, phi = .29; secondary vs primary: chi2(1, N = 280) = 16.87, p < .001, phi = .25; secondary vs tertiary: chi2(1, N = 284) = 0.72, p = .397, phi = .05. (p. 28)

### Experiment 3: confusion matrix (Table 3)

`res_conf_e3`

- **Result kind**: reported (p. 29, p. 51)
- **Proportions (rows = actual condition; columns = reported level primary, secondary, tertiary, common, unclassifiable)**: {"Primary": [0.931, 0.008, 0.008, 0.008, 0.046], "Secondary": [0.02, 0.899, 0.013, 0.007, 0.06], "Tertiary": [0, 0.23, 0.637, 0.044, 0.089], "Common": [0.007, 0, 0.015, 0.837, 0.141]} (p. 51)
- **Statistics**: More errors with tertiary than any other level: primary-tertiary chi2(1, N=266)=33.76, ϕ=.36; secondary-tertiary chi2(1, N=284)=27.91, ϕ=.31; common-tertiary chi2(1, N=270)=13.93, ϕ=.23 (all p<.001); secondary vs common errors n.s., chi2(1, N=284)=2.43, p=.119, ϕ=.09; tertiary taken for common 4%, differs from 23% mistaken for secondary (p<.001, sign test). (p. 29, p. 51)

### Experiment 3: Agreeableness

`res_agr_e3`

- **Result kind**: reported (p. 29, p. 30)
- **Statistics**: Knowledge x Agreeableness interaction, Wald chi2(3, N=550)=8.18, p=.043; coordinators more agreeable with secondary (t(147)=2.25, p=.013) and tertiary (t(133)=1.89, p=.030), not with primary or common (p>.60). (p. 29, p. 30)

### Experiment 3: Openness

`res_open_e3`

- **Result kind**: reported (p. 30)
- **Statistics**: Wald chi2(3, N=550)=13.64, p=.003; coordinators more open with secondary (t(147)=3.30, p<.001) and tertiary (t(133)=1.82, p=.036), not with primary or common (p>.80). No differences for the other three factors (all ps>.2). (p. 30)

## Simulation scope

### Experiment 3 simulation scope

`sim_e3`

- **Steps [implementation]**: Similarity questions, similarity statement and rating, scenario, decision, Big Five, comprehension test per AI participant.
- **Roles [implementation]**: One AI participant per original participant; the partner is described only.
- **Conditions [implementation]**: 4 knowledge x 2 similarity, $2 payoff; Big Five items are not printed in the paper.
- **Outcomes [implementation]**: Decision, perceived similarity, Big Five responses and level classification; Big Five scoring requires the original items.
- **Limits [implementation]**: No real stakes; AI Big Five answers are not a personality sample.

## Open issues

- **Eligibility not stated** [check, `iss_e3_elig`, part_e3]: Not stated. Suggested: None.
- **Assignment mechanism not described** [decision, `iss_e3_assign`, design_e3]: Evenly distributed across conditions only. Suggested: Confirm.
- **Condition sizes derived from pairwise Ns** [check, `iss_e3_n`, part_e3]: Per-condition analysed n derived arithmetically. Suggested: Confirm with data.
- **Order of similarity steps relative to scenario** [check, `iss_e3_order`, design_e3]: Similarity questions are 'at the beginning'; whether the statement and rating come before the scenario, and where the explanation/payoff comprehension items sit, is not stated. Suggested: Confirm.
- **Partner matching truth** [decision, `iss_e3_match`, mat_sim_told]: The paper does not say whether participants were really matched on answers. Suggested: Decide how to present.
- **Similarity statement wording** [check, `iss_e3_simtext`, mat_sim_told]: Wording not printed. Suggested: Supply original.
- **Dissimilar statement wording** [check, `iss_e3_simtext2`, mat_sim_told]: Wording not printed. Suggested: Supply original.
- **Rating prompt wording** [check, `iss_e3_rate`, mat_sim_told]: Not printed. Suggested: Supply original.
- **Big Five items and scale not printed** [decision, `iss_e3_big5`, mat_big5]: The 50 items and response format are not in the paper. Suggested: Supply the Goldberg (1999) 50-item inventory and scale from the researcher; no web lookup was done.
- **Big Five scoring not printed** [decision, `iss_e3_big5score`, cov_agree_e3]: Scoring not printed. Suggested: Supply scoring key.
- **Experiment 3 comprehension wording** [decision, `iss_e3_compitems`, mat_e3_comp]: Wording of the harder test not printed. Suggested: Approve or supply.
- **Scenario text inherited from Experiment 1** [check, `iss_e3_instr`, mat_e3_instr]: Derived from 'the $2 condition from Experiment 1'. Suggested: Confirm.
- **Per-cell percentages only in figures** [check, `iss_e3_cells`, res_sim_e3]: Figures 5-7. Suggested: Read figures.
- **Reported-level classification rule is an implementation reading** [decision, `iss_e3_levelrule`, mat_e3_comp]: Table 3 note describes unclassifiable patterns but not the exact answer pattern per level. Suggested: Confirm or supply the original coding.
- **AI participants cannot forget the earlier screens** [check, `iss_e3_memory`, mat_e3_comp]: Experiment 3 hid the information to make the comprehension test hard by relying on memory; a language model that receives its earlier screens in context has perfect recall. Suggested: Treat Table 3 replay as exploratory.
