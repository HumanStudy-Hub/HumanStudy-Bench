# Multisite replication of Risen & Gilovich (2008) Study 6 (tempting fate × cognitive load)

`s1`

Between-subjects 2 (tempting fate: tempted vs. not tempted) × 2 (cognitive load: counting backward vs. none) experiment run at 4 similar U.S. universities (updated protocol), 6 dissimilar universities, and on Amazon Mechanical Turk (RP:P protocol).

## Hypotheses

### H1: main effect of tempting fate

`h1`

- **Statement**: Subjects who imagine having tempted fate judge an ironic bad outcome (being called on in class) as more likely than subjects who have not tempted fate. Tested in subjects not under cognitive load; reported as the main effect. (p. 1, p. 2)

### H2 (focal): Tempting Fate × Cognitive Load interaction

`h2`

- **Statement**: Positive interaction of cognitive load and tempting fate on the perceived likelihood of an ironic bad outcome. (p. 2)
- **Scope**: The primary replication test is the interaction within the similar sites, and its difference from MTurk (three-way interaction). (p. 4)

## Participants

### Participants (AI participants replace human subjects in replay)

`p1`

- **Setting**: Updated protocol: controlled settings with reasonable isolation from other subjects (private lab room or cubicles); MTurk: online, no restriction on physical setting. (p. 3, p. 5)
- **Roles**: Single role: subject (every original participant is replayed as one AI participant). No confederates or scripted roles are described. (p. 2)
  - how it was derived: Paper describes only subjects reading a scenario; no other roles appear.
- **Recruitment and compensation by site**: MTurk: paid U.S. online workers. Penn: paid undergraduates from subject pool. Berkeley: undergraduate business majors given credit. UVA, Stanford: introductory-psychology students given credit. ELTE: psychology undergraduates given credit (Hungarian). Leuven: subject pool, credit or pay (Dutch). Porto: intro-psychology, no compensation (Portuguese). BYU-Idaho: intro-psychology, credit and raffle entry. URI: psychology undergraduates, credit. Rose-Hulman: uncompensated peers of research assistants. (p. 5)
- **Site eligibility (similar sites)**: Primary analyses: U.S. university sites with median SAT above the 90th percentile nationally (similar sites); other universities, or universities outside the U.S., are dissimilar sites. (p. 3, p. 4)
- **Analysed n by site type**: Similar sites 754 (Penn 335, Berkeley 200, UVA 151, Stanford 68); dissimilar sites 714 (ELTE 284, Leuven 118, Porto 91, BYU-Idaho 84, URI 81, Rose-Hulman 56); MTurk 2,973. (p. 6, p. 5)
- **Excluded n by site**: MTurk 162; Penn 24; Berkeley 23; UVA 5; Stanford 1; ELTE 7; Leuven 9; Porto 13; BYU-Idaho 6; URI 9; Rose-Hulman 2. (p. 5)
- **Number recruited / randomized**: Not stated in the paper. (p. 5)
- **A priori exclusion criteria**: Not stated in the paper. (p. 5, p. 3)
- **Subjects per condition**: Not stated in the paper. (p. 4)
- **Language of materials**: English at English-speaking sites; translated (verified by independent back-translation) at sites where subjects were not expected to speak fluent English: ELTE Hungarian, Leuven Dutch, Porto Portuguese. (p. 4, p. 5)
- **Unit of analysis**: Individual subject; subjects are clustered within site. (p. 8)
  - how it was derived: Mixed-model random effects by site imply subject-level observations nested in sites.
- **Sample-size criterion**: More than 80% power to detect a three-way interaction with SMD above 0.75; sites tried to reach their share but often could not. (p. 4)
- **Ethics**: Ethics-committee approval obtained where appropriate; Declaration of Helsinki. (p. 3)

## Design

### Design

`d1`

- **Type**: Between-subjects 2 (tempting fate) × 2 (cognitive load) factorial, replicated across sites; site type (similar / dissimilar / MTurk protocol) is a non-randomized between-site factor. (p. 2, p. 8)
- **Assignment**: Random assignment to tempting-fate scenario and, independently, to cognitive load. (p. 2)
- **Randomization procedure and allocation ratio**: Not stated in the paper. (p. 2)
- **Blinding**: Analyses by the first author, blinded to results until all sites finished data collection; audited by other authors. No subject blinding described. (p. 4)
- **Manipulation check**: Load-condition subjects rated effort and difficulty of the counting task (0–10). (p. 4)
- **Updated protocol vs. RP:P protocol**: Updated protocol: university undergraduates in controlled, isolated lab settings (similar sites). RP:P protocol: MTurk, no restriction on physical setting. Original: Cornell undergraduates in a low-distraction private lab. (p. 3, p. 4)
- **Order of conditions**: Between-subjects: each subject sees one scenario and one load condition. Experiment run alone or after pre-approved experiments (Many Labs 5). (p. 2, p. 20)

## Conditions

### Not tempted fate, no cognitive load

`c00`

- **Definition**: not tempted fate (read scenario in which the assigned reading was done); no counting task (p. 2)

### Tempted fate, no cognitive load

`c10`

- **Definition**: tempted fate (read scenario in which the assigned reading was not done); no counting task (p. 2)

### Not tempted fate, cognitive load

`c01`

- **Definition**: not tempted fate; counted backward by 3s from a large number while reading (p. 2)

### Tempted fate, cognitive load

`c11`

- **Definition**: tempted fate; counted backward by 3s from a large number while reading (p. 2)

## Materials

### Scenario text (tempted / not tempted fate)

`m_scn`

- **Tempted-fate scenario, full text (had not done the reading)** (verbatim, p. 2):
  > Imagine that you are in a large lecture with a few hundred students and you are sitting in the middle section, a little more than half-way back in the room. The professor asks a question about the readings, but no one raises his or her hand to answer. You have not done the reading and feel confident that you would not be able to answer the question. The class sits in silence for two minutes before the professor explains that if no one volunteers, he will choose someone.
- **Not-tempted-fate scenario, full text (had done the reading)** (verbatim, p. 2):
  > Imagine that you are in a large lecture with a few hundred students and you are sitting in the middle section, a little more than half-way back in the room. The professor asks a question about the readings, but no one raises his or her hand to answer. You have done the reading and feel confident that the professor would like your answer, but prefer not to volunteer answers in large classes. The class sits in silence for two minutes before the professor explains that if no one volunteers, he will choose someone.
- **Description**: Subjects imagine themselves in a class either having not done the assigned reading (tempted fate) or having done it (not tempted fate), then judge the likelihood of being called on. (p. 2)
- **Lead-in above the scenario, load conditions** (verbatim, p. 2):
  > Start with the number: 564
  > While counting backwards, read the following scenario.
- **Lead-in above the scenario, no-load conditions** (verbatim, p. 4):
  > Please read the following scenario carefully.

### Cognitive-load task (counting backward)

`m_load`

- **Description**: Count backward by 3s from a large number while reading the scenario. (p. 2)
- **Counting-task instruction wording (pre-scenario page, load conditions)** (verbatim, p. 1):
  > Please read the directions carefully. In this short study, you will be asked to complete two tasks simultaneously. Your first task will be to read the scenario on the following page and answer the question that follows it. Your second task will be to simultaneously count backwards by 3s. Both tasks are very important, so you should be sure to put effort into reading and into counting backwards. After you finish with the scenario, you will have to report the number that you end on. For example, if you are asked to start with 876 and you count backwards by 3s (873, 870, 867…) and finish reading the scenario and answering the question while you are on 819, you should write 819 at the bottom of the page. If you have been counting carefully, the number you report should be a multiple of three away from the original number.  The number that you should start with will appear on the top of the following page. Begin counting backwards by 3s immediately, and then proceed to the scenario. When you are ready to complete the two tasks, please proceed to the next page.
- **Starting number** (verbatim, p. 2, p. 1):
  > 564
- **End-number question (after the likelihood item, load conditions)** (verbatim, p. 3):
  > STOP COUNTING! What number did you end on?
- **Effort-split item with 0 to 6 scale (load conditions)** (verbatim, p. 3):
  > It is often very difficult to complete two tasks simultaneously. Please indicate how much effort you believe you put into each of the two tasks by choosing the appropriate number.
  > I put all my effort into reading | I split my effort evenly between the two tasks | I put all my effort into counting
  > 0 1 2 3 4 5 6

### Likelihood judgment item

`m_lkl`

- **Description**: Judgment of the likelihood of being called on in class, scale from 0 to 10 (11 options). (p. 2, p. 20)
- **Exact item wording and scale anchors** (verbatim, p. 2):
  > How likely do you believe it is that the professor will call on you? Mark your answer by choosing the appropriate number below.
  > Not at all likely | Extremely likely
  > 0 1 2 3 4 5 6 7 8 9 10

### New measures developed with the original authors (verbatim)

`m_new`

- **Effort (load condition only)** (verbatim, p. 3, p. 4):
  > How much effort did the counting task require?
  > Very little effort | A lot of effort
  > 0 1 2 3 4 5 6 7 8 9 10
- **Difficulty (load condition only)** (verbatim, p. 3, p. 4):
  > How difficult was the counting task?
  > Not at all difficult | Extremely difficult
  > 0 1 2 3 4 5 6 7 8 9 10
- **Importance of answering correctly (questionnaire wording)** (verbatim, p. 5, p. 4):
  > If you were a student in the class you just read about, how important would it be to you to answer questions correctly in class?
  > Not at all important | Extremely important
  > 0 1 2 3 4 5 6 7 8 9 10
- **Negativity of answering incorrectly** (verbatim, p. 5, p. 4):
  > If you were a student in the class, how bad would you feel if you were called on by the professor, but couldn’t answer the question?
  > Not at all bad | Extremely bad
  > 0 1 2 3 4 5 6 7 8 9 10
- **Response scale**: 0 to 10 for all four items. (p. 4)
- **Omission**: Effort and difficulty items were omitted at the University of California at Berkeley because of a data-collection error; Berkeley also did not administer the importance/negativity items. (p. 11)

## Procedure

Order of steps:

- Random assignment to scenario and load condition (p. 2)
  - when then:
    - Depending on condition: Read the imagined class scenario (tempted or not tempted fate) (participant sees: Scenario text (tempted / not tempted fate)) (p. 2)
      - Read scenario: not having done the assigned reading (tempted fate) (participant sees: Scenario text (tempted / not tempted fate)) (p. 2)
        - conditions: Tempted fate, no cognitive load; Tempted fate, cognitive load
      - Read scenario: having done the assigned reading (not tempted fate) (participant sees: Scenario text (tempted / not tempted fate)) (p. 2)
        - conditions: Not tempted fate, no cognitive load; Not tempted fate, cognitive load
      - when then judge:
        - Judge the likelihood of being called on in class (0–10) (participant sees: Likelihood judgment item) (p. 2)
          - when if cognitive load:
            - Report the ending number and the reading/counting effort split (load conditions only) (participant sees: Cognitive-load task (counting backward)) (p. 3)
              - conditions: Not tempted fate, cognitive load; Tempted fate, cognitive load
              - when then:
                - Rate effort and difficulty of the counting task (load conditions only) (participant sees: New measures developed with the original authors (verbatim)) (p. 4)
                  - conditions: Not tempted fate, cognitive load; Tempted fate, cognitive load
                  - when then:
                    - Rate importance of answering correctly and negativity of answering incorrectly (participant sees: New measures developed with the original authors (verbatim)) (p. 4)
          - when if no load:
            - (back to: Rate importance of answering correctly and negativity of answering incorrectly)
  - when if cognitive load:
    - Count backward by 3s from a large number while reading (participant sees: Cognitive-load task (counting backward)) (p. 2)
      - conditions: Not tempted fate, cognitive load; Tempted fate, cognitive load
      - **Concurrency with reading [to check]**: Counting occurs while reading the scenario; further timing rules not stated. (p. 2)
      - when then:
        - (back to: Judge the likelihood of being called on in class (0–10))

### Procedure

`pr1`

- **Task**: Read an imagined class scenario, (under load) count backward by 3s meanwhile, then judge the likelihood of being called on in class. (p. 2)
- **What the subject sees**: Own scenario only; no other subjects' responses. Between-subjects, individual completion. (p. 2)
  - how it was derived: Between-subjects design with independent subjects.
- **Feedback**: None reported. (p. 2)
  - how it was derived: No feedback is described in the paper.
- **Payoff**: No performance-contingent payoff reported; compensation varies by site. (p. 5)
- **Order of likelihood item and later measures [derived]**: As printed in the questionnaire: consent; (load only) directions page; scenario page with likelihood item; (load only) end number, effort split, difficulty, effort; importance and negativity. (p. 1, p. 2, p. 3, p. 5)
  - how it was derived: Page order of sources/RG_questionnaire_items.pdf (pages 1 to 5).
- **Complete questionnaire and instructions [to check]**: Printed in sources/RG_questionnaire_items.pdf except the consent text, which the file shows only as the placeholder “(Insert consent text here.)”. (p. 1)

## Variables

### Tempting fate

`v_tf`

- **role**: iv (p. 2)
- **type**: binary (0 = not tempted, 1 = tempted) (p. 17)
- **observation-unit**: subject (p. 8)

### Cognitive load

`v_load`

- **role**: iv (p. 2)
- **type**: binary (0 = none, 1 = counting backward) (p. 17)
- **observation-unit**: subject (p. 8)

### Site type / protocol

`v_site`

- **role**: moderator (p. 4)
- **type**: categorical: similar university (updated protocol), dissimilar university, MTurk (RP:P protocol) (p. 8)
- **observation-unit**: subject (site-level) (p. 8)

### Perceived likelihood of being called on in class

`v_lkl`

- **role**: dv (p. 2)
- **type**: ordinal rating, 0–10 (p. 2, p. 20)
- **scale**: 11-point scale, 0 to 10 (p. 20)
- **observation-unit**: subject (p. 8)

### Perceived effort of counting task

`v_effort`

- **role**: manipulation check / secondary measure (p. 4)
- **type**: rating 0–10 (p. 4)
- **observation-unit**: subject (p. 8)

### Perceived difficulty of counting task

`v_diff`

- **role**: manipulation check / secondary measure (p. 4)
- **type**: rating 0–10 (p. 4)
- **observation-unit**: subject (p. 8)

### Perceived importance of answering correctly

`v_imp`

- **role**: manipulation check / secondary measure (p. 4)
- **type**: rating 0–10 (p. 4)
- **observation-unit**: subject (p. 8)

### Perceived negativity of answering incorrectly

`v_neg`

- **role**: manipulation check / secondary measure (p. 4)
- **type**: rating 0–10 (p. 4)
- **observation-unit**: subject (p. 8)

## Analysis

### Within-site OLS regression

`a_site`

- **Aim**: Estimate the main effect of tempting fate and the focal interaction in each site. (p. 4)
- **Method**: OLS of perceived likelihood on tempting-fate condition, cognitive-load condition and their interaction; statistically equivalent to the original ANOVA. (p. 4)
- **Original-style ANOVA**: Original two-way ANOVA also refitted as a sensitivity analysis (Supplement). (p. 19)

### Primary mixed model (similar sites + MTurk)

`a_primary`

- **Aim**: Estimate the focal interaction at similar sites and under the RP:P protocol, and their difference. (p. 4)
- **Method**: Linear mixed model with fixed effects of tempting fate, cognitive load, protocol and their interactions; by-site random intercepts and random slopes for tempting fate, cognitive load and their interaction. (p. 8)
- **Sample**: 3,727 subjects (similar sites and MTurk). (p. 8)

### Secondary mixed models (all universities)

`a_sec`

- **Aim**: Same questions as the primary analyses, adding dissimilar sites and testing the similar–dissimilar difference. (p. 8)
- **Method**: Site type as categorical (MTurk, similar, dissimilar); N = 4,441. (p. 8)

### Evaluation of proposed explanations for the RP:P failure

`a_mech`

- **Aim**: Test whether load was less effective on MTurk (three-way interaction with MTurk indicator; effort/difficulty ratings among load subjects) and whether the scenario was less salient (importance and negativity ratings by site type). (p. 9, p. 10)
- **Method**: Mixed models; planned analysis was simplified post hoc because similar and dissimilar sites did not differ. (p. 9)
- **SAT moderator**: Linear effect of university median SAT on main effect and interaction (US universities only). (p. 10)
- **Statistical consistency with the original**: Post hoc check whether the original estimates are plausible draws from the replication distribution. (p. 8)

## Reported results

### Original study 6 estimates (as restated)

`r_orig`

- **result_kind**: reported (p. 2)
- **Main effect and interaction**: Main effect b = 1.03 [0.09, 1.97], p = .03 (n = 120); interaction 1.54 [0.05, 3.03], p = .04. (p. 6, p. 7)

### Within-site and pooled estimates (Figures 1–2)

`r_sites`

- **result_kind**: reported (p. 6)
- **Main effect**: MTurk 0.21 [−0.01, 0.44] p = .07; similar sites 0.11 [−0.34, 0.56] p = .64; dissimilar sites 0.42 [−0.07, 0.90] p = .09. (p. 6)
- **Interaction**: Similar sites −0.02 [−0.68, 0.63] p = .94; dissimilar sites −0.28 [−1.01, 0.45] p = .45; ELTE −0.99 [−1.96, −0.01] p = .05. (p. 7)

### Table 3: primary mixed model

`r_t3`

- **result_kind**: reported (p. 8)
- **Main effect**: MTurk 0.21 [−0.01, 0.43] p = .06; similar sites 0.11 [−0.34, 0.56] p = .64; site-type difference −0.11 [−0.61, 0.39] p = .68. (p. 8)
- **Interaction**: MTurk −0.20 [−0.53, 0.13] p = .24; similar sites −0.02 [−0.68, 0.63] p = .94; site-type difference 0.18 [−0.56, 0.91] p = .64. N = 3,727. (p. 8)

### Table 4 and S3: all-university models

`r_t4`

- **result_kind**: reported (p. 9)
- **Estimates**: Dissimilar sites main effect 0.42 [−0.07, 0.90] p = .09; interaction −0.28 [−1.01, 0.45] p = .45; similar sites interaction 0.01 [−0.73, 0.76] p = .97. All universities combined: main effect 0.28 [−0.08, 0.63] p = .12. (p. 9, p. 18)

### Cell means and SDs, similar sites (Table S2, naïvely pooled)

`r_cells`

- **result_kind**: reported (p. 17)
- **Descriptives**: Perceived likelihood M (SD): no tempt/no load 2.26 (1.89); tempt/no load 2.37 (1.96); no tempt/load 3.38 (2.16); tempt/load 3.47 (2.19). (p. 17)

### Secondary mechanism analyses

`r_mech`

- **result_kind**: reported (p. 9)
- **Load effectiveness: MTurk vs. university**: Three-way interaction b = −0.03 [−0.99, 0.93], p = .95 (N = 4,441). (p. 9)
- **Effort / difficulty (MTurk vs. university)**: Effort b = 0.63 [−0.42, 1.68], p = .24; difficulty b = 0.51 [−0.11, 1.14], p = .11. (p. 9)
- **SAT moderator**: Did not moderate main effect or interaction. (p. 10)
- **Consistency with original**: Probability the original main effect would be at least as extreme: .12 (similar sites). (p. 8)

## Simulation scope

### Simulation scope

`sim1`

- **Steps [implementation]**: Read scenario, counting task (load conditions), likelihood judgment, and the four 0–10 secondary measures can be replayed as text. Blocked until the scenario, counting-task instruction and exact item wording are supplied (not printed in the paper).
- **Roles [implementation]**: One AI participant per original subject; no scripted roles.
- **Conditions [implementation]**: Four cells (tempting fate × cognitive load). Site type, language and lab setting cannot be varied faithfully for text-only AI participants; the primary replay target is the similar-site, English population.
- **Outcomes [implementation]**: Likelihood rating 0–10; secondary ratings. Cognitive load is only nominal for a text model: counting backward cannot occupy working memory, so any load effect is not comparable with human subjects.
- **Limits [implementation]**: Physical lab context, real distraction and compensation cannot be reproduced. Agreement with the paper would not show that people behave the same way.

## Open issues

- **Number recruited not reported** [check, `i_nrec`, p1]: Only analysed and excluded counts are in Table 2. Suggested: Take from the OSF data or leave as is. (p. 5)
- **A priori exclusion criteria not listed in the paper** [check, `i_excl`, p1]: The paper refers to preregistered criteria without stating them. Suggested: Supply from the preregistration (https://osf.io/8y6st/). (p. 5)
- **Per-condition sample sizes not reported** [check, `i_ncond`, p1]: Only per-site analysed n is given. Suggested: Take from the OSF data. (p. 4)
- **Randomization procedure and allocation ratio not stated** [check, `i_ratio`, d1]: The paper says random assignment only. Suggested: Confirm equal allocation or supply the procedure. (p. 2)
- **Consent text not supplied** [check, `i_full`, pr1]: The questionnaire PDF shows the consent screen only as the placeholder “(Insert consent text here.)”. Suggested: Supply the site consent text if it should be shown to participants. (p. 1)
- **Timing rule of counting task relative to reading** [check, `i_step_load`, s1]: The paper says 'while reading' but gives no further rule (e.g., duration, response mode). Suggested: Confirm how load was delivered. (p. 2)
- **Importance item wording differs between the paper and the questionnaire** [check, `i_wording_imp`, m_new]: The paper quotes “in the scenario you just read about, how important would it be for you…”; the questionnaire prints “in the class you just read about, how important would it be to you…”. Suggested: Confirm that the printed questionnaire is the version that was administered. (p. 4, p. 5)
- **Protocol says new items come last; questionnaire prints difficulty and effort earlier** [check, `i_order_conflict`, pr1]: The protocol says the new items appear at the very end; in the printed questionnaire the difficulty and effort items follow the end-number question on page 3 and only importance and negativity are last. Suggested: Confirm which order was administered. (p. 1, p. 3)
