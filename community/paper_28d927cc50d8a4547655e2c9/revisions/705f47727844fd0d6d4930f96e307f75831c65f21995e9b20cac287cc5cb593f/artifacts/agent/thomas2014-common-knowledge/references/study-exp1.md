# Experiment 1: two-player butcher-baker coordination game (4 knowledge x 4 payoff, between subjects)

`exp1`

MTurk participants, 1600 recruited, 1033 analysed; each plays butcher or baker and decides whether to work alone or together given one of four knowledge states and one of four payoff structures.

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

### Rational actor hypothesis

`hyp_rat`

- **Statement**: As the ratio of costs to benefits increases, people will be less likely to try to work together. (p. 15)
- **Scope**: Experiment 1, Experiment 2 (p. 15)

### Knowledge-level heuristic hypothesis

`hyp_heur`

- **Statement**: Participants' decisions will not closely track the cost-benefit ratio of different payoffs. (p. 15)
- **Scope**: Experiment 1, Experiment 2 (p. 15)

## Participants

### Experimenter / interface (scripted role)

`scripted_researchers`

- **Role**: Experimenter and survey interface: presents the printed instructions, the private (messenger boy) and public (loudspeaker) boxes, and the comprehension questions; performs the offline random pairing and payment after data collection. (p. 19, p. 20)
- **Scripted [implementation]**: Yes: this is the paper's own scripted role (an interface and offline payment); it is not an original participant.

### Experiment 1 participants

`part_e1`

- **Setting**: Online study on Amazon Mechanical Turk; participants role-play a merchant on screen and are paid real money. (p. 18)
- **Roles**: Each participant plays either the butcher or the baker; a partner (the other role) is another participant paired offline. (p. 17, p. 18, p. 20)
- **Recruitment**: Amazon Mechanical Turk, from the United States, for a short study for a small payment. (p. 18)
- **Eligibility**: Located in the United States (MTurk). No other criteria reported. (p. 18)
- **Recruited**: 1600 (100 per condition, 16 conditions). (p. 18)
- **Completed**: Not stated in the paper.
- **Analysed**: 1033 after excluding participants who gave incorrect answers to the comprehension questions about the game's payoff structure (cell sizes in Table 1 sum to 1033: 276+272+236+249). (p. 18, p. 49)
- **Characteristics**: 58% female; mean age 32.8 years (SD = 15.0). (p. 18)
- **Sampling unit**: The individual participant (one decision each); pairing is arranged offline only to compute payments. (p. 20)
- **Compensation**: Minimum 50 cents; bonus depends on own and partner's decisions, paid after offline random pairing. (p. 18, p. 20)
- **Attrition and exclusion rule [to check]**: Excluded those with incorrect answers to the payoff comprehension questions; the exact criterion (any error vs. a threshold) and the number excluded per condition are not stated. (p. 18, p. 20)

## Design

### Experiment 1 design

`design_e1`

- **Design type**: 4 (knowledge state) x 4 (payoff structure) fully crossed between-subjects design, 16 cells of 100 recruited participants each. (p. 17, p. 18)
- **Control [derived]**: Primary knowledge condition is the baseline with no information about the partner's knowledge. (p. 17)
  - how it was derived: Primary knowledge is the lowest level of knowledge; the paper does not label a control condition explicitly.
- **Assignment**: Not stated in the paper. (p. 17)
- **Order**: Fixed sequence for all participants: instructions, reveal of the day's hot dog information, decision, explanation, comprehension questions, demographics. (p. 19, p. 20)
- **Blinding**: Not stated in the paper.
- **Manipulation check**: Four comprehension questions about what the participant and partner knew; used in Experiment 1 only to exclude payoff-comprehension failures; the paper states all participants answered them correctly in Experiments 1 and 2. (p. 20, p. 26)

## Conditions

### Experiment 1: Primary knowledge

`kc_prim_e1`

- **Definition**: Primary knowledge manipulation (what the participant is told about the partner's knowledge of the day's payoff). (p. 17)
- **Text shown**: See the matching material node. (p. 17)

### Experiment 1: Secondary knowledge

`kc_sec_e1`

- **Definition**: Secondary knowledge manipulation (what the participant is told about the partner's knowledge of the day's payoff). (p. 17)
- **Text shown**: See the matching material node. (p. 17)

### Experiment 1: Tertiary knowledge

`kc_ter_e1`

- **Definition**: Tertiary knowledge manipulation (what the participant is told about the partner's knowledge of the day's payoff). (p. 17)
- **Text shown**: See the matching material node. (p. 17)

### Experiment 1: Common knowledge

`kc_ck_e1`

- **Definition**: Common knowledge manipulation (what the participant is told about the partner's knowledge of the day's payoff). (p. 17)
- **Text shown**: See the matching material node. (p. 17)

### Experiment 1: payoff $1

`pc_1_e1`

- **Definition**: $1.00 alone / $1.10 together (p. 18, p. 52)

### Experiment 1: payoff $2

`pc_2_e1`

- **Definition**: $2.00 alone / $2.10 together (p. 18, p. 52)

### Experiment 1: payoff $5

`pc_5_e1`

- **Definition**: $5.00 alone / $5.10 together (p. 18, p. 52)

### Experiment 1: payoff $10

`pc_10_e1`

- **Definition**: $10.00 alone / $10.10 together (p. 18, p. 52)

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

### Experiment 1 instructions (as reported)

`mat_e1_instr`

- **Instruction content (paraphrase of the paper's description) [to check]**: Participants read that they would earn a minimum of 50 cents, which they could augment based on their decisions in their interaction with another participant on Mechanical Turk. One of them would play a butcher and the other a baker. Each could either work alone for a sure profit (the butcher could make chicken wings, the baker dinner rolls) or attempt to work with their partner, the butcher making hot dogs, the baker the buns. By choosing to work together the participant can earn a profit, but only if the partner also chooses to work together; if either decides to collaborate but the partner does not, that person doesn't earn anything, because they cannot sell a bun without a hot dog or vice versa. They would earn a certain amount ($1, $2, $5, or $10) if they decided to work alone, but the hot dog price varied from day to day, and thus the earnings for working together might be more than or less than this sure profit. The information about hot dog earnings might be conveyed by a messenger boy (displayed on their screen in a private box that only they could see), or by a loudspeaker (displayed in a public box that the other participant could see on his or her screen as well). (p. 18, p. 19)
- **Printed instruction wording**: Not stated in the paper. (p. 18)

### Reveal of the day's payoff

`mat_e1_reveal`

- **Reveal**: The participant clicked a button to reveal the day's information about the price of hot dogs and hence the potential profit for collaborating; in each case it was ten cents more than each would earn by working alone. (p. 19)
- **On-screen wording of the payoff reveal in the primary, secondary and tertiary conditions**: Not stated in the paper. (p. 19)

### Decision and explanation prompts

`mat_e1_resp`

- **Decision options**: Work alone or work with the partner, indicated with the keyboard. (p. 20)
- **Printed decision prompt**: Not stated in the paper. (p. 20, p. 10)
- **Open-ended explanation [to check]**: Asked to explain how they made the decision; wording not printed. (p. 20)

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

### Demographic questionnaire

`mat_demo`

- **Items**: Not stated in the paper. (p. 20)

## Procedure

Order of steps:

- Read instructions about roles, profits and information sources (participant sees: Experiment 1 instructions (as reported)) (p. 18, p. 19)
- Depending on condition: Click to reveal the day's hot dog information for the assigned knowledge condition (p. 19, p. 17)
  - **Which message is shown**: The message shown follows the participant's assigned knowledge condition: primary, secondary, tertiary or common. (p. 19, p. 17)
  - Primary knowledge message shown (participant sees: Primary knowledge message (baker version)) (p. 17, p. 19)
    - conditions: Experiment 1: Primary knowledge
  - Secondary knowledge message shown (participant sees: Secondary knowledge message (baker version)) (p. 17, p. 19)
    - conditions: Experiment 1: Secondary knowledge
  - Tertiary knowledge message shown (participant sees: Tertiary knowledge message (baker version)) (p. 17, p. 19)
    - conditions: Experiment 1: Tertiary knowledge
  - Common knowledge message shown (participant sees: Common knowledge message (baker version)) (p. 17, p. 20)
    - conditions: Experiment 1: Common knowledge
- Decide to work alone or work together (keyboard) (participant sees: Primary knowledge message (baker version); Secondary knowledge message (baker version); Tertiary knowledge message (baker version); Common knowledge message (baker version)) (p. 20)
  - **Visible at decision [to check]**: The participant still sees only the information for their own knowledge condition (the experiments 1-2 information was visible on screen during later questions). (p. 26, p. 20)
- Explain how the decision was made (p. 20)
- Answer three payoff comprehension questions (p. 20)
- Answer four knowledge comprehension questions (participant sees: Primary knowledge message (baker version); Secondary knowledge message (baker version); Tertiary knowledge message (baker version); Common knowledge message (baker version)) (p. 20, p. 26)
- Complete brief demographic questionnaire and submit (p. 20)
- Offline random pairing, profit calculation and bonus payment (scripted) (p. 20)

### Experiment 1 procedure

`proc_e1`

- **Task**: Role-played Stag Hunt: choose whether to work alone (certain profit) or to attempt to work with the partner (higher profit only if partner also chooses to work together; otherwise nothing). (p. 18, p. 12)
- **Goal [derived]**: Earn as much money as possible (bonus). (p. 18)
  - how it was derived: Participants are paid a bonus based on decisions; goal wording is not printed.
- **Actions**: Work alone or work together; open-text explanation; comprehension answers; demographics. (p. 20)
- **Visible information**: Own role, own sure profit, the day's potential profit, and the information box content for their knowledge condition. In Experiments 1 and 2 the knowledge information remained visible on screen during the comprehension questions. (p. 19, p. 26)
- **Feedback**: None reported before payment; pairing and payoffs computed offline. (p. 20)
- **Payoff**: Working alone: sure profit ($1, $2, $5 or $10). Both work together: sure profit + $0.10. Works together but partner alone: $0. Plus minimum 50 cents. (p. 52, p. 18)
- **Stopping [derived]**: Single one-shot decision per participant. (p. 20)
  - how it was derived: The paper describes one decision for the day; no repeated rounds are reported.

## Variables

### Knowledge state (IV)

`iv_kn_e1`

- **Role**: iv (p. 17)
- **Type**: categorical, 4 levels: primary, secondary, tertiary, common (p. 17)
- **Observation unit**: participant (p. 18)

### Payoff structure (IV)

`iv_pay_e1`

- **Role**: iv (p. 18)
- **Type**: categorical, 4 levels ($1, $2, $5, $10 alone; +$0.10 together) (p. 18)
- **Unit**: US dollars (p. 18)
- **Observation unit**: participant (p. 18)

### Tried to work together (DV)

`dv_dec_e1`

- **Role**: dv (p. 20)
- **Type**: binary: work together vs. work alone (p. 20)
- **Definition**: Participant decided to try to work together with the partner (hot dogs/buns) rather than alone. (p. 20)
- **Observation unit**: participant (one decision) (p. 20)
- **Aggregation**: Percentage of participants per cell who tried to work together (p. 53)

### Expected earnings (derived outcome)

`dv_eff_e1`

- **Role [derived]**: dv (p. 22)
  - how it was derived: Computed from all possible matchups rather than the arranged matchups.
- **Definition [to check]**: Sum of the proportion of participants who chose to work alone times the smaller certain payoff plus the proportion expected to coordinate successfully (the product of the proportion that tried to work together with itself) times the higher risky payoff; discoordination pays zero. Expressed as a percentage of maximum possible earnings. (p. 22, p. 54)
- **Observation unit [derived]**: knowledge x payoff cell (p. 22)
  - how it was derived: Computed per cell from the cell proportion.

## Analysis

### Experiment 1: knowledge-level comparisons within each payoff

`an_main_e1`

- **Aim**: Test whether the proportion trying to work together differs by knowledge state within each payoff condition. (p. 20)
- **Inputs [derived]**: dv_dec_e1 by iv_kn_e1 within each level of iv_pay_e1 (p. 49)
  - how it was derived: Table 1 columns are payoff conditions, rows knowledge-level comparisons.
- **Method**: Chi-square tests of proportions: an omnibus test across all four knowledge levels (df = 3) and planned comparisons of adjacent levels (primary-secondary, secondary-tertiary, tertiary-common; df = 1), phi as effect size. (p. 49, p. 20)
- **Criterion**: *p < .05, **p < .01, ***p < .001. (p. 49)
- **Coding**: Whether or not the participant tried to work together. (p. 20)
- **Robustness**: None reported.

### Experiment 1: payoff invariance within each knowledge level

`an_inv_e1`

- **Aim**: Test rational actor versus knowledge-level heuristic hypotheses: does the proportion trying to work together change across payoffs for a given knowledge state? (p. 21)
- **Method**: Chi-square (df = 3) across the four payoff conditions within each knowledge condition, with phi. (p. 21)
- **Inputs [derived]**: dv_dec_e1 by iv_pay_e1 within each level of iv_kn_e1 (p. 21)
  - how it was derived: Described in the text.

### Experiment 1: expected earnings (efficiency)

`an_eff_e1`

- **Aim**: Assess the overall rationality of choices by expected earnings over all possible matchups. (p. 22)
- **Method [to check]**: Expected earnings per cell = (proportion who chose to work alone x smaller certain payoff) + (proportion who would successfully work together x higher risky payoff); expressed as a percentage of maximum possible earnings (Figure 3a). (p. 22)
- **Criterion**: Descriptive comparison of efficiency across knowledge conditions; no test reported. (p. 22)

## Reported results

### Experiment 1: Table 1 knowledge-level comparisons

`res_main_e1`

- **Result kind**: reported (p. 49)
- **Descriptives [to check]**: Percentages per cell are shown only in Figure 2 (image); they are not stated in the text or tables. (p. 53)
- **Table 1**: chi-square, n and phi per payoff condition (p. 49)
- **Table 1 values**: {"payoff_conditions": ["$1.00", "$2.00", "$5.00", "$10.00"], "All Levels (df=3)": {"chi2": ["63.70***", "62.79***", "56.71***", "70.41***"], "n": [276, 272, 236, 249], "phi": [0.48, 0.48, 0.49, 0.53]}, "1o vs 2o": {"chi2": ["21.26***", "23.27***", "2.66", "19.06***"], "n": [147, 134, 121, 118], "phi": [0.38, 0.42, 0.15, 0.4]}, "2o vs 3o": {"chi2": ["0.69", "0.63", "10.88***", "0.01"], "n": [141, 139, 113, 126], "phi": [0.07, 0.07, 0.31, 0.01]}, "3o vs CK": {"chi2": ["8.84**", "7.12**", "4.02*", "23.05***"], "n": [129, 138, 115, 131], "phi": [0.26, 0.26, 0.19, 0.42]}} (p. 49)
- **Reported pattern**: In all four payoff conditions more participants tried to work together with common knowledge than tertiary; in three of four, more with secondary than primary (marginal in the $5 condition); coordination rates the same with secondary and tertiary except at $5, where the secondary rate was anomalously low. (p. 21)

### Experiment 1: invariance of coordination attempts across payoffs

`res_inv_e1`

- **Result kind**: reported (p. 21)
- **Chi-square (df = 3) across payoffs**: Primary: chi2(3, N=261)=3.13, p=.373, phi=.11; Secondary: chi2(3, N=259)=7.66, p=.054, phi=.17; Tertiary: chi2(3, N=260)=6.81, p=.078, phi=.16; Common: chi2(3, N=253)=.76, p=.859, phi=.05. (p. 21, p. 22)
- **Interpretation tie**: Proportions varied little across payoff conditions as the cost-benefit ratio rose 10-fold. (p. 21)

### Experiment 1: efficiency and coordination failure

`res_eff_e1`

- **Result kind**: reported (p. 22)
- **Descriptives [to check]**: Participants who chose to work together failed to coordinate around 50% of the time in the secondary and tertiary conditions; efficiency (Figure 3a) was higher with primary and common knowledge than with either level of shared knowledge for all payoffs. Values appear only in Figure 3. (p. 22, p. 54)

## Simulation scope

### Experiment 1 simulation scope

`sim_e1`

- **Steps [implementation]**: All steps through the decision, explanation, comprehension questions and demographics can be role-played by one AI participant per session; offline pairing and payment are not simulated, expected earnings are computed from all session pairs.
- **Roles [implementation]**: One AI participant per original participant, as butcher or baker; the partner is described in text only, as in the original, since decisions were independent. The experimenter/interface is scripted.
- **Conditions [implementation]**: All 16 knowledge x payoff cells; wording of primary/secondary/tertiary reveals for the payoff amount and the butcher-perspective texts is incomplete (see issues).
- **Outcomes [implementation]**: Decision, explanation, comprehension answers; not real MTurk behaviour and no real money.
- **Limits [implementation]**: AI participants have no real monetary stakes, no MTurk population, and may know the hypothesis from training data; exclusion is applied from comprehension answers.

## Open issues

- **Assignment of participants to cells and roles not described** [decision, `iss_e1_assign`, design_e1]: The paper says knowledge state and payoff were varied between subjects with 100 recruited per condition, but not how participants were assigned. Suggested: Confirm random, equal assignment to the 16 cells and to butcher/baker, or supply the mechanism.
- **Blinding not reported** [check, `iss_e1_blind`, design_e1]: Not stated. Suggested: None needed unless the authors report otherwise.
- **Number completing not reported** [check, `iss_e1_completed`, part_e1]: Only recruited (1600) and analysed (1033) are given. Suggested: Obtain completion counts if needed.
- **Exclusion criterion for comprehension failures** [check, `iss_e1_excl`, part_e1]: The paper excludes participants with incorrect answers to the payoff comprehension questions but does not say whether any single error excluded them. Suggested: Confirm the criterion with the authors.
- **Placeholder values in the common knowledge message** [decision, `iss_e1_ckvals`, mat_ck]: The printed message contains [today's price] and [earnings for working together]; the relation between the two amounts is not stated. Suggested: Decide the values (e.g. both set to the together-payoff). Recorded as a replay adaptation.
- **Butcher-perspective wording not printed** [decision, `iss_e1_butcher`, mat_persp]: Messages are given from the baker's perspective only. Suggested: Researcher to approve a role swap (Butcher<->Baker, bakery<->butcher shop) or supply the original.
- **Instruction wording not printed** [decision, `iss_e1_instr`, mat_e1_instr]: Instructions are described in prose, not printed. A supplied replication survey (Exeter, GBP) prints similar wording but is a different instrument. Suggested: Supply the original instruction text or approve the reported-content version.
- **Instructions are a paraphrase** [check, `iss_e1_instr_check`, mat_e1_instr]: Field is labelled reported because the wording is not printed. Suggested: Review the paraphrase against the original materials.
- **Payoff reveal wording in primary/secondary/tertiary not printed** [decision, `iss_e1_price`, mat_e1_reveal]: Only the common knowledge message prints the amount. Suggested: State the amount in the reveal (ten cents more than the sure profit); record as an adaptation.
- **Silent loudspeaker wording** [check, `iss_e1_silent`, mat_prim]: The public box 'stated that the loudspeaker was silent'; exact wording not given. Suggested: Supply the original wording if available.
- **Decision prompt not printed** [decision, `iss_e1_decprompt`, mat_e1_resp]: Prompt and option labels are not printed in the paper. Suggested: Supply original wording.
- **Full comprehension sets not printed** [decision, `iss_e1_comp`, mat_comp]: Only one example of each question set is printed; the survey source gives a full set for a replication. Suggested: Approve survey wording or supply the original.
- **Demographic items not printed** [check, `iss_e1_demo`, mat_demo]: Wording not printed. Suggested: None required for the replay.
- **Cell percentages only in figures** [check, `iss_e1_cells`, res_main_e1]: Percentages per cell are shown in Figure 2 and not in text. Suggested: Read Figure 2 or obtain data.
- **Expected earnings formula as printed** [check, `iss_e1_effdef`, dv_eff_e1]: The text says the successful-coordination proportion is 'the product of the proportion of participants that tried to work together and this proportion minus one', which is not a probability as written; p x p is the evident intent. Suggested: Confirm with the authors.
- **Visibility during decision** [check, `iss_e1_vis`, exp1]: The paper says information was visible while answering the comprehension questions in Experiments 1-2; visibility during the decision is inferred. Suggested: Confirm.
- **Efficiency values only in figure** [check, `iss_e1_effres`, res_eff_e1]: Figure 3a values are not in the text. Suggested: Read Figure 3a.
- **Printed completion of Table 1 note** [check, `iss_e1_ph`, res_main_e1]: Table 1 note: all-levels tests have df 3 and adjacent tests df 1; n per comparison is the combined n of the compared cells. Suggested: None.
- **Retained n per cell not reported** [check, `iss_e1_cellN`, part_e1]: Only the combined n of compared cells (Table 1) and per-knowledge totals are given; the retained n of each of the 16 cells after exclusions is not reported. Suggested: Obtain cell sizes from the data if needed.
