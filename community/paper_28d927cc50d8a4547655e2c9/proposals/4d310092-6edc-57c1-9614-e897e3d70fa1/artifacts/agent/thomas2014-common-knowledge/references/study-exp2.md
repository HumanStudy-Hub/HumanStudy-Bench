# Experiment 2: four-person superburger coordination game (4 knowledge x 4 payoff, between subjects)

`exp2`

MTurk participants, 1600 recruited, 1150 analysed; same design as Experiment 1 but success requires all three other merchants to also work together.

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

### Experiment 2 participants

`part_e2`

- **Setting**: Online study on Amazon Mechanical Turk, role-played four-merchant game. (p. 24)
- **Roles**: Four roles: butcher, baker, cheese maker and produce vender; one participant assigned to each role in a group of four. (p. 24)
- **Recruitment**: Amazon Mechanical Turk, as in Experiment 1. (p. 24)
- **Recruited**: 1600, evenly distributed across the sixteen combinations of four payoff and four knowledge conditions (100 per cell). (p. 24)
- **Analysed**: 1150 after excluding participants who did not understand the payoffs (Table 2 totals: 281+297+283+289). (p. 24, p. 50)
- **Characteristics**: 48% female; Mage = 31.9, SDage = 11.1. (p. 24)
- **Eligibility**: Not stated in the paper. (p. 24)
- **Sampling unit [derived]**: The individual participant; groups of four are formed offline for payment. (p. 24)
  - how it was derived: As in Experiment 1, 'all other aspects of the procedure were the same'.
- **Compensation [derived, to check]**: As in Experiment 1 (50 cents minimum plus a bonus). (p. 24)
  - how it was derived: Inherited from Experiment 1 by 'all other aspects'.

## Design

### Experiment 2 design

`design_e2`

- **Design type**: 4 (knowledge) x 4 (payoff) between-subjects, as Experiment 1, but with three partners. (p. 24)
- **Assignment**: Not stated in the paper. (p. 24)
- **Order [derived]**: Same as Experiment 1. (p. 24)
  - how it was derived: 'All other aspects of the procedure were the same as in Experiment 1.'
- **Manipulation check [derived]**: Comprehension questions as in Experiment 1; participants who did not understand the payoffs were excluded. (p. 24, p. 26)
  - how it was derived: Same procedure as Experiment 1.

## Conditions

### Experiment 2: Primary knowledge

`kc_prim_e2`

- **Definition**: Primary knowledge as in Experiment 1; the participant's three partners were said to have the same level of knowledge, with 'the other merchants' replacing 'the Butcher' or 'the Baker'. (p. 24)

### Experiment 2: Secondary knowledge

`kc_sec_e2`

- **Definition**: Secondary knowledge as in Experiment 1; the participant's three partners were said to have the same level of knowledge, with 'the other merchants' replacing 'the Butcher' or 'the Baker'. (p. 24)

### Experiment 2: Tertiary knowledge

`kc_ter_e2`

- **Definition**: Tertiary knowledge as in Experiment 1; the participant's three partners were said to have the same level of knowledge, with 'the other merchants' replacing 'the Butcher' or 'the Baker'. (p. 24)

### Experiment 2: Common knowledge

`kc_ck_e2`

- **Definition**: Common knowledge as in Experiment 1; the participant's three partners were said to have the same level of knowledge, with 'the other merchants' replacing 'the Butcher' or 'the Baker'. (p. 24)

### Experiment 2: payoff $1

`pc_1_e2`

- **Definition [to check]**: $1.00 alone; amount for success with all partners not printed. (p. 50)

### Experiment 2: payoff $2

`pc_2_e2`

- **Definition [to check]**: $2.00 alone; amount for success with all partners not printed. (p. 50)

### Experiment 2: payoff $5

`pc_5_e2`

- **Definition [to check]**: $5.00 alone; amount for success with all partners not printed. (p. 50)

### Experiment 2: payoff $10

`pc_10_e2`

- **Definition [to check]**: $10.00 alone; amount for success with all partners not printed. (p. 50)

## Materials

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

### Experiment 2 Primary knowledge message

`mat_prim_e2`

Conditions: Experiment 2: Primary knowledge

- **Private box (derived from the Experiment 1 text) [derived, to check]**: “The Messenger Boy has not seen the other merchants today, so he cannot tell you anything about what the other merchants knows.” (p. 19, p. 24)
  - how it was derived: Experiment 1 baker text with 'the Butcher' replaced by 'the other merchants', as the paper states; other grammar and shop names unchanged except where the Butcher's shop is named.

### Experiment 2 Secondary knowledge message

`mat_sec_e2`

Conditions: Experiment 2: Secondary knowledge

- **Private box (derived from the Experiment 1 text) [derived, to check]**: “The Messenger Boy says he stopped by the other merchants' shops before coming to your bakery. He tells you that the other merchants knows what today's hot dog price is. However, he says that he forgot to mention to the other merchants that he was coming to see you, so the other merchants is not aware that you know today's hot dog price.” (p. 19, p. 24)
  - how it was derived: Experiment 1 baker text with 'the Butcher' replaced by 'the other merchants', as the paper states; other grammar and shop names unchanged except where the Butcher's shop is named.

### Experiment 2 Tertiary knowledge message

`mat_ter_e2`

Conditions: Experiment 2: Tertiary knowledge

- **Private box (derived from the Experiment 1 text) [derived, to check]**: “The Messenger Boy mentions that he is heading over to the other merchants' shops, and will let the other merchants know today's price as well. The Messenger Boy will also tell the other merchants that he just came from your bakery and told you the price. However, the Messenger Boy will not inform the other merchants that he told you he would be heading over there. So, while the other merchants is aware that you know today's price, he is not aware that you know that he knows that.” (p. 19, p. 24)
  - how it was derived: Experiment 1 baker text with 'the Butcher' replaced by 'the other merchants', as the paper states; other grammar and shop names unchanged except where the Butcher's shop is named.

### Experiment 2 Common knowledge message

`mat_ck_e2`

Conditions: Experiment 2: Common knowledge

- **Private box (derived from the Experiment 1 text) [derived, to check]**: “The messenger boy did not come by. Because the market price was broadcast on the loudspeaker, the other merchants knows [today’s price], and he knows that you know this information as well.” (p. 20, p. 24)
  - how it was derived: Experiment 1 baker text with 'the Butcher' replaced by 'the other merchants', as the paper states; other grammar and shop names unchanged except where the Butcher's shop is named.
- **Public box [derived, to check]**: “The loudspeaker broadcast the market price of [today’s price] (of which you could earn [earnings for working together]).” (p. 20, p. 24)
  - how it was derived: As above.

### Experiment 2 instructions (as reported)

`mat_e2_instr`

- **Content [to check]**: Participants were told they could work together to make superburgers, which require a burger, a bun, cheese, and toppings from, respectively, a butcher, a baker, a cheese maker, and a produce vender, one participant per role. Each participant also had the option to make a food item on their own for a sure profit. They would receive a profit for contributing to superburgers only if all three of the other merchants made the same choice, and nothing otherwise. Remaining instructions as in Experiment 1. (p. 24)
- **Printed wording**: Not stated in the paper. (p. 24)
- **Amounts for working together [derived, to check]**: The payoff for successful four-way coordination is not stated; the paper says all other aspects are as in Experiment 1 (ten cents above the sure profit). (p. 24, p. 50)
  - how it was derived: Inferred from 'All other aspects of the procedure were the same as in Experiment 1' and the payoff headings of Table 2.

## Procedure

Order of steps:

- Read instructions about the four roles, profits and information sources (participant sees: Experiment 2 instructions (as reported)) (p. 24)
- Depending on condition: Reveal the day's information for the assigned knowledge condition (p. 24)
  - **Which message is shown**: The message follows the participant's knowledge condition. (p. 24)
  - Primary knowledge message shown (participant sees: Experiment 2 Primary knowledge message) (p. 24)
    - conditions: Experiment 2: Primary knowledge
  - Secondary knowledge message shown (participant sees: Experiment 2 Secondary knowledge message) (p. 24)
    - conditions: Experiment 2: Secondary knowledge
  - Tertiary knowledge message shown (participant sees: Experiment 2 Tertiary knowledge message) (p. 24)
    - conditions: Experiment 2: Tertiary knowledge
  - Common knowledge message shown (participant sees: Experiment 2 Common knowledge message) (p. 24)
    - conditions: Experiment 2: Common knowledge
- Decide to work alone or work together (participant sees: Experiment 2 Primary knowledge message; Experiment 2 Secondary knowledge message; Experiment 2 Tertiary knowledge message; Experiment 2 Common knowledge message) (p. 24)
- Explain the decision (p. 24)
- Answer payoff and knowledge comprehension questions (participant sees: Experiment 2 Primary knowledge message; Experiment 2 Secondary knowledge message; Experiment 2 Tertiary knowledge message; Experiment 2 Common knowledge message) (p. 24)
- Demographics and submit (p. 24)
- Offline grouping, profit calculation and bonus payment (scripted) (p. 24)

### Experiment 2 procedure

`proc_e2`

- **Task**: Choose to work alone for a sure profit or work together on superburgers; success needs all three other merchants to also choose to work together. (p. 24)
- **Actions [derived]**: As in Experiment 1: decision, explanation, comprehension questions, demographics. (p. 24)
  - how it was derived: Same procedure as Experiment 1.
- **Visible information**: As in Experiment 1; the three partners were said to have the same level of knowledge. (p. 24)
- **Payoff**: Profit only if all three others make the same choice; otherwise nothing. (p. 24)

## Variables

### Knowledge state (IV)

`iv_kn_e2`

- **Role**: iv (p. 24)
- **Type**: categorical, 4 levels (p. 24)
- **Observation unit**: participant (p. 24)

### Payoff structure (IV)

`iv_pay_e2`

- **Role**: iv (p. 24)
- **Type**: categorical, 4 levels ($1, $2, $5, $10 forgone payoff) (p. 50)
- **Unit**: US dollars (p. 50)
- **Observation unit**: participant (p. 24)

### Tried to work together (DV)

`dv_dec_e2`

- **Role**: dv (p. 25)
- **Type**: binary (p. 25)
- **Observation unit**: participant (p. 24)
- **Aggregation**: Percentage per cell (Figure 4) (p. 55)

### Expected earnings (derived outcome)

`dv_eff_e2`

- **Role [derived]**: dv (p. 54)
  - how it was derived: Computed as in Experiment 1 across all possible matchups.
- **Definition**: Not stated in the paper. (p. 54, p. 25)
- **Observation unit [derived]**: knowledge x payoff cell (p. 54)
  - how it was derived: Figure 3b is by knowledge condition.

## Analysis

### Experiment 2: knowledge-level comparisons within each payoff

`an_main_e2`

- **Aim**: Test knowledge-level effects on coordination attempts within each payoff. (p. 25)
- **Method**: Chi-square tests across all four levels (df 3) and adjacent-level planned comparisons (df 1), with phi. (p. 50, p. 25)
- **Criterion**: *p < .05, **p < .01, ***p < .001. (p. 50)

### Experiment 2: payoff sensitivity

`an_pay_e2`

- **Aim**: Assess whether coordination rates fall as the forgone payoff increases. (p. 25)
- **Method**: Not stated in the paper. (p. 25)

### Experiment 2: expected earnings

`an_eff_e2`

- **Aim**: Compare expected earnings by knowledge condition (Figure 3b). (p. 25)
- **Method**: Not stated in the paper. (p. 54)

## Reported results

### Experiment 2: Table 2 knowledge-level comparisons

`res_main_e2`

- **Result kind**: reported (p. 50)
- **Table 2 values**: {"payoff_conditions": ["$1.00", "$2.00", "$5.00", "$10.00"], "All Levels (df=3)": {"chi2": ["68.24***", "45.32***", "46.36***", "24.72***"], "n": [281, 297, 283, 289], "phi": [0.49, 0.39, 0.41, 0.29]}, "1o vs 2o": {"chi2": ["13.59***", "14.95***", "1.77", "11.54***"], "n": [142, 147, 148, 140], "phi": [0.31, 0.32, 0.11, 0.29]}, "2o vs 3o": {"chi2": ["2.50", "0.69", "2.26", "0.03"], "n": [132, 153, 137, 149], "phi": [0.11, 0.07, 0.13, 0.01]}, "3o vs CK": {"chi2": ["10.14**", "6.21*", "12.39***", "4.23*"], "n": [139, 150, 135, 149], "phi": [0.27, 0.2, 0.3, 0.17]}} (p. 50)
- **Reported pattern**: In all payoff conditions significantly more tried to work together with common than tertiary knowledge; in three of four, with secondary than primary; no significant secondary-tertiary difference. (p. 25)
- **Descriptives [to check]**: Per-cell percentages only in Figure 4; more than half of common-knowledge participants chose the risky higher payoff. (p. 55, p. 26)

### Experiment 2: payoff sensitivity and earnings

`res_pay_e2`

- **Result kind**: reported (p. 25)
- **Effect**: Increasing the relative cost of coordination failures lowered coordination rates in all knowledge conditions. The most profitable knowledge condition at all four payoffs was Primary Knowledge; Common Knowledge was less profitable than Shared Knowledge except at $1. (p. 25)
- **Descriptives [to check]**: Figure values not in text. (p. 54)

## Simulation scope

### Experiment 2 simulation scope

`sim_e2`

- **Steps [implementation]**: Decision, explanation, comprehension and demographics per AI participant; offline grouping and payment are not simulated.
- **Roles [implementation]**: One AI participant per original participant in one of four merchant roles; partners are described as having the same knowledge level, as in the original.
- **Conditions [implementation]**: 16 knowledge x payoff cells; sizes and wording of the four-way success payoff are inferred.
- **Outcomes [implementation]**: Decision, explanation, comprehension answers.
- **Limits [implementation]**: No real stakes or MTurk population; the grammatical form of partners ('the other merchants') is a mechanical substitution.

## Open issues

- **Eligibility not stated** [check, `iss_e2_elig`, part_e2]: US restriction is stated for Experiment 1 only. Suggested: None.
- **Assignment not described** [decision, `iss_e2_assign`, design_e2]: Same gap as Experiment 1. Suggested: Confirm.
- **Experiment 2 instruction wording not printed** [decision, `iss_e2_instr`, mat_e2_instr]: Only described in prose. Suggested: Supply or approve.
- **Experiment 2 instructions are a paraphrase** [check, `iss_e2_instr_chk`, mat_e2_instr]: Labelled reported. Suggested: Review.
- **Four-way success payoff not stated** [check, `iss_e2_amounts`, mat_e2_instr]: Inferred from 'all other aspects same as Experiment 1' and Table 2 headings. Suggested: Confirm +$0.10.
- **Message substitution is mechanical** [check, `iss_e2_subst`, mat_prim_e2]: The paper says 'the Butcher' or 'the Baker' was replaced with 'the other merchants'; verbatim output is ungrammatical in places (e.g. 'the other merchants knows'). Suggested: Supply original Experiment 2 text.
- **Four-person expected earnings computation not described** [check, `iss_e2_edef`, dv_eff_e2]: Figure 3b formula for four players is not given. Suggested: Confirm with the authors.
- **No test for payoff sensitivity** [check, `iss_e2_paytest`, an_pay_e2]: Text reports sensitivity without statistics. Suggested: Read Figure 4 or obtain the data.
- **Experiment 2 efficiency method** [check, `iss_e2_effmeth`, an_eff_e2]: Same gap. Suggested: Same.
- **Cell percentages only in figures** [check, `iss_e2_cells`, res_main_e2]: Figure 4. Suggested: Read Figure 4.
- **Efficiency values only in Figure 3** [check, `iss_e2_fig3`, res_pay_e2]: Not in text. Suggested: Read Figure 3b.
- **Role-specific wording for the four Experiment 2 merchants not printed** [decision, `iss_e2_roles`, mat_e2_instr]: Only the baker-perspective messages are printed for Experiment 1; the paper does not give the cheese maker, produce vender or butcher versions for Experiment 2. Suggested: Supply role-specific wording or approve a role-swap.
