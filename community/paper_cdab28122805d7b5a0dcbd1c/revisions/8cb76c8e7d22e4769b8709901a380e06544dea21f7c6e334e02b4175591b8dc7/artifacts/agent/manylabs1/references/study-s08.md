# Study 8: Quote attribution

`s08`

Quote attribution (Lorge & Curtiss, 1936)

## Hypotheses

### Hypothesis: Quote attribution

`hyp_s08`

- **Statement**: An identical quotation receives more agreement when it is attributed to a liked individual than to a disliked individual. (p. 5)

## Participants

### Quote attribution: participants analysed

`s08_pt`

- **N in the analysis (Table 2)**: Table 2 row: Quote attribution na ... 6,325 <.001 (N column); sample is the shared 36-site sample (p. 7)
- **Exclusions**: Reported in the supplement (see evidence) (p. 7)
- **Sampling unit [derived]**: participant
  - how it was derived: Each participant provides one response record per study

## Design

### Quote attribution: design

`s08_dz`

- **Design type [derived]**: Between-subjects, two conditions (p. 7)
  - how it was derived: The analysis compares two conditions with an independent-samples test, which implies a between-subjects comparison
- **Assignment**: Not stated in the paper. (p. 7)
- **Conditions**: Liked speaker (George Washington): Quote attributed to George Washington; Disliked speaker (Osama Bin Laden): Quote attributed to Osama Bin Laden (p. 5)

## Conditions

### Quote attribution: Liked speaker (George Washington)

`s08_c_liked`

- **Description**: Quote attributed to George Washington (p. 5)

### Quote attribution: Disliked speaker (Osama Bin Laden)

`s08_c_disliked`

- **Description**: Quote attributed to Osama Bin Laden (p. 5)

## Materials

### Quote attribution: participant materials

`s08_mat`

- **Description of the material as reported**: See evidence quote (paper description of the original and replication stimulus) (p. 5)
- **Printed text available in the sources** (verbatim, p. 5):
  > Quoted in the evidence record
- **Full participant-visible wording (instructions, items, response options)** (verbatim, sources/Measures_US_Version.docx, p. 5):
  > [liked]
  > A quote similar to what follows was once spoken by George Washington: "I have sworn to only live free, even if I find bitter the taste of death."
  >
  > How much do you agree with this quote?
  >
  > 1 – strongly agree
  > 2
  > 3
  > 4
  > 5 – neither agree nor disagree
  > 6
  > 7
  > 8
  > 9 – strongly disagree
  >
  > [disliked]
  > A quote similar to what follows was once spoken by Osama Bin Laden: "I have sworn to only live free, even if I find bitter the taste of death."
  >
  > How much do you agree with this quote?
  >
  > 1 – strongly agree
  > 2
  > 3
  > 4
  > 5 – neither agree nor disagree
  > 6
  > 7
  > 8
  > 9 – strongly disagree

## Procedure

Order of steps:

- Rate agreement with the quote attributed to Washington (participant sees: Quote attribution: participant materials) (p. 5)
  - conditions: Quote attribution: Liked speaker (George Washington)
- Rate agreement with the quotation (participant sees: Quote attribution: participant materials) (p. 5)
- Rate agreement with the quote attributed to Bin Laden (participant sees: Quote attribution: participant materials) (p. 5)
  - conditions: Quote attribution: Disliked speaker (Osama Bin Laden)
- (back to: Rate agreement with the quotation)

### Quote attribution: procedure

`s08_pc`

- **Task**: Rate agreement with the quotation (p. 5)

## Variables

### Quote attribution: independent variable

`s08_viv`

- **Role [derived]**: iv (p. 7)
  - how it was derived: Condition is the grouping variable of the confirmatory test
- **Type**: categorical (Liked speaker (George Washington), Disliked speaker (Osama Bin Laden)) (p. 5)
- **Observation unit [derived]**: participant
  - how it was derived: One response record per participant

### Quote attribution: Agreement with the quotation

`s08_vdv`

- **Role [derived]**: dv (p. 7)
  - how it was derived: Outcome of the confirmatory test
- **Type**: rating (p. 5)
- **Definition**: Rated agreement with the quotation (p. 5)
- **Observation unit [derived]**: participant
  - how it was derived: One response per participant (per topic for anchoring)
- **Response scale** (verbatim, sources/Measures_US_Version.docx, p. 5):
  > 1 – strongly agree
  > 2
  > 3
  > 4
  > 5 – neither agree nor disagree
  > 6
  > 7
  > 8
  > 9 – strongly disagree

## Analysis

### Quote attribution: confirmatory analysis

`s08_an`

- **Aim**: Test whether the original effect replicates (p. 5)
- **Method**: Independent samples t-test with condition (liked, disliked speaker) as the independent variable and agreement with the quote as the dependent variable (p. 7)
- **Inputs [derived]**: Records of Agreement with the quotation by condition after exclusions (p. 7)
  - how it was derived: Exclusions are those stated in the supplement

## Reported results

### Quote attribution: original finding

`s08_res_orig`

- **Result kind**: reported (p. 5)
- **Original result**: See evidence quote (p. 5)

### Quote attribution: replication result

`s08_res_rep`

- **Result kind**: reported (p. 7)
- **Test and descriptives (supplement)**: See evidence quotes (p. 7)
- **Table 2 row (effect sizes, CIs, proportions of samples)**: See evidence quote (p. 7)

## Simulation scope

### Simulation scope: text studies

`sim_text`

- **Steps [implementation]**: Text replay of the bundled session: each AI participant sees the studies in a randomized order and answers each in turn, seeing its own earlier answers. Studies whose manipulation is an image (flag priming, currency priming) are excluded until the image decision is made; the reaction-time IAT is excluded.
- **Roles [implementation]**: Participants only; laboratory experimenters did not interact with participants and are not simulated.
- **Conditions [implementation]**: Every condition of the two-condition designs; assignment rule is a researcher decision.
- **Outcomes [implementation]**: Ratings, choices, numeric estimates and item composites; evaluation reproduces the confirmatory test per study and reports not_ready when wording or scales are missing.
- **Limits [implementation]**: Not simulated: 36-site population variation, seven translations, lab versus online setting, timed imagination periods, image stimuli (flag photos, money background). Imagined scenarios have no real stakes.

## Open issues

- **How participants were allocated to conditions is not stated** [decision, `i_s08_dz_assignment`, s08_dz]: The paper reports a between-subjects comparison but not the allocation rule (random, balanced, other). Suggested: Confirm random assignment with equal probability, or supply the rule. (p. 7)
