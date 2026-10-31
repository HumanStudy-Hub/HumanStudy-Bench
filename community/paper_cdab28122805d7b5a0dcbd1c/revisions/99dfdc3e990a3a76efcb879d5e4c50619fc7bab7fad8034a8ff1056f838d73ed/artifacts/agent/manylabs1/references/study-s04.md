# Study 4: Retrospective gambler's fallacy

`s04`

Retrospective gambler's fallacy (Oppenheimer & Monin, 2009)

## Hypotheses

### Hypothesis: Retrospective gambler's fallacy

`hyp_s04`

- **Statement**: Observing a rare chance outcome (three 6's) leads people to estimate that more rolls occurred before it than observing a less rare outcome (two 6's and a 3). (p. 4, p. 5)

## Participants

### Retrospective gambler's fallacy: participants analysed

`s04_pt`

- **N in the analysis (Table 2)**: Table 2 row: Retrospective gambler fallacy 0.69 ... 5,942 <.001 (N column); sample is the shared 36-site sample (p. 7)
- **Exclusions**: Reported in the supplement (see evidence) (p. 6)
- **Sampling unit [derived]**: participant
  - how it was derived: Each participant provides one response record per study

## Design

### Retrospective gambler's fallacy: design

`s04_dz`

- **Design type [derived]**: Between-subjects, two conditions (p. 6)
  - how it was derived: The analysis compares two conditions with an independent-samples test, which implies a between-subjects comparison
- **Assignment**: Not stated in the paper. (p. 6)
- **Conditions**: Three 6's: Imagines three dice all coming up 6; Two 6's and a 3: Imagines three dice: two 6's and one 3 (p. 4)

## Conditions

### Retrospective gambler's fallacy: Three 6's

`s04_c_three6`

- **Description**: Imagines three dice all coming up 6 (p. 4)

### Retrospective gambler's fallacy: Two 6's and a 3

`s04_c_two6one3`

- **Description**: Imagines three dice: two 6's and one 3 (p. 4)

## Materials

### Retrospective gambler's fallacy: participant materials

`s04_mat`

- **Description of the material as reported**: See evidence quote (paper description of the original and replication stimulus) (p. 4)
- **Full participant-visible wording (instructions, items, response options)** (verbatim, sources/Measures_US_Version.docx, p. 4):
  > [two6one3]
  > Imagine that you are in a casino and you happen to pass a man rolling dice. You observe him roll three dice and one comes up 3, and two come up 6’s. Based on your imagined scenario, how many times do you think the man had rolled the dice before you walked by?
  >
  > [three6]
  > Imagine that you are in a casino and you happen to pass a man rolling dice. You observe him roll three dice and all three come up 6’s. Based on your imagined scenario, how many times do you think the man had rolled the dice before you walked by?

## Procedure

Order of steps:

- Imagine seeing three dice all come up 6 (participant sees: Retrospective gambler's fallacy: participant materials) (p. 4)
  - conditions: Retrospective gambler's fallacy: Three 6's
- Estimate how many times the man had rolled before (participant sees: Retrospective gambler's fallacy: participant materials) (p. 4)
- Imagine seeing three dice come up two 6's and a 3 (participant sees: Retrospective gambler's fallacy: participant materials) (p. 4)
  - conditions: Retrospective gambler's fallacy: Two 6's and a 3
- (back to: Estimate how many times the man had rolled before)

### Retrospective gambler's fallacy: procedure

`s04_pc`

- **Task**: Estimate how many times the man had rolled before (p. 4)
- **Changes in the replication**: See evidence quote (p. 4)

## Variables

### Retrospective gambler's fallacy: independent variable

`s04_viv`

- **Role [derived]**: iv (p. 6)
  - how it was derived: Condition is the grouping variable of the confirmatory test
- **Type**: categorical (Three 6's, Two 6's and a 3) (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response record per participant

### Retrospective gambler's fallacy: Estimated number of earlier rolls

`s04_vdv`

- **Role [derived]**: dv (p. 6)
  - how it was derived: Outcome of the confirmatory test
- **Type**: continuous (open numeric) (p. 4)
- **Definition**: Open-ended estimate of how many times the man had rolled the dice before the participant entered the room (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response per participant (per topic for anchoring)
- **Response scale**: Open-ended numeric estimate (p. 4)

## Analysis

### Retrospective gambler's fallacy: confirmatory analysis

`s04_an`

- **Aim**: Test whether the original effect replicates (p. 5)
- **Method**: Square-root transform of responses, 37 responses above 3 SD dropped, then independent t-test (p. 6)
- **Inputs [derived]**: Records of Estimated number of earlier rolls by condition after exclusions (p. 6)
  - how it was derived: Exclusions are those stated in the supplement

## Reported results

### Retrospective gambler's fallacy: original finding

`s04_res_orig`

- **Result kind**: reported (p. 5)
- **Original result**: See evidence quote (p. 5)

### Retrospective gambler's fallacy: replication result

`s04_res_rep`

- **Result kind**: reported (p. 5, p. 6)
- **Test and descriptives (supplement)**: See evidence quotes (p. 5, p. 6)
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

- **How participants were allocated to conditions is not stated** [decision, `i_s04_dz_assignment`, s04_dz]: The paper reports a between-subjects comparison but not the allocation rule (random, balanced, other). Suggested: Confirm random assignment with equal probability, or supply the rule. (p. 6)
