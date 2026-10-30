# Study 6: Norm of reciprocity

`s06`

Norm of reciprocity (Hyman & Sheatsley, 1950)

## Hypotheses

### Hypothesis: Norm of reciprocity

`hyp_s06`

- **Statement**: Support for allowing reporters from another (communist) country into the US is higher when the question follows one about allowing American reporters into that country than when it is asked first. (p. 4, p. 7)

## Participants

### Norm of reciprocity: participants analysed

`s06_pt`

- **N in the analysis (Table 2)**: Table 2 row: Norm of reciprocity 0.16 ... 6,276 <.001 (N column); sample is the shared 36-site sample (p. 7)
- **Exclusions**: Reported in the supplement (see evidence) (p. 7)
- **Sampling unit [derived]**: participant
  - how it was derived: Each participant provides one response record per study

## Design

### Norm of reciprocity: design

`s06_dz`

- **Design type**: Between-subjects, two conditions (p. 7)
- **Assignment**: Not stated in the paper. (p. 7)
- **Conditions**: American reporters question first: Question about allowing American reporters into the other country asked first; Other-country reporters question first: Question about allowing other-country (North Korea) reporters into the home country asked first (p. 4)

## Conditions

### Norm of reciprocity: American reporters question first

`s06_c_us_first`

- **Description**: Question about allowing American reporters into the other country asked first (p. 4)

### Norm of reciprocity: Other-country reporters question first

`s06_c_nk_first`

- **Description**: Question about allowing other-country (North Korea) reporters into the home country asked first (p. 4)

## Materials

### Norm of reciprocity: participant materials

`s06_mat`

- **Description of the material as reported**: See evidence quote (paper description of the original and replication stimulus) (p. 4)
- **Full participant-visible wording (instructions, items, response options)** (verbatim, sources/Measures_US_Version.docx, p. 4):
  > [nk_first]
  > Do you think the United States should let newspaper reporters from North Korea come in here and send back to their papers the news as they see it?  
  > Yes
  > No
  >
  > Do you think North Korea should let American newspaper reporters come in and send back to America the news as they see it?  
  > Yes
  > No
  >
  > [us_first]
  > Do you think North Korea should let American newspaper reporters come in and send back to America the news as they see it?  
  > Yes
  > No
  >
  > Do you think the United States should let newspaper reporters from North Korea come in here and send back to their papers the news as they see it?  
  > Yes
  > No

## Procedure

Order of steps:

- Answer the American-reporters question first (participant sees: Norm of reciprocity: participant materials) (p. 4)
  - conditions: Norm of reciprocity: American reporters question first
- Answer the remaining reporters question (participant sees: Norm of reciprocity: participant materials) (p. 4, p. 7)
- Answer the other-country-reporters question first (participant sees: Norm of reciprocity: participant materials) (p. 4)
  - conditions: Norm of reciprocity: Other-country reporters question first
- (back to: Answer the remaining reporters question)

### Norm of reciprocity: procedure

`s06_pc`

- **Task**: Answer the remaining reporters question (p. 4)
- **Changes in the replication**: See evidence quote (p. 4)

## Variables

### Norm of reciprocity: independent variable

`s06_viv`

- **Role [derived]**: iv (p. 7)
  - how it was derived: Condition is the grouping variable of the confirmatory test
- **Type**: categorical (American reporters question first, Other-country reporters question first) (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response record per participant

### Norm of reciprocity: Support for allowing other-country reporters

`s06_vdv`

- **Role [derived]**: dv (p. 7)
  - how it was derived: Outcome of the confirmatory test
- **Type**: binary (p. 7)
- **Definition**: Support (yes/no) for allowing reporters from the other country into the participant's country (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response per participant (per topic for anchoring)
- **Response scale [derived]**: Yes/no support (p. 7)
  - how it was derived: The supplement reports percentages supporting; the yes/no response format is inferred from them.

## Analysis

### Norm of reciprocity: confirmatory analysis

`s06_an`

- **Aim**: Test whether the original effect replicates (p. 7)
- **Method**: Chi-square on statement agreement with condition as a between-subjects variable (p. 7)
- **Inputs [derived]**: Records of Support for allowing other-country reporters by condition after exclusions (p. 7)
  - how it was derived: Exclusions are those stated in the supplement

## Reported results

### Norm of reciprocity: original finding

`s06_res_orig`

- **Result kind**: reported (p. 7)
- **Original result**: See evidence quote (p. 7)

### Norm of reciprocity: replication result

`s06_res_rep`

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

- **How participants were allocated to conditions is not stated** [decision, `i_s06_dz_assignment`, s06_dz]: The paper reports a between-subjects comparison but not the allocation rule (random, balanced, other). Suggested: Confirm random assignment with equal probability, or supply the rule. (p. 7)
