# Study 5: Low-versus-high category scales

`s05`

Low-versus-high category scales (Schwarz, Hippler, Deutsch, & Strack, 1985)

## Participants

### Low-versus-high category scales: participants analysed

`s05_pt`

- **N in the analysis (Table 2)**: Table 2 row: Low vs. high category scales 0.50 ... 5,899 <.001 (N column); sample is the shared 36-site sample (p. 7)
- **Exclusions**: Reported in the supplement (see evidence) (p. 7)
- **Sampling unit [derived]**: participant
  - how it was derived: Each participant provides one response record per study

## Design

### Low-versus-high category scales: design

`s05_dz`

- **Design type [derived]**: Between-subjects, two conditions (p. 7)
  - how it was derived: The analysis compares two conditions with an independent-samples test, which implies a between-subjects comparison
- **Assignment**: Not stated in the paper. (p. 7)
- **Conditions**: Low-frequency scale: Daily TV hours reported on a low-frequency category scale; High-frequency scale: Daily TV hours reported on a high-frequency category scale (p. 4)

## Conditions

### Low-versus-high category scales: Low-frequency scale

`s05_c_low`

- **Description**: Daily TV hours reported on a low-frequency category scale (p. 4)

### Low-versus-high category scales: High-frequency scale

`s05_c_high`

- **Description**: Daily TV hours reported on a high-frequency category scale (p. 4)

## Materials

### Low-versus-high category scales: participant materials

`s05_mat`

- **Description of the material as reported**: See evidence quote (paper description of the original and replication stimulus) (p. 4)
- **Full participant-visible wording (instructions, items, response options)** (verbatim, sources/Measures_US_Version.docx, p. 4):
  > [low]
  > How much television do you watch per day?  
  >
  > Drop-down scale with options:
  > Up to a half hour
  > Half an hour to an hour
  > One to one and a half hours
  > One and a half to two hours
  > Two to two and a half hours
  > More than two and a half hours
  >
  > [high]
  > How much television do you watch per day?  
  >
  > Drop-down scale with options:
  > Up to two and a half hours
  > Two and a half to three hours
  > Three to three and a half hours
  > Three and a half to four hours
  > Four to four and a half hours
  > More than four and a half hours

## Procedure

Order of steps:

- Report daily TV on the low-frequency category scale (participant sees: Low-versus-high category scales: participant materials) (p. 4)
  - conditions: Low-versus-high category scales: Low-frequency scale
- Choose a TV-hours category (participant sees: Low-versus-high category scales: participant materials) (p. 4)
- Report daily TV on the high-frequency category scale (participant sees: Low-versus-high category scales: participant materials) (p. 4)
  - conditions: Low-versus-high category scales: High-frequency scale
- (back to: Choose a TV-hours category)

### Low-versus-high category scales: procedure

`s05_pc`

- **Task**: Choose a TV-hours category (p. 4)

## Variables

### Low-versus-high category scales: independent variable

`s05_viv`

- **Role [derived]**: iv (p. 7)
  - how it was derived: Condition is the grouping variable of the confirmatory test
- **Type**: categorical (Low-frequency scale, High-frequency scale) (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response record per participant

### Low-versus-high category scales: Daily TV watching category

`s05_vdv`

- **Role [derived]**: dv (p. 7)
  - how it was derived: Outcome of the confirmatory test
- **Type**: categorical choice (p. 4)
- **Definition**: Category chosen for daily TV watching; analysed as reporting more than two and a half hours (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response per participant (per topic for anchoring)
- **Response scale** (verbatim, sources/Measures_US_Version.docx, p. 4):
  > Low frequency condition:
  > Up to a half hour
  > Half an hour to an hour
  > One to one and a half hours
  > One and a half to two hours
  > Two to two and a half hours
  > More than two and a half hours
  >
  > High frequency condition:
  > Up to two and a half hours
  > Two and a half to three hours
  > Three to three and a half hours
  > Three and a half to four hours
  > Four to four and a half hours
  > More than four and a half hours

## Analysis

### Low-versus-high category scales: confirmatory analysis

`s05_an`

- **Aim**: Test whether the original effect replicates (p. 7)
- **Method**: Chi-square on the share reporting more than two and a half hours of TV by scale condition (p. 7)
- **Inputs [derived]**: Records of Daily TV watching category by condition after exclusions (p. 7)
  - how it was derived: Exclusions are those stated in the supplement

## Reported results

### Low-versus-high category scales: original finding

`s05_res_orig`

- **Result kind**: reported (p. 7)
- **Original result**: See evidence quote (p. 7)

### Low-versus-high category scales: replication result

`s05_res_rep`

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

- **How participants were allocated to conditions is not stated** [decision, `i_s05_dz_assignment`, s05_dz]: The paper reports a between-subjects comparison but not the allocation rule (random, balanced, other). Suggested: Confirm random assignment with equal probability, or supply the rule. (p. 7)
