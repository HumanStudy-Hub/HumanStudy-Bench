# Study 7: Allowed/Forbidden

`s07`

Allowed/Forbidden (Rugg, 1941)

## Hypotheses

### Hypothesis: Allowed/forbidden

`hyp_s07`

- **Statement**: People are more willing to not allow speeches against democracy than to forbid them: more answer 'No' when asked whether to allow them than answer 'Yes' when asked whether to forbid them. (p. 4)

## Participants

### Allowed/Forbidden: participants analysed

`s07_pt`

- **N in the analysis (Table 2)**: Table 2 row: Allowed/forbidden 0.65 ... 6,292 <.001 (N column); sample is the shared 36-site sample (p. 7)
- **Exclusions**: Reported in the supplement (see evidence) (p. 7)
- **Sampling unit [derived]**: participant
  - how it was derived: Each participant provides one response record per study

## Design

### Allowed/Forbidden: design

`s07_dz`

- **Design type**: Between-subjects, two conditions (p. 7)
- **Assignment**: Not stated in the paper. (p. 7)
- **Conditions**: Allow wording: Should the country allow speeches against democracy?; Forbid wording: Should the country forbid speeches against democracy? (p. 4)

## Conditions

### Allowed/Forbidden: Allow wording

`s07_c_allow`

- **Description**: Should the country allow speeches against democracy? (p. 4)

### Allowed/Forbidden: Forbid wording

`s07_c_forbid`

- **Description**: Should the country forbid speeches against democracy? (p. 4)

## Materials

### Allowed/Forbidden: participant materials

`s07_mat`

- **Description of the material as reported**: See evidence quote (paper description of the original and replication stimulus) (p. 4)
- **Full participant-visible wording (instructions, items, response options)** (verbatim, sources/Measures_US_Version.docx, p. 4):
  > [forbid]
  > Do you think the United States should forbid public speeches against democracy?
  >
  > Yes
  > No
  >
  > [allow]
  > Do you think the United States should allow public speeches against democracy?
  >
  > Yes
  > No

## Procedure

Order of steps:

- Answer whether speeches against democracy should be allowed (participant sees: Allowed/Forbidden: participant materials) (p. 4)
  - conditions: Allowed/Forbidden: Allow wording
- Answer yes or no (participant sees: Allowed/Forbidden: participant materials) (p. 4)
- Answer whether speeches against democracy should be forbidden (participant sees: Allowed/Forbidden: participant materials) (p. 4)
  - conditions: Allowed/Forbidden: Forbid wording
- (back to: Answer yes or no)

### Allowed/Forbidden: procedure

`s07_pc`

- **Task**: Answer yes or no (p. 4)
- **Changes in the replication**: See evidence quote (p. 4)

## Variables

### Allowed/Forbidden: independent variable

`s07_viv`

- **Role [derived]**: iv (p. 7)
  - how it was derived: Condition is the grouping variable of the confirmatory test
- **Type**: categorical (Allow wording, Forbid wording) (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response record per participant

### Allowed/Forbidden: Yes/No response on speeches against democracy

`s07_vdv`

- **Role [derived]**: dv (p. 7)
  - how it was derived: Outcome of the confirmatory test
- **Type**: binary (p. 4)
- **Definition**: Yes/no answer to whether speeches against democracy should be allowed or forbidden (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response per participant (per topic for anchoring)
- **Response scale**: Yes/no (p. 4)

## Analysis

### Allowed/Forbidden: confirmatory analysis

`s07_an`

- **Aim**: Test whether the original effect replicates (p. 4)
- **Method**: Chi-square on policy choice with condition (allowed, forbidden) as a between-subjects variable (p. 7)
- **Inputs [derived]**: Records of Yes/No response on speeches against democracy by condition after exclusions (p. 7)
  - how it was derived: Exclusions are those stated in the supplement

## Reported results

### Allowed/Forbidden: original finding

`s07_res_orig`

- **Result kind**: reported (p. 4)
- **Original result**: See evidence quote (p. 4)

### Allowed/Forbidden: replication result

`s07_res_rep`

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

- **How participants were allocated to conditions is not stated** [decision, `i_s07_dz_assignment`, s07_dz]: The paper reports a between-subjects comparison but not the allocation rule (random, balanced, other). Suggested: Confirm random assignment with equal probability, or supply the rule. (p. 7)
