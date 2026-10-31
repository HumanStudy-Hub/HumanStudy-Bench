# Study 2: Gain versus loss framing

`s02`

Gain versus loss framing (Tversky & Kahneman, 1981)

## Hypotheses

### Hypothesis: Gain versus loss framing

`hyp_s02`

- **Statement**: Framing logically identical programs as gains (lives saved) rather than losses (lives lost) decreases willingness to take risks: the sure option is chosen more in the gain frame. (p. 4, p. 2)

## Participants

### Gain versus loss framing: participants analysed

`s02_pt`

- **N in the analysis (Table 2)**: Table 2 row: Gain vs. loss framing 1.13 ... 6,271 <.001 (N column); sample is the shared 36-site sample (p. 7)
- **Exclusions**: Reported in the supplement (see evidence) (p. 3)
- **Sampling unit [derived]**: participant
  - how it was derived: Each participant provides one response record per study

## Design

### Gain versus loss framing: design

`s02_dz`

- **Design type**: Between-subjects, two conditions (p. 3)
- **Assignment**: Not stated in the paper. (p. 3)
- **Conditions**: Gain frame: Programs described by lives saved (Program A / Program B); Loss frame: Programs described by lives lost (Program C / Program D) (p. 4)

## Conditions

### Gain versus loss framing: Gain frame

`s02_c_gain`

- **Description**: Programs described by lives saved (Program A / Program B) (p. 4)

### Gain versus loss framing: Loss frame

`s02_c_loss`

- **Description**: Programs described by lives lost (Program C / Program D) (p. 4)

## Materials

### Gain versus loss framing: participant materials

`s02_mat`

- **Description of the material as reported**: See evidence quote (paper description of the original and replication stimulus) (p. 4)
- **Full participant-visible wording (instructions, items, response options)** (verbatim, sources/Measures_US_Version.docx, p. 4):
  > [gain]
  > Imagine that the United States is preparing for the outbreak of an unusual disease, which is expected to kill 600 people. Two alternative programs to fight the disease have been proposed. Assume that the exact scientific estimate of the consequences of the program are as follows:
  > If Program A is adopted, 200 people will be saved 
  > If Program B is adopted, there is 1/3 probability that 600 people will be saved and 2/3 probability that no people will be saved.
  > Which program would you choose?
  > Choose program:
  > Program A
  > Program B
  >
  > [loss]
  > Imagine that the United States is preparing for the outbreak of an unusual disease, which is expected to kill 600 people. Two alternative programs to fight the disease have been proposed. Assume that the exact scientific estimate of the consequences of the program are as follows:
  >
  > If Program C is adopted, 400 people will die. 
  >
  > If Program D is adopted, there is 1/3 probability that no people will die and 2/3 probability that 600 people will die.
  >
  > Which program would you choose?
  > Choose program:
  > Program C
  > Program D

## Procedure

Order of steps:

- Read the disease scenario with gain-framed programs (participant sees: Gain versus loss framing: participant materials) (p. 4)
  - conditions: Gain versus loss framing: Gain frame
- Choose a program (participant sees: Gain versus loss framing: participant materials) (p. 4, p. 2)
- Read the disease scenario with loss-framed programs (participant sees: Gain versus loss framing: participant materials) (p. 4)
  - conditions: Gain versus loss framing: Loss frame
- (back to: Choose a program)

### Gain versus loss framing: procedure

`s02_pc`

- **Task**: Choose a program (p. 4)
- **Changes in the replication**: See evidence quote (p. 4)

## Variables

### Gain versus loss framing: independent variable

`s02_viv`

- **Role [derived]**: iv (p. 3)
  - how it was derived: Condition is the grouping variable of the confirmatory test
- **Type**: categorical (Gain frame, Loss frame) (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response record per participant

### Gain versus loss framing: Program choice (sure option vs gamble)

`s02_vdv`

- **Role [derived]**: dv (p. 3)
  - how it was derived: Outcome of the confirmatory test
- **Type**: binary choice (p. 2)
- **Definition**: Choice between a certain program and a probabilistic program (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response per participant (per topic for anchoring)
- **Response scale**: Two-alternative choice: gain frame Program A or B; loss frame Program C or D (p. 2)

## Analysis

### Gain versus loss framing: confirmatory analysis

`s02_an`

- **Aim**: Test whether the original effect replicates (p. 2)
- **Method**: Chi-square analysis on program choice with condition (gain-frame or loss-frame) as a between-subjects variable (p. 3)
- **Inputs [derived]**: Records of Program choice (sure option vs gamble) by condition after exclusions (p. 3)
  - how it was derived: Exclusions are those stated in the supplement

## Reported results

### Gain versus loss framing: original finding

`s02_res_orig`

- **Result kind**: reported (p. 2)
- **Original result**: See evidence quote (p. 2)

### Gain versus loss framing: replication result

`s02_res_rep`

- **Result kind**: reported (p. 2, p. 3)
- **Test and descriptives (supplement)**: See evidence quotes (p. 2, p. 3)
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

- **How participants were allocated to conditions is not stated** [decision, `i_s02_dz_assignment`, s02_dz]: The paper reports a between-subjects comparison but not the allocation rule (random, balanced, other). Suggested: Confirm random assignment with equal probability, or supply the rule. (p. 3)
