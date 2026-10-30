# Study 1: Sunk costs

`s01`

Sunk costs (Oppenheimer, Meyvis, & Davidenko, 2009)

## Hypotheses

### Hypothesis: Sunk costs

`hyp_s01`

- **Statement**: Having paid for a ticket (a sunk cost) makes people more likely to go to a game on a freezing day than having received the ticket for free. (p. 3, p. 2)

## Participants

### Sunk costs: participants analysed

`s01_pt`

- **N in the analysis (Table 2)**: Table 2 row: Sunk costs 0.23 ... 6,330 <.001 (N column); sample is the shared 36-site sample (p. 7)
- **Exclusions**: Reported in the supplement (see evidence) (p. 2)
- **Sampling unit [derived]**: participant
  - how it was derived: Each participant provides one response record per study

## Design

### Sunk costs: design

`s01_dz`

- **Design type [derived]**: Between-subjects, two conditions (p. 2)
  - how it was derived: The analysis compares two conditions with an independent-samples test, which implies a between-subjects comparison
- **Assignment**: Not stated in the paper. (p. 2)
- **Conditions**: Paid ticket: Imagines having paid for a ticket to a football game; Free ticket: Imagines having received the ticket for free (p. 3)

## Conditions

### Sunk costs: Paid ticket

`s01_c_paid`

- **Description**: Imagines having paid for a ticket to a football game (p. 3)

### Sunk costs: Free ticket

`s01_c_free`

- **Description**: Imagines having received the ticket for free (p. 3)

## Materials

### Sunk costs: participant materials

`s01_mat`

- **Description of the material as reported**: See evidence quote (paper description of the original and replication stimulus) (p. 3)
- **Full participant-visible wording (instructions, items, response options)** (verbatim, sources/Measures_US_Version.docx, p. 3):
  > [paid]
  > Imagine that your favorite football team is playing an important game. You have a ticket to the game that you have paid handsomely for. However, on the day of the game, it happens to be freezing cold. What do you do?  
  > 9 point drop-down scale: 1 – Definitely stay at home, 9 – Definitely go to the game
  >
  > [free]
  > Imagine that your favorite football team is playing an important game. You have a ticket to the game that you have received for free from a friend. However, on the day of the game, it happens to be freezing cold. What do you do?  
  > 9 point drop-down scale: 1 – Definitely stay at home, 9 – Definitely go to the game

## Procedure

Order of steps:

- Imagine having a paid ticket to a freezing-cold game (participant sees: Sunk costs: participant materials) (p. 3)
  - conditions: Sunk costs: Paid ticket
- Rate likelihood of going to the game (1-9) (participant sees: Sunk costs: participant materials) (p. 3)
- Imagine having a free ticket to a freezing-cold game (participant sees: Sunk costs: participant materials) (p. 3)
  - conditions: Sunk costs: Free ticket
- (back to: Rate likelihood of going to the game (1-9))

### Sunk costs: procedure

`s01_pc`

- **Task**: Rate likelihood of going to the game (1-9) (p. 3)

## Variables

### Sunk costs: independent variable

`s01_viv`

- **Role [derived]**: iv (p. 2)
  - how it was derived: Condition is the grouping variable of the confirmatory test
- **Type**: categorical (Paid ticket, Free ticket) (p. 3)
- **Observation unit [derived]**: participant
  - how it was derived: One response record per participant

### Sunk costs: Likelihood of attending the game

`s01_vdv`

- **Role [derived]**: dv (p. 2)
  - how it was derived: Outcome of the confirmatory test
- **Type**: ordinal rating (1-9) (p. 3)
- **Definition**: Rated likelihood of attending the game on a 9-point scale (p. 3)
- **Observation unit [derived]**: participant
  - how it was derived: One response per participant (per topic for anchoring)
- **Response scale**: 9-point scale (1 = definitely stay at home, 9 = definitely go to the game) (p. 3)

## Analysis

### Sunk costs: confirmatory analysis

`s01_an`

- **Aim**: Test whether the original effect replicates (p. 2)
- **Method**: Independent samples t-test with condition (paid ticket, free ticket) as the independent variable and likelihood of going to the game as the dependent variable (p. 2)
- **Inputs [derived]**: Records of Likelihood of attending the game by condition after exclusions (p. 2)
  - how it was derived: Exclusions are those stated in the supplement

## Reported results

### Sunk costs: original finding

`s01_res_orig`

- **Result kind**: reported (p. 2)
- **Original result**: See evidence quote (p. 2)

### Sunk costs: replication result

`s01_res_rep`

- **Result kind**: reported (p. 2)
- **Test and descriptives (supplement)**: See evidence quotes (p. 2)
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

- **How participants were allocated to conditions is not stated** [decision, `i_s01_dz_assignment`, s01_dz]: The paper reports a between-subjects comparison but not the allocation rule (random, balanced, other). Suggested: Confirm random assignment with equal probability, or supply the rule. (p. 2)
