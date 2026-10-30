# Study 9: Flag priming

`s09`

Flag priming (Carter, Ferguson, & Hassin, 2011; Study 2)

## Hypotheses

### Hypothesis: Flag priming

`hyp_s09`

- **Statement**: Subtle exposure to the American flag increases conservatism: flag-primed participants indicate more conservative political positions than controls. (p. 5, p. 8)

## Participants

### Flag priming: participants analysed

`s09_pt`

- **N in the analysis (Table 2)**: Table 2 row: Flag priming 0.50 ... 4,896 0.38 (N column); sample is the shared 36-site sample (p. 7)
- **Exclusions**: Reported in the supplement (see evidence) (p. 8)
- **Sampling unit [derived]**: participant
  - how it was derived: Each participant provides one response record per study

## Design

### Flag priming: design

`s09_dz`

- **Design type [derived]**: Between-subjects, two conditions (p. 8)
  - how it was derived: The analysis compares two conditions with an independent-samples test, which implies a between-subjects comparison
- **Assignment**: Not stated in the paper. (p. 8)
- **Conditions**: Flag prime: American flag appears in two of four photos; Control: Same photos without flags (p. 5)

## Conditions

### Flag priming: Flag prime

`s09_c_flag`

- **Description**: American flag appears in two of four photos (p. 5)

### Flag priming: Control

`s09_c_control`

- **Description**: Same photos without flags (p. 5)

## Materials

### Flag priming: participant materials

`s09_mat`

- **Description of the material as reported**: See evidence quote (paper description of the original and replication stimulus) (p. 5)
- **Printed text available in the sources** (verbatim, p. 15):
  > Quoted in the evidence record
- **Full participant-visible wording (instructions, items, response options)** (verbatim, sources/Measures_US_Version.docx, p. 5):
  > Flag priming
  > Screenshot from flag prime condition:
  > Screenshot from control condition:
  >
  > Flag priming DV:
  > All items were on 7-point scales (radio buttons):
  > For each statement below, indicate the degree to which you agree or disagree according to how you feel right now.
  > 1) Laws designed to protect the environment pose too high a cost on businesses that contribute to the economy. 
  > 1 – Completely disagree
  > 7 – Completely agree
  > 2) A woman should have the right to choose what to do with her body, even if that means getting an abortion. <<Reverse scored>>
  > 1 – Completely disagree
  > 7 – Completely agree
  > 3) The United States should not have invaded Iraq. <<Reverse scored>>
  > 1 – Completely disagree
  > 7 – Completely agree
  > 4) Homosexuals should have the same right to marriage as anyone else. <<Reverse scored>>
  > 1 – Completely disagree
  > 7 – Completely agree
  > 5) Affirmative action gives those groups with a history of oppression a chance to get ahead. <<Reverse scored>>
  > 1 – Completely disagree
  > 7 – Completely agree
  > 6) Gun control laws are not nearly strict enough. <<Reverse scored>>
  > 1 – Completely disagree
  > 7 – Completely agree
  > 7) It is important for our legal system to use the death penalty as punishment for heinous crimes.
  > 1 – Completely disagree
  > 7 – Completely agree
  > 8) Stem Cell research has important implications for medical advances, and should be pursued at all costs. <<Reverse scored>>
  > 1 – Completely disagree
  > 7 – Completely agree

## Procedure

Order of steps:

- Estimate time of day for four photos, two containing the American flag (participant sees: Flag priming: participant materials) (p. 5)
  - conditions: Flag priming: Flag prime
- Answer the 8 political-attitude items (participant sees: Flag priming: participant materials) (p. 5)
- Estimate time of day for four photos without flags (participant sees: Flag priming: participant materials) (p. 5)
  - conditions: Flag priming: Control
- (back to: Answer the 8 political-attitude items)
- Answer the three flag-priming moderator items at the end of the session (participant sees: Flag priming: moderator items (verbatim)) (p. 6)

### Flag priming: procedure

`s09_pc`

- **Task**: Answer the 8 political-attitude items (p. 5)
- **Changes in the replication**: See evidence quote (p. 5)

## Variables

### Flag priming: independent variable

`s09_viv`

- **Role [derived]**: iv (p. 8)
  - how it was derived: Condition is the grouping variable of the confirmatory test
- **Type**: categorical (Flag prime, Control) (p. 5)
- **Observation unit [derived]**: participant
  - how it was derived: One response record per participant

### Flag priming: Political attitude score (8 items)

`s09_vdv`

- **Role [derived]**: dv (p. 8)
  - how it was derived: Outcome of the confirmatory test
- **Type**: composite of 8 items (alpha=.68) (p. 5)
- **Definition**: 8-item questionnaire on views toward political issues (e.g. abortion, gun control, affirmative action); scored on a Republican-Democratic scale (p. 5)
- **Observation unit [derived]**: participant
  - how it was derived: One response per participant (per topic for anchoring)
- **Response scale** (verbatim, sources/Measures_US_Version.docx, p. 5):
  > All items were on 7-point scales (radio buttons):
  > 1 – Completely disagree
  > 7 – Completely agree
- **Scoring direction [derived]**: Items 2, 3, 4, 5, 6 and 8 are marked <<Reverse scored>> in the printed instrument. (sources/Measures_US_Version.docx)
  - how it was derived: Read from the <<Reverse scored>> markers printed after each item

### Flag priming: moderator items (verbatim)

`s09_vmod`

- **Role**: moderator (p. 15)
- **Items** (verbatim, p. 15):
  > Three items administered at the end of the session; wording in evidence
- **Observation unit [derived]**: participant
  - how it was derived: One response per participant

## Analysis

### Flag priming: confirmatory analysis

`s09_an`

- **Aim**: Test whether the original effect replicates (p. 8)
- **Method**: Independent t-test of the flag vs control attitude composite; US samples only; participants missing any of the four time-of-day estimates excluded (p. 8)
- **Inputs [derived]**: Records of Political attitude score (8 items) by condition after exclusions (p. 8)
  - how it was derived: Exclusions are those stated in the supplement

## Reported results

### Flag priming: original finding

`s09_res_orig`

- **Result kind**: reported (p. 8)
- **Original result**: See evidence quote (p. 8)

### Flag priming: replication result

`s09_res_rep`

- **Result kind**: reported (p. 8)
- **Test and descriptives (supplement)**: See evidence quotes (p. 8)
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

- **How participants were allocated to conditions is not stated** [decision, `i_s09_dz_assignment`, s09_dz]: The paper reports a between-subjects comparison but not the allocation rule (random, balanced, other). Suggested: Confirm random assignment with equal probability, or supply the rule. (p. 8)
- **Flag priming: the manipulation is a set of photographs** [decision, `i_s09_image`, s09_mat]: The four photos differ between flag and control conditions; the instrument document only has screenshot placeholders and the paper says only that the flag appears in two of four photos. A text-only AI participant cannot see them, and the sources hold no verbatim description. Suggested: Decide whether to run it with a vision-capable model and the images in sources/materials_html/images, or leave it out. (sources/Measures_US_Version.docx)
