# Study 3: Anchoring

`s03`

Anchoring (Jacowitz & Kahneman, 1995), four topics

## Participants

### Anchoring: participants analysed

`s03_pt`

- **N in the analysis (Table 2)**: Table 2 row: Anchoring – Chicago 0.93 ... 5,284 <.001 (N column); sample is the shared 36-site sample (p. 7)
- **Exclusions**: Reported in the supplement (see evidence) (p. 3)
- **Sampling unit [derived]**: participant
  - how it was derived: Each participant provides one response record per study

## Design

### Anchoring: design

`s03_dz`

- **Design type**: Between-subjects, two conditions (p. 5)
- **Assignment**: Random assignment (p. 7)
- **Conditions**: Low anchor: Receives a low anchor value before estimating; High anchor: Receives a high anchor value before estimating (p. 4)

## Conditions

### Anchoring: Low anchor

`s03_c_low`

- **Description**: Receives a low anchor value before estimating (p. 4)

### Anchoring: High anchor

`s03_c_high`

- **Description**: Receives a high anchor value before estimating (p. 4)

## Materials

### Anchoring: participant materials

`s03_mat`

- **Description of the material as reported**: See evidence quote (paper description of the original and replication stimulus) (p. 4)
- **Full participant-visible wording (instructions, items, response options)** (verbatim, sources/Measures_US_Version.docx, sources/materials_html/anchoring1b.html, p. 4):
  > [high]
  > We are interested in people's beliefs about the world around them. For each of the following 4 questions we would like you to give your best estimate. If you do not know the correct answer, just give your best guess.
  >
  > The distance from San Francisco to New York City is shorter than 6,000 miles. How far do you think it is?
  > (miles)
  >
  > The population of Chicago is less than 5,000,000. What do you think the population of Chicago is?
  > (people)
  >
  > Mount Everest is shorter than 45,500 feet. How tall do you think Mount Everest is?
  > (feet)
  >
  > Less than 50,000 babies are born per day in the United States. How many babies do you think are born in the U.S. each day?
  > (babies)
  >
  > [low]
  > We are interested in people's beliefs about the world around them. For each of the following 4 questions we would like you to give your best estimate. If you do not know the correct answer, just give your best guess.
  >
  > The distance from San Francisco to New York City is longer than 1,500 miles. How far do you think it is?
  > (miles)
  >
  > The population of Chicago is more than 200,000. What do you think the population of Chicago is?
  > (people)
  >
  > Mount Everest is taller than 2,000 feet. How tall do you think Mount Everest is?
  > (feet)
  >
  > More than 100 babies are born per day in the United States. How many babies do you think are born in the U.S. each day?
  > (babies)

## Procedure

Order of steps:

- See the low anchor for each topic (participant sees: Anchoring: participant materials) (p. 4)
  - conditions: Anchoring: Low anchor
- Estimate each of the four quantities (participant sees: Anchoring: participant materials) (p. 4)
- See the high anchor for each topic (participant sees: Anchoring: participant materials) (p. 4)
  - conditions: Anchoring: High anchor
- (back to: Estimate each of the four quantities)

### Anchoring: procedure

`s03_pc`

- **Task**: Estimate each of the four quantities (p. 4)

## Variables

### Anchoring: independent variable

`s03_viv`

- **Role [derived]**: iv (p. 5)
  - how it was derived: Condition is the grouping variable of the confirmatory test
- **Type**: categorical (Low anchor, High anchor) (p. 4)
- **Observation unit [derived]**: participant
  - how it was derived: One response record per participant

### Anchoring: Estimated quantity (four topics)

`s03_vdv`

- **Role [derived]**: dv (p. 5)
  - how it was derived: Outcome of the confirmatory test
- **Type**: continuous (open numeric) (p. 4)
- **Definition**: Open-ended numeric estimate for distance San Francisco to New York City, population of Chicago, height of Mt. Everest and babies born per day in the US (p. 4)
- **Observation unit [derived]**: participant x topic
  - how it was derived: One response per participant (per topic for anchoring)
- **Response scale**: Open-ended numeric estimate (no response scale) (p. 4)

## Analysis

### Anchoring: confirmatory analysis

`s03_an`

- **Aim**: Test whether the original effect replicates (p. 3)
- **Method**: t-tests on rank-transformed estimates; responses outside the anchor end points set to missing (p. 5)
- **Inputs [derived]**: Records of Estimated quantity (four topics) by condition after exclusions (p. 3)
  - how it was derived: Exclusions are those stated in the supplement

## Reported results

### Anchoring: original finding

`s03_res_orig`

- **Result kind**: reported (p. 3)
- **Original result**: See evidence quote (p. 3)

### Anchoring: replication result

`s03_res_rep`

- **Result kind**: reported (p. 3, p. 5)
- **Test and descriptives (supplement)**: See evidence quotes (p. 3, p. 5)
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

- **Rank-transformed anchoring statistics in the supplement look internally inconsistent** [check, `i_s03_suppdisc`, s03_res_rep]: The supplement lists Anchoring 1 (NYC) with t(5607)=90.49, d=1.17 and Anchoring 4 (babies) with the same t(5607)=90.49, while Table 2 gives NYC t=42.86 and babies t=90.49. Suggested: Use the Table 2 values as primary and check against the OSF data. (p. 5)
- **SF–NY anchoring: unit label differs between the instrument document and the study page** [check, `i_s03_unit_label`, s03_mat]: Measures_US_Version labels the answer box for the 'shorter than 6,000 miles' item “(babies)”, but the study page anchoring1b.html labels it “(miles)”. Materials use “(miles)”, as in the study page and the low-anchor version. Suggested: Confirm “(miles)” is correct, or supply the label as run. (sources/Measures_US_Version.docx, sources/materials_html/anchoring1b.html)
