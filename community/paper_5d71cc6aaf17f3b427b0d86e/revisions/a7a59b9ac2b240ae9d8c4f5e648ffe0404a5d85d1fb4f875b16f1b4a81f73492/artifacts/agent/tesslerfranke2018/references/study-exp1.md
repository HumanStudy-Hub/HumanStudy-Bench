# Experiment 1: Single utterances

`exp1`

Exploratory online rating study; each participant rated adjectives from morphological-antonym and lexical-antonym sets one sentence at a time.

## Hypotheses

### H1: Partial vs. full ordering by antonym type

`h-order`

- **Statement**: In isolation (single utterance), unhappy (morphological antonym) is not distinguished from not happy (partial ordering), whereas lexical antonyms (short, sad) are interpreted more negatively than negated positives (full ordering). (p. 3, p. 4)
- **Scope**: Interpretations of adjectives describing properties of people, rated on a slider between the most positive and most negative person. (p. 4)

## Participants

### Exp 1 participants

`p1`

- **Setting**: Online, Amazon Mechanical Turk. (p. 4)
- **Roles [derived]**: All participants are raters; no scripted roles are described. (p. 4)
  - how it was derived: The paper describes only raters; no confederates or experimenters are described.
- **Recruitment**: Mechanical Turk; U.S. IP addresses and at least a 95% work approval rating. (p. 4)
- **Eligibility**: Participants who self-reported a native language other than English were excluded. (p. 4)
- **N recruited**: 120 (p. 4)
- **N analysed**: 114 (p. 4)
- **N completed [to check]**: Not separately reported (recruited 120; 6 excluded for native language). (p. 4)
- **Sampling unit [derived]**: Individual participant (each rated 16 sentences). (p. 4)
  - how it was derived: One participant completed 16 trials.
- **Compensation**: $0.40 for an average of 3 minutes. (p. 4)
- **Characteristics**: Not stated in the paper. (p. 4)

## Design

### Exp 1 design

`d1`

- **Type [derived]**: Within-participant rating task: adjective type (positive, antonym, negated positive, negated antonym) crossed with antonym type (morphological vs. lexical), both varying within a participant. (p. 4)
  - how it was derived: Participants 'saw items from both antonym types throughout' and 16 trials = 2 repetitions of each adjective type for each antonym type.
- **Control**: Lexical antonyms (e.g., short) serve as a control: they lack overt negation markers. (p. 4)
- **Assignment [derived]**: No between-participant manipulation; all participants complete the same kind of 16-trial design. (p. 4)
  - how it was derived: No between-subject factor is described for Expt. 1.
- **Order**: Not stated in the paper. (p. 4)

## Materials

### Exp 1 adjective sets (Table 1)

`m1-items`

- **Morphological antonym sets** (verbatim, p. 4):
  > attractive, unattractive; educated, uneducated; friendly, unfriendly; happy, unhappy; honest, dishonest; intelligent, unintelligent; interesting, uninteresting; mature, immature; polite, impolite; successful, unsuccessful
- **Lexical antonym sets** (verbatim, p. 4):
  > beautiful, ugly; brave, cowardly; fat, skinny; hard-working, lazy; loud, quiet; proud, humble; rich, poor; strong, weak; tall, short; wise, foolish
- **Construction**: 10 adjective sets per antonym type (total 20) from an informal survey of the linguistics literature and a list of common opposites available online; each trial used a distinct adjective set. (p. 4)

### Exp 1 response scale

`m1-scale`

- **Slider endpoints** (verbatim, p. 4):
  > from "the most positive person" to "the most antonym person" (item-dependent)
- **Numeric range and start position**: Not stated in the paper. (p. 4)

### Exp 1 instructions and sentence frames

`m1-text`

- **Instructions**: Not stated in the paper. (p. 4)
- **Sentence frame and character names**: Not stated in the paper. (p. 4)

## Procedure

Order of steps:

- Exp 1 session (p. 4)
  - Repeated: 16 rating trials (p. 4)
    - **Repeat rule**: 16 trials, exactly 2 repetitions of each adjective type for each antonym type, each from a distinct adjective set. (p. 4)
    - Read one sentence and rate the person on the slider (participant sees: Exp 1 adjective sets (Table 1); Exp 1 response scale) (p. 4)

### Exp 1 procedure

`proc1`

- **Task**: On each trial, read a statement introducing a person using a gradable adjective and rate the character on a slider. (p. 4)
- **Actions**: One sentence at a time; items from both antonym types; 16 trials with exactly 2 repetitions of each adjective type for each antonym type; each trial from a distinct adjective set. (p. 4)
- **Visible [derived]**: One statement at a time; no responses of other participants are shown. (p. 4)
  - how it was derived: 'Participants rated one sentence at a time.'
- **Feedback [derived, to check]**: None described. (p. 4)
  - how it was derived: The paper describes no feedback.

## Variables

### Exp 1 slider rating

`v1-rating`

- **Role**: dv (p. 4)
- **Type**: Continuous slider rating (raw, unnormalized) from the most antonym person to the most positive person. (p. 4)
- **Observation unit [derived]**: One rating per participant per trial. (p. 4)
  - how it was derived: 16 trials per participant.
- **Normalization [to check]**: Participant-wise normalized response r'ij computed from the participant's mean and SD for visualization; formula glyphs are garbled in the text layer. (p. 4)

### Exp 1 adjective type

`v1-adj`

- **Role**: iv (p. 4)
- **Type** (verbatim, p. 4):
  > Categorical, 4 levels, Helmert coded in order: antonym, negated positive, negated antonym, positive.
- **Observation unit [derived]**: Trial (p. 4)
  - how it was derived: Varies by trial.

### Exp 1 antonym type

`v1-ant`

- **Role**: iv (p. 4)
- **Type**: Categorical: morphological vs. lexical (varies within participant, by item). (p. 4)
- **Observation unit [derived]**: Adjective set / trial (p. 4)
  - how it was derived: Varies by item.

## Analysis

### Exp 1 linear mixed model

`an1-lmm`

- **Aim**: Test whether the antonym vs. negated-positive contrast differs between morphological and lexical antonyms. (p. 4)
- **Inputs**: Raw unnormalized ratings, 114 analysed participants. (p. 4)
- **Method**: Linear mixed model: fixed effects of antonym type, adjective type (Helmert) and their interaction; random intercepts and random slopes of adjective type by participant and by item (the maximal model that converged and explained significantly more variance; lme4 in R). (p. 4)
- **Criterion [to check]**: Interaction of antonym vs. negated-positive contrast with antonym type; significance level not stated. (p. 4)

### Exp 1 normalized means

`an1-vis`

- **Aim**: Visualize the ordering of the four adjective types within morphological and lexical antonym sets. (p. 4)
- **Method**: Participant-wise normalized responses; mean and bootstrapped 95% confidence intervals per adjective type and antonym type. (p. 4)

## Reported results

### Exp 1: antonym type x (antonym vs. negated positive)

`res1-int`

- **Result kind**: reported (p. 4)
- **Test** (verbatim, p. 4):
  > b = 0.029, t(16) = 2.4, p = 0.029
- **Effect**: The antonym vs. negated positive difference interacted significantly with antonym type (morphological vs. lexical). (p. 4)

### Exp 1: ordering pattern

`res1-ord`

- **Result kind**: reported (p. 4)
- **Descriptives [to check]**: Morphological antonym sets show only a partial ordering; lexical antonym sets show a full ordering. Mean normalized values are shown only in Fig. 5A, not tabulated. (p. 4)
- **Themes**: Negated morphological antonyms rated lower than negated lexical antonyms; negated antonyms received a bimodal distribution (mostly slightly positive, a minority slightly negative), present in every item and most participants. (p. 5)

## Simulation scope

### Exp 1 simulation scope

`sim1`

- **Steps [implementation]**: Every rating trial can be replayed as a text prompt: statement plus slider description, one AI participant per original participant. (p. 4)
- **Roles [implementation]**: All original participants are AI participants; no scripted roles exist in the paper. (p. 4)
- **Conditions [implementation]**: All arms run as published. (p. 4)
- **Outcomes [implementation]**: Slider ratings are collected as numbers; the paper's linear mixed models are approximated offline (see evaluation), since lme4 is not available. (p. 4)
- **Limits [implementation]**: Slider is replaced by a typed number on an assumed 0-100 scale; instruction and sentence wording are replay adaptations; MTurk population, attention and the native-language screen cannot be reproduced; AI agreement does not show human behavior. (p. 4)

## Open issues

- **Exp 1 instruction text not printed** [decision, `i1-instr`, m1-text]: The paper does not print the instructions shown to participants. Suggested: Supply the original instructions from the authors' materials, or accept a minimal neutral framing recorded as a replay adaptation. (p. 4)
- **Exp 1 sentence frame and names not printed** [decision, `i1-frame`, m1-text]: Only Expt. 2 gives an example sentence ("Greg is not unhappy"); no frame or character names are listed for Expt. 1. Suggested: Supply the original sentences, or accept the 'NAME is (not) ADJECTIVE' frame from the Expt. 2 example as a replay adaptation. (p. 4)
- **Exp 1 slider numeric range not stated** [decision, `i1-slider`, m1-scale]: Slider range and start position are not stated, and how 'positive' and 'antonym' in the endpoint labels are filled in per item is not shown in text (Fig. 4A). Suggested: Supply the original slider range and endpoint wording, or accept a documented 0-100 range with the item's own adjective words in the endpoint labels as a replay adaptation. (p. 4)
- **Exp 1 trial order and item sampling not stated** [decision, `i1-order`, d1]: The order of the 16 trials and which 8 of 10 adjective sets per antonym type each participant sees are not stated. Suggested: Supply the original randomization, or accept random sampling without replacement per participant as a replay adaptation. (p. 4)
- **Exp 1 demographics not reported** [check, `i1-demo`, p1]: Age, gender and other participant characteristics are not reported. Suggested: Record that no persona is specified; do not invent demographics. (p. 4)
- **Exp 1 completion count not separately reported** [check, `i1-completed`, p1]: n completed is not given separately. Suggested: Treat 120 as recruited and 114 as analysed. (p. 4)
- **Exp 1 normalization formula garbled** [check, `i1-norm`, v1-rating]: The normalization formula is garbled in the PDF text layer. Suggested: Check the formula on the page image; the text suggests rating minus participant mean, divided by participant SD. (p. 4)
- **Exp 1 means shown only in a figure** [check, `i1-desc`, res1-ord]: Mean normalized ratings are plotted in Fig. 5A but not tabulated. Suggested: Obtain cell means from the authors' data, or compare orderings only. (p. 4)
- **Exp 1 significance criterion not stated** [check, `i1-crit`, an1-lmm]: The alpha level is not stated. Suggested: Use the reported p-values; assume the conventional .05 only if the researcher confirms. (p. 4)
- **Native-language exclusion cannot apply to AI participants** [check, `i1-excl`, p1]: Exclusion for self-reported non-English native language relies on a question not printed in the paper. Suggested: Record as a replay adaptation. (p. 4)
