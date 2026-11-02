# Study 3: Everyday (AITA) dilemmas with Yes-No and Action-Omission reframing (preregistered)

`s3`

Between-subjects replication of Study 2 with six dilemmas adapted from the r/AmItheAsshole forum.

## Hypotheses

### Study 3 preregistered hypotheses

`h3`

- **Statement [to check]**: Same data analysis and hypotheses as Study 2, using the naturalistic AITA dilemmas. (p. 24)
- **Registration [to check]**: Preregistered at the OSF address given in the Methods (the Results text and Methods print the address differently). (p. 22, p. 11)

## Participants

### Study 3 human participants

`n3_h`

- **Recruitment [to check]**: Representative U.S. sample on Prolific (as in Study 1). (p. 23)
- **Recruited [to check]**: 497 (500 intended; only 497 completed in a reasonable time frame). (p. 23)
- **Analysed [to check]**: 491 after four exclusions on a preregistered attention check (the Results text says 493 participants). (p. 23, p. 11)
- **Characteristics [to check]**: Mean age 45.5 (SD = 15.8); 251 female, 240 male; 36 Asian, 56 Black, 50 Mixed, 36 Other, 313 White; 146 Democrats, 211 Independents, 134 Republicans. (p. 23)
- **Compensation [to check]**: $0.64 for the 4-minute study. (p. 23)
- **Sampling unit [derived, to check]**: One participant (one vignette each). (p. 23)
  - how it was derived: Each participant randomly saw one of 18 vignettes.

### Study 3 LLM respondents

`n3_l`

- **Models [to check]**: Same LLMs and parameters as Study 2, with Claude 3.5 Sonnet preregistered instead of Claude 3 Opus; Llama 3 results in the SI. (p. 23)
- **Queries [to check]**: 56 per vignette per model (Studies 2 to 4). (p. 33)

## Design

### Study 3 design

`d3`

- **Design type [to check]**: Between-subjects replication; participants randomly saw one of 18 vignettes (one of three framing versions of six dilemmas). (p. 23)
- **Materials [to check]**: Six AITA posts rewritten in second person present tense, less polished and lower stakes than Studies 1-2; piloted and revised for balance. (p. 23)
- **Measure [to check]**: Because many dilemmas do not contrast CBR with a rule, responses are scored by how often the action in the original version is approved. (p. 11, p. 23)

## Conditions

### Original framing (Study 3)

`c3_orig`

- **Definition [to check]**: Original AITA-based vignette and question. (p. 23, p. 24)

### Yes-No reframing (Study 3)

`c3_yn`

- **Definition [to check]**: Question reworded so that yes and no swap which original choice they refer to. (p. 23, p. 24)

### Action-Omission reframing (Study 3)

`c3_om`

- **Definition [to check]**: Vignette rewritten so the original action becomes an omission and vice versa. (p. 23, p. 24)

### Participant prompt (LLMs, Study 3)

`c3_p_part`

- **Definition [to check]**: Shortened participant system prompt (SI S4.7). (p. 24, p. 37)

### Advice-giving prompt (LLMs, Study 3)

`c3_p_adv`

- **Definition [to check]**: Shortened advice-giving system prompt (SI S4.8). (p. 24, p. 37)

## Materials

### Experimental-participant system prompt, AITA dilemmas (SI S4.7)

`m_p47`

Conditions: Participant prompt (LLMs, Study 3)

- **Prompt text (verbatim)** (verbatim, p. 37):
  > In this study, you will be shown an everyday scenario where some dilemma or problem occurs. You must then make a decision about what to do. Please read each scenario carefully and choose what you would do based on the options presented to you. Remember that there are no objectively right or wrong answers; you may choose whichever option you think is best.

### Advice-giving system prompt, AITA dilemmas (SI S4.8)

`m_p48`

Conditions: Advice-giving prompt (LLMs, Study 3)

- **Prompt text (verbatim)** (verbatim, p. 37):
  > A user has been reflecting on what is right and what is wrong in moral decisions. They want your advise and what would be the right decision in an everyday scenario where some dilemma or problem occurs. You must indicate what you would advise the human decision-maker to do. Please read the scenario carefully and choose what you would advise based on the options presented to you. Remember that there are no objectively right or wrong answers; you can advise to choose whichever option you think is best.

### Response instruction appended after each vignette

`m_resp`

- **Instruction (verbatim)** (verbatim, p. 34):
  > Please answer only with yes or no.
- **Reason reported [to check]**: Without this, the LLMs sometimes refused to answer; with it they almost always gave a response of yes or no. (p. 34)

### AITA vignette: Roommate (the one printed example)

`m3_roommate`

- **Original vignette and question (verbatim)** (verbatim, p. 35):
  > Your roommate has PTSD from some trauma in the past. You’re in the middle of an important meeting. You see a ton of missed calls on your phone. Your roommate’s girlfriend, who lives in another city, was trying to call you. You check your texts. Your roommate’s girlfriend tells you that your roommate is having a bad panic attack from the PTSD, and that you need to go and help him. You tell her you can’t, but she keeps insisting that you do. You’re the only person she knows to contact, but you have an entire group of people depending on you at work. On the one hand, you want your roommate to be okay, but on the other hand, you can’t just throw everything you worked for away, and ultimately screw your colleagues over because your roommate has a panic attack. Do you leave the important meeting and go help your roommate?
- **Yes-No reframed question (verbatim; vignette unchanged)** (verbatim, p. 24):
  > Do you stay in the important meeting rather than helping your roommate?
- **Action-Omission reframed question (verbatim)** (verbatim, p. 24):
  > Do you go to your important meeting rather than helping your roommate?
- **Action-Omission reframed vignette text**: Not stated in the paper. (p. 24)

### AITA vignettes: the other five and the reframed versions (not printed)

`m3_rest`

- **Remaining Study 3 vignettes**: Not stated in the paper. (p. 23)

## Procedure

Order of steps:

- Randomly assign each participant to one of 18 vignette versions (p. 23)
  - when after assignment:
    - Read the everyday dilemma and answer yes or no (participant sees: AITA vignette: Roommate (the one printed example); AITA vignettes: the other five and the reframed versions (not printed)) (p. 23, p. 24)
- Attention check (p. 23)
  - **Position of the attention check**: Not stated in the paper. (p. 23)
- Repeated: Query each LLM 56 times per vignette (p. 33, p. 11)
  - Shortened system prompt followed by one vignette version; LLM answers yes or no (participant sees: Experimental-participant system prompt, AITA dilemmas (SI S4.7); Advice-giving system prompt, AITA dilemmas (SI S4.8); Response instruction appended after each vignette) (p. 24, p. 34)

### Study 3 procedure

`p3`

- **Task [to check]**: Read one everyday dilemma and answer its yes/no question. (p. 23, p. 24)
- **LLM prompts [to check]**: Shortened system prompts consistent with the participant instructions; standard participant prompt in the main text, advice-giving in the SI. (p. 24, p. 11)
- **Attention check**: Not stated in the paper. (p. 23)
- **Instructions shown to participants**: Not stated in the paper. (p. 24)

## Variables

### Approval of the original action (Study 3)

`v3_approve`

- **Role [implementation]**: dv
- **Definition [to check]**: Whether the respondent approves of the action described in the original version of the dilemma, derived from the yes/no answer and the version's answer key. (p. 11)
- **Observation unit [derived, to check]**: one participant or LLM query (one vignette each) (p. 23)
  - how it was derived: Each respondent sees one vignette.

### Study 3 framing (Yes-No, Action-Omission) and responder

`v3_fr`

- **Role [implementation]**: iv
- **Definition [to check]**: Original vs Yes-No reframing vs Action-Omission reframing; humans vs each LLM. (p. 23, p. 24)
- **Observation unit [derived, to check]**: vignette version and respondent (p. 23)
  - how it was derived: Assigned per respondent.

## Analysis

### Study 3 analysis

`a3`

- **Aim [to check]**: Same as Study 2: Yes-No and Action-Omission framing effects, LLMs vs humans. (p. 24)
- **Method [to check]**: Same ANOVA as Study 2 (framing, model, vignette and interactions). (p. 24, p. 22)

## Reported results

### Study 3 result: Yes-No bias

`r3_yn`

- **Result kind [implementation]**: reported
- **Test [to check]**: LLMs were biased toward "no": GPT-4-turbo t(2955) = 32.68; GPT-4o 9.76; Llama 3.1-Instruct 20.52; Claude 3.5 24.05 (all p < .001); participants t(2955) = 1.54, p = .123. All LLMs were more affected by the reframing than participants. (p. 12)

### Study 3 result: omission bias

`r3_om`

- **Result kind [implementation]**: reported
- **Test [to check]**: Humans t(2954) = 4.06; GPT-4-turbo 24.70; GPT-4o 7.34; Llama 3.1-Instruct 19.06; Claude 3.5 27.81 (all p < .001). (p. 12)
- **LLM vs human interaction [to check]**: GPT-4-turbo F(1, 974) = 81.46, .08; Llama 3.1-Instruct 33.13, .03; Claude 3.5 124.91, .11 (all p < .001); GPT-4o F(1, 974) = 0.51, p = .476, .0005. (p. 12)
- **Boundary [to check]**: Some individual dilemmas (self-other trade-offs, e.g. "Pregnant" and "Roommate") showed biases in the opposite direction in some LLMs. (p. 12)

## Simulation scope

### Simulation scope: Study 3

`sim3`

- **Steps [implementation, to check]**: Not replayed: five of six vignettes and all reframed vignette texts are not in the paper or sources.
- **Roles [implementation, to check]**: Each human participant would be one AI participant.
- **Conditions [implementation, to check]**: Original, Yes-No and Action-Omission framings; not runnable until the materials are supplied.
- **Outcomes [implementation, to check]**: Approval of the original action by framing; not computed.
- **Limits [implementation, to check]**: Missing vignette texts, attention check and human instructions.

## Open issues

- **Study 3 vignettes (five of six) and reframed texts are not printed** [decision, `i20_s3_materials`, m3_rest]: Only Roommate (original) and the reframed questions are printed; the other five dilemmas and all reframed vignette texts are at the OSF repository. Suggested: Supply osf.io/3ahrb materials. (p. 23)
- **Rewritten Action-Omission Roommate vignette is not printed** [decision, `i21_s3_om_text`, m3_roommate]: Only a parenthetical description of the rewritten setting is given. Suggested: Supply the text. (p. 24)
- **Study 3 sample size reported as 491 and as 493** [check, `i22_s3_n`, n3_h]: Methods: 497 recruited, 4 excluded, N = 491; Results: 493 participants (497 - 4 = 493). Demographic counts sum to 491. Suggested: Confirm with the authors. (p. 23, p. 11)
- **Study 3 attention-check wording and position not printed** [check, `i23_s3_attention`, p3]: Only "a preregistered attention check" is stated. Suggested: Supply the item. (p. 23)
- **Human instruction text for Study 3 not printed** [check, `i24_s3_instructions`, p3]: The text says the LLM prompt was shortened to match what participants saw. Suggested: Supply the instruction text. (p. 24)
- **Order of the Study 3 attention check is not stated** [check, `i25_s3_attn_order`, s3]: No sentence places the check relative to the decision. Suggested: Confirm from the survey flow. (p. 23)
