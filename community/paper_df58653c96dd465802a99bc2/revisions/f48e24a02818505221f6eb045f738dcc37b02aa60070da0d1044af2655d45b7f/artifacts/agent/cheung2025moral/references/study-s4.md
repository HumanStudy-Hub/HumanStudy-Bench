# Study 4: Effect of post-training on the biases (Llama 3.1 pre-trained, Llama 3.1-Instruct, Centaur)

`s4`

Three models answer the twelve dilemmas of Studies 2 and 3 and are compared with the human data already collected in those studies. No new human participants.

## Participants

### Study 4 model respondents

`n4_l`

- **Models [to check]**: Llama 3.1 (pre-trained), Llama 3.1-Instruct, and Centaur (Llama 3.1 post-trained on over 60,000 participants' behavior in over 160 psychological experiments). (p. 12)
- **Queries [to check]**: 56 per vignette per model (Studies 2 to 4). (p. 33)
- **Run environment [to check]**: Queried in Python by loading the low-rank adapter with unsloth, on RunPod with an H100 NVL GPU. (p. 33)
- **Model checkpoint links**: Not stated in the paper. (p. 33)

### Study 4 human reference data

`n4_h`

- **Source [to check]**: No new human participants: the human data collected in Studies 2 and 3 are reused. (p. 24)

## Design

### Study 4 design

`d4`

- **Design type [to check]**: Model comparison on the twelve dilemmas of Studies 2 and 3 with Yes-No and Action-Omission reframing, compared with the human data of those studies. (p. 13)
- **Prompting [to check]**: Participant prompt only. Centaur: adds "Press Y for answering yes or N for answering no. You press <<". Pre-trained Llama: yes/no put in quotation marks and "You answer with" appended; system and user prompts pasted together. Llama 3.1-Instruct: standard participant prompt (robustness check with the pre-trained model's prompt in the SI). (p. 24, p. 33)

## Conditions

### Llama 3.1 (pre-trained)

`c4_pre`

- **Definition [to check]**: Pre-trained Llama 3.1 model, no post-training. (p. 12)

### Llama 3.1-Instruct

`c4_inst`

- **Definition [to check]**: Llama 3.1 fine-tuned by Meta (RLHF and supervised learning) to follow instructions and align with human preferences. (p. 12)

### Centaur

`c4_cent`

- **Definition [to check]**: Llama 3.1 post-trained on over 60,000 participants' behavior in over 160 psychological experiments. (p. 12)

## Procedure

Order of steps:

- Repeated: Query each model 56 times per vignette (p. 33, p. 13)
  - Participant prompt (model-specific adjustments) followed by one vignette; model answers (participant sees: Response instruction appended after each vignette) (p. 24, p. 33)

### Study 4 procedure

`p4`

- **Task [to check]**: Each model answers one vignette per query with the participant prompt. (p. 13, p. 24)
- **Human protocol [to check]**: Humans are not re-run; see Studies 2 and 3. (p. 24)

## Variables

### Study 4 choice and model

`v4`

- **Role [implementation]**: dv and iv
- **Definition [to check]**: Proportion of CBR-option choices by framing and model (humans, Centaur, Llama 3.1 pre-trained, Llama 3.1-Instruct). (p. 13)
- **Observation unit [derived, to check]**: one model query or one participant (one vignette each) (p. 13, p. 24)
  - how it was derived: Models are queried per vignette; human data are one vignette per participant.

## Analysis

### Study 4 analysis

`a4`

- **Aim [to check]**: Compare the strength of Yes-No and omission biases across pre-trained Llama 3.1, Llama 3.1-Instruct and Centaur, and against humans. (p. 13)
- **Method [to check]**: F tests of the reframing effect between model pairs and against humans, and t tests of bias within a model, as reported; the model specification is not described in a Methods paragraph. (p. 13)

## Reported results

### Study 4 result: Yes-No bias

`r4_yn`

- **Result kind [implementation]**: reported
- **Test [to check]**: Llama 3.1-Instruct preferred "no", t(2247) = 11.52, p < .001, much stronger than pre-trained F(1, 1280) = 116.01 and Centaur F(1, 1320) = 109.60 (both p < .001, .08). Neither pre-trained Llama 3.1 nor Centaur was significantly more affected than humans. (p. 13)
- **Centaur [to check]**: Centaur showed very little variability between dilemmas, endorsing CBR about 50% of the time. (p. 13)

### Study 4 result: omission bias

`r4_om`

- **Result kind [implementation]**: reported
- **Test [to check]**: Llama 3.1-Instruct vs pre-trained F(1, 1275) = 141.32 (.10); vs Centaur F(1, 1320) = 100.94 (.07), both p < .001; no evidence Centaur differed from pre-trained Llama 3.1; all models differed from participants. (p. 13)

## Simulation scope

### Simulation scope: Study 4

`sim4`

- **Steps [implementation, to check]**: Not replayed.
- **Roles [implementation, to check]**: The subjects are specific models (pre-trained Llama 3.1, Centaur) that cannot be run with a command-line chat agent; there are no human participants to replace.
- **Conditions [implementation, to check]**: Three models; not replayed.
- **Outcomes [implementation, to check]**: None.
- **Limits [implementation, to check]**: Needs the pre-trained and Centaur checkpoints and their button-press prompts.

## Open issues

- **Study 4 model checkpoints are not specified here** [check, `i30_s4_checkpoint`, n4_l]: The paper points to Binz et al. for the Centaur links and does not name the pre-trained checkpoint size. Suggested: Supply checkpoint names. (p. 33)
- **Study 4 model specification only appears in the results** [check, `i31_s4_analysis`, a4]: No Methods paragraph describes the Study 4 tests. Suggested: Supply the analysis script. (p. 13)
