# Preferences for Advisor Agreement and Accuracy

**Authors:** Matt Jaquiery and Nick Yeung

**Year:** 2024

This package implements the continuous Dates Task from Experiments 3B and 3C. It is a stateful Judge-Advisor System: each participant repeatedly estimates historical dates, sees advice from two anonymous advisors with different hidden policies, revises their estimates, and later chooses an advisor.

## Implemented Scope

- Complete 74-event public question bank
- Ten unaided practice trials with correctness feedback
- Two practice-advisor trials with correctness feedback
- Fifteen familiarisation trials with each advisor
- Ten advisor-choice trials
- Attention checks at the original global trial indices 16 and 36
- Accurate advice centered on the true year
- Agreeing advice centered on the participant's initial estimate
- 13.5% reflected-control advice trials
- Feedback and no-feedback conditions
- Counterbalanced advisor order
- Persistent conversation state per participant

The two attention checks replace ordinary historical-event trials, matching the
original schedule. A completed participant therefore sees 52 trial slots and
answers 50 historical-event questions: 12 practice questions, 38 formal
historical-event questions, and two direct-instruction attention checks. A
failed attention check terminates the participant and preserves the failed
record for auditing.

Advisor avatars, browser drag interactions, and the browser debrief are
excluded. They are presentation mechanisms rather than the target
advisor-choice manipulation.

## Run

Mock smoke test with four participants:

    .venv/bin/python generation_pipeline/run.py \
      --settings config/settings.example.yaml \
      --stage 5 \
      --experiment studies/study_017 \
      --sim-models mock \
      --n-agents 4 \
      --mock-agent \
      --seed 42

For a real model, remove --mock-agent and select a configured model. A completed
participant requires 102 model calls before any response repairs: one response
for each unaided practice trial and attention check, and an initial plus final
response for each advisor trial. Validate with a small participant count first.

## Fidelity Boundary

Advisor identities shown to participants are anonymous and stable. The runtime
never labels a formal advisor as accurate or agreeing in an agent-visible
prompt. Correct years are shown during the 12 practice trials and, in the formal
task, only after final responses in the Experiment 3C feedback condition.

The runtime uses text fields instead of draggable markers. A center year plus marker width preserves the experimental response and scoring semantics without pretending to reproduce motor behavior.

## Files

- source/jaquiery_yeung_2024_preferences_advisor_agreement_accuracy.pdf
- source/materials/dates_task_3b_3c.json
- source/materials/question_bank.json
- source/metadata.json
- source/specification.json
- source/ground_truth.json
- scripts/config.py
- scripts/evaluator.py
- scripts/study_utils.py
