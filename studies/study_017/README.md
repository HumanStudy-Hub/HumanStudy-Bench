# Preferences for Advisor Agreement and Accuracy

**Authors:** Matt Jaquiery and Nick Yeung

**Year:** 2024

This package implements the continuous Dates Task from Experiments 3B and 3C. It is a stateful Judge-Advisor System: each participant repeatedly estimates historical dates, sees advice from two anonymous advisors with different hidden policies, revises their estimates, and later chooses an advisor.

## Implemented Scope

- Complete 74-event public question bank
- Fifteen familiarisation trials with each advisor
- Ten advisor-choice trials
- Accurate advice centered on the true year
- Agreeing advice centered on the participant's initial estimate
- 13.5% reflected-control advice trials
- Feedback and no-feedback conditions
- Counterbalanced advisor order
- Persistent conversation state per participant

Practice, attention-check termination, avatars, and browser drag interactions are excluded. They are interface or screening mechanisms rather than the target advisor-choice manipulation.

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

For a real model, remove --mock-agent and select a configured model. A full participant requires 90 model calls before any response repairs, so validate with a small participant count first.

## Fidelity Boundary

Advisor identities shown to participants are anonymous and stable. The runtime never labels an advisor as accurate or agreeing in an agent-visible prompt. Correct years are hidden until after final responses and are revealed during the task only in the Experiment 3C feedback condition.

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
