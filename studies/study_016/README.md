# Information Cascades in the Laboratory

**Authors:** Lisa R. Anderson and Charles A. Holt

**Year:** 1997

This package implements the paper's symmetric baseline as a stateful six-agent experiment. It is not a questionnaire: each period has a hidden urn, private draws, a randomized decision order, and a public decision history that changes before every participant acts.

## Implemented Scope

- Three six-participant baseline sessions by default
- Fifteen paid periods per session
- Equal prior probability for urn A and urn B
- Symmetric 2:1 and 1:2 light/dark urn compositions
- Private draws with replacement
- Sequential public predictions in random order
- $2 simulated earnings for each correct prediction
- Cascade-opportunity, Bayesian-agreement, accuracy, and earnings summaries

The public-draw variants and asymmetric urn treatment are deliberately excluded because the public article does not provide enough session-level material to reconstruct every detail without guessing.

## Run

Mock validation with one complete session:

```bash
.venv/bin/python generation_pipeline/run.py \
  --settings config/settings.example.yaml \
  --stage 5 \
  --experiment studies/study_016 \
  --sim-models mock \
  --n-agents 6 \
  --mock-agent \
  --seed 42
```

For a real model, remove `--mock-agent`, select a configured model, and keep the participant count at a multiple of six. One session requires 90 sequential decisions.

## Fidelity Boundary

The adapter reproduces the information available to each decision maker. Prompts never include the true urn or other participants' private draws. The true urn is revealed only after all six decisions in a period, as in the original procedure.

The paper's reported `41/56` cascade rate pools all six symmetric sessions, including public-draw variants. The runtime's default three-session baseline is therefore directly comparable in mechanism and direction, but not an exact raw-data rerun.

## Files

- `source/anderson_holt_1997_information_cascades.pdf`
- `source/materials/symmetric_baseline.json`
- `source/metadata.json`
- `source/specification.json`
- `source/ground_truth.json`
- `scripts/config.py`
- `scripts/evaluator.py`
- `scripts/study_utils.py`
