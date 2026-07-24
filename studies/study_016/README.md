# Information Cascades in the Laboratory

**Authors:** Lisa R. Anderson and Charles A. Holt

**Year:** 1997

This package implements the paper as a stateful six-agent experiment. It is not
a questionnaire: every paid period has a hidden urn, private draws, a newly
randomized decision order, and a public prediction history that changes before
each decision maker acts.

## Implemented Scope

The runtime implements 66 decision-maker slots across 11 evidence-complete
sessions:

- Published sessions 1-3: symmetric baseline, 18 decision makers
- Published sessions 4-5: symmetric urns with two public draws after prediction
  4, 12 decision makers
- Published sessions 7-12: asymmetric urns, 36 decision makers
- Fifteen paid periods per session
- Public unpaid practice until urn A and urn B have each appeared
- Private draws with replacement
- A newly randomized decision order in every paid period
- Sequentially announced predictions
- End-of-period urn feedback and $2 for each correct prediction

The article reports one additional public-draw session, published session 6,
but does not state when its public draws appeared or which positions observed
them. That session's six decision makers are intentionally excluded.

## Information Timing

During paid periods, every decision prompt contains only:

- the treatment rules and urn compositions;
- that decision maker's current private draw;
- predictions announced earlier in the current period;
- the two additional public draws, only for positions 5 and 6 in sessions 4-5;
- feedback from completed periods; and
- cumulative simulated earnings.

The selected urn and other decision makers' private draws are not exposed
before a decision. The selected urn is revealed only after all six predictions.

Practice is generated as part of each session. The die result, selected urn,
and all six draws are public; no agent response is requested and no earnings
are awarded. Practice continues until both urns have been demonstrated.

## Run

One-session mock smoke test:

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

Full 11-session environment test:

```bash
.venv/bin/python generation_pipeline/run.py \
  --settings config/settings.example.yaml \
  --stage 5 \
  --experiment studies/study_016 \
  --sim-models mock \
  --n-agents 66 \
  --mock-agent \
  --seed 42
```

For a real model, remove `--mock-agent` and select a configured simulation
model. Participant counts must be multiples of six and at most 66. A partial
run executes a prefix of the published evidence-complete schedule: 6-18 agents
exercise the symmetric baseline, 24-30 add the documented public-draw
treatment, and 36-66 progressively add asymmetric sessions. A full run makes
990 paid decision calls; practice itself makes no model calls.

## Evaluation Boundary

Pass/fail is based on environment integrity: complete sessions, treatment and
urn configuration, practice stopping rule, prediction history, public-draw
timing, feedback, and payoffs. Agent behavior is reported as a diagnostic and
is not required to reproduce the original human outcome distribution.

The paper's participant-level data appendix was available from the authors on
request and is not in the public PDF. This prevents exact replay of human
choices, but it does not prevent generative reconstruction of the randomized
urn-and-draw environment.

## Files

- `source/anderson_holt_1997_information_cascades.pdf`
- `source/materials/experiment_instructions.json`
- `source/materials/symmetric_baseline.json`
- `source/materials/symmetric_public_draw_after_position_4.json`
- `source/materials/asymmetric_baseline.json`
- `source/metadata.json`
- `source/specification.json`
- `source/ground_truth.json`
- `scripts/config.py`
- `scripts/evaluator.py`
- `scripts/study_utils.py`
