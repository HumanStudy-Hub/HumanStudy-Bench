# Social Influences in Sequential Decision Making

**Authors:** Markus Schöbel, Jörg Rieskamp, and Rafael Huber

**Year:** 2016

This package implements both scenario experiments from the paper. Each
participant completes one full sub-study in a randomized order. The tasks are
text-native and do not require synchronous groups: the previous decisions are
fixed scenario evidence taken from the published variable-coding documents.

## Implemented Experiments

### Study 1: Abstract Urn Scenarios

- 40 analyzed participants
- 24 scenarios per participant
- 12 distinct information structures and 12 mirrored presentations
- Up to three public predecessor decisions
- One private white or black ball
- Urn A/Urn B prediction
- Confidence judgment from 50% through 100%

Urn A contains two white and one black ball; Urn B contains one white and two
black balls. The urn prior is equal. Participants see predecessors' public urn
predictions but never their private draws. Every predecessor saw all earlier
public predictions before deciding, so later predictions may already reflect
earlier ones and are not independent ball draws.

### Study 2: Medical Authority Scenarios

- 40 participants
- 40 scenarios per participant
- One to three previous diagnoses
- Previous decision makers are assistant physicians or the medical director
- Medical-director diagnosis supports private evidence, opposes it, or is absent
- Appendicitis/sigmoid diverticulitis diagnosis
- Confidence judgment from 50% through 100%

The private symptom and a physician diagnosing without prior diagnoses are
described as 67% accurate. Physicians act sequentially and see diagnoses
already entered in the patient record, so later diagnoses are not independent
medical tests. Hierarchical role therefore changes authority while holding the
stated baseline diagnostic accuracy constant.

## Source Boundary

The article, PLOS JATS tables, supporting text, Figshare raw workbooks, and
variable-coding PDFs are committed under `source/`. `materials/scenarios.json`
is reproducibly compiled from the two public raw workbooks and explicit
scenario codes by `materials/build_scenarios.py`.

The public archive does not contain the original questionnaire or original
German wording. Instructions are therefore a semantic reconstruction of the
published Procedure sections. Scenario evidence, order pool, response scales,
and outcome data are source-grounded rather than reconstructed by an LLM.

## Run

Small mock test covering both studies:

```bash
.venv/bin/python generation_pipeline/run.py \
  --settings config/settings.example.yaml \
  --stage 5 \
  --experiment extended_study/study_019 \
  --sim-models mock \
  --n-agents 2 \
  --mock-agent \
  --seed 42
```

Full original analyzed sample:

```bash
.venv/bin/python generation_pipeline/run.py \
  --settings config/settings.example.yaml \
  --stage 5 \
  --experiment extended_study/study_019 \
  --sim-models mock \
  --n-agents 80 \
  --mock-agent \
  --seed 42
```

Participant counts must be even and at least two. Participants alternate
between Study 1 and Study 2, giving equal sub-study sample sizes. The full
original sample makes 2,560 model calls before any response repairs.

## Evaluation Boundary

`environment_passed` checks scenario completeness, source-material integrity,
randomized order, response domains, prompt visibility, and absence of answer
feedback. Top-level `passed` additionally requires at least 40 participants in
each sub-study and source-grounded alignment for broad choices, Study 1
probability-judgment groups, and Study 2 authority-condition choice and
probability-judgment cells.

Stage 5 writes both `behavioral_diagnostics` and the package evaluator result
to `full_benchmark.json`. A two-agent smoke run is not behaviorally evaluable
and therefore cannot report replication success. The source-derived mock policy
is only an execution regression fixture; its behavioral agreement is not
independent replication evidence.

## Files

- `source/schoebel_rieskamp_huber_2016.pdf`
- `source/schoebel_rieskamp_huber_2016.xml`
- `source/schoebel_rieskamp_huber_2016_s1.docx`
- `source/raw_data/`
- `source/materials/build_scenarios.py`
- `source/materials/scenarios.json`
- `source/metadata.json`
- `source/specification.json`
- `source/ground_truth.json`
- `scripts/config.py`
- `scripts/evaluator.py`
- `scripts/study_utils.py`
