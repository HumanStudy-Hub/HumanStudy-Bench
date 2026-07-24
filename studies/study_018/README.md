# Strategies for Integrating Disparate Social Information

**Authors:** Lucas Molleman, Alan N. Tump, Andrea Gradassi, Stefan M. Herzog, Bertrand Jayles, Ralf H. J. M. Kurvers, and Wouter van den Bos

**Year:** 2020

This package implements the paper's complete three-block behavioral task. The one-peer and main blocks ask participants to estimate animals in an image, observe one or three prerecorded estimates, and revise their judgment. The four-peer block asks participants to aggregate four estimates without seeing the image or forming a personal first estimate.

## Implemented Scope

- Five one-peer rounds with the exact species, counts, source pools, and peer-selection rule
- Thirty main rounds with the exact species, counts, and treatment order
- Five rounds each of LN, HN, HF, and HC, plus ten fillers
- Exact peer estimates for every valid first estimate from 1 through 150
- Five four-peer control rounds and published anchor pools
- All six original block-order permutations, assigned cyclically by participant number
- Original block-specific instructions and four-item comprehension checks
- Multimodal first-estimate prompts followed by text-only revisions
- Published social-information-use and strategy measures
- Condition-level evaluator and execution audit

Browser slider motor behavior, enforced client-side response timers, post-task scales, cognitive-model fitting, and downstream simulations are excluded.

## Run

Mock smoke test:

    .venv/bin/python generation_pipeline/run.py \
      --settings config/settings.example.yaml \
      --stage 5 \
      --experiment studies/study_018 \
      --sim-models mock \
      --n-agents 4 \
      --mock-agent \
      --seed 42

Real vision-capable model:

    .venv/bin/python generation_pipeline/run.py \
      --settings config/settings.yaml \
      --stage 5 \
      --experiment studies/study_018 \
      --sim-models gpt-5 \
      --n-agents 1 \
      --seed 42

A full participant requires at least 78 model calls: three comprehension checks, 10 one-peer calls, 60 main-task calls, and five four-peer calls. Start with one participant before scaling.

## Fidelity Boundary

The schedules, block orders, comprehension checks, and peer distributions come from the public LIONESS code, not an LLM reconstruction. The historical animal sprite URLs are unavailable, so the package contains deterministic regenerated silhouettes with the exact published species and counts. The placement follows the original seeded JavaScript formula using a documented reference viewport, but the silhouettes are not the original artwork. The runtime sends the image only for the first estimate and starts the social revision in a new, image-free conversation; the original six-second browser exposure and slider motor behavior are documented adaptations rather than simulated timing claims.

## Files

- `source/molleman_et_al_2020_strategies_disparate_social_information.pdf`
- `source/materials/disparate_social_information.json`
- `source/materials/peer_lookup.json`
- `source/materials/stimulus_manifest.json`
- `source/materials/build_peer_lookup.py`
- `source/stimuli/`
- `source/stimuli/generate_stimuli.py`
- `source/metadata.json`
- `source/specification.json`
- `source/ground_truth.json`
- `scripts/config.py`
- `scripts/evaluator.py`
- `scripts/study_utils.py`
