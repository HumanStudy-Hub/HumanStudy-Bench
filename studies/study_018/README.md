# Strategies for Integrating Disparate Social Information

**Authors:** Lucas Molleman, Alan N. Tump, Andrea Gradassi, Stefan M. Herzog, Bertrand Jayles, Ralf H. J. M. Kurvers, and Wouter van den Bos

**Year:** 2020

This package implements the paper's focal visual-estimation experiment. A participant first estimates the number of animals in an image, then sees three prerecorded peer estimates and revises the judgment. Peer variance and skewness vary across four core conditions. The package also includes the four-peer control with no image or personal first estimate.

## Implemented Scope

- Thirty main rounds with the exact species, counts, and treatment order
- Five rounds each of LN, HN, HF, and HC, plus ten fillers
- Exact peer estimates for every valid first estimate from 1 through 150
- Five four-peer control rounds and published anchor pools
- Multimodal first-estimate prompts followed by text-only revisions
- Published social-information-use and strategy measures
- Condition-level evaluator and execution audit

The one-peer control, browser slider motor behavior, comprehension checks, post-task scales, cognitive-model fitting, and downstream simulations are excluded.

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

A full participant requires at least 65 model calls. Start with one participant before scaling.

## Fidelity Boundary

The exact peer distributions come from the public LIONESS code, not an LLM reconstruction. The historical animal sprite URLs are unavailable, so the package contains deterministic regenerated silhouettes with the exact published species and counts. The placement follows the original seeded JavaScript formula using a documented reference viewport, but the silhouettes are not the original artwork. The runtime sends the image only for the first estimate and starts the social revision in a new, image-free conversation.

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
