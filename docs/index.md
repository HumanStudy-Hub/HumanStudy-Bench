# HumanStudy-Bench Documentation

Welcome to the HumanStudy-Bench docs. This site is for **contributors** who want to add or understand studies in the benchmark.

## What is HumanStudy-Bench?

HumanStudy-Bench is a standardized testbed for *replaying human-subject experiments* with LLM-simulated participants. It combines:

- An **Execution Engine** that reconstructs full experimental protocols from published studies
- A **Benchmark** with standardized evaluation metrics and alignment at the level of scientific inference

Each study lives under `studies/study_XXX/` with source data, scripts for trial generation and evaluation, and metadata.

## Where to start

| If you want to… | Go to |
|-----------------|--------|
| **Contribute a new study** (fork → create files → PR) | [Contributing](contributing.md) |
| **Follow a step-by-step guide** to create a study (with examples) | [Study Walkthrough](study_walkthrough.md) |
| **Extract data from a paper** (with AI assistance) | [Data Extraction Guide](extraction_guide.md) |
| **Look up required files and API contracts** | [File Reference](file_reference.md) |

## Quick contribution flow

1. Fork the repo and clone it.
2. Create a branch (e.g. `study_013`) and add `studies/study_013/` with the required files.
3. Run `bash scripts/verify_study.sh study_013`.
4. Open a Pull Request. After review, your study is merged and appears in the index and on the website.

For full details and the exact file layout, see the [Contributing](contributing.md) guide.
