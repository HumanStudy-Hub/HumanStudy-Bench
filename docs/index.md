# HumanStudy-Bench Documentation

HumanStudy-Bench is a standardized testbed for replaying human-subject
experiments with AI agents. Each runnable study lives under `studies/` with
source materials, metadata, execution code, and evaluation logic.

## Contribution paths

**Build Study** is a private-beta HumanStudy-Hub workflow for researchers who
begin with a paper PDF and optional open materials. It runs the staged pipeline,
persists researcher decisions, and produces a ZIP or GitHub pull request.

**Direct pull requests** are available now and remain fully supported:

1. Fork and clone the repository.
2. Add a new folder under `studies/`.
3. Run `bash scripts/verify_study.sh <study-folder>`.
4. Push the branch and open a pull request.
5. Complete automated checks and human review.

## Guides

- [What should I submit?](what_to_submit.md)
- [How to extract data from a paper](extract_from_paper.md)
- [How to build study files](build_study_files.md)
- [How to submit a study](submit_study.md)
