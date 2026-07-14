<div align="center">
  <img src="docs/img/new-HS-bench_logo.png" alt="HumanStudy-Bench Logo" width="300">

  <h1>HumanStudy-Bench: Community Edition</h1>
  <p><em>Open community-driven expansion of the HumanStudy-Bench benchmark</em></p>

  <a href="https://arxiv.org/abs/2602.00685"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="Read the Paper" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
  <a href="https://github.com/HumanStudy-Hub/HumanStudy-Bench/pulls"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?logo=opensourceinitiative&logoColor=white" alt="Contributions Welcome" /></a>
  <a href="https://www.hs-bench.clawder.ai"><img src="https://img.shields.io/badge/docs-website-blue?logo=readthedocs&logoColor=white" alt="Docs" /></a>

</div>

---

> LLMs are increasingly used to simulate human participants in social science research, but existing evaluations conflate base model capabilities with agent design choices, making it unclear whether results reflect the model or the configuration.

## <img src="https://api.iconify.design/lucide/book-open.svg?color=%230891b2" width="20" height="20" /> Overview

HumanStudy-Bench treats participant simulation as an *agent design problem* and provides a **standardized testbed** — combining an **Execution Engine** that reconstructs full experimental protocols from published studies and a **Benchmark** with standardized evaluation metrics — for *replaying human-subject experiments end-to-end* with alignment evaluation at the level of scientific inference.

## Automated Study Generation Pipeline

This repo also includes a staged generation pipeline for drafting a Hub study
folder from a paper PDF and optional OSF/supplementary sources.

The pipeline is intentionally human-in-the-loop:

- Stage 1 compiles a source-grounded inventory of empirical units and
  source-explicit comparison groups from any social-science topic.
- Stage 2 extracts study-level findings, effects, samples, and statistics.
- Stage 3 assembles and audits participant-facing materials from
  OSF/QSF/SAV/PDF sources.
- Stage 4 writes a Hub study folder under `studies/<study_id>/`.
- Stage 5 can run simulation after `scripts/config.py` exists and the materials
  pass the package-readiness gate.

Install the LLM and layout-aware PDF dependencies for generation:

```bash
python -m pip install -e ".[llm,pdf]"
```

Docling is the primary PDF parser. It preserves page/block provenance and table
structure; image-dominant PDFs additionally use Docling's RapidOCR path. The
first parse may download Docling model weights. If Docling is unavailable, the
pipeline records a degraded pypdf fallback and will not silently treat flat text
as layout-complete evidence.

Copy the settings template and configure API credentials locally:

```bash
cp config/settings.example.yaml config/settings.yaml
```

Do not commit `config/settings.yaml`.

### Stage 1: evidence-grounded study inventory

Stage 1 is domain-independent. It inventories every reported empirical unit in
a social-science paper, including experiments, surveys, pilots, validation
samples, field studies, and observational studies. It then determines whether
each human-participant task can be represented in HumanStudy-Bench. Psychology,
behavioral economics, organizational behavior, political science, sociology,
communication, marketing, education, and HCI use the same task definition; no
ethics-specific topic or outcome filter remains.

The complete PDF is never placed in one LLM prompt. Stage 1 instead uses this
evidence compiler:

```text
Docling layout-aware parse (once)
  -> targeted second OCR only for ambiguous numeric/statistical tokens
  -> bounded overlapping windows covering every parsed block
  -> parallel high-recall empirical-unit and relationship discovery
  -> deterministic reconciliation by source labels and assigned variants
  -> bounded global candidate-ledger adjudication of simulation task families
  -> shared sample/procedure evidence attached without merging different tasks
  -> one bounded, evidence-anchored extraction per task family
  -> provenance and empirical-support gate
  -> task families + nested material variants + source-explicit comparison groups
  -> bounded-window coverage audit; global ledger remains boundary authority
  -> complete cited-evidence verifier for each study's fields and eligibility
  -> deterministic numeric/OCR conflict hints that every study audit must adjudicate
  -> typed field and eligibility repair with boundary-audit reuse
  -> full recompilation only for grounded missing-unit or boundary changes
  -> accept a refinement only when deterministic audit quality improves
```

The three output levels are intentional. `experiments[]` contains coherent
simulation task families, `experiments[].material_variants[]` preserves stories,
forms, orders, stimuli, and conditions that Stage 3 may need to reconstruct
separately, and `comparison_groups[]` records task families that the paper
explicitly combines in one contrast or finding. Different tasks remain separate
even when they share one questionnaire and participant pool; repeated or
parameterized items remain together when they use one task template and response
format. Questionnaire-wide sample/procedure evidence is retained under
`shared_contexts` instead of becoming a fake study or comparison group. Table
rows, outcome groups, and individual effects are never promoted to studies. A
material variant must be a genuine alternative version assigned across
participants or occasions; response options shown together inside one question
remain part of that question rather than becoming separate variants.

Every accepted unit records PDF block/page evidence, `unit_provenance`, whether
it is a distinct empirical unit, and direct support for its sample, participant
task, and quantitative target. Cited prior work and isolated result fragments
are rejected from the current-paper inventory. Missing exact questionnaire
wording or supplementary files is recorded as a Stage 3 material gap and does
not by itself exclude an otherwise simulatable unit. A real task with no exact
numeric response result remains in the inventory but is labeled `NO`; stimulus
amounts, response-scale endpoints, and qualitative majority/significance claims
do not count as quantitative target results. `replicable` remains the legacy
JSON field name; it means simulation eligibility, not a guarantee that the
original result will reproduce.

`simulation_barriers` separately records source-grounded execution requirements
such as observed physical action or live interaction. A barrier only forces a
`NO` label when it affects the paper's original quantitative target; it cannot
be bypassed by silently replacing that target with a hypothetical proxy.

Run Stage 1 with the default four bounded-call workers:

```bash
python generation_pipeline/run.py \
  --settings config/settings.yaml \
  --stage 1 \
  --pdf /absolute/path/to/paper.pdf \
  --provider openai \
  --model gpt-5-mini \
  --stage1-workers 4 \
  --stage1-timeout 300 \
  --stage1-verifier-timeout 300
```

Inspect `generation_pipeline/outputs/<paper_id>/stage1.json` and `stage1.md`.
The JSON is ready to pass to Stage 2 only when
`stage1_evidence.extraction_complete` and
`stage1_evidence.all_comparison_relations_resolved` are true,
`stage1_study_contract.blocking_issue_count` is zero, and
`stage1_verification.study_audit.all_cited_evidence_included` is true, and
`stage1_verification.overall` is `pass`. A paper window is never allowed to
invalidate a study field merely because that field's evidence appears in a
different section; field checks run against the study's complete cited-evidence
bundle instead. Field-only refinement never reruns discovery or changes the
accepted boundary audit, and a failed refinement candidate cannot replace the
last valid result.

### PDF-only Stage 3

OSF and supplementary material handling remains the preferred path when those
sources exist. For PDF-only papers, Stage 3 uses this fail-closed flow:

```text
Docling/RapidOCR parse and page images
  -> page-grounded evidence index and study-specific retrieval
  -> participant blocks vs non-runtime source structures
  -> semantic table linking and visual cell recovery/adjudication
  -> source-axis runtime plan with an independent plan audit
  -> bounded parallel unit compilation and per-unit source audit
  -> whole-instrument semantic verification and constrained repair
  -> coverage ledger and ready/not-ready decision
```

Tables used to construct conditions are retained as non-runtime evidence. Only
source-supported questions, options, scales, matrices, and instructions become
participant-facing material. Missing questionnaire wording is reported as a
blocking source absence; construct names, results summaries, and placeholders
cannot be substituted merely to make a package runnable.

Example using `gpt-5-mini` after Stages 1 and 2 have produced `stage2.json`:

```bash
python generation_pipeline/run.py \
  --settings config/settings.yaml \
  --stage 3 \
  --json generation_pipeline/outputs/<paper_id>/stage2.json \
  --pdf /absolute/path/to/paper.pdf \
  --provider openai \
  --model gpt-5-mini \
  --stage3-select-votes 1 \
  --stage3-select-timeout 120 \
  --stage3-pdf-timeout 300 \
  --no-backup
```

Stage 3 writes `stage3.json`, `stage3.md`, and reusable parser/compiler evidence
under `generation_pipeline/outputs/<paper_id>/pdf_artifacts/`, including
`blocks.json`, `parser_report.json`, page images, table-link/recovery caches, and
runtime compiler plans/units. A selected material is not necessarily ready:
inspect `study_materials.<id>.readiness`, `coverage_ledger`, and
`source_trace.semantic_verifier` before Stage 4.

### Stage 4 Hub package

Example Stage 4 Hub package generation from an existing `stage3.json`:

```bash
python generation_pipeline/run.py \
  --settings config/settings.yaml \
  --stage 4 \
  --json generation_pipeline/outputs/<paper_id>/stage3.json \
  --pdf /absolute/path/to/paper.pdf \
  --study-id study_my_generated_study \
  --provider openai \
  --model gpt-5-mini \
  --hub-layout \
  --hub-studies-dir studies
```

In Hub layout, Stage 4 writes:

```text
studies/<study_id>/
  index.json
  README.md
  source/
    metadata.json
    specification.json
    ground_truth.json
    source_extraction.json
    audit.json
    materials/*.json
  scripts/
    study_utils.py
    config.py        # generated by the Stage 4 LLM config generator when configured
    evaluator.py     # placeholder; fill after inspecting Stage 5 responses
```

Stage 4 compiles `source/materials/*.json` deterministically from Stage 3
`study_materials`. It also derives metadata findings, specification coverage,
and ground-truth references deterministically from the complete structured Stage
3 payload, so a long extraction cannot be truncated to one sub-study. The LLM may
enrich metadata domain/keywords and generates adapter code, but it does not
regenerate participant-facing items or decide finding/material coverage.

`scripts/config.py` is Stage 5 input, not Stage 5 output. It is generated by
the Stage 4 config generator when a Stage 4 LLM is configured; otherwise Stage 4
marks config generation as skipped. `scripts/evaluator.py` is deliberately a
placeholder because reliable evaluator code usually needs response samples from
a Stage 5 smoke run plus human review.

Stage 5 enforces `source/audit.json` for pipeline-generated packages and refuses
to run while `ready_for_simulation` is false. For an explicit debugging-only
smoke run, pass `--allow-unready`; the override is recorded in the Stage 5 JSON.
Stage 4 also imports and validates the generated config adapter. Missing or
invalid adapter code makes the package unready even when its JSON files exist.

## <img src="https://api.iconify.design/lucide/git-pull-request.svg?color=%230891b2" width="20" height="20" /> How to Contribute a Study

### 1. Fork and clone

```bash
git clone https://github.com/<your-github-id>/HumanStudy-Bench.git
cd HumanStudy-Bench
git checkout -b contrib-<yourgithubid>-013
```

### 2. Create your study folder

Add a new directory under `studies/` with the required folders:

```
studies/<yourgithubid>_013/
  ├── index.json
  ├── source/
  ├── scripts/
  └── README.md
```

See the docs below for what goes inside each folder and the exact schemas:

| # | Guide | Description |
|---|-------|-------------|
| 1 | [What Should I Submit?](https://www.hs-bench.clawder.ai/docs/what_to_submit) | Overview of contribution, required folders and files |
| 2 | [How to Extract Data from a Paper](https://www.hs-bench.clawder.ai/docs/extract_from_paper) | Paper hierarchy, AI extraction prompt, walkthrough example |
| 3 | [How to Build Your Study Files](https://www.hs-bench.clawder.ai/docs/build_study_files) | Schemas, code examples, and contracts for each file |
| 4 | [How to Submit Your Study](https://www.hs-bench.clawder.ai/docs/submit_study) | Fork, verify, push, and open a PR |

### 3. Verify locally

```bash
bash scripts/verify_study.sh <yourgithubid>_013
```

### 4. Commit and push

```bash
git add studies/<yourgithubid>_013/
git commit -m "Add study: <Your Study Title>"
git push origin contrib-<yourgithubid>-013
```

### 5. Open a Pull Request

Open a PR on GitHub targeting the `main` branch. Maintainers assign final `study_XXX` numbering by merge order. CI runs validation automatically; confirmation is by human review.

You can also submit a study via **web upload** at [hs-bench.clawder.ai/contribute](https://www.hs-bench.clawder.ai/contribute).

## <img src="https://api.iconify.design/lucide/flask-conical.svg?color=%230891b2" width="20" height="20" /> Existing Studies

The 12 foundational studies (cognition, strategic interaction, social psychology) serve as reference examples. Browse them on the [website](https://www.hs-bench.clawder.ai/contribute#studies) or locally under `studies/`.

## <img src="https://api.iconify.design/lucide/quote.svg?color=%230891b2" width="20" height="20" /> Citation

If you use HumanStudy-Bench, please cite:

```bibtex
@misc{liu2026humanstudybenchaiagentdesign,
      title={HumanStudy-Bench: Towards AI Agent Design for Participant Simulation},
      author={Xuan Liu and Haoyang Shang and Zizhang Liu and Xinyan Liu and Yunze Xiao and Yiwen Tu and Haojian Jin},
      year={2026},
      eprint={2602.00685},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.00685},
}
```

## <img src="https://api.iconify.design/lucide/scale.svg?color=%230891b2" width="20" height="20" /> License

MIT License. See [LICENSE](LICENSE) for details.
