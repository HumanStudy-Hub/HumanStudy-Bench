<div align="center">
  <img src="docs/img/new-HS-bench_logo.png" alt="HumanStudy-Bench Logo" width="300">

  <h1>HumanStudy-Bench</h1>
  <p><em>A benchmark for evaluating AI agents through runnable human studies</em></p>

  <a href="https://arxiv.org/abs/2602.00685"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="Read the Paper"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://github.com/HumanStudy-Hub/HumanStudy-Bench/pulls"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions welcome"></a>
</div>

---

HumanStudy-Bench is a standardized testbed for replaying human-subject
experiments with AI agents. It combines:

- an execution engine that reconstructs published experimental protocols;
- a growing collection of runnable social-science studies; and
- evaluation at the level of behavior, effects, and scientific findings.

The 12 foundational studies cover cognition, strategic interaction, and social
psychology. Community studies live alongside them under [`studies/`](studies/).

## Add a study

There are two contribution paths.

| Path | Best for | Current status |
|---|---|---|
| **Build Study** | Researchers starting from a paper PDF and optional open materials | **Private beta.** HumanStudy-Hub runs this repository's pipeline, persists review decisions, and produces a ZIP or GitHub pull request. |
| **Direct pull request** | Contributors who want full control over study files | **Available now.** Build the folder locally, run validation, and open a PR. |

The intended Build Study workflow is:

```text
paper PDF + optional OSF/open materials
  -> evidence and study extraction
  -> researcher review at uncertain decisions
  -> runnable HumanStudy-Bench folder
  -> download ZIP and/or save to GitHub
```

Build Study is currently operated through the private HumanStudy-Hub web
application. The direct contribution documentation remains supported for
researchers who prefer to author or inspect every file themselves. The manual
entry point is the [direct pull request guide](docs/submit_study.md).

### Direct pull request

1. Fork this repository and clone your fork.

```bash
git clone https://github.com/<your-github-id>/HumanStudy-Bench.git
cd HumanStudy-Bench
git checkout -b contrib-<your-github-id>-<study-name>
```

2. Add a study folder.

```text
studies/<contributor-id>_<study-id>/
  index.json
  README.md
  source/
  scripts/
```

3. Validate it locally.

```bash
bash scripts/verify_study.sh <contributor-id>_<study-id>
```

4. Commit the study, push your branch, and open a pull request targeting
   `main`. CI validates the package and contributor attribution; maintainers
   complete the human review and assign the final study ID.

Contributor guides:

- [What should I submit?](docs/what_to_submit.md)
- [How to extract data from a paper](docs/extract_from_paper.md)
- [How to build study files](docs/build_study_files.md)
- [How to submit a study](docs/submit_study.md)

## Agent study builder

Build Study uses Claude Code as an agent runtime in GitHub Actions. The agent
reads the submitted PDF, searches for optional OSF and supplementary material,
extracts the study, builds a runnable package, validates it, and then opens one
final researcher review in HumanStudy-Hub.

Claude Code is routed through OpenRouter. The default model is
`anthropic/claude-sonnet-5`; it can be changed without editing the workflow.

Repository settings required by
[`run-humanstudy-pipeline.yml`](.github/workflows/run-humanstudy-pipeline.yml):

| Type | Name | Value |
|---|---|---|
| Actions secret | `OPENROUTER_API_KEY` | OpenRouter API key |
| Actions secret | `HUMANSTUDY_PIPELINE_TOKEN` | Fine-grained GitHub token with read/write access to the private jobs repository |
| Actions variable | `OPENROUTER_MODEL` | Optional; defaults to `anthropic/claude-sonnet-5` |

The agent instructions and package contract live in
[`agent_pipeline/CLAUDE.md`](agent_pipeline/CLAUDE.md). The validator requires a
single paper folder containing the study overview, source evidence, materials,
runnable task adapter, evaluation logic, provenance, and a researcher-facing
missing-information checklist.

The older staged Python implementation remains in `generation_pipeline/` for
reference and migration tests, but the web builder no longer dispatches it.

## Playground

The playground replays a single study with a model of your choice and scores it
against the paper, so you can see where an agent behaves like the original
participants and where it does not.

```bash
OPENROUTER_API_KEY=... python playground/run_playground.py --run <run-dir>
```

It drives each study's own trial builder and evaluator, so a playground run is
scored exactly like a benchmark run; the model, the participant prompt, and the
number of participants are what you change. Results are charted and interpreted
by Claude Code, with deterministic charts as the fallback. HumanStudy-Hub
dispatches the same code through
[`run-playground.yml`](.github/workflows/run-playground.yml). See
[`playground/README.md`](playground/README.md).

## Repository structure

```text
studies/               Runnable benchmark studies
src/                   Execution and evaluation code
playground/            On-demand single-study runs and result comparison
agent_pipeline/        Claude Code package contract and validation
generation_pipeline/   Legacy staged extraction implementation
scripts/               Study validation and index utilities
docs/                  Manual contribution guides
tests/                 Pipeline and benchmark tests
```

## Citation

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

## License

MIT License. See [LICENSE](LICENSE).
