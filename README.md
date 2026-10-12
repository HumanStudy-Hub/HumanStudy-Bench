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
| **Build Study** | Researchers starting from a paper PDF and optional open materials | **Private beta.** Contact the HumanStudy-Hub team for access. |
| **Direct pull request** | Contributors who want to build a study from the repository templates | **Available now.** Build the folder locally, run validation, and open a PR. |

The intended Build Study workflow is:

```text
paper PDF + optional OSF/open materials
  -> evidence and study extraction
  -> researcher review at uncertain decisions
  -> runnable HumanStudy-Bench folder
  -> download ZIP and/or save to GitHub
```

Build Study is currently operated through the private HumanStudy-Hub web
application. Contributors without access can use the direct pull request path
below.

### Direct pull request

1. Fork this repository and clone your fork.

```bash
git clone https://github.com/<your-github-id>/HumanStudy-Bench.git
cd HumanStudy-Bench
git checkout -b contrib-<your-github-id>-<study-name>
```

2. Copy an existing study such as [`study_001`](studies/study_001/) and adapt
   its legacy study package structure.

```text
studies/<your-github-id>_<study-name>/
  index.json
  README.md
  source/
  scripts/
```

3. Validate it locally.

```bash
bash scripts/verify_study.sh <your-github-id>_<study-name>
```

4. Make sure `index.json` lists your GitHub account under `contributors`, then
   commit the study, push your branch, and open a pull request targeting
   `main`. Maintainers review the study and rename its folder to the final
   numbered ID, such as `study_016`, before merging.

Contributor guides:

- [What should I submit?](docs/what_to_submit.md)
- [How to extract data from a paper](docs/extract_from_paper.md)
- [How to build study files](docs/build_study_files.md)
- [How to submit a study](docs/submit_study.md)

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
