<div align="center">
  <img src="docs/img/new-HS-bench_logo.png" alt="HumanStudy-Bench Logo" width="300">

  <h1>HumanStudy-Bench</h1>

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

The initial set includes 12 studies spanning cognition, strategic interaction,
and social psychology. Further community-contributed studies are added under
[`studies/`](studies/) as the benchmark grows.

## Add a study

There are two contribution paths.

| Path | Best for | Current status |
|---|---|---|
| **HumanStudy-Hub platform** | An automated study builder on the HumanStudy-Hub platform designed for social scientists, turning a paper and optional research materials into a runnable study through guided human review and calibration | **Currently being upgraded.** Contact the HumanStudy-Hub team for early access. |
| **Direct pull request** | An alternative for contributors who want to build a study from the repository templates | **Available now.** Build the folder locally, run validation, and open a PR. |

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
