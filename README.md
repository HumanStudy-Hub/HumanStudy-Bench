<div align="center">
  <img src="docs/img/new-HS-bench_logo.png" alt="HumanStudy-Bench Logo" width="300">

  <h1>HumanStudy-Bench: Community Edition</h1>
  <p><em>Open community-driven expansion of the HumanStudy-Bench benchmark</em></p>

  <h3><a href="https://arxiv.org/abs/2602.00685">📖 Read the Paper</a> | <a href="https://github.com/AISmithLab/HumanStudy-Bench">🔬 Original Research Repo</a></h3>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/HumanStudy-Hub/HumanStudy-Bench/pulls)

</div>

---

> LLMs are increasingly used to simulate human participants in social science research, but existing evaluations conflate base model capabilities with agent design choices, making it unclear whether results reflect the model or the configuration.

## 👋 Overview

HumanStudy-Bench treats participant simulation as an *agent design problem* and provides a **standardized testbed** — combining an **Execution Engine** that reconstructs full experimental protocols from published studies and a **Benchmark** with standardized evaluation metrics — for *replaying human-subject experiments end-to-end* with alignment evaluation at the level of scientific inference.

## How to Contribute a Study

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

## Existing Studies

The 12 foundational studies (cognition, strategic interaction, social psychology) serve as reference examples. Browse them on the [website](https://www.hs-bench.clawder.ai/contribute#studies) or locally under `studies/`.

## Citation & Hugging Face

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

**Hugging Face:** Benchmark and resources are available on the [Hugging Face Hub](https://huggingface.co/) — `fuyyckwhy/HS-Bench-results`.

## Reproduction (Paper Results)

To reproduce the exact benchmark and results reported in the paper, see the [original research repo](https://github.com/AISmithLab/HumanStudy-Bench) at tag `v1.0.0`.

## License

MIT License. See [LICENSE](LICENSE) for details.
