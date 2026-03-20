# What Should I Submit?

HumanStudy-Bench is a standardized testbed for replaying human-subject experiments with LLM agents. The 12 foundational studies are just the beginning — and we need your help to grow the community. Every new study you contribute makes the evaluation richer and more honest about what AI can (and can't) replicate.

The good news? You don't need to run any experiments yourself. You just need to translate an existing published study into a format our system understands. Think of it as turning a paper into a recipe that an AI agent can follow.

---

## Your study folder at a glance

Each study lives in its own directory under `studies/study_XXX/`. At minimum, you need **2 folders** and **2 files**:

```
studies/study_XXX/
├── index.json           # Who wrote it, what year, what it's about
├── README.md            # A friendly description for humans browsing the repo
├── source/              # The paper's data, however you want to structure it
│   ├── ground_truth.json          (common — extracted findings + stats)
│   ├── specification.json         (common — participants, design)
│   └── materials/                 (common — questions, scenarios, stimuli)
│       └── *.json
└── scripts/             # The replication engine
    ├── config.py                  (common — trial generation + prompts)
    └── evaluator.py               (common — statistical tests on responses)
```

!!! tip "Minimal requirements"
    The verifier only enforces `source/`, `scripts/`, `index.json`, and `README.md`. Everything else — `ground_truth.json`, `config.py`, `evaluator.py` — is convention, not law. Structure your `source/` however makes sense for your study.

---

## The contribution process

Contributing a study follows three primary phases: extraction, building, and submission.

1. **[How to Extract Data from a Paper](extract_from_paper.md)**
   Understand the paper's structure and systematically extract the required information, including findings, statistical tests, and experimental materials.

2. **[How to Build Your Study Files](build_study_files.md)**
   Translate the extracted data into the standardized JSON and Python file formats required by the benchmark.

3. **[How to Submit Your Study](submit_study.md)**
   Commit your changes and open a pull request for review. CI runs the verifier automatically.

---

<details>
<summary><strong>A real example to follow</strong></summary>

The reference study — `studies/study_001/` — replicates Ross et al. 1977's *False Consensus Effect*. It has three sub-studies, five findings, and all the trimmings:

```
studies/study_001/
├── index.json
├── README.md
├── source/
│   ├── ground_truth.json
│   ├── specification.json
│   ├── metadata.json
│   ├── materials/
│   │   ├── study_1_hypothetical_stories.json
│   │   ├── study_2_personal_description_items.json
│   │   └── study_3_sandwich_board_hypothetical.json
│   └── 4705-Ross-et-al-False-Consensus-Effect.pdf
└── scripts/
    ├── config.py
    ├── evaluator.py
    ├── study_utils.py
    └── stats_lib.py
```

We reference it throughout the guides. When in doubt, peek at `study_001`.

</details>

<details>
<summary><strong>Studies come in all shapes</strong></summary>

Not every study is a simple questionnaire. The benchmark already includes some pretty wild designs:

- **study_001** — Classic single-participant survey (Ross et al. 1977, "doesn't everyone agree with me?")
- **study_009** — Multi-participant, multi-round game where everyone's choices affect the next round (Nagel 1995, p-beauty contest)
- **study_012** — Two-stage trust game where one participant's decision becomes another's input (Berg et al. 1995)

Your study might be a straightforward survey, a strategic interaction, or something entirely new. The benchmark grows more interesting with diversity.

</details>

Ready? Start with [How to Extract Data from a Paper](extract_from_paper.md).
