# How to Extract Data from a Paper

Before you write any code, you need to understand what you're looking at. Social science papers have a particular structure, and knowing how to read them makes the rest of this process much easier.

---

## What typically goes in a social science paper

A social science paper reports one or more experiments designed to test hypotheses about human behavior. The authors recruit participants, put them through some kind of task — filling out questionnaires, making economic decisions, solving puzzles — and then analyze the results with statistical tests.

What we need from each paper:

- **What were they testing?** The hypotheses and expected effects.
- **What did participants do, and who were they?** The tasks, instructions, stimuli, sample sizes, demographics, and how participants were grouped or assigned to conditions.
- **How was the experiment run?** The procedure — number of rounds, phases, interactions between participants, timing, and any feedback given.
- **What did they find?** The statistical results — test names, test statistics, p-values, means, standard deviations, and raw data points.

Our job is to pull these things out of the paper so an AI agent can attempt the same experiment and we can compare its behavior to the original human results.

---

## How papers are organized

Most social science papers follow a hierarchical structure:

- A **paper** contains one or more **studies** (sometimes called experiments)
- Each study tests one or more **findings** (hypotheses the authors wanted to prove or disprove)
- Each finding is supported by one or more **statistical tests**
- Each test comes with raw data: means, SDs, sample sizes, p-values

Here's how that looks for **study_001** — Ross et al. 1977, the "False Consensus Effect" (spoiler: people really do think everyone agrees with them):

```
Paper: Ross et al. (1977) — False Consensus Effect
│
├── Study 1 (n = 320): Questionnaire with hypothetical situations
│   ├── Finding 1: Consensus estimates differ by what you chose
│   │   └── 4 scenarios, each with an F-test
│   └── Finding 2: Trait ratings differ by what you chose
│       └── 4 scenarios, each with an F-test
│
├── Study 2 (n = 80): Personal description items
│   └── Finding 3: Category estimates differ by your own category
│       └── 17 items, t-tests
│
└── Study 3 (n = 104): The sandwich board situation
    ├── Finding 4: Consensus for behavioral choices
    └── Finding 5: What you think about people who made the other choice
```

Each level of this tree maps to a part of our file format. Keep this picture in mind — it'll make everything click.

!!! info "Not every paper is this tidy"
    Some papers have just one study with one finding. Others have complex multi-round interactions (like study_009's beauty contest game) or multi-stage designs (like study_012's trust game where one player's decision feeds into another's). Some papers don't report full statistical details — they might only give raw values, partial test statistics, or describe results in prose rather than tables. Extract what's there and note what's missing — don't force it into a shape it doesn't fit.

---

<details>
<summary><strong>Extraction checklist and file mapping</strong></summary>

Here's what to extract and where it typically ends up:

| What to extract | What to look for | Common target file |
|----------------|-----------------|-------------------|
| **Study structure** | Studies, findings, hypotheses, conditions | `source/ground_truth.json` |
| **Materials** | Question text, scenarios, instructions, response options | `source/materials/*.json` |
| **Participants & procedure** | Sample sizes, demographics, group assignments, experimental steps | `source/specification.json` |
| **Statistical results** | Test names, statistics, p-values, means, SDs, raw data | `source/ground_truth.json` |
| **Title, authors, year** | Paper metadata | `index.json` |

These file names and structures are conventions from existing studies, not enforced rules. Organize your `source/` directory however makes sense for your paper — the verifier only checks that `source/` exists.

For a complete real-world example, browse `studies/study_001/source/`.

</details>

<details>
<summary><strong>AI-assisted extraction prompt</strong></summary>

Reading a 20-page methods section cover to cover is slow. Here's a prompt you can give to an LLM (with the paper attached) to extract most of what you need in one shot:

> **Identify all findings in [PAPER TITLE] and identify all statistical tests supporting the findings. Include every single replication detail of each experiment, including the tests, participants, and raw data.**
>
> **Extract complete information for each replicable experiment/study to enable replication and evaluation.**
>
> **EXTRACTION REQUIREMENTS:**
> 1. Label each finding as "Finding 1", "Finding 2", etc. (or use paper's notation like "F1", "F2")
> 2. Extract all statistical tests for each finding — significant, non-significant, marginal, interactions, follow-ups
> 3. Include complete raw data for each test: means, SDs, sample sizes, differences
>
> **For EACH study/experiment, extract:**
>
> **1. STUDY STRUCTURE:**
> - Study ID, name, phenomenon tested
> - Findings: list with IDs and hypotheses
> - All sub-studies, scenarios, conditions
>
> **2. MATERIALS & PROCEDURE:**
> - Actual text of questions, scenarios, instructions, stimuli
> - Item-level details: question text, response options, scales
> - Experimental procedure: steps, rounds, phases, feedback mechanisms
>
> **3. PARTICIPANTS:**
> - Sample sizes, demographics, group assignments, exclusion criteria
>
> **4. STATISTICAL RESULTS:**
> For each test:
> - **finding_id**: Which finding this addresses (e.g. "Finding 1", "F2")
> - **test_name**: e.g. "t-test", "ANOVA", "correlation"
> - **statistic**: Complete string, e.g. "t(23) = 4.66", "F(1, 68) = 6.38"
> - **p_value**: Exact value, e.g. "p < .001", "p = .04", "not significant"
> - **raw_data**: Means, SDs, sample sizes for all groups
> - **claim**: What the test evaluates
> - **location**: Page and section, e.g. "Page 489, Table 1"

You'll still want to double-check the output against the paper — LLMs occasionally hallucinate statistics, which is exactly the kind of thing you don't want in a benchmark.

</details>

---

## Next step

You've got the data. Now let's turn it into actual files — head to [How to Build Your Study Files](build_study_files.md).
