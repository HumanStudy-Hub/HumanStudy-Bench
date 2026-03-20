# How to Build Your Study Files

Time to turn your extracted data into a proper study folder. This guide covers the files most studies include. Only `source/`, `scripts/`, `index.json`, and `README.md` are strictly required — everything else is convention.

!!! tip "Complex experiments"
    If your study doesn't fit neatly into this template, feel free to improvise — the only hard rule is passing `scripts/verify_study.sh study_XXX`. Multi-round games, multi-stage interactions, custom participant pools — it's all welcome. See study_009 and study_012 for inspiration.

---

## Set up your directory

From the repo root:

```bash
mkdir -p studies/study_XXX/source/materials
mkdir -p studies/study_XXX/scripts
```

Replace `study_XXX` with something descriptive for now. Maintainers assign final numbering when they merge.

---

## index.json

This file lives at the root of your study directory. The website reads it for the study catalog, and CI validates it.

**Required fields:** `title` (string), `authors` (string[]), `year` (number | null), `description` (string) — all non-empty.

**Optional:** `contributors` — array of objects so you get credit on the site.

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Your full name |
| `github` | No | Your GitHub profile URL |
| `institution` | No | Your university or organization. Use `"Independent Researcher"` if unaffiliated |

```json
{
  "title": "False Consensus Effect",
  "authors": ["Lee Ross", "David Greene", "Pamela House"],
  "year": 1977,
  "description": "People overestimate how many others share their beliefs and behaviors.",
  "contributors": [
    {
      "name": "Your Name",
      "github": "https://github.com/your-username",
      "institution": "Stanford University"
    }
  ]
}
```

---

## README.md

A short description at the study root: what the study is, who the participants were, what the main tests are. This is for people browsing the repo — the website uses `index.json`.

---

## source/ — your data, your way

**No required schema or template** — structure it however makes sense for your study.

<details>
<summary><strong>ground_truth.json</strong> — common convention for findings and statistical results</summary>

Most studies include a `ground_truth.json` with findings and statistical results. A typical shape:

```json
{
  "study_id": "study_001",
  "title": "False Consensus Effect",
  "authors": ["Ross", "Greene", "House"],
  "year": 1977,
  "studies": [
    {
      "study_id": "study_1_hypothetical_stories",
      "study_name": "Hypothetical Stories Questionnaire",
      "findings": [
        {
          "finding_id": "F1",
          "main_hypothesis": "Consensus estimates differ...",
          "statistical_tests": [
            {
              "test_name": "F-test",
              "reported_statistics": "F(1, 68) = 6.38",
              "significance_level": 0.05,
              "expected_direction": "choosers_higher"
            }
          ],
          "original_data_points": { }
        }
      ]
    }
  ]
}
```

This isn't mandatory. Some studies use a flat list, others use a completely different layout. The evaluator is your code — it reads whatever format you write.

</details>

<details>
<summary><strong>specification.json</strong> — common convention for participant and design details</summary>

Participant details and experimental design. Config scripts often load this via `load_specification()`.

```json
{
  "participants": {
    "n": 320,
    "population": "Stanford undergraduates"
  },
  "design": { "type": "between-subjects", "factors": ["choice"] }
}
```

</details>

<details>
<summary><strong>materials/*.json</strong> — common convention for stimuli and instructions</summary>

One JSON file per sub-study or stimulus set — questions, scenarios, and instructions. Config loads them with `load_material("filename_without_extension")`.

```json
{
  "items": [
    {
      "id": "scenario_1",
      "text": "A large university is considering a ban on...",
      "options": ["Comply with the request", "Refuse"]
    }
  ]
}
```

See `studies/study_001/source/materials/` for three real examples.

</details>

---

## scripts/ — bringing it to life

 **No required template** — implement whatever fits your study.

<details>
<summary><strong>config.py</strong> — trial generation and prompt building</summary>

Your config generates experiment trials and builds prompts for the AI agent. The typical pattern uses two classes:

1. A **PromptBuilder** (subclass of `PromptBuilder` from `study_utils`) that implements `build_trial_prompt(trial_metadata) -> str`
2. A **Config** (subclass of `BaseStudyConfig` from `study_utils`) that sets `prompt_builder_class` and implements `create_trials(self, n_trials=None) -> list`

**BaseStudyConfig gives you:**
- `self.load_specification()` — reads `source/specification.json`
- `self.load_material(name)` — reads `source/materials/<name>.json`

```python
from study_utils import BaseStudyConfig, PromptBuilder

class CustomPromptBuilder(PromptBuilder):
    def build_trial_prompt(self, trial_metadata):
        # Craft a prompt from the trial's items, scenario, etc.
        # Return a single string the agent will receive.
        ...

class StudyConfig(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder

    def create_trials(self, n_trials=None):
        spec = self.load_specification()
        materials = self.load_material("study_1_hypothetical_stories")
        trials = []
        # Build trial dicts with sub_study_id, scenario_id, items, etc.
        return trials
```

For multi-participant studies (games, negotiations, group decisions), set `REQUIRES_GROUP_TRIALS = True` and implement `run_group_experiment()`. See study_009 and study_012 for working examples.

</details>

<details>
<summary><strong>evaluator.py</strong> — statistical tests on agent responses</summary>

Your evaluator parses the agent's responses, groups them by condition, and runs statistical tests to compare against the original human results.

```python
def evaluate_study(results):
    """
    results: dict with at least {"individual_data": [...]}
    Each element typically has response_text and trial_info.

    Returns: {"test_results": [...]}
    """
    test_results = []
    individual_data = results.get("individual_data", [])
    # Parse responses, group by scenario, run statistical tests
    # Compare agent behavior to original paper findings
    return {"test_results": test_results}
```

Common fields in each test result: `study_id`, `sub_study_id`, `finding_id`, `t_stat`, `p_value`, `significant`, `direction_match`, `human_p_value`, `replication`. The exact fields depend on your study — there's no enforced schema, just return a dict with a `test_results` key.

</details>

---

All set? Head to [How to Submit Your Study](submit_study.md).
