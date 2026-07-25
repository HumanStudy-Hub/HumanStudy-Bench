# How to Submit Your Study

There are two ways to contribute a study to HumanStudy-Bench.

## Build Study

HumanStudy-Hub is developing a guided Build Study workflow for social-science
researchers. It is designed to accept a published paper and optional open
materials, pause for researcher review, and produce a runnable experiment
folder that can be downloaded as a ZIP or saved to HumanStudy-Bench.

Build Study is currently available through the private-beta HumanStudy-Hub web
application. Direct pull requests remain available to every contributor and are
the preferred path when authors want full control over each study file.

## Direct pull request

### 1. Fork and clone

Fork [HumanStudy-Bench](https://github.com/HumanStudy-Hub/HumanStudy-Bench),
then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/HumanStudy-Bench.git
cd HumanStudy-Bench
git remote add upstream https://github.com/HumanStudy-Hub/HumanStudy-Bench.git
```

### 2. Create a branch

```bash
git checkout -b contrib-yourgithubid-study-name
```

### 3. Add and verify the study

Follow [What should I submit?](what_to_submit.md) and
[How to build study files](build_study_files.md), then run:

```bash
bash scripts/verify_study.sh yourgithubid_study_id
```

### 4. Commit and push

```bash
git add studies/yourgithubid_study_id/
git commit -m "Add study: Your Study Title"
git push origin contrib-yourgithubid-study-name
```

### 5. Open a pull request

Open a pull request targeting `main`. Include a short study description and
anything reviewers should inspect carefully.

Your `index.json` must identify the contributor:

```json
"contributors": [
  {
    "name": "Your Name",
    "github": "https://github.com/your-username",
    "institution": "Your Institution"
  }
]
```

Use `"Independent Researcher"` when no institution applies.

After submission:

- CI runs the study validator and rebuilds the study index.
- A maintainer reviews structure, source fidelity, and data quality.
- After approval, the study is merged and receives its final repository ID.
