# How to Submit Your Study

You've extracted the data and built the files. The hard part is over — now you just need to get it into the repo.

---

## Option A: GitHub Pull Request

### 1. Fork & clone

Fork the [HumanStudy-Bench](https://github.com/HumanStudy-Hub/HumanStudy-Bench) repo on GitHub first, then clone your fork. If you haven't already:

```bash
git clone https://github.com/YOUR_USERNAME/HumanStudy-Bench.git
cd HumanStudy-Bench
git remote add upstream https://github.com/HumanStudy-Hub/HumanStudy-Bench.git
```

### 2. Create a branch

```bash
git checkout -b contrib-yourgithubid-013
```

### 3. Commit & push

```bash
git add studies/study_XXX/
git commit -m "Add study: Your Study Title"
git push origin contrib-yourgithubid-013
```

### 4. Open a Pull Request

Head to GitHub and open a **Pull Request** targeting the `main` branch. Include a brief description of the study and any notes for reviewers.

Make sure your `index.json` includes a `contributors` field with your name, GitHub link, and institution — CI will verify your GitHub ID matches the PR author:

```json
"contributors": [
  {
    "name": "Your Name",
    "github": "https://github.com/your-username",
    "institution": "Stanford University"
  }
]
```

If you are not affiliated with a university or organization, use `"Independent Researcher"` as your institution.

### What happens after you hit "Create PR"

- **CI** runs `verify_study.sh` and `build_studies_index.py` automatically. It also checks that the `contributors` GitHub ID in your `index.json` matches the PR author.
- **Human review**: A maintainer will look over your study structure, data quality, and description.
- **Merge**: Once approved, your PR gets merged. The studies index rebuilds, and your study goes live on the [project website](https://www.hs-bench.clawder.ai). Maintainers assign the final `study_XXX` number by merge order.

---

## Option B: Web upload

Head to [hs-bench.clawder.ai/contribute](https://www.hs-bench.clawder.ai/contribute), upload a `.zip` of your study folder, and we'll create a GitHub PR on your behalf.

---
