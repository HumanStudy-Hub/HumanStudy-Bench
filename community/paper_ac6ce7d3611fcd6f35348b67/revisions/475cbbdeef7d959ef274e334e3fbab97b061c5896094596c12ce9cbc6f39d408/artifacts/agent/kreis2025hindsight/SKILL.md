---
name: kreis2025hindsight
description: "Kreis et al. Hindsight Bias Through Knowledge Updating: A Conceptual Replication of Groß et al. (2023): 1 study as a HumanStudy program, with participants, design, conditions, verbatim materials, procedure, analysis, reported results and open issues, each cited to its page. Use when asked about this study, to brief or run AI participants in it, or to compare agent answers with the reported human results."
---

# Hindsight Bias Through Knowledge Updating: A Conceptual Replication of Groß et al. (2023)

Barbara K. Kreis, Antje Hermann, Thorsten Pachur, and Julia Groß. Hindsight Bias Through Knowledge Updating: A Conceptual Replication of Groß et al. (2023).

This folder is the study rebuilt as a program: what each participant saw and did, in order, and how the paper analysed it. Every fact comes from the paper and carries its page. `program.json` is the full machine-readable version.

## Studies

| Study | Title | Read | Conditions |
| --- | --- | --- | --- |
| `s1` | Preregistered online experiment: hindsight bias and transfer learning for the sugar content of food items | [study-s1.md](references/study-s1.md) | `c_control`, `c_feedback`, `c_domain`, `c_irrelevant` |

Also: [paper.md](references/paper.md) (background, findings, limitations), [open-issues.md](references/open-issues.md) (what the paper does not state) and [sources.md](references/sources.md) (source files and hashes) and [accuracy.md](references/accuracy.md) (what has been checked).

Search everything: `python scripts/study_core.py search <words>`.

## Run AI participants

Not ready to run: the study still needs decisions (iss_filler). Reading and querying work.

The replay package `package/kreis2025hindsight/` holds the participant prompts (`task/adapter.py`) and the paper's analysis (`evaluation/evaluation.py`).

Each AI participant must answer in its own fresh context that sees only its prompt. Never answer several prompts yourself in this conversation: you know the hypothesis, so those answers would not be independent participants.

With an agent CLI you are signed in to (no key needed), one new process per participant:

```bash
python scripts/study_core.py run --n 5 --cli "claude -p --model haiku"
python scripts/study_core.py run --n 5 --cli "codex exec --skip-git-repo-check -s read-only -"
```

Each process gets only the prompt on stdin, in an empty folder, so it cannot read this study. With an API key instead: `--model <model-id>` (`OPENROUTER_API_KEY`, or `OPENAI_API_KEY` with `OPENAI_BASE_URL`). To collect answers some other way: `plan --n 5` lists every prompt, and `replay replies.json --n 5` scores the replies given in the same order.

## Rules

- Answer only from these files and cite the page given. If something is not here, say so.
- Materials marked verbatim are shown to participants exactly as written. Never paraphrase them.
- Items that read "Not stated in the paper" and open issues stay open: ask the researcher; never fill them with plausible text.
- Results from AI participants are a simulation. They are not observations of people.
