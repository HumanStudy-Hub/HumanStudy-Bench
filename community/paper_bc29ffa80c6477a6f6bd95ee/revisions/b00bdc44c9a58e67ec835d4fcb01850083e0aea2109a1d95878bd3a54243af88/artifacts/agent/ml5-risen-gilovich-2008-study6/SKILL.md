---
name: ml5-risen-gilovich-2008-study6
description: "Mathur et al. (2020) Many Labs 5: Registered Multisite Replication of the Tempting-Fate Effects in Risen and Gilovich (2008): 2 studies as a HumanStudy program, with participants, design, conditions, verbatim materials, procedure, analysis, reported results and open issues, each cited to its page. Use when asked about this study, to brief or run AI participants in it, or to compare agent answers with the reported human results."
---

# Many Labs 5: Registered Multisite Replication of the Tempting-Fate Effects in Risen and Gilovich (2008)

Maya B. Mathur, Diane-Jo Bart-Plange, Balazs Aczel, et al., and Michael C. Frank (2020). Many Labs 5: Registered Multisite Replication of the Tempting-Fate Effects in Risen and Gilovich (2008). Advances in Methods and Practices in Psychological Science. https://doi.org/10.1177/2515245918785165.

This folder is the study rebuilt as a program: what each participant saw and did, in order, and how the paper analysed it. Every fact comes from the paper and carries its page. `program.json` is the full machine-readable version.

## Studies

| Study | Title | Read | Conditions |
| --- | --- | --- | --- |
| `s1` | Multisite replication of Risen & Gilovich (2008) Study 6 (tempting fate × cognitive load) | [study-s1.md](references/study-s1.md) | `c00`, `c10`, `c01`, `c11` |
| `s2` | Post hoc comparison with Many Labs 2 (analysis only) | [study-s2.md](references/study-s2.md) | none recorded |

Also: [paper.md](references/paper.md) (background, findings, limitations), [open-issues.md](references/open-issues.md) (what the paper does not state) and [sources.md](references/sources.md) (source files and hashes) and [accuracy.md](references/accuracy.md) (111 quotes checked against their pages: 111 unchecked) and [capabilities.md](references/capabilities.md) (what can and cannot be done with this study, and why).

Search everything: `python scripts/study_core.py search <words>`.

## Run AI participants

Ready to run the parts the replay package covers (see `package/ml5-risen-gilovich-2008/task/task.json`). Results from AI participants are a simulation, not observations of people.

The replay package `package/ml5-risen-gilovich-2008/` holds the participant prompts (`task/adapter.py`) and the paper's analysis (`evaluation/evaluation.py`).

Each AI participant must answer in its own fresh context that sees only its prompt. Never answer several prompts yourself in this conversation: you know the hypothesis, so those answers would not be independent participants.

With an agent CLI you are signed in to (no key needed), one new process per participant:

```bash
python scripts/study_core.py run --n 5 --cli "claude -p --model haiku"
python scripts/study_core.py run --n 5 --cli "codex exec --skip-git-repo-check -s read-only -"
```

Each process gets only the prompt on stdin, in an empty folder, so it cannot read this study. With an API key instead: `--model <model-id>` (`OPENROUTER_API_KEY`, or `OPENAI_API_KEY` with `OPENAI_BASE_URL`). To collect answers some other way: `plan --n 5` lists every prompt, and `replay replies.json --n 5` scores the replies given in the same order.

## Human data

The original human data are not in this folder. `python scripts/study_core.py human-data` downloads them from their source and checks every file against its pinned SHA-256. `python scripts/study_core.py reproduce` recomputes the paper's key results from them with the package's analysis: trust the analysis on AI participants only as far as it recovers the paper here. If the data are a replication sample, say so when you report.

## Rules

- Answer only from these files and cite the page given. If something is not here, say so.
- Materials marked verbatim are shown to participants exactly as written. Never paraphrase them.
- Items that read "Not stated in the paper" and open issues stay open: ask the researcher; never fill them with plausible text.
- Results from AI participants are a simulation. They are not observations of people.
- Do only what capabilities.md allows. A refused request (status `refused`) is reported with its reason; never replace it with an approximation of your own.
